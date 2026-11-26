"""Time series analysis for multi-timepoint trials.

This module extends sctrial's capabilities beyond 2-timepoint DiD to handle
longitudinal studies with 3+ timepoints.

Mathematical Background
-----------------------
For studies with >2 timepoints, several approaches are available:

1. **Generalized DiD (Event Study)**:
   Compare each post-treatment timepoint to baseline:

       Y_it = α_i + Σ_k β_k × D_it^k + ε_it

   where D_it^k = 1 if participant i is treated and at timepoint k.

2. **Linear Trend Interaction**:
   Test if treatment changes the slope over time:

       Y_it = α_i + β₁×Time + β₂×Treat×Time + ε_it

   H₀: β₂ = 0 (treatment doesn't change trajectory)

3. **Quadratic/Polynomial Trends**:
   Allow for non-linear trajectories:

       Y_it = α_i + β₁×Time + β₂×Time² + β₃×Treat×Time + β₄×Treat×Time² + ε_it

4. **Spline Models**:
   Flexible non-parametric trend modeling using basis splines.

Model Selection Guidance
------------------------

- 3-4 timepoints: Linear trend interaction is usually sufficient
- 5+ timepoints: Consider polynomial or spline models
- Irregular intervals: Use actual time values, not ordinal indices
- Treatment onset varies: Use time relative to treatment start

Key Assumptions
---------------
1. Parallel trends in pre-treatment period (for causal inference)
2. No anticipation effects
3. Treatment effect timing is correctly specified
4. Panel is balanced or data are missing completely at random (MCAR)
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ..adata_tools import subset_arms
from ..design import TrialDesign
from ._utils import apply_fdr, drop_unused_levels, fit_identified, fits_exactly, unidentified_terms
from .did import MIN_CLUSTERS_FOR_ROBUST_SE, AggregateMode

__all__ = [
    "trend_interaction",
    "event_study_did",
    "polynomial_trend",
    "test_parallel_trends",
    "TrendModel",
]

TrendModel = Literal["linear", "quadratic", "cubic"]


def _prepare_longitudinal_data(
    adata: AnnData,
    design: TrialDesign,
    visits: Sequence[str],
    features: Sequence[str],
    layer: str | None = None,
    exclude_crossovers: bool = True,
) -> pd.DataFrame:
    """Prepare data for a two-arm longitudinal analysis: the visits and the two named arms."""
    from ._extract import extract_gene_matrix

    # Filter to specified visits
    mask = adata.obs[design.visit_col].isin(visits)
    if exclude_crossovers and design.crossover_col:
        mask &= ~adata.obs[design.crossover_col].astype(bool)

    ad = subset_arms(adata[mask].copy(), design, stacklevel=4)

    # Build dataframe
    df = ad.obs[[design.participant_col, design.visit_col, design.arm_col]].copy()

    # Add features
    obs_feats = [f for f in features if f in ad.obs.columns]
    gene_feats = [f for f in features if f in ad.var_names and f not in ad.obs.columns]
    missing = [f for f in features if f not in ad.obs.columns and f not in ad.var_names]
    if missing:
        raise KeyError(f"Features not found in obs or var_names: {missing[:5]}")

    for feat in obs_feats:
        df[feat] = ad.obs[feat].values

    if gene_feats:
        mat = extract_gene_matrix(ad, gene_feats, layer=layer)
        df_genes = pd.DataFrame(mat, columns=gene_feats, index=df.index)
        df = pd.concat([df, df_genes], axis=1)

    return df


def trend_interaction(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: Sequence[str],
    time_values: Sequence[float] | None = None,
    model: TrendModel = "linear",
    aggregate: AggregateMode = "participant_visit",
    layer: str | None = None,
    exclude_crossovers: bool = True,
) -> pd.DataFrame:
    """Test treatment × time trend interactions for multi-timepoint data.

    This function fits a model where treatment modifies the trajectory
    over time:

    Linear model:
        Y_it = α_i + β₁×Time + β₂×Treat×Time + ε_it

        H₀: β₂ = 0 (treatment doesn't change the slope)

    Quadratic model:
        Y_it = α_i + β₁×Time + β₂×Time² + β₃×Treat×Time + β₄×Treat×Time² + ε_it

        H₀: β₃ = β₄ = 0 (no treatment effect on trajectory)

    Parameters
    ----------
    adata
        AnnData object with longitudinal data.
    features
        Features to test.
    design
        TrialDesign object.
    visits
        Ordered list of visit labels (3+ visits).
    time_values
        Numeric time values corresponding to visits.
        If None, uses ordinal indices (0, 1, 2, ...).
    model
        "linear", "quadratic", or "cubic".
    aggregate
        Aggregation mode.
    layer
        Expression layer.
    exclude_crossovers
        Exclude crossover participants.

    Returns
    -------
    pd.DataFrame
        Results with columns:

        - feature: Feature name
        - beta_trend: Main time effect (in control)
        - beta_treat_trend: Treatment × time interaction (linear)
        - se_treat_trend: Standard error of the linear interaction (cluster-robust by
          participant)
        - p_treat_trend: P-value for linear interaction
        - n_units: Number of participants
        - n_timepoints: Number of visits
        - beta_treat_trend2: Treatment × time² interaction (quadratic and cubic models)
        - p_treat_trend2: P-value for quadratic interaction (quadratic and cubic models)
        - beta_treat_trend3: Treatment × time³ interaction (cubic model)
        - p_treat_trend3: P-value for cubic interaction (cubic model)
        - FDR_treat_trend: Benjamini-Hochberg FDR of the linear interaction across features
        - error: Message of a fit that raised (present only when a fit raised)

        A feature has no estimate (missing values in its row) when both arms are not
        among its participants, or when the design does not identify the interaction,
        for example a cubic model on three visits.

    Examples
    --------
    >>> # 4-timepoint study: Day 0, 7, 14, 28
    >>> res = trend_interaction(
    ...     adata, features=genes, design=design,
    ...     visits=["D0", "D7", "D14", "D28"],
    ...     time_values=[0, 7, 14, 28],
    ...     model="linear"
    ... )
    >>> print(res[res["p_treat_trend"] < 0.05])
    """
    import statsmodels.formula.api as smf

    if len(visits) < 3:
        raise ValueError("trend_interaction requires 3+ timepoints. Use did_table for 2.")

    # Prepare time values
    if time_values is None:
        time_values = list(range(len(visits)))
    time_map = dict(zip(visits, time_values))

    # Prepare data
    df = _prepare_longitudinal_data(adata, design, visits, features, layer, exclude_crossovers)

    # Add numeric time
    df["time_num"] = df[design.visit_col].map(time_map).astype(float)
    df["arm_bin"] = design.arm_bin(df)

    # Aggregate to participant-visit level
    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, design.visit_col, design.arm_col, "time_num", "arm_bin"]
        df = df.groupby(grp_cols, observed=True)[list(features)].mean().reset_index()

    # Enforce balanced panel: keep only participants observed at all visits
    visit_counts = df.groupby(design.participant_col, observed=True)[design.visit_col].nunique()
    balanced_pids = visit_counts[visit_counts == len(visits)].index
    df = drop_unused_levels(df[df[design.participant_col].isin(balanced_pids)].copy())

    n_timepoints = len(visits)
    no_estimate = {
        "beta_trend": np.nan,
        "beta_treat_trend": np.nan,
        "se_treat_trend": np.nan,
        "p_treat_trend": np.nan,
    }
    higher_terms = {"linear": [], "quadratic": ["2"], "cubic": ["2", "3"]}.get(model, [])
    for power in higher_terms:
        no_estimate[f"beta_treat_trend{power}"] = np.nan
        no_estimate[f"p_treat_trend{power}"] = np.nan
    interaction = "arm_bin:time_num"

    model_cols = [design.participant_col, design.visit_col, "time_num", "arm_bin"]
    rows = []
    for feat in features:
        # Only the columns of this feature's model: a frame with a column for every feature,
        # copied once per feature, makes the cost per feature grow with the number of features.
        df_feat = df[list(dict.fromkeys([*model_cols, feat]))]
        # The panel is balanced for each feature: a participant without a value at one visit
        # is not in the model of that feature.
        df_feat = df_feat.dropna(subset=[feat])
        visits_seen = df_feat.groupby(design.participant_col, observed=True)[
            design.visit_col
        ].nunique()
        balanced = visits_seen.index[visits_seen == n_timepoints]
        df_feat = drop_unused_levels(df_feat[df_feat[design.participant_col].isin(balanced)].copy())
        n_units = int(df_feat[design.participant_col].nunique())
        df_feat["outcome_std"] = df_feat[feat].astype(float)

        # Standardize
        y_std = df_feat["outcome_std"].std(ddof=1)
        if not np.isfinite(y_std) or y_std < 1e-12:
            rows.append(
                {"feature": feat, **no_estimate, "n_units": n_units, "n_timepoints": n_timepoints}
            )
            continue
        df_feat["outcome_std"] = (df_feat["outcome_std"] - df_feat["outcome_std"].mean()) / y_std

        # Build formula based on model
        if model == "linear":
            formula = f"outcome_std ~ time_num + arm_bin:time_num + C({design.participant_col})"
        elif model == "quadratic":
            df_feat["time_num2"] = df_feat["time_num"] ** 2
            formula = f"outcome_std ~ time_num + time_num2 + arm_bin:time_num + arm_bin:time_num2 + C({design.participant_col})"
        elif model == "cubic":
            df_feat["time_num2"] = df_feat["time_num"] ** 2
            df_feat["time_num3"] = df_feat["time_num"] ** 3
            formula = f"outcome_std ~ time_num + time_num2 + time_num3 + arm_bin:time_num + arm_bin:time_num2 + arm_bin:time_num3 + C({design.participant_col})"
        else:
            raise ValueError(f"Unknown model: {model}")

        try:
            if n_units < MIN_CLUSTERS_FOR_ROBUST_SE:
                warnings.warn(
                    f"Only {n_units} clusters (participants) available. Cluster-robust "
                    f"standard errors are unreliable with fewer than {MIN_CLUSTERS_FOR_ROBUST_SE} "
                    f"clusters.",
                    UserWarning,
                    stacklevel=2,
                )
            trend_model = smf.ols(formula, data=df_feat)
            fit = fit_identified(
                trend_model,
                interaction,
                cov_type="cluster",
                cov_kwds={"groups": df_feat[design.participant_col]},
            )
            if fit is None:
                # One arm among the participants of this feature (the interaction column
                # then equals the time column or is zero), more terms than the visits
                # allow, or a time coding that repeats a value.
                rows.append(
                    {
                        "feature": feat,
                        **no_estimate,
                        "n_units": n_units,
                        "n_timepoints": n_timepoints,
                    }
                )
                continue

            # An exact fit gives the estimates and no test; so does a term that the other
            # regressors determine.
            exact = fits_exactly(fit)
            blank = unidentified_terms(trend_model, [f"{interaction}{p}" for p in higher_terms])
            errors = fit.bse * np.nan if exact else fit.bse.drop(blank, errors="ignore")
            tested = fit.pvalues * np.nan if exact else fit.pvalues.drop(blank, errors="ignore")
            estimates = fit.params.drop(blank, errors="ignore")
            result = {
                "feature": feat,
                "beta_trend": float(fit.params.get("time_num", np.nan)),
                "beta_treat_trend": float(fit.params.get("arm_bin:time_num", np.nan)),
                "se_treat_trend": float(errors.get("arm_bin:time_num", np.nan)),
                "p_treat_trend": float(tested.get("arm_bin:time_num", np.nan)),
                "n_units": n_units,
                "n_timepoints": n_timepoints,
            }

            if model in ["quadratic", "cubic"]:
                result["beta_treat_trend2"] = float(estimates.get("arm_bin:time_num2", np.nan))
                result["p_treat_trend2"] = float(tested.get("arm_bin:time_num2", np.nan))

            if model == "cubic":
                result["beta_treat_trend3"] = float(estimates.get("arm_bin:time_num3", np.nan))
                result["p_treat_trend3"] = float(tested.get("arm_bin:time_num3", np.nan))

            rows.append(result)

        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            rows.append(
                {
                    "feature": feat,
                    **no_estimate,
                    "n_units": n_units,
                    "n_timepoints": n_timepoints,
                    "error": str(e),
                }
            )

    res = pd.DataFrame(rows)
    if rows:
        # one column order, whichever feature comes first and whether or not it has an estimate
        order = [
            "feature",
            *list(no_estimate)[:4],
            "n_units",
            "n_timepoints",
            *list(no_estimate)[4:],
        ]
        res = res[[*order, *(c for c in res.columns if c not in order)]]

    # FDR correction for linear trend
    if "p_treat_trend" in res.columns:
        res = apply_fdr(res, p_col="p_treat_trend", fdr_col="FDR_treat_trend")

    return res


def event_study_did(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: Sequence[str],
    reference_visit: str | None = None,
    layer: str | None = None,
    exclude_crossovers: bool = True,
) -> pd.DataFrame:
    """Run event study DiD comparing each visit to a reference baseline.

    This is a generalization of 2-period DiD to multiple periods.
    For each post-baseline visit, we estimate a separate DiD effect.

    Model:
        Y_it = α_i + Σ_k (γ_k × Visit_k) + Σ_k (β_k × Treat × Visit_k) + ε_it

    β_k captures the DiD effect at visit k relative to baseline.

    Parameters
    ----------
    adata
        AnnData object.
    features
        Features to test.
    design
        TrialDesign object.
    visits
        Ordered visit labels.
    reference_visit
        Baseline visit for comparison. If None, uses first visit.
    layer
        Expression layer.
    exclude_crossovers
        Exclude crossover participants.

    Returns
    -------
    pd.DataFrame
        Long-format results, one row per feature and post-baseline visit, with the
        columns of :func:`did_table` and:

        - feature: Feature name
        - visit: Post-baseline visit
        - reference: Reference visit
        - beta_DiD: DiD effect vs reference
        - se_DiD: Standard error
        - p_DiD: P-value
        - FDR_DiD: Benjamini-Hochberg FDR across all visit-feature tests (recomputed over
          the combined table, not within each visit)
        - n_units: Sample size
        - error: Message of a visit whose table raised (present only in that case)

    Examples
    --------
    >>> # Compare each follow-up to baseline
    >>> res = event_study_did(
    ...     adata, features=genes, design=design,
    ...     visits=["Baseline", "Week4", "Week8", "Week12"],
    ...     reference_visit="Baseline"
    ... )
    >>> # Visualize event study plot
    >>> for feat in genes[:3]:
    ...     sub = res[res["feature"] == feat]
    ...     plt.errorbar(sub["visit"], sub["beta_DiD"], yerr=1.96*sub["se_DiD"])
    """
    from .did import did_table

    if reference_visit is None:
        reference_visit = visits[0]

    if reference_visit not in visits:
        raise ValueError(f"Reference visit '{reference_visit}' not in visits list")

    # Restrict once, so that rows outside the two arms are reported once and a design
    # without two arms in the data is an error here, not an error row per visit.
    adata = subset_arms(
        adata[adata.obs[design.visit_col].isin(list(visits)).to_numpy()], design, stacklevel=3
    )

    # Run DiD for each post-baseline visit
    all_results = []
    post_visits = [v for v in visits if v != reference_visit]

    for post_visit in post_visits:
        try:
            res = did_table(
                adata,
                features=features,
                design=design,
                visits=(reference_visit, post_visit),
                exclude_crossovers=exclude_crossovers,
                layer=layer,
                aggregate="participant_visit",
            )
            res["visit"] = post_visit
            res["reference"] = reference_visit
            all_results.append(res)
        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            # Create empty result for this visit
            empty = pd.DataFrame({"feature": features})
            empty["visit"] = post_visit
            empty["reference"] = reference_visit
            empty["error"] = str(e)
            all_results.append(empty)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        # Recalculate FDR across all visit-feature tests to maintain proper
        # family-wise error control (each did_table() only corrects within its visit).
        if "p_DiD" in combined.columns and "FDR_DiD" in combined.columns:
            from ._utils import apply_fdr

            combined = apply_fdr(combined, p_col="p_DiD", fdr_col="FDR_DiD")
    else:
        combined = pd.DataFrame()

    return combined


def polynomial_trend(
    adata: AnnData,
    feature: str,
    design: TrialDesign,
    visits: Sequence[str],
    time_values: Sequence[float] | None = None,
    degree: int = 2,
    layer: str | None = None,
) -> dict:
    """Fit polynomial trend model for a single feature.

    This function fits a polynomial model and returns detailed diagnostics
    including predicted trajectories.

    Parameters
    ----------
    adata
        AnnData object.
    feature
        Single feature to model.
    design
        TrialDesign object.
    visits
        Ordered visit labels.
    time_values
        Numeric time values (or None for ordinal).
    degree
        Polynomial degree (1=linear, 2=quadratic, 3=cubic).
    layer
        Expression layer.

    Returns
    -------
    dict
        Model results:

        - coefficients: All model coefficients
        - pvalues: P-value of each coefficient
        - predictions: Predicted values per arm×time
        - aic, bic: Model fit statistics
        - rsquared: R-squared of the model
        - residuals: Model residuals

    Raises
    ------
    ValueError
        If the data do not identify a time term or an arm-by-time term: fewer distinct
        time points than the degree plus one, or one arm only with a value of the feature.
        Least squares would still return coefficients, p-values and a curve between the
        time points, each split arbitrarily among the terms that determine each other.
    """
    import statsmodels.formula.api as smf

    if time_values is None:
        time_values = list(range(len(visits)))
    time_map = dict(zip(visits, time_values))

    df = _prepare_longitudinal_data(
        adata, design, visits, [feature], layer, exclude_crossovers=True
    )

    # Aggregate
    grp_cols = [design.participant_col, design.visit_col, design.arm_col]
    df = drop_unused_levels(df.groupby(grp_cols, observed=True)[[feature]].mean().reset_index())

    df["time_num"] = df[design.visit_col].map(time_map).astype(float)
    df["arm_bin"] = design.arm_bin(df)
    df["outcome_std"] = df[feature].astype(float)

    # Build polynomial terms
    terms = ["time_num"]
    for d in range(2, degree + 1):
        df[f"time_num{d}"] = df["time_num"] ** d
        terms.append(f"time_num{d}")

    # Build formula with interactions
    fixed = " + ".join(terms)
    interact = " + ".join([f"arm_bin:{t}" for t in terms])
    formula = f"outcome_std ~ {fixed} + {interact} + C({design.participant_col})"

    polynomial = smf.ols(formula, data=df)
    undetermined = unidentified_terms(polynomial, [*terms, *(f"arm_bin:{t}" for t in terms)])
    if undetermined:
        raise ValueError(
            f"polynomial_trend: {', '.join(undetermined)} not identified for '{feature}'. A "
            f"polynomial of degree {degree} needs {degree + 1} distinct time points, and "
            "participants of both arms with a value of the feature."
        )
    fit = polynomial.fit()

    # Generate population-average predictions by averaging across all
    # participant fixed effects.
    time_grid = np.linspace(min(time_values), max(time_values), 50)
    all_pids = df[design.participant_col].unique()
    pred_records = []
    for treat in [0, 1]:
        for t in time_grid:
            preds_for_t = []
            for pid in all_pids:
                row_dict = {"time_num": t, "arm_bin": treat, design.participant_col: pid}
                for d_i in range(2, degree + 1):
                    row_dict[f"time_num{d_i}"] = t**d_i
                preds_for_t.append(row_dict)
            try:
                pid_df = pd.DataFrame(preds_for_t)
                mean_pred = float(fit.predict(pid_df).mean())
            except (ValueError, KeyError, TypeError):
                mean_pred = np.nan
            pred_records.append({"time_num": t, "arm_bin": treat, "predicted": mean_pred})

    pred_df = pd.DataFrame(pred_records)

    pred_df["arm"] = pred_df["arm_bin"].map({0: design.arm_control, 1: design.arm_treated})

    return {
        "coefficients": fit.params.to_dict(),
        "pvalues": fit.pvalues.to_dict(),
        "predictions": pred_df[["time_num", "arm", "predicted"]],
        "aic": fit.aic,
        "bic": fit.bic,
        "rsquared": fit.rsquared,
        "residuals": fit.resid.values,
    }


def test_parallel_trends(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    pre_visits: Sequence[str],
    layer: str | None = None,
) -> pd.DataFrame:
    """Test the parallel trends assumption using pre-treatment data.

    A key assumption of DiD is that treatment and control groups would
    have followed parallel trajectories in the absence of treatment.
    This can be partially tested using pre-treatment data.

    Model:
        Y_it = α_i + β₁×Time + β₂×Treat×Time + ε_it  (pre-treatment only)

    H₀: β₂ = 0 (parallel pre-trends)

    A significant β₂ suggests the parallel trends assumption may be violated.

    Parameters
    ----------
    adata
        AnnData object.
    features
        Features to test.
    design
        TrialDesign object.
    pre_visits
        List of pre-treatment visits (2+ visits).
    layer
        Expression layer.

    Returns
    -------
    pd.DataFrame
        Results with:

        - feature: Feature name
        - beta_pretrend: Pre-treatment trend difference
        - p_pretrend: P-value for non-parallel trends
        - FDR_pretrend: Benjamini-Hochberg FDR of ``p_pretrend`` across features
        - warning: "Potential violation" if p < 0.10, "OK" otherwise, and
          "Not estimable" when the feature has no test (see trend_interaction)
        - n_units: Number of participants

    Examples
    --------
    >>> # Check parallel trends in screening visits
    >>> pre_test = test_parallel_trends(
    ...     adata, features=genes, design=design,
    ...     pre_visits=["Screen1", "Screen2", "Baseline"]
    ... )
    >>> violations = pre_test[pre_test["warning"] == "Potential violation"]
    >>> print(f"Features with potential violations: {len(violations)}")
    """
    if len(pre_visits) < 2:
        raise ValueError("Need at least 2 pre-treatment visits to test parallel trends")

    # Use linear trend interaction on pre-treatment data only
    res = trend_interaction(
        adata,
        features=features,
        design=design,
        visits=pre_visits,
        model="linear",
        layer=layer,
        exclude_crossovers=True,
    )

    # Rename columns for clarity
    res = res.rename(
        columns={
            "beta_treat_trend": "beta_pretrend",
            "p_treat_trend": "p_pretrend",
            "FDR_treat_trend": "FDR_pretrend",
        }
    )

    # Flag potential violations
    # A feature without a test is neither a pass nor a violation.
    res["warning"] = np.select(
        [res["p_pretrend"].isna(), res["p_pretrend"] < 0.10],
        ["Not estimable", "Potential violation"],
        default="OK",
    )

    return res[["feature", "beta_pretrend", "p_pretrend", "FDR_pretrend", "warning", "n_units"]]
