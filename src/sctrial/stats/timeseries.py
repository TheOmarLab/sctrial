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
4. Panel is balanced or missingness is random (MCAR)
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from statsmodels.stats.multitest import multipletests

from ..design import TrialDesign
from .did import AggregateMode

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
    """Prepare data for longitudinal analysis."""
    from ._extract import extract_gene_vector

    # Filter to specified visits
    mask = adata.obs[design.visit_col].isin(visits)
    if exclude_crossovers and design.crossover_col:
        mask &= ~adata.obs[design.crossover_col].astype(bool)

    ad = adata[mask].copy()

    # Build dataframe
    df = ad.obs[[design.participant_col, design.visit_col, design.arm_col]].copy()

    # Add features
    for feat in features:
        if feat in ad.obs.columns:
            df[feat] = ad.obs[feat].values
        elif feat in ad.var_names:
            df[feat] = extract_gene_vector(ad, feat, layer=layer)

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
        - p_treat_trend: P-value for linear interaction
        - beta_treat_trend2: Treatment × time² interaction (if quadratic)
        - p_treat_trend2: P-value for quadratic interaction
        - n_units: Number of participants
        - n_timepoints: Number of visits

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
    df = _prepare_longitudinal_data(
        adata, design, visits, features, layer, exclude_crossovers
    )

    # Add numeric time
    df["_time"] = df[design.visit_col].map(time_map).astype(float)
    df["_treat"] = (df[design.arm_col] == design.arm_treated).astype(int)

    # Aggregate to participant-visit level
    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, design.visit_col, design.arm_col, "_time", "_treat"]
        df = df.groupby(grp_cols, observed=True)[list(features)].mean().reset_index()

    n_units = df[design.participant_col].nunique()
    n_timepoints = len(visits)

    rows = []
    for feat in features:
        df_feat = df.copy()
        df_feat["_y"] = df_feat[feat].astype(float)

        # Standardize
        y_std = df_feat["_y"].std(ddof=1)
        if y_std < 1e-12:
            rows.append({"feature": feat, "n_units": n_units, "n_timepoints": n_timepoints})
            continue
        df_feat["_y"] = (df_feat["_y"] - df_feat["_y"].mean()) / y_std

        # Build formula based on model
        if model == "linear":
            formula = f"_y ~ _time + _treat:_time + C({design.participant_col})"
        elif model == "quadratic":
            df_feat["_time2"] = df_feat["_time"] ** 2
            formula = f"_y ~ _time + _time2 + _treat:_time + _treat:_time2 + C({design.participant_col})"
        elif model == "cubic":
            df_feat["_time2"] = df_feat["_time"] ** 2
            df_feat["_time3"] = df_feat["_time"] ** 3
            formula = f"_y ~ _time + _time2 + _time3 + _treat:_time + _treat:_time2 + _treat:_time3 + C({design.participant_col})"
        else:
            raise ValueError(f"Unknown model: {model}")

        try:
            fit = smf.ols(formula, data=df_feat).fit(
                cov_type="cluster",
                cov_kwds={"groups": df_feat[design.participant_col]}
            )

            result = {
                "feature": feat,
                "beta_trend": float(fit.params.get("_time", np.nan)),
                "beta_treat_trend": float(fit.params.get("_treat:_time", np.nan)),
                "se_treat_trend": float(fit.bse.get("_treat:_time", np.nan)),
                "p_treat_trend": float(fit.pvalues.get("_treat:_time", np.nan)),
                "n_units": n_units,
                "n_timepoints": n_timepoints,
            }

            if model in ["quadratic", "cubic"]:
                result["beta_treat_trend2"] = float(fit.params.get("_treat:_time2", np.nan))
                result["p_treat_trend2"] = float(fit.pvalues.get("_treat:_time2", np.nan))

            if model == "cubic":
                result["beta_treat_trend3"] = float(fit.params.get("_treat:_time3", np.nan))
                result["p_treat_trend3"] = float(fit.pvalues.get("_treat:_time3", np.nan))

            rows.append(result)

        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            rows.append({
                "feature": feat,
                "n_units": n_units,
                "n_timepoints": n_timepoints,
                "error": str(e),
            })

    res = pd.DataFrame(rows)

    # FDR correction for linear trend
    if "p_treat_trend" in res.columns:
        mask = res["p_treat_trend"].notna()
        res["FDR_treat_trend"] = np.nan
        if mask.sum() > 0:
            res.loc[mask, "FDR_treat_trend"] = multipletests(
                res.loc[mask, "p_treat_trend"], method="fdr_bh"
            )[1]

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
        Long-format results with columns:
        - feature: Feature name
        - visit: Post-baseline visit
        - beta_DiD: DiD effect vs reference
        - se_DiD: Standard error
        - p_DiD: P-value
        - n_units: Sample size

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
        Model results including:
        - coefficients: All model coefficients
        - predictions: Predicted values per arm×time
        - aic, bic: Model fit statistics
        - residuals: Model residuals
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
    df = df.groupby(grp_cols, observed=True)[[feature]].mean().reset_index()

    df["_time"] = df[design.visit_col].map(time_map).astype(float)
    df["_treat"] = (df[design.arm_col] == design.arm_treated).astype(int)
    df["_y"] = df[feature].astype(float)

    # Build polynomial terms
    terms = ["_time"]
    for d in range(2, degree + 1):
        df[f"_time{d}"] = df["_time"] ** d
        terms.append(f"_time{d}")

    # Build formula with interactions
    fixed = " + ".join(terms)
    interact = " + ".join([f"_treat:{t}" for t in terms])
    formula = f"_y ~ {fixed} + {interact} + C({design.participant_col})"

    fit = smf.ols(formula, data=df).fit()

    # Generate predictions
    time_grid = np.linspace(min(time_values), max(time_values), 50)
    pred_data = []
    for treat in [0, 1]:
        for t in time_grid:
            row = {"_time": t, "_treat": treat}
            for d in range(2, degree + 1):
                row[f"_time{d}"] = t ** d
            # Use mean participant (approximate)
            row[design.participant_col] = df[design.participant_col].iloc[0]
            pred_data.append(row)

    pred_df = pd.DataFrame(pred_data)
    try:
        pred_df["predicted"] = fit.predict(pred_df)
    except (ValueError, KeyError, TypeError):
        pred_df["predicted"] = np.nan

    pred_df["arm"] = pred_df["_treat"].map({0: design.arm_control, 1: design.arm_treated})

    return {
        "coefficients": fit.params.to_dict(),
        "pvalues": fit.pvalues.to_dict(),
        "predictions": pred_df[["_time", "arm", "predicted"]],
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
        - warning: "Violation" if p < 0.10

    Examples
    --------
    >>> # Check parallel trends in screening visits
    >>> pre_test = test_parallel_trends(
    ...     adata, features=genes, design=design,
    ...     pre_visits=["Screen1", "Screen2", "Baseline"]
    ... )
    >>> violations = pre_test[pre_test["warning"] == "Violation"]
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
    res = res.rename(columns={
        "beta_treat_trend": "beta_pretrend",
        "p_treat_trend": "p_pretrend",
        "FDR_treat_trend": "FDR_pretrend",
    })

    # Flag potential violations
    res["warning"] = np.where(
        res["p_pretrend"] < 0.10,
        "Potential violation",
        "OK"
    )

    return res[["feature", "beta_pretrend", "p_pretrend", "FDR_pretrend", "warning", "n_units"]]
