"""Mixed effects models for trial-aware single-cell inference.

This module provides linear mixed effects model alternatives to the fixed effects
approach used in the standard DiD functions.

Mathematical Background
-----------------------
**Fixed Effects vs Mixed Effects**:

Fixed Effects (current did_table approach):
    Y_it = α_i + β₁×Post + β₂×Treat×Post + ε_it

    - α_i: participant-specific intercept (absorbed as dummy variables)
    - Pros: No distributional assumptions on random effects
    - Cons: Cannot estimate participant-level variance; loses df

Mixed Effects:
    Y_it = (α + u_i) + β₁×Post + β₂×Treat×Post + ε_it
    u_i ~ N(0, σ²_u)  (random intercept)
    ε_it ~ N(0, σ²_ε)

    - Pros: Estimates variance components; more efficient when assumptions hold
    - Cons: Requires distributional assumptions; can be biased if misspecified

**When to Use Each**:

Use Fixed Effects when:

- Participant effects may correlate with treatment assignment
- Small number of time points
- Interest is purely in within-participant changes

Use Mixed Effects when:

- Random sample of participants from a population
- Interest in generalizing to new participants
- Want to estimate variance components
- Large number of clusters with few observations each

**Recommendation**: For randomized trials, both approaches yield similar
treatment effect estimates. Fixed effects is more robust to misspecification.
Report both as sensitivity analysis.

Model Specification
-------------------
The full DiD mixed model:

    Y_ijt = β₀ + β₁×Treat_j + β₂×Post_t + β₃×(Treat×Post)_jt + u_j + ε_ijt

    Random effects: u_j ~ N(0, σ²_u)  (participant)
    Residuals: ε_ijt ~ N(0, σ²_ε)

The DiD effect is β₃, same as in fixed effects.

Optional random slopes:
    Y_ijt = β₀ + β₁×Treat_j + β₂×Post_t + β₃×(Treat×Post)_jt + u_j + v_j×Post_t + ε_ijt

    This allows the time effect to vary by participant.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from ..design import TrialDesign
from ._utils import (
    aggregate_features,
    apply_fdr,
    covariate_aggregator,
    drop_unused_levels,
    standardize_series,
)
from .did import AggregateFunc, AggregateMode, _ensure_paired

__all__ = [
    "did_mixed",
    "did_table_mixed",
    "compare_fixed_vs_mixed",
]


def did_mixed(
    df: pd.DataFrame,
    outcome: str,
    participant_col: str,
    time_col: str,
    arm_col: str,
    arm_treated: str,
    random_slope: bool = False,
    covariates: Sequence[str] | None = None,
    visits: tuple[str, str] | None = None,
    *,
    arm_control: str | None = None,
) -> dict:
    """Fit a single DiD mixed effects model.

    Model specification:
        Y ~ Treat + Post + Treat:Post + (1 | Participant)

    Or with random slopes:
        Y ~ Treat + Post + Treat:Post + (1 + Post | Participant)

    Parameters
    ----------
    df
        DataFrame with outcome and design columns.
    outcome
        Name of the outcome variable column.
    participant_col
        Column identifying participants (random effect grouping).
    time_col
        Column that holds the two visit labels. Any two distinct values are accepted;
        ``visits`` names which one is the baseline.
    arm_col
        Column with treatment arm labels.
    arm_treated
        Label for the treated arm.
    random_slope
        If True, include random slope for time.
    covariates
        Columns of ``df`` to include as fixed effects. A column that is not numeric
        enters as a categorical term. A name that is not a column raises ``KeyError``.
    visits
        (baseline, follow-up) labels of ``time_col``. If None, the two labels are
        sorted.
    arm_control
        Label for the control arm. When given, the table is restricted to the two
        arms as by :func:`sctrial.subset_arms`: rows whose arm is neither
        ``arm_treated`` nor ``arm_control`` are excluded with a warning, and a
        ``ValueError`` is raised when such rows exist and one of the two arms has
        none. When None, every row that is not ``arm_treated`` forms the reference
        group; a warning is raised if that group holds more than one label or
        missing labels.

    Notes
    -----
    The model is fitted on the rows that hold a value for the outcome, the
    participant, the visit and every covariate in the model. A participant left
    with one visit stays in the model. The result holds missing values, with the
    reason under ``error``, when those rows do not hold the two visits.

    Returns
    -------
    dict
        Results dictionary with keys:

        - beta_DiD: DiD coefficient
        - se_DiD: Standard error
        - p_DiD: P-value
        - ci_lower, ci_upper: 95% CI
        - var_participant: Random intercept variance
        - var_residual: Residual variance
        - icc: Intraclass correlation
        - converged: Model convergence status
        - var_slope: Random slope variance (only with ``random_slope=True``)
        - error: Reason, when the model was not fitted (only in that case)
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels is required for mixed effects models")

    def _not_estimated(reason: str) -> dict:
        return {
            "beta_DiD": np.nan,
            "se_DiD": np.nan,
            "p_DiD": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "var_participant": np.nan,
            "var_residual": np.nan,
            "icc": np.nan,
            "converged": False,
            "error": reason,
        }

    df = df.copy()

    if arm_control is not None:
        from ..adata_tools import subset_arms

        df = subset_arms(
            df,
            TrialDesign(
                participant_col=participant_col,
                arm_col=arm_col,
                arm_treated=arm_treated,
                arm_control=arm_control,
            ),
            stacklevel=3,
        )
    else:
        reference = df.loc[~df[arm_col].isin([arm_treated]), arm_col]
        labels = sorted(str(v) for v in pd.unique(reference.dropna()))
        n_missing = int(reference.isna().sum())
        if len(labels) > 1 or n_missing:
            found = ([f"labels {labels}"] if labels else []) + (
                [f"{n_missing} rows with a missing label"] if n_missing else []
            )
            warnings.warn(
                f"The reference group of did_mixed pools every row that is not "
                f"{arm_treated!r}: {' and '.join(found)} in {arm_col!r}. Pass "
                f"arm_control to compare two arms only.",
                UserWarning,
                stacklevel=2,
            )

    # The rows of the model hold every variable of it. The formula interface would drop the
    # others from the model matrices only, not from the participant labels passed beside it.
    model_covariates = list(covariates or [])
    absent = [cov for cov in model_covariates if cov not in df.columns]
    if absent:
        raise KeyError(f"covariates not found in the table: {absent}")
    df = df.dropna(
        subset=list(dict.fromkeys([outcome, participant_col, time_col, *model_covariates]))
    )

    # Encode binary variables
    df["arm_bin"] = df[arm_col].isin([arm_treated]).astype(int)

    # Ensure time is numeric
    time_vals = df[time_col].unique()
    if len(time_vals) != 2:
        return _not_estimated("Requires exactly 2 time points")

    # Map time to 0/1 (pre=0, post=1)
    if visits is not None:
        time_sorted = list(visits)
        if set(time_vals) != set(time_sorted):
            return _not_estimated(
                f"{time_col!r} holds {sorted(map(str, time_vals))}, not the visits "
                f"{[str(v) for v in time_sorted]}"
            )
    else:
        time_sorted = sorted(time_vals)
    df["post_num"] = df[time_col].map({time_sorted[0]: 0, time_sorted[1]: 1}).astype(float)
    df = drop_unused_levels(df)

    # Build formula
    fixed_part = f"{outcome} ~ arm_bin + post_num + arm_bin:post_num"
    for cov in model_covariates:
        if pd.api.types.is_numeric_dtype(df[cov]):
            fixed_part += f" + {cov}"
        else:
            fixed_part += f" + C({cov})"

    # Random effects specification
    if random_slope:
        re_formula = "1 + post_num"
    else:
        re_formula = "1"

    # Fit model
    try:
        model = smf.mixedlm(
            fixed_part,
            df,
            groups=df[participant_col],
            re_formula=re_formula,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            fit = model.fit(method="powell", maxiter=500, full_output=False)

        # Determine convergence: trust fit.converged as primary indicator,
        # fall back to caught warnings only if the attribute is unavailable.
        if hasattr(fit, "converged") and fit.converged is not None:
            converged = fit.converged
        else:
            converged = not bool(caught)

        # Fallback optimizer: retry with lbfgs if Powell did not converge
        if not converged:
            with warnings.catch_warnings(record=True) as caught2:
                warnings.simplefilter("always", ConvergenceWarning)
                fit = model.fit(method="lbfgs", maxiter=500, full_output=False)
            if hasattr(fit, "converged") and fit.converged is not None:
                converged = fit.converged
            else:
                converged = not bool(caught2)
    except (ValueError, np.linalg.LinAlgError) as e:
        return _not_estimated(str(e))

    # Extract DiD coefficient
    did_term = "arm_bin:post_num"
    if did_term not in fit.params.index:
        did_term = "post_num:arm_bin"  # Alternative ordering

    if did_term in fit.params.index:
        beta = float(fit.params[did_term])
        se = float(fit.bse[did_term])
        pval = float(fit.pvalues[did_term])
        ci = fit.conf_int().loc[did_term]
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    else:
        beta = se = pval = ci_lower = ci_upper = np.nan

    # Variance components
    var_resid = float(fit.scale)
    var_intercept = np.nan
    var_slope = np.nan

    if hasattr(fit, "cov_re") and fit.cov_re is not None:
        cov_re = fit.cov_re
        var_intercept = float(cov_re.iloc[0, 0])
        # Extract slope variance if random slopes were fitted (2x2 matrix)
        if cov_re.shape[0] > 1 and cov_re.shape[1] > 1:
            var_slope = float(cov_re.iloc[1, 1])

    # ICC (based on intercept variance only - standard definition)
    if not np.isnan(var_intercept) and (var_intercept + var_resid) > 0:
        icc = var_intercept / (var_intercept + var_resid)
    else:
        icc = np.nan

    result = {
        "beta_DiD": beta,
        "se_DiD": se,
        "p_DiD": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "var_participant": var_intercept,
        "var_residual": var_resid,
        "icc": icc,
        "converged": converged,
    }

    # Add slope variance if available (when random_slope=True)
    if not np.isnan(var_slope):
        result["var_slope"] = var_slope

    return result


def did_table_mixed(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    exclude_crossovers: bool = True,
    aggregate: AggregateMode = "participant_visit",
    layer: str | None = None,
    agg: AggregateFunc = "mean",
    standardize: bool = True,
    random_slope: bool = False,
    covariates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run DiD analysis using mixed effects models for multiple features.

    This is the mixed effects analog of did_table. The key difference is that
    participant effects are modeled as random draws from a normal distribution
    rather than fixed parameters.

    Model:
        Y_it ~ Treat + Post + Treat×Post + (1 | Participant)

    H₀: β_{Treat×Post} = 0 (no differential treatment effect)

    Parameters
    ----------
    adata
        AnnData object containing expression data.
    features
        List of features (genes or module scores) to test.
    design
        TrialDesign object specifying column mappings.
    visits
        Tuple of (baseline, followup) visit labels.
    exclude_crossovers
        Whether to exclude crossover participants.
    aggregate
        Aggregation mode: "participant_visit" or "cell".
    layer
        Expression layer to use (None for adata.X).
    agg
        Aggregation function ("mean", "median").
    standardize
        Whether to z-score outcomes before fitting.
    random_slope
        Include random slope for time (allows time effect to vary by participant).
    covariates
        Columns of ``adata.obs`` to include as fixed effects. With
        ``aggregate="participant_visit"`` a numeric covariate is its mean over the
        cells of a participant-visit (its median with ``agg="median"``); any other
        covariate must be the same for every cell of a participant-visit.

    Returns
    -------
    pd.DataFrame
        Results with columns:

        - feature: Feature name
        - beta_DiD, se_DiD, p_DiD, FDR_DiD: DiD statistics
        - ci_lower, ci_upper: 95% CI
        - var_participant, var_residual, icc: Variance components
        - n_units: Number of participants in the model of the feature
        - converged: Model convergence

    Notes
    -----
    As in :func:`did_table`, each feature is fitted on the participants that have
    a value for it at both visits. A row without a value for the feature, or for
    a covariate of the model, is not in the model, and neither is the other visit
    of that participant.

    Examples
    --------
    >>> res_mixed = did_table_mixed(
    ...     adata, features=genes, design=design, visits=("Pre", "Post")
    ... )
    >>> print(res_mixed[["feature", "beta_DiD", "icc", "converged"]])

    See Also
    --------
    did_table : Fixed effects version (recommended for most applications).
    compare_fixed_vs_mixed : Compare both approaches.
    """
    from ..adata_tools import subset_arms, subset_primary
    from ._extract import extract_gene_matrix

    if design.arm_col is None:
        raise ValueError("mixed-effects DiD requires a two-arm design (arm_col must not be None)")

    # Subset to analysis population: the two visits and the two named arms
    ad = subset_primary(adata, design, visits, exclude_crossovers=exclude_crossovers)
    ad = subset_arms(ad, design, stacklevel=3)

    # Build dataframe
    obs = ad.obs.copy()
    cols = [design.participant_col, design.visit_col, design.arm_col]
    if design.celltype_col is not None:
        cols.append(design.celltype_col)

    df = obs[cols].copy()

    # Add feature values
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

    # Covariates travel with the rows of the model: a numeric one as its summary over the cells
    # of a participant-visit, any other as its value, which must be the same for those cells.
    cov_agg: dict[str, str] = {}
    for cov in covariates or []:
        if cov in features:
            raise ValueError(f"covariate '{cov}' is also a feature of the model.")
        if cov not in ad.obs.columns:
            raise KeyError(f"covariate '{cov}' not found in adata.obs")
        df[cov] = ad.obs[cov].to_numpy()
        cov_agg[cov] = (
            covariate_aggregator(agg) if pd.api.types.is_numeric_dtype(df[cov]) else "first"
        )

    # Aggregate if requested
    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, design.visit_col, design.arm_col]
        cov_rows = None
        if cov_agg:
            grouped = df.groupby(grp_cols, observed=True)
            for cov, how in cov_agg.items():
                if how == "first" and grouped[cov].nunique().max() > 1:
                    raise ValueError(
                        f"Covariate '{cov}' varies within participant-visit; "
                        "use numeric or constant covariates only."
                    )
            cov_rows = grouped[list(cov_agg)].agg(cov_agg).reset_index()
        df = aggregate_features(df, grp_cols=grp_cols, features=list(features), agg=agg)
        if cov_rows is not None:
            df = df.merge(cov_rows, on=grp_cols, how="left")

    # Ensure paired data
    df = _ensure_paired(df, unit=design.participant_col, time=design.visit_col, visits=visits)

    # Run mixed model for each feature, on a frame of the columns of its own model: a frame
    # with a column for every feature, copied once per feature, makes the cost per feature
    # grow with the number of features.
    model_covariates = list(cov_agg)
    model_cols = [design.participant_col, design.visit_col, design.arm_col, *model_covariates]
    rows = []
    for feat in features:
        df_feat = df[list(dict.fromkeys([*model_cols, feat]))]
        # A participant without a value at one visit has no change for this feature.
        df_feat = df_feat.dropna(subset=list(dict.fromkeys([feat, *model_covariates])))
        visits_seen = df_feat.groupby(design.participant_col, observed=True)[
            design.visit_col
        ].nunique()
        paired = visits_seen.index[visits_seen >= 2]
        df_feat = df_feat[df_feat[design.participant_col].isin(paired)].copy()
        n_units = int(df_feat[design.participant_col].nunique())

        if standardize:
            y_std, ok = standardize_series(df_feat, feat, min_std=1e-12)
            if not ok:
                rows.append(
                    {
                        "feature": feat,
                        "beta_DiD": np.nan,
                        "se_DiD": np.nan,
                        "p_DiD": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "var_participant": np.nan,
                        "var_residual": np.nan,
                        "icc": np.nan,
                        "n_units": n_units,
                        "converged": False,
                    }
                )
                continue
            df_feat["outcome_std"] = y_std
        else:
            df_feat["outcome_std"] = df_feat[feat].astype(float)

        result = did_mixed(
            df_feat,
            outcome="outcome_std",
            participant_col=design.participant_col,
            time_col=design.visit_col,
            arm_col=design.arm_col,
            arm_treated=design.arm_treated,
            random_slope=random_slope,
            covariates=covariates,
            visits=visits,
        )

        rows.append(
            {
                "feature": feat,
                "beta_DiD": result["beta_DiD"],
                "se_DiD": result["se_DiD"],
                "p_DiD": result["p_DiD"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "var_participant": result["var_participant"],
                "var_residual": result["var_residual"],
                "icc": result["icc"],
                "n_units": n_units,
                "converged": result["converged"],
            }
        )

    res = pd.DataFrame(rows)
    res = apply_fdr(res, p_col="p_DiD", fdr_col="FDR_DiD")
    return res


def compare_fixed_vs_mixed(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    **kwargs,
) -> pd.DataFrame:
    """Compare fixed effects and mixed effects DiD results.

    This function runs both approaches and returns a combined DataFrame
    for comparison. This is useful for sensitivity analysis.

    Parameters
    ----------
    adata
        AnnData object.
    features
        Features to test.
    design
        TrialDesign object.
    visits
        (baseline, followup) visit labels.
    **kwargs
        Additional arguments passed to both did_table and did_table_mixed.

    Returns
    -------
    pd.DataFrame
        Combined results with columns:

        - feature
        - beta_fixed, se_fixed, p_fixed: Fixed effects results
        - beta_mixed, se_mixed, p_mixed, icc: Mixed effects results
        - beta_diff: Difference in estimates
        - agreement: Whether both methods agree on significance and direction. Missing
          (``pd.NA``) when either model gave no estimate or no test.

    Examples
    --------
    >>> comparison = compare_fixed_vs_mixed(
    ...     adata, features=genes, design=design, visits=("Pre", "Post")
    ... )
    >>> # Check agreement
    >>> print(f"Methods agree: {comparison['agreement'].mean():.0%}")
    """
    from ..adata_tools import subset_arms
    from .did import did_table

    if design.arm_col is None:
        raise ValueError("compare_fixed_vs_mixed requires a two-arm design (arm_col must not be None)")
    # Restrict once, so that both fits use the same rows and report excluded rows once.
    adata = subset_arms(
        adata[adata.obs[design.visit_col].isin(list(visits)).to_numpy()], design, stacklevel=3
    )

    # Split kwargs: 'random_slope' is mixed-only and would cause TypeError in did_table
    _MIXED_ONLY_KEYS = {"random_slope"}
    fixed_kwargs = {k: v for k, v in kwargs.items() if k not in _MIXED_ONLY_KEYS}

    # Run fixed effects
    res_fixed = did_table(adata, features, design, visits, **fixed_kwargs)

    # Run mixed effects
    res_mixed = did_table_mixed(adata, features, design, visits, **kwargs)

    # Merge results
    comparison = res_fixed[["feature", "beta_DiD", "se_DiD", "p_DiD"]].copy()
    comparison = comparison.rename(
        columns={
            "beta_DiD": "beta_fixed",
            "se_DiD": "se_fixed",
            "p_DiD": "p_fixed",
        }
    )

    mixed_cols = res_mixed[["feature", "beta_DiD", "se_DiD", "p_DiD", "icc"]].copy()
    mixed_cols = mixed_cols.rename(
        columns={
            "beta_DiD": "beta_mixed",
            "se_DiD": "se_mixed",
            "p_DiD": "p_mixed",
        }
    )

    comparison = comparison.merge(mixed_cols, on="feature", how="outer")

    # Compute agreement metrics
    comparison["beta_diff"] = comparison["beta_fixed"] - comparison["beta_mixed"]

    # Agreement on direction and significance
    alpha = 0.05
    fixed_sig = comparison["p_fixed"] < alpha
    mixed_sig = comparison["p_mixed"] < alpha
    same_direction = np.sign(comparison["beta_fixed"]) == np.sign(comparison["beta_mixed"])

    # Neither an agreement nor a disagreement when one of the models gave no estimate or
    # no test: a missing p-value compares as "not significant" and would agree with one.
    compared = comparison[["beta_fixed", "beta_mixed", "p_fixed", "p_mixed"]].notna().all(axis=1)
    comparison["agreement"] = ((fixed_sig == mixed_sig) & same_direction).astype("boolean").where(compared)

    return comparison
