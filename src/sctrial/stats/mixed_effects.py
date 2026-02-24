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
from ._utils import aggregate_features, apply_fdr, standardize_series
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
        Column with time/visit indicators (binary: 0=pre, 1=post).
    arm_col
        Column with treatment arm labels.
    arm_treated
        Label for the treated arm.
    random_slope
        If True, include random slope for time.
    covariates
        Additional covariates to include as fixed effects.

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
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels is required for mixed effects models")

    df = df.copy()

    # Encode binary variables
    df["arm_bin"] = (df[arm_col] == arm_treated).astype(int)

    # Ensure time is numeric
    time_vals = df[time_col].unique()
    if len(time_vals) != 2:
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
            "error": "Requires exactly 2 time points",
        }

    # Map time to 0/1 (pre=0, post=1)
    if visits is not None:
        time_sorted = list(visits)
    else:
        time_sorted = sorted(time_vals)
    df["post_num"] = df[time_col].map({time_sorted[0]: 0, time_sorted[1]: 1}).astype(float)

    # Build formula
    fixed_part = f"{outcome} ~ arm_bin + post_num + arm_bin:post_num"
    if covariates:
        for cov in covariates:
            if cov in df.columns:
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
            "error": str(e),
        }

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
        Additional covariates to include.

    Returns
    -------
    pd.DataFrame
        Results with columns:
        - feature: Feature name
        - beta_DiD, se_DiD, p_DiD, FDR_DiD: DiD statistics
        - ci_lower, ci_upper: 95% CI
        - var_participant, var_residual, icc: Variance components
        - n_units: Number of participants
        - converged: Model convergence

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
    from ..adata_tools import subset_primary
    from ._extract import extract_gene_matrix

    # Subset to analysis population
    ad = subset_primary(adata, design, visits, exclude_crossovers=exclude_crossovers)

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

    # Aggregate if requested
    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, design.visit_col, design.arm_col]
        df = aggregate_features(df, grp_cols=grp_cols, features=list(features), agg=agg)

    # Ensure paired data
    df = _ensure_paired(
        df, unit=design.participant_col, time=design.visit_col, visits=visits
    )

    n_units = df[design.participant_col].nunique()

    # Run mixed model for each feature
    rows = []
    for feat in features:
        df_feat = df.copy()

        if standardize:
            y_std, ok = standardize_series(df_feat, feat, min_std=1e-12)
            if not ok:
                rows.append({
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
                })
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

        rows.append({
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
        })

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
        - agreement: Whether both methods agree on significance direction

    Examples
    --------
    >>> comparison = compare_fixed_vs_mixed(
    ...     adata, features=genes, design=design, visits=("Pre", "Post")
    ... )
    >>> # Check agreement
    >>> print(f"Methods agree: {comparison['agreement'].mean():.0%}")
    """
    from .did import did_table

    # Split kwargs: 'random_slope' is mixed-only and would cause TypeError in did_table
    _MIXED_ONLY_KEYS = {"random_slope"}
    fixed_kwargs = {k: v for k, v in kwargs.items() if k not in _MIXED_ONLY_KEYS}

    # Run fixed effects
    res_fixed = did_table(adata, features, design, visits, **fixed_kwargs)

    # Run mixed effects
    res_mixed = did_table_mixed(adata, features, design, visits, **kwargs)

    # Merge results
    comparison = res_fixed[["feature", "beta_DiD", "se_DiD", "p_DiD"]].copy()
    comparison = comparison.rename(columns={
        "beta_DiD": "beta_fixed",
        "se_DiD": "se_fixed",
        "p_DiD": "p_fixed",
    })

    mixed_cols = res_mixed[["feature", "beta_DiD", "se_DiD", "p_DiD", "icc"]].copy()
    mixed_cols = mixed_cols.rename(columns={
        "beta_DiD": "beta_mixed",
        "se_DiD": "se_mixed",
        "p_DiD": "p_mixed",
    })

    comparison = comparison.merge(mixed_cols, on="feature", how="outer")

    # Compute agreement metrics
    comparison["beta_diff"] = comparison["beta_fixed"] - comparison["beta_mixed"]

    # Agreement on direction and significance
    alpha = 0.05
    fixed_sig = comparison["p_fixed"] < alpha
    mixed_sig = comparison["p_mixed"] < alpha
    same_direction = np.sign(comparison["beta_fixed"]) == np.sign(comparison["beta_mixed"])

    comparison["agreement"] = (fixed_sig == mixed_sig) & same_direction

    return comparison
