"""Effect size calculations for trial-aware inference.

This module provides standardized effect size measures (Cohen's d, Hedge's g)
and confidence intervals for DiD analyses.

Statistical Background
----------------------
**Cohen's d** and **Hedge's g** are standardized effect size measures that express
the difference between groups in standard deviation units, enabling comparison
across studies with different scales.

**Cohen's d vs Bootstrap CI**: These serve complementary purposes:
- Cohen's d: Standardized effect size for meta-analysis and interpretation
- Bootstrap CI: Uncertainty quantification on any estimator

**Recommendation**: Report BOTH standardized effect sizes (for interpretation/meta-analysis)
AND confidence intervals (for statistical inference). Bootstrap CI on Cohen's d
provides the best of both worlds.

**Hedge's g vs Cohen's d**: Hedge's g applies a small-sample correction factor
(J = 1 - 3/(4(n₁+n₂-2)-1) = 1 - 3/(4*df - 1)) that reduces upward bias in small samples. Use Hedge's g when
n < 20 per group; for larger samples they are nearly identical.

Mathematical Definitions
------------------------
Cohen's d for DiD:
    d = β_DiD / s_pooled

    where s_pooled = sqrt(((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2))

Hedge's g:
    g = d × J

    where J = 1 - 3/(4(n₁+n₂-2)-1) = 1 - 3/(4*df - 1)  (Hedges' correction factor)

95% CI via noncentral t-distribution:
    SE(d) ≈ sqrt(n₁+n₂/(n₁×n₂) + d²/(2(n₁+n₂)))
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from statsmodels.regression.linear_model import RegressionResultsWrapper

__all__ = [
    "cohens_d",
    "hedges_g",
    "cohens_d_from_did",
    "effect_size_ci",
    "add_effect_sizes_to_did",
    "EffectSizeMethod",
]

EffectSizeMethod = Literal["cohens_d", "hedges_g"]


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True,
) -> float:
    """Calculate Cohen's d effect size between two groups.

    Cohen's d expresses the difference between group means in standard
    deviation units:

        d = (mean₁ - mean₂) / s_pooled

    Parameters
    ----------
    group1
        First group values (e.g., treatment group).
    group2
        Second group values (e.g., control group).
    pooled
        If True (default), use pooled standard deviation.
        If False, use only group1's standard deviation (Glass's delta).

    Returns
    -------
    float
        Cohen's d effect size. Positive values indicate group1 > group2.

    Notes
    -----
    Interpretation guidelines (Cohen, 1988):
    - ``|d|`` ≈ 0.2: small effect
    - ``|d|`` ≈ 0.5: medium effect
    - ``|d|`` ≈ 0.8: large effect

    These are rough guidelines; practical significance depends on context.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    # Remove NaNs
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan

    mean_diff = np.mean(g1) - np.mean(g2)

    if pooled:
        var1 = np.var(g1, ddof=1)
        var2 = np.var(g2, ddof=1)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_sd = np.sqrt(pooled_var)
    else:
        # Glass's delta: use only group1's SD
        pooled_sd = np.std(g1, ddof=1)

    if pooled_sd < 1e-12:
        return np.nan

    return mean_diff / pooled_sd


def hedges_g(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """Calculate Hedge's g effect size (bias-corrected Cohen's d).

    Hedge's g applies a correction factor J to Cohen's d that reduces
    upward bias in small samples:

        g = d × J
        J = 1 - 3/(4(n₁+n₂-2) - 1)  (equivalently 1 - 3/(4*df - 1))

    Parameters
    ----------
    group1
        First group values.
    group2
        Second group values.

    Returns
    -------
    float
        Hedge's g effect size.

    Notes
    -----
    Use Hedge's g when sample sizes are small (n < 20 per group).
    For larger samples, Cohen's d and Hedge's g are nearly identical.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1, n2 = len(g1), len(g2)

    d = cohens_d(g1, g2, pooled=True)
    if np.isnan(d):
        return np.nan

    # Hedges' correction factor
    df = n1 + n2 - 2
    if df <= 0:
        return np.nan

    # Exact correction using gamma function
    # J = Γ(df/2) / (√(df/2) × Γ((df-1)/2))
    # Approximation: J ≈ 1 - 3/(4df - 1)
    j = 1 - 3 / (4 * df - 1)

    return d * j


def effect_size_ci(
    d: float,
    n1: int,
    n2: int,
    alpha: float = 0.05,
    method: Literal["nct", "bootstrap"] = "nct",
) -> tuple[float, float]:
    """Calculate confidence interval for an effect size.

    Parameters
    ----------
    d
        Effect size (Cohen's d or Hedge's g).
    n1
        Sample size of group 1.
    n2
        Sample size of group 2.
    alpha
        Significance level (default 0.05 for 95% CI).
    method
        - "nct": Noncentral t-distribution (analytical, fast)
        - "bootstrap": Not implemented here; use bootstrap_effect_size_ci

    Returns
    -------
    tuple[float, float]
        (lower, upper) confidence interval bounds.

    Notes
    -----
    The noncentral t-distribution method finds the noncentrality parameters
    λ_L and λ_U such that:

        P(t > t_obs | λ=λ_L) = α/2
        P(t < t_obs | λ=λ_U) = α/2

    Then: d_L = λ_L × √(1/n₁ + 1/n₂), d_U = λ_U × √(1/n₁ + 1/n₂)
    """
    if np.isnan(d) or n1 < 2 or n2 < 2:
        return (np.nan, np.nan)

    if method == "nct":
        # Approximate SE using Hedges & Olkin (1985) formula
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + (d ** 2) / (2 * (n1 + n2 - 2)))
        df = n1 + n2 - 2

        t_crit = stats.t.ppf(1 - alpha / 2, df)

        lower = d - t_crit * se_d
        upper = d + t_crit * se_d

        return (lower, upper)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'nct'.")


def cohens_d_from_did(
    beta_did: float,
    residual_std: float,
) -> float:
    """Calculate Cohen's d from a DiD regression coefficient.

    For DiD models, the effect size is the interaction coefficient
    divided by the residual standard deviation:

        d = β_DiD / σ_residual

    Parameters
    ----------
    beta_did
        The DiD interaction coefficient (treatment × time).
    residual_std
        Residual standard deviation from the regression.

    Returns
    -------
    float
        Cohen's d effect size.

    Notes
    -----
    This is an approximation. For more precise effect sizes, compute
    Cohen's d directly from the change scores in each group.
    """
    if residual_std < 1e-12 or np.isnan(beta_did):
        return np.nan
    return beta_did / residual_std


def _compute_effect_size_from_fit(
    fit: RegressionResultsWrapper,
    term_name: str,
    method: EffectSizeMethod = "hedges_g",
) -> dict:
    """Extract effect size from a fitted regression model.

    Parameters
    ----------
    fit
        Fitted statsmodels regression object.
    term_name
        Name of the coefficient to compute effect size for.
    method
        "cohens_d" or "hedges_g".

    Returns
    -------
    dict
        Dictionary with 'd', 'se_d', 'd_lower', 'd_upper' keys.
    """
    if term_name not in fit.params.index:
        return {"d": np.nan, "se_d": np.nan, "d_lower": np.nan, "d_upper": np.nan}

    beta = fit.params[term_name]
    resid_std = np.sqrt(fit.scale)  # Residual standard deviation

    d = beta / resid_std if resid_std > 1e-12 else np.nan

    # Apply Hedge's correction if requested
    if method == "hedges_g" and not np.isnan(d):
        df = fit.df_resid
        if df > 0:
            j = 1 - 3 / (4 * df - 1)
            d = d * j

    # Approximate CI using delta method
    n = fit.nobs
    se_d = np.sqrt(2 / n + (d ** 2) / (2 * n)) if not np.isnan(d) else np.nan

    if not np.isnan(se_d):
        t_crit = stats.t.ppf(0.975, fit.df_resid)
        d_lower = d - t_crit * se_d
        d_upper = d + t_crit * se_d
    else:
        d_lower = d_upper = np.nan

    return {"d": d, "se_d": se_d, "d_lower": d_lower, "d_upper": d_upper}


def add_effect_sizes_to_did(
    df: pd.DataFrame,
    beta_col: str = "beta_DiD",
    se_col: str = "se_DiD",
    n_col: str = "n_units",
    method: EffectSizeMethod = "hedges_g",
) -> pd.DataFrame:
    """Add standardized effect sizes to a DiD results DataFrame.

    This function computes Cohen's d or Hedge's g from the DiD coefficients
    and adds columns for the effect size and its confidence interval.

    Parameters
    ----------
    df
        DataFrame with DiD results (from did_table).
    beta_col
        Name of the coefficient column.
    se_col
        Name of the standard error column.
    n_col
        Name of the sample size column.
    method
        "cohens_d" or "hedges_g" (recommended for small samples).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        - 'effect_size': Cohen's d or Hedge's g
        - 'effect_size_lower': Lower 95% CI bound
        - 'effect_size_upper': Upper 95% CI bound
        - 'effect_size_interpretation': "small", "medium", "large"

    Examples
    --------
    >>> res = did_table(adata, features=genes, design=design, visits=visits)
    >>> res = add_effect_sizes_to_did(res)
    >>> print(res[["feature", "beta_DiD", "effect_size", "effect_size_interpretation"]])
    """
    df = df.copy()

    effect_sizes = []
    effect_lower = []
    effect_upper = []
    interpretations = []

    for _, row in df.iterrows():
        beta = row.get(beta_col, np.nan)
        se = row.get(se_col, np.nan)
        n = row.get(n_col, 10)  # Default assumption

        if pd.isna(beta) or pd.isna(se) or se < 1e-12:
            effect_sizes.append(np.nan)
            effect_lower.append(np.nan)
            effect_upper.append(np.nan)
            interpretations.append("")
            continue

        # Approximate residual SD from SE and sample size
        # For DiD: SE ≈ residual_SD × √(4/n) for balanced design
        # So: residual_SD ≈ SE × √(n/4)
        approx_resid_sd = se * np.sqrt(n / 4) if n > 0 else se

        d = beta / approx_resid_sd if approx_resid_sd > 1e-12 else np.nan

        # Apply Hedge's correction
        if method == "hedges_g" and not np.isnan(d) and n > 2:
            df = n - 2
            j = 1 - 3 / (4 * df - 1) if df > 1 else 1.0
            d = d * j

        # CI via approximate SE
        if not np.isnan(d) and n > 2:
            se_d = np.sqrt(4 / n + (d ** 2) / (2 * n))
            t_crit = stats.t.ppf(0.975, max(n - 2, 1))
            lower = d - t_crit * se_d
            upper = d + t_crit * se_d
        else:
            lower = upper = np.nan

        effect_sizes.append(d)
        effect_lower.append(lower)
        effect_upper.append(upper)

        # Interpretation
        abs_d = abs(d) if not np.isnan(d) else 0
        if abs_d < 0.2:
            interp = "negligible"
        elif abs_d < 0.5:
            interp = "small"
        elif abs_d < 0.8:
            interp = "medium"
        else:
            interp = "large"
        interpretations.append(interp)

    df["effect_size"] = effect_sizes
    df["effect_size_lower"] = effect_lower
    df["effect_size_upper"] = effect_upper
    df["effect_size_interpretation"] = interpretations

    return df


def bootstrap_effect_size_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    method: EffectSizeMethod = "hedges_g",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    *,
    n_bootstrap: int | None = None,
    ci: float | None = None,
) -> tuple[float, float, float]:
    """Compute effect size with bootstrap confidence interval.

    This provides the most robust CI for effect sizes, especially
    with non-normal distributions or small samples.

    Parameters
    ----------
    group1, group2
        Arrays of values for each group.
    method
        "cohens_d" or "hedges_g".
    n_boot
        Number of bootstrap resamples.
    alpha
        Significance level for CI (e.g., 0.05 for 95% CI).
    seed
        Random seed for reproducibility.
    n_bootstrap
        Deprecated alias for n_boot.
    ci
        Deprecated alias for confidence level. If provided, alpha = 1 - ci.

    Returns
    -------
    tuple[float, float, float]
        (effect_size, ci_lower, ci_upper)
    """
    # Handle deprecated parameter aliases
    if n_bootstrap is not None:
        n_boot = n_bootstrap
    if ci is not None:
        alpha = 1 - ci
    rng = np.random.default_rng(seed)

    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return (np.nan, np.nan, np.nan)

    # Point estimate
    if method == "hedges_g":
        d_obs = hedges_g(g1, g2)
    else:
        d_obs = cohens_d(g1, g2)

    # Bootstrap
    boot_d = np.empty(n_boot)
    for b in range(n_boot):
        b1 = rng.choice(g1, size=n1, replace=True)
        b2 = rng.choice(g2, size=n2, replace=True)

        if method == "hedges_g":
            boot_d[b] = hedges_g(b1, b2)
        else:
            boot_d[b] = cohens_d(b1, b2)

    # Percentile CI
    lower = np.nanpercentile(boot_d, 100 * alpha / 2)
    upper = np.nanpercentile(boot_d, 100 * (1 - alpha / 2))

    return (float(d_obs), float(lower), float(upper))
