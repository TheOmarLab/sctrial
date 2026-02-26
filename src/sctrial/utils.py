from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = [
    "safe_filename",
    "intersect_preserve_order",
    "ensure_unique_index",
    "looks_like_counts",
    "get_counts_matrix",
    "wild_cluster_bootstrap_t",
    "permutation_pvalue",
    "permutation_pvalue_paired",
    "resolve_feature",
]


def safe_filename(s: str, maxlen: int = 180) -> str:
    """Return a filesystem-safe filename slug.

    Parameters
    ----------
    s
        Input string to sanitize.
    maxlen
        Maximum length of the output string.
    """
    s = str(s)
    s = s.replace("γ", "gamma").replace("δ", "delta")
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s[:maxlen] if len(s) > maxlen else s


def intersect_preserve_order(items: Sequence[str], universe: Iterable[str]) -> list[str]:
    """Return items that appear in universe, preserving original order."""
    u = set(universe)
    return [x for x in items if x in u]


def ensure_unique_index(df: pd.DataFrame, *, agg: str = "mean") -> pd.DataFrame:
    """If df.index has duplicates, aggregate duplicates and return a new df.

    agg: "mean" or "sum" (extend later if needed).
    """
    if df.index.is_unique:
        return df
    if agg == "mean":
        return df.groupby(level=0).mean(numeric_only=True)
    if agg == "sum":
        return df.groupby(level=0).sum(numeric_only=True)
    raise ValueError(f"Unsupported agg='{agg}'. Use 'mean' or 'sum'.")


def looks_like_counts(X, sample: int = 10000, seed: int = 0) -> bool:
    """Check if matrix appears to be raw counts."""
    rng = np.random.default_rng(seed)
    if X is None:
        return False
    if hasattr(X, "toarray"):
        data = X.data if hasattr(X, "data") else np.asarray(X).ravel()
    else:
        data = np.asarray(X).ravel()
    if data.size == 0:
        return False
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    if data.size > sample:
        data = rng.choice(data, size=sample, replace=False)
    return bool(np.all(data >= 0) and np.allclose(data, np.round(data), atol=1e-6))


def get_counts_matrix(adata: AnnData) -> tuple[np.ndarray | None, str | None]:
    """Return a raw-counts matrix and its source label, if available."""
    if "counts" in adata.layers and looks_like_counts(adata.layers["counts"]):
        return adata.layers["counts"], "layers['counts']"
    if getattr(adata, "raw", None) is not None:
        if list(adata.raw.var_names) == list(adata.var_names) and looks_like_counts(adata.raw.X):
            return adata.raw.X, "adata.raw.X"
    if "raw" in adata.layers and looks_like_counts(adata.layers["raw"]):
        return adata.layers["raw"], "layers['raw']"
    if looks_like_counts(adata.X):
        return adata.X, "adata.X"
    return None, None


def wild_cluster_bootstrap_t(
    fit: RegressionResultsWrapper,
    X: np.ndarray,
    clusters: np.ndarray,
    term_name: str,
    B: int = 999,
    seed: int = 42,
) -> float:
    r"""Wild cluster bootstrap (Rademacher) for one coefficient.

    Notes
    -----
    Implements a **wild cluster bootstrap-t** using Rademacher weights at the
    cluster level. This is recommended when the number of clusters is small
    and standard cluster-robust inference may be unreliable.

    Each bootstrap draw perturbs the **restricted** residuals (imposing
    H0: beta_j = 0) with cluster-level Rademacher weights (±1 with equal
    probability), re-fits the full model via OLS (or WLS when the
    original fit used weights) with **per-iteration cluster-robust SE**,
    and forms a bootstrap t-statistic.  The two-sided p-value is the
    fraction of bootstrap \|t*\| values that exceed the observed \|t\|.

    Reference:
    Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2008).
    Bootstrap-based improvements for inference with clustered errors.
    The Review of Economics and Statistics, 90(3), 414–427.

    Parameters
    ----------
    fit
        Statsmodels regression results (with cluster-robust SE).
    X
        Design matrix (fit.model.exog).
    clusters
        Array of cluster IDs.
    term_name
        Name of the coefficient to test.
    B
        Number of bootstrap draws.
    seed
        Random seed.

    Returns
    -------
    p_boot : float
        Two-sided wild cluster bootstrap p-value.
    """
    rng = np.random.default_rng(seed)
    coef_names = fit.model.exog_names
    if term_name not in coef_names:
        return np.nan

    j = coef_names.index(term_name)
    beta_hat = fit.params.iloc[j]
    se_hat = fit.bse.iloc[j]

    if not np.isfinite(se_hat) or se_hat == 0:
        return np.nan

    t_obs = beta_hat / se_hat
    uniq_cl = np.unique(clusters)
    G = len(uniq_cl)

    # Detect WLS vs OLS
    weights = getattr(fit.model, "weights", None)
    use_wls = weights is not None

    import statsmodels.api as sm

    # Restricted residuals: impose H0 (beta_j = 0) by subtracting only the
    # non-null components of the fit.
    restricted_fitted = fit.fittedvalues - beta_hat * X[:, j]
    resid_r = fit.model.endog - restricted_fitted

    t_boot = np.empty(B, dtype=float)

    for b in range(B):
        w_g = rng.choice([-1, 1], size=G)
        w_map = dict(zip(uniq_cl, w_g))
        w_i = np.array([w_map[g] for g in clusters])

        e_star = resid_r * w_i
        y_star = restricted_fitted + e_star

        if use_wls:
            fit_b = sm.WLS(y_star, X, weights=weights).fit(
                cov_type="cluster", cov_kwds={"groups": clusters}
            )
        else:
            fit_b = sm.OLS(y_star, X).fit(
                cov_type="cluster", cov_kwds={"groups": clusters}
            )

        beta_b = fit_b.params.iloc[j]
        se_b = fit_b.bse.iloc[j]
        if np.isfinite(se_b) and se_b > 0:
            t_boot[b] = beta_b / se_b
        else:
            t_boot[b] = np.nan

    # Drop failed draws (non-finite SE) before computing p-value
    valid = t_boot[np.isfinite(t_boot)]
    if len(valid) == 0:
        return np.nan

    # +1 correction (same as permutation_pvalue) to avoid p=0 and ensure
    # the observed statistic is included in the reference distribution.
    count = np.sum(np.abs(valid) >= np.abs(t_obs))
    p_boot = (count + 1) / (len(valid) + 1)
    return float(p_boot)


def permutation_pvalue(
    group1: np.ndarray,
    group2: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sample permutation test for difference in means.
    Parameters
    ----------
    group1
        First group of values.
    group2
        Second group of values.
    n_perm
        Number of permutations.
    seed
        Random seed.

    Returns
    -------
    float
        Two-sided permutation p-value in ``[0, 1]``.
    Notes
    -----
    H0: mean(group1) = mean(group2)
    """
    rng = np.random.default_rng(seed)
    obs_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        p_diff = np.mean(perm[:n1]) - np.mean(perm[n1:])
        if abs(p_diff) >= abs(obs_diff):
            count += 1

    return (count + 1) / (n_perm + 1)


def permutation_pvalue_paired(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """Paired permutation test (sign-flip test) for difference in means.

    Parameters
    ----------
    x
        First group of values.
    y
        Second group of values.
    n_perm
        Number of permutations.
    seed
        Random seed.

    Returns
    -------
    float
        Two-sided permutation p-value in ``[0, 1]``.
    Notes
    -----
    H0: mean(y - x) = 0
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(y) - np.asarray(x)
    obs_mean = np.mean(diff)

    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        p_mean = np.mean(diff * signs)
        if abs(p_mean) >= abs(obs_mean):
            count += 1

    return (count + 1) / (n_perm + 1)


def resolve_feature(adata: AnnData, query: str) -> str:
    """Resolve a feature name in adata.var_names or adata.obs.columns (case-insensitive).

    Parameters
    ----------
    adata
        AnnData object.
    query
        Feature name to resolve.

    Returns
    -------
    str
        The exact name string to use.

    Raises
    ------
    KeyError
        If the feature is not found in adata.var_names or adata.obs.columns.
    ValueError
        If the feature query is an empty string.
    """
    if query is None or (isinstance(query, str) and query.strip() == ""):
        raise ValueError("Feature query must be a non-empty string.")
    # 1. Exact matches
    if query in adata.obs.columns:
        return query
    if query in adata.var_names:
        return query

    # 2. Case-insensitive obs
    obs_matches = [c for c in adata.obs.columns if c.lower() == query.lower()]
    if len(obs_matches) == 1:
        return obs_matches[0]

    # 3. Case-insensitive var
    var_matches = [g for g in adata.var_names if g.lower() == query.lower()]
    if len(var_matches) == 1:
        return var_matches[0]

    if len(obs_matches) > 1 or len(var_matches) > 1:
        raise KeyError(f"Feature '{query}' is ambiguous (multiple case-insensitive matches).")

    raise KeyError(f"Feature '{query}' not found in obs or var_names.")
