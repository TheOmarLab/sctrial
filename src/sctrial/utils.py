from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = [
    "safe_filename",
    "intersect_preserve_order",
    "ensure_unique_index",
    "looks_like_counts",
    "wild_cluster_bootstrap_t",
    "permutation_pvalue",
    "permutation_pvalue_paired",
    "resolve_feature",
]


def safe_filename(s: str, maxlen: int = 180) -> str:
    s = str(s)
    s = s.replace("γ", "gamma").replace("δ", "delta")
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s[:maxlen] if len(s) > maxlen else s


def intersect_preserve_order(items: Sequence[str], universe: Iterable[str]) -> list[str]:
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
    return np.all(data >= 0) and np.allclose(data, np.round(data), atol=1e-6)


def wild_cluster_bootstrap_t(
    fit,
    X: np.ndarray,
    clusters: np.ndarray,
    term_name: str,
    B: int = 999,
    seed: int = 42,
) -> float:
    """Wild cluster bootstrap (Rademacher) for one coefficient.

    Notes
    -----
    Implements a **wild cluster bootstrap-t** using Rademacher weights at the
    cluster level. This is recommended when the number of clusters is small
    and standard cluster-robust inference may be unreliable.

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
    resid = fit.resid
    fitted = fit.fittedvalues
    uniq_cl = np.unique(clusters)
    G = len(uniq_cl)

    # Detect WLS vs OLS
    weights = getattr(fit.model, "weights", None)
    use_wls = weights is not None

    import statsmodels.api as sm
    t_boot = np.empty(B, dtype=float)

    for b in range(B):
        w_g = rng.choice([-1, 1], size=G)
        w_map = dict(zip(uniq_cl, w_g))
        w_i = np.array([w_map[g] for g in clusters])

        e_star = resid * w_i
        y_star = fitted + e_star

        if use_wls:
            fit_b = sm.WLS(y_star, X, weights=weights).fit()
        else:
            fit_b = sm.OLS(y_star, X).fit()

        beta_b = fit_b.params.iloc[j]
        t_boot[b] = (beta_b - beta_hat) / se_hat

    p_boot = np.mean(np.abs(t_boot) >= np.abs(t_obs))
    return float(p_boot)


def permutation_pvalue(
    group1: np.ndarray,
    group2: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sample permutation test for difference in means.

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

    Returns the exact name string to use. Raises KeyError if not found.
    """
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
