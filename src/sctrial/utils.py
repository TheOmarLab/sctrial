"""Shared internal utilities: logging, formatting, helpers."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = [
    "BootstrapResult",
    "safe_filename",
    "intersect_preserve_order",
    "ensure_unique_index",
    "looks_like_counts",
    "get_counts_matrix",
    "wild_cluster_bootstrap_t",
    "warn_unbounded_bootstrap_intervals",
    "permutation_pvalue",
    "permutation_pvalue_paired",
    "resolve_feature",
]


class BootstrapResult(NamedTuple):
    """Result of a wild cluster bootstrap procedure.

    Attributes
    ----------
    p_boot : float
        Two-sided p-value of the restricted wild cluster bootstrap-t test of a zero coefficient.
    se_boot : float
        Bootstrap standard error (SD of the unrestricted bootstrap coefficient estimates).
    ci_lo : float
        Lower bound of the confidence interval obtained by inverting the bootstrap test
        (``-inf`` when the test cannot reject at the requested level).
    ci_hi : float
        Upper bound of that interval (``inf`` when the test cannot reject).
    boot_distribution : np.ndarray
        Unrestricted bootstrap coefficient estimates, centred on the point estimate.
    """

    p_boot: float
    se_boot: float
    ci_lo: float
    ci_hi: float
    boot_distribution: np.ndarray


def safe_filename(s: str, maxlen: int = 180) -> str:
    """Return a filesystem-safe filename slug.

    Parameters
    ----------
    s
        Input string to sanitize.
    maxlen
        Maximum length of the output string.

    Returns
    -------
    str
        A filesystem-safe filename.
    """
    s = str(s)
    s = s.replace("γ", "gamma").replace("δ", "delta")
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s[:maxlen] if len(s) > maxlen else s


def intersect_preserve_order(items: Sequence[str], universe: Iterable[str]) -> list[str]:
    """Return items that appear in universe, preserving original order.

    Parameters
    ----------
    items
        List of items to intersect.
    universe
        List of items to intersect with.

    Returns
    -------
    list[str]
        A list of items that appear in universe, preserving original order.
    """
    u = set(universe)
    return [x for x in items if x in u]


def ensure_unique_index(df: pd.DataFrame, *, agg: str = "mean") -> pd.DataFrame:
    """If df.index has duplicates, aggregate duplicates and return a new df.
    Parameters
    ----------
    df
        DataFrame to ensure unique index.
    agg
        Aggregation method: "mean" or "sum" (extend later if needed).
    Returns
    -------
    pd.DataFrame
        A DataFrame with unique index.
    """
    if df.index.is_unique:
        return df
    if agg == "mean":
        return df.groupby(level=0).mean(numeric_only=True)
    if agg == "sum":
        return df.groupby(level=0).sum(numeric_only=True)
    raise ValueError(f"Unsupported agg='{agg}'. Use 'mean' or 'sum'.")


def looks_like_counts(X, sample: int = 10000, seed: int = 0) -> bool:
    """Check if matrix appears to be raw counts.

    Parameters
    ----------
    X
        Matrix to check.
    sample
        Number of samples to check.
    seed
        Random seed.

    Returns
    -------
    bool
        True if matrix appears to be raw counts, False otherwise.
    """
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
    """Return a raw-counts matrix and its source label, if available.

    Parameters
    ----------
    adata
        AnnData object.

    Returns
    -------
    tuple[np.ndarray | None, str | None]
        A tuple containing the counts matrix and its source label.
        - The counts matrix is the raw counts matrix.
        - The source label is the layer name where the counts matrix is stored.
        - If no counts matrix is found, returns (None, None).
    """

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


def warn_unbounded_bootstrap_intervals(
    res: pd.DataFrame, lo_col: str = "ci_lo_boot", hi_col: str = "ci_hi_boot"
) -> None:
    """Warn once when a results table holds unbounded bootstrap intervals."""
    if lo_col not in res.columns or hi_col not in res.columns:
        return
    unbounded = np.isinf(res[lo_col].to_numpy(dtype=float)) | np.isinf(
        res[hi_col].to_numpy(dtype=float)
    )
    if unbounded.any():
        warnings.warn(
            f"{int(unbounded.sum())} of {len(res)} bootstrap confidence intervals are unbounded: "
            "with this few participants the sign-flip bootstrap cannot reject at the 5% level.",
            UserWarning,
            stacklevel=3,
        )


_BOOT_COV_TYPES = ("cluster", "nonrobust")
_BOOT_CI_MAX_RADIUS_SE = 1e4


def _sum_by_cluster(values: np.ndarray, order: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Sum the rows of ``values`` within each cluster (rows of the result follow cluster order)."""
    return np.add.reduceat(values[order], starts, axis=0)


def wild_cluster_bootstrap_t(
    fit: RegressionResultsWrapper,
    X: np.ndarray,
    clusters: np.ndarray,
    term_name: str,
    B: int = 999,
    seed: int = 42,
    cov_type: str = "cluster",
    ci_level: float = 0.95,
) -> BootstrapResult:
    r"""Restricted wild cluster bootstrap (Rademacher) for one coefficient.

    Notes
    -----
    Tests :math:`H_0: \beta_j = \beta_0` with the restricted wild cluster bootstrap-t
    (WCR; Cameron, Gelbach and Miller 2008). For each hypothesised value the model is
    **refitted under the null**: the tested regressor is removed, the outcome
    :math:`y - \beta_0 x_j` is regressed on the remaining regressors, and the
    restricted fitted values and restricted residuals define the bootstrap
    data-generating process. Each draw multiplies the restricted residuals of every
    cluster by one Rademacher weight (±1 with equal probability), the unrestricted
    model is fitted to the bootstrap outcome, and the statistic
    :math:`t^* = (\hat\beta^*_j - \beta_0) / \mathrm{se}^*` is formed with the same
    variance estimator as the observed statistic. The model is refitted in closed
    form, so no regression is re-run per draw.

    When the clusters admit no more than ``B`` distinct sign assignments
    (:math:`2^G \le B`), all of them are enumerated instead of sampled, which makes
    the test exact and independent of ``seed``; the two-sided p-value for
    :math:`\beta_0 = 0` is then the fraction of assignments with
    :math:`|t^*| \ge |t|`. Otherwise ``B`` assignments are drawn and the p-value is
    :math:`(1 + \#\{|t^*| \ge |t|\}) / (1 + B)`. Neither depends on which level of a
    binary regressor is coded as 1.

    The confidence interval is obtained by **inverting the bootstrap test**: it is the
    set of values :math:`\beta_0` around the estimate whose bootstrap p-value is at
    least ``1 - ci_level``, found with the same bootstrap weights for every
    :math:`\beta_0`. The interval therefore excludes zero exactly when the p-value is
    below ``1 - ci_level``. The two assignments that keep or flip every cluster
    reproduce the observed statistic, so under enumeration the p-value cannot fall
    below :math:`2^{1-G}`. With five or fewer clusters (enumerated whenever
    ``B >= 32``) that floor exceeds 0.05, the test cannot reject at the 5% level, and
    the interval is unbounded (``-inf``, ``inf``).

    The bootstrap standard error is that of the **unrestricted** wild cluster
    bootstrap (unrestricted fitted values plus sign-flipped unrestricted residuals),
    because a standard error must not depend on the hypothesis being tested. With
    Rademacher weights it has a closed form, the square root of the sum of squared
    cluster scores, which is the cluster-robust standard error without a small-sample
    correction; it is returned exactly, not estimated from the draws.
    ``boot_distribution`` holds the unrestricted estimates for the same assignments.

    References
    ----------
    Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2008). Bootstrap-based improvements
    for inference with clustered errors. The Review of Economics and Statistics,
    90(3), 414–427.

    Parameters
    ----------
    fit
        Statsmodels OLS or WLS results for the unrestricted model.
    X
        Design matrix (``fit.model.exog``).
    clusters
        Array of cluster IDs, one per row of ``X``.
    term_name
        Name of the coefficient to test.
    B
        Number of bootstrap draws.
    seed
        Random seed.
    cov_type
        Variance estimator used to studentize the observed and the bootstrap
        statistics: ``"cluster"`` (cluster-robust, as statsmodels computes it) or
        ``"nonrobust"`` (homoskedastic; used when participant fixed effects leave two
        observations per cluster and the cluster-robust estimate is degenerate).
    ci_level
        Confidence level of the interval (default 0.95).

    Returns
    -------
    BootstrapResult
        Named tuple with ``p_boot``, ``se_boot``, ``ci_lo``, ``ci_hi``,
        and ``boot_distribution``.
    """
    if cov_type not in _BOOT_COV_TYPES:
        raise ValueError(f"cov_type must be one of {_BOOT_COV_TYPES}, got {cov_type!r}.")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be strictly between 0 and 1.")

    _nan_result = BootstrapResult(
        p_boot=np.nan,
        se_boot=np.nan,
        ci_lo=np.nan,
        ci_hi=np.nan,
        boot_distribution=np.array([], dtype=float),
    )

    coef_names = fit.model.exog_names
    if term_name not in coef_names:
        return _nan_result
    j = coef_names.index(term_name)

    y = np.asarray(fit.model.endog, dtype=float)
    X = np.asarray(X, dtype=float)
    n_obs, k_params = X.shape

    # WLS is OLS on the outcome and regressors scaled by the square root of the weights,
    # and a cluster-level sign flip commutes with that scaling.
    weights = getattr(fit.model, "weights", None)
    if weights is not None:
        root_w = np.sqrt(np.broadcast_to(np.asarray(weights, dtype=float), y.shape))
        y = y * root_w
        X = X * root_w[:, None]

    uniq_cl, cluster_idx = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = len(uniq_cl)
    df_resid = float(fit.df_resid)
    if n_clusters < 2 or df_resid <= 0:
        return _nan_result
    order = np.argsort(cluster_idx, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(cluster_idx[order]) != 0])

    pinv_x = np.linalg.pinv(X)
    a = pinv_x[j]
    beta_hat = float(a @ y)
    resid_u = y - X @ (pinv_x @ y)

    x_j = X[:, j]
    X_r = np.delete(X, j, axis=1)
    pinv_r = np.linalg.pinv(X_r)
    resid_r_y = y - X_r @ (pinv_r @ y)
    resid_r_x = x_j - X_r @ (pinv_r @ x_j)

    enumerate_all = n_clusters < 31 and (1 << n_clusters) <= B
    if enumerate_all:
        codes = np.arange(1 << n_clusters)
        flips = (codes[:, None] >> np.arange(n_clusters)) & 1
    else:
        flips = np.random.default_rng(seed).integers(0, 2, size=(B, n_clusters))
    signs = flips.astype(float) * 2.0 - 1.0
    # Keeping or flipping every cluster reproduces the observed statistic for every hypothesised
    # value, so those draws are ties by construction and are counted without a comparison.
    is_tie = np.all(signs == signs[:, :1], axis=1)
    n_ties = int(is_tie.sum())

    # A draw's outcome is the restricted fit plus sign-flipped restricted residuals, and the
    # restricted fit has no component along the tested regressor. The bootstrap coefficients,
    # residual sum of squares and cluster scores are therefore linear or quadratic forms in the
    # cluster signs, with per-cluster sums computed once.
    ax_by_cluster = _sum_by_cluster(X * a[:, None], order, starts)

    def _draw_terms(resid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coef = signs @ _sum_by_cluster(pinv_x.T * resid[:, None], order, starts)
        xty = signs @ _sum_by_cluster(X * resid[:, None], order, starts)
        score = signs * _sum_by_cluster(a * resid, order, starts)[None, :] - coef @ ax_by_cluster.T
        return coef, xty, score

    coef_y, xty_y, score_y = _draw_terms(resid_r_y)
    coef_x, xty_x, score_x = _draw_terms(resid_r_x)
    num_y, num_x = coef_y[:, j], coef_x[:, j]
    score_u = _sum_by_cluster(a * resid_u, order, starts)

    if cov_type == "nonrobust":
        var_scale = float((pinv_x @ pinv_x.T)[j, j]) / df_resid
        q_yy = float(resid_r_y @ resid_r_y) - np.einsum("bk,bk->b", coef_y, xty_y)
        q_xy = float(resid_r_y @ resid_r_x) - np.einsum("bk,bk->b", coef_y, xty_x)
        q_xx = float(resid_r_x @ resid_r_x) - np.einsum("bk,bk->b", coef_x, xty_x)
        var_obs = var_scale * float(resid_u @ resid_u)
    else:
        if n_obs <= k_params:
            return _nan_result
        var_scale = n_clusters / (n_clusters - 1.0) * (n_obs - 1.0) / (n_obs - k_params)
        q_yy = np.einsum("bg,bg->b", score_y, score_y)
        q_xy = np.einsum("bg,bg->b", score_y, score_x)
        q_xx = np.einsum("bg,bg->b", score_x, score_x)
        var_obs = var_scale * float(score_u @ score_u)

    if not np.isfinite(var_obs) or var_obs <= 0:
        return _nan_result
    se_obs = float(np.sqrt(var_obs))

    def _p_value(beta_null: float) -> float:
        var_star = var_scale * (q_yy - 2.0 * beta_null * q_xy + beta_null**2 * q_xx)
        valid = np.isfinite(var_star) & (var_star > 0) & ~is_tie
        n_valid = int(valid.sum()) + n_ties
        if n_valid == 0:
            return np.nan
        t_star = (num_y[valid] - beta_null * num_x[valid]) / np.sqrt(var_star[valid])
        n_extreme = np.count_nonzero(np.abs(t_star) >= abs(beta_hat - beta_null) / se_obs)
        if enumerate_all:
            return float((n_extreme + n_ties) / n_valid)
        return float((n_extreme + n_ties + 1) / (n_valid + 1))

    p_boot = _p_value(0.0)
    if not np.isfinite(p_boot):
        return _nan_result

    # 1 - 0.95 is 0.05000000000000004 in floating point; a p-value of exactly 0.05 must not
    # count as below the level.
    alpha = float(np.round(1.0 - ci_level, 12))

    def _bound(direction: float) -> float:
        accepted = 0.0
        rejected = np.nan
        distance = 0.0
        step = 0.25 * se_obs
        while distance <= _BOOT_CI_MAX_RADIUS_SE * se_obs:
            distance = distance + step if distance < 10.0 * se_obs else 2.0 * distance
            if _p_value(beta_hat + direction * distance) >= alpha:
                accepted = distance
            else:
                rejected = distance
                break
        if not np.isfinite(rejected):
            return direction * np.inf
        for _ in range(80):
            if rejected - accepted <= 1e-10 * se_obs:
                break
            mid = 0.5 * (accepted + rejected)
            if _p_value(beta_hat + direction * mid) >= alpha:
                accepted = mid
            else:
                rejected = mid
        return beta_hat + direction * accepted

    ci_lo, ci_hi = _bound(-1.0), _bound(1.0)

    return BootstrapResult(
        p_boot=p_boot,
        se_boot=float(np.sqrt(score_u @ score_u)),
        ci_lo=float(ci_lo),
        ci_hi=float(ci_hi),
        boot_distribution=beta_hat + signs @ score_u,
    )


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
