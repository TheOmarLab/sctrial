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
        Aggregation method: "mean" or "sum".
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
    """Warn once when a results table holds unbounded bootstrap intervals.

    Parameters
    ----------
    res
        Results table. Nothing happens when it lacks either column.
    lo_col, hi_col
        Columns with the lower and upper bounds of the bootstrap interval.
    """
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
# Hypothesised values further than this many standard errors from the estimate are not examined:
# an interval that reaches that far is reported as unbounded.
_BOOT_CI_MAX_RADIUS_SE = 1e4
# Two statistics that are equal in exact arithmetic can differ in the last bits once computed
# along different paths. A bootstrap t statistic within this relative margin of the observed
# one is a tie; the margin is far above rounding error and far below any difference between
# two statistics that are not equal.
_TIE_RTOL = 1e-9
# A bootstrap variance below this fraction of the largest variance among the draws is too small
# for the closed form, which obtains it as a difference of two larger numbers. It is computed
# again from the residuals of the draw, and is zero when those do not exceed the rounding error
# of the outcome they are residuals of.
_SMALL_VARIANCE_RTOL = 1e-12


def _tie_tolerance(centred: np.ndarray) -> float:
    """Bound on the rounding error of a difference of two means of ``centred`` values.

    Summing n values of magnitude at most m in two different orders gives results that differ
    by at most about n * eps * m. The values must be centred: the difference of two means does
    not depend on the origin of the measurements, and neither may the bound.
    """
    scale = float(np.max(np.abs(centred), initial=0.0))
    return 8.0 * centred.size * float(np.finfo(float).eps) * scale


def _sum_by_cluster(values: np.ndarray, order: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Sum the rows of ``values`` within each cluster (rows of the result follow cluster order)."""
    return np.add.reduceat(values[order], starts, axis=0)


def _real_roots_by_row(coefs: np.ndarray) -> np.ndarray:
    """Real roots of the polynomial in each row of ``coefs`` (highest degree first).

    Returns one row per polynomial, padded with NaN where a root is not real or the row holds a
    non-finite coefficient.
    """
    n_rows, degree = coefs.shape[0], coefs.shape[1] - 1
    roots = np.full((n_rows, degree), np.nan + 0.0j)
    finite = np.all(np.isfinite(coefs), axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        first_row = -coefs[:, 1:] / coefs[:, :1]
    regular = finite & np.all(np.isfinite(first_row), axis=1)
    companion = np.zeros((int(regular.sum()), degree, degree))
    companion[:, 0, :] = first_row[regular]
    companion[:, np.arange(1, degree), np.arange(degree - 1)] = 1.0
    roots[regular] = np.linalg.eigvals(companion)
    # A vanishing leading coefficient lowers the degree; np.roots drops leading zeros.
    for i in np.flatnonzero(finite & ~regular):
        row = coefs[i].copy()
        row[0] = 0.0
        found = np.roots(row)
        roots[i, : found.size] = found
    # A statistic that touches the observed one without crossing it gives a double root, which
    # rounding turns into a conjugate pair with a small imaginary part.
    real = np.abs(roots.imag) <= 1e-6 * np.maximum(1.0, np.abs(roots.real))
    return np.where(real, roots.real, np.nan)


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

    A bootstrap statistic within ``1e-9`` of the observed one (relative to the observed
    statistic when that exceeds one) counts as a tie, so assignments that reproduce
    the observed statistic in exact arithmetic (for example a flipped cluster whose
    outcome did not change) are counted whatever the rounding of the two computations.

    An assignment that the model fits exactly has no residual variance and an
    unbounded statistic, and counts as at least as extreme as the observed one. Its
    residuals are zero within the rounding error of the outcome (``8 * n * eps`` times
    the largest absolute outcome, measured from the mean when the other regressors
    hold a constant). An assignment with a small positive variance is not such a case:
    its statistic is computed from its own residuals and compared as any other. When
    the model fits the observed outcome exactly by the same criterion, there is no
    statistic to compare with and every result is NaN.

    The confidence interval is obtained by **inverting the bootstrap test** with the
    same bootstrap weights for every :math:`\beta_0`. The bootstrap p-value is a step
    function of :math:`\beta_0` that changes only where a draw's statistic equals the
    observed one, which is at the real roots of one quartic per draw. The interval
    runs from the smallest to the largest :math:`\beta_0` whose p-value is at least
    ``1 - ci_level``, so it contains every value the test does not reject, and it
    contains zero whenever the p-value is at least ``1 - ci_level``. In small or
    unbalanced designs the values that are not rejected can form more than one
    interval; the reported interval then spans them, includes the rejected values in
    between, and can contain zero although the p-value is below ``1 - ci_level``. The
    two assignments that keep or flip every cluster reproduce the observed statistic,
    so under enumeration the p-value cannot fall below :math:`2^{1-G}`. With five or
    fewer clusters (enumerated whenever ``B >= 32``) that floor exceeds 0.05, the
    test cannot reject at the 5% level, and the interval is unbounded (``-inf``,
    ``inf``).

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
    rounding = 8.0 * n_obs * float(np.finfo(float).eps)

    # When the other regressors hold a constant, the estimate and every residual are the same
    # wherever the origin of the outcome lies. The outcome is then measured from its mean, which
    # keeps rounding error at the scale of its spread: far from zero, two statistics that are
    # equal in exact arithmetic would differ by more than the tie margin. An outcome whose mean
    # is zero within rounding is left as it is, and one whose spread is zero within rounding
    # does not vary.
    others = np.delete(X, j, axis=1)
    if others.shape[1]:
        ones = np.ones(n_obs)
        beyond = ones - others @ np.linalg.lstsq(others, ones, rcond=None)[0]
        if np.linalg.norm(beyond) <= 1e-8 * np.sqrt(n_obs):
            mean = float(y.mean())
            centred = y - mean
            spread = float(np.max(np.abs(centred), initial=0.0))
            if spread <= rounding * abs(mean):
                y = np.zeros(n_obs)
            elif abs(mean) > rounding * spread:
                y = centred

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

    def _residual_of(values: np.ndarray) -> np.ndarray:
        """Residual of the rows of ``values`` on the regressors, projected twice."""
        once = values - (values @ pinv_x.T) @ X.T
        return once - (once @ pinv_x.T) @ X.T

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

    # A model that fits the outcome exactly leaves no variance to studentize with. Rounding
    # leaves its residuals a hair away from zero, so they are compared with the rounding error
    # of the outcome, as those of a draw are below. Projecting the residuals once more removes
    # the error that a badly scaled design adds to them (it lies along the regressors).
    refined_u = _residual_of(resid_u)
    observed_parts = refined_u if cov_type == "nonrobust" else _sum_by_cluster(a * refined_u, order, starts)
    observed_floor = rounding * float(np.max(np.abs(y), initial=0.0))
    if cov_type == "cluster":
        observed_floor *= float(np.max(np.abs(a)))
    if not np.isfinite(var_obs) or np.max(np.abs(observed_parts), initial=0.0) <= observed_floor:
        return _nan_result
    se_obs = float(np.sqrt(var_obs))

    free = ~is_tie
    num_y_f, num_x_f = num_y[free], num_x[free]
    q_yy_f, q_xy_f, q_xx_f = q_yy[free], q_xy[free], q_xx[free]
    signs_f = signs[free]
    # The restricted residuals, projected once more on the restricted regressors: only the
    # draws recomputed below use them, so every other number is that of the closed form.
    refined_r_y = resid_r_y - X_r @ (pinv_r @ resid_r_y)
    refined_r_x = resid_r_x - X_r @ (pinv_r @ resid_r_x)

    def _variance_from_residuals(
        beta_null: float, draws: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Variance of the given draws from their own residuals, and whether it is zero."""
        restricted = refined_r_y - beta_null * refined_r_x
        resid = _residual_of(signs_f[draws][:, cluster_idx] * restricted)
        floor = rounding * float(np.max(np.abs(y - beta_null * x_j), initial=0.0))
        if cov_type == "nonrobust":
            parts = resid
        else:
            parts = np.add.reduceat((resid * a)[:, order], starts, axis=1)
            floor *= float(np.max(np.abs(a)))
        variance = var_scale * np.einsum("bi,bi->b", parts, parts)
        return variance, np.max(np.abs(parts), axis=1, initial=0.0) <= floor

    def _p_values(beta_nulls: np.ndarray) -> np.ndarray:
        b = np.asarray(beta_nulls, dtype=float)[:, None]
        var_star = var_scale * (q_yy_f - 2.0 * b * q_xy_f + b**2 * q_xx_f)
        valid = np.isfinite(var_star)
        # A draw that the model fits exactly has no residual variance and an unbounded
        # statistic, so it is at least as extreme as any observed one. The closed form leaves
        # such a variance a hair above or below zero, and gives a small positive variance no
        # better: both are computed again from the residuals of the draw, where a draw without
        # variance is told from one with little.
        largest = np.max(np.where(valid, var_star, 0.0), axis=1, keepdims=True, initial=var_obs)
        small = valid & (var_star <= _SMALL_VARIANCE_RTOL * largest)
        unbounded = np.zeros_like(small)
        for row in np.flatnonzero(small.any(axis=1)):
            draws = np.flatnonzero(small[row])
            var_star[row, draws], unbounded[row, draws] = _variance_from_residuals(
                float(b[row, 0]), draws
            )
        regular = valid & ~unbounded
        t_star = (num_y_f - b * num_x_f) / np.sqrt(np.where(regular, var_star, 1.0))
        t_obs = np.abs(beta_hat - b) / se_obs
        extreme = unbounded | (
            regular & (np.abs(t_star) >= t_obs - _TIE_RTOL * np.maximum(1.0, t_obs))
        )
        n_extreme = np.count_nonzero(extreme, axis=1) + n_ties
        n_valid = np.count_nonzero(valid, axis=1) + n_ties
        if enumerate_all:
            p = n_extreme / np.maximum(n_valid, 1)
        else:
            p = (n_extreme + 1) / (n_valid + 1)
        return np.where(n_valid > 0, p, np.nan)

    def _p_value(beta_null: float) -> float:
        return float(_p_values(np.array([beta_null]))[0])

    p_boot = _p_value(0.0)
    if not np.isfinite(p_boot):
        return _nan_result

    # 1 - 0.95 is 0.05000000000000004 in floating point; a p-value of exactly 0.05 must not
    # count as below the level.
    alpha = float(np.round(1.0 - ci_level, 12))

    # With s = (beta_null - beta_hat) / se_obs, a draw's statistic equals the observed one where
    # (a0 - s * b1)**2 = s**2 * (c0 + c1 * s + c2 * s**2). The p-value is constant between
    # consecutive real roots, so one value per gap decides the whole gap.
    a0 = num_y_f - beta_hat * num_x_f
    b1 = se_obs * num_x_f
    c0 = var_scale * (q_yy_f - 2.0 * beta_hat * q_xy_f + beta_hat**2 * q_xx_f)
    c1 = 2.0 * var_scale * se_obs * (beta_hat * q_xx_f - q_xy_f)
    c2 = var_scale * se_obs**2 * q_xx_f
    crossings = _real_roots_by_row(np.stack([c2, c1, c0 - b1**2, 2.0 * a0 * b1, -(a0**2)], axis=1))
    # A draw whose variance is positive at every hypothesised value is no longer as extreme as
    # the observed statistic beyond its outermost crossing, because the observed statistic grows
    # without bound and the draw's does not. The other draws are allowed anything.
    with np.errstate(invalid="ignore"):
        settled = (c2 > 0) & (c1**2 < (1.0 - 1e-9) * 4.0 * c0 * c2)
    n_settled = int(settled.sum())
    monte_carlo = 0 if enumerate_all else 1
    # Number of settled draws that must be extreme for a p-value of at least alpha.
    n_needed = int(
        np.ceil(
            alpha * (n_settled + n_ties + monte_carlo)
            - (len(settled) - n_settled)
            - n_ties
            - monte_carlo
            - 1e-9
        )
    )

    def _bound(direction: float) -> float:
        signed = direction * crossings
        positive = np.isfinite(signed) & (signed > 0)
        roots = np.unique(signed[positive])[::-1]
        remote = roots[roots > _BOOT_CI_MAX_RADIUS_SE]
        roots = roots[roots <= _BOOT_CI_MAX_RADIUS_SE]
        if roots.size == 0:
            return direction * np.inf
        # From beyond the outermost root inward: each root, where a tie can raise the p-value
        # at that single value, followed by the gap between it and the next root (or the estimate).
        points = np.empty(2 * roots.size + 1)
        points[0] = 2.0 * roots[0] + 1.0
        if remote.size:
            points[0] = min(points[0], 0.5 * (roots[0] + remote[-1]))
        points[1::2] = roots
        points[2::2] = 0.5 * (roots + np.append(roots[1:], 0.0))

        # Beyond the n_needed-th largest outermost crossing too few draws can be extreme, so
        # every point out there is rejected without being evaluated. The margin covers the
        # stretch in which a draw still counts as tied.
        start = 0
        outermost = np.where(positive, signed, 0.0).max(axis=1)[settled]
        if 0 < n_needed <= np.count_nonzero(outermost > 0):
            limit = np.partition(outermost, -n_needed)[-n_needed]
            start = int(np.searchsorted(-points, -(limit * (1.0 + 1e-3) + 1e-3)))

        inner, outer = 0.0, float(points[-1])
        size = 32
        while start < points.size:
            chunk = points[start : start + size]
            accepted = _p_values(beta_hat + direction * se_obs * chunk) >= alpha
            if accepted.any():
                first = start + int(np.argmax(accepted))
                if first == 0:
                    return direction * np.inf
                inner, outer = float(points[first]), float(points[first - 1])
                break
            start, size = start + size, min(2 * size, 256)
        # ``inner`` is accepted and ``outer`` rejected; close in on the last accepted value.
        while True:
            mid = 0.5 * (inner + outer)
            if not inner < mid < outer:
                break
            if _p_value(beta_hat + direction * se_obs * mid) >= alpha:
                inner = mid
            else:
                outer = mid
        return beta_hat + direction * se_obs * inner

    ci_lo, ci_hi = _bound(-1.0), _bound(1.0)
    if p_boot >= alpha:
        ci_lo, ci_hi = min(ci_lo, 0.0), max(ci_hi, 0.0)

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
    n1 = len(group1)
    # The difference of two means does not depend on the origin of the measurements, so the
    # pooled values are centred: the contrasts keep their precision whatever the origin.
    combined = np.concatenate([np.asarray(group1, dtype=float), np.asarray(group2, dtype=float)])
    combined = combined - np.mean(combined)
    obs_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
    # A permutation that reproduces the observed split sums the same values in another order.
    tol = _tie_tolerance(combined)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        p_diff = np.mean(perm[:n1]) - np.mean(perm[n1:])
        if abs(p_diff) >= abs(obs_diff) - tol:
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
    diff = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
    obs_mean = np.mean(diff)
    # The paired differences are free of the origin already; a sign flip keeps their size.
    tol = _tie_tolerance(diff)

    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        p_mean = np.mean(diff * signs)
        if abs(p_mean) >= abs(obs_mean) - tol:
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
