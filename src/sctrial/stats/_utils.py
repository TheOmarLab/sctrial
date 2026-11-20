from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

__all__ = [
    "aggregate_features",
    "apply_fdr",
    "drop_unused_levels",
    "encode_visit",
    "fit_identified",
    "standardize_series",
]


def _column_is_identified(exog: np.ndarray, j: int) -> bool:
    """Whether column ``j`` of a design is linearly independent of the other columns."""
    others = np.delete(exog, j, axis=1)
    unexplained = exog[:, j] - others @ np.linalg.lstsq(others, exog[:, j], rcond=None)[0]
    return bool(np.linalg.norm(unexplained) > 1e-8 * np.linalg.norm(exog[:, j]))


def fit_identified(model: Any, term: str, **fit_kwargs: Any) -> Any | None:
    """Fit a least-squares model whose design may hold redundant columns.

    A full-rank design is fitted as it is. Otherwise the fit returns ``None`` when the
    regressor ``term`` is a linear combination of the others: least squares would still
    report a number for it, split arbitrarily among the collinear regressors. When
    ``term`` is identified, the model is refitted on a maximal set of independent columns.
    The coefficients of identified regressors are the same, but statsmodels counts the
    columns of the design, not its rank, in the small-sample factor of a cluster-robust
    covariance, so a regressor that the others determine (a covariate that is constant
    within participant, beside participant effects) would enlarge every standard error.
    """
    fit = model.fit(**fit_kwargs)
    exog = np.asarray(fit.model.exog, dtype=float)
    names = list(fit.model.exog_names)
    rank = int(fit.model.rank)
    if rank == exog.shape[1]:
        return fit
    if term in names and not _column_is_identified(exog, names.index(term)):
        return None
    pivots = scipy.linalg.qr(exog, mode="r", pivoting=True)[1]
    keep = np.sort(pivots[:rank])
    rows = fit.model.data.row_labels
    design = pd.DataFrame(exog[:, keep], columns=[names[i] for i in keep], index=rows)
    outcome = pd.Series(np.asarray(fit.model.endog, dtype=float), index=rows, name=fit.model.endog_names)
    if isinstance(fit.model, sm.OLS):
        reduced = sm.OLS(outcome, design)
    else:
        reduced = sm.WLS(outcome, design, weights=np.asarray(fit.model.weights, dtype=float))
    return reduced.fit(**fit_kwargs)


def drop_unused_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the unused categories of every categorical column of a model frame.

    A formula term ``C(col)`` builds one indicator per category of a categorical
    column, including categories that no row of the frame carries. anndata keeps
    every category of an ``obs`` column through subsetting and ``groupby``, so a
    frame restricted to the participants in a model still lists all the others.
    Their empty indicators count as parameters, which inflates (or makes
    undefined) the small-sample correction of a cluster-robust covariance.

    Parameters
    ----------
    df
        Model frame. Modified in place and returned.
    """
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].cat.remove_unused_categories()
    return df


def standardize_series(
    df: pd.DataFrame,
    col: str,
    *,
    min_std: float = 1e-12,
) -> tuple[pd.Series, bool]:
    """Z-score a column and report if standardization is valid.

    Parameters
    ----------
    df
        DataFrame containing the column to standardize.
    col
        Name of the column to standardize.
    min_std
        Minimum standard deviation to consider the column for standardization.
    Returns
    -------
    tuple[pd.Series, bool]
        A tuple containing the standardized column and a boolean indicating if standardization is valid.
        - The standardized column is a Series with the same index as df.
        - The boolean indicates if standardization is valid.
        - If standardization is not valid, the standardized column is a Series with NaN values.
    """
    y = df[col].astype(float)
    y_std = y.std(ddof=1)
    if not np.isfinite(y_std) or y_std < min_std:
        return pd.Series(np.nan, index=df.index), False
    return (y - y.mean()) / y_std, True


def encode_visit(
    df: pd.DataFrame,
    visit_col: str,
    visits: tuple[str, str],
) -> pd.DataFrame:
    """Encode visit as ordered categorical and numeric 0/1.

    Parameters
    ----------
    df
        DataFrame containing the visit column.
    visit_col
        Name of the column containing visit labels.
    visits
        Tuple of (baseline, followup) visit labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded visit column.
    """
    out = df.copy()
    out[visit_col] = pd.Categorical(out[visit_col], categories=list(visits), ordered=True)
    out["visit_num"] = out[visit_col].map({visits[0]: 0, visits[1]: 1}).astype(float)
    return out


def aggregate_features(
    df: pd.DataFrame,
    grp_cols: list[str],
    features: Sequence[str],
    agg: str,
) -> pd.DataFrame:
    """Aggregate features within grp_cols using *agg*.

    Parameters
    ----------
    df
        Long-form DataFrame with one row per cell.
    grp_cols
        Columns to group by (e.g. subject, visit, arm).
    features
        Feature columns to aggregate.
    agg
        Aggregation method: ``'mean'``, ``'median'``, or ``'pct_pos'``.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame.

    Notes
    -----
    ``'mean'`` and ``'median'`` return values on the original feature scale.
    ``'pct_pos'`` returns the percentage (0–100) of cells with values > 0,
    *not* a proportion (0–1).
    """
    if agg == "mean":
        return df.groupby(grp_cols, observed=True)[list(features)].mean().reset_index()
    if agg == "median":
        return df.groupby(grp_cols, observed=True)[list(features)].median().reset_index()
    if agg == "pct_pos":
        out = (
            df.groupby(grp_cols, observed=True)[list(features)]
            .apply(lambda x: (x > 0).mean() * 100.0)
            .reset_index()
        )
        return out
    raise ValueError(f"Unsupported agg='{agg}'. Use 'mean', 'median', or 'pct_pos'.")


def apply_fdr(
    df: pd.DataFrame,
    p_col: str,
    fdr_col: str = "FDR",
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Apply FDR correction to a p-value column in a DataFrame.

    Parameters
    ----------
    df
        DataFrame containing the p-value column.
    p_col
        Name of the column containing p-values.
    fdr_col
        Name of the output column for FDR-corrected values.
    method
        Multiple testing correction method (default: 'fdr_bh' for
        Benjamini-Hochberg).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added FDR column.

    Examples
    --------
    >>> df = apply_fdr(results, p_col="p_DiD", fdr_col="FDR_DiD")
    """
    df = df.copy()
    mask = df[p_col].notna()
    df[fdr_col] = np.nan
    if mask.sum() > 0:
        df.loc[mask, fdr_col] = multipletests(df.loc[mask, p_col], method=method)[1]
    return df
