from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

__all__ = [
    "aggregate_features",
    "apply_fdr",
    "encode_visit",
    "standardize_series",
]


def standardize_series(
    df: pd.DataFrame,
    col: str,
    *,
    min_std: float = 1e-12,
) -> tuple[pd.Series, bool]:
    """Z-score a column and report if standardization is valid."""
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
    """Encode visit as ordered categorical and numeric 0/1."""
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
