from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "aggregate_features",
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
    """Aggregate features within grp_cols using agg."""
    if agg == "mean":
        return df.groupby(grp_cols, observed=True).mean(numeric_only=True).reset_index()
    if agg == "median":
        return df.groupby(grp_cols, observed=True).median(numeric_only=True).reset_index()
    if agg == "pct_pos":
        out = (
            df.groupby(grp_cols, observed=True)[list(features)]
            .apply(lambda x: (x > 0).mean() * 100.0)
            .reset_index()
        )
        return out
    raise ValueError(f"Unsupported agg='{agg}'. Use 'mean', 'median', or 'pct_pos'.")
