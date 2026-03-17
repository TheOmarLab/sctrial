from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from .design import TrialDesign

__all__ = ["subset_primary", "subset_cells", "profile_features"]

import logging

logger = logging.getLogger(__name__)


def _require_cols(obs: pd.DataFrame, cols: list[str]) -> None:
    """Require specific columns in the DataFrame."""
    missing = [c for c in cols if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}. Available: {list(obs.columns)}")


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Best-effort conversion of a metadata column to boolean.

    Handles bool, numeric (any non-zero → True), and string/categorical
    columns (recognises "true", "t", "yes", "y", "1" as truthy).

    Non-finite numeric values (NaN, inf) are treated as ``False``.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    # numeric: any non-zero finite value is truthy
    if pd.api.types.is_numeric_dtype(s):
        n_nonfinite = int(s.isna().sum() + np.isinf(s.fillna(0)).sum())
        if n_nonfinite > 0:
            logger.warning(
                "%d non-finite value(s) (NaN/inf) in boolean column '%s'; treating as False.",
                n_nonfinite,
                s.name,
            )
        filled = s.fillna(0)
        finite_mask = np.isfinite(filled)
        return (finite_mask & (filled != 0)).astype(bool)

    # strings / categoricals
    ss = s.astype(str).str.strip().str.lower()
    true_vals = {"1", "true", "t", "yes", "y"}
    out = ss.isin(true_vals)
    return out.fillna(False).astype(bool)


def subset_primary(
    adata: AnnData,
    design: TrialDesign,
    visits: tuple[str, str],
    exclude_crossovers: bool = True,
) -> AnnData:
    """Subset AnnData to the primary (baseline, followup) visits.

    Parameters
    ----------
    visits:
        Tuple of (baseline_visit, followup_visit), e.g. ("3/T0", "6/T12w").
    exclude_crossovers:
        If True and design.crossover_col is provided, drop rows where crossover_col is truthy.

    Returns
    -------
    AnnData
        Subsetted AnnData object.
    """
    obs = adata.obs
    _require_cols(obs, [design.visit_col])

    mask = obs[design.visit_col].isin(list(visits)).to_numpy(dtype=bool)

    if exclude_crossovers and design.crossover_col:
        _require_cols(obs, [design.crossover_col])
        cross = _to_bool_series(obs[design.crossover_col]).to_numpy(dtype=bool)
        mask &= ~cross

    return adata[mask].copy()


def subset_cells(
    adata: AnnData,
    design: TrialDesign,
    arm: str | None = None,
    visit: str | None = None,
    celltype: str | None = None,
    exclude_crossovers: bool = False,
) -> AnnData:
    """General-purpose subsetting helper by arm/visit/celltype (+ optional crossover exclusion).

    Parameters
    ----------
    adata
        AnnData object.
    design
        TrialDesign object.
    arm
        Arm to subset by.
    visit
        Visit to subset by.
    celltype
        Celltype to subset by.
    exclude_crossovers
        If True, exclude crossovers.

    Returns
    -------
    AnnData
        Subsetted AnnData object.

    """
    obs = adata.obs

    required = []
    if arm is not None and design.arm_col is not None:
        required.append(design.arm_col)
    if visit is not None:
        required.append(design.visit_col)
    if celltype is not None:
        if not design.celltype_col:
            raise ValueError("celltype_col must be set in TrialDesign when filtering by celltype.")
        required.append(design.celltype_col)
    if exclude_crossovers and design.crossover_col:
        required.append(design.crossover_col)

    if required:
        _require_cols(obs, required)

    mask = np.ones(obs.shape[0], dtype=bool)

    if arm is not None and design.arm_col is not None:
        mask &= obs[design.arm_col].to_numpy() == arm

    if visit is not None:
        mask &= obs[design.visit_col].to_numpy() == visit

    if celltype is not None:
        mask &= obs[design.celltype_col].to_numpy() == celltype

    if exclude_crossovers and design.crossover_col:
        cross = _to_bool_series(obs[design.crossover_col]).to_numpy(dtype=bool)
        mask &= ~cross

    return adata[mask].copy()


def profile_features(
    adata: AnnData,
    features: Sequence[str],
    groupby: str,
    layer: str | None = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Calculate aggregate expression of features across groups.

    Useful for profiling marker sets across clusters or trial arms.

    Parameters
    ----------
    adata
        AnnData object.
    features
        Genes or obs columns to aggregate.
    groupby
        Column in `adata.obs` to group by.
    layer
        Expression layer to use for genes.
    agg
        Aggregation function ('mean', 'median', etc. supported by pandas).

    Returns
    -------
    pd.DataFrame
        Table with index `groupby` and columns `features`, containing
        aggregated feature values.

    Notes
    -----
    Aggregation is performed at the **cell level**.  When grouping by
    treatment arm or participant, groups with more cells will dominate
    the aggregate.  For participant-level or balanced comparisons, use
    :func:`~sctrial.stats.pseudobulk.pseudobulk_expression` or
    pre-aggregate to pseudobulk before calling this function.
    """
    from .stats._extract import extract_gene_vector

    _require_cols(adata.obs, [groupby])

    res = {}
    for feat in features:
        if feat in adata.obs.columns:
            res[feat] = adata.obs[feat].values
        elif feat in adata.var_names:
            res[feat] = extract_gene_vector(adata, feat, layer=layer)
        else:
            raise KeyError(f"Feature '{feat}' not found in obs or var_names.")

    df = pd.DataFrame(res, index=adata.obs_names)
    df[groupby] = adata.obs[groupby].values

    return df.groupby(groupby, observed=True).agg(agg)
