from __future__ import annotations

from typing import Optional, Tuple, Sequence, List
import numpy as np
import pandas as pd
from anndata import AnnData
from .design import TrialDesign


def _require_cols(obs: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}. Available: {list(obs.columns)}")


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Best-effort conversion of a metadata column to boolean."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    # numeric 0/1
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)

    # strings / categoricals
    ss = s.astype(str).str.strip().str.lower()
    true_vals = {"1", "true", "t", "yes", "y"}
    false_vals = {"0", "false", "f", "no", "n", "nan", "none"}
    out = ss.map(lambda x: True if x in true_vals else (False if x in false_vals else False))
    return out.fillna(False).astype(bool)


def subset_primary(
        adata: AnnData,
        design: TrialDesign,
        visits: Tuple[str, str],
        exclude_crossovers: bool = True,
) -> AnnData:
    """Subset AnnData to the primary (baseline, followup) visits.

    Parameters
    ----------
    visits:
        Tuple of (baseline_visit, followup_visit), e.g. ("3/T0", "6/T12w").
    exclude_crossovers:
        If True and design.crossover_col is provided, drop rows where crossover_col is truthy.
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
        arm: Optional[str] = None,
        visit: Optional[str] = None,
        celltype: Optional[str] = None,
        exclude_crossovers: bool = False,
) -> AnnData:
    """General-purpose subsetting helper by arm/visit/celltype (+ optional crossover exclusion)."""
    obs = adata.obs

    required = []
    if arm is not None:
        required.append(design.arm_col)
    if visit is not None:
        required.append(design.visit_col)
    if celltype is not None and design.celltype_col:
        required.append(design.celltype_col)
    if exclude_crossovers and design.crossover_col:
        required.append(design.crossover_col)

    if required:
        _require_cols(obs, required)

    mask = np.ones(obs.shape[0], dtype=bool)

    if arm is not None:
        mask &= (obs[design.arm_col].to_numpy() == arm)

    if visit is not None:
        mask &= (obs[design.visit_col].to_numpy() == visit)

    if celltype is not None and design.celltype_col:
        mask &= (obs[design.celltype_col].to_numpy() == celltype)

    if exclude_crossovers and design.crossover_col:
        cross = _to_bool_series(obs[design.crossover_col]).to_numpy(dtype=bool)
        mask &= ~cross

    return adata[mask].copy()


def profile_features(
    adata: AnnData,
    features: Sequence[str],
    groupby: str,
    layer: Optional[str] = None,
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
        Table with index `groupby` and columns `features`.
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