from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


__all__ = ["pseudobulk_expression", "pseudobulk_within_arm"]


def _get_layer(adata: AnnData, layer: Optional[str]) -> np.ndarray:
    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        X = X.tocsr()
    return X


def pseudobulk_expression(
    adata: AnnData,
    genes: Sequence[str],
    groupby: Sequence[str],
    counts_layer: Optional[str] = "counts",
    scale: float = 1e6,
    log1p: bool = True,
) -> pd.DataFrame:
    """Compute pseudobulk log1p-CPM for a gene panel per group.

    Parameters
    ----------
    adata
        AnnData object.
    genes
        List of genes to summarize.
    groupby
        Columns in `adata.obs` used for grouping (e.g., participant, visit, cell type).
    counts_layer
        Layer containing raw counts. Defaults to "counts".
    scale
        CPM scale factor (default 1e6).
    log1p
        If True, apply log1p to CPM.
    """
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        return pd.DataFrame()

    X = _get_layer(adata, counts_layer)
    if sp.issparse(X):
        X = X.toarray()

    gene_idx = [int(adata.var_names.get_loc(g)) for g in genes]
    X_panel = X[:, gene_idx]

    df_expr = pd.DataFrame(X_panel, columns=genes, index=adata.obs_names)
    df_meta = adata.obs[list(groupby)].copy()
    df_meta["total_counts"] = X.sum(axis=1)
    df = df_meta.join(df_expr, how="left")

    df_sum = (
        df.groupby(list(groupby), observed=True)
        .sum(numeric_only=True)
        .reset_index()
    )

    totals = df_sum["total_counts"].values.reshape(-1, 1)
    cpm = df_sum[genes].values / (totals + 1e-12) * scale
    if log1p:
        cpm = np.log1p(cpm)
    df_sum[genes] = cpm
    return df_sum


def pseudobulk_within_arm(
    adata: AnnData,
    genes: Sequence[str],
    participant_col: str,
    visit_col: str,
    visits: Sequence[str],
    celltype_col: str,
    counts_layer: Optional[str] = "counts",
    min_paired: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute within-arm pseudobulk deltas and Wilcoxon tests.

    Returns (summary_df, delta_long_df).
    """
    pb = pseudobulk_expression(
        adata,
        genes=genes,
        groupby=[participant_col, visit_col, celltype_col],
        counts_layer=counts_layer,
        log1p=True,
    )
    if pb.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    deltas = []
    for ct in pb[celltype_col].unique():
        sub = pb[pb[celltype_col] == ct].copy()
        wide = sub.pivot_table(index=participant_col, columns=visit_col, aggfunc="size", fill_value=0, observed=True)
        keep = wide[(wide.get(visits[0], 0) > 0) & (wide.get(visits[1], 0) > 0)].index
        sub = sub[sub[participant_col].isin(keep)].copy()
        if sub[participant_col].nunique() < min_paired:
            continue
        for g in genes:
            w = sub.pivot_table(index=participant_col, columns=visit_col, values=g, aggfunc="mean")
            if visits[0] not in w.columns or visits[1] not in w.columns:
                continue
            delta = (w[visits[1]] - w[visits[0]]).dropna()
            for pid, dv in delta.items():
                deltas.append({
                    "celltype": ct,
                    "feature": g,
                    "participant_id": pid,
                    "delta": float(dv),
                })
            if len(delta) < min_paired:
                p_val = np.nan
            else:
                try:
                    _, p_val = wilcoxon(delta.values)
                except Exception:
                    p_val = np.nan
            rows.append({
                "celltype": ct,
                "feature": g,
                "n_units": int(len(delta)),
                "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                "median_delta": float(delta.median()) if len(delta) else np.nan,
                "p_time": float(p_val),
            })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        mask = summary["p_time"].notna()
        summary["FDR_time"] = np.nan
        if mask.sum() > 0:
            summary.loc[mask, "FDR_time"] = multipletests(summary.loc[mask, "p_time"], method="fdr_bh")[1]
    delta_long = pd.DataFrame(deltas)
    return summary, delta_long
