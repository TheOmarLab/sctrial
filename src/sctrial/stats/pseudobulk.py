from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sctrial.design import TrialDesign
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from ._utils import encode_visit
from .did import did_fit

__all__ = ["pseudobulk_expression", "pseudobulk_within_arm", "pseudobulk_did"]


def _get_layer(adata: AnnData, layer: str | None):
    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        X = X.tocsr()
    return X


def pseudobulk_expression(
    adata: AnnData,
    genes: Sequence[str],
    groupby: Sequence[str],
    counts_layer: str | None = "counts",
    scale: float = 1e6,
    log1p: bool = True,
    include_n_cells: bool = True,
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
    gene_idx = [int(adata.var_names.get_loc(g)) for g in genes]

    if sp.issparse(X):
        X_panel = X[:, gene_idx].toarray()
        total_counts = np.asarray(X.sum(axis=1)).ravel()
    else:
        X_panel = np.asarray(X[:, gene_idx])
        total_counts = np.asarray(X.sum(axis=1)).ravel()

    df_expr = pd.DataFrame(X_panel, columns=genes, index=adata.obs_names)
    df_meta = adata.obs[list(groupby)].copy()
    df_meta["total_counts"] = total_counts
    df = df_meta.join(df_expr, how="left")

    df_sum = (
        df.groupby(list(groupby), observed=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    if include_n_cells:
        df_sum["n_cells"] = df.groupby(list(groupby), observed=True).size().values

    totals = df_sum["total_counts"].values.reshape(-1, 1)
    cpm = df_sum[genes].values / (totals + 1e-12) * scale
    if log1p:
        cpm = np.log1p(cpm)
    df_sum[genes] = cpm
    return df_sum


def pseudobulk_did(
    adata: AnnData,
    genes: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    celltype_col: str | None = None,
    counts_layer: str | None = "counts",
    min_cells_per_group: int = 5,
    min_paired: int = 4,
    use_bootstrap: bool = False,
    n_boot: int = 999,
    seed: int = 42,
) -> pd.DataFrame:
    """Run DiD on pseudobulk expression (participant-level aggregates).

    This mirrors subject-level pseudobulk DiD workflows where each participant×visit
    (optionally per cell type) is one observation.
    """
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        return pd.DataFrame()

    groupby = [design.participant_col, design.visit_col, design.arm_col]
    if celltype_col is not None:
        groupby.append(celltype_col)

    pb = pseudobulk_expression(
        adata,
        genes=genes,
        groupby=groupby,
        counts_layer=counts_layer,
        log1p=True,
        include_n_cells=True,
    )
    if pb.empty:
        return pd.DataFrame()

    if "n_cells" in pb.columns:
        pb = pb[pb["n_cells"] >= min_cells_per_group].copy()

    pb = pb[pb[design.visit_col].isin(visits)].copy()
    pb["arm_bin"] = (pb[design.arm_col] == design.arm_treated).astype(int)
    pb = encode_visit(pb, design.visit_col, visits)

    rows = []
    if celltype_col is None:
        pools = [None]
    else:
        pools = sorted(pb[celltype_col].dropna().unique())

    for pool in pools:
        if pool is not None:
            df_pool = pb[pb[celltype_col] == pool].copy()
        else:
            df_pool = pb.copy()

        # paired participants
        wide = df_pool.pivot_table(
            index=design.participant_col,
            columns=design.visit_col,
            values=genes[0],
            aggfunc="size",
        )
        keep = wide[(wide.get(visits[0], 0) > 0) & (wide.get(visits[1], 0) > 0)].index
        df_pool = df_pool[df_pool[design.participant_col].isin(keep)].copy()

        if df_pool[design.participant_col].nunique() < min_paired:
            continue

        for g in genes:
            out = did_fit(
                df_pool,
                y=g,
                unit=design.participant_col,
                time="visit_num",
                arm_bin="arm_bin",
                covariates=None,
                standardize=True,
                use_bootstrap=use_bootstrap,
                n_boot=n_boot,
                seed=seed,
            )
            out["feature"] = g
            out["n_units"] = int(df_pool[design.participant_col].nunique())
            if pool is not None:
                out["celltype"] = pool
            rows.append(out)

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    mask = res["p_DiD"].notna()
    res["FDR_DiD"] = np.nan
    if mask.sum() > 0:
        res.loc[mask, "FDR_DiD"] = multipletests(res.loc[mask, "p_DiD"], method="fdr_bh")[1]

    if celltype_col is not None:
        res["FDR_DiD_celltype"] = np.nan
        for ct, sub in res.groupby("celltype", observed=True):
            m = sub["p_DiD"].notna()
            if m.sum() > 0:
                res.loc[sub.index[m], "FDR_DiD_celltype"] = multipletests(
                    sub.loc[m, "p_DiD"], method="fdr_bh"
                )[1]

    return res.reset_index(drop=True)


def pseudobulk_within_arm(
    adata: AnnData,
    genes: Sequence[str],
    participant_col: str,
    visit_col: str,
    visits: Sequence[str],
    celltype_col: str,
    counts_layer: str | None = "counts",
    min_paired: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                except (ValueError, TypeError):
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
