from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from ..design import TrialDesign
from ._utils import apply_fdr, encode_visit
from .did import did_fit

logger = logging.getLogger(__name__)

__all__ = ["pseudobulk_expression", "pseudobulk_within_arm", "pseudobulk_did", "pseudobulk_export"]


def _get_layer(adata: AnnData, layer: str | None) -> np.ndarray | sp.csr_matrix:
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
    min_cells_per_group: int = 1,
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

    # Group encoding (vectorized aggregation)
    group_df = adata.obs[list(groupby)].copy()

    if min_cells_per_group > 1:
        counts = group_df.value_counts().rename("n_cells")
        keep_groups = counts[counts >= min_cells_per_group].index
        # Use isin mask instead of merge to preserve the original cell index
        group_mi = pd.MultiIndex.from_frame(group_df[list(groupby)])
        mask = group_mi.isin(keep_groups)
        group_df = group_df[mask]
        adata = adata[group_df.index].copy()
        if group_df.empty:
            return pd.DataFrame()

    # Extract expression matrix AFTER any subsetting so dimensions match
    X = _get_layer(adata, counts_layer)
    gene_idx = [int(adata.var_names.get_loc(g)) for g in genes]

    group_index = pd.MultiIndex.from_frame(group_df)
    group_index.names = list(groupby)
    group_codes, groups = pd.factorize(group_index)
    n_groups = len(groups)

    # Total counts per cell
    if sp.issparse(X):
        X_sparse = cast(sp.csr_matrix, X)
        X_panel = X_sparse[:, gene_idx].tocsr()
        total_counts = np.asarray(X.sum(axis=1)).ravel()

        # Sparse group aggregation: G @ X_panel
        rows = group_codes
        cols = np.arange(X_panel.shape[0], dtype=int)
        data = np.ones(X_panel.shape[0], dtype=float)
        G = sp.csr_matrix((data, (rows, cols)), shape=(n_groups, X_panel.shape[0]))

        sums = (G @ X_panel).toarray()
        totals = np.asarray(G @ total_counts).ravel()
        n_cells = np.asarray(G @ np.ones(X_panel.shape[0], dtype=float)).ravel()
    else:
        X_panel = np.asarray(X[:, gene_idx])
        total_counts = np.asarray(X.sum(axis=1)).ravel()

        sums = np.zeros((n_groups, len(genes)), dtype=float)
        totals = np.zeros(n_groups, dtype=float)
        n_cells = np.zeros(n_groups, dtype=float)
        np.add.at(sums, group_codes, X_panel)
        np.add.at(totals, group_codes, total_counts)
        np.add.at(n_cells, group_codes, 1.0)

    # Build group summary dataframe
    df_sum = groups.to_frame(index=False)
    df_sum.columns = list(groupby)
    for i, g in enumerate(genes):
        df_sum[g] = sums[:, i]
    df_sum["total_counts"] = totals
    if include_n_cells:
        df_sum["n_cells"] = n_cells.astype(int)

    totals = df_sum["total_counts"].to_numpy(dtype=float).reshape(-1, 1)
    zero_mask = (totals.ravel() == 0)
    if zero_mask.any():
        n_dropped = int(zero_mask.sum())
        logger.warning(
            "Dropped %d group(s) with zero total counts before CPM normalization.",
            n_dropped,
        )
        df_sum = df_sum[~zero_mask].copy()
        totals = totals[~zero_mask]
    cpm = df_sum[genes].values / totals * scale
    if log1p:
        cpm = np.log1p(cpm)
    df_sum[genes] = cpm
    return df_sum


def pseudobulk_export(
    adata: AnnData,
    genes: Sequence[str],
    design: TrialDesign,
    *,
    visits: tuple[str, str] | None = None,
    celltype_col: str | None = None,
    counts_layer: str | None = "counts",
    min_cells_per_group: int = 1,
    log1p: bool = True,
) -> AnnData:
    """Export pseudobulk expression as a new AnnData object.

    Parameters
    ----------
    adata
        Input AnnData.
    genes
        List of genes to include.
    design
        TrialDesign object.
    visits
        Optional (baseline, followup) visits to subset.
    celltype_col
        If provided, aggregate per participant-visit-celltype.
    counts_layer
        Layer to use for counts (default: "counts").
    min_cells_per_group
        Minimum cells per group to include.
    log1p
        Whether to log1p-transform CPM values.

    Returns
    -------
    AnnData
        AnnData with pseudobulk expression in .X and group metadata in .obs.
    """
    groupby = [design.participant_col, design.visit_col]
    if design.arm_col in adata.obs.columns:
        groupby.append(design.arm_col)
    if celltype_col is not None:
        groupby.append(celltype_col)

    if visits is not None:
        adata = adata[adata.obs[design.visit_col].isin(visits)].copy()

    df_sum = pseudobulk_expression(
        adata,
        genes=genes,
        groupby=groupby,
        counts_layer=counts_layer,
        min_cells_per_group=min_cells_per_group,
        log1p=log1p,
    )

    X = df_sum[genes].to_numpy(dtype=float)
    obs = df_sum[groupby].copy()
    pb = AnnData(X=X, obs=obs)
    pb.var_names = list(genes)
    return pb


def pseudobulk_did(
    adata: AnnData,
    genes: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    celltype_col: str | None = None,
    counts_layer: str | None = "counts",
    log1p: bool = True,
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
        log1p=log1p,
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

        # paired participants: check visit presence using row counts (not a
        # single gene) so that pairing is consistent across all genes.
        visit_counts = df_pool.groupby(
            [design.participant_col, design.visit_col], observed=True
        ).size().unstack(fill_value=0)
        has_v0 = visit_counts.get(visits[0], pd.Series(0, index=visit_counts.index)) > 0
        has_v1 = visit_counts.get(visits[1], pd.Series(0, index=visit_counts.index)) > 0
        keep = visit_counts.index[has_v0 & has_v1]
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
            row = dict(out)
            row["feature"] = g
            row["n_units"] = int(df_pool[design.participant_col].nunique())
            if pool is not None:
                row["celltype"] = pool
            rows.append(row)

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    res = apply_fdr(res, p_col="p_DiD", fdr_col="FDR_DiD")

    # Per-celltype FDR correction
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
        summary = apply_fdr(summary, p_col="p_time", fdr_col="FDR_time")
    delta_long = pd.DataFrame(deltas)
    return summary, delta_long
