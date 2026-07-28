"""Contracted input representations for real-data benchmark analyses.

The permutation and subsampling analyses must hand each method the identical
representation that the simulation benchmark uses. Previously they built their
own pseudobulk and normalised within the tested panel, characterising methods
under different normalisation scope from the simulation. This module is the
shared code path that prevents the two from drifting apart.

The normalisation scope bug
---------------------------
The CPM denominator must be the FULL TRANSCRIPTOME, not the tested panel.
When a coordinated signal is present across the panel, normalising within the
panel shifts the denominator, giving every null gene an apparent offsetting
effect. That artifact was previously attributed to empirical-Bayes moderation.

Panel selection happens AFTER normalisation, which is what a real workflow does
and what makes panel size separable from normalisation scope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "prepare_inputs_from_adata",
    "participant_log1p_cpm",
]

_META_COLS = ("participant", "visit", "arm")


def participant_log1p_cpm(
    pseudobulk_counts: pd.DataFrame,
    panel_genes: list[str],
    gene_cols: list[str] | None = None,
) -> pd.DataFrame:
    """log(1 + CPM) per participant-visit, normalised on the FULL TRANSCRIPTOME.

    The denominator is the sum over ``gene_cols`` (all transcriptome genes),
    not over ``panel_genes``. Panel selection happens AFTER normalisation.
    """
    genes = list(pseudobulk_counts.columns) if gene_cols is None else list(gene_cols)
    genes = [g for g in genes if g not in _META_COLS and g != "n_cells"]
    mat = pseudobulk_counts[genes].to_numpy(dtype=np.float64)
    total = mat.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    cpm = mat / total * 1e6

    pos = {g: i for i, g in enumerate(genes)}
    keep = [g for g in panel_genes if g in pos]
    meta = pseudobulk_counts[[c for c in _META_COLS if c in pseudobulk_counts.columns]]
    panel_df = pd.DataFrame(
        np.log1p(cpm[:, [pos[g] for g in keep]]),
        columns=keep,
        index=pseudobulk_counts.index,
    )
    return pd.concat([meta, panel_df], axis=1)


def prepare_inputs_from_adata(
    adata,
    panel_genes: list[str],
    participant_col: str = "participant",
    visit_col: str = "visit",
    arm_col: str = "arm",
    counts_layer: str = "counts",
) -> dict:
    """Build contracted method inputs from real data.

    Returns a dict with keys:
        participant_log1p_cpm  outcome frame for sctrial / Wilcoxon
        pseudobulk_counts      panel raw counts for count-based models
        lib_size               full-transcriptome total per pseudobulk sample
        cell_counts            cell-level AnnData restricted to the panel
        cell_lib_size          full-transcriptome library size per cell (linear scale)
        panel_genes            the tested panel (genes present in both panel_genes and adata)
    """
    obs = adata.obs
    X = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
    gene_cols = [str(g) for g in adata.var_names]
    panel = [g for g in panel_genes if g in set(gene_cols)]

    keys = [k for k in [participant_col, visit_col, arm_col] if k in obs.columns]
    groups = obs.groupby(keys, observed=True).indices

    rows, mats = [], []
    for key, idx in groups.items():
        key = key if isinstance(key, tuple) else (key,)
        sub = X[idx]
        block = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
        mats.append(block.sum(axis=0))
        meta = dict(zip(keys, [str(k) for k in key]))
        rows.append(
            {
                "participant": meta.get(participant_col, ""),
                "visit": meta.get(visit_col, ""),
                "arm": meta.get(arm_col, "Treated"),
                "n_cells": len(idx),
            }
        )

    mat = np.vstack(mats)
    pb = pd.concat(
        [pd.DataFrame(rows), pd.DataFrame(mat, columns=gene_cols)], axis=1
    )

    # Full-transcriptome library size per pseudobulk sample
    lib_size = mat.sum(axis=1)

    # Panel-restricted pseudobulk counts for count-based models
    gene_to_idx = {g: i for i, g in enumerate(gene_cols)}
    panel_idx = [gene_to_idx[g] for g in panel]
    meta_df = pd.DataFrame(rows)[["participant", "visit", "arm"]]
    panel_counts = pd.concat(
        [meta_df, pd.DataFrame(mat[:, panel_idx], columns=panel)], axis=1
    )

    # Full-transcriptome CPM normalised outcome for sctrial/Wilcoxon
    outcome = participant_log1p_cpm(pb, panel, gene_cols=gene_cols)

    # Cell-level AnnData for NEBULA, restricted to panel genes, canonical column names
    cell_lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    # Non-finite lib_size (NaN from QC-flagged cells, Inf from corrupted data)
    # would silently become NA in the R CSV, causing NEBULA to throw
    # "missing value where TRUE/FALSE needed". Zero is safe: R filters those out.
    cell_lib[~np.isfinite(cell_lib)] = 0.0
    adata_canonical = _canonical_adata(adata, participant_col, visit_col, arm_col, counts_layer)
    cell_view = adata_canonical[:, panel]

    return {
        "participant_log1p_cpm": outcome,
        "pseudobulk_counts": panel_counts,
        "lib_size": lib_size,
        "cell_counts": cell_view,
        "cell_lib_size": cell_lib,
        "panel_genes": panel,
    }


def _canonical_adata(adata, participant_col, visit_col, arm_col, counts_layer):
    """AnnData with canonical obs column names (participant, visit, arm)."""
    import anndata as ad

    obs = adata.obs.rename(
        columns={participant_col: "participant", visit_col: "visit", arm_col: "arm"}
    )
    keep = [c for c in ("participant", "visit", "arm") if c in obs.columns]
    X = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
    out = ad.AnnData(X=X, obs=obs[keep].copy())
    out.var_names = adata.var_names
    return out
