"""
Supplementary Figure 1: QC Metrics Across Datasets
===================================================

Per-dataset QC visualisations for all five datasets
(Sade-Feldman, Stephenson, Vaccine, AML, CAR-T).

Row 1 (panels A-E):  UMAP coloured by genes detected.
Row 2 (panels F-J):  UMAP coloured by total expression (UMI counts or TPM).
Panel K:             Cells per participant across datasets.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    apply_style,
    despine,
    save_panel,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    load_clinical_trial_dataset,
    clear_cache,
)

FIGURE_NAME = "SuppFig1_qc_metrics"


# ── helpers ───────────────────────────────────────────────────────────

def _get_expression_matrix(adata):
    """Return the best available non-negative expression matrix for QC.

    Priority: layers['counts'] > layers['tpm'] > layers['cpm'].
    Raises if no recognised layer exists, since falling back to adata.X
    risks using scaled/batch-corrected values that invalidate QC metrics.
    """
    for layer in ("counts", "tpm", "cpm"):
        if layer in adata.layers:
            return adata.layers[layer]
    raise ValueError(
        "No suitable expression layer for QC: need 'counts', 'tpm', or 'cpm' "
        f"in adata.layers. Available layers: {list(adata.layers.keys())}"
    )


def _get_ngenes(adata) -> np.ndarray:
    """Return genes-detected per cell.

    Counting nonzero entries is valid from any expression matrix
    (counts, TPM, CPM) since a gene is detected iff expression > 0.
    """
    obs = adata.obs
    for col in ("n_genes_by_counts", "n_genes", "n_genes_detected", "NumberOfGenes"):
        if col in obs.columns:
            return np.asarray(obs[col], dtype=float)
    import scipy.sparse as sp
    X = _get_expression_matrix(adata)
    if sp.issparse(X):
        return np.asarray((X > 0).sum(axis=1), dtype=float).ravel()
    return np.asarray((X > 0).sum(axis=1), dtype=float).ravel()


def _get_counts(adata) -> np.ndarray:
    """Return total expression per cell.

    Prefers raw UMI counts; falls back to TPM/CPM sums as a QC proxy.
    """
    obs = adata.obs
    for col in ("total_counts", "n_counts", "total_UMI", "TranscriptomeUMIs"):
        if col in obs.columns:
            return np.asarray(obs[col], dtype=float)
    import scipy.sparse as sp
    X = _get_expression_matrix(adata)
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1), dtype=float).ravel()
    return np.asarray(X.sum(axis=1), dtype=float).ravel()


def _get_pid_col(obs: pd.DataFrame) -> str:
    """Return participant ID column name."""
    for col in ("participant_id", "patient_id", "donor_id", "subject_id"):
        if col in obs.columns:
            return col
    raise KeyError("No participant ID column found")


def _ensure_umap(adata):
    """Ensure UMAP coordinates exist in adata.obsm['X_umap'].

    Uses the same HVG / PCA / neighbors parameters as supp_fig2 so that
    UMAP embeddings are consistent across supplementary figures.
    """
    if "X_umap" in adata.obsm:
        return adata

    import scanpy as sc

    # Need PCA first
    if "X_pca" not in adata.obsm:
        print("    Computing PCA...")
        adata_work = adata.copy()
        if "log1p_tpm" in adata_work.layers:
            adata_work.X = adata_work.layers["log1p_tpm"]
        elif "log1p_cpm" in adata_work.layers:
            adata_work.X = adata_work.layers["log1p_cpm"]
        elif "log1p_norm" in adata_work.layers:
            adata_work.X = adata_work.layers["log1p_norm"]
        elif "counts" in adata_work.layers:
            adata_work.X = adata_work.layers["counts"]
            sc.pp.normalize_total(adata_work, target_sum=1e4)
            sc.pp.log1p(adata_work)
        sc.pp.highly_variable_genes(adata_work, n_top_genes=3000,
                                     flavor="seurat")
        adata_work = adata_work[:, adata_work.var["highly_variable"]].copy()
        sc.pp.scale(adata_work, max_value=10)
        sc.tl.pca(adata_work, n_comps=50)
        adata.obsm["X_pca"] = adata_work.obsm["X_pca"]
        del adata_work
        gc.collect()

    has_conn = ("connectivities" in adata.obsp) if adata.obsp else False
    if "neighbors" not in adata.uns or not has_conn:
        print("    Computing neighbors...")
        use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15)

    print("    Computing UMAP...")
    sc.tl.umap(adata)
    return adata


def _load_all_datasets() -> dict:
    """Load all five datasets and extract QC + UMAP data."""
    datasets = {}
    loaders = [
        ("Sade-Feldman", get_sade_feldman, None),
        ("Stephenson", get_stephenson, None),
        ("Vaccine", get_vaccine, None),
        ("AML", lambda: load_clinical_trial_dataset("aml"), None),
        ("CAR-T", lambda: load_clinical_trial_dataset("cart"), None),
    ]
    for name, loader, _ in loaders:
        try:
            adata = loader()
            adata = _ensure_umap(adata)

            pid_col = _get_pid_col(adata.obs)
            cells_per_pid = adata.obs.groupby(pid_col).size()

            has_raw = "counts" in adata.layers or any(
                c in adata.obs.columns
                for c in ("total_counts", "n_counts", "total_UMI")
            )
            datasets[name] = {
                "umap": adata.obsm["X_umap"],
                "n_genes": _get_ngenes(adata),
                "total_counts": _get_counts(adata),
                "cells_per_participant": cells_per_pid,
                "n_cells": adata.n_obs,
                "n_participants": cells_per_pid.shape[0],
                "has_raw_counts": has_raw,
            }
            print(f"  {name}: {adata.n_obs:,} cells, "
                  f"{cells_per_pid.shape[0]} participants")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")

    return datasets


# ── UMAP QC panels ───────────────────────────────────────────────────

def _panel_umap_qc(ax, data: dict, title: str, metric: str, cbar_label: str):
    """UMAP coloured by a QC metric (genes detected or total counts)."""
    umap = data["umap"]
    values = data[metric]

    # Random shuffle for overplotting fairness
    rng = np.random.default_rng(42)
    order = rng.permutation(len(values))
    umap = umap[order]
    values = values[order]

    # Use log scale for expression totals
    if metric == "total_counts":
        values = np.log10(values + 1)
        if data.get("has_raw_counts", True):
            cbar_label = r"$\log_{10}$(UMI counts)"
        else:
            cbar_label = r"$\log_{10}$(TPM)"

    sc = ax.scatter(
        umap[:, 0], umap[:, 1],
        c=values,
        s=1.5, alpha=0.7, rasterized=True,
        cmap="magma",
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    if metric == "n_genes":
        metric_label = "Genes Detected"
    elif data.get("has_raw_counts", True):
        metric_label = "Total UMI Counts"
    else:
        metric_label = "Total TPM"
    ax.set_title(f"{title}: {metric_label}", fontweight="bold", fontsize=10)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

    # Summary text
    ax.text(
        0.97, 0.03,
        f"n = {data['n_cells']:,}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"],
                  alpha=0.8),
    )
    despine(ax)


# ── Cells per participant panel ──────────────────────────────────────

def _panel_cells_per_participant(ax, datasets: dict):
    """Box plot + jittered points of cells per participant across all datasets."""
    rows = []
    for ds_name, data in datasets.items():
        cpp = data["cells_per_participant"]
        for v in cpp.values:
            rows.append({"Dataset": ds_name, "Cells": int(v)})

    df = pd.DataFrame(rows)
    if df.empty:
        ax.set_title("Cells per Participant")
        return

    order = list(datasets.keys())
    palette = sns.color_palette("Set2", n_colors=len(order))

    # x-axis labels with participant counts
    x_labels = [
        f"{name}\n(n={datasets[name]['n_participants']})" for name in order
    ]

    sns.boxplot(
        data=df, x="Dataset", y="Cells", order=order,
        palette=palette, linewidth=0.8, fliersize=0,
        ax=ax, boxprops=dict(alpha=0.6),
    )
    sns.stripplot(
        data=df, x="Dataset", y="Cells", order=order,
        color="black", size=4, alpha=0.6, jitter=0.2, ax=ax,
    )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Cells per participant")
    ax.set_title("Cells per Participant", fontweight="bold")
    despine(ax)


# ======================================================================
# Generate individual panels
# ======================================================================

def generate():
    """Create and save Supplementary Figure 1 individual panels."""
    print("Supplementary Figure 1: QC Metrics")
    datasets = _load_all_datasets()

    if not datasets:
        print("  No datasets loaded; skipping figure.")
        return

    ds_names = list(datasets.keys())

    # ── Row 1: UMAP coloured by genes detected ───────────────────────
    for i, ds_name in enumerate(ds_names):
        fig, ax = plt.subplots(figsize=(5, 4))
        _panel_umap_qc(ax, datasets[ds_name], ds_name,
                        "n_genes", "Genes detected")
        fig.tight_layout()
        save_panel(fig, f"panel_{chr(65 + i)}", FIGURE_NAME, SUPP_OUTPUT)

    # ── Row 2: UMAP coloured by total UMI counts ─────────────────────
    offset = len(ds_names)
    for i, ds_name in enumerate(ds_names):
        fig, ax = plt.subplots(figsize=(5, 4))
        _panel_umap_qc(ax, datasets[ds_name], ds_name,
                        "total_counts", "Total counts")
        fig.tight_layout()
        save_panel(fig, f"panel_{chr(65 + offset + i)}", FIGURE_NAME, SUPP_OUTPUT)

    # ── Panel K: Cells per participant ────────────────────────────────
    fig_k, ax_k = plt.subplots(figsize=(6, 4))
    _panel_cells_per_participant(ax_k, datasets)
    fig_k.tight_layout()
    save_panel(fig_k, "panel_K", FIGURE_NAME, SUPP_OUTPUT)

    # ── Cleanup ───────────────────────────────────────────────────────
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
