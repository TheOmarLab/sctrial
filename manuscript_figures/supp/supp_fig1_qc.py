"""
Supplementary Figure 1 — QC Metrics Across Datasets.
=====================================================

Per-dataset QC visualisations for all five datasets
(Sade-Feldman, Stephenson, Vaccine, AML, CAR-T).

Row 1 (panels A-E):  UMAP coloured by genes detected.
Row 2 (panels F-J):  UMAP coloured by total UMI counts.
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

def _get_ngenes(adata) -> np.ndarray:
    """Return gene-count array (whichever name exists), compute if needed."""
    obs = adata.obs
    for col in ("n_genes_by_counts", "n_genes", "n_genes_detected", "NumberOfGenes"):
        if col in obs.columns:
            return np.asarray(obs[col], dtype=float)
    # Compute from X (sparse-safe)
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        return np.asarray((X > 0).sum(axis=1), dtype=float).ravel()
    return np.asarray((X > 0).sum(axis=1), dtype=float).ravel()


def _get_counts(adata) -> np.ndarray:
    """Return total counts array (sparse-safe)."""
    obs = adata.obs
    for col in ("total_counts", "n_counts", "total_UMI", "TranscriptomeUMIs"):
        if col in obs.columns:
            return np.asarray(obs[col], dtype=float)
    # Compute from X (sparse-safe)
    import scipy.sparse as sp
    X = adata.X
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
    """Ensure UMAP coordinates exist in adata.obsm['X_umap']."""
    if "X_umap" in adata.obsm:
        return adata

    import scanpy as sc

    # Need PCA first
    if "X_pca" not in adata.obsm:
        print("    Computing PCA...")
        # Use log1p layer if available, else X
        if "log1p_tpm" in adata.layers:
            adata_work = adata.copy()
            adata_work.X = adata_work.layers["log1p_tpm"]
        elif "log1p_cpm" in adata.layers:
            adata_work = adata.copy()
            adata_work.X = adata_work.layers["log1p_cpm"]
        else:
            adata_work = adata.copy()
            import scipy.sparse as sp
            if sp.issparse(adata_work.X):
                sc.pp.normalize_total(adata_work, target_sum=1e4)
                sc.pp.log1p(adata_work)
            else:
                # Already normalized (e.g. TPM)
                pass

        # Use seurat_v3 only when raw counts are available
        hvg_flavor = "seurat_v3" if "counts" in adata.layers else "seurat"
        sc.pp.highly_variable_genes(adata_work, n_top_genes=2000, flavor=hvg_flavor)
        sc.tl.pca(adata_work, n_comps=50)
        adata.obsm["X_pca"] = adata_work.obsm["X_pca"]
        del adata_work
        gc.collect()

    print("    Computing UMAP...")
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)
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

            datasets[name] = {
                "umap": adata.obsm["X_umap"],
                "n_genes": _get_ngenes(adata),
                "total_counts": _get_counts(adata),
                "cells_per_participant": cells_per_pid,
                "n_cells": adata.n_obs,
                "n_participants": cells_per_pid.shape[0],
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

    # Use log scale for UMI counts
    if metric == "total_counts":
        values = np.log10(values + 1)
        cbar_label = r"$\log_{10}$" + f"({cbar_label})"

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

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontweight="bold")
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
    """Violin + strip plot of cells per participant across all datasets."""
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

    sns.violinplot(
        data=df, x="Dataset", y="Cells", order=order,
        palette=palette, inner=None, linewidth=0.8,
        cut=0, ax=ax, alpha=0.6,
    )
    sns.stripplot(
        data=df, x="Dataset", y="Cells", order=order,
        color="black", size=3, alpha=0.6, jitter=0.15, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Cells per participant")
    ax.set_title("Cells per Participant", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
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
