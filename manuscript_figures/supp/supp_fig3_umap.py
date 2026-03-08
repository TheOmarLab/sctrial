"""
Supplementary Figure 3 — Sade-Feldman UMAP with Cell Types & Response.
======================================================================

Two panels using real Sade-Feldman immunotherapy data:

A  UMAP coloured by cell type.
B  UMAP coloured by treatment response (Responder vs Non-responder).
"""

from __future__ import annotations

import gc

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    apply_style,
    clear_cache,
    despine,
    get_sade_feldman,
    harmonize_response,
    save_panel,
)

FIGURE_NAME = "SuppFig3_umap"


# ── helpers ───────────────────────────────────────────────────────────

def _ensure_umap_and_clusters(adata):
    """Ensure UMAP coordinates and Leiden clusters exist."""
    # PCA
    if "X_pca" not in adata.obsm:
        print("    Computing PCA...")
        adata_work = adata.copy()
        if "log1p_tpm" in adata.layers:
            adata_work.X = adata_work.layers["log1p_tpm"]
        sc.pp.highly_variable_genes(adata_work, n_top_genes=2000, flavor="seurat")
        sc.tl.pca(adata_work, n_comps=50)
        adata.obsm["X_pca"] = adata_work.obsm["X_pca"]
        del adata_work
        gc.collect()

    # Neighbors (needed for both UMAP and Leiden)
    if "neighbors" not in adata.uns:
        print("    Computing neighbors...")
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)

    # UMAP
    if "X_umap" not in adata.obsm:
        print("    Computing UMAP...")
        sc.tl.umap(adata)

    # Leiden clusters as proxy for cell types
    if "leiden" not in adata.obs.columns:
        print("    Computing Leiden clusters...")
        sc.tl.leiden(adata, resolution=0.8)

    return adata


# ── panels ────────────────────────────────────────────────────────────

def _panel_celltypes(ax, adata):
    """UMAP coloured by cell type."""
    umap = adata.obsm["X_umap"]
    labels = adata.obs["cell_type"].astype(str).values

    # Shuffle for overplotting fairness
    rng = np.random.default_rng(42)
    order = rng.permutation(len(labels))

    unique_cts = sorted(set(labels))
    n_ct = len(unique_cts)
    palette = dict(zip(unique_cts,
                       sns.color_palette("husl", n_colors=n_ct)))
    colors = np.array([palette[labels[i]] for i in order])

    ax.scatter(
        umap[order, 0], umap[order, 1],
        c=colors, s=3.0, alpha=0.7, edgecolors="none", rasterized=True,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Cell Types", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend
    handles = [mpatches.Patch(facecolor=palette[c], label=c,
                              edgecolor="none")
               for c in unique_cts]
    ncol = 2 if n_ct > 5 else 1
    ax.legend(handles=handles, fontsize=6, loc="lower right", frameon=True,
              framealpha=0.9, ncol=ncol, markerscale=0.8,
              handlelength=1.0, handletextpad=0.4)
    despine(ax)


def _panel_response(ax, adata):
    """UMAP coloured by treatment response."""
    umap = adata.obsm["X_umap"]
    response = adata.obs["response_harmonized"].values

    rng = np.random.default_rng(42)
    order = rng.permutation(len(response))

    color_map = {
        "Responder": COLORS["treated"],
        "Non-responder": COLORS["control"],
    }
    colors = np.array([color_map.get(response[i], "#cccccc") for i in order])

    ax.scatter(
        umap[order, 0], umap[order, 1],
        c=colors, s=3.0, alpha=0.7, edgecolors="none", rasterized=True,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Treatment Response", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Summary text
    n_r = (response == "Responder").sum()
    n_nr = (response == "Non-responder").sum()
    ax.text(
        0.97, 0.03,
        f"R: {n_r:,}  NR: {n_nr:,}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"],
                  alpha=0.8),
    )

    # Legend
    handles = [
        mpatches.Patch(facecolor=COLORS["treated"], label="Responder", edgecolor="none"),
        mpatches.Patch(facecolor=COLORS["control"], label="Non-responder", edgecolor="none"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right", frameon=True,
              framealpha=0.9, title="Response", title_fontsize=9)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 3 panels."""
    print("Supplementary Figure 3: Sade-Feldman UMAP")

    adata = get_sade_feldman()
    adata = _ensure_umap_and_clusters(adata)
    adata = harmonize_response(adata)

    # Panel A: Cell types
    fig_a, ax_a = plt.subplots(figsize=(7, 6))
    _panel_celltypes(ax_a, adata)
    fig_a.tight_layout()
    save_panel(fig_a, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: Response
    fig_b, ax_b = plt.subplots(figsize=(7, 6))
    _panel_response(ax_b, adata)
    fig_b.tight_layout()
    save_panel(fig_b, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Cleanup
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
