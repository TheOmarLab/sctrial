"""
Supplementary Figure 2: Dataset UMAP Embeddings.
=================================================

Grid layout (5 datasets × 2 columns):

Column 1: UMAP coloured by cell type.
Column 2: Parallel categories (cell type × response/visit/timepoint).

Datasets: Sade-Feldman, Stephenson, Vaccine, AML, CAR-T.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    apply_style,
    despine,
    save_panel,
    load_clinical_trial_dataset,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    clear_cache,
)

FIGURE_NAME = "SuppFig2_dataset_umaps"

DOT_SIZE = 4.0
CELLTYPE_LABEL_SIZE = 9
LEGEND_FONTSIZE = 8

DATASETS = [
    ("Sade-Feldman", get_sade_feldman),
    ("Stephenson", get_stephenson),
    ("Vaccine", get_vaccine),
    ("AML", lambda: load_clinical_trial_dataset("aml")),
    ("CAR-T", lambda: load_clinical_trial_dataset("cart")),
]

# Per-dataset: which column to use for the response/condition panel
LABEL_COL2 = {
    "Sade-Feldman": ("response_harmonized", "Response"),
    "Stephenson": ("Status_on_day_collection_summary", "Severity"),
    "Vaccine": ("visit", "Visit"),
    "AML": ("response", "Treatment Arm"),
    "CAR-T": ("timepoint", "Timepoint"),
}


# ── helpers ───────────────────────────────────────────────────────────

def _find_celltype_col(obs):
    for col in ("cell_type", "celltype", "CellType", "cell_type_fine",
                "cell_type_coarse", "celltype_major", "clustnm"):
        if col in obs.columns:
            # Skip if only 1 unique value (e.g. "Immune")
            if obs[col].nunique() > 1:
                return col
    # Fall back to leiden if available
    if "leiden" in obs.columns:
        return "leiden"
    return None


def _find_visit_col(obs):
    for col in ("visit", "timepoint", "time_point", "condition"):
        if col in obs.columns:
            return col
    return None


def _ensure_umap_and_clusters(adata):
    """Ensure UMAP + Leiden clusters exist."""
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
        # Prefer harmony-corrected PCA if available
        use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15)

    if "X_umap" not in adata.obsm:
        print("    Computing UMAP...")
        sc.tl.umap(adata)

    if "leiden" not in adata.obs.columns:
        print("    Computing Leiden clusters...")
        sc.tl.leiden(adata, resolution=0.8)

    return adata


# ── panel functions ───────────────────────────────────────────────────

def _build_ct_palette(unique_cts):
    """Build a colour palette for cell types (tab10/tab20/husl)."""
    n = len(unique_cts)
    if n <= 10:
        colors = sns.color_palette("tab10", n_colors=n)
    elif n <= 20:
        colors = sns.color_palette("tab20", n_colors=n)
    else:
        colors = sns.color_palette("husl", n_colors=n)
    return dict(zip(unique_cts, colors))


def _panel_umap_celltype(ax, adata, title: str):
    """UMAP coloured by cell type with a separate legend.

    Returns ``(ct_col, palette)`` so the parallel-categories panel can
    reuse exactly the same colour mapping.
    """
    ct_col = _find_celltype_col(adata.obs)
    if ct_col is None:
        ax.text(0.5, 0.5, "No cell-type\nannotation", ha="center",
                va="center", transform=ax.transAxes, fontsize=11,
                color=COLORS["gray"])
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
        return ct_col, {}

    mask = adata.obs[ct_col].notna()
    adata_sub = adata[mask] if not mask.all() else adata
    labels = adata_sub.obs[ct_col].astype(str).values
    emb = adata_sub.obsm["X_umap"]

    rng = np.random.default_rng(42)
    order = rng.permutation(len(labels))

    unique_cts = sorted(set(labels))
    palette = _build_ct_palette(unique_cts)
    colors = np.array([palette[labels[i]] for i in order])

    ax.scatter(emb[order, 0], emb[order, 1],
               c=colors, s=DOT_SIZE, alpha=0.7, edgecolors="none",
               rasterized=True)

    # Separate legend
    n_ct = len(unique_cts)
    handles = [mpatches.Patch(facecolor=palette[ct], edgecolor="none",
                              label=(f"Cluster {ct}" if ct_col == "leiden"
                                     else ct))
               for ct in unique_cts]
    ncol = 2 if n_ct > 10 else 1
    leg_fs = max(5, CELLTYPE_LABEL_SIZE - max(0, n_ct - 10))
    ax.legend(handles=handles, fontsize=leg_fs, loc="best",
              frameon=True, framealpha=0.9, ncol=ncol,
              handlelength=1.0, handleheight=0.8,
              borderpad=0.4, labelspacing=0.25)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    despine(ax)
    return ct_col, palette


def _panel_parallel_categories(ax, adata, ct_col: str, label_col: str,
                                label_title: str, title: str,
                                ct_palette_in: dict | None = None,
                                max_types: int = 15):
    """Parallel categories (alluvial) plot: cell type → label.

    Left axis  = cell type (coloured to match the UMAP panel).
    Right axis = response / visit / timepoint label.
    Ribbons coloured by cell-type palette.

    Parameters
    ----------
    ct_palette_in : dict, optional
        Colour mapping ``{cell_type: colour}`` from the UMAP panel so
        colours stay consistent across both panels.

    Groups rare cell types (beyond *max_types*) into 'Other'.
    """
    from matplotlib.path import Path
    import matplotlib.patches as mpatches_path

    if ct_col is None or label_col not in adata.obs.columns:
        ax.text(0.5, 0.5, "Insufficient\nannotation", ha="center",
                va="center", transform=ax.transAxes, fontsize=11,
                color=COLORS["gray"])
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
        return

    obs = adata.obs.copy()
    ct_vals = obs[ct_col].astype(str)
    label_vals = obs[label_col].astype(str)

    valid = (ct_vals != "nan") & (label_vals != "nan")
    ct_vals = ct_vals[valid].copy()
    label_vals = label_vals[valid]

    # Group rare cell types
    ct_counts = ct_vals.value_counts()
    if len(ct_counts) > max_types:
        keep = set(ct_counts.head(max_types).index)
        ct_vals = ct_vals.apply(lambda x: x if x in keep else "Other")

    # Category ordering — sort by descending count
    ct_order = ct_vals.value_counts().index.tolist()      # LEFT axis
    label_order = label_vals.value_counts().index.tolist() # RIGHT axis

    n_ct = len(ct_order)
    n_lab = len(label_order)

    # Cell-type palette — reuse UMAP colours when available
    if ct_palette_in:
        ct_palette = {}
        for ct in ct_order:
            if ct in ct_palette_in:
                ct_palette[ct] = ct_palette_in[ct]
            else:
                # "Other" or new categories get a grey
                ct_palette[ct] = (0.6, 0.6, 0.6)
    else:
        ct_palette = _build_ct_palette(ct_order)

    # Label palette (fixed colours for known labels)
    fixed = {
        "Responder": COLORS["treated"],
        "Non-responder": COLORS["control"],
        "Treatment": COLORS["treated"],
        "Control": COLORS["control"],
        "Pre": COLORS["control"],
        "Post": COLORS["treated"],
        "Severe": "#d62728",
        "Mild": "#2ca02c",
        "Critical": "#8c0000",
        "Critical ": "#8c0000",
        "Moderate": "#ff7f0e",
        "Death": "#1f1f1f",
    }
    fallback_colors = sns.color_palette("Set2", n_colors=max(n_lab, 8))
    label_palette = {}
    fi = 0
    for v in label_order:
        if v in fixed:
            label_palette[v] = fixed[v]
        else:
            label_palette[v] = fallback_colors[fi % len(fallback_colors)]
            fi += 1

    # Contingency table: rows = cell type (left), cols = label (right)
    cross = pd.crosstab(ct_vals, label_vals)
    cross = cross.reindex(index=ct_order, columns=label_order, fill_value=0)

    total = cross.values.sum()
    gap = 0.02          # gap fraction between blocks
    bar_w = 0.12        # half-width of category bars
    x_left = 0.0        # left axis  (cell type)
    x_right = 1.0       # right axis (label)

    # ── compute block positions (normalised to [0, 1]) ──────────────
    def _block_positions(counts, n):
        total_gap = gap * (n - 1) if n > 1 else 0
        usable = 1.0 - total_gap
        heights = [(c / total) * usable for c in counts]
        positions = []
        y = 0.0
        for i, h in enumerate(heights):
            positions.append((y, h))
            y += h + (gap if i < n - 1 else 0)
        return positions

    left_counts = [cross.loc[ct].sum() for ct in ct_order]
    right_counts = [cross[lab].sum() for lab in label_order]

    left_pos = _block_positions(left_counts, n_ct)
    right_pos = _block_positions(right_counts, n_lab)

    # ── draw ribbons ────────────────────────────────────────────────
    left_offsets = [0.0] * n_ct
    right_offsets = [0.0] * n_lab

    for ci, ct in enumerate(ct_order):
        for li, lab in enumerate(label_order):
            count = cross.loc[ct, lab]
            if count == 0:
                continue

            usable_left = 1.0 - gap * (n_ct - 1) if n_ct > 1 else 1.0
            usable_right = 1.0 - gap * (n_lab - 1) if n_lab > 1 else 1.0

            ribbon_h_left = (count / total) * usable_left
            ribbon_h_right = (count / total) * usable_right

            y0_left = left_pos[ci][0] + left_offsets[ci]
            y0_right = right_pos[li][0] + right_offsets[li]

            left_offsets[ci] += ribbon_h_left
            right_offsets[li] += ribbon_h_right

            x0 = x_left + bar_w
            x1 = x_right - bar_w
            xm = (x0 + x1) / 2

            verts = [
                (x0, y0_left),
                (xm, y0_left),
                (xm, y0_right),
                (x1, y0_right),
                (x1, y0_right + ribbon_h_right),
                (xm, y0_right + ribbon_h_right),
                (xm, y0_left + ribbon_h_left),
                (x0, y0_left + ribbon_h_left),
                (x0, y0_left),
            ]
            codes = [
                Path.MOVETO,
                Path.CURVE4, Path.CURVE4, Path.CURVE4,
                Path.LINETO,
                Path.CURVE4, Path.CURVE4, Path.CURVE4,
                Path.CLOSEPOLY,
            ]
            path = Path(verts, codes)
            color = ct_palette[ct]
            patch = mpatches_path.PathPatch(
                path, facecolor=(*color[:3], 0.45),
                edgecolor=(*color[:3], 0.15), linewidth=0.3)
            ax.add_patch(patch)

    # ── draw category bars ──────────────────────────────────────────
    ct_fsize = max(5, min(8, 90 // max(n_ct, 1)))
    label_fsize = max(6, min(9, 90 // max(n_lab, 1)))

    # Left axis — cell type
    for i, ct in enumerate(ct_order):
        y_bot, h = left_pos[i]
        color = ct_palette[ct]
        rect = plt.Rectangle((x_left - bar_w, y_bot), bar_w * 2, h,
                              facecolor=color, edgecolor="white",
                              linewidth=0.5, zorder=3)
        ax.add_patch(rect)
        display_label = f"Cl. {ct}" if ct_col == "leiden" else ct
        ax.text(x_left - bar_w - 0.02, y_bot + h / 2, display_label,
                ha="right", va="center", fontsize=ct_fsize)

    # Right axis — label (response / visit / timepoint)
    for i, lab in enumerate(label_order):
        y_bot, h = right_pos[i]
        color = label_palette[lab]
        rect = plt.Rectangle((x_right - bar_w, y_bot), bar_w * 2, h,
                              facecolor=color, edgecolor="white",
                              linewidth=0.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x_right + bar_w + 0.02, y_bot + h / 2, lab,
                ha="left", va="center", fontsize=label_fsize,
                fontweight="bold")

    # ── axis labels ─────────────────────────────────────────────────
    ax.text(x_left, -0.06, "Cell Type", ha="center", va="top",
            fontsize=8, fontweight="bold", transform=ax.transData)
    ax.text(x_right, -0.06, label_title, ha="center", va="top",
            fontsize=8, fontweight="bold", transform=ax.transData)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.08, 1.02)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Supplementary Figure 2 individual panels."""
    print("Supplementary Figure 2: Dataset UMAP Embeddings")

    loaded = {}
    for name, loader in DATASETS:
        try:
            adata = loader()
            if name == "Sade-Feldman":
                adata = harmonize_response(adata)
            adata = _ensure_umap_and_clusters(adata)
            loaded[name] = adata
            print(f"  {name}: {adata.n_obs:,} cells")
        except Exception as exc:
            print(f"  {name}: failed to load ({exc})")

    if not loaded:
        print("  No datasets available; skipping figure.")
        return

    label_idx = 0
    label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for ds_name, adata in loaded.items():
        col2, col2_title = LABEL_COL2[ds_name]

        # Panel 1: Cell types UMAP
        fig, ax = plt.subplots(figsize=(7, 6))
        ct_col, ct_palette = _panel_umap_celltype(ax, adata,
                                                    f"{ds_name}, Cell Types")
        fig.tight_layout()
        save_panel(fig, f"panel_{label_chars[label_idx]}",
                   FIGURE_NAME, SUPP_OUTPUT)
        label_idx += 1

        # Panel 2: Parallel categories (cell type × label)
        fig, ax = plt.subplots(figsize=(7, 6))
        _panel_parallel_categories(ax, adata, ct_col, col2,
                                    col2_title,
                                    f"{ds_name}, Composition",
                                    ct_palette_in=ct_palette)
        fig.tight_layout()
        save_panel(fig, f"panel_{label_chars[label_idx]}",
                   FIGURE_NAME, SUPP_OUTPUT)
        label_idx += 1

    # Cleanup
    for adata in loaded.values():
        del adata
    loaded.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
