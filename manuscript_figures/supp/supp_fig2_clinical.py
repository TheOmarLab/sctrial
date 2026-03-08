"""
Supplementary Figure 2 — Cellular Atlas and Annotation Confidence.
==================================================================

Establish cell-type annotation quality and embedding reliability.

Panels:
  A  UMAP per dataset coloured by cell type (5 mini-UMAPs).
  B  Cell-type proportions per dataset (stacked horizontal bars).
  C  Marker gene dot plot (top 3 markers per cell type, pooled).
  D  Annotation confidence: per-cell max marker score distribution.
  E  Silhouette scores per dataset (embedding quality per cell type).
  F  Cell-type × dataset cross-tabulation heatmap (normalised).
  G  Cells per cell type per dataset (grouped bars, log-scale).
  H  Embedding neighbourhood purity per dataset.

Non-overlap guardrail: no treatment-effect claims, no DiD estimates.
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

DOT_SIZE = 1.8
CELLTYPE_LABEL_SIZE = 9

_DS_PALETTE = dict(zip(
    ["Sade-Feldman", "Stephenson", "Vaccine", "AML", "CAR-T"],
    sns.color_palette("Set2", 5),
))

DATASETS = [
    ("Sade-Feldman", get_sade_feldman),
    ("Stephenson", get_stephenson),
    ("Vaccine", get_vaccine),
    ("AML", lambda: load_clinical_trial_dataset("aml")),
    ("CAR-T", lambda: load_clinical_trial_dataset("cart")),
]


# ── helpers ──────────────────────────────────────────────────────────

def _find_celltype_col(obs):
    for col in ("cell_type", "celltype", "CellType", "cell_type_fine",
                "cell_type_coarse", "celltype_major", "clustnm"):
        if col in obs.columns and obs[col].nunique() > 1:
            return col
    if "leiden" in obs.columns:
        return "leiden"
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
        use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15)

    if "X_umap" not in adata.obsm:
        print("    Computing UMAP...")
        sc.tl.umap(adata)

    if "leiden" not in adata.obs.columns:
        print("    Computing Leiden clusters...")
        sc.tl.leiden(adata, resolution=0.8)

    return adata


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


def _get_log_expr(adata):
    """Return log-normalised expression matrix (cells × genes)."""
    import scipy.sparse as sp
    for layer in ("log1p_tpm", "log1p_cpm", "log1p_norm"):
        if layer in adata.layers:
            X = adata.layers[layer]
            if sp.issparse(X):
                return X
            return X
    # Fallback: normalise counts
    if "counts" in adata.layers:
        X = adata.layers["counts"].copy()
        if sp.issparse(X):
            X = X.toarray()
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X = np.log1p(X / row_sums * 1e4)
        return X
    return None


# ── Panel A: UMAP per dataset (5 mini-UMAPs) ─────────────────────

def _panel_umap_grid(fig, axes, loaded: dict):
    """5 mini-UMAPs coloured by cell type."""
    ds_names = list(loaded.keys())
    for i, name in enumerate(ds_names):
        ax = axes[i]
        adata = loaded[name]["adata"]
        ct_col = loaded[name]["ct_col"]

        if ct_col is None:
            ax.text(0.5, 0.5, "No annotation", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, fontstyle="italic",
                    color="#888888")
            ax.set_title(name, fontweight="bold", fontsize=9)
            ax.axis("off")
            continue

        emb = adata.obsm["X_umap"]
        labels = adata.obs[ct_col].astype(str).values
        unique_cts = sorted(set(labels))
        palette = _build_ct_palette(unique_cts)

        rng = np.random.default_rng(42)
        order = rng.permutation(len(labels))

        colors = np.array([palette[labels[j]] for j in order])
        ax.scatter(emb[order, 0], emb[order, 1],
                   c=colors, s=DOT_SIZE, alpha=0.7, edgecolors="none",
                   rasterized=True)

        # Compact legend
        handles = [mpatches.Patch(facecolor=palette[ct], edgecolor="none",
                                  label=ct) for ct in unique_cts[:12]]
        if len(unique_cts) > 12:
            handles.append(mpatches.Patch(facecolor="grey", edgecolor="none",
                                          label=f"+ {len(unique_cts)-12} more"))
        ax.legend(handles=handles, fontsize=4.5, loc="best", frameon=True,
                  framealpha=0.8, ncol=1, handlelength=0.7, handleheight=0.6,
                  borderpad=0.3, labelspacing=0.15, markerscale=0.4)

        ax.set_title(name, fontweight="bold", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)

    # Hide unused axes
    for i in range(len(ds_names), len(axes)):
        axes[i].axis("off")


# ── Panel B: cell-type proportions (stacked horizontal bars) ──────

def _panel_ct_proportions(ax, loaded: dict):
    """Stacked horizontal bars: cell-type proportion per dataset."""
    ds_names = list(loaded.keys())
    all_cts = set()
    ct_fracs = {}

    for name in ds_names:
        ct_col = loaded[name]["ct_col"]
        if ct_col is None:
            ct_fracs[name] = {}
            continue
        cts = loaded[name]["adata"].obs[ct_col].astype(str).value_counts(normalize=True)
        ct_fracs[name] = cts.to_dict()
        all_cts.update(cts.index)

    all_cts_sorted = sorted(all_cts)
    palette = _build_ct_palette(all_cts_sorted)

    y_pos = np.arange(len(ds_names))
    for name_i, name in enumerate(ds_names):
        left = 0.0
        for ct in all_cts_sorted:
            frac = ct_fracs.get(name, {}).get(ct, 0)
            if frac > 0:
                ax.barh(name_i, frac, left=left, height=0.6,
                        color=palette[ct], edgecolor="white", linewidth=0.3)
                if frac > 0.06:
                    ax.text(left + frac / 2, name_i, ct, ha="center",
                            va="center", fontsize=4.5, color="white",
                            fontweight="bold")
                left += frac

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ds_names, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of cells")
    ax.set_title("Cell-Type Proportions", fontweight="bold")

    # Legend for top cell types
    top_cts = sorted(all_cts, key=lambda ct: sum(
        ct_fracs.get(n, {}).get(ct, 0) for n in ds_names), reverse=True)[:15]
    handles = [mpatches.Patch(facecolor=palette[ct], edgecolor="none",
                              label=ct) for ct in top_cts]
    ax.legend(handles=handles, fontsize=5, loc="lower right", frameon=True,
              ncol=2, handlelength=0.8, handleheight=0.6,
              borderpad=0.3, labelspacing=0.15)
    despine(ax)


# ── Panel C: Marker gene dot plot ─────────────────────────────────

def _panel_marker_dotplot(fig, ax, loaded: dict):
    """Top 3 marker genes per major cell type, shown as dot plot."""
    # Canonical markers for major cell types
    canonical_markers = {
        "T cells": ["CD3D", "CD3E", "TRAC"],
        "CD8 T": ["CD8A", "CD8B", "GZMB"],
        "CD4 T": ["CD4", "IL7R", "CCR7"],
        "B cells": ["CD19", "MS4A1", "CD79A"],
        "NK cells": ["NKG7", "GNLY", "KLRD1"],
        "Monocytes": ["CD14", "LYZ", "S100A9"],
        "DC": ["FCER1A", "CLEC10A", "CD1C"],
        "Plasma": ["MZB1", "JCHAIN", "SDC1"],
    }

    # Pool all datasets and compute fraction expressing + mean expression
    rows = []
    for name, data in loaded.items():
        adata = data["adata"]
        X = _get_log_expr(adata)
        if X is None:
            continue
        import scipy.sparse as sp
        gene_names = list(adata.var_names)
        gene_idx_map = {g: i for i, g in enumerate(gene_names)}

        ct_col = data["ct_col"]
        if ct_col is None:
            continue

        labels = adata.obs[ct_col].astype(str).values

        for ct_name, markers in canonical_markers.items():
            for marker in markers:
                # Case-insensitive lookup
                idx = gene_idx_map.get(marker)
                if idx is None:
                    # Try uppercase
                    marker_up = marker.upper()
                    for g, gi in gene_idx_map.items():
                        if g.upper() == marker_up:
                            idx = gi
                            break
                if idx is None:
                    continue

                if sp.issparse(X):
                    col = np.asarray(X[:, idx].todense()).ravel()
                else:
                    col = X[:, idx].ravel()

                for ct_label in sorted(set(labels)):
                    mask = labels == ct_label
                    if mask.sum() < 10:
                        continue
                    vals = col[mask]
                    frac_expr = (vals > 0).mean()
                    mean_expr = vals[vals > 0].mean() if (vals > 0).any() else 0
                    rows.append({
                        "Cell type": ct_label, "Marker": marker,
                        "Marker group": ct_name,
                        "Fraction expressing": frac_expr,
                        "Mean expression": mean_expr,
                        "Dataset": name,
                    })

    if not rows:
        ax.text(0.5, 0.5, "No marker data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        ax.set_title("Marker Gene Expression", fontweight="bold")
        return

    df = pd.DataFrame(rows)
    # Average across datasets
    agg = df.groupby(["Cell type", "Marker", "Marker group"]).agg({
        "Fraction expressing": "mean",
        "Mean expression": "mean",
    }).reset_index()

    # Select top cell types by frequency across datasets
    all_ct_counts = {}
    for data in loaded.values():
        if data["ct_col"]:
            for ct, cnt in data["adata"].obs[data["ct_col"]].value_counts().items():
                all_ct_counts[str(ct)] = all_ct_counts.get(str(ct), 0) + cnt
    top_cts = sorted(all_ct_counts, key=all_ct_counts.get, reverse=True)[:10]

    agg_filt = agg[agg["Cell type"].isin(top_cts)]
    if agg_filt.empty:
        ax.text(0.5, 0.5, "No overlap", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        return

    # Pivot for dot plot
    markers_used = sorted(agg_filt["Marker"].unique())
    ct_used = sorted(agg_filt["Cell type"].unique())

    pivot_frac = agg_filt.pivot_table(
        index="Cell type", columns="Marker",
        values="Fraction expressing", fill_value=0)
    pivot_expr = agg_filt.pivot_table(
        index="Cell type", columns="Marker",
        values="Mean expression", fill_value=0)

    # Reindex
    markers_used = [m for m in markers_used if m in pivot_frac.columns]
    ct_used = [c for c in ct_used if c in pivot_frac.index]
    pivot_frac = pivot_frac.reindex(index=ct_used, columns=markers_used, fill_value=0)
    pivot_expr = pivot_expr.reindex(index=ct_used, columns=markers_used, fill_value=0)

    # Draw dot plot
    for yi, ct in enumerate(ct_used):
        for xi, marker in enumerate(markers_used):
            frac = pivot_frac.loc[ct, marker]
            expr = pivot_expr.loc[ct, marker]
            if frac > 0.01:
                size = frac * 200  # scale dot size
                ax.scatter(xi, yi, s=size, c=expr, cmap="Reds",
                           vmin=0, vmax=pivot_expr.values.max(),
                           edgecolors="grey", linewidth=0.3, zorder=3)

    ax.set_xticks(range(len(markers_used)))
    ax.set_xticklabels(markers_used, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ct_used)))
    ax.set_yticklabels(ct_used, fontsize=7)
    ax.set_xlim(-0.5, len(markers_used) - 0.5)
    ax.set_ylim(-0.5, len(ct_used) - 0.5)
    ax.set_title("Canonical Marker Expression", fontweight="bold")

    # Size legend
    for frac_val in [0.25, 0.5, 0.75]:
        ax.scatter([], [], s=frac_val * 200, c="grey", alpha=0.5,
                   edgecolors="grey", label=f"{frac_val:.0%}")
    ax.legend(title="Frac. expr.", fontsize=5, title_fontsize=6,
              loc="upper right", frameon=True, handletextpad=0.1)

    despine(ax)


# ── Panel D: Annotation confidence ────────────────────────────────

def _panel_annotation_confidence(ax, loaded: dict):
    """Per-cell max marker score distribution per dataset.

    Uses a simple approach: for each cell, compute the max z-score
    of canonical marker expression across cell types as a confidence proxy.
    """
    canonical_markers = {
        "T cells": ["CD3D", "CD3E", "TRAC"],
        "B cells": ["CD19", "MS4A1", "CD79A"],
        "NK cells": ["NKG7", "GNLY", "KLRD1"],
        "Monocytes": ["CD14", "LYZ", "S100A9"],
    }
    all_markers = []
    for markers in canonical_markers.values():
        all_markers.extend(markers)

    import scipy.sparse as sp

    rows = []
    for name, data in loaded.items():
        adata = data["adata"]
        X = _get_log_expr(adata)
        if X is None:
            continue

        gene_names = list(adata.var_names)
        gene_idx_map = {g.upper(): i for i, g in enumerate(gene_names)}

        marker_scores = []
        for marker in all_markers:
            idx = gene_idx_map.get(marker.upper())
            if idx is None:
                continue
            if sp.issparse(X):
                col = np.asarray(X[:, idx].todense()).ravel()
            else:
                col = X[:, idx].ravel()
            marker_scores.append(col)

        if not marker_scores:
            continue

        # Max marker expression per cell (proxy for annotation confidence)
        max_scores = np.max(np.column_stack(marker_scores), axis=1)

        # Subsample for violin
        n = len(max_scores)
        if n > 10000:
            idx = np.random.default_rng(42).choice(n, 10000, replace=False)
            max_scores = max_scores[idx]

        rows.append(pd.DataFrame({
            "Dataset": name,
            "Max marker score": max_scores,
        }))

    if not rows:
        ax.text(0.5, 0.5, "No marker data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        ax.set_title("Annotation Confidence", fontweight="bold")
        return

    df = pd.concat(rows, ignore_index=True)
    sns.violinplot(data=df, x="Dataset", y="Max marker score",
                   order=list(loaded.keys()), cut=0, inner="quartile",
                   linewidth=0.5, palette="Set2", density_norm="width", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Max canonical marker expression")
    ax.set_title("Annotation Confidence (Marker Proxy)", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel E: Silhouette scores ────────────────────────────────────

def _panel_silhouette(ax, loaded: dict):
    """Silhouette score per dataset (embedding quality per cell type)."""
    from sklearn.metrics import silhouette_score

    ds_names = []
    scores = []

    for name, data in loaded.items():
        adata = data["adata"]
        ct_col = data["ct_col"]
        if ct_col is None or "X_umap" not in adata.obsm:
            continue

        labels = adata.obs[ct_col].astype(str).values
        if len(set(labels)) < 2:
            continue

        # Subsample for speed
        n = adata.n_obs
        if n > 10000:
            idx = np.random.default_rng(42).choice(n, 10000, replace=False)
        else:
            idx = np.arange(n)

        emb = adata.obsm["X_umap"][idx]
        labs = labels[idx]

        try:
            s = silhouette_score(emb, labs, sample_size=min(5000, len(idx)),
                                 random_state=42)
            ds_names.append(name)
            scores.append(s)
        except Exception:
            pass

    if not ds_names:
        ax.text(0.5, 0.5, "No silhouette data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        ax.set_title("Silhouette Scores", fontweight="bold")
        return

    colors = [_DS_PALETTE.get(n, "grey") for n in ds_names]
    bars = ax.bar(ds_names, scores, color=colors, edgecolor="white", width=0.6)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.01, f"{s:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Silhouette score")
    ax.set_title("Embedding Quality (Silhouette)", fontweight="bold")
    ax.set_ylim(0, max(scores) * 1.2 if scores else 1)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel F: Cell type × dataset cross-tabulation ─────────────────

def _panel_ct_crosstab(ax, loaded: dict):
    """Heatmap: cell type × dataset (normalised within dataset)."""
    rows = []
    for name, data in loaded.items():
        ct_col = data["ct_col"]
        if ct_col is None:
            continue
        cts = data["adata"].obs[ct_col].astype(str).value_counts(normalize=True)
        for ct, frac in cts.items():
            rows.append({"Dataset": name, "Cell type": str(ct),
                         "Fraction": frac})

    if not rows:
        ax.text(0.5, 0.5, "No cell-type data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Cell type", columns="Dataset",
                           values="Fraction", fill_value=0)

    # Sort by total frequency
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    # Limit to top 15
    if len(pivot) > 15:
        pivot = pivot.iloc[:15]

    sns.heatmap(pivot, cmap="YlOrBr", annot=True, fmt=".2f",
                linewidths=0.5, ax=ax, vmin=0, vmax=pivot.values.max(),
                cbar_kws={"label": "Fraction", "shrink": 0.7})
    ax.set_title("Cell-Type × Dataset", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    despine(ax)


# ── Panel G: Cells per cell type per dataset ──────────────────────

def _panel_cells_per_ct(ax, loaded: dict):
    """Grouped bars: cell count per cell type per dataset (log-scale)."""
    rows = []
    for name, data in loaded.items():
        ct_col = data["ct_col"]
        if ct_col is None:
            continue
        cts = data["adata"].obs[ct_col].astype(str).value_counts()
        for ct, cnt in cts.items():
            rows.append({"Dataset": name, "Cell type": str(ct), "Count": cnt})

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)

    # Top 10 cell types by total count
    top_cts = df.groupby("Cell type")["Count"].sum().nlargest(10).index.tolist()
    df_filt = df[df["Cell type"].isin(top_cts)]

    ds_names = [n for n in loaded.keys() if loaded[n]["ct_col"] is not None]

    x = np.arange(len(top_cts))
    width = 0.15
    for di, ds in enumerate(ds_names):
        sub = df_filt[df_filt["Dataset"] == ds].set_index("Cell type")
        vals = [sub.loc[ct, "Count"] if ct in sub.index else 0 for ct in top_cts]
        offset = (di - (len(ds_names) - 1) / 2) * width
        ax.bar(x + offset, vals, width * 0.9, label=ds,
               color=_DS_PALETTE.get(ds, "grey"), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(top_cts, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Cell count")
    ax.set_yscale("log")
    ax.set_title("Cells per Cell Type", fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", frameon=True)
    despine(ax)


# ── Panel H: Embedding neighbourhood purity ──────────────────────

def _panel_knn_purity(ax, loaded: dict):
    """kNN label purity per dataset: fraction of k nearest neighbours
    with the same cell type label."""
    from sklearn.neighbors import NearestNeighbors

    ds_names = []
    purities = []

    for name, data in loaded.items():
        adata = data["adata"]
        ct_col = data["ct_col"]
        if ct_col is None or "X_umap" not in adata.obsm:
            continue

        labels = adata.obs[ct_col].astype(str).values
        emb = adata.obsm["X_umap"]

        # Subsample for speed
        n = adata.n_obs
        if n > 10000:
            idx = np.random.default_rng(42).choice(n, 10000, replace=False)
        else:
            idx = np.arange(n)

        emb_sub = emb[idx]
        labs_sub = labels[idx]

        try:
            nn = NearestNeighbors(n_neighbors=16, algorithm="ball_tree")
            nn.fit(emb_sub)
            _, indices = nn.kneighbors(emb_sub)
            # Exclude self (first column)
            neighbour_labels = labs_sub[indices[:, 1:]]
            self_labels = labs_sub[:, None]
            purity = (neighbour_labels == self_labels).mean()

            ds_names.append(name)
            purities.append(purity)
        except Exception:
            pass

    if not ds_names:
        ax.text(0.5, 0.5, "No embedding data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        ax.set_title("Neighbourhood Purity", fontweight="bold")
        return

    colors = [_DS_PALETTE.get(n, "grey") for n in ds_names]
    bars = ax.bar(ds_names, purities, color=colors, edgecolor="white", width=0.6)
    for bar, p in zip(bars, purities):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.01, f"{p:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("kNN label purity (k=15)")
    ax.set_title("Embedding Neighbourhood Purity", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 2 panels."""
    print("Supplementary Figure 2: Cellular Atlas and Annotation Confidence")

    loaded = {}
    for name, loader in DATASETS:
        try:
            adata = loader()
            if name == "Sade-Feldman":
                adata = harmonize_response(adata)
            adata = _ensure_umap_and_clusters(adata)
            ct_col = _find_celltype_col(adata.obs)
            loaded[name] = {"adata": adata, "ct_col": ct_col}
            ct_info = f", ct_col={ct_col}" if ct_col else ", no cell types"
            print(f"  {name}: {adata.n_obs:,} cells{ct_info}")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")

    if not loaded:
        print("  No datasets available; skipping.")
        return

    # Panel A: UMAP grid (5 mini-UMAPs)
    n_ds = len(loaded)
    ncols = min(n_ds, 3)
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    _panel_umap_grid(fig, axes_flat, loaded)
    fig.suptitle("Cell-Type UMAPs", fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: Cell-type proportions
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_ct_proportions(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: Marker dot plot
    fig, ax = plt.subplots(figsize=(10, 7))
    _panel_marker_dotplot(fig, ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: Annotation confidence
    fig, ax = plt.subplots(figsize=(8, 5))
    _panel_annotation_confidence(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: Silhouette scores
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_silhouette(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: Cell type × dataset crosstab
    fig, ax = plt.subplots(figsize=(8, 7))
    _panel_ct_crosstab(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Panel G: Cells per cell type
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_cells_per_ct(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # Panel H: kNN purity
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_knn_purity(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    # Cleanup
    for data in loaded.values():
        del data["adata"]
    loaded.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
