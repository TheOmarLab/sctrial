"""
Supplementary Figure 2 — Cellular Atlas and Annotation Confidence.
==================================================================

Establish cell-type annotation quality and embedding reliability.

Panels:
  A  UMAP per dataset coloured by cell type (5 mini-UMAPs).
  B  UMAP per dataset coloured by study grouping variable.
  C  Marker gene dot plot (top 3 markers per cell type, pooled).
  D  Cluster→cell-type purity per dataset.
  E  Centroid-silhouette score on PCA/Harmony embedding.
  F  Embedding neighbourhood purity from full kNN graph.
  G  Cell-type × dataset cross-tabulation heatmap (normalised).
  H  Annotation uncertainty (cluster entropy) per dataset.

Non-overlap guardrail: no treatment-effect claims, no DiD estimates.
"""

from __future__ import annotations

import gc

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from .._shared import (
    HARMONIZED_CELLTYPE_ORDER,
    SUPP_OUTPUT,
    apply_style,
    clear_cache,
    despine,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_celltype,
    harmonize_response,
    load_clinical_trial_dataset,
    save_panel,
)

FIGURE_NAME = "SuppFig2_cell_annotation"

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


def _find_group_col(obs):
    for col in ("response_harmonized", "response", "severity", "visit", "timepoint"):
        if col in obs.columns and obs[col].nunique() > 1:
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


def _panel_umap_group_grid(fig, axes, loaded: dict):
    """5 mini-UMAPs coloured by dataset-specific grouping variable."""
    ds_names = list(loaded.keys())
    for i, name in enumerate(ds_names):
        ax = axes[i]
        adata = loaded[name]["adata"]
        gcol = loaded[name]["group_col"]
        if gcol is None or "X_umap" not in adata.obsm:
            ax.text(0.5, 0.5, "No grouping var", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, fontstyle="italic",
                    color="#888888")
            ax.set_title(name, fontweight="bold", fontsize=9)
            ax.axis("off")
            continue

        labels = adata.obs[gcol].astype(str).values
        emb = adata.obsm["X_umap"]
        uniq = sorted(set(labels))
        pal = dict(zip(uniq, sns.color_palette("Set1", max(3, len(uniq)))[:len(uniq)]))
        order = np.random.default_rng(42).permutation(len(labels))
        colors = np.array([pal[labels[j]] for j in order])
        ax.scatter(emb[order, 0], emb[order, 1], c=colors, s=DOT_SIZE, alpha=0.75,
                   edgecolors="none", rasterized=True)
        handles = [mpatches.Patch(facecolor=pal[k], edgecolor="none", label=k) for k in uniq]
        ax.legend(handles=handles, fontsize=5, loc="best", frameon=True,
                  framealpha=0.85, ncol=1, handlelength=0.8, borderpad=0.3)
        ax.set_title(f"{name} ({gcol})", fontweight="bold", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)

    for i in range(len(ds_names), len(axes)):
        axes[i].axis("off")


# ── Panel B: cell-type proportions (stacked horizontal bars) ──────

def _panel_ct_proportions(ax, loaded: dict):
    """Stacked horizontal bars: cell-type proportion per dataset."""
    ds_names = list(loaded.keys())
    ct_fracs: dict[str, dict[str, float]] = {}

    for name in ds_names:
        ct_col = loaded[name]["ct_col"]
        if ct_col is None:
            ct_fracs[name] = {}
            continue
        raw = loaded[name]["adata"].obs[ct_col].astype(str).map(harmonize_celltype)
        cts = raw.value_counts(normalize=True)
        ct_fracs[name] = cts.to_dict()

    all_cts_sorted = [c for c in HARMONIZED_CELLTYPE_ORDER
                      if any(c in ct_fracs.get(n, {}) for n in ds_names)]
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
    top_cts = sorted(all_cts_sorted, key=lambda ct: sum(
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

    # Select top cell types per dataset (top 5 each) to avoid
    # bias toward the largest dataset (Stephenson)
    top_cts_set: set[str] = set()
    for data in loaded.values():
        if data["ct_col"]:
            per_ds = data["adata"].obs[data["ct_col"]].value_counts().head(5).index
            top_cts_set.update(str(ct) for ct in per_ds)
    top_cts = sorted(top_cts_set)

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


# ── Panel D: cluster→cell-type purity ─────────────────────────────

def _panel_annotation_confidence(ax, loaded: dict):
    """Distribution of cluster-level purity mapped to cells."""
    rows = []
    for name, data in loaded.items():
        adata = data["adata"]
        ct_col = data["ct_col"]
        if ct_col is None or "leiden" not in adata.obs.columns:
            continue

        tab = pd.crosstab(adata.obs["leiden"], adata.obs[ct_col].astype(str))
        purity_by_cluster = (tab.max(axis=1) / tab.sum(axis=1)).to_dict()
        purity = adata.obs["leiden"].map(purity_by_cluster).astype(float).values

        rows.append(pd.DataFrame({
            "Dataset": name,
            "Cluster purity": purity,
        }))

    if not rows:
        ax.text(0.5, 0.5, "No cluster purity data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        ax.set_title("Cluster Purity", fontweight="bold")
        return

    df = pd.concat(rows, ignore_index=True)
    sns.violinplot(data=df, x="Dataset", y="Cluster purity",
                   order=list(loaded.keys()), cut=0, inner="quartile",
                   linewidth=0.5, palette="Set2", density_norm="width", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Dominant cell-type fraction within Leiden cluster")
    ax.set_title("Cluster→Cell-Type Purity", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel E: Silhouette scores ────────────────────────────────────

def _panel_silhouette(ax, loaded: dict):
    """Centroid-silhouette approximation on PCA/Harmony (full data)."""

    ds_names = []
    scores = []

    for name, data in loaded.items():
        adata = data["adata"]
        ct_col = data["ct_col"]
        if ct_col is None:
            continue
        emb_key = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
        if emb_key not in adata.obsm:
            continue

        labels = adata.obs[ct_col].astype(str).values
        if len(set(labels)) < 2:
            continue

        try:
            emb = adata.obsm[emb_key]
            df_emb = pd.DataFrame(emb[:, : min(20, emb.shape[1])])
            df_emb["label"] = labels
            centroids = df_emb.groupby("label", observed=True).mean()
            centroid_map = {k: v.values for k, v in centroids.iterrows()}
            centroid_keys = list(centroid_map.keys())
            emb_np = df_emb.drop(columns=["label"]).values
            a_vals = np.empty(len(df_emb), dtype=float)
            b_vals = np.empty(len(df_emb), dtype=float)
            lab_arr = df_emb["label"].values
            for i in range(len(df_emb)):
                own = centroid_map[lab_arr[i]]
                a_vals[i] = float(np.linalg.norm(emb_np[i] - own))
                d_other = [
                    float(np.linalg.norm(emb_np[i] - centroid_map[k]))
                    for k in centroid_keys if k != lab_arr[i]
                ]
                b_vals[i] = min(d_other) if d_other else a_vals[i]
            denom = np.maximum(a_vals, b_vals)
            sil = np.where(denom > 0, (b_vals - a_vals) / denom, 0.0)
            s = float(np.nanmean(sil))
            ds_names.append(name)
            scores.append(s)
        except Exception as exc:
            print(f"    {name}: silhouette failed ({exc})")

    if not ds_names:
        ax.text(0.5, 0.5, "No silhouette data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        ax.set_title("Embedding Separation", fontweight="bold")
        return

    colors = [_DS_PALETTE.get(n, "grey") for n in ds_names]
    bars = ax.bar(ds_names, scores, color=colors, edgecolor="white", width=0.6)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.01, f"{s:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Centroid-silhouette score")
    ax.set_title("Embedding Quality (PCA/Harmony, Full Data)", fontweight="bold")
    ax.set_ylim(min(scores) - 0.05, max(scores) * 1.2 if scores else 1)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel F: Cell type × dataset cross-tabulation ─────────────────

def _panel_ct_crosstab(ax, loaded: dict):
    """Heatmap: harmonised cell type × dataset (normalised within dataset)."""
    rows = []
    for name, data in loaded.items():
        ct_col = data["ct_col"]
        if ct_col is None:
            continue
        harmonised = data["adata"].obs[ct_col].astype(str).map(harmonize_celltype)
        cts = harmonised.value_counts(normalize=True)
        for ct, frac in cts.items():
            rows.append({"Dataset": name, "Cell type": ct,
                         "Fraction": frac})

    if not rows:
        ax.text(0.5, 0.5, "No cell-type data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Cell type", columns="Dataset",
                           values="Fraction", aggfunc="sum", fill_value=0)

    # Sort by canonical order, keep only present types
    order = [c for c in HARMONIZED_CELLTYPE_ORDER if c in pivot.index]
    pivot = pivot.loc[order]

    sns.heatmap(pivot, cmap="YlOrBr", annot=True, fmt=".2f",
                linewidths=0.5, ax=ax, vmin=0, vmax=pivot.values.max(),
                cbar_kws={"label": "Fraction", "shrink": 0.7})
    ax.set_title("Cell-Type × Dataset (harmonised)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    despine(ax)


# ── Panel G: Cells per cell type per dataset ──────────────────────

def _panel_cells_per_ct(ax, loaded: dict):
    """Grouped bars: cell count per harmonised cell type per dataset (log)."""
    rows = []
    for name, data in loaded.items():
        ct_col = data["ct_col"]
        if ct_col is None:
            continue
        harmonised = data["adata"].obs[ct_col].astype(str).map(harmonize_celltype)
        cts = harmonised.value_counts()
        for ct, cnt in cts.items():
            rows.append({"Dataset": name, "Cell type": ct, "Count": cnt})

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)

    # Use canonical order, keep only present types
    present = df.groupby("Cell type")["Count"].sum()
    top_cts = [c for c in HARMONIZED_CELLTYPE_ORDER if c in present.index]

    ds_names = [n for n in loaded.keys() if loaded[n]["ct_col"] is not None]

    x = np.arange(len(top_cts))
    width = 0.15
    for di, ds in enumerate(ds_names):
        sub = df[df["Dataset"] == ds].set_index("Cell type")
        vals = [sub.loc[ct, "Count"] if ct in sub.index else 0 for ct in top_cts]
        offset = (di - (len(ds_names) - 1) / 2) * width
        ax.bar(x + offset, vals, width * 0.9, label=ds,
               color=_DS_PALETTE.get(ds, "grey"), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(top_cts, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Cell count")
    ax.set_yscale("log")
    ax.set_title("Cells per Cell Type (harmonised)", fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", frameon=True)
    despine(ax)


# ── Panel H: Embedding neighbourhood purity ──────────────────────

def _panel_knn_purity(ax, loaded: dict):
    """Label purity from full Scanpy neighbour graph (no subsampling)."""

    ds_names = []
    purities = []

    for name, data in loaded.items():
        adata = data["adata"]
        ct_col = data["ct_col"]
        if ct_col is None:
            continue

        try:
            conn = adata.obsp.get("connectivities", None)
            if conn is None:
                continue
            conn = conn.tocsr()
            labels = pd.Categorical(adata.obs[ct_col].astype(str)).codes
            per_cell = []
            for i in range(adata.n_obs):
                start, end = conn.indptr[i], conn.indptr[i + 1]
                nbrs = conn.indices[start:end]
                w = conn.data[start:end]
                if nbrs.size == 0:
                    continue
                mask = nbrs != i
                nbrs = nbrs[mask]
                w = w[mask]
                if nbrs.size == 0 or w.sum() == 0:
                    continue
                same = w[labels[nbrs] == labels[i]].sum()
                per_cell.append(float(same / w.sum()))
            if not per_cell:
                continue
            purity = float(np.mean(per_cell))
            ds_names.append(name)
            purities.append(purity)
        except Exception as exc:
            print(f"    {name}: kNN purity failed ({exc})")

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

    ax.set_ylabel("Graph label purity (connectivities)")
    ax.set_title("Embedding Neighbourhood Purity (Full Graph)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


def _panel_annotation_uncertainty(ax, loaded: dict):
    """Annotation uncertainty from normalized cluster entropy."""
    rows = []
    for name, data in loaded.items():
        adata = data["adata"]
        ct_col = data["ct_col"]
        if ct_col is None or "leiden" not in adata.obs.columns:
            continue
        tab = pd.crosstab(adata.obs["leiden"], adata.obs[ct_col].astype(str))
        p = tab.div(tab.sum(axis=1), axis=0).replace(0, np.nan)
        entropy = -(p * np.log(p)).sum(axis=1)
        max_h = np.log(tab.shape[1]) if tab.shape[1] > 1 else 1.0
        norm_entropy = (entropy / max_h).fillna(0.0).to_dict()
        cell_unc = adata.obs["leiden"].map(norm_entropy).astype(float).values
        rows.append(pd.DataFrame({"Dataset": name, "Uncertainty": cell_unc}))

    if not rows:
        ax.text(0.5, 0.5, "No uncertainty data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        return

    df = pd.concat(rows, ignore_index=True)
    sns.violinplot(data=df, x="Dataset", y="Uncertainty",
                   order=list(loaded.keys()), cut=0, inner="quartile",
                   linewidth=0.5, palette="Set2", density_norm="width", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Normalized cluster entropy")
    ax.set_title("Annotation Uncertainty (Cluster Entropy)", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 2 panels (A–H)."""
    print("Supplementary Figure 2: Annotation & Embedding Reliability")

    loaded = {}
    for name, loader in DATASETS:
        try:
            adata = loader()
            if name == "Sade-Feldman":
                adata = harmonize_response(adata)
            adata = _ensure_umap_and_clusters(adata)
            ct_col = _find_celltype_col(adata.obs)
            gcol = _find_group_col(adata.obs)
            loaded[name] = {"adata": adata, "ct_col": ct_col, "group_col": gcol}
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

    # Panel B: UMAP by study grouping variable
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    _panel_umap_group_grid(fig, axes_flat, loaded)
    fig.suptitle("Grouping-Variable UMAPs", fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: Marker dot plot
    fig, ax = plt.subplots(figsize=(10, 7))
    _panel_marker_dotplot(fig, ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: Cluster purity
    fig, ax = plt.subplots(figsize=(8, 5))
    _panel_annotation_confidence(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: Embedding silhouette
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_silhouette(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: kNN graph purity
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_knn_purity(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Panel G: Cell-type proportions cross-tab
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_ct_crosstab(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # Panel H: Annotation uncertainty
    fig, ax = plt.subplots(figsize=(8, 5))
    _panel_annotation_uncertainty(ax, loaded)
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
