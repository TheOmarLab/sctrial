"""
Supplementary Figure 2 — Cell Annotation and Baseline Comparability.
====================================================================

Establish cell-type annotation quality and demonstrate that pre-treatment
groups are comparable before DiD analysis.

Panels:
  A  UMAP per dataset coloured by cell type (5 mini-UMAPs).
  B  UMAP per dataset coloured by grouping variable (arm/visit).
  C  Marker gene dot plot (top 3 markers per cell type, pooled).
  D  Embedding quality: silhouette + kNN purity (merged 1×2).
  E  Cell-type × dataset cross-tabulation heatmap (normalised).
  F  Quantitative baseline arm overlap (kNN arm-mixing score).
  G  Per-dataset parallel categories (separate PNGs, participant-weighted).

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

FIGURE_NAME = "SuppFig2_annotation_baseline"

DOT_SIZE = 1.8

_DS_PALETTE = dict(zip(
    ["Sade-Feldman", "Stephenson", "Vaccine", "AML", "CAR-T"],
    ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"],
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
            return adata.layers[layer]
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


def _pid_col(obs):
    for c in ("participant_id", "patient_id", "donor_id", "pt_id"):
        if c in obs.columns:
            return c
    return None


def _visit_col(obs):
    for c in ("visit", "Collection_Day", "dfo_bin", "timepoint"):
        if c in obs.columns and obs[c].nunique() > 1:
            return c
    return None


def _arm_col(obs):
    for c in ("response", "severity", "therapy", "condition"):
        if c in obs.columns and obs[c].nunique() > 1:
            return c
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


# ── Panel B: UMAP per dataset coloured by grouping variable ──────

def _panel_umap_grouping(fig, axes, loaded: dict):
    """5 mini-UMAPs coloured by arm_col or visit_col."""
    ds_names = list(loaded.keys())
    for i, name in enumerate(ds_names):
        ax = axes[i]
        adata = loaded[name]["adata"]
        arm = loaded[name].get("arm_col")
        vis = loaded[name].get("visit_col")

        grp_col = arm if arm else vis
        if grp_col is None:
            ax.text(0.5, 0.5, "No grouping", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, fontstyle="italic",
                    color="#888888")
            ax.set_title(name, fontweight="bold", fontsize=9)
            ax.axis("off")
            continue

        emb = adata.obsm["X_umap"]
        labels = adata.obs[grp_col].astype(str).values
        unique_vals = sorted(set(labels))
        palette = dict(zip(unique_vals,
                           sns.color_palette("Set1", len(unique_vals))))

        rng = np.random.default_rng(42)
        order = rng.permutation(len(labels))
        colors = np.array([palette[labels[j]] for j in order])
        ax.scatter(emb[order, 0], emb[order, 1],
                   c=colors, s=DOT_SIZE, alpha=0.7, edgecolors="none",
                   rasterized=True)

        handles = [mpatches.Patch(facecolor=palette[v], edgecolor="none",
                                  label=v) for v in unique_vals]
        ax.legend(handles=handles, fontsize=5, loc="best", frameon=True,
                  framealpha=0.8, handlelength=0.7, handleheight=0.6,
                  borderpad=0.3, labelspacing=0.15)

        grp_label = "arm" if arm else "visit"
        ax.set_title(f"{name} ({grp_label})", fontweight="bold", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)

    for i in range(len(ds_names), len(axes)):
        axes[i].axis("off")


# ── Panel C: Marker gene dot plot ─────────────────────────────────

def _panel_marker_dotplot(fig, ax, loaded: dict):
    """Top 3 marker genes per major cell type, shown as dot plot."""
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
                idx = gene_idx_map.get(marker)
                if idx is None:
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
    agg = df.groupby(["Cell type", "Marker", "Marker group"]).agg({
        "Fraction expressing": "mean",
        "Mean expression": "mean",
    }).reset_index()

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

    markers_used = sorted(agg_filt["Marker"].unique())
    ct_used = sorted(agg_filt["Cell type"].unique())

    pivot_frac = agg_filt.pivot_table(
        index="Cell type", columns="Marker",
        values="Fraction expressing", fill_value=0)
    pivot_expr = agg_filt.pivot_table(
        index="Cell type", columns="Marker",
        values="Mean expression", fill_value=0)

    markers_used = [m for m in markers_used if m in pivot_frac.columns]
    ct_used = [c for c in ct_used if c in pivot_frac.index]
    pivot_frac = pivot_frac.reindex(index=ct_used, columns=markers_used, fill_value=0)
    pivot_expr = pivot_expr.reindex(index=ct_used, columns=markers_used, fill_value=0)

    for yi, ct in enumerate(ct_used):
        for xi, marker in enumerate(markers_used):
            frac = pivot_frac.loc[ct, marker]
            expr = pivot_expr.loc[ct, marker]
            if frac > 0.01:
                size = frac * 200
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

    for frac_val in [0.25, 0.5, 0.75]:
        ax.scatter([], [], s=frac_val * 200, c="grey", alpha=0.5,
                   edgecolors="grey", label=f"{frac_val:.0%}")
    ax.legend(title="Frac. expr.", fontsize=5, title_fontsize=6,
              loc="upper right", frameon=True, handletextpad=0.1)

    despine(ax)


# ── Panel D: Silhouette scores ────────────────────────────────────

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


# ── Panel E: Embedding neighbourhood purity ──────────────────────

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


# ── Panel D merged: Silhouette + kNN purity in 1×2 ───────────────

def _panel_embedding_quality(fig_merged, loaded: dict):
    """D: Combined silhouette + kNN purity in 1×2 subplot."""
    ax1, ax2 = fig_merged.subplots(1, 2)
    _panel_silhouette(ax1, loaded)
    _panel_knn_purity(ax2, loaded)


# ── Panel F: Quantitative baseline arm overlap (kNN mixing) ──────

def _panel_arm_mixing(ax, loaded: dict):
    """Quantitative baseline arm overlap via kNN arm-mixing score.

    For each dataset with an arm column, at baseline: build kNN graph
    (k=15) in PCA space, compute fraction of neighbors from opposite arm
    per cell. Bar chart: arm-mixing score per dataset.
    Higher = better overlap = arms are well mixed.
    """
    ds_names = []
    mix_scores = []

    for name, data in loaded.items():
        arm = data.get("arm_col")
        vis = data.get("visit_col")
        if arm is None:
            continue
        adata = data["adata"]
        obs = adata.obs

        # Get baseline cells
        if vis:
            pre_mask = obs[vis].astype(str).str.lower().isin(
                ["pre", "baseline", "d0", "day0", "0", "d000"])
            if pre_mask.sum() < 50:
                continue
            idx = np.where(pre_mask.values)[0]
        else:
            idx = np.arange(adata.n_obs)

        emb_key = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
        if emb_key not in adata.obsm:
            continue

        emb = adata.obsm[emb_key][idx, :min(20, adata.obsm[emb_key].shape[1])]
        arms = obs[arm].values[idx].astype(str)
        unique_arms = sorted(set(arms))
        if len(unique_arms) < 2:
            continue

        # Build kNN (k=15) via brute-force on PCA embedding
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(16, len(idx)),
                              metric="euclidean", algorithm="auto")
        nn.fit(emb)
        _, nbr_idx = nn.kneighbors(emb)
        nbr_idx = nbr_idx[:, 1:]  # exclude self

        # Fraction of neighbors from different arm
        mix_per_cell = []
        for i in range(len(idx)):
            nbr_arms = arms[nbr_idx[i]]
            frac_diff = np.mean(nbr_arms != arms[i])
            mix_per_cell.append(float(frac_diff))

        ds_names.append(name)
        mix_scores.append(float(np.mean(mix_per_cell)))

    if not ds_names:
        ax.text(0.5, 0.5, "No arm-overlap data", ha="center", va="center",
                transform=ax.transAxes)
        return

    colors = [_DS_PALETTE.get(n, "grey") for n in ds_names]
    bars = ax.bar(ds_names, mix_scores, color=colors, edgecolor="white",
                  width=0.6)
    for bar, s in zip(bars, mix_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.01, f"{s:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Arm-mixing score (frac. opposite-arm neighbors)")
    ax.set_title("Baseline Arm Overlap (kNN mixing, k=15)", fontweight="bold")
    ax.set_ylim(0, min(max(mix_scores) * 1.3, 1.0) if mix_scores else 1)
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--", alpha=0.5,
               label="Perfect mixing")
    ax.legend(fontsize=7, frameon=True)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel E: Cell type × dataset cross-tabulation ─────────────────

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

    order = [c for c in HARMONIZED_CELLTYPE_ORDER if c in pivot.index]
    pivot = pivot.loc[order]

    sns.heatmap(pivot, cmap="YlOrBr", annot=True, fmt=".2f",
                linewidths=0.5, ax=ax, vmin=0, vmax=pivot.values.max(),
                cbar_kws={"label": "Fraction", "shrink": 0.7})
    ax.set_title("Cell-Type × Dataset (harmonised)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    despine(ax)


# ── Panel G: Baseline cell-type composition — per-dataset alluvial ──

def _draw_sigmoid_ribbon(ax, x0, x1, y0_bot, y0_top, y1_bot, y1_top,
                         color, alpha=0.45):
    """Draw a smooth sigmoid ribbon between two vertical bars."""
    n_pts = 80
    t = np.linspace(0, 1, n_pts)
    s = 1.0 / (1.0 + np.exp(-10 * (t - 0.5)))
    top = y0_top + s * (y1_top - y0_top)
    bot = y0_bot + s * (y1_bot - y0_bot)
    xs = x0 + t * (x1 - x0)
    verts = np.concatenate([
        np.column_stack([xs, top]),
        np.column_stack([xs[::-1], bot[::-1]]),
    ])
    poly = plt.Polygon(verts, closed=True, fc=color, ec="none", alpha=alpha)
    ax.add_patch(poly)


def _single_parcats(ax, ds_name: str, obs_sub, right_col: str, ct_col: str,
                    right_label: str = "Arm", pid_col: str | None = None):
    """One parallel-categories subplot: Cell type (left) → right_col (right).

    *right_col* is typically the arm column for two-arm datasets or
    the visit column for single-arm datasets.

    When *pid_col* is given, flows are participant-weighted (count unique
    participants per Cell-type × Right combination) instead of cell-weighted.
    """
    mapped = obs_sub[ct_col].astype(str).map(harmonize_celltype)
    right_vals = obs_sub[right_col].astype(str)

    # Count flows: Cell type → Right
    flow = pd.DataFrame({"Cell type": mapped.values, "Right": right_vals.values})
    if pid_col and pid_col in obs_sub.columns:
        flow["pid"] = obs_sub[pid_col].values
        counts = (
            flow.groupby(["Cell type", "Right"])["pid"]
            .nunique()
            .reset_index(name="n")
        )
    else:
        counts = flow.groupby(["Cell type", "Right"]).size().reset_index(name="n")
    total = counts["n"].sum()
    if total == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Orders — cap at top 15 cell types to keep figure renderable
    max_ct = 15
    ct_order_local = (
        flow["Cell type"].value_counts().sort_values(ascending=False).index
        .tolist()
    )
    if len(ct_order_local) > max_ct:
        top_cts_set = set(ct_order_local[:max_ct])
        flow.loc[~flow["Cell type"].isin(top_cts_set), "Cell type"] = "Other"
        # Recompute counts after collapsing
        if pid_col and pid_col in obs_sub.columns:
            counts = (
                flow.groupby(["Cell type", "Right"])["pid"]
                .nunique()
                .reset_index(name="n")
            )
        else:
            counts = flow.groupby(["Cell type", "Right"]).size().reset_index(name="n")
        total = counts["n"].sum()
        ct_order_local = (
            flow["Cell type"].value_counts().sort_values(ascending=False).index
            .tolist()
        )
    right_order = sorted(flow["Right"].unique())

    # Palette: cell types use tab20, right values use Set2
    pal20 = sns.color_palette("tab20", max(len(ct_order_local), 1))
    ct_palette = dict(zip(ct_order_local, pal20))
    right_pal = dict(zip(right_order,
                         sns.color_palette("Set2", len(right_order))))

    col_x = [0.0, 1.0]
    bar_w = 0.10
    gap = 0.012

    def _positions(categories, counts_map):
        usable = 1.0 - gap * max(len(categories) - 1, 0)
        pos = {}
        y = 0.0
        for cat in categories:
            h = usable * counts_map.get(cat, 0) / total
            pos[cat] = (y, y + h)
            y += h + gap
        return pos

    # Derive marginal counts from the (possibly participant-weighted) counts df
    ct_counts = counts.groupby("Cell type")["n"].sum().to_dict()
    right_counts = counts.groupby("Right")["n"].sum().to_dict()
    ct_pos = _positions(ct_order_local, ct_counts)
    right_pos = _positions(right_order, right_counts)

    # Draw bars and collect label y-positions for repulsion
    # Left column: Cell type
    left_labels: list[tuple[float, str]] = []  # (y_centre, name)
    for ct_name in ct_order_local:
        yb, yt = ct_pos[ct_name]
        ax.fill_between([col_x[0] - bar_w, col_x[0] + bar_w], yb, yt,
                        color=ct_palette[ct_name], alpha=0.85,
                        edgecolor="white", lw=0.5)
        left_labels.append(((yb + yt) / 2, ct_name))

    # Right column
    right_labels: list[tuple[float, str]] = []
    for rv in right_order:
        yb, yt = right_pos[rv]
        ax.fill_between([col_x[1] - bar_w, col_x[1] + bar_w], yb, yt,
                        color=right_pal[rv], alpha=0.85,
                        edgecolor="white", lw=0.5)
        right_labels.append(((yb + yt) / 2, rv))

    # Greedy repulsion to avoid overlapping labels (bounded to [0, 1])
    def _repel(labels, min_gap=0.028):
        """Shift label y-positions so they don't overlap, staying in [0,1]."""
        if not labels:
            return labels
        out = [(y, name) for y, name in labels]
        for _ in range(50):
            moved = False
            for i in range(1, len(out)):
                dy = out[i][0] - out[i - 1][0]
                if dy < min_gap:
                    shift = (min_gap - dy) / 2 + 0.001
                    out[i - 1] = (max(0.0, out[i - 1][0] - shift), out[i - 1][1])
                    out[i] = (min(1.0, out[i][0] + shift), out[i][1])
                    moved = True
            if not moved:
                break
        return out

    for y_pos, ct_name in _repel(left_labels):
        ax.text(col_x[0] - bar_w - 0.03, y_pos, ct_name,
                ha="right", va="center", fontsize=6.5, fontweight="bold")

    for y_pos, rv in _repel(right_labels):
        ax.text(col_x[1] + bar_w + 0.03, y_pos, rv,
                ha="left", va="center", fontsize=8, fontweight="bold")

    # Draw ribbons: Cell type → Right
    ct_cursor = {c: ct_pos[c][0] for c in ct_order_local}
    right_cursor = {r: right_pos[r][0] for r in right_order}

    usable_frac = 1.0 - gap * max(len(ct_order_local) - 1, 0)
    for _, row in counts.iterrows():
        ct_name, rv, cnt = row["Cell type"], row["Right"], row["n"]
        h = usable_frac * cnt / total
        y0_bot = ct_cursor[ct_name]
        y0_top = y0_bot + h
        y1_bot = right_cursor[rv]
        y1_top = y1_bot + h
        _draw_sigmoid_ribbon(
            ax, col_x[0] + bar_w, col_x[1] - bar_w,
            y0_bot, y0_top, y1_bot, y1_top,
            color=ct_palette[ct_name], alpha=0.35,
        )
        ct_cursor[ct_name] = y0_top
        right_cursor[rv] = y1_top

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(col_x)
    ax.set_xticklabels(["Cell type", right_label], fontsize=9,
                       fontweight="bold")
    ax.set_yticks([])
    ax.set_title(ds_name, fontweight="bold", fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _panel_baseline_ct_by_arm_separate(loaded: dict):
    """Per-dataset parallel categories saved as separate PNGs.

    Two-arm datasets: Cell type → Arm (baseline cells only).
    Single-arm datasets: Cell type → Visit (all cells).
    Flows are participant-weighted (count unique participants per flow).
    """
    ds_entries: list[tuple[str, pd.DataFrame, str, str, str, str | None]] = []
    for name, data in loaded.items():
        ct = data.get("ct_col")
        if ct is None:
            continue
        obs = data["adata"].obs
        arm = data.get("arm_col")
        vis = data.get("visit_col")
        pid = data.get("pid_col")

        if arm is not None:
            # Two-arm: restrict to baseline, flow Cell type → Arm
            if vis:
                pre_mask = obs[vis].astype(str).str.lower().isin(
                    ["pre", "baseline", "d0", "day0", "0", "d000"])
                if pre_mask.sum() < 50:
                    continue
                sub = obs.loc[pre_mask]
            else:
                sub = obs
            ds_entries.append((name, sub, arm, ct, "Arm", pid))
        elif vis is not None:
            # Single-arm: flow Cell type → Visit
            ds_entries.append((name, obs, vis, ct, "Visit", pid))

    panel_dir = SUPP_OUTPUT / f"{FIGURE_NAME}_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for ds_name, sub, right_col, ct_col, right_label, pid_col in ds_entries:
        fig, ax = plt.subplots(figsize=(7.0, 7.0))
        _single_parcats(ax, ds_name, sub, right_col, ct_col,
                        right_label=right_label, pid_col=pid_col)
        fig.subplots_adjust(left=0.30, right=0.75, top=0.92, bottom=0.05)
        safe_name = ds_name.replace(" ", "_").replace("-", "_")
        path = panel_dir / f"panel_G_{safe_name}.png"
        fig.savefig(str(path), format="png", dpi=600,
                    facecolor="white", edgecolor="none")
        print(f"    Saved panel: panel_G_{safe_name}")
        plt.close(fig)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 2 panels (A–G)."""
    print("Supplementary Figure 2: Cell Annotation and Baseline Comparability")

    loaded = {}
    for name, loader in DATASETS:
        try:
            adata = loader()
            if name == "Sade-Feldman":
                adata = harmonize_response(adata)
            adata = _ensure_umap_and_clusters(adata)
            ct_col = _find_celltype_col(adata.obs)
            pid = _pid_col(adata.obs)
            vis = _visit_col(adata.obs)
            arm = _arm_col(adata.obs)

            loaded[name] = {
                "adata": adata,
                "ct_col": ct_col,
                "pid_col": pid,
                "visit_col": vis,
                "arm_col": arm,
            }
            ct_info = f", ct_col={ct_col}" if ct_col else ", no cell types"
            print(f"  {name}: {adata.n_obs:,} cells{ct_info}")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")

    if not loaded:
        print("  No datasets available; skipping.")
        return

    # Panel A: UMAP grid coloured by cell type (5 mini-UMAPs)
    n_ds = len(loaded)
    ncols = min(n_ds, 3)
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    _panel_umap_grid(fig, axes_flat, loaded)
    fig.suptitle("Cell-Type UMAPs", fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: UMAP grid coloured by grouping variable (arm/visit)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    _panel_umap_grouping(fig, axes_flat, loaded)
    fig.suptitle("UMAPs by Grouping Variable", fontweight="bold",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: Marker gene dot plot
    fig, ax = plt.subplots(figsize=(10, 7))
    _panel_marker_dotplot(fig, ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: Embedding quality — silhouette + kNN purity (1×2)
    fig = plt.figure(figsize=(14, 5))
    _panel_embedding_quality(fig, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: Cell-type × dataset cross-tabulation heatmap
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_ct_crosstab(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: Quantitative baseline arm overlap (kNN mixing score)
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_arm_mixing(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Panel G: Per-dataset parallel categories (separate PNGs)
    _panel_baseline_ct_by_arm_separate(loaded)

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
