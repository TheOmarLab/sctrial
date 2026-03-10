"""
Supplementary Figure 2 — Cell Annotation and Baseline Comparability.
====================================================================

Establish cell-type annotation quality and demonstrate that pre-treatment
groups are comparable before DiD analysis.

Panels:
  A  UMAP per dataset coloured by cell type (5 mini-UMAPs).
  B  Marker gene dot plot (top 3 markers per cell type, pooled).
  C  Cluster→cell-type purity per dataset.
  D  Centroid-silhouette score on PCA/Harmony embedding.
  E  Embedding neighbourhood purity from full kNN graph.
  F  Cell-type × dataset cross-tabulation heatmap (normalised).
  G  Baseline PCA overlap between arms per dataset.
  H  Baseline cell-type composition by arm.

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


def _ct_col(obs):
    for c in ("cell_type", "celltype", "CellType", "cell_type_fine",
              "cell_type_coarse", "celltype_major", "clustnm"):
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


# ── Panel B: Marker gene dot plot ─────────────────────────────────

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


# ── Panel C: cluster→cell-type purity ─────────────────────────────

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
    order = [n for n in loaded.keys() if n in df["Dataset"].values]
    sns.boxplot(data=df, x="Dataset", y="Cluster purity",
                order=order, palette=_DS_PALETTE,
                fliersize=0, linewidth=0.8, width=0.5, ax=ax)
    sns.stripplot(data=df, x="Dataset", y="Cluster purity",
                  order=order, palette=_DS_PALETTE,
                  size=1.5, alpha=0.25, jitter=0.2, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Dominant cell-type fraction within Leiden cluster")
    ax.set_title("Cluster→Cell-Type Purity", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
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

    order = [c for c in HARMONIZED_CELLTYPE_ORDER if c in pivot.index]
    pivot = pivot.loc[order]

    sns.heatmap(pivot, cmap="YlOrBr", annot=True, fmt=".2f",
                linewidths=0.5, ax=ax, vmin=0, vmax=pivot.values.max(),
                cbar_kws={"label": "Fraction", "shrink": 0.7})
    ax.set_title("Cell-Type × Dataset (harmonised)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    despine(ax)


# ── Panel G: Baseline PCA overlap between arms ───────────────────

def _panel_baseline_pca(fig_parent, axes, loaded: dict):
    """PCA of baseline (pre-treatment) cells coloured by arm."""
    import scipy.sparse as sp

    ds_with_baseline = []
    for name, data in loaded.items():
        vis = data.get("visit_col")
        arm = data.get("arm_col")
        if arm is None:
            continue  # Need arm to show overlap
        obs = data["adata"].obs
        if vis:
            pre_mask = obs[vis].astype(str).str.lower().isin(
                ["pre", "baseline", "d0", "day0", "0", "d000"])
            if pre_mask.sum() > 50:
                ds_with_baseline.append((name, True))  # has baseline
        else:
            # No visit column: use all cells as baseline
            if obs.shape[0] > 50:
                ds_with_baseline.append((name, False))  # use all cells

    n_baseline = len(ds_with_baseline)

    for ax_i, ax in enumerate(axes):
        if ax_i >= n_baseline:
            ax.axis("off")
            continue

        name, has_visit = ds_with_baseline[ax_i]
        data = loaded[name]
        adata = data["adata"]
        obs = adata.obs
        arm = data["arm_col"]

        if has_visit:
            vis = data["visit_col"]
            pre_mask = obs[vis].astype(str).str.lower().isin(
                ["pre", "baseline", "d0", "day0", "0", "d000"])
            adata_pre = adata[pre_mask]
        else:
            adata_pre = adata  # use all cells

        if "X_pca" in adata_pre.obsm:
            pca = adata_pre.obsm["X_pca"][:, :2]
        else:
            for layer in ("log1p_tpm", "log1p_cpm", "log1p_norm"):
                if layer in adata_pre.layers:
                    X = adata_pre.layers[layer]
                    break
            else:
                if "counts" in adata_pre.layers:
                    X = adata_pre.layers["counts"]
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes)
                    continue

            if sp.issparse(X):
                X = X.toarray()
            var_genes = np.var(X, axis=0)
            top_genes = np.argsort(var_genes)[-500:]
            X_sub = X[:, top_genes]
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42).fit_transform(X_sub)

        arms = adata_pre.obs[arm].astype(str).values
        unique_arms = sorted(set(arms))
        arm_palette = dict(zip(unique_arms,
                                sns.color_palette("Set1", len(unique_arms))))

        rng = np.random.default_rng(42)
        order = rng.permutation(len(arms))

        for a in unique_arms:
            mask = arms[order] == a
            ax.scatter(pca[order[mask], 0], pca[order[mask], 1],
                       c=[arm_palette[a]], s=3, alpha=0.5, label=a,
                       edgecolors="none", rasterized=True)

        ax.set_title(f"{name} (baseline)", fontweight="bold", fontsize=11)
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.legend(fontsize=8, loc="best", frameon=True, markerscale=2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)


# ── Panel H: Baseline cell-type composition by arm ───────────────

def _panel_baseline_ct_by_arm(fig_parent, loaded: dict):
    """Parallel-categories plot: Dataset → Arm → Cell type (baseline)."""
    frac_rows: list[dict] = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        vis = data.get("visit_col")
        arm = data.get("arm_col")
        ct = data.get("ct_col")
        if arm is None or ct is None:
            continue

        if vis:
            pre_mask = obs[vis].astype(str).str.lower().isin(
                ["pre", "baseline", "d0", "day0", "0", "d000"])
            if pre_mask.sum() < 50:
                continue
            sub = obs.loc[pre_mask]
        else:
            sub = obs

        # Vectorised: build cell-type fractions per arm
        mapped = sub[ct].astype(str).map(harmonize_celltype)
        for a in sorted(sub[arm].dropna().unique()):
            a_mask = sub[arm] == a
            ct_frac = mapped[a_mask].value_counts(normalize=True)
            for c, f in ct_frac.items():
                frac_rows.append({"Dataset": name, "Arm": str(a),
                                  "Cell type": str(c), "Fraction": f})

    if not frac_rows:
        ax = fig_parent.add_subplot(111)
        ax.text(0.5, 0.5, "No baseline cell-type data", ha="center",
                va="center", transform=ax.transAxes, fontsize=10,
                fontstyle="italic")
        ax.set_title("Baseline Cell-Type Composition", fontweight="bold")
        return

    df = pd.DataFrame(frac_rows)

    # Build cell-type palette
    present = set(df["Cell type"].unique())
    all_cts = [c for c in HARMONIZED_CELLTYPE_ORDER if c in present]
    for c in sorted(present - set(all_cts)):
        all_cts.append(c)
    pal = sns.color_palette("tab20", max(len(all_cts), 1))
    ct_palette = dict(zip(all_cts, pal))

    # Build stacked horizontal bar
    ds_arm_groups = sorted(
        df[["Dataset", "Arm"]].drop_duplicates().values.tolist(),
        key=lambda x: (x[0], x[1]),
    )
    ax = fig_parent.add_subplot(111)

    y_pos = np.arange(len(ds_arm_groups))
    for gi, (ds, arm_val) in enumerate(ds_arm_groups):
        gsub = df[(df["Dataset"] == ds) & (df["Arm"] == arm_val)]
        left = 0.0
        for ct_name in all_cts:
            frac = gsub.loc[gsub["Cell type"] == ct_name, "Fraction"].sum()
            if frac > 0:
                ax.barh(gi, frac, left=left, height=0.65,
                        color=ct_palette[ct_name], edgecolor="white",
                        linewidth=0.3)
                if frac > 0.05:
                    ax.text(left + frac / 2, gi, ct_name, ha="center",
                            va="center", fontsize=5.5, color="black",
                            fontweight="bold")
                left += frac

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{ds}\n{arm_val}" for ds, arm_val in ds_arm_groups],
                       fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of cells", fontsize=9)
    ax.set_title("Baseline Cell-Type Composition by Arm", fontweight="bold")

    # Legend for top cell types
    top_cts = (df.groupby("Cell type")["Fraction"].sum()
               .sort_values(ascending=False).head(12).index)
    handles = [mpatches.Patch(facecolor=ct_palette[c], label=c)
               for c in all_cts if c in top_cts]
    ax.legend(handles=handles, fontsize=6, loc="lower right", frameon=True,
              ncol=2, title="Cell type", title_fontsize=7)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 2 panels (A–H)."""
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

    # Panel B: Marker dot plot
    fig, ax = plt.subplots(figsize=(10, 7))
    _panel_marker_dotplot(fig, ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: Cluster purity
    fig, ax = plt.subplots(figsize=(8, 5))
    _panel_annotation_confidence(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: Embedding silhouette
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_silhouette(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: kNN graph purity
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_knn_purity(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: Cell-type cross-tab
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_ct_crosstab(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Panel G: Baseline PCA overlap (faceted)
    n_baseline = sum(1 for data in loaded.values()
                     if data.get("arm_col"))
    ncols_bl = min(n_baseline, 3) if n_baseline > 0 else 1
    nrows_bl = max(1, (n_baseline + ncols_bl - 1) // ncols_bl)
    fig, axes = plt.subplots(nrows_bl, ncols_bl,
                              figsize=(5.5 * ncols_bl, 5.0 * nrows_bl))
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    else:
        axes = axes.ravel()
    _panel_baseline_pca(fig, axes, loaded)
    fig.suptitle("Baseline PCA by Arm", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # Panel H: Baseline cell-type composition by arm (parallel categories)
    fig = plt.figure(figsize=(11, 7))
    _panel_baseline_ct_by_arm(fig, loaded)
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
