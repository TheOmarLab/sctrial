"""
Supplementary Figure 2 — Cell Annotation and Baseline Comparability.
====================================================================

Establish cell-type annotation quality and demonstrate that pre-treatment
groups are comparable before DiD analysis.

Panels:
  A  UMAP per dataset coloured by cell type (mini-UMAPs).
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
    get_aml,
    get_cart,
    get_tnbc_zhang,
    save_panel,
)

FIGURE_NAME = "SuppFig2_annotation_baseline"

DOT_SIZE = 1.8

_DS_PALETTE = dict(zip(
    ["TNBC", "Melanoma", "COVID-19", "Vaccine", "AML", "CAR-T"],
    ["#996633", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"],
))

DATASETS = [
    ("TNBC", lambda: get_tnbc_zhang()),
    ("Melanoma", get_sade_feldman),
    ("COVID-19", get_stephenson),
    ("Vaccine", get_vaccine),
    ("AML", lambda: get_aml()),
    ("CAR-T", lambda: get_cart()),
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

    # Always recompute neighbors uniformly (n_neighbors=15, same rep) so that
    # purity and silhouette scores are comparable across datasets.
    use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    print("    (Re)computing neighbors uniformly (n_neighbors=15)...")
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
    for c in ("arm", "response", "severity", "therapy", "condition"):
        if c in obs.columns and obs[c].nunique() > 1:
            return c
    return None


# ── Panel A: UMAP per dataset (mini-UMAPs) ─────────────────────

def _panel_umap_grid(fig, axes, loaded: dict):
    """Mini-UMAPs coloured by cell type."""
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
        ax.legend(handles=handles, fontsize=6, loc="best", frameon=True,
                  framealpha=0.8, ncol=1, handlelength=0.8, handleheight=0.7,
                  borderpad=0.4, labelspacing=0.2, markerscale=0.5)

        ax.set_title(name, fontweight="bold", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)

    # Hide unused axes
    for i in range(len(ds_names), len(axes)):
        axes[i].axis("off")


# ── Panel B: UMAP per dataset coloured by grouping variable ──────

def _panel_umap_grouping(fig, axes, loaded: dict):
    """Mini-UMAPs coloured by arm_col or visit_col."""
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
        ax.legend(handles=handles, fontsize=6.5, loc="best", frameon=True,
                  framealpha=0.8, handlelength=0.8, handleheight=0.7,
                  borderpad=0.4, labelspacing=0.2)

        grp_label = "arm" if arm else "visit"
        ax.set_title(f"{name} ({grp_label})", fontweight="bold", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)

    for i in range(len(ds_names), len(axes)):
        axes[i].axis("off")


# ── Panel C: Marker gene dot plot ─────────────────────────────────

def _panel_marker_dotplot(fig, ax, loaded: dict):
    """Top 3 marker genes per cell type, dot plot.

    Color = mean log-expression (averaged across datasets).
    Size  = fraction of cells expressing the marker.
    """
    # canonical_markers keys must match HARMONIZED_CELLTYPE_ORDER labels
    canonical_markers = {
        "T other": ["CD3D", "CD3E", "TRAC"],
        "CD8+ T": ["CD8A", "CD8B", "GZMB"],
        "CD4+ T": ["CD4", "IL7R", "CCR7"],
        "B cell": ["CD19", "MS4A1", "CD79A"],
        "NK": ["NKG7", "GNLY", "KLRD1"],
        "Monocyte": ["CD14", "LYZ", "S100A9"],
        "DC": ["FCER1A", "CLEC10A", "CD1C"],
        "Plasma": ["MZB1", "JCHAIN", "SDC1"],
    }
    all_markers = [m for mks in canonical_markers.values() for m in mks]

    import scipy.sparse as sp

    rows = []
    for name, data in loaded.items():
        adata = data["adata"]
        X = _get_log_expr(adata)
        if X is None:
            continue
        gene_names = list(adata.var_names)
        gene_idx_map = {g.upper(): i for i, g in enumerate(gene_names)}

        ct_col = data["ct_col"]
        if ct_col is None:
            continue

        raw_labels = adata.obs[ct_col].astype(str).values
        harm_labels = np.array([harmonize_celltype(lbl) for lbl in raw_labels])

        if sp.issparse(X):
            X_arr = X.toarray()
        else:
            X_arr = np.asarray(X, dtype=float)

        for marker in all_markers:
            idx = gene_idx_map.get(marker.upper())
            if idx is None:
                continue
            raw_col = X_arr[:, idx].ravel()

            for ct_label in HARMONIZED_CELLTYPE_ORDER:
                mask = harm_labels == ct_label
                if mask.sum() < 10:
                    continue
                raw_vals = raw_col[mask]
                rows.append({
                    "Dataset": name,
                    "Marker": marker,
                    "Cell type": ct_label,
                    "Mean expr": float(raw_vals.mean()),
                    "Fraction expressing": float((raw_vals > 0).mean()),
                })

    if not rows:
        ax.text(0.5, 0.5, "No marker data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        ax.set_title("Marker Gene Expression", fontweight="bold")
        return

    # TEMP DIAGNOSTIC: confirm TNBC actually contributed rows to panel C.
    _tnbc_rows = [r for r in rows if r["Dataset"] == "TNBC"]
    print(f"    [panel C diagnostic] TNBC contributed {len(_tnbc_rows)} "
          f"(marker, cell type) rows out of {len(rows)} total.")
    if _tnbc_rows:
        print(f"    [panel C diagnostic] TNBC sample rows: {_tnbc_rows[:3]}")

    df = pd.DataFrame(rows)
    agg = df.groupby(["Marker", "Cell type"]).agg(
        mean_expr=("Mean expr", "mean"),
        frac_expr=("Fraction expressing", "mean"),
    ).reset_index()

    ct_used = [ct for ct in HARMONIZED_CELLTYPE_ORDER if ct in agg["Cell type"].values]
    marker_order = [m for mks in canonical_markers.values() for m in mks
                    if m in agg["Marker"].values]
    markers_used = list(dict.fromkeys(marker_order))

    if not ct_used or not markers_used:
        ax.text(0.5, 0.5, "No overlap", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        return

    pivot_expr = agg.pivot_table(
        index="Cell type", columns="Marker", values="mean_expr", fill_value=0)
    pivot_frac = agg.pivot_table(
        index="Cell type", columns="Marker", values="frac_expr", fill_value=0)

    pivot_expr = pivot_expr.reindex(index=ct_used, columns=markers_used, fill_value=0)
    pivot_frac = pivot_frac.reindex(index=ct_used, columns=markers_used, fill_value=0)

    expr_max = max(float(pivot_expr.values.max()), 0.01)

    sc_handle = None
    for yi, ct in enumerate(ct_used):
        for xi, marker in enumerate(markers_used):
            frac = float(pivot_frac.loc[ct, marker])
            expr = float(pivot_expr.loc[ct, marker])
            if frac > 0.01:
                size = max(frac * 200, 2)
                sc_handle = ax.scatter(
                    xi, yi, s=size, c=expr,
                    cmap="Reds", vmin=0, vmax=expr_max,
                    edgecolors="grey", linewidth=0.3, zorder=3,
                )

    if sc_handle is not None:
        cbar = fig.colorbar(sc_handle, ax=ax, shrink=0.4, pad=0.02, aspect=15)
        cbar.set_label("Mean log\nexpression", fontsize=5)
        cbar.ax.tick_params(labelsize=5)

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
            conn.eliminate_zeros()  # ensure .data matches structural nnz exactly
            labels = pd.Categorical(adata.obs[ct_col].astype(str)).codes
            # Vectorised kNN purity: build same-label weight matrix, row-sum
            label_arr = np.asarray(labels)
            # Derive row/col indices from the CSR structure itself so they are
            # guaranteed to align 1:1 with conn.data (conn.nonzero() instead
            # value-filters and can return a different length if a matrix has
            # explicit zero entries stored, causing a shape mismatch).
            cols = conn.indices
            rows = np.repeat(np.arange(conn.shape[0]), np.diff(conn.indptr))
            same_label = label_arr[rows] == label_arr[cols]
            # Build sparse matrix of same-label weights only
            from scipy import sparse as sp
            same_w = sp.csr_matrix(
                (conn.data * same_label, (rows, cols)), shape=conn.shape
            )
            total_w = np.asarray(conn.sum(axis=1)).ravel()
            same_w_sum = np.asarray(same_w.sum(axis=1)).ravel()
            # Mask out cells with no neighbours
            valid = total_w > 0
            if not valid.any():
                continue
            purity = float(np.mean(same_w_sum[valid] / total_w[valid]))
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
    (k=15) in PCA space, compute participant-weighted fraction of neighbors
    from the opposite arm.  The expected value under random mixing is
    2*p*(1-p) (for two arms with proportions p and 1-p), NOT 0.5, which
    only holds when arms are equal-sized.  Each dataset gets its own
    dataset-specific reference line.
    """
    from sklearn.neighbors import NearestNeighbors

    ds_names = []
    mix_scores = []
    null_lines = []  # dataset-specific expected mixing under random assignment

    for name, data in loaded.items():
        arm = data.get("arm_col")
        vis = data.get("visit_col")
        pid = data.get("pid_col")
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

        # Build kNN (k=15)
        nn = NearestNeighbors(n_neighbors=min(16, len(idx)),
                              metric="euclidean", algorithm="auto")
        nn.fit(emb)
        _, nbr_idx = nn.kneighbors(emb)
        nbr_idx = nbr_idx[:, 1:]  # exclude self

        # Participant-weighted mixing: average per-participant mean mixing score
        obs_sub = obs.iloc[idx].copy()
        obs_sub["_mix"] = [float(np.mean(arms[nbr_idx[i]] != arms[i]))
                           for i in range(len(idx))]
        if pid and pid in obs_sub.columns:
            pid_scores = obs_sub.groupby(pid)["_mix"].mean()
            mix_score = float(pid_scores.mean())
        else:
            mix_score = float(obs_sub["_mix"].mean())

        # Expected mixing under random assignment: 2*p*(1-p)
        # Use participant-level arm proportions to match the observed
        # score which is also participant-weighted.
        if pid and pid in obs_sub.columns:
            pid_arms = obs_sub.groupby(pid)[arm].first()
            arm_counts = pid_arms.value_counts(normalize=True)
        else:
            arm_counts = pd.Series(arms).value_counts(normalize=True)
        p = float(arm_counts.iloc[0])
        expected_null = 2.0 * p * (1.0 - p)

        ds_names.append(name)
        mix_scores.append(mix_score)
        null_lines.append(expected_null)

    if not ds_names:
        ax.text(0.5, 0.5, "No arm-overlap data", ha="center", va="center",
                transform=ax.transAxes)
        return

    x_pos = np.arange(len(ds_names))
    colors = [_DS_PALETTE.get(n, "grey") for n in ds_names]
    bars = ax.bar(x_pos, mix_scores, color=colors, edgecolor="white", width=0.6)
    for bar, s in zip(bars, mix_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.01, f"{s:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Per-dataset expected null line (tick marks spanning the bar width)
    bar_w = 0.6
    for xi, null in enumerate(null_lines):
        ax.plot([xi - bar_w / 2, xi + bar_w / 2], [null, null],
                color="black", linewidth=0.6, linestyle="--", zorder=4)
    # Add a single legend entry for the null lines
    ax.plot([], [], color="black", linewidth=0.6, linestyle="--",
            label="Expected (random, 2p(1−p))")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ds_names)
    ax.set_ylabel("Arm-mixing score (participant-weighted)")
    ax.set_title("Baseline Arm Overlap (kNN mixing, k=15)", fontweight="bold")
    ax.set_ylim(0, min(max(mix_scores + null_lines) * 1.3, 1.0) if mix_scores else 1)
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

    # Enforce manuscript dataset order in columns
    ds_order = [name for name, _ in DATASETS if name in pivot.columns]
    pivot = pivot.reindex(columns=ds_order, fill_value=0)

    order = [c for c in HARMONIZED_CELLTYPE_ORDER if c in pivot.index]
    pivot = pivot.loc[order]

    sns.heatmap(pivot, cmap="YlOrBr", annot=True, fmt=".2f",
                linewidths=0.5, ax=ax, vmin=0, vmax=pivot.values.max(),
                cbar_kws={"label": "Fraction", "shrink": 0.7})
    ax.set_title("Cell-Type × Dataset (harmonised)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    despine(ax)


# ── Panel G: Baseline cell-type composition — plotly parallel categories ──


def _make_plotly_parcats(ds_name: str, obs_sub, right_col: str, ct_col: str,
                         right_label: str = "Arm", pid_col: str | None = None):
    """Build a plotly parallel-categories figure for one dataset.

    Returns a plotly Figure or None if insufficient data.
    """
    import plotly.express as px

    mapped = obs_sub[ct_col].astype(str).map(harmonize_celltype)
    right_vals = obs_sub[right_col].astype(str)

    # Build per-cell dataframe with harmonised labels
    df = pd.DataFrame({"Cell type": mapped.values, right_label: right_vals.values})
    if pid_col and pid_col in obs_sub.columns:
        df["pid"] = obs_sub[pid_col].values

    if len(df) == 0:
        return None

    # Cap at top 15 cell types
    max_ct = 15
    ct_order = df["Cell type"].value_counts().index.tolist()
    if len(ct_order) > max_ct:
        top_set = set(ct_order[:max_ct])
        df.loc[~df["Cell type"].isin(top_set), "Cell type"] = "Other"

    # Participant-weighted counts per flow
    if pid_col and "pid" in df.columns:
        flow = (
            df.groupby(["Cell type", right_label])["pid"]
            .nunique()
            .reset_index(name="count")
        )
    else:
        flow = df.groupby(["Cell type", right_label]).size().reset_index(name="count")

    # Expand flow counts back to a long dataframe for px.parallel_categories
    # (plotly weights by row count, so repeat rows by participant count)
    rows = []
    for _, r in flow.iterrows():
        for _ in range(int(r["count"])):
            rows.append({"Cell type": r["Cell type"], right_label: r[right_label]})
    df_expanded = pd.DataFrame(rows)
    if len(df_expanded) == 0:
        return None

    # Color by cell type using Dark24 palette
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    ct_unique = sorted(df_expanded["Cell type"].unique())
    ct_color_map = {ct: palette[i % len(palette)] for i, ct in enumerate(ct_unique)}
    color_vals = df_expanded["Cell type"].map(ct_color_map)

    fig = px.parallel_categories(
        df_expanded,
        dimensions=["Cell type", right_label],
        color=color_vals,
        #title=ds_name,
        width=850,
        height=720,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        font=dict(size=32),
        #title=dict(text=ds_name, font=dict(size=22, family="Arial"),
        #           x=0.5, xanchor="center", y=0.98, yanchor="top"),
        margin=dict(l=160, r=100, t=70, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _panel_baseline_ct_by_arm_separate(loaded: dict):
    """Per-dataset parallel categories saved as HTML + PNG via plotly.

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
        fig = _make_plotly_parcats(
            ds_name, sub, right_col, ct_col,
            right_label=right_label, pid_col=pid_col,
        )
        if fig is None:
            continue
        safe_name = ds_name.replace(" ", "_").replace("-", "_")
        # Save interactive HTML
        html_path = panel_dir / f"panel_G_{safe_name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        # Save static PNG (600 DPI equivalent at 3× scale)
        png_path = panel_dir / f"panel_G_{safe_name}.png"
        fig.write_image(str(png_path), format="png", scale=3)
        print(f"    Saved panel: panel_G_{safe_name}")


# ======================================================================
# Composite  (180 mm × ≤215 mm)
# ======================================================================
#   Row 0: A  — UMAP by cell type (1 × n_ds)
#   Row 2: B  — UMAP by grouping  (1 × n_ds)
#   Row 4: C (marker dot plot, left) | D (silhouette + purity, right)
#   Row 6: E (ct × ds heatmap, left) | F (arm mixing, right)
#   Row 8: G  — parallel-categories PNGs (1 × n_ds)
# ======================================================================

_SMALL_RC = {
    "font.size": 4.5,
    "axes.titlesize": 5,
    "axes.labelsize": 4.5,
    "xtick.labelsize": 4,
    "ytick.labelsize": 4,
    "legend.fontsize": 2.5,
    "legend.title_fontsize": 2.5,
}
_MAX_FONT = 5.5


def _cap_fontsize(fig_obj, maximum):
    """Clamp all text elements in *fig_obj* to *maximum* pt."""
    for ax_i in fig_obj.get_axes():
        for txt in ([ax_i.title, ax_i.xaxis.label, ax_i.yaxis.label]
                    + ax_i.get_xticklabels() + ax_i.get_yticklabels()
                    + ax_i.texts):
            if txt.get_fontsize() > maximum:
                txt.set_fontsize(maximum)
        leg = ax_i.get_legend()
        if leg:
            for txt in leg.get_texts():
                if txt.get_fontsize() > maximum:
                    txt.set_fontsize(maximum)
            t = leg.get_title()
            if t and t.get_fontsize() > maximum:
                t.set_fontsize(maximum)


def _build_composite(loaded: dict):
    """Assemble all panels A–G into a single 180 mm × ≤215 mm figure."""
    import matplotlib.image as mpimg

    print("  Building composite figure (panels A–G) ...")
    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm = 1.0 / 25.4
    n_ds = len(loaded)
    fig_c = plt.figure(figsize=(180 * _mm, 215 * _mm))

    outer = fig_c.add_gridspec(
        9, 1,
        height_ratios=[
            0.28,   # row 0: A (UMAPs cell type)
            0.17,   # spacer (room for A legends below)
            0.28,   # row 2: B (UMAPs grouping)
            0.09,   # spacer (B → C|D)
            0.36,   # row 4: C | D  (reduced height)
            0.14,   # spacer (C|D → E|F)
            0.32,   # row 6: E | F  (reduced height)
            0.07,   # spacer (E|F → G)
            0.38,   # row 8: G (PNGs)
        ],
        hspace=0.0,
        left=0.06, right=0.98, top=0.97, bottom=0.03,
    )

    # ── Row 0: A — UMAPs coloured by cell type ─────────────────────
    gs_a = outer[0].subgridspec(1, n_ds, wspace=0.30)
    axes_a = [fig_c.add_subplot(gs_a[0, i]) for i in range(n_ds)]
    _panel_umap_grid(fig_c, axes_a, loaded)
    for _ax in axes_a:
        for _coll in _ax.collections:
            if hasattr(_coll, "set_sizes"):
                _coll.set_sizes([0.15])
        _leg = _ax.get_legend()
        if _leg:
            _handles = _leg.legend_handles
            _labels = [t.get_text() for t in _leg.get_texts()]
            _leg.remove()
            _ax.legend(
                handles=_handles, labels=_labels,
                fontsize=3.5, loc="upper center",
                bbox_to_anchor=(0.5, -0.02), ncol=2,
                frameon=False, handlelength=0.6, handleheight=0.5,
                borderpad=0.1, labelspacing=0.12, columnspacing=0.4,
                markerscale=0.35,
            )

    # ── Row 2: B — UMAPs coloured by grouping variable ─────────────
    gs_b = outer[2].subgridspec(1, n_ds, wspace=0.30)
    axes_b = [fig_c.add_subplot(gs_b[0, i]) for i in range(n_ds)]
    _panel_umap_grouping(fig_c, axes_b, loaded)
    for _ax in axes_b:
        for _coll in _ax.collections:
            if hasattr(_coll, "set_sizes"):
                _coll.set_sizes([0.15])
        _leg_b = _ax.get_legend()
        if _leg_b:
            for _txt in _leg_b.get_texts():
                _txt.set_fontsize(3.5)

    # ── Row 4: C (marker dot plot, left) | D (sil + purity, right) ─
    gs_cd = outer[4].subgridspec(1, 2, width_ratios=[0.55, 0.45],
                                 wspace=0.15)
    ax_c = fig_c.add_subplot(gs_cd[0])
    _panel_marker_dotplot(fig_c, ax_c, loaded)
    ax_c.tick_params(axis="x", labelsize=4)
    ax_c.tick_params(axis="y", labelsize=4)
    for _coll in ax_c.collections:
        if hasattr(_coll, "get_sizes") and hasattr(_coll, "set_sizes"):
            _coll.set_sizes(_coll.get_sizes() * 0.35)
    _leg_c = ax_c.get_legend()
    if _leg_c:
        for _lh in _leg_c.legend_handles:
            if hasattr(_lh, "set_sizes"):
                _lh.set_sizes([s * 0.1 for s in _lh.get_sizes()])
        for _txt in _leg_c.get_texts():
            _txt.set_fontsize(3.5)
        _t_c = _leg_c.get_title()
        if _t_c:
            _t_c.set_fontsize(3.5)
        _leg_c.set_bbox_to_anchor((1.02, 1.05))
        _leg_c._loc = 1  # top right, above colorbar

    gs_d = gs_cd[1].subgridspec(2, 1, hspace=0.80)
    ax_d1 = fig_c.add_subplot(gs_d[0])
    ax_d2 = fig_c.add_subplot(gs_d[1])
    _panel_silhouette(ax_d1, loaded)
    _panel_knn_purity(ax_d2, loaded)
    ax_d1.set_ylabel("Centroid-silhouette\nscore", fontsize=3.5)
    ax_d2.set_ylabel("Graph label purity\n(connectivities)", fontsize=3.5)

    # ── Row 6: E (ct × ds heatmap, left) | F (arm mixing, right) ──
    gs_ef = outer[6].subgridspec(1, 2, width_ratios=[0.55, 0.45],
                                 wspace=0.15)
    ax_e = fig_c.add_subplot(gs_ef[0])
    _panel_ct_crosstab(ax_e, loaded)
    ax_f = fig_c.add_subplot(gs_ef[1])
    _panel_arm_mixing(ax_f, loaded)
    for _txt in ax_f.texts:
        _txt.set_fontsize(4.5)
    _leg_f = ax_f.get_legend()
    if _leg_f:
        for _txt in _leg_f.get_texts():
            _txt.set_fontsize(4.5)

    # ── Row 8: G — parallel-categories PNGs ────────────────────────
    panel_dir = SUPP_OUTPUT / f"{FIGURE_NAME}_panels"
    ds_names = list(loaded.keys())
    gs_g = outer[8].subgridspec(1, n_ds, wspace=0.05)
    axes_g: list[plt.Axes] = []
    for i, name in enumerate(ds_names):
        ax_gi = fig_c.add_subplot(gs_g[0, i])
        axes_g.append(ax_gi)
        safe_name = name.replace(" ", "_").replace("-", "_")
        png_path = panel_dir / f"panel_G_{safe_name}.png"
        if png_path.exists():
            img = mpimg.imread(str(png_path))
            ax_gi.imshow(img)
            ax_gi.set_title(name, fontweight="bold", fontsize=6, pad=2)
        else:
            ax_gi.text(0.5, 0.5, f"{name}\n(PNG not found)",
                       ha="center", va="center",
                       transform=ax_gi.transAxes, fontsize=4)
        ax_gi.set_xticks([])
        ax_gi.set_yticks([])
        ax_gi.axis("off")

    # ── Post-processing ────────────────────────────────────────────
    _cap_fontsize(fig_c, _MAX_FONT)

    _lbl_fs = 8
    _lbl_x = -0.12
    _lbl_y = 1.15

    _label_pairs: list[tuple] = [
        (axes_a[0], "A"),
        (axes_b[0], "B"),
        (ax_c, "C"),
        (ax_d1, "D"),
        (ax_e, "E"),
        (ax_f, "F"),
    ]
    if axes_g:
        _label_pairs.append((axes_g[0], "G"))

    for ax_lbl, lbl in _label_pairs:
        ax_lbl.text(
            _lbl_x, _lbl_y, lbl,
            transform=ax_lbl.transAxes,
            fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
        )

    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, SUPP_OUTPUT, close=False)
    pdf_path = panel_dir / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 2 panels (A–G) + composite."""
    print("Supplementary Figure 2: Cell Annotation and Baseline Comparability")

    loaded = {}
    for name, loader in DATASETS:
        try:
            adata = loader()
            if name == "Melanoma":
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

    # Panel A: UMAP grid coloured by cell type (mini-UMAPs)
    n_ds = len(loaded)
    ncols = min(n_ds, 3)
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    _panel_umap_grid(fig, axes_flat, loaded)
    fig.suptitle("Cell-Type UMAPs", fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Save individual cell-type UMAPs (legend below figure, 2 cols)
    for name, data in loaded.items():
        fig_ind, ax_ind = plt.subplots(figsize=(5.5, 6.0))
        _panel_umap_grid(fig_ind, [ax_ind], {name: data})
        _leg_ind = ax_ind.get_legend()
        if _leg_ind:
            _h = _leg_ind.legend_handles
            _l = [t.get_text() for t in _leg_ind.get_texts()]
            _leg_ind.remove()
            ax_ind.legend(
                handles=_h, labels=_l,
                fontsize=7, loc="upper center",
                bbox_to_anchor=(0.5, -0.05), ncol=2,
                frameon=False, handlelength=1.0, handleheight=0.8,
                borderpad=0.3, labelspacing=0.3, columnspacing=0.8,
            )
        fig_ind.tight_layout()
        fig_ind.subplots_adjust(bottom=0.22)
        safe = name.replace(" ", "_").replace("-", "_")
        save_panel(fig_ind, f"panel_A_{safe}", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: UMAP grid coloured by grouping variable (arm/visit)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    _panel_umap_grouping(fig, axes_flat, loaded)
    fig.suptitle("UMAPs by Grouping Variable", fontweight="bold",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Save individual grouping UMAPs
    for name, data in loaded.items():
        fig_ind, ax_ind = plt.subplots(figsize=(5.5, 5.0))
        _panel_umap_grouping(fig_ind, [ax_ind], {name: data})
        fig_ind.tight_layout()
        safe = name.replace(" ", "_").replace("-", "_")
        save_panel(fig_ind, f"panel_B_{safe}", FIGURE_NAME, SUPP_OUTPUT)

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
    try:
        _panel_baseline_ct_by_arm_separate(loaded)
    except Exception as exc:
        print(f"    Skipping plotly panel G PNGs: {exc}")

    # ── Composite artboard ─────────────────────────────────────────
    _build_composite(loaded)

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