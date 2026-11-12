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
    get_aml,
    get_cart,
    get_sade_feldman,
    get_stephenson,
    get_tnbc_zhang,
    get_vaccine,
    harmonize_celltype,
    harmonize_response,
    key_below_axes,
    letter_at_panel_corner,
    report_layout_problems,
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


# ── Prepared panel data ───────────────────────────────────────────
#
# Every statistic is computed once from the loaded AnnData objects; the standalone
# panels and the composite draw from the same prepared values.


def _umap_data(loaded: dict) -> dict:
    """UMAP coordinates with cell-type and grouping labels, per dataset."""
    out = {}
    for name, data in loaded.items():
        adata = data["adata"]
        ct_col = data["ct_col"]
        arm = data.get("arm_col")
        grp_col = arm if arm else data.get("visit_col")
        out[name] = {
            "emb": adata.obsm["X_umap"],
            "ct": adata.obs[ct_col].astype(str).values if ct_col is not None else None,
            "grp": adata.obs[grp_col].astype(str).values if grp_col is not None else None,
            "grp_label": "arm" if arm else "visit",
        }
    return out


# ── Panel A: UMAP per dataset (mini-UMAPs) ─────────────────────

def _panel_umap_grid(fig, axes, umap: dict):
    """Mini-UMAPs coloured by cell type."""
    ds_names = list(umap.keys())
    for i, name in enumerate(ds_names):
        ax = axes[i]
        labels = umap[name]["ct"]

        if labels is None:
            ax.text(0.5, 0.5, "No annotation", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, fontstyle="italic",
                    color="#888888")
            ax.set_title(name, fontweight="bold", fontsize=9)
            ax.axis("off")
            continue

        emb = umap[name]["emb"]
        unique_cts = sorted(set(labels))
        palette = _build_ct_palette(unique_cts)

        rng = np.random.default_rng(42)
        order = rng.permutation(len(labels))

        colors = np.array([palette[labels[j]] for j in order])
        ax.scatter(emb[order, 0], emb[order, 1],
                   c=colors, s=DOT_SIZE, alpha=0.7, edgecolors="none",
                   rasterized=True)

        handles = [mpatches.Patch(facecolor=palette[ct], edgecolor="none",
                                  label=ct) for ct in unique_cts[:12]]
        if len(unique_cts) > 12:
            handles.append(mpatches.Patch(facecolor="grey", edgecolor="none",
                                          label=f"+ {len(unique_cts)-12} more"))
        ax.legend(handles=handles, fontsize=6, loc="upper center",
                  bbox_to_anchor=(0.5, -0.02), frameon=False, ncol=2,
                  handlelength=0.8, handleheight=0.7, borderpad=0.4,
                  labelspacing=0.2, columnspacing=0.8)

        ax.set_title(name, fontweight="bold", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)

    for i in range(len(ds_names), len(axes)):
        axes[i].axis("off")


# ── Panel B: UMAP per dataset coloured by grouping variable ──────

def _panel_umap_grouping(fig, axes, umap: dict):
    """Mini-UMAPs coloured by arm or visit; the key is placed below by the caller."""
    ds_names = list(umap.keys())
    for i, name in enumerate(ds_names):
        ax = axes[i]
        labels = umap[name]["grp"]
        if labels is None:
            ax.text(0.5, 0.5, "No grouping", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, fontstyle="italic",
                    color="#888888")
            ax.set_title(name, fontweight="bold", fontsize=9)
            ax.axis("off")
            continue

        emb = umap[name]["emb"]
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
        ax.legend(handles=handles, fontsize=6.5)

        ax.set_title(f"{name} ({umap[name]['grp_label']})", fontweight="bold", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)

    for i in range(len(ds_names), len(axes)):
        axes[i].axis("off")


# ── Panel C: Marker gene dot plot ─────────────────────────────────

# canonical_markers keys must match HARMONIZED_CELLTYPE_ORDER labels
_CANONICAL_MARKERS = {
    "T other": ["CD3D", "CD3E", "TRAC"],
    "CD8+ T": ["CD8A", "CD8B", "GZMB"],
    "CD4+ T": ["CD4", "IL7R", "CCR7"],
    "B cell": ["CD19", "MS4A1", "CD79A"],
    "NK": ["NKG7", "GNLY", "KLRD1"],
    "Monocyte": ["CD14", "LYZ", "S100A9"],
    "DC": ["FCER1A", "CLEC10A", "CD1C"],
    "Plasma": ["MZB1", "JCHAIN", "SDC1"],
}


def _marker_dotplot_data(loaded: dict):
    """Mean log-expression and fraction expressing per marker x harmonised cell type.

    Each dataset contributes its own mean and fraction; the panel shows their average
    across datasets. Returns ``(pivot_expr, pivot_frac)``, or a message string when there
    is nothing to draw.
    """
    import scipy.sparse as sp

    all_markers = [m for mks in _CANONICAL_MARKERS.values() for m in mks]
    rows = []
    for name, data in loaded.items():
        adata = data["adata"]
        X = _get_log_expr(adata)
        if X is None:
            continue
        if sp.issparse(X) and X.format == "coo":
            X = X.tocsr()
        gene_idx_map = {g.upper(): i for i, g in enumerate(adata.var_names)}

        ct_col = data["ct_col"]
        if ct_col is None:
            continue

        raw_labels = adata.obs[ct_col].astype(str).values
        harm_labels = np.array([harmonize_celltype(lbl) for lbl in raw_labels])

        present = [(m, gene_idx_map[m.upper()]) for m in all_markers
                   if m.upper() in gene_idx_map]
        if not present:
            continue
        cols = [idx for _, idx in present]
        X_sub = X[:, cols]
        X_arr = X_sub.toarray() if sp.issparse(X_sub) else np.asarray(X_sub, dtype=float)

        for j, (marker, _) in enumerate(present):
            raw_col = X_arr[:, j].ravel()
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
        return "No marker data"

    df = pd.DataFrame(rows)
    agg = df.groupby(["Marker", "Cell type"]).agg(
        mean_expr=("Mean expr", "mean"),
        frac_expr=("Fraction expressing", "mean"),
    ).reset_index()

    ct_used = [ct for ct in HARMONIZED_CELLTYPE_ORDER if ct in agg["Cell type"].values]
    marker_order = [m for mks in _CANONICAL_MARKERS.values() for m in mks
                    if m in agg["Marker"].values]
    markers_used = list(dict.fromkeys(marker_order))
    if not ct_used or not markers_used:
        return "No overlap"

    pivot_expr = agg.pivot_table(
        index="Cell type", columns="Marker", values="mean_expr", fill_value=0)
    pivot_frac = agg.pivot_table(
        index="Cell type", columns="Marker", values="frac_expr", fill_value=0)
    pivot_expr = pivot_expr.reindex(index=ct_used, columns=markers_used, fill_value=0)
    pivot_frac = pivot_frac.reindex(index=ct_used, columns=markers_used, fill_value=0)
    return pivot_expr, pivot_frac


def _panel_marker_dotplot(fig, ax, dot):
    """Top 3 marker genes per cell type, dot plot.

    Color = mean log-expression (averaged across datasets).
    Size  = fraction of cells expressing the marker.
    The size key sits right of the axes, below the colour bar, off the data.
    """
    if isinstance(dot, str):
        ax.text(0.5, 0.5, dot, ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        if dot == "No marker data":
            ax.set_title("Marker Gene Expression", fontweight="bold")
        return
    pivot_expr, pivot_frac = dot
    ct_used = list(pivot_expr.index)
    markers_used = list(pivot_expr.columns)

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
        cbar = fig.colorbar(sc_handle, ax=ax, shrink=0.4, pad=0.02, aspect=15,
                            anchor=(0.0, 1.0))
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
              loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=True,
              handletextpad=0.1)

    despine(ax)


# ── Panel D: Silhouette scores ────────────────────────────────────

def _silhouette_scores(loaded: dict) -> tuple[list[str], list[float]]:
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
    return ds_names, scores


def _panel_silhouette(ax, sil: tuple[list[str], list[float]]):
    ds_names, scores = sil
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


# ── Panel D (lower): Embedding neighbourhood purity ──────────────

def _knn_purities(loaded: dict) -> tuple[list[str], list[float]]:
    """Label purity from the full Scanpy neighbour graph (no subsampling)."""
    from scipy import sparse as sp

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
            label_arr = np.asarray(labels)
            # Row/col indices from the CSR structure itself, so they align 1:1 with
            # conn.data (conn.nonzero() value-filters and can return another length).
            cols = conn.indices
            rows = np.repeat(np.arange(conn.shape[0]), np.diff(conn.indptr))
            same_label = label_arr[rows] == label_arr[cols]
            same_w = sp.csr_matrix(
                (conn.data * same_label, (rows, cols)), shape=conn.shape
            )
            total_w = np.asarray(conn.sum(axis=1)).ravel()
            same_w_sum = np.asarray(same_w.sum(axis=1)).ravel()
            valid = total_w > 0
            if not valid.any():
                continue
            purity = float(np.mean(same_w_sum[valid] / total_w[valid]))
            ds_names.append(name)
            purities.append(purity)
        except Exception as exc:
            print(f"    {name}: kNN purity failed ({exc})")
    return ds_names, purities


def _panel_knn_purity(ax, pur: tuple[list[str], list[float]]):
    ds_names, purities = pur
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
    # Headroom above 1 so a value label near 1 stays inside the axes, below the title.
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.5, 1.0])
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel F: Quantitative baseline arm overlap (kNN mixing) ──────

def _arm_mixing_scores(loaded: dict):
    """Baseline kNN arm-mixing score per dataset and its random-mixing expectation.

    For each dataset with an arm column, at baseline: build a kNN graph (k=15) in PCA
    space and compute the participant-weighted fraction of neighbours from the other
    arm. The expected value under random mixing is 2*p*(1-p) for two arms with
    participant proportions p and 1-p (0.5 only when the arms are equal-sized).
    Returns (names, scores, expected).
    """
    from sklearn.neighbors import NearestNeighbors

    ds_names = []
    mix_scores = []
    null_lines = []
    for name, data in loaded.items():
        arm = data.get("arm_col")
        vis = data.get("visit_col")
        pid = data.get("pid_col")
        if arm is None:
            continue
        adata = data["adata"]
        obs = adata.obs

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

        nn = NearestNeighbors(n_neighbors=min(16, len(idx)),
                              metric="euclidean", algorithm="auto")
        nn.fit(emb)
        _, nbr_idx = nn.kneighbors(emb)
        nbr_idx = nbr_idx[:, 1:]  # exclude self

        obs_sub = obs.iloc[idx].copy()
        obs_sub["_mix"] = [float(np.mean(arms[nbr_idx[i]] != arms[i]))
                           for i in range(len(idx))]
        if pid and pid in obs_sub.columns:
            pid_scores = obs_sub.groupby(pid)["_mix"].mean()
            mix_score = float(pid_scores.mean())
        else:
            mix_score = float(obs_sub["_mix"].mean())

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
    return ds_names, mix_scores, null_lines


def _panel_arm_mixing(ax, mixing):
    ds_names, mix_scores, null_lines = mixing
    if not ds_names:
        ax.text(0.5, 0.5, "No arm-overlap data", ha="center", va="center",
                transform=ax.transAxes)
        return

    x_pos = np.arange(len(ds_names))
    colors = [_DS_PALETTE.get(n, "grey") for n in ds_names]
    bars = ax.bar(x_pos, mix_scores, color=colors, edgecolor="white", width=0.6)
    y_top = min(max(mix_scores + null_lines) * 1.3, 1.0)
    ax.set_ylim(0, y_top)
    # A value sits on its bar, unless the label would reach the expected-value line
    # just above the bar; then it sits above the line. The label height is taken as
    # 6% of the axis range (a 7 pt label is under 4% in the standalone panel and 5.5%
    # in the composite).
    label_h = 0.06 * y_top
    for bar, s, null in zip(bars, mix_scores, null_lines):
        y = s + 0.01
        if s < null and y + label_h > null - 0.01:
            y = null + 0.012
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{s:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    bar_w = 0.6
    for xi, null in enumerate(null_lines):
        ax.plot([xi - bar_w / 2, xi + bar_w / 2], [null, null],
                color="black", linewidth=0.6, linestyle="--", zorder=4)
    ax.plot([], [], color="black", linewidth=0.6, linestyle="--",
            label="Expected (random, 2p(1−p))")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ds_names)
    ax.set_ylabel("Arm-mixing score (participant-weighted)")
    ax.set_title("Baseline Arm Overlap (kNN mixing, k=15)", fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel E: Cell type × dataset cross-tabulation ─────────────────

def _celltype_crosstab(loaded: dict):
    """Harmonised cell-type fractions within each dataset (cell type x dataset)."""
    rows = []
    for name, data in loaded.items():
        ct_col = data["ct_col"]
        if ct_col is None:
            continue
        harmonised = data["adata"].obs[ct_col].astype(str).map(harmonize_celltype)
        cts = harmonised.value_counts(normalize=True)
        for ct, frac in cts.items():
            rows.append({"Dataset": name, "Cell type": ct, "Fraction": frac})
    if not rows:
        return None

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Cell type", columns="Dataset",
                           values="Fraction", aggfunc="sum", fill_value=0)
    ds_order = [name for name, _ in DATASETS if name in pivot.columns]
    pivot = pivot.reindex(columns=ds_order, fill_value=0)
    order = [c for c in HARMONIZED_CELLTYPE_ORDER if c in pivot.index]
    return pivot.loc[order]


def _panel_ct_crosstab(ax, pivot):
    """Heatmap: harmonised cell type × dataset (normalised within dataset)."""
    if pivot is None:
        ax.text(0.5, 0.5, "No cell-type data", ha="center", va="center",
                transform=ax.transAxes)
        return
    sns.heatmap(pivot, cmap="YlOrBr", annot=True, fmt=".2f",
                linewidths=0.5, ax=ax, vmin=0, vmax=pivot.values.max(),
                cbar_kws={"label": "Fraction", "shrink": 0.7})
    ax.set_title("Cell-Type × Dataset (harmonised)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    despine(ax)


# ── Panel G: Baseline cell-type composition (participant-weighted alluvial) ──

def _alluvial_flows(obs_sub, right_col: str, ct_col: str, right_label: str,
                    pid_col: str | None) -> pd.DataFrame | None:
    """Participants per (harmonised cell type, arm or visit) flow.

    The 15 most frequent cell types are kept and the rest pooled as "Other". A flow's
    weight is the number of participants with cells of that type in that group (the
    number of cells when no participant column exists).
    """
    mapped = obs_sub[ct_col].astype(str).map(harmonize_celltype)
    df = pd.DataFrame({"Cell type": mapped.values,
                       right_label: obs_sub[right_col].astype(str).values})
    if pid_col and pid_col in obs_sub.columns:
        df["pid"] = obs_sub[pid_col].values
    if len(df) == 0:
        return None

    max_ct = 15
    ct_order = df["Cell type"].value_counts().index.tolist()
    if len(ct_order) > max_ct:
        top_set = set(ct_order[:max_ct])
        df.loc[~df["Cell type"].isin(top_set), "Cell type"] = "Other"

    if pid_col and "pid" in df.columns:
        flow = (df.groupby(["Cell type", right_label])["pid"]
                .nunique().reset_index(name="count"))
    else:
        flow = df.groupby(["Cell type", right_label]).size().reset_index(name="count")
    flow = flow[flow["count"] > 0].reset_index(drop=True)
    return flow if len(flow) else None


def _alluvial_data(loaded: dict) -> dict:
    """Flows for every dataset with cell types.

    Two-arm datasets: cell type to arm, baseline cells only. Single-arm datasets: cell
    type to visit, all cells.
    """
    out = {}
    for name, data in loaded.items():
        ct = data.get("ct_col")
        if ct is None:
            continue
        obs = data["adata"].obs
        arm = data.get("arm_col")
        vis = data.get("visit_col")
        pid = data.get("pid_col")
        if arm is not None:
            if vis:
                pre_mask = obs[vis].astype(str).str.lower().isin(
                    ["pre", "baseline", "d0", "day0", "0", "d000"])
                if pre_mask.sum() < 50:
                    continue
                sub = obs.loc[pre_mask]
            else:
                sub = obs
            right_col, right_label = arm, "Arm"
        elif vis is not None:
            sub, right_col, right_label = obs, vis, "Visit"
        else:
            continue
        flow = _alluvial_flows(sub, right_col, ct, right_label, pid)
        if flow is not None:
            out[name] = {"flow": flow, "right_label": right_label}
    return out


def _alluvial_colors(cell_types) -> dict:
    """Plotly Dark24 then Light24, assigned in sorted cell-type order."""
    import plotly.express as px

    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    return {ct: palette[i % len(palette)] for i, ct in enumerate(sorted(cell_types))}


def _write_alluvial_html(path, flow: pd.DataFrame, right_label: str) -> None:
    """Interactive plotly parallel-categories view of the same flows."""
    import plotly.express as px

    rows = [{"Cell type": r["Cell type"], right_label: r[right_label]}
            for _, r in flow.iterrows() for _ in range(int(r["count"]))]
    df_expanded = pd.DataFrame(rows)
    colors = _alluvial_colors(df_expanded["Cell type"].unique())
    fig = px.parallel_categories(
        df_expanded, dimensions=["Cell type", right_label],
        color=df_expanded["Cell type"].map(colors), width=850, height=720,
    )
    fig.update_layout(coloraxis_showscale=False, font=dict(size=18, color="black"),
                      margin=dict(l=160, r=220, t=70, b=30),
                      paper_bgcolor="white", plot_bgcolor="white")
    fig.write_html(str(path), include_plotlyjs="cdn")


def _spread_labels(y, min_gap: float, lo: float, hi: float) -> np.ndarray:
    """Move label centres apart by at least ``min_gap``, keeping them within [lo, hi]."""
    y = np.asarray(y, dtype=float)
    order = np.argsort(y)
    s = y[order].copy()
    for i in range(1, len(s)):
        s[i] = max(s[i], s[i - 1] + min_gap)
    if len(s) and s[-1] > hi:
        s[-1] = hi
        for i in range(len(s) - 2, -1, -1):
            s[i] = min(s[i], s[i + 1] - min_gap)
    if len(s) and s[0] < lo:
        s += lo - s[0]
    out = np.empty_like(s)
    out[order] = s
    return out


def _draw_alluvial(ax, flow: pd.DataFrame, right_label: str, *, fontsize: float,
                   header_fontsize: float | None = None) -> None:
    """Two-column alluvial: cell types (left) to arm or visit (right).

    Bar and ribbon heights are proportional to the participant-weighted flow. Ribbons
    are coloured by cell type; categories are in sorted order on both sides. Labels are
    spread apart where neighbouring categories are thinner than a line of text, with a
    short leader to their bar.
    """
    header_fontsize = header_fontsize or fontsize
    left = sorted(flow["Cell type"].unique())
    right = sorted(flow[right_label].unique())
    colors = _alluvial_colors(left)
    total = float(flow["count"].sum())
    gap = 0.02
    bar_w = 0.05

    # One height per participant on both sides; the side with fewer categories (and
    # so fewer gaps) is centred.
    unit = (1.0 - gap * (max(len(left), len(right)) - 1)) / total

    def _stack(cats, key):
        heights = [flow.loc[flow[key] == c, "count"].sum() * unit for c in cats]
        y = (1.0 - sum(heights) - gap * (len(cats) - 1)) / 2
        pos = {}
        for c, h in zip(cats, heights):
            pos[c] = (y, y + h)
            y += h + gap
        return pos

    lpos = _stack(left, "Cell type")
    rpos = _stack(right, right_label)
    lcur = {c: lpos[c][0] for c in left}
    rcur = {c: rpos[c][0] for c in right}
    seg = {}
    for c in left:
        for r in right:
            n = flow.loc[(flow["Cell type"] == c) & (flow[right_label] == r), "count"].sum()
            if n > 0:
                seg[(c, r)] = n * unit
    for c in left:
        for r in right:
            if (c, r) in seg:
                h = seg[(c, r)]
                seg[(c, r)] = [lcur[c], lcur[c] + h, None, None]
                lcur[c] += h
    for r in right:
        for c in left:
            if (c, r) in seg:
                s = seg[(c, r)]
                hr = s[1] - s[0]
                s[2], s[3] = rcur[r], rcur[r] + hr
                rcur[r] += hr

    t = np.linspace(0.0, 1.0, 60)
    ease = (1 - np.cos(np.pi * t)) / 2
    x = bar_w + (1 - 2 * bar_w) * t
    for (c, r), (l0, l1, r0, r1) in seg.items():
        ax.fill_between(x, l0 + (r0 - l0) * ease, l1 + (r1 - l1) * ease,
                        color=colors[c], alpha=0.55, linewidth=0, zorder=1)
        ax.add_patch(mpatches.Rectangle((1 - bar_w, r0), bar_w, r1 - r0,
                                        facecolor=colors[c], edgecolor="none", zorder=2))
    for c in left:
        y0, y1 = lpos[c]
        ax.add_patch(mpatches.Rectangle((0, y0), bar_w, y1 - y0, facecolor=colors[c],
                                        edgecolor="black", linewidth=0.3, zorder=2))
    for r in right:
        y0, y1 = rpos[r]
        ax.add_patch(mpatches.Rectangle((1 - bar_w, y0), bar_w, y1 - y0, facecolor="none",
                                        edgecolor="black", linewidth=0.3, zorder=3))

    ax.set_xlim(0, 1)
    ax.set_ylim(1.0, 0.0)
    ax.axis("off")

    fig = ax.figure
    height_in = ax.get_position().height * fig.get_figheight()
    min_gap = fontsize * 1.15 / 72 / height_in
    pad = 0.012
    for cats, pos, xb, xt, ha in ((left, lpos, 0.0, -0.05, "right"),
                                  (right, rpos, 1.0, 1.05, "left")):
        centres = np.array([(pos[c][0] + pos[c][1]) / 2 for c in cats])
        placed = _spread_labels(centres, min_gap, min_gap / 2, 1.0 - min_gap / 2)
        for c, yc, yl in zip(cats, centres, placed):
            ax.text(xt, yl, c, ha=ha, va="center", fontsize=fontsize, clip_on=False)
            if abs(yl - yc) > min_gap / 4:
                xs = (xb - pad, xt + pad) if ha == "right" else (xb + pad, xt - pad)
                ax.plot(xs, (yc, yl), color="#777777", linewidth=0.3, clip_on=False)
    ax.text(bar_w / 2, -0.03, "Cell type", ha="center", va="bottom",
            fontsize=header_fontsize, fontweight="bold", clip_on=False)
    ax.text(1 - bar_w / 2, -0.03, right_label, ha="center", va="bottom",
            fontsize=header_fontsize, fontweight="bold", clip_on=False)


def _alluvial_standalone(name: str, item: dict):
    """Standalone panel G figure for one dataset (title above the column headers)."""
    fig = plt.figure(figsize=(4.2, 4.0))
    ax = fig.add_axes([0.3, 0.04, 0.4, 0.80])
    _draw_alluvial(ax, item["flow"], item["right_label"], fontsize=8)
    ax.set_title(name, fontweight="bold", fontsize=10, pad=26)
    return fig


def _save_alluvial_panels(alluvial: dict) -> None:
    """Standalone PNG and interactive HTML of panel G for each dataset."""
    panel_dir = SUPP_OUTPUT / f"{FIGURE_NAME}_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for name, item in alluvial.items():
        safe_name = name.replace(" ", "_").replace("-", "_")
        fig = _alluvial_standalone(name, item)
        report_layout_problems(fig, f"panel_G_{safe_name}")
        save_panel(fig, f"panel_G_{safe_name}", FIGURE_NAME, SUPP_OUTPUT)
        try:
            _write_alluvial_html(panel_dir / f"panel_G_{safe_name}.html",
                                 item["flow"], item["right_label"])
        except Exception as exc:
            print(f"    panel_G_{safe_name}.html not written ({exc})")


def _prepare_panel_data(loaded: dict) -> dict:
    """Everything panels A-G draw, computed once from the loaded datasets."""
    return {
        "umap": _umap_data(loaded),
        "dotplot": _marker_dotplot_data(loaded),
        "silhouette": _silhouette_scores(loaded),
        "purity": _knn_purities(loaded),
        "crosstab": _celltype_crosstab(loaded),
        "mixing": _arm_mixing_scores(loaded),
        "alluvial": _alluvial_data(loaded),
    }


# ======================================================================
# Composite (180 mm wide)
# ======================================================================
#   Row 0: A, UMAP by cell type (1 x n_ds), keys below
#   Row 1: B, UMAP by grouping (1 x n_ds), keys below
#   Row 2: C (marker dot plot, left) | D (silhouette + purity, right)
#   Row 3: E (ct x ds heatmap, left) | F (arm mixing, right)
#   Row 4: G, participant-weighted alluvials (1 x n_ds)
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


def _truncate_to_column(fig, ax, labels, fontsize):
    """Shorten legend labels that would be wider than half the axes (two-column key)."""
    import matplotlib.text as _mtext

    renderer = fig.canvas.get_renderer()
    col_width = ax.get_window_extent(renderer).width / 2 - 10 * fig.dpi / 72
    probe = _mtext.Text(0, 0, "W", fontsize=fontsize)
    probe.figure = fig
    char_w = probe.get_window_extent(renderer).width
    max_chars = max(4, int(col_width / char_w) + 4)
    return [s if len(s) <= max_chars else s[:max_chars - 1] + "…" for s in labels]


def _fit_row_to_ink(fig, axes, *, left: float, right: float, gap_mm: float) -> None:
    """Place a row of equal-width axes so their label ink fills [left, right] with
    ``gap_mm`` between neighbours' ink.

    Each axes' ink overhang (labels left and right of the frame) is measured after a
    draw, so wide labels get the room they need and narrow ones give theirs away. Two
    passes, because labels placed in axes-fraction units move slightly with the width.
    """
    if not axes:
        return
    mm = fig.dpi / 25.4
    width_px = fig.bbox.width
    for _ in range(2):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        over = []
        for ax in axes:
            frame = ax.get_window_extent(r)
            ink = ax.get_tightbbox(r)
            over.append((max(frame.x0 - ink.x0, 0.0), max(ink.x1 - frame.x1, 0.0)))
        avail = (right - left) * width_px - gap_mm * mm * (len(axes) - 1)
        w = (avail - sum(lo + ro for lo, ro in over)) / len(axes)
        x = left * width_px
        for ax, (lo, ro) in zip(axes, over):
            pos = ax.get_position()
            ax.set_position([(x + lo) / width_px, pos.y0, w / width_px, pos.height])
            x += lo + w + ro + gap_mm * mm


def _build_composite(prep: dict):
    """Assemble panels A-G on a 180 mm wide artboard from the prepared panel data."""
    print("  Building composite figure (panels A-G) ...")
    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    umap = prep["umap"]
    ds_names = list(umap.keys())
    n_ds = len(ds_names)

    # Heights in mm: A, its keys, B, its keys and the C|D titles, C|D, the gap holding
    # C's gene names and the E|F titles, E|F, the gap holding E|F tick labels and the G
    # titles, and G.
    _mm = 1.0 / 25.4
    rows_mm = [30.0, 17.0, 30.0, 12.0, 40.0, 17.0, 38.0, 16.0, 42.0]
    top_mm, bottom_mm = 6.0, 3.0
    height_mm = top_mm + sum(rows_mm) + bottom_mm
    fig_c = plt.figure(figsize=(180 * _mm, height_mm * _mm))
    outer = fig_c.add_gridspec(len(rows_mm), 1, height_ratios=rows_mm, hspace=0.0,
                               left=0.06, right=0.98,
                               top=1 - top_mm / height_mm, bottom=bottom_mm / height_mm)

    # Row 0: A, UMAPs coloured by cell type
    gs_a = outer[0].subgridspec(1, n_ds, wspace=0.30)
    axes_a = [fig_c.add_subplot(gs_a[0, i]) for i in range(n_ds)]
    _panel_umap_grid(fig_c, axes_a, umap)
    fig_c.canvas.draw()
    for _ax in axes_a:
        for _coll in _ax.collections:
            _coll.set_sizes([0.15])
        _leg = _ax.get_legend()
        if _leg:
            _handles = list(_leg.legend_handles)
            _labels = [t.get_text() for t in _leg.get_texts()]
            _leg.remove()
            _ax.legend(
                handles=_handles, labels=_truncate_to_column(fig_c, _ax, _labels, 3.5),
                fontsize=3.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
                frameon=False, handlelength=0.6, handleheight=0.5,
                borderpad=0.1, labelspacing=0.12, columnspacing=0.8,
            )

    # Row 1: B, UMAPs coloured by grouping variable
    gs_b = outer[2].subgridspec(1, n_ds, wspace=0.30)
    axes_b = [fig_c.add_subplot(gs_b[0, i]) for i in range(n_ds)]
    _panel_umap_grouping(fig_c, axes_b, umap)
    for _ax in axes_b:
        for _coll in _ax.collections:
            _coll.set_sizes([0.15])

    # Row 2: C (marker dot plot, left) | D (silhouette + purity, right)
    gs_cd = outer[4].subgridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.22)
    ax_c = fig_c.add_subplot(gs_cd[0])
    _panel_marker_dotplot(fig_c, ax_c, prep["dotplot"])
    ax_c.tick_params(axis="x", labelsize=4)
    ax_c.tick_params(axis="y", labelsize=4)
    for _coll in ax_c.collections:
        _coll.set_sizes(_coll.get_sizes() * 0.35)
    _leg_c = ax_c.get_legend()
    if _leg_c:
        for _lh in _leg_c.legend_handles:
            if hasattr(_lh, "set_sizes"):
                _lh.set_sizes([s * 0.1 for s in _lh.get_sizes()])
        for _txt in _leg_c.get_texts():
            _txt.set_fontsize(4.5)
        _leg_c.get_title().set_fontsize(4.5)

    gs_d = gs_cd[1].subgridspec(2, 1, hspace=0.80)
    ax_d1 = fig_c.add_subplot(gs_d[0])
    ax_d2 = fig_c.add_subplot(gs_d[1])
    _panel_silhouette(ax_d1, prep["silhouette"])
    _panel_knn_purity(ax_d2, prep["purity"])
    ax_d1.set_ylabel("Centroid-silhouette\nscore", fontsize=4.5)
    ax_d2.set_ylabel("Graph label purity\n(connectivities)", fontsize=4.5)

    # Row 3: E (ct x ds heatmap, left) | F (arm mixing, right)
    gs_ef = outer[6].subgridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.22)
    ax_e = fig_c.add_subplot(gs_ef[0])
    _panel_ct_crosstab(ax_e, prep["crosstab"])
    ax_f = fig_c.add_subplot(gs_ef[1])
    _panel_arm_mixing(ax_f, prep["mixing"])
    for _txt in ax_f.texts:
        _txt.set_fontsize(4.5)

    # Row 4: G, participant-weighted alluvials
    gs_g = outer[8].subgridspec(1, n_ds, wspace=1.15)
    axes_g: list[plt.Axes] = []
    for i, name in enumerate(ds_names):
        ax_gi = fig_c.add_subplot(gs_g[0, i])
        axes_g.append(ax_gi)
        item = prep["alluvial"].get(name)
        if item is None:
            ax_gi.text(0.5, 0.5, f"{name}\n(no flows)", ha="center", va="center",
                       transform=ax_gi.transAxes, fontsize=4)
            ax_gi.axis("off")
            continue
        _draw_alluvial(ax_gi, item["flow"], item["right_label"], fontsize=4,
                       header_fontsize=4.5)
        ax_gi.set_title(name, fontweight="bold", fontsize=5.5, pad=12)
    _fit_row_to_ink(fig_c, axes_g, left=0.02, right=0.98, gap_mm=2.5)

    # Post-processing: fonts final, then keys and letters, then the layout check
    _cap_fontsize(fig_c, _MAX_FONT)

    for _ax in axes_b:
        key_below_axes(_ax, ncol=2, fontsize=3.5, gap_pt=1.0)
    key_below_axes(ax_f, ncol=1, fontsize=4.5)

    for ax_lbl, lbl in [(axes_a[0], "A"), (axes_b[0], "B"), (ax_c, "C"), (ax_d1, "D"),
                        (ax_e, "E"), (ax_f, "F"), (axes_g[0], "G")]:
        letter_at_panel_corner(ax_lbl, lbl, fontsize=8)

    report_layout_problems(fig_c, FIGURE_NAME)
    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, SUPP_OUTPUT, close=False)
    pdf_path = SUPP_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 2 panels (A-G) + composite."""
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

    prep = _prepare_panel_data(loaded)
    for data in loaded.values():
        del data["adata"]
    loaded.clear()
    clear_cache()
    gc.collect()
    umap = prep["umap"]

    # Panel A: UMAP grid coloured by cell type (mini-UMAPs)
    n_ds = len(umap)
    ncols = min(n_ds, 3)
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    _panel_umap_grid(fig, axes_flat, umap)
    fig.suptitle("Cell-Type UMAPs", fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Individual cell-type UMAPs (key below, two columns)
    for name, item in umap.items():
        fig_ind, ax_ind = plt.subplots(figsize=(5.5, 6.0))
        _panel_umap_grid(fig_ind, [ax_ind], {name: item})
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
    _panel_umap_grouping(fig, axes_flat, umap)
    fig.suptitle("UMAPs by Grouping Variable", fontweight="bold",
                 fontsize=12, y=1.02)
    fig.tight_layout(h_pad=3.0)
    for _ax in axes_flat:
        key_below_axes(_ax, ncol=2)
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Individual grouping UMAPs
    for name, item in umap.items():
        fig_ind, ax_ind = plt.subplots(figsize=(5.5, 5.0))
        _panel_umap_grouping(fig_ind, [ax_ind], {name: item})
        fig_ind.tight_layout()
        key_below_axes(ax_ind, ncol=2)
        safe = name.replace(" ", "_").replace("-", "_")
        save_panel(fig_ind, f"panel_B_{safe}", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: Marker gene dot plot
    fig, ax = plt.subplots(figsize=(10, 7))
    _panel_marker_dotplot(fig, ax, prep["dotplot"])
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: Embedding quality, silhouette + kNN purity (1 x 2)
    fig = plt.figure(figsize=(14, 5))
    ax1, ax2 = fig.subplots(1, 2)
    _panel_silhouette(ax1, prep["silhouette"])
    _panel_knn_purity(ax2, prep["purity"])
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: Cell-type x dataset cross-tabulation heatmap
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_ct_crosstab(ax, prep["crosstab"])
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: Quantitative baseline arm overlap (kNN mixing score)
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_arm_mixing(ax, prep["mixing"])
    fig.tight_layout()
    key_below_axes(ax, ncol=1)
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Panel G: per-dataset alluvials (PNG + interactive HTML)
    _save_alluvial_panels(prep["alluvial"])

    _build_composite(prep)
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
