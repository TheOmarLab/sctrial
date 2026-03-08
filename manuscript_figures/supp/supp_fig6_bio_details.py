"""
Supplementary Figure 6 — Cross-Dataset Biological Context.
==========================================================

Compare biological signal across datasets using gene-set scoring,
cross-dataset effect correlations, and pathway-level analyses.

Panels:
  A  Gene-set score distributions per dataset (violin per signature).
  B  Cross-dataset effect correlation (SF vs AML DiD betas).
  C  Top differentially affected genes (horizontal bar, ranked by |β|).
  D  Gene-level effect distribution histogram per dataset.
  E  Exhaustion signature forest plot (across cell types, SF only).
  F  Effect heterogeneity (SD) per cell type per signature.
  G  Signature score pre vs post (paired trajectories).
  H  Gene-set enrichment summary heatmap.

Non-overlap guardrail: biological context, no methods/sensitivity.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    apply_style,
    despine,
    save_panel,
    get_sade_feldman,
    harmonize_response,
    load_clinical_trial_dataset,
    clear_cache,
)

FIGURE_NAME = "SuppFig6_biological_details"

# Immune-related features for cross-dataset comparison
_FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19",
    "CD14", "LYZ", "NKG7",
]

# Gene sets for scoring
_GENE_SETS = {
    "Exhaustion": ["PDCD1", "HAVCR2", "LAG3", "CTLA4", "TIGIT", "TOX", "ENTPD1"],
    "Cytotoxicity": ["GZMB", "PRF1", "GZMA", "GZMK", "NKG7", "GNLY", "FASLG"],
    "Activation": ["IFNG", "TNF", "IL2", "CD69", "CD25", "HLA-DRA"],
    "T cell": ["CD3D", "CD3E", "CD8A", "CD4", "TCF7", "IL7R"],
}

_DS_PALETTE = dict(zip(
    ["Sade-Feldman", "AML"],
    sns.color_palette("Set2", 2),
))


def _load_data():
    """Load SF and AML datasets, run DiD, compute gene set scores."""
    import sctrial

    datasets = {}

    # Sade-Feldman
    adata_sf = get_sade_feldman()
    adata_sf = harmonize_response(adata_sf)
    if "log1p_tpm" not in adata_sf.layers and "tpm" in adata_sf.layers:
        adata_sf.layers["log1p_tpm"] = np.log1p(adata_sf.layers["tpm"])

    design_sf = sctrial.TrialDesign(
        participant_col="participant_id", visit_col="visit",
        arm_col="response", arm_treated="Responder",
        arm_control="Non-responder",
    )
    feats_sf = [f for f in _FEATURES if f in adata_sf.var_names]

    did_sf = sctrial.did_table(
        adata_sf, feats_sf, design_sf, ("Pre", "Post"),
        layer="log1p_tpm", aggregate="participant_visit", standardize=True,
    )

    # Gene set scores
    gs_scores_sf = _score_gene_sets(adata_sf, "log1p_tpm")

    datasets["Sade-Feldman"] = {
        "adata": adata_sf, "did": did_sf, "design": design_sf,
        "features": feats_sf, "gs_scores": gs_scores_sf,
    }

    # AML
    adata_aml = load_clinical_trial_dataset("aml")
    design_aml = sctrial.TrialDesign(
        participant_col="participant_id", visit_col="visit",
        arm_col="response", arm_treated="Treatment",
        arm_control="Control",
    )
    feats_aml = [f for f in _FEATURES if f in adata_aml.var_names]

    did_aml = sctrial.did_table(
        adata_aml, feats_aml, design_aml, ("Pre", "Post"),
        layer="log1p_norm", aggregate="participant_visit", standardize=True,
    )

    gs_scores_aml = _score_gene_sets(adata_aml, "log1p_norm")

    datasets["AML"] = {
        "adata": adata_aml, "did": did_aml, "design": design_aml,
        "features": feats_aml, "gs_scores": gs_scores_aml,
    }

    return datasets


def _score_gene_sets(adata, layer):
    """Score gene sets via z-mean of available genes."""
    import warnings
    scores = {}
    X = adata.layers[layer] if layer in adata.layers else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    for gs_name, genes in _GENE_SETS.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 2:
            continue
        idx = [list(adata.var_names).index(g) for g in present]
        vals = X[:, idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zmean = np.nanmean(
                (vals - np.nanmean(vals, axis=0)) / (np.nanstd(vals, axis=0) + 1e-8),
                axis=1,
            )
        scores[gs_name] = zmean
    return scores


# ── Panel A: Gene-set score violins ──────────────────────────────

def _panel_gs_violins(ax, datasets: dict):
    """Violin plot: gene-set scores per dataset."""
    rows = []
    for ds_name, ds in datasets.items():
        for gs_name, vals in ds["gs_scores"].items():
            # Subsample for speed
            n = min(len(vals), 2000)
            idx = np.random.RandomState(42).choice(len(vals), n, replace=False)
            for v in vals[idx]:
                rows.append({"Dataset": ds_name, "Gene set": gs_name, "Score": v})

    if not rows:
        ax.text(0.5, 0.5, "No gene-set scores", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    sns.violinplot(data=df, x="Gene set", y="Score", hue="Dataset",
                   palette="Set2", split=True, inner="quartile",
                   linewidth=0.5, ax=ax, cut=0)
    ax.set_xlabel("")
    ax.set_ylabel("Z-mean score")
    ax.set_title("Gene-Set Score Distributions", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    despine(ax)


# ── Panel B: Cross-dataset DiD correlation ───────────────────────

def _panel_cross_ds_corr(ax, datasets: dict):
    """Scatter: SF β vs AML β for shared features."""
    ds_names = list(datasets.keys())
    if len(ds_names) < 2:
        ax.text(0.5, 0.5, "Need ≥2 datasets", ha="center", va="center",
                transform=ax.transAxes)
        return

    d1, d2 = ds_names[0], ds_names[1]
    b1 = datasets[d1]["did"].set_index("feature")["beta_DiD"]
    b2 = datasets[d2]["did"].set_index("feature")["beta_DiD"]
    common = b1.index.intersection(b2.index)
    mask = np.isfinite(b1[common]) & np.isfinite(b2[common])
    common = common[mask]

    if len(common) < 3:
        ax.text(0.5, 0.5, "Insufficient overlap", ha="center", va="center",
                transform=ax.transAxes)
        return

    x, y = b1[common].values, b2[common].values
    ax.scatter(x, y, s=30, alpha=0.7, color="#8E44AD",
               edgecolors="grey", linewidth=0.3)
    for feat in common:
        ax.annotate(feat, (b1[feat], b2[feat]),
                    fontsize=5.5, alpha=0.7, xytext=(3, 2),
                    textcoords="offset points")

    r, p = sp_stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.3f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.8))
    lims = [min(min(x), min(y)) - 0.2, max(max(x), max(y)) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel(f"β ({d1})")
    ax.set_ylabel(f"β ({d2})")
    ax.set_title("Cross-Dataset Effect Correlation", fontweight="bold")
    despine(ax)


# ── Panel C: Top genes by |β| ────────────────────────────────────

def _panel_top_genes(ax, datasets: dict):
    """Horizontal bar: top genes ranked by |β| across datasets."""
    rows = []
    for ds_name, ds in datasets.items():
        did_df = ds["did"]
        for _, row in did_df.iterrows():
            rows.append({"Dataset": ds_name, "Feature": row["feature"],
                         "beta": row["beta_DiD"]})

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    df["abs_beta"] = df["beta"].abs()
    # Top 10 by max |β| across datasets
    top = df.groupby("Feature")["abs_beta"].max().nlargest(12).index
    df = df[df["Feature"].isin(top)]

    # Pivot for grouped bar
    piv = df.pivot(index="Feature", columns="Dataset", values="beta").reindex(top)
    piv = piv.sort_values(piv.columns[0], ascending=True)

    y = np.arange(len(piv))
    h = 0.35
    for i, col in enumerate(piv.columns):
        ax.barh(y + i * h, piv[col].values, height=h,
                color=_DS_PALETTE.get(col, "grey"), alpha=0.8,
                edgecolor="white", linewidth=0.5, label=col)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y + h / 2)
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xlabel("β (DiD)")
    ax.set_title("Top Differentially Affected Genes", fontweight="bold")
    ax.legend(fontsize=7, loc="best", frameon=True)
    despine(ax)


# ── Panel D: Effect distribution histogram ────────────────────────

def _panel_effect_hist(ax, datasets: dict):
    """Histogram of DiD betas across features per dataset."""
    for ds_name, ds in datasets.items():
        did_df = ds["did"]
        vals = did_df["beta_DiD"].dropna().values
        ax.hist(vals, bins=15, alpha=0.5, label=ds_name,
                color=_DS_PALETTE.get(ds_name, "grey"), edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("β (DiD)")
    ax.set_ylabel("Count")
    ax.set_title("Effect Size Distribution", fontweight="bold")
    ax.legend(fontsize=7, frameon=True)
    despine(ax)


# ── Panel E: Exhaustion forest by cell type ───────────────────────

def _panel_exhaustion_forest(ax, datasets: dict):
    """Forest plot: exhaustion-related gene effects by cell type (SF)."""
    import sctrial

    ds = datasets.get("Sade-Feldman")
    if ds is None:
        ax.text(0.5, 0.5, "No SF data", ha="center", va="center",
                transform=ax.transAxes)
        return

    adata = ds["adata"]
    design = ds["design"]
    exh_genes = [g for g in ["PDCD1", "HAVCR2", "LAG3", "CTLA4", "TOX"]
                 if g in adata.var_names]
    ct_col = next((c for c in ["cell_type", "celltype"]
                   if c in adata.obs.columns), None)
    if not ct_col or not exh_genes:
        ax.text(0.5, 0.5, "No cell-type or exhaustion genes", ha="center",
                va="center", transform=ax.transAxes)
        return

    top_cts = adata.obs[ct_col].value_counts().head(4).index.tolist()
    rows = []
    for ct in top_cts:
        sub = adata[adata.obs[ct_col] == ct].copy()
        if sub.n_obs < 50:
            continue
        try:
            ct_did = sctrial.did_table(
                sub, exh_genes, design, ("Pre", "Post"),
                layer="log1p_tpm", aggregate="participant_visit",
                standardize=True,
            )
            for _, r in ct_did.iterrows():
                rows.append({"Cell type": ct, "Gene": r["feature"],
                             "beta": r["beta_DiD"], "se": r["se_DiD"]})
        except Exception:
            pass

    if not rows:
        ax.text(0.5, 0.5, "No exhaustion results", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Exhaustion Genes by Cell Type", fontweight="bold")
        despine(ax)
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["Cell type", "beta"]).reset_index(drop=True)

    y = np.arange(len(df))
    colors = sns.color_palette("tab10", df["Cell type"].nunique())
    ct_colors = dict(zip(df["Cell type"].unique(), colors))

    for i, (_, row) in enumerate(df.iterrows()):
        c = ct_colors[row["Cell type"]]
        ci = 1.96 * row["se"] if np.isfinite(row["se"]) else 0
        ax.errorbar(row["beta"], i, xerr=ci, fmt="o", markersize=4,
                    color=c, elinewidth=1, capsize=2)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['Gene']} ({r['Cell type'][:8]})"
                        for _, r in df.iterrows()], fontsize=6)
    ax.set_xlabel("β (DiD)")
    ax.set_title("Exhaustion Genes by Cell Type (SF)", fontweight="bold")
    despine(ax)


# ── Panel F: Effect heterogeneity SD heatmap ─────────────────────

def _panel_heterogeneity_sd(ax, datasets: dict):
    """Heatmap: std dev of DiD betas across datasets per feature."""
    all_betas = {}
    for ds_name, ds in datasets.items():
        did_df = ds["did"]
        all_betas[ds_name] = did_df.set_index("feature")["beta_DiD"]

    df = pd.DataFrame(all_betas)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Sort by mean |β|
    df["mean_abs"] = df.abs().mean(axis=1)
    df = df.sort_values("mean_abs", ascending=False).drop(columns="mean_abs")
    df = df.head(12)

    sns.heatmap(df, ax=ax, cmap="RdBu_r", center=0, linewidths=0.5,
                linecolor="white", annot=True, fmt=".2f",
                annot_kws={"fontsize": 7},
                cbar_kws={"shrink": 0.6, "label": "β"})
    ax.set_title("Effect Sizes Across Datasets", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    ax.tick_params(axis="y", labelsize=7)


# ── Panel G: Pre vs Post trajectories ────────────────────────────

def _panel_pre_post_trajectories(ax, datasets: dict):
    """Paired trajectories: mean gene-set score Pre→Post by arm."""
    rows = []
    for ds_name, ds in datasets.items():
        adata = ds["adata"]
        arm_col = ds["design"].arm_col
        for gs_name, vals in ds["gs_scores"].items():
            adata.obs[f"_gs_{gs_name}"] = vals
            for visit in ["Pre", "Post"]:
                for arm in adata.obs[arm_col].unique():
                    mask = (adata.obs["visit"] == visit) & (adata.obs[arm_col] == arm)
                    if mask.sum() > 0:
                        rows.append({
                            "Dataset": ds_name, "Gene set": gs_name,
                            "Visit": visit, "Arm": arm,
                            "Score": float(np.nanmean(vals[mask])),
                        })

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    # Focus on SF only for clarity
    df = df[df["Dataset"] == "Sade-Feldman"]

    gs_names = df["Gene set"].unique()
    arms = df["Arm"].unique()
    arm_colors = {"Responder": COLORS.get("treated", "#E07B54"),
                  "Non-responder": COLORS.get("control", "#5B9BD5")}

    x_pos = {"Pre": 0, "Post": 1}
    offsets = np.linspace(-0.15, 0.15, len(gs_names))

    for arm in arms:
        for j, gs in enumerate(gs_names):
            sub = df[(df["Arm"] == arm) & (df["Gene set"] == gs)]
            if len(sub) < 2:
                continue
            pre = sub[sub["Visit"] == "Pre"]["Score"].values
            post = sub[sub["Visit"] == "Post"]["Score"].values
            if len(pre) == 0 or len(post) == 0:
                continue
            c = arm_colors.get(arm, "grey")
            ax.plot([0 + offsets[j], 1 + offsets[j]],
                    [pre[0], post[0]], "o-", color=c,
                    markersize=4, linewidth=1, alpha=0.7)
            if arm == arms[0]:  # label gene set only once
                ax.annotate(gs, (1 + offsets[j] + 0.05, post[0]),
                            fontsize=5.5, alpha=0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Mean Z-score")
    ax.set_title("Gene-Set Trajectories (SF)", fontweight="bold")

    legend_h = [mpatches.Patch(color=c, label=a) for a, c in arm_colors.items()
                if a in arms]
    if legend_h:
        ax.legend(handles=legend_h, fontsize=6, loc="best", frameon=True)
    despine(ax)


# ── Panel H: Gene-set enrichment summary ─────────────────────────

def _panel_gs_enrichment_summary(ax, datasets: dict):
    """Heatmap: mean gene-set z-score by dataset × arm × visit."""
    rows = []
    for ds_name, ds in datasets.items():
        adata = ds["adata"]
        arm_col = ds["design"].arm_col
        for gs_name, vals in ds["gs_scores"].items():
            for visit in ["Pre", "Post"]:
                for arm in [ds["design"].arm_treated, ds["design"].arm_control]:
                    mask = (adata.obs["visit"] == visit) & (adata.obs[arm_col] == arm)
                    if mask.sum() > 0:
                        rows.append({
                            "label": f"{ds_name}\n{arm}\n{visit}",
                            "Gene set": gs_name,
                            "Score": float(np.nanmean(vals[mask])),
                        })

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    piv = df.pivot(index="Gene set", columns="label", values="Score")

    sns.heatmap(piv, ax=ax, cmap="RdBu_r", center=0, linewidths=0.5,
                linecolor="white", annot=True, fmt=".2f",
                annot_kws={"fontsize": 6},
                cbar_kws={"shrink": 0.6, "label": "Z-score"})
    ax.set_title("Gene-Set Scores: Dataset × Arm × Visit", fontweight="bold")
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    ax.tick_params(axis="y", labelsize=7)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 6 panels."""
    print("Supplementary Figure 6: Cross-Dataset Biological Context")
    datasets = _load_data()
    print(f"  Loaded {len(datasets)} datasets")

    panels = [
        ("panel_A", _panel_gs_violins, (9, 5)),
        ("panel_B", _panel_cross_ds_corr, (6.5, 6)),
        ("panel_C", _panel_top_genes, (8, 6)),
        ("panel_D", _panel_effect_hist, (7, 5)),
        ("panel_E", _panel_exhaustion_forest, (7, 7)),
        ("panel_F", _panel_heterogeneity_sd, (8, 5.5)),
        ("panel_G", _panel_pre_post_trajectories, (7, 5.5)),
        ("panel_H", _panel_gs_enrichment_summary, (10, 5)),
    ]

    for name, func, figsize in panels:
        fig, ax = plt.subplots(figsize=figsize)
        func(ax, datasets)
        fig.tight_layout()
        save_panel(fig, name, FIGURE_NAME, SUPP_OUTPUT)

    # Cleanup
    for ds in datasets.values():
        if "adata" in ds:
            del ds["adata"]
    datasets.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
