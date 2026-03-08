"""
Supplementary Figure 6 — Cross-Dataset Biological Context.
==========================================================

Compare biological signal across datasets using gene-set scoring,
cross-dataset effect correlations, and pathway-level analyses.

Panels:
  A  Gene-set score distributions per dataset (violin per signature).
  B  Cross-dataset effect correlation (pairwise DiD betas).
  C  Top differentially affected genes (horizontal bar, ranked by |beta|).
  D  Gene-level effect distribution histogram per dataset.
  E  Exhaustion signature forest plot (across cell types, SF only).
  F  Effect heterogeneity (SD) per cell type per signature.
  G  Signature score pre vs post (paired trajectories).
  H  Gene-set enrichment summary heatmap.

Non-overlap guardrail: biological context, no methods/sensitivity.
"""

from __future__ import annotations

import gc

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from scipy import stats as sp_stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    apply_style,
    clear_cache,
    despine,
    get_sade_feldman,
    harmonize_response,
    load_clinical_trial_dataset,
    save_panel,
)

FIGURE_NAME = "SuppFig5_biological_context"

# Immune-related features for cross-dataset comparison
_FEATURES = [
    "CD8A",
    "CD4",
    "PDCD1",
    "HAVCR2",
    "LAG3",
    "CTLA4",
    "GZMB",
    "PRF1",
    "IFNG",
    "TNF",
    "IL2",
    "CD19",
    "CD14",
    "LYZ",
    "NKG7",
]

# Gene sets for scoring
_GENE_SETS = {
    "Exhaustion": ["PDCD1", "HAVCR2", "LAG3", "CTLA4", "TIGIT", "TOX", "ENTPD1"],
    "Cytotoxicity": ["GZMB", "PRF1", "GZMA", "GZMK", "NKG7", "GNLY", "FASLG"],
    "Activation": ["IFNG", "TNF", "IL2", "CD69", "CD25", "HLA-DRA"],
    "T cell": ["CD3D", "CD3E", "CD8A", "CD4", "TCF7", "IL7R"],
}

_DS_PALETTE = dict(
    zip(
        ["Sade-Feldman", "AML", "CAR-T", "Melanoma"],
        sns.color_palette("Set2", 4),
    )
)

# Dataset configurations: loader, layer, and TrialDesign kwargs
_DATASET_CONFIGS = {
    "Sade-Feldman": {
        "loader": "sade_feldman",
        "layer": "log1p_tpm",
        "design_kw": {
            "participant_col": "participant_id",
            "visit_col": "visit",
            "arm_col": "response",
            "arm_treated": "Responder",
            "arm_control": "Non-responder",
        },
    },
    "AML": {
        "loader": "aml",
        "layer": "log1p_norm",
        "design_kw": {
            "participant_col": "participant_id",
            "visit_col": "visit",
            "arm_col": "response",
            "arm_treated": "Treatment",
            "arm_control": "Control",
        },
    },
    "CAR-T": {
        "loader": "cart",
        "layer": "log1p_norm",
        "design_kw": {
            "participant_col": "participant_id",
            "visit_col": "visit",
            "arm_col": "response",
            "arm_treated": "CAR-T",
            "arm_control": None,  # single-arm: DiD is skipped
        },
    },
    "Melanoma": {
        "loader": "melanoma",
        "layer": "log1p_tpm",
        "design_kw": {
            "participant_col": "participant_id",
            "visit_col": "visit",
            "arm_col": "Cohort",
            "arm_treated": "New",
            "arm_control": "Tirosh",
        },
    },
}


def _load_data():
    """Load all datasets, attempt DiD, compute gene set scores."""
    import sctrial

    datasets = {}

    for ds_name, cfg in _DATASET_CONFIGS.items():
        print(f"  Loading {ds_name}...")
        try:
            # Load adata
            if cfg["loader"] == "sade_feldman":
                adata = get_sade_feldman()
                adata = harmonize_response(adata)
                if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
                    adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
            else:
                adata = load_clinical_trial_dataset(cfg["loader"])

            layer = cfg["layer"]
            design = sctrial.TrialDesign(**cfg["design_kw"])
            feats = [f for f in _FEATURES if f in adata.var_names]

            # Gene set scores (always possible)
            gs_scores = _score_gene_sets(adata, layer)

            entry = {
                "adata": adata,
                "design": design,
                "features": feats,
                "gs_scores": gs_scores,
                "did": None,  # may be filled below
            }

            # Attempt DiD — may fail for single-arm datasets
            try:
                did_df = sctrial.did_table(
                    adata,
                    feats,
                    design,
                    ("Pre", "Post"),
                    layer=layer,
                    aggregate="participant_visit",
                    standardize=True,
                )
                if did_df is not None and len(did_df) > 0:
                    entry["did"] = did_df
            except Exception as exc:
                print(f"    DiD skipped for {ds_name}: {exc}")

            datasets[ds_name] = entry
        except Exception as exc:
            print(f"    Could not load {ds_name}: {exc}")

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


def _datasets_with_did(datasets: dict) -> dict:
    """Return subset of datasets that have valid DiD results."""
    return {k: v for k, v in datasets.items() if v.get("did") is not None}


# ── Panel A: Gene-set score violins ──────────────────────────────


def _panel_gs_violins(ax, datasets: dict):
    """Violin plot: gene-set scores per dataset."""
    rows = []
    for ds_name, ds in datasets.items():
        for gs_name, vals in ds["gs_scores"].items():
            for v in vals:
                rows.append({"Dataset": ds_name, "Gene set": gs_name, "Score": v})

    if not rows:
        ax.text(0.5, 0.5, "No gene-set scores", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    n_ds = df["Dataset"].nunique()
    sns.violinplot(
        data=df,
        x="Gene set",
        y="Score",
        hue="Dataset",
        palette=_DS_PALETTE,
        split=(n_ds == 2),
        inner="quartile",
        linewidth=0.5,
        ax=ax,
        cut=0,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Z-mean score")
    ax.set_title("Gene-Set Score Distributions", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    despine(ax)


# ── Panel B: Cross-dataset DiD correlation ───────────────────────


def _panel_cross_ds_corr(ax, datasets: dict):
    """Scatter: pairwise DiD beta correlation for datasets with valid DiD."""
    did_ds = _datasets_with_did(datasets)
    ds_names = list(did_ds.keys())
    if len(ds_names) < 2:
        ax.text(
            0.5,
            0.5,
            "Need >= 2 datasets with DiD",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Use first two datasets with DiD
    d1, d2 = ds_names[0], ds_names[1]
    b1 = did_ds[d1]["did"].set_index("feature")["beta_DiD"]
    b2 = did_ds[d2]["did"].set_index("feature")["beta_DiD"]
    common = b1.index.intersection(b2.index)
    mask = np.isfinite(b1[common]) & np.isfinite(b2[common])
    common = common[mask]

    if len(common) < 3:
        ax.text(0.5, 0.5, "Insufficient overlap", ha="center", va="center", transform=ax.transAxes)
        return

    x, y = b1[common].values, b2[common].values
    ax.scatter(x, y, s=50, alpha=0.7, color="#8E44AD", edgecolors="grey", linewidth=0.3)

    # Annotate with adjustText to avoid overlap
    texts = []
    for feat in common:
        texts.append(ax.text(b1[feat], b2[feat], feat, fontsize=7, alpha=0.8))
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        force_text=(2.0, 2.0),
        force_points=(2.0, 2.0),
        expand=(1.5, 1.5),
    )

    r, p = sp_stats.pearsonr(x, y)
    ax.text(
        0.05,
        0.95,
        f"r = {r:.2f}\np = {p:.3f}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.8),
    )
    lims = [min(min(x), min(y)) - 0.2, max(max(x), max(y)) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel(f"\u03b2 ({d1})")
    ax.set_ylabel(f"\u03b2 ({d2})")
    ax.set_title("Cross-Dataset Effect Correlation", fontweight="bold")
    despine(ax)


# ── Panel C: Top genes by |beta| ────────────────────────────────


def _panel_top_genes(ax, datasets: dict):
    """Horizontal bar: top genes ranked by |beta| across all DiD datasets."""
    did_ds = _datasets_with_did(datasets)
    rows = []
    for ds_name, ds in did_ds.items():
        did_df = ds["did"]
        for _, row in did_df.iterrows():
            rows.append({"Dataset": ds_name, "Feature": row["feature"], "beta": row["beta_DiD"]})

    if not rows:
        ax.text(0.5, 0.5, "No DiD data", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    df["abs_beta"] = df["beta"].abs()
    # Top 12 by max |beta| across datasets
    top = df.groupby("Feature")["abs_beta"].max().nlargest(12).index
    df = df[df["Feature"].isin(top)]

    # Pivot for grouped bar
    piv = df.pivot(index="Feature", columns="Dataset", values="beta").reindex(top)
    # Sort by first column
    first_col = piv.columns[0]
    piv = piv.sort_values(first_col, ascending=True)

    n_ds = len(piv.columns)
    y = np.arange(len(piv))
    h = 0.8 / max(n_ds, 1)
    for i, col in enumerate(piv.columns):
        ax.barh(
            y + i * h - (n_ds - 1) * h / 2,
            piv[col].values,
            height=h,
            color=_DS_PALETTE.get(col, "grey"),
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            label=col,
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xlabel("\u03b2 (DiD)")
    ax.set_title("Top Differentially Affected Genes", fontweight="bold")
    ax.legend(fontsize=7, loc="best", frameon=True)
    despine(ax)


# ── Panel D: Effect distribution histogram ────────────────────────


def _panel_effect_hist(ax, datasets: dict):
    """Histogram of DiD betas across features per dataset (all with DiD)."""
    did_ds = _datasets_with_did(datasets)
    for ds_name, ds in did_ds.items():
        did_df = ds["did"]
        vals = did_df["beta_DiD"].dropna().values
        ax.hist(
            vals,
            bins=15,
            alpha=0.45,
            label=ds_name,
            color=_DS_PALETTE.get(ds_name, "grey"),
            edgecolor="white",
        )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("\u03b2 (DiD)")
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
        ax.text(0.5, 0.5, "No SF data", ha="center", va="center", transform=ax.transAxes)
        return

    adata = ds["adata"]
    design = ds["design"]
    exh_genes = [g for g in ["PDCD1", "HAVCR2", "LAG3", "CTLA4", "TOX"] if g in adata.var_names]
    ct_col = next((c for c in ["cell_type", "celltype"] if c in adata.obs.columns), None)
    if not ct_col or not exh_genes:
        ax.text(
            0.5,
            0.5,
            "No cell-type or exhaustion genes",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    top_cts = adata.obs[ct_col].value_counts().head(4).index.tolist()
    rows = []
    for ct in top_cts:
        sub = adata[adata.obs[ct_col] == ct].copy()
        if sub.n_obs < 50:
            continue
        try:
            ct_did = sctrial.did_table(
                sub,
                exh_genes,
                design,
                ("Pre", "Post"),
                layer="log1p_tpm",
                aggregate="participant_visit",
                standardize=True,
            )
            for _, r in ct_did.iterrows():
                rows.append(
                    {
                        "Cell type": ct,
                        "Gene": r["feature"],
                        "beta": r["beta_DiD"],
                        "se": r["se_DiD"],
                    }
                )
        except Exception:
            pass

    if not rows:
        ax.text(0.5, 0.5, "No exhaustion results", ha="center", va="center", transform=ax.transAxes)
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
        ax.errorbar(
            row["beta"], i, xerr=ci, fmt="o", markersize=4, color=c, elinewidth=1, capsize=2
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['Gene']} ({r['Cell type']})" for _, r in df.iterrows()], fontsize=5.5)
    ax.set_xlabel("\u03b2 (DiD)")
    ax.set_title("Exhaustion Genes by Cell Type (SF)", fontweight="bold")
    despine(ax)


# ── Panel F: Effect heterogeneity SD heatmap ─────────────────────


def _panel_heterogeneity_sd(ax, datasets: dict):
    """Heatmap: DiD betas across all datasets with valid DiD per feature."""
    did_ds = _datasets_with_did(datasets)
    all_betas = {}
    for ds_name, ds in did_ds.items():
        did_df = ds["did"]
        all_betas[ds_name] = did_df.set_index("feature")["beta_DiD"]

    df = pd.DataFrame(all_betas)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Sort by mean |beta|
    df["mean_abs"] = df.abs().mean(axis=1)
    df = df.sort_values("mean_abs", ascending=False).drop(columns="mean_abs")
    df = df.head(12)

    sns.heatmap(
        df,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"shrink": 0.6, "label": "\u03b2"},
    )
    ax.set_title("Effect Sizes Across Datasets", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    ax.tick_params(axis="y", labelsize=7)


# ── Panel G: Pre vs Post trajectories ────────────────────────────


def _panel_pre_post_trajectories(ax, datasets: dict):
    """Paired trajectories: mean gene-set score Pre vs Post by arm."""
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
                        rows.append(
                            {
                                "Dataset": ds_name,
                                "Gene set": gs_name,
                                "Visit": visit,
                                "Arm": arm,
                                "Score": float(np.nanmean(vals[mask])),
                            }
                        )

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    # Focus on SF only for clarity
    df = df[df["Dataset"] == "Sade-Feldman"]

    gs_names = df["Gene set"].unique()
    arms = df["Arm"].unique()
    arm_colors = {
        "Responder": COLORS.get("treated", "#E07B54"),
        "Non-responder": COLORS.get("control", "#5B9BD5"),
    }

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
            ax.plot(
                [0 + offsets[j], 1 + offsets[j]],
                [pre[0], post[0]],
                "o-",
                color=c,
                markersize=4,
                linewidth=1,
                alpha=0.7,
            )
            if arm == arms[0]:  # label gene set only once
                ax.annotate(gs, (1 + offsets[j] + 0.05, post[0]), fontsize=5.5, alpha=0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Mean Z-score")
    ax.set_title("Gene-Set Trajectories (SF)", fontweight="bold")

    legend_h = [mpatches.Patch(color=c, label=a) for a, c in arm_colors.items() if a in arms]
    if legend_h:
        ax.legend(handles=legend_h, fontsize=6, loc="best", frameon=True)
    despine(ax)


# ── Panel H: Gene-set enrichment summary ─────────────────────────


def _panel_gs_enrichment_summary(ax, datasets: dict):
    """Heatmap: mean gene-set z-score by dataset x arm x visit."""
    rows = []
    for ds_name, ds in datasets.items():
        adata = ds["adata"]
        arm_col = ds["design"].arm_col
        # Use actual arm values from the data (not just design config)
        # to ensure single-arm datasets are included
        design_arms = list(dict.fromkeys([ds["design"].arm_treated, ds["design"].arm_control]))
        # Also include any arm values that actually appear in the data
        if arm_col in adata.obs.columns:
            data_arms = adata.obs[arm_col].unique().tolist()
            all_arms = list(dict.fromkeys(design_arms + data_arms))
        else:
            all_arms = design_arms
        # Determine available visit values
        if "visit" in adata.obs.columns:
            available_visits = [v for v in ["Pre", "Post"] if v in adata.obs["visit"].values]
            if not available_visits:
                # Fall back to all unique visit values
                available_visits = adata.obs["visit"].unique().tolist()
        else:
            continue
        for gs_name, vals in ds["gs_scores"].items():
            for visit in available_visits:
                for arm in all_arms:
                    mask = (adata.obs["visit"] == visit) & (adata.obs[arm_col] == arm)
                    if mask.sum() > 0:
                        rows.append(
                            {
                                "label": f"{ds_name}\n{arm}\n{visit}",
                                "Gene set": gs_name,
                                "Score": float(np.nanmean(vals[mask])),
                            }
                        )

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="Gene set", columns="label", values="Score", aggfunc="mean")

    sns.heatmap(
        piv,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"shrink": 0.6, "label": "Z-score"},
    )
    ax.set_title("Gene-Set Scores: Dataset \u00d7 Arm \u00d7 Visit", fontweight="bold")
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7)


# ======================================================================
# Generate
# ======================================================================


def generate():
    """Create and save Supplementary Figure 5 panels.

    Reorganised layout (5 panels):
      A  Gene-set score violins          (was SF6-A)
      B  Cross-dataset correlation       (was SF6-B)
      C  Exhaustion forest               (was SF6-E)
      D  Heterogeneity SD heatmap        (was SF6-F)
      E  Pre/post trajectories           (was SF6-G)
    """
    print("Supplementary Figure 5: Biological Context of Treatment Effects")
    datasets = _load_data()
    print(f"  Loaded {len(datasets)} datasets")

    panels = [
        ("panel_A", _panel_gs_violins, (11, 5)),
        ("panel_B", _panel_cross_ds_corr, (7, 6.5)),
        ("panel_C", _panel_exhaustion_forest, (9, 7)),
        ("panel_D", _panel_heterogeneity_sd, (10, 5.5)),
        ("panel_E", _panel_pre_post_trajectories, (7, 5.5)),
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
