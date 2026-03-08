"""
Supplementary Figure 4 -- Outcome Correlation Details
=====================================================

Three-panel figure (1x3) exploring signature-level associations with
treatment response in the Sade-Feldman immunotherapy dataset.  These
panels were originally Figure 7 panels A, B, and D in the 12-panel
manuscript.

Panels
------
A  Signature changes: grouped bars of mean Pre->Post change for
   Responders vs Non-responders, top 10 by Cohen's d.
B  Effect-size forest plot: Cohen's d with direction colouring and
   FDR-scaled marker size.
C  Predictive power: AUC (from Mann-Whitney U) for each signature.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from .._shared import (
    COLORS,
    GENE_SIGNATURES,
    SUPP_OUTPUT,
    TrialDesign,
    apply_style,
    clear_cache,
    despine,
    get_sade_feldman,
    harmonize_response,
    save_panel,
    score_signatures,
    sig_display,
)

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "SuppFig4_outcome_correlation"
FIGSIZE = (18, 6)


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_data() -> dict:
    """Load Sade-Feldman, score signatures, compute participant-level
    Pre->Post changes, and derive response-correlation statistics.

    Returns a dict with:
        change_df  : per-participant per-signature change (Post - Pre)
        stats_df   : per-signature Cohen's d, AUC, FDR, p-value
        sig_cols   : list of scored signature column names
    """
    adata = get_sade_feldman()

    # Ensure log1p_tpm
    if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])

    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")
    adata = harmonize_response(adata)

    obs = adata.obs.copy()

    # ── Participant-level means per visit ──────────────────────────────
    group_cols = ["participant_id", "visit", "response_harmonized"]
    pid_means = obs.groupby(group_cols, observed=True)[sig_cols].mean().reset_index()

    # ── Pre -> Post change per participant ─────────────────────────────
    pre = pid_means[pid_means["visit"] == "Pre"].set_index("participant_id")
    post = pid_means[pid_means["visit"] == "Post"].set_index("participant_id")
    common_pids = pre.index.intersection(post.index)

    if len(common_pids) == 0:
        print("  WARNING: No paired participants found")
        return dict(change_df=None, stats_df=None, sig_cols=sig_cols, adata=adata)

    change = post.loc[common_pids, sig_cols] - pre.loc[common_pids, sig_cols]
    change["response"] = pre.loc[common_pids, "response_harmonized"]
    change = change.reset_index()

    # ── Per-signature statistics ───────────────────────────────────────
    records = []
    for col in sig_cols:
        vals_r = change.loc[change["response"] == "Responder", col].dropna()
        vals_nr = change.loc[change["response"] == "Non-responder", col].dropna()

        if len(vals_r) < 2 or len(vals_nr) < 2:
            continue

        # Mean change
        mean_r = vals_r.mean()
        mean_nr = vals_nr.mean()
        sem_r = vals_r.sem()
        sem_nr = vals_nr.sem()

        # Cohen's d (pooled)
        n_r, n_nr = len(vals_r), len(vals_nr)
        pooled_std = np.sqrt(
            ((n_r - 1) * vals_r.std(ddof=1) ** 2 +
             (n_nr - 1) * vals_nr.std(ddof=1) ** 2)
            / (n_r + n_nr - 2)
        )
        d = (mean_r - mean_nr) / pooled_std if pooled_std > 0 else 0.0

        # Point-biserial correlation (t-test p-value)
        _, p_val = stats.ttest_ind(vals_r, vals_nr, equal_var=False)

        # AUC from Mann-Whitney U
        u_stat, _ = stats.mannwhitneyu(vals_r, vals_nr, alternative="two-sided")
        auc = u_stat / (n_r * n_nr)
        # Ensure AUC reflects Responder > Non-responder direction
        if mean_r < mean_nr:
            auc = 1.0 - auc

        records.append(dict(
            signature=col,
            display=sig_display(col),
            mean_R=mean_r,
            mean_NR=mean_nr,
            sem_R=sem_r,
            sem_NR=sem_nr,
            cohens_d=d,
            abs_d=abs(d),
            p_value=p_val,
            auc=auc,
        ))

    stats_df = pd.DataFrame(records)

    # FDR correction (Benjamini-Hochberg)
    if len(stats_df) > 0:
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(stats_df["p_value"], method="fdr_bh")
        stats_df["fdr"] = fdr
    else:
        stats_df["fdr"] = np.nan

    return dict(
        change_df=change,
        stats_df=stats_df,
        sig_cols=sig_cols,
        adata=adata,
    )


# ======================================================================
# Panel A -- Signature changes comparison (grouped bars)
# ======================================================================

def panel_A(ax, data: dict):
    """Grouped bar chart: mean change (Post-Pre) for Responders vs
    Non-responders, top 10 signatures sorted by |Cohen's d|."""
    stats_df = data["stats_df"]

    if stats_df is None or len(stats_df) == 0:
        ax.text(0.5, 0.5, "No data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df = stats_df.sort_values("abs_d", ascending=False).head(10).copy()
    df = df.sort_values("abs_d", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    bar_h = 0.35

    # Responder bars
    ax.barh(y - bar_h / 2, df["mean_R"], height=bar_h,
            xerr=df["sem_R"], capsize=2,
            color=COLORS["treated"], alpha=0.85, edgecolor="white",
            linewidth=0.5, label="Responder",
            error_kw=dict(lw=0.8, capthick=0.8))

    # Non-responder bars
    ax.barh(y + bar_h / 2, df["mean_NR"], height=bar_h,
            xerr=df["sem_NR"], capsize=2,
            color=COLORS["control"], alpha=0.85, edgecolor="white",
            linewidth=0.5, label="Non-responder",
            error_kw=dict(lw=0.8, capthick=0.8))

    # FDR markers
    for i, (_, row) in enumerate(df.iterrows()):
        fdr_val = row["fdr"]
        if pd.notna(fdr_val) and fdr_val < 0.25:
            star = "***" if fdr_val < 0.001 else "**" if fdr_val < 0.01 else "*"
            x_max = max(abs(row["mean_R"]), abs(row["mean_NR"]))
            ax.text(x_max + 0.02, i, star, ha="left", va="center",
                    fontsize=10, fontweight="bold", color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(df["display"], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean Change (Post - Pre)", fontsize=10)
    ax.set_title("Signature Changes by Response\n(sorted by |Cohen's d|)",
                 fontsize=11)

    ax.legend(fontsize=9, loc="lower right", frameon=True, framealpha=0.9)
    ax.text(0.97, 0.12, "* FDR < 0.25  ** < 0.01  *** < 0.001",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, fontstyle="italic", color=COLORS["gray"])
    despine(ax)


# ======================================================================
# Panel B -- Effect size forest plot (Cohen's d)
# ======================================================================

def panel_B(ax, data: dict):
    """Horizontal forest plot of Cohen's d for each signature."""
    stats_df = data["stats_df"]

    if stats_df is None or len(stats_df) == 0:
        ax.text(0.5, 0.5, "No data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df = stats_df.sort_values("cohens_d", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    # Colours by direction
    colors = [
        COLORS["treated"] if d > 0 else COLORS["control"]
        for d in df["cohens_d"]
    ]

    # Marker size by FDR significance
    sizes = []
    for fdr_val in df["fdr"]:
        if pd.notna(fdr_val) and fdr_val < 0.25:
            sizes.append(100)
        else:
            sizes.append(40)

    ax.scatter(df["cohens_d"], y, c=colors, s=sizes, edgecolors="white",
               linewidths=0.8, zorder=3)

    # Connect dots to zero with thin lines
    for i, d in enumerate(df["cohens_d"]):
        ax.plot([0, d], [i, i], color=colors[i], alpha=0.4, lw=1.2, zorder=1)

    # Reference lines
    ax.axvline(0, color="black", lw=0.8, zorder=2)
    for ref in [-0.5, 0.5]:
        ax.axvline(ref, color=COLORS["gray"], ls="--", lw=0.7, alpha=0.6,
                   zorder=1)
    ax.text(0.5, len(df) + 0.3, "d = 0.5", ha="center", va="bottom",
            fontsize=7, color=COLORS["gray"])
    ax.text(-0.5, len(df) + 0.3, "d = -0.5", ha="center", va="bottom",
            fontsize=7, color=COLORS["gray"])

    ax.set_yticks(y)
    ax.set_yticklabels(df["display"], fontsize=9)
    ax.set_xlabel("Cohen's d (Responder vs Non-responder)", fontsize=10)
    ax.set_title("Effect Sizes: Responder vs Non-responder", fontsize=11)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=COLORS["treated"], label="Favours Responder"),
        mpatches.Patch(facecolor=COLORS["control"], label="Favours Non-resp"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COLORS["gray"], markersize=10,
                   label="FDR < 0.25 (large)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COLORS["gray"], markersize=6,
                   label="FDR >= 0.25 (small)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel C -- Signature predictive power (AUC)
# ======================================================================

def panel_C(ax, data: dict):
    """Horizontal bar chart of AUC for each signature."""
    stats_df = data["stats_df"]

    if stats_df is None or len(stats_df) == 0:
        ax.text(0.5, 0.5, "No data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df = stats_df.sort_values("auc", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    # Colour by threshold
    auc_threshold = 0.6
    colors = [
        COLORS["treated"] if a >= auc_threshold else COLORS["gray"]
        for a in df["auc"]
    ]

    ax.barh(y, df["auc"], color=colors, alpha=0.85, edgecolor="white",
            linewidth=0.5, height=0.7)

    # Reference line at 0.5 (random)
    ax.axvline(0.5, color=COLORS["highlight"], ls="--", lw=1.0, zorder=1)
    ax.text(0.5, len(df) + 0.3, "Random (0.5)", ha="center", va="bottom",
            fontsize=8, color=COLORS["highlight"])

    # Reference line at threshold
    ax.axvline(auc_threshold, color=COLORS["gray"], ls=":", lw=0.8,
               alpha=0.6, zorder=1)
    ax.text(auc_threshold, -0.8, f"AUC = {auc_threshold}",
            ha="center", va="top", fontsize=7, color=COLORS["gray"])

    # Annotate AUC values on bars
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["auc"] + 0.01, i, f"{row['auc']:.2f}",
                ha="left", va="center", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(df["display"], fontsize=9)
    ax.set_xlabel("Area Under the Curve (AUC)", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_title("Response Discrimination (AUC)", fontsize=11)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=COLORS["treated"], alpha=0.85,
                       label=f"AUC >= {auc_threshold}"),
        mpatches.Patch(facecolor=COLORS["gray"], alpha=0.85,
                       label=f"AUC < {auc_threshold}"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Supplementary Figure 4 individual panels."""
    print("Supplementary Figure 4: Outcome Correlation Details")

    try:
        data = _prepare_data()
    except Exception as exc:
        print(f"  ERROR preparing data: {exc}")
        gc.collect()
        print("  Done (no data).\n")
        return

    # ── Save individual panels ────────────────────────────────────────
    for panel_label, panel_func in [("A", panel_A), ("B", panel_B),
                                     ("C", panel_C)]:
        fig_p, ax_p = plt.subplots(figsize=(7, 6))
        panel_func(ax_p, data)
        fig_p.tight_layout()
        save_panel(fig_p, f"panel_{panel_label}", FIGURE_NAME, SUPP_OUTPUT)

    # ── Cleanup ───────────────────────────────────────────────────────
    if "adata" in data:
        del data["adata"]
    del data
    clear_cache()
    gc.collect()
    print("  Done.\n")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    apply_style()
    generate()
