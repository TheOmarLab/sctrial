"""
Supplementary Figure 8 — Individual Participant Heterogeneity
=============================================================

Three-panel figure (GridSpec: top spans full width, bottom 2 panels)
showing individual-level treatment effect heterogeneity from the
Sade-Feldman immunotherapy dataset.

Panels
------
A  Strip plot of individual participant effects across signatures.
B  Box plots by response status for the most variable signatures.
C  Horizontal bar chart of effect heterogeneity (SD) per signature.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .._shared import *  # noqa: F401,F403

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "SuppFig8_individual_heterogeneity"
FIGSIZE = (18, 12)


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_data() -> dict:
    """Load Sade-Feldman, score signatures, compute individual effects."""
    adata = get_sade_feldman()

    # Ensure log1p_tpm layer
    if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])

    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")
    adata = harmonize_response(adata)

    # Compute pseudobulk per participant x visit
    obs = adata.obs.copy()
    pb = (
        obs.groupby(["participant_id", "visit"], observed=True)[sig_cols]
        .mean()
        .reset_index()
    )

    # Add response column
    pid_resp = obs.groupby("participant_id", observed=True)["response_harmonized"].first()
    pb["response"] = pb["participant_id"].map(pid_resp)

    # Compute individual effects: Post - Pre for each participant
    pre = pb[pb["visit"] == "Pre"].set_index("participant_id")
    post = pb[pb["visit"] == "Post"].set_index("participant_id")
    common_pids = pre.index.intersection(post.index)

    if len(common_pids) == 0:
        raise RuntimeError("No paired participants found (Pre & Post)")

    effects = post.loc[common_pids, sig_cols] - pre.loc[common_pids, sig_cols]
    effects["response"] = pid_resp.loc[common_pids]
    effects = effects.reset_index()

    print(f"  Individual effects: {len(common_pids)} participants, {len(sig_cols)} signatures")

    return dict(
        adata=adata,
        sig_cols=sig_cols,
        effects=effects,
    )


# ======================================================================
# Panel A — Strip plot of individual effects
# ======================================================================

def panel_A(ax, data: dict):
    """Strip plot: individual participant effects across signatures."""
    effects = data["effects"]
    sig_cols = data["sig_cols"]

    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "Individual effects data unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=COLORS["gray"])
        ax.axis("off")
        return

    # Melt to long format
    df_long = effects.melt(
        id_vars=["participant_id", "response"],
        value_vars=sig_cols,
        var_name="signature",
        value_name="effect",
    )
    df_long["display"] = df_long["signature"].map(sig_display)

    # Sort signatures by mean effect
    sig_order = (
        df_long.groupby("display", observed=True)["effect"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    # Color by response
    resp_colors = {
        "Responder": COLORS["treated"],
        "Non-responder": COLORS["control"],
    }

    x_positions = {sig: i for i, sig in enumerate(sig_order)}
    jitter_width = 0.2
    rng = np.random.default_rng(42)

    for resp, color in resp_colors.items():
        sub = df_long[df_long["response"] == resp]
        x_vals = sub["display"].map(x_positions).values
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(sub))
        ax.scatter(
            x_vals + jitter, sub["effect"].values,
            c=color, alpha=0.5, s=20, edgecolors="none",
            label=resp, zorder=2,
        )

    # Horizontal mean lines
    for sig in sig_order:
        sub = df_long[df_long["display"] == sig]
        mean_val = sub["effect"].mean()
        x = x_positions[sig]
        ax.hlines(mean_val, x - 0.35, x + 0.35, color="black",
                  linewidth=2.0, zorder=3)

    # Reference line at 0
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0,
               alpha=0.5)

    ax.set_xticks(range(len(sig_order)))
    ax.set_xticklabels(sig_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Individual effect (Post - Pre)")
    ax.set_title("Individual Participant Effects Across Signatures",
                 fontsize=11, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.6, label="Responder"),
        mpatches.Patch(color=COLORS["control"], alpha=0.6, label="Non-responder"),
        Line2D([0], [0], color="black", linewidth=2.0, label="Group mean"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel B — Box plots by response for most variable signatures
# ======================================================================

def panel_B(ax, data: dict):
    """Grouped box plots: Responder vs Non-responder for top 6 variable sigs."""
    effects = data["effects"]
    sig_cols = data["sig_cols"]

    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "Individual effects data unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=COLORS["gray"])
        ax.axis("off")
        return

    # Find top 6 most variable signatures
    sds = effects[sig_cols].std().sort_values(ascending=False)
    top_sigs = sds.head(6).index.tolist()

    df_long = effects.melt(
        id_vars=["participant_id", "response"],
        value_vars=top_sigs,
        var_name="signature",
        value_name="effect",
    )
    df_long["display"] = df_long["signature"].map(sig_display)

    # Ensure consistent ordering by SD
    display_order = [sig_display(s) for s in top_sigs]
    df_long["display"] = pd.Categorical(df_long["display"],
                                         categories=display_order,
                                         ordered=True)

    resp_colors = {
        "Responder": COLORS["treated"],
        "Non-responder": COLORS["control"],
    }

    # Plot grouped boxplot
    x_positions = {sig: i for i, sig in enumerate(display_order)}
    box_width = 0.35

    for resp_idx, (resp, color) in enumerate(resp_colors.items()):
        sub = df_long[df_long["response"] == resp]
        offset = -0.2 if resp_idx == 0 else 0.2

        for sig in display_order:
            sig_data = sub[sub["display"] == sig]["effect"].dropna()
            if len(sig_data) == 0:
                continue
            x = x_positions[sig] + offset
            bp = ax.boxplot(
                sig_data, positions=[x], widths=box_width,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                markerfacecolor=color, markeredgecolor="none"),
                medianprops=dict(color="white", linewidth=1.5),
                boxprops=dict(facecolor=color, alpha=0.7, edgecolor=color),
                whiskerprops=dict(color=color, linewidth=1.0),
                capprops=dict(color=color, linewidth=1.0),
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0,
               alpha=0.5)

    ax.set_xticks(range(len(display_order)))
    ax.set_xticklabels(display_order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Individual effect (Post - Pre)")
    ax.set_title("Effects by Response (Top 6 Most Variable)",
                 fontsize=11, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.7, label="Responder"),
        mpatches.Patch(color=COLORS["control"], alpha=0.7, label="Non-responder"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="best",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel C — Effect heterogeneity (SD)
# ======================================================================

def panel_C(ax, data: dict):
    """Horizontal bar chart of SD of individual effects per signature."""
    effects = data["effects"]
    sig_cols = data["sig_cols"]

    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "Individual effects data unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=COLORS["gray"])
        ax.axis("off")
        return

    sds = effects[sig_cols].std().sort_values(ascending=True)
    display_names = [sig_display(s) for s in sds.index]

    y_pos = np.arange(len(sds))
    median_sd = sds.median()

    colors = [
        COLORS["treated"] if v >= median_sd else COLORS["control"]
        for v in sds.values
    ]

    ax.barh(y_pos, sds.values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(median_sd, color="black", linewidth=0.8, linestyle="--",
               zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel("SD of individual effects (Post - Pre)")
    ax.set_title("Effect Heterogeneity Across Signatures",
                 fontsize=11, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.85,
                       label="Above median"),
        mpatches.Patch(color=COLORS["control"], alpha=0.85,
                       label="Below median"),
        Line2D([0], [0], color="black", linewidth=0.8, linestyle="--",
               label=f"Median SD = {median_sd:.3f}"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Supplementary Figure 8 individual panels."""
    print("Supplementary Figure 8: Individual Participant Heterogeneity")
    apply_style()

    try:
        data = _prepare_data()
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return

    # ── Save individual panels ────────────────────────────────────────
    for panel_label, panel_func, psize in [
        ("A_strip_plot", panel_A, (16, 6)),
        ("B_boxplots_response", panel_B, (8, 6)),
        ("C_heterogeneity_sd", panel_C, (8, 6)),
    ]:
        fig_p, ax_p = plt.subplots(figsize=psize)
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
