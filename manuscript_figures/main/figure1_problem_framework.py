"""
Figure 1 -- The Problem & sctrial Framework
============================================

Four-panel figure (2x2) combining the trial design overview with
empirical demonstration of pseudoreplication bias.

Panels
------
A  Conceptual schematic of a longitudinal two-arm trial design.
B  Paired-participant verification (cells per participant x visit).
C  Coefficient comparison: cell-level vs participant-level aggregation.
D  P-value comparison: -log10 scale, illustrating inflation at cell level.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from .._shared import *  # noqa: F401,F403

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "Figure1_problem_framework"
FIGSIZE = (16, 11)


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_data() -> dict:
    """Load Sade-Feldman, score signatures, run DiD at both aggregation levels."""
    adata = get_sade_feldman()

    # Ensure log1p_tpm layer exists
    if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])

    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="response",
        arm_treated="Responder",
        arm_control="Non-responder",
    )
    visits = ("Pre", "Post")

    # Paired-participant verification
    pair_info = verify_paired_participants(
        adata.obs,
        visit_col="visit",
        visits=visits,
        participant_col="participant_id",
    )

    # DiD at cell level
    res_cell = did_table(
        adata,
        features=sig_cols,
        design=design,
        visits=visits,
        layer="log1p_tpm",
        standardize=True,
        aggregate="cell",
    )

    # DiD at participant-visit level (recommended)
    res_part = did_table(
        adata,
        features=sig_cols,
        design=design,
        visits=visits,
        layer="log1p_tpm",
        standardize=True,
        aggregate="participant_visit",
    )

    return dict(
        adata=adata,
        sig_cols=sig_cols,
        design=design,
        visits=visits,
        pair_info=pair_info,
        res_cell=res_cell,
        res_part=res_part,
    )


# ======================================================================
# Panel A -- Conceptual trial schematic
# ======================================================================

def panel_A(ax):
    """Draw a publication-quality schematic of a two-arm longitudinal trial."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    col_treat = COLORS["treated"]
    col_ctrl = COLORS["control"]

    # --- Title ---
    ax.text(5, 9.7, "Longitudinal Two-Arm Trial Design",
            ha="center", va="top", fontsize=13, fontweight="bold")

    # --- Timepoint column headers ---
    ax.text(3.2, 9.05, "Baseline", ha="center", va="center",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f0f0",
                      edgecolor="#cccccc", linewidth=0.8))
    ax.text(7.8, 9.05, "Follow-up", ha="center", va="center",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f0f0",
                      edgecolor="#cccccc", linewidth=0.8))

    # --- Arrow connecting Baseline -> Follow-up ---
    ax.annotate(
        "", xy=(7.0, 9.05), xytext=(4.0, 9.05),
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#555555",
                        shrinkA=15, shrinkB=15),
    )

    # --- Arm group boxes (rounded rectangles behind each arm) ---
    treat_bg = mpatches.FancyBboxPatch(
        (1.2, 5.55), 8.3, 3.1,
        boxstyle="round,pad=0.15", facecolor=col_treat, alpha=0.06,
        edgecolor=col_treat, linewidth=1.5, linestyle="--",
    )
    ax.add_patch(treat_bg)
    ctrl_bg = mpatches.FancyBboxPatch(
        (1.2, 1.45), 8.3, 3.1,
        boxstyle="round,pad=0.15", facecolor=col_ctrl, alpha=0.06,
        edgecolor=col_ctrl, linewidth=1.5, linestyle="--",
    )
    ax.add_patch(ctrl_bg)

    # --- Arm labels (inside group boxes, left side) ---
    ax.text(1.55, 7.1, "Treatment", ha="center", va="center",
            fontsize=10, fontweight="bold", color=col_treat, rotation=90)
    ax.text(1.55, 3.0, "Control", ha="center", va="center",
            fontsize=10, fontweight="bold", color=col_ctrl, rotation=90)

    # --- Draw participant boxes and arrows ---
    def _draw_arm(y_center, color, n_participants=4):
        """Draw participant rectangles at Pre and Post with connecting arrows."""
        box_w, box_h = 1.6, 0.52
        gap = 0.68
        y_start = y_center + (n_participants - 1) * gap / 2

        for i in range(n_participants):
            y = y_start - i * gap

            # Pre box — solid fill, white text
            pre_rect = mpatches.FancyBboxPatch(
                (2.7, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.06", facecolor=color, alpha=0.20,
                edgecolor=color, linewidth=1.0,
            )
            ax.add_patch(pre_rect)
            ax.text(2.7 + box_w / 2, y, f"Patient {i + 1}",
                    ha="center", va="center", fontsize=7.5, color=color,
                    fontweight="semibold")

            # Post box
            post_rect = mpatches.FancyBboxPatch(
                (6.7, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.06", facecolor=color, alpha=0.20,
                edgecolor=color, linewidth=1.0,
            )
            ax.add_patch(post_rect)
            ax.text(6.7 + box_w / 2, y, f"Patient {i + 1}",
                    ha="center", va="center", fontsize=7.5, color=color,
                    fontweight="semibold")

            # Arrow connecting pre -> post (solid, clean)
            ax.annotate(
                "", xy=(6.65, y), xytext=(4.35, y),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color,
                                alpha=0.5, shrinkA=2, shrinkB=2),
            )

    _draw_arm(y_center=7.1, color=col_treat, n_participants=4)
    _draw_arm(y_center=3.0, color=col_ctrl, n_participants=4)

    # --- DiD equation at the bottom ---
    ax.text(5, 0.7,
            r"$\beta_{\mathrm{DiD}} = "
            r"(\bar{Y}_{\mathrm{treat,post}} - \bar{Y}_{\mathrm{treat,pre}})"
            r" - (\bar{Y}_{\mathrm{ctrl,post}} - \bar{Y}_{\mathrm{ctrl,pre}})$",
            ha="center", va="center", fontsize=10, color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                      edgecolor="#cccccc", linewidth=0.8))


# ======================================================================
# Panel B -- Paired-participant verification
# ======================================================================

def panel_B(ax, data: dict):
    """Grouped bar chart of cells per participant x visit, colored by response."""
    adata = data["adata"]
    obs = adata.obs.copy()

    # Count cells per participant x visit
    counts = (
        obs.groupby(["participant_id", "visit", "response"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    # Ensure consistent visit ordering
    counts["visit"] = pd.Categorical(
        counts["visit"], categories=["Pre", "Post"], ordered=True,
    )
    counts = counts.sort_values(["response", "participant_id", "visit"])

    # Assign x-positions: group by participant, offset by visit
    participants = counts["participant_id"].unique()
    pid_order = {pid: i for i, pid in enumerate(participants)}
    bar_width = 0.35

    for _, row in counts.iterrows():
        x_base = pid_order[row["participant_id"]]
        offset = -bar_width / 2 if row["visit"] == "Pre" else bar_width / 2
        color = COLORS["treated"] if row["response"] == "Responder" else COLORS["control"]
        alpha = 1.0 if row["visit"] == "Post" else 0.6
        ax.bar(x_base + offset, row["n_cells"], width=bar_width,
               color=color, alpha=alpha, edgecolor="white", linewidth=0.5)

    # Axis formatting
    ax.set_xticks(range(len(participants)))
    ax.set_xticklabels(
        [str(p)[:6] for p in participants],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_xlabel("Participant")
    ax.set_ylabel("Number of cells")
    ax.set_title("Paired Participants: Cells per Visit", fontsize=11)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=COLORS["treated"], label="Responder"),
        mpatches.Patch(facecolor=COLORS["control"], label="Non-responder"),
        mpatches.Patch(facecolor=COLORS["gray"], alpha=0.6, label="Pre"),
        mpatches.Patch(facecolor=COLORS["gray"], alpha=1.0, label="Post"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
              frameon=True, framealpha=0.9)

    # Annotation: paired count
    pair_info = data["pair_info"]
    ax.text(
        0.02, 0.95,
        f"{pair_info['n_paired']}/{pair_info['n_total']} participants paired",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.8),
    )
    despine(ax)


# ======================================================================
# Panel C -- Coefficient comparison (cell vs participant)
# ======================================================================

def panel_C(ax, data: dict):
    """Scatter of cell-level vs participant-level beta_DiD with identity line."""
    res_cell = data["res_cell"].set_index("feature")
    res_part = data["res_part"].set_index("feature")
    common = res_cell.index.intersection(res_part.index)

    beta_cell = res_cell.loc[common, "beta_DiD"].values
    beta_part = res_part.loc[common, "beta_DiD"].values

    # Colour by direction of participant-level effect (treated=blue, control=orange)
    colors = [COLORS["treated"] if b > 0 else COLORS["control"] for b in beta_part]

    ax.scatter(beta_cell, beta_part, c=colors, s=60, edgecolors="white",
               linewidths=0.5, zorder=3)

    # Identity line
    lim_lo = min(beta_cell.min(), beta_part.min()) * 1.15
    lim_hi = max(beta_cell.max(), beta_part.max()) * 1.15
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color=COLORS["gray"],
            lw=1, zorder=1, label="Identity")
    ax.axhline(0, color=COLORS["gray"], lw=0.5, ls=":", zorder=0)
    ax.axvline(0, color=COLORS["gray"], lw=0.5, ls=":", zorder=0)

    # Annotate points using adjustText to prevent overlaps
    texts = []
    for feat, xv, yv in zip(common, beta_cell, beta_part):
        t = ax.text(xv, yv, sig_display(feat), fontsize=7, alpha=0.85)
        texts.append(t)

    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color=COLORS["gray"], lw=0.4,
                            shrinkA=5, shrinkB=3),
            force_points=(0.6, 0.6),
            force_text=(1.0, 1.0),
            expand_points=(2.0, 2.0),
            expand_text=(1.3, 1.3),
        )
    except ImportError:
        pass  # fall back to raw placement

    # Correlation
    r, p = stats.pearsonr(beta_cell, beta_part)
    ax.text(
        0.05, 0.95,
        f"r = {r:.2f}, p = {p:.1e}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.8),
    )

    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (cell-level)")
    ax.set_ylabel(r"$\beta_{\mathrm{DiD}}$ (participant-level)")
    ax.set_title("Effect Size: Cell vs Participant Aggregation", fontsize=11)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=COLORS["treated"], label="Positive effect"),
        mpatches.Patch(facecolor=COLORS["control"], label="Negative effect"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel D -- P-value comparison (inflation demonstration)
# ======================================================================

def panel_D(ax, data: dict):
    """Horizontal bar chart of -log10(p) at cell vs participant level."""
    res_cell = data["res_cell"].set_index("feature")
    res_part = data["res_part"].set_index("feature")
    common = res_cell.index.intersection(res_part.index)

    # Build comparison DataFrame
    df = pd.DataFrame({
        "feature": common,
        "p_cell": res_cell.loc[common, "p_DiD"].values,
        "p_part": res_part.loc[common, "p_DiD"].values,
    })
    df["nlog10_cell"] = -np.log10(df["p_cell"].clip(lower=1e-300))
    df["nlog10_part"] = -np.log10(df["p_part"].clip(lower=1e-300))
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("nlog10_cell", ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(df))
    bar_h = 0.35

    # Bars
    ax.barh(y_pos - bar_h / 2, df["nlog10_cell"], height=bar_h,
            color=COLORS["highlight"], alpha=0.8, label="Cell-level", zorder=2)
    ax.barh(y_pos + bar_h / 2, df["nlog10_part"], height=bar_h,
            color=COLORS["treated"], alpha=0.8, label="Participant-level", zorder=2)

    # Threshold line: p = 0.05
    thresh = -np.log10(0.05)
    ax.axvline(thresh, color=COLORS["gray"], ls="--", lw=1, zorder=1)
    ax.text(thresh + 0.1, len(df) - 0.5, "p = 0.05", fontsize=8,
            va="bottom", color=COLORS["gray"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"], fontsize=8)
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.set_title("P-value Inflation: Cell vs Participant Level", fontsize=11)

    ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Figure 1 individual panels."""
    print("Figure 1: The Problem & sctrial Framework")
    data = _prepare_data()

    # ── Save individual panels ────────────────────────────────────────
    for panel_label, panel_func, needs_data in [
        ("A", panel_A, False),
        ("B", panel_B, True),
        ("C", panel_C, True),
        ("D", panel_D, True),
    ]:
        fig_p, ax_p = plt.subplots(figsize=(8, 6))
        if needs_data:
            panel_func(ax_p, data)
        else:
            panel_func(ax_p)
        fig_p.tight_layout()
        save_panel(fig_p, f"panel_{panel_label}", FIGURE_NAME, MAIN_OUTPUT)

    # ── Cleanup ───────────────────────────────────────────────────────
    del data["adata"]
    del data
    gc.collect()
    print("  Done.\n")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    apply_style()
    generate()
