"""
Figure 1 -- The Problem & sctrial Framework
============================================

Eight-panel figure combining the sctrial analytical pipeline overview,
pseudoreplication mechanism, multi-design support, design validation,
and empirical demonstration of pseudoreplication bias.

Layout:
  Row 1: [────────────── A: Pipeline (full width) ──────────────]
  Row 2: [  B: Pseudobulk  ] [  C: Mechanism  ] [  D: Designs  ] [ E: Validation ]
  Row 3: [      F: Paired verification      ] [     G: Coeff comparison     ]
  Row 4: [────────── H: P-value inflation (full width) ─────────]

Panels (reading order)
----------------------
A  sctrial analytical pipeline: Input → Pseudobulk + Analysis → Outputs.
B  Pseudobulk aggregation visual: cells → participant-level means.
C  Pseudoreplication mechanism: why cell-level analysis inflates.
D  Multi-design support: two-arm, single-arm, cross-sectional schematics.
E  Design validation: baseline balance between arms (parallel trends).
F  Paired-participant verification (cells per participant x visit).
G  Coefficient comparison: cell-level vs participant-level aggregation.
H  P-value comparison: -log10 scale, illustrating inflation at cell level.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy import stats

from .._shared import *  # noqa: F401,F403

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "Figure1_problem_framework"

# Pipeline color palette (matching README aesthetic but print-friendly)
_C = {
    "input_bg": "#E8F4FD",
    "input_border": "#0077B6",
    "input_text": "#023E8A",
    "analysis_bg": "#EDE9FE",
    "analysis_border": "#4338CA",
    "analysis_text": "#3730A3",
    "output_bg": "#ECFDF5",
    "output_border": "#059669",
    "output_text": "#047857",
    "did_fill": "#4338CA",
    "paired_fill": "#7C3AED",
    "cross_fill": "#E11D48",
    "arrow": "#334155",
    "muted": "#94A3B8",
    "bg_light": "#F8FAFC",
}


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_data() -> dict:
    """Load Sade-Feldman, score signatures, run DiD at both aggregation levels."""
    adata = get_sade_feldman()
    adata = harmonize_response(adata)

    # Ensure log1p_tpm layer exists
    if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])

    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="response_harmonized",
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
# Helper: rounded box with optional gradient effect
# ======================================================================

def _rounded_box(ax, xy, width, height, facecolor, edgecolor,
                 label=None, sublabel=None, label_size: float = 11, sublabel_size=8.5,
                 label_color="white", sublabel_color=None, alpha=1.0,
                 linewidth=1.5, pad=0.12, zorder=2):
    """Draw a rounded rectangle with centered label text."""
    x, y = xy
    rect = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad={pad}", facecolor=facecolor, alpha=alpha,
        edgecolor=edgecolor, linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(rect)
    cy = y + height / 2
    if sublabel_color is None:
        sublabel_color = label_color
    if label and sublabel:
        ax.text(x + width / 2, cy + height * 0.12, label,
                ha="center", va="center", fontsize=label_size,
                fontweight="bold", color=label_color, zorder=zorder + 1)
        ax.text(x + width / 2, cy - height * 0.18, sublabel,
                ha="center", va="center", fontsize=sublabel_size,
                color=sublabel_color, zorder=zorder + 1)
    elif label:
        ax.text(x + width / 2, cy, label,
                ha="center", va="center", fontsize=label_size,
                fontweight="bold", color=label_color, zorder=zorder + 1)


def _connect(ax, x1, y1, x2, y2, color=None, lw=2.0, style="-|>"):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    if color is None:
        color = _C["arrow"]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                        shrinkA=3, shrinkB=3),
        zorder=5,
    )


# ======================================================================
# Panel A -- Analytical pipeline (redesigned, full width)
# ======================================================================

def panel_A(ax):
    """sctrial analytical pipeline: Input -> Pseudobulk + Analysis -> Outputs."""
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # ── Column 1: INPUT (x = 0.2 – 3.6) ─────────────────────────────
    ax.text(1.9, 5.7, "INPUT", ha="center", va="center",
            fontsize=12, fontweight="bold", color=_C["input_text"],
            family="sans-serif")

    # AnnData box — clean, no internal matrix
    _rounded_box(ax, (0.2, 3.4), 3.4, 1.9,
                 facecolor=_C["input_bg"], edgecolor=_C["input_border"],
                 label="AnnData", sublabel="scRNA-seq\ncells x genes matrix",
                 label_size=13, sublabel_size=9,
                 label_color=_C["input_text"], sublabel_color=_C["muted"],
                 linewidth=1.8)

    # TrialDesign box — clean, metadata fields listed
    _rounded_box(ax, (0.2, 0.7), 3.4, 2.3,
                 facecolor=_C["input_bg"], edgecolor=_C["input_border"],
                 label="TrialDesign", sublabel=None,
                 label_size=13, label_color=_C["input_text"],
                 linewidth=1.8)
    # Metadata fields as a clean list
    fields = [
        ("participant_col", "#334155"),
        ("arm_col", COLORS["treated"]),
        ("visit_col", "#334155"),
        ("celltype_col", "#334155"),
    ]
    for j, (field, clr) in enumerate(fields):
        y_f = 2.35 - j * 0.38
        ax.text(1.9, y_f, field, fontsize=8.5, color=clr,
                ha="center", family="monospace", zorder=4)

    # ── Arrow: Input → Analysis ──────────────────────────────────────
    _connect(ax, 3.7, 3.1, 4.7, 3.1, lw=2.5)

    # ── Column 2: PSEUDOBULK + ANALYSIS (x = 4.8 – 10.8) ────────────
    ax.text(7.8, 5.7, "PSEUDOBULK  +  ANALYSIS", ha="center", va="center",
            fontsize=12, fontweight="bold", color=_C["analysis_text"],
            family="sans-serif")

    # Pseudobulk aggregation
    _rounded_box(ax, (4.8, 4.3), 6.0, 1.0,
                 facecolor="#0096C7", edgecolor="#0077B6",
                 label="Pseudobulk Aggregation",
                 sublabel="cells  -->  participant x visit means",
                 label_size=12, sublabel_size=8.5,
                 label_color="white", sublabel_color=(1, 1, 1, 0.85),
                 linewidth=1.8)
    _connect(ax, 7.8, 4.25, 7.8, 3.85, lw=2.5)

    # Statistical framework
    _rounded_box(ax, (4.8, 2.8), 6.0, 1.0,
                 facecolor=_C["did_fill"], edgecolor="#3730A3",
                 label="Statistical Framework",
                 sublabel="DiD / paired / cross-sectional",
                 label_size=12, sublabel_size=8.5,
                 label_color="white", sublabel_color=(1, 1, 1, 0.85),
                 linewidth=1.8)
    _connect(ax, 7.8, 2.75, 7.8, 2.3, lw=2.5)

    # Three analysis boxes side by side
    box_w = 1.87
    box_h = 0.95
    x_start = 4.8
    gap = 0.2
    labels_sub = [
        ("Paired\nContrasts", _C["paired_fill"], "#6D28D9"),
        ("Between-Arm\nTests", _C["cross_fill"], "#BE123C"),
        ("Gene Set\nEnrichment", "#D97706", "#B45309"),
    ]
    for i, (lab, fc, ec) in enumerate(labels_sub):
        x = x_start + i * (box_w + gap)
        _rounded_box(ax, (x, 1.2), box_w, box_h,
                     facecolor=fc, edgecolor=ec,
                     label=lab, label_size=10, label_color="white",
                     linewidth=1.5)
    _connect(ax, 7.8, 1.15, 7.8, 0.75, lw=2.5)

    # Power analysis (outlined)
    _rounded_box(ax, (4.8, -0.1), 6.0, 0.8,
                 facecolor="white", edgecolor=_C["analysis_border"],
                 label="Power Analysis & Study Planning",
                 label_size=11, label_color=_C["analysis_text"],
                 linewidth=1.8)

    # ── Arrow: Analysis → Outputs ────────────────────────────────────
    _connect(ax, 10.9, 3.1, 11.6, 3.1, lw=2.5)

    # ── Column 3: OUTPUTS (x = 11.7 – 15.8) ─────────────────────────
    ax.text(13.75, 5.7, "OUTPUTS", ha="center", va="center",
            fontsize=12, fontweight="bold", color=_C["output_text"],
            family="sans-serif")

    out_items = [
        ("Statistical Results", "effect sizes, p-values, FDR"),
        ("Volcano & Forest Plots", "gene- and signature-level"),
        ("GSEA Enrichment", "pathway-level analysis"),
        ("Power Curves", "sample size planning"),
    ]
    out_h = 0.95
    out_gap = 0.22
    for i, (title, sub) in enumerate(out_items):
        y = 4.3 - i * (out_h + out_gap)
        _rounded_box(ax, (11.7, y), 4.1, out_h,
                     facecolor=_C["output_bg"], edgecolor=_C["output_border"],
                     label=title, sublabel=sub,
                     label_size=10.5, sublabel_size=8.5,
                     label_color=_C["output_text"], sublabel_color=_C["muted"],
                     linewidth=1.3)


# ======================================================================
# Panel E -- Pseudobulk aggregation visual
# ======================================================================

def panel_E(ax, data: dict):
    """Scatter showing cells → participant-level aggregation."""
    adata = data["adata"]
    obs = adata.obs.copy()

    # Get cells from Pre visit only for clarity
    obs_pre = obs[obs["visit"] == "Pre"].copy()

    # Assign colors per participant
    pids = obs_pre["participant_id"].unique()
    pid_colors = {}
    cmap = plt.colormaps["Set2"]
    for i, pid in enumerate(pids):
        pid_colors[pid] = cmap(i / max(len(pids) - 1, 1))

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")

    # LEFT: many cells per participant (jittered scatter)
    rng = np.random.RandomState(0)
    n_show = min(80, len(obs_pre))  # subsample for visual clarity
    sample_idx = rng.choice(len(obs_pre), n_show, replace=False)
    sample = obs_pre.iloc[sample_idx]

    x_jitter = rng.uniform(0.0, 1.2, n_show)
    y_jitter = rng.uniform(0.15, 0.95, n_show)
    colors_left = [pid_colors[pid] for pid in sample["participant_id"]]

    ax.scatter(x_jitter, y_jitter, c=colors_left, s=8, alpha=0.6,
               edgecolors="none", zorder=3, rasterized=True)
    ax.text(0.6, 1.1, "Cell-level", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#334155")
    ax.text(0.6, 0.03, f"n = {len(obs_pre):,} cells", ha="center",
            va="center", fontsize=8, color=_C["muted"])

    # Arrow
    ax.annotate("aggregate", xy=(2.1, 0.55), xytext=(1.5, 0.55),
                fontsize=9, ha="center", va="bottom", color=_C["arrow"],
                arrowprops=dict(arrowstyle="-|>", lw=2, color=_C["arrow"],
                                shrinkA=5, shrinkB=5),
                zorder=5)

    # RIGHT: one dot per participant
    n_pids = len(pids)
    y_positions = np.linspace(0.2, 0.9, n_pids)
    for i, pid in enumerate(pids):
        ax.scatter(2.8, y_positions[i], c=[pid_colors[pid]], s=120,
                   edgecolors="white", linewidths=1.0, zorder=4)

    ax.text(2.8, 1.1, "Participant-level", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#334155")
    ax.text(2.8, 0.03, f"n = {n_pids} means", ha="center",
            va="center", fontsize=8, color=_C["muted"])

    # Bounding emphasis
    left_box = mpatches.FancyBboxPatch(
        (-0.2, 0.08), 1.6, 1.0,
        boxstyle="round,pad=0.08", facecolor="none",
        edgecolor=_C["muted"], linewidth=0.8, linestyle=":", zorder=1,
    )
    ax.add_patch(left_box)
    right_box = mpatches.FancyBboxPatch(
        (2.2, 0.08), 1.1, 1.0,
        boxstyle="round,pad=0.08", facecolor="none",
        edgecolor=_C["output_border"], linewidth=1.2, zorder=1,
    )
    ax.add_patch(right_box)


# ======================================================================
# Panel F -- Pseudoreplication mechanism
# ======================================================================

def panel_F(ax):
    """Conceptual diagram: why cell-level analysis inflates statistics."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5, 9.6, "Why Pseudoreplication Inflates",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="#334155")

    col_bad = COLORS["highlight"]
    col_good = COLORS["treated"]

    # ── LEFT: Cell-level (inflated) ──────────────────────────────────
    ax.text(2.5, 8.6, "Cell-level", ha="center", fontsize=10,
            fontweight="bold", color=col_bad)

    # Two participant blobs with many small dots
    for p_idx, (cx, cy) in enumerate([(1.8, 6.5), (3.2, 6.5)]):
        # Participant circle outline
        circ = Circle((cx, cy), 1.0, facecolor=col_bad, alpha=0.06,
                           edgecolor=col_bad, linewidth=1.2, linestyle="--",
                           zorder=1)
        ax.add_patch(circ)
        ax.text(cx, cy + 1.2, f"Participant {p_idx + 1}",
                ha="center", fontsize=7, color=col_bad)
        # Many small dots inside
        rng = np.random.RandomState(p_idx + 10)
        n_dots = 30
        angles = rng.uniform(0, 2 * np.pi, n_dots)
        radii = rng.uniform(0, 0.75, n_dots)
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)
        ax.scatter(xs, ys, s=6, c=col_bad, alpha=0.5, edgecolors="none",
                   zorder=2)

    # Annotations
    ax.text(2.5, 4.9, "n = 1,000+ observations", ha="center", fontsize=8,
            color=col_bad, fontweight="bold")
    ax.text(2.5, 4.3, "SE artificially small", ha="center", fontsize=8,
            color=col_bad)

    # Small SE bar
    ax.plot([1.5, 3.5], [3.6, 3.6], color=col_bad, lw=2.5, zorder=3)
    ax.plot([2.5, 2.5], [3.4, 3.8], color=col_bad, lw=1.5, zorder=3)
    # CI whiskers (narrow)
    ax.plot([2.2, 2.8], [3.6, 3.6], color=col_bad, lw=6, alpha=0.3, zorder=2)

    ax.text(2.5, 2.8, "p < 0.001", ha="center", fontsize=10,
            fontweight="bold", color=col_bad,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=col_bad,
                      alpha=0.1, edgecolor=col_bad, linewidth=0.8))

    # Cross mark
    ax.text(2.5, 1.8, "X  False significance", ha="center", fontsize=9,
            color=col_bad, fontweight="bold")

    # ── RIGHT: Participant-level (correct) ───────────────────────────
    ax.text(7.5, 8.6, "Participant-level", ha="center", fontsize=10,
            fontweight="bold", color=col_good)

    # Two large dots (one per participant)
    for p_idx, (cx, cy) in enumerate([(6.8, 6.5), (8.2, 6.5)]):
        ax.scatter(cx, cy, s=300, c=col_good, edgecolors="white",
                   linewidths=1.5, zorder=3)
        ax.text(cx, cy + 1.2, f"Participant {p_idx + 1}",
                ha="center", fontsize=7, color=col_good)

    ax.text(7.5, 4.9, "n = 2 observations", ha="center", fontsize=8,
            color=col_good, fontweight="bold")
    ax.text(7.5, 4.3, "SE reflects true variability", ha="center", fontsize=8,
            color=col_good)

    # Large SE bar
    ax.plot([6.5, 8.5], [3.6, 3.6], color=col_good, lw=2.5, zorder=3)
    ax.plot([7.5, 7.5], [3.4, 3.8], color=col_good, lw=1.5, zorder=3)
    # CI whiskers (wide)
    ax.plot([6.2, 8.8], [3.6, 3.6], color=col_good, lw=6, alpha=0.3, zorder=2)

    ax.text(7.5, 2.8, "p = 0.42", ha="center", fontsize=10,
            fontweight="bold", color=col_good,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=col_good,
                      alpha=0.1, edgecolor=col_good, linewidth=0.8))

    # Check mark
    ax.text(7.5, 1.8, "Correct inference", ha="center", fontsize=9,
            color=col_good, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=col_good,
                      alpha=0.08, edgecolor=col_good, linewidth=0.6))

    # Divider
    ax.plot([5, 5], [2.0, 9.0], color=_C["muted"], lw=0.8,
            linestyle=":", zorder=0)

    # Bottom message
    ax.text(5, 0.8, "Cells within a participant share the same biology",
            ha="center", va="center", fontsize=9, color="#555555",
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F8F8",
                      edgecolor="#CCCCCC", linewidth=0.8))


# ======================================================================
# Panel G -- Multi-design support
# ======================================================================

def panel_G(ax):
    """Three mini-schematics showing sctrial's supported study designs."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5, 9.6, "Supported Study Designs",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="#334155")

    col_tx = COLORS["treated"]
    col_ctrl = COLORS["control"]

    designs = [
        {
            "y_center": 7.5,
            "title": "Two-Arm Longitudinal",
            "formula": r"$\beta_{\mathrm{DiD}}$",
            "datasets": "Melanoma",
            "color1": col_tx, "color2": col_ctrl,
            "labels": ("Treatment", "Control"),
            "timepoints": ("Pre", "Post"),
        },
        {
            "y_center": 5.0,
            "title": "Single-Arm Paired",
            "formula": r"$\beta_{\Delta}$",
            "datasets": "Vaccine, AML, CAR-T",
            "color1": col_tx, "color2": None,
            "labels": ("Treatment",),
            "timepoints": ("Pre", "Post"),
        },
        {
            "y_center": 2.5,
            "title": "Cross-Sectional",
            "formula": "Hedges' g",
            "datasets": "COVID-19",
            "color1": "#E11D48", "color2": "#0077B6",
            "labels": ("Severe", "Mild"),
            "timepoints": None,
        },
    ]

    for d in designs:
        yc = d["y_center"]

        # Background band
        band = mpatches.FancyBboxPatch(
            (0.3, yc - 1.0), 9.4, 2.0,
            boxstyle="round,pad=0.1", facecolor="#F8FAFC",
            edgecolor="#E2E8F0", linewidth=0.8, zorder=0,
        )
        ax.add_patch(band)

        # Title and formula
        ax.text(0.6, yc + 0.7, d["title"], fontsize=9.5,
                fontweight="bold", color="#334155", va="center", zorder=3)
        ax.text(9.4, yc + 0.7, d["formula"], fontsize=10,
                color=d["color1"], ha="right", va="center",
                fontweight="bold", zorder=3)

        # Dataset names
        ax.text(9.4, yc - 0.7, d["datasets"], fontsize=7.5,
                color=_C["muted"], ha="right", va="center",
                fontstyle="italic", zorder=3)

        if d["timepoints"] is not None:
            # Longitudinal design: arrows from Pre to Post
            pre_x, post_x = 3.5, 7.0
            ax.text(pre_x, yc + 0.7, d["timepoints"][0], fontsize=8,
                    ha="center", color=_C["muted"], fontweight="bold", zorder=3)
            ax.text(post_x, yc + 0.7, d["timepoints"][1], fontsize=8,
                    ha="center", color=_C["muted"], fontweight="bold", zorder=3)

            for arm_idx, (lbl, clr) in enumerate(
                zip(d["labels"], [d["color1"], d["color2"]])
            ):
                if clr is None:
                    break
                y_arm = yc + 0.05 - arm_idx * 0.65

                # Pre circle
                ax.scatter(pre_x, y_arm, s=80, c=clr, edgecolors="white",
                           linewidths=0.8, zorder=4)
                # Post circle
                ax.scatter(post_x, y_arm, s=80, c=clr, edgecolors="white",
                           linewidths=0.8, zorder=4)
                # Arrow
                ax.annotate(
                    "", xy=(post_x - 0.3, y_arm), xytext=(pre_x + 0.3, y_arm),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color=clr,
                                    alpha=0.6, shrinkA=2, shrinkB=2),
                    zorder=3,
                )
                # Arm label
                ax.text(2.0, y_arm, lbl, fontsize=7.5, color=clr,
                        va="center", fontweight="semibold", zorder=3)

            if len(d["labels"]) == 1:
                # Single arm — show just the one arm
                pass
        else:
            # Cross-sectional: two groups at one timepoint
            tp_x = 5.5
            ax.text(tp_x, yc + 0.7, "Single timepoint", fontsize=8,
                    ha="center", color=_C["muted"], fontweight="bold", zorder=3)

            for arm_idx, (lbl, clr) in enumerate(
                zip(d["labels"], [d["color1"], d["color2"]])
            ):
                y_arm = yc + 0.05 - arm_idx * 0.65
                # Single circle
                ax.scatter(tp_x, y_arm, s=100, c=clr, edgecolors="white",
                           linewidths=0.8, zorder=4)
                ax.text(2.0, y_arm, lbl, fontsize=7.5, color=clr,
                        va="center", fontweight="semibold", zorder=3)

            # Double-headed arrow between groups
            ax.annotate(
                "", xy=(tp_x, yc + 0.05), xytext=(tp_x, yc - 0.6),
                arrowprops=dict(arrowstyle="<->", lw=1.5,
                                color=_C["muted"], shrinkA=8, shrinkB=8),
                zorder=3,
            )


# ======================================================================
# Panel H -- Design validation: baseline balance
# ======================================================================

def panel_H(ax, data: dict):
    """Baseline balance: Pre-treatment signature scores by response group."""
    adata = data["adata"]
    sig_cols = data["sig_cols"]

    # Filter to Pre-treatment cells only
    pre_mask = adata.obs["visit"] == "Pre"
    obs_pre = adata.obs.loc[pre_mask].copy()
    for c in sig_cols:
        if c in adata.obs.columns:
            obs_pre[c] = adata.obs.loc[pre_mask, c].values

    # Aggregate to participant level
    pid_col = "participant_id"
    arm_col = "response_harmonized"
    pb = obs_pre.groupby(pid_col).agg(
        {arm_col: "first", **{c: "mean" for c in sig_cols if c in obs_pre.columns}}
    )

    # Select top 6 signatures for readability
    top_sigs = sig_cols[:6]
    display_names = [sig_display(s) for s in top_sigs]

    # Build tidy data for boxplot
    rows = []
    for sig, dname in zip(top_sigs, display_names):
        if sig not in pb.columns:
            continue
        for _, row in pb.iterrows():
            rows.append({
                "signature": dname,
                "response": row[arm_col],
                "score": row[sig],
            })
    df = pd.DataFrame(rows)

    # Paired boxplot
    import seaborn as sns
    sns.boxplot(
        data=df, x="signature", y="score", hue="response",
        palette={"Responder": COLORS["treated"],
                 "Non-responder": COLORS["control"]},
        width=0.6, linewidth=0.8, fliersize=3,
        ax=ax, zorder=2,
    )
    sns.stripplot(
        data=df, x="signature", y="score", hue="response",
        palette={"Responder": COLORS["treated"],
                 "Non-responder": COLORS["control"]},
        dodge=True, size=3, alpha=0.5, linewidth=0.3,
        ax=ax, zorder=3, legend=False,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Pre-treatment score\n(participant mean)")
    ax.set_title("Baseline Balance Between Arms", fontsize=11)
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=8)

    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], fontsize=8, loc="upper right",
              frameon=True, framealpha=0.9)

    # Mann-Whitney test annotation for each signature
    for i, (sig, dname) in enumerate(zip(top_sigs, display_names)):
        if sig not in pb.columns:
            continue
        resp = pb.loc[pb[arm_col] == "Responder", sig].dropna().values
        nresp = pb.loc[pb[arm_col] == "Non-responder", sig].dropna().values
        if len(resp) >= 2 and len(nresp) >= 2:
            _, p = stats.mannwhitneyu(resp, nresp, alternative="two-sided")
            star = "n.s." if p > 0.05 else ("*" if p > 0.01 else "**")
            y_max = df.loc[df["signature"] == dname, "score"].max()
            ax.text(i, y_max + 0.02, star, ha="center", fontsize=7,
                    color=_C["muted"])

    despine(ax)


# ======================================================================
# Panel B -- Paired-participant verification (KEPT AS-IS)
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
# Panel C -- Coefficient comparison (cell vs participant) (KEPT AS-IS)
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
# Panel D -- P-value comparison (inflation demonstration) (KEPT AS-IS)
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
    """Create and save Figure 1 as individual panels.

    Panel order follows reading order (left-to-right, top-to-bottom):
      Row 1: A — Pipeline (full width)
      Row 2: B — Pseudobulk aggregation, C — Mechanism, D — Designs, E — Validation
      Row 3: F — Paired verification, G — Coefficient comparison
      Row 4: H — P-value inflation (full width)
    """
    print("Figure 1: The Problem & sctrial Framework")
    data = _prepare_data()

    # ── Panel A: Pipeline (full width, wide) ─────────────────────────
    fig_a, ax_a = plt.subplots(figsize=(16, 6))
    panel_A(ax_a)
    fig_a.tight_layout()
    save_panel(fig_a, "panel_A", FIGURE_NAME, MAIN_OUTPUT)

    # ── Row 2: B, C, D, E (conceptual + data panels) ────────────────
    # Map: reading-order label -> (function, needs_data, figsize)
    panel_specs = [
        ("B", panel_E, True, (6, 4.5)),     # pseudobulk aggregation
        ("C", panel_F, False, (6, 6)),       # mechanism
        ("D", panel_G, False, (6, 6)),       # designs
        ("E", panel_H, True, (6, 5)),        # validation
    ]
    for label, func, needs_data, figsize in panel_specs:
        fig_p, ax_p = plt.subplots(figsize=figsize)
        if needs_data:
            func(ax_p, data)
        else:
            func(ax_p)
        fig_p.tight_layout()
        save_panel(fig_p, f"panel_{label}", FIGURE_NAME, MAIN_OUTPUT)

    # ── Row 3–4: F, G, H (empirical panels) ─────────────────────────
    empirical = [
        ("F", panel_B, (8, 6)),    # paired verification
        ("G", panel_C, (8, 6)),    # coefficient comparison
        ("H", panel_D, (10, 6)),   # p-value inflation (wider)
    ]
    for label, func, figsize in empirical:
        fig_p, ax_p = plt.subplots(figsize=figsize)
        func(ax_p, data)
        fig_p.tight_layout()
        save_panel(fig_p, f"panel_{label}", FIGURE_NAME, MAIN_OUTPUT)

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
