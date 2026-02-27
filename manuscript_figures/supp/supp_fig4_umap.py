"""
Supplementary Figure 4 -- Sade-Feldman UMAP (Hierarchical Data Structure)
=========================================================================

Single panel illustrating the hierarchical structure of single-cell
trial data: cells cluster by participant, with participants grouped by
treatment response.  This was previously Figure 1B in the 12-panel
manuscript.

Panel
-----
UMAP-like scatter showing simulated participant clusters colored by
response status (Responder vs Non-responder).
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    apply_style,
    clear_cache,
    despine,
    save_figure,
    save_panel,
)

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "SuppFig4_umap"
FIGSIZE = (8, 7)

N_PARTICIPANTS = 6      # 3 R + 3 NR
CELLS_PER_PARTICIPANT = 40
CLUSTER_SPREAD = 0.8     # within-participant spread
RNG_SEED = 42


# ======================================================================
# Data simulation
# ======================================================================

def _simulate_hierarchical_data() -> dict:
    """Create simulated UMAP-like coordinates with participant clusters.

    Returns dict with arrays: x, y, response, participant_label.
    """
    rng = np.random.default_rng(RNG_SEED)

    labels = ["R1", "R2", "R3", "NR1", "NR2", "NR3"]
    responses = ["Responder"] * 3 + ["Non-responder"] * 3

    # Place cluster centres on a rough 2D layout
    # Responders upper-left, Non-responders lower-right
    centres = np.array([
        [-3.0,  2.5],   # R1
        [-1.0,  3.5],   # R2
        [-2.5,  4.5],   # R3
        [ 2.0, -1.5],   # NR1
        [ 3.5, -0.5],   # NR2
        [ 1.5, -3.0],   # NR3
    ])

    xs, ys, resp_arr, pid_arr = [], [], [], []
    for i, (lbl, resp) in enumerate(zip(labels, responses)):
        cx, cy = centres[i]
        x = rng.normal(cx, CLUSTER_SPREAD, CELLS_PER_PARTICIPANT)
        y = rng.normal(cy, CLUSTER_SPREAD, CELLS_PER_PARTICIPANT)
        xs.append(x)
        ys.append(y)
        resp_arr.extend([resp] * CELLS_PER_PARTICIPANT)
        pid_arr.extend([lbl] * CELLS_PER_PARTICIPANT)

    return dict(
        x=np.concatenate(xs),
        y=np.concatenate(ys),
        response=np.array(resp_arr),
        participant=np.array(pid_arr),
        centres=centres,
        labels=labels,
        responses=responses,
    )


# ======================================================================
# Panel -- UMAP scatter
# ======================================================================

def _panel_umap(ax, sim: dict):
    """Scatter plot of simulated UMAP with participant clusters."""
    x, y = sim["x"], sim["y"]
    response = sim["response"]
    centres = sim["centres"]
    labels = sim["labels"]
    responses = sim["responses"]

    # Map response to colour
    color_map = {
        "Responder": COLORS["treated"],
        "Non-responder": COLORS["control"],
    }
    colors = np.array([color_map[r] for r in response])

    # Plot cells
    ax.scatter(x, y, c=colors, s=12, alpha=0.55, edgecolors="none",
               rasterized=True, zorder=2)

    # Annotate participant labels with coloured background
    for i, (lbl, resp) in enumerate(zip(labels, responses)):
        cx, cy = centres[i]
        bg_color = color_map[resp]
        ax.annotate(
            lbl,
            xy=(cx, cy),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=bg_color,
                edgecolor="white",
                linewidth=1.2,
                alpha=0.9,
            ),
        )

    # Axes
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title("Hierarchical Data Structure: Cells Clustered by Participant",
                 fontsize=12, pad=12)

    # Annotation
    ax.text(
        0.5, 0.02,
        "Each cluster = cells from one participant",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=10, fontstyle="italic",
        color=COLORS["gray"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.8),
    )

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=COLORS["treated"], label="Responder",
                       edgecolor="white"),
        mpatches.Patch(facecolor=COLORS["control"], label="Non-responder",
                       edgecolor="white"),
    ]
    ax.legend(
        handles=legend_handles, fontsize=9, loc="upper right",
        frameon=True, framealpha=0.9,
        title="Response", title_fontsize=10,
    )

    # Clean up axis ticks (UMAP axes are unitless)
    ax.set_xticks([])
    ax.set_yticks([])
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Supplementary Figure 4."""
    print("Supplementary Figure 4: UMAP Hierarchical Data Structure")

    sim = _simulate_hierarchical_data()

    # ── Composite (single panel) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _panel_umap(ax, sim)
    fig.tight_layout()
    save_figure(fig, FIGURE_NAME, SUPP_OUTPUT)

    # ── Individual panel ──────────────────────────────────────────────
    fig_p, ax_p = plt.subplots(figsize=FIGSIZE)
    _panel_umap(ax_p, sim)
    fig_p.tight_layout()
    save_panel(fig_p, "panel_umap", FIGURE_NAME, SUPP_OUTPUT)

    # ── Cleanup ───────────────────────────────────────────────────────
    clear_cache()
    gc.collect()
    print("  Done.\n")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    apply_style()
    generate()
