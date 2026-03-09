"""
Figure 12 — Temporal Dynamics of Treatment Effects
====================================================

Four-panel figure analysing how severity-associated immune programs
evolve over the disease course in the Stephenson COVID-19 cohort,
using days-from-onset (DFO) bins.

Panels
------
A : Mean signature trajectories over DFO bins, stratified by severity.
B : Severity divergence (Severe − Mild) over DFO bins for four
    representative signatures.
C : Time-specific Hedges' g effect sizes per signature × DFO bin.
D : Heatmap of severity divergence (all signatures × DFO bins).
"""

from __future__ import annotations

import gc
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage

from .._shared import (
    COLORS,
    MAIN_OUTPUT,
    add_log1p_cpm_layer,
    apply_style,
    despine,
    get_stephenson,
    save_panel,
    score_signatures,
    sig_display,
)

warnings.filterwarnings("ignore")

FIGURE_NAME = "Figure12_temporal_dynamics"

COL_SEVERE = COLORS["control"]    # orange
COL_MILD = COLORS["treated"]      # blue

DFO_BINS = ["DFO_0-7", "DFO_8-14", "DFO_15+"]
DFO_LABELS = ["0–7 d", "8–14 d", "15+ d"]


# ── data preparation ────────────────────────────────────────────────────

def _prepare_data() -> dict:
    """Load Stephenson, score signatures, compute per-participant-bin means."""
    adata = get_stephenson()

    # Ensure log1p_cpm layer
    if "log1p_cpm" not in adata.layers:
        if "counts" in adata.layers:
            adata = add_log1p_cpm_layer(
                adata, counts_layer="counts", out_layer="log1p_cpm",
            )
        else:
            raise RuntimeError("No counts layer for log1p_cpm creation.")

    adata, sig_cols = score_signatures(adata, layer="log1p_cpm")

    # Keep only cells in the three DFO bins with severity info
    mask = (
        adata.obs["dfo_bin"].isin(DFO_BINS)
        & adata.obs["severity"].isin(["Mild", "Severe"])
    )
    adata_sub = adata[mask].copy()
    print(f"  Cells with DFO bin + severity: {adata_sub.n_obs:,}")

    # Per-participant-bin pseudobulk means
    grp_cols = ["participant_id", "dfo_bin", "severity"]
    pb = (
        adata_sub.obs[grp_cols + sig_cols]
        .groupby(grp_cols, observed=True)[sig_cols]
        .mean()
        .reset_index()
    )

    return {
        "adata": adata,
        "sig_cols": sig_cols,
        "pb": pb,
    }


def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g  (x − y)  with small-sample correction."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    if pooled == 0:
        return np.nan
    g = (np.mean(x) - np.mean(y)) / pooled
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return g * correction


# ── Panel A: Mean trajectories by severity ──────────────────────────────

def _panel_a(ax, data: dict) -> None:
    """Line plot of mean signature score over DFO bins for Mild vs Severe."""
    pb = data["pb"]
    sig_cols = data["sig_cols"]

    # Pick 4 representative signatures
    targets = []
    for keyword in ["Cytotoxic", "Interferon", "Memory", "Antigen"]:
        for col in sig_cols:
            if keyword in col:
                targets.append(col)
                break
    if len(targets) < 4:
        targets = sig_cols[:4]

    markers = ["o", "s", "^", "D"]
    for idx, col in enumerate(targets):
        for sev, color, ls in [("Severe", COL_SEVERE, "-"), ("Mild", COL_MILD, "--")]:
            means = []
            for b in DFO_BINS:
                vals = pb.loc[
                    (pb["severity"] == sev) & (pb["dfo_bin"] == b), col
                ]
                means.append(vals.mean() if len(vals) > 0 else np.nan)
            ax.plot(
                range(len(DFO_BINS)), means,
                color=color, ls=ls, lw=1.8, marker=markers[idx],
                markersize=6, markeredgecolor="white", markeredgewidth=0.5,
                alpha=0.85, label=None,
            )
            # Only put text at last point for readability
            if not np.isnan(means[-1]):
                ha = "left"
                ax.annotate(
                    sig_display(col) if sev == "Severe" else "",
                    (len(DFO_BINS) - 1, means[-1]),
                    fontsize=6, ha=ha, va="center",
                    xytext=(6, 0), textcoords="offset points",
                    color=color,
                )

    ax.set_xticks(range(len(DFO_BINS)))
    ax.set_xticklabels(DFO_LABELS)
    ax.set_xlabel("Days from Onset")
    ax.set_ylabel("Mean Signature Score")
    ax.set_title("Signature Trajectories by Severity", fontsize=10)

    handles = [
        Line2D([0], [0], color=COL_SEVERE, lw=2, ls="-", label="Severe"),
        Line2D([0], [0], color=COL_MILD, lw=2, ls="--", label="Mild"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel B: Severity divergence for 4 representative signatures ────────

def _panel_b(ax, data: dict) -> None:
    """Line plot of (Severe − Mild) mean per DFO bin for 4 signatures."""
    pb = data["pb"]
    sig_cols = data["sig_cols"]

    targets = []
    for keyword in ["Cytotoxic", "Interferon", "Memory", "Antigen"]:
        for col in sig_cols:
            if keyword in col:
                targets.append(col)
                break
    if len(targets) < 4:
        targets = sig_cols[:4]

    palette = [COLORS["highlight"], COLORS["neutral"], COLORS["success"], COLORS["treated"]]
    markers = ["o", "s", "^", "D"]

    for idx, col in enumerate(targets):
        divs = []
        for b in DFO_BINS:
            sev_vals = pb.loc[
                (pb["severity"] == "Severe") & (pb["dfo_bin"] == b), col
            ]
            mild_vals = pb.loc[
                (pb["severity"] == "Mild") & (pb["dfo_bin"] == b), col
            ]
            div = sev_vals.mean() - mild_vals.mean()
            divs.append(div)

        ax.plot(
            range(len(DFO_BINS)), divs,
            color=palette[idx], lw=2, marker=markers[idx],
            markersize=7, markeredgecolor="white", markeredgewidth=0.5,
            label=sig_display(col),
        )

    ax.axhline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_xticks(range(len(DFO_BINS)))
    ax.set_xticklabels(DFO_LABELS)
    ax.set_xlabel("Days from Onset")
    ax.set_ylabel("Divergence (Severe − Mild)")
    ax.set_title("Severity Divergence Over Time", fontsize=10)
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel C: Time-specific Hedges' g ────────────────────────────────────

def _panel_c(ax, data: dict) -> None:
    """Grouped bar chart of Hedges' g per signature per DFO bin."""
    pb = data["pb"]
    sig_cols = data["sig_cols"]

    records = []
    for col in sig_cols:
        for b in DFO_BINS:
            sev = pb.loc[
                (pb["severity"] == "Severe") & (pb["dfo_bin"] == b), col
            ].dropna().values
            mild = pb.loc[
                (pb["severity"] == "Mild") & (pb["dfo_bin"] == b), col
            ].dropna().values
            g = _hedges_g(sev, mild)
            records.append({
                "feature": col, "display": sig_display(col),
                "dfo_bin": b, "g": g,
            })

    df = pd.DataFrame(records)
    pivot = df.pivot(index="display", columns="dfo_bin", values="g")
    pivot = pivot.reindex(columns=DFO_BINS)

    # Sort by absolute mean effect
    pivot["abs_mean"] = pivot.abs().mean(axis=1)
    pivot = pivot.sort_values("abs_mean", ascending=True).drop(columns="abs_mean")

    y_pos = np.arange(len(pivot))
    n_bins = len(DFO_BINS)
    bar_h = 0.8 / n_bins
    bin_colors = [COLORS["treated"], COLORS["neutral"], COLORS["highlight"]]

    for i, (b, color) in enumerate(zip(DFO_BINS, bin_colors)):
        offset = (i - (n_bins - 1) / 2) * bar_h
        vals = pivot[b].values
        ax.barh(
            y_pos + offset, vals,
            height=bar_h, color=color, alpha=0.8,
            label=DFO_LABELS[i], edgecolor="none",
        )

    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Hedges' g (Severe − Mild)")
    ax.set_title("Time-Specific Effect Sizes", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel D: Divergence heatmap ─────────────────────────────────────────

def _panel_d(ax, data: dict) -> None:
    """Heatmap of severity divergence (Severe − Mild) × DFO bin."""
    pb = data["pb"]
    sig_cols = data["sig_cols"]

    records = []
    for col in sig_cols:
        for b in DFO_BINS:
            sev_mean = pb.loc[
                (pb["severity"] == "Severe") & (pb["dfo_bin"] == b), col
            ].mean()
            mild_mean = pb.loc[
                (pb["severity"] == "Mild") & (pb["dfo_bin"] == b), col
            ].mean()
            records.append({
                "display": sig_display(col),
                "dfo_bin": b,
                "divergence": sev_mean - mild_mean,
            })

    df = pd.DataFrame(records)
    pivot = df.pivot(index="display", columns="dfo_bin", values="divergence")
    pivot = pivot.reindex(columns=DFO_BINS)
    pivot.columns = DFO_LABELS

    # Cluster rows by similarity
    vals = pivot.fillna(0).values
    if vals.shape[0] > 2:
        link = linkage(vals, method="ward")
        order = leaves_list(link)
        pivot = pivot.iloc[order]

    vmax = np.nanmax(np.abs(pivot.values)) * 0.9
    sns.heatmap(
        pivot, ax=ax, cmap="RdBu_r", center=0,
        vmin=-vmax, vmax=vmax,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Divergence (Severe − Mild)", "shrink": 0.8},
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
    )
    ax.set_xlabel("Days from Onset")
    ax.set_ylabel("")
    ax.set_title("Temporal Divergence Heatmap", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)


# ── Composite generation ────────────────────────────────────────────────

def generate() -> None:
    """Create and save all Figure 12 panels."""
    apply_style()
    print("Figure 12: Temporal Dynamics of Treatment Effects")
    data = _prepare_data()

    panel_funcs = [
        ("panel_A_trajectories_by_severity", _panel_a),
        ("panel_B_severity_divergence", _panel_b),
        ("panel_C_time_specific_effect_sizes", _panel_c),
        ("panel_D_divergence_heatmap", _panel_d),
    ]
    for panel_name, func in panel_funcs:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        func(ax, data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    adata = data.get("adata")
    if adata is not None:
        del adata
    del data
    gc.collect()

    print(f"  Figure 12 complete: {FIGURE_NAME}")
