"""
Supplementary Figure 9 — Temporal Dynamics
===========================================

Three-panel figure (GridSpec: 2x2, bottom row spans full width)
showing temporal dynamics of immune signatures across COVID-19
severity groups using the Stephenson dataset.

Panels
------
A  Temporal trajectories of 4 key signatures by severity.
B  Severity divergence (Severe - Mild) for top 4 divergent signatures.
C  Heatmap of divergence across all signatures x time bins.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from .._shared import *  # noqa: F401,F403

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "SuppFig9_temporal_dynamics"
FIGSIZE = (18, 12)

# Key signatures for Panel A
KEY_SIGNATURES = [
    "Interferon Response",
    "Immune Exhaustion",
    "Cytotoxic T Cell Activity",
    "Inflammatory Response",
]

KEY_DISPLAY = {
    "Interferon Response": "IFN Response",
    "Immune Exhaustion": "T Cell Exhaustion",
    "Cytotoxic T Cell Activity": "Cytotoxic T Cells",
    "Inflammatory Response": "Inflammation",
}


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_data() -> dict:
    """Load Stephenson, score signatures, compute per-bin severity means."""
    adata = get_stephenson()

    # Ensure log1p_cpm layer
    add_log1p_cpm_layer(adata)

    adata, sig_cols = score_signatures(adata)
    print(f"  Scored {len(sig_cols)} signatures")

    obs = adata.obs.copy()

    # Ensure severity and dfo_bin columns exist
    if "severity" not in obs.columns:
        for candidate in ("Status_on_day_collection_summary", "status", "Status"):
            if candidate in obs.columns:
                obs["severity"] = obs[candidate].astype(str)
                break

    if "dfo_bin" not in obs.columns:
        if "dfo" not in obs.columns:
            for candidate in ("Days_from_onset", "days_from_onset"):
                if candidate in obs.columns:
                    obs["dfo"] = pd.to_numeric(obs[candidate], errors="coerce")
                    break
        if "dfo" in obs.columns:
            obs["dfo_bin"] = pd.cut(
                obs["dfo"],
                bins=[-np.inf, 7, 14, np.inf],
                labels=["DFO_0-7", "DFO_8-14", "DFO_15+"],
            ).astype(str)

    if "severity" not in obs.columns or "dfo_bin" not in obs.columns:
        raise RuntimeError("Cannot find severity/dfo_bin columns in Stephenson data")

    # Filter to Mild and Severe
    obs = obs[obs["severity"].isin(["Mild", "Severe"])].copy()
    valid_bins = ["DFO_0-7", "DFO_8-14", "DFO_15+"]
    obs = obs[obs["dfo_bin"].isin(valid_bins)].copy()

    # Detect participant column
    pid_col = None
    for candidate in ("participant_id", "patient_id", "donor_id", "sample_id"):
        if candidate in obs.columns:
            pid_col = candidate
            break
    if pid_col is None:
        pid_col = obs.columns[0]  # fallback

    # Compute participant-level mean scores per time bin x severity
    available_sig_cols = [c for c in sig_cols if c in obs.columns]
    group_cols = [pid_col, "severity", "dfo_bin"]
    participant_means = (
        obs.groupby(group_cols, observed=True)[available_sig_cols]
        .mean()
        .reset_index()
    )

    # Compute group-level means (severity x time bin)
    group_means = (
        participant_means.groupby(["severity", "dfo_bin"], observed=True)[available_sig_cols]
        .agg(["mean", "sem"])
    )

    # Pivot for easier access
    # Structure: group_means.loc[("Severe", "DFO_0-7"), ("sig_X", "mean")]

    # Sort time bins
    sorted_bins = sorted(valid_bins, key=dfo_sort_key)

    # Compute divergence matrix (Severe - Mild) for each time bin x signature
    divergence = {}
    for tbin in sorted_bins:
        for sig in available_sig_cols:
            try:
                sev_val = group_means.loc[("Severe", tbin), (sig, "mean")]
                mild_val = group_means.loc[("Mild", tbin), (sig, "mean")]
                divergence[(sig, tbin)] = sev_val - mild_val
            except KeyError:
                divergence[(sig, tbin)] = np.nan

    div_df = pd.DataFrame(
        {tbin: {sig: divergence.get((sig, tbin), np.nan) for sig in available_sig_cols}
         for tbin in sorted_bins}
    )
    div_df.index = [sig_display(s) for s in div_df.index]

    print(f"  Divergence matrix: {div_df.shape[0]} signatures x {div_df.shape[1]} time bins")

    return dict(
        adata=adata,
        sig_cols=available_sig_cols,
        group_means=group_means,
        participant_means=participant_means,
        sorted_bins=sorted_bins,
        divergence_df=div_df,
    )


# ======================================================================
# Panel A — Temporal trajectories
# ======================================================================

def panel_A(ax, data: dict):
    """Line plot of 4 key signatures over time by severity."""
    group_means = data["group_means"]
    sorted_bins = data["sorted_bins"]
    sig_cols = data["sig_cols"]

    # Map key signatures to sig_ columns
    key_sig_cols = []
    for name in KEY_SIGNATURES:
        col = f"sig_{name}"
        if col in sig_cols:
            key_sig_cols.append(col)

    if len(key_sig_cols) == 0:
        # Fallback: use first 4 available
        key_sig_cols = sig_cols[:4]

    linestyles = ["-", "--", "-.", ":"]
    severity_colors = {
        "Severe": COLORS["treated"],
        "Mild": COLORS["control"],
    }
    x = np.arange(len(sorted_bins))

    for i, sig in enumerate(key_sig_cols):
        display = sig_display(sig)
        ls = linestyles[i % len(linestyles)]

        for sev, color in severity_colors.items():
            vals = []
            for tbin in sorted_bins:
                try:
                    vals.append(group_means.loc[(sev, tbin), (sig, "mean")])
                except KeyError:
                    vals.append(np.nan)

            label = f"{display} ({sev})" if sev == "Severe" else None
            ax.plot(x, vals, color=color, linestyle=ls, linewidth=1.8,
                    marker="o", markersize=5, markeredgecolor="white",
                    markeredgewidth=0.5, alpha=0.85,
                    label=label if sev == "Severe" else None)

            # Only label Mild on first signature for legend clarity
            if sev == "Mild" and i == 0:
                pass  # handled in custom legend

    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("DFO_", "DFO ") for b in sorted_bins],
                       fontsize=9)
    ax.set_xlabel("Days from onset")
    ax.set_ylabel("Mean signature score")
    ax.set_title("Temporal Trajectories by Severity", fontsize=11,
                 fontweight="bold")

    # Custom legend
    legend_handles = []
    # Severity colors
    legend_handles.append(Line2D([0], [0], color=COLORS["treated"],
                                 linewidth=2, label="Severe"))
    legend_handles.append(Line2D([0], [0], color=COLORS["control"],
                                 linewidth=2, label="Mild"))
    # Linestyles for signatures
    for i, sig in enumerate(key_sig_cols):
        display = sig_display(sig)
        ls = linestyles[i % len(linestyles)]
        legend_handles.append(
            Line2D([0], [0], color="gray", linestyle=ls, linewidth=1.5,
                   label=display)
        )

    ax.legend(handles=legend_handles, fontsize=7.5, loc="best",
              frameon=True, framealpha=0.9, ncol=2)
    despine(ax)


# ======================================================================
# Panel B — Severity divergence over time
# ======================================================================

def panel_B(ax, data: dict):
    """Line plot of (Severe - Mild) divergence for top 4 divergent signatures."""
    div_df = data["divergence_df"]
    sorted_bins = data["sorted_bins"]

    if div_df is None or len(div_df) == 0:
        ax.text(0.5, 0.5, "Divergence data unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=COLORS["gray"])
        ax.axis("off")
        return

    # Select top 4 most divergent signatures (by max absolute divergence)
    max_div = div_df.abs().max(axis=1).sort_values(ascending=False)
    top_sigs = max_div.head(4).index.tolist()

    x = np.arange(len(sorted_bins))
    linestyles = ["-", "--", "-.", ":"]
    palette = [COLORS["treated"], COLORS["control"],
               COLORS["highlight"], COLORS["neutral"]]

    for i, sig in enumerate(top_sigs):
        vals = div_df.loc[sig, sorted_bins].values.astype(float)
        ax.plot(x, vals, color=palette[i % len(palette)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2.0, marker="s", markersize=5,
                markeredgecolor="white", markeredgewidth=0.5,
                label=sig, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0,
               alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("DFO_", "DFO ") for b in sorted_bins],
                       fontsize=9)
    ax.set_xlabel("Days from onset")
    ax.set_ylabel("Divergence (Severe - Mild)")
    ax.set_title("Severity Divergence Over Time", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel C — Temporal divergence heatmap
# ======================================================================

def panel_C(ax, data: dict):
    """Heatmap of (Severe - Mild) divergence across signatures x time bins."""
    div_df = data["divergence_df"]
    sorted_bins = data["sorted_bins"]

    if div_df is None or len(div_df) == 0:
        ax.text(0.5, 0.5, "Divergence data unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=COLORS["gray"])
        ax.axis("off")
        return

    # Ensure correct column order
    plot_df = div_df[sorted_bins].copy()

    # Clean column labels
    plot_df.columns = [b.replace("DFO_", "DFO ") for b in plot_df.columns]

    # Sort signatures by mean divergence
    plot_df = plot_df.loc[plot_df.mean(axis=1).sort_values().index]

    # Symmetric colormap centered at 0
    vmax = np.nanmax(np.abs(plot_df.values))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    import seaborn as sns
    hm = sns.heatmap(
        plot_df,
        ax=ax,
        cmap="RdBu_r",
        norm=norm,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Divergence (Severe - Mild)", "shrink": 0.8},
    )

    ax.set_xlabel("Days from onset", fontsize=10)
    ax.set_ylabel("")
    ax.set_title("Temporal Divergence Heatmap (Severe - Mild)",
                 fontsize=11, fontweight="bold")

    # Rotate y-axis labels for readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Supplementary Figure 9 individual panels."""
    print("Supplementary Figure 9: Temporal Dynamics")
    apply_style()

    try:
        data = _prepare_data()
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return

    # ── Save individual panels ────────────────────────────────────────
    for panel_label, panel_func, psize in [
        ("A_temporal_trajectories", panel_A, (9, 6)),
        ("B_severity_divergence", panel_B, (9, 6)),
        ("C_divergence_heatmap", panel_C, (16, 6)),
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
