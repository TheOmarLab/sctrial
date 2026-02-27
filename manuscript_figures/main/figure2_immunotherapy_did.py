"""
Figure 2 — Immunotherapy DiD Analysis.

Three-panel figure (top: A wide; bottom: B + C) showing the primary
Difference-in-Differences analysis on the Sade-Feldman melanoma
immunotherapy dataset:

    A  Forest plot of DiD effects across all 12 gene signatures
    B  Small-multiple interaction plots for the top 6 signatures
    C  Per-participant change heatmap across signatures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from .._shared import (
    COLORS,
    MAIN_OUTPUT,
    TrialDesign,
    apply_style,
    despine,
    did_table,
    get_sade_feldman,
    harmonize_response,
    save_panel,
    score_signatures,
    sig_display,
)

# ── Colour palette ───────────────────────────────────────────────────────

COL_RESP = COLORS["treated"]       # #4C72B0 — blue  (Responder)
COL_NRESP = COLORS["control"]      # #E1812C — orange (Non-responder)
COL_GRAY = COLORS["gray"]          # #8C8C8C

# ── Trial design ─────────────────────────────────────────────────────────

DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",          # ← harmonized labels (fixes P0)
    arm_treated="Responder",
    arm_control="Non-responder",
)
VISITS: tuple[str, str] = ("Pre", "Post")
FIGURE_NAME = "Figure2_immunotherapy_did"


# ── Data preparation ─────────────────────────────────────────────────────

def _prepare_data() -> tuple:
    """Load Sade-Feldman data, score signatures, and run DiD."""
    adata = get_sade_feldman()
    if "log1p_tpm" not in adata.layers:
        raise RuntimeError("log1p_tpm layer missing from Sade-Feldman dataset.")
    adata = harmonize_response(adata)
    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

    did_res = did_table(
        adata,
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer="log1p_tpm",
        standardize=True,
        aggregate="participant_visit",
        use_bootstrap=True,
        n_boot=999,
        seed=42,
    )
    return adata, sig_cols, did_res


def _pseudobulk(adata, sig_col: str) -> pd.DataFrame:
    """Per-participant-visit pseudobulk means for *sig_col*."""
    df = adata.obs[[
        DESIGN.participant_col, DESIGN.visit_col,
        DESIGN.arm_col, sig_col,
    ]].copy()
    return (
        df.groupby(
            [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col],
            observed=True,
        )[sig_col]
        .mean()
        .reset_index()
    )


def _pseudobulk_all(adata, sig_cols: list[str]) -> pd.DataFrame:
    """Per-participant-visit pseudobulk means for ALL signature columns."""
    cols = [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col] + sig_cols
    df = adata.obs[cols].copy()
    return (
        df.groupby(
            [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col],
            observed=True,
        )[sig_cols]
        .mean()
        .reset_index()
    )


# ── Panel A: Forest plot ─────────────────────────────────────────────────

def panel_forest(ax: plt.Axes, did_res: pd.DataFrame) -> None:
    """Horizontal forest plot."""
    df = did_res.sort_values("beta_DiD").reset_index(drop=True)
    y_pos = np.arange(len(df))

    # Use bootstrap-t CIs when available (from use_bootstrap=True),
    # otherwise fall back to analytical ±1.96·SE.
    # Per-row fallback: if bootstrap CI is NaN for a given feature
    # (e.g. failed bootstrap draw), use analytical CI for that row.
    analytical_lo = df["beta_DiD"] - 1.96 * df["se_DiD"]
    analytical_hi = df["beta_DiD"] + 1.96 * df["se_DiD"]
    if "ci_lo_boot" in df.columns and "ci_hi_boot" in df.columns:
        ci_lo = df["ci_lo_boot"].fillna(analytical_lo)
        ci_hi = df["ci_hi_boot"].fillna(analytical_hi)
    else:
        ci_lo = analytical_lo
        ci_hi = analytical_hi

    for i, (_, row) in enumerate(df.iterrows()):
        color = COL_RESP if row["beta_DiD"] > 0 else COL_NRESP
        ax.hlines(
            y_pos[i], ci_lo.iloc[i], ci_hi.iloc[i],
            color=color, linewidth=2.0, alpha=1.0, zorder=1,
        )
        ax.scatter(
            row["beta_DiD"], y_pos[i], color=color, s=55,
            edgecolors="white", linewidths=0.8, alpha=1.0, zorder=2,
        )

    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", zorder=0,
               alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sig_display(f) for f in df["feature"]], fontsize=10)
    ax.set_xlabel(r"DiD coefficient ($\beta$, standardised)", fontsize=11)
    ax.set_title("DiD effects across signatures", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(-0.6, len(df) - 0.4)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_RESP, markersize=8,
               label=r"Responder $\uparrow$"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_NRESP, markersize=8,
               label=r"Non-responder $\uparrow$"),
    ]
    ax.legend(
        handles=legend_handles, fontsize=9, loc="lower right",
        frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
        handletextpad=0.4, borderpad=0.5,
    )
    despine(ax)


# ── Panel B: Small-multiple interaction plots ────────────────────────────

def panel_interaction_grid(
    fig: plt.Figure,
    gs_parent: gridspec.SubplotSpec,
    adata,
    did_res: pd.DataFrame,
    n_sigs: int = 6,
) -> list[plt.Axes]:
    """2×3 grid of interaction plots for the top *n_sigs* signatures (by p)."""
    # Prefer bootstrap p-values for ranking but fall back to analytical
    # per-row when bootstrap p is NaN (e.g. failed bootstrap draw).
    if "p_DiD_boot" in did_res.columns:
        rank_p = did_res["p_DiD_boot"].fillna(did_res["p_DiD"])
    else:
        rank_p = did_res["p_DiD"]
    top = did_res.assign(_rank_p=rank_p).sort_values("_rank_p").head(n_sigs).drop(columns="_rank_p")

    nrows, ncols = 2, 3
    gs_inner = gs_parent.subgridspec(nrows, ncols, hspace=0.55, wspace=0.35)
    axes = []

    arm_colors = {
        DESIGN.arm_treated: COL_RESP,
        DESIGN.arm_control: COL_NRESP,
    }
    x_map = {VISITS[0]: 0.0, VISITS[1]: 1.0}

    for idx, (_, row) in enumerate(top.iterrows()):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs_inner[r, c])
        axes.append(ax)

        sig_col = row["feature"]
        pb = _pseudobulk(adata, sig_col)

        # Individual participant traces
        for arm, arm_df in pb.groupby(DESIGN.arm_col, observed=True):
            color = arm_colors.get(arm, COL_GRAY)
            for _, pid_df in arm_df.groupby(DESIGN.participant_col,
                                            observed=True):
                pid_df = pid_df.sort_values(
                    DESIGN.visit_col, key=lambda s: s.map(x_map),
                )
                if len(pid_df) == 2:
                    ax.plot(
                        pid_df[DESIGN.visit_col].map(x_map),
                        pid_df[sig_col],
                        color=color, alpha=0.22, linewidth=0.8, zorder=1,
                    )

        # Group means — bold lines
        group_means = (
            pb.groupby([DESIGN.arm_col, DESIGN.visit_col], observed=True)
            [sig_col].mean().reset_index()
        )
        for arm, gdf in group_means.groupby(DESIGN.arm_col, observed=True):
            color = arm_colors.get(arm, COL_GRAY)
            gdf = gdf.sort_values(DESIGN.visit_col,
                                  key=lambda s: s.map(x_map))
            ax.plot(
                gdf[DESIGN.visit_col].map(x_map), gdf[sig_col],
                color=color, linewidth=2.8, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3,
            )

        # Formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels(VISITS, fontsize=9)
        ax.set_xlim(-0.35, 1.35)
        ax.tick_params(axis="y", labelsize=8)

        # Title with bootstrap p-value when available; fall back to
        # analytical if bootstrap p is NaN for this feature.
        p_val = row.get("p_DiD_boot", np.nan)
        if pd.isna(p_val):
            p_val = row["p_DiD"]
        p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else f"p = {p_val:.1e}"
        ax.set_title(
            f"{sig_display(sig_col)}\n{p_str}",
            fontsize=9.5, fontweight="bold", pad=6,
        )
        despine(ax)

    # Shared legend on the last axis
    legend_handles = [
        Line2D([0], [0], color=COL_RESP, linewidth=2.5, marker="o",
               markersize=6, markeredgecolor="white", label="Responder"),
        Line2D([0], [0], color=COL_NRESP, linewidth=2.5, marker="o",
               markersize=6, markeredgecolor="white", label="Non-responder"),
        Line2D([0], [0], color=COL_GRAY, linewidth=0.8, alpha=0.4,
               label="Individual"),
    ]
    axes[-1].legend(
        handles=legend_handles, fontsize=7.5, loc="best",
        frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
    )

    return axes


# ── Panel C: Per-participant change heatmap ──────────────────────────────

def panel_heatmap(
    ax: plt.Axes,
    adata,
    sig_cols: list[str],
    did_res: pd.DataFrame,
) -> None:
    """Heatmap of per-participant pre→post Δ across all signatures.

    Note: the colour scale shows raw score change (Post − Pre) per
    participant, NOT the DiD regression coefficient shown in Panel A.
    """
    pb = _pseudobulk_all(adata, sig_cols)

    # Build pre/post DataFrames keyed by participant (mean across cells)
    pre_mask = pb[DESIGN.visit_col] == VISITS[0]
    post_mask = pb[DESIGN.visit_col] == VISITS[1]

    # Separate numeric and categorical columns for aggregation
    num_cols = sig_cols
    pre_num = (
        pb.loc[pre_mask]
        .groupby(DESIGN.participant_col, observed=True)[num_cols]
        .mean()
    )
    post_num = (
        pb.loc[post_mask]
        .groupby(DESIGN.participant_col, observed=True)[num_cols]
        .mean()
    )
    # Get arm labels (take mode per participant — consistent after harmonization)
    pre_arm = (
        pb.loc[pre_mask]
        .groupby(DESIGN.participant_col, observed=True)[DESIGN.arm_col]
        .first()
    )
    pre = pre_num.join(pre_arm)
    post = post_num
    common_pids = sorted(set(pre.index) & set(post.index))

    if len(common_pids) == 0:
        ax.text(0.5, 0.5, "No paired participants", ha="center",
                va="center", transform=ax.transAxes, fontsize=10)
        ax.axis("off")
        return

    delta = pd.DataFrame(
        post.loc[common_pids, sig_cols].values
        - pre.loc[common_pids, sig_cols].values,
        index=common_pids,
        columns=[sig_display(c) for c in sig_cols],
    )
    arms = pre.loc[common_pids, DESIGN.arm_col]

    # Sort: Responders first, then by mean Δ
    mean_delta = delta.mean(axis=1)
    sort_df = pd.DataFrame({
        "arm_order": arms.map({"Responder": 0, "Non-responder": 1}).values,
        "mean_delta": -mean_delta.values,
    }, index=common_pids)
    ordered_pids = sort_df.sort_values(["arm_order", "mean_delta"]).index.tolist()
    delta = delta.loc[ordered_pids]
    arms = arms.loc[ordered_pids]

    # Sort columns by DiD effect (matches forest plot)
    col_order = [sig_display(f) for f in
                 did_res.sort_values("beta_DiD")["feature"]]
    col_order = [c for c in col_order if c in delta.columns]
    delta = delta[col_order]

    # Diverging colourmap centred at 0
    vmax = np.nanpercentile(np.abs(delta.values), 95)
    vmax = max(vmax, 0.1)

    # Draw heatmap
    im = ax.imshow(
        delta.values, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )

    # Row annotations — arm colour sidebar
    sidebar_w = 0.4
    for i, pid in enumerate(ordered_pids):
        arm = arms.loc[pid]
        color = COL_RESP if arm == "Responder" else COL_NRESP
        ax.add_patch(plt.Rectangle(
            (-sidebar_w - 0.55, i - 0.5), sidebar_w, 1.0,
            color=color, clip_on=False,
        ))

    # Horizontal separator between responder / non-responder groups
    n_resp = sum(1 for p in ordered_pids if arms.loc[p] == "Responder")
    if 0 < n_resp < len(ordered_pids):
        ax.axhline(n_resp - 0.5, color="white", linewidth=2.5, zorder=5)

    # ── Axis labels ──
    ax.set_xticks(np.arange(len(delta.columns)))
    ax.set_xticklabels(delta.columns, fontsize=8.5, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(delta.index)))

    # Colour-code y-tick labels by arm for a polished look
    ylabels = []
    for pid in ordered_pids:
        ylabels.append(str(pid))
    ax.set_yticklabels(ylabels, fontsize=8.5,
                       fontfamily="monospace", fontweight="medium")
    for i, tick in enumerate(ax.get_yticklabels()):
        arm = arms.iloc[i]
        tick.set_color(COL_RESP if arm == "Responder" else COL_NRESP)
    ax.set_title("Per-participant score change (Post − Pre)", fontsize=12,
                 fontweight="bold")

    # Colour bar — label clearly as raw score change, NOT DiD β
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score change", fontsize=9)
    cbar.ax.tick_params(labelsize=7.5)

    # Sidebar legend — centred below the heatmap
    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_RESP,
               markersize=9, markeredgewidth=0, label="Responder"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_NRESP,
               markersize=9, markeredgewidth=0, label="Non-responder"),
    ]
    ax.legend(
        handles=legend_handles, fontsize=10,
        loc="lower center", bbox_to_anchor=(0.45, -0.22),
        ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.5,
    )


# ── Composite figure ─────────────────────────────────────────────────────

def generate() -> None:
    """Create and save Figure 2 individual panels."""
    print("Figure 2: Immunotherapy DiD Analysis")
    adata, sig_cols, did_res = _prepare_data()

    # ── Individual panels ────────────────────────────────────────────────
    pfig_a, pax_a = plt.subplots(figsize=(12, 5))
    panel_forest(pax_a, did_res)
    save_panel(pfig_a, "A_forest", FIGURE_NAME, MAIN_OUTPUT)

    pfig_b = plt.figure(figsize=(14, 8))
    gs_bs = pfig_b.add_gridspec(1, 1)[0, 0]
    panel_interaction_grid(pfig_b, gs_bs, adata, did_res, n_sigs=6)
    save_panel(pfig_b, "B_interaction_grid", FIGURE_NAME, MAIN_OUTPUT)

    pfig_c, pax_c = plt.subplots(figsize=(11, 7))
    panel_heatmap(pax_c, adata, sig_cols, did_res)
    save_panel(pfig_c, "C_heatmap", FIGURE_NAME, MAIN_OUTPUT)

    print("  Done.\n")


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    generate()
