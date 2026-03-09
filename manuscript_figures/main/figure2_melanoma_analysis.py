"""
Figure 2 — Melanoma Primary Analysis & Clinical Outcome
=========================================================

Seven-panel figure combining the primary DiD analysis on the
Sade-Feldman melanoma immunotherapy dataset with clinical outcome
correlation.

Panels
------
A : Forest plot of DiD effects across all 12 gene signatures.
B : Small-multiple interaction plots for the top 6 signatures.
C : Per-participant change heatmap across signatures.
D : Bar plot of mean Δ score (post − pre) by response group.
E : Cohen's d effect sizes (responder − non-responder) on Δ scores.
F : Individual participant trajectories for memory T cell signature.
G : ROC AUC for response prediction from each signature's Δ score.
"""

from __future__ import annotations

import gc
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

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

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────

FIGURE_NAME = "Figure2_melanoma_analysis"
VISITS: tuple[str, str] = ("Pre", "Post")

COL_RESP = COLORS["treated"]     # blue  (Responder)
COL_NRESP = COLORS["control"]    # orange (Non-responder)
COL_GRAY = COLORS["gray"]

DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)


# ── Data preparation ─────────────────────────────────────────────────────

def _prepare_data() -> dict:
    """Load Sade-Feldman, score signatures, run DiD and compute Δ scores."""
    adata = get_sade_feldman()
    if "log1p_tpm" not in adata.layers:
        raise RuntimeError("log1p_tpm layer missing from Sade-Feldman dataset.")
    adata = harmonize_response(adata)
    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

    # DiD with bootstrap CIs
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

    # Pseudobulk: per-participant-visit means
    grp_cols = [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col]
    pb = (
        adata.obs[grp_cols + sig_cols]
        .groupby(grp_cols, observed=True)[sig_cols]
        .mean()
        .reset_index()
    )

    # Keep only paired participants
    visit_counts = pb.groupby(DESIGN.participant_col)[DESIGN.visit_col].nunique()
    paired_pids = visit_counts[visit_counts == 2].index
    pb = pb[pb[DESIGN.participant_col].isin(paired_pids)].copy()

    # Compute Δ = Post − Pre for each participant
    pre = pb[pb[DESIGN.visit_col] == "Pre"].set_index(DESIGN.participant_col)
    post = pb[pb[DESIGN.visit_col] == "Post"].set_index(DESIGN.participant_col)
    delta = post[sig_cols].subtract(pre[sig_cols])
    delta[DESIGN.arm_col] = pre[DESIGN.arm_col]
    delta = delta.reset_index()

    n_r = (delta[DESIGN.arm_col] == "Responder").sum()
    n_nr = (delta[DESIGN.arm_col] == "Non-responder").sum()
    print(f"  Paired participants: {len(delta)} (R={n_r}, NR={n_nr})")

    return {
        "adata": adata,
        "sig_cols": sig_cols,
        "did_res": did_res,
        "pb": pb,
        "delta": delta,
    }


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


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges-corrected Cohen's d  (x − y)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_sd = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    if pooled_sd == 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / pooled_sd
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return d * correction


# ── Panel A: Forest plot ─────────────────────────────────────────────────

def _panel_a(ax: plt.Axes, data: dict) -> None:
    """Horizontal forest plot of DiD effects across all signatures."""
    did_res = data["did_res"]
    df = did_res.sort_values("beta_DiD").reset_index(drop=True)
    y_pos = np.arange(len(df))

    # Use bootstrap-t CIs when available, fall back to analytical
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

def _panel_b(
    fig: plt.Figure,
    gs_parent: gridspec.SubplotSpec,
    data: dict,
    n_sigs: int = 6,
) -> list[plt.Axes]:
    """2×3 grid of interaction plots for the top *n_sigs* signatures."""
    adata = data["adata"]
    did_res = data["did_res"]

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

        # Group means
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

        ax.set_xticks([0, 1])
        ax.set_xticklabels(VISITS, fontsize=9)
        ax.set_xlim(-0.35, 1.35)
        ax.tick_params(axis="y", labelsize=8)

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

def _panel_c(ax: plt.Axes, data: dict) -> None:
    """Heatmap of per-participant pre→post Δ across all signatures."""
    adata = data["adata"]
    sig_cols = data["sig_cols"]
    did_res = data["did_res"]

    pb = _pseudobulk_all(adata, sig_cols)

    pre_mask = pb[DESIGN.visit_col] == VISITS[0]
    post_mask = pb[DESIGN.visit_col] == VISITS[1]

    pre_num = (
        pb.loc[pre_mask]
        .groupby(DESIGN.participant_col, observed=True)[sig_cols]
        .mean()
    )
    post_num = (
        pb.loc[post_mask]
        .groupby(DESIGN.participant_col, observed=True)[sig_cols]
        .mean()
    )
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

    # Sort columns by DiD effect
    col_order = [sig_display(f) for f in
                 did_res.sort_values("beta_DiD")["feature"]]
    col_order = [c for c in col_order if c in delta.columns]
    delta = delta[col_order]

    vmax = np.nanpercentile(np.abs(delta.values), 95)
    vmax = max(vmax, 0.1)

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

    # Separator between groups
    n_resp = sum(1 for p in ordered_pids if arms.loc[p] == "Responder")
    if 0 < n_resp < len(ordered_pids):
        ax.axhline(n_resp - 0.5, color="white", linewidth=2.5, zorder=5)

    ax.set_xticks(np.arange(len(delta.columns)))
    ax.set_xticklabels(delta.columns, fontsize=8.5, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(delta.index)))
    ylabels = [str(pid) for pid in ordered_pids]
    ax.set_yticklabels(ylabels, fontsize=8.5,
                       fontfamily="monospace", fontweight="medium")
    for i, tick in enumerate(ax.get_yticklabels()):
        arm = arms.iloc[i]
        tick.set_color(COL_RESP if arm == "Responder" else COL_NRESP)
    ax.set_title("Per-participant score change (Post − Pre)", fontsize=12,
                 fontweight="bold")

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score change", fontsize=9)
    cbar.ax.tick_params(labelsize=7.5)

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


# ── Panel D: Mean Δ score by response group ─────────────────────────────

def _panel_d(ax: plt.Axes, data: dict) -> None:
    """Grouped bar plot of mean Δ scores for responders vs non-responders."""
    delta = data["delta"]
    sig_cols = data["sig_cols"]

    resp_mask = delta[DESIGN.arm_col] == "Responder"
    nresp_mask = ~resp_mask

    means_r = delta.loc[resp_mask, sig_cols].mean()
    means_nr = delta.loc[nresp_mask, sig_cols].mean()
    sems_r = delta.loc[resp_mask, sig_cols].sem()
    sems_nr = delta.loc[nresp_mask, sig_cols].sem()

    order = means_r.sort_values().index
    display_names = [sig_display(s) for s in order]

    y_pos = np.arange(len(order))
    bar_h = 0.35

    ax.barh(
        y_pos + bar_h / 2, means_r[order].values,
        height=bar_h, color=COL_RESP, alpha=0.85,
        xerr=sems_r[order].values, capsize=2, ecolor=COLORS["gray"],
        label="Responder", edgecolor="none",
    )
    ax.barh(
        y_pos - bar_h / 2, means_nr[order].values,
        height=bar_h, color=COL_NRESP, alpha=0.85,
        xerr=sems_nr[order].values, capsize=2, ecolor=COLORS["gray"],
        label="Non-responder", edgecolor="none",
    )

    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_xlabel("Mean Δ score (Post − Pre)")
    ax.set_title("Signature Changes by Response", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel E: Cohen's d ──────────────────────────────────────────────────

def _panel_e(ax: plt.Axes, data: dict) -> None:
    """Horizontal lollipop of Cohen's d (responder − non-responder) on Δ."""
    delta = data["delta"]
    sig_cols = data["sig_cols"]
    arm = DESIGN.arm_col

    records = []
    for col in sig_cols:
        x = delta.loc[delta[arm] == "Responder", col].dropna().values
        y = delta.loc[delta[arm] == "Non-responder", col].dropna().values
        d = _cohens_d(x, y)
        records.append({"feature": col, "display": sig_display(col), "d": d})

    df = pd.DataFrame(records).dropna(subset=["d"]).sort_values("d")
    y_pos = np.arange(len(df))

    colors = [COL_RESP if v > 0 else COL_NRESP for v in df["d"].values]

    ax.hlines(y_pos, 0, df["d"].values, colors=colors, lw=2, zorder=2)
    ax.scatter(df["d"].values, y_pos, c=colors, s=50,
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel("Cohen's d (Responder − Non-responder)")
    ax.set_title("Effect Size of Response Separation", fontsize=10)
    despine(ax)


# ── Panel F: Individual trajectories for top signature ──────────────────

def _panel_f(ax: plt.Axes, data: dict) -> None:
    """Pre→Post spaghetti plot for the memory T cell signature."""
    pb = data["pb"]
    sig_cols = data["sig_cols"]

    target = None
    for col in sig_cols:
        if "Memory" in col:
            target = col
            break
    if target is None:
        target = sig_cols[0]

    visit_counts = pb.groupby(DESIGN.participant_col)[DESIGN.visit_col].nunique()
    paired_pids = visit_counts[visit_counts == 2].index
    pb_paired = pb[pb[DESIGN.participant_col].isin(paired_pids)].copy()

    for pid in paired_pids:
        rows = pb_paired[pb_paired[DESIGN.participant_col] == pid]
        if len(rows) != 2:
            continue
        arm = rows[DESIGN.arm_col].iloc[0]
        color = COL_RESP if arm == "Responder" else COL_NRESP
        pre_val = rows.loc[rows[DESIGN.visit_col] == "Pre", target].values
        post_val = rows.loc[rows[DESIGN.visit_col] == "Post", target].values
        if len(pre_val) == 0 or len(post_val) == 0:
            continue
        ax.plot(
            [0, 1], [pre_val[0], post_val[0]],
            color=color, alpha=0.5, lw=1.2, zorder=2,
        )
        ax.scatter(
            [0, 1], [pre_val[0], post_val[0]],
            color=color, s=25, edgecolor="white", linewidth=0.4, zorder=3,
        )

    # Group means
    for arm_label, color in [
        ("Responder", COL_RESP),
        ("Non-responder", COL_NRESP),
    ]:
        arm_rows = pb_paired[pb_paired[DESIGN.arm_col] == arm_label]
        pre_mean = arm_rows.loc[
            arm_rows[DESIGN.visit_col] == "Pre", target
        ].mean()
        post_mean = arm_rows.loc[
            arm_rows[DESIGN.visit_col] == "Post", target
        ].mean()
        ax.plot([0, 1], [pre_mean, post_mean], color=color, lw=3, zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel(f"{sig_display(target)} Score")
    ax.set_title(f"Individual Trajectories — {sig_display(target)}", fontsize=10)

    handles = [
        Line2D([0], [0], color=COL_RESP, lw=2, label="Responder"),
        Line2D([0], [0], color=COL_NRESP, lw=2, label="Non-responder"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", frameon=True,
              framealpha=0.9)
    despine(ax)


# ── Panel G: ROC AUC per signature ──────────────────────────────────────

def _panel_g(ax: plt.Axes, data: dict) -> None:
    """Horizontal bar chart of ROC AUC for predicting response from Δ score."""
    delta = data["delta"]
    sig_cols = data["sig_cols"]
    arm = DESIGN.arm_col

    y_true = (delta[arm] == "Responder").astype(int).values

    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "Insufficient class balance",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    records = []
    for col in sig_cols:
        scores = delta[col].values
        if np.isnan(scores).all():
            continue
        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            continue
        auc_disp = max(auc, 1 - auc)
        records.append({
            "feature": col,
            "display": sig_display(col),
            "auc": auc_disp,
        })

    df = pd.DataFrame(records).sort_values("auc", ascending=True)
    y_pos = np.arange(len(df))

    colors = []
    for a in df["auc"].values:
        if a >= 0.8:
            colors.append(COLORS["highlight"])
        elif a >= 0.7:
            colors.append(COLORS["treated"])
        else:
            colors.append(COLORS["gray"])

    ax.barh(y_pos, df["auc"].values, color=colors, alpha=0.85,
            edgecolor="none", height=0.6)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["auc"] + 0.01, i, f"{row['auc']:.2f}",
                va="center", fontsize=7, color="#333333")

    ax.axvline(0.5, ls="--", color=COLORS["gray"], lw=0.8, label="Chance")
    ax.set_xlim(0.35, 1.02)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel("ROC AUC")
    ax.set_title("Predictive Power of Signature Changes", fontsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["highlight"], label="AUC ≥ 0.80"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["treated"], label="AUC ≥ 0.70"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["gray"], label="AUC < 0.70"),
        Line2D([0], [0], ls="--", color=COLORS["gray"], label="Chance"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ── Composite generation ────────────────────────────────────────────────

def generate() -> None:
    """Create and save all Figure 2 panels."""
    print("Figure 2: Melanoma Primary Analysis & Clinical Outcome")
    data = _prepare_data()

    # Panel A: Forest plot
    fig_a, ax_a = plt.subplots(figsize=(12, 5))
    _panel_a(ax_a, data)
    save_panel(fig_a, "panel_A_forest", FIGURE_NAME, MAIN_OUTPUT)

    # Panel B: Interaction grid (needs figure + gridspec)
    fig_b = plt.figure(figsize=(14, 8))
    gs_b = fig_b.add_gridspec(1, 1)[0, 0]
    _panel_b(fig_b, gs_b, data, n_sigs=6)
    save_panel(fig_b, "panel_B_interaction_grid", FIGURE_NAME, MAIN_OUTPUT)

    # Panel C: Heatmap
    fig_c, ax_c = plt.subplots(figsize=(11, 7))
    _panel_c(ax_c, data)
    save_panel(fig_c, "panel_C_heatmap", FIGURE_NAME, MAIN_OUTPUT)

    # Panels D-G: Simple single-axis panels
    simple_panels = [
        ("panel_D_delta_by_response", _panel_d),
        ("panel_E_cohens_d", _panel_e),
        ("panel_F_individual_trajectories", _panel_f),
        ("panel_G_roc_auc", _panel_g),
    ]
    for panel_name, func in simple_panels:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        func(ax, data)
        save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    # Cleanup
    del data["adata"]
    del data
    gc.collect()
    print("  Done.\n")


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    generate()
