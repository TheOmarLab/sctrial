"""
Figure 7 — Clinical Outcome Correlation
=========================================

Four-panel figure demonstrating how participant-level signature changes
correlate with clinical response in the Sade-Feldman melanoma
immunotherapy cohort.

Panels
------
A : Bar plot of mean Δ score (post − pre) for responders vs
    non-responders across all 12 immune signatures.
B : Cohen's d effect sizes (responder − non-responder) on Δ scores.
C : Individual participant trajectories for the memory T cell signature,
    stratified by response.
D : ROC AUC for response prediction based on each signature's Δ score.
"""

from __future__ import annotations

import gc
import warnings

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
    get_sade_feldman,
    harmonize_response,
    save_panel,
    score_signatures,
    sig_display,
)

warnings.filterwarnings("ignore")

FIGURE_NAME = "Figure7_clinical_outcome"
VISITS: tuple[str, str] = ("Pre", "Post")

COL_RESP = COLORS["treated"]     # blue  (Responder)
COL_NRESP = COLORS["control"]    # orange (Non-responder)

DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _prepare_data() -> dict:
    """Load, score, compute per-participant Δ scores."""
    adata = get_sade_feldman()
    if "log1p_tpm" not in adata.layers:
        raise RuntimeError("log1p_tpm layer missing from Sade-Feldman dataset.")
    adata = harmonize_response(adata)
    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

    # ── pseudobulk: per-participant-visit means ─────────────────────────
    grp_cols = [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col]
    pb = (
        adata.obs[grp_cols + sig_cols]
        .groupby(grp_cols, observed=True)[sig_cols]
        .mean()
        .reset_index()
    )

    # Keep only paired participants (have both Pre and Post)
    visit_counts = (
        pb.groupby(DESIGN.participant_col)[DESIGN.visit_col]
        .nunique()
    )
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
        "pb": pb,
        "delta": delta,
    }


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
    # Hedges' correction for small samples
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return d * correction


# ── Panel A: Mean Δ score by response group ─────────────────────────────

def _panel_a(ax, data: dict) -> None:
    """Grouped bar plot of mean Δ scores for responders vs non-responders."""
    delta = data["delta"]
    sig_cols = data["sig_cols"]

    resp_mask = delta[DESIGN.arm_col] == "Responder"
    nresp_mask = ~resp_mask

    means_r = delta.loc[resp_mask, sig_cols].mean()
    means_nr = delta.loc[nresp_mask, sig_cols].mean()
    sems_r = delta.loc[resp_mask, sig_cols].sem()
    sems_nr = delta.loc[nresp_mask, sig_cols].sem()

    # Sort by responder mean for readability
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


# ── Panel B: Cohen's d ──────────────────────────────────────────────────

def _panel_b(ax, data: dict) -> None:
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

    # Colour by direction
    colors = [
        COL_RESP if v > 0 else COL_NRESP for v in df["d"].values
    ]

    ax.hlines(y_pos, 0, df["d"].values, colors=colors, lw=2, zorder=2)
    ax.scatter(df["d"].values, y_pos, c=colors, s=50,
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel("Cohen's d (Responder − Non-responder)")
    ax.set_title("Effect Size of Response Separation", fontsize=10)
    despine(ax)


# ── Panel C: Individual trajectories for top signature ──────────────────

def _panel_c(ax, data: dict) -> None:
    """Pre→Post spaghetti plot for the memory T cell signature."""
    pb = data["pb"]
    sig_cols = data["sig_cols"]

    # Find the memory T cell column
    target = None
    for col in sig_cols:
        if "Memory" in col:
            target = col
            break
    if target is None:
        target = sig_cols[0]

    # Filter to paired participants only
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
    for arm_label, color, ls in [
        ("Responder", COL_RESP, "-"),
        ("Non-responder", COL_NRESP, "-"),
    ]:
        arm_rows = pb_paired[pb_paired[DESIGN.arm_col] == arm_label]
        pre_mean = arm_rows.loc[
            arm_rows[DESIGN.visit_col] == "Pre", target
        ].mean()
        post_mean = arm_rows.loc[
            arm_rows[DESIGN.visit_col] == "Post", target
        ].mean()
        ax.plot(
            [0, 1], [pre_mean, post_mean],
            color=color, lw=3, ls=ls, zorder=5,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel(f"{sig_display(target)} Score")
    ax.set_title(f"Individual Trajectories — {sig_display(target)}", fontsize=10)

    handles = [
        Line2D([0], [0], color=COL_RESP, lw=2, label="Responder"),
        Line2D([0], [0], color=COL_NRESP, lw=2, label="Non-responder"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel D: ROC AUC per signature ──────────────────────────────────────

def _panel_d(ax, data: dict) -> None:
    """Horizontal bar chart of ROC AUC for predicting response from Δ score."""
    delta = data["delta"]
    sig_cols = data["sig_cols"]
    arm = DESIGN.arm_col

    y_true = (delta[arm] == "Responder").astype(int).values

    # Need at least both classes
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
        # Flip if AUC < 0.5 (use |AUC - 0.5| + 0.5 to report best direction)
        auc_disp = max(auc, 1 - auc)
        records.append({
            "feature": col,
            "display": sig_display(col),
            "auc": auc_disp,
        })

    df = pd.DataFrame(records).sort_values("auc", ascending=True)
    y_pos = np.arange(len(df))

    # Color intensity by AUC
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

    # AUC value labels
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
    """Create and save all Figure 7 panels."""
    apply_style()
    print("Figure 7: Clinical Outcome Correlation")
    data = _prepare_data()

    panel_funcs = [
        ("panel_A_delta_by_response", _panel_a),
        ("panel_B_cohens_d", _panel_b),
        ("panel_C_individual_trajectories", _panel_c),
        ("panel_D_roc_auc", _panel_d),
    ]
    for panel_name, func in panel_funcs:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        func(ax, data)
        save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    # ── Cleanup ─────────────────────────────────────────────────────────
    adata = data.get("adata")
    if adata is not None:
        del adata
    del data
    gc.collect()

    print(f"  Figure 7 complete: {FIGURE_NAME}")
