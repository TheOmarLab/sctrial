"""
Supplementary Figure 3 — Pseudoreplication Bias & Melanoma/TNBC Primary Analysis
==================================================================================

Eleven-panel figure: empirical demonstration of pseudoreplication (A–C),
TNBC cross-dataset comparison panels (D–F), primary DiD analysis (G–H),
per-participant effects (I–J), and clinical outcome correlation (K).

Panels A–C, G–K use the Sade-Feldman melanoma immunotherapy dataset.
Panels D–F use the Zhang et al. TNBC dataset (GSE169246).

Panels
------
A : Paired-participant verification (cells per participant × visit) [Melanoma].
B : Coefficient comparison: cell-level vs participant-level aggregation [Melanoma].
C : P-value comparison: −log₁₀ scale, illustrating inflation at cell level [Melanoma].
D : Mean Δ score (post − pre) by treatment arm [TNBC].
E : Within-arm response DID forest plot — Chemo arm [TNBC].
F : Within-arm response DID forest plot — anti-PDL1+Chemo arm [TNBC].
G : Forest plot of DiD effects across all gene signatures [Melanoma].
H : Small-multiple interaction plots for the top 6 signatures [Melanoma].
I : Per-participant change heatmap across signatures [Melanoma].
J : Bar plot of mean Δ score (post − pre) by response group [Melanoma].
K : Cohen's d effect sizes (responder − non-responder) on Δ scores [Melanoma].
"""

from __future__ import annotations

import gc
import warnings
from contextlib import contextmanager

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy import stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    TrialDesign,
    apply_style,
    despine,
    did_table,
    get_sade_feldman,
    get_tnbc_zhang,
    harmonize_response,
    save_panel,
    score_signatures,
    sig_display,
    verify_paired_participants,
)

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────

FIGURE_NAME = "SuppFig3_melanoma_tnbc_analysis"
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

# TNBC design (Zhang et al., Cancer Cell 2021)
TNBC_DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="arm",
    arm_treated="anti-PDL1+Chemo",
    arm_control="Chemo",
)
_TNBC_RESP_COL = "response"
_TNBC_R_VAL    = "R"
_TNBC_NR_VAL   = "NR"


# ── Data preparation ─────────────────────────────────────────────────────

def _prepare_data() -> dict:
    """Load Sade-Feldman, score signatures, run DiD and compute Δ scores."""
    adata = get_sade_feldman()
    if "log1p_tpm" not in adata.layers:
        raise RuntimeError("log1p_tpm layer missing from Sade-Feldman dataset.")
    adata = harmonize_response(adata)
    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

    # DiD at cell level (for pseudoreplication panels A-C)
    res_cell = did_table(
        adata,
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer="log1p_tpm",
        standardize=True,
        aggregate="cell",
    )

    # Paired-participant verification
    pair_info = verify_paired_participants(
        adata.obs,
        visit_col="visit",
        visits=VISITS,
        participant_col="participant_id",
    )

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
        "res_cell": res_cell,
        "pair_info": pair_info,
        "did_res": did_res,
        "pb": pb,
        "delta": delta,
    }


def _prepare_tnbc_data() -> dict:
    """Load TNBC h5ad, score signatures, compute per-participant delta scores."""
    adata = get_tnbc_zhang()
    if "log1p_norm" not in adata.layers:
        raise RuntimeError("log1p_norm layer missing from TNBC dataset.")
    adata, sig_cols = score_signatures(adata, layer="log1p_norm")
    print(f"  [TNBC] Scored {len(sig_cols)} signatures")

    grp_cols = [TNBC_DESIGN.participant_col, TNBC_DESIGN.visit_col, TNBC_DESIGN.arm_col]
    pb = (
        adata.obs[grp_cols + sig_cols]
        .groupby(grp_cols, observed=True)[sig_cols]
        .mean()
        .reset_index()
    )

    visit_counts = pb.groupby(TNBC_DESIGN.participant_col)[TNBC_DESIGN.visit_col].nunique()
    paired_pids = visit_counts[visit_counts == 2].index
    pb = pb[pb[TNBC_DESIGN.participant_col].isin(paired_pids)].copy()

    pre = pb[pb[TNBC_DESIGN.visit_col] == "Pre"].set_index(TNBC_DESIGN.participant_col)
    post = pb[pb[TNBC_DESIGN.visit_col] == "Post"].set_index(TNBC_DESIGN.participant_col)
    delta = post[sig_cols].subtract(pre[sig_cols])
    delta[TNBC_DESIGN.arm_col] = pre[TNBC_DESIGN.arm_col]
    delta = delta.reset_index()

    response_map = (
        adata.obs
        .groupby(TNBC_DESIGN.participant_col, observed=True)[_TNBC_RESP_COL]
        .first()
    )
    delta[_TNBC_RESP_COL] = delta[TNBC_DESIGN.participant_col].map(response_map)

    n_t = (delta[TNBC_DESIGN.arm_col] == TNBC_DESIGN.arm_treated).sum()
    n_c = (delta[TNBC_DESIGN.arm_col] == TNBC_DESIGN.arm_control).sum()
    print(f"  [TNBC] Paired participants: {len(delta)} (treated={n_t}, control={n_c})")

    return {"sig_cols": sig_cols, "delta": delta}


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


# ── Panel A: Paired-participant verification (from Figure 1B) ────────────

def _panel_a_paired_verification(ax: plt.Axes, data: dict) -> None:
    """Grouped bar chart of cells per participant × visit, colored by response."""
    adata = data["adata"]
    obs = adata.obs.copy()

    counts = (
        obs.groupby(["participant_id", "visit", "response"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    counts["visit"] = pd.Categorical(
        counts["visit"], categories=["Pre", "Post"], ordered=True,
    )
    counts = counts.sort_values(["response", "participant_id", "visit"])

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

    ax.set_xticks(range(len(participants)))
    ax.set_xticklabels(
        [f"P{i+1}" for i in range(len(participants))],
        rotation=90, ha="center", fontsize=5,
    )
    ax.set_xlabel("Participant", fontsize=9)
    ax.set_ylabel("Number of cells", fontsize=12)
    ax.set_title("Paired Participants: Cells per Visit (Melanoma)", fontsize=11,
                 fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["treated"],
               markersize=5, markeredgewidth=0, label="Responder"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["control"],
               markersize=5, markeredgewidth=0, label="Non-responder"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["gray"],
               markersize=5, markeredgewidth=0, alpha=0.6, label="Pre"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["gray"],
               markersize=5, markeredgewidth=0, label="Post"),
    ]
    ax.legend(handles=legend_handles, fontsize=16,
              loc="upper center", bbox_to_anchor=(0.5, -0.38),
              ncol=4, frameon=True, framealpha=0.9,
              handletextpad=0.3, columnspacing=0.8)

    pair_info = data["pair_info"]
    ax.text(
        0.02, 0.98,
        f"{pair_info['n_paired']}/{pair_info['n_total']} participants paired",
        transform=ax.transAxes, fontsize=5.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.8),
    )
    despine(ax)


# ── Panel B: Coefficient comparison (from Figure 1C) ────────────────────

def _panel_b_beta_comparison(ax: plt.Axes, data: dict) -> None:
    """Scatter of cell-level vs participant-level beta_DiD with identity line."""
    res_cell = data["res_cell"].set_index("feature")
    res_part = data["did_res"].set_index("feature")
    common = res_cell.index.intersection(res_part.index)

    beta_cell = res_cell.loc[common, "beta_DiD"].values
    beta_part = res_part.loc[common, "beta_DiD"].values

    colors = [COLORS["treated"] if b > 0 else COLORS["control"] for b in beta_part]

    ax.scatter(beta_cell, beta_part, c=colors, s=20, edgecolors="white",
               linewidths=0.5, zorder=3)

    lim_lo = min(beta_cell.min(), beta_part.min(), -1.0) * 1.15
    lim_hi = max(beta_cell.max(), beta_part.max()) * 1.15
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color=COLORS["gray"],
            lw=1, zorder=1, label="Identity")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.axhline(0, color=COLORS["gray"], lw=0.5, ls=":", zorder=0)
    ax.axvline(0, color=COLORS["gray"], lw=0.5, ls=":", zorder=0)

    # adjustText with leader lines resolves the dense near-origin label cluster
    # (Memory/Regulatory T Cells previously overprinted each other).
    from adjustText import adjust_text
    texts = [ax.text(xv, yv, sig_display(feat), fontsize=8, alpha=0.9)
             for feat, xv, yv in zip(common, beta_cell, beta_part)]
    adjust_text(
        texts, ax=ax, expand=(1.05, 1.1),
        arrowprops=dict(arrowstyle="-", color=COLORS["gray"], lw=0.4, alpha=0.6),
        ensure_inside_axes=True,
    )

    r, p = stats.pearsonr(beta_cell, beta_part)
    ax.text(
        0.05, 0.95,
        f"r = {r:.2f}, p = {p:.1e}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.8),
    )

    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (cell-level)", fontsize=12)
    ax.set_ylabel(r"$\beta_{\mathrm{DiD}}$ (participant-level)", fontsize=12)
    ax.set_title("Effect Size: Cell vs Participant Aggregation (Melanoma)", fontsize=11,
                 fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["treated"], label="Positive effect"),
        mpatches.Patch(facecolor=COLORS["control"], label="Negative effect"),
    ]
    ax.legend(handles=legend_handles, fontsize=16,
              loc="lower left",
              frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel C: P-value inflation (from Figure 1D) ─────────────────────────

def _panel_c_pvalue_inflation(ax: plt.Axes, data: dict) -> None:
    """Horizontal bar chart of -log10(p) at cell vs participant level."""
    res_cell = data["res_cell"].set_index("feature")
    res_part = data["did_res"].set_index("feature")
    common = res_cell.index.intersection(res_part.index)

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

    ax.barh(y_pos - bar_h / 2, df["nlog10_cell"], height=bar_h,
            color=COLORS["highlight"], alpha=0.8, label="Cell-level", zorder=2)
    ax.barh(y_pos + bar_h / 2, df["nlog10_part"], height=bar_h,
            color=COLORS["treated"], alpha=0.8, label="Participant-level", zorder=2)

    thresh = -np.log10(0.05)
    ax.axvline(thresh, color=COLORS["gray"], ls="--", lw=1, zorder=1)
    ax.text(thresh + 0.1, len(df) - 0.5, "p = 0.05", fontsize=8,
            va="bottom", color=COLORS["gray"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"], fontsize=3.5)
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.set_title("P-value Inflation: Cell vs Participant Level (Melanoma)", fontsize=11,
                 fontweight="bold")

    ax.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=True, framealpha=0.9)
    despine(ax)


# ── TNBC helper ──────────────────────────────────────────────────────────

def _within_arm_response_forest_tnbc(
    ax: plt.Axes,
    tnbc_data: dict,
    arm_label: str,
    arm_color: str,
    show_stars: bool = True,
    legend_loc: str = "upper left",
) -> None:
    """Forest plot of DID_response = (Post-Pre)_R − (Post-Pre)_NR within one TNBC arm."""
    delta    = tnbc_data["delta"]
    sig_cols = tnbc_data["sig_cols"]

    mask_R  = (delta[TNBC_DESIGN.arm_col] == arm_label) & (delta[_TNBC_RESP_COL] == _TNBC_R_VAL)
    mask_NR = (delta[TNBC_DESIGN.arm_col] == arm_label) & (delta[_TNBC_RESP_COL] == _TNBC_NR_VAL)
    n_R  = int(mask_R.sum())
    n_NR = int(mask_NR.sum())

    rng = np.random.default_rng(42)
    records = []
    for col in sig_cols:
        d_R  = delta.loc[mask_R,  col].dropna().values
        d_NR = delta.loc[mask_NR, col].dropna().values
        if len(d_R) == 0 or len(d_NR) == 0:
            continue
        point = d_R.mean() - d_NR.mean()
        boots = []
        for _ in range(999):
            b_R  = rng.choice(d_R,  len(d_R),  replace=True)
            b_NR = rng.choice(d_NR, len(d_NR), replace=True)
            boots.append(b_R.mean() - b_NR.mean())
        boots = np.array(boots)
        records.append({
            "feature": col,
            "DID":    point,
            "ci_lo":  float(np.percentile(boots, 2.5)),
            "ci_hi":  float(np.percentile(boots, 97.5)),
        })

    df = pd.DataFrame(records).sort_values("DID").reset_index(drop=True)
    y_pos = np.arange(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        color = COL_RESP if row["DID"] > 0 else COL_NRESP
        ax.hlines(y_pos[i], row["ci_lo"], row["ci_hi"],
                  color=color, linewidth=1.2, alpha=1.0, zorder=1)
        ax.scatter(row["DID"], y_pos[i], color=color, s=12,
                   edgecolors="white", linewidths=0.5, zorder=2)
        if show_stars and not (row["ci_lo"] < 0 < row["ci_hi"]):
            ax.text(row["ci_hi"] + 0.02, y_pos[i], "*",
                    va="center", fontsize=10, fontweight="bold", color=color)

    ax.axvline(0, color="#333333", lw=0.9, ls="--", zorder=0, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sig_display(f) for f in df["feature"]], fontsize=3.5)
    ax.set_xlabel(
        r"DID$_{\mathrm{response}}$ = $\Delta$R $-$ $\Delta$NR",
        fontsize=9,
    )
    ax.set_title(
        f"Response DID within {arm_label}\n"
        r"(Post$-$Pre)$_R$ $-$ (Post$-$Pre)$_{NR}$ (TNBC)",
        fontsize=9, fontweight="bold",
    )
    ax.set_ylim(-0.6, len(df) - 0.4)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_RESP, markersize=5,
               label=r"Responder $\uparrow$"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_NRESP, markersize=5,
               label=r"Non-responder $\uparrow$"),
    ]
    ax.legend(handles=handles, fontsize=16, loc=legend_loc,
              frameon=True, framealpha=0.9,
              borderpad=0.3, labelspacing=0.2, handlelength=1.0)
    despine(ax)


# ── Panel D: Mean Δ by arm (TNBC) ─────────────────────────────────────────

def _panel_d_tnbc_mean_delta_by_arm(ax: plt.Axes, tnbc_data: dict) -> None:
    """Bar chart of mean post−pre Δ per arm for the TNBC dataset."""
    delta    = tnbc_data["delta"]
    sig_cols = tnbc_data["sig_cols"]
    arm      = TNBC_DESIGN.arm_col

    treat_mask = delta[arm] == TNBC_DESIGN.arm_treated
    ctrl_mask  = ~treat_mask

    means_t = delta.loc[treat_mask, sig_cols].mean()
    means_c = delta.loc[ctrl_mask,  sig_cols].mean()
    sems_t  = delta.loc[treat_mask, sig_cols].sem()
    sems_c  = delta.loc[ctrl_mask,  sig_cols].sem()

    order         = means_t.sort_values().index
    display_names = [sig_display(s) for s in order]
    y_pos  = np.arange(len(order))
    bar_h  = 0.35

    ax.barh(y_pos + bar_h / 2, means_t[order].values, height=bar_h,
            color=COL_RESP, alpha=0.85,
            xerr=sems_t[order].values, capsize=1, ecolor=COL_GRAY,
            error_kw={"linewidth": 0.4}, label=TNBC_DESIGN.arm_treated)
    ax.barh(y_pos - bar_h / 2, means_c[order].values, height=bar_h,
            color=COL_NRESP, alpha=0.85,
            xerr=sems_c[order].values, capsize=1, ecolor=COL_GRAY,
            error_kw={"linewidth": 0.4}, label=TNBC_DESIGN.arm_control)

    ax.axvline(0, ls=":", color=COL_GRAY, lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=3.5)
    ax.set_xlabel("Mean Δ score (Post − Pre)")
    ax.set_title("Signature Changes by Arm (TNBC)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=16, loc="center left", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel E: Response DID — Chemo arm (TNBC) ─────────────────────────────

def _panel_e_tnbc_response_did_chemo(ax: plt.Axes, tnbc_data: dict) -> None:
    """Panel E: response DID within the Chemo arm (TNBC)."""
    _within_arm_response_forest_tnbc(
        ax, tnbc_data, TNBC_DESIGN.arm_control, COL_NRESP,
        show_stars=False, legend_loc="upper left",
    )


# ── Panel F: Response DID — anti-PDL1+Chemo arm (TNBC) ───────────────────

def _panel_f_tnbc_response_did_antipdl1(ax: plt.Axes, tnbc_data: dict) -> None:
    """Panel F: response DID within the anti-PDL1+Chemo arm (TNBC)."""
    _within_arm_response_forest_tnbc(
        ax, tnbc_data, TNBC_DESIGN.arm_treated, COL_RESP,
    )


# ── Panel G: Forest plot (Melanoma) ──────────────────────────────────────

def _panel_g_forest(ax: plt.Axes, data: dict) -> None:
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
            color=color, linewidth=1.2, alpha=1.0, zorder=1,
        )
        ax.scatter(
            row["beta_DiD"], y_pos[i], color=color, s=12,
            edgecolors="white", linewidths=0.5, alpha=1.0, zorder=2,
        )

    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", zorder=0,
               alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sig_display(f) for f in df["feature"]], fontsize=3.5)
    ax.set_xlabel(r"DiD coefficient ($\beta$, standardised)", fontsize=11)
    ax.set_title("DiD Effects Across Signatures (Melanoma)", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(-0.6, len(df) - 0.4)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_RESP, markersize=5,
               label=r"Responder $\uparrow$"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_NRESP, markersize=5,
               label=r"Non-responder $\uparrow$"),
    ]
    # Bottom rows (most negative) have short right whiskers, so the lower-right
    # corner is clear; upper-left collided with the top rows' left whiskers.
    ax.legend(
        handles=legend_handles, fontsize=12,
        loc="lower right",
        ncol=1, frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
        handletextpad=0.4, borderpad=0.5,
    )
    despine(ax)


# ── Panel E: Small-multiple interaction plots ────────────────────────────

def _panel_h_interaction_grid(
    fig: plt.Figure,
    gs_parent: gridspec.SubplotSpec,
    data: dict,
    n_sigs: int = 6,
    *,
    inner_hspace: float = 0.70,
    inner_wspace: float = 0.35,
    inner_top: float = 1.0,
    group_title: str = "",
) -> tuple[list[plt.Axes], plt.Axes | None]:
    """2×3 grid of interaction plots for the top *n_sigs* signatures.

    Returns (axes, ax_hdr) where ax_hdr is the invisible header axes that
    holds the group title (None when inner_top == 1.0).  The caller should
    place the panel letter label on ax_hdr *after* any font-size capping so
    the label matches the size of neighbouring panel letters.
    """
    adata = data["adata"]
    did_res = data["did_res"]

    if "p_DiD_boot" in did_res.columns:
        rank_p = did_res["p_DiD_boot"].fillna(did_res["p_DiD"])
    else:
        rank_p = did_res["p_DiD"]
    top = did_res.assign(_rank_p=rank_p).sort_values("_rank_p").head(n_sigs).drop(columns="_rank_p")

    nrows, ncols = 2, 3
    ax_hdr: plt.Axes | None = None
    if inner_top < 1.0:
        # Reserve (1-inner_top) fraction at top as a title/label slot;
        # panels occupy the bottom inner_top fraction.
        _gs_split = gs_parent.subgridspec(
            2, 1,
            height_ratios=[1.0 - inner_top, inner_top],
            hspace=0.0,
        )
        ax_hdr = fig.add_subplot(_gs_split[0])
        ax_hdr.axis("off")
        if group_title:
            # y=1.0 va='bottom' → bottom of text sits at top of ax_hdr,
            # which equals the top of the H region ≈ top of neighbouring ax_f.
            ax_hdr.text(
                0.5, 1.05, group_title,
                transform=ax_hdr.transAxes,
                ha="center", va="bottom",
                fontsize=7, fontweight="bold",
                clip_on=False,
            )
        gs_inner = _gs_split[1].subgridspec(
            nrows, ncols, hspace=inner_hspace, wspace=inner_wspace,
        )
    else:
        gs_inner = gs_parent.subgridspec(
            nrows, ncols, hspace=inner_hspace, wspace=inner_wspace,
        )
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
                color=color, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
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
            fontsize=6, fontweight="bold", pad=-2,
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
        handles=legend_handles, fontsize=16,
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=3, frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
    )
    return axes, ax_hdr


# ── Panel F: Per-participant change heatmap ──────────────────────────────

def _panel_i_heatmap(ax: plt.Axes, data: dict) -> None:
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

    # Row annotations — arm colour sidebar. Keep it INSIDE the axes (right of the
    # left spine) so it never overprints the participant-ID y-tick labels, which
    # live in the margin to the left of the spine. The widened left xlim below
    # reserves the strip's column.
    sidebar_w = 0.35
    _sb_left = -0.5 - sidebar_w            # right edge abuts the first data column
    for i, pid in enumerate(ordered_pids):
        arm = arms.loc[pid]
        color = COL_RESP if arm == "Responder" else COL_NRESP
        ax.add_patch(plt.Rectangle(
            (_sb_left, i - 0.5), sidebar_w, 1.0,
            color=color, clip_on=False,
        ))
    ax.set_xlim(_sb_left - 0.15, len(delta.columns) - 0.5)

    # Separator between groups
    n_resp = sum(1 for p in ordered_pids if arms.loc[p] == "Responder")
    if 0 < n_resp < len(ordered_pids):
        ax.axhline(n_resp - 0.5, color="white", linewidth=2.5, zorder=5)

    ax.set_xticks(np.arange(len(delta.columns)))
    ax.set_xticklabels(delta.columns, fontsize=4.5, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(delta.index)))
    ylabels = [str(pid) for pid in ordered_pids]
    ax.set_yticklabels(ylabels, fontsize=8.5,
                       fontfamily="monospace", fontweight="medium")
    for i, tick in enumerate(ax.get_yticklabels()):
        arm = arms.iloc[i]
        tick.set_color(COL_RESP if arm == "Responder" else COL_NRESP)
    ax.set_title("Per-participant Score Change (Post − Pre) (Melanoma)", fontsize=12,
                 fontweight="bold")

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score Δ", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_RESP,
               markersize=9, markeredgewidth=0, label="Responder"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_NRESP,
               markersize=9, markeredgewidth=0, label="Non-responder"),
    ]
    # Anchor well below the rotated x-tick labels so the key never overlaps them.
    ax.legend(
        handles=legend_handles, fontsize=10,
        loc="upper center", bbox_to_anchor=(0.5, -0.32),
        ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.5,
    )


# ── Panel G: Mean Δ score by response group ─────────────────────────────

def _panel_j_delta_by_response(ax: plt.Axes, data: dict) -> None:
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
        xerr=sems_r[order].values, capsize=1, ecolor=COLORS["gray"],
        error_kw={"linewidth": 0.4},
        label="Responder", edgecolor="none",
    )
    ax.barh(
        y_pos - bar_h / 2, means_nr[order].values,
        height=bar_h, color=COL_NRESP, alpha=0.85,
        xerr=sems_nr[order].values, capsize=1, ecolor=COLORS["gray"],
        error_kw={"linewidth": 0.4},
        label="Non-responder", edgecolor="none",
    )

    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=3.5)
    ax.set_xlabel("Mean Δ score (Post − Pre)")
    ax.set_title("Signature Changes by Response (Melanoma)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel H: Cohen's d ──────────────────────────────────────────────────

def _panel_k_cohens_d(ax: plt.Axes, data: dict) -> None:
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

    ax.hlines(y_pos, 0, df["d"].values, colors=colors, lw=1.2, zorder=2)
    ax.scatter(df["d"].values, y_pos, c=colors, s=12,
               edgecolor="white", linewidth=0.4, zorder=3)

    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=3.5)
    ax.set_xlabel("Cohen's d (Responder − Non-responder)")
    ax.set_title("Effect Size of Response Separation (Melanoma)", fontsize=10,
                 fontweight="bold")
    despine(ax)


# ── Panel I: Individual trajectories for top signature ──────────────────

def _panel_i_individual_trajectories(ax: plt.Axes, data: dict) -> None:
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
    ax.set_title(f"Individual Trajectories — {sig_display(target)}", fontsize=10,
                 fontweight="bold")

    handles = [
        Line2D([0], [0], color=COL_RESP, lw=2, label="Responder"),
        Line2D([0], [0], color=COL_NRESP, lw=2, label="Non-responder"),
    ]
    ax.legend(handles=handles, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=True, framealpha=0.9)
    despine(ax)


# ── Composite generation ────────────────────────────────────────────────

_BIG_FONT_RC = {
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
}

_MIN_FONT = 11  # floor: every text element will be at least this big


@contextmanager
def _big_fonts():
    """Temporarily raise all font sizes so standalone panels are readable."""
    prev = {k: plt.rcParams[k] for k in _BIG_FONT_RC}
    plt.rcParams.update(_BIG_FONT_RC)
    try:
        yield
    finally:
        plt.rcParams.update(prev)


def _enforce_min_fontsize(fig, minimum: float = _MIN_FONT) -> None:
    """Walk every text element in *fig* and raise any font size below *minimum*."""
    for ax in fig.get_axes():
        for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                    + ax.get_xticklabels() + ax.get_yticklabels()
                    + ax.texts):
            if txt.get_fontsize() < minimum:
                txt.set_fontsize(minimum)
        if ax.get_legend():
            for txt in ax.get_legend().get_texts():
                if txt.get_fontsize() < minimum:
                    txt.set_fontsize(minimum)
    for txt in fig.texts:
        if txt.get_fontsize() < minimum:
            txt.set_fontsize(minimum)


def generate() -> None:
    """Create and save all Figure 2 panels (A–K)."""
    print("Supp Fig 3: Pseudoreplication Bias & Melanoma/TNBC Primary Analysis")
    data      = _prepare_data()
    tnbc_data = _prepare_tnbc_data()

    with _big_fonts():
        # Panels A–C: Pseudoreplication demonstration (Melanoma)
        pseudo_panels = [
            ("panel_A_paired_verification", _panel_a_paired_verification, (11, 6)),
            ("panel_B_beta_comparison",     _panel_b_beta_comparison,     (11, 6)),
            ("panel_C_pvalue_inflation",    _panel_c_pvalue_inflation,    (8, 6)),
        ]
        for panel_name, func, size in pseudo_panels:
            fig, ax = plt.subplots(figsize=size)
            func(ax, data)
            _enforce_min_fontsize(fig)
            fig.tight_layout()
            save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

        # Panel D: mean Δ by arm (TNBC)
        fig_d, ax_d_solo = plt.subplots(figsize=(10, 7))
        _panel_d_tnbc_mean_delta_by_arm(ax_d_solo, tnbc_data)
        _enforce_min_fontsize(fig_d)
        fig_d.tight_layout()
        save_panel(fig_d, "panel_D_tnbc_mean_delta_by_arm", FIGURE_NAME, SUPP_OUTPUT)

        # Panel E: response DID within Chemo arm (TNBC)
        fig_e, ax_e_solo = plt.subplots(figsize=(10, 7))
        _panel_e_tnbc_response_did_chemo(ax_e_solo, tnbc_data)
        _enforce_min_fontsize(fig_e)
        fig_e.tight_layout()
        save_panel(fig_e, "panel_E_tnbc_response_did_chemo", FIGURE_NAME, SUPP_OUTPUT)

        # Panel F: response DID within anti-PDL1+Chemo arm (TNBC)
        fig_f_tnbc, ax_f_solo = plt.subplots(figsize=(10, 7))
        _panel_f_tnbc_response_did_antipdl1(ax_f_solo, tnbc_data)
        _enforce_min_fontsize(fig_f_tnbc)
        fig_f_tnbc.tight_layout()
        save_panel(fig_f_tnbc, "panel_F_tnbc_response_did_antipdl1", FIGURE_NAME, SUPP_OUTPUT)

        # Panel G: Forest plot (Melanoma)
        fig_g, ax_g_solo = plt.subplots(figsize=(12, 6))
        _panel_g_forest(ax_g_solo, data)
        _enforce_min_fontsize(fig_g)
        fig_g.tight_layout()
        save_panel(fig_g, "panel_G_forest", FIGURE_NAME, SUPP_OUTPUT)

        # Panel H: Interaction grid (Melanoma, needs figure + gridspec)
        fig_h = plt.figure(figsize=(14, 5.5))
        gs_h  = fig_h.add_gridspec(1, 1)[0, 0]
        _panel_h_interaction_grid(fig_h, gs_h, data, n_sigs=6, inner_top=0.92)  # ax_hdr unused in standalone
        fig_h.suptitle("Participant-Level Trajectories (Melanoma)",
                       fontsize=11, fontweight="bold", y=0.97)
        _enforce_min_fontsize(fig_h)
        fig_h.tight_layout(rect=[0, 0, 1, 0.90])
        save_panel(fig_h, "panel_H_interaction_grid", FIGURE_NAME, SUPP_OUTPUT)

        # Panel I: Heatmap (Melanoma)
        fig_i, ax_i_solo = plt.subplots(figsize=(12, 8))
        _panel_i_heatmap(ax_i_solo, data)
        _enforce_min_fontsize(fig_i)
        fig_i.tight_layout()
        save_panel(fig_i, "panel_I_heatmap", FIGURE_NAME, SUPP_OUTPUT)

        # Panels J, K: simple single-axis panels (Melanoma)
        simple_panels = [
            ("panel_J_delta_by_response", _panel_j_delta_by_response),
            ("panel_K_cohens_d",          _panel_k_cohens_d),
        ]
        for panel_name, func in simple_panels:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            func(ax, data)
            _enforce_min_fontsize(fig)
            fig.tight_layout()
            save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

    # ── Combined artboard (180 × 270 mm, taller to fit new TNBC row) ──────
    _SMALL_RC = {
        "font.size":             5,
        "axes.titlesize":        5.5,
        "axes.labelsize":        5,
        "xtick.labelsize":       4.5,
        "ytick.labelsize":       4.5,
        "legend.fontsize":       4,
        "legend.title_fontsize": 4,
    }
    _MAX_FONT_COMPOSITE = 6

    def _cap_fontsize(fig, maximum):
        for ax in fig.get_axes():
            for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                        + ax.get_xticklabels() + ax.get_yticklabels()
                        + ax.texts):
                if txt.get_fontsize() > maximum:
                    txt.set_fontsize(maximum)
            if ax.get_legend():
                for txt in ax.get_legend().get_texts():
                    if txt.get_fontsize() > maximum:
                        txt.set_fontsize(maximum)
        for txt in fig.texts:
            if txt.get_fontsize() > maximum:
                txt.set_fontsize(maximum)

    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm  = 1.0 / 25.4
    fig_c = plt.figure(figsize=(180 * _mm, 215 * _mm))

    #   Row 0: A | B
    #   Row 1: C | D | E  (3 columns)
    #   Row 2: F (top-left) + G (bottom-left) | H (right, spans both)
    #   Row 3: I | J
    #   Row 4: K (centred)
    outer = fig_c.add_gridspec(
        5, 1,
        height_ratios=[1, 1.4, 2.8, 1.3, 1.4],
        hspace=0.70,
        left=0.10, right=0.95, top=0.97, bottom=0.05,
    )

    # Row 0: A | B
    gs0  = outer[0].subgridspec(1, 2, wspace=0.28, width_ratios=[1, 1.4])
    ax_a = fig_c.add_subplot(gs0[0])
    ax_b = fig_c.add_subplot(gs0[1])

    # Row 1: C | D | E  (C and E narrower, wide gutters so C's title and its
    # p=0.05 annotation do not bleed into panel D's letter / y-tick labels)
    gs1  = outer[1].subgridspec(1, 3, wspace=0.95, width_ratios=[0.72, 1, 0.72])
    ax_c = fig_c.add_subplot(gs1[0])
    ax_d = fig_c.add_subplot(gs1[1])
    ax_e = fig_c.add_subplot(gs1[2])

    # Row 2: F (top-left) + G (bottom-left) | H (right, spans both)
    gs2       = outer[2].subgridspec(2, 2, width_ratios=[1, 1.6],
                                     hspace=0.70, wspace=0.40)
    ax_f      = fig_c.add_subplot(gs2[0, 0])    # Panel F (TNBC anti-PDL1+Chemo DID)
    ax_g_comp = fig_c.add_subplot(gs2[1, 0])    # Panel G (Melanoma forest)

    # Row 3: I | J
    gs3  = outer[3].subgridspec(1, 2, wspace=0.55)
    ax_i = fig_c.add_subplot(gs3[0])
    ax_j = fig_c.add_subplot(gs3[1])

    # Row 4: K (centred)
    gs4  = outer[4].subgridspec(1, 3, width_ratios=[0.6, 1.8, 0.6], wspace=0.40)
    ax_k = fig_c.add_subplot(gs4[1])

    # ── Draw all panels ────────────────────────────────────────────────────
    _panel_a_paired_verification(ax_a, data)
    _panel_b_beta_comparison(ax_b, data)
    _panel_c_pvalue_inflation(ax_c, data)
    _panel_d_tnbc_mean_delta_by_arm(ax_d, tnbc_data)
    _panel_e_tnbc_response_did_chemo(ax_e, tnbc_data)
    _panel_f_tnbc_response_did_antipdl1(ax_f, tnbc_data)
    _panel_g_forest(ax_g_comp, data)
    axes_h, ax_h_hdr = _panel_h_interaction_grid(
        fig_c, gs2[:, 1], data, n_sigs=6,
        inner_hspace=0.80, inner_wspace=0.30, inner_top=0.88,
        group_title="Participant-Level Trajectories (Melanoma)",
    )
    _panel_i_heatmap(ax_i, data)
    _panel_j_delta_by_response(ax_j, data)
    _panel_k_cohens_d(ax_k, data)

    # ── Combined-panel-only adjustments ───────────────────────────────────

    # Move below-figure legends inside the plot area for space
    _inside = {
        ax_a:      "upper right",
        ax_b:      "lower left",
        ax_c:      "lower right",    # Melanoma p-value inflation
        ax_d:      "center left",    # TNBC mean delta by arm
        ax_e:      "upper left",     # TNBC Chemo DID
        ax_f:      "upper left",     # TNBC anti-PDL1 DID
        ax_g_comp: "upper left",     # Melanoma forest
        ax_j:      "lower right",    # Melanoma delta by response
    }
    for ax_target, loc in _inside.items():
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels  = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_target.legend(
                handles=handles, labels=labels,
                fontsize=4.5, loc=loc,
                frameon=True, framealpha=0.85,
                handlelength=1, handletextpad=0.3,
                borderpad=0.3, labelspacing=0.2,
            )

    # Panel H (interaction grid): consolidate legend at centre-bottom
    if axes_h:
        leg_h = axes_h[-1].get_legend()
        if leg_h:
            leg_h.remove()
        mid_ax = axes_h[4]
        _hh = [
            Line2D([0], [0], color=COL_RESP, linewidth=1.5, marker="o",
                   markersize=3, markeredgecolor="white", label="Responder"),
            Line2D([0], [0], color=COL_NRESP, linewidth=1.5, marker="o",
                   markersize=3, markeredgecolor="white",
                   label="Non-responder"),
            Line2D([0], [0], color=COL_GRAY, linewidth=0.6, alpha=0.4,
                   label="Individual"),
        ]
        mid_ax.legend(
            handles=_hh, fontsize=4.5,
            loc="upper center", bbox_to_anchor=(0.5, -0.25),
            ncol=3, frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
        )

    # Panel B: shrink annotation text in composite
    for txt in ax_b.texts:
        txt.set_fontsize(max(txt.get_fontsize() * 0.55, 3.0))

    # Panel C: nudge "p = 0.05" annotation down
    for txt in ax_c.texts:
        if "0.05" in txt.get_text():
            x, y = txt.get_position()
            txt.set_position((x, y - 2))

    # Panel I (heatmap): replace below-legend with y-axis group labels
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.colors import to_rgba
    leg_i = ax_i.get_legend()
    if leg_i:
        leg_i.remove()
    _trans_i   = blended_transform_factory(ax_i.transAxes, ax_i.transData)
    _resp_rgba = to_rgba(COL_RESP)
    fig_c.canvas.draw_idle()
    _n_resp_i  = sum(
        1 for t in ax_i.get_yticklabels()
        if np.allclose(to_rgba(t.get_color()), _resp_rgba, atol=0.02)
    )
    _n_total_i = len(ax_i.get_yticklabels())
    _n_nresp_i = _n_total_i - _n_resp_i
    if _n_resp_i > 0:
        ax_i.text(-0.16, (_n_resp_i - 1) / 2, "Resp.",
                  transform=_trans_i, color=COL_RESP,
                  fontsize=4, fontweight="bold",
                  ha="right", va="center", rotation=90, clip_on=False)
    if _n_nresp_i > 0:
        ax_i.text(-0.16, _n_resp_i + (_n_nresp_i - 1) / 2, "Non-resp.",
                  transform=_trans_i, color=COL_NRESP,
                  fontsize=4, fontweight="bold",
                  ha="right", va="center", rotation=90, clip_on=False)

    # Panel I colorbar: shrink to prevent overlap
    for _child_ax in fig_c.get_axes():
        if _child_ax.get_ylabel() == "Score Δ":
            _child_ax.set_ylabel("Score Δ", fontsize=3.5, labelpad=1)
            _child_ax.tick_params(labelsize=3, pad=1)
            break

    # Cap all font sizes to composite maximum
    _cap_fontsize(fig_c, _MAX_FONT_COMPOSITE)

    # Bold panel labels (after cap so they stay prominent)
    _lbl_fs = 7
    _lbl_pos = {
        "C": (-0.55, 1.22),   # more left, and up
        "D": (-0.15, 1.22),   # up
        "E": (-0.28, 1.22),   # left and up
        "F": (-0.15, 1.22),   # up
        "G": (-0.15, 1.22),   # up
    }
    _lbl_default = (-0.15, 1.12)
    for ax, lbl in [
        (ax_a, "A"), (ax_b, "B"), (ax_c, "C"),
        (ax_d, "D"), (ax_e, "E"), (ax_f, "F"),
        (ax_g_comp, "G"), (ax_i, "I"), (ax_j, "J"), (ax_k, "K"),
    ]:
        _lx, _ly = _lbl_pos.get(lbl, _lbl_default)
        ax.text(_lx, _ly, lbl, transform=ax.transAxes,
                fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    # H label: placed on ax_h_hdr (top of H region) after cap so fontsize
    # matches all other panel letters.  y=1.68 calibrated for gs2 hspace=0.70
    # and inner_top=0.88 so the label sits at the same absolute height as
    # F's label at y=1.22.
    if ax_h_hdr is not None:
        ax_h_hdr.text(
            -0.15, 1.68, "H",
            transform=ax_h_hdr.transAxes,
            fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
            clip_on=False,
        )
    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, SUPP_OUTPUT, close=False)
    pdf_path = SUPP_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")

    # Cleanup
    del data["adata"]
    del data
    gc.collect()
    print("  Supp Fig 3 complete: 11 individual panels + combined (A–K)\n")


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    generate()
