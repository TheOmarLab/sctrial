"""
Figure 2 — TNBC Immunotherapy Primary Analysis
===============================================

Eight-panel figure mirroring the melanoma Figure 2 structure,
adapted for the Zhang et al. (Cancer Cell 2021) TNBC dataset (GSE169246).

Dataset: PTX ± atezolizumab (anti-PD-L1), 12 paired tumor biopsies (6 per arm).
Reference: Zhang Y et al. Single-cell analyses reveal key immune cell subsets
associated with response to PD-L1 blockade in triple-negative breast cancer.
Cancer Cell 2021 Dec 13;39(12):1578-1593. PMID: 34653365.

Panels
------
A : Paired-participant verification (cells per participant × visit).
B : Coefficient comparison: cell-level vs participant-level aggregation.
C : P-value comparison: -log10 scale, illustrating inflation at cell level.
D : Forest plot of DiD effects across all gene signatures.
E : Small-multiple interaction plots for the top 6 signatures.
F : Per-participant change heatmap across signatures.
G : Signature changes by arm and response (four-group bar chart).
H : Cohen's d effect sizes (anti-PDL1+Chemo - Chemo) on delta scores.
I : Response-stratified second-order DiD forest plot (DID₂ = DID_R − DID_NR).
J : Bar plot of mean delta score (post - pre) by arm.
K : Within-arm response DID forest plot — Chemo arm.
L : Within-arm response DID forest plot — anti-PDL1+Chemo arm.

"""

from __future__ import annotations

import gc
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
import sctrial as st
from sctrial import TrialDesign, did_table
from sctrial.stats.effect_size import cohens_d_from_did

warnings.filterwarnings("ignore")

# ── Try to import from _shared; fall back to local definitions ────────────
# Once _shared.py has been updated with TNBC helpers, replace the
# try/except block below with a direct import like the melanoma script uses:
#
#   from .._shared import (
#       COLORS, MAIN_OUTPUT, apply_style, despine, save_panel,
#       score_signatures, sig_display, verify_paired_participants,
#       get_tnbc_zhang, TNBC_DESIGN, TNBC_GENE_SIGNATURES,
#   )

try:
    from .._shared import (  # type: ignore[import]
        COLORS,
        MAIN_OUTPUT,
        apply_style,
        despine,
        get_tnbc_zhang,
        save_panel,
        score_signatures,
        sig_display,
        verify_paired_participants,
    )
    _SHARED_AVAILABLE = True
except ImportError:
    _SHARED_AVAILABLE = False

    # ── Local fallbacks (remove once _shared.py is updated) ──────────────

    COLORS = {
        "treated":   "#4878CF",   # blue  -- anti-PDL1+Chemo
        "control":   "#D65F5F",   # red   -- Chemo only
        "gray":      "#888888",
        "highlight": "#E8A838",
    }


    def apply_style() -> None:
        """Minimal publication style — replace with _shared version."""
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor":   "white",
            "font.family":      "sans-serif",
        })

    def despine(ax: plt.Axes) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def save_panel(
        fig: plt.Figure,
        name: str,
        figure_name: str,
        output_dir: Path,
        *,
        close: bool = True,
    ) -> None:
        panel_dir = output_dir / f"{figure_name}_panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        path = panel_dir / f"{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        if close:
            plt.close(fig)
        print(f"  Saved {path.name}")

    def sig_display(col: str) -> str:
        return col.replace("sig_", "")

    def verify_paired_participants(
        obs: pd.DataFrame,
        visit_col: str,
        visits: tuple[str, str],
        participant_col: str,
    ) -> dict:
        counts = obs.groupby(participant_col)[visit_col].nunique()
        n_paired = int((counts == 2).sum())
        n_total  = int(len(counts))
        return {"n_paired": n_paired, "n_total": n_total}

    def score_signatures(
        adata,
        gene_sets: dict,
        layer: str | None = None,
        min_genes: int = 3,
        prefix: str = "sig_",
    ) -> tuple:
        """Score gene sets using st.score_gene_sets (zmean method)."""
        st.score_gene_sets(
            adata,
            gene_sets=gene_sets,
            method="zmean",
            layer=layer,
            min_genes=min_genes,
            prefix=prefix,
            overwrite=True,
        )
        sig_cols = [f"{prefix}{name}" for name in gene_sets
                    if f"{prefix}{name}" in adata.obs.columns]
        return adata, sig_cols


FIGURE_NAME = "Figure2_tnbc_analysis"

# ── Design & constants ────────────────────────────────────────────────────
VISITS: tuple[str, str] = ("Pre", "Post")
SEED = 42

DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="arm",
    arm_treated="anti-PDL1+Chemo",
    arm_control="Chemo",
    celltype_col="cell_type",
)

COL_TREAT = COLORS["treated"]
COL_CTRL  = COLORS["control"]
COL_GRAY  = COLORS["gray"]


# ── Data preparation ──────────────────────────────────────────────────────
def _prepare_data() -> dict:
    """Load TNBC h5ad, score signatures, run DiD, compute delta scores."""
    adata = get_tnbc_zhang()
    if "log1p_norm" not in adata.layers:
        raise RuntimeError("log1p_norm layer missing from TNBC dataset.")

    adata, sig_cols = score_signatures(adata, layer="log1p_norm")
    print(f"  Scored {len(sig_cols)} signatures")

    # Paired-participant verification
    pair_info = verify_paired_participants(
        adata.obs,
        visit_col=DESIGN.visit_col,
        visits=VISITS,
        participant_col=DESIGN.participant_col,
    )

    # DiD at cell level (for pseudoreplication panels A-C)
    print("Running DiD (cell-level, for pseudoreplication demonstration)...")
    res_cell = did_table(
        adata,
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer="log1p_norm",
        standardize=True,
        aggregate="cell",
    )

    # DiD with bootstrap CIs (primary analysis)
    print("Running DiD (participant-level, primary analysis)...")
    res_part = did_table(
        adata,
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer="log1p_norm",
        standardize=True,
        aggregate="participant_visit",
        use_bootstrap=True,
        n_boot=999,
        seed=SEED,
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
    pre = pb[pb[DESIGN.visit_col] == VISITS[0]].set_index(DESIGN.participant_col)
    post = pb[pb[DESIGN.visit_col] == VISITS[1]].set_index(DESIGN.participant_col)
    delta = post[sig_cols].subtract(pre[sig_cols])
    delta[DESIGN.arm_col] = pre[DESIGN.arm_col]
    delta = delta.reset_index()

    n_t = (delta[DESIGN.arm_col] == DESIGN.arm_treated).sum()
    n_c = (delta[DESIGN.arm_col] == DESIGN.arm_control).sum()
    print(
        f"  Paired participants: {len(delta)} "
        f"({DESIGN.arm_treated}={n_t}, {DESIGN.arm_control}={n_c})"
    )

    # Add clinical response label to delta (for second-order DID panels I/J/K)
    response_map = (
        adata.obs
        .groupby(DESIGN.participant_col, observed=True)["response"]
        .first()
    )
    delta["response"] = delta[DESIGN.participant_col].map(response_map)

    return {
        "adata":    adata,
        "sig_cols": sig_cols,
        "res_cell": res_cell,
        "pair_info": pair_info,
        "res_part": res_part,
        "pb":       pb,
        "delta":    delta,
    }

def _pseudobulk(adata, sig_col: str) -> pd.DataFrame:
    cols = [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col, sig_col]
    return (
        adata.obs[cols]
        .groupby(
            [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col],
            observed=True,
        )[sig_col]
        .mean()
        .reset_index()
    )


def _pseudobulk_all(adata, sig_cols: list[str]) -> pd.DataFrame:
    cols = [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col] + sig_cols
    return (
        adata.obs[cols]
        .groupby(
            [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col],
            observed=True,
        )[sig_cols]
        .mean()
        .reset_index()
    )


# ── Panel A: Paired-participant verification ──────────────────────────────

def _panel_a(ax: plt.Axes, data: dict) -> None:
    adata    = data["adata"]
    pair_info = data["pair_info"]

    counts = (
        adata.obs
        .groupby(
            [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col],
            observed=True,
        )
        .size()
        .reset_index(name="n_cells")
    )
    counts[DESIGN.visit_col] = pd.Categorical(
        counts[DESIGN.visit_col], ["Pre", "Post"], ordered=True
    )
    counts = counts.sort_values(
        [DESIGN.arm_col, DESIGN.participant_col, DESIGN.visit_col]
    )

    participants = counts[DESIGN.participant_col].unique()
    pid_order  = {pid: i for i, pid in enumerate(participants)}
    bar_width  = 0.35

    for _, row in counts.iterrows():
        x_base = pid_order[row[DESIGN.participant_col]]
        offset = -bar_width / 2 if row[DESIGN.visit_col] == "Pre" else bar_width / 2
        color  = COL_TREAT if row[DESIGN.arm_col] == DESIGN.arm_treated else COL_CTRL
        alpha  = 0.6 if row[DESIGN.visit_col] == "Pre" else 1.0
        ax.bar(
            x_base + offset, row["n_cells"],
            width=bar_width, color=color, alpha=alpha,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(range(len(participants)))
    ax.set_xticklabels(
        [f"P{i + 1}" for i in range(len(participants))],
        rotation=90, ha="center", fontsize=7,
    )

    ax.set_xlabel("Participant", fontsize=9)
    ax.set_ylabel("Number of cells", fontsize=9)
    ax.set_title("Paired Participants: Cells per Visit",
                 fontsize=11, fontweight="bold")

    # FIX: uses verify_paired_participants output like melanoma Panel A
    ax.text(
        0.055, 0.52,
        f"{pair_info['n_paired']}/{pair_info['n_total']} participants paired",
        transform=ax.transAxes, fontsize=4, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COL_GRAY, alpha=0.8),
    )

    handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COL_TREAT, markersize=3,
               label=DESIGN.arm_treated),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COL_CTRL, markersize=3,
               label=DESIGN.arm_control),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COL_GRAY, markersize=3, alpha=0.6,
               label="Pre"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COL_GRAY, markersize=3,
               label="Post"),
    ]
    ax.legend(
        handles=handles, fontsize=5,
        loc="upper center", bbox_to_anchor=(0.5, -0.28),
        ncol=4, frameon=True, framealpha=0.9,
    )
    despine(ax)


# ── Panel B: Beta comparison ──────────────────────────────────────────────

def _panel_b(ax: plt.Axes, data: dict) -> None:
    rc = data["res_cell"].set_index("feature")
    rp = data["res_part"].set_index("feature")
    common = rc.index.intersection(rp.index)

    beta_cell = rc.loc[common, "beta_DiD"].values
    beta_part = rp.loc[common, "beta_DiD"].values
    colors    = [COL_TREAT if b > 0 else COL_CTRL for b in beta_part]

    ax.scatter(beta_cell, beta_part, c=colors, s=30,
               edgecolors="white", linewidths=0.5, zorder=3)

    lim_lo = min(beta_cell.min(), beta_part.min()) * 1.2
    lim_hi = max(beta_cell.max(), beta_part.max()) * 1.2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--",
            color=COL_GRAY, lw=1, zorder=1)
    ax.axhline(0, color=COL_GRAY, lw=0.5, ls=":", zorder=0)
    ax.axvline(0, color=COL_GRAY, lw=0.5, ls=":", zorder=0)

    x_range = max(beta_cell) - min(beta_cell)
    y_range = max(beta_part) - min(beta_part)
    dx = x_range * 0.04
    dy = y_range * 0.04

    # Explicit positions for each signature to avoid overlap
    # Format: "substring of label": (x_offset, y_offset, ha)
    _label_offsets = {
        "antigen": (-dx, -dy * 3.5, "right"),
        "cytotoxic": (-dx, dy * 2, "right"),
        "memory": (-dx * 2, dy, "right"),
        "immune exh": (-dx, dy * 3, "right"),
        "exhaustion": (-dx * 2, dy * 0.5, "right"),
        "interferon": (-dx * 3, -dy, "right"),
        "cell prolif": (-dx * 2, dy, "right"),
        "oxidative": (dx, -dy, "left"),
        "inflammatory": (dx, dy * 2, "left"),
        "regulatory": (dx * 2, dy, "left"),
        "apoptosis": (dx, -dy * 2, "left"),
        "t cell activ": (dx, dy * 3, "left"),
        "nk cell": (dx * 2, -dy, "left"),
    }

    for feat, xv, yv in zip(common, beta_cell, beta_part):
        label = sig_display(feat)
        ll = label.lower()
        ox, oy, ha = dx, 0, "left"  # default
        for key, (ox_k, oy_k, ha_k) in _label_offsets.items():
            if key in ll:
                ox, oy, ha = ox_k, oy_k, ha_k
                break
        ax.annotate(
            label, (xv, yv),
            xytext=(xv + ox, yv + oy),
            fontsize=4, alpha=0.85,
            ha=ha, va="center",
           # arrowprops=dict(arrowstyle="-", color=COL_GRAY,
            #                lw=0.4, alpha=0.5),
        )
    # FIX v1 Issue 6: Spearman rho not Pearson r
    rho, p = stats.spearmanr(beta_cell, beta_part)
    ax.text(
        0.05, 0.95, f"ρ = {rho:.2f}, p = {p:.1e}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COL_GRAY, alpha=0.8),
    )

    ax.set_xlabel(r'$\beta_{DID}$ (cell-level)', fontsize=9)
    ax.set_ylabel(r'$\beta_{DID}$ (participant-level)', fontsize=9)
    ax.set_title("Effect Size: Cell vs Participant Aggregation",
                 fontsize=11, fontweight="bold")

    handles = [
        mpatches.Patch(facecolor=COL_TREAT, label="Positive effect"),
        mpatches.Patch(facecolor=COL_CTRL,  label="Negative effect"),
    ]
    ax.legend(handles=handles, fontsize=5, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel C: P-value inflation ────────────────────────────────────────────

def _panel_c(ax: plt.Axes, data: dict) -> None:
    rc = data["res_cell"].set_index("feature")
    rp = data["res_part"].set_index("feature")
    common = rc.index.intersection(rp.index)

    # FIX v1 Issue 5: use p_DiD_perm for participant-level bars
    perm_p = rp["p_DiD_perm"] if "p_DiD_perm" in rp.columns else rp["p_DiD"]

    df = pd.DataFrame({
        "feature":     common,
        "nlog10_cell": -np.log10(rc.loc[common, "p_DiD"].clip(lower=1e-300).values),
        "nlog10_part": -np.log10(perm_p.loc[common].clip(lower=1e-300).values),
        "display":     [sig_display(f) for f in common],
    }).sort_values("nlog10_cell", ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(df))
    bar_h = 0.35

    ax.barh(y_pos - bar_h / 2, df["nlog10_cell"], height=bar_h,
            color=COLORS["highlight"], alpha=0.8, label="Cell-level", zorder=2)
    ax.barh(y_pos + bar_h / 2, df["nlog10_part"], height=bar_h,
            color=COL_TREAT, alpha=0.8, label="Participant-level",
            zorder=2)

    thresh = -np.log10(0.05)
    ax.axvline(thresh, color=COL_GRAY, ls="--", lw=1, zorder=1)
    ax.text(thresh + 0.05, len(df) - 0.5, "p = 0.05",
            fontsize=7, va="bottom", color=COL_GRAY)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"], fontsize=5)
    ax.set_xlabel(r'$-\log_{10}(p)$', fontsize=9)
    ax.set_title("P-value Inflation: Cell vs Participant Level",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel D: Forest plot ──────────────────────────────────────────────────
def _panel_d(ax: plt.Axes, data: dict) -> None:
    df = data["res_part"].copy()

    rank_p = df["p_DiD_perm"].fillna(df["p_DiD"]) if "p_DiD_perm" in df.columns else df["p_DiD"]
    df = df.assign(_rank_p=rank_p).sort_values("beta_DiD").drop(columns="_rank_p")
    df = df.reset_index(drop=True)
    y_pos = np.arange(len(df))

    # Key CI values by feature name after sorting to guarantee alignment
    res_part_indexed = data["res_part"].set_index("feature")

    analytical_lo = df["beta_DiD"] - 1.96 * df["se_DiD"]
    analytical_hi = df["beta_DiD"] + 1.96 * df["se_DiD"]

    if "ci_lo_boot" in res_part_indexed.columns and "ci_hi_boot" in res_part_indexed.columns:
        ci_lo = df["feature"].map(res_part_indexed["ci_lo_boot"]).fillna(analytical_lo)
        ci_hi = df["feature"].map(res_part_indexed["ci_hi_boot"]).fillna(analytical_hi)
    else:
        ci_lo = analytical_lo
        ci_hi = analytical_hi

    for i, (_, row) in enumerate(df.iterrows()):
        color = COL_TREAT if row["beta_DiD"] > 0 else COL_CTRL
        ax.hlines(y_pos[i], ci_lo.iloc[i], ci_hi.iloc[i],
                  color=color, linewidth=2.0, alpha=1.0, zorder=1)
        ax.scatter(row["beta_DiD"], y_pos[i], color=color, s=30,
                   edgecolors="white", linewidths=0.8, zorder=2)

        fdr_col = "FDR_DiD_perm" if "FDR_DiD_perm" in df.columns else "FDR_DiD"
        if pd.notna(row.get(fdr_col)) and row[fdr_col] < 0.25:
            ax.text(ci_hi.iloc[i] + 0.02, y_pos[i], "*",
                    va="center", fontsize=10, fontweight="bold", color=color)

    ax.axvline(0, color="#333333", lw=0.9, ls="--", zorder=0, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sig_display(f) for f in df["feature"]], fontsize=5)
    ax.set_xlabel("DiD coefficient (β, standardised)", fontsize=5)
    ax.set_title(
        "DiD effects across signatures",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(-0.6, len(df) - 0.4)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_TREAT, markersize=3,
               label=f"{DESIGN.arm_treated} ↑"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_CTRL, markersize=3,
               label=f"{DESIGN.arm_control} ↑"),
    ]
    ax.legend(handles=handles,
              fontsize=5,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.05),
              frameon=True,
              framealpha=0.9,
              borderpad=0.3,
              labelspacing=0.2,
              handlelength=1.0,
              handletextpad=0.3,
              columnspacing=0.8)
    despine(ax)
# ── Panel E: Interaction grid ─────────────────────────────────────────────

def _panel_e(
    fig: plt.Figure,
    gs_parent: gridspec.SubplotSpec,
    data: dict,
    n_sigs: int = 6,
    inner_hspace: float = 0.55,
    inner_wspace: float = 0.35,
) -> list[plt.Axes]:
    adata    = data["adata"]
    res_part = data["res_part"]

    # FIX v1 Issue 5: rank by permutation p-value with fallback
    if "p_DiD_perm" in res_part.columns:
        rank_p = res_part["p_DiD_perm"].fillna(res_part["p_DiD"])
    else:
        rank_p = res_part["p_DiD"]
    top = (
        res_part.assign(_rank_p=rank_p)
        .sort_values("_rank_p")
        .head(n_sigs)
        .drop(columns="_rank_p")
    )

    nrows, ncols = 2, 3
    gs_inner = gs_parent.subgridspec(nrows, ncols,
                                     hspace=inner_hspace, wspace=inner_wspace)
    axes      = []
    x_map     = {VISITS[0]: 0.0, VISITS[1]: 1.0}
    arm_colors = {DESIGN.arm_treated: COL_TREAT, DESIGN.arm_control: COL_CTRL}

    for idx, (_, row) in enumerate(top.iterrows()):
        r, c = divmod(idx, ncols)
        ax   = fig.add_subplot(gs_inner[r, c])
        axes.append(ax)

        sig_col = row["feature"]
        pb      = _pseudobulk(adata, sig_col)

        for arm, arm_df in pb.groupby(DESIGN.arm_col, observed=True):
            color = arm_colors.get(arm, COL_GRAY)
            for _, pid_df in arm_df.groupby(DESIGN.participant_col, observed=True):
                pid_df = pid_df.sort_values(
                    DESIGN.visit_col, key=lambda s: s.map(x_map)
                )
                if len(pid_df) == 2:
                    ax.plot(pid_df[DESIGN.visit_col].map(x_map), pid_df[sig_col],
                            color=color, alpha=0.22, linewidth=0.8, zorder=1)

        gm = (
            pb.groupby([DESIGN.arm_col, DESIGN.visit_col], observed=True)
            [sig_col].mean().reset_index()
        )
        for arm, gdf in gm.groupby(DESIGN.arm_col, observed=True):
            color = arm_colors.get(arm, COL_GRAY)
            gdf   = gdf.sort_values(DESIGN.visit_col, key=lambda s: s.map(x_map))
            ax.plot(gdf[DESIGN.visit_col].map(x_map), gdf[sig_col],
                    color=color, linewidth=2.0, marker="o", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(VISITS, fontsize=8)
        ax.set_xlim(-0.35, 1.35)
        ax.tick_params(axis="y", labelsize=7)

        # FIX v1 Issue 5: use permutation p-value for title
        p_val = row.get("p_DiD_perm", np.nan)
        if pd.isna(p_val):
            p_val = row["p_DiD"]
        p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else f"p = {p_val:.1e}"
        ax.set_title(f"{sig_display(sig_col)}\n{p_str}",
                     fontsize=9, fontweight="bold", pad=5)
        despine(ax)

    handles = [
        Line2D([0], [0], color=COL_TREAT, lw=4, marker="o", markersize=4,
               markeredgecolor="white", label=DESIGN.arm_treated),
        Line2D([0], [0], color=COL_CTRL, lw=4, marker="o", markersize=4,
               markeredgecolor="white", label=DESIGN.arm_control),
        Line2D([0], [0], color=COL_GRAY, lw=0.8, alpha=0.4, label="Individual"),
    ]
    axes[-1].legend(handles=handles, fontsize=5,
                    loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    ncol=3, frameon=True, framealpha=0.95)
    return axes


# ── Panel F: Per-participant change heatmap ───────────────────────────────
def _panel_f(ax: plt.Axes, data: dict) -> None:
    adata    = data["adata"]
    sig_cols = data["sig_cols"]
    res_part = data["res_part"]

    pb = _pseudobulk_all(adata, sig_cols)

    pre_mask  = pb[DESIGN.visit_col] == VISITS[0]
    post_mask = pb[DESIGN.visit_col] == VISITS[1]

    pre_num = (
        pb.loc[pre_mask]
        .groupby(DESIGN.participant_col, observed=True)[sig_cols].mean()
    )
    post_num = (
        pb.loc[post_mask]
        .groupby(DESIGN.participant_col, observed=True)[sig_cols].mean()
    )
    pre_arm = (
        pb.loc[pre_mask]
        .groupby(DESIGN.participant_col, observed=True)[DESIGN.arm_col].first()
    )
    pre = pre_num.join(pre_arm)

    common_pids = sorted(set(pre.index) & set(post_num.index))
    if not common_pids:
        ax.text(0.5, 0.5, "No paired participants", ha="center",
                va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    delta_mat = pd.DataFrame(
        post_num.loc[common_pids, sig_cols].values
        - pre.loc[common_pids, sig_cols].values,
        index=common_pids,
        columns=[sig_display(c) for c in sig_cols],
    )
    arms = pre.loc[common_pids, DESIGN.arm_col]

    sort_df = pd.DataFrame({
        "arm_order":  arms.map({DESIGN.arm_treated: 0, DESIGN.arm_control: 1}).values,
        "mean_delta": -delta_mat.mean(axis=1).values,
    }, index=common_pids)
    ordered = sort_df.sort_values(["arm_order", "mean_delta"]).index.tolist()
    delta_mat = delta_mat.loc[ordered]
    arms      = arms.loc[ordered]

    col_order = [sig_display(f) for f in
                 res_part.sort_values("beta_DiD")["feature"]]
    col_order = [c for c in col_order if c in delta_mat.columns]
    delta_mat = delta_mat[col_order]

    vmax = max(np.nanpercentile(np.abs(delta_mat.values), 95), 0.1)
    im   = ax.imshow(delta_mat.values, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, interpolation="nearest")

    sidebar_w = 0.35
    for i, pid in enumerate(ordered):
        color = COL_TREAT if arms.loc[pid] == DESIGN.arm_treated else COL_CTRL
        ax.add_patch(plt.Rectangle(
            (-sidebar_w - 0.35, i - 0.5), sidebar_w, 1.0,
            color=color, clip_on=False,
        ))

    n_treated = sum(1 for p in ordered if arms.loc[p] == DESIGN.arm_treated)
    if 0 < n_treated < len(ordered):
        ax.axhline(n_treated - 0.5, color="white", linewidth=2.5, zorder=5)

    # Build same P1-P12 mapping as panel A
    counts_a = (
        adata.obs
        .groupby(
            [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col],
            observed=True,
        )
        .size()
        .reset_index(name="n_cells")
        .sort_values([DESIGN.arm_col, DESIGN.participant_col, DESIGN.visit_col])
    )
    panel_a_participants = counts_a[DESIGN.participant_col].unique()
    pid_to_label = {pid: f"P{i+1}" for i, pid in enumerate(panel_a_participants)}

    ax.set_xticks(np.arange(len(delta_mat.columns)))
    ax.set_xticklabels(delta_mat.columns, fontsize=4, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(
        [pid_to_label.get(p, str(p)) + "   " for p in ordered],
        fontsize=6, fontfamily="monospace",
    )
    ax.yaxis.set_tick_params(pad=2)

    # Build responder map: one response value per participant ("R" / "NR")
    if "response" in adata.obs.columns:
        responder_map = (
            adata.obs
            .groupby(DESIGN.participant_col, observed=True)["response"]
            .first()
        )
    else:
        responder_map = pd.Series(dtype=str)

    from matplotlib.transforms import blended_transform_factory
    star_trans = blended_transform_factory(ax.transAxes, ax.transData)

    # Stars: use scatter marker="*" — avoids Unicode tofu from missing font glyphs
    star_xs = []
    star_ys = []
    for i, pid in enumerate(ordered):
        if responder_map.get(pid) == "R":
            star_xs.append(-0.065)
            star_ys.append(i)
    if star_xs:
        star_handle = ax.scatter(
            star_xs, star_ys,
            marker="*", s=40, color="black",
            transform=star_trans,
            clip_on=False, zorder=5,
        )
        ax.legend(
            handles=[star_handle],
            labels=["Responder"],
            fontsize=6,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            frameon=False,
            handletextpad=0.3, borderpad=0.4,
            markerscale=0.6,
        )

    # Tick label colors: separate loop, guarded against pre-render empty list
    for i, tick in enumerate(ax.get_yticklabels()):
        if i < len(ordered):
            tick.set_color(COL_TREAT if arms.iloc[i] == DESIGN.arm_treated else COL_CTRL)

    ax.set_title("Per-participant score change (Post - Pre)",
                 fontsize=11, fontweight="bold", pad=2)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score Δ", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    n_treated = sum(1 for p in ordered if arms.loc[p] == DESIGN.arm_treated)
    n_control = len(ordered) - n_treated

    if n_treated > 0:
        ax.text(
            -0.22, (n_treated - 1) / 2,
            "Chemo",
            transform=trans,
            color=COL_CTRL,
            fontsize=4, fontweight="bold",
            ha="right", va="center",
            rotation=90, clip_on=False,
        )

    if n_control > 0:
        ax.text(
            -0.22, n_treated + (n_control - 1) / 2,
            "anti-PDL1+\nChemo",
            transform=trans,
            color=COL_TREAT,
            fontsize=4, fontweight="bold",
            ha="right", va="center",
            rotation=90, clip_on=False,
        )
# ── Panel J: Mean delta by arm ────────────────────────────────────────────

def _panel_j(ax: plt.Axes, data: dict) -> None:
    delta    = data["delta"]
    sig_cols = data["sig_cols"]
    arm      = DESIGN.arm_col

    treat_mask = delta[arm] == DESIGN.arm_treated
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
            color=COL_TREAT, alpha=0.85,
            xerr=sems_t[order].values, capsize=2, ecolor=COL_GRAY,
            error_kw={"linewidth": 0.8}, label=DESIGN.arm_treated)
    ax.barh(y_pos - bar_h / 2, means_c[order].values, height=bar_h,
            color=COL_CTRL, alpha=0.85,
            xerr=sems_c[order].values, capsize=2, ecolor=COL_GRAY,
            error_kw={"linewidth": 0.8}, label=DESIGN.arm_control)

    ax.axvline(0, ls=":", color=COL_GRAY, lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=5)
    ax.set_xlabel("Mean Δ score (Post - Pre)", fontsize=5)
    ax.set_title("Signature Changes by Arm", fontsize=11, fontweight="bold")
    ax.legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel H: Cohen's d / Hedge's g ───────────────────────────────────────

def _panel_h(ax: plt.Axes, data: dict) -> None:
    delta    = data["delta"]
    sig_cols = data["sig_cols"]
    arm      = DESIGN.arm_col

    # FIX v1 Issue 7: use st.cohens_d_from_did (exact gamma correction)
    records = []
    for col in sig_cols:
        x = delta.loc[delta[arm] == DESIGN.arm_treated, col].dropna().values
        y = delta.loc[delta[arm] == DESIGN.arm_control, col].dropna().values
        d = cohens_d_from_did(x, y)   # Hedge's g via exact gamma correction
        records.append({"feature": col, "display": sig_display(col), "d": d})

    df     = pd.DataFrame(records).dropna(subset=["d"]).sort_values("d")
    y_pos  = np.arange(len(df))
    colors = [COL_TREAT if v > 0 else COL_CTRL for v in df["d"].values]

    ax.hlines(y_pos, 0, df["d"].values, colors=colors, lw=2, zorder=2)
    ax.scatter(df["d"].values, y_pos, c=colors, s=35,
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, ls=":", color=COL_GRAY, lw=0.8, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel(
        f"Hedge's g ({DESIGN.arm_treated} - {DESIGN.arm_control})",
        fontsize=9,
    )
    ax.set_title("Effect Size of Arm Separation", fontsize=11,
                 fontweight="bold")
    despine(ax)


# ── Second-order DID helpers & panels I / J / K ───────────────────────────

_RESP_COL = "response"
_R_VAL    = "R"
_NR_VAL   = "NR"

# 4-group colours: (arm, response)
_COL_TR  = COLORS["treated"]      # anti-PDL1+Chemo + Responder
_COL_TNR = "#9FB8DB"               # anti-PDL1+Chemo + Non-responder (muted blue)
_COL_CR  = COLORS["control"]      # Chemo + Responder
_COL_CNR = "#F0BF8A"               # Chemo + Non-responder (muted orange)


def _compute_response_did(
    delta: pd.DataFrame,
    sig_cols: list[str],
    n_boot: int = 999,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compute second-order DID (DID_R − DID_NR) with stratified bootstrap CIs.

    Parameters
    ----------
    delta
        Per-participant delta frame with columns: participant_id, arm, response,
        and one column per signature.
    sig_cols
        Signature columns to iterate over.
    n_boot, seed
        Bootstrap parameters.

    Returns
    -------
    DataFrame with one row per signature and columns:
        feature, DID_R, DID_NR, DID2, ci_lo, ci_hi, n_TR, n_CR, n_TNR, n_CNR.
    """
    arm  = DESIGN.arm_col
    resp = _RESP_COL

    mask_TR  = (delta[arm] == DESIGN.arm_treated) & (delta[resp] == _R_VAL)
    mask_TNR = (delta[arm] == DESIGN.arm_treated) & (delta[resp] == _NR_VAL)
    mask_CR  = (delta[arm] == DESIGN.arm_control) & (delta[resp] == _R_VAL)
    mask_CNR = (delta[arm] == DESIGN.arm_control) & (delta[resp] == _NR_VAL)

    rows = []
    rng  = np.random.default_rng(seed)

    for col in sig_cols:
        d_TR  = delta.loc[mask_TR,  col].dropna().values
        d_TNR = delta.loc[mask_TNR, col].dropna().values
        d_CR  = delta.loc[mask_CR,  col].dropna().values
        d_CNR = delta.loc[mask_CNR, col].dropna().values

        DID_R  = d_TR.mean()  - d_CR.mean()  if len(d_TR)  > 0 and len(d_CR)  > 0 else np.nan
        DID_NR = d_TNR.mean() - d_CNR.mean() if len(d_TNR) > 0 and len(d_CNR) > 0 else np.nan
        DID2   = DID_R - DID_NR if not (np.isnan(DID_R) or np.isnan(DID_NR)) else np.nan

        # Stratified bootstrap — resample each stratum independently
        boot_vals = []
        for _ in range(n_boot):
            b_TR  = rng.choice(d_TR,  len(d_TR),  replace=True) if len(d_TR)  > 0 else np.array([np.nan])
            b_TNR = rng.choice(d_TNR, len(d_TNR), replace=True) if len(d_TNR) > 0 else np.array([np.nan])
            b_CR  = rng.choice(d_CR,  len(d_CR),  replace=True) if len(d_CR)  > 0 else np.array([np.nan])
            b_CNR = rng.choice(d_CNR, len(d_CNR), replace=True) if len(d_CNR) > 0 else np.array([np.nan])
            b_DID_R  = b_TR.mean()  - b_CR.mean()
            b_DID_NR = b_TNR.mean() - b_CNR.mean()
            boot_vals.append(b_DID_R - b_DID_NR)

        boot_arr = np.array(boot_vals)
        ci_lo = float(np.nanpercentile(boot_arr, 2.5))
        ci_hi = float(np.nanpercentile(boot_arr, 97.5))

        rows.append({
            "feature": col,
            "DID_R":   DID_R,
            "DID_NR":  DID_NR,
            "DID2":    DID2,
            "ci_lo":   ci_lo,
            "ci_hi":   ci_hi,
            "n_TR":    len(d_TR),
            "n_TNR":   len(d_TNR),
            "n_CR":    len(d_CR),
            "n_CNR":   len(d_CNR),
        })

    return pd.DataFrame(rows)


# ── Panel I: Second-order DID forest plot ─────────────────────────────────

def _panel_i(ax: plt.Axes, data: dict, *, show_note: bool = True) -> None:
    delta    = data["delta"]
    sig_cols = data["sig_cols"]

    df = _compute_response_did(delta, sig_cols).dropna(subset=["DID2"])
    df = df.sort_values("DID2").reset_index(drop=True)
    y_pos = np.arange(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        color = COL_TREAT if row["DID2"] > 0 else COL_CTRL
        ax.hlines(y_pos[i], row["ci_lo"], row["ci_hi"],
                  color=color, linewidth=2.0, alpha=1.0, zorder=1)
        ax.scatter(row["DID2"], y_pos[i], color=color, s=30,
                   edgecolors="white", linewidths=0.8, zorder=2)
        if not (row["ci_lo"] < 0 < row["ci_hi"]):
            ax.text(row["ci_hi"] + 0.02, y_pos[i], "*",
                    va="center", fontsize=10, fontweight="bold", color=color)

    ax.axvline(0, color="#333333", lw=0.9, ls="--", zorder=0, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sig_display(f) for f in df["feature"]], fontsize=5)
    ax.set_xlabel(
        r"DID$_2$ = DID$_{\mathrm{R}}$ $-$ DID$_{\mathrm{NR}}$ (standardised $\Delta$)",
        fontsize=5,
    )
    ax.set_title(
        "Response-stratified DID\n(second-order DiD)", fontsize=11, fontweight="bold",
    )
    ax.set_ylim(-0.6, len(df) - 0.4)

    # Sample-size warning (standalone only)
    if show_note:
        n_cnr = int(df["n_CNR"].iloc[0]) if len(df) > 0 else 1
        ax.text(
            0.98, 0.02,
            f"Chemo+NR: n={n_cnr} participant",
            transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor=COL_GRAY, alpha=0.9),
        )

    from matplotlib.lines import Line2D as _L2D
    handles = [
        _L2D([0], [0], marker="o", color="w",
             markerfacecolor=COL_TREAT, markersize=4,
             label="Responder ↑"),
        _L2D([0], [0], marker="o", color="w",
             markerfacecolor=COL_CTRL, markersize=4,
             label="Non-responder ↑"),
    ]
    ax.legend(handles=handles, fontsize=5,
              loc="upper left",
              frameon=True, framealpha=0.9,
              borderpad=0.3, labelspacing=0.2, handlelength=1.0)
    despine(ax)


# ── Panel G: Mean delta by arm × response ─────────────────────────────────

def _panel_g(ax: plt.Axes, data: dict) -> None:
    delta    = data["delta"]
    sig_cols = data["sig_cols"]

    arm  = DESIGN.arm_col
    resp = _RESP_COL

    mask_TR  = (delta[arm] == DESIGN.arm_treated) & (delta[resp] == _R_VAL)
    mask_TNR = (delta[arm] == DESIGN.arm_treated) & (delta[resp] == _NR_VAL)
    mask_CR  = (delta[arm] == DESIGN.arm_control) & (delta[resp] == _R_VAL)
    mask_CNR = (delta[arm] == DESIGN.arm_control) & (delta[resp] == _NR_VAL)

    n_TR  = int(mask_TR.sum())
    n_TNR = int(mask_TNR.sum())
    n_CR  = int(mask_CR.sum())
    n_CNR = int(mask_CNR.sum())

    means_TR  = delta.loc[mask_TR,  sig_cols].mean()
    means_TNR = delta.loc[mask_TNR, sig_cols].mean()
    means_CR  = delta.loc[mask_CR,  sig_cols].mean()
    means_CNR = delta.loc[mask_CNR, sig_cols].mean()

    sems_TR  = delta.loc[mask_TR,  sig_cols].sem()
    sems_TNR = delta.loc[mask_TNR, sig_cols].sem()
    sems_CR  = delta.loc[mask_CR,  sig_cols].sem()
    sems_CNR = delta.loc[mask_CNR, sig_cols].sem()

    order = means_TR.sort_values().index
    display_names = [sig_display(s) for s in order]
    y_pos  = np.arange(len(order))
    bar_h  = 0.2
    offsets = [1.5 * bar_h, 0.5 * bar_h, -0.5 * bar_h, -1.5 * bar_h]

    groups = [
        (means_TR,  sems_TR,  _COL_TR,  f"anti-PDL1+Chemo, R (n={n_TR})"),
        (means_TNR, sems_TNR, _COL_TNR, f"anti-PDL1+Chemo, NR (n={n_TNR})"),
        (means_CR,  sems_CR,  _COL_CR,  f"Chemo, R (n={n_CR})"),
        (means_CNR, sems_CNR, _COL_CNR, f"Chemo, NR (n={n_CNR})"),
    ]

    for (means, sems, color, label), offset in zip(groups, offsets):
        ax.barh(
            y_pos + offset, means[order].values, height=bar_h,
            color=color, alpha=0.90,
            xerr=sems[order].values, capsize=0.5, ecolor=COL_GRAY,
            error_kw={"linewidth": 0.4}, label=label,
        )

    ax.axvline(0, ls=":", color=COL_GRAY, lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=5)
    ax.set_xlabel(r"Mean $\Delta$ score (Post $-$ Pre)", fontsize=5)
    ax.set_title("Signature changes by arm and response",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=True, framealpha=0.9)
    despine(ax)


# ── Within-arm response DID forest plots (panels K & L) ───────────────────

def _within_arm_response_forest(
    ax: plt.Axes,
    data: dict,
    arm_label: str,
    arm_color: str,
) -> None:
    """Forest plot of DID_response = (Post-Pre)_R − (Post-Pre)_NR within one arm.

    Parameters
    ----------
    arm_label  : one of DESIGN.arm_treated / DESIGN.arm_control
    arm_color  : colour for positive-effect bars
    """
    delta    = data["delta"]
    sig_cols = data["sig_cols"]

    arm  = DESIGN.arm_col
    resp = _RESP_COL

    mask_R  = (delta[arm] == arm_label) & (delta[resp] == _R_VAL)
    mask_NR = (delta[arm] == arm_label) & (delta[resp] == _NR_VAL)
    n_R  = int(mask_R.sum())
    n_NR = int(mask_NR.sum())

    rng = np.random.default_rng(SEED)
    records = []
    for col in sig_cols:
        d_R  = delta.loc[mask_R,  col].dropna().values
        d_NR = delta.loc[mask_NR, col].dropna().values
        if len(d_R) == 0 or len(d_NR) == 0:
            continue
        point = d_R.mean() - d_NR.mean()

        # Stratified bootstrap CI
        boots = []
        for _ in range(999):
            b_R  = rng.choice(d_R,  len(d_R),  replace=True)
            b_NR = rng.choice(d_NR, len(d_NR), replace=True)
            boots.append(b_R.mean() - b_NR.mean())
        boots = np.array(boots)
        records.append({
            "feature": col,
            "DID":     point,
            "ci_lo":   float(np.percentile(boots, 2.5)),
            "ci_hi":   float(np.percentile(boots, 97.5)),
        })

    df = pd.DataFrame(records).sort_values("DID").reset_index(drop=True)
    y_pos = np.arange(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        color = arm_color if row["DID"] > 0 else COL_GRAY
        ax.hlines(y_pos[i], row["ci_lo"], row["ci_hi"],
                  color=color, linewidth=2.0, alpha=1.0, zorder=1)
        ax.scatter(row["DID"], y_pos[i], color=color, s=30,
                   edgecolors="white", linewidths=0.8, zorder=2)
        if not (row["ci_lo"] < 0 < row["ci_hi"]):
            ax.text(row["ci_hi"] + 0.02, y_pos[i], "*",
                    va="center", fontsize=10, fontweight="bold", color=color)

    ax.axvline(0, color="#333333", lw=0.9, ls="--", zorder=0, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sig_display(f) for f in df["feature"]], fontsize=5)
    ax.set_xlabel(
        r"DID$_{\mathrm{response}}$ = $\Delta$Responders $-$ $\Delta$Non-responders",
        fontsize=5,
    )
    ax.set_title(
        f"Response DID within {arm_label}\n"
        r"(Post$-$Pre)$_R$ $-$ (Post$-$Pre)$_{NR}$",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(-0.6, len(df) - 0.4)

    ax.text(
        0.98, 0.02,
        f"n(R)={n_R}, n(NR)={n_NR}",
        transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor=COL_GRAY, alpha=0.9),
    )

    from matplotlib.lines import Line2D as _L2D
    handles = [
        _L2D([0], [0], marker="o", color="w",
             markerfacecolor=arm_color, markersize=4, label="R > NR"),
        _L2D([0], [0], marker="o", color="w",
             markerfacecolor=COL_GRAY, markersize=4, label="NR > R"),
    ]
    ax.legend(handles=handles, fontsize=5, loc="upper left",
              frameon=True, framealpha=0.9,
              borderpad=0.3, labelspacing=0.2, handlelength=1.0)
    despine(ax)


def _panel_k(ax: plt.Axes, data: dict) -> None:
    """Panel K: DID_response within Chemo arm."""
    _within_arm_response_forest(ax, data, DESIGN.arm_control, COL_CTRL)


def _panel_l(ax: plt.Axes, data: dict) -> None:
    """Panel L: DID_response within anti-PDL1+Chemo arm."""
    _within_arm_response_forest(ax, data, DESIGN.arm_treated, COL_TREAT)


# ── Font helpers ──────────────────────────────────────────────────────────

_BIG_FONT_RC = {
    "font.size":              14,
    "axes.titlesize":         14,
    "axes.titleweight":       "bold",
    "axes.labelsize":         12,
    "xtick.labelsize":        10,
    "ytick.labelsize":        10,
    "legend.fontsize":        10,
    "legend.title_fontsize":  10,
}

_SMALL_RC = {
    "font.size":             5,
    "axes.titlesize":        5.5,
    "axes.labelsize":        5,
    "xtick.labelsize":       4.5,
    "ytick.labelsize":       4.5,
    "legend.fontsize":       4,
    "legend.title_fontsize": 4,
}


@contextmanager
def _big_fonts():
    prev = {k: plt.rcParams[k] for k in _BIG_FONT_RC}
    plt.rcParams.update(_BIG_FONT_RC)
    try:
        yield
    finally:
        plt.rcParams.update(prev)


def _enforce_min_fontsize(fig: plt.Figure, minimum: float = 10) -> None:
    for ax in fig.get_axes():
        for txt in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
            + ax.texts
        ):
            if txt.get_fontsize() < minimum:
                txt.set_fontsize(minimum)
        if ax.get_legend():
            for txt in ax.get_legend().get_texts():
                if txt.get_fontsize() < minimum:
                    txt.set_fontsize(minimum)
    for txt in fig.texts:
        if txt.get_fontsize() < minimum:
            txt.set_fontsize(minimum)


def _cap_fontsize(fig: plt.Figure, maximum: float) -> None:
    for ax in fig.get_axes():
        for txt in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
            + ax.texts
        ):
            if txt.get_fontsize() > maximum:
                txt.set_fontsize(maximum)
        if ax.get_legend():
            for txt in ax.get_legend().get_texts():
                if txt.get_fontsize() > maximum:
                    txt.set_fontsize(maximum)
    for txt in fig.texts:
        if txt.get_fontsize() > maximum:
            txt.set_fontsize(maximum)


# ── Main generation ───────────────────────────────────────────────────────
def generate() -> None:
    print("=" * 60)
    print("Figure 2 -- TNBC Immunotherapy Primary Analysis")
    print("Zhang et al., Cancer Cell 2021 (GSE169246)")
    print("=" * 60)
    data = _prepare_data()

    with _big_fonts():
        # Individual panels A-H
        panel_specs = [
            ("panel_A_paired_verification",      _panel_a,  (11, 6)),
            ("panel_B_beta_comparison",          _panel_b,  (9,  7)),
            ("panel_C_pvalue_inflation",         _panel_c,  (9,  7)),
            ("panel_D_forest",                   _panel_d,  (10, 7)),
            ("panel_H_cohens_d",                 _panel_h,  (8,  6)),
            ("panel_I_response_did_forest",      _panel_i,  (10, 7)),
            ("panel_G_delta_by_arm_response",    _panel_g,  (10, 8)),
            ("panel_J_mean_delta_by_arm",        _panel_j,  (10, 7)),
            ("panel_K_response_did_chemo",        _panel_k,  (10, 7)),
            ("panel_L_response_did_antipdl1",     _panel_l,  (10, 7)),
        ]
        for panel_name, func, size in panel_specs:
            fig, ax = plt.subplots(figsize=size)
            func(ax, data)
            _enforce_min_fontsize(fig)
            fig.tight_layout()
            save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

        # Panel E — needs figure + gridspec
        fig_e = plt.figure(figsize=(14, 9))
        gs_e  = fig_e.add_gridspec(1, 1)[0, 0]
        _panel_e(fig_e, gs_e, data, n_sigs=6)
        _enforce_min_fontsize(fig_e)
        fig_e.tight_layout()
        save_panel(fig_e, "panel_E_interaction_grid", FIGURE_NAME, MAIN_OUTPUT)

        # Panel F — heatmap
        fig_f, ax_f = plt.subplots(figsize=(12, 7))
        _panel_f(ax_f, data)
        _enforce_min_fontsize(fig_f)
        fig_f.tight_layout()
        save_panel(fig_f, "panel_F_heatmap", FIGURE_NAME, MAIN_OUTPUT)

    # ── Combined artboard (180 × 215 mm) ──────────────────────────────────
    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm  = 1.0 / 25.4
    _MAX = 6

    fig_c = plt.figure(figsize=(180 * _mm, 215 * _mm))

    outer = fig_c.add_gridspec(
        4, 1,
        height_ratios=[1, 2.2, 1.8, 1],
        hspace=0.55,
        left=0.10, right=0.95, top=0.97, bottom=0.05,
    )

    # Row 0: A | B
    gs0  = outer[0].subgridspec(1, 2, wspace=0.28, width_ratios=[1, 1.4])
    ax_a = fig_c.add_subplot(gs0[0])
    ax_b = fig_c.add_subplot(gs0[1])

    # Row 1: C (top-left) + D (bottom-left) | E (right, spans both)
    gs1  = outer[1].subgridspec(2, 2, width_ratios=[1, 1.6],
                                 hspace=0.70, wspace=0.40)
    ax_c = fig_c.add_subplot(gs1[0, 0])
    ax_d = fig_c.add_subplot(gs1[1, 0])

    # Row 2: F | G
    gs2  = outer[2].subgridspec(1, 2, wspace=0.55, width_ratios=[1, 1.5])
    ax_f = fig_c.add_subplot(gs2[0])
    ax_g = fig_c.add_subplot(gs2[1])

    # Row 3: H | I
    gs3  = outer[3].subgridspec(1, 2, wspace=0.45)
    ax_h = fig_c.add_subplot(gs3[0])
    ax_i = fig_c.add_subplot(gs3[1])

    # ── Draw all panels first ──────────────────────────────────────────────
    _panel_a(ax_a, data)
    _panel_b(ax_b, data)
    _panel_c(ax_c, data)
    _panel_d(ax_d, data)
    axes_e = _panel_e(fig_c, gs1[:, 1], data, n_sigs=6,
                      inner_hspace=0.45, inner_wspace=0.40)
    _panel_f(ax_f, data)
    _panel_g(ax_g, data)
    _panel_h(ax_h, data)
    _panel_i(ax_i, data, show_note=False)

    # ── Legend overrides (all must come AFTER panels are drawn) ───────────

    # Panels B, C, G — move below-axis legends inside plot
    _inside = {
        ax_b: "lower right",
        ax_c: "lower right",
    }
    for ax_target, loc in _inside.items():
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels  = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_target.legend(
                handles=handles, labels=labels,
                fontsize=5, loc=loc,
                frameon=True, framealpha=0.85,
                handlelength=0.8, handletextpad=0.2,
                borderpad=0.2, labelspacing=0.15,
                columnspacing=0.5,
            )

    # Panel G legend — single row above axes, title pushed up to make room
    leg_g = ax_g.get_legend()
    if leg_g:
        handles_g = leg_g.legend_handles
        labels_g  = [t.get_text() for t in leg_g.get_texts()]
        leg_g.remove()
        ax_g.legend(
            handles=handles_g, labels=labels_g,
            fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 0.98),
            ncol=4, frameon=True, framealpha=0.85,
            handlelength=0.8, handletextpad=0.2,
            borderpad=0.2, labelspacing=0.15, columnspacing=0.5,
        )
    ax_g.set_title(ax_g.get_title(), pad=12,
                   fontsize=ax_g.title.get_fontsize(), fontweight="bold")

    # Panel A legend — nudged right via bbox_to_anchor to avoid overlapping bars
    leg_a = ax_a.get_legend()
    if leg_a:
        handles_a = leg_a.legend_handles
        labels_a  = [t.get_text() for t in leg_a.get_texts()]
        leg_a.remove()
        ax_a.legend(
            handles=handles_a, labels=labels_a,
            fontsize=5,
            loc="upper left",
            bbox_to_anchor=(0.04, 1.0),  # increase 0.08 to nudge further right
            frameon=True, framealpha=0.85,
            handlelength=0.8, handletextpad=0.2,
            borderpad=0.2, labelspacing=0.15,
            columnspacing=0.5,
        )

    # Panel D legend — tighter sizing
    leg_d = ax_d.get_legend()
    if leg_d:
        handles_d = leg_d.legend_handles
        labels_d  = [t.get_text() for t in leg_d.get_texts()]
        leg_d.remove()
        ax_d.legend(
            handles=handles_d, labels=labels_d,
            fontsize=5, loc="lower right",
            frameon=True, framealpha=0.85,
            handlelength=0.8, handletextpad=0.2,
            borderpad=0.2, labelspacing=0.15,
            columnspacing=0.5,
        )

    # Panel E legend
    if axes_e:
        leg_e = axes_e[-1].get_legend()
        if leg_e:
            leg_e.remove()
        mid_ax = axes_e[min(4, len(axes_e) - 1)]
        mid_ax.legend(
            handles=[
                Line2D([0], [0], color=COL_TREAT, lw=1.5, marker="o",
                       markersize=3, markeredgecolor="white",
                       label=DESIGN.arm_treated),
                Line2D([0], [0], color=COL_CTRL, lw=1.5, marker="o",
                       markersize=3, markeredgecolor="white",
                       label=DESIGN.arm_control),
                Line2D([0], [0], color=COL_GRAY, lw=0.6, alpha=0.4,
                       label="Individual"),
            ],
            fontsize=5,
            loc="upper center", bbox_to_anchor=(0.5, -0.25),
            ncol=3, frameon=True, framealpha=0.95,
            handlelength=2.5,  # ← wider line segments
            handletextpad=0.6,  # ← space between line and label
            borderpad=0.2, labelspacing=0.15,
            columnspacing=1.5,  # ← space between columns
        )

    # Panel F — shrink stars, move legend above heatmap, push title up
    for coll in ax_f.collections:
        coll.set_sizes([12])  # smaller stars in composite

    leg_f = ax_f.get_legend()
    if leg_f:
        handles_f = leg_f.legend_handles
        labels_f  = [t.get_text() for t in leg_f.get_texts()]
        leg_f.remove()
        ax_f.legend(
            handles=handles_f, labels=labels_f,
            fontsize=5,
            loc="lower left",
            bbox_to_anchor=(0.01, 1.01),
            frameon=False,
            handletextpad=0.2, borderpad=0.3,
            markerscale=0.8,
        )
    ax_f.set_title(ax_f.get_title(), fontsize=ax_f.title.get_fontsize(),
                   fontweight="bold", pad=12)

    # Panel C y-axis tick labels — kept at same size as D (5 pt)

    # Panel B in composite: smaller markers + slightly larger labels
    for coll in ax_b.collections:
        coll.set_sizes([s * 0.5 for s in coll.get_sizes()])
    for txt in ax_b.texts:
        txt.set_fontsize(max(txt.get_fontsize() * 0.7, 4.0))

    _cap_fontsize(fig_c, _MAX)

    # Bold panel labels
    _lbl_fs = 7
    for ax, lbl in [
        (ax_a, "A"), (ax_b, "B"), (ax_c, "C"),
        (ax_d, "D"), (ax_f, "F"), (ax_g, "G"), (ax_h, "H"), (ax_i, "I"),
    ]:
        ax.text(-0.15, 1.12, lbl, transform=ax.transAxes,
                fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    if axes_e:
        axes_e[0].text(-0.10, 1.15, "E", transform=axes_e[0].transAxes,
                       fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")

    plt.rcParams.update(_prev_rc)

    # Save composite PNG
    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, MAIN_OUTPUT, close=False)

    # Save composite PDF
    pdf_dir  = MAIN_OUTPUT / f"{FIGURE_NAME}_panels"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print(f"  Saved combined artboard (PNG + PDF)")

    del data
    gc.collect()
    print(f"\nAll panels saved to {MAIN_OUTPUT}")
    print("Figure 2 complete: 11 individual panels (A-K) + combined artboard\n")
# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # FIX v1 Issue 8: apply_style() added, matches melanoma CLI entry point
    apply_style()
    generate()