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
G : Bar plot of mean delta score (post - pre) by arm.
H : Cohen's d effect sizes (anti-PDL1+Chemo - Chemo) on delta scores.

Changes from v1
---------------
- Fixed citation: Cancer Cell 2021, not Cell 2025
- Fixed bootstrap CI: uses percentile method not 1.96 × SE
- Fixed scoring: uses st.score_gene_sets(method="zmean") not sc.tl.score_genes
- Fixed Panel B: uses Spearman rho not Pearson r
- Fixed Panel D: percentile CI with fallback, no crash on missing columns
- Fixed Panel E: explicit permutation p-value column, same fallback as melanoma
- Fixed Panel H: uses st.cohens_d_from_did not local reimplementation
- Fixed seed handling: single RNG outside loop, SeedSequence per feature
- Fixed redundant participant filter in _prepare_data
- Added verify_paired_participants for Panel A annotation
- Added combined artboard (180 x 215 mm, PNG + PDF)
- Added apply_style() CLI entry point
- Imports COLORS from _shared.py (falls back to local if _shared not yet present)
- save_panel signature matches melanoma (fig, name, figure_name, output_dir)
- Permutation p-values stored in p_DiD_perm column, not overwriting p_DiD
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
import scanpy as sc
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.stats.multitest import multipletests

import sctrial as st
from sctrial import TrialDesign, did_table
from sctrial.stats.effect_size import cohens_d_from_did
from sctrial.utils import permutation_pvalue

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

    MAIN_OUTPUT = Path(
        "/Users/valenciai/Documents/Research/projects/TNBC/figures/outs/datatnbc_processed_responces.h5ad"
    )

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


# ── Paths ─────────────────────────────────────────────────────────────────
H5AD_PATH  = Path(
    "/Users/valenciai/Documents/Research/projects/TNBC/outs/datatnbc_processed_responces.h5ad"
)
OUTPUT_DIR = MAIN_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# ── Gene signatures ───────────────────────────────────────────────────────
# Same 12 signatures as v1 — move to _shared.TNBC_GENE_SIGNATURES once ready.
GENE_SIGNATURES = {
    "Cytotoxic T Cell": [
        "GZMB", "GZMA", "GZMH", "GZMK", "PRF1", "GNLY", "NKG7", "KLRK1",
        "KLRD1", "FASLG", "IFNG",
    ],
    "Immune Exhaustion": [
        "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "TOX2",
        "ENTPD1", "CD244", "CD160", "BTLA",
    ],
    "Interferon Response": [
        "ISG15", "IFI6", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2",
        "OAS1", "OAS2", "OAS3", "STAT1", "IRF7", "IRF9",
    ],
    "Memory T Cell": [
        "IL7R", "TCF7", "LEF1", "CCR7", "SELL", "CD27", "CD28",
        "BCL2", "EOMES", "ID3",
    ],
    "T Cell Activation": [
        "CD69", "CD44", "IL2RA", "ICOS", "TNFRSF4", "TNFRSF9",
        "CD40LG", "HLA-DRA", "HLA-DRB1",
    ],
    "Inflammatory Response": [
        "IL1B", "IL6", "TNF", "CXCL8", "CCL2", "CCL3", "CCL4",
        "NFKB1", "NLRP3", "CASP1",
    ],
    "Antigen Presentation": [
        "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2",
        "PSMB8", "PSMB9", "CD74",
    ],
    "Cell Proliferation": [
        "MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "CCNA2",
        "MCM2", "MCM7", "TYMS",
    ],
    "Regulatory T Cell": [
        "FOXP3", "IL2RA", "CTLA4", "TNFRSF18", "IKZF2", "IKZF4",
        "IL10", "TGFB1", "ENTPD1",
    ],
    "NK Cell Activity": [
        "NCAM1", "FCGR3A", "NCR1", "NCR3", "KLRF1", "KLRC1",
        "KIR2DL1", "KIR2DL3", "KIR3DL1",
    ],
    "Apoptosis": [
        "BCL2", "BAX", "BAK1", "CASP3", "CASP8", "CASP9",
        "FAS", "FASLG", "BID", "PARP1",
    ],
    "Oxidative Stress": [
        "NFE2L2", "HMOX1", "NQO1", "GCLC", "GCLM", "GSR",
        "SOD1", "SOD2", "CAT", "GPX1",
    ],
}

MIN_GENES = 3


# ── Data preparation ──────────────────────────────────────────────────────
def _prepare_data() -> dict:
    """Load TNBC h5ad, score signatures, run DiD, compute delta scores."""
    print("Loading data...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")

    print("Scoring signatures (zmean method)...")
    available = set(adata.var_names)
    valid_gene_sets = {}
    for name, genes in GENE_SIGNATURES.items():
        found = [g for g in genes if g in available]
        if len(found) >= MIN_GENES:
            valid_gene_sets[name] = found
            print(f"  {name}: {len(found)}/{len(genes)} genes")
        else:
            print(f"  {name}: SKIP ({len(found)} genes found, need {MIN_GENES})")

    adata, sig_cols = score_signatures(
        adata,
        gene_sets=valid_gene_sets,
        layer="log1p_norm",
        min_genes=MIN_GENES,
        prefix="sig_",
    )
    print(f"\n  Scored {len(sig_cols)} signatures")

    pair_info = verify_paired_participants(
        adata.obs,
        visit_col=DESIGN.visit_col,
        visits=VISITS,
        participant_col=DESIGN.participant_col,
    )

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

    print("Running DiD (participant-level, primary analysis)...")
    res_part = did_table(
        adata,
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer="log1p_norm",
        standardize=True,
        aggregate="participant_visit",
    )

    grp_cols = [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col]
    df_agg = (
        adata.obs[grp_cols + sig_cols]
        .groupby(grp_cols, observed=True)[sig_cols]
        .mean()
        .reset_index()
    )

    # ── Permutation p-values and bootstrap CIs ────────────────────────────
    participant_arm_map = (
        adata.obs
        .groupby(DESIGN.participant_col)[DESIGN.arm_col]
        .first()
    )

    ss = np.random.SeedSequence(SEED)
    child_seeds = ss.spawn(len(sig_cols))

    perm_pvals = []
    boot_ses   = []
    ci_los     = []
    ci_his     = []

    for feat, child_seed in zip(sig_cols, child_seeds):
        wide = df_agg.pivot_table(
            index=DESIGN.participant_col,
            columns=DESIGN.visit_col,
            values=feat,
            aggfunc="mean",
        )
        if VISITS[0] not in wide.columns or VISITS[1] not in wide.columns:
            perm_pvals.append(np.nan)
            boot_ses.append(np.nan)
            ci_los.append(np.nan)
            ci_his.append(np.nan)
            continue

        wide["delta"] = wide[VISITS[1]] - wide[VISITS[0]]
        wide = wide.dropna(subset=["delta"])
        wide["arm"] = wide.index.map(participant_arm_map)

        dt = wide.loc[wide["arm"] == DESIGN.arm_treated, "delta"].values
        dc = wide.loc[wide["arm"] == DESIGN.arm_control, "delta"].values

        p = permutation_pvalue(dt, dc, n_perm=9999, seed=SEED)
        perm_pvals.append(p)

        rng = np.random.default_rng(child_seed)
        if len(dt) >= 2 and len(dc) >= 2:
            # Standardise deltas to match the scale of beta_DiD from did_table
            all_deltas = np.concatenate([dt, dc])
            pooled_std = np.std(all_deltas, ddof=1)
            if pooled_std > 0:
                dt_std = dt / pooled_std
                dc_std = dc / pooled_std
            else:
                dt_std = dt
                dc_std = dc

            boots = np.array([
                rng.choice(dt_std, len(dt_std), replace=True).mean()
                - rng.choice(dc_std, len(dc_std), replace=True).mean()
                for _ in range(999)
            ])
            boot_ses.append(float(np.std(boots, ddof=1)))
            ci_los.append(float(np.percentile(boots, 2.5)))
            ci_his.append(float(np.percentile(boots, 97.5)))
        else:
            boot_ses.append(np.nan)
            ci_los.append(np.nan)
            ci_his.append(np.nan)

    # Build a keyed dataframe and merge on feature name to guarantee alignment
    boot_df = pd.DataFrame({
        "feature":     sig_cols,
        "p_DiD_perm":  perm_pvals,
        "se_DiD_boot": boot_ses,
        "ci_lo_boot":  ci_los,
        "ci_hi_boot":  ci_his,
    })

    for col in ["p_DiD_perm", "se_DiD_boot", "ci_lo_boot", "ci_hi_boot"]:
        if col in res_part.columns:
            res_part = res_part.drop(columns=col)

    res_part = res_part.merge(boot_df, on="feature", how="left")

    mask = res_part["p_DiD_perm"].notna()
    res_part["FDR_DiD_perm"] = np.nan
    if mask.sum() > 0:
        res_part.loc[mask, "FDR_DiD_perm"] = multipletests(
            res_part.loc[mask, "p_DiD_perm"], method="fdr_bh"
        )[1]

    # ── Delta scores per participant ──────────────────────────────────────
    pb = df_agg.copy()
    visit_counts = pb.groupby(DESIGN.participant_col)[DESIGN.visit_col].nunique()
    paired_pids  = visit_counts[visit_counts == 2].index
    pb = pb[pb[DESIGN.participant_col].isin(paired_pids)].copy()

    pre   = pb[pb[DESIGN.visit_col] == VISITS[0]].set_index(DESIGN.participant_col)
    post  = pb[pb[DESIGN.visit_col] == VISITS[1]].set_index(DESIGN.participant_col)
    delta = post[sig_cols].subtract(pre[sig_cols])
    delta[DESIGN.arm_col] = pre[DESIGN.arm_col]
    delta = delta.reset_index()

    n_t = (delta[DESIGN.arm_col] == DESIGN.arm_treated).sum()
    n_c = (delta[DESIGN.arm_col] == DESIGN.arm_control).sum()
    print(
        f"  Paired participants: {len(delta)} "
        f"({DESIGN.arm_treated}={n_t}, {DESIGN.arm_control}={n_c})"
    )

    return {
        "adata":               adata,
        "sig_cols":            sig_cols,
        "res_cell":            res_cell,
        "res_part":            res_part,
        "pb":                  pb,
        "delta":               delta,
        "pair_info":           pair_info,
        "participant_arm_map": participant_arm_map,
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
        0.055, 0.68,
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
        handles=handles, fontsize=6,
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
        "antigen": (-dx, -dy * 2, "right"),
        "cytotoxic": (-dx, dy * 2, "right"),
        "memory": (-dx * 3, -dy, "right"),
        "immune exh": (-dx, dy * 3, "right"),
        "interferon": (-dx, -dy * 3, "right"),
        "cell prolif": (dx, dy, "left"),
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
    ax.legend(handles=handles, fontsize=7, loc="lower right",
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
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28),
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
              fontsize=4,
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
    axes[-1].legend(handles=handles, fontsize=7,
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

    # Build responder map: one response value per participant
    responder_map = (
        adata.obs
        .groupby(DESIGN.participant_col, observed=True)["response"]
        .first()
    )

    # Debug — remove after confirming
    for pid in ordered:
        print(f"  {pid}: {responder_map.get(pid)}")

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

    from matplotlib.transforms import blended_transform_factory
    star_trans = blended_transform_factory(ax.transAxes, ax.transData)

    for i, (tick, pid) in enumerate(zip(ax.get_yticklabels(), ordered)):
        tick.set_color(COL_TREAT if arms.iloc[i] == DESIGN.arm_treated else COL_CTRL)
        if responder_map.get(pid) == "R":
            ax.text(
                -0.065, i,
                "★",
                transform=star_trans,
                fontsize=6, color="black",
                va="center", ha="center",
                clip_on=False,
            )

    ax.set_title("Per-participant score change (Post - Pre)",
                 fontsize=11, fontweight="bold")

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score Δ", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    n_treated = sum(1 for p in ordered if arms.loc[p] == DESIGN.arm_treated)
    n_control = len(ordered) - n_treated

    if n_treated > 0:
        ax.text(
            -0.18, (n_treated - 1) / 2,
            "anti-PDL1+\nChemo",
            transform=trans,
            color=COL_CTRL,
            fontsize=4, fontweight="bold",
            ha="right", va="center",
            rotation=90, clip_on=False,
        )

    if n_control > 0:
        ax.text(
            -0.18, n_treated + (n_control - 1) / 2,
            "Chemo",
            transform=trans,
            color=COL_TREAT,
            fontsize=4, fontweight="bold",
            ha="right", va="center",
            rotation=90, clip_on=False,
        )
# ── Panel G: Mean delta by arm ────────────────────────────────────────────

def _panel_g(ax: plt.Axes, data: dict) -> None:
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
    ax.set_yticklabels(display_names, fontsize=4)
    ax.set_xlabel("Mean Δ score (Post - Pre)", fontsize=5)
    ax.set_title("Signature Changes by Arm", fontsize=11, fontweight="bold")
    ax.legend(fontsize=4, loc="upper center", bbox_to_anchor=(0.5, -0.22),
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
            ("panel_A_paired_verification", _panel_a,  (11, 6)),
            ("panel_B_beta_comparison",     _panel_b,  (9,  7)),
            ("panel_C_pvalue_inflation",    _panel_c,  (9,  7)),
            ("panel_D_forest",              _panel_d,  (10, 7)),
            ("panel_G_delta_by_arm",        _panel_g,  (8,  6)),
            ("panel_H_cohens_d",            _panel_h,  (8,  6)),
        ]
        for panel_name, func, size in panel_specs:
            fig, ax = plt.subplots(figsize=size)
            func(ax, data)
            _enforce_min_fontsize(fig)
            fig.tight_layout()
            save_panel(fig, panel_name, FIGURE_NAME, OUTPUT_DIR)

        # Panel E — needs figure + gridspec
        fig_e = plt.figure(figsize=(14, 9))
        gs_e  = fig_e.add_gridspec(1, 1)[0, 0]
        _panel_e(fig_e, gs_e, data, n_sigs=6)
        _enforce_min_fontsize(fig_e)
        fig_e.tight_layout()
        save_panel(fig_e, "panel_E_interaction_grid", FIGURE_NAME, OUTPUT_DIR)

        # Panel F — heatmap
        fig_f, ax_f = plt.subplots(figsize=(12, 7))
        _panel_f(ax_f, data)
        _enforce_min_fontsize(fig_f)
        fig_f.tight_layout()
        save_panel(fig_f, "panel_F_heatmap", FIGURE_NAME, OUTPUT_DIR)

    # ── Combined artboard (180 × 215 mm) ──────────────────────────────────
    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm  = 1.0 / 25.4
    _MAX = 6

    fig_c = plt.figure(figsize=(180 * _mm, 215 * _mm))

    outer = fig_c.add_gridspec(
        4, 1,
        height_ratios=[1, 2.2, 1.3, 1],
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
    gs2  = outer[2].subgridspec(1, 2, wspace=0.55)
    ax_f = fig_c.add_subplot(gs2[0])
    ax_g = fig_c.add_subplot(gs2[1])

    # Row 3: H centred
    gs3  = outer[3].subgridspec(1, 3, width_ratios=[0.6, 1.8, 0.6], wspace=0.40)
    ax_h = fig_c.add_subplot(gs3[1])

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

    # ── Legend overrides (all must come AFTER panels are drawn) ───────────

    # Panels B, C, G — move below-axis legends inside plot
    _inside = {
        ax_b: "lower right",
        ax_c: "lower right",
        ax_g: "lower left",
    }
    for ax_target, loc in _inside.items():
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels  = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_target.legend(
                handles=handles, labels=labels,
                fontsize=3.5, loc=loc,
                frameon=True, framealpha=0.85,
                handlelength=0.8, handletextpad=0.2,
                borderpad=0.2, labelspacing=0.15,
                columnspacing=0.5,
            )

    # Panel A legend — nudged right via bbox_to_anchor to avoid overlapping bars
    leg_a = ax_a.get_legend()
    if leg_a:
        handles_a = leg_a.legend_handles
        labels_a  = [t.get_text() for t in leg_a.get_texts()]
        leg_a.remove()
        ax_a.legend(
            handles=handles_a, labels=labels_a,
            fontsize=3.5,
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
            fontsize=3.0, loc="lower right",
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
            fontsize=3.5,
            loc="upper center", bbox_to_anchor=(0.5, -0.25),
            ncol=3, frameon=True, framealpha=0.95,
            handlelength=2.5,  # ← wider line segments
            handletextpad=0.6,  # ← space between line and label
            borderpad=0.2, labelspacing=0.15,
            columnspacing=1.5,  # ← space between columns
        )

    # Panel F — remove legend box; replace with rotated arm labels
    leg_f = ax_f.get_legend()
    if leg_f:
        leg_f.remove()
    from matplotlib.transforms import blended_transform_factory
    trans     = blended_transform_factory(ax_f.transAxes, ax_f.transData)
    # recompute ordered/arms from the heatmap data for label placement
    pb_f      = _pseudobulk_all(data["adata"], data["sig_cols"])
    pre_mask  = pb_f[DESIGN.visit_col] == VISITS[0]
    pre_arm   = (
        pb_f.loc[pre_mask]
        .groupby(DESIGN.participant_col, observed=True)[DESIGN.arm_col]
        .first()
    )
    n_treated_f = int((pre_arm == DESIGN.arm_treated).sum())
    n_control_f = int((pre_arm == DESIGN.arm_control).sum())
    if n_treated_f > 0:
        ax_f.text(
            -0.18, (n_treated_f - 1) / 2,
            "anti-PDL1+\nChemo",
            transform=trans, color=COL_TREAT,
            fontsize=4, fontweight="bold",
            ha="right", va="center",
            rotation=90, clip_on=False,
        )
    if n_control_f > 0:
        ax_f.text(
            -0.18, n_treated_f + (n_control_f - 1) / 2,
            "Chemo",
            transform=trans, color=COL_CTRL,
            fontsize=4, fontweight="bold",
            ha="right", va="center",
            rotation=90, clip_on=False,
        )

    # Shrink Panel C y-axis tick labels in composite
    for tick in ax_c.get_yticklabels():
        tick.set_fontsize(3.5)

    # Shrink annotation text in Panel B
    for txt in ax_b.texts:
        txt.set_fontsize(max(txt.get_fontsize() * 0.55, 3.0))

    _cap_fontsize(fig_c, _MAX)

    # Bold panel labels
    _lbl_fs = 7
    for ax, lbl in [
        (ax_a, "A"), (ax_b, "B"), (ax_c, "C"),
        (ax_d, "D"), (ax_f, "F"), (ax_g, "G"), (ax_h, "H"),
    ]:
        ax.text(-0.15, 1.12, lbl, transform=ax.transAxes,
                fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    if axes_e:
        axes_e[0].text(-0.10, 1.15, "E", transform=axes_e[0].transAxes,
                       fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")

    plt.rcParams.update(_prev_rc)

    # Save composite PNG
    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, OUTPUT_DIR, close=False)

    # Save composite PDF
    pdf_dir  = OUTPUT_DIR / f"{FIGURE_NAME}_panels"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print(f"  Saved combined artboard (PNG + PDF)")

    del data
    gc.collect()
    print(f"\nAll panels saved to {OUTPUT_DIR}")
    print("Figure 2 complete: 8 individual panels (A-H) + combined artboard\n")
# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # FIX v1 Issue 8: apply_style() added, matches melanoma CLI entry point
    apply_style()
    generate()