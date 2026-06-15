"""
Figure 3 — Statistical Robustness & Method Benchmarking (TNBC)
==============================================================

Adapted from the melanoma Figure 3 for the Zhang et al. (Cancer Cell 2021)
TNBC dataset (GSE169246). Panels A, B, F, G use TNBC data.
Panels C, D, E use the NatMeth simulation benchmark CSV (method-agnostic).

Panels
------
A   Bootstrap vs analytical SE (TNBC).
B   Leave-one-out participant sensitivity (TNBC).
C   Null-gene FPR vs signal fraction (simulator benchmark; faceted by panel size).
D   Effect-size bias and RMSE on signal genes (simulator benchmark).
E   Genomic inflation λ_GC under pure null (simulator benchmark).
F   Standard-error comparison (cell vs participant level) (TNBC).
G   Cross-dataset signed Cohen's d forest (pre-specified endpoints, TNBC added).
"""

from __future__ import annotations

import gc
import hashlib
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.ticker import MultipleLocator
from scipy import stats
from statsmodels.stats.multitest import multipletests

import sctrial as st
from sctrial import TrialDesign, did_table
from sctrial.stats.effect_size import cohens_d_from_did, effect_size_ci

warnings.filterwarnings("ignore")

# ── Shared imports via direct path ────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, "/Users/valenciai/Documents/Research/projects/sctrial_breast/sctrial/manuscript_figures")
from _shared import (  # noqa: E402
    COLORS,
    apply_style,
    despine,
    save_panel,
    score_signatures,
    sig_display,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    get_aml,
    get_cart,
    harmonize_response,
    between_arm_comparison,
    within_arm_comparison,
    clear_cache,
)


MAIN_OUTPUT = Path("/Users/valenciai/Documents/Research/projects/TNBC/figures/outs")

# ── Paths ──────────────────────────────────────────────────────────────────
H5AD_PATH = Path(
    "/Users/valenciai/Documents/Research/projects/TNBC/outs/datatnbc_processed_responces.h5ad"
)
OUTPUT_DIR = MAIN_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_NAME = "Figure3_tnbc_robustness_benchmarking"

# ── TNBC design ────────────────────────────────────────────────────────────
VISITS: tuple[str, str] = ("Pre", "Post")
N_BOOT = 999
SEED = 42

DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="arm",
    arm_treated="anti-PDL1+Chemo",
    arm_control="Chemo",
)

# ── Gene signatures ────────────────────────────────────────────────────────
GENE_SIGNATURES = {
    "Cytotoxic T Cell Activity": [
        "GZMB", "GZMA", "GZMH", "GZMK", "PRF1", "GNLY", "NKG7",
        "KLRK1", "KLRD1", "FASLG", "IFNG",
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
    "Oxidative Stress Response": [
        "NFE2L2", "HMOX1", "NQO1", "GCLC", "GCLM", "GSR",
        "SOD1", "SOD2", "CAT", "GPX1",
    ],
}
MIN_GENES = 3

# ── Benchmark CSV (simulation, method-agnostic) ────────────────────────────
_BENCHMARK_CSV = Path(
    "/Users/valenciai/Documents/Research/projects/sctrial_breast/sctrial"
    "/temp/simulation/sensitivity/sensitivity_combined.csv"
)

# ── Cache ──────────────────────────────────────────────────────────────────
_CODE_VERSION = "v1_tnbc"
_CACHE_DIR = OUTPUT_DIR / "_cache"


def _cache_key(*args):
    payload = "|".join([_CODE_VERSION] + list(args))
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _load_cache(tag):
    path = _CACHE_DIR / f"{tag}.json"
    if path.exists():
        try:
            return pd.read_json(path, orient="records")
        except Exception:
            return None
    return None


def _save_cache(tag, df):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(_CACHE_DIR / f"{tag}.json", orient="records", indent=2)


# ── Benchmark method styles ────────────────────────────────────────────────
_BENCH_METHODS = ["wilcoxon_paired", "nebula", "dreamlet", "sctrial_did"]
_BENCH_METHOD_LABELS = {
    "sctrial_did":      "sctrial (DiD)",
    "dreamlet":         "dreamlet",
    "nebula":           "NEBULA",
    "wilcoxon_paired":  "Wilcoxon (Δ scores)",
}
_BENCH_METHOD_COLORS = {
    "sctrial_did":     "#1f77b4",
    "dreamlet":        "#d62728",
    "nebula":          "#ff7f0e",
    "wilcoxon_paired": "#2ca02c",
}
_BENCH_METHOD_MARKERS = {
    "sctrial_did":     "o",
    "dreamlet":        "D",
    "nebula":          "s",
    "wilcoxon_paired": "^",
}
_PANEL_SIZES      = [50, 200, 500, 2000]
_SIGNAL_FRACTIONS = [1, 5, 10, 20]

# ── Cross-dataset colors ───────────────────────────────────────────────────
DATASET_COLORS = {
    "TNBC":         "#e91e8c",
    "Sade-Feldman": COLORS["control"],
    "Vaccine":      COLORS["treated"],
    "AML":          COLORS["success"],
    "CAR-T":        COLORS["neutral"],
    "COVID-19":     COLORS["highlight"],
}
_DATASET_DISPLAY_NAMES = {"Sade-Feldman": "Melanoma"}

_PRESPECIFIED_ENDPOINTS = [
    "sig_Cytotoxic T Cell Activity",
    "sig_Interferon Response",
    "sig_Immune Exhaustion",
    "sig_T Cell Activation",
    "sig_Inflammatory Response",
]


# ======================================================================
# TNBC data preparation (panels A, B, F)
# ======================================================================

def _prepare_tnbc_data() -> dict:
    """Load TNBC h5ad, score signatures, run DiD, bootstrap, LOO."""
    print("Loading TNBC data...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")

    # Score signatures using st directly (avoids _shared.py signature mismatch)
    available = set(adata.var_names)
    valid_gene_sets = {}
    for name, genes in GENE_SIGNATURES.items():
        found = [g for g in genes if g in available]
        if len(found) >= MIN_GENES:
            valid_gene_sets[name] = found
    st.score_gene_sets(adata, gene_sets=valid_gene_sets, method="zmean",
                       layer="log1p_norm", min_genes=MIN_GENES,
                       prefix="sig_", overwrite=True)
    sig_cols = [f"sig_{n}" for n in valid_gene_sets
                if f"sig_{n}" in adata.obs.columns]
    print(f"  Scored {len(sig_cols)} signatures")

    common_kw = dict(
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer="log1p_norm",
        standardize=True,
    )

    print("  Cell-level DiD...")
    df_cell = did_table(adata, aggregate="cell", **common_kw)

    print("  Participant-level DiD...")
    df_part = did_table(adata, aggregate="participant_visit", **common_kw)

    print("  Bootstrap DiD...")
    df_boot = did_table(
        adata, aggregate="participant_visit",
        use_bootstrap=True, n_boot=N_BOOT, seed=SEED,
        **common_kw,
    )

    print("  Leave-one-out analysis...")
    loo_records = _run_loo(adata, sig_cols, common_kw)

    return {
        "adata":     adata,
        "sig_cols":  sig_cols,
        "df_cell":   df_cell,
        "df_part":   df_part,
        "df_boot":   df_boot,
        "loo_records": loo_records,
    }


def _run_loo(adata, sig_cols, common_kw) -> pd.DataFrame:
    pid_col  = DESIGN.participant_col
    all_pids = adata.obs[pid_col].unique()
    records  = []
    for i, drop_pid in enumerate(all_pids):
        mask = adata.obs[pid_col] != drop_pid
        sub  = adata[mask]
        try:
            res = did_table(sub, aggregate="participant_visit", **common_kw)
            for _, row in res.iterrows():
                records.append({
                    "dropped_pid": drop_pid,
                    "feature":     row["feature"],
                    "beta_DiD":    row["beta_DiD"],
                    "se_DiD":      row["se_DiD"],
                    "p_DiD":       row["p_DiD"],
                })
        except Exception:
            pass
        if (i + 1) % 3 == 0:
            print(f"    LOO {i + 1}/{len(all_pids)}")
    print(f"  LOO complete: {len(all_pids)} participants")
    return pd.DataFrame(records)


# ======================================================================
# Panel A: Bootstrap vs Analytical SE
# ======================================================================

def _panel_a(ax, data: dict) -> None:
    df_boot = data["df_boot"]

    feats, analytical, bootstrap = [], [], []
    for _, row in df_boot.iterrows():
        se_an = row["se_DiD"]
        se_bt = row.get("se_DiD_boot", np.nan)
        if np.isfinite(se_an) and np.isfinite(se_bt):
            feats.append(row["feature"])
            analytical.append(se_an)
            bootstrap.append(se_bt)

    analytical = np.array(analytical)
    bootstrap  = np.array(bootstrap)

    if len(analytical) == 0:
        ax.text(0.5, 0.5, "Insufficient bootstrap data",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    lo   = min(analytical.min(), bootstrap.min()) * 0.85
    hi   = max(analytical.max(), bootstrap.max()) * 1.15
    x_lo = lo - 0.08 * (hi - lo)
    ax.plot([x_lo, hi], [x_lo, hi], ls="--", color=COLORS["gray"], lw=1, zorder=1)
    ax.scatter(analytical, bootstrap, s=50, color=COLORS["treated"],
               edgecolor="white", linewidth=0.5, zorder=3)

    for feat, xv, yv in zip(feats, analytical, bootstrap):
        ax.annotate(sig_display(feat), (xv, yv),
                    xytext=(4, 2), textcoords="offset points",
                    fontsize=5.5, alpha=0.85)

    r, p = stats.pearsonr(analytical, bootstrap)
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.1e}",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    ax.set_xlabel("Analytical SE (cluster-robust)")
    ax.set_ylabel("Bootstrap SE (wild cluster)")
    ax.set_title("Bootstrap vs Analytical SE (TNBC)", fontsize=10, fontweight="bold")
    ax.set_xlim(x_lo, hi)
    ax.set_ylim(lo, hi)
    despine(ax)


# ======================================================================
# Panel B: Leave-one-out sensitivity
# ======================================================================

def _panel_b(ax, data: dict) -> None:
    loo_df  = data["loo_records"]
    df_part = data["df_part"]

    if loo_df is None or len(loo_df) == 0:
        ax.text(0.5, 0.5, "LOO results unavailable",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    top_sigs = (
        df_part.assign(_abs=df_part["beta_DiD"].abs())
        .nlargest(4, "_abs")["feature"].tolist()
    )
    palette = [COLORS["treated"], COLORS["control"],
               COLORS["highlight"], COLORS["neutral"]]

    for idx, feat in enumerate(top_sigs):
        full_beta = df_part.loc[df_part["feature"] == feat, "beta_DiD"].values[0]
        loo_betas = loo_df.loc[loo_df["feature"] == feat, "beta_DiD"].values
        if len(loo_betas) == 0:
            continue
        color  = palette[idx % len(palette)]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(loo_betas))
        ax.scatter(loo_betas, np.full_like(loo_betas, idx) + jitter,
                   s=20, color=color, alpha=0.6, edgecolor="none", zorder=2)
        ax.scatter(full_beta, idx, s=64, color=color, marker="D",
                   edgecolor="black", linewidth=0.8, zorder=4)
        ax.hlines(idx, loo_betas.min(), loo_betas.max(),
                  colors=color, lw=1.5, alpha=0.4, zorder=1)

    ax.set_yticks(range(len(top_sigs)))
    ax.set_yticklabels([sig_display(s) for s in top_sigs], fontsize=8)
    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (standardized)")
    ax.set_title("Leave-One-Out Sensitivity (TNBC)", fontsize=10, fontweight="bold")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=palette[i % len(palette)], edgecolor="#333",
                     linewidth=0.5, label=sig_display(f))
               for i, f in enumerate(top_sigs)]
    handles += [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS["gray"], markersize=4, label="LOO estimate"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=COLORS["gray"], markeredgecolor="black",
               markersize=4.8, label="Full sample"),
    ]
    ax.legend(handles=handles, fontsize=7.6, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel F: SE comparison cell vs participant
# ======================================================================

def _panel_f(ax, data: dict) -> None:
    df_cell = data["df_cell"]
    df_boot = data["df_boot"]

    se_cell = df_cell[["feature", "se_DiD"]].copy().rename(
        columns={"se_DiD": "se_cell"})
    se_part = df_boot[["feature"]].copy()
    se_part["se_part"] = df_boot.get("se_DiD_boot", df_boot["se_DiD"])

    merged = se_cell.merge(se_part, on="feature")
    merged["display"] = merged["feature"].apply(sig_display)
    merged = merged.sort_values("se_part", ascending=True)

    y_pos = np.arange(len(merged))
    bar_h = 0.35
    ax.barh(y_pos - bar_h / 2, merged["se_cell"].values,
            height=bar_h, color=COLORS["highlight"], alpha=0.8,
            label="Cell-level SE", edgecolor="none")
    ax.barh(y_pos + bar_h / 2, merged["se_part"].values,
            height=bar_h, color=COLORS["treated"], alpha=0.8,
            label="Participant-level SE (bootstrap)", edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["display"].values, fontsize=8)
    ax.set_xlabel("Standard Error")
    ax.set_title("Precision: Cell vs Participant Level (TNBC)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel G: Cross-dataset Cohen's d forest (TNBC added)
# ======================================================================

def _paired_cohens_d(participant_deltas):
    n  = len(participant_deltas)
    sd = float(np.std(participant_deltas, ddof=1))
    if sd < 1e-12:
        return 0.0, 0.0, 0.0
    d      = float(np.mean(participant_deltas) / sd)
    se_d   = np.sqrt(1 / n + d**2 / (2 * n))
    t_crit = stats.t.ppf(0.975, n - 1)
    return d, d - t_crit * se_d, d + t_crit * se_d


def _compute_tnbc_effects(data: dict) -> pd.DataFrame:
    """Compute signed Cohen's d for TNBC pre-specified endpoints."""
    adata    = data["adata"]
    sig_cols = data["sig_cols"]
    pid_col  = DESIGN.participant_col

    participant_arm_map = (
        adata.obs.groupby(pid_col)[DESIGN.arm_col].first()
    )
    grp_cols = [pid_col, DESIGN.visit_col, DESIGN.arm_col]
    df_agg   = (
        adata.obs[grp_cols + sig_cols]
        .groupby(grp_cols, observed=True)[sig_cols]
        .mean()
        .reset_index()
    )

    records = []
    target  = [s for s in _PRESPECIFIED_ENDPOINTS if s in sig_cols]
    for sig in target:
        pb = df_agg[[pid_col, DESIGN.visit_col, DESIGN.arm_col, sig]].copy()
        deltas = {}
        for arm in [DESIGN.arm_treated, DESIGN.arm_control]:
            arm_pb = pb[pb[DESIGN.arm_col] == arm]
            arm_d  = []
            for _, pdf in arm_pb.groupby(pid_col):
                if set(VISITS).issubset(set(pdf[DESIGN.visit_col])):
                    pre  = pdf.loc[pdf[DESIGN.visit_col] == VISITS[0], sig].values[0]
                    post = pdf.loc[pdf[DESIGN.visit_col] == VISITS[1], sig].values[0]
                    arm_d.append(post - pre)
            deltas[arm] = arm_d

        n1 = len(deltas[DESIGN.arm_treated])
        n2 = len(deltas[DESIGN.arm_control])
        if n1 < 2 or n2 < 2:
            continue
        d_val         = cohens_d_from_did(np.array(deltas[DESIGN.arm_treated]),
                                          np.array(deltas[DESIGN.arm_control]))
        ci_lo, ci_hi  = effect_size_ci(d_val, n1, n2)
        records.append({
            "dataset":       "TNBC",
            "signature":     sig.replace("sig_", ""),
            "d":             d_val,
            "d_lower":       ci_lo,
            "d_upper":       ci_hi,
            "design_type":   "two_arm_did",
            "n_participants": n1 + n2,
        })
        print(f"    TNBC/{sig.replace('sig_', '')}: "
              f"d={d_val:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}] (n={n1 + n2})")

    return pd.DataFrame(records)


DatasetInfo = tuple[str, object, object, tuple, list[str], str]

SF_DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)
SF_VISITS: tuple[str, str] = ("Pre", "Post")


def _load_sf() -> DatasetInfo:
    sf = get_sade_feldman()
    sf = harmonize_response(sf)
    sf, sf_sigs = score_signatures(sf, layer="log1p_tpm")
    return ("Sade-Feldman", sf, SF_DESIGN, SF_VISITS, sf_sigs, "two_arm_did")


def _load_vaccine() -> DatasetInfo:
    vacc = get_vaccine()
    vacc, vacc_sigs = score_signatures(vacc, layer="counts")
    vacc_design = TrialDesign(
        participant_col="participant_id", visit_col="visit", arm_col=None)
    return ("Vaccine", vacc, vacc_design, ("Pre", "Post"), vacc_sigs, "paired")


def _load_aml() -> DatasetInfo:
    aml = get_aml()
    aml, aml_sigs = score_signatures(aml, layer="counts")
    pid_col    = "participant_id" if "participant_id" in aml.obs.columns else "patient_id"
    aml_design = TrialDesign(participant_col=pid_col, visit_col="visit", arm_col=None)
    return ("AML", aml, aml_design, ("Pre", "Post"), aml_sigs, "paired")


def _load_cart() -> DatasetInfo:
    cart = get_cart()
    cart, cart_sigs = score_signatures(cart, layer="counts")
    pid_col     = "participant_id" if "participant_id" in cart.obs.columns else "patient_id"
    cart_design = TrialDesign(participant_col=pid_col, visit_col="visit", arm_col=None)
    return ("CAR-T", cart, cart_design, ("Pre", "Post"), cart_sigs, "paired")


def _load_covid() -> DatasetInfo:
    covid = get_stephenson()
    covid, covid_sigs = score_signatures(covid, layer="counts")
    top_bin = (covid.obs["dfo_bin"].value_counts().idxmax()
               if "dfo_bin" in covid.obs.columns else "Pre")
    covid_design = TrialDesign(
        participant_col="participant_id", visit_col="dfo_bin",
        arm_col="severity", arm_treated="Severe", arm_control="Mild")
    return ("COVID-19", covid, covid_design, (top_bin,), covid_sigs, "cross_sectional")


def _compute_other_dataset_effects() -> pd.DataFrame:
    """Compute Cohen's d for Sade-Feldman, Vaccine, AML, CAR-T, COVID."""
    cache_tag = "effects_other_" + _cache_key("v1")
    cached    = _load_cache(cache_tag)
    if cached is not None:
        print("  Other-dataset effect sizes (cached)")
        return cached

    loaders = [_load_sf, _load_vaccine, _load_aml, _load_cart, _load_covid]
    frames  = []

    for loader in loaders:
        try:
            ds = loader()
        except Exception as exc:
            print(f"    Dataset failed to load: {exc}")
            continue
        name, adata, design, visits, sigs, dtype = ds
        print(f"  {name}: {adata.n_obs:,} cells")

        target = [s for s in _PRESPECIFIED_ENDPOINTS if s in sigs
                  and s in adata.obs.columns]
        pid_col = design.participant_col
        records = []

        for sig in target:
            try:
                if dtype == "two_arm_did":
                    pb = (adata.obs.groupby(
                              [pid_col, design.visit_col, design.arm_col],
                              observed=True)[sig].mean().reset_index())
                    deltas = {}
                    for arm in [design.arm_treated, design.arm_control]:
                        arm_pb = pb[pb[design.arm_col] == arm]
                        arm_d  = []
                        for _, pdf in arm_pb.groupby(pid_col):
                            if set(visits).issubset(set(pdf[design.visit_col])):
                                pre  = pdf.loc[pdf[design.visit_col] == visits[0], sig].values[0]
                                post = pdf.loc[pdf[design.visit_col] == visits[1], sig].values[0]
                                arm_d.append(post - pre)
                        deltas[arm] = arm_d
                    n1 = len(deltas[design.arm_treated])
                    n2 = len(deltas[design.arm_control])
                    if n1 < 2 or n2 < 2:
                        continue
                    d_val        = cohens_d_from_did(
                        np.array(deltas[design.arm_treated]),
                        np.array(deltas[design.arm_control]))
                    ci_lo, ci_hi = effect_size_ci(d_val, n1, n2)
                    n_ppt        = n1 + n2

                elif dtype == "paired":
                    pb = (adata.obs.groupby(
                              [pid_col, design.visit_col], observed=True
                          )[sig].mean().reset_index())
                    ds_list = []
                    for _, pdf in pb.groupby(pid_col):
                        if set(visits).issubset(set(pdf[design.visit_col])):
                            pre  = pdf.loc[pdf[design.visit_col] == visits[0], sig].values[0]
                            post = pdf.loc[pdf[design.visit_col] == visits[1], sig].values[0]
                            ds_list.append(post - pre)
                    if len(ds_list) < 3:
                        continue
                    d_val, ci_lo, ci_hi = _paired_cohens_d(np.array(ds_list))
                    n_ppt = len(ds_list)

                elif dtype == "cross_sectional":
                    visit_mask = adata.obs[design.visit_col] == visits[0]
                    obs_v      = adata.obs.loc[visit_mask]
                    pb_t = obs_v.loc[obs_v[design.arm_col] == design.arm_treated].groupby(pid_col)[sig].mean()
                    pb_c = obs_v.loc[obs_v[design.arm_col] == design.arm_control].groupby(pid_col)[sig].mean()
                    n1, n2 = len(pb_t), len(pb_c)
                    if n1 < 3 or n2 < 3:
                        continue
                    pooled_sd = np.sqrt(
                        ((n1 - 1) * pb_t.std()**2 + (n2 - 1) * pb_c.std()**2)
                        / (n1 + n2 - 2))
                    if pooled_sd < 1e-12:
                        continue
                    d_val        = float((pb_t.mean() - pb_c.mean()) / pooled_sd)
                    ci_lo, ci_hi = effect_size_ci(d_val, n1, n2)
                    n_ppt        = n1 + n2
                else:
                    continue

                records.append({
                    "dataset":        name,
                    "signature":      sig.replace("sig_", ""),
                    "d":              d_val,
                    "d_lower":        ci_lo,
                    "d_upper":        ci_hi,
                    "design_type":    dtype,
                    "n_participants": n_ppt,
                })
            except Exception as exc:
                print(f"    {name}/{sig}: FAILED ({exc})")

        if records:
            frames.append(pd.DataFrame(records))
        del adata
        gc.collect()

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _save_cache(cache_tag, result)
    return result


def _panel_g(ax, tnbc_effects: pd.DataFrame,
             other_effects: pd.DataFrame, *, composite: bool = False) -> None:
    """Forest plot — TNBC first, then other datasets."""
    if tnbc_effects.empty and other_effects.empty:
        ax.text(0.5, 0.5, "No effect size data available",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    effect_df = pd.concat([tnbc_effects, other_effects], ignore_index=True)
    ds_order  = list(dict.fromkeys(effect_df["dataset"]))

    _SIG_ORDER = [
        "Cytotoxic T Cell Activity",
        "Immune Exhaustion",
        "Inflammatory Response",
        "Interferon Response",
        "T Cell Activation",
    ]

    rows: list[dict] = []
    for ds in ds_order:
        grp = effect_df[effect_df["dataset"] == ds]
        if grp.empty:
            continue
        sig_map = {s: i for i, s in enumerate(_SIG_ORDER)}
        grp     = grp.copy()
        grp["_sort_key"] = grp["signature"].map(
            lambda s, m=sig_map: m.get(s, len(_SIG_ORDER)))
        grp = grp.sort_values("_sort_key")
        rows.append({"_group_label": ds})
        for _, row in grp.iterrows():
            rows.append(row.to_dict())

    if not rows:
        return

    y, y_positions, y_labels, data_rows = 0, [], [], []
    for r in reversed(rows):
        if "_group_label" in r:
            y_positions.append(y)
            y_labels.append(r["_group_label"])
            y += 1
        else:
            y_positions.append(y)
            y_labels.append(f"  {r.get('signature', '').replace('_', ' ')}")
            data_rows.append((len(y_positions) - 1, r))
            y += 1

    ax.grid(True, axis="x", color="#f0f0f0", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#444", linewidth=1.0, linestyle="-", zorder=1, alpha=0.5)

    _pt  = 46 if composite else 80
    _lw  = 1.7 if composite else 2.2
    _afs = 5.0 if composite else 7
    _ytf = 6.4 if composite else 8.5
    _ygt = 7.2 if composite else 9.5
    _xlf = 7.6 if composite else 11
    _ttf = 8.6 if composite else 13
    _xtf = 6.4 if composite else 9.5

    for idx, row in data_rows:
        yp    = y_positions[idx]
        color = DATASET_COLORS.get(row["dataset"], COLORS["gray"])
        ax.hlines(yp, row["d_lower"], row["d_upper"],
                  color=color, linewidth=_lw, zorder=2, alpha=0.7)
        ax.scatter(row["d"], yp, color=color, s=_pt, zorder=3,
                   edgecolors="white", linewidths=1.0)
        ax.text(row["d_upper"] + 0.08, yp,
                f"{row['d']:+.2f}  (nₚ={row['n_participants']})",
                fontsize=_afs, va="center", ha="left", color="#444")

    for i, lbl in enumerate(y_labels):
        if lbl in ds_order:
            ax.axhline(y_positions[i] - 0.5, color=COLORS["gray"],
                       linewidth=0.4, linestyle="-", alpha=0.3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=_ytf)
    for tick in ax.get_yticklabels():
        txt = tick.get_text()
        if txt in ds_order:
            tick.set_fontweight("bold")
            tick.set_fontsize(_ygt)
            tick.set_color(DATASET_COLORS.get(txt, COLORS["gray"]))

    for ref_d in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        ax.axvline(ref_d, color=COLORS["gray"], linewidth=0.4,
                   linestyle=":", zorder=0, alpha=0.25)

    ax.set_xlabel("Cohen's d  (signed effect size)", fontsize=_xlf)
    ax.set_title("Effect sizes — pre-specified endpoints",
                 fontsize=_ttf, fontweight="bold", pad=12 if not composite else 8)
    ax.tick_params(axis="x", labelsize=_xtf)
    all_vals = effect_df[["d_lower", "d_upper", "d"]].values.flatten()
    x_margin = max(abs(all_vals.min()), abs(all_vals.max())) + 0.5
    ax.set_xlim(-x_margin, x_margin)
    ax.set_ylim(-0.6, max(y_positions) + 0.6)
    despine(ax)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)


# ======================================================================
# Benchmark panels C, D, E (simulation CSV — method-agnostic)
# ======================================================================

def _load_benchmark_data() -> pd.DataFrame:
    if not _BENCHMARK_CSV.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found at {_BENCHMARK_CSV}.\n"
            "Run the sensitivity benchmark on HPC and rsync locally.")
    df = pd.read_csv(_BENCHMARK_CSV, low_memory=False)
    df["n_genes"]        = df["scenario"].str.extract(r"_g(\d+)")[0].astype(int)
    frac                 = df["scenario"].str.extract(r"_f(\d+)")
    df["signal_pct"]     = pd.to_numeric(frac[0], errors="coerce").fillna(0).astype(int)
    df["is_null_scenario"] = df["scenario"].str.contains("sens_null")
    return df


def _method_style(method, is_focal=False, alpha=1.0, *, composite=False):
    ms_hi, ms_lo = (5.6, 4.3) if composite else (9, 7)
    lw_hi, lw_lo = (1.45, 1.1) if composite else (2.5, 1.8)
    mew          = 0.48 if composite else 0.6
    return {
        "color":           _BENCH_METHOD_COLORS[method],
        "marker":          _BENCH_METHOD_MARKERS[method],
        "markersize":      ms_hi if is_focal else ms_lo,
        "markeredgecolor": "white",
        "markeredgewidth": mew,
        "linewidth":       lw_hi if is_focal else lw_lo,
        "alpha":           alpha,
    }


def _add_nominal_band(ax, level=0.05, low=0.03, high=0.07, color="#d62728"):
    ax.axhspan(low, high, color=color, alpha=0.06, zorder=0)
    ax.axhline(level, color=color, linestyle="--", linewidth=1.0,
               alpha=0.65, zorder=1)


def _style_axis(ax):
    ax.grid(axis="y", linestyle=":", color="#b0b0b0", alpha=0.45,
            linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", color="#333333",
                   width=0.8, length=4)


def _panel_c_fpr_curves(fig, bench_df, *, composite=False):
    """Panel C — null-gene FPR vs signal fraction."""
    null = bench_df[bench_df["true_beta"] == 0.0].copy()
    rows = []
    for (method, n_g, frac), grp in null.groupby(["method", "n_genes", "signal_pct"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({"method": method, "n_genes": int(n_g),
                     "signal_pct": int(frac),
                     "fpr": float((pvals < 0.05).mean()),
                     "n_tests": int(len(pvals))})
    fpr_df = pd.DataFrame(rows)
    fpr_df = fpr_df[fpr_df["signal_pct"] > 0].copy()

    axes   = fig.subplots(1, 4, sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    _ttl = 5.75 if composite else 12
    _axf = 5.15 if composite else 11
    _tkf = 4.65 if composite else 10
    _lgf = 5.2  if composite else 9
    if composite:
        fig.suptitle("Null-gene FPR vs signal fraction",
                     x=0.5, y=0.99, fontsize=5.8, fontweight="bold")

    x_pos   = np.arange(len(_SIGNAL_FRACTIONS), dtype=float)
    frac2x  = dict(zip(_SIGNAL_FRACTIONS, x_pos))
    offsets = {"wilcoxon_paired": -0.08, "nebula": -0.03,
               "sctrial_did": +0.03, "dreamlet": +0.08}

    for ax_idx, (ax, n_g) in enumerate(zip(axes, _PANEL_SIZES)):
        sub = fpr_df[fpr_df["n_genes"] == n_g]
        for method in _BENCH_METHODS:
            m = sub[sub["method"] == method].sort_values("signal_pct")
            if m.empty:
                continue
            is_focal = method == "sctrial_did"
            style    = _method_style(method, is_focal=is_focal, composite=composite)
            x        = np.array([frac2x[int(f)] for f in m["signal_pct"].values]) \
                       + offsets[method]
            ax.plot(x, m["fpr"],
                    label=_BENCH_METHOD_LABELS[method] if ax_idx == 0 else None,
                    zorder=10 if is_focal else 3, **style)
        _add_nominal_band(ax)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{f}%" for f in _SIGNAL_FRACTIONS], fontsize=_tkf)
        ax.set_xlim(-0.4, len(_SIGNAL_FRACTIONS) - 0.6)
        ax.set_xlabel("Signal fraction", fontsize=_axf)
        ax.set_title(f"{n_g:,} genes", fontsize=_ttl, fontweight="bold",
                     color="#222222", pad=4 if composite else 8)
        ax.set_ylim(0.0, 0.7)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.tick_params(axis="y", labelsize=_tkf)
        _style_axis(ax)

    axes[0].set_ylabel("Null-gene FPR (p < 0.05)", fontsize=_axf)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
                   ncol=1, frameon=True, framealpha=0.95,
                   edgecolor="#cccccc", fontsize=_lgf,
                   handlelength=0.85, columnspacing=0.55, markerscale=0.65)


def _panel_e_lambda_gc(ax, bench_df, *, composite=False):
    """Panel E — genomic inflation λ_GC."""
    null_s  = bench_df[bench_df["is_null_scenario"]]
    pure    = null_s[null_s["true_beta"] == 0.0]
    rows    = []
    for (method, n_g), grp in pure.groupby(["method", "n_genes"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) < 50:
            continue
        chi2 = stats.chi2.isf(pvals, df=1)
        chi2 = chi2[np.isfinite(chi2)]
        if len(chi2) < 50:
            continue
        lam  = float(np.median(chi2) / stats.chi2.ppf(0.5, df=1))
        rows.append({"method": method, "n_genes": int(n_g), "lambda_gc": lam})
    lam_df  = pd.DataFrame(rows)
    x_pos   = np.arange(len(_PANEL_SIZES), dtype=float)
    n_to_x  = dict(zip(_PANEL_SIZES, x_pos))

    _lf  = 5.15 if composite else 11
    _ttf = 6.0  if composite else 12
    _lgf = 5.2  if composite else 9

    for method in _BENCH_METHODS:
        sub = lam_df[lam_df["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_focal = method == "sctrial_did"
        style    = _method_style(method, is_focal=is_focal, composite=composite)
        xs       = [n_to_x[int(n)] for n in sub["n_genes"].values]
        ax.plot(xs, sub["lambda_gc"], label=_BENCH_METHOD_LABELS[method],
                zorder=10 if is_focal else 3, **style)

    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0,
               alpha=0.65, zorder=1)
    ax.axhspan(0.95, 1.05, color="#d62728", alpha=0.06, zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{p:,}" for p in _PANEL_SIZES], fontsize=_lf)
    ax.set_xlim(-0.35, len(_PANEL_SIZES) - 0.65)
    ax.set_xlabel("Panel size (genes)", fontsize=_lf)
    ax.set_ylabel(r"Genomic inflation factor ($\lambda_{\mathrm{GC}}$)",
                  fontsize=_lf)
    ax.set_title("Pure-null calibration across panel sizes",
                 fontsize=_ttf, fontweight="bold",
                 pad=5 if composite else 10)
    ax.set_ylim(0.88, 1.18)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.tick_params(axis="y", labelsize=_lf)
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.03),
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              fontsize=_lgf, markerscale=0.52, handlelength=1.0, ncol=1)
    _style_axis(ax)


def _compute_signal_bias_rmse(bench_df):
    mixed = bench_df[~bench_df["is_null_scenario"]].copy()
    sig   = mixed[mixed["true_beta"] != 0.0].dropna(subset=["estimated_beta"]).copy()
    sig["err"]    = sig["estimated_beta"] - sig["true_beta"]
    sig["sq_err"] = sig["err"] ** 2
    rows = []
    for (method, n_g, frac), grp in sig.groupby(["method", "n_genes", "signal_pct"]):
        if grp.empty:
            continue
        rows.append({"method": method, "n_genes": int(n_g), "signal_pct": int(frac),
                     "bias": float(grp["err"].mean()),
                     "rmse": float(np.sqrt(grp["sq_err"].mean())),
                     "n_tests": int(len(grp))})
    return pd.DataFrame(rows)


def _panel_d_signal_rmse(fig, bench_df, *, composite=False):
    """Panel D — bias and RMSE on signal genes."""
    df = _compute_signal_bias_rmse(bench_df)
    if hasattr(fig, "set_constrained_layout"):
        fig.set_constrained_layout(False)
    gs = fig.add_gridspec(
        2, 4,
        hspace=0.52 if composite else 0.38,
        wspace=0.18 if composite else 0.22,
        left=0.07  if composite else 0.08,
        right=0.99 if composite else 0.985,
        top=0.82   if composite else 0.84,
        bottom=0.16 if composite else 0.11,
    )
    _ttf  = 6.35 if composite else 12
    _ylf  = 6.2  if composite else 11
    _axf  = 5.35 if composite else 10
    _xlbf = 5.45 if composite else 10
    bw    = 0.20
    morder = ["sctrial_did", "wilcoxon_paired", "nebula", "dreamlet"]
    x_pos  = np.arange(len(_SIGNAL_FRACTIONS))
    bias_lo = min(df["bias"].min(), 0) - 0.02
    bias_hi = max(df["bias"].max(), 0.02) * 1.12
    rmse_hi = df["rmse"].max() * 1.18
    bias_axes, rmse_axes = [], []

    for col, n_g in enumerate(_PANEL_SIZES):
        ax_bias = fig.add_subplot(gs[0, col])
        ax_rmse = fig.add_subplot(gs[1, col])
        bias_axes.append(ax_bias)
        rmse_axes.append(ax_rmse)
        sub = df[df["n_genes"] == n_g]
        for mi, method in enumerate(morder):
            bv, rv = [], []
            for frac in _SIGNAL_FRACTIONS:
                cell = sub[(sub["method"] == method) & (sub["signal_pct"] == frac)]
                bv.append(float(cell["bias"].iloc[0]) if len(cell) else np.nan)
                rv.append(float(cell["rmse"].iloc[0]) if len(cell) else np.nan)
            offset = (mi - (len(morder) - 1) / 2) * bw
            ax_bias.bar(x_pos + offset, bv, bw,
                        color=_BENCH_METHOD_COLORS[method],
                        edgecolor="white", linewidth=0.6, zorder=3)
            ax_rmse.bar(x_pos + offset, rv, bw,
                        color=_BENCH_METHOD_COLORS[method],
                        edgecolor="white", linewidth=0.6, zorder=3)
        ax_bias.axhline(0, color="#222", linestyle="--",
                        linewidth=0.9, alpha=0.7, zorder=2)
        ax_bias.set_xticks(x_pos)
        ax_bias.set_xticklabels([])
        ax_bias.set_ylim(bias_lo, bias_hi)
        ax_bias.yaxis.set_major_locator(MultipleLocator(0.05))
        ax_bias.set_title(f"{n_g:,} genes", fontsize=_ttf, fontweight="bold",
                          color="#1a1a1a", pad=-7 if composite else 8)
        _style_axis(ax_bias)
        ax_rmse.set_xticks(x_pos)
        ax_rmse.set_xticklabels([f"{f}%" for f in _SIGNAL_FRACTIONS], fontsize=_axf)
        ax_rmse.set_xlabel("Signal fraction", fontsize=_xlbf)
        ax_rmse.set_ylim(0, rmse_hi)
        ax_rmse.yaxis.set_major_locator(MultipleLocator(0.05))
        ax_rmse.tick_params(axis="both", labelsize=_axf)
        _style_axis(ax_rmse)
        ax_bias.tick_params(axis="y", labelsize=_axf)
        if col > 0:
            ax_bias.set_yticklabels([])
            ax_rmse.set_yticklabels([])

    bias_axes[0].set_ylabel(r"Mean bias ($\hat{\beta} - \beta$)", fontsize=_ylf)
    rmse_axes[0].set_ylabel(r"RMSE of $\hat{\beta}$", fontsize=_ylf)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_BENCH_METHOD_COLORS[m],
                       edgecolor="white", linewidth=0.6,
                       label=_BENCH_METHOD_LABELS[m])
        for m in morder
    ]
    _lgf = 5.2 if composite else 10
    if composite:
        _anchor = bias_axes[0]
        _cx     = 2.0
        _anchor.text(_cx, 1.25, "Effect-size estimation accuracy",
                     transform=_anchor.transAxes, ha="center", va="bottom",
                     fontsize=5.9, fontweight="bold")
        _anchor.legend(handles=legend_handles, loc="upper center", ncol=4,
                       bbox_to_anchor=(_cx, -0.15),
                       bbox_transform=_anchor.transAxes,
                       frameon=True, framealpha=0.93, edgecolor="#cccccc",
                       fontsize=_lgf, handlelength=0.85, handleheight=0.42,
                       handletextpad=0.35, columnspacing=0.55, borderpad=0.35)
    else:
        fig.legend(handles=legend_handles, loc="upper center", ncol=4,
                   bbox_to_anchor=(0.53, 0.94),
                   frameon=True, framealpha=0.95, edgecolor="#cccccc",
                   fontsize=_lgf)


# ======================================================================
# Generate
# ======================================================================

def generate() -> None:
    apply_style()
    print("Figure 3 (TNBC): Robustness & Benchmarking")

    # ── TNBC data (panels A, B, F, G-TNBC) ───────────────────────────
    data = _prepare_tnbc_data()

    # ── TNBC effect sizes for panel G ────────────────────────────────
    print("  Computing TNBC effect sizes...")
    tnbc_effects = _compute_tnbc_effects(data)

    # ── Other-dataset effect sizes for panel G ────────────────────────
    print("  Computing other-dataset effect sizes...")
    try:
        other_effects = _compute_other_dataset_effects()
    except Exception as exc:
        print(f"  Warning: other datasets failed ({exc})")
        other_effects = pd.DataFrame()

    # ── Benchmark CSV (panels C, D, E) ───────────────────────────────
    bench_df = None
    try:
        bench_df = _load_benchmark_data()
        print(f"  Benchmark CSV: {len(bench_df):,} rows, "
              f"{bench_df.scenario.nunique()} scenarios")
    except FileNotFoundError as exc:
        print(f"  Warning: {exc}")

    # ── Individual panels ─────────────────────────────────────────────
    for panel_name, func, size in [
        ("panel_A_bootstrap_validation", _panel_a, (6.5, 5)),
        ("panel_B_loo_sensitivity",      _panel_b, (6.5, 5)),
        ("panel_F_se_comparison",        _panel_f, (6.5, 5)),
    ]:
        fig, ax = plt.subplots(figsize=size)
        func(ax, data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, OUTPUT_DIR)

    fig_g, ax_g = plt.subplots(figsize=(10, 8))
    _panel_g(ax_g, tnbc_effects, other_effects)
    fig_g.tight_layout()
    save_panel(fig_g, "panel_G_cross_dataset_effects", FIGURE_NAME, OUTPUT_DIR)

    if bench_df is not None:
        fig_c = plt.figure(figsize=(14, 4.2))
        _panel_c_fpr_curves(fig_c, bench_df)
        fig_c.suptitle("Null-gene FPR scales with signal fraction, not panel size",
                        fontsize=13, fontweight="bold", y=1.04)
        fig_c.tight_layout()
        save_panel(fig_c, "panel_C_benchmark_fpr_curves", FIGURE_NAME, OUTPUT_DIR)

        fig_d = plt.figure(figsize=(14, 6.8))
        _panel_d_signal_rmse(fig_d, bench_df)
        fig_d.suptitle("Effect-size estimation accuracy on signal genes",
                        fontsize=13, fontweight="bold", y=0.995)
        save_panel(fig_d, "panel_D_benchmark_signal_rmse", FIGURE_NAME, OUTPUT_DIR)

        fig_e, ax_e = plt.subplots(figsize=(7.2, 5.0))
        _panel_e_lambda_gc(ax_e, bench_df)
        fig_e.tight_layout()
        save_panel(fig_e, "panel_E_benchmark_lambda_gc", FIGURE_NAME, OUTPUT_DIR)

    # ── Combined artboard ─────────────────────────────────────────────
    _SMALL_RC = {
        "font.size": 5, "axes.titlesize": 5.5, "axes.labelsize": 5,
        "xtick.labelsize": 4.5, "ytick.labelsize": 4.5,
        "legend.fontsize": 4, "legend.title_fontsize": 4,
    }
    _MAX = 6

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

    def _match_height(ref_ax, subfig, height_frac=1.0):
        axes = [a for a in subfig.get_axes() if a.get_visible()]
        if not axes:
            return
        ref_bb  = ref_ax.get_position()
        target_h = max(ref_bb.height * height_frac, 1e-6)
        block_y0 = min(a.get_position().y0 for a in axes)
        block_y1 = max(a.get_position().y1 for a in axes)
        block_h  = max(block_y1 - block_y0, 1e-6)
        scale    = target_h / block_h
        for a in axes:
            bb    = a.get_position()
            new_y = ref_bb.y0 + (bb.y0 - block_y0) * scale
            a.set_position([bb.x0, new_y, bb.width, bb.height * scale])

    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm   = 1.0 / 25.4
    fig_c = plt.figure(figsize=(180 * _mm, 235 * _mm))

    outer = fig_c.add_gridspec(
        6, 1,
        height_ratios=[1.0, 0.11, 0.95, 1.28, 0.11, 1.25],
        hspace=0.28,
        left=0.07, right=0.97, top=0.97, bottom=0.04,
    )

    gs0  = outer[0].subgridspec(1, 2, wspace=0.35)
    ax_a = fig_c.add_subplot(gs0[0])
    ax_b = fig_c.add_subplot(gs0[1])

    sp1  = fig_c.add_subplot(outer[1])
    sp1.set_axis_off()

    sub_c = fig_c.add_subfigure(outer[2])
    if bench_df is not None:
        _panel_c_fpr_curves(sub_c, bench_df, composite=True)
        sub_c.subplots_adjust(left=0.06, right=0.985,
                              top=0.82, bottom=0.22, wspace=0.18)
    else:
        ax_sc = sub_c.subplots(1, 1)
        ax_sc.text(0.5, 0.5, "Benchmark data not available",
                   ha="center", va="center", transform=ax_sc.transAxes, fontsize=6)
        ax_sc.set_axis_off()

    gs_mid = outer[3].subgridspec(1, 2, wspace=0.50, width_ratios=[1.35, 0.95])
    sub_d  = fig_c.add_subfigure(gs_mid[0])
    ax_e   = fig_c.add_subplot(gs_mid[1])
    if bench_df is not None:
        _panel_d_signal_rmse(sub_d, bench_df, composite=True)
        _panel_e_lambda_gc(ax_e, bench_df, composite=True)
    else:
        ax_e.text(0.5, 0.5, "—", ha="center", va="center",
                  transform=ax_e.transAxes, fontsize=6)
        ax_e.set_axis_off()

    sp2  = fig_c.add_subplot(outer[4])
    sp2.set_axis_off()

    gs_bot = outer[5].subgridspec(1, 2, wspace=0.38, width_ratios=[1, 1.45])
    ax_f   = fig_c.add_subplot(gs_bot[0])
    ax_g   = fig_c.add_subplot(gs_bot[1])

    _panel_a(ax_a, data)
    _panel_b(ax_b, data)
    _panel_f(ax_f, data)
    _panel_g(ax_g, tnbc_effects, other_effects, composite=True)

    fig_c.canvas.draw()
    if bench_df is not None:
        _match_height(ax_e, sub_d, height_frac=1.0)

    # Legend overrides
    for ax_tgt, loc in {ax_a: "upper right", ax_b: "lower right",
                         ax_f: "lower right", ax_g: "lower right"}.items():
        leg = ax_tgt.get_legend()
        if leg:
            handles = leg.legend_handles
            labels  = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_tgt.legend(handles=handles, labels=labels, fontsize=4.6,
                          loc=loc, frameon=True, framealpha=0.85,
                          handlelength=1, handletextpad=0.3,
                          borderpad=0.3, labelspacing=0.2)

    _cap_fontsize(fig_c, _MAX)

    _lbl_fs = 7
    for ax, lbl in [(ax_a, "A"), (ax_b, "B"), (ax_e, "E"),
                    (ax_f, "F"), (ax_g, "G")]:
        ax.text(-0.15, 1.12, lbl, transform=ax.transAxes,
                fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_c_list = sub_c.get_axes()
    if ax_c_list:
        ax_c_list[0].text(-0.02, 1.18, "C", transform=ax_c_list[0].transAxes,
                          fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_d_list = sub_d.get_axes()
    if ax_d_list:
        ax_d_list[0].text(-0.08, 1.14, "D", transform=ax_d_list[0].transAxes,
                          fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")

    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, OUTPUT_DIR, close=False)
    pdf_path = OUTPUT_DIR / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")

    del data
    clear_cache()
    gc.collect()
    print("  Figure 3 (TNBC) complete: 7 individual panels + combined (A–G)")


if __name__ == "__main__":
    apply_style()
    generate()

import pandas as pd
df = pd.read_csv("/Users/valenciai/Documents/Research/projects/sctrial_breast/sctrial/temp/simulation/sensitivity/sensitivity_combined.csv", low_memory=False)
print(df.groupby("method")["pvalue"].count())
print(df.groupby("method")["pvalue"].isna().sum())

df["n_genes"] = df["scenario"].str.extract(r"_g(\d+)")[0].astype(float)
frac = df["scenario"].str.extract(r"_f(\d+)")
df["signal_pct"] = pd.to_numeric(frac[0], errors="coerce").fillna(0)

nebula = df[df["method"] == "nebula"]
print(nebula.groupby(["n_genes", "signal_pct"])["pvalue"].count().unstack())