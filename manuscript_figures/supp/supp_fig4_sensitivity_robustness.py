"""
Supplementary Figure 4 — Sensitivity and Robustness.
=====================================================

Show how DiD results change under different analytical decisions.

Panels
------
  A  Analytical vs bootstrap SE (all 5 datasets, forest plot).
  B  Standardised vs unstandardised effect sizes (Melanoma).
  C  Mean vs median aggregation comparison (Melanoma).
  D  Log-transform sensitivity (Melanoma).
  E  Cell-type-stratified DiD heatmap (Melanoma).
  F  Rank-order concordance across preprocessing choices (Melanoma).
  G  Leave-one-out stability matrix (max influence, all datasets).
  --- NatMeth benchmark (sctrial_did, dreamlet, NEBULA, Wilcoxon) ---
  H  Benchmark: two-arm signal-gene power curves (faceted by n).
  I  Benchmark: null calibration FPR (dot-and-whisker, both designs).
  J  Benchmark: genomic inflation factor λ_GC (two-arm null).
  K  Benchmark: faceted p-value QQ plots with 95% Beta envelope (two-arm, n=40).
  L  Benchmark: null-gene FPR in DE scenarios (dreamlet inflation finding).
  M  Benchmark: single-arm signal-gene power curves.
  N  Benchmark: runtime comparison across methods.

Non-overlap guardrail: methodological sensitivity only, not biological claims.
"""

from __future__ import annotations

import gc
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy import stats as sp_stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    add_log1p_cpm_layer,
    apply_style,
    clear_cache,
    despine,
    get_aml,
    get_cart,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    save_panel,
)

FIGURE_NAME = "SuppFig4_sensitivity_robustness"

# Composite artboard: panel E (_panel_ct_heatmap, composite) uses these sizes.
_COMPOSITE_E_AXIS_LABEL_FS = 5
_COMPOSITE_E_XTICK_FS = 4
_COMPOSITE_E_YTICK_FS = 4.5
_COMPOSITE_G_CBAR_LABEL_FS = 3.5


def _apply_composite_axis_typography_panel_e(fig) -> None:
    """Set axis label and tick label *font sizes* to panel E values (sizes only)."""
    for ax in fig.get_axes():
        ylab = ax.yaxis.label
        _is_loo_cbar = (
            ylab is not None and ylab.get_text() == "Max LOO deviation"
        )
        xlab = ax.xaxis.label
        if xlab is not None and xlab.get_text():
            xlab.set_fontsize(_COMPOSITE_E_AXIS_LABEL_FS)
        if ylab is not None and ylab.get_text():
            if _is_loo_cbar:
                ylab.set_fontsize(_COMPOSITE_G_CBAR_LABEL_FS)
            else:
                ylab.set_fontsize(_COMPOSITE_E_AXIS_LABEL_FS)
        for _tl in ax.get_xticklabels():
            _tl.set_fontsize(_COMPOSITE_E_XTICK_FS)
        _yt_fs = (
            _COMPOSITE_G_CBAR_LABEL_FS if _is_loo_cbar else _COMPOSITE_E_YTICK_FS
        )
        for _tl in ax.get_yticklabels():
            _tl.set_fontsize(_yt_fs)


# Features for sensitivity tests
_FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19",
    "CD14", "LYZ", "NKG7",
]

_PAL = {"cell": COLORS.get("highlight", "#5B9BD5"),
        "participant": COLORS.get("treated", "#E07B54")}

# Try to import adjustText; fall back to manual offsets if unavailable
try:
    from adjustText import adjust_text as _adjust_text
    _HAS_ADJUSTTEXT = True
except ImportError:
    _HAS_ADJUSTTEXT = False


# ======================================================================
# Disk cache — avoids re-running expensive computations across sessions
# ======================================================================

def _cache_dir():
    d = SUPP_OUTPUT / f"{FIGURE_NAME}_panels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_cache(tag: str):
    """Return cached object for *tag*, or None on miss."""
    p = _cache_dir() / f"_cache_{tag}.pkl"
    if not p.exists():
        return None
    print(f"    [cache hit] {tag}")
    with open(p, "rb") as f:
        return pickle.load(f)


def _save_cache(tag: str, obj):
    """Persist *obj* under *tag*."""
    p = _cache_dir() / f"_cache_{tag}.pkl"
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    [cached]    {tag}")


def _add_gene_labels(ax, features, x_series, y_series, *, fontsize=7):
    """Add non-overlapping gene labels to a scatter plot.

    Uses adjustText when available; otherwise falls back to manual offsets.
    """
    if _HAS_ADJUSTTEXT:
        texts = []
        for feat in features:
            texts.append(
                ax.text(x_series[feat], y_series[feat], feat,
                        fontsize=fontsize, fontweight="bold", ha="left")
            )
        _adjust_text(
            texts, ax=ax,
            force_text=(2.5, 2.5), force_points=(2.5, 2.5),
            expand=(1.8, 1.8),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
        )
    else:
        for feat in features:
            ax.annotate(
                feat, (x_series[feat], y_series[feat]),
                fontsize=fontsize, fontweight="bold", alpha=0.85, ha="left",
                xytext=(5, 3), textcoords="offset points",
            )


# ======================================================================
# Data loading
# ======================================================================

def _run_sensitivity():
    """Run DiD under several preprocessing choices."""
    import sctrial

    adata = get_sade_feldman()
    adata = harmonize_response(adata)

    if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])

    design = sctrial.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="response",
        arm_treated="Responder",
        arm_control="Non-responder",
    )
    visits = ("Pre", "Post")
    feats = [f for f in _FEATURES if f in adata.var_names]

    out = {"adata": adata, "design": design, "visits": visits, "features": feats}

    # 1. Cell-level
    print("  cell-level DiD ...")
    out["cell"] = sctrial.did_table(
        adata, feats, design, visits,
        layer="log1p_tpm", aggregate="cell", standardize=True,
    )

    # 2. Participant-level (analytical SE)
    print("  participant-level DiD ...")
    out["part"] = sctrial.did_table(
        adata, feats, design, visits,
        layer="log1p_tpm", aggregate="participant_visit", standardize=True,
    )

    # 3. Participant-level bootstrap
    print("  participant bootstrap DiD ...")
    out["boot"] = sctrial.did_table(
        adata, feats, design, visits,
        layer="log1p_tpm", aggregate="participant_visit", standardize=True,
        use_bootstrap=True, n_boot=200, seed=42,
    )

    # 4. Unstandardised
    print("  unstandardised DiD ...")
    out["unstd"] = sctrial.did_table(
        adata, feats, design, visits,
        layer="log1p_tpm", aggregate="participant_visit", standardize=False,
    )

    # 5. Median aggregation
    print("  median aggregation DiD ...")
    out["median"] = sctrial.did_table(
        adata, feats, design, visits,
        layer="log1p_tpm", aggregate="participant_visit", standardize=True,
        agg="median",
    )

    # 6. Raw TPM (no log) — only if tpm layer exists
    if "tpm" in adata.layers:
        print("  raw TPM DiD ...")
        out["raw"] = sctrial.did_table(
            adata, feats, design, visits,
            layer="tpm", aggregate="participant_visit", standardize=True,
        )
    else:
        out["raw"] = None

    # 7. Cell-type stratified
    ct_col = next((c for c in ["cell_type", "celltype"]
                   if c in adata.obs.columns), None)
    ct_results = {}
    if ct_col:
        top_cts = adata.obs[ct_col].value_counts().head(5).index.tolist()
        short_feats = feats[:8]
        for ct in top_cts:
            try:
                sub = adata[adata.obs[ct_col] == ct].copy()
                # Need enough cells in both arms and visits
                if sub.n_obs < 50:
                    continue
                ct_df = sctrial.did_table(
                    sub, short_feats, design, visits,
                    layer="log1p_tpm", aggregate="participant_visit",
                    standardize=True,
                )
                ct_results[ct] = ct_df
            except Exception:
                pass
    out["ct_results"] = ct_results
    out["ct_col"] = ct_col

    return out


# ── Panel A: Multi-dataset analytical vs bootstrap SE ──────────────

def _run_multi_bootstrap():
    """Run analytical + bootstrap DiD on all 5 datasets."""
    import sctrial

    results = {}
    for name, cfg in _MDE_DATASET_CFG.items():
        try:
            adata = cfg["loader"]()
            if cfg.get("harmonize", False):
                adata = harmonize_response(adata)
            layer = cfg["layer"]
            if layer == "log1p_cpm" and "log1p_cpm" not in adata.layers:
                if "counts" in adata.layers:
                    adata = add_log1p_cpm_layer(
                        adata, counts_layer="counts", out_layer="log1p_cpm",
                    )
                else:
                    continue

            feats = [f for f in _FEATURES if f in adata.var_names]
            if len(feats) < 3:
                continue

            arm_col = cfg.get("arm_col")
            design_type = cfg.get("design", "two_arm")

            if design_type == "two_arm" and arm_col:
                design = sctrial.TrialDesign(
                    participant_col=cfg["participant_col"],
                    visit_col=cfg["visit_col"],
                    arm_col=arm_col,
                    arm_treated=cfg["arm_treated"],
                    arm_control=cfg["arm_control"],
                )
                print(f"  bootstrap {name}: analytical ...")
                part = sctrial.did_table(
                    adata, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit", standardize=True,
                )
                print(f"  bootstrap {name}: bootstrap ...")
                boot = sctrial.did_table(
                    adata, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit", standardize=True,
                    use_bootstrap=True, n_boot=200, seed=42,
                )
                results[name] = {"part": part, "boot": boot}
            else:
                # Single-arm: use within_arm_comparison
                arm_filter = cfg.get("arm_filter")
                arm_value = arm_filter or "All"
                if arm_filter and arm_col and arm_col in adata.obs.columns:
                    adata = adata[adata.obs[arm_col] == arm_filter].copy()
                # Build a design for within-arm comparison
                design = sctrial.TrialDesign(
                    participant_col=cfg["participant_col"],
                    visit_col=cfg["visit_col"],
                    arm_col=arm_col if arm_col and arm_col in adata.obs.columns else None,
                )
                print(f"  bootstrap {name}: within-arm analytical ...")
                part = sctrial.within_arm_comparison(
                    adata, arm_value, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit",
                )
                print(f"  bootstrap {name}: within-arm bootstrap ...")
                boot = sctrial.within_arm_comparison(
                    adata, arm_value, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit",
                    use_bootstrap=True, n_boot=200, seed=42,
                )
                results[name] = {"part": part, "boot": boot}
            del adata
        except Exception as exc:
            print(f"  bootstrap {name}: failed ({exc})")
    return results


def _panel_bootstrap_multi(fig, boot_data: dict, *, composite: bool = False):
    """A: Faceted forest plot — analytical vs bootstrap SE per dataset."""
    names = [n for n in boot_data if boot_data[n].get("part") is not None]
    if not names:
        ax = fig.subplots(1, 1)
        ax.text(0.5, 0.5, "No bootstrap data", ha="center", va="center",
                transform=ax.transAxes)
        return

    ncols = len(names)
    axes = fig.subplots(1, ncols, sharey=False)
    if ncols == 1:
        axes = [axes]

    _ms = 2 if composite else 3
    _elw = 0.5 if composite else 0.8
    _cs = 1.0 if composite else 1.5
    _ytick_fs = 4.5 if composite else 6
    _title_fs = 5.5 if composite else 9

    def _beta_col(df):
        for c in ("beta_DiD", "beta_delta", "beta_time", "beta"):
            if c in df.columns:
                return c
        return None

    def _se_col(df):
        for c in ("se_DiD", "se_delta", "se_time", "se"):
            if c in df.columns:
                return c
        return None

    def _se_boot_col(df):
        for c in ("se_DiD_boot", "se_delta_boot", "se_time_boot"):
            if c in df.columns:
                return c
        return None

    for ax, name in zip(axes, names):
        part = boot_data[name]["part"]
        boot = boot_data[name]["boot"]

        bc = _beta_col(part)
        sc_a = _se_col(part)
        sc_b = _se_boot_col(boot)
        if bc is None or sc_a is None or sc_b is None:
            ax.set_title(name, fontweight="bold", fontsize=_title_fs)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=_title_fs)
            continue

        df_a = part.set_index("feature")[[bc, sc_a]].rename(
            columns={bc: "beta", sc_a: "se_analytical"})
        df_b = boot.set_index("feature")[[sc_b]].rename(
            columns={sc_b: "se_boot"})
        df = df_a.join(df_b, how="inner").reset_index()
        df = df.sort_values("beta", ascending=True).reset_index(drop=True)

        y = np.arange(len(df))
        off = 0.15

        ax.errorbar(df["beta"], y - off, xerr=1.96 * df["se_analytical"],
                     fmt="s", markersize=_ms, color=_PAL["cell"],
                     elinewidth=_elw, capsize=_cs, label="Analytical")
        ax.errorbar(df["beta"], y + off, xerr=1.96 * df["se_boot"],
                     fmt="o", markersize=_ms, color=_PAL["participant"],
                     elinewidth=_elw, capsize=_cs, label="Bootstrap")
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(df["feature"], fontsize=_ytick_fs)
        ax.set_title(name, fontweight="bold", fontsize=_title_fs)

        if composite:
            if name == "Vaccine":
                ax.legend(fontsize=4, loc="lower right", frameon=True)
            if name == "CAR-T":
                ax.set_xlabel("β with 95% CI", fontsize=5)
            else:
                ax.set_xlabel("")
        else:
            if ax == axes[0]:
                ax.set_xlabel("β with 95% CI", fontsize=8)
                ax.legend(fontsize=6, loc="lower right", frameon=True)
            else:
                ax.set_xlabel("")
        despine(ax)

    if composite:
        fig.suptitle("Analytical vs Bootstrap SE", fontweight="bold",
                     fontsize=5.5, y=1.06)
    else:
        fig.suptitle("Analytical vs Bootstrap SE", fontweight="bold",
                     fontsize=11)


# ── Panel C: Standardised vs Unstandardised ───────────────────────

def _panel_std_vs_unstd(ax, data: dict, *, composite: bool = False):
    """Scatter: standardised vs unstandardised effect sizes."""
    std = data["part"].set_index("feature")["beta_DiD"]
    unstd = data["unstd"].set_index("feature")["beta_DiD"]
    common = std.index.intersection(unstd.index)
    if len(common) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    _s = 18 if composite else 50
    _lbl_fs = 4.5 if composite else 7
    _r_fs = 5.5 if composite else 7

    x, y = std[common].values, unstd[common].values
    ax.scatter(x, y, s=_s, alpha=0.85, color=COLORS.get("treated", "#E07B54"),
               edgecolors="grey", linewidth=0.3)

    _add_gene_labels(ax, common, std, unstd, fontsize=_lbl_fs)

    r, _ = sp_stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=_r_fs, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.8))
    ax.set_xlabel("β (standardised)")
    ax.set_ylabel("β (unstandardised)")
    ax.set_title("Standardised vs Unstandardised (Melanoma)",
                 fontweight="bold")
    despine(ax)


# ── Panel D: Mean vs Median aggregation ───────────────────────────

def _panel_mean_vs_median(ax, data: dict, *, composite: bool = False):
    """Scatter: mean-aggregation vs median-aggregation betas."""
    mean_df = data["part"].set_index("feature")["beta_DiD"]
    med_res = data.get("median")

    if med_res is None or med_res.empty:
        ax.text(0.5, 0.5, "No median-aggregation results", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Mean vs Median Aggregation (Melanoma)",
                 fontweight="bold")
        despine(ax)
        return

    med_df = med_res.set_index("feature")["beta_DiD"]
    common = mean_df.index.intersection(med_df.index)
    mask = np.isfinite(mean_df[common]) & np.isfinite(med_df[common])
    common = common[mask]
    if len(common) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Mean vs Median Aggregation (Melanoma)",
                 fontweight="bold")
        despine(ax)
        return

    _s = 18 if composite else 50
    _lbl_fs = 4.5 if composite else 7
    _r_fs = 5.5 if composite else 7

    x, y = mean_df[common].values, med_df[common].values
    ax.scatter(x, y, s=_s, alpha=0.85, color="#7B68EE",
               edgecolors="grey", linewidth=0.3)

    _add_gene_labels(ax, common, mean_df, med_df, fontsize=_lbl_fs)

    lims = [min(min(x), min(y)) - 0.1, max(max(x), max(y)) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.3)
    r, _ = sp_stats.pearsonr(x, y)

    if composite:
        ax.text(0.95, 0.05, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=_r_fs, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#ccc", alpha=0.8))
        ax.set_xlim(right=0)
    else:
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=_r_fs, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#ccc", alpha=0.8))

    ax.set_xlabel("β (mean aggregation)")
    ax.set_ylabel("β (median aggregation)")
    ax.set_title("Mean vs Median Aggregation (Melanoma)",
                 fontweight="bold")
    despine(ax)


# ── Panel E: Log-transform sensitivity ────────────────────────────

def _panel_log_sensitivity(ax, data: dict, *, composite: bool = False):
    """Scatter: log1p_tpm betas vs raw TPM betas."""
    log_df = data["part"].set_index("feature")["beta_DiD"]
    raw_res = data.get("raw")
    if raw_res is None or raw_res.empty:
        ax.text(0.5, 0.5, "No raw-TPM results", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Log-Transform Sensitivity (Melanoma)",
                 fontweight="bold")
        despine(ax)
        return

    raw_df = raw_res.set_index("feature")["beta_DiD"]
    common = log_df.index.intersection(raw_df.index)
    if len(common) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    _s = 18 if composite else 50
    _lbl_fs = 4.5 if composite else 7
    _r_fs = 5.5 if composite else 7

    x, y = log_df[common].values, raw_df[common].values
    ax.scatter(x, y, s=_s, alpha=0.85, color="#2ECC71",
               edgecolors="grey", linewidth=0.3)

    _add_gene_labels(ax, common, log_df, raw_df, fontsize=_lbl_fs)

    r, _ = sp_stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=_r_fs, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.8))
    ax.set_xlabel("β (log1p TPM)")
    ax.set_ylabel("β (raw TPM)")
    ax.set_title("Log-Transform Sensitivity (Melanoma)",
                 fontweight="bold")
    despine(ax)


# ── Panel F: Cell-type stratified heatmap ─────────────────────────

def _panel_ct_heatmap(ax, data: dict, *, composite: bool = False):
    """Heatmap: DiD effect sizes stratified by top cell types."""
    ct_results = data.get("ct_results", {})
    if not ct_results:
        ax.text(0.5, 0.5, "No cell-type-stratified results", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Cell-Type Stratified DiD (Melanoma)",
                 fontweight="bold")
        despine(ax)
        return

    rows = {}
    for ct, df in ct_results.items():
        if "beta_DiD" in df.columns and "feature" in df.columns:
            rows[ct] = df.set_index("feature")["beta_DiD"]
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    mat = pd.DataFrame(rows)
    mat["_mean_abs"] = mat.abs().mean(axis=1)
    mat = mat.sort_values("_mean_abs", ascending=False).head(12).drop(columns="_mean_abs")

    _annot_fs = 4 if composite else 6
    sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, linewidths=0.5,
                linecolor="white", cbar_kws={"shrink": 0.6, "label": "β"},
                annot=True, fmt=".2f", annot_kws={"fontsize": _annot_fs})
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Feature")
    ax.set_title("Cell-Type Stratified DiD (Melanoma)",
                 fontweight="bold")
    if composite:
        ax.tick_params(axis="x", labelsize=4, rotation=25, pad=1.0)
        ax.tick_params(axis="y", labelsize=4.5)
    else:
        ax.tick_params(axis="x", labelsize=7, rotation=45, pad=1.5)
        ax.tick_params(axis="y", labelsize=7)

    # Slight extra nudge for this long label only (toward the x-axis).
    _mono_ct = "Monocyte/Macrophage"
    _dy_pt = 3.5 if composite else 4.5
    _fig = ax.figure
    for _tl in ax.get_xticklabels():
        if " ".join(_tl.get_text().split()) == _mono_ct:
            _tl.set_transform(
                _tl.get_transform()
                + mtransforms.ScaledTranslation(
                    0, _dy_pt / 72.0, _fig.dpi_scale_trans
                )
            )
            break


# ── Panel G: Rank concordance ────────────────────────────────────

def _panel_rank_concordance(ax, data: dict):
    """Bar chart: Spearman rank correlation of feature rankings
    across preprocessing choices."""
    # Get rankings from different configs
    configs = {}
    for key, label in [("cell", "Cell-level"),
                       ("part", "Participant"),
                       ("boot", "Bootstrap"),
                       ("unstd", "Unstandardised"),
                       ("median", "Median agg")]:
        df = data.get(key)
        # Guard: skip if df is None, empty, or has too few valid betas
        if (df is not None
                and not df.empty
                and "beta_DiD" in df.columns
                and df["beta_DiD"].notna().sum() >= 2):
            configs[label] = df.set_index("feature")["beta_DiD"].rank()

    if ("raw" in data
            and data["raw"] is not None
            and not data["raw"].empty
            and "beta_DiD" in data["raw"].columns
            and data["raw"]["beta_DiD"].notna().sum() >= 2):
        configs["Raw TPM"] = data["raw"].set_index("feature")["beta_DiD"].rank()

    if len(configs) < 2:
        ax.text(0.5, 0.5, "Insufficient configs", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Use "Participant" as reference
    ref_key = "Participant"
    if ref_key not in configs:
        ref_key = list(configs.keys())[0]

    ref = configs[ref_key]
    labels, rhos = [], []
    for label, ranks in configs.items():
        if label == ref_key:
            continue
        common = ref.index.intersection(ranks.index)
        # Drop NaN ranks before correlation
        valid = ref[common].notna() & ranks[common].notna()
        common = common[valid]
        if len(common) < 3:
            continue
        rho, _ = sp_stats.spearmanr(ref[common], ranks[common])
        if np.isnan(rho):
            continue
        labels.append(label)
        rhos.append(rho)

    if not labels:
        ax.text(0.5, 0.5, "No comparisons", ha="center", va="center",
                transform=ax.transAxes)
        return

    colors = [COLORS.get("highlight", "#5B9BD5") if r > 0.8
              else COLORS.get("treated", "#E07B54") if r > 0.5
              else "#E74C3C" for r in rhos]

    y = np.arange(len(labels))
    ax.barh(y, rhos, color=colors, edgecolor="white", alpha=0.85, height=0.6)
    ax.axvline(1.0, color="grey", linewidth=0.5, alpha=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Spearman ρ (vs {ref_key})")
    ax.set_xlim(0, 1.15)
    ax.set_title("Rank Concordance Across Choices (Melanoma)",
                 fontweight="bold")

    for i, rho in enumerate(rhos):
        ax.text(rho + 0.02, i, f"{rho:.2f}", va="center", fontsize=7)

    despine(ax)


# ── Panel G: Leave-one-out stability ──────────────────────────────

def _find_beta_col(df):
    """Return the first beta column present in *df*."""
    for c in ("beta_DiD", "beta_delta", "beta_time", "beta_arm"):
        if c in df.columns:
            return c
    raise KeyError(f"No beta column in {list(df.columns)}")


def _compute_loo_data():
    """Compute LOO max-deviation matrix (features × datasets).

    Returns a DataFrame or None if no data could be computed.
    """
    import sctrial

    rows = {}
    for name, cfg in _MDE_DATASET_CFG.items():
        try:
            adata = cfg["loader"]()
            if cfg.get("harmonize", False):
                adata = harmonize_response(adata)
            layer = cfg["layer"]
            if layer == "log1p_cpm" and "log1p_cpm" not in adata.layers:
                if "counts" in adata.layers:
                    adata = add_log1p_cpm_layer(
                        adata, counts_layer="counts", out_layer="log1p_cpm",
                    )
                else:
                    continue

            feats = [f for f in _FEATURES if f in adata.var_names]
            if len(feats) < 2:
                continue

            adata = adata[:, feats].copy()

            arm_col = cfg.get("arm_col")
            design_type = cfg.get("design", "two_arm")
            pid_col = cfg["participant_col"]
            vis_col = cfg["visit_col"]

            obs = adata.obs
            pids = obs[pid_col].unique().tolist()
            if len(pids) < 4:
                continue

            if design_type == "two_arm" and arm_col:
                design = sctrial.TrialDesign(
                    participant_col=pid_col, visit_col=vis_col,
                    arm_col=arm_col, arm_treated=cfg["arm_treated"],
                    arm_control=cfg["arm_control"],
                )
                full_df = sctrial.did_table(
                    adata, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit", standardize=True,
                )
                beta_col = "beta_DiD"
            else:
                arm_filter = cfg.get("arm_filter")
                arm_value = arm_filter or "All"
                sub = adata
                if arm_filter and arm_col and arm_col in obs.columns:
                    sub = adata[obs[arm_col] == arm_filter].copy()
                design = sctrial.TrialDesign(
                    participant_col=pid_col, visit_col=vis_col,
                    arm_col=arm_col if arm_col and arm_col in obs.columns else None,
                )
                full_df = sctrial.within_arm_comparison(
                    sub, arm_value, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit",
                )
                beta_col = _find_beta_col(full_df)

            full_betas = full_df.set_index("feature")[beta_col]

            loo_betas = []
            for pid in pids:
                mask = obs[pid_col] != pid
                sub = adata[mask].copy()
                try:
                    if design_type == "two_arm" and arm_col:
                        df_loo = sctrial.did_table(
                            sub, feats, design, cfg["visits"],
                            layer=layer, aggregate="participant_visit",
                            standardize=True,
                        )
                        loo_betas.append(
                            df_loo.set_index("feature")[_find_beta_col(df_loo)]
                        )
                    else:
                        if arm_filter and arm_col and arm_col in sub.obs.columns:
                            sub = sub[sub.obs[arm_col] == arm_filter].copy()
                        loo_design = sctrial.TrialDesign(
                            participant_col=pid_col, visit_col=vis_col,
                            arm_col=arm_col if arm_col and arm_col in sub.obs.columns else None,
                        )
                        df_loo = sctrial.within_arm_comparison(
                            sub, arm_value, feats, loo_design, cfg["visits"],
                            layer=layer, aggregate="participant_visit",
                        )
                        loo_betas.append(
                            df_loo.set_index("feature")[_find_beta_col(df_loo)]
                        )
                except Exception:
                    pass

            if len(loo_betas) < 3:
                continue

            loo_mat = pd.DataFrame(loo_betas)
            deviations = loo_mat.subtract(full_betas, axis=1).abs()
            max_dev = deviations.max() / (full_betas.abs() + 0.01)
            rows[name] = max_dev
            print(f"  LOO {name}: {len(pids)} pids, {len(feats)} feats")
            del adata
        except Exception as exc:
            print(f"  LOO {name}: failed ({exc})")

    if not rows:
        return None
    return pd.DataFrame(rows)


def _draw_loo_heatmap(
    ax, mat, *, annot_fs: float = 7, cbar_label_fs: float | None = None,
):
    """Draw LOO stability heatmap on *ax*.

    If *cbar_label_fs* is set, color bar label uses that size (e.g. composite G).
    """
    sns.heatmap(mat, ax=ax, cmap="YlOrRd", linewidths=0.5,
                linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Max LOO deviation"},
                annot=True, fmt=".2f", annot_kws={"fontsize": annot_fs})
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Feature")
    ax.set_title("Leave-One-Out Stability (max influence)", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.tick_params(axis="y", labelsize=8)
    if cbar_label_fs is not None and ax.collections:
        cb = getattr(ax.collections[0], "colorbar", None)
        if cb is not None:
            cb.set_label("Max LOO deviation", fontsize=cbar_label_fs)
            cb.ax.tick_params(axis="y", labelsize=cbar_label_fs)


def _panel_loo_stability(ax):
    """G: LOO max-deviation of betas — heatmap of features × datasets."""
    mat = _compute_loo_data()
    if mat is None:
        ax.text(0.5, 0.5, "No LOO data", ha="center", va="center",
                transform=ax.transAxes)
        return
    _draw_loo_heatmap(ax, mat)


# ======================================================================
# Legacy dataset config (shared by panels G, H, and bootstrap)
# ======================================================================

_MDE_DATASET_CFG = {
    "Melanoma": {
        "design": "two_arm",
        "loader": get_sade_feldman,
        "harmonize": True,
        "layer": "log1p_tpm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_treated": "Responder",
        "arm_control": "Non-responder",
        "visits": ("Pre", "Post"),
    },
    "AML": {
        "design": "single_arm_paired",
        "loader": lambda: get_aml(),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_filter": "Treatment",
        "visits": ("Pre", "Post"),
    },
    "CAR-T": {
        "design": "single_arm_paired",
        "loader": lambda: get_cart(),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_filter": "CAR-T",
        "visits": ("Pre", "Post"),
    },
    "COVID-19": {
        "design": "two_arm",
        "loader": get_stephenson,
        "harmonize": False,
        "layer": "log1p_cpm",
        "participant_col": "participant_id",
        "visit_col": "Collection_Day",
        "arm_col": "severity",
        "arm_treated": "Severe",
        "arm_control": "Mild",
        "visits": ("D0", "D28"),
    },
    "Vaccine": {
        "design": "single_arm_paired",
        "loader": get_vaccine,
        "harmonize": False,
        "layer": "log1p_cpm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": None,
        "visits": ("Pre", "Post"),
    },
}




_BENCHMARK_CSV = (
    Path(__file__).resolve().parents[4]
    / "manuscript"
    / "benchmark"
    / "sensitivity"
    / "sensitivity_combined.csv"
)

_BENCH_METHODS = ["wilcoxon_paired", "nebula", "dreamlet", "sctrial_did"]
_BENCH_METHOD_LABELS = {
    "sctrial_did": "sctrial (DiD)",
    "dreamlet": "dreamlet",
    "nebula": "NEBULA",
    "wilcoxon_paired": "Wilcoxon (Δ scores)",
}
_BENCH_METHOD_COLORS = {
    "sctrial_did": "#1f77b4",
    "dreamlet": "#d62728",
    "nebula": "#ff7f0e",
    "wilcoxon_paired": "#2ca02c",
}
_BENCH_METHOD_MARKERS = {
    "sctrial_did": "o",
    "dreamlet": "D",
    "nebula": "s",
    "wilcoxon_paired": "^",
}

_PANEL_SIZES = [50, 200, 500, 2000]
_SIGNAL_FRACTIONS = [1, 5, 10, 20]


def _load_benchmark_data():
    if not _BENCHMARK_CSV.exists():
        raise FileNotFoundError(
            f"Benchmark results not found at {_BENCHMARK_CSV}.\n"
            "Run the signal-fraction sensitivity benchmark on HPC first, "
            "then rsync results locally."
        )
    df = pd.read_csv(_BENCHMARK_CSV, low_memory=False)
    df["n_genes"] = df["scenario"].str.extract(r"_g(\d+)")[0].astype(int)
    frac = df["scenario"].str.extract(r"_f(\d+)")
    df["signal_pct"] = pd.to_numeric(frac[0], errors="coerce").fillna(0).astype(int)
    df["is_null_scenario"] = df["scenario"].str.contains("sens_null")
    return df


def _method_style(method, is_focal=False, alpha=1.0):
    return {
        "color": _BENCH_METHOD_COLORS[method],
        "marker": _BENCH_METHOD_MARKERS[method],
        "markersize": 9 if is_focal else 7,
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
        "linewidth": 2.5 if is_focal else 1.8,
        "alpha": alpha,
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
    ax.tick_params(axis="both", which="major",
                   color="#333333", width=0.8, length=4)


def _compute_null_fpr_table(bench_df):
    null = bench_df[bench_df["true_beta"] == 0.0].copy()
    rows = []
    for (method, n_g, frac), grp in null.groupby(
        ["method", "n_genes", "signal_pct"]
    ):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({
            "method": method,
            "n_genes": int(n_g),
            "signal_pct": int(frac),
            "fpr": float((pvals < 0.05).mean()),
            "n_tests": int(len(pvals)),
        })
    return pd.DataFrame(rows)


def _compute_signal_bias_rmse_table(bench_df):
    mixed = bench_df[~bench_df["is_null_scenario"]].copy()
    sig = mixed[mixed["true_beta"] != 0.0]
    sig = sig.dropna(subset=["estimated_beta"]).copy()
    sig["err"] = sig["estimated_beta"] - sig["true_beta"]
    sig["sq_err"] = sig["err"] ** 2

    rows = []
    for (method, n_g, frac), grp in sig.groupby(
        ["method", "n_genes", "signal_pct"]
    ):
        if grp.empty:
            continue
        rows.append({
            "method": method,
            "n_genes": int(n_g),
            "signal_pct": int(frac),
            "bias": float(grp["err"].mean()),
            "rmse": float(np.sqrt(grp["sq_err"].mean())),
            "n_tests": int(len(grp)),
        })
    return pd.DataFrame(rows)


def _panel_bench_fpr_curves(fig, bench_df):
    fpr_df = _compute_null_fpr_table(bench_df)
    fpr_df = fpr_df[fpr_df["signal_pct"] > 0].copy()
    axes = fig.subplots(1, 4, sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    x_positions = np.arange(len(_SIGNAL_FRACTIONS), dtype=float)
    frac_to_x = dict(zip(_SIGNAL_FRACTIONS, x_positions))
    method_offsets = {
        "wilcoxon_paired": -0.08,
        "nebula": -0.03,
        "sctrial_did": +0.03,
        "dreamlet": +0.08,
    }

    for ax_idx, (ax, n_g) in enumerate(zip(axes, _PANEL_SIZES)):
        sub = fpr_df[fpr_df["n_genes"] == n_g]
        for method in _BENCH_METHODS:
            m = sub[sub["method"] == method].sort_values("signal_pct")
            if m.empty:
                continue
            is_focal = method == "sctrial_did"
            style = _method_style(method, is_focal=is_focal)
            x = np.array([frac_to_x[int(f)] for f in m["signal_pct"].values]) + method_offsets[method]
            ax.plot(
                x, m["fpr"],
                label=_BENCH_METHOD_LABELS[method] if ax_idx == 0 else None,
                zorder=10 if is_focal else 3,
                **style,
            )
        _add_nominal_band(ax)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{f}%" for f in _SIGNAL_FRACTIONS])
        ax.set_xlim(-0.4, len(_SIGNAL_FRACTIONS) - 0.6)
        ax.set_xlabel("Signal fraction")
        ax.set_title(f"{n_g:,} genes", fontsize=12, fontweight="bold", color="#222222", pad=8)
        ax.set_ylim(0.0, 0.7)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        _style_axis(ax)

    axes[0].set_ylabel("Null-gene FPR (p < 0.05)", fontsize=11)
    axes[0].legend(
        loc="upper left", frameon=True, framealpha=0.95,
        edgecolor="#cccccc", fontsize=8,
    )


def _panel_bench_fpr_heatmap(fig, bench_df):
    fpr_df = _compute_null_fpr_table(bench_df)
    mixed = fpr_df[fpr_df["signal_pct"] > 0].copy()
    n_methods = len(_BENCH_METHODS)
    axes = fig.subplots(1, n_methods, sharey=False)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    fractions = sorted(_SIGNAL_FRACTIONS)
    panel_sizes = sorted(_PANEL_SIZES)
    cmap = plt.cm.RdYlGn_r
    im = None

    for mi, (ax, method) in enumerate(zip(axes, _BENCH_METHODS)):
        mat = np.full((len(fractions), len(panel_sizes)), np.nan)
        for fi, frac in enumerate(fractions):
            for pi, ps in enumerate(panel_sizes):
                cell = mixed[
                    (mixed["method"] == method)
                    & (mixed["signal_pct"] == frac)
                    & (mixed["n_genes"] == ps)
                ]
                if len(cell) == 1:
                    mat[fi, pi] = cell["fpr"].iloc[0]
        im = ax.imshow(
            mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.7,
            origin="lower", interpolation="nearest",
        )
        for fi in range(len(fractions)):
            for pi in range(len(panel_sizes)):
                v = mat[fi, pi]
                if np.isnan(v):
                    continue
                text_color = "white" if (v > 0.35 or v < 0.02) else "#1a1a1a"
                ax.text(pi, fi, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, color=text_color, fontweight="bold")

        ax.set_xticks(range(len(panel_sizes)))
        ax.set_xticklabels([f"{p:,}" for p in panel_sizes], fontsize=9)
        ax.set_xlabel("Panel size (genes)", fontsize=10)
        ax.set_yticks(range(len(fractions)))
        ax.set_yticklabels([f"{f}%" for f in fractions], fontsize=9)
        if mi == 0:
            ax.set_ylabel("Signal fraction", fontsize=11, fontweight="bold")
        ax.set_title(
            _BENCH_METHOD_LABELS[method],
            fontsize=12, fontweight="bold",
            color=_BENCH_METHOD_COLORS[method], pad=8,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(0.9)
        ax.tick_params(axis="both", which="major", length=0)

    cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.015, aspect=20)
    cbar.set_label("Null-gene FPR (p < 0.05)", fontsize=10, rotation=270, labelpad=16)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("#333333")


def _panel_bench_lambda_gc(ax, bench_df):
    null_scenarios = bench_df[bench_df["is_null_scenario"]]
    pvals_pure = null_scenarios[null_scenarios["true_beta"] == 0.0]
    rows = []
    for (method, n_g), grp in pvals_pure.groupby(["method", "n_genes"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) < 50:
            continue
        chi2_obs = sp_stats.chi2.isf(pvals, df=1)
        chi2_obs = chi2_obs[np.isfinite(chi2_obs)]
        if len(chi2_obs) < 50:
            continue
        lam = float(np.median(chi2_obs) / sp_stats.chi2.ppf(0.5, df=1))
        rows.append({"method": method, "n_genes": int(n_g), "lambda_gc": lam})
    lam_df = pd.DataFrame(rows)
    x_positions = np.arange(len(_PANEL_SIZES), dtype=float)
    n_to_x = dict(zip(_PANEL_SIZES, x_positions))

    for method in _BENCH_METHODS:
        sub = lam_df[lam_df["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_focal = method == "sctrial_did"
        style = _method_style(method, is_focal=is_focal)
        xs = [n_to_x[int(n)] for n in sub["n_genes"].values]
        ax.plot(xs, sub["lambda_gc"], label=_BENCH_METHOD_LABELS[method],
                zorder=10 if is_focal else 3, **style)

    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.65, zorder=1)
    ax.axhspan(0.95, 1.05, color="#d62728", alpha=0.06, zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p:,}" for p in _PANEL_SIZES])
    ax.set_xlim(-0.35, len(_PANEL_SIZES) - 0.65)
    ax.set_xlabel("Panel size (genes)", fontsize=11)
    ax.set_ylabel(r"Genomic inflation factor ($\lambda_{\mathrm{GC}}$)", fontsize=11)
    ax.set_title("Pure-null calibration across panel sizes", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0.88, 1.18)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=9)
    _style_axis(ax)


def _match_subfig_axes_height_to_ref(
    ref_ax, subfig, *, height_frac: float = 1.0,
) -> None:
    """Scale axes inside *subfig* to match *ref_ax* height (figure-normalized).

    Faceted subfigures (power curves, QQ panels) expand axes to fill the
    subfigure's cell, so they look vertically stretched next to single-axis
    neighbours. This rescales each child axis's vertical extent and aligns the
    block's bottom to ``ref_ax``.

    *height_frac* (< 1) uses only that fraction of *ref_ax*'s height, bottom
    aligned, leaving space above (e.g. composite H/M suptitles).
    """
    pos_ref = ref_ax.get_position()
    h_ref = pos_ref.height * float(np.clip(height_frac, 0.05, 1.0))
    y0_ref = pos_ref.y0
    axes_sf = subfig.get_axes()
    if not axes_sf:
        return
    positions = [ax.get_position() for ax in axes_sf]
    y0_u = min(p.y0 for p in positions)
    y1_u = max(p.y1 for p in positions)
    h_old = y1_u - y0_u
    if h_old <= 1e-9:
        return
    for ax in axes_sf:
        p = ax.get_position()
        rel_lo = (p.y0 - y0_u) / h_old
        rel_hi = (p.y1 - y0_u) / h_old
        new_y0 = y0_ref + rel_lo * h_ref
        new_h = (rel_hi - rel_lo) * h_ref
        ax.set_position([p.x0, new_y0, p.width, new_h])


def _subfig_bbox_in_figure_coords(fig, subfig) -> mtransforms.Bbox:
    """Return *subfig* bounds in normalized figure coordinates (0–1).

    Prefer the **gridspec cell** for this subfigure (``SubplotSpec.get_position``)
    so titles sit above the correct panel (H vs M). Fall back when unavailable.
    """
    spec = getattr(subfig, "_subplotspec", None)
    if spec is not None:
        get_cell = getattr(spec, "get_position", None)
        if callable(get_cell):
            bb = None
            try:
                bb = get_cell(fig)
            except TypeError:
                try:
                    bb = get_cell(figure=fig)
                except TypeError:
                    bb = None
            if bb is not None and all(
                np.isfinite(x) for x in (bb.x0, bb.y0, bb.width, bb.height)
            ):
                return bb

    gpf = getattr(subfig, "get_position", None)
    if callable(gpf):
        pos = gpf()
        return mtransforms.Bbox.from_bounds(
            pos.x0, pos.y0, pos.width, pos.height,
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    try:
        bb_disp = subfig.get_window_extent(renderer=renderer)
        return bb_disp.transformed(fig.transFigure.inverted())
    except (AttributeError, RuntimeError):
        pass

    axes_sf = subfig.get_axes()
    if not axes_sf:
        return mtransforms.Bbox.from_extents(0.0, 0.0, 1.0, 1.0)
    u = None
    for ax in axes_sf:
        p = ax.get_position()
        bb = mtransforms.Bbox.from_bounds(p.x0, p.y0, p.width, p.height)
        u = bb if u is None else u.union(bb)
    return u


def _figure_title_above_subfig(
    fig,
    subfig,
    title: str,
    *,
    fontsize: float = 6.5,
    pad_frac: float = 0.012,
) -> None:
    """Draw *title* just above *subfig* in figure coordinates.

    ``SubFigure.suptitle`` is often clipped or invisible inside nested grids;
    anchoring to the parent figure avoids that.
    """
    pos = _subfig_bbox_in_figure_coords(fig, subfig)
    xc = pos.x0 + 0.5 * pos.width
    y = pos.y1 + pad_frac
    fig.text(
        xc,
        y,
        title,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        transform=fig.transFigure,
        clip_on=False,
    )


def _panel_bench_fpr(ax, bench_df, *, composite: bool = False):
    """Panel I: Type I error calibration — dot-and-whisker by method × n.

    Both two-arm and single-arm null scenarios. Methods grouped on y-axis,
    dots per sample size with Wilson CIs. Red vertical line at 0.05.
    """
    # Use only pure-null scenarios (null_n* and null_hetero_n*)
    null_data = bench_df[
        (bench_df["true_beta"] == 0.0)
        & (bench_df["scenario"].str.contains("null"))
    ].copy()

    rows = []
    for (method, design, n, it), grp in null_data.groupby(
        ["method", "design", "n_per_arm", "iteration"]
    for (method, design, n, it), grp in null_data.groupby(
        ["method", "design", "n_per_arm", "iteration"]
    ):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({
            "method": method, "design": design, "n_per_arm": n,
            "fpr": (pvals < 0.05).mean(),
        })
        rows.append({
            "method": method, "design": design, "n_per_arm": n,
            "fpr": (pvals < 0.05).mean(),
        })
    fpr_df = pd.DataFrame(rows)

    # Aggregate across iterations (pool both designs for cleaner plot)

    # Aggregate across iterations (pool both designs for cleaner plot)
    fpr_agg = (
        fpr_df.groupby(["method", "n_per_arm"])["fpr"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    fpr_agg["se"] = fpr_agg["std"] / np.sqrt(fpr_agg["count"])
    fpr_agg["ci_lo"] = (fpr_agg["mean"] - 1.96 * fpr_agg["se"]).clip(0)
    fpr_agg["ci_hi"] = fpr_agg["mean"] + 1.96 * fpr_agg["se"]

    ns = sorted(fpr_agg["n_per_arm"].unique())
    marker_list = ["o", "s", "D", "^", "v"]
    _ms = 1.75 if composite else 7
    _caps = 1.2 if composite else 4
    _lw_e = 1.0 if composite else 1.5
    _lw_v = 0.85 if composite else 1.2
    _ms_leg = 1.55 if composite else 6
    _yt_fs = 7 if composite else 9
    _leg_fs = 3.6 if composite else 7

    y_pos = 0
    y_ticks, y_labels = [], []
    for method in _BENCH_METHODS:
        for ni, n_val in enumerate(ns):
            sub = fpr_agg[
                (fpr_agg["method"] == method) & (fpr_agg["n_per_arm"] == n_val)
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            ax.errorbar(
                row["mean"], y_pos,
                xerr=[[row["mean"] - row["ci_lo"]], [row["ci_hi"] - row["mean"]]],
                marker=marker_list[ni % len(marker_list)],
                markersize=_ms, capsize=_caps, linewidth=_lw_e,
                color=_BENCH_METHOD_COLORS[method],
                label=f"n={n_val}" if method == _BENCH_METHODS[0] else None,
            )
            y_pos += 1
        center = y_pos - len(ns) / 2
        y_ticks.append(center)
        y_labels.append(_BENCH_METHOD_LABELS[method])
        y_pos += 0.5

    ax.axvline(0.05, color="red", linestyle="--", linewidth=_lw_v, alpha=0.8)
    ax.axvspan(0.03, 0.07, color="red", alpha=0.06)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=_yt_fs)
    ax.set_xlabel("Type I Error Rate (p < 0.05)")
    ax.set_title("Null Calibration (both designs)", fontweight="bold")
    _x0 = 0.02 if composite else 0.0
    if composite:
        ax.set_xlim(_x0, 0.10)
    else:
        ax.set_xlim(_x0, max(0.12, fpr_agg["ci_hi"].max() * 1.1))

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=marker_list[i], color="gray",
               markersize=_ms_leg, linestyle="none", label=f"n={n}")
        for i, n in enumerate(ns)
    ]
    handles.append(Line2D([0], [0], color="red", linestyle="--",
                          linewidth=_lw_v, label="Nominal 5%"))
    ax.legend(
        handles=handles, fontsize=_leg_fs, loc="upper right",
        markerscale=0.55 if composite else 1.0,
    )
    ax.invert_yaxis()
    despine(ax)


def _panel_bench_lambda(ax, bench_df, *, composite: bool = False):
    """Panel J: Genomic inflation factor (λ_GC) under null.

    λ_GC = median(χ²_obs) / 0.456 per method × n. Well-calibrated ≈ 1.0.
    Uses two-arm null scenarios only.
    """
    null_data = _filter_bench(bench_df, "two_arm", r"null_n\d+$")
    pure_null = null_data[null_data["true_beta"] == 0.0]

    rows = []
    for (method, n), grp in pure_null.groupby(["method", "n_per_arm"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        chi2_obs = sp_stats.chi2.ppf(1 - pvals.clip(1e-300, 1), df=1)
        lambda_gc = np.median(chi2_obs) / sp_stats.chi2.ppf(0.5, df=1)
        rows.append({"method": method, "n_per_arm": n, "lambda_gc": lambda_gc})

    lambda_df = pd.DataFrame(rows)
    ns = sorted(lambda_df["n_per_arm"].unique())
    _ms_hi, _ms_lo = (3.2, 2.6) if composite else (10, 8)
    _lw_hi, _lw_lo = (1.35, 0.95) if composite else (2.5, 1.8)
    _lw_ref = 0.85 if composite else 1.2
    _leg_fs = 3.6 if composite else 7  # match panel I (composite)

    for zi, method in enumerate(_BENCH_METHODS):
        sub = lambda_df[lambda_df["method"] == method].sort_values("n_per_arm")
        if sub.empty:
            continue
        is_sctrial = method == "sctrial_did"
        ax.plot(
            sub["n_per_arm"], sub["lambda_gc"],
            marker=_BENCH_METHOD_MARKERS[method],
            markersize=_ms_hi if is_sctrial else _ms_lo,
            linewidth=_lw_hi if is_sctrial else _lw_lo,
            label=_BENCH_METHOD_LABELS[method],
            color=_BENCH_METHOD_COLORS[method],
            zorder=10 if is_sctrial else zi + 1,
        )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=_lw_ref, alpha=0.7,
               label=r"Ideal ($\lambda$ = 1)")
    ax.axhspan(0.95, 1.05, color="red", alpha=0.06)
    ax.set_xlabel("Sample size (participants per arm)")
    ax.set_ylabel(r"Genomic inflation factor ($\lambda_{\mathrm{GC}}$)")
    ax.set_title("Null Calibration Summary (two-arm)", fontweight="bold")
    ax.set_xticks(ns)
    if composite:
        ax.set_ylim(0.86, 1.20)
    else:
        ax.set_ylim(0.90, 1.15)
    if composite:
        ax.legend(
            fontsize=_leg_fs, ncol=2, loc="upper right",
            markerscale=0.45,
        )
    else:
        ax.legend(fontsize=_leg_fs, markerscale=1.0)
    despine(ax)


def _panel_bench_qq(fig, bench_df, n_genes=200, signal_pct=10):
    scenario_name = f"two_arm__sens_g{n_genes}_f{signal_pct}"
    sub_all = bench_df[bench_df["scenario"] == scenario_name]
    if sub_all.empty:
        print(f"    WARNING: scenario {scenario_name} not found for panel K")
        return
    null = sub_all[sub_all["true_beta"] == 0.0]
    axes = fig.subplots(1, len(_BENCH_METHODS), sharex=True, sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for mi, (ax, method) in enumerate(zip(axes, _BENCH_METHODS)):
        pvals = (
            null.loc[null["method"] == method, "pvalue"]
            .dropna()
            .sort_values()
            .values
        )
        if len(pvals) == 0:
            continue
        n = len(pvals)
        ranks = np.arange(1, n + 1)
        expected = (ranks - 0.5) / n
        obs_log = -np.log10(pvals + 1e-300)
        exp_log = -np.log10(expected + 1e-300)
        lo_env = -np.log10(sp_stats.beta.ppf(0.975, ranks, n - ranks + 1) + 1e-300)
        hi_env = -np.log10(sp_stats.beta.ppf(0.025, ranks, n - ranks + 1) + 1e-300)
        ax.fill_between(
            exp_log, lo_env, hi_env, color="#b0b0b0", alpha=0.22, zorder=1,
            label="95% Beta envelope" if mi == 0 else None,
        )
        ax.scatter(
            exp_log, obs_log, s=8, alpha=0.55,
            color=_BENCH_METHOD_COLORS[method], edgecolors="none",
            rasterized=True, zorder=3,
        )
        lim = max(exp_log.max(), obs_log.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#333333", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)
        ax.set_title(_BENCH_METHOD_LABELS[method], fontsize=12, fontweight="bold",
                     color=_BENCH_METHOD_COLORS[method], pad=8)
        ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=10)
        if mi == 0:
            ax.set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=10)
        _style_axis(ax)
    fig.suptitle(
        f"Null-gene p-value calibration at {n_genes:,} genes, {signal_pct}% signal",
        fontsize=13, fontweight="bold", y=1.03,
    )
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=8)


def _panel_bench_pure_null_fpr(ax, bench_df):
    null = bench_df[(bench_df["is_null_scenario"]) & (bench_df["true_beta"] == 0.0)]
    rows = []
    for (method, n_g), grp in null.groupby(["method", "n_genes"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        k = int((pvals < 0.05).sum())
        n = len(pvals)
        p = k / n
        ci = sp_stats.binomtest(k, n, p=0.05).proportion_ci(confidence_level=0.95, method="wilson")
        rows.append({
            "method": method, "n_genes": int(n_g),
            "fpr": p, "ci_lo": ci.low, "ci_hi": ci.high,
        })
    df = pd.DataFrame(rows)
    ax.axhspan(0.03, 0.07, color="#d62728", alpha=0.08, zorder=0, label="Nominal 5% ± 2%")
    ax.axhline(0.05, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    panel_sizes = sorted(_PANEL_SIZES)
    x_positions = np.arange(len(panel_sizes), dtype=float)
    n_to_x = dict(zip(panel_sizes, x_positions))
    method_offsets = {
        "wilcoxon_paired": -0.09,
        "nebula": -0.03,
        "sctrial_did": +0.03,
        "dreamlet": +0.09,
    }
    for method in _BENCH_METHODS:
        sub = df[df["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        xs = np.array([n_to_x[int(n)] for n in sub["n_genes"].values]) + method_offsets[method]
        ys = sub["fpr"].values
        lo = sub["ci_lo"].values
        hi = sub["ci_hi"].values
        is_focal = method == "sctrial_did"
        ax.errorbar(
            xs, ys, yerr=[ys - lo, hi - ys], fmt=_BENCH_METHOD_MARKERS[method],
            markersize=10 if is_focal else 8, color=_BENCH_METHOD_COLORS[method],
            markerfacecolor=_BENCH_METHOD_COLORS[method], markeredgecolor="white",
            markeredgewidth=0.8, ecolor=_BENCH_METHOD_COLORS[method],
            elinewidth=1.4, capsize=4, capthick=1.2,
            linestyle="-", linewidth=2.0 if is_focal else 1.4,
            label=_BENCH_METHOD_LABELS[method], alpha=0.92,
            zorder=10 if is_focal else 4,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p:,}" for p in panel_sizes])
    ax.set_xlim(-0.35, len(panel_sizes) - 0.65)
    ax.set_xlabel("Panel size (genes)", fontsize=11)
    ax.set_ylabel("Pure-null Type I error (p < 0.05)", fontsize=11)
    ax.set_ylim(0.025, 0.085)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.set_title("All methods are calibrated under pure-null conditions", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=8)
    _style_axis(ax)


def _panel_bench_signal_rmse(fig, bench_df):
    df = _compute_signal_bias_rmse_table(bench_df)
    if hasattr(fig, "set_constrained_layout"):
        fig.set_constrained_layout(False)
    gs = fig.add_gridspec(
        2, 4,
        hspace=0.38, wspace=0.22,
        left=0.08, right=0.985, top=0.84, bottom=0.11,
    )
    bar_width = 0.20
    method_order = ["sctrial_did", "wilcoxon_paired", "nebula", "dreamlet"]
    x_positions = np.arange(len(_SIGNAL_FRACTIONS))
    bias_lo = min(df["bias"].min(), 0) - 0.02
    bias_hi = max(df["bias"].max(), 0.02) * 1.12
    rmse_hi = df["rmse"].max() * 1.18
    bias_axes = []
    rmse_axes = []
    for col, n_g in enumerate(_PANEL_SIZES):
        ax_bias = fig.add_subplot(gs[0, col])
        ax_rmse = fig.add_subplot(gs[1, col])
        bias_axes.append(ax_bias)
        rmse_axes.append(ax_rmse)
        sub = df[df["n_genes"] == n_g]
        for mi, method in enumerate(method_order):
            bias_vals = []
            rmse_vals = []
            for frac in _SIGNAL_FRACTIONS:
                cell = sub[(sub["method"] == method) & (sub["signal_pct"] == frac)]
                if len(cell):
                    bias_vals.append(float(cell["bias"].iloc[0]))
                    rmse_vals.append(float(cell["rmse"].iloc[0]))
                else:
                    bias_vals.append(np.nan)
                    rmse_vals.append(np.nan)
            offset = (mi - (len(method_order) - 1) / 2) * bar_width
            ax_bias.bar(x_positions + offset, bias_vals, bar_width,
                        color=_BENCH_METHOD_COLORS[method], edgecolor="white",
                        linewidth=0.6, zorder=3)
            ax_rmse.bar(x_positions + offset, rmse_vals, bar_width,
                        color=_BENCH_METHOD_COLORS[method], edgecolor="white",
                        linewidth=0.6, zorder=3)
        ax_bias.axhline(0.0, color="#222222", linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
        ax_bias.set_xticks(x_positions)
        ax_bias.set_xticklabels([])
        ax_bias.set_ylim(bias_lo, bias_hi)
        ax_bias.yaxis.set_major_locator(MultipleLocator(0.05))
        ax_bias.set_title(f"{n_g:,} genes", fontsize=12, fontweight="bold", color="#222222", pad=8)
        _style_axis(ax_bias)
        ax_rmse.set_xticks(x_positions)
        ax_rmse.set_xticklabels([f"{f}%" for f in _SIGNAL_FRACTIONS])
        ax_rmse.set_xlabel("Signal fraction")
        ax_rmse.set_ylim(0, rmse_hi)
        ax_rmse.yaxis.set_major_locator(MultipleLocator(0.05))
        _style_axis(ax_rmse)
        if col > 0:
            ax_bias.set_yticklabels([])
            ax_rmse.set_yticklabels([])
    bias_axes[0].set_ylabel(r"Mean bias ($\hat{\beta} - \beta$)", fontsize=11)
    rmse_axes[0].set_ylabel(r"RMSE of $\hat{\beta}$", fontsize=11)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_BENCH_METHOD_COLORS[m], edgecolor="white", linewidth=0.6,
                      label=_BENCH_METHOD_LABELS[m])
        for m in method_order
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 0.94),
               frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=9)


def _panel_bench_runtime(ax, bench_df):
    rt = (
        bench_df.groupby(["method", "scenario", "n_genes", "iteration"])["runtime_seconds"]
        .first()
        .reset_index()
    )
    summary = rt.groupby(["method", "n_genes"])["runtime_seconds"].median().reset_index()
    x_positions = np.arange(len(_PANEL_SIZES), dtype=float)
    n_to_x = dict(zip(_PANEL_SIZES, x_positions))
    for method in _BENCH_METHODS:
        sub = summary[summary["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_focal = method == "sctrial_did"
        style = _method_style(method, is_focal=is_focal)
        xs = [n_to_x[int(n)] for n in sub["n_genes"].values]
        ax.plot(xs, sub["runtime_seconds"], label=_BENCH_METHOD_LABELS[method],
                zorder=10 if is_focal else 3, **style)
    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p:,}" for p in _PANEL_SIZES])
    ax.set_xlim(-0.35, len(_PANEL_SIZES) - 0.65)
    ax.set_xlabel("Panel size (genes)", fontsize=11)
    ax.set_ylabel("Median runtime per iteration (s)", fontsize=11)
    ax.set_title("Computational cost", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=9)
    _style_axis(ax)


def _panel_bench_runtime(ax, bench_df):
    """Panel M: Runtime comparison across methods.

    Boxplot of per-iteration runtime (seconds) grouped by method.
    Runtime is recorded per method × iteration (duplicated across genes).
    """
    # Deduplicate: one runtime per method × scenario × iteration
    rt = (
        bench_df.groupby(["method", "scenario", "iteration"])["runtime_seconds"]
        .first().reset_index()
    )

    import matplotlib.patches as mpatches

    methods_data = []
    labels = []
    colors = []
    for method in _BENCH_METHODS:
        sub = rt[rt["method"] == method]["runtime_seconds"].dropna()
        if len(sub) == 0:
            continue
        methods_data.append(sub.values)
        labels.append(_BENCH_METHOD_LABELS[method])
        colors.append(_BENCH_METHOD_COLORS[method])

    bp = ax.boxplot(methods_data, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_yscale("log")
    ax.set_ylabel("Runtime per iteration (seconds, log scale)")
    ax.set_title("Computational Cost", fontweight="bold")
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 4 panels (A–N) + composite.

    Layout:
      A  Analytical vs bootstrap SE (all 5 datasets, faceted forest plot)
      B  Standardised vs unstandardised effect sizes (Melanoma)
      C  Mean vs median aggregation comparison (Melanoma)
      D  Log-transform sensitivity (Melanoma)
      E  Cell-type-stratified DiD heatmap (Melanoma)
      F  Rank-order concordance across choices (Melanoma)
      G  Leave-one-out stability matrix (all datasets)
      --- NatMeth benchmark (4 methods) ---
      H  Two-arm signal-gene power (faceted by n)
      I  Null calibration FPR (both designs)
      J  Genomic inflation λ_GC (two-arm null)
      K  Faceted QQ + 95% envelope (two-arm, n=40)
      L  Null-gene FPR in DE scenarios
      M  Single-arm signal-gene power
      N  Runtime comparison

    Composite (180×215 mm): row1 A | row2 B–E | row3 F–H | row4 I–K | row5 L–N.
    """
    print("Supplementary Figure 4: Sensitivity to Modeling and Preprocessing")

    # ── Sensitivity (panels B–F) — cache minus large adata ────────────
    data = _load_cache("sensitivity")
    if data is None:
        data = _run_sensitivity()
        cacheable = {k: v for k, v in data.items() if k != "adata"}
        _save_cache("sensitivity", cacheable)
        if "adata" in data:
            del data["adata"]
    # (cached version never contains adata — panels B–F don't need it)

    # ── Bootstrap (panel A) ───────────────────────────────────────────
    boot_data = _load_cache("bootstrap")
    if boot_data is None:
        print("  Computing multi-dataset bootstrap ...")
        boot_data = _run_multi_bootstrap()
        _save_cache("bootstrap", boot_data)
    boot_ok = {k: v for k, v in boot_data.items() if v.get("part") is not None}
    if boot_ok:
        ncols_a = len(boot_ok)
        fig = plt.figure(figsize=(4.5 * ncols_a, 6.5))
        _panel_bootstrap_multi(fig, boot_ok)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panels B–F: single-dataset sensitivity (Sade-Feldman only)
    panels_bf = [
        ("panel_B", _panel_std_vs_unstd, (7.0, 6.0)),
        ("panel_C", _panel_mean_vs_median, (7.0, 6.0)),
        ("panel_D", _panel_log_sensitivity, (7.0, 6.0)),
        ("panel_E", _panel_ct_heatmap, (8.8, 5.8)),
        ("panel_F", _panel_rank_concordance, (7.2, 5.8)),
    ]
    for panel_name, fn, size in panels_bf:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

    # ── LOO stability (panel G) ───────────────────────────────────────
    loo_mat = _load_cache("loo")
    if loo_mat is None:
        print("  Computing LOO stability ...")
        loo_mat = _compute_loo_data()
        _save_cache("loo", loo_mat)
    fig, ax = plt.subplots(figsize=(9, 6))
    if loo_mat is not None:
        _draw_loo_heatmap(ax, loo_mat)
    else:
        ax.text(0.5, 0.5, "No LOO data", ha="center", va="center",
                transform=ax.transAxes)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # ── Benchmark (panels H–N) ──────────────────────────────────────
    print("  Loading signal-fraction sensitivity benchmark results ...")
    bench_df = _load_benchmark_data()
    print(
        f"    {len(bench_df):,} rows, "
        f"{bench_df.scenario.nunique()} scenarios, "
        f"panel sizes = {sorted(bench_df['n_genes'].unique())}"
    )

    fig_h = plt.figure(figsize=(14, 4.2))
    _panel_bench_fpr_curves(fig_h, bench_df)
    fig_h.suptitle(
        "Null-gene FPR scales with signal fraction, not panel size",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig_h.tight_layout()
    save_panel(fig_h, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    fig_i = plt.figure(figsize=(15.5, 4.2))
    _panel_bench_fpr_heatmap(fig_i, bench_df)
    save_panel(fig_i, "panel_I", FIGURE_NAME, SUPP_OUTPUT)

    fig_j, ax_j = plt.subplots(figsize=(7.2, 5.0))
    _panel_bench_lambda_gc(ax_j, bench_df)
    fig_j.tight_layout()
    save_panel(fig_j, "panel_J", FIGURE_NAME, SUPP_OUTPUT)

    fig_k = plt.figure(figsize=(15.0, 4.0))
    _panel_bench_qq(fig_k, bench_df, n_genes=200, signal_pct=10)
    fig_k.tight_layout()
    save_panel(fig_k, "panel_K", FIGURE_NAME, SUPP_OUTPUT)

    fig_l, ax_l = plt.subplots(figsize=(7.5, 4.8))
    _panel_bench_pure_null_fpr(ax_l, bench_df)
    fig_l.tight_layout()
    save_panel(fig_l, "panel_L", FIGURE_NAME, SUPP_OUTPUT)

    fig_m = plt.figure(figsize=(14, 6.8))
    _panel_bench_signal_rmse(fig_m, bench_df)
    fig_m.suptitle(
        "Effect-size estimation accuracy on signal genes",
        fontsize=13, fontweight="bold", y=0.995,
    )
    save_panel(fig_m, "panel_M", FIGURE_NAME, SUPP_OUTPUT)

    fig_n, ax_n = plt.subplots(figsize=(7.2, 5.0))
    _panel_bench_runtime(ax_n, bench_df)
    fig_n.tight_layout()
    save_panel(fig_n, "panel_N", FIGURE_NAME, SUPP_OUTPUT)

    # ==================================================================
    # Composite artboard  (180 mm × ≤215 mm)
    # ==================================================================
    #   Row 1: A (full width)
    #   Row 2: B | C | D | E
    #   Row 3: F | G | H
    #   Row 4: I | J | K
    #   Row 5: L | M | N
    # ==================================================================
    print("  Building composite figure ...")

    _SMALL_RC = {
        "font.size": 5,
        "axes.titlesize": 5.5,
        "axes.labelsize": _COMPOSITE_E_AXIS_LABEL_FS,
        "xtick.labelsize": _COMPOSITE_E_XTICK_FS,
        "ytick.labelsize": _COMPOSITE_E_YTICK_FS,
        "legend.fontsize": 4,
        "legend.title_fontsize": 4,
    }
    _MAX_FONT = 6

    def _cap_fontsize(fig_obj, maximum):
        for ax_i in fig_obj.get_axes():
            for txt in ([ax_i.title, ax_i.xaxis.label, ax_i.yaxis.label]
                        + ax_i.get_xticklabels() + ax_i.get_yticklabels()
                        + ax_i.texts):
                if txt.get_fontsize() > maximum:
                    txt.set_fontsize(maximum)
            leg = ax_i.get_legend()
            if leg:
                for txt in leg.get_texts():
                    if txt.get_fontsize() > maximum:
                        txt.set_fontsize(maximum)
                t = leg.get_title()
                if t and t.get_fontsize() > maximum:
                    t.set_fontsize(maximum)

    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm = 1.0 / 25.4
    fig_c = plt.figure(figsize=(180 * _mm, 215 * _mm))

    # Top: A stacked above rows 2–4 with a *small* hspace (tighter than mid↔LMN).
    # Bottom: row 5 (L–N); gap from row 4 is outer hspace only.
    outer = fig_c.add_gridspec(
        2, 1,
        height_ratios=[0.48 + 2.15, 0.46],
        hspace=0.18,
        left=0.065, right=0.985, top=0.97, bottom=0.028,
    )
    top_stack = outer[0].subgridspec(
        2, 1,
        height_ratios=[0.48, 2.15],
        hspace=0.16,
    )
    mid = top_stack[1].subgridspec(
        3, 1,
        height_ratios=[0.44, 0.46, 0.46],
        hspace=0.52,
    )

    # ── Row 1: A (forest plot) ───────────────────────────────────────
    subfig_a = fig_c.add_subfigure(top_stack[0])
    if boot_ok:
        _panel_bootstrap_multi(subfig_a, boot_ok, composite=True)
    else:
        ax_a_tmp = subfig_a.subplots(1, 1)
        ax_a_tmp.text(0.5, 0.5, "No bootstrap data", ha="center",
                      va="center", transform=ax_a_tmp.transAxes)
    subfig_a.subplots_adjust(wspace=0.42, left=0.04, right=0.98, top=0.88, bottom=0.14)

    # ── Row 2: B | C | D | E ─────────────────────────────────────────
    gs_r2 = mid[0].subgridspec(1, 4, wspace=0.52)
    ax_b = fig_c.add_subplot(gs_r2[0])
    ax_cc = fig_c.add_subplot(gs_r2[1])
    ax_d = fig_c.add_subplot(gs_r2[2])
    ax_e = fig_c.add_subplot(gs_r2[3])

    _panel_std_vs_unstd(ax_b, data, composite=True)
    _panel_mean_vs_median(ax_cc, data, composite=True)
    _panel_log_sensitivity(ax_d, data, composite=True)
    _panel_ct_heatmap(ax_e, data, composite=True)

    # ── Row 3: F | G | H (cols 1 & 3 are spacers: tighter F–G, wider G–H)
    gs_r3 = mid[1].subgridspec(
        1, 5,
        wspace=0.32,
        width_ratios=[0.72, 0.035, 1.0, 0.12, 1.42],
    )
    ax_f = fig_c.add_subplot(gs_r3[0])
    ax_g = fig_c.add_subplot(gs_r3[2])
    subfig_h = fig_c.add_subfigure(gs_r3[4])

    _panel_rank_concordance(ax_f, data)
    if loo_mat is not None:
        _draw_loo_heatmap(
            ax_g, loo_mat, annot_fs=4,
            cbar_label_fs=_COMPOSITE_G_CBAR_LABEL_FS,
        )
    else:
        ax_g.text(0.5, 0.5, "No LOO data", ha="center", va="center",
                  transform=ax_g.transAxes)
    _panel_bench_fpr_curves(subfig_h, bench_df)
    subfig_h.subplots_adjust(left=0.11, right=0.985, top=0.82, bottom=0.18)

    # ── Row 4: I | J | K (spacers: tighter I–J, wider J–K)
    gs_r4 = mid[2].subgridspec(
        1, 5,
        wspace=0.28,
        width_ratios=[0.78, 0.012, 0.68, 0.22, 1.46],
    )
    subfig_i = fig_c.add_subfigure(gs_r4[0])
    ax_j = fig_c.add_subplot(gs_r4[2])
    subfig_k = fig_c.add_subfigure(gs_r4[4])

    _panel_bench_fpr_heatmap(subfig_i, bench_df)
    _panel_bench_lambda_gc(ax_j, bench_df)
    _panel_bench_qq(subfig_k, bench_df, n_genes=200, signal_pct=10)
    subfig_i.subplots_adjust(left=0.03, right=0.99, top=0.82, bottom=0.16)
    subfig_k.subplots_adjust(left=0.05, right=0.992, top=0.82, bottom=0.46)

    # ── Row 5: L | M | N ────────────────────────────────────────────
    gs_r5 = outer[1].subgridspec(1, 3, wspace=0.45, width_ratios=[0.9, 1.22, 0.92])
    ax_l = fig_c.add_subplot(gs_r5[0])
    subfig_m = fig_c.add_subfigure(gs_r5[1])
    ax_n = fig_c.add_subplot(gs_r5[2])

    _panel_bench_pure_null_fpr(ax_l, bench_df)
    _panel_bench_signal_rmse(subfig_m, bench_df)
    subfig_m.subplots_adjust(left=0.11, right=0.985, top=0.80, bottom=0.22)
    _panel_bench_runtime(ax_n, bench_df)
    # Final layout so get_position() is correct; shrink faceted subfigures to
    # match same-row neighbour axis height (G for H; J for K; L for M).
    fig_c.canvas.draw()
    _COMP_HM_HEIGHT_FRAC = 0.88
    _match_subfig_axes_height_to_ref(
        ax_g, subfig_h, height_frac=_COMP_HM_HEIGHT_FRAC,
    )
    _match_subfig_axes_height_to_ref(ax_j, subfig_k)
    _match_subfig_axes_height_to_ref(
        ax_l, subfig_m, height_frac=_COMP_HM_HEIGHT_FRAC,
    )

    # ── Post-processing ───────────────────────────────────────────────
    for ax_pp in fig_c.get_axes():
        leg = ax_pp.get_legend()
        if leg:
            leg.get_frame().set_alpha(0.85)
            leg.get_frame().set_edgecolor("#CCCCCC")

    _cap_fontsize(fig_c, _MAX_FONT)
    for _sf in (subfig_a, subfig_h, subfig_i, subfig_k, subfig_m):
        _cap_fontsize(_sf, _MAX_FONT)

    _apply_composite_axis_typography_panel_e(fig_c)

    # Bold panel labels (placed after cap so they stay prominent)
    _lbl_fs = 9
    _lbl_xy = (-0.25, 1.12)
    _lbl_x_left = -0.38  # B, E, F, K, M nudged further left

    subfig_axes = subfig_a.get_axes()
    if subfig_axes:
        subfig_axes[0].text(
            -0.22, 1.15, "A",
            transform=subfig_axes[0].transAxes,
            fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
        )

    for ax_lbl, lbl in [
        (ax_b, "B"), (ax_cc, "C"), (ax_d, "D"),
        (ax_f, "F"),
        (ax_j, "J"),
        (ax_l, "L"), (ax_n, "N"),
    ]:
        _x = _lbl_x_left if lbl in ("B", "F") else _lbl_xy[0]
        ax_lbl.text(
            _x, _lbl_xy[1], lbl,
            transform=ax_lbl.transAxes,
            fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
        )

    def _label_subfig_panel(sf, letter: str, *, x: float = -0.2, y: float = 1.08):
        ax0 = sf.get_axes()
        if ax0:
            ax0[0].text(
                x, y, letter,
                transform=ax0[0].transAxes,
                fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
            )

    _label_subfig_panel(subfig_h, "H", y=1.20)
    _label_subfig_panel(subfig_i, "I", x=-0.30, y=1.15)
    _label_subfig_panel(subfig_k, "K", x=-0.46, y=1.06)
    _label_subfig_panel(subfig_m, "M", x=_lbl_x_left, y=1.20)

    # E & G: heatmaps — label slightly lower to clear title/colorbar
    _heat_y = 1.08
    ax_e.text(
        _lbl_x_left, _heat_y, "E",
        transform=ax_e.transAxes,
        fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
    )
    ax_g.text(
        _lbl_xy[0], _heat_y, "G",
        transform=ax_g.transAxes,
        fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
    )

    # H/M row titles: same size as panel G axis title (axes.titlesize in composite).
    _figure_title_above_subfig(
        fig_c,
        subfig_h,
        "Null-gene FPR vs signal fraction",
        fontsize=_SMALL_RC["axes.titlesize"],
        pad_frac=0.004,
    )
    _figure_title_above_subfig(
        fig_c,
        subfig_m,
        "Effect-size estimation accuracy",
        fontsize=_SMALL_RC["axes.titlesize"],
    )

    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, SUPP_OUTPUT, close=False)
    pdf_path = SUPP_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")

    # ── Cleanup ───────────────────────────────────────────────────────
    if "adata" in data:
        del data["adata"]
    data.clear()
    boot_data.clear()
    del bench_df, loo_mat

    clear_cache()
    gc.collect()
    print("  SuppFig4 complete: 14 individual panels + combined (A–N)\n")


if __name__ == "__main__":
    apply_style()
    generate()
