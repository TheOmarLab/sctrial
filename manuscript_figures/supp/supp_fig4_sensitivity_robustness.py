"""
Supplementary Figure 4 — Sensitivity, Robustness, and Benchmarking.
===================================================================

Panels A–G characterize the sensitivity and robustness of sctrial's
participant-level inference on real datasets.  NatMeth benchmark
panels H–J (pure-null FPR, runtime, faceted QQ) use the same four
methods (dreamlet, NEBULA, Wilcoxon on change scores, sctrial DiD) on
a hierarchical gamma-Poisson simulator (panel sizes 50–2000,
signal fractions 1–20%); the FPR-curve, λ_GC, and signal-RMSE
benchmark panels were promoted to Figure 3.  Panel K shows empirical
power curves on real datasets (formerly Figure 3 panel C).

Panels (letters match the composite artboard, left-to-right and top-to-bottom)
--------------------------------------------------------------------------------
  A  Analytical vs bootstrap SE (all 5 datasets, forest plot).
  B  Standardised vs unstandardised effect sizes (Melanoma).
  C  Mean vs median aggregation comparison (Melanoma).
  D  Log-transform sensitivity (Melanoma).
  E  Cell-type-stratified DiD heatmap (Melanoma).
  F  Rank-order concordance across preprocessing choices (Melanoma).
  G  Leave-one-out stability matrix (max influence, all datasets).
  --- composite row 3 (right of F|G) ---
  H  Benchmark: pure-null Type I error vs panel size.
  I  Benchmark: runtime comparison across methods.
  --- composite row 4 ---
  J  Benchmark: faceted p-value QQ plots with 95% Beta envelope (two-arm, n=40).
  K  Empirical power curves (participant subsampling; 3+2 facet grid).

Non-overlap guardrail: methodological sensitivity only, not biological claims.
"""

from __future__ import annotations

import gc
import hashlib
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy import stats as sp_stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    TrialDesign,
    add_log1p_cpm_layer,
    apply_style,
    between_arm_comparison,
    clear_cache,
    despine,
    did_table,
    get_aml,
    get_cart,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    save_panel,
    score_signatures,
    within_arm_comparison,
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

    if not composite:
        fig.suptitle("Analytical vs Bootstrap SE", fontweight="bold",
                     fontsize=11)
    # Composite: title is drawn on the parent figure via
    # _figure_title_above_subfig (SubFigure.suptitle can anchor incorrectly).


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

def _panel_rank_concordance(ax, data: dict, *, composite: bool = False):
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
    _yt_fs = 6 if composite else 8
    ax.set_yticklabels(labels, fontsize=_yt_fs)
    ax.set_xlabel(f"Spearman ρ (vs {ref_key})")
    ax.set_xlim(0, 1.15)
    ax.set_title("Rank Concordance Across Choices (Melanoma)",
                 fontweight="bold")

    _rho_lbl_fs = 5.0 if composite else 7
    for i, rho in enumerate(rhos):
        ax.text(rho + 0.02, i, f"{rho:.2f}", va="center", fontsize=_rho_lbl_fs)

    despine(ax)
    if composite:
        ax.tick_params(axis="x", labelsize=5.0)


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
# Legacy dataset config (shared by panel G LOO heatmap and panel A bootstrap)
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


# ======================================================================
# Multi-dataset power constants, loaders, and helpers (panel K)
# (duplicated from manuscript_figures/main/figure3_robustness_benchmarking.py
#  per design choice; json cache helpers renamed to *_json_* to avoid
#  collision with the pickle-based _load_cache/_save_cache above)
# ======================================================================

N_POWER_ITERATIONS = 200
POWER_ALPHA = 0.05
RNG_SEED = 42
_CODE_VERSION = "v14"

DatasetInfo = tuple[str, object, object, tuple, list[str], str]

_DATASET_TAGS: dict[str, str] = {
    "Sade-Feldman": "SF",
    "AML": "AML",
    "CAR-T": "CAR-T",
    "Vaccine": "VAX",
    "COVID-19": "COVID",
}

_METHOD_FAMILY: dict[str, str] = {
    "two_arm_did": "did_table",
    "paired": "did_table",
    "cross_sectional": "between_arm",
}

_PRESPECIFIED_ENDPOINTS = [
    "sig_Cytotoxic T Cell Activity",
    "sig_Interferon Response",
    "sig_Immune Exhaustion",
    "sig_T Cell Activation",
    "sig_Inflammatory Response",
]

_DATASET_PRIMARY_ENDPOINT: dict[str, str] = {
    "Sade-Feldman": "sig_Interferon Response",
    "AML":          "sig_Cytotoxic T Cell Activity",
    "CAR-T":        "sig_Cytotoxic T Cell Activity",
    "Vaccine":      "sig_Cytotoxic T Cell Activity",
    "COVID-19":     "sig_Inflammatory Response",
}

DATASET_COLORS = {
    "Sade-Feldman": COLORS["control"],
    "Vaccine":      COLORS["treated"],
    "AML":          COLORS["success"],
    "CAR-T":        COLORS["neutral"],
    "COVID-19":     COLORS["highlight"],
}

_DATASET_DISPLAY_NAMES: dict[str, str] = {
    "Sade-Feldman": "Melanoma",
}

SF_DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)
SF_VISITS: tuple[str, str] = ("Pre", "Post")


# ---- JSON cache helpers (separate from pickle cache above) -------------

_JSON_CACHE_DIR = Path(__file__).resolve().parent.parent / "_cache"


def _cache_key_json(*args: str) -> str:
    payload = "|".join([_CODE_VERSION] + list(args))
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _load_json_cache(tag: str) -> pd.DataFrame | None:
    path = _JSON_CACHE_DIR / f"{tag}.json"
    if path.exists():
        try:
            return pd.read_json(path, orient="records")
        except Exception:
            return None
    return None


def _save_json_cache(tag: str, df: pd.DataFrame) -> None:
    _JSON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(_JSON_CACHE_DIR / f"{tag}.json", orient="records", indent=2)


# ---- Per-dataset loaders (memory-efficient, one at a time) -------------

def _load_sf() -> DatasetInfo:
    sf = get_sade_feldman()
    sf = harmonize_response(sf)
    sf, sf_sigs = score_signatures(sf, layer="log1p_tpm")
    return ("Sade-Feldman", sf, SF_DESIGN, SF_VISITS, sf_sigs, "two_arm_did")


def _load_vaccine() -> DatasetInfo:
    vacc = get_vaccine()
    vacc, vacc_sigs = score_signatures(vacc, layer="counts")
    vacc_design = TrialDesign(
        participant_col="participant_id", visit_col="visit", arm_col=None,
    )
    return ("Vaccine", vacc, vacc_design, ("Pre", "Post"), vacc_sigs, "paired")


def _load_aml() -> DatasetInfo:
    aml = get_aml()
    aml, aml_sigs = score_signatures(aml, layer="counts")
    pid_col = "participant_id" if "participant_id" in aml.obs.columns else "patient_id"
    aml_design = TrialDesign(
        participant_col=pid_col, visit_col="visit", arm_col=None,
    )
    return ("AML", aml, aml_design, ("Pre", "Post"), aml_sigs, "paired")


def _load_cart() -> DatasetInfo:
    cart = get_cart()
    cart, cart_sigs = score_signatures(cart, layer="counts")
    pid_col = "participant_id" if "participant_id" in cart.obs.columns else "patient_id"
    cart_design = TrialDesign(
        participant_col=pid_col, visit_col="visit", arm_col=None,
    )
    return ("CAR-T", cart, cart_design, ("Pre", "Post"), cart_sigs, "paired")


def _load_covid() -> DatasetInfo:
    covid = get_stephenson()
    covid, covid_sigs = score_signatures(covid, layer="counts")
    if "dfo_bin" in covid.obs.columns:
        top_bin = covid.obs["dfo_bin"].value_counts().idxmax()
    else:
        top_bin = "Pre"
    covid_design = TrialDesign(
        participant_col="participant_id",
        visit_col="dfo_bin",
        arm_col="severity",
        arm_treated="Severe",
        arm_control="Mild",
    )
    return ("COVID-19", covid, covid_design, (top_bin,), covid_sigs, "cross_sectional")


def _load_dataset_by_index(idx: int) -> DatasetInfo | None:
    """Load a single dataset by index (0-4), returning None on failure."""
    loaders = [_load_sf, _load_vaccine, _load_aml, _load_cart, _load_covid]
    if idx >= len(loaders):
        return None
    try:
        return loaders[idx]()
    except Exception as exc:
        print(f"    Dataset {idx}: FAILED to load ({exc})")
        return None


# ---- Power computation helpers -----------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half_width = (
        z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    )
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def _stratified_subsample_pids(
    adata,
    design: "TrialDesign",
    dtype: str,
    n_sub: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Subsample participants preserving arm/visit balance where possible."""
    pid_col = design.participant_col
    all_pids = adata.obs[pid_col].unique()

    if dtype in ("two_arm_did", "cross_sectional") and n_sub >= 6:
        arm_pids: dict[str, np.ndarray] = {}
        for arm in [design.arm_treated, design.arm_control]:
            arm_pids[arm] = adata.obs.loc[
                adata.obs[design.arm_col] == arm, pid_col
            ].unique()
        n_t = len(arm_pids[design.arm_treated])
        n_c = len(arm_pids[design.arm_control])
        frac_t = n_t / (n_t + n_c)
        n_sub_t = max(3, round(n_sub * frac_t))
        n_sub_c = n_sub - n_sub_t
        if n_sub_c < 3:
            n_sub_c = 3
            n_sub_t = n_sub - n_sub_c
        n_sub_t = min(n_sub_t, n_t)
        n_sub_c = min(n_sub_c, n_c)
        if n_sub_t < 3 or n_sub_c < 3:
            return None
        sampled = np.concatenate([
            rng.choice(arm_pids[design.arm_treated], n_sub_t, replace=False),
            rng.choice(arm_pids[design.arm_control], n_sub_c, replace=False),
        ])
        return sampled

    return rng.choice(all_pids, size=min(n_sub, len(all_pids)), replace=False)


def _compute_subsampling_power(
    datasets: list[DatasetInfo],
) -> pd.DataFrame:
    """Compute empirical power via participant subsampling on real datasets."""
    cache_tag = "power_" + _cache_key_json(
        *[f"{n}:{a.n_obs}" for n, a, *_ in datasets],
        str(N_POWER_ITERATIONS),
    )
    cached = _load_json_cache(cache_tag)
    if cached is not None:
        print("  Empirical power curves (cached)")
        return cached

    print("  Computing empirical power curves (participant subsampling) ...")
    rng = np.random.default_rng(RNG_SEED)
    records: list[dict] = []
    import warnings as _warnings

    for name, adata, design, visits, sigs, dtype in datasets:
        pid_col = design.participant_col
        vis_col = design.visit_col

        if dtype in ("paired", "two_arm_did"):
            pid_visits = adata.obs.groupby(pid_col, observed=True)[vis_col].apply(set)
            analyzable_pids = pid_visits[
                pid_visits.apply(lambda s: set(visits).issubset(s))
            ].index.tolist()
        else:
            analyzable_pids = adata.obs[pid_col].unique().tolist()

        n_total = len(analyzable_pids)
        if n_total < 4:
            print(f"    {name}: too few analyzable participants ({n_total}), skipping")
            continue

        adata = adata[adata.obs[pid_col].isin(analyzable_pids)].copy()

        feat = _DATASET_PRIMARY_ENDPOINT.get(name)
        if feat and feat in sigs and feat in adata.obs.columns:
            pass
        else:
            feat = next((ep for ep in _PRESPECIFIED_ENDPOINTS
                         if ep in sigs and ep in adata.obs.columns), None)
        if feat is None:
            print(f"    {name}: no valid endpoint available, skipping")
            continue
        print(f"      Primary endpoint: {feat}")

        adata = adata[:, adata.var_names[:1]].copy()
        gc.collect()

        min_sub = 6 if dtype in ("two_arm_did", "cross_sectional") else 4
        max_sub = n_total - 1 if n_total > min_sub else n_total
        sub_sizes = sorted(set(
            [min_sub, min_sub + 2, min_sub + 4]
            + list(range(min_sub, min(max_sub, 30) + 1, 5))
            + [max_sub]
        ))
        sub_sizes = [s for s in sub_sizes if min_sub <= s <= max_sub]

        if adata.n_obs > 100_000:
            n_iter = 100
        elif adata.n_obs > 50_000:
            n_iter = N_POWER_ITERATIONS // 2
        else:
            n_iter = N_POWER_ITERATIONS

        print(f"    {name} ({n_total} ppts, {dtype}, feat={feat}, "
              f"n_iter={n_iter}): ", end="", flush=True)

        for n_sub in sub_sizes:
            n_sig = 0
            n_valid = 0
            n_fail = 0
            for _ in range(n_iter):
                sampled_pids = _stratified_subsample_pids(
                    adata, design, dtype, n_sub, rng,
                )
                if sampled_pids is None:
                    n_fail += 1
                    continue
                mask = adata.obs[pid_col].isin(sampled_pids)
                sub_adata = adata[mask]

                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    try:
                        if dtype == "cross_sectional":
                            res = between_arm_comparison(
                                sub_adata,
                                visit=visits[0],
                                features=[feat],
                                design=design,
                                aggregate="participant_visit",
                                standardize=True,
                            )
                        elif dtype == "paired":
                            res = within_arm_comparison(
                                sub_adata,
                                arm="All",
                                features=[feat],
                                design=design,
                                visits=visits,
                                aggregate="participant_visit",
                                standardize=True,
                            )
                        else:
                            res = did_table(
                                sub_adata,
                                features=[feat],
                                design=design,
                                visits=visits,
                                aggregate="participant_visit",
                                standardize=True,
                            )
                        p_col = next(
                            (c for c in ("p_time", "p_DiD", "p_arm", "p_value")
                             if c in res.columns),
                            None,
                        )
                        if p_col is None or res.empty:
                            n_fail += 1
                            continue
                        n_valid += 1
                        if res[p_col].iloc[0] < POWER_ALPHA:
                            n_sig += 1
                    except Exception:
                        n_fail += 1

            gc.collect()
            power = n_sig / n_iter
            records.append({
                "n_participants": n_sub,
                "dataset": name,
                "power": power,
                "n_valid": n_valid,
                "n_failures": n_fail,
                "feature": feat,
                "design_type": dtype,
                "n_analyzable": n_total,
            })

        ds_records = [r for r in records if r["dataset"] == name]
        powers = [r["power"] for r in ds_records if not np.isnan(r["power"])]
        total_fail = sum(r["n_failures"] for r in ds_records)
        if powers:
            print(f"range [{min(powers):.2f}, {max(powers):.2f}], "
                  f"{total_fail} fit failures")
        else:
            print("no valid results")

    power_df = pd.DataFrame(records)
    _save_json_cache(cache_tag, power_df)
    return power_df


def _prepare_power_data() -> dict:
    """Load each dataset, compute power curves, free dataset between iterations."""
    print("  Loading datasets one at a time for power curves ...")

    power_frames: list[pd.DataFrame] = []
    for idx in range(5):
        ds = _load_dataset_by_index(idx)
        if ds is None:
            continue
        name = ds[0]
        print(f"  {name}: {ds[1].n_obs:,} cells, {ds[1].n_vars:,} genes")

        pdf = _compute_subsampling_power([ds])
        if pdf is not None and not pdf.empty:
            power_frames.append(pdf)

        del ds
        gc.collect()

    return {
        "power_df": pd.concat(power_frames, ignore_index=True) if power_frames else pd.DataFrame(),
    }


# ---- Power-curve panel functions ---------------------------------------

def _panel_power_grid(
    fig: plt.Figure,
    gs_parent,
    data: dict,
    *,
    inner_wspace: float = 0.30,
    composite: bool = False,
) -> list[plt.Axes]:
    """Draw power curves into a gridspec area, returning created axes.

    Styling is intentionally matched to Figure 3C (legacy `fig3_c.py`), with
    only the panel arrangement changed to support a 2-row 3+2 layout when
    five datasets are present.
    """
    from sklearn.isotonic import IsotonicRegression
    from matplotlib.ticker import MaxNLocator

    power_data = data.get("power_data")
    if power_data is None:
        ax = fig.add_subplot(gs_parent)
        ax.text(0.5, 0.5, "Power data unavailable",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return [ax]

    power_df = power_data["power_df"]
    if power_df.empty:
        ax = fig.add_subplot(gs_parent)
        ax.text(0.5, 0.5, "No power data",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return [ax]

    _DESIGN_LABELS = {
        "two_arm_did": "Two-arm DiD",
        "paired": "Paired pre/post",
        "cross_sectional": "Cross-sectional",
    }

    ds_names = list(dict.fromkeys(power_df["dataset"]))
    n_ds = len(ds_names)

    if n_ds == 5:
        _hspace = 1.08 if composite else 1.3
        gs_inner = gs_parent.subgridspec(
            2, 8,
            wspace=inner_wspace,
            hspace=_hspace,
            height_ratios=[1.0, 1.0],
        )
        slots = [
            gs_inner[0, 0:2], gs_inner[0, 3:5], gs_inner[0, 6:8],
            gs_inner[1, 1:3], gs_inner[1, 5:7],
        ]
    else:
        gs_inner = gs_parent.subgridspec(1, n_ds, wspace=inner_wspace)
        slots = [gs_inner[0, i] for i in range(n_ds)]

    axes: list[plt.Axes] = []

    for i, ds_name in enumerate(ds_names):
        ax = fig.add_subplot(slots[i])
        axes.append(ax)
        grp = power_df[power_df["dataset"] == ds_name].sort_values("n_participants")
        color = DATASET_COLORS.get(ds_name, COLORS["gray"])

        x = grp["n_participants"].values.astype(float)
        y = grp["power"].values.astype(float)

        ci_lo, ci_hi = [], []
        for _, row in grp.iterrows():
            n_total_iter = int(row.get("n_valid", 0)) + int(row.get("n_failures", 0))
            if n_total_iter == 0:
                n_total_iter = N_POWER_ITERATIONS
            k = round(row["power"] * n_total_iter) if not np.isnan(row["power"]) else 0
            lo, hi = _wilson_ci(k, n_total_iter)
            ci_lo.append(lo)
            ci_hi.append(hi)

        ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.15, zorder=1, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2.0, zorder=3,
                solid_capstyle="round", marker="o", markersize=3.5)

        if len(x) >= 3:
            iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0,
                                     out_of_bounds="clip")
            y_iso = iso.fit_transform(x, y)
            ax.plot(x, y_iso, color=color, linewidth=1.2, linestyle="--",
                    alpha=0.5, zorder=2)

        ax.axhline(0.80, color="#bbb", linewidth=0.7,
                   linestyle="--", zorder=1, alpha=0.5)
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_ylim(-0.02, 1.05)

        if "feature" in grp.columns and not grp.empty:
            feat = grp["feature"].iloc[0].replace("sig_", "").replace("_", " ")
        else:
            feat = ""
        dtype = grp["design_type"].iloc[0] if "design_type" in grp.columns else ""
        design_label = _DESIGN_LABELS.get(dtype, dtype)
        analyzable_n = int(grp["n_analyzable"].iloc[0]) if "n_analyzable" in grp.columns else int(x.max()) + 1

        display_name = _DATASET_DISPLAY_NAMES.get(ds_name, ds_name)
        title_lines = display_name
        if feat:
            title_lines += f"\n{feat}"
        title_lines += f"\n{design_label}, n={analyzable_n}"

        _title_fs = 3.15 if composite else 3.5
        _title_pad = 2 if composite else 3
        ax.set_title(title_lines, fontsize=_title_fs, fontweight="bold",
                     color=color, pad=_title_pad, linespacing=1.25 if composite else 1.3)

        _x_fs = 8.5 if composite else 9.5
        _y_fs = 9 if composite else 10
        _tick_fs = 7.5 if composite else 8.5
        ax.set_xlabel("Analyzable participants", fontsize=_x_fs)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(r"Power (1 − $\beta$)", fontsize=_y_fs)
        ax.tick_params(axis="both", which="major", labelsize=_tick_fs, labelleft=True)

        ax.grid(True, which="major", axis="y", color="#f0f0f0", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.spines["left"].set_linewidth(1.0)

    return axes


def _panel_power_curves(data: dict) -> plt.Figure | None:
    """Standalone figure for panel K: power curves across datasets."""
    power_data = data.get("power_data")
    if power_data is None:
        return None
    df = power_data["power_df"]
    if df.empty:
        return None

    panel_order = list(dict.fromkeys(df["dataset"]))
    n_panels = len(panel_order)
    if n_panels == 0:
        return None

    fig = plt.figure(figsize=(10.8, 6.3))
    outer = fig.add_gridspec(1, 1, left=0.07, right=0.985, top=0.88, bottom=0.12)
    _panel_power_grid(fig, outer[0], data, inner_wspace=0.26)
    fig.tight_layout()
    return fig


# ======================================================================
# NatMeth signal-fraction benchmark CSV (H: pure-null FPR; I: runtime; J: QQ)
# ======================================================================

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
            "Run the signal-fraction sensitivity benchmark on HPC first, "
            "then rsync results locally."
        )
    df = pd.read_csv(_BENCHMARK_CSV, low_memory=False)
    df["n_genes"] = df["scenario"].str.extract(r"_g(\d+)")[0].astype(int)
    frac = df["scenario"].str.extract(r"_f(\d+)")
    df["signal_pct"] = pd.to_numeric(frac[0], errors="coerce").fillna(0).astype(int)
    df["is_null_scenario"] = df["scenario"].str.contains("sens_null")
    df["n_genes"] = df["scenario"].str.extract(r"_g(\d+)")[0].astype(int)
    frac = df["scenario"].str.extract(r"_f(\d+)")
    df["signal_pct"] = pd.to_numeric(frac[0], errors="coerce").fillna(0).astype(int)
    df["is_null_scenario"] = df["scenario"].str.contains("sens_null")
    return df


def _method_style(method, is_focal=False, alpha=1.0, *, composite=False):
    """Line/marker style for benchmark method curves.

    *composite*: smaller glyphs for the tight multi-panel artboard.
    """
    if composite:
        ms_hi, ms_lo = 5.6, 4.3
        lw_hi, lw_lo = 1.45, 1.1
        mew = 0.48
    else:
        ms_hi, ms_lo = 9, 7
        lw_hi, lw_lo = 2.5, 1.8
        mew = 0.6
    return {
        "color": _BENCH_METHOD_COLORS[method],
        "marker": _BENCH_METHOD_MARKERS[method],
        "markersize": ms_hi if is_focal else ms_lo,
        "markeredgecolor": "white",
        "markeredgewidth": mew,
        "linewidth": lw_hi if is_focal else lw_lo,
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


# ======================================================================
# Panel J — Faceted p-value QQ plots with 95% Beta envelope
# (was panel K before reshuffle)
# ======================================================================

def _panel_bench_qq(
    fig, bench_df,
    n_genes: int = 200,
    signal_pct: int = 10,
    *,
    composite: bool = False,
    gs_parent=None,
):
    """Draw faceted null QQ panels.

    If *gs_parent* (a SubplotSpec) is provided, axes are created via a nested
    gridspec inside it — preferred for composite use because plain SubFigures in
    narrow cells stretch their internal gridspecs using figure-coord
    subplotpars and overflow neighbouring rows.
    """
    scenario_name = f"two_arm__sens_g{n_genes}_f{signal_pct}"
    sub_all = bench_df[bench_df["scenario"] == scenario_name]
    if sub_all.empty:
        print(f"    WARNING: scenario {scenario_name} not found for panel J")
        return []
    null = sub_all[sub_all["true_beta"] == 0.0]
    if hasattr(fig, "set_constrained_layout"):
        fig.set_constrained_layout(False)
    if gs_parent is not None:
        gs_inner = gs_parent.subgridspec(2, 2, hspace=0.35, wspace=0.28)
        axes = []
        ref_ax = None
        for r in range(2):
            for c in range(2):
                kw = {}
                if ref_ax is not None:
                    kw["sharex"] = ref_ax
                    kw["sharey"] = ref_ax
                ax = fig.add_subplot(gs_inner[r, c], **kw)
                if ref_ax is None:
                    ref_ax = ax
                axes.append(ax)
    elif composite:
        ax_grid = fig.subplots(
            2,
            2,
            sharex=True,
            sharey=True,
            gridspec_kw={
                "hspace": 0.35,
                "wspace": 0.28,
            },
        )
        axes = ax_grid.flatten()
    else:
        axes = fig.subplots(1, len(_BENCH_METHODS), sharex=True, sharey=True)
        if not hasattr(axes, "__len__"):
            axes = [axes]

    _sct = 5.5 if composite else 8
    _ttl_fs = 6.0 if composite else 12
    _ttl_pad = 1 if composite else 8
    _axlbl_fs = 5.1 if composite else 10
    _leg_fs = 3.95 if composite else 8

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
            exp_log, lo_env, hi_env, color="#9a9a9a",
            alpha=0.32 if composite else 0.22, zorder=1,
            label=None if composite else ("95% Beta envelope" if mi == 0 else None),
        )
        ax.scatter(
            exp_log, obs_log, s=_sct, alpha=0.55,
            color=_BENCH_METHOD_COLORS[method], edgecolors="none",
            rasterized=True, zorder=3,
        )
        lim = max(exp_log.max(), obs_log.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#333333", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)
        ax.set_title(
            _BENCH_METHOD_LABELS[method], fontsize=_ttl_fs, fontweight="bold",
            color=_BENCH_METHOD_COLORS[method], pad=_ttl_pad,
        )
        if composite:
            if mi >= 2:
                ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=_axlbl_fs)
            else:
                ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=_axlbl_fs)
            if mi == 0:
                ax.set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=_axlbl_fs)
        _style_axis(ax)
        ax.tick_params(axis="both", which="major", labelsize=_axlbl_fs - 0.5)
        if composite:
            ax.tick_params(axis="x", labelbottom=(mi >= 2))
            ax.tick_params(axis="y", labelleft=(mi in (0, 2)))
    if not composite:
        fig.suptitle(
            f"Null-gene p-value calibration at {n_genes:,} genes, {signal_pct}% signal",
            fontsize=13, fontweight="bold", y=1.03,
        )
        leg = axes[0].legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True, framealpha=0.92, edgecolor="#cccccc",
            fontsize=_leg_fs,
        )
    else:
        # Keep y-labels attached to the actual left-column y-axes so they align
        # exactly with panel-J y ticks in the composite layout.
        for yi in (0, 2):
            axes[yi].set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=_axlbl_fs)
        # Place the envelope key inside the upper-left of the first QQ panel —
        # that corner has no scatter points, so the legend is always visible
        # (subfigure-level fig.legend gets clipped behind the panel letter/title).
        leg = axes[0].legend(
            handles=[
                Patch(
                    facecolor="#9a9a9a",
                    alpha=0.32,
                    edgecolor="none",
                    label="95% Beta envelope",
                ),
            ],
            loc="upper left",
            bbox_to_anchor=(0.03, 0.98),
            ncol=1,
            frameon=True,
            framealpha=0.9,
            edgecolor="#cccccc",
            fontsize=_leg_fs,
            handleheight=0.6,
            handlelength=1.0,
            handletextpad=0.35,
            borderpad=0.3,
            borderaxespad=0.0,
        )
    if composite and leg is not None:
        leg.get_frame().set_linewidth(0.55)
    return list(axes)


# ======================================================================
# Panel H — Pure-null Type I error vs panel size
# (was panel L before reshuffle)
# ======================================================================

def _panel_bench_pure_null_fpr(ax, bench_df, *, composite: bool = False):
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
    ax.axhspan(
        0.03, 0.07, color="#d62728", alpha=0.08, zorder=0,
        label=None if composite else "Nominal 5% ± 2%",
    )
    ax.axhline(
        0.05, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1,
        label=("Nominal 5%" if composite else None),
    )
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
        _ms_hi, _ms_lo = (5.0, 3.85) if composite else (9.0, 7.2)
        _lw_hi, _lw_lo = (1.25, 0.95) if composite else (2.0, 1.4)
        _cap_w = (2.0, 0.85) if composite else (4, 1.2)
        ax.errorbar(
            xs, ys, yerr=[ys - lo, hi - ys], fmt=_BENCH_METHOD_MARKERS[method],
            markersize=_ms_hi if is_focal else _ms_lo,
            color=_BENCH_METHOD_COLORS[method],
            markerfacecolor=_BENCH_METHOD_COLORS[method], markeredgecolor="white",
            markeredgewidth=0.6 if composite else 0.8,
            ecolor=_BENCH_METHOD_COLORS[method],
            elinewidth=1.0 if composite else 1.4,
            capsize=_cap_w[0], capthick=_cap_w[1],
            linestyle="-",
            linewidth=_lw_hi if is_focal else _lw_lo,
            label=_BENCH_METHOD_LABELS[method], alpha=0.92,
            zorder=10 if is_focal else 4,
        )
    _tk_fs = 5.05 if composite else 11
    _ttl_fs = 6.0 if composite else 12
    _leg_fs = 3.45 if composite else 8

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p:,}" for p in panel_sizes], fontsize=_tk_fs, rotation=0)
    ax.set_xlim(-0.35, len(panel_sizes) - 0.65)
    ax.set_xlabel("Panel size (genes)", fontsize=_tk_fs)
    ax.set_ylabel("Pure-null Type I error (p < 0.05)", fontsize=_tk_fs)
    ax.set_ylim(0.025, 0.095 if composite else 0.085)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.tick_params(axis="y", labelsize=_tk_fs)
    ax.set_title(
        "Pure-null Type I error",
        fontsize=_ttl_fs, fontweight="bold", pad=(5 if composite else 10),
    )
    if composite:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=True, framealpha=0.93, edgecolor="#cccccc",
            fontsize=_leg_fs, columnspacing=0.75, handlelength=0.85,
            markerscale=0.55,
        )
    else:
        ax.legend(
            loc="upper left", frameon=True, framealpha=0.95,
            edgecolor="#cccccc", fontsize=_leg_fs,
        )
    _style_axis(ax)


# ======================================================================
# Panel I — Runtime comparison across methods
# (was panel N before reshuffle)
# ======================================================================

def _panel_bench_runtime(ax, bench_df, *, composite: bool = False):
    """Per-iteration runtime by method × panel size (log y).

    X-axis uses evenly-spaced categorical positions so the 4 panel sizes
    are ticked at equal intervals, independent of their raw values.
    """
    rt = (
        bench_df.groupby(["method", "scenario", "n_genes", "iteration"])[
            "runtime_seconds"
        ]
        .first()
        .reset_index()
    )

    summary = (
        rt.groupby(["method", "n_genes"])["runtime_seconds"]
        .median()
        .reset_index()
    )

    x_positions = np.arange(len(_PANEL_SIZES), dtype=float)
    n_to_x = dict(zip(_PANEL_SIZES, x_positions))

    _lbl_fs = 5.05 if composite else 11
    _ttl_fs = 6.0 if composite else 12
    _ttl_pad = 5 if composite else 10
    _leg_fs = 3.45 if composite else 9

    for method in _BENCH_METHODS:
        sub = summary[summary["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_focal = method == "sctrial_did"
        style = _method_style(method, is_focal=is_focal, composite=composite)
        xs = [n_to_x[int(n)] for n in sub["n_genes"].values]
        ax.plot(
            xs, sub["runtime_seconds"],
            label=_BENCH_METHOD_LABELS[method],
            zorder=10 if is_focal else 3,
            **style,
        )

    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p:,}" for p in _PANEL_SIZES], fontsize=_lbl_fs)
    ax.set_xlim(-0.35, len(_PANEL_SIZES) - 0.65)
    ax.set_xlabel("Panel size (genes)", fontsize=_lbl_fs)
    ax.set_ylabel("Median runtime per iteration (s)", fontsize=_lbl_fs)
    ax.set_title(
        "Computational cost", fontsize=_ttl_fs, fontweight="bold", pad=_ttl_pad,
    )
    ax.tick_params(axis="y", labelsize=_lbl_fs)
    if composite:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.72),
            frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=_leg_fs,
            markerscale=0.52, handlelength=1.0,
        )
    else:
        ax.legend(
            loc="upper left", frameon=True, framealpha=0.95,
            edgecolor="#cccccc", fontsize=_leg_fs,
            markerscale=1.0, handlelength=1.5,
        )
    _style_axis(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 4 panels (A–K) + composite.

    Layout (same order as the composite artboard):
      A  Analytical vs bootstrap SE (all 5 datasets, faceted forest plot)
      B  Standardised vs unstandardised effect sizes (Melanoma)
      C  Mean vs median aggregation comparison (Melanoma)
      D  Log-transform sensitivity (Melanoma)
      E  Cell-type-stratified DiD heatmap (Melanoma)
      F  Rank-order concordance across choices (Melanoma)
      G  Leave-one-out stability matrix (all datasets)
      H  Pure-null Type I error vs panel size (NatMeth benchmark, 4 methods)
      I  Runtime comparison (NatMeth benchmark)
      J  Faceted QQ + 95% envelope (two-arm, n=40, 200 genes, 10% signal)
      K  Empirical power curves (participant subsampling; 3+2 facet grid)

    Composite (180 mm × ≤215 mm): row1 A | row2 B|C|D|E | row3 F|G|H|I |
    row4 J|K.
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

    # ── Benchmark (panels H–J) — H: pure-null FPR; I: runtime; J: QQ ───
    print("  Loading signal-fraction sensitivity benchmark results ...")
    bench_df = _load_benchmark_data()
    print(
        f"    {len(bench_df):,} rows, "
        f"{bench_df.scenario.nunique()} scenarios, "
        f"panel sizes = {sorted(bench_df['n_genes'].unique())}"
    )

    # Panel H: Pure-null FPR
    fig_h, ax_h_ind = plt.subplots(figsize=(7.5, 4.8))
    _panel_bench_pure_null_fpr(ax_h_ind, bench_df)
    fig_h.tight_layout()
    save_panel(fig_h, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    # Panel I: Runtime
    fig_i, ax_i_ind = plt.subplots(figsize=(7.2, 5.0))
    _panel_bench_runtime(ax_i_ind, bench_df)
    fig_i.tight_layout()
    save_panel(fig_i, "panel_I", FIGURE_NAME, SUPP_OUTPUT)

    # Panel J: Faceted QQ panels
    fig_j = plt.figure(figsize=(15.0, 4.0))
    _panel_bench_qq(fig_j, bench_df, n_genes=200, signal_pct=10)
    fig_j.tight_layout()
    save_panel(fig_j, "panel_J", FIGURE_NAME, SUPP_OUTPUT)

    # ── Empirical power curves on real datasets (panel K) ─────────────
    print("  Computing empirical power curves (panel K) ...")
    try:
        power_data = _prepare_power_data()
    except Exception as exc:
        print(f"  Warning: Could not compute power curves: {exc}")
        power_data = None
    composite_data = {**data, "power_data": power_data}

    fig_k = _panel_power_curves(composite_data)
    if fig_k is not None:
        save_panel(fig_k, "panel_K_power_curves", FIGURE_NAME, SUPP_OUTPUT)

    # ==================================================================
    # Composite artboard  (180 mm × ≤215 mm)
    # ==================================================================
    #   Row 1: A (full width)
    #   Row 2: B | C | D | E
    #   Row 3: F | G | H | I  (rank concordance | LOO | pure-null FPR | runtime)
    #   Row 4: J | K  (QQ panels | power curves 3+2)
    # ==================================================================
    print("  Building composite figure ...")

    _SMALL_RC = {
        "font.size": 5,
        "axes.titlesize": 5.5,
        "axes.labelsize": _COMPOSITE_E_AXIS_LABEL_FS,
        "xtick.labelsize": _COMPOSITE_E_XTICK_FS,
        "ytick.labelsize": _COMPOSITE_E_YTICK_FS,
        "legend.fontsize": 4.25,
        "legend.title_fontsize": 4.25,
    }
    _MAX_FONT = 6.5

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
        for leg in list(getattr(fig_obj, "legends", []) or []):
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

    outer = fig_c.add_gridspec(
        4, 1,
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.42,
        left=0.065, right=0.985, top=0.97, bottom=0.068,
    )

    # ── Row 1: A (forest plot) ───────────────────────────────────────
    subfig_a = fig_c.add_subfigure(outer[0])
    if boot_ok:
        _panel_bootstrap_multi(subfig_a, boot_ok, composite=True)
    else:
        ax_a_tmp = subfig_a.subplots(1, 1)
        ax_a_tmp.text(0.5, 0.5, "No bootstrap data", ha="center",
                      va="center", transform=ax_a_tmp.transAxes)
    subfig_a.subplots_adjust(wspace=0.42, left=0.04, right=0.98, top=0.80, bottom=0.24)

    # ── Row 2: B | C | D | E ─────────────────────────────────────────
    gs_r2 = outer[1].subgridspec(1, 4, wspace=0.52)
    ax_b = fig_c.add_subplot(gs_r2[0])
    ax_cc = fig_c.add_subplot(gs_r2[1])
    ax_d = fig_c.add_subplot(gs_r2[2])
    ax_e = fig_c.add_subplot(gs_r2[3])

    _panel_std_vs_unstd(ax_b, data, composite=True)
    _panel_mean_vs_median(ax_cc, data, composite=True)
    _panel_log_sensitivity(ax_d, data, composite=True)
    _panel_ct_heatmap(ax_e, data, composite=True)

    # ── Row 3: F | G | H | I  (nested so F–G wspace can exceed G–H / H–I)
    _w_fg, _w_hi, _w_mid = 0.77, 0.32, 0.24
    gs_r3 = outer[2].subgridspec(1, 2, width_ratios=[1.72, 2.0], wspace=_w_mid)
    gs_fg = gs_r3[0].subgridspec(1, 2, width_ratios=[0.72, 1.0], wspace=_w_fg)
    gs_hi = gs_r3[1].subgridspec(1, 2, width_ratios=[0.92, 0.82], wspace=_w_hi)
    ax_f = fig_c.add_subplot(gs_fg[0])
    ax_g = fig_c.add_subplot(gs_fg[1])
    ax_pure_null = fig_c.add_subplot(gs_hi[0])
    ax_runtime = fig_c.add_subplot(gs_hi[1])

    _panel_rank_concordance(ax_f, data, composite=True)
    if loo_mat is not None:
        _draw_loo_heatmap(
            ax_g, loo_mat, annot_fs=4,
            cbar_label_fs=_COMPOSITE_G_CBAR_LABEL_FS,
        )
    else:
        ax_g.text(0.5, 0.5, "No LOO data", ha="center", va="center",
                  transform=ax_g.transAxes)
    _panel_bench_pure_null_fpr(ax_pure_null, bench_df, composite=True)
    _panel_bench_runtime(ax_runtime, bench_df, composite=True)

    # ── Row 4: J | K — plain nested gridspecs (no SubFigures).
    # SubFigures in narrow half-row cells inherit figure-coord subplotpars for
    # their internal gridspecs and overflow neighbouring rows. Direct nested
    # subgridspecs respect their parent SubplotSpec bounds → no overflow.
    #
    # _R4_INSET controls J/K vertical inset within row 4 (blank top, content,
    # blank bottom). Edit the first/last values to grow/shrink J,K together.
    _R4_INSET = (0.08, 1.0, 0.12)
    gs_r4_outer = outer[3].subgridspec(3, 1, height_ratios=list(_R4_INSET), hspace=0)
    gs_r4 = gs_r4_outer[1].subgridspec(1, 2, wspace=0.22, width_ratios=[1.0, 1.0])

    qq_axes = _panel_bench_qq(
        fig_c, bench_df, n_genes=200, signal_pct=10,
        composite=True, gs_parent=gs_r4[0],
    )

    if power_data is not None and not power_data["power_df"].empty:
        power_axes = _panel_power_grid(
            fig_c, gs_r4[1], composite_data,
            inner_wspace=0.30, composite=True,
        )
    else:
        power_axes = None
        ax_power_stub = fig_c.add_subplot(gs_r4[1])
        ax_power_stub.text(
            0.5, 0.5, "No power data available",
            ha="center", va="center",
            transform=ax_power_stub.transAxes,
            color=COLORS.get("gray", "#666"),
        )
        ax_power_stub.set_axis_off()

    fig_c.canvas.draw()

    # ── Post-processing ───────────────────────────────────────────────
    for ax_pp in fig_c.get_axes():
        leg = ax_pp.get_legend()
        if leg:
            leg.get_frame().set_alpha(0.85)
            leg.get_frame().set_edgecolor("#CCCCCC")
    _composite_subfigs = [subfig_a]
    for _sf in _composite_subfigs:
        for leg in list(getattr(_sf, "legends", []) or []):
            leg.get_frame().set_alpha(0.85)
            leg.get_frame().set_edgecolor("#CCCCCC")

    _cap_fontsize(fig_c, _MAX_FONT)
    for _sf in _composite_subfigs:
        _cap_fontsize(_sf, _MAX_FONT)

    _apply_composite_axis_typography_panel_e(fig_c)

    # Bold panel labels (placed after cap so they stay prominent)
    _lbl_fs = 9
    _lbl_xy = (-0.25, 1.12)
    _lbl_y_r3 = 1.17  # F, G, H, I — slightly above default row-2 y
    _lbl_x_left = -0.38  # B, E, F nudged further left

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
        (ax_pure_null, "H"),
        (ax_runtime, "I"),
    ]:
        _x = _lbl_x_left if lbl in ("B", "F") else _lbl_xy[0]
        _y = _lbl_y_r3 if lbl in ("F", "H", "I") else _lbl_xy[1]
        ax_lbl.text(
            _x, _y, lbl,
            transform=ax_lbl.transAxes,
            fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
        )

    def _label_subfig_panel(sf, letter: str, *, x: float = -0.2, y: float = 1.08):
        if sf is None:
            return
        ax0 = sf.get_axes()
        if ax0:
            ax0[0].text(
                x, y, letter,
                transform=ax0[0].transAxes,
                fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
            )

    def _label_axes_panel(axes_list, letter: str, *, x: float = -0.22, y: float = 1.10):
        if not axes_list:
            return
        axes_list[0].text(
            x, y, letter,
            transform=axes_list[0].transAxes,
            fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
        )

    _label_axes_panel(qq_axes, "J", x=-0.30, y=1.31)
    _label_axes_panel(power_axes, "K", x=-0.30, y=1.31)

    # E & G: heatmaps — label slightly lower to clear title/colorbar
    _heat_y = 1.08
    _heat_y_g = _lbl_y_r3
    ax_e.text(
        _lbl_x_left, _heat_y, "E",
        transform=ax_e.transAxes,
        fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
    )
    ax_g.text(
        _lbl_xy[0], _heat_y_g, "G",
        transform=ax_g.transAxes,
        fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
    )

    _figure_title_above_subfig(
        fig_c,
        subfig_a,
        "Analytical vs Bootstrap SE",
        fontsize=_SMALL_RC["axes.titlesize"],
        pad_frac=0.006,
    )
    # Anchor J's figure title just above its QQ axes (inside row 4), so the
    # text stays clear of row 3 / panel F.
    if qq_axes:
        _qq_top_y = max(ax.get_position().y1 for ax in qq_axes)
        _qq_xs = [ax.get_position().x0 for ax in qq_axes] + [
            ax.get_position().x1 for ax in qq_axes
        ]
        _qq_xc = 0.5 * (min(_qq_xs) + max(_qq_xs))
        fig_c.text(
            _qq_xc,
            _qq_top_y + 0.014,
            "Null-gene p-value calibration at 200 genes, 10% signal",
            ha="center",
            va="bottom",
            fontsize=_SMALL_RC["axes.titlesize"],
            fontweight="bold",
            transform=fig_c.transFigure,
            clip_on=False,
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
    composite_data.clear()
    del bench_df, loo_mat
    if power_data is not None:
        power_data.clear()

    clear_cache()
    gc.collect()
    print("  SuppFig4 complete: 11 individual panels + combined (A–K)\n")


if __name__ == "__main__":
    apply_style()
    generate()
