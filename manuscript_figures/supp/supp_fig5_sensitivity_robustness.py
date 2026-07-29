"""
Supplementary Figure 5 — Sensitivity, Robustness, and Benchmarking.
===================================================================

Panels A–G characterize the sensitivity and robustness of sctrial's
participant-level inference on real datasets.  NatMeth benchmark
panels H–I use the same five methods (sctrial (DiD), Wilcoxon (Δ scores),
limma-voom, dreamlet, NEBULA) on a hierarchical gamma-Poisson simulator
(panel sizes 50–2000, signal fractions 2–20%: 2, 4, 10, 20%); the
λ_GC-calibration (Figure 3 panel H) and QQ-calibration benchmark panels
were promoted to Figure 3.
Panel J shows empirical power curves on real datasets.

Panels (letters match the composite artboard, left-to-right and top-to-bottom)
--------------------------------------------------------------------------------
  A  Analytical vs bootstrap SE (all 5 datasets, forest plot).
  B  Standardized vs unstandardized effect sizes (TNBC).
  C  Mean vs median aggregation comparison (TNBC).
  D  Log-transform sensitivity (TNBC).
  E  Cell-type-stratified DiD heatmap (TNBC).
  F  Rank-order concordance across preprocessing choices (TNBC).
  G  Leave-one-out stability matrix (max influence, all datasets).
  --- composite row 4 ---
  H  Benchmark: mixed-signal null-gene FPR vs signal fraction (expanded from Figure 3 panel D).
  I  Benchmark: pure-null Type I error vs panel size.
  J  Empirical power curves (participant subsampling; 3+3 facet grid).

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
    get_tnbc_zhang,
    get_vaccine,
    harmonize_response,
    save_panel,
    score_signatures,
    within_arm_comparison,
)

FIGURE_NAME = "SuppFig5_sensitivity_robustness"

# Composite artboard: panel E (_panel_ct_heatmap, composite) uses these sizes.
_COMPOSITE_E_AXIS_LABEL_FS = 5
_COMPOSITE_E_XTICK_FS = 4
_COMPOSITE_E_YTICK_FS = 4.5
_COMPOSITE_G_CBAR_LABEL_FS = 5.0


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


# Features for sensitivity tests — tuned for TNBC TME.
# Cytokines (IFNG, TNF, IL2) and CD19 were dropped: near-zero expression in most
# TNBC pseudobulks collapses median-aggregation covariance → NaN betas.
_FEATURES = [
    # Pan-immune (high, reliable expression across all immune cells)
    "PTPRC", "CD3E", "CD3D",
    # T cell / checkpoint
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4", "TIGIT",
    # Treg
    "FOXP3",
    # Cytotoxic / effector
    "GZMB", "PRF1", "NKG7", "CCL5",
    # Myeloid / innate
    "CD14", "LYZ", "S100A8", "S100A9", "CD68", "SPP1",
    # IFN / antigen-presentation pathway (anti-PDL1 relevant)
    "STAT1", "CXCL9", "CXCL10", "ISG15", "HLA-A", "B2M",
    # Tumor / proliferation
    "EPCAM", "KRT18", "MKI67",
    # PD-L1 (direct drug target) / immune evasion
    "CD274", "IDO1",
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
            force_text=(1.2, 1.2), force_static=(0.3, 0.3),
            force_explode=(0.3, 0.3),
            expand=(1.3, 1.5),
            max_move=(40, 40),
            time_lim=5,
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

    adata = get_tnbc_zhang()

    _layer = "log1p_norm"

    design = sctrial.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="anti-PDL1+Chemo",
        arm_control="Chemo",
    )
    visits = ("Pre", "Post")
    feats = [f for f in _FEATURES if f in adata.var_names]

    out = {"adata": adata, "design": design, "visits": visits, "features": feats}

    # 1. Cell-level
    print("  cell-level DiD ...")
    out["cell"] = sctrial.did_table(
        adata, feats, design, visits,
        layer=_layer, aggregate="cell", standardize=True,
    )

    # 2. Participant-level (analytical SE)
    print("  participant-level DiD ...")
    out["part"] = sctrial.did_table(
        adata, feats, design, visits,
        layer=_layer, aggregate="participant_visit", standardize=True,
    )

    # 3. Participant-level bootstrap
    print("  participant bootstrap DiD ...")
    out["boot"] = sctrial.did_table(
        adata, feats, design, visits,
        layer=_layer, aggregate="participant_visit", standardize=True,
        use_bootstrap=True, n_boot=200, seed=42,
    )

    # 4. Unstandardized
    print("  unstandardized DiD ...")
    out["unstd"] = sctrial.did_table(
        adata, feats, design, visits,
        layer=_layer, aggregate="participant_visit", standardize=False,
    )

    # 5. Median aggregation
    print("  median aggregation DiD ...")
    out["median"] = sctrial.did_table(
        adata, feats, design, visits,
        layer=_layer, aggregate="participant_visit", standardize=True,
        agg="median",
    )

    # 6. Raw counts (no log) — only if counts layer exists
    if "counts" in adata.layers:
        print("  raw counts DiD ...")
        out["raw"] = sctrial.did_table(
            adata, feats, design, visits,
            layer="counts", aggregate="participant_visit", standardize=True,
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
                    layer=_layer, aggregate="participant_visit",
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
    """Run analytical + bootstrap DiD on all datasets."""
    import sctrial

    results = {}
    for name, cfg in _MDE_DATASET_CFG.items():
        try:
            adata = cfg["loader"]()
            if cfg.get("harmonize", False):
                adata = harmonize_response(adata)
            layer = cfg["layer"]
            if layer not in adata.layers:
                if layer == "log1p_cpm" and "counts" in adata.layers:
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
            gc.collect()
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
        df = df.drop_duplicates(subset="feature")
        if len(df) > 15:
            df = df.loc[df["beta"].abs().nlargest(15).index]
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
        ax.set_xlabel("")

        if composite:
            if name == "Vaccine":
                ax.legend(fontsize=4, loc="lower right", frameon=True)
        else:
            if ax == axes[0]:
                ax.legend(fontsize=6, loc="lower right", frameon=True)
        despine(ax)

    # Centered x-axis label under the full row, rather than anchored to one
    # subplot — robust to dataset count/order changes (e.g. adding TNBC),
    # unlike hardcoding a specific dataset as the "middle" anchor.
    #
    # NOTE: in composite mode, *fig* here is a SubFigure (subfig_a). Calling
    # subfig.text(..., transform=subfig.transFigure) does NOT position
    # relative to the subfigure's own local 0-1 space as the name implies —
    # it resolves in the parent figure's coordinate frame, landing the label
    # at the bottom of the whole page instead of just below panel A. The fix
    # (same pattern as _figure_title_above_subfig elsewhere in this file) is
    # to get axes positions, which ARE local to the subfigure, then place the
    # text on the true parent figure using those same local coordinates only
    # when fig is a real top-level Figure; for a SubFigure we must instead
    # anchor using the parent figure's transform directly.
    _xlbl_fs = 5 if composite else 8
    fig.canvas.draw()
    _ax_xs = [a.get_position().x0 for a in axes] + [
        a.get_position().x1 for a in axes
    ]
    _xc = 0.5 * (min(_ax_xs) + max(_ax_xs))
    _y0 = min(a.get_position().y0 for a in axes)
    _parent_fig = getattr(fig, "figure", fig)
    if _parent_fig is not fig:
        # fig is a SubFigure: translate its local axes-position fractions
        # into true parent-figure fractions before placing text, then draw
        # on the parent figure (not the subfigure) so the transform behaves.
        _sf_pos = _subfig_bbox_in_figure_coords(_parent_fig, fig)
        _xc_parent = _sf_pos.x0 + _xc * _sf_pos.width
        _y0_parent = _sf_pos.y0 + _y0 * _sf_pos.height
        _parent_fig.text(
            _xc_parent, _y0_parent - 0.050, "β with 95% CI",
            ha="center", va="top", fontsize=_xlbl_fs,
            transform=_parent_fig.transFigure,
        )
    else:
        fig.text(
            _xc, _y0 - (0.06 if composite else 0.09), "β with 95% CI",
            ha="center", va="top", fontsize=_xlbl_fs,
            transform=fig.transFigure,
        )

    if not composite:
        fig.suptitle("Analytical vs bootstrap SE", fontweight="bold",
                     fontsize=11)
    # Composite: title is drawn on the parent figure via
    # _figure_title_above_subfig (SubFigure.suptitle can anchor incorrectly).


# ── Panel B: Standardized vs Unstandardized ───────────────────────

def _panel_std_vs_unstd(ax, data: dict, *, composite: bool = False):
    """Scatter: standardized vs unstandardized effect sizes."""
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
    ax.text(0.95, 0.05, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=_r_fs, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.8))
    ax.set_xlabel("β (standardized)")
    ax.set_ylabel("β (unstandardized)")
    ax.set_title("Standardized vs unstandardized (TNBC)",
                 fontweight="bold")
    despine(ax)


# ── Panel C: Mean vs Median aggregation ───────────────────────────

def _panel_mean_vs_median(ax, data: dict, *, composite: bool = False):
    """Scatter: mean-aggregation vs median-aggregation betas."""
    mean_df = data["part"].set_index("feature")["beta_DiD"]
    med_res = data.get("median")

    if med_res is None or med_res.empty:
        ax.text(0.5, 0.5, "No median-aggregation results", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Mean vs median aggregation (TNBC)",
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
        ax.set_title("Mean vs median aggregation (TNBC)",
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
    else:
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=_r_fs, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#ccc", alpha=0.8))

    ax.set_xlabel("β (mean aggregation)")
    ax.set_ylabel("β (median aggregation)")
    ax.set_title("Mean vs median aggregation (TNBC)",
                 fontweight="bold")
    despine(ax)


# ── Panel D: Log-transform sensitivity ────────────────────────────

def _panel_log_sensitivity(ax, data: dict, *, composite: bool = False):
    """Scatter: log1p_norm betas vs raw counts betas."""
    log_df = data["part"].set_index("feature")["beta_DiD"]
    raw_res = data.get("raw")
    if raw_res is None or raw_res.empty:
        ax.text(0.5, 0.5, "No raw counts results", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Log-transform sensitivity (TNBC)",
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
    ax.set_xlabel("β (log1p norm)")
    ax.set_ylabel("β (raw counts)")
    ax.set_title("Log-transform sensitivity (TNBC)",
                 fontweight="bold")
    despine(ax)


# ── Panel E: Cell-type stratified heatmap ─────────────────────────

def _panel_ct_heatmap(ax, data: dict, *, composite: bool = False):
    """Heatmap: DiD effect sizes stratified by top cell types."""
    ct_results = data.get("ct_results", {})
    if not ct_results:
        ax.text(0.5, 0.5, "No cell-type-stratified results", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Cell-type stratified DiD (TNBC)",
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

    _annot_fs = 5 if composite else 6
    sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, linewidths=0.5,
                linecolor="white", cbar_kws={"shrink": 0.6, "label": "β"},
                annot=True, fmt=".2f", annot_kws={"fontsize": _annot_fs})
    ax.set_ylabel("Feature")
    ax.set_title("Cell-type stratified DiD (TNBC)",
                 fontweight="bold")
    if composite:
        ax.tick_params(axis="x", labelsize=3.2, rotation=35, pad=1.0)
        ax.tick_params(axis="y", labelsize=4.5)
        for _tl in ax.get_xticklabels():
            _tl.set_ha("right")
        #ax.set_xlabel("Cell type", labelpad=-3)
    else:
        ax.tick_params(axis="x", labelsize=5.5, rotation=55, pad=1.5)
        ax.tick_params(axis="y", labelsize=7)
        for _tl in ax.get_xticklabels():
            _tl.set_ha("right")
        ax.set_xlabel("Cell type")

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


# ── Panel F: Rank concordance ────────────────────────────────────

def _panel_rank_concordance(ax, data: dict, *, composite: bool = False):
    """Bar chart: Spearman rank correlation of feature rankings
    across preprocessing choices."""
    # Get rankings from different configs
    configs = {}
    for key, label in [("cell", "Cell-level"),
                       ("part", "Participant"),
                       ("boot", "Bootstrap"),
                       ("unstd", "Unstandardized"),
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
        configs["Raw counts"] = data["raw"].set_index("feature")["beta_DiD"].rank()

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
    ax.set_title("Rank concordance across choices (TNBC)",
                 fontweight="bold", fontsize=5.5 if composite else 11)

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
            if layer not in adata.layers:
                if layer == "log1p_cpm" and "counts" in adata.layers:
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
                finally:
                    del sub
                    gc.collect()

            if len(loo_betas) < 3:
                continue

            loo_mat = pd.DataFrame(loo_betas)
            deviations = loo_mat.subtract(full_betas, axis=1).abs()
            max_dev = deviations.max() / (full_betas.abs() + 0.01)
            rows[name] = max_dev
            print(f"  LOO {name}: {len(pids)} pids, {len(feats)} feats")
            del adata
            gc.collect()
        except Exception as exc:
            print(f"  LOO {name}: failed ({exc})")

    if not rows:
        return None
    return pd.DataFrame(rows)


def _draw_loo_heatmap(
    ax, mat, *, annot_fs: float = 7, cbar_label_fs: float | None = None,
    title_fs: float | None = None, xlabel_pad: float | None = None,
):
    """Draw LOO stability heatmap on *ax*.

    If *cbar_label_fs* is set, color bar label uses that size (e.g. composite G).
    """
    sns.heatmap(mat, ax=ax, cmap="YlOrRd", linewidths=0.5,
                linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Max LOO deviation"},
                annot=True, fmt=".2f", annot_kws={"fontsize": annot_fs})
    ax.set_xlabel("Dataset", labelpad=xlabel_pad)
    ax.set_ylabel("Feature")
    ax.set_title("Leave-one-out stability (max influence)", fontweight="bold",
                 fontsize=title_fs)
    ax.tick_params(axis="x", labelsize=8, rotation=35)
    for _tl in ax.get_xticklabels():
        _tl.set_ha("right")
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
    mat = mat.drop_duplicates()
    if len(mat) > 15:
        mat = mat.loc[mat.mean(axis=1).nlargest(15).index]
    _draw_loo_heatmap(ax, mat)


# ======================================================================
# Legacy dataset config (shared by panel G LOO heatmap and panel A bootstrap)
# ======================================================================

_MDE_DATASET_CFG = {
    "TNBC": {
        "design": "two_arm",
        "loader": get_tnbc_zhang,
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "arm",
        "arm_treated": "anti-PDL1+Chemo",
        "arm_control": "Chemo",
        "visits": ("Pre", "Post"),
    },
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
        # Single-arm: every patient received CAR-T, so there is no arm to
        # filter on. (response now holds LtR/R/NR/Unknown from the loader, so the
        # old arm_filter="CAR-T" selected zero cells.)
        "arm_col": None,
        "arm_filter": None,
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
# Multi-dataset power constants, loaders, and helpers (panel J)
# (duplicated from manuscript_figures/main/figure3_robustness_benchmarking.py
#  per design choice; json cache helpers renamed to *_json_* to avoid
#  collision with the pickle-based _load_cache/_save_cache above)
# ======================================================================

N_POWER_ITERATIONS = 200
POWER_ALPHA = 0.05
RNG_SEED = 42
_CODE_VERSION = "v15"

DatasetInfo = tuple[str, object, object, tuple, list[str], str]

_DATASET_TAGS: dict[str, str] = {
    "Sade-Feldman": "SF",
    "TNBC": "TNBC",
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
    "TNBC":         "sig_Cytotoxic T Cell Activity",
    "AML":          "sig_Cytotoxic T Cell Activity",
    "CAR-T":        "sig_Cytotoxic T Cell Activity",
    "Vaccine":      "sig_Cytotoxic T Cell Activity",
    "COVID-19":     "sig_Inflammatory Response",
}

DATASET_COLORS = {
    "Sade-Feldman": COLORS["control"],
    "TNBC":         "#996633",
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

TNBC_DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="arm",
    arm_treated="anti-PDL1+Chemo",
    arm_control="Chemo",
    celltype_col="cell_type",
)
TNBC_VISITS: tuple[str, str] = ("Pre", "Post")


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


def _load_tnbc() -> DatasetInfo:
    tnbc = get_tnbc_zhang()
    tnbc, tnbc_sigs = score_signatures(tnbc, layer="log1p_norm")
    return ("TNBC", tnbc, TNBC_DESIGN, TNBC_VISITS, tnbc_sigs, "two_arm_did")


def _load_vaccine() -> DatasetInfo:
    vacc = get_vaccine()
    vacc, vacc_sigs = score_signatures(vacc, layer="log1p_norm")
    vacc_design = TrialDesign(
        participant_col="participant_id", visit_col="visit", arm_col=None,
    )
    return ("Vaccine", vacc, vacc_design, ("Pre", "Post"), vacc_sigs, "paired")


def _load_aml() -> DatasetInfo:
    aml = get_aml()
    aml, aml_sigs = score_signatures(aml, layer="log1p_norm")
    pid_col = "participant_id" if "participant_id" in aml.obs.columns else "patient_id"
    aml_design = TrialDesign(
        participant_col=pid_col, visit_col="visit", arm_col=None,
    )
    return ("AML", aml, aml_design, ("Pre", "Post"), aml_sigs, "paired")


def _load_cart() -> DatasetInfo:
    cart = get_cart()
    cart, cart_sigs = score_signatures(cart, layer="log1p_norm")
    pid_col = "participant_id" if "participant_id" in cart.obs.columns else "patient_id"
    cart_design = TrialDesign(
        participant_col=pid_col, visit_col="visit", arm_col=None,
    )
    return ("CAR-T", cart, cart_design, ("Pre", "Post"), cart_sigs, "paired")


def _load_covid() -> DatasetInfo:
    covid = get_stephenson()
    if "log1p_cpm" not in covid.layers:
        add_log1p_cpm_layer(covid, counts_layer="counts", out_layer="log1p_cpm")
    covid, covid_sigs = score_signatures(covid, layer="log1p_cpm")
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
    """Load a single dataset by index (0-5), returning None on failure."""
    loaders = [_load_tnbc, _load_sf, _load_vaccine, _load_aml, _load_cart, _load_covid]
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
    design: TrialDesign,
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
    for idx in range(6):
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
    inner_hspace: float | None = None,
    composite: bool = False,
) -> list[plt.Axes]:
    """Draw power curves into a gridspec area, returning created axes.

    Styling is intentionally matched to the legacy power-curve panel (`fig3_c.py`), with
    only the panel arrangement changed to support a 2-row 3+3 layout when
    six datasets are present (previously 3+2 for five).
    """
    from matplotlib.ticker import MaxNLocator
    from sklearn.isotonic import IsotonicRegression

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
        "paired": "Single-arm (paired)",
        "cross_sectional": "Cross-sectional",
    }

    ds_names = list(dict.fromkeys(power_df["dataset"]))
    n_ds = len(ds_names)

    if n_ds == 6:
        _hspace_default = 1.6 if composite else 1.9
        _hspace = inner_hspace if inner_hspace is not None else _hspace_default
        _wspace_6 = 0.42 if composite else 0.5
        gs_inner = gs_parent.subgridspec(
            2, 9,
            wspace=_wspace_6,
            hspace=_hspace,
            height_ratios=[1.0, 1.0],
        )
        slots = [
            gs_inner[0, 0:3], gs_inner[0, 3:6], gs_inner[0, 6:9],
            gs_inner[1, 0:3], gs_inner[1, 3:6], gs_inner[1, 6:9],
        ]
    elif n_ds == 5:
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
        if composite:
            # Tighter 2-line title for the composite artboard, where row 4
            # has far less vertical room than the standalone figure.
            title_lines = display_name
            if feat:
                title_lines += f", {feat}"
            title_lines += f"\n{design_label}, n={analyzable_n}"
        else:
            title_lines = display_name
            if feat:
                title_lines += f"\n{feat}"
            title_lines += f"\n{design_label}, n={analyzable_n}"

        _title_fs = 5.5 if composite else 6.0
        _title_pad = 3 if composite else 8
        ax.set_title(title_lines, fontsize=_title_fs, fontweight="bold",
                     color=color, pad=_title_pad, linespacing=1.3 if composite else 1.4)

        _x_fs = 5.0 if composite else 9.5
        _y_fs = 5.2 if composite else 10
        _tick_fs = 4.6 if composite else 8.5
        ax.set_xlabel("Analyzable participants", fontsize=_x_fs)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # For the 6-dataset 3-wide grid, only the leftmost column in each row
        # keeps a y-axis label/ticks — duplicating a full y-axis on every
        # subplot left no room between columns and caused titles/labels to
        # visually collide with the neighboring subplot's y-axis.
        _is_left_col = (n_ds != 6) or (i % 3 == 0)
        if _is_left_col:
            ax.set_ylabel(r"Power (1 − $\beta$)", fontsize=_y_fs,
                           labelpad=6 if composite else 4)
            ax.tick_params(axis="y", which="major", labelsize=_tick_fs, labelleft=True,
                            pad=2.5 if composite else 2)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="major", labelleft=False)
        ax.tick_params(axis="x", which="major", labelsize=_tick_fs)

        ax.grid(True, which="major", axis="y", color="#f0f0f0", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.spines["left"].set_linewidth(1.0)

    return axes


def _panel_power_curves(data: dict) -> plt.Figure | None:
    """Standalone figure for panel J: power curves across datasets."""
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

    fig = plt.figure(figsize=(13.5, 8.5))
    outer = fig.add_gridspec(1, 1, left=0.07, right=0.985, top=0.88, bottom=0.12)
    _panel_power_grid(fig, outer[0], data, inner_wspace=0.42)
    fig.tight_layout()
    return fig


# ======================================================================
# NatMeth signal-fraction benchmark CSV (H: mixed-signal null-gene FPR; I: pure-null Type I error; J: power curves)
# ======================================================================

# See figure3: derive from MANUSCRIPT_DIR so the HPC checkout depth is honoured.
# Path resolution lives in figure3's loader, which this delegates to.

# Imported from Figure 3, not restated. These were hand-maintained duplicates, so
# a method added to CORE_METHODS would have appeared in the main figure and
# silently vanished from the supplement -- with both looking internally
# consistent. Supp Fig 5 and Figure 3 must describe the same benchmark.
from .._benchmark import (  # noqa: E402
    _BENCH_METHOD_COLORS,
    _BENCH_METHOD_LABELS,
    _BENCH_METHOD_MARKERS,
    _BENCH_METHODS,
    _panel_bench_mixed_fpr,
    _panel_bench_pure_null_fpr,
)

_PANEL_SIZES = [50, 200, 500, 2000]
_SIGNAL_FRACTIONS = [2, 4, 10, 20]


def _load_benchmark_data():
    """Delegate to Figure 3's loader — do not re-implement it here.

    This was a second, independent copy of the same parsing logic. When the
    scenario labelling was found to be wrong (nominal versus realised signal
    fraction), fixing one copy would have left the supplementary panels
    disagreeing with the main figure while both looked internally consistent.
    Supp Fig 5 and Figure 3 must describe the same benchmark.
    """
    from .._benchmark import (
        _load_benchmark_data as _load_from_figure3,
    )

    return _load_from_figure3()


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
# Panel H — Mixed-signal null-gene FPR vs signal fraction (expanded from Figure 3 panel D)
# ======================================================================

def _compute_null_fpr_table(bench_df) -> pd.DataFrame:
    null = bench_df[bench_df["true_beta"] == 0.0].copy()
    rows = []
    for (method, n_g, frac), grp in null.groupby(["method", "n_genes", "signal_pct"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({"method": method, "n_genes": int(n_g), "signal_pct": int(frac),
                     "fpr": float((pvals < 0.05).mean()), "n_tests": int(len(pvals))})
    return pd.DataFrame(rows)


def _panel_bench_fpr_curves(fig, bench_df, *, composite: bool = False, axes=None) -> None:
    fpr_df = _compute_null_fpr_table(bench_df)
    fpr_df = fpr_df[fpr_df["signal_pct"] > 0].copy()
    if axes is None:
        axes = fig.subplots(1, 4, sharey=True)
        if not hasattr(axes, "__len__"):
            axes = [axes]
        if composite:
            fig.suptitle("Mixed-signal null-gene FPR vs signal fraction", x=0.5, y=0.99, fontsize=5.8, fontweight="bold")

    _ttl_fs = 5.75 if composite else 12
    _ax_fs = 5.15 if composite else 11
    _tk_fs = 4.65 if composite else 10
    _leg_fs = 5.2 if composite else 9

    x_positions = np.arange(len(_SIGNAL_FRACTIONS), dtype=float)
    frac_to_x = dict(zip(_SIGNAL_FRACTIONS, x_positions))
    # Five methods (limma-voom was added to the benchmark); offsets spread them so
    # the near-nominal calibrated curves stay legible where they overlap.
    method_offsets = {"wilcoxon_paired": -0.10, "nebula": -0.05, "limma_voom": 0.0,
                      "sctrial_did": +0.05, "dreamlet": +0.10}

    for ax_idx, (ax, n_g) in enumerate(zip(axes, _PANEL_SIZES)):
        sub = fpr_df[fpr_df["n_genes"] == n_g]
        for method in _BENCH_METHODS:
            m = sub[sub["method"] == method].sort_values("signal_pct")
            if m.empty:
                continue
            is_focal = method == "sctrial_did"
            style = _method_style(method, is_focal=is_focal, composite=composite)
            x = np.array([frac_to_x[int(f)] for f in m["signal_pct"].values]) + method_offsets[method]
            ax.plot(x, m["fpr"], label=_BENCH_METHOD_LABELS[method] if ax_idx == 0 else None,
                    zorder=10 if is_focal else 3, **style)
        _add_nominal_band(ax)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{f}%" for f in _SIGNAL_FRACTIONS], fontsize=_tk_fs)
        ax.set_xlim(-0.4, len(_SIGNAL_FRACTIONS) - 0.6)
        ax.set_xlabel("Signal fraction", fontsize=_ax_fs)
        ax.set_title(f"{n_g:,} genes", fontsize=_ttl_fs, fontweight="bold",
                     color="#222222", pad=4 if composite else 8)
        ax.set_ylim(0.0, 0.7)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.tick_params(axis="y", labelsize=_tk_fs)
        _style_axis(ax)

    axes[0].set_ylabel("Null-gene FPR (p < 0.05)", fontsize=_ax_fs)
    if composite:
        h, lab = axes[0].get_legend_handles_labels()
        for ax in axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        axes[0].legend(h, lab, loc="upper left", bbox_to_anchor=(0.02, 0.98),
                       ncol=1, frameon=True, framealpha=0.95, edgecolor="#cccccc",
                       fontsize=_leg_fs, handlelength=0.85, columnspacing=0.55, markerscale=0.65)
    else:
        axes[0].legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=_leg_fs)


# ======================================================================
# Panel I — Pure-null Type I error vs panel size
# (was panel H before reshuffle)
# ======================================================================

# ======================================================================
# (unused) Runtime comparison across methods — corresponds to Figure 3 panel G
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
    _leg_fs = 4.5 if composite else 9

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
    """Create and save Supplementary Figure 5 panels (A–J) + composite.

    Layout (same order as the composite artboard):
      A  Analytical vs bootstrap SE (all datasets, faceted forest plot)
      B  Standardized vs unstandardized effect sizes (TNBC)
      C  Mean vs median aggregation comparison (TNBC)
      D  Log-transform sensitivity (TNBC)
      E  Cell-type-stratified DiD heatmap (TNBC)
      F  Rank-order concordance across choices (TNBC)
      G  Leave-one-out stability matrix (all datasets)
      H  Mixed-signal null-gene FPR vs signal fraction (NatMeth benchmark, 5 methods)
      I  Pure-null Type I error vs panel size (NatMeth benchmark, 5 methods)
      J  Empirical power curves (participant subsampling; 3+3 facet grid)

    Composite (180 mm × ≤215 mm): row1 A | row2 B|C|D | row3 E|F|G |
    row4 H|I | row5 J.
    """
    print("Supplementary Figure 5: Sensitivity to Modeling and Preprocessing")

    # ── Sensitivity (panels B–F) — cache minus large adata ────────────
    data = _load_cache("sensitivity_tnbc")
    if data is None:
        data = _run_sensitivity()
        cacheable = {k: v for k, v in data.items() if k != "adata"}
        _save_cache("sensitivity_tnbc", cacheable)
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

    # Panels B–F: single-dataset sensitivity (TNBC)
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
    if loo_mat is not None:
        loo_mat = loo_mat.drop_duplicates()
        if len(loo_mat) > 15:
            loo_mat = loo_mat.loc[loo_mat.mean(axis=1).nlargest(15).index]
    fig, ax = plt.subplots(figsize=(9, 6))
    if loo_mat is not None:
        _draw_loo_heatmap(ax, loo_mat)
    else:
        ax.text(0.5, 0.5, "No LOO data", ha="center", va="center",
                transform=ax.transAxes)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # ── Benchmark (panels H–I) — H: null-gene FPR curves; I: pure-null Type I error ──
    print("  Loading signal-fraction sensitivity benchmark results ...")
    bench_df = _load_benchmark_data()
    print(
        f"    {len(bench_df):,} rows, "
        f"{bench_df.scenario.nunique()} scenarios, "
        f"panel sizes = {sorted(bench_df['n_genes'].unique())}"
    )

    # Panel H: Mixed-signal null-gene FPR vs signal fraction (the full four-facet
    # grid that Figure 3 panel D condenses). Delegated to Figure 3's panel so the
    # NEBULA broken-axis strip and the per-replicate aggregation stay identical
    # between the main figure and the supplement.
    fig_h_stand = plt.figure(figsize=(13, 4.6))
    _panel_bench_mixed_fpr(fig_h_stand, bench_df)
    save_panel(fig_h_stand, "panel_H_benchmark_fpr_curves", FIGURE_NAME, SUPP_OUTPUT)

    # Panel I: Pure-null Type I error vs gene panel size (NEBULA on a broken
    # upper axis). Unique to the supplement (Figure 3 panel C is vs participants).
    fig_i = plt.figure(figsize=(7.5, 5.0))
    _panel_bench_pure_null_fpr(fig_i, bench_df)
    save_panel(fig_i, "panel_I_benchmark_pure_null_fpr", FIGURE_NAME, SUPP_OUTPUT)

    # ── Empirical power curves on real datasets (panel J) ─────────────
    print("  Computing empirical power curves (panel J) ...")
    try:
        power_data = _prepare_power_data()
    except Exception as exc:
        print(f"  Warning: Could not compute power curves: {exc}")
        power_data = None
    composite_data = {**data, "power_data": power_data}

    fig_j = _panel_power_curves(composite_data)
    if fig_j is not None:
        save_panel(fig_j, "panel_J_power_curves", FIGURE_NAME, SUPP_OUTPUT)

    # ==================================================================
    # Composite artboard  (180 mm × 215 mm)
    # ==================================================================
    #   Row 1: A (full width)
    #   Row 2: B | C | D
    #   Row 3: E | F | G  (CT heatmap | rank concordance | LOO)
    #   Row 4: H | I  (FPR curves | pure-null FPR)
    #   Row 5: J  (centered — empirical power curves)
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
        5, 1,
        height_ratios=[1.0, 1.0, 0.80, 0.80, 1.10],
        hspace=0.52,
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
    subfig_a.subplots_adjust(wspace=0.42, left=0.04, right=0.98, top=0.80, bottom=0.14)

    # ── Row 2: B | C | D ─────────────────────────────────────────────
    # (E moved to row 3)
    gs_r2 = outer[1].subgridspec(1, 3, wspace=0.52)
    ax_b = fig_c.add_subplot(gs_r2[0])
    ax_cc = fig_c.add_subplot(gs_r2[1])
    ax_d = fig_c.add_subplot(gs_r2[2])

    _panel_std_vs_unstd(ax_b, data, composite=True)
    _panel_mean_vs_median(ax_cc, data, composite=True)
    _panel_log_sensitivity(ax_d, data, composite=True)

    # ── Row 3: E | F | G  (CT heatmap | rank concordance | LOO)
    # E moved down from row 2; pure-null FPR (was H) moves to row 4 as I.
    gs_r3 = outer[2].subgridspec(1, 3, width_ratios=[1.15, 0.80, 1.40], wspace=0.42)
    ax_e = fig_c.add_subplot(gs_r3[0])
    ax_f = fig_c.add_subplot(gs_r3[1])
    ax_g = fig_c.add_subplot(gs_r3[2])

    _panel_ct_heatmap(ax_e, data, composite=True)
    _panel_rank_concordance(ax_f, data, composite=True)
    if loo_mat is not None:
        _draw_loo_heatmap(
            ax_g, loo_mat, annot_fs=5,
            cbar_label_fs=_COMPOSITE_G_CBAR_LABEL_FS,
            title_fs=5.5, xlabel_pad=-1,
        )
    else:
        ax_g.text(0.5, 0.5, "No LOO data", ha="center", va="center",
                  transform=ax_g.transAxes)

    # ── Row 4: H (FPR curves) | I (pure-null FPR) ────────────────────
    # H = mixed-signal null-gene FPR vs signal fraction (expanded from Figure 3 panel D).
    # I = pure-null Type I error vs panel size (was H in row 3).
    # Use a subgridspec (not a SubFigure) so H's 4 axes get the same
    # automatic margin behaviour as ax_i, preventing vertical stretch.
    gs_r4 = outer[3].subgridspec(1, 2, width_ratios=[2.8, 1.0], wspace=0.25)
    # H (mixed-signal null-gene FPR, four facets, NEBULA broken-axis strip) is
    # delegated to Figure 3's panel; I (pure-null Type I error vs gene panel size)
    # is drawn as a broken pair and returns its (main, strip) axes.
    _cell_h = gs_r4[0]
    _panel_bench_mixed_fpr(fig_c, bench_df, composite=True, gs_parent=_cell_h)
    _ax_i_main, ax_i_strip = _panel_bench_pure_null_fpr(
        fig_c, bench_df, composite=True, gs_parent=gs_r4[1])

    # ── Row 5: J centered (empirical power curves) ────────────────────────
    # Centered by flanking with empty columns; wider than before because it
    # now has the full figure width to draw from.
    _R5_INSET = (0.12, 0.95, 0.08)
    gs_r5_outer = outer[4].subgridspec(3, 1, height_ratios=list(_R5_INSET), hspace=0)
    gs_r5 = gs_r5_outer[1].subgridspec(1, 3, wspace=0, width_ratios=[0.10, 1.0, 0.10])

    if power_data is not None and not power_data["power_df"].empty:
        power_axes = _panel_power_grid(
            fig_c, gs_r5[1], composite_data,
            inner_wspace=0.30, inner_hspace=3.2, composite=True,
        )
    else:
        power_axes = None
        ax_power_stub = fig_c.add_subplot(gs_r5[1])
        ax_power_stub.text(
            0.5, 0.5, "No power data available",
            ha="center", va="center",
            transform=ax_power_stub.transAxes,
            color=COLORS.get("gray", "#666"),
        )
        ax_power_stub.set_axis_off()

    fig_c.canvas.draw()

    # H panel group title — placed above the H cell (positions resolved post-draw)
    _h_pos = _cell_h.get_position(fig_c)
    fig_c.text(
        0.5 * (_h_pos.x0 + _h_pos.x1), _h_pos.y1 + 0.006,
        "Mixed-signal null-gene FPR vs signal fraction",
        ha="center", va="bottom", fontsize=5.8, fontweight="bold",
        transform=fig_c.transFigure, clip_on=False,
    )

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
    _lbl_y_r3 = 1.17  # F, G — slightly above default row-3 y
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
    ]:
        _x = _lbl_x_left if lbl in ("B", "F") else _lbl_xy[0]
        _y = _lbl_y_r3 if lbl in ("F",) else _lbl_xy[1]
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

    # H label at the upper-left of the H cell (figure coords, since H's axes are
    # owned by the delegated Figure 3 panel).
    fig_c.text(
        max(_h_pos.x0 - 0.028, 0.004), min(_h_pos.y1 + 0.006, 0.997), "H",
        transform=fig_c.transFigure, fontsize=_lbl_fs, fontweight="bold",
        va="bottom", ha="left",
    )
    # I label on the pure-null strip axis (top of the broken pair).
    ax_i_strip.text(
        -0.25, 1.35, "I",
        transform=ax_i_strip.transAxes,
        fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
    )
    _label_axes_panel(power_axes, "J", x=-0.30, y=1.42)

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
        "Analytical vs bootstrap SE",
        fontsize=_SMALL_RC["axes.titlesize"],
        pad_frac=0.006,
    )
    if power_axes:
        _pw_top_y = max(ax.get_position().y1 for ax in power_axes)
        _pw_xs = [ax.get_position().x0 for ax in power_axes] + [
            ax.get_position().x1 for ax in power_axes
        ]
        _pw_xc = 0.5 * (min(_pw_xs) + max(_pw_xs))
        fig_c.text(
            _pw_xc,
            _pw_top_y + 0.030,
            "Empirical power",
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
    print("  SuppFig5 complete: 10 individual panels + combined (A–J)\n")


if __name__ == "__main__":
    apply_style()
    generate()
