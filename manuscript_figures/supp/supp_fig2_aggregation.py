"""
Supplementary Figure 2 — Aggregation Sensitivity Analysis.
==========================================================

2x3 grid (16x10 inches) comparing mean, median, and trimmed-mean
pseudobulk aggregation strategies on the Sade-Feldman dataset.

Panels
------
A  Scatter: mean vs median effect sizes.
B  Scatter: mean vs trimmed-mean effect sizes.
C  Scatter: -log10(p) mean vs median.
D  Scatter: -log10(p) mean vs trimmed-mean.
E  Forest plot of ranked effect sizes by method.
F  Correlation summary bar chart.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    TrialDesign,
    apply_style,
    despine,
    did_table,
    get_sade_feldman,
    harmonize_response,
    save_panel,
    score_signatures,
    sig_display,
    clear_cache,
    SCTRIAL_AVAILABLE,
)

FIGURE_NAME = "SuppFig2_aggregation_sensitivity"
FIGSIZE = (16, 10)

DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)
VISITS = ("Pre", "Post")


# ── data preparation ─────────────────────────────────────────────────

def _prepare_data() -> dict | None:
    """Run DiD with mean, median, and trimmed_mean aggregation."""
    if not SCTRIAL_AVAILABLE:
        print("  sctrial not available; skipping.")
        return None

    adata = get_sade_feldman()
    if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")
    adata = harmonize_response(adata)

    results = {}
    for agg_name in ("mean", "median"):
        try:
            res = did_table(
                adata,
                features=sig_cols,
                design=DESIGN,
                visits=VISITS,
                layer="log1p_tpm",
                standardize=True,
                aggregate="participant_visit",
                agg=agg_name,
            )
            res["label"] = res["feature"].apply(sig_display)
            results[agg_name] = res
            print(f"    {agg_name}: {len(res)} features")
        except Exception as exc:
            print(f"    {agg_name} failed: {exc}")

    # Trimmed mean: use scipy.stats.trim_mean via custom agg callable
    try:
        from functools import partial
        from scipy.stats import trim_mean as _trim_mean

        def trimmed_mean_agg(x):
            """20% trimmed mean."""
            arr = np.asarray(x, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                return np.nan
            return _trim_mean(arr, proportiontocut=0.1)

        res_tm = did_table(
            adata,
            features=sig_cols,
            design=DESIGN,
            visits=VISITS,
            layer="log1p_tpm",
            standardize=True,
            aggregate="participant_visit",
            agg=trimmed_mean_agg,
        )
        res_tm["label"] = res_tm["feature"].apply(sig_display)
        results["trimmed_mean"] = res_tm
        print(f"    trimmed_mean: {len(res_tm)} features")
    except Exception as exc:
        print(f"    trimmed_mean failed: {exc} — skipping trimmed-mean panels")

    return results


# ── panel helpers ─────────────────────────────────────────────────────

def _effect_col(df: pd.DataFrame) -> str:
    """Find the DiD effect-size column."""
    for col in ("beta_DiD", "beta_did", "hedges_g", "effect_size"):
        if col in df.columns:
            return col
    # Fallback to first numeric column with 'beta' in name
    for col in df.columns:
        if "beta" in col.lower():
            return col
    return df.select_dtypes(include="number").columns[0]


def _pval_col(df: pd.DataFrame) -> str:
    """Find the p-value column."""
    for col in ("p_DiD", "p_did", "p_value", "pvalue"):
        if col in df.columns:
            return col
    for col in df.columns:
        if col.startswith("p_") or col == "p":
            return col
    return "p_DiD"


def _scatter_comparison(ax, df_x, df_y, *, col: str, xlabel: str, ylabel: str,
                         title: str, log_scale: bool = False):
    """Generic scatter comparing a metric between two aggregation methods."""
    merged = df_x[["feature", col]].merge(
        df_y[["feature", col]], on="feature", suffixes=("_x", "_y"),
    )
    if merged.empty:
        ax.set_title(title)
        return

    x = merged[f"{col}_x"].values
    y = merged[f"{col}_y"].values

    if log_scale:
        x = -np.log10(np.clip(x, 1e-300, 1))
        y = -np.log10(np.clip(y, 1e-300, 1))

    ax.scatter(x, y, s=40, alpha=0.7, color=COLORS["treated"],
               edgecolors="white", linewidth=0.5, zorder=3)

    # Identity line
    lims = [min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "--", color=COLORS["gray"], linewidth=1, alpha=0.6,
            zorder=1)

    # Correlation
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 3:
        r, p = stats.pearsonr(x[mask], y[mask])
        ax.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.2e}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=COLORS["gray"], alpha=0.8))

    # Labels per point (with adjustText to avoid overlaps)
    labels = merged.merge(df_x[["feature", "label"]], on="feature",
                          how="left")
    if "label" in labels.columns:
        try:
            from adjustText import adjust_text
            texts = []
            for _, row in labels.iterrows():
                vx = -np.log10(max(row[f"{col}_x"], 1e-300)) if log_scale else row[f"{col}_x"]
                vy = -np.log10(max(row[f"{col}_y"], 1e-300)) if log_scale else row[f"{col}_y"]
                texts.append(ax.text(vx, vy, row["label"], fontsize=6, alpha=0.7))
            adjust_text(texts, ax=ax,
                        force_points=(0.5, 0.5),
                        force_text=(0.3, 0.3),
                        expand_points=(1.5, 1.5),
                        arrowprops=dict(arrowstyle="-", color="#bbb",
                                        lw=0.6, shrinkA=0, shrinkB=4))
        except ImportError:
            for _, row in labels.iterrows():
                vx = -np.log10(max(row[f"{col}_x"], 1e-300)) if log_scale else row[f"{col}_x"]
                vy = -np.log10(max(row[f"{col}_y"], 1e-300)) if log_scale else row[f"{col}_y"]
                ax.annotate(
                    row["label"], (vx, vy), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points",
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    despine(ax)


def _panel_forest(ax, results: dict):
    """Forest plot comparing effect sizes across aggregation methods."""
    method_colors = {
        "mean": COLORS["treated"],
        "median": COLORS["control"],
        "trimmed_mean": COLORS["neutral"],
    }

    # Use mean results as baseline ordering
    if "mean" not in results:
        ax.set_title("Effect Sizes by Method")
        return

    base = results["mean"].copy()
    ecol = _effect_col(base)
    base = base.sort_values(ecol, ascending=True)
    features = base["feature"].tolist()
    labels = base["label"].tolist()

    y_positions = np.arange(len(features))
    offset_map = {"mean": -0.15, "median": 0.0, "trimmed_mean": 0.15}

    for method, df in results.items():
        color = method_colors.get(method, COLORS["gray"])
        offset = offset_map.get(method, 0)
        for i, feat in enumerate(features):
            row = df[df["feature"] == feat]
            if row.empty:
                continue
            val = row[ecol].values[0]
            ax.plot(val, y_positions[i] + offset, "o",
                    color=color, markersize=6, alpha=0.8, zorder=3)

    ax.axvline(0, color=COLORS["gray"], linewidth=0.8, linestyle="--",
               alpha=0.5, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("DiD Effect Size")
    ax.set_title("Ranked Effect Sizes by Method", fontweight="bold")

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", color=method_colors[m], linestyle="",
               markersize=6, label=m.replace("_", " ").title())
        for m in results
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=True,
              framealpha=0.9)
    despine(ax)


def _panel_correlation_bar(ax, results: dict):
    """Bar chart of pairwise Pearson r between aggregation methods."""
    ecol = _effect_col(results.get("mean", pd.DataFrame()))
    if not ecol:
        ax.set_title("Correlation Summary")
        return

    pairs = [
        ("mean", "median"),
        ("mean", "trimmed_mean"),
        ("median", "trimmed_mean"),
    ]
    names = []
    correlations = []

    for m1, m2 in pairs:
        if m1 not in results or m2 not in results:
            continue
        merged = results[m1][["feature", ecol]].merge(
            results[m2][["feature", ecol]], on="feature", suffixes=("_1", "_2"),
        )
        mask = np.isfinite(merged[f"{ecol}_1"]) & np.isfinite(merged[f"{ecol}_2"])
        if mask.sum() >= 3:
            r, _ = stats.pearsonr(merged.loc[mask, f"{ecol}_1"],
                                  merged.loc[mask, f"{ecol}_2"])
            correlations.append(r)
            label = (f"{m1.replace('_', ' ').title()}\nvs\n"
                     f"{m2.replace('_', ' ').title()}")
            names.append(label)

    if not correlations:
        ax.set_title("Correlation Summary")
        return

    bars = ax.bar(names, correlations, color=[COLORS["treated"], COLORS["control"],
                                               COLORS["neutral"]][:len(names)],
                  edgecolor="white", linewidth=1.2, width=0.5)

    for bar, r in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{r:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Pearson r (effect sizes)")
    ax.set_title("Correlation Summary", fontweight="bold")
    ax.axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=0.5,
               alpha=0.4)
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Supplementary Figure 2 individual panels."""
    print("Supplementary Figure 2: Aggregation Sensitivity")
    results = _prepare_data()
    if results is None or not results:
        print("  No data; skipping figure.")
        return

    ecol = _effect_col(results.get("mean", pd.DataFrame()))
    pcol = _pval_col(results.get("mean", pd.DataFrame()))

    # ── Save individual panels ────────────────────────────────────────
    panel_specs = [
        ("A", lambda ax: _scatter_comparison(
            ax, results["mean"], results["median"], col=ecol,
            xlabel="Effect size (mean)", ylabel="Effect size (median)",
            title="Mean vs Median")),
        ("C", lambda ax: _scatter_comparison(
            ax, results["mean"], results["median"], col=pcol,
            xlabel=r"$-\log_{10}(p)$ (mean)", ylabel=r"$-\log_{10}(p)$ (median)",
            title="P-values: Mean vs Median", log_scale=True)),
        ("E", lambda ax: _panel_forest(ax, results)),
        ("F", lambda ax: _panel_correlation_bar(ax, results)),
    ]

    # Trimmed-mean panels only if trimmed_mean succeeded
    if "trimmed_mean" in results:
        panel_specs.insert(1, ("B", lambda ax: _scatter_comparison(
            ax, results["mean"], results["trimmed_mean"], col=ecol,
            xlabel="Effect size (mean)", ylabel="Effect size (trimmed mean)",
            title="Mean vs Trimmed Mean")))
        panel_specs.insert(3, ("D", lambda ax: _scatter_comparison(
            ax, results["mean"], results["trimmed_mean"], col=pcol,
            xlabel=r"$-\log_{10}(p)$ (mean)",
            ylabel=r"$-\log_{10}(p)$ (trimmed mean)",
            title="P-values: Mean vs Trimmed Mean", log_scale=True)))
    for label, func in panel_specs:
        try:
            fig_p, ax_p = plt.subplots(figsize=(6, 5))
            func(ax_p)
            fig_p.tight_layout()
            save_panel(fig_p, f"panel_{label}", FIGURE_NAME, SUPP_OUTPUT)
        except Exception:
            pass

    # ── Cleanup ───────────────────────────────────────────────────────
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
