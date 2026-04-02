"""
Supplementary Figure 7 — Signal-Fraction Sensitivity Analysis.
===============================================================

Shows how null-gene calibration depends on gene-panel size and signal
fraction.  Addresses the reviewer concern that the 50-gene / 20% signal
benchmark may be regime-specific.

Panels
------
  A  Null-gene FPR heatmap (panel size × signal fraction × method).
  B  Null-gene FPR line plots per method across signal fractions,
     faceted by panel size.
  C  Pure-null calibration (λ_GC) across panel sizes.
  D  QQ plots for selected conditions (50-gene/20% vs 2000-gene/1%).
"""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._shared import (
    SUPP_OUTPUT,
    apply_style,
    clear_cache,
    despine,
    save_panel,
)

FIGURE_NAME = "SuppFig7_signal_fraction_sensitivity"

_SENSITIVITY_CSV = (
    Path(__file__).resolve().parents[4]
    / "manuscript"
    / "benchmark"
    / "sensitivity"
    / "sensitivity_combined.csv"
)

# Reuse method display config from SF4
_METHODS = ["wilcoxon_paired", "nebula", "dreamlet", "sctrial_did"]
_METHOD_LABELS = {
    "sctrial_did": "sctrial (DiD)",
    "dreamlet": "dreamlet",
    "nebula": "NEBULA",
    "wilcoxon_paired": "Wilcoxon (Δ scores)",
}
_METHOD_COLORS = {
    "sctrial_did": "#1f77b4",
    "dreamlet": "#d62728",
    "nebula": "#ff7f0e",
    "wilcoxon_paired": "#2ca02c",
}
_METHOD_MARKERS = {
    "sctrial_did": "o",
    "dreamlet": "D",
    "nebula": "s",
    "wilcoxon_paired": "^",
}


def _load_sensitivity_data() -> pd.DataFrame:
    """Load sensitivity benchmark results."""
    if not _SENSITIVITY_CSV.exists():
        raise FileNotFoundError(
            f"Sensitivity results not found at {_SENSITIVITY_CSV}.\n"
            "Run the sensitivity benchmark on HPC first."
        )
    df = pd.read_csv(_SENSITIVITY_CSV, low_memory=False)
    df["design"] = df["scenario"].str.split("__").str[0]

    # Extract panel size and signal fraction from scenario name
    # Format: two_arm__sens_g{n_genes}_f{frac_pct} or two_arm__sens_null_g{n_genes}
    df["n_genes"] = df["scenario"].str.extract(r"_g(\d+)")[0].astype(float)
    frac_extract = df["scenario"].str.extract(r"_f(\d+)")
    df["signal_fraction_pct"] = pd.to_numeric(frac_extract[0], errors="coerce")
    # Pure-null scenarios have NaN signal_fraction_pct → set to 0
    df.loc[df["scenario"].str.contains("sens_null"), "signal_fraction_pct"] = 0.0
    return df


def _panel_fpr_heatmap(fig, sens_df):
    """Panel A: Null-gene FPR heatmap — method × (panel size, signal fraction).

    Rows = methods, columns = (panel_size, signal_fraction) combos.
    Color = null-gene FPR. Shows the gradient from safe to inflated.
    """
    # Compute null-gene FPR per method × panel × fraction
    null_genes = sens_df[sens_df["true_beta"] == 0.0].copy()
    # Exclude pure-null scenarios (we want mixed-signal null-gene FPR)
    mixed = null_genes[null_genes["signal_fraction_pct"] > 0]

    rows = []
    for (method, n_genes, frac), grp in mixed.groupby(
        ["method", "n_genes", "signal_fraction_pct"]
    ):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({
            "method": method, "n_genes": int(n_genes),
            "signal_fraction": int(frac),
            "null_fpr": (pvals < 0.05).mean(),
        })
    fpr_df = pd.DataFrame(rows)

    n_methods = len(_METHODS)
    axes = fig.subplots(1, n_methods, sharey=True)
    if n_methods == 1:
        axes = [axes]

    panel_sizes = sorted(fpr_df["n_genes"].unique())
    fractions = sorted(fpr_df["signal_fraction"].unique())

    for mi, method in enumerate(_METHODS):
        ax = axes[mi]
        msub = fpr_df[fpr_df["method"] == method]

        # Build matrix: rows = signal fractions, cols = panel sizes
        mat = np.full((len(fractions), len(panel_sizes)), np.nan)
        for fi, frac in enumerate(fractions):
            for pi, ps in enumerate(panel_sizes):
                val = msub[
                    (msub["signal_fraction"] == frac) & (msub["n_genes"] == ps)
                ]["null_fpr"]
                if len(val) > 0:
                    mat[fi, pi] = val.values[0]

        im = ax.imshow(
            mat, aspect="auto", cmap="RdYlGn_r",
            vmin=0.0, vmax=max(0.10, np.nanmax(mat)),
            origin="lower",
        )
        ax.set_xticks(range(len(panel_sizes)))
        ax.set_xticklabels([str(int(p)) for p in panel_sizes], fontsize=8)
        ax.set_xlabel("Panel size (genes)", fontsize=9)
        if mi == 0:
            ax.set_yticks(range(len(fractions)))
            ax.set_yticklabels([f"{int(f)}%" for f in fractions], fontsize=8)
            ax.set_ylabel("Signal fraction", fontsize=9)
        else:
            ax.set_yticks([])

        ax.set_title(_METHOD_LABELS[method], fontweight="bold", fontsize=10)

        # Annotate cells with values
        for fi in range(len(fractions)):
            for pi in range(len(panel_sizes)):
                val = mat[fi, pi]
                if not np.isnan(val):
                    color = "white" if val > 0.15 else "black"
                    ax.text(pi, fi, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")

    fig.colorbar(im, ax=axes, label="Null-gene FPR (p < 0.05)",
                 shrink=0.6, pad=0.02)


def _panel_fpr_lines(fig, sens_df):
    """Panel B: Null-gene FPR line plots — one subplot per panel size.

    x = signal fraction, y = null-gene FPR, one line per method.
    Shows how FPR scales with signal fraction at each panel size.
    """
    null_genes = sens_df[sens_df["true_beta"] == 0.0].copy()
    mixed = null_genes[null_genes["signal_fraction_pct"] > 0]

    rows = []
    for (method, n_genes, frac), grp in mixed.groupby(
        ["method", "n_genes", "signal_fraction_pct"]
    ):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({
            "method": method, "n_genes": int(n_genes),
            "signal_fraction": int(frac),
            "null_fpr": (pvals < 0.05).mean(),
        })
    fpr_df = pd.DataFrame(rows)

    panel_sizes = sorted(fpr_df["n_genes"].unique())
    n_panels = len(panel_sizes)
    axes = fig.subplots(1, n_panels, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for pi, (ax, ps) in enumerate(zip(axes, panel_sizes)):
        for zi, method in enumerate(_METHODS):
            msub = fpr_df[
                (fpr_df["method"] == method) & (fpr_df["n_genes"] == ps)
            ].sort_values("signal_fraction")
            if msub.empty:
                continue
            is_sctrial = method == "sctrial_did"
            ax.plot(
                msub["signal_fraction"], msub["null_fpr"],
                marker=_METHOD_MARKERS[method],
                markersize=9 if is_sctrial else 7,
                linewidth=2.5 if is_sctrial else 1.8,
                color=_METHOD_COLORS[method],
                label=_METHOD_LABELS[method] if pi == 0 else None,
                zorder=10 if is_sctrial else zi + 1,
            )

        ax.axhline(0.05, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Signal fraction (%)")
        ax.set_title(f"{int(ps)} genes", fontweight="bold")
        ax.set_xticks([1, 5, 10, 20])
        despine(ax)

    axes[0].set_ylabel("Null-gene FPR (p < 0.05)")
    axes[0].legend(fontsize=7, loc="upper left")


def _panel_null_lambda(ax, sens_df):
    """Panel C: Pure-null λ_GC across panel sizes.

    Shows that all methods remain calibrated under pure null
    regardless of panel size.
    """
    null_scenarios = sens_df[sens_df["scenario"].str.contains("sens_null")]
    null_pvals = null_scenarios[null_scenarios["true_beta"] == 0.0]

    rows = []
    for (method, n_genes), grp in null_pvals.groupby(["method", "n_genes"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) < 10:
            continue
        chi2_obs = sp_stats.chi2.isf(pvals, df=1)
        chi2_obs = chi2_obs[np.isfinite(chi2_obs)]
        if len(chi2_obs) < 10:
            continue
        lambda_gc = np.median(chi2_obs) / sp_stats.chi2.ppf(0.5, df=1)
        rows.append({"method": method, "n_genes": int(n_genes), "lambda_gc": lambda_gc})

    lambda_df = pd.DataFrame(rows)
    panel_sizes = sorted(lambda_df["n_genes"].unique())

    for zi, method in enumerate(_METHODS):
        sub = lambda_df[lambda_df["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_sctrial = method == "sctrial_did"
        ax.plot(
            sub["n_genes"], sub["lambda_gc"],
            marker=_METHOD_MARKERS[method],
            markersize=10 if is_sctrial else 8,
            linewidth=2.5 if is_sctrial else 1.8,
            label=_METHOD_LABELS[method],
            color=_METHOD_COLORS[method],
            zorder=10 if is_sctrial else zi + 1,
        )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label=r"Ideal ($\lambda$ = 1)")
    ax.axhspan(0.95, 1.05, color="red", alpha=0.06)
    ax.set_xlabel("Panel size (genes)")
    ax.set_ylabel(r"Genomic inflation factor ($\lambda_{\mathrm{GC}}$)")
    ax.set_title("Pure-null calibration across panel sizes", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(panel_sizes)
    ax.set_xticklabels([str(int(p)) for p in panel_sizes])
    ax.set_ylim(0.90, 1.15)
    ax.legend(fontsize=7)
    despine(ax)


def _panel_qq_contrast(fig, sens_df):
    """Panel D: QQ contrast — worst case (50g/20%) vs best case (2000g/1%).

    Two columns, four methods each. Directly shows the regime dependence.
    """
    conditions = [
        ("50-gene, 20% signal", 50, 20),
        ("2000-gene, 1% signal", 2000, 1),
    ]

    n_methods = len(_METHODS)
    axes = fig.subplots(len(conditions), n_methods, sharex=True, sharey=True)

    for ci, (label, n_genes, frac_pct) in enumerate(conditions):
        # Get null-gene p-values for this condition
        mask = (
            (sens_df["n_genes"] == n_genes)
            & (sens_df["signal_fraction_pct"] == frac_pct)
            & (sens_df["true_beta"] == 0.0)
        )
        cond_data = sens_df[mask]

        for mi, method in enumerate(_METHODS):
            ax = axes[ci, mi]
            pvals = cond_data.loc[
                cond_data["method"] == method, "pvalue"
            ].dropna().sort_values().values

            if len(pvals) == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", fontsize=8)
                continue

            n = len(pvals)
            expected = (np.arange(1, n + 1) - 0.5) / n
            obs_log = -np.log10(pvals + 1e-300)
            exp_log = -np.log10(expected + 1e-300)

            # 95% Beta envelope
            ranks = np.arange(1, n + 1)
            lo_env = -np.log10(
                sp_stats.beta.ppf(0.975, ranks, n - ranks + 1) + 1e-300
            )
            hi_env = -np.log10(
                sp_stats.beta.ppf(0.025, ranks, n - ranks + 1) + 1e-300
            )
            ax.fill_between(exp_log, lo_env, hi_env, color="gray", alpha=0.12)
            ax.scatter(exp_log, obs_log, s=3, alpha=0.5,
                       color=_METHOD_COLORS[method], rasterized=True)
            lim = max(exp_log.max(), obs_log.max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)

            if ci == 0:
                ax.set_title(_METHOD_LABELS[method], fontweight="bold", fontsize=9)
            if mi == 0:
                ax.set_ylabel(f"{label}\n" + r"Obs $-\log_{10}(p)$", fontsize=8)
            if ci == len(conditions) - 1:
                ax.set_xlabel(r"Exp $-\log_{10}(p)$", fontsize=8)
            despine(ax)


def generate():
    """Generate Supplementary Figure 7 panels."""
    apply_style()

    print("SuppFig7: Signal-fraction sensitivity")
    print("  Loading sensitivity data ...")
    sens_df = _load_sensitivity_data()
    print(f"    {len(sens_df):,} rows, {sens_df.scenario.nunique()} scenarios")

    # Panel A: FPR heatmap
    print("  Panel A (FPR heatmap) ...")
    fig_a = plt.figure(figsize=(16, 4.5))
    _panel_fpr_heatmap(fig_a, sens_df)
    fig_a.tight_layout()
    save_panel(fig_a, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: FPR line plots
    print("  Panel B (FPR lines) ...")
    fig_b = plt.figure(figsize=(16, 4.5))
    _panel_fpr_lines(fig_b, sens_df)
    fig_b.tight_layout()
    save_panel(fig_b, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: Pure-null λ_GC
    print("  Panel C (null lambda) ...")
    fig_c, ax_c = plt.subplots(figsize=(7.0, 5.0))
    _panel_null_lambda(ax_c, sens_df)
    fig_c.tight_layout()
    save_panel(fig_c, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: QQ contrast
    print("  Panel D (QQ contrast) ...")
    fig_d = plt.figure(figsize=(16, 8))
    _panel_qq_contrast(fig_d, sens_df)
    fig_d.tight_layout()
    save_panel(fig_d, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    del sens_df
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
