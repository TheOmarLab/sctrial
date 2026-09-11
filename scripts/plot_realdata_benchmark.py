#!/usr/bin/env python
"""Plot real-data benchmark results (permutation + subsampling on TNBC).

Reads:
  manuscript/benchmark/realdata/permutation_tnbc.csv
  manuscript/benchmark/realdata/subsampling_tnbc.csv

Saves figures to:
  manuscript/benchmark/realdata/figures/

Usage::
    python scripts/plot_realdata_benchmark.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "manuscript" / "benchmark" / "realdata"
OUT_DIR = DATA_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "sctrial_did":     "sctrial (DiD)",
    "dreamlet":        "dreamlet",
    "nebula":          "NEBULA",
    "wilcoxon_paired": "Wilcoxon (paired)",
}

METHOD_COLORS = {
    "sctrial_did":     "#4C72B0",
    "dreamlet":        "#DD8452",
    "nebula":          "#55A868",
    "wilcoxon_paired": "#C44E52",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    for fmt in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{fmt}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    print(f"  Saved → {OUT_DIR / name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Permutation: p-value calibration
# ---------------------------------------------------------------------------

perm_path = DATA_DIR / "permutation_tnbc.csv"
if not perm_path.exists():
    print(f"Not found: {perm_path}  — skipping permutation plots")
else:
    perm = pd.read_csv(perm_path)
    methods = [m for m in METHOD_LABELS if m in perm["method"].unique()]
    print(f"Permutation: {perm['permutation'].nunique()} permutations, "
          f"{len(methods)} methods, {perm['gene'].nunique()} genes")

    # --- Panel A: QQ plot (observed vs expected uniform) ---
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Uniform (expected)")

    for method in methods:
        pvals = perm.loc[perm["method"] == method, "pvalue"].dropna().values
        pvals = np.sort(pvals)
        n = len(pvals)
        if n == 0:
            continue
        expected = np.linspace(1 / n, 1, n)
        ax.plot(expected, pvals,
                color=METHOD_COLORS.get(method, "gray"),
                lw=1.5, alpha=0.85,
                label=METHOD_LABELS.get(method, method))

    ax.set_xlabel("Expected p-value (uniform)")
    ax.set_ylabel("Observed p-value")
    ax.set_title("P-value calibration under permuted null\n(TNBC, real data)")
    ax.legend(frameon=False)
    despine(ax)
    fig.tight_layout()
    save(fig, "permutation_qq")

    # --- Panel B: FPR bar chart (rejection rate at α=0.05) ---
    alpha = 0.05
    fpr_rows = []
    for method in methods:
        pvals = perm.loc[perm["method"] == method, "pvalue"].dropna()
        fpr = (pvals < alpha).mean()
        fpr_rows.append({"method": method, "fpr": fpr})
    fpr_df = pd.DataFrame(fpr_rows)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(
        [METHOD_LABELS.get(m, m) for m in fpr_df["method"]],
        fpr_df["fpr"],
        color=[METHOD_COLORS.get(m, "gray") for m in fpr_df["method"]],
        edgecolor="white", linewidth=0.5,
    )
    ax.axhline(alpha, color="black", linestyle="--", lw=1.2,
               label=f"Nominal α = {alpha}")
    ax.set_ylabel(f"FPR (p < {alpha})")
    ax.set_title("False positive rate under permuted null\n(TNBC, real data)")
    ax.set_ylim(0, max(fpr_df["fpr"].max() * 1.3, alpha * 2))
    ax.legend(frameon=False)
    plt.xticks(rotation=20, ha="right")
    despine(ax)
    fig.tight_layout()
    save(fig, "permutation_fpr")

    # --- Panel C: P-value histogram per method ---
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(3 * n_methods, 3.5),
                             sharey=True)
    if n_methods == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        pvals = perm.loc[perm["method"] == method, "pvalue"].dropna()
        ax.hist(pvals, bins=20, range=(0, 1),
                color=METHOD_COLORS.get(method, "gray"), alpha=0.8, edgecolor="white")
        ax.axhline(len(pvals) / 20, color="black", linestyle="--", lw=1,
                   alpha=0.6, label="Uniform")
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("p-value")
        despine(ax)
    axes[0].set_ylabel("Count")
    fig.suptitle("P-value histograms under permuted null (TNBC)", y=1.02)
    fig.tight_layout()
    save(fig, "permutation_histograms")

    # --- Panel D: Runtime boxplot per method ---
    if "runtime_seconds" in perm.columns:
        # One runtime value per (permutation × method), not per gene — deduplicate
        rt = (perm[["permutation", "method", "runtime_seconds"]]
              .drop_duplicates(subset=["permutation", "method"]))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        data = [rt.loc[rt["method"] == m, "runtime_seconds"].dropna().values
                for m in methods]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="white", lw=2))
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS.get(method, "gray"))
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=15, ha="right")
        ax.set_ylabel("Runtime (seconds per permutation)")
        ax.set_title("Per-method runtime under permuted null\n(TNBC, real data)")
        despine(ax)
        fig.tight_layout()
        save(fig, "permutation_runtime")

    # --- Panel E: Beta distribution under null — one subplot per method ---
    # Methods report betas on different scales (NEBULA logFC ≈ 0.001,
    # others ≈ 0.3–0.5 SD), so a shared x-axis makes some invisible.
    if "beta" in perm.columns:
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(3.5 * n_methods, 3.5),
                                 sharey=False)
        if n_methods == 1:
            axes = [axes]
        for ax, method in zip(axes, methods):
            betas = perm.loc[perm["method"] == method, "beta"].dropna().values
            if len(betas) == 0:
                continue
            ax.hist(betas, bins=40,
                    color=METHOD_COLORS.get(method, "gray"),
                    edgecolor="white", linewidth=0.3,
                    density=True, alpha=0.85)
            ax.axvline(0, color="black", linestyle="--", lw=1, alpha=0.6)
            ax.set_xlabel("β under permuted null")
            ax.set_ylabel("Density" if ax is axes[0] else "")
            ax.set_title(METHOD_LABELS.get(method, method))
            mean_val = betas.mean()
            ax.text(0.97, 0.95, f"mean={mean_val:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="black")
            despine(ax)
        fig.suptitle("β calibration under permuted null (TNBC)\nmean ≈ 0 expected",
                     y=1.02)
        fig.tight_layout()
        save(fig, "permutation_beta")

    # --- Panel F: Convergence rate per method ---
    if "converged" in perm.columns:
        conv_rows = []
        for method in methods:
            sub = perm[perm["method"] == method]["converged"].dropna()
            conv_rows.append({
                "method": method,
                "convergence_rate": sub.mean(),
                "n": len(sub),
            })
        conv_df = pd.DataFrame(conv_rows)

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.bar(
            [METHOD_LABELS.get(m, m) for m in conv_df["method"]],
            conv_df["convergence_rate"],
            color=[METHOD_COLORS.get(m, "gray") for m in conv_df["method"]],
            edgecolor="white",
        )
        ax.axhline(1.0, color="gray", linestyle=":", lw=1, alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Convergence rate")
        ax.set_title("Method convergence rate\n(TNBC, real data permutations)")
        plt.xticks(rotation=15, ha="right")
        despine(ax)
        fig.tight_layout()
        save(fig, "permutation_convergence")

    # --- Panel G: Per-gene FPR distribution ---
    # For each gene, compute the fraction of permutations where p < 0.05.
    # A well-calibrated method → gene FPRs uniform around 0.05.
    # Conservative method → spike near 0 (NEBULA).
    # Anti-conservative → rightward shift (dreamlet).
    if "gene" in perm.columns:
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(3.5 * n_methods, 3.5),
                                 sharey=False, sharex=True)
        if n_methods == 1:
            axes = [axes]
        for ax, method in zip(axes, methods):
            sub_m = perm[perm["method"] == method]
            if sub_m["permutation"].nunique() == 0:
                continue
            gene_fpr = (
                sub_m.groupby("gene")["pvalue"]
                .apply(lambda x: (x.dropna() < 0.05).mean())
            )
            gene_fpr = gene_fpr.dropna()
            if len(gene_fpr) == 0:
                continue
            ax.hist(gene_fpr, bins=20, range=(0, 1),
                    color=METHOD_COLORS.get(method, "gray"),
                    alpha=0.8, edgecolor="white")
            ax.axvline(0.05, color="black", linestyle="--", lw=1.2, alpha=0.7)
            ax.set_title(METHOD_LABELS.get(method, method))
            ax.set_xlabel("Per-gene FPR")
            despine(ax)
        axes[0].set_ylabel("Gene count")
        fig.suptitle(
            "Per-gene FPR distribution under permuted null (TNBC)\n"
            "Dashed line = nominal α = 0.05",
            y=1.02,
        )
        fig.tight_layout()
        save(fig, "permutation_gene_fpr")

    # --- Panel H: KS calibration statistic (distance from Uniform) ---
    # Lower KS = better calibrated. Integrates the full QQ curve into one number.
    try:
        from scipy.stats import kstest as _kstest

        ks_rows = []
        for method in methods:
            pvals = perm.loc[perm["method"] == method, "pvalue"].dropna().values
            if len(pvals) == 0:
                ks_rows.append({"method": method, "ks_stat": np.nan})
                continue
            stat, _ = _kstest(pvals, "uniform")
            ks_rows.append({"method": method, "ks_stat": stat})
        ks_df = pd.DataFrame(ks_rows)

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.bar(
            [METHOD_LABELS.get(m, m) for m in ks_df["method"]],
            ks_df["ks_stat"],
            color=[METHOD_COLORS.get(m, "gray") for m in ks_df["method"]],
            edgecolor="white",
        )
        ax.set_ylabel("KS distance from Uniform(0, 1)")
        ax.set_title("Calibration quality: KS statistic\n(TNBC — lower = better calibrated)")
        plt.xticks(rotation=15, ha="right")
        despine(ax)
        fig.tight_layout()
        save(fig, "permutation_ks_calibration")
    except ImportError:
        print("  scipy not available — skipping KS calibration plot")

    # --- Panel I: Failure mode stacked bar ---
    # Shows fraction of (permutation × gene) pairs by outcome:
    # success | numerical | convergence | timeout
    if "failure_mode" in perm.columns:
        perm_fm = perm.copy()
        perm_fm["failure_mode_clean"] = perm_fm["failure_mode"].fillna("success")

        failure_cats = ["success", "numerical", "convergence", "timeout"]
        cat_colors = {
            "success":     "#aec7e8",
            "numerical":   "#fd8d3c",
            "convergence": "#e6550d",
            "timeout":     "#756bb1",
        }

        method_fracs = {}
        for method in methods:
            counts = (
                perm_fm[perm_fm["method"] == method]["failure_mode_clean"]
                .value_counts(normalize=True)
            )
            method_fracs[method] = counts

        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(len(methods))
        bottom = np.zeros(len(methods))

        for cat in failure_cats:
            vals = np.array([method_fracs.get(m, pd.Series()).get(cat, 0.0)
                             for m in methods])
            if vals.sum() == 0:
                continue
            ax.bar(x, vals, bottom=bottom,
                   label=cat, color=cat_colors[cat], edgecolor="white", linewidth=0.5)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Fraction of gene-permutation pairs")
        ax.set_title("Failure mode breakdown\n(TNBC, real data permutations)")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        despine(ax)
        fig.tight_layout()
        save(fig, "permutation_failure_modes")

# ---------------------------------------------------------------------------
# Subsampling: reproducibility
# ---------------------------------------------------------------------------

sub_path = DATA_DIR / "subsampling_tnbc.csv"
if not sub_path.exists():
    print(f"Not found: {sub_path}  — skipping subsampling plots")
else:
    sub = pd.read_csv(sub_path)
    methods = [m for m in METHOD_LABELS if m in sub["method"].unique()]
    fractions = sorted(sub["fraction"].unique())
    print(f"Subsampling: {sub['resample'].nunique()} resamples, "
          f"{len(fractions)} fractions, {len(methods)} methods")

    # --- Panel D: Spearman ρ vs fraction (grouped boxplot) ---
    fig, ax = plt.subplots(figsize=(5.5, 4))

    x_positions = np.arange(len(fractions))
    width = 0.8 / len(methods)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(methods))

    for i, method in enumerate(methods):
        mdata = sub[sub["method"] == method]
        vals = [mdata.loc[mdata["fraction"] == f, "spearman_rho"].dropna().values
                for f in fractions]
        bp = ax.boxplot(
            vals,
            positions=x_positions + offsets[i],
            widths=width * 0.85,
            patch_artist=True,
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(color=METHOD_COLORS.get(method, "gray")),
            capprops=dict(color=METHOD_COLORS.get(method, "gray")),
            flierprops=dict(marker="o", markersize=2,
                            color=METHOD_COLORS.get(method, "gray"), alpha=0.4),
            boxprops=dict(facecolor=METHOD_COLORS.get(method, "gray"), alpha=0.8),
        )
        bp["boxes"][0].set_label(METHOD_LABELS.get(method, method))

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
    ax.set_xlabel("Participant fraction")
    ax.set_ylabel("Spearman ρ (vs full cohort)")
    ax.set_title("Ranking reproducibility under subsampling\n(TNBC, real data)")
    ax.axhline(1.0, color="gray", linestyle=":", lw=1, alpha=0.5)
    ax.set_ylim(-0.1, 1.1)
    handles = [plt.Rectangle((0, 0), 1, 1,
                              color=METHOD_COLORS.get(m, "gray"), alpha=0.8)
               for m in methods]
    ax.legend(handles, [METHOD_LABELS.get(m, m) for m in methods],
              frameon=False, loc="lower right")
    despine(ax)
    fig.tight_layout()
    save(fig, "subsampling_spearman")

    # --- Panel E: Jaccard@20 vs fraction ---
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, method in enumerate(methods):
        mdata = sub[sub["method"] == method]
        vals = [mdata.loc[mdata["fraction"] == f, "jaccard_top20"].dropna().values
                for f in fractions]
        bp = ax.boxplot(
            vals,
            positions=x_positions + offsets[i],
            widths=width * 0.85,
            patch_artist=True,
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(color=METHOD_COLORS.get(method, "gray")),
            capprops=dict(color=METHOD_COLORS.get(method, "gray")),
            flierprops=dict(marker="o", markersize=2,
                            color=METHOD_COLORS.get(method, "gray"), alpha=0.4),
            boxprops=dict(facecolor=METHOD_COLORS.get(method, "gray"), alpha=0.8),
        )
        bp["boxes"][0].set_label(METHOD_LABELS.get(method, method))

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
    ax.set_xlabel("Participant fraction")
    ax.set_ylabel("Jaccard overlap (top 20 genes)")
    ax.set_title("Top-gene overlap under subsampling\n(TNBC, real data)")
    ax.set_ylim(-0.05, 1.05)
    handles = [plt.Rectangle((0, 0), 1, 1,
                              color=METHOD_COLORS.get(m, "gray"), alpha=0.8)
               for m in methods]
    ax.legend(handles, [METHOD_LABELS.get(m, m) for m in methods],
              frameon=False, loc="lower right")
    despine(ax)
    fig.tight_layout()
    save(fig, "subsampling_jaccard")

    # --- Panel: gene failure rate vs fraction (method failure diagnostic) ---
    # Plotting failure rate (1 - n_valid/total) instead of raw n_valid so that
    # methods with zero failures are visible at the bottom rather than hidden at
    # the ceiling line.
    if "n_valid_genes" in sub.columns:
        n_genes_total = sub["n_valid_genes"].max()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for i, method in enumerate(methods):
            mdata = sub[sub["method"] == method]
            vals = [1 - mdata.loc[mdata["fraction"] == f, "n_valid_genes"].dropna().values / n_genes_total
                    for f in fractions]
            bp = ax.boxplot(
                vals,
                positions=x_positions + offsets[i],
                widths=width * 0.85,
                patch_artist=True,
                medianprops=dict(color="white", lw=2),
                whiskerprops=dict(color=METHOD_COLORS.get(method, "gray")),
                capprops=dict(color=METHOD_COLORS.get(method, "gray")),
                flierprops=dict(marker="o", markersize=2,
                                color=METHOD_COLORS.get(method, "gray"), alpha=0.4),
                boxprops=dict(facecolor=METHOD_COLORS.get(method, "gray"), alpha=0.8),
            )
            bp["boxes"][0].set_label(METHOD_LABELS.get(method, method))

        ax.axhline(0, color="gray", linestyle=":", lw=1, alpha=0.6, label="No failures")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
        ax.set_xlabel("Participant fraction")
        ax.set_ylabel(f"Gene failure rate (out of {int(n_genes_total)})")
        ax.set_title("Method reliability at small sample sizes\n(TNBC, real data)")
        handles = [plt.Rectangle((0, 0), 1, 1,
                                 color=METHOD_COLORS.get(m, "gray"), alpha=0.8)
                   for m in methods]
        ax.legend(handles, [METHOD_LABELS.get(m, m) for m in methods],
                  frameon=False, loc="upper right")
        despine(ax)
        fig.tight_layout()
        save(fig, "subsampling_n_valid")

    # --- Panel: runtime vs fraction per method ---
    if "runtime_seconds" in sub.columns:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for i, method in enumerate(methods):
            mdata = sub[sub["method"] == method]
            vals = [mdata.loc[mdata["fraction"] == f, "runtime_seconds"].dropna().values
                    for f in fractions]
            bp = ax.boxplot(
                vals,
                positions=x_positions + offsets[i],
                widths=width * 0.85,
                patch_artist=True,
                medianprops=dict(color="white", lw=2),
                whiskerprops=dict(color=METHOD_COLORS.get(method, "gray")),
                capprops=dict(color=METHOD_COLORS.get(method, "gray")),
                flierprops=dict(marker="o", markersize=2,
                                color=METHOD_COLORS.get(method, "gray"), alpha=0.4),
                boxprops=dict(facecolor=METHOD_COLORS.get(method, "gray"), alpha=0.8),
            )
            bp["boxes"][0].set_label(METHOD_LABELS.get(method, method))

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
        ax.set_xlabel("Participant fraction")
        ax.set_yscale("log")
        ax.set_ylabel("Runtime (seconds, log scale)")
        ax.set_title("Per-method runtime vs sample size\n(TNBC, real data)")
        handles = [plt.Rectangle((0, 0), 1, 1,
                                 color=METHOD_COLORS.get(m, "gray"), alpha=0.8)
                   for m in methods]
        ax.legend(handles, [METHOD_LABELS.get(m, m) for m in methods],
                  frameon=False, loc="upper left")
        despine(ax)
        fig.tight_layout()
        save(fig, "subsampling_runtime")

    # --- Panel: summary line plot (median ± IQR) ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    metrics = [("spearman_rho", "Spearman ρ", axes[0]),
               ("jaccard_top20", "Jaccard (top 20)", axes[1])]

    for col, ylabel, ax in metrics:
        for method in methods:
            mdata = sub[sub["method"] == method]
            medians, q25s, q75s = [], [], []
            for f in fractions:
                v = mdata.loc[mdata["fraction"] == f, col].dropna()
                medians.append(v.median())
                q25s.append(v.quantile(0.25))
                q75s.append(v.quantile(0.75))
            xs = [int(f * 100) for f in fractions]
            color = METHOD_COLORS.get(method, "gray")
            ax.plot(xs, medians, "o-", color=color, lw=2,
                    label=METHOD_LABELS.get(method, method))
            ax.fill_between(xs, q25s, q75s, color=color, alpha=0.15)
        ax.set_xlabel("Participant fraction (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([int(f * 100) for f in fractions])
        despine(ax)

    axes[0].set_title("Ranking reproducibility (TNBC)")
    axes[1].set_title("Top-gene overlap (TNBC)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "subsampling_summary")

print(f"\nAll figures saved to {OUT_DIR}")
