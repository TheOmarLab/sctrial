"""
Supplementary Figure 1 — Data Quality and Cohort Integrity Across Datasets.
============================================================================

Establish that all datasets are QC-sound and comparable before inference.

Panels:
  A  Cells / genes / participants / samples per dataset (faceted bars).
  B  QC retention waterfall per dataset (raw → filtered).
  C  n_genes per cell distributions by dataset and visit/arm.
  D  total_counts per cell distributions by dataset and visit/arm.
  E  % mito and % ribosomal distributions with threshold overlays.
  F  Cells-per-participant inequality (Lorenz curve + Gini per dataset).
  G  Participant-by-visit completeness heatmaps (pairedness map).
  H  QC metric correlation matrix per dataset (depth vs quality metrics).

Non-overlap guardrail: no treatment-effect claims, no DiD estimates.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    apply_style,
    despine,
    save_panel,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    load_clinical_trial_dataset,
    clear_cache,
)

FIGURE_NAME = "SuppFig1_qc_metrics"

# ── dataset registry ─────────────────────────────────────────────────

_DS_PALETTE = dict(zip(
    ["Sade-Feldman", "Stephenson", "Vaccine", "AML", "CAR-T"],
    sns.color_palette("Set2", 5),
))


def _loaders():
    return [
        ("Sade-Feldman", get_sade_feldman),
        ("Stephenson", get_stephenson),
        ("Vaccine", get_vaccine),
        ("AML", lambda: load_clinical_trial_dataset("aml")),
        ("CAR-T", lambda: load_clinical_trial_dataset("cart")),
    ]


# ── helpers ──────────────────────────────────────────────────────────

def _pid_col(obs):
    for c in ("participant_id", "patient_id", "donor_id", "pt_id"):
        if c in obs.columns:
            return c
    return None


def _visit_col(obs):
    for c in ("visit",):
        if c in obs.columns:
            return c
    return None


def _arm_col(obs):
    for c in ("response", "severity", "therapy"):
        if c in obs.columns and obs[c].nunique() > 1:
            return c
    return None


def _get_ngenes(adata) -> np.ndarray:
    obs = adata.obs
    for col in ("n_genes_by_counts", "n_genes", "n_genes_detected", "NumberOfGenes"):
        if col in obs.columns:
            return np.asarray(obs[col], dtype=float)
    import scipy.sparse as sp
    for layer in ("counts", "tpm", "cpm"):
        if layer in adata.layers:
            X = adata.layers[layer]
            if sp.issparse(X):
                return np.asarray((X > 0).sum(axis=1), dtype=float).ravel()
            return np.asarray((X > 0).sum(axis=1), dtype=float).ravel()
    return np.full(adata.n_obs, np.nan)


def _get_counts(adata) -> np.ndarray:
    obs = adata.obs
    for col in ("total_counts", "n_counts", "total_UMI"):
        if col in obs.columns:
            return np.asarray(obs[col], dtype=float)
    import scipy.sparse as sp
    for layer in ("counts", "tpm", "cpm"):
        if layer in adata.layers:
            X = adata.layers[layer]
            if sp.issparse(X):
                return np.asarray(X.sum(axis=1), dtype=float).ravel()
            return np.asarray(X.sum(axis=1), dtype=float).ravel()
    return np.full(adata.n_obs, np.nan)


def _get_pct_mito(adata) -> np.ndarray:
    obs = adata.obs
    for col in ("pct_counts_mt", "pct_mito", "percent_mito"):
        if col in obs.columns:
            return np.asarray(obs[col], dtype=float)
    # Compute from counts if possible
    import scipy.sparse as sp
    for layer in ("counts", "tpm"):
        if layer in adata.layers:
            mt_mask = adata.var_names.str.upper().str.startswith("MT-")
            if mt_mask.sum() == 0:
                return np.full(adata.n_obs, np.nan)
            X = adata.layers[layer]
            if sp.issparse(X):
                mt_sum = np.asarray(X[:, mt_mask].sum(axis=1), dtype=float).ravel()
                total = np.asarray(X.sum(axis=1), dtype=float).ravel()
            else:
                mt_sum = np.asarray(X[:, mt_mask].sum(axis=1), dtype=float).ravel()
                total = np.asarray(X.sum(axis=1), dtype=float).ravel()
            return np.where(total > 0, mt_sum / total * 100, 0.0)
    return np.full(adata.n_obs, np.nan)


def _get_pct_ribo(adata) -> np.ndarray:
    obs = adata.obs
    for col in ("pct_counts_ribo", "pct_ribo", "percent_ribo"):
        if col in obs.columns:
            return np.asarray(obs[col], dtype=float)
    # Compute from counts if possible
    import scipy.sparse as sp
    for layer in ("counts", "tpm"):
        if layer in adata.layers:
            ribo_mask = (adata.var_names.str.upper().str.startswith("RPS") |
                         adata.var_names.str.upper().str.startswith("RPL"))
            if ribo_mask.sum() == 0:
                return np.full(adata.n_obs, np.nan)
            X = adata.layers[layer]
            if sp.issparse(X):
                rb_sum = np.asarray(X[:, ribo_mask].sum(axis=1), dtype=float).ravel()
                total = np.asarray(X.sum(axis=1), dtype=float).ravel()
            else:
                rb_sum = np.asarray(X[:, ribo_mask].sum(axis=1), dtype=float).ravel()
                total = np.asarray(X.sum(axis=1), dtype=float).ravel()
            return np.where(total > 0, rb_sum / total * 100, 0.0)
    return np.full(adata.n_obs, np.nan)


def _load_all(compute_umap: bool = False) -> dict:
    """Load all datasets and extract QC metrics (no UMAP needed)."""
    loaded = {}
    for name, loader in _loaders():
        try:
            adata = loader()
            pid = _pid_col(adata.obs)
            vis = _visit_col(adata.obs)
            arm = _arm_col(adata.obs)
            pid_counts = adata.obs.groupby(pid).size() if pid else pd.Series(dtype=int)

            loaded[name] = {
                "adata": adata,
                "n_cells": adata.n_obs,
                "n_genes_total": adata.n_vars,
                "n_participants": pid_counts.shape[0] if pid else 0,
                "n_samples": adata.obs.groupby([pid, vis]).ngroups
                    if pid and vis and vis in adata.obs.columns
                    else (pid_counts.shape[0] if pid else 0),
                "pid_col": pid,
                "visit_col": vis,
                "arm_col": arm,
                "cells_per_pid": pid_counts,
                "ngenes": _get_ngenes(adata),
                "total_counts": _get_counts(adata),
                "pct_mito": _get_pct_mito(adata),
                "pct_ribo": _get_pct_ribo(adata),
            }
            print(f"  {name}: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
    return loaded


# ── Panel A: dataset summary bars ────────────────────────────────────

def _panel_summary_bars(ax, loaded: dict):
    """Faceted bars: cells, genes, participants, samples per dataset."""
    ds_names = list(loaded.keys())
    metrics = {
        "Cells": [loaded[n]["n_cells"] for n in ds_names],
        "Genes": [loaded[n]["n_genes_total"] for n in ds_names],
        "Participants": [loaded[n]["n_participants"] for n in ds_names],
        "Samples": [loaded[n]["n_samples"] for n in ds_names],
    }

    x = np.arange(len(ds_names))
    n_metrics = len(metrics)
    width = 0.18
    metric_colors = sns.color_palette("tab10", n_metrics)

    for mi, (metric_name, vals) in enumerate(metrics.items()):
        offset = (mi - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=metric_name,
                      color=metric_colors[mi], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:,}" if v < 10000 else f"{v / 1000:.0f}k",
                    ha="center", va="bottom", fontsize=5.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title("Dataset Overview", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2, frameon=True)
    despine(ax)


# ── Panel B: QC retention waterfall ──────────────────────────────────

def _panel_retention_waterfall(ax, loaded: dict):
    """Show retention after QC filtering steps per dataset.

    Since we only have post-filter data, we estimate pre-filter counts
    from the gene-level: genes with zero expression across all cells
    were likely filtered, and we can compute the ratio of expressed genes.
    """
    ds_names = list(loaded.keys())
    import scipy.sparse as sp

    rows = []
    for name in ds_names:
        adata = loaded[name]["adata"]
        n_cells = adata.n_obs
        n_genes_total = adata.n_vars

        # Compute genes expressed (>0 in at least 1 cell)
        for layer in ("counts", "tpm", "cpm"):
            if layer in adata.layers:
                X = adata.layers[layer]
                if sp.issparse(X):
                    genes_expressed = int((np.asarray(X.sum(axis=0)).ravel() > 0).sum())
                else:
                    genes_expressed = int((X.sum(axis=0) > 0).sum())
                break
        else:
            genes_expressed = n_genes_total

        # Compute cells passing mito filter (if we have mito data)
        pct_mito = loaded[name]["pct_mito"]
        if not np.all(np.isnan(pct_mito)):
            cells_low_mito = int((pct_mito < 20).sum())
        else:
            cells_low_mito = n_cells

        rows.append({
            "Dataset": name,
            "Total genes": n_genes_total,
            "Expressed genes": genes_expressed,
            "Total cells": n_cells,
            "Cells (mito < 20%)": cells_low_mito,
        })

    df = pd.DataFrame(rows).set_index("Dataset")

    # Plot as grouped horizontal bars: genes and cells
    y = np.arange(len(ds_names))
    h = 0.35

    # Gene retention
    total_g = df["Total genes"].values.astype(float)
    expr_g = df["Expressed genes"].values.astype(float)
    pct_expr = expr_g / total_g * 100

    ax.barh(y + h / 2, total_g, height=h, label="Total genes",
            color="#aec7e8", edgecolor="white")
    ax.barh(y + h / 2, expr_g, height=h, label="Expressed genes",
            color="#1f77b4", edgecolor="white")

    # Cell retention
    total_c = df["Total cells"].values.astype(float)
    mito_c = df["Cells (mito < 20%)"].values.astype(float)

    ax.barh(y - h / 2, total_c, height=h, label="Total cells",
            color="#ffbb78", edgecolor="white")
    ax.barh(y - h / 2, mito_c, height=h, label="Cells (mito < 20%)",
            color="#ff7f0e", edgecolor="white")

    # Annotate percentages
    for i in range(len(ds_names)):
        ax.text(total_g[i], y[i] + h / 2, f" {pct_expr[i]:.0f}%",
                va="center", ha="left", fontsize=6)
        pct_cell = mito_c[i] / total_c[i] * 100 if total_c[i] > 0 else 100
        ax.text(total_c[i], y[i] - h / 2, f" {pct_cell:.0f}%",
                va="center", ha="left", fontsize=6)

    ax.set_yticks(y)
    ax.set_yticklabels(ds_names)
    ax.set_xlabel("Count")
    ax.set_xscale("log")
    ax.set_title("QC Retention Overview", fontweight="bold")
    ax.legend(fontsize=6, loc="lower right", frameon=True, ncol=2)
    despine(ax)


# ── Panel C: n_genes distributions by dataset + visit/arm ────────────

def _panel_ngenes_dist(ax, loaded: dict):
    """Violin: genes detected per cell, split by visit or arm."""
    rows = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        ngenes = data["ngenes"]
        vis = data["visit_col"]
        arm = data["arm_col"]

        # Subsample large datasets for violin performance
        n = len(ngenes)
        if n > 15000:
            idx = np.random.default_rng(42).choice(n, 15000, replace=False)
        else:
            idx = np.arange(n)

        split_col = vis if vis and vis in obs.columns and obs[vis].nunique() > 1 else arm
        if split_col and split_col in obs.columns:
            for val in sorted(obs[split_col].dropna().unique()):
                mask = obs[split_col].values == val
                sub_idx = idx[mask[idx]]
                if len(sub_idx) > 0:
                    rows.append(pd.DataFrame({
                        "Dataset": name, "Group": str(val),
                        "Genes": ngenes[sub_idx],
                    }))
        else:
            rows.append(pd.DataFrame({
                "Dataset": name, "Group": "All",
                "Genes": ngenes[idx],
            }))

    df = pd.concat(rows, ignore_index=True)
    order = list(loaded.keys())

    sns.violinplot(data=df, x="Dataset", y="Genes", hue="Group",
                   order=order, cut=0, inner="quartile", linewidth=0.5,
                   palette="pastel", density_norm="width", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Genes detected per cell")
    ax.set_title("Gene Detection by Dataset & Group", fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", title="Group", title_fontsize=7,
              frameon=True, ncol=2)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel D: total_counts distributions by dataset + visit/arm ───────

def _panel_counts_dist(ax, loaded: dict):
    """Violin: total counts per cell, split by visit or arm."""
    rows = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        counts = data["total_counts"]
        vis = data["visit_col"]
        arm = data["arm_col"]

        n = len(counts)
        if n > 15000:
            idx = np.random.default_rng(42).choice(n, 15000, replace=False)
        else:
            idx = np.arange(n)

        split_col = vis if vis and vis in obs.columns and obs[vis].nunique() > 1 else arm
        if split_col and split_col in obs.columns:
            for val in sorted(obs[split_col].dropna().unique()):
                mask = obs[split_col].values == val
                sub_idx = idx[mask[idx]]
                if len(sub_idx) > 0:
                    rows.append(pd.DataFrame({
                        "Dataset": name, "Group": str(val),
                        "Counts": np.log10(counts[sub_idx] + 1),
                    }))
        else:
            rows.append(pd.DataFrame({
                "Dataset": name, "Group": "All",
                "Counts": np.log10(counts[idx] + 1),
            }))

    df = pd.concat(rows, ignore_index=True)
    order = list(loaded.keys())

    sns.violinplot(data=df, x="Dataset", y="Counts", hue="Group",
                   order=order, cut=0, inner="quartile", linewidth=0.5,
                   palette="pastel", density_norm="width", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(r"$\log_{10}$(total counts + 1)")
    ax.set_title("Sequencing Depth by Dataset & Group", fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", title="Group", title_fontsize=7,
              frameon=True, ncol=2)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel E: % mito and % ribosomal distributions ───────────────────

def _panel_mito_ribo(ax, loaded: dict):
    """Side-by-side violins for % mito and % ribo with threshold overlays."""
    rows = []
    for name, data in loaded.items():
        pct_mt = data["pct_mito"]
        pct_rb = data["pct_ribo"]
        n = len(pct_mt)
        if n > 10000:
            idx = np.random.default_rng(42).choice(n, 10000, replace=False)
        else:
            idx = np.arange(n)
        if not np.all(np.isnan(pct_mt)):
            rows.append(pd.DataFrame({
                "Dataset": name, "Metric": "% Mito", "Value": pct_mt[idx]}))
        if not np.all(np.isnan(pct_rb)):
            rows.append(pd.DataFrame({
                "Dataset": name, "Metric": "% Ribo", "Value": pct_rb[idx]}))

    if not rows:
        ax.text(0.5, 0.5, "No mito/ribo data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, fontstyle="italic", color="#888888")
        ax.set_title("Mitochondrial & Ribosomal Content", fontweight="bold")
        return

    df = pd.concat(rows, ignore_index=True)
    order = list(loaded.keys())

    sns.violinplot(data=df, x="Dataset", y="Value", hue="Metric",
                   order=order, cut=0, inner="quartile", linewidth=0.5,
                   palette={"% Mito": "#e74c3c", "% Ribo": "#3498db"},
                   density_norm="width", split=False, ax=ax)

    # Threshold overlays
    ax.axhline(20, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Mito threshold (20%)")
    ax.axhline(50, color="#3498db", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Ribo threshold (50%)")

    ax.set_xlabel("")
    ax.set_ylabel("Percentage")
    ax.set_title("Mitochondrial & Ribosomal Content", fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", frameon=True)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel F: Lorenz curve + Gini per dataset ────────────────────────

def _panel_lorenz_gini(ax, loaded: dict):
    """Lorenz curve of cells-per-participant inequality per dataset."""
    for name, data in loaded.items():
        cpp = np.sort(data["cells_per_pid"].values.astype(float))
        if len(cpp) == 0:
            continue
        n = len(cpp)
        cum = np.cumsum(cpp) / cpp.sum()
        x_lorenz = np.concatenate([[0], np.arange(1, n + 1) / n])
        y_lorenz = np.concatenate([[0], cum])

        # Gini coefficient
        gini = 1 - 2 * np.trapz(y_lorenz, x_lorenz)

        ax.plot(x_lorenz, y_lorenz, linewidth=1.5,
                color=_DS_PALETTE.get(name, "black"),
                label=f"{name} (Gini={gini:.2f})")

    # Equality line
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Perfect equality")

    ax.set_xlabel("Cumulative fraction of participants")
    ax.set_ylabel("Cumulative fraction of cells")
    ax.set_title("Cell Allocation Inequality (Lorenz)", fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    despine(ax)


# ── Panel G: participant-by-visit completeness heatmaps ──────────────

def _panel_completeness(ax, loaded: dict):
    """Heatmap of participant × visit completeness across datasets."""
    # Collect per-dataset completeness
    ds_names = list(loaded.keys())
    rows = []
    for name in ds_names:
        adata = loaded[name]["adata"]
        obs = adata.obs
        pid = loaded[name]["pid_col"]
        vis = loaded[name]["visit_col"]
        if pid is None:
            continue
        if vis and vis in obs.columns and obs[vis].nunique() > 1:
            visits = sorted(obs[vis].dropna().unique())
            participants = sorted(obs[pid].dropna().unique())
            for p in participants:
                for v in visits:
                    n_cells = ((obs[pid] == p) & (obs[vis] == v)).sum()
                    rows.append({"Dataset": name, "Participant": p,
                                 "Visit": v, "Cells": int(n_cells)})
        else:
            participants = sorted(obs[pid].dropna().unique())
            for p in participants:
                n_cells = (obs[pid] == p).sum()
                rows.append({"Dataset": name, "Participant": p,
                             "Visit": "All", "Cells": int(n_cells)})

    if not rows:
        ax.text(0.5, 0.5, "No visit data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)

    # Show as summary: fraction of participants with cells at each visit
    summary_rows = []
    for name in ds_names:
        sub = df[df["Dataset"] == name]
        if sub.empty:
            continue
        for visit in sorted(sub["Visit"].unique()):
            vs = sub[sub["Visit"] == visit]
            n_with = (vs["Cells"] > 0).sum()
            n_total = vs.shape[0]
            summary_rows.append({
                "Dataset": name, "Visit": visit,
                "Fraction": n_with / n_total if n_total > 0 else 0,
                "Count": f"{n_with}/{n_total}",
            })

    sdf = pd.DataFrame(summary_rows)
    if sdf.empty:
        return

    # Pivot for heatmap
    pivot = sdf.pivot(index="Dataset", columns="Visit", values="Fraction")
    annot = sdf.pivot(index="Dataset", columns="Visit", values="Count")

    sns.heatmap(pivot, annot=annot, fmt="", cmap="YlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Fraction with cells",
                                                    "shrink": 0.7})
    ax.set_title("Participant × Visit Completeness", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    despine(ax)


# ── Panel H: QC metric correlation matrix ────────────────────────────

def _panel_qc_correlation(ax, loaded: dict):
    """Correlation heatmap of QC metrics (pooled across datasets)."""
    rows = []
    for name, data in loaded.items():
        n = data["n_cells"]
        if n > 10000:
            idx = np.random.default_rng(42).choice(n, 10000, replace=False)
        else:
            idx = np.arange(n)

        entry = {"Dataset": name}
        ng = data["ngenes"]
        tc = data["total_counts"]
        pm = data["pct_mito"]
        pr = data["pct_ribo"]

        for i in idx:
            row = dict(entry)
            row["Genes detected"] = ng[i]
            row["Total counts"] = tc[i]
            if not np.isnan(pm[i]):
                row["% Mito"] = pm[i]
            if not np.isnan(pr[i]):
                row["% Ribo"] = pr[i]
            rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c not in ("Dataset",)]
    corr = df[numeric_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                square=True, cbar_kws={"shrink": 0.7})
    ax.set_title("QC Metric Correlations (Pooled)", fontweight="bold")
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 1 panels."""
    print("Supplementary Figure 1: Data Quality and Cohort Integrity")
    loaded = _load_all()

    if not loaded:
        print("  No datasets loaded; skipping figure.")
        return

    # Panel A: Dataset summary bars
    fig, ax = plt.subplots(figsize=(8, 5))
    _panel_summary_bars(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: QC retention waterfall
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _panel_retention_waterfall(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: n_genes distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_ngenes_dist(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: total_counts distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_counts_dist(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: % mito and % ribosomal
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_mito_ribo(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: Lorenz + Gini
    fig, ax = plt.subplots(figsize=(6, 6))
    _panel_lorenz_gini(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Panel G: Completeness heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_completeness(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # Panel H: QC correlation matrix
    fig, ax = plt.subplots(figsize=(5.5, 5))
    _panel_qc_correlation(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    # Cleanup
    del loaded
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
