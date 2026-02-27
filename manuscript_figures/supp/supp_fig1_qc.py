"""
Supplementary Figure 1 — QC Metrics Across Datasets.
=====================================================

2x3 grid (14x9 inches):

Row 1  Per-dataset QC distributions (Sade-Feldman, Stephenson, Vaccine).
       Each panel shows genes detected vs total UMI counts as a scatter,
       coloured by cells per participant.
Row 2  Cross-dataset comparisons via violin/box plots:
       (D) genes detected, (E) total UMI counts, (F) cells per participant.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
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
    clear_cache,
)

FIGURE_NAME = "SuppFig1_qc_metrics"
FIGSIZE = (14, 9)


# ── helpers ───────────────────────────────────────────────────────────

def _get_ngenes(obs: pd.DataFrame) -> pd.Series | None:
    """Return gene-count column (whichever name exists)."""
    for col in ("n_genes_by_counts", "n_genes", "n_genes_detected"):
        if col in obs.columns:
            return obs[col]
    return None


def _get_counts(obs: pd.DataFrame) -> pd.Series | None:
    """Return total counts column."""
    for col in ("total_counts", "n_counts", "total_UMI"):
        if col in obs.columns:
            return obs[col]
    return None


def _cells_per_participant(obs: pd.DataFrame) -> pd.Series:
    """Return cells-per-participant Series."""
    pid_col = "participant_id" if "participant_id" in obs.columns else "patient_id"
    return obs.groupby(pid_col).size()


def _load_qc_data() -> dict:
    """Load all three datasets and extract QC columns."""
    datasets = {}
    loaders = [
        ("Sade-Feldman", get_sade_feldman),
        ("Stephenson", get_stephenson),
        ("Vaccine", get_vaccine),
    ]
    for name, loader in loaders:
        try:
            adata = loader()
            obs = adata.obs.copy()
            n_genes = _get_ngenes(obs)
            total_counts = _get_counts(obs)

            # If columns not in obs, try to compute from X
            if n_genes is None:
                X = adata.X
                if hasattr(X, "toarray"):
                    X = X.toarray()
                n_genes = pd.Series((X > 0).sum(axis=1), index=obs.index,
                                    name="n_genes")
            if total_counts is None:
                X = adata.X
                if hasattr(X, "toarray"):
                    X = X.toarray()
                total_counts = pd.Series(X.sum(axis=1), index=obs.index,
                                         name="total_counts")

            cells_per_pid = _cells_per_participant(obs)

            datasets[name] = {
                "n_genes": np.asarray(n_genes, dtype=float),
                "total_counts": np.asarray(total_counts, dtype=float),
                "cells_per_participant": cells_per_pid,
                "n_cells": adata.n_obs,
                "n_participants": cells_per_pid.shape[0],
            }
            print(f"  {name}: {adata.n_obs:,} cells, "
                  f"{cells_per_pid.shape[0]} participants")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")

    return datasets


# ── Row 1 panels: per-dataset scatter ─────────────────────────────────

def _panel_scatter(ax, data: dict, title: str):
    """Scatter of genes vs total counts, coloured by density."""
    n_genes = data["n_genes"]
    total_counts = data["total_counts"]

    # Subsample for large datasets
    n = len(n_genes)
    if n > 10_000:
        idx = np.random.default_rng(42).choice(n, 10_000, replace=False)
        n_genes = n_genes[idx]
        total_counts = total_counts[idx]

    ax.scatter(
        total_counts, n_genes,
        s=2, alpha=0.15, color=COLORS["treated"], rasterized=True,
    )
    ax.set_xlabel("Total counts (UMI)")
    ax.set_ylabel("Genes detected")
    ax.set_title(title, fontweight="bold")

    # Summary text
    ax.text(
        0.97, 0.05,
        f"n = {data['n_cells']:,} cells\n"
        f"{data['n_participants']} participants",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"],
                  alpha=0.8),
    )
    despine(ax)


# ── Row 2 panels: cross-dataset violins ──────────────────────────────

def _panel_violin(ax, datasets: dict, metric: str, ylabel: str, title: str):
    """Violin + strip plot comparing a QC metric across datasets."""
    rows = []
    for ds_name, data in datasets.items():
        values = data[metric]
        if isinstance(values, pd.Series):
            values = values.values
        for v in values:
            rows.append({"Dataset": ds_name, "value": float(v)})

    df = pd.DataFrame(rows)
    if df.empty:
        ax.set_title(title)
        return

    palette = [COLORS["treated"], COLORS["control"], COLORS["neutral"]]
    order = list(datasets.keys())

    sns.violinplot(
        data=df, x="Dataset", y="value", order=order,
        palette=palette[:len(order)], inner=None, linewidth=0.8,
        cut=0, ax=ax, alpha=0.6,
    )
    sns.boxplot(
        data=df, x="Dataset", y="value", order=order,
        width=0.15, showcaps=False, showfliers=False,
        boxprops=dict(facecolor="white", linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


def _panel_cells_per_participant(ax, datasets: dict):
    """Bar chart of cells per participant across datasets."""
    rows = []
    for ds_name, data in datasets.items():
        cpp = data["cells_per_participant"]
        for v in cpp.values:
            rows.append({"Dataset": ds_name, "Cells": int(v)})

    df = pd.DataFrame(rows)
    if df.empty:
        ax.set_title("Cells per Participant")
        return

    order = list(datasets.keys())
    palette = [COLORS["treated"], COLORS["control"], COLORS["neutral"]]

    sns.violinplot(
        data=df, x="Dataset", y="Cells", order=order,
        palette=palette[:len(order)], inner=None, linewidth=0.8,
        cut=0, ax=ax, alpha=0.6,
    )
    sns.stripplot(
        data=df, x="Dataset", y="Cells", order=order,
        color="black", size=3, alpha=0.6, jitter=0.15, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Cells per participant")
    ax.set_title("F  Cells per Participant", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Supplementary Figure 1 individual panels."""
    print("Supplementary Figure 1: QC Metrics")
    datasets = _load_qc_data()

    if not datasets:
        print("  No datasets loaded; skipping figure.")
        return

    ds_names = list(datasets.keys())

    # ── Save individual panels ────────────────────────────────────────
    # Row 1 panels
    for i, ds_name in enumerate(ds_names[:3]):
        fig_p, ax_p = plt.subplots(figsize=(5, 4))
        _panel_scatter(ax_p, datasets[ds_name], ds_name)
        fig_p.tight_layout()
        save_panel(fig_p, f"panel_{chr(65 + i)}", FIGURE_NAME, SUPP_OUTPUT)

    # Row 2 panels
    fig_d, ax_d2 = plt.subplots(figsize=(5, 4))
    _panel_violin(ax_d2, datasets, "n_genes", "Genes detected",
                  "Genes Detected")
    fig_d.tight_layout()
    save_panel(fig_d, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    fig_e, ax_e2 = plt.subplots(figsize=(5, 4))
    _panel_violin(ax_e2, datasets, "total_counts", "Total UMI counts",
                  "Total UMI Counts")
    fig_e.tight_layout()
    save_panel(fig_e, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    fig_f, ax_f2 = plt.subplots(figsize=(5, 4))
    _panel_cells_per_participant(ax_f2, datasets)
    fig_f.tight_layout()
    save_panel(fig_f, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # ── Cleanup ───────────────────────────────────────────────────────
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
