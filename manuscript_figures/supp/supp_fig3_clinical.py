"""
Supplementary Figure 3 — Clinical Trial Dataset Details.
========================================================

Per-dataset panels for AML (GSE116256) and CAR-T (GSE290722):

Per dataset (3 panels each, 6 total):
  Cell-type composition (horizontal bar).
  QC distributions (side-by-side violins: genes detected + total UMI).
  Participant structure (box + strip by timepoint).
"""

from __future__ import annotations

import gc
import re as _re

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
    load_clinical_trial_dataset,
    clear_cache,
)

FIGURE_NAME = "SuppFig3_clinical_datasets"
DATASETS = ["aml", "cart"]
DATASET_LABELS = {"aml": "AML (GSE116256)", "cart": "CAR-T (GSE290722)"}
FIGSIZE = (16, 5 * len(DATASETS))


# ── helpers ───────────────────────────────────────────────────────────

def _find_celltype_col(obs: pd.DataFrame) -> str | None:
    """Locate the cell-type column."""
    for col in ("cell_type", "celltype", "CellType", "cell_type_fine",
                "cell_type_coarse", "celltype_major"):
        if col in obs.columns:
            return col
    return None


def _find_pid_col(obs: pd.DataFrame) -> str:
    """Locate participant ID column."""
    for col in ("participant_id", "patient_id", "donor_id", "sample_id"):
        if col in obs.columns:
            return col
    raise KeyError("No participant ID column found in obs")


def _find_visit_col(obs: pd.DataFrame) -> str | None:
    """Locate the most granular visit/timepoint column."""
    # Prefer the more detailed column (timepoint > timepoint_category > visit)
    for col in ("timepoint", "timepoint_category", "time_point",
                 "visit", "condition"):
        if col in obs.columns:
            return col
    return None


def _get_ngenes(obs: pd.DataFrame, adata=None) -> np.ndarray | None:
    """Extract gene-count array."""
    for col in ("n_genes_by_counts", "n_genes", "n_genes_detected"):
        if col in obs.columns:
            return np.asarray(obs[col], dtype=float)
    if adata is not None:
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return (X > 0).sum(axis=1).astype(float)
    return None


def _get_counts(obs: pd.DataFrame, adata=None) -> np.ndarray | None:
    """Extract total counts array."""
    for col in ("total_counts", "n_counts", "total_UMI"):
        if col in obs.columns:
            return np.asarray(obs[col], dtype=float)
    if adata is not None:
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X.sum(axis=1).astype(float)
    return None


# ── panel functions ───────────────────────────────────────────────────

def _panel_celltype(ax, adata, title: str):
    """Horizontal bar chart of cell-type composition."""
    ct_col = _find_celltype_col(adata.obs)
    if ct_col is None:
        ax.text(0.5, 0.5, "No cell-type\nannotation", ha="center",
                va="center", transform=ax.transAxes, fontsize=11,
                color=COLORS["gray"])
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
        return

    counts = adata.obs[ct_col].value_counts().sort_values(ascending=True)
    # Limit to top 15 for readability
    if len(counts) > 15:
        other = counts.iloc[:-15].sum()
        counts = counts.iloc[-15:]
        counts["Other"] = other
        counts = counts.sort_values(ascending=True)

    n_ct = len(counts)
    if n_ct <= 10:
        colors = sns.color_palette("tab10", n_colors=n_ct)
    elif n_ct <= 20:
        colors = sns.color_palette("tab20", n_colors=n_ct)
    else:
        colors = sns.color_palette("husl", n_colors=n_ct)
    ax.barh(range(len(counts)), counts.values, color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=7)
    ax.set_xlabel("Number of cells")
    ax.set_title(title, fontweight="bold")

    # Add count labels
    for i, (ct, cnt) in enumerate(counts.items()):
        ax.text(cnt + counts.max() * 0.01, i, f"{cnt:,}",
                va="center", fontsize=6, color=COLORS["gray"])

    despine(ax)


def _panel_qc(axes, adata, title: str):
    """Violin plots of genes detected and total UMI on separate subplots.

    Parameters
    ----------
    axes : sequence of two Axes
        Left axis for genes detected, right for total UMI.
    """
    obs = adata.obs
    n_genes = _get_ngenes(obs, adata)
    total_counts = _get_counts(obs, adata)

    metrics = []
    if n_genes is not None:
        metrics.append(("Genes detected", n_genes, COLORS["treated"]))
    if total_counts is not None:
        metrics.append(("Total UMI", total_counts, COLORS["control"]))

    if not metrics:
        for ax in axes:
            ax.text(0.5, 0.5, "No QC metrics\navailable", ha="center",
                    va="center", transform=ax.transAxes, fontsize=11,
                    color=COLORS["gray"])
            ax.set_title(title, fontweight="bold")
        return

    for ax, (label, vals, color) in zip(axes, metrics):
        # Subsample large arrays for plotting speed
        if len(vals) > 20_000:
            idx = np.random.default_rng(42).choice(len(vals), 20_000,
                                                    replace=False)
            vals = vals[idx]

        # Use log-scale for UMI counts (values span orders of magnitude)
        use_log = label == "Total UMI" and vals.max() / (np.median(vals) + 1) > 10
        plot_vals = np.log10(vals + 1) if use_log else vals

        parts = ax.violinplot([plot_vals], positions=[0], showmedians=True,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")

        med = np.median(vals)
        med_plot = np.log10(med + 1) if use_log else med
        ax.text(0, med_plot, f"  {med:,.0f}", ha="left", va="bottom",
                fontsize=8, fontweight="bold")
        ax.set_xticks([0])
        ax.set_xticklabels([label], fontsize=9)
        y_label = rf"$\log_{{10}}$({label})" if use_log else label
        ax.set_ylabel(y_label)
        ax.set_title(f"{title}: {label}" if len(metrics) > 1 else title,
                     fontweight="bold")
        despine(ax)


def _panel_structure(ax, adata, title: str):
    """Participant structure: cells per participant, grouped by visit."""
    obs = adata.obs
    pid_col = _find_pid_col(obs)
    visit_col = _find_visit_col(obs)

    if visit_col is not None and obs[visit_col].nunique() > 1:
        # Grouped bar: cells per participant per visit
        grouped = (
            obs.groupby([pid_col, visit_col], observed=True)
            .size()
            .reset_index(name="n_cells")
        )
        _VISIT_ORDER = {
            "leukapheresis": 0, "pre": 0, "baseline": 0,
            "diagnosis": 0,
            "early_post": 1, "post": 1,
            "mid_post": 2,
            "late_post": 3,
            "follow-up": 4,
        }
        def _visit_sort_key(v):
            s = str(v).lower()
            if s in _VISIT_ORDER:
                return _VISIT_ORDER[s]
            digits = _re.findall(r"\d+", s)
            return int(digits[0]) if digits else 0
        visits = sorted(grouped[visit_col].unique(), key=_visit_sort_key)

        palette = [COLORS["treated"], COLORS["control"], COLORS["neutral"],
                   COLORS["highlight"]]

        sns.boxplot(
            data=grouped, x=visit_col, y="n_cells", order=visits,
            palette=palette[:len(visits)], width=0.5,
            showcaps=True, showfliers=False,
            boxprops=dict(alpha=0.5),
            ax=ax,
        )
        sns.stripplot(
            data=grouped, x=visit_col, y="n_cells", order=visits,
            color="black", size=4, alpha=0.6, jitter=0.15, ax=ax,
        )
        ax.set_xlabel("Visit / Timepoint")
        ax.set_ylabel("Cells per participant")
        ax.tick_params(axis="x", rotation=30)
    else:
        # Simple bar: cells per participant
        cpp = obs.groupby(pid_col, observed=True).size().sort_values(
            ascending=False)

        ax.bar(range(len(cpp)), cpp.values, color=COLORS["treated"],
               edgecolor="white", linewidth=0.3, width=0.8)
        ax.set_xlabel("Participant (sorted)")
        ax.set_ylabel("Number of cells")
        ax.set_xticks([])

    # Summary text
    n_pid = obs[pid_col].nunique()
    n_visits = obs[visit_col].nunique() if visit_col else 1
    ax.text(
        0.97, 0.95,
        f"{n_pid} participants\n{n_visits} timepoint(s)\n{adata.n_obs:,} cells",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"],
                  alpha=0.8),
    )

    ax.set_title(title, fontweight="bold")
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Supplementary Figure 3 individual panels."""
    print("Supplementary Figure 3: Clinical Dataset Details")

    loaded = {}
    for name in DATASETS:
        try:
            loaded[name] = load_clinical_trial_dataset(name)
        except Exception as exc:
            print(f"  {name}: failed to load ({exc})")

    if not loaded:
        print("  No clinical datasets available; skipping figure.")
        return

    label_idx = 0
    label_chars = "ABCDEFGHIJKLMNOP"

    # ── Save individual panels ────────────────────────────────────────
    for name, adata in loaded.items():
        ds_label = DATASET_LABELS.get(name, name.upper())

        # Panel: Cell types
        fig_ct, ax_ct = plt.subplots(figsize=(6, 5))
        _panel_celltype(ax_ct, adata, f"{ds_label} — Cell Types")
        fig_ct.tight_layout()
        save_panel(fig_ct, f"panel_{label_chars[label_idx]}",
                   FIGURE_NAME, SUPP_OUTPUT)
        label_idx += 1

        # Panels: QC (genes detected + total UMI as separate panels)
        fig_qc, axes_qc = plt.subplots(1, 2, figsize=(8, 4))
        _panel_qc(axes_qc, adata, ds_label)
        fig_qc.tight_layout()
        save_panel(fig_qc, f"panel_{label_chars[label_idx]}",
                   FIGURE_NAME, SUPP_OUTPUT)
        label_idx += 1

        # Panel: Sample structure
        fig_st, ax_st = plt.subplots(figsize=(6, 5))
        _panel_structure(ax_st, adata, f"{ds_label} — Sample Structure")
        fig_st.tight_layout()
        save_panel(fig_st, f"panel_{label_chars[label_idx]}",
                   FIGURE_NAME, SUPP_OUTPUT)
        label_idx += 1

    # ── Cleanup ───────────────────────────────────────────────────────
    for adata in loaded.values():
        del adata
    loaded.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
