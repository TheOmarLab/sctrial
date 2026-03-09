"""
Supplementary Figure 1 — Data Quality and Cohort Characterisation.
===================================================================

Establish that all datasets are QC-sound and well-characterised before
inference.  Supports Main Figure 1 (Problem & Framework).

Panels
------
  A  Study design summary table (design, pairing, arms, N).
  B  Participant pairing structure per dataset.
  C  Cells per participant per arm (box + strip).
  D  Genes detected per cell distributions by dataset and group.
  E  Total counts per cell distributions by dataset and group.
  F  Mitochondrial & ribosomal content with threshold overlays.
  G  Lorenz curve + Gini inequality per dataset.
  H  Attrition + visit completeness (side-by-side).

Non-overlap guardrail: no treatment-effect claims, no DiD estimates.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

from .._shared import (
    SUPP_OUTPUT,
    apply_style,
    clear_cache,
    despine,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    load_clinical_trial_dataset,
    save_panel,
)

FIGURE_NAME = "SuppFig1_data_quality_cohort"

# ── dataset registry ─────────────────────────────────────────────────

_DS_PALETTE = dict(zip(
    ["Sade-Feldman", "Stephenson", "Vaccine", "AML", "CAR-T"],
    ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"],
))

DATASETS = [
    ("Sade-Feldman", get_sade_feldman),
    ("Stephenson", get_stephenson),
    ("Vaccine", get_vaccine),
    ("AML", lambda: load_clinical_trial_dataset("aml")),
    ("CAR-T", lambda: load_clinical_trial_dataset("cart")),
]

# ── dataset metadata ──────────────────────────────────────────────────

_DESIGN_META = {
    "Sade-Feldman": {
        "design": "Pre/post anti-PD-1",
        "pairing": "Partially paired",
        "arms": "Responder vs Non-responder",
        "indication": "Melanoma",
    },
    "Stephenson": {
        "design": "Cross-sectional COVID-19",
        "pairing": "Single observation",
        "arms": "Severity groups",
        "indication": "COVID-19",
    },
    "Vaccine": {
        "design": "Pre/post vaccination",
        "pairing": "Paired",
        "arms": "Single arm",
        "indication": "Influenza",
    },
    "AML": {
        "design": "Pre/post treatment",
        "pairing": "Paired (partial)",
        "arms": "Treatment vs Control",
        "indication": "AML",
    },
    "CAR-T": {
        "design": "Pre/post CAR-T infusion",
        "pairing": "Paired",
        "arms": "Single arm (CAR-T)",
        "indication": "B-ALL / DLBCL",
    },
}

# ── helpers ──────────────────────────────────────────────────────────


def _pid_col(obs):
    for c in ("participant_id", "patient_id", "donor_id", "pt_id"):
        if c in obs.columns:
            return c
    return None


def _visit_col(obs):
    for c in ("visit", "Collection_Day", "dfo_bin", "timepoint"):
        if c in obs.columns and obs[c].nunique() > 1:
            return c
    return None


def _arm_col(obs):
    for c in ("response", "severity", "therapy", "condition"):
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


# ── unified data loader ──────────────────────────────────────────────


def _load_all() -> dict:
    """Load all datasets with both QC metrics and clinical metadata."""
    loaded = {}
    for name, loader in DATASETS:
        try:
            adata = loader()
            if name == "Sade-Feldman":
                adata = harmonize_response(adata)
            obs = adata.obs
            pid = _pid_col(obs)
            vis = _visit_col(obs)
            arm = _arm_col(obs)
            pid_counts = obs.groupby(pid).size() if pid else pd.Series(dtype=int)

            loaded[name] = {
                "adata": adata,
                "n_cells": adata.n_obs,
                "n_genes_total": adata.n_vars,
                "n_participants": pid_counts.shape[0] if pid else 0,
                "n_samples": (adata.obs.groupby([pid, vis]).ngroups
                              if pid and vis and vis in obs.columns
                              else (pid_counts.shape[0] if pid else 0)),
                "pid_col": pid,
                "visit_col": vis,
                "arm_col": arm,
                "cells_per_pid": pid_counts,
                "ngenes": _get_ngenes(adata),
                "total_counts": _get_counts(adata),
                "pct_mito": _get_pct_mito(adata),
                "pct_ribo": _get_pct_ribo(adata),
            }
            print(f"  {name}: {adata.n_obs:,} cells, {adata.n_vars:,} genes, "
                  f"pid={pid}, vis={vis}, arm={arm}")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
    return loaded


# ── Panel A: Study design summary table ──────────────────────────────


def _panel_design_table(ax, loaded: dict):
    """Study design overview: cells, participants, visits per dataset."""
    ds_names = list(loaded.keys())
    n_cells = [loaded[n]["n_cells"] for n in ds_names]
    n_parts = [loaded[n]["n_participants"] for n in ds_names]

    x = np.arange(len(ds_names))
    colors = [_DS_PALETTE.get(n, "grey") for n in ds_names]

    bars = ax.bar(x, n_cells, color=colors, edgecolor="white", width=0.6)
    for bar, nc, np_ in zip(bars, n_cells, n_parts):
        ax.text(bar.get_x() + bar.get_width() / 2, nc + nc * 0.02,
                f"{nc:,}\n({np_} pts)",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontsize=9)
    ax.set_ylabel("Number of cells")
    ax.set_title("Dataset Overview: Cells and Participants", fontweight="bold")
    ax.set_yscale("log")
    # Powers-of-10 ticks only
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    despine(ax)


# ── Panel B: Participant pairing structure ────────────────────────────


def _panel_pairing(ax, loaded: dict):
    """Participant pairing structure: paired vs unpaired counts."""
    ds_names = []
    n_paired_list = []
    n_unpaired_list = []

    for name, data in loaded.items():
        obs = data["adata"].obs
        pid = data["pid_col"]
        vis = data["visit_col"]
        ds_names.append(name)

        if vis is None or pid is None:
            n_total = obs[pid].nunique() if pid else 0
            n_paired_list.append(0)
            n_unpaired_list.append(n_total)
            continue

        visits = sorted(obs[vis].dropna().unique())
        if len(visits) < 2:
            n_total = obs[pid].nunique() if pid else 0
            n_paired_list.append(0)
            n_unpaired_list.append(n_total)
            continue

        participants = sorted(obs[pid].dropna().unique())
        paired_count = 0
        for p in participants:
            p_visits = set(obs.loc[obs[pid] == p, vis].dropna().unique())
            if len(p_visits) >= 2:
                paired_count += 1

        n_paired_list.append(paired_count)
        n_unpaired_list.append(len(participants) - paired_count)

    x = np.arange(len(ds_names))
    w = 0.35

    ax.bar(x - w/2, n_paired_list, w, color="#2ecc71", edgecolor="white",
           label="Paired (\u22652 visits)")
    ax.bar(x + w/2, n_unpaired_list, w, color="#e74c3c", edgecolor="white",
           label="Unpaired (1 visit)")

    for i, (p, u) in enumerate(zip(n_paired_list, n_unpaired_list)):
        if p > 0:
            ax.text(i - w/2, p + 0.5, str(p), ha="center", va="bottom",
                    fontsize=7, fontweight="bold")
        if u > 0:
            ax.text(i + w/2, u + 0.5, str(u), ha="center", va="bottom",
                    fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontsize=9)
    ax.set_ylabel("Number of participants")
    ax.set_title("Participant Pairing Structure", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


# ── Panel C: Cells per participant per arm ────────────────────────────


def _panel_cells_per_pid_arm(ax, loaded: dict):
    """Box + strip: cells per participant, split by arm."""
    rows = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        pid = data["pid_col"]
        arm = data["arm_col"]
        if pid is None:
            continue

        if arm and arm in obs.columns:
            per_pid = obs.groupby([pid, arm]).size().reset_index(name="Cells")
            per_pid.rename(columns={arm: "Arm"}, inplace=True)
        else:
            per_pid = obs.groupby(pid).size().reset_index(name="Cells")
            per_pid["Arm"] = "All"

        per_pid["Dataset"] = name
        rows.append(per_pid)

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.concat(rows, ignore_index=True)
    df["Cells_log"] = np.log10(df["Cells"] + 1)

    sns.boxplot(data=df, x="Dataset", y="Cells_log", hue="Arm",
                order=list(loaded.keys()), fliersize=0, linewidth=0.5,
                palette="Dark2", ax=ax)
    sns.stripplot(data=df, x="Dataset", y="Cells_log", hue="Arm",
                  order=list(loaded.keys()), dodge=True, size=2, alpha=0.5,
                  palette="Dark2", ax=ax, legend=False)

    ax.set_xlabel("")
    ax.set_ylabel(r"$\log_{10}$(cells per participant + 1)")
    ax.set_title("Cells per Participant by Arm", fontweight="bold")
    ax.legend(fontsize=6, title="Arm", title_fontsize=7, loc="upper right",
              frameon=True, ncol=2)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel D: n_genes distributions ──────────────────────────────────


def _panel_ngenes_dist(ax, loaded: dict):
    """Violin: genes detected per cell, split by visit or arm."""
    rows = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        ngenes = data["ngenes"]
        vis = data["visit_col"]
        arm = data["arm_col"]

        n = len(ngenes)
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
                   palette="Dark2", density_norm="width", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Genes detected per cell")
    ax.set_title("Gene Detection by Dataset & Group", fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", title="Group", title_fontsize=7,
              frameon=True, ncol=2)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel E: total_counts distributions ──────────────────────────────


def _panel_counts_dist(ax, loaded: dict):
    """Violin: total counts per cell, split by visit or arm."""
    rows = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        counts = data["total_counts"]
        vis = data["visit_col"]
        arm = data["arm_col"]

        n = len(counts)
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
                   palette="Dark2", density_norm="width", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(r"$\log_{10}$(total counts + 1)")
    ax.set_title("Sequencing Depth by Dataset & Group", fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", title="Group", title_fontsize=7,
              frameon=True, ncol=2)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel F: % mito and % ribosomal ─────────────────────────────────


def _panel_mito_ribo(ax, loaded: dict):
    """Side-by-side violins for % mito and % ribo with threshold overlays."""
    rows = []
    for name, data in loaded.items():
        pct_mt = data["pct_mito"]
        pct_rb = data["pct_ribo"]
        n = len(pct_mt)
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


# ── Panel G: Lorenz curve + Gini per dataset ─────────────────────────


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
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4,
            label="Perfect equality")

    ax.set_xlabel("Cumulative fraction of participants")
    ax.set_ylabel("Cumulative fraction of cells")
    ax.set_title("Cell Allocation Inequality (Lorenz)", fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    despine(ax)


# ── Panel H helpers: Dropout + Completeness ──────────────────────────


def _panel_dropout(ax, loaded: dict):
    """Dropout rates: fraction of participants missing post-treatment."""
    rows = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        pid = data["pid_col"]
        vis = data["visit_col"]

        if pid is None or vis is None:
            continue

        visits = sorted(obs[vis].dropna().unique())
        if len(visits) < 2:
            continue

        # Identify pre and post
        pre_visits = [v for v in visits
                      if str(v).lower() in ("pre", "baseline", "d0", "day0", "0")]
        post_visits = [v for v in visits if v not in pre_visits]

        if not pre_visits or not post_visits:
            pre_visits = [visits[0]]
            post_visits = visits[1:]

        pre_pids = set(obs.loc[obs[vis].isin(pre_visits), pid].dropna().unique())
        post_pids = set(obs.loc[obs[vis].isin(post_visits), pid].dropna().unique())

        n_pre = len(pre_pids)
        n_post = len(pre_pids & post_pids)
        dropout_rate = 1 - (n_post / n_pre) if n_pre > 0 else 0

        rows.append({
            "Dataset": name,
            "N pre": n_pre,
            "N retained": n_post,
            "Dropout rate": dropout_rate,
        })

    # Include cross-sectional datasets as 0% dropout
    for name, data in loaded.items():
        if name not in [r["Dataset"] for r in rows]:
            pid = data["pid_col"]
            if pid:
                n = data["adata"].obs[pid].nunique()
                rows.append({
                    "Dataset": name,
                    "N pre": n,
                    "N retained": n,
                    "Dropout rate": 0.0,
                })

    if not rows:
        ax.text(0.5, 0.5, "No longitudinal data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        ax.set_title("Attrition Rates", fontweight="bold")
        return

    df = pd.DataFrame(rows)
    colors = [_DS_PALETTE.get(n, "grey") for n in df["Dataset"]]

    bars = ax.bar(df["Dataset"], df["Dropout rate"], color=colors,
                  edgecolor="white", width=0.6)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                row["Dropout rate"] + 0.02,
                f"{row['Dropout rate']:.0%}\n({row['N retained']}/{row['N pre']})",
                ha="center", va="bottom", fontsize=6)

    ax.set_ylabel("Dropout rate")
    ax.set_title("Participant Attrition (Pre → Post)", fontweight="bold")
    ax.set_ylim(0, min(1.0, df["Dropout rate"].max() * 1.5 + 0.1))
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


def _panel_completeness_detailed(ax, loaded: dict):
    """Bar chart: fraction of participants with cells at each visit."""
    rows = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        pid = data["pid_col"]
        vis = data["visit_col"]
        if pid is None:
            continue

        if vis and vis in obs.columns and obs[vis].nunique() > 1:
            visits = sorted(obs[vis].dropna().unique())
            participants = sorted(obs[pid].dropna().unique())
            for v in visits:
                n_total = len(participants)
                n_with = len(set(obs.loc[(obs[vis] == v) &
                                         obs[pid].notna(), pid].unique()))
                label = f"{name}\n{v}"
                rows.append({
                    "Label": label,
                    "Fraction": n_with / n_total if n_total > 0 else 0,
                    "Count": f"{n_with}/{n_total}",
                })
        else:
            participants = sorted(obs[pid].dropna().unique())
            rows.append({
                "Label": f"{name}\nAll",
                "Fraction": 1.0,
                "Count": f"{len(participants)}/{len(participants)}",
            })

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)

    y_pos = np.arange(len(df))
    colors = ["#27ae60" if f >= 0.9 else "#f39c12" if f >= 0.5 else "#e74c3c"
              for f in df["Fraction"]]

    ax.barh(y_pos, df["Fraction"], color=colors, height=0.6,
            edgecolor="white")
    for i, row in df.iterrows():
        ax.text(row["Fraction"] + 0.02, i, row["Count"],
                va="center", ha="left", fontsize=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Label"], fontsize=6)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Fraction of participants with cells")
    ax.set_title("Visit Completeness", fontweight="bold")
    despine(ax)


# ======================================================================
# Generate
# ======================================================================


def generate():
    """Create and save Supplementary Figure 1 panels.

    Layout (8 panels):
      A  Study design summary table
      B  Participant pairing structure
      C  Cells per participant by arm
      D  Genes detected per cell distributions
      E  Total counts per cell distributions
      F  Mitochondrial & ribosomal content
      G  Lorenz curve + Gini inequality
      H  Attrition + visit completeness (side-by-side)
    """
    print("Supplementary Figure 1: Data Quality & Cohort Characterisation")
    loaded = _load_all()

    if not loaded:
        print("  No datasets loaded; skipping figure.")
        return

    # Panel A: Study design summary table
    fig, ax = plt.subplots(figsize=(10, 4))
    _panel_design_table(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: Participant pairing structure
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_pairing(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: Cells per participant by arm
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_cells_per_pid_arm(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: Genes detected per cell distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_ngenes_dist(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: Total counts per cell distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_counts_dist(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: Mitochondrial & ribosomal content
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_mito_ribo(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Panel G: Lorenz curve + Gini inequality
    fig, ax = plt.subplots(figsize=(6, 6))
    _panel_lorenz_gini(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # Panel H: Attrition + visit completeness (side-by-side)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    _panel_dropout(ax_left, loaded)
    _panel_completeness_detailed(ax_right, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    # Cleanup
    for data in loaded.values():
        del data["adata"]
    loaded.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
