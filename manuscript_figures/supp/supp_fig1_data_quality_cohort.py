"""
Supplementary Figure 1 — Data Quality and Cohort Characterisation.
===================================================================

Establish that all datasets are QC-sound and well-characterised before
inference.  Supports Main Figure 1 (Problem & Framework).

Panels
------
  A  Participant pairing structure per dataset.
  B  Participant counts per arm × visit (grouped bar chart).
  C  Cells per participant by arm (box + strip).
  D  Genes detected per cell distributions by dataset and group.
  E  Total counts + mito/ribo QC merged (1×2).
  F  Lorenz curve + Gini inequality per dataset.
  G  Post-QC threshold compliance per dataset.
  H  Visit completeness per dataset.

Non-overlap guardrail: no treatment-effect claims, no DiD estimates.
"""

from __future__ import annotations

import gc
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from .._shared import (
    SUPP_OUTPUT,
    apply_style,
    clear_cache,
    despine,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    get_aml,
    get_cart,
    get_tnbc_zhang,
    save_panel,
)

FIGURE_NAME = "SuppFig1_data_quality_cohort"

# ── dataset registry ─────────────────────────────────────────────────

_DS_PALETTE = dict(zip(
    ["TNBC", "Melanoma", "COVID-19", "Vaccine", "AML", "CAR-T"],
    ["#996633", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"],
))

DATASETS = [
    ("TNBC", lambda: get_tnbc_zhang()),
    ("Melanoma", get_sade_feldman),
    ("COVID-19", get_stephenson),
    ("Vaccine", get_vaccine),
    ("AML", lambda: get_aml()),
    ("CAR-T", lambda: get_cart()),
]

# ── dataset metadata ──────────────────────────────────────────────────

_DESIGN_META = {
    "Melanoma": {
        "design": "Pre/post anti-PD-1",
        "pairing": "Partially paired",
        "arms": "Responder vs Non-responder",
        "indication": "Melanoma",
        "visits": "Pre, Post",
        "estimand": "DiD (two-arm)",
    },
    "COVID-19": {
        "design": "COVID-19 severity comparison",
        "pairing": "Multi-visit (grouped by severity)",
        "arms": "Severity groups",
        "indication": "COVID-19",
        "visits": "Multiple collection days",
        "estimand": "DiD (two-arm)",
    },
    "Vaccine": {
        "design": "Pre/post vaccination",
        "pairing": "Paired",
        "arms": "Single arm",
        "indication": "Influenza",
        "visits": "Pre, Post",
        "estimand": "Δ (single-arm)",
    },
    "AML": {
        "design": "Pre/post treatment",
        "pairing": "Paired (partial); control largely pre-only",
        "arms": "Treatment vs Control",
        "indication": "AML",
        "visits": "Pre, Post",
        "estimand": "Δ (two-arm)",
    },
    "CAR-T": {
        "design": "Pre/post CAR-T infusion",
        "pairing": "Partially paired",
        "arms": "Single arm (CAR-T)",
        "indication": "B-ALL / DLBCL",
        "visits": "Pre, Post",
        "estimand": "Δ (single-arm)",
    },
    "TNBC": {
        "design": "Pre/post anti-PD-L1 + chemo vs chemo",
        "pairing": "Paired",
        "arms": "anti-PDL1+Chemo vs Chemo",
        "indication": "Triple-negative breast cancer",
        "visits": "Pre, Post",
        "estimand": "DiD (two-arm)",
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
    for c in ("arm", "response", "severity", "therapy", "condition"):
        if c in obs.columns and obs[c].nunique() > 1:
            return c
    return None


# Known chronological visit-label orderings. Plain alphabetical sort puts
# "Post" before "Pre" (P-o < P-r), which is backwards — this table lets
# panel B (and anywhere else ordering matters) request the correct order.
_VISIT_ORDER_KNOWN = [
    ["Pre", "Post"],
    ["Baseline", "Pre", "Post"],
    ["D0", "D28"],
]


def _visit_sort_order(values) -> list:
    """Return *values* in chronological order.

    Checks known pre/post-style orderings first; falls back to natural
    (numeric-aware) sort for things like day-numbered visits not already
    covered, so e.g. "D2" sorts before "D10".
    """
    vals = list(values)
    val_set = set(vals)
    for known in _VISIT_ORDER_KNOWN:
        if val_set == set(known):
            return known

    def _natural_key(v):
        import re
        parts = re.split(r"(\d+)", str(v))
        return [int(p) if p.isdigit() else p for p in parts]

    return sorted(vals, key=_natural_key)


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
            if name == "Melanoma":
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


# ── Panel A: Participant pairing structure ────────────────────────────


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

    ax.bar(x - w / 2, n_paired_list, w, color="#2ecc71", edgecolor="white",
           label="Paired (\u22652 visits)")
    ax.bar(x + w / 2, n_unpaired_list, w, color="#e74c3c", edgecolor="white",
           label="Unpaired (1 visit)")

    for i, (p, u) in enumerate(zip(n_paired_list, n_unpaired_list)):
        if p > 0:
            ax.text(i - w / 2, p + 0.5, str(p), ha="center", va="bottom",
                    fontsize=7, fontweight="bold")
        if u > 0:
            ax.text(i + w / 2, u + 0.5, str(u), ha="center", va="bottom",
                    fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Number of participants")
    ax.set_title("Participant Pairing Structure", fontweight="bold")
    ax.legend(fontsize=8, frameon=True, loc="upper left")
    despine(ax)


# ── Panel B: Participant counts per arm × visit ──────────────────────


def _panel_participant_counts(fig, loaded: dict):
    """Faceted bar chart: participant N per arm × visit (one subplot per dataset)."""
    names = list(loaded.keys())
    ncols = len(names)
    axes = fig.subplots(1, ncols, sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        data = loaded[name]
        obs = data["adata"].obs
        pid = data["pid_col"]
        arm = data["arm_col"]
        vis = data["visit_col"]

        if pid is None:
            ax.set_title(name, fontweight="bold", fontsize=9)
            ax.axis("off")
            continue

        # Build grouped counts (observed=True to avoid phantom bins).
        # Cast grouping columns to str to strip categorical phantom levels.
        if arm and vis and arm in obs.columns and vis in obs.columns:
            grp = (obs.assign(**{arm: obs[arm].astype(str),
                                 vis: obs[vis].astype(str)})
                   .groupby([arm, vis], observed=True)[pid]
                   .nunique().reset_index(name="N"))
            grp.rename(columns={arm: "Arm", vis: "Visit"}, inplace=True)
            x_col, hue_col = "Visit", "Arm"
        elif vis and vis in obs.columns:
            grp = (obs.assign(**{vis: obs[vis].astype(str)})
                   .groupby(vis, observed=True)[pid]
                   .nunique().reset_index(name="N"))
            grp.rename(columns={vis: "Visit"}, inplace=True)
            grp["Arm"] = "All"
            x_col, hue_col = "Visit", None
        elif arm and arm in obs.columns:
            grp = (obs.assign(**{arm: obs[arm].astype(str)})
                   .groupby(arm, observed=True)[pid]
                   .nunique().reset_index(name="N"))
            grp.rename(columns={arm: "Arm"}, inplace=True)
            grp["Visit"] = "All"
            x_col, hue_col = "Arm", None
        else:
            ax.set_title(name, fontweight="bold", fontsize=9)
            ax.axis("off")
            continue

        # Drop zero-count bins from unobserved categorical combos
        grp = grp[grp["N"] > 0]

        _order = _visit_sort_order(grp[x_col].unique()) if x_col == "Visit" else None

        if hue_col:
            sns.barplot(data=grp, x=x_col, y="N", hue=hue_col, order=_order,
                        palette="Dark2", edgecolor="white", ax=ax)
            ax.legend(fontsize=6, title=hue_col, title_fontsize=7,
                      loc="upper right", frameon=True)
        else:
            sns.barplot(data=grp, x=x_col, y="N", order=_order,
                        color="#1b9e77", edgecolor="white", ax=ax)

        # Annotate bar values
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{int(h)}", ha="center", va="bottom", fontsize=6)

        ax.set_title(name, fontweight="bold", fontsize=9)
        ax.set_xlabel("")
        if ax == axes[0]:
            ax.set_ylabel("Participants")
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        despine(ax)

    fig.suptitle("Participants per Arm × Visit", fontweight="bold", fontsize=11)


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
            per_pid = (
                obs.groupby([pid, arm], observed=True)
                .size()
                .reset_index(name="Cells")
            )
            per_pid = per_pid[per_pid["Cells"] > 0]
            per_pid.rename(columns={arm: "Arm"}, inplace=True)
        else:
            per_pid = obs.groupby(pid, observed=True).size().reset_index(name="Cells")
            per_pid = per_pid[per_pid["Cells"] > 0]
            per_pid["Arm"] = "All"

        per_pid["Dataset"] = name
        rows.append(per_pid)

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.concat(rows, ignore_index=True)
    df["Cells_log"] = np.log10(df["Cells"] + 1)

    # Use categorical positions explicitly to avoid seaborn 0.13 dodge centering bug
    ds_order = list(loaded.keys())
    arms_present = sorted(df["Arm"].unique())
    n_arms = len(arms_present)
    # "All" is a placeholder for single-arm datasets; keep it neutral grey
    # so it never clashes with real arm labels like "anti-PDL1+Chemo".
    _non_all = sorted(a for a in arms_present if a != "All")
    _arm_pal = sns.color_palette("Dark2", max(len(_non_all), 1))
    arm_colors = {"All": "#AAAAAA"}
    for _a, _c in zip(_non_all, _arm_pal):
        arm_colors[_a] = _c

    # Box width based on max arms any single dataset has
    max_arms_per_ds = max(
        len(df[df["Dataset"] == ds]["Arm"].unique())
        for ds in ds_order
    )
    box_width = 0.9 / max_arms_per_ds
    for tick_idx, ds_name in enumerate(ds_order):
        ds_df = df[df["Dataset"] == ds_name]
        ds_arms = sorted(ds_df["Arm"].unique())
        n_ds_arms = len(ds_arms)
        # Center this dataset's boxes around tick_idx
        offsets = np.linspace(-(n_ds_arms - 1) / 2 * box_width,
                              (n_ds_arms - 1) / 2 * box_width,
                              n_ds_arms)
        for arm, off in zip(ds_arms, offsets):
            arm_df = ds_df[ds_df["Arm"] == arm]["Cells_log"].dropna()
            if len(arm_df) == 0:
                continue
            color = arm_colors.get(arm, "grey")
            bp = ax.boxplot(arm_df, positions=[tick_idx + off],
                            widths=box_width * 0.85,
                            patch_artist=True,
                            showfliers=False,
                            medianprops=dict(color="black", linewidth=1.5),
                            boxprops=dict(facecolor=color, alpha=0.8,
                                          linewidth=0.8),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8))
            # Strip dots
            jitter = np.random.default_rng(42).uniform(
                -box_width * 0.3, box_width * 0.3, len(arm_df))
            ax.scatter(tick_idx + off + jitter, arm_df.values,
                       s=4, color=color, alpha=0.5, zorder=3)

    ax.set_xticks(range(len(ds_order)))
    ax.set_xticklabels(ds_order)

    ax.set_xticks(range(len(ds_order)))
    ax.set_xticklabels(ds_order)
    ax.set_xlabel("")
    ax.set_ylabel(r"$\log_{10}$(cells per participant + 1)")
    ax.set_title("Cells per Participant by Arm", fontweight="bold")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=arm_colors[a], label=a) for a in arms_present]
    ax.legend(handles=handles, fontsize=6, title="Arm", title_fontsize=7,
              loc="upper right", frameon=True, ncol=3)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)
    ax.set_ylim(ax.get_ylim()[0], 5.7)


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
            _vals = obs[split_col].dropna().unique()
            _ordered = _visit_sort_order(_vals) if split_col == vis else sorted(_vals)
            for val in _ordered:
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
                   palette="Dark2", density_norm="width", width=0.8, ax=ax)

    # Re-center violins: seaborn 0.13 offsets groups as if all hue levels
    # are present at every tick; shift each dataset's violin block to center.
    from matplotlib.collections import PolyCollection
    n_datasets = len(order)
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    # Compute offset per poly and store (original_center -> offset) mapping
    offsets_map = {}
    for tick_idx in range(n_datasets):
        tick_polys = [p for p in polys
                      if abs(p.get_paths()[0].vertices[:, 0].mean() - tick_idx) < 0.6]
        if not tick_polys:
            continue
        xs = [p.get_paths()[0].vertices[:, 0].mean() for p in tick_polys]
        group_center = (min(xs) + max(xs)) / 2
        offset = tick_idx - group_center
        for p in tick_polys:
            orig_center = p.get_paths()[0].vertices[:, 0].mean()
            offsets_map[round(orig_center, 6)] = offset
            verts = p.get_paths()[0].vertices
            verts[:, 0] += offset
    # Shift quartile lines stored in ax.lines
    for line in ax.lines:
        xdata = np.array(line.get_xdata(), dtype=float)
        if len(xdata) == 0:
            continue
        x_center = round(float(xdata.mean()), 6)
        # Find closest key in offsets_map
        best_key = min(offsets_map.keys(), key=lambda k: abs(k - x_center), default=None)
        if best_key is not None and abs(best_key - x_center) < 0.5:
            line.set_xdata(xdata + offsets_map[best_key])
    ax.set_xlabel("")
    ax.set_ylabel("Genes detected per cell")
    ax.set_title("Gene Detection by Dataset & Group", fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", title="Group", title_fontsize=7,
              frameon=True, ncol=2)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel E helper: total_counts distributions ──────────────────────


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
            _vals = obs[split_col].dropna().unique()
            _ordered = _visit_sort_order(_vals) if split_col == vis else sorted(_vals)
            for val in _ordered:
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
                   palette="Dark2", density_norm="width", width=0.8, ax=ax)

    # Re-center violins per dataset tick
    from matplotlib.collections import PolyCollection
    n_datasets = len(order)
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    offsets_map = {}
    for tick_idx in range(n_datasets):
        tick_polys = [p for p in polys
                      if abs(p.get_paths()[0].vertices[:, 0].mean() - tick_idx) < 0.6]
        if not tick_polys:
            continue
        xs = [p.get_paths()[0].vertices[:, 0].mean() for p in tick_polys]
        group_center = (min(xs) + max(xs)) / 2
        offset = tick_idx - group_center
        for p in tick_polys:
            orig_center = p.get_paths()[0].vertices[:, 0].mean()
            offsets_map[round(orig_center, 6)] = offset
            verts = p.get_paths()[0].vertices
            verts[:, 0] += offset
    for line in ax.lines:
        xdata = np.array(line.get_xdata(), dtype=float)
        if len(xdata) == 0:
            continue
        x_center = round(float(xdata.mean()), 6)
        best_key = min(offsets_map.keys(), key=lambda k: abs(k - x_center), default=None)
        if best_key is not None and abs(best_key - x_center) < 0.5:
            line.set_xdata(xdata + offsets_map[best_key])
    ax.set_xlabel("")
    ax.set_ylabel(r"$\log_{10}$(total counts + 1)")
    ax.set_title("Sequencing Depth by Dataset & Group", fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", frameon=True, ncol=2)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel E: % mito and % ribosomal ─────────────────────────────────


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
    ax.legend(fontsize=6, loc="upper right", frameon=True, ncol=2)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel E merged: Total counts + mito/ribo in 1×2 ──────────────────


def _panel_counts_mito_merged(fig_merged, loaded: dict):
    """E: Combined total counts + mito/ribo QC in 1×2 subplot."""
    ax1, ax2 = fig_merged.subplots(1, 2)
    _panel_counts_dist(ax1, loaded)
    _panel_mito_ribo(ax2, loaded)


# ── Panel G: QC attrition waterfall ──────────────────────────────────


def _panel_qc_waterfall(ax, loaded: dict):
    """Post-QC threshold check: cells meeting each criterion in processed data."""
    thresholds = [
        ("All cells", None),
        ("genes ≥ 200", lambda ng, tc, mt: ng >= 200),
        ("counts ≥ 500", lambda ng, tc, mt: tc >= 500),
        ("mito < 20%", lambda ng, tc, mt: mt < 20),
    ]

    ds_names = list(loaded.keys())
    x = np.arange(len(thresholds))
    width = 0.15
    offsets = np.linspace(-(len(ds_names) - 1) * width / 2,
                          (len(ds_names) - 1) * width / 2,
                          len(ds_names))

    for di, name in enumerate(ds_names):
        data = loaded[name]
        ngenes = data["ngenes"]
        total_counts = data["total_counts"]
        pct_mito = data["pct_mito"]

        # Start with all cells (already QC-filtered upstream)
        mask = np.ones(len(ngenes), dtype=bool)
        counts_at_stage = []

        for label, filt_fn in thresholds:
            if filt_fn is not None:
                stage_mask = filt_fn(ngenes, total_counts, pct_mito)
                # NaN values → fail the threshold (conservative)
                stage_mask = np.where(np.isnan(stage_mask), False, stage_mask)
                mask = mask & stage_mask
            counts_at_stage.append(int(mask.sum()))

        color = _DS_PALETTE.get(name, "grey")
        is_orange = (color == "#d95f02")
        bars = ax.bar(x + offsets[di], counts_at_stage, width,
                      color=color, edgecolor="white", label=name)
        for bar, cnt in zip(bars, counts_at_stage):
            ax.text(bar.get_x() + bar.get_width() / 2, cnt + cnt * 0.01,
                    f"{cnt:,}", ha="center", va="bottom", fontsize=5,
                    rotation=0 if is_orange else 90)

    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in thresholds], fontsize=8,
                       rotation=20, ha="right")
    ax.set_ylabel("Cells meeting threshold")
    ax.set_title("Post-QC Threshold Compliance", fontweight="bold")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else "0"))
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.30)
    ax.legend(fontsize=5, frameon=True, ncol=len(ds_names),
              loc="upper center", borderpad=0.2,
              handlelength=0.8, handletextpad=0.2, columnspacing=0.5)
    despine(ax)


# ── Panel F: Lorenz curve + Gini per dataset ─────────────────────────


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
        try:  # for numpy < 1.25
            gini = 1 - 2 * np.trapz(y_lorenz, x_lorenz)
        except Exception:
            gini = 1 - 2 * np.trapezoid(y_lorenz, x_lorenz)

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


# ── Panel H: Visit completeness ──────────────────────────────────────


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
            visits = _visit_sort_order(obs[vis].dropna().unique())
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
    ax.set_yticklabels(df["Label"], fontsize=4.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Fraction of participants with cells")
    ax.set_title("Visit Completeness", fontweight="bold")
    ax.invert_yaxis()
    despine(ax)


# ======================================================================
# Generate
# ======================================================================


def generate():
    """Create and save Supplementary Figure 1 panels.

    Layout (8 panels, A–H):
      A  Participant pairing structure
      B  Participant counts per arm × visit
      C  Cells per participant by arm
      D  Genes detected per cell distributions
      E  Total counts + mito/ribo QC merged (1×2)
      F  Lorenz curve + Gini inequality
      G  Post-QC threshold compliance
      H  Visit completeness
    """
    print("Supplementary Figure 1: Data Quality & Cohort Characterisation")
    loaded = _load_all()

    if not loaded:
        print("  No datasets loaded; skipping figure.")
        return

    # Panel A: Participant pairing structure
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_pairing(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: Participant counts per arm × visit (faceted)
    ncols_b = len(loaded)
    fig = plt.figure(figsize=(3.5 * ncols_b, 4))
    _panel_participant_counts(fig, loaded)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: Cells per participant by arm
    fig, ax = plt.subplots(figsize=(13, 5))
    _panel_cells_per_pid_arm(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: Genes detected per cell distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_ngenes_dist(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: Total counts + mito/ribo QC merged (1×2)
    fig = plt.figure(figsize=(18, 5.5))
    _panel_counts_mito_merged(fig, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: Lorenz curve + Gini inequality
    fig, ax = plt.subplots(figsize=(6, 6))
    _panel_lorenz_gini(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Panel G: QC threshold compliance
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _panel_qc_waterfall(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # Panel H: Visit completeness
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _panel_completeness_detailed(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    # ── Combined artboard (180 × ≤215 mm) ────────────────────────────────
    _SMALL_RC = {
        "font.size": 5,
        "axes.titlesize": 5.5,
        "axes.labelsize": 5,
        "xtick.labelsize": 4.5,
        "ytick.labelsize": 4.5,
        "legend.fontsize": 4,
        "legend.title_fontsize": 4,
    }
    _MAX_FONT_COMPOSITE = 6

    def _cap_fontsize(fig_obj, maximum):
        for ax_i in fig_obj.get_axes():
            for txt in ([ax_i.title, ax_i.xaxis.label, ax_i.yaxis.label]
                        + ax_i.get_xticklabels() + ax_i.get_yticklabels()
                        + ax_i.texts):
                if txt.get_fontsize() > maximum:
                    txt.set_fontsize(maximum)
            if ax_i.get_legend():
                for txt in ax_i.get_legend().get_texts():
                    if txt.get_fontsize() > maximum:
                        txt.set_fontsize(maximum)
        for txt in fig_obj.texts:
            if txt.get_fontsize() > maximum:
                txt.set_fontsize(maximum)

    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm = 1.0 / 25.4
    fig_c = plt.figure(figsize=(180 * _mm, 190 * _mm))

    #   Row 0: A (pairing)   | B (first half of datasets)
    #   Row 1: C (cells/pid) | B (remaining datasets, centred)
    #   Row 2: D (genes) | E_counts | E_mito
    #   Row 3: F (Lorenz) | G (QC waterfall) | H (completeness)
    n_ds = len(loaded)
    ds_names = list(loaded.keys())
    n_b_cols = math.ceil(n_ds / 2)  # datasets per row
    b_rows = [ds_names[i:i + n_b_cols] for i in range(0, n_ds, n_b_cols)]

    outer = fig_c.add_gridspec(
        4, 1,
        height_ratios=[0.75, 0.75, 1, 1],
        hspace=0.32,
        left=0.08, right=0.95, top=0.97, bottom=0.05,
    )

    # Rows 0–1: left panel (A/C) | right B subfigs
    # Every row uses width_ratios that sum identically so left panel
    # widths are equal and each B subplot has the same width.
    ax_b_list = []

    # Row 0: A | B (full row of datasets)
    _left_w = 2.2  # wider left panel → narrower B subplots
    b_row0 = b_rows[0]
    gs0 = outer[0].subgridspec(
        1, 1 + n_b_cols, wspace=0.40,
        width_ratios=[_left_w] + [1] * n_b_cols,
    )
    ax_a = fig_c.add_subplot(gs0[0])
    for j, nm in enumerate(b_row0):
        ax_b_list.append((fig_c.add_subplot(gs0[1 + j]), nm))

    # Spanning invisible axis used only to display a common B title
    ax_b_title = fig_c.add_subplot(gs0[0, 1:])
    ax_b_title.axis("off")
    ax_b_title.text(0.5, 1.10, "Participants per Arm × Visit",
                    transform=ax_b_title.transAxes,
                    ha="center", va="bottom",
                    fontsize=6, fontweight="bold")

    # Row 1: C | B (remaining datasets, centred if fewer than n_b_cols)
    b_row1 = b_rows[1] if len(b_rows) > 1 else []
    n_have = len(b_row1)
    n_gap = n_b_cols - n_have
    if n_gap == 0:
        gs1 = outer[1].subgridspec(
            1, 1 + n_b_cols, wspace=0.40,
            width_ratios=[_left_w] + [1] * n_b_cols,
        )
        ax_c = fig_c.add_subplot(gs1[0])
        for j, nm in enumerate(b_row1):
            ax_b_list.append((fig_c.add_subplot(gs1[1 + j]), nm))
    else:
        half_pad = n_gap / 2.0
        ratios = [_left_w, half_pad] + [1] * n_have + [half_pad]
        n_cols = len(ratios)
        gs1 = outer[1].subgridspec(
            1, n_cols, wspace=0.40,
            width_ratios=ratios,
        )
        ax_c = fig_c.add_subplot(gs1[0])
        ax_pad_l = fig_c.add_subplot(gs1[1])
        ax_pad_l.axis("off")
        for j, nm in enumerate(b_row1):
            ax_b_list.append((fig_c.add_subplot(gs1[2 + j]), nm))
        ax_pad_r = fig_c.add_subplot(gs1[n_cols - 1])
        ax_pad_r.axis("off")

    # Row 2: D | E_counts | E_mito
    gs2 = outer[2].subgridspec(1, 3, wspace=0.45)
    ax_d = fig_c.add_subplot(gs2[0])
    ax_e1 = fig_c.add_subplot(gs2[1])
    ax_e2 = fig_c.add_subplot(gs2[2])

    # Row 3: F | G | H
    gs3 = outer[3].subgridspec(1, 3, wspace=0.50)
    ax_f = fig_c.add_subplot(gs3[0])
    ax_g = fig_c.add_subplot(gs3[1])
    ax_h = fig_c.add_subplot(gs3[2])

    # Draw left panels
    _panel_pairing(ax_a, loaded)
    _panel_cells_per_pid_arm(ax_c, loaded)
    _panel_ngenes_dist(ax_d, loaded)

    # Draw Panel B subfigs across rows
    for ax_bi, name_bi in ax_b_list:
        data_bi = loaded[name_bi]
        obs_bi = data_bi["adata"].obs
        pid_bi = data_bi["pid_col"]
        arm_bi = data_bi["arm_col"]
        vis_bi = data_bi["visit_col"]
        if pid_bi is None:
            ax_bi.set_title(name_bi, fontweight="bold", fontsize=5, pad=-8)
            ax_bi.axis("off")
            continue
        if arm_bi and vis_bi and arm_bi in obs_bi.columns and vis_bi in obs_bi.columns:
            if name_bi == "TNBC" and "response" in obs_bi.columns:
                # Stacked bars: arm × visit × R/NR
                grp3 = (obs_bi
                        .assign(**{arm_bi: obs_bi[arm_bi].astype(str),
                                   vis_bi: obs_bi[vis_bi].astype(str),
                                   "response": obs_bi["response"].astype(object).fillna("unknown").astype(str)})
                        .groupby([arm_bi, vis_bi, "response"], observed=True)[pid_bi]
                        .nunique().reset_index(name="N"))
                grp3.rename(columns={arm_bi: "Arm", vis_bi: "Visit", "response": "Response"}, inplace=True)
                grp3 = grp3[grp3["N"] > 0]

                visits_t = _visit_sort_order(grp3["Visit"].unique())
                arms_t = sorted(grp3["Arm"].unique())
                _known_resp = [r for r in ("R", "NR") if r in grp3["Response"].unique()]
                resps_t = _known_resp + sorted(r for r in grp3["Response"].unique()
                                               if r not in _known_resp)
                n_arms_t = len(arms_t)
                bar_w_t = 0.30
                _arm_pal_t = sns.color_palette("Dark2", n_arms_t)
                _arm_colors_t = dict(zip(sorted(arms_t), _arm_pal_t))
                _hatch_t = {"R": "", "NR": "///"}

                from matplotlib.patches import Patch as _Patch
                for ai, arm_t in enumerate(arms_t):
                    _off_t = (ai - (n_arms_t - 1) / 2) * bar_w_t
                    for vi_t, visit_t in enumerate(visits_t):
                        bottom_t = 0
                        for resp_t in resps_t:
                            row_t = grp3[(grp3["Arm"] == arm_t) &
                                         (grp3["Visit"] == visit_t) &
                                         (grp3["Response"] == resp_t)]
                            n_t = int(row_t["N"].values[0]) if len(row_t) > 0 else 0
                            if n_t > 0:
                                ax_bi.bar(vi_t + _off_t, n_t, bar_w_t,
                                          bottom=bottom_t,
                                          color=_arm_colors_t[arm_t],
                                          hatch=_hatch_t.get(resp_t, "xx"),
                                          alpha=1.0 if resp_t == "R" else 0.50,
                                          edgecolor="white", linewidth=0.5)
                                bottom_t += n_t

                ax_bi.set_xticks(range(len(visits_t)))
                ax_bi.set_xticklabels(visits_t)

                _tnbc_handles = (
                    [_Patch(facecolor=_arm_colors_t[a], edgecolor="#555555", label=a)
                     for a in arms_t] +
                    # Use white fill + dark edge so the hatch pattern is unambiguous
                    [_Patch(facecolor="white",
                            hatch=_hatch_t.get(r, "xx"),
                            edgecolor="#333333",
                            linewidth=0.8,
                            label="Responder" if r == "R" else "Non-Responder")
                     for r in resps_t]
                )
                ax_bi.legend(handles=_tnbc_handles, fontsize=4.5,
                             title=None,
                             loc="upper right", frameon=True, framealpha=0.85,
                             edgecolor="#CCCCCC", borderpad=0.3,
                             handlelength=2.5, handletextpad=0.3, labelspacing=0.15,
                             ncol=1)
            else:
                grp = (obs_bi.assign(**{arm_bi: obs_bi[arm_bi].astype(str),
                                        vis_bi: obs_bi[vis_bi].astype(str)})
                       .groupby([arm_bi, vis_bi], observed=True)[pid_bi]
                       .nunique().reset_index(name="N"))
                grp.rename(columns={arm_bi: "Arm", vis_bi: "Visit"}, inplace=True)
                sns.barplot(data=grp, x="Visit", y="N", hue="Arm",
                            order=_visit_sort_order(grp["Visit"].unique()),
                            palette="Dark2", edgecolor="white", ax=ax_bi)
                leg_bi = ax_bi.get_legend()
                if leg_bi:
                    handles_bi = leg_bi.legend_handles
                    labels_bi = [t.get_text() for t in leg_bi.get_texts()]
                    leg_bi.remove()
                    ax_bi.legend(
                        handles=handles_bi, labels=labels_bi,
                        fontsize=5, title=None,
                        loc="upper right", frameon=True, framealpha=0.85,
                        edgecolor="#CCCCCC", borderpad=0.3,
                        handlelength=0.8, handletextpad=0.3,
                        labelspacing=0.2, ncol=1,
                    )
        elif vis_bi and vis_bi in obs_bi.columns:
            grp = (obs_bi.assign(**{vis_bi: obs_bi[vis_bi].astype(str)})
                   .groupby(vis_bi, observed=True)[pid_bi]
                   .nunique().reset_index(name="N"))
            grp.rename(columns={vis_bi: "Visit"}, inplace=True)
            sns.barplot(data=grp, x="Visit", y="N", width=0.4,
                        order=_visit_sort_order(grp["Visit"].unique()),
                        color="#1b9e77", edgecolor="white", ax=ax_bi)
        elif arm_bi and arm_bi in obs_bi.columns:
            grp = (obs_bi.assign(**{arm_bi: obs_bi[arm_bi].astype(str)})
                   .groupby(arm_bi, observed=True)[pid_bi]
                   .nunique().reset_index(name="N"))
            grp.rename(columns={arm_bi: "Arm"}, inplace=True)
            sns.barplot(data=grp, x="Arm", y="N", width=0.4,
                        color="#1b9e77", edgecolor="white", ax=ax_bi)
        else:
            ax_bi.axis("off")
            continue
        ax_bi.set_title(name_bi, fontweight="bold", fontsize=5, pad=-8)
        ax_bi.set_xlabel("")
        ax_bi.set_ylabel("Participants", fontsize=5)
        ax_bi.tick_params(axis="x", rotation=30, labelsize=4.5)
        ax_bi.tick_params(axis="y", labelsize=4.5)
        despine(ax_bi)

    # Fixed y-axis limits for TNBC and Melanoma to make space for the legend
    _b_ylim = {"TNBC": 8.8, "Melanoma": 17.5}
    for ax_bi, nm_bi in ax_b_list:
        if nm_bi in _b_ylim:
            ax_bi.set_ylim(ax_bi.get_ylim()[0], _b_ylim[nm_bi])

    # Draw remaining panels
    _panel_counts_dist(ax_e1, loaded)
    _panel_mito_ribo(ax_e2, loaded)
    _panel_lorenz_gini(ax_f, loaded)
    _panel_qc_waterfall(ax_g, loaded)
    _panel_completeness_detailed(ax_h, loaded)

    # Move legends inside plots for the composite
    _inside = {
        ax_a: dict(loc="upper left", ncol=1),
        ax_c: dict(loc="upper right", ncol=3),
        ax_d: dict(loc="upper right", ncol=3),
        ax_f: dict(loc="upper left", ncol=1),
    }
    for ax_target, kw in _inside.items():
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_target.legend(
                handles=handles, labels=labels,
                fontsize=5,
                frameon=True, framealpha=0.85,
                edgecolor="#CCCCCC", borderpad=0.3,
                handlelength=1, handletextpad=0.3,
                labelspacing=0.2,
                **kw,
            )

    # G legend: single horizontal row along top
    leg_g = ax_g.get_legend()
    if leg_g:
        handles_g = leg_g.legend_handles
        labels_g = [t.get_text() for t in leg_g.get_texts()]
        leg_g.remove()
        ax_g.legend(
            handles=handles_g, labels=labels_g,
            fontsize=5, loc="upper center",
            ncol=3,
            frameon=True, framealpha=0.85,
            edgecolor="#CCCCCC", borderpad=0.2,
            handlelength=0.8, handletextpad=0.2,
            columnspacing=0.5, labelspacing=0.2,
        )

    # Shrink legends in E subpanels
    for ax_ei in [ax_e1, ax_e2]:
        leg = ax_ei.get_legend()
        if leg:
            handles_e = leg.legend_handles
            labels_e = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_ei.legend(
                handles=handles_e, labels=labels_e,
                fontsize=5, loc="upper right",
                frameon=True, framealpha=0.85,
                edgecolor="#CCCCCC", borderpad=0.3,
                handlelength=0.6, handleheight=0.5,
                handletextpad=0.3, labelspacing=0.2,
                ncol=2,
            )

    _cap_fontsize(fig_c, _MAX_FONT_COMPOSITE)

    # Match A and G xtick labelsize to C (4.5) — these panels set explicit
    # fontsize in their draw functions which the cap doesn't reduce that far.
    for _ax_fix in (ax_a, ax_g):
        for _lbl in _ax_fix.get_xticklabels():
            _lbl.set_fontsize(4.5)

    # Bold panel labels (after cap so they stay prominent)
    _lbl_fs = 9
    for ax_lbl, lbl in [
        (ax_a, "A"), (ax_c, "C"), (ax_d, "D"),
        (ax_e1, "E"), (ax_f, "F"), (ax_g, "G"),
        (ax_h, "H"),
    ]:
        ax_lbl.text(-0.20, 1.12, lbl, transform=ax_lbl.transAxes,
                    fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")

    # Panel B label on the first B axes
    if ax_b_list:
        ax_b_list[0][0].text(-0.30, 1.12, "B", transform=ax_b_list[0][0].transAxes,
                             fontsize=_lbl_fs, fontweight="bold", va="top",
                             ha="left")

    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, SUPP_OUTPUT, close=False)
    pdf_path = SUPP_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")

    # Cleanup
    for data in loaded.values():
        del data["adata"]
    loaded.clear()
    clear_cache()
    gc.collect()
    print("  SuppFig1 complete: 8 individual panels + combined (A–H)\n")


if __name__ == "__main__":
    apply_style()
    generate()