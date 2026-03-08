"""
Supplementary Figure 3 — Study Cohort Overview.
================================================

Multi-dataset summary characterising the five clinical trial cohorts
used throughout the manuscript.

Panels:
  A  Dataset summary table (design type, cells, participants, timepoints).
  B  Participant pairing completeness (pre-only / post-only / paired).
  C  Treatment arm or severity distribution per dataset.
  D  Cells per participant stratified by timepoint (all datasets).
  E  Gene detection breadth across datasets (violin).
  F  Cell-type proportions (stacked horizontal bar, all datasets).
"""

from __future__ import annotations

import gc

import matplotlib.patheffects as pe
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
    harmonize_response,
    clear_cache,
)

FIGURE_NAME = "SuppFig3_cohort_overview"

# ── dataset registry ──────────────────────────────────────────────────
# Each entry: (loader, design_type, arm_col, arm_labels)

_REGISTRY: dict[str, dict] = {
    "Sade-Feldman": dict(
        loader=get_sade_feldman,
        design="Two-arm DiD",
        disease="Melanoma",
        arm_col="response",
        arm_labels=["Responder", "Non-responder"],
    ),
    "Stephenson": dict(
        loader=get_stephenson,
        design="Cross-sectional",
        disease="COVID-19",
        arm_col="severity",
        arm_labels=["Severe", "Mild"],
    ),
    "Vaccine": dict(
        loader=get_vaccine,
        design="Single-arm paired",
        disease="COVID-19 vaccine",
        arm_col=None,
        arm_labels=[],
    ),
    "AML": dict(
        loader=lambda: load_clinical_trial_dataset("aml"),
        design="Two-arm paired",
        disease="AML",
        arm_col="response",
        arm_labels=["Treatment", "Control"],
    ),
    "CAR-T": dict(
        loader=lambda: load_clinical_trial_dataset("cart"),
        design="Single-arm paired",
        disease="DLBCL",
        arm_col=None,
        arm_labels=[],
    ),
}


# ── helpers ───────────────────────────────────────────────────────────

def _pid_col(obs):
    for c in ("participant_id", "patient_id", "donor_id"):
        if c in obs.columns:
            return c
    return None


def _visit_col(obs):
    for c in ("visit",):
        if c in obs.columns:
            return c
    return None


def _timepoint_col(obs):
    """Most granular timepoint column."""
    for c in ("timepoint", "timepoint_category", "dfo_bin"):
        if c in obs.columns and obs[c].nunique() > 1:
            return c
    return _visit_col(obs)


def _ct_col(obs):
    for c in ("cell_type", "celltype", "CellType"):
        if c in obs.columns:
            return c
    return None


def _ngenes(obs, adata):
    for c in ("n_genes_by_counts", "n_genes", "n_genes_detected"):
        if c in obs.columns:
            return np.asarray(obs[c], dtype=float)
    import scipy.sparse as sp
    X = adata.layers.get("counts", adata.X)
    if sp.issparse(X):
        return np.asarray((X > 0).sum(axis=1), dtype=float).ravel()
    return np.asarray((X > 0).sum(axis=1), dtype=float).ravel()


# ── Panel A: summary table ───────────────────────────────────────────

def _panel_table(ax, loaded: dict, meta: dict):
    """Formatted table summarising all datasets."""
    ax.axis("off")
    rows = []
    for name in loaded:
        a = loaded[name]
        m = meta[name]
        pid = _pid_col(a.obs)
        n_pid = a.obs[pid].nunique() if pid else "?"
        tp = _timepoint_col(a.obs)
        n_tp = a.obs[tp].nunique() if tp else 1
        rows.append([
            name,
            m["disease"],
            m["design"],
            f"{a.n_obs:,}",
            str(n_pid),
            str(n_tp),
            f"{a.n_vars:,}",
        ])

    col_labels = ["Dataset", "Disease", "Study Design",
                   "Cells", "Participants", "Timepoints", "Genes"]
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", color="white")
        cell.set_facecolor("#4C72B0")
        cell.set_edgecolor("white")

    # Style body rows with alternating shading
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_edgecolor("white")
            if i % 2 == 0:
                cell.set_facecolor("#f0f0f0")
            else:
                cell.set_facecolor("white")

    ax.set_title("Dataset Summary", fontweight="bold", fontsize=12, pad=10)


# ── Panel B: pairing completeness ────────────────────────────────────

def _panel_pairing(ax, loaded: dict):
    """Stacked bar: pre-only / post-only / paired participants per dataset."""
    ds_names = list(loaded.keys())
    pre_only, post_only, paired, single_obs = [], [], [], []

    for name in ds_names:
        obs = loaded[name].obs
        pid = _pid_col(obs)
        vis = _visit_col(obs)
        if pid is None or vis is None or obs[vis].nunique() < 2:
            # Cross-sectional — single observation per participant
            n = obs[pid].nunique() if pid else 0
            pre_only.append(0)
            post_only.append(0)
            paired.append(0)
            single_obs.append(n)
            continue

        per_pid = obs.groupby(pid, observed=True)[vis].apply(
            lambda x: set(x.dropna().unique())
        )
        n_pre = sum(1 for s in per_pid if {"Pre"} == s)
        n_post = sum(1 for s in per_pid if {"Post"} == s)
        n_both = sum(1 for s in per_pid if "Pre" in s and "Post" in s)
        pre_only.append(n_pre)
        post_only.append(n_post)
        paired.append(n_both)
        single_obs.append(0)

    y = np.arange(len(ds_names))
    h = 0.55

    ax.barh(y, paired, height=h, label="Paired (Pre + Post)",
            color=COLORS["treated"], edgecolor="white")
    left1 = paired
    ax.barh(y, pre_only, height=h, left=left1, label="Pre only",
            color=COLORS["control"], edgecolor="white")
    left2 = [p + pr for p, pr in zip(paired, pre_only)]
    ax.barh(y, post_only, height=h, left=left2,
            label="Post only", color=COLORS["neutral"], edgecolor="white")
    left3 = [a + b for a, b in zip(left2, post_only)]
    ax.barh(y, single_obs, height=h, left=left3,
            label="Single observation", color="#999999", edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(ds_names)
    ax.set_xlabel("Number of participants")
    ax.set_title("Sample Pairing Completeness", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)

    # Annotate totals
    for i, name in enumerate(ds_names):
        total = paired[i] + pre_only[i] + post_only[i] + single_obs[i]
        ax.text(total + 0.3, i, f"n={total}", va="center", fontsize=8)

    despine(ax)


# ── Panel C: arm / response / severity distribution ──────────────────

def _panel_arm_distribution(ax, loaded: dict, meta: dict):
    """Grouped bar: participants per arm/condition per dataset."""
    rows = []
    for name in loaded:
        obs = loaded[name].obs
        m = meta[name]
        pid = _pid_col(obs)
        arm_col = m["arm_col"]

        if arm_col and arm_col in obs.columns and pid:
            per_pid = obs.drop_duplicates(subset=pid)
            counts = per_pid[arm_col].value_counts()
            for label, cnt in counts.items():
                rows.append({"Dataset": name, "Group": str(label), "Count": cnt})
        elif pid:
            n = obs[pid].nunique()
            rows.append({"Dataset": name, "Group": "All", "Count": n})

    df = pd.DataFrame(rows)
    if df.empty:
        ax.text(0.5, 0.5, "No arm data", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Fixed colour mapping for known groups
    group_colors = {
        "Responder": COLORS["treated"],
        "Non-responder": COLORS["control"],
        "Severe": "#d62728",
        "Mild": "#2ca02c",
        "Treatment": COLORS["treated"],
        "Control": COLORS["control"],
        "All": COLORS["neutral"],
    }
    fallback = sns.color_palette("Set2", 8)

    ds_names = list(loaded.keys())
    groups = df["Group"].unique()
    n_groups = len(groups)
    x = np.arange(len(ds_names))
    width = 0.7 / max(n_groups, 1)

    for gi, grp in enumerate(groups):
        vals = []
        for ds in ds_names:
            sub = df[(df["Dataset"] == ds) & (df["Group"] == grp)]
            vals.append(sub["Count"].sum() if len(sub) else 0)
        offset = (gi - (n_groups - 1) / 2) * width
        color = group_colors.get(grp, fallback[gi % len(fallback)])
        bars = ax.bar(x + offset, vals, width * 0.9, label=grp, color=color,
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                        str(v), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontsize=9)
    ax.set_ylabel("Number of participants")
    ax.set_title("Treatment Arm / Condition Distribution", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True, ncol=2)
    despine(ax)


# ── Panel D: cells per participant by timepoint ──────────────────────

def _panel_cells_per_pid(ax, loaded: dict):
    """Box + strip: cells per participant × timepoint, faceted by dataset."""
    rows = []
    for name, adata in loaded.items():
        obs = adata.obs
        pid = _pid_col(obs)
        tp = _timepoint_col(obs)
        if pid is None:
            continue
        if tp and obs[tp].nunique() > 1:
            grouped = obs.groupby([pid, tp], observed=True).size().reset_index(name="n")
            grouped["Dataset"] = name
            grouped.rename(columns={tp: "Timepoint"}, inplace=True)
            rows.append(grouped[["Dataset", "Timepoint", "n"]])
        else:
            grouped = obs.groupby(pid, observed=True).size().reset_index(name="n")
            grouped["Dataset"] = name
            grouped["Timepoint"] = "All"
            rows.append(grouped[["Dataset", "Timepoint", "n"]])

    df = pd.concat(rows, ignore_index=True)

    # Cells per (participant × timepoint) across datasets
    order = list(loaded.keys())
    palette = sns.color_palette("Set2", len(order))

    sns.boxplot(data=df, x="Dataset", y="n", order=order,
                palette=palette, width=0.5, showfliers=False,
                boxprops=dict(alpha=0.5), ax=ax)
    sns.stripplot(data=df, x="Dataset", y="n", order=order,
                  color="black", size=3, alpha=0.5, jitter=0.2, ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("Cells per participant × timepoint")
    ax.set_title("Cells per Sample", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel E: gene detection breadth ──────────────────────────────────

def _panel_gene_coverage(ax, loaded: dict):
    """Violin: genes detected per cell across datasets."""
    rows = []
    for name, adata in loaded.items():
        vals = _ngenes(adata.obs, adata)
        # Subsample for speed
        if len(vals) > 10_000:
            idx = np.random.default_rng(42).choice(len(vals), 10_000,
                                                    replace=False)
            vals = vals[idx]
        rows.append(pd.DataFrame({"Dataset": name, "Genes": vals}))

    df = pd.concat(rows, ignore_index=True)
    order = list(loaded.keys())
    palette = sns.color_palette("Set2", len(order))

    parts = ax.violinplot(
        [df.loc[df["Dataset"] == ds, "Genes"].values for ds in order],
        positions=range(len(order)), showmedians=True, showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    # Annotate medians
    for i, ds in enumerate(order):
        med = df.loc[df["Dataset"] == ds, "Genes"].median()
        ax.text(i, med, f"  {med:,.0f}", ha="left", va="bottom",
                fontsize=7, fontweight="bold")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("Genes detected per cell")
    ax.set_title("Gene Detection Breadth", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel F: cell-type proportions ───────────────────────────────────

def _panel_celltype_proportions(ax, loaded: dict):
    """Stacked horizontal bar: cell-type proportions per dataset."""
    ds_names = list(loaded.keys())

    # Collect proportions
    all_cts = set()
    props = {}
    for name in ds_names:
        col = _ct_col(loaded[name].obs)
        if col is None:
            continue
        cts = loaded[name].obs[col].astype(str)
        cts = cts[cts != "nan"]
        counts = cts.value_counts(normalize=True)
        # Group rare types
        if len(counts) > 12:
            keep = counts.head(11)
            other = counts.iloc[11:].sum()
            counts = pd.concat([keep, pd.Series({"Other": other})])
        props[name] = counts
        all_cts.update(counts.index)

    # Build consistent palette — grey for unknown/unassigned/other
    _GREY_TYPES = {"Unknown", "Unassigned", "Other", "nan"}
    real_cts = sorted(c for c in all_cts if c not in _GREY_TYPES)
    n = len(real_cts)
    if n <= 10:
        pal = sns.color_palette("tab10", n)
    elif n <= 20:
        pal = sns.color_palette("tab20", n)
    else:
        pal = sns.color_palette("husl", n)
    ct_palette = dict(zip(real_cts, pal))
    for gt in _GREY_TYPES:
        if gt in all_cts:
            ct_palette[gt] = "#bbbbbb"

    y = np.arange(len(ds_names))
    h = 0.6
    for name_idx, name in enumerate(ds_names):
        if name not in props:
            ax.text(0.5, name_idx, "No cell-type annotation",
                    ha="center", va="center", fontsize=7, fontstyle="italic",
                    color="#888888")
            continue
        left = 0.0
        for ct, frac in props[name].items():
            ax.barh(name_idx, frac, height=h, left=left,
                    color=ct_palette[ct], edgecolor="white", linewidth=0.3)
            # Label cell types with > 10% proportion
            if frac > 0.10:
                ax.text(left + frac / 2, name_idx, ct,
                        ha="center", va="center", fontsize=5.5,
                        color="white", fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            left += frac

    ax.set_yticks(y)
    ax.set_yticklabels(ds_names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion")
    ax.set_title("Cell-Type Composition", fontweight="bold")

    # Legend for cell types that appear in multiple datasets
    # Show top cell types only to avoid clutter
    all_cts_sorted = sorted(all_cts)
    top_cts = sorted(all_cts_sorted, key=lambda ct: sum(
        props.get(ds, pd.Series()).get(ct, 0) for ds in ds_names
    ), reverse=True)[:15]
    handles = [mpatches.Patch(facecolor=ct_palette[ct], label=ct, edgecolor="none")
               for ct in top_cts]
    ax.legend(handles=handles, fontsize=5, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1,
              handlelength=1.0, handleheight=0.8, labelspacing=0.2)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 3 panels."""
    print("Supplementary Figure 3: Study Cohort Overview")

    # Load all datasets
    loaded = {}
    for name, info in _REGISTRY.items():
        try:
            adata = info["loader"]()
            # Harmonize response for Sade-Feldman
            if name == "Sade-Feldman":
                adata = harmonize_response(adata)
            loaded[name] = adata
            print(f"  {name}: {adata.n_obs:,} cells, "
                  f"{adata.n_vars:,} genes")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")

    if not loaded:
        print("  No datasets; skipping.")
        return

    meta = {n: _REGISTRY[n] for n in loaded}

    # Panel A: Summary table
    fig, ax = plt.subplots(figsize=(10, 3.5))
    _panel_table(ax, loaded, meta)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: Pairing completeness
    fig, ax = plt.subplots(figsize=(7, 4))
    _panel_pairing(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: Arm / condition distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    _panel_arm_distribution(ax, loaded, meta)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: Cells per participant × timepoint
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _panel_cells_per_pid(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: Gene detection breadth
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _panel_gene_coverage(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: Cell-type proportions
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _panel_celltype_proportions(ax, loaded)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Cleanup
    del loaded
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
