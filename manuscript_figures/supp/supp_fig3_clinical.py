"""
Supplementary Figure 3 — Trial Design and Baseline Comparability.
=================================================================

Demonstrate that pre-treatment groups are comparable and trial designs
are well-characterised before DiD analysis.

Panels:
  A  Study design summary table (design, pairing, arms, N).
  B  Participant pairing structure per dataset (paired / unpaired / single).
  C  Cells per participant per arm (box + strip, showing balance).
  D  Baseline gene-expression PCA overlap between arms per dataset.
  E  Genes detected at baseline: arm comparison (violins).
  F  Cell-type composition at baseline: arm comparison (stacked bars).
  G  Dropout / attrition rates per arm per dataset.
  H  Participant × visit completeness (bar chart).

Non-overlap guardrail: no treatment-effect claims, no DiD estimates.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
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
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    clear_cache,
)

FIGURE_NAME = "SuppFig3_cohort_overview"

_DS_PALETTE = dict(zip(
    ["Sade-Feldman", "Stephenson", "Vaccine", "AML", "CAR-T"],
    sns.color_palette("Set2", 5),
))

DATASETS = [
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
        if c in obs.columns and obs[c].nunique() > 1:
            return c
    return None


def _arm_col(obs):
    for c in ("response", "severity", "therapy"):
        if c in obs.columns and obs[c].nunique() > 1:
            return c
    return None


def _ct_col(obs):
    for c in ("cell_type", "celltype", "CellType", "cell_type_fine",
              "cell_type_coarse", "celltype_major", "clustnm"):
        if c in obs.columns and obs[c].nunique() > 1:
            return c
    return None


def _get_ngenes(adata) -> np.ndarray:
    obs = adata.obs
    for col in ("n_genes_by_counts", "n_genes", "n_genes_detected"):
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


def _load_all():
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
            ct = _ct_col(obs)
            loaded[name] = {
                "adata": adata,
                "pid_col": pid,
                "visit_col": vis,
                "arm_col": arm,
                "ct_col": ct,
            }
            print(f"  {name}: {adata.n_obs:,} cells, pid={pid}, vis={vis}, arm={arm}")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
    return loaded


# ── dataset metadata ──────────────────────────────────────────────

_DESIGN_META = {
    "Sade-Feldman": {
        "design": "Pre/post anti-PD-1",
        "pairing": "Paired",
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
        "arms": "Responder vs Non-responder",
        "indication": "B-ALL / DLBCL",
    },
}


# ── Panel A: Design summary table ─────────────────────────────────

def _panel_design_table(ax, loaded: dict):
    """Study design summary table."""
    rows = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        pid = data["pid_col"]
        vis = data["visit_col"]

        n_cells = data["adata"].n_obs
        n_participants = obs[pid].nunique() if pid else 0
        n_visits = obs[vis].nunique() if vis else 1

        meta = _DESIGN_META.get(name, {})
        rows.append([
            name,
            meta.get("indication", ""),
            meta.get("design", ""),
            meta.get("pairing", ""),
            f"{n_participants}",
            f"{n_visits}",
            f"{n_cells:,}",
        ])

    col_labels = ["Dataset", "Indication", "Design", "Pairing",
                   "Participants", "Visits", "Cells"]

    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")
            else:
                cell.set_facecolor("white")

    ax.set_title("Study Design Summary", fontweight="bold", pad=20)


# ── Panel B: Pairing structure ────────────────────────────────────

def _panel_pairing(ax, loaded: dict):
    """Participant pairing structure per dataset."""
    paired_ds = []
    single_ds = []
    partial_ds = []
    cross_sectional = []

    for name, data in loaded.items():
        obs = data["adata"].obs
        pid = data["pid_col"]
        vis = data["visit_col"]

        if vis is None or pid is None:
            cross_sectional.append(name)
            continue

        visits = sorted(obs[vis].dropna().unique())
        if len(visits) < 2:
            cross_sectional.append(name)
            continue

        participants = sorted(obs[pid].dropna().unique())
        paired_count = 0
        for p in participants:
            p_visits = set(obs.loc[obs[pid] == p, vis].dropna().unique())
            if len(p_visits) >= 2:
                paired_count += 1

        frac_paired = paired_count / len(participants) if participants else 0
        if frac_paired > 0.9:
            paired_ds.append(name)
        elif frac_paired > 0.3:
            partial_ds.append(name)
        else:
            single_ds.append(name)

    categories = {
        "Fully paired": paired_ds,
        "Partially paired": partial_ds,
        "Pre only": single_ds,
        "Single observation": cross_sectional,
    }
    cat_colors = {
        "Fully paired": COLORS.get("treated", "#2ecc71"),
        "Partially paired": "#f39c12",
        "Pre only": COLORS.get("control", "#e74c3c"),
        "Single observation": "#95a5a6",
    }

    ds_names = list(loaded.keys())
    y_pos = np.arange(len(ds_names))
    bar_colors = []
    bar_labels = []
    for name in ds_names:
        for cat, members in categories.items():
            if name in members:
                bar_colors.append(cat_colors[cat])
                bar_labels.append(cat)
                break
        else:
            bar_colors.append("#bdc3c7")
            bar_labels.append("Unknown")

    ax.barh(y_pos, [1] * len(ds_names), color=bar_colors, height=0.6,
            edgecolor="white")
    for i, (name, label) in enumerate(zip(ds_names, bar_labels)):
        ax.text(0.5, i, label, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ds_names, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Participant Pairing Structure", fontweight="bold")

    # Legend
    handles = [mpatches.Patch(facecolor=c, label=k)
               for k, c in cat_colors.items() if any(n in categories[k]
                                                       for n in ds_names)]
    ax.legend(handles=handles, fontsize=7, loc="lower right", frameon=True)
    despine(ax)


# ── Panel C: Cells per participant per arm ────────────────────────

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
                palette="Set2", ax=ax)
    sns.stripplot(data=df, x="Dataset", y="Cells_log", hue="Arm",
                  order=list(loaded.keys()), dodge=True, size=2, alpha=0.5,
                  palette="Set2", ax=ax, legend=False)

    ax.set_xlabel("")
    ax.set_ylabel(r"$\log_{10}$(cells per participant + 1)")
    ax.set_title("Cells per Participant by Arm", fontweight="bold")
    ax.legend(fontsize=6, title="Arm", title_fontsize=7, loc="upper right",
              frameon=True, ncol=2)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel D: Baseline PCA overlap between arms ───────────────────

def _panel_baseline_pca(fig_parent, axes, loaded: dict):
    """PCA of baseline (pre-treatment) cells coloured by arm."""
    import scipy.sparse as sp

    ds_with_baseline = []
    for name, data in loaded.items():
        vis = data["visit_col"]
        arm = data["arm_col"]
        if vis and arm:
            obs = data["adata"].obs
            pre_mask = obs[vis].astype(str).str.lower().isin(
                ["pre", "baseline", "d0", "day0", "0"])
            if pre_mask.sum() > 50:
                ds_with_baseline.append(name)

    for ax_i, ax in enumerate(axes):
        if ax_i >= len(ds_with_baseline):
            ax.axis("off")
            continue

        name = ds_with_baseline[ax_i]
        data = loaded[name]
        adata = data["adata"]
        obs = adata.obs
        vis = data["visit_col"]
        arm = data["arm_col"]

        pre_mask = obs[vis].astype(str).str.lower().isin(
            ["pre", "baseline", "d0", "day0", "0"])
        adata_pre = adata[pre_mask]

        if "X_pca" in adata_pre.obsm:
            pca = adata_pre.obsm["X_pca"][:, :2]
        else:
            # Quick PCA
            for layer in ("log1p_tpm", "log1p_cpm", "log1p_norm"):
                if layer in adata_pre.layers:
                    X = adata_pre.layers[layer]
                    break
            else:
                if "counts" in adata_pre.layers:
                    X = adata_pre.layers["counts"]
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes)
                    continue

            if sp.issparse(X):
                X = X.toarray()
            # Subsample genes
            var_genes = np.var(X, axis=0)
            top_genes = np.argsort(var_genes)[-500:]
            X_sub = X[:, top_genes]
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42).fit_transform(X_sub)

        arms = adata_pre.obs[arm].astype(str).values
        unique_arms = sorted(set(arms))
        arm_palette = dict(zip(unique_arms,
                                sns.color_palette("Set1", len(unique_arms))))

        rng = np.random.default_rng(42)
        order = rng.permutation(len(arms))

        for a in unique_arms:
            mask = arms[order] == a
            ax.scatter(pca[order[mask], 0], pca[order[mask], 1],
                       c=[arm_palette[a]], s=3, alpha=0.5, label=a,
                       edgecolors="none", rasterized=True)

        ax.set_title(f"{name} (baseline)", fontweight="bold", fontsize=9)
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.legend(fontsize=5, loc="best", frameon=True)
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)


# ── Panel E: Genes detected at baseline by arm ───────────────────

def _panel_baseline_ngenes(ax, loaded: dict):
    """Violins: genes detected per cell at baseline, split by arm."""
    rows = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        vis = data["visit_col"]
        arm = data["arm_col"]
        ngenes = _get_ngenes(data["adata"])

        if vis and arm:
            pre_mask = obs[vis].astype(str).str.lower().isin(
                ["pre", "baseline", "d0", "day0", "0"])
            if pre_mask.sum() < 50:
                continue
            arms = obs.loc[pre_mask, arm].astype(str).values
            ng = ngenes[pre_mask.values]
        elif arm:
            arms = obs[arm].astype(str).values
            ng = ngenes
        else:
            continue

        # Subsample
        n = len(ng)
        if n > 10000:
            idx = np.random.default_rng(42).choice(n, 10000, replace=False)
            arms = arms[idx]
            ng = ng[idx]

        for a in sorted(set(arms)):
            mask = arms == a
            rows.append(pd.DataFrame({
                "Dataset": name, "Arm": a, "Genes": ng[mask],
            }))

    if not rows:
        ax.text(0.5, 0.5, "No baseline data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontstyle="italic")
        ax.set_title("Baseline Gene Detection by Arm", fontweight="bold")
        return

    df = pd.concat(rows, ignore_index=True)
    ds_order = [n for n in loaded.keys()
                if n in df["Dataset"].unique()]

    sns.violinplot(data=df, x="Dataset", y="Genes", hue="Arm",
                   order=ds_order, cut=0, inner="quartile", linewidth=0.5,
                   palette="Set1", density_norm="width", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Genes detected per cell")
    ax.set_title("Baseline Gene Detection by Arm", fontweight="bold")
    ax.legend(fontsize=6, title="Arm", title_fontsize=7, loc="upper right",
              frameon=True, ncol=2)
    ax.tick_params(axis="x", rotation=15)
    despine(ax)


# ── Panel F: Cell-type composition at baseline by arm ─────────────

def _panel_baseline_ct_by_arm(ax, loaded: dict):
    """Stacked bars: baseline cell-type composition per arm per dataset."""
    rows = []
    for name, data in loaded.items():
        obs = data["adata"].obs
        vis = data["visit_col"]
        arm = data["arm_col"]
        ct = data["ct_col"]
        if arm is None or ct is None:
            continue

        if vis:
            pre_mask = obs[vis].astype(str).str.lower().isin(
                ["pre", "baseline", "d0", "day0", "0"])
            if pre_mask.sum() < 50:
                continue
            sub = obs.loc[pre_mask]
        else:
            sub = obs

        for a in sorted(sub[arm].dropna().unique()):
            a_sub = sub[sub[arm] == a]
            ct_frac = a_sub[ct].astype(str).value_counts(normalize=True)
            for c, f in ct_frac.items():
                rows.append({"Dataset": name, "Arm": str(a),
                             "Cell type": str(c), "Fraction": f})

    if not rows:
        ax.text(0.5, 0.5, "No baseline cell-type data", ha="center",
                va="center", transform=ax.transAxes, fontsize=10,
                fontstyle="italic")
        ax.set_title("Baseline Cell-Type Composition", fontweight="bold")
        return

    df = pd.DataFrame(rows)

    # Create combined label
    df["Group"] = df["Dataset"] + "\n" + df["Arm"]
    groups = sorted(df["Group"].unique())

    # Get all cell types and build palette
    all_cts = sorted(df["Cell type"].unique())
    n_ct = len(all_cts)
    if n_ct <= 10:
        pal = sns.color_palette("tab10", n_ct)
    elif n_ct <= 20:
        pal = sns.color_palette("tab20", n_ct)
    else:
        pal = sns.color_palette("husl", n_ct)
    ct_palette = dict(zip(all_cts, pal))

    y_pos = np.arange(len(groups))
    for gi, group in enumerate(groups):
        gsub = df[df["Group"] == group]
        left = 0.0
        for ct in all_cts:
            frac = gsub.loc[gsub["Cell type"] == ct, "Fraction"].sum()
            if frac > 0:
                ax.barh(gi, frac, left=left, height=0.6,
                        color=ct_palette[ct], edgecolor="white",
                        linewidth=0.2)
                if frac > 0.08:
                    ax.text(left + frac / 2, gi, ct, ha="center",
                            va="center", fontsize=4, color="white",
                            fontweight="bold",
                            path_effects=[pe.withStroke(linewidth=1,
                                                         foreground="black")])
                left += frac

    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups, fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of cells")
    ax.set_title("Baseline Cell-Type Composition by Arm", fontweight="bold")
    despine(ax)


# ── Panel G: Dropout / attrition ──────────────────────────────────

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


# ── Panel H: Detailed completeness ────────────────────────────────

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
    """Create and save Supplementary Figure 3 panels."""
    print("Supplementary Figure 3: Trial Design and Baseline Comparability")
    loaded = _load_all()

    if not loaded:
        print("  No datasets loaded; skipping.")
        return

    # Panel A: Design table
    fig, ax = plt.subplots(figsize=(10, 4))
    _panel_design_table(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: Pairing structure
    fig, ax = plt.subplots(figsize=(8, 4))
    _panel_pairing(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: Cells per participant per arm
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_cells_per_pid_arm(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: Baseline PCA overlap
    n_baseline = sum(1 for data in loaded.values()
                     if data["visit_col"] and data["arm_col"])
    ncols = min(n_baseline, 3) if n_baseline > 0 else 1
    nrows = max(1, (n_baseline + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    else:
        axes = axes.ravel()
    _panel_baseline_pca(fig, axes, loaded)
    fig.suptitle("Baseline PCA by Arm", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: Baseline gene detection by arm
    fig, ax = plt.subplots(figsize=(9, 5))
    _panel_baseline_ngenes(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: Baseline cell-type composition by arm
    fig, ax = plt.subplots(figsize=(10, 6))
    _panel_baseline_ct_by_arm(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Panel G: Dropout rates
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_dropout(ax, loaded)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # Panel H: Completeness
    fig, ax = plt.subplots(figsize=(8, 6))
    _panel_completeness_detailed(ax, loaded)
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
