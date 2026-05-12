"""
Supplementary Figure 5 - Cross-dataset biological consistency.

Panels:
  A  Gene-set score distributions (within-dataset z-score).
  B  All-pairs cross-dataset effect correlation (DiD/Δ labelled).
  C  Shared top genes with concordant direction (|β|>0.05 threshold).
  D  Gene-level effect distributions (DiD/Δ labelled).
  E  Exhaustion effects by cell type (proper two-sample SE for DiD).
  F  Effect heatmap across datasets (DiD/Δ labelled).
  G  Participant-level paired gene-set trajectories.
  H  Enrichment summary heatmap (within-dataset z-score).

Design-type handling:
  Two-arm datasets (Sade-Feldman, Stephenson) use DiD estimand
  (treated Δ − control Δ).  Single-arm datasets (AML, CAR-T, Vaccine)
  use within-arm pre→post change (Δ).  Panels mixing both estimands
  label each dataset with (DiD) or (Δ) to avoid silent conflation.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    add_log1p_cpm_layer,
    apply_style,
    clear_cache,
    despine,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    get_aml,
    get_cart,
    save_panel,
)

FIGURE_NAME = "SuppFig5_cross_dataset_biology"

_FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19",
    "CD14", "LYZ", "NKG7", "CD3D", "FOXP3", "IL7R", "TOX",
]

_GENE_SETS = {
    "T cell exhaustion\n(PD-1/TIM-3/LAG-3)": ["PDCD1", "HAVCR2", "LAG3", "CTLA4", "TIGIT", "TOX", "ENTPD1"],
    "CD8+ cytotoxicity\n(granzyme/perforin)": ["GZMB", "PRF1", "GZMA", "GZMK", "NKG7", "GNLY", "FASLG"],
    "Pro-inflammatory\nactivation (IFNγ/TNF)": ["IFNG", "TNF", "IL2", "CD69", "IL2RA", "HLA-DRA"],
    "T cell identity\n(CD3/CD4/CD8)": ["CD3D", "CD3E", "CD4", "CD8A", "TCF7", "IL7R"],
}

_DATASET_CFG = {
    "Melanoma": {
        "design": "two_arm",
        "loader": get_sade_feldman,
        "harmonize": True,
        "layer": "log1p_tpm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_treated": "Responder",
        "arm_control": "Non-responder",
        "visits": ("Pre", "Post"),
    },
    "AML": {
        "design": "single_arm",
        "loader": lambda: get_aml(),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_filter": "Treatment",
        "visits": ("Pre", "Post"),
    },
    "CAR-T": {
        "design": "single_arm",
        "loader": lambda: get_cart(),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_filter": "CAR-T",
        "visits": ("Pre", "Post"),
    },
    "COVID-19": {
        "design": "two_arm",
        "loader": get_stephenson,
        "harmonize": False,
        "layer": "log1p_cpm",
        "participant_col": "participant_id",
        "visit_col": "Collection_Day",
        "arm_col": "severity",
        "arm_treated": "Severe",
        "arm_control": "Mild",
        "visits": ("D0", "D28"),
    },
    "Vaccine": {
        "design": "single_arm",
        "loader": get_vaccine,
        "harmonize": False,
        "layer": "log1p_cpm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": None,
        "visits": ("Pre", "Post"),
    },
}

_DS_PALETTE = dict(zip(_DATASET_CFG.keys(),
    ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]))

# Design-type label for legend annotations: DiD = two-arm difference-in-differences,
# Δ = single-arm pre/post change.
_DESIGN_LABEL: dict[str, str] = {
    "Melanoma": "DiD",
    "COVID-19": "DiD",
    "AML": "Δ",
    "CAR-T": "Δ",
    "Vaccine": "Δ",
}


def _ds_label(name: str) -> str:
    """Return dataset name with design-type suffix for legends."""
    tag = _DESIGN_LABEL.get(name, "")
    return f"{name} ({tag})" if tag else name


def _to_array(mat) -> np.ndarray:
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _score_gene_sets(adata, layer: str) -> dict[str, np.ndarray]:
    X = adata.layers[layer] if layer in adata.layers else adata.X
    X = _to_array(X)
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    out = {}
    for gs_name, genes in _GENE_SETS.items():
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(idx) < 2:
            continue
        vals = X[:, idx]
        z = (vals - np.nanmean(vals, axis=0)) / (np.nanstd(vals, axis=0) + 1e-8)
        out[gs_name] = np.nanmean(z, axis=1)
    return out


def _participant_delta(adata, cfg: dict, features: list[str]) -> pd.DataFrame | None:
    """Compute per-participant pre→post deltas.

    For single-arm datasets with ``arm_filter``, subset to that arm first.
    Returns a DataFrame with columns = features + ["participant_id", "arm"].

    Deltas are indexed by (participant, arm) to prevent arm misassignment
    when participant IDs are not strictly unique per arm stratum.
    """
    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    arm_col = cfg.get("arm_col")
    arm_filter = cfg.get("arm_filter")
    pre_v, post_v = cfg["visits"]

    required = [pid_col, visit_col]
    if arm_col and arm_col in adata.obs.columns:
        required.append(arm_col)
    if not all(c in adata.obs.columns for c in required):
        return None

    if not features:
        return None

    # Single-arm: filter to treatment arm before computing deltas
    ad = adata
    if arm_filter and arm_col and arm_col in adata.obs.columns:
        ad = adata[adata.obs[arm_col] == arm_filter]

    X = _to_array(ad[:, features].layers[cfg["layer"]] if cfg["layer"] in ad.layers else ad[:, features].X)
    df = pd.DataFrame(X, columns=features, index=ad.obs_names)
    df[pid_col] = ad.obs[pid_col].values
    df[visit_col] = ad.obs[visit_col].values
    if arm_col and arm_col in ad.obs.columns:
        df["arm"] = ad.obs[arm_col].values
    else:
        df["arm"] = "All"

    # Group by (participant, visit, arm) to get unique pseudobulk per stratum
    pv = (
        df.groupby([pid_col, visit_col, "arm"], observed=True)[features]
        .mean()
        .reset_index()
    )
    pv = pv[pv[visit_col].isin([pre_v, post_v])].copy()

    # Index by (participant, arm) to ensure correct arm pairing
    pre = pv[pv[visit_col] == pre_v].set_index([pid_col, "arm"])
    post = pv[pv[visit_col] == post_v].set_index([pid_col, "arm"])
    common = pre.index.intersection(post.index)
    if len(common) < 3:
        return None

    delta = post.loc[common, features] - pre.loc[common, features]
    delta = delta.reset_index().rename(columns={pid_col: "participant_id"})
    return delta


def _participant_visit_gs(adata, cfg: dict, gs_scores: dict[str, np.ndarray]) -> pd.DataFrame | None:
    if not gs_scores:
        return None
    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    arm_col = cfg["arm_col"]

    required = [pid_col, visit_col]
    if arm_col:
        required.append(arm_col)
    if not all(c in adata.obs.columns for c in required):
        return None

    cols = [pid_col, visit_col]
    if arm_col and arm_col in adata.obs.columns:
        cols.append(arm_col)
    obs = adata.obs[cols].copy()
    col_map = {pid_col: "participant_id", visit_col: "visit"}
    if arm_col:
        col_map[arm_col] = "arm"
    obs = obs.rename(columns=col_map)
    if "arm" not in obs.columns:
        obs["arm"] = "All"
    for k, v in gs_scores.items():
        obs[k] = v

    pv = obs.groupby(["participant_id", "visit", "arm"], observed=True).mean().reset_index()
    return pv


def _effect_vector(delta: pd.DataFrame, cfg: dict, features: list[str]) -> pd.Series:
    """Mean effect per feature.

    Two-arm: treated mean Δ − control mean Δ  (difference-in-differences).
    Single-arm: mean Δ across all participants in the treated arm.
    """
    if delta is None or delta.empty:
        return pd.Series(dtype=float)

    if cfg.get("design") == "two_arm":
        treated = cfg["arm_treated"]
        control = cfg["arm_control"]
        t = delta[delta["arm"] == treated][features].mean(axis=0)
        c = delta[delta["arm"] == control][features].mean(axis=0)
        return t - c

    # Single-arm: mean pre→post delta
    return delta[features].mean(axis=0)


def _load_all() -> dict[str, dict]:
    out = {}
    for name, cfg in _DATASET_CFG.items():
        try:
            adata = cfg["loader"]()
            if cfg.get("harmonize", False):
                adata = harmonize_response(adata)

            layer = cfg["layer"]
            if layer == "log1p_cpm" and "log1p_cpm" not in adata.layers:
                if "counts" in adata.layers:
                    adata = add_log1p_cpm_layer(
                        adata, counts_layer="counts", out_layer="log1p_cpm",
                    )

            features = [f for f in _FEATURES if f in adata.var_names]
            gs_scores = _score_gene_sets(adata, cfg["layer"])
            delta = _participant_delta(adata, cfg, features)
            gs_pv = _participant_visit_gs(adata, cfg, gs_scores)
            effect = _effect_vector(delta, cfg, features) if delta is not None else pd.Series(dtype=float)

            out[name] = {
                "adata": adata,
                "cfg": cfg,
                "features": features,
                "gs_scores": gs_scores,
                "gs_pv": gs_pv,
                "delta": delta,
                "effect": effect,
            }
            n_part = 0 if delta is None else delta["participant_id"].nunique()
            print(f"  {name}: {adata.n_obs} cells, {n_part} paired participants")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
    return out


def _effect_matrix(data: dict[str, dict]) -> pd.DataFrame:
    cols = {}
    for name, ds in data.items():
        eff = ds.get("effect", pd.Series(dtype=float))
        if isinstance(eff, pd.Series) and len(eff) > 0:
            cols[name] = eff
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


def _panel_gs_distributions(ax, data: dict[str, dict]):
    rows = []
    for name, ds in data.items():
        pv = ds.get("gs_pv")
        if pv is None or pv.empty:
            continue
        gs_cols = [g for g in _GENE_SETS if g in pv.columns]
        if not gs_cols:
            continue
        # Participant-level means across visits to avoid cell count imbalance.
        per_pid = pv.groupby("participant_id", observed=True)[gs_cols].mean().reset_index()
        for gs in gs_cols:
            rows.extend({"Dataset": name, "Gene set": gs, "Score": float(v)} for v in per_pid[gs].values)

    if not rows:
        ax.text(0.5, 0.5, "No gene-set data", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    # Map dataset names to design-labelled versions for legend
    df["Dataset"] = df["Dataset"].map(_ds_label)
    palette = {_ds_label(k): v for k, v in _DS_PALETTE.items()}
    sns.violinplot(data=df, x="Gene set", y="Score", hue="Dataset", palette=palette,
                   cut=0, linewidth=0.6, ax=ax)
    ax.set_title("Gene-set score distributions (within-dataset z-score)", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.25)
    ax.legend(fontsize=4, frameon=True, ncol=2, loc="upper right")
    despine(ax)


def _panel_pairwise_corr(ax, data: dict[str, dict]):
    mat = _effect_matrix(data)
    if mat.empty or mat.shape[1] < 2:
        ax.text(0.5, 0.5, "Need >=2 datasets", ha="center", va="center", transform=ax.transAxes)
        return

    # Rename columns/index to include design-type label
    mat = mat.rename(columns=_ds_label)
    corr = pd.DataFrame(index=mat.columns, columns=mat.columns, dtype=float)
    for a in mat.columns:
        for b in mat.columns:
            if a == b:
                corr.loc[a, b] = 1.0
                continue
            common = mat[[a, b]].dropna()
            if len(common) < 3:
                corr.loc[a, b] = np.nan
            else:
                # Spearman rank correlation — more appropriate when comparing
                # effect estimates from different statistical frameworks
                # (DiD beta vs Hedges' g vs paired delta)
                rho, _ = sp_stats.spearmanr(common.iloc[:, 0], common.iloc[:, 1])
                corr.loc[a, b] = rho

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": r"Spearman $\rho$"}, ax=ax)
    ax.set_title("All-pairs cross-dataset effect correlation", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)


def _panel_concordant_top_genes(ax, data: dict[str, dict]):
    mat = _effect_matrix(data)
    if mat.empty:
        ax.text(0.5, 0.5, "No effects", ha="center", va="center", transform=ax.transAxes)
        return

    # Minimum |effect| threshold to count toward concordance — prevents
    # near-zero noisy estimates from inflating the concordance fraction.
    abs_thresh = 0.05

    stats_df = []
    for feat, row in mat.iterrows():
        vals = row.dropna().values
        if len(vals) < 2:
            continue
        # Only count effects above threshold for concordance
        sig = vals[np.abs(vals) > abs_thresh]
        if len(sig) < 2:
            continue
        pos = int(np.sum(sig > 0))
        neg = int(np.sum(sig < 0))
        concord = max(pos, neg) / len(sig)
        stats_df.append({
            "feature": feat,
            "mean_abs": float(np.mean(np.abs(vals))),
            "concordance": float(concord),
            "n_datasets": int(len(sig)),
            "direction": "up" if pos >= neg else "down",
        })

    if not stats_df:
        ax.text(0.5, 0.5, "No concordance data", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.DataFrame(stats_df).sort_values(["concordance", "mean_abs"], ascending=False).head(12)
    df = df.sort_values("mean_abs", ascending=True)
    colors = [COLORS["treated"] if d == "up" else COLORS["control"] for d in df["direction"]]

    ax.barh(df["feature"], df["mean_abs"], color=colors, alpha=0.85)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["mean_abs"] + 0.005, i, f"{r['concordance']:.0%}", va="center", fontsize=7)
    ax.set_xlabel("Mean |effect| across datasets")
    ax.set_title("Shared top genes with concordant direction (|β|>0.05)", fontweight="bold")
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=COLORS["treated"], label="Upregulated"),
        mpatches.Patch(facecolor=COLORS["control"], label="Downregulated"),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=True)
    despine(ax)


def _panel_gene_dist(ax, data: dict[str, dict]):
    rows = []
    for name, ds in data.items():
        eff = ds.get("effect")
        if eff is None or len(eff) == 0:
            continue
        for v in eff.dropna().values:
            rows.append({"Dataset": name, "Effect": float(v)})
    if not rows:
        ax.text(0.5, 0.5, "No effect distributions", ha="center", va="center", transform=ax.transAxes)
        return
    df = pd.DataFrame(rows)
    df["Dataset"] = df["Dataset"].map(_ds_label)
    palette = {_ds_label(k): v for k, v in _DS_PALETTE.items()}
    sns.violinplot(data=df, x="Dataset", y="Effect", palette=palette, cut=0, inner="quartile", ax=ax)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Gene-level effect distributions", fontweight="bold")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=_DS_PALETTE[n], label=_ds_label(n))
               for n in _DS_PALETTE if _ds_label(n) in df["Dataset"].values]
    ax.legend(handles=handles, fontsize=5, frameon=True, loc="lower right")
    despine(ax)


def _panel_exhaustion_by_celltype(ax, data: dict[str, dict]):
    """E: Exhaustion effects by cell type across datasets.

    Two-arm: effect = mean(treated Δ) − mean(control Δ),
             SE = sqrt(var_t/n_t + var_c/n_c)  (two-sample).
    Single-arm: effect = mean(Δ),
                SE = sd(Δ) / sqrt(n)  (one-sample).
    """
    ex_genes_full = ["PDCD1", "HAVCR2", "LAG3", "CTLA4", "TOX"]
    rows = []

    for ds_name, ds in data.items():
        adata = ds["adata"]
        cfg = ds["cfg"]
        ct_col = next((c for c in [
            "cell_type", "celltype", "cell_type_annot",
            "CellType", "cell_type_original", "cell_label",
            "clustnm", "leiden",
        ] if c in adata.obs.columns), None)
        ex_genes = [g for g in ex_genes_full if g in adata.var_names]
        if ct_col is None or len(ex_genes) < 2:
            continue

        # Use top 5 cell types with ≥50 cells to ensure reasonable estimates
        cts = adata.obs[ct_col].value_counts()
        top_ct = cts[cts >= 50].head(5).index.tolist()
        for ct in top_ct:
            sub = adata[adata.obs[ct_col] == ct]
            delta = _participant_delta(sub, cfg, ex_genes)
            if delta is None or len(delta) < 3:
                continue

            # Per-participant mean exhaustion score (average across genes)
            if cfg.get("design") == "two_arm":
                treated = cfg["arm_treated"]
                control = cfg["arm_control"]
                t_scores = delta[delta["arm"] == treated][ex_genes].mean(axis=1).values
                c_scores = delta[delta["arm"] == control][ex_genes].mean(axis=1).values
                if len(t_scores) < 2 or len(c_scores) < 2:
                    continue
                eff = float(np.mean(t_scores) - np.mean(c_scores))
                # Two-sample SE: sqrt(var_t/n_t + var_c/n_c)
                se = float(np.sqrt(
                    np.var(t_scores, ddof=1) / len(t_scores)
                    + np.var(c_scores, ddof=1) / len(c_scores)
                ))
            else:
                # Single-arm: one-sample mean and SE
                per_pid = delta[ex_genes].mean(axis=1).values
                eff = float(np.mean(per_pid))
                se = float(np.std(per_pid, ddof=1) / np.sqrt(len(per_pid))) if len(per_pid) > 1 else np.nan

            rows.append({"Dataset": ds_name, "Cell type": ct,
                        "Effect": float(eff),
                        "SE": float(se) if np.isfinite(se) else np.nan})

    if not rows:
        ax.text(0.5, 0.5, "No cell-type effects", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    # Color by dataset
    y = np.arange(len(df))
    colors = [_DS_PALETTE.get(d, "grey") for d in df["Dataset"]]
    ax.errorbar(df["Effect"], y, xerr=1.96 * df["SE"], fmt="none",
                ecolor="grey", capsize=2, lw=0.8, zorder=1)
    ax.scatter(df["Effect"], y, c=colors, s=12, edgecolors="white",
               linewidth=0.3, zorder=3)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['Cell type']} [{r['Dataset']}]" for _, r in df.iterrows()],
                       fontsize=4)
    ax.set_xlabel("Exhaustion effect (treatment)")
    ax.set_title("T cell exhaustion effects by cell type", fontweight="bold")

    # Legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=_DS_PALETTE[n], label=_ds_label(n))
               for n in _DS_PALETTE if n in df["Dataset"].values]
    ax.legend(handles=handles, fontsize=4, loc="best", frameon=True)
    despine(ax)


def _panel_effect_heatmap(ax, data: dict[str, dict]):
    mat = _effect_matrix(data)
    if mat.empty:
        ax.text(0.5, 0.5, "No effect matrix", ha="center", va="center", transform=ax.transAxes)
        return
    # Top 15 most variable features across datasets
    vv = mat.var(axis=1, skipna=True).sort_values(ascending=False)
    top = vv.head(15).index.tolist()
    plot_df = mat.loc[top].rename(columns=_ds_label)
    sns.heatmap(plot_df, cmap="RdBu_r", center=0, linewidths=0.4, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 7}, ax=ax,
                cbar_kws={"label": "Effect"})
    ax.set_title("Effect heatmap across datasets", fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=8)


def _panel_paired_trajectories(ax, data: dict[str, dict]):
    exhaustion_col = next((k for k in _GENE_SETS if "exhaustion" in k.lower()), None)
    if exhaustion_col is None:
        ax.text(0.5, 0.5, "No exhaustion gene set", ha="center", va="center", transform=ax.transAxes)
        return

    x_tick = []
    x_tick_lab = []
    xpos = 0
    for name, ds in data.items():
        pv = ds.get("gs_pv")
        cfg = ds.get("cfg", {})
        if pv is None or pv.empty or exhaustion_col not in pv.columns:
            continue
        pre_v, post_v = cfg.get("visits", ("Pre", "Post"))
        for arm in pv["arm"].dropna().unique():
            sub = pv[pv["arm"] == arm]
            pre = sub[sub["visit"] == pre_v].set_index("participant_id")
            post = sub[sub["visit"] == post_v].set_index("participant_id")
            common = pre.index.intersection(post.index)
            if len(common) < 2:
                continue
            x0, x1 = xpos, xpos + 1
            for pid in common:
                y0 = float(pre.loc[pid, exhaustion_col])
                y1 = float(post.loc[pid, exhaustion_col])
                ax.plot([x0, x1], [y0, y1], color=_DS_PALETTE.get(name, "grey"), alpha=0.35, lw=0.8)
            med0 = np.median(pre.loc[common, exhaustion_col].values)
            med1 = np.median(post.loc[common, exhaustion_col].values)
            ax.plot([x0, x1], [med0, med1], color=_DS_PALETTE.get(name, "black"), lw=2.8, zorder=5)
            x_tick.extend([x0, x1])
            lbl = _ds_label(name)
            x_tick_lab.extend([f"{lbl}\n{arm}\n{pre_v}", f"{lbl}\n{arm}\n{post_v}"])
            xpos += 2.5

    if not x_tick:
        ax.text(0.5, 0.5, "No paired trajectory data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.set_xticks(x_tick)
    ax.set_xticklabels(x_tick_lab, rotation=30, ha="right", fontsize=4.5)
    ax.set_ylabel("Exhaustion score")
    ax.set_title("Participant-level paired trajectories (Exhaustion)", fontweight="bold")
    despine(ax)


def _panel_enrichment_heatmap(ax, data: dict[str, dict]):
    rows = []
    for name, ds in data.items():
        pv = ds.get("gs_pv")
        if pv is None or pv.empty:
            continue
        gs_cols = [g for g in _GENE_SETS if g in pv.columns]
        if not gs_cols:
            continue
        for arm in sorted(pv["arm"].dropna().unique()):
            for visit in sorted(pv["visit"].dropna().unique()):
                sub = pv[(pv["arm"] == arm) & (pv["visit"] == visit)]
                if sub.empty:
                    continue
                label = f"{_ds_label(name)}\n{arm}\n{visit}"
                means = sub[gs_cols].mean(axis=0)
                for gs in gs_cols:
                    rows.append({"Gene set": gs, "Group": label, "Score": float(means[gs])})

    if not rows:
        ax.text(0.5, 0.5, "No enrichment summary", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    piv = df.pivot(index="Gene set", columns="Group", values="Score")
    sns.heatmap(piv, cmap="RdBu_r", center=0, linewidths=0.3, linecolor="white",
                ax=ax, cbar_kws={"label": "Mean z-score (within-dataset)"})
    ax.set_title("Enrichment summary (within-dataset z-score)", fontweight="bold")
    ax.tick_params(axis="x", labelsize=4.5, rotation=45)
    ax.tick_params(axis="y", labelsize=5)


def generate():
    """Create and save Supplementary Figure 5 panels (A–H) + composite."""
    print("Supplementary Figure 5: Cross-Dataset Biological Consistency")

    data = _load_all()
    if not data:
        print("  No datasets available; skipping.")
        return

    # ── Individual panels ─────────────────────────────────────────────
    panels = [
        ("panel_A", _panel_gs_distributions, (11.0, 5.8)),
        ("panel_B", _panel_pairwise_corr, (6.8, 6.0)),
        ("panel_C", _panel_concordant_top_genes, (8.8, 5.8)),
        ("panel_D", _panel_gene_dist, (7.2, 5.8)),
        ("panel_E", _panel_exhaustion_by_celltype, (7.8, 6.8)),
        ("panel_F", _panel_effect_heatmap, (8.8, 6.0)),
        ("panel_G", _panel_paired_trajectories, (12.0, 6.2)),
        ("panel_H", _panel_enrichment_heatmap, (12.0, 6.5)),
    ]
    for panel_name, fn, size in panels:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

    # ==================================================================
    # Composite artboard  (180 mm × ≤ 215 mm)
    # ==================================================================
    #   Row 0: A | B
    #   Row 1: C | D
    #   Row 2: E | F
    #   Row 3: G  (full width)
    #   Row 4: H  (full width)
    # ==================================================================
    print("  Building composite figure ...")

    _SMALL_RC = {
        "font.size": 5,
        "axes.titlesize": 5.5,
        "axes.labelsize": 5,
        "xtick.labelsize": 4.5,
        "ytick.labelsize": 4.5,
        "legend.fontsize": 4,
        "legend.title_fontsize": 4,
    }
    _MAX_FONT = 6

    def _cap_fontsize(fig_obj, maximum):
        for ax_i in fig_obj.get_axes():
            for txt in ([ax_i.title, ax_i.xaxis.label, ax_i.yaxis.label]
                        + ax_i.get_xticklabels() + ax_i.get_yticklabels()
                        + ax_i.texts):
                if txt.get_fontsize() > maximum:
                    txt.set_fontsize(maximum)
            leg = ax_i.get_legend()
            if leg:
                for txt in leg.get_texts():
                    if txt.get_fontsize() > maximum:
                        txt.set_fontsize(maximum)
                t = leg.get_title()
                if t and t.get_fontsize() > maximum:
                    t.set_fontsize(maximum)

    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm = 1.0 / 25.4
    fig_c = plt.figure(figsize=(180 * _mm, 215 * _mm))

    # 9 rows: 5 content rows interleaved with 4 spacer rows
    outer = fig_c.add_gridspec(
        9, 1,
        height_ratios=[
            0.60,   # row 0: A | B
            0.40,   # spacer
            0.60,   # row 2: C | D
            0.30,   # spacer
            0.65,   # row 4: E | F
            0.35,   # spacer
            0.55,   # row 6: G (full width)
            0.35,   # spacer
            0.55,   # row 8: H (full width)
        ],
        hspace=0.0,
        left=0.04, right=0.99, top=0.97, bottom=0.04,
    )

    # ── Row 0: A | B ─────────────────────────────────────────────────
    gs0 = outer[0].subgridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.50)
    ax_a = fig_c.add_subplot(gs0[0])
    ax_b = fig_c.add_subplot(gs0[1])

    _panel_gs_distributions(ax_a, data)
    _panel_pairwise_corr(ax_b, data)

    # ── Row 2: C | D ─────────────────────────────────────────────────
    gs1 = outer[2].subgridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.50)
    ax_cc = fig_c.add_subplot(gs1[0])
    ax_d = fig_c.add_subplot(gs1[1])

    _panel_concordant_top_genes(ax_cc, data)
    _panel_gene_dist(ax_d, data)

    # ── Row 4: E | F (shifted right with left padding) ────────────
    gs2 = outer[4].subgridspec(1, 3, width_ratios=[0.01, 1.0, 1.2],
                               wspace=0.45)
    ax_e = fig_c.add_subplot(gs2[1])
    ax_f = fig_c.add_subplot(gs2[2])

    _panel_exhaustion_by_celltype(ax_e, data)
    _panel_effect_heatmap(ax_f, data)

    # ── Row 6: G (full width) ────────────────────────────────────────
    ax_g = fig_c.add_subplot(outer[6])
    _panel_paired_trajectories(ax_g, data)

    # ── Row 8: H (right-aligned with left padding) ────────────────
    gs3 = outer[8].subgridspec(1, 2, width_ratios=[0.10, 1.0], wspace=0.0)
    ax_h = fig_c.add_subplot(gs3[1])
    _panel_enrichment_heatmap(ax_h, data)

    # ── Post-processing ───────────────────────────────────────────────
    for ax_pp in fig_c.get_axes():
        leg = ax_pp.get_legend()
        if leg:
            leg.get_frame().set_alpha(0.85)
            leg.get_frame().set_edgecolor("#CCCCCC")

    _cap_fontsize(fig_c, _MAX_FONT)

    # Bold panel labels — consistent offset for all panels
    _lbl_fs = 9
    _lbl_y = 1.12
    _lbl_x = -0.10

    for ax_lbl, lbl, lx in [
        (ax_a, "A", _lbl_x), (ax_b, "B", _lbl_x),
        (ax_cc, "C", _lbl_x), (ax_d, "D", _lbl_x),
        (ax_e, "E", -0.16), (ax_f, "F", _lbl_x),
        (ax_g, "G", -0.05), (ax_h, "H", _lbl_x),
    ]:
        ax_lbl.text(
            lx, _lbl_y, lbl,
            transform=ax_lbl.transAxes,
            fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
        )

    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, SUPP_OUTPUT, close=False)
    pdf_path = SUPP_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")

    # ── Cleanup ───────────────────────────────────────────────────────
    data.clear()
    clear_cache()
    gc.collect()
    print("  SuppFig5 complete: 8 individual panels + combined (A–H)\n")


if __name__ == "__main__":
    apply_style()
    generate()
