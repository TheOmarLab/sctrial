"""
Supplementary Figure 4 — Sensitivity and Robustness.
=====================================================

Show how DiD results change under different analytical decisions.

Panels:
  A  Cell-level vs participant-level aggregation (beta comparison).
  B  Analytical vs bootstrap SE (forest-style CI comparison).
  C  Standardised vs unstandardised effect sizes.
  D  Mean vs median aggregation comparison.
  E  Log-transform sensitivity (raw vs log1p betas).
  F  Cell-type-stratified DiD heatmap.
  G  Rank-order concordance across preprocessing choices.
  H  Power curves: minimum detectable effect vs sample size.

Non-overlap guardrail: methodological sensitivity only, not biological claims.
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
    load_clinical_trial_dataset,
    save_panel,
)

FIGURE_NAME = "SuppFig4_sensitivity_robustness"

# Features for sensitivity tests
_FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19",
    "CD14", "LYZ", "NKG7",
]

_PAL = {"cell": COLORS.get("highlight", "#5B9BD5"),
        "participant": COLORS.get("treated", "#E07B54")}

# Try to import adjustText; fall back to manual offsets if unavailable
try:
    from adjustText import adjust_text as _adjust_text
    _HAS_ADJUSTTEXT = True
except ImportError:
    _HAS_ADJUSTTEXT = False


def _add_gene_labels(ax, features, x_series, y_series):
    """Add non-overlapping gene labels to a scatter plot.

    Uses adjustText when available; otherwise falls back to manual offsets.
    """
    if _HAS_ADJUSTTEXT:
        texts = []
        for feat in features:
            texts.append(
                ax.text(x_series[feat], y_series[feat], feat,
                        fontsize=7, fontweight="bold", ha="left")
            )
        _adjust_text(
            texts, ax=ax,
            force_text=(2.5, 2.5), force_points=(2.5, 2.5),
            expand=(1.8, 1.8),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
        )
    else:
        for feat in features:
            ax.annotate(
                feat, (x_series[feat], y_series[feat]),
                fontsize=7, fontweight="bold", alpha=0.85, ha="left",
                xytext=(5, 3), textcoords="offset points",
            )


# ======================================================================
# Data loading
# ======================================================================

def _run_sensitivity():
    """Run DiD under several preprocessing choices."""
    import sctrial

    adata = get_sade_feldman()
    adata = harmonize_response(adata)

    if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])

    design = sctrial.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="response",
        arm_treated="Responder",
        arm_control="Non-responder",
    )
    visits = ("Pre", "Post")
    feats = [f for f in _FEATURES if f in adata.var_names]

    out = {"adata": adata, "design": design, "visits": visits, "features": feats}

    # 1. Cell-level
    print("  cell-level DiD ...")
    out["cell"] = sctrial.did_table(
        adata, feats, design, visits,
        layer="log1p_tpm", aggregate="cell", standardize=True,
    )

    # 2. Participant-level (analytical SE)
    print("  participant-level DiD ...")
    out["part"] = sctrial.did_table(
        adata, feats, design, visits,
        layer="log1p_tpm", aggregate="participant_visit", standardize=True,
    )

    # 3. Participant-level bootstrap
    print("  participant bootstrap DiD ...")
    out["boot"] = sctrial.did_table(
        adata, feats, design, visits,
        layer="log1p_tpm", aggregate="participant_visit", standardize=True,
        use_bootstrap=True, n_boot=200, seed=42,
    )

    # 4. Unstandardised
    print("  unstandardised DiD ...")
    out["unstd"] = sctrial.did_table(
        adata, feats, design, visits,
        layer="log1p_tpm", aggregate="participant_visit", standardize=False,
    )

    # 5. Median aggregation
    print("  median aggregation DiD ...")
    out["median"] = sctrial.did_table(
        adata, feats, design, visits,
        layer="log1p_tpm", aggregate="participant_visit", standardize=True,
        agg="median",
    )

    # 6. Raw TPM (no log) — only if tpm layer exists
    if "tpm" in adata.layers:
        print("  raw TPM DiD ...")
        out["raw"] = sctrial.did_table(
            adata, feats, design, visits,
            layer="tpm", aggregate="participant_visit", standardize=True,
        )
    else:
        out["raw"] = None

    # 7. Cell-type stratified
    ct_col = next((c for c in ["cell_type", "celltype"]
                   if c in adata.obs.columns), None)
    ct_results = {}
    if ct_col:
        top_cts = adata.obs[ct_col].value_counts().head(5).index.tolist()
        short_feats = feats[:8]
        for ct in top_cts:
            try:
                sub = adata[adata.obs[ct_col] == ct].copy()
                # Need enough cells in both arms and visits
                if sub.n_obs < 50:
                    continue
                ct_df = sctrial.did_table(
                    sub, short_feats, design, visits,
                    layer="log1p_tpm", aggregate="participant_visit",
                    standardize=True,
                )
                ct_results[ct] = ct_df
            except Exception:
                pass
    out["ct_results"] = ct_results
    out["ct_col"] = ct_col

    return out


# ── Panel A: Cell vs Participant betas ────────────────────────────

def _panel_cell_vs_part(ax, data: dict):
    """Scatter comparing cell-level vs participant-level beta_DiD."""
    cell = data["cell"].set_index("feature")["beta_DiD"]
    part = data["part"].set_index("feature")["beta_DiD"]
    common = cell.index.intersection(part.index)
    if len(common) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        return

    x, y = cell[common].values, part[common].values
    ax.scatter(x, y, s=50, alpha=0.85, color=COLORS.get("highlight", "#5B9BD5"),
               edgecolors="grey", linewidth=0.3)

    # Labels — non-overlapping
    _add_gene_labels(ax, common, cell, part)

    # Identity + correlation
    lims = [min(min(x), min(y)) - 0.1, max(max(x), max(y)) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.3)
    r, p = sp_stats.pearsonr(x, y)
    p_str = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p_str}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.8))
    ax.set_xlabel("β (cell-level)")
    ax.set_ylabel("β (participant-level)")
    ax.set_title("Cell vs Participant Aggregation", fontweight="bold")
    despine(ax)


# ── Panel B: Analytical vs Bootstrap SE ───────────────────────────

def _panel_bootstrap_ci(ax, data: dict):
    """Forest plot: analytical SE vs bootstrap SE."""
    part = data["part"]
    boot = data["boot"]
    if part is None or boot is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df_a = part.set_index("feature")[["beta_DiD", "se_DiD"]].rename(
        columns={"beta_DiD": "beta", "se_DiD": "se_analytical"})
    df_b = boot.set_index("feature")[["se_DiD"]].rename(
        columns={"se_DiD": "se_boot"})
    df = df_a.join(df_b, how="inner").reset_index()
    df = df.sort_values("beta", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    off = 0.15

    ax.errorbar(df["beta"], y - off, xerr=1.96 * df["se_analytical"],
                fmt="s", markersize=4, color=_PAL["cell"], elinewidth=1,
                capsize=2, label="Analytical SE")
    ax.errorbar(df["beta"], y + off, xerr=1.96 * df["se_boot"],
                fmt="o", markersize=4, color=_PAL["participant"], elinewidth=1,
                capsize=2, label="Bootstrap SE")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=7)
    ax.set_xlabel("β with 95% CI")
    ax.set_title("Analytical vs Bootstrap SE", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", frameon=True)
    despine(ax)


# ── Panel C: Standardised vs Unstandardised ───────────────────────

def _panel_std_vs_unstd(ax, data: dict):
    """Scatter: standardised vs unstandardised effect sizes."""
    std = data["part"].set_index("feature")["beta_DiD"]
    unstd = data["unstd"].set_index("feature")["beta_DiD"]
    common = std.index.intersection(unstd.index)
    if len(common) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    x, y = std[common].values, unstd[common].values
    ax.scatter(x, y, s=50, alpha=0.85, color=COLORS.get("treated", "#E07B54"),
               edgecolors="grey", linewidth=0.3)

    _add_gene_labels(ax, common, std, unstd)

    r, _ = sp_stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=7,
            va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor="#ccc", alpha=0.8))
    ax.set_xlabel("β (standardised)")
    ax.set_ylabel("β (unstandardised)")
    ax.set_title("Standardised vs Unstandardised", fontweight="bold")
    despine(ax)


# ── Panel D: Mean vs Median aggregation ───────────────────────────

def _panel_mean_vs_median(ax, data: dict):
    """Scatter: mean-aggregation vs median-aggregation betas."""
    mean_df = data["part"].set_index("feature")["beta_DiD"]
    med_res = data.get("median")

    # Guard: median agg may produce NaN betas
    if med_res is None or med_res.empty:
        ax.text(0.5, 0.5, "No median-aggregation results", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Mean vs Median Aggregation", fontweight="bold")
        despine(ax)
        return

    med_df = med_res.set_index("feature")["beta_DiD"]
    common = mean_df.index.intersection(med_df.index)
    # Drop NaN/Inf
    mask = np.isfinite(mean_df[common]) & np.isfinite(med_df[common])
    common = common[mask]
    if len(common) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Mean vs Median Aggregation", fontweight="bold")
        despine(ax)
        return

    x, y = mean_df[common].values, med_df[common].values
    ax.scatter(x, y, s=50, alpha=0.85, color="#7B68EE",
               edgecolors="grey", linewidth=0.3)

    _add_gene_labels(ax, common, mean_df, med_df)

    lims = [min(min(x), min(y)) - 0.1, max(max(x), max(y)) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.3)
    r, _ = sp_stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=7,
            va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor="#ccc", alpha=0.8))
    ax.set_xlabel("β (mean aggregation)")
    ax.set_ylabel("β (median aggregation)")
    ax.set_title("Mean vs Median Aggregation", fontweight="bold")
    despine(ax)


# ── Panel E: Log-transform sensitivity ────────────────────────────

def _panel_log_sensitivity(ax, data: dict):
    """Scatter: log1p_tpm betas vs raw TPM betas."""
    log_df = data["part"].set_index("feature")["beta_DiD"]
    raw_res = data.get("raw")
    if raw_res is None or raw_res.empty:
        ax.text(0.5, 0.5, "No raw-TPM results", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Log-Transform Sensitivity", fontweight="bold")
        despine(ax)
        return

    raw_df = raw_res.set_index("feature")["beta_DiD"]
    common = log_df.index.intersection(raw_df.index)
    if len(common) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    x, y = log_df[common].values, raw_df[common].values
    ax.scatter(x, y, s=50, alpha=0.85, color="#2ECC71",
               edgecolors="grey", linewidth=0.3)

    _add_gene_labels(ax, common, log_df, raw_df)

    r, _ = sp_stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=7,
            va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor="#ccc", alpha=0.8))
    ax.set_xlabel("β (log1p TPM)")
    ax.set_ylabel("β (raw TPM)")
    ax.set_title("Log-Transform Sensitivity", fontweight="bold")
    despine(ax)


# ── Panel F: Cell-type stratified heatmap ─────────────────────────

def _panel_ct_heatmap(ax, data: dict):
    """Heatmap: DiD effect sizes stratified by top cell types."""
    ct_results = data.get("ct_results", {})
    if not ct_results:
        ax.text(0.5, 0.5, "No cell-type-stratified results", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Cell-Type Stratified DiD", fontweight="bold")
        despine(ax)
        return

    # Build matrix
    rows = {}
    for ct, df in ct_results.items():
        if "beta_DiD" in df.columns and "feature" in df.columns:
            rows[ct] = df.set_index("feature")["beta_DiD"]
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    mat = pd.DataFrame(rows)
    # Limit display
    mat = mat.iloc[:8]

    sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, linewidths=0.5,
                linecolor="white", cbar_kws={"shrink": 0.6, "label": "β"},
                annot=True, fmt=".2f", annot_kws={"fontsize": 6})
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Feature")
    ax.set_title("Cell-Type Stratified DiD", fontweight="bold")
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7)


# ── Panel G: Rank concordance ────────────────────────────────────

def _panel_rank_concordance(ax, data: dict):
    """Bar chart: Spearman rank correlation of feature rankings
    across preprocessing choices."""
    # Get rankings from different configs
    configs = {}
    for key, label in [("cell", "Cell-level"),
                       ("part", "Participant"),
                       ("boot", "Bootstrap"),
                       ("unstd", "Unstandardised"),
                       ("median", "Median agg")]:
        df = data.get(key)
        # Guard: skip if df is None, empty, or has too few valid betas
        if (df is not None
                and not df.empty
                and "beta_DiD" in df.columns
                and df["beta_DiD"].notna().sum() >= 2):
            configs[label] = df.set_index("feature")["beta_DiD"].rank()

    if ("raw" in data
            and data["raw"] is not None
            and not data["raw"].empty
            and "beta_DiD" in data["raw"].columns
            and data["raw"]["beta_DiD"].notna().sum() >= 2):
        configs["Raw TPM"] = data["raw"].set_index("feature")["beta_DiD"].rank()

    if len(configs) < 2:
        ax.text(0.5, 0.5, "Insufficient configs", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Use "Participant" as reference
    ref_key = "Participant"
    if ref_key not in configs:
        ref_key = list(configs.keys())[0]

    ref = configs[ref_key]
    labels, rhos = [], []
    for label, ranks in configs.items():
        if label == ref_key:
            continue
        common = ref.index.intersection(ranks.index)
        # Drop NaN ranks before correlation
        valid = ref[common].notna() & ranks[common].notna()
        common = common[valid]
        if len(common) < 3:
            continue
        rho, _ = sp_stats.spearmanr(ref[common], ranks[common])
        if np.isnan(rho):
            continue
        labels.append(label)
        rhos.append(rho)

    if not labels:
        ax.text(0.5, 0.5, "No comparisons", ha="center", va="center",
                transform=ax.transAxes)
        return

    colors = [COLORS.get("highlight", "#5B9BD5") if r > 0.8
              else COLORS.get("treated", "#E07B54") if r > 0.5
              else "#E74C3C" for r in rhos]

    y = np.arange(len(labels))
    ax.barh(y, rhos, color=colors, edgecolor="white", alpha=0.85, height=0.6)
    ax.axvline(1.0, color="grey", linewidth=0.5, alpha=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Spearman ρ (vs {ref_key})")
    ax.set_xlim(0, 1.05)
    ax.set_title("Rank Concordance Across Choices", fontweight="bold")

    for i, rho in enumerate(rhos):
        ax.text(rho + 0.02, i, f"{rho:.2f}", va="center", fontsize=7)

    despine(ax)


# ======================================================================
# Panel H: Power curves (MDE) — data from OLS DiD across datasets
# ======================================================================

_MDE_FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19",
    "CD14", "LYZ", "NKG7", "CD3D", "FOXP3", "IL7R",
]

_MDE_DATASET_CFG = {
    "Sade-Feldman": {
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
        "design": "single_arm_paired",
        "loader": lambda: load_clinical_trial_dataset("aml"),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_filter": "Treatment",
        "visits": ("Pre", "Post"),
    },
    "CAR-T": {
        "design": "single_arm_paired",
        "loader": lambda: load_clinical_trial_dataset("cart"),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_filter": "CAR-T",
        "visits": ("Pre", "Post"),
    },
    "Stephenson": {
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
        "design": "single_arm_paired",
        "loader": get_vaccine,
        "harmonize": False,
        "layer": "log1p_cpm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": None,
        "visits": ("Pre", "Post"),
    },
}

_MDE_PALETTE = dict(
    zip(_MDE_DATASET_CFG.keys(), ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"])
)


def _mde_curve(
    n: np.ndarray,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.8,
    paired: bool = False,
) -> np.ndarray:
    """Minimum detectable effect for a comparison.

    Two-arm unpaired: MDE = (z_a + z_b) * sigma * sqrt(2/n)
    Single-arm paired: MDE = (z_a + z_b) * sigma * sqrt(1/n)
    """
    z_a = sp_stats.norm.ppf(1 - alpha / 2)
    z_b = sp_stats.norm.ppf(power)
    scale = np.sqrt(1.0 / n) if paired else np.sqrt(2.0 / n)
    return (z_a + z_b) * sigma * scale


def _load_mde_data() -> dict[str, dict]:
    """Load datasets and compute sigma + n_per_group for MDE curves."""
    out: dict[str, dict] = {}
    for name, cfg in _MDE_DATASET_CFG.items():
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
                else:
                    continue

            features = [f for f in _MDE_FEATURES if f in adata.var_names]
            if len(features) < 4:
                continue

            pid_col = cfg["participant_col"]
            visit_col = cfg["visit_col"]
            arm_col = cfg.get("arm_col")
            design = cfg.get("design", "two_arm")

            # Get participant-visit means
            mat = (
                adata[:, features].layers[layer]
                if layer in adata.layers
                else adata[:, features].X
            )
            mat = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
            expr = pd.DataFrame(mat, columns=features, index=adata.obs_names)
            expr[pid_col] = adata.obs[pid_col].values
            expr[visit_col] = adata.obs[visit_col].values
            if arm_col and arm_col in adata.obs.columns:
                expr[arm_col] = adata.obs[arm_col].values

            arm_filter = cfg.get("arm_filter")
            if arm_filter and arm_col and arm_col in expr.columns:
                expr = expr[expr[arm_col] == arm_filter].copy()

            group_cols = [pid_col, visit_col]
            if arm_col and arm_col in expr.columns:
                group_cols.append(arm_col)
            pv = expr.groupby(group_cols, observed=True)[features].mean().reset_index()

            # Paired filter
            counts = pv.groupby(pid_col)[visit_col].nunique()
            paired = counts[counts >= 2].index
            pv = pv[pv[pid_col].isin(paired)].copy()

            if design == "two_arm":
                pv = pv[
                    pv[arm_col].isin([cfg["arm_treated"], cfg["arm_control"]])
                ].copy()
                arm_counts = pv.groupby(arm_col)[pid_col].nunique()
                n_per_group = int(min(
                    arm_counts.get(cfg["arm_treated"], 0),
                    arm_counts.get(cfg["arm_control"], 0),
                ))
            else:
                n_per_group = int(pv[pid_col].nunique())

            if n_per_group < 2:
                continue

            # Compute median residual sigma from simple OLS per feature
            sigmas = []
            pre_v, post_v = cfg["visits"]
            pv_sub = pv[pv[visit_col].isin([pre_v, post_v])].copy()
            post = (pv_sub[visit_col].values == post_v).astype(float)
            if design == "two_arm":
                treated = (pv_sub[arm_col].values == cfg["arm_treated"]).astype(float)
            else:
                treated = None
            for feat in features:
                y = pv_sub[feat].values.astype(float)
                if treated is not None:
                    X = np.column_stack(
                        [np.ones_like(post), post, treated, post * treated]
                    )
                else:
                    X = np.column_stack([np.ones_like(post), post])
                if np.linalg.matrix_rank(X) < X.shape[1]:
                    continue
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                n, p = X.shape
                dof = n - p
                if dof > 0:
                    sigmas.append(float(np.sqrt(np.sum(resid**2) / dof)))

            sigma = float(np.nanmedian(sigmas)) if sigmas else 1.0
            out[name] = {
                "n_per_group": n_per_group,
                "sigma": sigma,
                "design": design,
            }
            print(f"  MDE {name}: n/group={n_per_group}, sigma={sigma:.3f}, design={design}")
        except Exception as exc:
            print(f"  MDE {name}: failed ({exc})")
    return out


def _panel_mde(ax, mde_data: dict[str, dict]):
    """H: Power curves — minimum detectable effect vs sample size."""
    # Determine grid range to include all observed n values
    all_n = [info["n_per_group"] for info in mde_data.values() if info["n_per_group"] > 0]
    n_max = max(max(all_n) + 10, 61) if all_n else 61
    n_grid = np.arange(3, n_max, 1)

    for name, info in mde_data.items():
        sigma = info["sigma"]
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        paired = info.get("design", "two_arm") == "single_arm_paired"
        y = _mde_curve(n_grid, sigma, paired=paired)
        ls = "--" if paired else "-"
        ax.plot(
            n_grid, y, lw=1.8, ls=ls,
            color=_MDE_PALETTE.get(name, "grey"), label=name,
        )
        n_actual = info["n_per_group"]
        if n_actual >= 3:
            mde_val = _mde_curve(np.array([n_actual]), sigma, paired=paired)[0]
            ax.scatter(
                [n_actual], [mde_val],
                color=_MDE_PALETTE.get(name, "grey"),
                edgecolors="black", zorder=5, s=60,
                linewidth=1.0,
            )
            # Add label near dot
            ax.annotate(
                f"n={n_actual}",
                (n_actual, mde_val),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=6,
                fontweight="bold",
                color=_MDE_PALETTE.get(name, "grey"),
            )
    ax.set_xlabel("Participants per group")
    ax.set_ylabel("Minimum detectable effect")
    ax.set_title("Power Curves: MDE vs Sample Size", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 4 panels (A-H)."""
    print("Supplementary Figure 4: Sensitivity to Modeling and Preprocessing")
    data = _run_sensitivity()

    panels = [
        ("panel_A", _panel_cell_vs_part, (7.0, 6.0)),
        ("panel_B", _panel_bootstrap_ci, (8.6, 6.6)),
        ("panel_C", _panel_std_vs_unstd, (7.0, 6.0)),
        ("panel_D", _panel_mean_vs_median, (7.0, 6.0)),
        ("panel_E", _panel_log_sensitivity, (7.0, 6.0)),
        ("panel_F", _panel_ct_heatmap, (8.8, 5.8)),
        ("panel_G", _panel_rank_concordance, (7.2, 5.8)),
    ]

    for panel_name, fn, size in panels:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

    if "adata" in data:
        del data["adata"]
    data.clear()

    # H: Power curves (MDE) — separate data pipeline
    print("  Loading MDE data ...")
    mde_data = _load_mde_data()
    if mde_data:
        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        _panel_mde(ax, mde_data)
        fig.tight_layout()
        save_panel(fig, "panel_H", FIGURE_NAME, SUPP_OUTPUT)
    mde_data.clear()

    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
