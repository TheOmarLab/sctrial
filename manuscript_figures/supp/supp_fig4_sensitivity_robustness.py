"""
Supplementary Figure 4 — Sensitivity and Robustness.
=====================================================

Show how DiD results change under different analytical decisions.

Panels
------
  A  Analytical vs bootstrap SE (all 5 datasets, forest plot).
  B  Standardised vs unstandardised effect sizes (Melanoma).
  C  Mean vs median aggregation comparison (Melanoma).
  D  Log-transform sensitivity (Melanoma).
  E  Cell-type-stratified DiD heatmap (Melanoma).
  F  Rank-order concordance across preprocessing choices (Melanoma).
  G  Leave-one-out stability matrix (max influence, all datasets).
  H  Simulation: power (TPR) across effect sizes.
  I  Simulation: type I error calibration across sample sizes.
  J  Simulation: effect-size bias (estimated vs true beta).
  K  Simulation: p-value calibration QQ plot.

Non-overlap guardrail: methodological sensitivity only, not biological claims.
"""

from __future__ import annotations

import gc
from pathlib import Path

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
    get_aml,
    get_cart,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
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


# ── Panel A: Multi-dataset analytical vs bootstrap SE ──────────────

def _run_multi_bootstrap():
    """Run analytical + bootstrap DiD on all 5 datasets."""
    import sctrial

    results = {}
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

            feats = [f for f in _FEATURES if f in adata.var_names]
            if len(feats) < 3:
                continue

            arm_col = cfg.get("arm_col")
            design_type = cfg.get("design", "two_arm")

            if design_type == "two_arm" and arm_col:
                design = sctrial.TrialDesign(
                    participant_col=cfg["participant_col"],
                    visit_col=cfg["visit_col"],
                    arm_col=arm_col,
                    arm_treated=cfg["arm_treated"],
                    arm_control=cfg["arm_control"],
                )
                print(f"  bootstrap {name}: analytical ...")
                part = sctrial.did_table(
                    adata, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit", standardize=True,
                )
                print(f"  bootstrap {name}: bootstrap ...")
                boot = sctrial.did_table(
                    adata, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit", standardize=True,
                    use_bootstrap=True, n_boot=200, seed=42,
                )
                results[name] = {"part": part, "boot": boot}
            else:
                # Single-arm: use within_arm_comparison
                arm_filter = cfg.get("arm_filter")
                arm_value = arm_filter or "All"
                if arm_filter and arm_col and arm_col in adata.obs.columns:
                    adata = adata[adata.obs[arm_col] == arm_filter].copy()
                # Build a design for within-arm comparison
                design = sctrial.TrialDesign(
                    participant_col=cfg["participant_col"],
                    visit_col=cfg["visit_col"],
                    arm_col=arm_col if arm_col and arm_col in adata.obs.columns else None,
                )
                print(f"  bootstrap {name}: within-arm analytical ...")
                part = sctrial.within_arm_comparison(
                    adata, arm_value, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit",
                )
                print(f"  bootstrap {name}: within-arm bootstrap ...")
                boot = sctrial.within_arm_comparison(
                    adata, arm_value, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit",
                    use_bootstrap=True, n_boot=200, seed=42,
                )
                results[name] = {"part": part, "boot": boot}
            del adata
        except Exception as exc:
            print(f"  bootstrap {name}: failed ({exc})")
    return results


def _panel_bootstrap_multi(fig, boot_data: dict):
    """A: Faceted forest plot — analytical vs bootstrap SE per dataset."""
    names = [n for n in boot_data if boot_data[n].get("part") is not None]
    if not names:
        ax = fig.subplots(1, 1)
        ax.text(0.5, 0.5, "No bootstrap data", ha="center", va="center",
                transform=ax.transAxes)
        return

    ncols = len(names)
    axes = fig.subplots(1, ncols, sharey=False)
    if ncols == 1:
        axes = [axes]

    # Detect which column holds the beta / SE
    def _beta_col(df):
        for c in ("beta_DiD", "beta_delta", "beta_time", "beta"):
            if c in df.columns:
                return c
        return None

    def _se_col(df):
        for c in ("se_DiD", "se_delta", "se_time", "se"):
            if c in df.columns:
                return c
        return None

    def _se_boot_col(df):
        """Find the bootstrap SE column. Returns None if not present."""
        for c in ("se_DiD_boot", "se_delta_boot", "se_time_boot"):
            if c in df.columns:
                return c
        return None

    for ax, name in zip(axes, names):
        part = boot_data[name]["part"]
        boot = boot_data[name]["boot"]

        bc = _beta_col(part)
        sc_a = _se_col(part)
        sc_b = _se_boot_col(boot)
        if bc is None or sc_a is None or sc_b is None:
            ax.set_title(name, fontweight="bold", fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            continue

        df_a = part.set_index("feature")[[bc, sc_a]].rename(
            columns={bc: "beta", sc_a: "se_analytical"})
        df_b = boot.set_index("feature")[[sc_b]].rename(
            columns={sc_b: "se_boot"})
        df = df_a.join(df_b, how="inner").reset_index()
        df = df.sort_values("beta", ascending=True).reset_index(drop=True)

        y = np.arange(len(df))
        off = 0.15

        ax.errorbar(df["beta"], y - off, xerr=1.96 * df["se_analytical"],
                     fmt="s", markersize=3, color=_PAL["cell"], elinewidth=0.8,
                     capsize=1.5, label="Analytical")
        ax.errorbar(df["beta"], y + off, xerr=1.96 * df["se_boot"],
                     fmt="o", markersize=3, color=_PAL["participant"],
                     elinewidth=0.8, capsize=1.5, label="Bootstrap")
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(df["feature"], fontsize=6)
        ax.set_title(name, fontweight="bold", fontsize=9)
        if ax == axes[0]:
            ax.set_xlabel("β with 95% CI", fontsize=8)
            ax.legend(fontsize=6, loc="lower right", frameon=True)
        else:
            ax.set_xlabel("")
        despine(ax)

    fig.suptitle("Analytical vs Bootstrap SE", fontweight="bold", fontsize=11)


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
    ax.set_title("Standardised vs Unstandardised (Melanoma)",
                 fontweight="bold")
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
        ax.set_title("Mean vs Median Aggregation (Melanoma)",
                 fontweight="bold")
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
        ax.set_title("Mean vs Median Aggregation (Melanoma)",
                 fontweight="bold")
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
    ax.set_title("Mean vs Median Aggregation (Melanoma)",
                 fontweight="bold")
    despine(ax)


# ── Panel E: Log-transform sensitivity ────────────────────────────

def _panel_log_sensitivity(ax, data: dict):
    """Scatter: log1p_tpm betas vs raw TPM betas."""
    log_df = data["part"].set_index("feature")["beta_DiD"]
    raw_res = data.get("raw")
    if raw_res is None or raw_res.empty:
        ax.text(0.5, 0.5, "No raw-TPM results", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Log-Transform Sensitivity (Melanoma)",
                 fontweight="bold")
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
    ax.set_title("Log-Transform Sensitivity (Melanoma)",
                 fontweight="bold")
    despine(ax)


# ── Panel F: Cell-type stratified heatmap ─────────────────────────

def _panel_ct_heatmap(ax, data: dict):
    """Heatmap: DiD effect sizes stratified by top cell types."""
    ct_results = data.get("ct_results", {})
    if not ct_results:
        ax.text(0.5, 0.5, "No cell-type-stratified results", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("Cell-Type Stratified DiD (Melanoma)",
                 fontweight="bold")
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
    # Show top features by mean absolute effect across cell types
    mat["_mean_abs"] = mat.abs().mean(axis=1)
    mat = mat.sort_values("_mean_abs", ascending=False).head(12).drop(columns="_mean_abs")

    sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, linewidths=0.5,
                linecolor="white", cbar_kws={"shrink": 0.6, "label": "β"},
                annot=True, fmt=".2f", annot_kws={"fontsize": 6})
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Feature")
    ax.set_title("Cell-Type Stratified DiD (Melanoma)",
                 fontweight="bold")
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
    ax.set_title("Rank Concordance Across Choices (Melanoma)",
                 fontweight="bold")

    for i, rho in enumerate(rhos):
        ax.text(rho + 0.02, i, f"{rho:.2f}", va="center", fontsize=7)

    despine(ax)


# ── Panel G: Leave-one-out stability ──────────────────────────────

def _find_beta_col(df):
    """Return the first beta column present in *df*."""
    for c in ("beta_DiD", "beta_delta", "beta_time", "beta_arm"):
        if c in df.columns:
            return c
    raise KeyError(f"No beta column in {list(df.columns)}")


def _panel_loo_stability(ax):
    """G: LOO max-deviation of betas — heatmap of features × datasets.

    Metric: max_i |beta_LOO_i - beta_full| / (|beta_full| + 0.01)
    This measures the worst-case influence of any single participant,
    normalised by the full-data effect size. Stable near zero unlike CV.
    Uses all participants per dataset (no data subsampling).
    """
    import sctrial

    rows = {}
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

            feats = [f for f in _FEATURES if f in adata.var_names]
            if len(feats) < 2:
                continue

            # Slice to only needed genes — reduces memory ~6000×
            # so LOO copies are tiny (~34 participants × 3 genes)
            adata = adata[:, feats].copy()

            arm_col = cfg.get("arm_col")
            design_type = cfg.get("design", "two_arm")
            pid_col = cfg["participant_col"]
            vis_col = cfg["visit_col"]

            # Use all participants — no data subsampling
            obs = adata.obs
            pids = obs[pid_col].unique().tolist()
            if len(pids) < 4:
                continue

            # Full-data betas
            if design_type == "two_arm" and arm_col:
                design = sctrial.TrialDesign(
                    participant_col=pid_col, visit_col=vis_col,
                    arm_col=arm_col, arm_treated=cfg["arm_treated"],
                    arm_control=cfg["arm_control"],
                )
                full_df = sctrial.did_table(
                    adata, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit", standardize=True,
                )
                beta_col = "beta_DiD"
            else:
                arm_filter = cfg.get("arm_filter")
                arm_value = arm_filter or "All"
                sub = adata
                if arm_filter and arm_col and arm_col in obs.columns:
                    sub = adata[obs[arm_col] == arm_filter].copy()
                design = sctrial.TrialDesign(
                    participant_col=pid_col, visit_col=vis_col,
                    arm_col=arm_col if arm_col and arm_col in obs.columns else None,
                )
                full_df = sctrial.within_arm_comparison(
                    sub, arm_value, feats, design, cfg["visits"],
                    layer=layer, aggregate="participant_visit",
                )
                beta_col = _find_beta_col(full_df)

            full_betas = full_df.set_index("feature")[beta_col]

            # LOO: drop each participant, recompute betas
            loo_betas = []
            for pid in pids:
                mask = obs[pid_col] != pid
                sub = adata[mask].copy()
                try:
                    if design_type == "two_arm" and arm_col:
                        df_loo = sctrial.did_table(
                            sub, feats, design, cfg["visits"],
                            layer=layer, aggregate="participant_visit",
                            standardize=True,
                        )
                        loo_betas.append(
                            df_loo.set_index("feature")[_find_beta_col(df_loo)]
                        )
                    else:
                        if arm_filter and arm_col and arm_col in sub.obs.columns:
                            sub = sub[sub.obs[arm_col] == arm_filter].copy()
                        loo_design = sctrial.TrialDesign(
                            participant_col=pid_col, visit_col=vis_col,
                            arm_col=arm_col if arm_col and arm_col in sub.obs.columns else None,
                        )
                        df_loo = sctrial.within_arm_comparison(
                            sub, arm_value, feats, loo_design, cfg["visits"],
                            layer=layer, aggregate="participant_visit",
                        )
                        loo_betas.append(
                            df_loo.set_index("feature")[_find_beta_col(df_loo)]
                        )
                except Exception:
                    pass

            if len(loo_betas) < 3:
                continue

            loo_mat = pd.DataFrame(loo_betas)
            # Max-deviation: max_i |beta_LOO_i - beta_full| / (|beta_full| + 0.01)
            deviations = loo_mat.subtract(full_betas, axis=1).abs()
            max_dev = deviations.max() / (full_betas.abs() + 0.01)
            rows[name] = max_dev
            print(f"  LOO {name}: {len(pids)} pids, {len(feats)} feats")
            del adata
        except Exception as exc:
            print(f"  LOO {name}: failed ({exc})")

    if not rows:
        ax.text(0.5, 0.5, "No LOO data", ha="center", va="center",
                transform=ax.transAxes)
        return

    mat = pd.DataFrame(rows)

    sns.heatmap(mat, ax=ax, cmap="YlOrRd", linewidths=0.5,
                linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Max LOO deviation"},
                annot=True, fmt=".2f", annot_kws={"fontsize": 7})
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Feature")
    ax.set_title("Leave-One-Out Stability (max influence)", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.tick_params(axis="y", labelsize=8)


# ======================================================================
# Legacy dataset config (shared by panels G, H, and bootstrap)
# ======================================================================

_MDE_DATASET_CFG = {
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
        "design": "single_arm_paired",
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
        "design": "single_arm_paired",
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




# ======================================================================
# NatMeth Benchmark panels H–N
# ======================================================================

# Benchmark data lives in the manuscript directory (generated on HPC)
_BENCHMARK_CSV = Path(__file__).resolve().parents[4] / "manuscript" / "benchmark" / "simulation" / "benchmark_combined.csv"

# Method display configuration
# Plot order: sctrial last so it renders on top and stays visible
_BENCH_METHODS = ["wilcoxon_paired", "nebula", "dreamlet", "sctrial_did"]
_BENCH_METHOD_LABELS = {
    "sctrial_did": "sctrial (DiD)",
    "dreamlet": "dreamlet",
    "nebula": "NEBULA",
    "wilcoxon_paired": "Wilcoxon (paired)",
}
_BENCH_METHOD_COLORS = {
    "sctrial_did": "#1f77b4",   # strong blue
    "dreamlet": "#d62728",      # red
    "nebula": "#ff7f0e",        # orange
    "wilcoxon_paired": "#2ca02c",  # green
}
_BENCH_METHOD_MARKERS = {
    "sctrial_did": "o",
    "dreamlet": "D",
    "nebula": "s",
    "wilcoxon_paired": "^",
}


def _load_benchmark_data():
    """Load NatMeth benchmark results from HPC output."""
    if not _BENCHMARK_CSV.exists():
        raise FileNotFoundError(
            f"Benchmark results not found at {_BENCHMARK_CSV}.\n"
            "Run the benchmark on HPC first, then rsync results locally."
        )
    df = pd.read_csv(_BENCHMARK_CSV, low_memory=False)
    df["design"] = df["scenario"].str.split("__").str[0]
    # Extract effect size from scenario name for DE scenarios
    beta_extract = df["scenario"].str.extract(r"b(\d+\.?\d*)")
    df["scenario_beta"] = pd.to_numeric(beta_extract[0], errors="coerce")
    return df


def _filter_bench(df, design="two_arm", scenario_pattern=None):
    """Filter benchmark data by design and optional scenario regex."""
    sub = df[df["design"] == design]
    if scenario_pattern:
        sub = sub[sub["scenario"].str.contains(scenario_pattern, regex=True)]
    return sub


def _panel_bench_power(fig_or_ax, bench_df, design="two_arm"):
    """Panel H/M: Power curves — signal-gene detection rate vs effect size.

    Faceted by sample size. Shows power on signal genes ONLY (not pooled
    across all genes). Lines per method with ±SE bands.
    """
    de_data = _filter_bench(bench_df, design, "de_pos")
    signal = de_data[de_data["is_signal"] == True].copy()
    if signal.empty:
        return

    # Compute per-iteration power
    rows = []
    for (method, n, beta, it), grp in signal.groupby(
        ["method", "n_per_arm", "scenario_beta", "iteration"]
    ):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({
            "method": method, "n_per_arm": n, "beta": beta,
            "power": (pvals < 0.05).mean(),
        })
    power_df = pd.DataFrame(rows)
    agg = (
        power_df.groupby(["method", "n_per_arm", "beta"])["power"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    ns = sorted(agg["n_per_arm"].unique())
    n_panels = len(ns)

    if hasattr(fig_or_ax, "subplots"):
        axes = fig_or_ax.subplots(1, n_panels, sharey=True)
        if n_panels == 1:
            axes = [axes]
    else:
        axes = [fig_or_ax]
        ns = ns[:1]

    for ax_idx, (ax, n_val) in enumerate(zip(axes, ns)):
        for zi, method in enumerate(_BENCH_METHODS):
            sub = agg[(agg["method"] == method) & (agg["n_per_arm"] == n_val)].sort_values("beta")
            if sub.empty:
                continue
            is_sctrial = method == "sctrial_did"
            ax.plot(
                sub["beta"], sub["mean"],
                marker=_BENCH_METHOD_MARKERS[method],
                markersize=9 if is_sctrial else 7,
                label=_BENCH_METHOD_LABELS[method] if ax_idx == 0 else None,
                color=_BENCH_METHOD_COLORS[method],
                linewidth=2.5 if is_sctrial else 1.8,
                zorder=10 if is_sctrial else zi + 1,
            )
            ax.fill_between(
                sub["beta"],
                (sub["mean"] - 1.96 * sub["se"]).clip(0, 1),
                (sub["mean"] + 1.96 * sub["se"]).clip(0, 1),
                color=_BENCH_METHOD_COLORS[method], alpha=0.12,
                zorder=0,
            )
        ax.set_xlabel(r"True effect size ($\beta$)")
        ax.set_title(f"n = {n_val} per arm", fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        despine(ax)

    design_label = "Two-arm" if design == "two_arm" else "Single-arm"
    axes[0].set_ylabel(f"Power (p < 0.05 on signal genes)\n{design_label}")
    axes[0].legend(fontsize=7, loc="lower right")


def _panel_bench_fpr(ax, bench_df):
    """Panel I: Type I error calibration — dot-and-whisker by method × n.

    Both two-arm and single-arm null scenarios. Methods grouped on y-axis,
    dots per sample size with Wilson CIs. Red vertical line at 0.05.
    """
    # Use only pure-null scenarios (null_n* and null_hetero_n*)
    null_data = bench_df[
        (bench_df["true_beta"] == 0.0)
        & (bench_df["scenario"].str.contains("null"))
    ].copy()

    rows = []
    for (method, design, n, it), grp in null_data.groupby(
        ["method", "design", "n_per_arm", "iteration"]
    ):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({
            "method": method, "design": design, "n_per_arm": n,
            "fpr": (pvals < 0.05).mean(),
        })
    fpr_df = pd.DataFrame(rows)

    # Aggregate across iterations (pool both designs for cleaner plot)
    fpr_agg = (
        fpr_df.groupby(["method", "n_per_arm"])["fpr"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    fpr_agg["se"] = fpr_agg["std"] / np.sqrt(fpr_agg["count"])
    fpr_agg["ci_lo"] = (fpr_agg["mean"] - 1.96 * fpr_agg["se"]).clip(0)
    fpr_agg["ci_hi"] = fpr_agg["mean"] + 1.96 * fpr_agg["se"]

    ns = sorted(fpr_agg["n_per_arm"].unique())
    marker_list = ["o", "s", "D", "^", "v"]

    y_pos = 0
    y_ticks, y_labels = [], []
    for method in _BENCH_METHODS:
        for ni, n_val in enumerate(ns):
            sub = fpr_agg[
                (fpr_agg["method"] == method) & (fpr_agg["n_per_arm"] == n_val)
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            ax.errorbar(
                row["mean"], y_pos,
                xerr=[[row["mean"] - row["ci_lo"]], [row["ci_hi"] - row["mean"]]],
                marker=marker_list[ni % len(marker_list)],
                markersize=7, capsize=4, linewidth=1.5,
                color=_BENCH_METHOD_COLORS[method],
                label=f"n={n_val}" if method == _BENCH_METHODS[0] else None,
            )
            y_pos += 1
        center = y_pos - len(ns) / 2
        y_ticks.append(center)
        y_labels.append(_BENCH_METHOD_LABELS[method])
        y_pos += 0.5

    ax.axvline(0.05, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvspan(0.03, 0.07, color="red", alpha=0.06)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Type I Error Rate (p < 0.05)")
    ax.set_title("Null Calibration (both designs)", fontweight="bold")
    ax.set_xlim(0, max(0.12, fpr_agg["ci_hi"].max() * 1.1))

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=marker_list[i], color="gray",
               markersize=6, linestyle="none", label=f"n={n}")
        for i, n in enumerate(ns)
    ]
    handles.append(Line2D([0], [0], color="red", linestyle="--",
                          linewidth=1.2, label="Nominal 5%"))
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.invert_yaxis()
    despine(ax)


def _panel_bench_lambda(ax, bench_df):
    """Panel J: Genomic inflation factor (λ_GC) under null.

    λ_GC = median(χ²_obs) / 0.456 per method × n. Well-calibrated ≈ 1.0.
    Uses two-arm null scenarios only.
    """
    null_data = _filter_bench(bench_df, "two_arm", r"null_n\d+$")
    pure_null = null_data[null_data["true_beta"] == 0.0]

    rows = []
    for (method, n), grp in pure_null.groupby(["method", "n_per_arm"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        chi2_obs = sp_stats.chi2.ppf(1 - pvals.clip(1e-300, 1), df=1)
        lambda_gc = np.median(chi2_obs) / sp_stats.chi2.ppf(0.5, df=1)
        rows.append({"method": method, "n_per_arm": n, "lambda_gc": lambda_gc})

    lambda_df = pd.DataFrame(rows)
    ns = sorted(lambda_df["n_per_arm"].unique())

    for zi, method in enumerate(_BENCH_METHODS):
        sub = lambda_df[lambda_df["method"] == method].sort_values("n_per_arm")
        if sub.empty:
            continue
        is_sctrial = method == "sctrial_did"
        ax.plot(
            sub["n_per_arm"], sub["lambda_gc"],
            marker=_BENCH_METHOD_MARKERS[method],
            markersize=10 if is_sctrial else 8,
            linewidth=2.5 if is_sctrial else 1.8,
            label=_BENCH_METHOD_LABELS[method],
            color=_BENCH_METHOD_COLORS[method],
            zorder=10 if is_sctrial else zi + 1,
        )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label=r"Ideal ($\lambda$ = 1)")
    ax.axhspan(0.95, 1.05, color="red", alpha=0.06)
    ax.set_xlabel("Sample size (participants per arm)")
    ax.set_ylabel(r"Genomic inflation factor ($\lambda_{\mathrm{GC}}$)")
    ax.set_title("Null Calibration Summary (two-arm)", fontweight="bold")
    ax.set_xticks(ns)
    ax.set_ylim(0.90, 1.15)
    ax.legend(fontsize=7)
    despine(ax)


def _panel_bench_qq(fig_or_ax, bench_df, n_target=None):
    """Panel K: Faceted QQ — one panel per method, colored by scenario type.

    Within each method panel, null-gene p-values are split into:
      • Pure-null scenarios (all genes null) — shown in method color
      • DE null-genes (null genes coexisting with signal genes) — shown in gray

    This reveals whether empirical Bayes variance moderation contaminates
    null-gene p-values when signal genes are present.
    """
    null_all = bench_df[bench_df["true_beta"] == 0.0].copy()
    null_all["is_de_scenario"] = null_all["scenario"].str.contains("de_pos")

    n_methods = len(_BENCH_METHODS)
    if hasattr(fig_or_ax, "subplots"):
        axes = fig_or_ax.subplots(1, n_methods, sharey=True)
        if n_methods == 1:
            axes = [axes]
    else:
        axes = None

    for mi, method in enumerate(_BENCH_METHODS):
        mdata = null_all[null_all["method"] == method]
        ax = axes[mi] if axes is not None else fig_or_ax

        # Split into pure-null vs DE-scenario null genes
        pure = mdata.loc[~mdata["is_de_scenario"], "pvalue"].dropna().sort_values().values
        de_null = mdata.loc[mdata["is_de_scenario"], "pvalue"].dropna().sort_values().values

        # Plot each category with its own QQ
        max_obs = 0
        max_exp = 0
        for pvals, color, alpha, label, zorder in [
            (de_null, "#888888", 0.35, "DE-scenario null genes", 1),
            (pure, _BENCH_METHOD_COLORS[method], 0.55, "Pure-null scenarios", 2),
        ]:
            if len(pvals) == 0:
                continue
            n = len(pvals)
            expected = (np.arange(1, n + 1) - 0.5) / n
            obs_log = -np.log10(pvals + 1e-300)
            exp_log = -np.log10(expected + 1e-300)
            ax.scatter(exp_log, obs_log, s=3, alpha=alpha, color=color,
                       rasterized=True, zorder=zorder,
                       label=label if mi == 0 else None)
            max_obs = max(max_obs, obs_log.max())
            max_exp = max(max_exp, exp_log.max())

        # 95% Beta envelope based on the larger set
        n_env = max(len(pure), len(de_null), 1)
        ranks = np.arange(1, n_env + 1)
        exp_env = -np.log10((ranks - 0.5) / n_env + 1e-300)
        lo_env = -np.log10(
            sp_stats.beta.ppf(0.975, ranks, n_env - ranks + 1) + 1e-300
        )
        hi_env = -np.log10(
            sp_stats.beta.ppf(0.025, ranks, n_env - ranks + 1) + 1e-300
        )
        ax.fill_between(exp_env, lo_env, hi_env, color="gray", alpha=0.1,
                        zorder=0, label="95% envelope" if mi == 0 else None)

        lim = max(max_exp, max_obs) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5, zorder=3)
        ax.set_title(_BENCH_METHOD_LABELS[method], fontweight="bold", fontsize=10)
        ax.set_xlabel(r"Expected $-\log_{10}(p)$")
        if mi == 0:
            ax.set_ylabel(r"Observed $-\log_{10}(p)$")
        despine(ax)

    # Shared legend across all panels
    if axes is not None and len(axes) > 0:
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
        handles = [
            mlines.Line2D([], [], marker="o", color="w",
                          markerfacecolor=_BENCH_METHOD_COLORS["sctrial_did"],
                          markersize=8, alpha=0.7,
                          label="Null genes — pure-null scenarios (method color)"),
            mlines.Line2D([], [], marker="o", color="w",
                          markerfacecolor="#888888",
                          markersize=8, alpha=0.5,
                          label="Null genes — DE scenarios (gray)"),
            mpatches.Patch(facecolor="gray", alpha=0.15,
                           label="95% Beta envelope"),
        ]
        fig = axes[0].get_figure()
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   fontsize=9, frameon=True, fancybox=True,
                   bbox_to_anchor=(0.5, -0.02))


def _panel_bench_de_null_fpr(ax, bench_df):
    """Panel L: Null-gene FPR in DE scenarios — the key benchmark finding.

    Shows how each method's false positive rate on null genes scales
    with signal strength. sctrial_did and wilcoxon_paired stay near
    nominal; dreamlet shows inflation from empirical Bayes variance
    moderation on the small (50-gene) benchmark panel.
    """
    de_data = _filter_bench(bench_df, "two_arm", "de_pos")
    null_genes = de_data[de_data["true_beta"] == 0.0].copy()

    rows = []
    for (method, n, beta), grp in null_genes.groupby(
        ["method", "n_per_arm", "scenario_beta"]
    ):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({
            "method": method, "n_per_arm": n, "beta": beta,
            "null_fpr": (pvals < 0.05).mean(),
        })
    fpr_df = pd.DataFrame(rows)

    # Plot: x = effect size, y = null-gene FPR, faceted by n
    # Use n=40 for the main display
    n_show = 40
    sub = fpr_df[fpr_df["n_per_arm"] == n_show].sort_values("beta")

    for zi, method in enumerate(_BENCH_METHODS):
        msub = sub[sub["method"] == method]
        if msub.empty:
            continue
        is_sctrial = method == "sctrial_did"
        ax.plot(
            msub["beta"], msub["null_fpr"],
            marker=_BENCH_METHOD_MARKERS[method],
            markersize=10 if is_sctrial else 8,
            linewidth=2.5 if is_sctrial else 1.8,
            label=_BENCH_METHOD_LABELS[method],
            color=_BENCH_METHOD_COLORS[method],
            zorder=10 if is_sctrial else zi + 1,
        )

    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Nominal 5%")
    ax.axhspan(0.03, 0.07, color="red", alpha=0.06)
    ax.set_xlabel(r"True effect size on signal genes ($\beta$)")
    ax.set_ylabel("FPR on null genes (p < 0.05)")
    ax.set_title(f"Null-gene FPR in DE scenarios (n={n_show}, two-arm)",
                 fontweight="bold")
    ax.set_ylim(0, min(1.0, fpr_df["null_fpr"].max() * 1.1))
    ax.legend(fontsize=7)
    despine(ax)


def _panel_bench_runtime(ax, bench_df):
    """Panel M: Runtime comparison across methods.

    Boxplot of per-iteration runtime (seconds) grouped by method.
    Runtime is recorded per method × iteration (duplicated across genes).
    """
    # Deduplicate: one runtime per method × scenario × iteration
    rt = (
        bench_df.groupby(["method", "scenario", "iteration"])["runtime_seconds"]
        .first().reset_index()
    )

    import matplotlib.patches as mpatches

    methods_data = []
    labels = []
    colors = []
    for method in _BENCH_METHODS:
        sub = rt[rt["method"] == method]["runtime_seconds"].dropna()
        if len(sub) == 0:
            continue
        methods_data.append(sub.values)
        labels.append(_BENCH_METHOD_LABELS[method])
        colors.append(_BENCH_METHOD_COLORS[method])

    bp = ax.boxplot(methods_data, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_yscale("log")
    ax.set_ylabel("Runtime per iteration (seconds, log scale)")
    ax.set_title("Computational Cost", fontweight="bold")
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 4 panels (A–N).

    Layout:
      A  Analytical vs bootstrap SE (all 5 datasets, faceted forest plot)
      B  Standardised vs unstandardised effect sizes (Melanoma)
      C  Mean vs median aggregation comparison (Melanoma)
      D  Log-transform sensitivity (Melanoma)
      E  Cell-type-stratified DiD heatmap (Melanoma)
      F  Rank-order concordance across choices (Melanoma)
      G  Leave-one-out stability matrix (all datasets)
      --- NatMeth Benchmark (4 methods: sctrial_did, dreamlet, NEBULA, Wilcoxon) ---
      H  Benchmark: two-arm signal-gene power curves (faceted by n)
      I  Benchmark: null calibration FPR (dot-and-whisker, both designs)
      J  Benchmark: genomic inflation factor λ_GC (two-arm null)
      K  Benchmark: faceted QQ plots with 95% Beta envelope (two-arm n=40)
      L  Benchmark: null-gene FPR in DE scenarios (dreamlet inflation finding)
      M  Benchmark: single-arm signal-gene power curves
      N  Benchmark: runtime comparison across methods
    """
    print("Supplementary Figure 4: Sensitivity to Modeling and Preprocessing")
    data = _run_sensitivity()

    # Panel A: Multi-dataset bootstrap SE (separate pipeline)
    print("  Loading multi-dataset bootstrap data ...")
    boot_data = _run_multi_bootstrap()
    # Filter to datasets that actually returned results
    boot_ok = {k: v for k, v in boot_data.items() if v.get("part") is not None}
    if boot_ok:
        ncols_a = len(boot_ok)
        fig = plt.figure(figsize=(4.5 * ncols_a, 6.5))
        _panel_bootstrap_multi(fig, boot_ok)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)
    boot_data.clear()

    # Panels B–F: single-dataset sensitivity (Sade-Feldman only)
    panels_bf = [
        ("panel_B", _panel_std_vs_unstd, (7.0, 6.0)),
        ("panel_C", _panel_mean_vs_median, (7.0, 6.0)),
        ("panel_D", _panel_log_sensitivity, (7.0, 6.0)),
        ("panel_E", _panel_ct_heatmap, (8.8, 5.8)),
        ("panel_F", _panel_rank_concordance, (7.2, 5.8)),
    ]

    for panel_name, fn, size in panels_bf:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

    if "adata" in data:
        del data["adata"]
    data.clear()

    # Panel G: LOO stability (heavy computation)
    print("  Computing LOO stability ...")
    fig, ax = plt.subplots(figsize=(9, 6))
    _panel_loo_stability(ax)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # Panels H–M: NatMeth benchmark panels (from HPC simulation)
    print("  Loading NatMeth benchmark results ...")
    bench_df = _load_benchmark_data()
    print(f"    {len(bench_df):,} rows, {bench_df.scenario.nunique()} scenarios")

    # Panel H: Two-arm power curves (signal genes, faceted by n)
    fig_h = plt.figure(figsize=(14, 4.5))
    _panel_bench_power(fig_h, bench_df, design="two_arm")
    fig_h.suptitle("Two-arm DiD: Signal-gene power", fontsize=13, y=1.02)
    fig_h.tight_layout()
    save_panel(fig_h, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    # Panel I: FPR calibration (dot-and-whisker, both designs)
    fig_i, ax_i = plt.subplots(figsize=(7.5, 6.0))
    _panel_bench_fpr(ax_i, bench_df)
    fig_i.tight_layout()
    save_panel(fig_i, "panel_I", FIGURE_NAME, SUPP_OUTPUT)

    # Panel J: Genomic inflation factor (λ_GC)
    fig_j, ax_j = plt.subplots(figsize=(7.0, 5.0))
    _panel_bench_lambda(ax_j, bench_df)
    fig_j.tight_layout()
    save_panel(fig_j, "panel_J", FIGURE_NAME, SUPP_OUTPUT)

    # Panel K: Faceted QQ with 95% envelopes (two-arm, n=40)
    fig_k = plt.figure(figsize=(16, 4.0))
    _panel_bench_qq(fig_k, bench_df, n_target=40)
    fig_k.tight_layout()
    save_panel(fig_k, "panel_K", FIGURE_NAME, SUPP_OUTPUT)

    # Panel L: Null-gene FPR in DE scenarios (key finding)
    fig_l, ax_l = plt.subplots(figsize=(7.5, 5.5))
    _panel_bench_de_null_fpr(ax_l, bench_df)
    fig_l.tight_layout()
    save_panel(fig_l, "panel_L", FIGURE_NAME, SUPP_OUTPUT)

    # Panel M: Single-arm power curves
    fig_m = plt.figure(figsize=(14, 4.5))
    _panel_bench_power(fig_m, bench_df, design="single_arm")
    fig_m.suptitle("Single-arm paired: Signal-gene power", fontsize=13, y=1.02)
    fig_m.tight_layout()
    save_panel(fig_m, "panel_M", FIGURE_NAME, SUPP_OUTPUT)

    # Panel N: Runtime comparison
    fig_n, ax_n = plt.subplots(figsize=(7.0, 5.0))
    _panel_bench_runtime(ax_n, bench_df)
    fig_n.tight_layout()
    save_panel(fig_n, "panel_N", FIGURE_NAME, SUPP_OUTPUT)

    del bench_df

    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
