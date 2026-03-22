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
# Simulation panels H–K
# ======================================================================

# Simulation grid parameters
_SIM_ITERATIONS = 200
_SIM_SAMPLE_SIZES = [20, 40, 60]
_SIM_EFFECT_SIZES = [0.0, 0.2, 0.5, 1.0]
_SIM_N_GENES = 50
_SIM_N_SIGNAL = 10  # first 10 genes get the effect
_SIM_NOISE_SD = 1.0
_SIM_METHODS = ["sctrial_did", "pseudobulk_ols", "wilcoxon"]
_SIM_METHOD_LABELS = {
    "sctrial_did": "sctrial DiD",
    "pseudobulk_ols": "Pseudobulk OLS",
    "wilcoxon": "Wilcoxon",
}
_SIM_METHOD_COLORS = {
    "sctrial_did": "#2c3e50",
    "pseudobulk_ols": "#e67e22",
    "wilcoxon": "#27ae60",
}


def _run_simulation_grid():
    """Run full simulation grid. Returns combined results DataFrame."""
    from sctrial.stats.simulation import run_method_comparison

    all_dfs = []
    for n in _SIM_SAMPLE_SIZES:
        for beta in _SIM_EFFECT_SIZES:
            print(f"    n={n}, beta={beta} ...")
            effects = {f"gene_{i}": beta for i in range(_SIM_N_SIGNAL)}
            df = run_method_comparison(
                n_participants=n,
                n_genes=_SIM_N_GENES,
                effect_sizes=effects,
                noise_sd=_SIM_NOISE_SD,
                n_iterations=_SIM_ITERATIONS,
                methods=_SIM_METHODS,
                seed=42 + n * 100 + int(beta * 10),
            )
            df["n_participants"] = n
            df["target_beta"] = beta
            df["is_signal"] = df["gene"].isin(
                [f"gene_{i}" for i in range(_SIM_N_SIGNAL)]
            )
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def _panel_sim_tpr(ax, results):
    """Panel H: True positive rate at FDR < 0.05 across effect sizes."""
    from statsmodels.stats.multitest import multipletests

    n_default = _SIM_SAMPLE_SIZES[1]  # middle sample size
    # Include ALL genes (null + signal) for proper FDR correction scope,
    # but only for iterations that have target_beta > 0
    subset = results[
        (results["target_beta"] > 0) & (results["n_participants"] == n_default)
    ].copy()

    rows = []
    for (method, beta, it), grp in subset.groupby(
        ["method", "target_beta", "iteration"]
    ):
        pvals = grp["pvalue"].dropna().values
        is_sig = grp.loc[grp["pvalue"].notna(), "is_signal"].values
        if len(pvals) == 0:
            continue
        # FDR correction over ALL genes (null + signal) per iteration
        reject = multipletests(pvals, alpha=0.05, method="fdr_bh")[0]
        # TPR = fraction of signal genes rejected
        if is_sig.sum() > 0:
            rows.append(
                {"method": method, "target_beta": beta,
                 "tpr": reject[is_sig].mean()}
            )
    tpr_df = pd.DataFrame(rows)
    tpr_agg = (
        tpr_df.groupby(["method", "target_beta"])["tpr"]
        .agg(["mean", "std"])
        .reset_index()
    )

    betas = sorted(tpr_agg["target_beta"].unique())
    x = np.arange(len(betas))
    width = 0.25
    for i, method in enumerate(_SIM_METHODS):
        sub = tpr_agg[tpr_agg["method"] == method]
        vals, errs = [], []
        for b in betas:
            s = sub[sub["target_beta"] == b]
            vals.append(s["mean"].values[0] if len(s) else 0)
            errs.append(s["std"].values[0] if len(s) else 0)
        ax.bar(
            x + i * width, vals, width, yerr=errs,
            label=_SIM_METHOD_LABELS[method],
            color=_SIM_METHOD_COLORS[method], alpha=0.85, capsize=3,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels([fr"$\beta$={b}" for b in betas])
    ax.set_xlabel(r"True effect size ($\beta$)")
    ax.set_ylabel("True Positive Rate (FDR < 0.05)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Power (n={n_default})", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    despine(ax)


def _panel_sim_fpr(ax, results):
    """Panel I: Per-test type I error rate under pure null (target_beta=0).

    Uses uncorrected p < 0.05 rejection rate (not BH-adjusted) so that the
    expected rate under a well-calibrated method is exactly 0.05, producing
    visible bars near the nominal line.
    """
    # Pure null: only iterations where target_beta=0 (all genes are null)
    pure_null = results[results["target_beta"] == 0.0].copy()

    rows = []
    for (method, n, it), grp in pure_null.groupby(
        ["method", "n_participants", "iteration"]
    ):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        # Uncorrected: fraction of raw p < 0.05
        rows.append(
            {"method": method, "n_participants": n,
             "fpr": (pvals < 0.05).mean()}
        )
    fpr_df = pd.DataFrame(rows)
    fpr_agg = (
        fpr_df.groupby(["method", "n_participants"])["fpr"]
        .agg(["mean", "std"])
        .reset_index()
    )

    ns = sorted(fpr_agg["n_participants"].unique())
    x = np.arange(len(ns))
    width = 0.25
    for i, method in enumerate(_SIM_METHODS):
        sub = fpr_agg[fpr_agg["method"] == method]
        vals, errs = [], []
        for n_val in ns:
            s = sub[sub["n_participants"] == n_val]
            vals.append(s["mean"].values[0] if len(s) else 0)
            errs.append(s["std"].values[0] if len(s) else 0)
        ax.bar(
            x + i * width, vals, width, yerr=errs,
            label=_SIM_METHOD_LABELS[method],
            color=_SIM_METHOD_COLORS[method], alpha=0.85, capsize=3,
        )

    ax.axhline(0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_xlabel("Sample size (participants)")
    ax.set_ylabel("Type I Error Rate (uncorrected)")
    ax.set_ylim(0, max(0.15, fpr_agg["mean"].max() * 1.3))
    ax.set_title("Type I Error Calibration", fontweight="bold")
    ax.legend(fontsize=7)
    despine(ax)


def _panel_sim_bias(ax, results):
    """Panel J: Effect-size bias (estimated vs true beta).

    All three methods are nearly unbiased, so points overlap on the y=x
    line.  We use distinct marker shapes and decreasing sizes so every
    method is visible without shifting x (which would create false
    apparent bias against the identity line).
    """
    n_default = _SIM_SAMPLE_SIZES[1]
    # Only non-zero effects for bias assessment
    signal = results[
        results["is_signal"] & (results["target_beta"] > 0)
        & (results["n_participants"] == n_default)
    ].copy()

    # Distinct markers + decreasing size so later-drawn methods don't
    # fully occlude earlier ones.  Draw order: largest first.
    markers = ["o", "s", "^"]
    sizes = [9, 7, 5]

    for idx, method in enumerate(_SIM_METHODS):
        sub = signal[signal["method"] == method]
        agg = (
            sub.groupby("true_beta")
            .agg(est_mean=("estimated_beta", "mean"),
                 est_se=("estimated_beta", "sem"))
            .reset_index()
        )
        ax.errorbar(
            agg["true_beta"],
            agg["est_mean"],
            yerr=1.96 * agg["est_se"],
            marker=markers[idx],
            markersize=sizes[idx],
            label=_SIM_METHOD_LABELS[method],
            color=_SIM_METHOD_COLORS[method],
            capsize=4,
            linewidth=1.5,
            zorder=3 + idx,
        )

    lims = [0, max(_SIM_EFFECT_SIZES) + 0.3]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.set_xlabel(r"True $\beta_{\mathrm{DiD}}$")
    ax.set_ylabel(r"Estimated $\beta$")
    ax.set_title(f"Effect-Size Bias (n={n_default})", fontweight="bold")
    ax.legend(fontsize=7)
    despine(ax)


def _panel_sim_coverage(ax, results):
    """Panel K: P-value QQ plot under pure null (target_beta=0).

    Plots observed vs expected p-value quantiles for each method under
    the null hypothesis. Well-calibrated methods follow the diagonal;
    anti-conservative methods curve above it.
    """
    n_default = _SIM_SAMPLE_SIZES[1]
    pure_null = results[
        (results["target_beta"] == 0.0)
        & (results["n_participants"] == n_default)
    ].copy()

    for method in _SIM_METHODS:
        pvals = pure_null.loc[
            pure_null["method"] == method, "pvalue"
        ].dropna().sort_values().values
        if len(pvals) == 0:
            continue
        n = len(pvals)
        expected = (np.arange(1, n + 1) - 0.5) / n
        # Convert to -log10 for visual clarity
        obs_log = -np.log10(pvals + 1e-300)
        exp_log = -np.log10(expected + 1e-300)
        ax.scatter(
            exp_log, obs_log, s=3, alpha=0.3,
            color=_SIM_METHOD_COLORS[method], rasterized=True,
        )
        # Full-opacity handle for legend
        ax.scatter([], [], s=25, alpha=1.0,
                   color=_SIM_METHOD_COLORS[method],
                   label=_SIM_METHOD_LABELS[method])

    # Diagonal reference
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_title(f"P-value Calibration QQ (n={n_default})", fontweight="bold")
    ax.legend(fontsize=7)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 4 panels (A–K).

    Layout:
      A  Analytical vs bootstrap SE (all 5 datasets, faceted forest plot)
      B  Standardised vs unstandardised effect sizes (Melanoma)
      C  Mean vs median aggregation comparison (Melanoma)
      D  Log-transform sensitivity (Melanoma)
      E  Cell-type-stratified DiD heatmap (Melanoma)
      F  Rank-order concordance across choices (Melanoma)
      G  Leave-one-out stability matrix (all datasets)
      H  Simulation: power (TPR) across effect sizes
      I  Simulation: type I error calibration across sample sizes
      J  Simulation: effect-size bias (estimated vs true beta)
      K  Simulation: p-value calibration QQ plot
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

    # Panels H–K: Monte Carlo simulation study
    print("  Running simulation grid (this may take several minutes) ...")
    sim_results = _run_simulation_grid()

    # Save raw simulation results for reproducibility
    out_dir = SUPP_OUTPUT / f"{FIGURE_NAME}_panels"
    out_dir.mkdir(exist_ok=True)
    sim_results.to_csv(out_dir / "simulation_results.csv", index=False)
    print(f"    Saved raw results → {out_dir / 'simulation_results.csv'}")

    sim_panels = [
        ("panel_H", _panel_sim_tpr, (7.0, 5.0)),
        ("panel_I", _panel_sim_fpr, (7.0, 5.0)),
        ("panel_J", _panel_sim_bias, (6.0, 5.0)),
        ("panel_K", _panel_sim_coverage, (7.0, 5.0)),
    ]
    for panel_name, fn, size in sim_panels:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, sim_results)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)
    del sim_results

    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
