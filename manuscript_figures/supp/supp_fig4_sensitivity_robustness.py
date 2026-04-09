"""
Supplementary Figure 4 — Sensitivity, Robustness, and Benchmarking.
===================================================================

Panels A–G characterize the sensitivity and robustness of sctrial's
participant-level inference on real datasets.  Panels H–N benchmark
sctrial against established multi-subject methods (dreamlet, NEBULA,
Wilcoxon on change scores) on a hierarchical gamma-Poisson simulator
across gene-panel sizes (50–2000) and signal fractions (1–20%).

Panels
------
  A  Analytical vs bootstrap SE (all 5 datasets, forest plot).
  B  Standardised vs unstandardised effect sizes (Melanoma).
  C  Mean vs median aggregation comparison (Melanoma).
  D  Log-transform sensitivity (Melanoma).
  E  Cell-type-stratified DiD heatmap (Melanoma).
  F  Rank-order concordance across preprocessing choices (Melanoma).
  G  Leave-one-out stability matrix (max influence, all datasets).
  H  Benchmark: null-gene FPR curves faceted by panel size.
  I  Benchmark: null-gene FPR heatmap (method × panel size × signal).
  J  Benchmark: pure-null λ_GC across panel sizes.
  K  Benchmark: QQ plots at 200 genes / 10% signal (challenging regime).
  L  Benchmark: pure-null FPR dot-and-whisker.
  M  Benchmark: signal-gene power across panel sizes.
  N  Benchmark: runtime scaling.

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
# Benchmark panels H–N (signal-fraction sensitivity)
# ======================================================================
#
# These panels characterize the statistical calibration and signal
# detection of sctrial against established multi-subject methods
# (dreamlet, NEBULA, Wilcoxon on change scores) across a 4 x 5 grid
# of gene-panel sizes and signal fractions.
#
# Data source: hierarchical gamma-Poisson simulator calibrated from TNBC
# (Phase 5 sensitivity grid, n = 11M per-gene test rows).

_BENCHMARK_CSV = (
    Path(__file__).resolve().parents[4]
    / "manuscript"
    / "benchmark"
    / "sensitivity"
    / "sensitivity_combined.csv"
)

# Method display configuration — plot order: sctrial last so it renders
# on top of overlapping curves and stays visible.
_BENCH_METHODS = ["wilcoxon_paired", "nebula", "dreamlet", "sctrial_did"]
_BENCH_METHOD_LABELS = {
    "sctrial_did": "sctrial (DiD)",
    "dreamlet": "dreamlet",
    "nebula": "NEBULA",
    "wilcoxon_paired": "Wilcoxon (Δ scores)",
}
_BENCH_METHOD_COLORS = {
    "sctrial_did": "#1f77b4",        # steel blue
    "dreamlet": "#d62728",           # brick red
    "nebula": "#ff7f0e",             # safety orange
    "wilcoxon_paired": "#2ca02c",    # forest green
}
_BENCH_METHOD_MARKERS = {
    "sctrial_did": "o",
    "dreamlet": "D",
    "nebula": "s",
    "wilcoxon_paired": "^",
}

_PANEL_SIZES = [50, 200, 500, 2000]
_SIGNAL_FRACTIONS = [1, 5, 10, 20]  # percent; pure null (0) handled separately


def _load_benchmark_data():
    """Load signal-fraction sensitivity results from HPC output.

    Returns a DataFrame with columns added for panel size (n_genes) and
    signal fraction (signal_pct, 0 for pure-null scenarios).
    """
    if not _BENCHMARK_CSV.exists():
        raise FileNotFoundError(
            f"Benchmark results not found at {_BENCHMARK_CSV}.\n"
            "Run the signal-fraction sensitivity benchmark on HPC first, "
            "then rsync results locally."
        )
    df = pd.read_csv(_BENCHMARK_CSV, low_memory=False)
    df["n_genes"] = df["scenario"].str.extract(r"_g(\d+)")[0].astype(int)
    frac = df["scenario"].str.extract(r"_f(\d+)")
    df["signal_pct"] = pd.to_numeric(frac[0], errors="coerce").fillna(0).astype(int)
    df["is_null_scenario"] = df["scenario"].str.contains("sens_null")
    return df


def _method_style(method, is_focal=False, alpha=1.0):
    """Return standardized styling kwargs for a method's line/marker.

    Focal method (sctrial) gets slightly heavier styling for visibility.
    """
    return {
        "color": _BENCH_METHOD_COLORS[method],
        "marker": _BENCH_METHOD_MARKERS[method],
        "markersize": 9 if is_focal else 7,
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
        "linewidth": 2.5 if is_focal else 1.8,
        "alpha": alpha,
    }


def _add_nominal_band(ax, level=0.05, low=0.03, high=0.07, color="#d62728"):
    """Overlay a subtle reference band at the nominal α level."""
    ax.axhspan(low, high, color=color, alpha=0.06, zorder=0)
    ax.axhline(level, color=color, linestyle="--", linewidth=1.0,
               alpha=0.65, zorder=1)


def _style_axis(ax):
    """Apply consistent publication styling to an axis."""
    ax.grid(axis="y", linestyle=":", color="#b0b0b0", alpha=0.45,
            linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major",
                   color="#333333", width=0.8, length=4)


# ----------------------------------------------------------------------
# Summary tables (computed once, reused across panels)
# ----------------------------------------------------------------------

def _compute_null_fpr_table(bench_df):
    """Per (method, panel_size, signal_pct) null-gene FPR.

    Null genes are those with ``true_beta == 0``.  The FPR is the fraction
    of valid (non-NaN) p-values below 0.05.  Returns a long DataFrame with
    columns: method, n_genes, signal_pct, fpr, n_tests.
    """
    null = bench_df[bench_df["true_beta"] == 0.0].copy()
    rows = []
    for (method, n_g, frac), grp in null.groupby(
        ["method", "n_genes", "signal_pct"]
    ):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({
            "method": method,
            "n_genes": int(n_g),
            "signal_pct": int(frac),
            "fpr": float((pvals < 0.05).mean()),
            "n_tests": int(len(pvals)),
        })
    return pd.DataFrame(rows)


def _compute_signal_power_table(bench_df):
    """Per (method, panel_size, signal_pct) power on signal genes.

    Signal genes are those with ``true_beta != 0``.  Power is the fraction
    of valid p-values below 0.05.  Pure-null scenarios are excluded.
    """
    mixed = bench_df[~bench_df["is_null_scenario"]].copy()
    sig = mixed[mixed["true_beta"] != 0.0]
    rows = []
    for (method, n_g, frac), grp in sig.groupby(
        ["method", "n_genes", "signal_pct"]
    ):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        rows.append({
            "method": method,
            "n_genes": int(n_g),
            "signal_pct": int(frac),
            "power": float((pvals < 0.05).mean()),
            "n_tests": int(len(pvals)),
        })
    return pd.DataFrame(rows)


def _compute_signal_bias_rmse_table(bench_df):
    """Per (method, panel_size, signal_pct) bias & RMSE of β̂ on signal genes.

    Signal genes are those with ``true_beta != 0``.  RMSE and mean bias
    are computed on rows with valid (non-NaN) ``estimated_beta``.
    """
    mixed = bench_df[~bench_df["is_null_scenario"]].copy()
    sig = mixed[mixed["true_beta"] != 0.0]
    sig = sig.dropna(subset=["estimated_beta"]).copy()
    sig["err"] = sig["estimated_beta"] - sig["true_beta"]
    sig["sq_err"] = sig["err"] ** 2

    rows = []
    for (method, n_g, frac), grp in sig.groupby(
        ["method", "n_genes", "signal_pct"]
    ):
        if grp.empty:
            continue
        rows.append({
            "method": method,
            "n_genes": int(n_g),
            "signal_pct": int(frac),
            "bias": float(grp["err"].mean()),
            "rmse": float(np.sqrt(grp["sq_err"].mean())),
            "n_tests": int(len(grp)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Panel H — FPR curves faceted by panel size
# ----------------------------------------------------------------------

def _panel_bench_fpr_curves(fig, bench_df):
    """Null-gene FPR as a function of signal fraction, faceted by panel size.

    4 small multiples (one per panel size).  One line per method.  The
    red dashed reference at α = 0.05 marks the nominal level.  Shared
    y-axis so cross-panel comparisons are visually straightforward.

    sctrial, Wilcoxon, and NEBULA all cluster near the nominal 5%, so a
    tiny horizontal offset per method is applied to prevent the lines
    from literally overlapping and hiding each other.
    """
    fpr_df = _compute_null_fpr_table(bench_df)
    fpr_df = fpr_df[fpr_df["signal_pct"] > 0].copy()

    axes = fig.subplots(1, 4, sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    # Small x-offsets (% of signal fraction) to separate overlapping lines
    method_offsets = {
        "wilcoxon_paired": -0.30,
        "nebula":          -0.10,
        "sctrial_did":     +0.10,
        "dreamlet":        +0.30,
    }

    for ax_idx, (ax, n_g) in enumerate(zip(axes, _PANEL_SIZES)):
        sub = fpr_df[fpr_df["n_genes"] == n_g]
        for method in _BENCH_METHODS:
            m = sub[sub["method"] == method].sort_values("signal_pct")
            if m.empty:
                continue
            is_focal = method == "sctrial_did"
            style = _method_style(method, is_focal=is_focal)
            x = m["signal_pct"].values + method_offsets[method]
            ax.plot(
                x, m["fpr"],
                label=_BENCH_METHOD_LABELS[method] if ax_idx == 0 else None,
                zorder=10 if is_focal else 3,
                **style,
            )

        _add_nominal_band(ax)
        ax.set_xticks(_SIGNAL_FRACTIONS)
        ax.set_xticklabels([f"{f}%" for f in _SIGNAL_FRACTIONS])
        ax.set_xlabel("Signal fraction")
        ax.set_title(
            f"{n_g:,} genes",
            fontsize=12, fontweight="bold", color="#222222", pad=8,
        )
        ax.set_ylim(-0.02, 0.72)
        _style_axis(ax)

    axes[0].set_ylabel("Null-gene FPR (p < 0.05)", fontsize=11)
    axes[0].legend(
        loc="upper left", frameon=True, framealpha=0.95,
        edgecolor="#cccccc", fontsize=8,
    )


# ----------------------------------------------------------------------
# Panel I — FPR heatmap across method × (panel size × signal fraction)
# ----------------------------------------------------------------------

def _panel_bench_fpr_heatmap(fig, bench_df):
    """Null-gene FPR heatmap — one subplot per method.

    Rows = signal fraction (ascending bottom→top), columns = panel size.
    Cells annotated with the FPR value; annotation colour flips from
    dark→white as the cell intensity crosses a threshold so values stay
    readable against any background.
    """
    fpr_df = _compute_null_fpr_table(bench_df)
    mixed = fpr_df[fpr_df["signal_pct"] > 0].copy()

    n_methods = len(_BENCH_METHODS)
    # No sharey: each subplot needs its own y-tick labels for readability
    axes = fig.subplots(1, n_methods, sharey=False)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    fractions = sorted(_SIGNAL_FRACTIONS)
    panel_sizes = sorted(_PANEL_SIZES)
    cmap = plt.cm.RdYlGn_r

    im = None
    for mi, (ax, method) in enumerate(zip(axes, _BENCH_METHODS)):
        mat = np.full((len(fractions), len(panel_sizes)), np.nan)
        for fi, frac in enumerate(fractions):
            for pi, ps in enumerate(panel_sizes):
                cell = mixed[
                    (mixed["method"] == method)
                    & (mixed["signal_pct"] == frac)
                    & (mixed["n_genes"] == ps)
                ]
                if len(cell) == 1:
                    mat[fi, pi] = cell["fpr"].iloc[0]

        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=0.7,
            origin="lower",
            interpolation="nearest",
        )

        for fi in range(len(fractions)):
            for pi in range(len(panel_sizes)):
                v = mat[fi, pi]
                if np.isnan(v):
                    continue
                text_color = "white" if (v > 0.35 or v < 0.02) else "#1a1a1a"
                ax.text(
                    pi, fi, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold",
                )

        ax.set_xticks(range(len(panel_sizes)))
        ax.set_xticklabels([f"{p:,}" for p in panel_sizes], fontsize=9)
        ax.set_xlabel("Panel size (genes)", fontsize=10)
        ax.set_yticks(range(len(fractions)))
        ax.set_yticklabels([f"{f}%" for f in fractions], fontsize=9)
        if mi == 0:
            ax.set_ylabel("Signal fraction", fontsize=11, fontweight="bold")
        ax.set_title(
            _BENCH_METHOD_LABELS[method],
            fontsize=12, fontweight="bold",
            color=_BENCH_METHOD_COLORS[method], pad=8,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(0.9)
        ax.tick_params(axis="both", which="major", length=0)

    cbar = fig.colorbar(
        im, ax=axes, shrink=0.78, pad=0.015, aspect=20,
    )
    cbar.set_label(
        "Null-gene FPR (p < 0.05)",
        fontsize=10, rotation=270, labelpad=16,
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("#333333")


# ----------------------------------------------------------------------
# Panel J — Pure-null λ_GC across panel sizes
# ----------------------------------------------------------------------

def _panel_bench_lambda_gc(ax, bench_df):
    """Genomic inflation factor (λ_GC) under pure-null conditions."""
    null_scenarios = bench_df[bench_df["is_null_scenario"]]
    pvals_pure = null_scenarios[null_scenarios["true_beta"] == 0.0]

    rows = []
    for (method, n_g), grp in pvals_pure.groupby(["method", "n_genes"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) < 50:
            continue
        chi2_obs = sp_stats.chi2.isf(pvals, df=1)
        chi2_obs = chi2_obs[np.isfinite(chi2_obs)]
        if len(chi2_obs) < 50:
            continue
        lam = float(np.median(chi2_obs) / sp_stats.chi2.ppf(0.5, df=1))
        rows.append({"method": method, "n_genes": int(n_g), "lambda_gc": lam})
    lam_df = pd.DataFrame(rows)

    for method in _BENCH_METHODS:
        sub = lam_df[lam_df["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_focal = method == "sctrial_did"
        style = _method_style(method, is_focal=is_focal)
        ax.plot(
            sub["n_genes"], sub["lambda_gc"],
            label=_BENCH_METHOD_LABELS[method],
            zorder=10 if is_focal else 3,
            **style,
        )

    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0,
               alpha=0.65, zorder=1)
    ax.axhspan(0.95, 1.05, color="#d62728", alpha=0.06, zorder=0)

    ax.set_xscale("log")
    ax.set_xticks(_PANEL_SIZES)
    ax.set_xticklabels([f"{p:,}" for p in _PANEL_SIZES])
    ax.set_xlabel("Panel size (genes)", fontsize=11)
    ax.set_ylabel(r"Genomic inflation factor ($\lambda_{\mathrm{GC}}$)",
                  fontsize=11)
    ax.set_title("Pure-null calibration across panel sizes",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0.88, 1.18)
    ax.legend(
        loc="upper left", frameon=True, framealpha=0.95,
        edgecolor="#cccccc", fontsize=9,
    )
    _style_axis(ax)


# ----------------------------------------------------------------------
# Panel K — QQ plots at a representative challenging condition
# ----------------------------------------------------------------------

def _panel_bench_qq(fig, bench_df, n_genes=200, signal_pct=10):
    """Null-gene QQ plots at a representative mixed-signal condition."""
    scenario_name = f"two_arm__sens_g{n_genes}_f{signal_pct}"
    sub_all = bench_df[bench_df["scenario"] == scenario_name]
    if sub_all.empty:
        print(f"    WARNING: scenario {scenario_name} not found for panel K")
        return
    null = sub_all[sub_all["true_beta"] == 0.0]

    axes = fig.subplots(1, len(_BENCH_METHODS), sharex=True, sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for mi, (ax, method) in enumerate(zip(axes, _BENCH_METHODS)):
        pvals = (
            null.loc[null["method"] == method, "pvalue"]
            .dropna()
            .sort_values()
            .values
        )
        if len(pvals) == 0:
            continue
        n = len(pvals)
        ranks = np.arange(1, n + 1)
        expected = (ranks - 0.5) / n
        obs_log = -np.log10(pvals + 1e-300)
        exp_log = -np.log10(expected + 1e-300)

        lo_env = -np.log10(
            sp_stats.beta.ppf(0.975, ranks, n - ranks + 1) + 1e-300
        )
        hi_env = -np.log10(
            sp_stats.beta.ppf(0.025, ranks, n - ranks + 1) + 1e-300
        )
        ax.fill_between(exp_log, lo_env, hi_env,
                        color="#b0b0b0", alpha=0.22, zorder=1,
                        label="95% Beta envelope" if mi == 0 else None)

        ax.scatter(
            exp_log, obs_log,
            s=8, alpha=0.55,
            color=_BENCH_METHOD_COLORS[method],
            edgecolors="none", rasterized=True, zorder=3,
        )

        lim = max(exp_log.max(), obs_log.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#333333",
                linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)

        ax.set_title(
            _BENCH_METHOD_LABELS[method],
            fontsize=12, fontweight="bold",
            color=_BENCH_METHOD_COLORS[method], pad=8,
        )
        ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=10)
        if mi == 0:
            ax.set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=10)
        _style_axis(ax)

    fig.suptitle(
        f"Null-gene p-value calibration at {n_genes:,} genes, "
        f"{signal_pct}% signal",
        fontsize=13, fontweight="bold", y=1.03,
    )
    axes[0].legend(
        loc="upper left", frameon=True, framealpha=0.95,
        edgecolor="#cccccc", fontsize=8,
    )


# ----------------------------------------------------------------------
# Panel L — Pure-null FPR dot-and-whisker
# ----------------------------------------------------------------------

def _panel_bench_pure_null_fpr(ax, bench_df):
    """Per-scenario empirical Type I error under pure-null conditions.

    x-axis: panel size (log scale).  y-axis: observed FPR.  One line per
    method, connecting the four panel sizes.  95% Wilson CIs are drawn
    as vertical whiskers at each point.  The nominal 5% band is shown
    in a faint red strip.
    """
    null = bench_df[
        (bench_df["is_null_scenario"]) & (bench_df["true_beta"] == 0.0)
    ]

    rows = []
    for (method, n_g), grp in null.groupby(["method", "n_genes"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        k = int((pvals < 0.05).sum())
        n = len(pvals)
        p = k / n
        ci = sp_stats.binomtest(k, n, p=0.05).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        rows.append({
            "method": method, "n_genes": int(n_g),
            "fpr": p, "ci_lo": ci.low, "ci_hi": ci.high,
        })
    df = pd.DataFrame(rows)

    # Nominal 5% band and reference line
    ax.axhspan(0.03, 0.07, color="#d62728", alpha=0.08, zorder=0,
               label="Nominal 5% ± 2%")
    ax.axhline(0.05, color="#d62728", linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=1)

    panel_sizes = sorted(_PANEL_SIZES)
    x_vals = np.array(panel_sizes, dtype=float)
    # Tiny log-scale offsets so error-bars don't perfectly overlap
    log_offsets = {
        "wilcoxon_paired": 0.93,
        "nebula":          0.97,
        "sctrial_did":     1.03,
        "dreamlet":        1.07,
    }

    for method in _BENCH_METHODS:
        sub = df[df["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        xs = sub["n_genes"].values * log_offsets[method]
        ys = sub["fpr"].values
        lo = sub["ci_lo"].values
        hi = sub["ci_hi"].values
        is_focal = method == "sctrial_did"

        ax.errorbar(
            xs, ys,
            yerr=[ys - lo, hi - ys],
            fmt=_BENCH_METHOD_MARKERS[method],
            markersize=10 if is_focal else 8,
            color=_BENCH_METHOD_COLORS[method],
            markerfacecolor=_BENCH_METHOD_COLORS[method],
            markeredgecolor="white", markeredgewidth=0.8,
            ecolor=_BENCH_METHOD_COLORS[method],
            elinewidth=1.4, capsize=4, capthick=1.2,
            linestyle="-", linewidth=2.0 if is_focal else 1.4,
            label=_BENCH_METHOD_LABELS[method],
            alpha=0.92, zorder=10 if is_focal else 4,
        )

    ax.set_xscale("log")
    ax.set_xticks(panel_sizes)
    ax.set_xticklabels([f"{p:,}" for p in panel_sizes])
    ax.set_xlabel("Panel size (genes)", fontsize=11)
    ax.set_ylabel("Pure-null Type I error (p < 0.05)", fontsize=11)
    ax.set_ylim(0.025, 0.085)
    ax.set_title(
        "All methods are calibrated under pure-null conditions",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(
        loc="upper left", frameon=True, framealpha=0.95,
        edgecolor="#cccccc", fontsize=8,
    )
    _style_axis(ax)


# ----------------------------------------------------------------------
# Panel M — Effect-size estimation accuracy on signal genes
# ----------------------------------------------------------------------

def _panel_bench_signal_rmse(fig, bench_df):
    """Effect-size estimation accuracy on signal genes, faceted by panel size.

    Top row: mean bias (β̂ − β, closer to 0 is better).  Dashed reference
    at 0 marks unbiased estimation.  Y-axis is set asymmetrically to
    accommodate dreamlet's large positive bias while still leaving room
    to see the tiny deviations of the other methods around zero.

    Bottom row: RMSE of β̂ (lower is better).  The y-axis spans 0 to the
    max observed RMSE so dreamlet's 3–5× higher error is visually obvious.

    Each bar is annotated with its numeric value so small differences
    between sctrial, Wilcoxon, and NEBULA remain readable even when
    dreamlet's bars dominate the y-scale.
    """
    df = _compute_signal_bias_rmse_table(bench_df)

    fig.set_constrained_layout(False)
    gs = fig.add_gridspec(
        2, 4,
        hspace=0.38, wspace=0.22,
        left=0.08, right=0.985, top=0.84, bottom=0.11,
    )

    bar_width = 0.20
    method_order = ["sctrial_did", "wilcoxon_paired", "nebula", "dreamlet"]
    x_positions = np.arange(len(_SIGNAL_FRACTIONS))

    # Asymmetric bias range: use actual min/max with small padding
    bias_lo = min(df["bias"].min(), 0) - 0.02
    bias_hi = max(df["bias"].max(), 0.02) * 1.12
    rmse_hi = df["rmse"].max() * 1.18  # extra room for top annotation

    bias_axes = []
    rmse_axes = []

    for col, n_g in enumerate(_PANEL_SIZES):
        ax_bias = fig.add_subplot(gs[0, col])
        ax_rmse = fig.add_subplot(gs[1, col])
        bias_axes.append(ax_bias)
        rmse_axes.append(ax_rmse)
        sub = df[df["n_genes"] == n_g]

        for mi, method in enumerate(method_order):
            bias_vals = []
            rmse_vals = []
            for frac in _SIGNAL_FRACTIONS:
                cell = sub[
                    (sub["method"] == method) & (sub["signal_pct"] == frac)
                ]
                if len(cell):
                    bias_vals.append(float(cell["bias"].iloc[0]))
                    rmse_vals.append(float(cell["rmse"].iloc[0]))
                else:
                    bias_vals.append(np.nan)
                    rmse_vals.append(np.nan)
            offset = (mi - (len(method_order) - 1) / 2) * bar_width
            ax_bias.bar(
                x_positions + offset, bias_vals, bar_width,
                color=_BENCH_METHOD_COLORS[method],
                edgecolor="white", linewidth=0.6,
                zorder=3,
            )
            ax_rmse.bar(
                x_positions + offset, rmse_vals, bar_width,
                color=_BENCH_METHOD_COLORS[method],
                edgecolor="white", linewidth=0.6,
                zorder=3,
            )

        # Top row — bias with 0 reference line
        ax_bias.axhline(
            0.0, color="#222222", linestyle="--", linewidth=0.9,
            alpha=0.7, zorder=2,
        )
        ax_bias.set_xticks(x_positions)
        ax_bias.set_xticklabels([])
        ax_bias.set_ylim(bias_lo, bias_hi)
        ax_bias.set_title(
            f"{n_g:,} genes",
            fontsize=12, fontweight="bold", color="#222222", pad=8,
        )
        _style_axis(ax_bias)

        # Bottom row — RMSE
        ax_rmse.set_xticks(x_positions)
        ax_rmse.set_xticklabels([f"{f}%" for f in _SIGNAL_FRACTIONS])
        ax_rmse.set_xlabel("Signal fraction")
        ax_rmse.set_ylim(0, rmse_hi)
        _style_axis(ax_rmse)

        if col > 0:
            ax_bias.set_yticklabels([])
            ax_rmse.set_yticklabels([])

    bias_axes[0].set_ylabel(
        r"Mean bias ($\hat{\beta} - \beta$)", fontsize=11,
    )
    rmse_axes[0].set_ylabel(
        r"RMSE of $\hat{\beta}$", fontsize=11,
    )

    # Figure-level legend above the subplots (no overlap with data)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=_BENCH_METHOD_COLORS[m],
                      edgecolor="white", linewidth=0.6,
                      label=_BENCH_METHOD_LABELS[m])
        for m in method_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center", ncol=4,
        bbox_to_anchor=(0.53, 0.94),
        frameon=True, framealpha=0.95,
        edgecolor="#cccccc", fontsize=9,
    )


# ----------------------------------------------------------------------
# Panel N — Runtime comparison (log scale)
# ----------------------------------------------------------------------

def _panel_bench_runtime(ax, bench_df):
    """Per-iteration runtime by method × panel size (log scale)."""
    rt = (
        bench_df.groupby(["method", "scenario", "n_genes", "iteration"])[
            "runtime_seconds"
        ]
        .first()
        .reset_index()
    )

    summary = (
        rt.groupby(["method", "n_genes"])["runtime_seconds"]
        .median()
        .reset_index()
    )

    for method in _BENCH_METHODS:
        sub = summary[summary["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_focal = method == "sctrial_did"
        style = _method_style(method, is_focal=is_focal)
        ax.plot(
            sub["n_genes"], sub["runtime_seconds"],
            label=_BENCH_METHOD_LABELS[method],
            zorder=10 if is_focal else 3,
            **style,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(_PANEL_SIZES)
    ax.set_xticklabels([f"{p:,}" for p in _PANEL_SIZES])
    ax.set_xlabel("Panel size (genes)", fontsize=11)
    ax.set_ylabel("Median runtime per iteration (s)", fontsize=11)
    ax.set_title("Computational cost",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(
        loc="upper left", frameon=True, framealpha=0.95,
        edgecolor="#cccccc", fontsize=9,
    )
    _style_axis(ax)


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
      --- Benchmark (4 methods: sctrial, dreamlet, NEBULA, Wilcoxon) ---
      --- across 4 panel sizes (50–2000) × 5 signal fractions (0–20%) ---
      H  Benchmark: null-gene FPR curves vs signal fraction, faceted by panel size
      I  Benchmark: null-gene FPR heatmap (method × panel size × signal fraction)
      J  Benchmark: pure-null λ_GC across panel sizes
      K  Benchmark: QQ plots at 200 genes / 10% signal (challenging regime)
      L  Benchmark: pure-null FPR dot-and-whisker (all panel sizes per method)
      M  Benchmark: signal-gene power faceted by panel size
      N  Benchmark: runtime scaling across panel sizes
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

    # ----------------------------------------------------------------
    # Panels H–N — Benchmark across panel sizes and signal fractions
    # ----------------------------------------------------------------
    print("  Loading signal-fraction sensitivity benchmark results ...")
    bench_df = _load_benchmark_data()
    print(
        f"    {len(bench_df):,} rows, "
        f"{bench_df.scenario.nunique()} scenarios, "
        f"panel sizes = {sorted(bench_df['n_genes'].unique())}"
    )

    # Panel H — Null-gene FPR curves faceted by panel size
    fig_h = plt.figure(figsize=(14, 4.2))
    _panel_bench_fpr_curves(fig_h, bench_df)
    fig_h.suptitle(
        "Null-gene FPR scales with signal fraction, not panel size",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig_h.tight_layout()
    save_panel(fig_h, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    # Panel I — FPR heatmap (method × panel size × signal fraction)
    fig_i = plt.figure(figsize=(15.5, 4.2))
    _panel_bench_fpr_heatmap(fig_i, bench_df)
    save_panel(fig_i, "panel_I", FIGURE_NAME, SUPP_OUTPUT)

    # Panel J — Pure-null λ_GC across panel sizes
    fig_j, ax_j = plt.subplots(figsize=(7.2, 5.0))
    _panel_bench_lambda_gc(ax_j, bench_df)
    fig_j.tight_layout()
    save_panel(fig_j, "panel_J", FIGURE_NAME, SUPP_OUTPUT)

    # Panel K — QQ plots at a representative challenging condition
    fig_k = plt.figure(figsize=(15.0, 4.0))
    _panel_bench_qq(fig_k, bench_df, n_genes=200, signal_pct=10)
    fig_k.tight_layout()
    save_panel(fig_k, "panel_K", FIGURE_NAME, SUPP_OUTPUT)

    # Panel L — Pure-null FPR dot-and-whisker (all 4 panel sizes)
    fig_l, ax_l = plt.subplots(figsize=(7.5, 4.8))
    _panel_bench_pure_null_fpr(ax_l, bench_df)
    fig_l.tight_layout()
    save_panel(fig_l, "panel_L", FIGURE_NAME, SUPP_OUTPUT)

    # Panel M — Effect-size estimation accuracy on signal genes
    fig_m = plt.figure(figsize=(14, 6.8))
    _panel_bench_signal_rmse(fig_m, bench_df)
    fig_m.suptitle(
        "Effect-size estimation accuracy on signal genes",
        fontsize=13, fontweight="bold", y=0.995,
    )
    save_panel(fig_m, "panel_M", FIGURE_NAME, SUPP_OUTPUT)

    # Panel N — Runtime scaling by method × panel size
    fig_n, ax_n = plt.subplots(figsize=(7.2, 5.0))
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
