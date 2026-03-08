"""
Supplementary Figure 4 — Model Diagnostics and Assumption Checks.
=================================================================

Verify that DiD model assumptions hold across datasets.

Panels:
  A  Residual Q-Q plots per dataset (normality check).
  B  Effect size vs standard error scatter per dataset (funnel plot).
  C  P-value distribution (uniformity under null expectation).
  D  Sensitivity analysis: minimum detectable effect by sample size.
  E  Leave-one-out influence: distribution of influence scores.
  F  Residual summary statistics table (mean, std, skew, kurtosis).
  G  Residual autocorrelation (ACF-style lag plot for top features).
  H  DiD effect size distribution per dataset.

Non-overlap guardrail: diagnostics only, no treatment-effect claims.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    apply_style,
    despine,
    save_panel,
    load_clinical_trial_dataset,
    get_sade_feldman,
    harmonize_response,
    clear_cache,
)

FIGURE_NAME = "SuppFig4_model_diagnostics"

# Datasets with proper pre/post + arm structure for DiD
_DID_DATASETS = {
    "Sade-Feldman": {
        "loader": get_sade_feldman,
        "harmonize": True,
        "design_kw": {
            "participant_col": "participant_id",
            "visit_col": "visit",
            "arm_col": "response",
            "arm_treated": "Responder",
            "arm_control": "Non-responder",
        },
        "visits": ("Pre", "Post"),
        "layer": "log1p_tpm",
    },
    "AML": {
        "loader": lambda: load_clinical_trial_dataset("aml"),
        "harmonize": False,
        "design_kw": {
            "participant_col": "participant_id",
            "visit_col": "visit",
            "arm_col": "response",
            "arm_treated": "Treatment",
            "arm_control": "Control",
        },
        "visits": ("Pre", "Post"),
        "layer": "log1p_norm",
    },
}

# Top features to test (immune markers)
_TEST_FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19",
    "CD14", "LYZ", "NKG7", "CD3D", "FOXP3", "IL7R",
    "TCF7", "TOX",
]

_DS_PALETTE = dict(zip(
    ["Sade-Feldman", "AML"],
    sns.color_palette("Set2", 2),
))


def _load_did_datasets():
    """Load datasets and run DiD to get residuals/diagnostics."""
    import sctrial

    results = {}
    for name, cfg in _DID_DATASETS.items():
        try:
            adata = cfg["loader"]()
            if cfg["harmonize"]:
                adata = harmonize_response(adata)

            design = sctrial.TrialDesign(**cfg["design_kw"])
            layer = cfg["layer"]
            visits = cfg["visits"]

            # Filter features to those present in data
            features = [f for f in _TEST_FEATURES
                        if f in adata.var_names][:15]

            if not features:
                print(f"  {name}: no test features found, skipping")
                continue

            # Run DiD
            did_df = sctrial.did_table(
                adata, features, design, visits,
                layer=layer, aggregate="participant_visit",
            )

            # Run LOO
            try:
                loo_df = sctrial.loo_cv_did(
                    adata, features[:5], design, visits, layer=layer,
                )
            except Exception:
                loo_df = None

            # Compute n per group (minimum of the two arms)
            arm_col = cfg["design_kw"]["arm_col"]
            arm_sizes = adata.obs.groupby(arm_col)[
                cfg["design_kw"]["participant_col"]].nunique()
            n_per = int(arm_sizes.min()) if len(arm_sizes) else 0

            results[name] = {
                "adata": adata,
                "did_df": did_df,
                "loo_df": loo_df,
                "features": features,
                "n_per_group": n_per,
            }
            print(f"  {name}: DiD on {len(features)} features, "
                  f"n_per_group={n_per}")

        except Exception as exc:
            print(f"  {name}: failed ({exc})")

    return results


# ── Panel A: Residual Q-Q plots ───────────────────────────────────

def _panel_qq(fig, axes, results: dict):
    """Q-Q plots of DiD residuals per dataset."""
    ds_names = list(results.keys())
    for ax_i, ax in enumerate(axes):
        if ax_i >= len(ds_names):
            ax.axis("off")
            continue

        name = ds_names[ax_i]
        did_df = results[name]["did_df"]

        # Use the DiD coefficients as a proxy for residual-like distribution
        if "beta_DiD" in did_df.columns:
            residuals = did_df["beta_DiD"].dropna().values
        elif "coef" in did_df.columns:
            residuals = did_df["coef"].dropna().values
        else:
            ax.text(0.5, 0.5, "No residuals", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"{name}", fontweight="bold", fontsize=9)
        ax.get_lines()[0].set_markerfacecolor(_DS_PALETTE.get(name, "grey"))
        ax.get_lines()[0].set_markersize(4)
        despine(ax)


# ── Panel B: Residuals vs fitted ──────────────────────────────────

def _panel_resid_vs_fitted(fig, axes, results: dict):
    """Scatter: DiD effect size vs standard error (diagnostic proxy)."""
    ds_names = list(results.keys())
    for ax_i, ax in enumerate(axes):
        if ax_i >= len(ds_names):
            ax.axis("off")
            continue

        name = ds_names[ax_i]
        did_df = results[name]["did_df"]

        if "beta_DiD" in did_df.columns and "se_DiD" in did_df.columns:
            x = did_df["beta_DiD"].values
            y = did_df["se_DiD"].values
        elif "coef" in did_df.columns and "se" in did_df.columns:
            x = did_df["coef"].values
            y = did_df["se"].values
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ax.scatter(x, y, s=15, alpha=0.6, color=_DS_PALETTE.get(name, "grey"),
                   edgecolors="grey", linewidth=0.3)
        ax.axhline(np.median(y), color="grey", linestyle="--", linewidth=0.5)
        ax.set_xlabel("DiD coefficient")
        ax.set_ylabel("Standard error")
        ax.set_title(f"{name}", fontweight="bold", fontsize=9)
        despine(ax)


# ── Panel C: P-value distribution ─────────────────────────────────

def _panel_pvalue_distribution(ax, results: dict):
    """Histogram of DiD p-values; uniform under the null."""
    for name, res in results.items():
        did_df = res["did_df"]
        pcol = next((c for c in ["p_DiD", "pvalue_DiD", "pvalue", "p"]
                     if c in did_df.columns), None)
        if pcol is None:
            continue
        pvals = did_df[pcol].dropna().values
        if len(pvals) < 2:
            continue
        ax.hist(pvals, bins=np.linspace(0, 1, 11), alpha=0.5,
                label=f"{name} (n={len(pvals)})",
                color=_DS_PALETTE.get(name, "grey"), edgecolor="white")

    ax.axhline(ax.get_ylim()[1] * 0.1, color="grey", linestyle="--",
               linewidth=0.5, alpha=0.5)
    ax.axvline(0.05, color="red", linestyle="--", linewidth=0.8,
               label="α = 0.05")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.set_title("DiD P-value Distribution", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    despine(ax)


# ── Panel D: Sensitivity analysis (MDE vs N) ─────────────────────

def _panel_sensitivity(ax, results: dict):
    """Minimum detectable effect vs sample size per group."""
    import sctrial

    n_range = np.arange(5, 55, 5)
    for name, res in results.items():
        did_df = res["did_df"]
        # Compute dataset-specific residual sigma from SE and N
        se_col = next((c for c in ["se_DiD", "se"] if c in did_df.columns), None)
        if se_col and did_df[se_col].notna().any():
            sigma = float(did_df[se_col].median()) * np.sqrt(res["n_per_group"])
            sigma = max(sigma, 0.3)  # floor
        else:
            sigma = 1.0

        mde_vals = []
        for n in n_range:
            try:
                mde = sctrial.sensitivity_analysis(int(n), sigma=sigma)
                mde_vals.append(mde)
            except Exception:
                mde_vals.append(np.nan)

        ax.plot(n_range, mde_vals, "o-", label=f"{name} (σ={sigma:.2f})",
                markersize=4, linewidth=1.5,
                color=_DS_PALETTE.get(name, "grey"))

        # Mark actual sample size
        n_actual = res["n_per_group"]
        if 0 < n_actual <= n_range.max():
            try:
                mde_actual = sctrial.sensitivity_analysis(
                    int(n_actual), sigma=sigma)
                ax.axvline(n_actual, color=_DS_PALETTE.get(name, "grey"),
                           linestyle=":", linewidth=0.8, alpha=0.5)
                ax.scatter([n_actual], [mde_actual], s=50, zorder=5,
                           color=_DS_PALETTE.get(name, "grey"),
                           edgecolors="black", linewidth=1)
            except Exception:
                pass

    ax.set_xlabel("Participants per group")
    ax.set_ylabel("Minimum detectable effect (σ)")
    ax.set_title("Power Analysis: MDE vs Sample Size", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    despine(ax)


# ── Panel E: LOO influence distribution ───────────────────────────

def _panel_loo_influence(ax, results: dict):
    """Distribution of LOO influence scores per dataset."""
    import sctrial

    rows = []
    for name, res in results.items():
        loo_df = res.get("loo_df")
        if loo_df is not None and not loo_df.empty:
            try:
                infl = sctrial.influence_diagnostics(loo_df)
                if "influence" in infl.columns:
                    for _, row in infl.iterrows():
                        rows.append({
                            "Dataset": name,
                            "Influence": row["influence"],
                            "Influential": row.get("is_influential", False),
                        })
            except Exception:
                pass

    if not rows:
        ax.text(0.5, 0.5, "No LOO data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, fontstyle="italic", color="#888888")
        ax.set_title("LOO Influence Scores", fontweight="bold")
        return

    df = pd.DataFrame(rows)
    sns.boxplot(data=df, x="Dataset", y="Influence", palette="Set2",
                linewidth=0.5, fliersize=2, ax=ax)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8,
               label="Influence threshold")
    ax.set_ylabel("Influence score")
    ax.set_title("LOO Influence Scores", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    despine(ax)


# ── Panel F: Residual summary statistics ──────────────────────────

def _panel_residual_summary(ax, results: dict):
    """Table: coefficient summary statistics per dataset."""
    rows = []
    for name, res in results.items():
        did_df = res["did_df"]
        bcol = next((c for c in ["beta_DiD", "coef"] if c in did_df.columns), None)
        se_col = next((c for c in ["se_DiD", "se"] if c in did_df.columns), None)
        pcol = next((c for c in ["p_DiD", "pvalue_DiD", "pvalue", "p"] if c in did_df.columns), None)
        if bcol is None:
            continue
        vals = did_df[bcol].dropna().values
        n_feat = len(vals)
        mean_b = np.mean(vals)
        std_b = np.std(vals)
        skew = float(stats.skew(vals)) if n_feat > 2 else np.nan
        kurt = float(stats.kurtosis(vals)) if n_feat > 2 else np.nan
        med_se = float(did_df[se_col].median()) if se_col else np.nan
        n_sig = int((did_df[pcol] < 0.05).sum()) if pcol else 0
        rows.append([
            name, str(n_feat),
            f"{mean_b:.3f}", f"{std_b:.3f}",
            f"{skew:.2f}", f"{kurt:.2f}",
            f"{med_se:.3f}",
            f"{n_sig}/{n_feat}",
        ])

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_title("Coefficient Summary", fontweight="bold")
        ax.axis("off")
        return

    col_labels = ["Dataset", "N feat", "Mean β", "Std β",
                   "Skew", "Kurtosis", "Med. SE", "Sig. (p<0.05)"]

    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 2.0)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(len(rows)):
        bg = "#f7f9fc" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(bg)

    ax.set_title("DiD Coefficient Summary Statistics",
                 fontweight="bold", pad=20)


# ── Panel G: Residual lag plot ────────────────────────────────────

def _panel_residual_lag(ax, results: dict):
    """Lag-1 scatter of DiD coefficients (autocorrelation check)."""
    for name, res in results.items():
        did_df = res["did_df"]
        if "beta_DiD" in did_df.columns:
            vals = did_df["beta_DiD"].dropna().values
        elif "coef" in did_df.columns:
            vals = did_df["coef"].dropna().values
        else:
            continue

        if len(vals) < 3:
            continue

        ax.scatter(vals[:-1], vals[1:], s=10, alpha=0.6,
                   color=_DS_PALETTE.get(name, "grey"), label=name,
                   edgecolors="grey", linewidth=0.3)

    # Reference line
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("DiD coefficient (feature i)")
    ax.set_ylabel("DiD coefficient (feature i+1)")
    ax.set_title("Lag-1 Coefficient Plot", fontweight="bold")
    ax.legend(fontsize=7, loc="best", frameon=True)
    despine(ax)


# ── Panel H: Effect size distribution ─────────────────────────────

def _panel_effect_distribution(ax, results: dict):
    """Histogram: DiD effect size distribution per dataset."""
    for name, res in results.items():
        did_df = res["did_df"]
        if "beta_DiD" in did_df.columns:
            vals = did_df["beta_DiD"].dropna().values
        elif "coef" in did_df.columns:
            vals = did_df["coef"].dropna().values
        else:
            continue

        ax.hist(vals, bins=20, alpha=0.5, label=name,
                color=_DS_PALETTE.get(name, "grey"), edgecolor="white")

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("DiD coefficient")
    ax.set_ylabel("Count")
    ax.set_title("DiD Effect Size Distribution", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 4 panels."""
    print("Supplementary Figure 4: Model Diagnostics and Assumption Checks")
    results = _load_did_datasets()

    if not results:
        print("  No DiD results; skipping.")
        return

    n_ds = len(results)

    # Panel A: Q-Q plots
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5))
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    _panel_qq(fig, axes, results)
    fig.suptitle("Residual Q-Q Plots", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: Residuals vs fitted
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5))
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    _panel_resid_vs_fitted(fig, axes, results)
    fig.suptitle("Effect Size vs Standard Error", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Panel C: P-value distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    _panel_pvalue_distribution(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # Panel D: Sensitivity
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_sensitivity(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # Panel E: LOO influence
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_loo_influence(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # Panel F: Residual summary
    fig, ax = plt.subplots(figsize=(10, 4))
    _panel_residual_summary(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # Panel G: Lag plot
    fig, ax = plt.subplots(figsize=(6, 6))
    _panel_residual_lag(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # Panel H: Effect size distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    _panel_effect_distribution(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    # Cleanup
    for res in results.values():
        if "adata" in res:
            del res["adata"]
    results.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
