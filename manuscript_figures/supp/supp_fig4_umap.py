"""
Supplementary Figure 4 — Model Diagnostics and Assumption Checks.
=================================================================

Verify that DiD model assumptions hold across datasets, plus
outcome-correlation panels from Sade-Feldman.

Panels:
  A  Coefficient Q-Q plots per dataset (normality check).
  B  Effect size vs standard error scatter per dataset (funnel plot).
  C  P-value distribution (uniformity under null expectation).
  D  Sensitivity analysis: minimum detectable effect by sample size.
  E  Leave-one-out influence: distribution of influence scores.
  F  Residual summary statistics table (mean, std, skew, kurtosis).
  G  Residual autocorrelation (ACF-style lag plot for top features).
  H  DiD effect size distribution per dataset.
  I  Signature changes: grouped bars (Responders vs Non-responders).
  J  Cohen's d forest plot with direction colouring and FDR-scaled marker.
  K  AUC horizontal bar chart.

Non-overlap guardrail: diagnostics only, no treatment-effect claims.
"""

from __future__ import annotations

import gc

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    apply_style,
    clear_cache,
    despine,
    get_sade_feldman,
    harmonize_response,
    load_clinical_trial_dataset,
    save_panel,
    score_signatures,
    sig_display,
)

FIGURE_NAME = "SuppFig4_diagnostics_robustness"

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
    # CAR-T excluded: single-arm study (arm_treated == arm_control)
    # is not a valid two-arm DiD design. Within-arm CAR-T comparisons
    # are shown in the temporal dynamics panels (SF6).
}

# Top features to test (immune markers)
_TEST_FEATURES = [
    "CD8A",
    "CD4",
    "PDCD1",
    "HAVCR2",
    "LAG3",
    "CTLA4",
    "GZMB",
    "PRF1",
    "IFNG",
    "TNF",
    "IL2",
    "CD19",
    "CD14",
    "LYZ",
    "NKG7",
    "CD3D",
    "FOXP3",
    "IL7R",
    "TCF7",
    "TOX",
]

_DS_PALETTE = dict(
    zip(
        ["Sade-Feldman", "AML", "CAR-T", "Melanoma"],
        sns.color_palette("Set2", 4),
    )
)


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
            features = [f for f in _TEST_FEATURES if f in adata.var_names][:15]

            if not features:
                print(f"  {name}: no test features found, skipping")
                continue

            # Run DiD
            did_df = sctrial.did_table(
                adata,
                features,
                design,
                visits,
                layer=layer,
                aggregate="participant_visit",
            )

            # Run LOO
            try:
                loo_df = sctrial.loo_cv_did(
                    adata,
                    features[:5],
                    design,
                    visits,
                    layer=layer,
                )
            except Exception:
                loo_df = None

            # Compute n per group (minimum of the two arms)
            arm_col = cfg["design_kw"]["arm_col"]
            arm_sizes = adata.obs.groupby(arm_col)[cfg["design_kw"]["participant_col"]].nunique()
            n_per = int(arm_sizes.min()) if len(arm_sizes) else 0

            results[name] = {
                "adata": adata,
                "did_df": did_df,
                "loo_df": loo_df,
                "features": features,
                "n_per_group": n_per,
            }
            print(f"  {name}: DiD on {len(features)} features, n_per_group={n_per}")

        except Exception as exc:
            print(f"  {name}: failed ({exc})")

    return results


# ── Panel A: Coefficient Q-Q plots ────────────────────────────────


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
            ax.text(0.5, 0.5, "No residuals", ha="center", va="center", transform=ax.transAxes)
            continue

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"{name}", fontweight="bold", fontsize=9)
        # Bold markers: larger size, full-opacity face colour
        line = ax.get_lines()[0]
        line.set_markerfacecolor(_DS_PALETTE.get(name, "grey"))
        line.set_markeredgecolor("black")
        line.set_markeredgewidth(0.5)
        line.set_markersize(8)
        line.set_alpha(0.9)
        # Make the reference line bolder too
        ref_line = ax.get_lines()[1]
        ref_line.set_linewidth(2.0)
        ref_line.set_color("#333333")
        despine(ax)


# ── Panel B: Effect size vs SE (funnel) ──────────────────────────


def _panel_funnel(fig, axes, results: dict):
    """Scatter: DiD effect size vs standard error (funnel plot)."""
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
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.scatter(
            x,
            y,
            s=50,
            alpha=0.85,
            color=_DS_PALETTE.get(name, "grey"),
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.axhline(np.median(y), color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("DiD coefficient")
        ax.set_ylabel("Standard error")
        ax.set_title(f"{name}", fontweight="bold", fontsize=9)
        despine(ax)


# ── Panel C: P-value distribution ─────────────────────────────────


def _panel_pvalue_distribution(ax, results: dict):
    """Histogram of DiD p-values; uniform under the null."""
    for name, res in results.items():
        did_df = res["did_df"]
        pcol = next(
            (c for c in ["p_DiD", "pvalue_DiD", "pvalue", "p"] if c in did_df.columns), None
        )
        if pcol is None:
            continue
        pvals = did_df[pcol].dropna().values
        if len(pvals) < 2:
            continue
        ax.hist(
            pvals,
            bins=np.linspace(0, 1, 11),
            alpha=0.6,
            label=f"{name} (n={len(pvals)})",
            color=_DS_PALETTE.get(name, "grey"),
            edgecolor="white",
            linewidth=0.8,
        )

    ax.axhline(ax.get_ylim()[1] * 0.1, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0.05, color="red", linestyle="--", linewidth=1.0, label="\u03b1 = 0.05")
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

        ax.plot(
            n_range,
            mde_vals,
            "o-",
            label=f"{name} (\u03c3={sigma:.2f})",
            markersize=8,
            linewidth=2.5,
            color=_DS_PALETTE.get(name, "grey"),
        )

        # Mark actual sample size
        n_actual = res["n_per_group"]
        if 0 < n_actual <= n_range.max():
            try:
                mde_actual = sctrial.sensitivity_analysis(int(n_actual), sigma=sigma)
                ax.axvline(
                    n_actual,
                    color=_DS_PALETTE.get(name, "grey"),
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.6,
                )
                ax.scatter(
                    [n_actual],
                    [mde_actual],
                    s=80,
                    zorder=5,
                    color=_DS_PALETTE.get(name, "grey"),
                    edgecolors="black",
                    linewidth=1.2,
                )
            except Exception:
                pass

    ax.set_xlabel("Participants per group")
    ax.set_ylabel("Minimum detectable effect (\u03c3)")
    ax.set_title("Power Analysis: MDE vs Sample Size", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    despine(ax)


# ── Panel E: LOO influence distribution (strip plot) ──────────────


def _panel_loo_influence(ax, results: dict):
    """Strip/swarm plot of LOO influence scores per dataset."""
    import sctrial

    rows = []
    for name, res in results.items():
        loo_df = res.get("loo_df")
        if loo_df is not None and not loo_df.empty:
            try:
                infl = sctrial.influence_diagnostics(loo_df)
                if "influence" in infl.columns:
                    for _, row in infl.iterrows():
                        rows.append(
                            {
                                "Dataset": name,
                                "Influence": row["influence"],
                                "Influential": row.get("is_influential", False),
                            }
                        )
            except Exception:
                pass

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No LOO data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            fontstyle="italic",
            color="#888888",
        )
        ax.set_title("LOO Influence Scores", fontweight="bold")
        despine(ax)
        return

    df = pd.DataFrame(rows)
    # Use strip plot for better visibility
    sns.stripplot(
        data=df,
        x="Dataset",
        y="Influence",
        palette=_DS_PALETTE,
        size=7,
        alpha=0.8,
        jitter=0.25,
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
        zorder=3,
    )
    # Overlay a boxplot outline for context
    sns.boxplot(
        data=df,
        x="Dataset",
        y="Influence",
        color="white",
        linewidth=1.2,
        fliersize=0,
        ax=ax,
        zorder=2,
        boxprops=dict(facecolor="none", edgecolor="#555555"),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
        medianprops=dict(color="red", linewidth=1.5),
    )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="Influence threshold")
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
        pcol = next(
            (c for c in ["p_DiD", "pvalue_DiD", "pvalue", "p"] if c in did_df.columns), None
        )
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
        rows.append(
            [
                name,
                str(n_feat),
                f"{mean_b:.3f}",
                f"{std_b:.3f}",
                f"{skew:.2f}",
                f"{kurt:.2f}",
                f"{med_se:.3f}",
                f"{n_sig}/{n_feat}",
            ]
        )

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="#888888",
        )
        ax.set_title("Coefficient Summary", fontweight="bold")
        ax.axis("off")
        return

    col_labels = [
        "Dataset",
        "N feat",
        "Mean \u03b2",
        "Std \u03b2",
        "Skew",
        "Kurtosis",
        "Med. SE",
        "Sig. (p<0.05)",
    ]

    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
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

    ax.set_title("DiD Coefficient Summary Statistics", fontweight="bold", pad=20)


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

        ax.scatter(
            vals[:-1],
            vals[1:],
            s=40,
            alpha=0.8,
            color=_DS_PALETTE.get(name, "grey"),
            label=name,
            edgecolors="black",
            linewidth=0.4,
            zorder=3,
        )

    # Reference line
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4)
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

        ax.hist(
            vals,
            bins=20,
            alpha=0.6,
            label=name,
            color=_DS_PALETTE.get(name, "grey"),
            edgecolor="white",
            linewidth=0.8,
        )

    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("DiD coefficient")
    ax.set_ylabel("Count")
    ax.set_title("DiD Effect Size Distribution", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    despine(ax)


# ======================================================================
# Outcome correlation panels (I, J, K) — from Sade-Feldman
# ======================================================================


def _prepare_outcome_data() -> dict:
    """Load Sade-Feldman, score signatures, compute participant-level
    Pre->Post changes, and derive response-correlation statistics.

    Returns a dict with:
        change_df  : per-participant per-signature change (Post - Pre)
        stats_df   : per-signature Cohen's d, AUC, FDR, p-value
        sig_cols   : list of scored signature column names
    """
    adata = get_sade_feldman()

    # Ensure log1p_tpm
    if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])

    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")
    adata = harmonize_response(adata)

    obs = adata.obs.copy()

    # Participant-level means per visit
    group_cols = ["participant_id", "visit", "response_harmonized"]
    pid_means = obs.groupby(group_cols, observed=True)[sig_cols].mean().reset_index()

    # Pre -> Post change per participant
    pre = pid_means[pid_means["visit"] == "Pre"].set_index("participant_id")
    post = pid_means[pid_means["visit"] == "Post"].set_index("participant_id")
    common_pids = pre.index.intersection(post.index)

    if len(common_pids) == 0:
        print("  WARNING: No paired participants found for outcome panels")
        return dict(change_df=None, stats_df=None, sig_cols=sig_cols, adata=adata)

    change = post.loc[common_pids, sig_cols] - pre.loc[common_pids, sig_cols]
    change["response"] = pre.loc[common_pids, "response_harmonized"]
    change = change.reset_index()

    # Per-signature statistics
    records = []
    for col in sig_cols:
        vals_r = change.loc[change["response"] == "Responder", col].dropna()
        vals_nr = change.loc[change["response"] == "Non-responder", col].dropna()

        if len(vals_r) < 2 or len(vals_nr) < 2:
            continue

        # Mean change
        mean_r = vals_r.mean()
        mean_nr = vals_nr.mean()
        sem_r = vals_r.sem()
        sem_nr = vals_nr.sem()

        # Cohen's d (pooled)
        n_r, n_nr = len(vals_r), len(vals_nr)
        pooled_std = np.sqrt(
            ((n_r - 1) * vals_r.std(ddof=1) ** 2 + (n_nr - 1) * vals_nr.std(ddof=1) ** 2)
            / (n_r + n_nr - 2)
        )
        d = (mean_r - mean_nr) / pooled_std if pooled_std > 0 else 0.0

        # t-test p-value
        _, p_val = stats.ttest_ind(vals_r, vals_nr, equal_var=False)

        # AUC from Mann-Whitney U
        u_stat, _ = stats.mannwhitneyu(vals_r, vals_nr, alternative="two-sided")
        auc = u_stat / (n_r * n_nr)
        # Ensure AUC reflects Responder > Non-responder direction
        if mean_r < mean_nr:
            auc = 1.0 - auc

        records.append(
            dict(
                signature=col,
                display=sig_display(col),
                mean_R=mean_r,
                mean_NR=mean_nr,
                sem_R=sem_r,
                sem_NR=sem_nr,
                cohens_d=d,
                abs_d=abs(d),
                p_value=p_val,
                auc=auc,
            )
        )

    stats_df = pd.DataFrame(records)

    # FDR correction (Benjamini-Hochberg)
    if len(stats_df) > 0:
        from statsmodels.stats.multitest import multipletests

        _, fdr, _, _ = multipletests(stats_df["p_value"], method="fdr_bh")
        stats_df["fdr"] = fdr
    else:
        stats_df["fdr"] = np.nan

    return dict(
        change_df=change,
        stats_df=stats_df,
        sig_cols=sig_cols,
        adata=adata,
    )


# ── Panel I: Signature changes comparison (grouped bars) ──────────


def _panel_signature_changes(ax, data: dict):
    """Grouped bar chart: mean change (Post-Pre) for Responders vs
    Non-responders, top 10 signatures sorted by |Cohen's d|."""
    stats_df = data["stats_df"]

    if stats_df is None or len(stats_df) == 0:
        ax.text(
            0.5,
            0.5,
            "No data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=COLORS["gray"],
        )
        ax.axis("off")
        return

    df = stats_df.sort_values("abs_d", ascending=False).head(10).copy()
    df = df.sort_values("abs_d", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    bar_h = 0.35

    # Responder bars
    ax.barh(
        y - bar_h / 2,
        df["mean_R"],
        height=bar_h,
        xerr=df["sem_R"],
        capsize=2,
        color=COLORS["treated"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Responder",
        error_kw=dict(lw=0.8, capthick=0.8),
    )

    # Non-responder bars
    ax.barh(
        y + bar_h / 2,
        df["mean_NR"],
        height=bar_h,
        xerr=df["sem_NR"],
        capsize=2,
        color=COLORS["control"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Non-responder",
        error_kw=dict(lw=0.8, capthick=0.8),
    )

    # FDR markers
    for i, (_, row) in enumerate(df.iterrows()):
        fdr_val = row["fdr"]
        if pd.notna(fdr_val) and fdr_val < 0.25:
            star = "***" if fdr_val < 0.001 else "**" if fdr_val < 0.01 else "*"
            x_max = max(abs(row["mean_R"]), abs(row["mean_NR"]))
            ax.text(
                x_max + 0.02,
                i,
                star,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(df["display"], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean Change (Post \u2212 Pre)", fontsize=10)
    ax.set_title("Signature Changes by Response\n(sorted by |Cohen's d|)", fontsize=11)

    ax.legend(fontsize=9, loc="lower right", frameon=True, framealpha=0.9)
    ax.text(
        0.97,
        0.12,
        "* FDR < 0.25  ** < 0.01  *** < 0.001",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        fontstyle="italic",
        color=COLORS["gray"],
    )
    despine(ax)


# ── Panel J: Cohen's d forest plot ────────────────────────────────


def _panel_cohens_d_forest(ax, data: dict):
    """Horizontal forest plot of Cohen's d for each signature."""
    stats_df = data["stats_df"]

    if stats_df is None or len(stats_df) == 0:
        ax.text(
            0.5,
            0.5,
            "No data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=COLORS["gray"],
        )
        ax.axis("off")
        return

    df = stats_df.sort_values("cohens_d", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    # Colours by direction
    colors = [COLORS["treated"] if d > 0 else COLORS["control"] for d in df["cohens_d"]]

    # Marker size by FDR significance
    sizes = []
    for fdr_val in df["fdr"]:
        if pd.notna(fdr_val) and fdr_val < 0.25:
            sizes.append(100)
        else:
            sizes.append(40)

    ax.scatter(df["cohens_d"], y, c=colors, s=sizes, edgecolors="white", linewidths=0.8, zorder=3)

    # Connect dots to zero with lines
    for i, d in enumerate(df["cohens_d"]):
        ax.plot([0, d], [i, i], color=colors[i], alpha=0.5, lw=1.5, zorder=1)

    # Reference lines
    ax.axvline(0, color="black", lw=0.8, zorder=2)
    for ref in [-0.5, 0.5]:
        ax.axvline(ref, color=COLORS["gray"], ls="--", lw=0.7, alpha=0.6, zorder=1)
    ax.text(
        0.5, len(df) + 0.3, "d = 0.5", ha="center", va="bottom", fontsize=7, color=COLORS["gray"]
    )
    ax.text(
        -0.5,
        len(df) + 0.3,
        "d = \u22120.5",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLORS["gray"],
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["display"], fontsize=9)
    ax.set_xlabel("Cohen's d (Responder vs Non-responder)", fontsize=10)
    ax.set_title("Effect Sizes: Responder vs Non-responder", fontsize=11)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=COLORS["treated"], label="Favours Responder"),
        mpatches.Patch(facecolor=COLORS["control"], label="Favours Non-resp"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["gray"],
            markersize=10,
            label="FDR < 0.25 (large)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["gray"],
            markersize=6,
            label="FDR >= 0.25 (small)",
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel K: AUC horizontal bar chart ─────────────────────────────


def _panel_auc_bars(ax, data: dict):
    """Horizontal bar chart of AUC for each signature."""
    stats_df = data["stats_df"]

    if stats_df is None or len(stats_df) == 0:
        ax.text(
            0.5,
            0.5,
            "No data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=COLORS["gray"],
        )
        ax.axis("off")
        return

    df = stats_df.sort_values("auc", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    # Colour by threshold
    auc_threshold = 0.6
    colors = [COLORS["treated"] if a >= auc_threshold else COLORS["gray"] for a in df["auc"]]

    ax.barh(y, df["auc"], color=colors, alpha=0.85, edgecolor="white", linewidth=0.5, height=0.7)

    # Reference line at 0.5 (random)
    ax.axvline(0.5, color=COLORS["highlight"], ls="--", lw=1.0, zorder=1)
    ax.text(
        0.5,
        len(df) + 0.3,
        "Random (0.5)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLORS["highlight"],
    )

    # Reference line at threshold
    ax.axvline(auc_threshold, color=COLORS["gray"], ls=":", lw=0.8, alpha=0.6, zorder=1)
    ax.text(
        auc_threshold,
        -0.8,
        f"AUC = {auc_threshold}",
        ha="center",
        va="top",
        fontsize=7,
        color=COLORS["gray"],
    )

    # Annotate AUC values on bars
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["auc"] + 0.01, i, f"{row['auc']:.2f}", ha="left", va="center", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(df["display"], fontsize=9)
    ax.set_xlabel("Area Under the Curve (AUC)", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_title("Response Discrimination (AUC)", fontsize=11)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=COLORS["treated"], alpha=0.85, label=f"AUC >= {auc_threshold}"),
        mpatches.Patch(facecolor=COLORS["gray"], alpha=0.85, label=f"AUC < {auc_threshold}"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================


def generate():
    """Create and save Supplementary Figure 4 panels.

    Reorganised layout (6 panels):
      A  Coefficient Q-Q plots           (was SF4-A)
      B  LOO influence                   (was SF4-E)
      C  Cell vs participant betas       (was SF5-A, imported)
      D  Mean vs median                  (was SF5-E, imported)
      E  Cell-type heatmap               (was SF5-G, imported)
      F  Rank concordance                (was SF5-H, imported)
    """
    print("Supplementary Figure 4: Model Diagnostics & Robustness")
    results = _load_did_datasets()

    if not results:
        print("  No DiD results; skipping.")
        return

    n_ds = len(results)

    # Panel A: Q-Q plots (one per dataset)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5))
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    _panel_qq(fig, axes, results)
    fig.suptitle("Coefficient Q-Q Plots", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # Panel B: LOO influence (was SF4-E)
    fig, ax = plt.subplots(figsize=(7, 5))
    _panel_loo_influence(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # Free DiD data
    for res in results.values():
        if "adata" in res:
            del res["adata"]
    results.clear()
    gc.collect()

    # Panels C-F: from SF5 robustness
    from .supp_fig5_robustness_details import (
        _panel_cell_vs_part,
        _panel_ct_heatmap,
        _panel_mean_vs_median,
        _panel_rank_concordance,
        _run_sensitivity,
    )

    data = _run_sensitivity()
    panel_map = [
        ("panel_C", _panel_cell_vs_part, (6.5, 6)),
        ("panel_D", _panel_mean_vs_median, (6.5, 6)),
        ("panel_E", _panel_ct_heatmap, (8, 5.5)),
        ("panel_F", _panel_rank_concordance, (7, 5)),
    ]
    for name, func, figsize in panel_map:
        fig, ax = plt.subplots(figsize=figsize)
        func(ax, data)
        fig.tight_layout()
        save_panel(fig, name, FIGURE_NAME, SUPP_OUTPUT)

    if "adata" in data:
        del data["adata"]
    data.clear()

    # Final cleanup
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
