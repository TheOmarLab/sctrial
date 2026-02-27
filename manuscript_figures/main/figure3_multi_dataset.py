"""
Figure 3 — Multi-Dataset Generalization.

Six panels demonstrating that sctrial analyses generalise across
heterogeneous study designs and disease contexts.

Layout
------
Top row : A (COVID-19 cross-sectional), B (Vaccine paired), C (AML)
Bottom row : D (CAR-T), E (Melanoma DiD), F (Cross-dataset effect-size heatmap)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .._shared import (
    COLORS,
    MAIN_OUTPUT,
    SCTRIAL_AVAILABLE,
    apply_style,
    despine,
    save_panel,
    # data loading
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    load_clinical_trial_dataset,
    harmonize_response,
    # scoring
    score_signatures,
    score_clinical_signatures,
    sig_display,
    # sctrial API
    TrialDesign,
    did_table,
    hedges_g,
    within_arm_comparison,
    between_arm_comparison,
    add_log1p_cpm_layer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def effect_size_ci(g: float, n1: int, n2: int, alpha: float = 0.05):
    """Approximate CI for Hedges' g (Hedges & Olkin 1985, normal approx)."""
    from scipy.stats import norm
    se = np.sqrt(1 / n1 + 1 / n2 + g ** 2 / (2 * (n1 + n2)))
    z = norm.ppf(1 - alpha / 2)
    return g - z * se, g + z * se

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIG_NAME = "Figure3_multi_dataset"

_DATASET_COLORS = {
    "COVID-19":  "#3498DB",
    "Vaccine":   "#27AE60",
    "AML":       "#8E44AD",
    "CAR-T":     "#E67E22",
    "Melanoma":  "#E74C3C",
}


# ── helpers ────────────────────────────────────────────────────────────────

def _stars(p: float) -> str:
    """Return significance stars for a p-value.

    Uses conventional thresholds plus a marginal indicator (†) for
    0.05 ≤ FDR < 0.1 to highlight near-significant trends.
    """
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "†"
    return ""


def _forest_plot(
    ax,
    df: pd.DataFrame,
    *,
    effect_col: str,
    ci_lo_col: str,
    ci_hi_col: str,
    label_col: str,
    fdr_col: str | None = None,
    xlabel: str = "Effect size",
    color_pos: str = COLORS["treated"],
    color_neg: str = COLORS["control"],
    legend_pos_label: str = "Positive",
    legend_neg_label: str = "Negative",
) -> None:
    """Draw a horizontal forest plot on *ax*."""
    df = df.sort_values(effect_col, ascending=True).reset_index(drop=True)
    n_rows = len(df)

    for i, row in df.iterrows():
        es = row[effect_col]
        lo, hi = row[ci_lo_col], row[ci_hi_col]
        color = color_pos if es > 0 else color_neg

        lw = 1.8
        ms = 6.5

        ax.plot([lo, hi], [i, i], color=color, lw=lw, solid_capstyle="round")
        ax.plot(
            es, i, "o", color=color, markersize=ms,
            markeredgecolor="white", markeredgewidth=0.8,
        )

    # Set axis limits *before* placing stars so we know the data range
    ax.axvline(0, color="black", ls="-", lw=0.8, alpha=0.5)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(df[label_col].values, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="x", labelsize=7)

    # Expand x-limits to leave room for stars on the right
    x_lo, x_hi = ax.get_xlim()
    x_range = x_hi - x_lo
    ax.set_xlim(x_lo, x_hi + 0.12 * x_range)

    # Place significance stars (after xlim is set)
    if fdr_col:
        for i, row in df.iterrows():
            if fdr_col in row.index:
                star = _stars(row[fdr_col])
                if star:
                    hi = row[ci_hi_col]
                    es = row[effect_col]
                    x_txt = max(hi, es) + 0.03 * x_range
                    ax.text(x_txt, i, star, fontsize=8, va="center",
                            fontweight="bold", color=(
                                color_pos if es > 0 else color_neg))

    # Compact legend
    legend_elements = [
        Line2D([0], [0], marker="o", color=color_pos, lw=1.8, markersize=6,
               markeredgecolor="white", label=legend_pos_label),
        Line2D([0], [0], marker="o", color=color_neg, lw=1.8, markersize=6,
               markeredgecolor="white", label=legend_neg_label),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True,
              facecolor="white", edgecolor="0.85", fontsize=7,
              handlelength=1.5, borderpad=0.4, labelspacing=0.3)
    despine(ax)


# ── data preparation ──────────────────────────────────────────────────────

def _prepare_data() -> dict[str, Any]:
    """Load all five datasets, score signatures, run analyses.

    Returns a dict consumed by individual panel functions.
    """
    data: dict[str, Any] = {}

    # ── Panel A: COVID-19 Stephenson (cross-sectional) ────────────────────
    try:
        print("  [A] Loading Stephenson COVID-19 ...")
        adata_covid = get_stephenson()
        if "log1p_cpm" not in adata_covid.layers and "counts" in adata_covid.layers:
            adata_covid = add_log1p_cpm_layer(
                adata_covid, counts_layer="counts", out_layer="log1p_cpm",
            )
        adata_covid, sig_cols = score_signatures(adata_covid, layer="log1p_cpm")

        # Pick a DFO bin where both Mild & Severe have patients
        target_visit = "DFO_8-14"
        if target_visit not in adata_covid.obs["dfo_bin"].unique():
            target_visit = adata_covid.obs["dfo_bin"].unique()[0]

        ad_visit = adata_covid[adata_covid.obs["dfo_bin"] == target_visit].copy()

        # Use sctrial between_arm_comparison API
        covid_design = TrialDesign(
            participant_col="participant_id",
            visit_col="dfo_bin",
            arm_col="severity",
            arm_treated="Severe",
            arm_control="Mild",
        )
        res_covid = between_arm_comparison(
            ad_visit,
            visit=target_visit,
            features=sig_cols,
            design=covid_design,
            layer="log1p_cpm",
            standardize=True,
        )
        res_covid["label"] = res_covid["feature"].apply(sig_display)

        # Compute Hedges' g effect sizes + Welch t-test p-values
        # (both from the same participant-level means for consistency)
        df_agg = (
            ad_visit.obs
            .groupby(["participant_id", "severity"], observed=True)[sig_cols]
            .mean()
            .reset_index()
        )
        g_rows = []
        for _, row in res_covid.iterrows():
            sig = row["feature"]
            mild = df_agg.loc[df_agg["severity"] == "Mild", sig].dropna().values
            severe = df_agg.loc[df_agg["severity"] == "Severe", sig].dropna().values
            if len(mild) >= 3 and len(severe) >= 3:
                g = hedges_g(severe, mild)
                n1, n2 = len(severe), len(mild)
                ci_lo, ci_hi = effect_size_ci(g, n1, n2)
                _, p_welch = stats.ttest_ind(severe, mild, equal_var=False)
                g_rows.append({
                    "feature": sig,
                    "hedges_g": g,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_welch": p_welch,
                })
            else:
                g_rows.append({
                    "feature": sig,
                    "hedges_g": row["beta_arm"],
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "p_welch": np.nan,
                })

        g_df = pd.DataFrame(g_rows)
        # FDR-correct the Welch p-values (consistent with Hedges' g)
        valid_p = g_df["p_welch"].dropna()
        if len(valid_p):
            _, fdr_vals, *_ = multipletests(valid_p, method="fdr_bh")
            g_df.loc[valid_p.index, "fdr_welch"] = fdr_vals
        else:
            g_df["fdr_welch"] = np.nan

        res_covid = res_covid.merge(g_df, on="feature", how="left")
        res_covid["fdr"] = res_covid["fdr_welch"]
        data["covid_effects"] = res_covid

        print(f"       {adata_covid.n_obs:,} cells, "
              f"{adata_covid.obs['participant_id'].nunique()} participants")
    except Exception as exc:
        print(f"  [A] COVID-19 error: {exc}")
        import traceback; traceback.print_exc()
        data["covid_effects"] = None

    # ── Panel B: Vaccine within-arm paired ────────────────────────────────
    try:
        print("  [B] Loading Vaccine (GSE171964) ...")
        adata_vax = get_vaccine()
        adata_vax, sig_cols_vax = score_signatures(adata_vax, layer="counts")

        # Build a single-arm design (all participants treated)
        if "arm" not in adata_vax.obs.columns:
            adata_vax.obs["arm"] = "Treated"

        vax_design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            arm_control="Treated",   # single-arm
        )

        # Use sctrial within_arm_comparison
        try:
            res_vax = within_arm_comparison(
                adata_vax,
                arm="Treated",
                features=sig_cols_vax,
                design=vax_design,
                visits=("Pre", "Post"),
                layer=None,
                standardize=True,
            )
            res_vax["label"] = res_vax["feature"].apply(sig_display)
            # Use SE/CI columns returned by within_arm_comparison
            res_vax["ci_lo"] = res_vax["ci_lo_time"]
            res_vax["ci_hi"] = res_vax["ci_hi_time"]
            data["vax_effects"] = res_vax
        except Exception as exc_inner:
            # Fallback: manual paired computation
            warnings.warn(
                f"within_arm_comparison failed ({exc_inner}); "
                "falling back to manual paired stats."
            )
            df_agg = (
                adata_vax.obs
                .groupby(["participant_id", "visit"], observed=True)[sig_cols_vax]
                .mean()
                .reset_index()
            )
            paired = df_agg.groupby("participant_id").size()
            paired_ids = paired[paired >= 2].index
            deltas: dict[str, list[float]] = {s: [] for s in sig_cols_vax}
            for pid in paired_ids:
                sub = df_agg[df_agg["participant_id"] == pid]
                pre_row = sub[sub["visit"] == "Pre"]
                post_row = sub[sub["visit"] == "Post"]
                if len(pre_row) and len(post_row):
                    for s in sig_cols_vax:
                        deltas[s].append(
                            float(post_row[s].values[0] - pre_row[s].values[0])
                        )
            rows_fb = []
            for s in sig_cols_vax:
                d = np.array(deltas[s])
                if len(d) >= 3:
                    m, se = np.mean(d), np.std(d, ddof=1) / np.sqrt(len(d))
                    t_crit = stats.t.ppf(0.975, len(d) - 1)
                    t_stat, p = stats.ttest_1samp(d, 0)
                    rows_fb.append({
                        "feature": s,
                        "label": sig_display(s),
                        "beta_time": m,
                        "p_time": p,
                        "n_units": len(d),
                        "ci_lo": m - t_crit * se,
                        "ci_hi": m + t_crit * se,
                    })
            if rows_fb:
                res_fb = pd.DataFrame(rows_fb)
                _, fdr_fb, *_ = multipletests(res_fb["p_time"], method="fdr_bh")
                res_fb["FDR_time"] = fdr_fb
                data["vax_effects"] = res_fb
            else:
                data["vax_effects"] = None

        print(f"       {adata_vax.n_obs:,} cells, "
              f"{adata_vax.obs['participant_id'].nunique()} participants")
    except Exception as exc:
        print(f"  [B] Vaccine error: {exc}")
        data["vax_effects"] = None

    # ── Panels C & D: AML and CAR-T ──────────────────────────────────────
    for tag, name, panel_label in [("aml", "aml", "C"), ("cart", "cart", "D")]:
        try:
            print(f"  [{panel_label}] Loading {name.upper()} ...")
            adata_clin = load_clinical_trial_dataset(name)
            adata_clin, sig_cols_clin = score_clinical_signatures(adata_clin)

            # Harmonise column names
            pid_col = (
                "participant_id"
                if "participant_id" in adata_clin.obs.columns
                else "patient_id"
            )
            if "visit" not in adata_clin.obs.columns:
                if "timepoint" in adata_clin.obs.columns:
                    adata_clin.obs["visit"] = adata_clin.obs["timepoint"]
            visit_col = "visit"

            # Determine Pre / Post visits
            visits_avail = list(adata_clin.obs[visit_col].unique())
            if "Pre" in visits_avail and "Post" in visits_avail:
                pre_v, post_v = "Pre", "Post"
            elif "Diagnosis" in visits_avail:
                pre_v = "Diagnosis"
                others = [v for v in visits_avail if v != "Diagnosis"]
                post_v = others[0] if others else visits_avail[-1]
            else:
                import re as _re

                def _sort_key(v):
                    nums = _re.findall(r"\d+", str(v))
                    return int(nums[0]) if nums else 0

                visits_sorted = sorted(visits_avail, key=_sort_key)
                pre_v, post_v = visits_sorted[0], visits_sorted[-1]

            # Detect whether this is a two-arm or single-arm dataset
            arm_col = "response" if "response" in adata_clin.obs.columns else "arm"
            arm_values = list(adata_clin.obs[arm_col].dropna().unique())
            is_two_arm = len(arm_values) >= 2

            if is_two_arm:
                # Two-arm design (e.g. AML: Treatment vs Control) → use did_table
                # Explicit arm mapping to avoid order-dependent sign flips
                _KNOWN_TREATED = {"Treatment", "Treated", "Responder"}
                _KNOWN_CONTROL = {"Control", "Untreated", "Non-responder"}
                treated = next(
                    (v for v in arm_values if v in _KNOWN_TREATED),
                    arm_values[0],
                )
                control = next(
                    (v for v in arm_values if v in _KNOWN_CONTROL),
                    arm_values[1],
                )
                clin_design = TrialDesign(
                    participant_col=pid_col,
                    visit_col=visit_col,
                    arm_col=arm_col,
                    arm_treated=treated,
                    arm_control=control,
                )
                res_clin = did_table(
                    adata_clin,
                    features=sig_cols_clin,
                    design=clin_design,
                    visits=(pre_v, post_v),
                    layer=None,
                    standardize=True,
                    aggregate="participant_visit",
                )
                res_clin["label"] = res_clin["feature"].apply(sig_display)
                # Compute CIs from SE
                res_clin["ci_lo"] = res_clin["beta_DiD"] - 1.96 * res_clin["se_DiD"]
                res_clin["ci_hi"] = res_clin["beta_DiD"] + 1.96 * res_clin["se_DiD"]
                data[f"{tag}_effects"] = res_clin
            else:
                # Single-arm design (e.g. CAR-T) → use within_arm_comparison
                if "arm" not in adata_clin.obs.columns:
                    adata_clin.obs["arm"] = arm_values[0]
                clin_design = TrialDesign(
                    participant_col=pid_col,
                    visit_col=visit_col,
                    arm_col="arm" if "arm" in adata_clin.obs.columns else arm_col,
                    arm_treated=arm_values[0],
                    arm_control=arm_values[0],
                )
                res_clin = within_arm_comparison(
                    adata_clin,
                    arm=arm_values[0],
                    features=sig_cols_clin,
                    design=clin_design,
                    visits=(pre_v, post_v),
                    layer=None,
                    standardize=True,
                )
                res_clin["label"] = res_clin["feature"].apply(sig_display)
                # Use SE/CI columns returned by within_arm_comparison
                res_clin["ci_lo"] = res_clin["ci_lo_time"]
                res_clin["ci_hi"] = res_clin["ci_hi_time"]
                data[f"{tag}_effects"] = res_clin

            print(f"       {adata_clin.n_obs:,} cells, "
                  f"{adata_clin.obs[pid_col].nunique()} participants")
        except Exception as exc:
            print(f"  [{panel_label}] {name.upper()} error: {exc}")
            import traceback; traceback.print_exc()
            data[f"{tag}_effects"] = None

    # ── Melanoma (Sade-Feldman): DiD responder vs non-responder ──────────
    try:
        print("  [E] Loading Melanoma (Sade-Feldman) ...")
        adata_mel = get_sade_feldman()
        if "log1p_tpm" not in adata_mel.layers:
            raise RuntimeError("log1p_tpm layer missing from Sade-Feldman dataset.")
        adata_mel = harmonize_response(adata_mel)
        adata_mel, sig_cols_mel = score_signatures(adata_mel, layer="log1p_tpm")

        mel_design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="response_harmonized",
            arm_treated="Responder",
            arm_control="Non-responder",
        )

        res_mel = did_table(
            adata_mel,
            features=sig_cols_mel,
            design=mel_design,
            visits=("Pre", "Post"),
            layer="log1p_tpm",
            standardize=True,
            aggregate="participant_visit",
            use_bootstrap=True,
        )
        res_mel["label"] = res_mel["feature"].apply(sig_display)
        data["mel_effects"] = res_mel

        print(f"       {adata_mel.n_obs:,} cells, "
              f"{adata_mel.obs['participant_id'].nunique()} participants")
    except Exception as exc:
        print(f"  [E] Melanoma error: {exc}")
        import traceback; traceback.print_exc()
        data["mel_effects"] = None

    # ── Compute CIs for melanoma DiD results ────────────────────────────
    if data.get("mel_effects") is not None:
        mel = data["mel_effects"]
        # Prefer bootstrap CIs, fall back to analytical per-row
        analytical_lo = mel["beta_DiD"] - 1.96 * mel["se_DiD"]
        analytical_hi = mel["beta_DiD"] + 1.96 * mel["se_DiD"]
        if "ci_lo_boot" in mel.columns and "ci_hi_boot" in mel.columns:
            mel["ci_lo"] = mel["ci_lo_boot"].fillna(analytical_lo)
            mel["ci_hi"] = mel["ci_hi_boot"].fillna(analytical_hi)
        else:
            mel["ci_lo"] = analytical_lo
            mel["ci_hi"] = analytical_hi
        # Prefer bootstrap p-values / FDR where available
        if "p_DiD_boot" in mel.columns:
            mel["p_DiD"] = mel["p_DiD_boot"].fillna(mel["p_DiD"])
        if "FDR_DiD_boot" in mel.columns:
            mel["FDR_DiD"] = mel["FDR_DiD_boot"].fillna(mel["FDR_DiD"])
        data["mel_effects"] = mel

    # ── Panel F: cross-dataset effect-size matrix ─────────────────────────
    data["heatmap_matrix"], data["heatmap_stars"] = _build_heatmap_data(data)

    return data


def _build_heatmap_data(
    data: dict[str, Any],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Compile standardised effect sizes across datasets.

    COVID-19 uses Hedges' g; Vaccine/CAR-T use within-arm β (standardised);
    AML/Melanoma use DiD β (standardised).  All are on a roughly comparable
    standardised scale but derive from different estimators — the panel
    footnote communicates this.
    """

    records: list[dict[str, Any]] = []

    # COVID-19: Hedges' g (cross-sectional)
    df = data.get("covid_effects")
    if df is not None and len(df):
        for _, row in df.iterrows():
            records.append({
                "dataset": "COVID-19",
                "signature": row["label"],
                "effect": row["hedges_g"],
                "p": row.get("fdr", np.nan),
            })

    # Vaccine / CAR-T: within-arm beta_time
    for tag, ds_name in [("vax", "Vaccine"), ("cart", "CAR-T")]:
        df = data.get(f"{tag}_effects")
        if df is not None and len(df):
            for _, row in df.iterrows():
                lbl = row.get("label", sig_display(row["feature"]))
                records.append({
                    "dataset": ds_name,
                    "signature": lbl,
                    "effect": row["beta_time"],
                    "p": row.get("FDR_time", row.get("p_time", np.nan)),
                })

    # AML / Melanoma: DiD beta
    for tag, ds_name in [("aml", "AML"), ("mel", "Melanoma")]:
        df = data.get(f"{tag}_effects")
        if df is not None and len(df):
            for _, row in df.iterrows():
                lbl = row.get("label", sig_display(row["feature"]))
                records.append({
                    "dataset": ds_name,
                    "signature": lbl,
                    "effect": row.get("beta_DiD", np.nan),
                    "p": row.get("FDR_DiD", row.get("p_DiD", np.nan)),
                })

    if not records:
        return None, None

    df_all = pd.DataFrame(records)

    # Pivot to matrix form
    mat = df_all.pivot_table(
        index="dataset", columns="signature", values="effect", aggfunc="first",
    )
    pmat = df_all.pivot_table(
        index="dataset", columns="signature", values="p", aggfunc="first",
    )

    # Order datasets consistently
    ds_order = [d for d in ["COVID-19", "Vaccine", "AML", "CAR-T", "Melanoma"]
                if d in mat.index]
    mat = mat.loc[ds_order]
    pmat = pmat.loc[ds_order]

    # Build star annotation matrix
    star_mat = pmat.map(lambda v: _stars(v) if pd.notna(v) else "")

    return mat, star_mat


# ── panel functions ───────────────────────────────────────────────────────

def panel_a_covid(ax, data: dict[str, Any]) -> None:
    """Panel A: COVID-19 Stephenson cross-sectional (Severe vs Mild)."""
    ax.set_title("COVID-19 (Stephenson)", fontsize=10, loc="left", pad=8)
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom")

    df = data.get("covid_effects")
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "COVID-19 data not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    _forest_plot(
        ax, df,
        effect_col="hedges_g",
        ci_lo_col="ci_lo",
        ci_hi_col="ci_hi",
        label_col="label",
        xlabel="Hedge's g (Severe vs Mild)",
        color_pos=COLORS["treated"],
        color_neg=COLORS["control"],
        legend_pos_label="Severe $\\uparrow$",
        legend_neg_label="Mild $\\uparrow$",
    )


def panel_b_vaccine(ax, data: dict[str, Any]) -> None:
    """Panel B: Vaccine within-arm paired Pre->Post."""
    ax.set_title("Vaccine (GSE171964)", fontsize=10, loc="left", pad=8)
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom")

    df = data.get("vax_effects")
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "Vaccine data not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    _forest_plot(
        ax, df,
        effect_col="beta_time",
        ci_lo_col="ci_lo",
        ci_hi_col="ci_hi",
        label_col="label",
        xlabel="Standardised $\\Delta$ (Post $-$ Pre)",
        color_pos=COLORS["treated"],
        color_neg=COLORS["control"],
        legend_pos_label="Post $\\uparrow$",
        legend_neg_label="Pre $\\uparrow$",
    )


def panel_c_aml(ax, data: dict[str, Any]) -> None:
    """Panel C: AML clinical dataset (two-arm DiD)."""
    ax.set_title("AML (GSE116256)", fontsize=10, loc="left", pad=8)
    ax.text(-0.12, 1.05, "C", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom")

    df = data.get("aml_effects")
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "AML data not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    _forest_plot(
        ax, df,
        effect_col="beta_DiD",
        ci_lo_col="ci_lo",
        ci_hi_col="ci_hi",
        label_col="label",
        xlabel="DiD effect (Treatment vs Control)",
        color_pos=COLORS["treated"],
        color_neg=COLORS["control"],
        legend_pos_label="Treatment $\\uparrow$",
        legend_neg_label="Control $\\uparrow$",
    )


def panel_d_cart(ax, data: dict[str, Any]) -> None:
    """Panel D: CAR-T clinical dataset (within-arm)."""
    ax.set_title("CAR-T (GSE290722)", fontsize=10, loc="left", pad=8)
    ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom")

    df = data.get("cart_effects")
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "CAR-T data not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    _forest_plot(
        ax, df,
        effect_col="beta_time",
        ci_lo_col="ci_lo",
        ci_hi_col="ci_hi",
        label_col="label",
        xlabel="Standardised $\\Delta$ (Post $-$ Pre)",
        color_pos=COLORS["treated"],
        color_neg=COLORS["control"],
        legend_pos_label="Post $\\uparrow$",
        legend_neg_label="Pre $\\uparrow$",
    )


def panel_e_melanoma(ax, data: dict[str, Any]) -> None:
    """Panel E: Melanoma (Sade-Feldman) DiD — Responder vs Non-responder."""
    ax.set_title("Melanoma (Sade-Feldman)", fontsize=10, loc="left", pad=8)
    ax.text(-0.12, 1.05, "E", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom")

    df = data.get("mel_effects")
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "Melanoma data not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    _forest_plot(
        ax, df,
        effect_col="beta_DiD",
        ci_lo_col="ci_lo",
        ci_hi_col="ci_hi",
        label_col="label",
        xlabel="DiD effect (Responder vs Non-responder)",
        color_pos=COLORS["treated"],
        color_neg=COLORS["control"],
        legend_pos_label="Responder $\\uparrow$",
        legend_neg_label="Non-resp. $\\uparrow$",
    )


def panel_f_heatmap(ax, data: dict[str, Any]) -> None:
    """Panel F: Cross-dataset standardised effect-size heatmap."""
    import seaborn as sns

    ax.set_title("Cross-Dataset Effect Sizes", fontsize=10, loc="left", pad=8)
    ax.text(-0.12, 1.05, "F", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom")

    mat = data.get("heatmap_matrix")
    star_mat = data.get("heatmap_stars")
    if mat is None or mat.empty:
        ax.text(0.5, 0.5, "Insufficient data for heatmap",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    # Build combined annotation: effect size + stars
    annot_combined = mat.copy().astype(object)
    for r in mat.index:
        for c in mat.columns:
            val = mat.loc[r, c]
            star = star_mat.loc[r, c] if star_mat is not None else ""
            if pd.isna(val):
                annot_combined.loc[r, c] = ""
            else:
                annot_combined.loc[r, c] = f"{val:.1f}{star}"

    # Determine colour limits symmetrically
    vmax = max(abs(np.nanmin(mat.values)), abs(np.nanmax(mat.values)), 0.5)

    sns.heatmap(
        mat,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Standardised effect", "shrink": 0.7, "aspect": 20},
        annot=annot_combined.values,
        fmt="",
        annot_kws={"fontsize": 7, "fontweight": "bold"},
        mask=mat.isna(),  # grey out missing cells
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                       fontsize=7.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)

    # Footnote: metrics differ across datasets
    ax.text(
        0.5, -0.32,
        "Hedges' g (COVID-19); within-arm β (Vaccine, CAR-T); DiD β (AML, Melanoma)",
        transform=ax.transAxes, fontsize=6, ha="center", va="top",
        fontstyle="italic", color="0.4",
    )


# ── composite figure ──────────────────────────────────────────────────────

def generate(*, save: bool = True) -> None:
    """Create and save Figure 3 individual panels."""
    print("=" * 60)
    print("Figure 3: Multi-Dataset Generalization")
    print("=" * 60)

    data = _prepare_data()

    if save:
        # Adaptive height based on the number of features in each panel
        def _n_features(key: str) -> int:
            df = data.get(key)
            return len(df) if df is not None else 6

        panel_specs = [
            (panel_a_covid, "A_covid_severity", _n_features("covid_effects")),
            (panel_b_vaccine, "B_vaccine_paired", _n_features("vax_effects")),
            (panel_c_aml, "C_aml_clinical", _n_features("aml_effects")),
            (panel_d_cart, "D_cart_clinical", _n_features("cart_effects")),
            (panel_e_melanoma, "E_melanoma_did", _n_features("mel_effects")),
            (panel_f_heatmap, "F_heatmap", 8),  # heatmap uses fixed size
        ]

        for panel_fn, panel_name, n_feat in panel_specs:
            # Adaptive height: ~0.38 inches per feature row, min 2.8
            h = max(2.8, 0.38 * n_feat + 1.1)
            fig_p, ax_p = plt.subplots(figsize=(6.5, h))
            panel_fn(ax_p, data)
            fig_p.tight_layout(pad=0.6)
            save_panel(fig_p, panel_name, FIG_NAME, MAIN_OUTPUT)


# ── entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    generate()
