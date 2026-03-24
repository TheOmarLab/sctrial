"""
Figure 5 — Multi-Dataset Generalization.

Seven panels demonstrating that sctrial analyses generalise across
heterogeneous study designs and disease contexts.

Layout
------
Top row : A (COVID-19 cross-sectional), B (Vaccine paired), C (AML within-arm)
Mid row : D (CAR-T), E (Melanoma DiD), F (Cross-dataset effect-size heatmap)
Bottom  : G (Cross-dataset GSEA heatmap — replicated pathways)
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
    TrialDesign,
    add_log1p_cpm_layer,
    apply_style,
    between_arm_comparison,
    despine,
    did_table,
    get_aml,
    get_cart,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    hedges_g,
    save_panel,
    score_signatures,
    sig_display,
    within_arm_comparison,
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

FIG_NAME = "Figure5_multi_dataset"

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

        lw = 1.2
        ms = 4.5

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
        Line2D([0], [0], marker="o", color=color_pos, lw=1.2, markersize=3,
               markeredgecolor="white", label=legend_pos_label),
        Line2D([0], [0], marker="o", color=color_neg, lw=1.2, markersize=3,
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
        available_bins = sorted(adata_covid.obs["dfo_bin"].dropna().unique())
        if target_visit not in available_bins:
            # Fallback: pick the first bin (sorted) with both severity groups
            for _bin in available_bins:
                _sub = adata_covid[adata_covid.obs["dfo_bin"] == _bin]
                if set(_sub.obs["severity"].unique()) >= {"Mild", "Severe"}:
                    target_visit = _bin
                    break
            else:
                # Last resort: pick any bin (will likely fail downstream)
                target_visit = available_bins[0]

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
        import traceback
        traceback.print_exc()
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
                use_bootstrap=True,
            )
            res_vax["label"] = res_vax["feature"].apply(sig_display)
            # Prefer bootstrap CIs, fall back to analytical per-row
            if "ci_lo_boot" in res_vax.columns:
                res_vax["ci_lo"] = res_vax["ci_lo_boot"].fillna(
                    res_vax["ci_lo_time"])
                res_vax["ci_hi"] = res_vax["ci_hi_boot"].fillna(
                    res_vax["ci_hi_time"])
            else:
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
    # Both use within-arm (Treatment only) Pre→Post comparisons.
    # AML has two nominal arms but Control has no Post timepoint
    # (healthy BM donors at baseline only), so a DiD interaction is
    # degenerate (beta_DiD == beta_time).  We therefore analyse the
    # Treatment arm longitudinally, matching CAR-T's single-arm design.
    _TREATED_ARM = {"aml": "Treatment", "cart": None}  # None → auto-detect
    _LOADERS = {"aml": get_aml, "cart": get_cart}
    for tag, name, panel_label in [("aml", "aml", "C"), ("cart", "cart", "D")]:
        try:
            print(f"  [{panel_label}] Loading {name.upper()} ...")
            adata_clin = _LOADERS[name]()
            adata_clin, sig_cols_clin = score_signatures(adata_clin)

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

            # Identify the treated arm for within-arm analysis
            arm_col = "response" if "response" in adata_clin.obs.columns else "arm"
            arm_values = list(adata_clin.obs[arm_col].dropna().unique())
            treated_arm = _TREATED_ARM.get(tag)
            if treated_arm is None:
                treated_arm = arm_values[0]

            # Within-arm (treated only) Pre→Post comparison
            if "arm" not in adata_clin.obs.columns:
                adata_clin.obs["arm"] = adata_clin.obs[arm_col]
            clin_design = TrialDesign(
                participant_col=pid_col,
                visit_col=visit_col,
                arm_col="arm" if arm_col != "arm" else arm_col,
                arm_treated=treated_arm,
                arm_control=treated_arm,  # single-arm
            )
            res_clin = within_arm_comparison(
                adata_clin,
                arm=treated_arm,
                features=sig_cols_clin,
                design=clin_design,
                visits=(pre_v, post_v),
                layer=None,
                standardize=True,
                use_bootstrap=True,
            )
            res_clin["label"] = res_clin["feature"].apply(sig_display)
            # Prefer bootstrap CIs, fall back to analytical per-row
            if "ci_lo_boot" in res_clin.columns:
                res_clin["ci_lo"] = res_clin["ci_lo_boot"].fillna(
                    res_clin["ci_lo_time"])
                res_clin["ci_hi"] = res_clin["ci_hi_boot"].fillna(
                    res_clin["ci_hi_time"])
            else:
                res_clin["ci_lo"] = res_clin["ci_lo_time"]
                res_clin["ci_hi"] = res_clin["ci_hi_time"]
            data[f"{tag}_effects"] = res_clin

            print(f"       {adata_clin.n_obs:,} cells, "
                  f"{adata_clin.obs[pid_col].nunique()} participants "
                  f"(analysing '{treated_arm}' arm)")
        except Exception as exc:
            print(f"  [{panel_label}] {name.upper()} error: {exc}")
            import traceback
            traceback.print_exc()
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
        import traceback
        traceback.print_exc()
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

    COVID-19 uses Hedges' g; Vaccine/AML/CAR-T use within-arm β
    (standardised); Melanoma uses DiD β (standardised).  AML is
    analysed within-arm (Treatment only) because the Control arm
    lacks Post timepoint data, making DiD degenerate.  All metrics
    are on a roughly comparable standardised scale — the panel
    footnote communicates the estimator differences.
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

    # AML: within-arm beta_time (no valid Control-Post for DiD)
    df = data.get("aml_effects")
    if df is not None and len(df):
        for _, row in df.iterrows():
            lbl = row.get("label", sig_display(row["feature"]))
            records.append({
                "dataset": "AML",
                "signature": lbl,
                "effect": row["beta_time"],
                "p": row.get("FDR_time", row.get("p_time", np.nan)),
            })

    # Melanoma: DiD beta (true two-arm: Responder vs Non-responder)
    df = data.get("mel_effects")
    if df is not None and len(df):
        for _, row in df.iterrows():
            lbl = row.get("label", sig_display(row["feature"]))
            records.append({
                "dataset": "Melanoma",
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
    ax.set_title("COVID-19", fontsize=6, fontweight="bold", loc="left", pad=8)
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
    ax.set_title("Vaccine (GSE171964)", fontsize=6, fontweight="bold", loc="left", pad=8)
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
    """Panel C: AML clinical dataset (within-arm Pre→Post)."""
    ax.set_title("AML (GSE116256)", fontsize=6, fontweight="bold", loc="left", pad=8)
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


def panel_d_cart(ax, data: dict[str, Any]) -> None:
    """Panel D: CAR-T clinical dataset (within-arm)."""
    ax.set_title("CAR-T (GSE290722)", fontsize=6, fontweight="bold", loc="left", pad=8)
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
    ax.set_title("Melanoma", fontsize=6, fontweight="bold", loc="left", pad=8)
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

    ax.set_title("Cross-Dataset Effect Sizes", fontsize=8, fontweight="bold", loc="center", pad=8)
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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right",
                       fontsize=7.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)


# ── composite figure ──────────────────────────────────────────────────────

def generate(*, save: bool = True) -> None:
    """Create and save Figure 5 individual panels."""
    print("=" * 60)
    print("Figure 5: Multi-Dataset Generalization")
    print("=" * 60)

    data = _prepare_data()

    if save:
        # Adaptive height based on the number of features in each panel
        def _n_features(key: str) -> int:
            df = data.get(key)
            return len(df) if df is not None else 6

        # Import cross-dataset GSEA panel from figure4 (where the GSEA
        # infrastructure lives) — we display it here in the multi-dataset figure.
        from .figure4_biological_discovery import (
            panel_C_replicated as _panel_gsea_cross,
            _run_multi_dataset_gsea,
        )

        # Run multi-dataset GSEA (uses cache from figure4 if available)
        gsea_multi = _run_multi_dataset_gsea()
        data["gsea_multi_dataset"] = gsea_multi

        panel_specs = [
            (panel_a_covid, "A_covid_severity", _n_features("covid_effects")),
            (panel_b_vaccine, "B_vaccine_paired", _n_features("vax_effects")),
            (panel_c_aml, "C_aml_clinical", _n_features("aml_effects")),
            (panel_d_cart, "D_cart_clinical", _n_features("cart_effects")),
            (panel_e_melanoma, "E_melanoma_did", _n_features("mel_effects")),
            (panel_f_heatmap, "F_heatmap", 8),  # heatmap uses fixed size
            (_panel_gsea_cross, "G_cross_dataset_gsea", 10),  # cross-dataset GSEA heatmap
        ]

        for panel_fn, panel_name, n_feat in panel_specs:
            # Adaptive height: ~0.38 inches per feature row, min 2.8
            h = max(2.8, 0.38 * n_feat + 1.1)
            # Wider figure for GSEA heatmap to avoid pathway name truncation
            w = 9.5 if "gsea" in panel_name.lower() else 6.5
            fig_p, ax_p = plt.subplots(figsize=(w, h))
            panel_fn(ax_p, data)
            fig_p.tight_layout(pad=0.6)
            save_panel(fig_p, panel_name, FIG_NAME, MAIN_OUTPUT)

    # ── Combined artboard (180 × ≤215 mm) ────────────────────────────────
    _SMALL_RC = {
        "font.size": 5,
        "axes.titlesize": 5.5,
        "axes.labelsize": 5,
        "xtick.labelsize": 4.5,
        "ytick.labelsize": 4.5,
        "legend.fontsize": 4,
        "legend.title_fontsize": 4,
    }
    _MAX_FONT_COMPOSITE = 6

    def _cap_fontsize(fig, maximum):
        """Shrink every text element in *fig* that exceeds *maximum*."""
        for ax in fig.get_axes():
            for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                        + ax.get_xticklabels() + ax.get_yticklabels()
                        + ax.texts):
                if txt.get_fontsize() > maximum:
                    txt.set_fontsize(maximum)
            if ax.get_legend():
                for txt in ax.get_legend().get_texts():
                    if txt.get_fontsize() > maximum:
                        txt.set_fontsize(maximum)
        for txt in fig.texts:
            if txt.get_fontsize() > maximum:
                txt.set_fontsize(maximum)

    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm = 1.0 / 25.4
    fig_c = plt.figure(figsize=(180 * _mm, 210 * _mm))

    #   Row 0: A | B | C    (forest plots)
    #   Row 1: D | E | F    (forest plots + heatmap)
    #   Row 2: G             (full-width GSEA heatmap)
    outer = fig_c.add_gridspec(
        3, 1,
        height_ratios=[1, 1, 1.3],
        hspace=0.55,
        left=0.10, right=0.95, top=0.97, bottom=0.06,
    )

    gs0 = outer[0].subgridspec(1, 3, wspace=1.05)
    ax_a = fig_c.add_subplot(gs0[0])
    ax_b = fig_c.add_subplot(gs0[1])
    ax_cc = fig_c.add_subplot(gs0[2])

    gs1 = outer[1].subgridspec(1, 3, wspace=0.65, width_ratios=[1, 1.3, 2.8])
    ax_d = fig_c.add_subplot(gs1[0])
    ax_e = fig_c.add_subplot(gs1[1])
    ax_f = fig_c.add_subplot(gs1[2])

    gs2 = outer[2].subgridspec(1, 2, width_ratios=[0.5, 1])
    ax_g = fig_c.add_subplot(gs2[1])

    panel_a_covid(ax_a, data)
    panel_b_vaccine(ax_b, data)
    panel_c_aml(ax_cc, data)
    panel_d_cart(ax_d, data)
    panel_e_melanoma(ax_e, data)
    panel_f_heatmap(ax_f, data)
    _panel_gsea_cross(ax_g, data)

    # Remove panel labels embedded by panel functions — will re-add below
    for ax in [ax_a, ax_b, ax_cc, ax_d, ax_e, ax_f, ax_g]:
        to_remove = [
            t for t in ax.texts
            if len(t.get_text()) == 1 and t.get_text().isupper()
        ]
        for t in to_remove:
            t.remove()

    # Move legends inside plots for the composite
    _inside = {
        ax_a: "lower right", ax_b: "lower right", ax_cc: "lower right",
        ax_d: "lower right", ax_e: "lower right",
    }
    for ax_target, loc in _inside.items():
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_target.legend(
                handles=handles, labels=labels,
                fontsize=4.5, loc=loc,
                frameon=True, framealpha=0.85,
                edgecolor="#CCCCCC", borderpad=0.3,
                handlelength=1, handletextpad=0.3,
                labelspacing=0.2,
            )

    # Move G ylabel slightly away from figure
    ax_g.yaxis.set_label_coords(-0.75, 0.5)

    # Reduce heatmap annotation font size in F
    for txt in ax_f.texts:
        txt.set_fontsize(max(txt.get_fontsize() * 0.75, 3.0))

    _cap_fontsize(fig_c, _MAX_FONT_COMPOSITE)

    # Uniform title font size across all panels
    for ax in [ax_a, ax_b, ax_cc, ax_d, ax_e]:
        ax.title.set_fontsize(2)
        ax.title.set_fontweight("bold")
    ax_f.title.set_fontsize(4.5)
    ax_f.title.set_fontweight("bold")
    ax_g.title.set_fontsize(7)
    ax_g.title.set_fontweight("bold")

    # Bold panel labels (after cap so they stay prominent)
    _lbl_fs = 9
    for ax, lbl in [
        (ax_a, "A"), (ax_b, "B"), (ax_cc, "C"),
        (ax_d, "D"), (ax_e, "E"), (ax_f, "F"),
        (ax_g, "G"),
    ]:
        ax.text(-0.25, 1.12, lbl, transform=ax.transAxes,
                fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")

    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIG_NAME, FIG_NAME, MAIN_OUTPUT, close=False)
    pdf_path = MAIN_OUTPUT / f"{FIG_NAME}_panels" / f"{FIG_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print(f"    Saved combined artboard (PNG + PDF)")
    print("  Figure 5 complete: 7 individual panels + combined (A–G)\n")


# ── entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    generate()
