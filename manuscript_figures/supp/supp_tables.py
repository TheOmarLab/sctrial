"""
Supplementary Tables 1–8.
=========================

Table 1  Gene signature definitions (name, gene count, genes).
Table 2  Complete effect-size results across all signatures and datasets.
Table 3  GSEA pre-ranked results (one sheet per dataset).
Table 4  Permutation test results.
Table 5  Power analysis results.
Table 6  Gene-level DiD results (TNBC / Zhang).
Table 7  Gene-level DiD results (Sade-Feldman / Melanoma).
Table 8  Dataset metadata summary.
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Collection
from contextlib import contextmanager

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .._relabeling import (
    between_arm_nulls,
    did_permutation_nulls,
    did_relabeling_cached,
    permutation_summary,
    within_arm_nulls,
)
from .._shared import (
    CLINICAL_SIGNATURES,
    GENE_SIGNATURES,
    SCTRIAL_AVAILABLE,
    SIGNATURE_DISPLAY_NAMES,
    STATS_FINGERPRINT,
    SUPP_OUTPUT,
    TrialDesign,
    add_log1p_cpm_layer,
    apply_style,
    between_arm_fit_beta,
    clear_cache,
    did_table,
    get_aml,
    get_between_arm_aggregated_df,
    get_cart,
    get_sade_feldman,
    get_stephenson,
    get_tnbc_zhang,
    get_vaccine,
    get_within_arm_aggregated_df,
    harmonize_response,
    hedges_g,
    load_or_run_gsea_cross_sectional,
    load_or_run_gsea_did,
    load_or_run_gsea_within_arm,
    score_clinical_signatures,
    score_signatures,
    sig_display,
    within_arm_comparison,
)


@contextmanager
def _quiet():
    """Silence warnings in a block; ``catch_warnings(action=...)`` needs Python 3.11."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ======================================================================
# Table 1 — Gene signature definitions
# ======================================================================


def _selected(name: str, datasets: Collection[str] | None) -> bool:
    """True when *name* is among the requested datasets (every dataset when None).

    Table builders raise when a selected dataset fails, so a table is never
    written with a dataset silently missing.
    """
    return datasets is None or name in datasets


def table1_gene_signatures() -> pd.DataFrame:
    """Gene signature definitions: name, gene count, gene list."""
    print("  Table 1: Gene signature definitions")

    rows = []
    # Main signatures
    for name, genes in GENE_SIGNATURES.items():
        display = SIGNATURE_DISPLAY_NAMES.get(name, name)
        rows.append({
            "Signature": name,
            "Display Name": display,
            "Set": "Main",
            "Gene Count": len(genes),
            "Genes": ", ".join(sorted(genes)),
        })
    # Clinical signatures
    for name, genes in CLINICAL_SIGNATURES.items():
        rows.append({
            "Signature": name,
            "Display Name": name.replace("_", " "),
            "Set": "Clinical",
            "Gene Count": len(genes),
            "Genes": ", ".join(sorted(genes)),
        })

    df = pd.DataFrame(rows)
    path = SUPP_OUTPUT / "Supp_Table1_gene_signatures.csv"
    df.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(df)} signatures)")
    return df


# ======================================================================
# Table 2 — Complete results across ALL datasets (unified)
# ======================================================================

_KEEP_COLS = [
    "dataset", "estimand", "label", "feature", "signature_set",
    "effect", "se", "se_bootstrap", "pvalue", "FDR", "ci_lower", "ci_upper", "inference",
    "n_units", "n_focal", "n_reference", "cov_type_used",
]

_SET_IMMUNE = "immune"
_SET_CLINICAL = "clinical"


def _by_signature_set(run, immune: list[str], clinical: list[str]) -> pd.DataFrame:
    """Run ``run(features)`` once per signature set and stack the results.

    The package functions correct for multiple testing over the features of one call,
    so each set forms its own FDR family: the immune signatures, which are the family
    of every figure, and the clinical signatures.
    """
    parts = []
    for name, feats in ((_SET_IMMUNE, list(immune)),
                        (_SET_CLINICAL, [f for f in clinical if f not in set(immune)])):
        if not feats:
            continue
        res = run(feats)
        res["signature_set"] = name
        parts.append(res)
    return pd.concat(parts, ignore_index=True)


def _paired_group_sizes(adata, design: TrialDesign, visits: tuple[str, str]) -> tuple[int, int]:
    """Participants with both visits in the focal and in the reference group.

    ``did_table`` excludes a participant that carries neither group label, with a
    warning. The table reports the group sizes of the cohort, so an unexpected label
    among the paired participants raises here.
    """
    obs = adata.obs[adata.obs[design.visit_col].isin(visits)]
    pid = design.participant_col
    n_visits = obs.groupby(pid, observed=True)[design.visit_col].nunique()
    paired = obs[obs[pid].isin(n_visits[n_visits == 2].index)]
    arm = paired.groupby(pid, observed=True)[design.arm_col].first().astype(str)
    other = sorted(set(arm) - {design.arm_treated, design.arm_control})
    if other:
        raise ValueError(
            f"paired participants with {design.arm_col} in {other}, neither "
            f"{design.arm_treated!r} nor {design.arm_control!r}")
    return int((arm == design.arm_treated).sum()), int((arm == design.arm_control).sum())

_COL_MAP = {
    "beta_time": "effect",
    "se_time": "se",
    "se_time_boot": "se_bootstrap",
    "p_time": "pvalue",
    "FDR_time": "FDR",
    "ci_lo_time": "ci_lower",
    "ci_hi_time": "ci_upper",
    "beta_DiD": "effect",
    "se_DiD": "se",
    "se_DiD_boot": "se_bootstrap",
    "p_DiD": "pvalue",
    "FDR_DiD": "FDR",
}

_INFERENCE_BOOTSTRAP = "wild cluster bootstrap"
_INFERENCE_ANALYTICAL = "analytical"


def _harmonise(df: pd.DataFrame, dataset: str, estimand: str) -> pd.DataFrame:
    """Rename columns to a common schema and tag with dataset/estimand.

    A row's p-value and interval come from one procedure, named in ``inference``: the wild
    cluster bootstrap where it was run (the interval inverts the bootstrap test), otherwise
    the model's analytical interval, or effect ± 1.96 SE where the model reports none.
    ``se`` is always the analytical standard error and ``se_bootstrap`` the bootstrap one.
    """
    out = df.rename(columns=_COL_MAP).copy()
    # Drop any duplicate columns created by rename
    out = out.loc[:, ~out.columns.duplicated()]
    out["dataset"] = dataset
    out["estimand"] = estimand
    if "label" not in out.columns:
        out["label"] = out["feature"].apply(sig_display)

    for col in ("ci_lower", "ci_upper"):
        if col not in out.columns:
            out[col] = np.nan
    no_interval = out["ci_lower"].isna() | out["ci_upper"].isna()
    out.loc[no_interval, "ci_lower"] = out["effect"] - 1.96 * out["se"]
    out.loc[no_interval, "ci_upper"] = out["effect"] + 1.96 * out["se"]
    out["inference"] = _INFERENCE_ANALYTICAL
    if {"ci_lo_boot", "ci_hi_boot"} <= set(out.columns):
        boot = out["ci_lo_boot"].notna() & out["ci_hi_boot"].notna()
        out.loc[boot, "ci_lower"] = out["ci_lo_boot"]
        out.loc[boot, "ci_upper"] = out["ci_hi_boot"]
        out.loc[boot, "inference"] = _INFERENCE_BOOTSTRAP

    avail = [c for c in _KEEP_COLS if c in out.columns]
    return out[avail].reset_index(drop=True)


def _single_arm_design(pid_col: str = "participant_id") -> TrialDesign:
    """Design object for single-arm pre/post datasets."""
    return TrialDesign(
        participant_col=pid_col,
        visit_col="visit",
        arm_col=None,
    )


def _detect_visits(adata) -> tuple[str, str]:
    """Auto-detect (pre, post) visit labels."""
    visits_avail = list(adata.obs["visit"].unique())
    if "Pre" in visits_avail and "Post" in visits_avail:
        return ("Pre", "Post")
    if "Diagnosis" in visits_avail:
        others = [v for v in visits_avail if v != "Diagnosis"]
        return ("Diagnosis", others[0] if others else visits_avail[-1])
    return (visits_avail[0], visits_avail[-1])


def table2_all_results(datasets: Collection[str] | None = None) -> pd.DataFrame:
    """Complete results across all 5 datasets (immune + clinical sigs)."""
    print("  Table 2: Complete results across datasets")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return pd.DataFrame()

    all_results: list[pd.DataFrame] = []

    # ── Sade-Feldman (two-arm DiD) ────────────────────────────────────
    if _selected("Melanoma", datasets):
        try:
            adata_sf = get_sade_feldman()
            if "log1p_tpm" not in adata_sf.layers and "tpm" in adata_sf.layers:
                adata_sf.layers["log1p_tpm"] = np.log1p(adata_sf.layers["tpm"])
            adata_sf, sig_cols = score_signatures(adata_sf, layer="log1p_tpm")
            adata_sf = harmonize_response(adata_sf)

            design = TrialDesign(
                participant_col="participant_id",
                visit_col="visit",
                arm_col="response_harmonized",
                arm_treated="Responder",
                arm_control="Non-responder",
            )
            res = did_table(
                adata_sf, features=sig_cols, design=design,
                visits=("Pre", "Post"), layer="log1p_tpm",
                standardize=True, aggregate="participant_visit",
                use_bootstrap=True, n_boot=999, seed=42,
            )
            res["signature_set"] = _SET_IMMUNE
            res["n_focal"], res["n_reference"] = _paired_group_sizes(adata_sf, design, ("Pre", "Post"))
            res["label"] = res["feature"].apply(sig_display)
            all_results.append(_harmonise(res, "Melanoma", "DiD"))
            print(f"    Melanoma: {len(res)} features (DiD)")
            del adata_sf
            gc.collect()
        except Exception as exc:
            raise RuntimeError("Supplementary Table 2: Melanoma failed") from exc

    # ── TNBC / Zhang (two-arm DiD: anti-PDL1+Chemo vs Chemo) ─────────
    if _selected("TNBC", datasets):
        try:
            adata_tnbc = get_tnbc_zhang()
            adata_tnbc, sig_cols_tnbc = score_signatures(adata_tnbc, layer="log1p_norm")
            try:
                adata_tnbc, sig_cols_clin = score_clinical_signatures(adata_tnbc, layer="log1p_norm")
            except Exception:
                sig_cols_clin = []
            tnbc_design = TrialDesign(
                participant_col="participant_id",
                visit_col="visit",
                arm_col="arm",
                arm_treated="anti-PDL1+Chemo",
                arm_control="Chemo",
            )
            res = _by_signature_set(
                lambda feats: did_table(
                    adata_tnbc, features=feats, design=tnbc_design,
                    visits=("Pre", "Post"), layer="log1p_norm",
                    standardize=True, aggregate="participant_visit",
                    use_bootstrap=True, n_boot=999, seed=42,
                ),
                sig_cols_tnbc, sig_cols_clin,
            )
            res["n_focal"], res["n_reference"] = _paired_group_sizes(
                adata_tnbc, tnbc_design, ("Pre", "Post"))
            res["label"] = res["feature"].apply(sig_display)
            all_results.append(_harmonise(res, "TNBC", "DiD"))
            print(f"    TNBC: {len(res)} features (DiD)")
            del adata_tnbc
            gc.collect()
        except Exception as exc:
            raise RuntimeError("Supplementary Table 2: TNBC failed") from exc

    # ── Stephenson / COVID-19 (cross-sectional Hedges' g at DFO_8-14)
    # Must match Figure 5 exactly: DFO_8-14 bin, log1p_cpm layer,
    # between_arm_comparison + hedges_g on participant-level means.
    if _selected("COVID-19", datasets):
        try:
            from scipy import stats as sp_stats

            adata_st = get_stephenson()
            # Add log1p_cpm layer (matching Figure 5 and Supp Table 3)
            if "log1p_cpm" not in adata_st.layers:
                adata_st = add_log1p_cpm_layer(
                    adata_st, counts_layer="counts", out_layer="log1p_cpm",
                )
            adata_st, sig_cols = score_signatures(adata_st, layer="log1p_cpm")

            # Filter to DFO_8-14 bin (matching Figure 5)
            target_visit = _covid_target_visit(adata_st)
            ad_visit = adata_st[adata_st.obs["dfo_bin"] == target_visit].copy()

            # Compute Hedges' g on participant-level means (matching Figure 5)
            df_agg = (
                ad_visit.obs
                .groupby(["participant_id", "severity"], observed=True)[sig_cols]
                .mean()
                .reset_index()
            )
            g_rows: list[dict] = []
            for feat in sig_cols:
                x_mild = df_agg.loc[df_agg["severity"] == "Mild", feat].dropna().values
                x_sev = df_agg.loc[df_agg["severity"] == "Severe", feat].dropna().values
                if len(x_mild) < 3 or len(x_sev) < 3:
                    continue
                g = hedges_g(x_sev, x_mild)
                n1, n2 = len(x_sev), len(x_mild)
                se_g = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
                _, p = sp_stats.ttest_ind(x_sev, x_mild, equal_var=False)
                g_rows.append({
                    "feature": feat, "signature_set": _SET_IMMUNE,
                    "effect": g, "se": se_g, "pvalue": p,
                    "ci_lower": g - 1.96 * se_g, "ci_upper": g + 1.96 * se_g,
                    "n_units": n1 + n2, "n_focal": n1, "n_reference": n2,
                    "cov_type_used": "hedges_g",
                })
            if g_rows:
                res_covid = pd.DataFrame(g_rows)
                # Drop NaN p-values before BH correction (degenerate signatures
                # with zero variance cause NaN from ttest_ind; a single NaN
                # propagates to all FDR values in multipletests).
                valid_p = res_covid["pvalue"].dropna()
                res_covid["FDR"] = np.nan
                if len(valid_p) > 0:
                    _, fdr, _, _ = multipletests(valid_p, method="fdr_bh")
                    res_covid.loc[valid_p.index, "FDR"] = fdr
                res_covid["label"] = res_covid["feature"].apply(sig_display)
                all_results.append(
                    _harmonise(res_covid, "COVID-19", f"Hedges' g ({target_visit})")
                )
                print(f"    COVID-19: {len(res_covid)} features "
                      f"(Hedges' g, {target_visit})")
            del adata_st, ad_visit
            gc.collect()
        except Exception as exc:
            raise RuntimeError("Supplementary Table 2: COVID-19 failed") from exc

    # ── Single-arm datasets: Vaccine, AML, CAR-T ─────────────────────
    single_arm = [
        ("Vaccine", get_vaccine, ("Pre", "Post")),
        ("AML", get_aml, None),
        ("CAR-T", get_cart, None),
    ]
    for ds_name, loader, visits_override in single_arm:
        if not _selected(ds_name, datasets):
            continue
        try:
            adata = loader()
            # Score both immune and clinical signatures
            adata, sig_cols_imm = score_signatures(adata)
            try:
                adata, sig_cols_clin = score_clinical_signatures(adata)
            except Exception:
                sig_cols_clin = []

            visits = visits_override or _detect_visits(adata)
            design = _single_arm_design()

            res = _by_signature_set(
                lambda feats, adata=adata, design=design, visits=visits: within_arm_comparison(
                    adata, arm="All", features=feats,
                    design=design, visits=visits, standardize=True,
                    use_bootstrap=True, n_boot=999, seed=42,
                ),
                sig_cols_imm, sig_cols_clin,
            )
            res["label"] = res["feature"].apply(sig_display)
            all_results.append(_harmonise(res, ds_name, "within-arm Δ"))
            print(f"    {ds_name}: {len(res)} features (within-arm Δ)")
            del adata
            gc.collect()
        except Exception as exc:
            raise RuntimeError(f"Supplementary Table 2: {ds_name} failed") from exc

    if not all_results:
        print("    No results generated")
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)
    path = SUPP_OUTPUT / "Supp_Table2_all_results.csv"
    df.to_csv(path, index=False)
    _table2_fingerprint_path().write_text(STATS_FINGERPRINT)
    print(f"    Saved {path.name} ({len(df)} rows)")
    return df


def _table2_fingerprint_path():
    """Sidecar recording the estimator code that produced Supp Table 2."""
    return SUPP_OUTPUT / "Supp_Table2_all_results.stats_fingerprint"


# ======================================================================
# Table 3 — GSEA results (per-dataset sheets)
# ======================================================================


def _stephenson_ranking() -> pd.Series:
    """Compute per-gene Welch t-stat ranking (severe vs mild) for Stephenson."""
    from scipy import stats as sp_stats

    adata_st = get_stephenson()
    pid_col, sev_col = "participant_id", "severity"

    layer_st = "log1p_cpm" if "log1p_cpm" in adata_st.layers else None
    expr = adata_st.layers[layer_st] if layer_st else adata_st.X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()

    expr_df = pd.DataFrame(expr, columns=adata_st.var_names)
    expr_df[pid_col] = adata_st.obs[pid_col].values
    expr_df[sev_col] = adata_st.obs[sev_col].values
    pb = expr_df.groupby(pid_col).agg(
        {sev_col: "first", **{g: "mean" for g in adata_st.var_names}}
    )

    sev_vals = pb[sev_col].unique()
    severe = "severe" if "severe" in sev_vals else sev_vals[0]
    mild = "mild" if "mild" in sev_vals else sev_vals[-1]
    grp_s = pb[pb[sev_col] == severe]
    grp_m = pb[pb[sev_col] == mild]

    ranking = {}
    for gene in adata_st.var_names:
        xs = grp_s[gene].dropna().values
        xm = grp_m[gene].dropna().values
        if len(xs) < 3 or len(xm) < 3:
            continue
        t, _ = sp_stats.ttest_ind(xs, xm, equal_var=False)
        if np.isfinite(t):
            ranking[gene] = t

    del adata_st
    gc.collect()
    return pd.Series(ranking).sort_values(ascending=False)


def table3_gsea_results(datasets: Collection[str] | None = None) -> dict[str, pd.DataFrame]:
    """Run GSEA pre-ranked analysis for all 5 datasets, save as multi-sheet Excel.

    Uses shared ``load_or_run_gsea_did`` / ``load_or_run_gsea_within_arm``
    helpers that cache results under ``manuscript/GSEA/{dataset}/{library}/``.
    If cached CSV files already exist they are loaded instantly; otherwise
    GSEA is run fresh and results are saved for next time.
    """
    print("  Table 3: GSEA results (per-dataset)")

    sheets: dict[str, pd.DataFrame] = {}

    # ── Sade-Feldman (two-arm DiD) ────────────────────────────────────
    if _selected("Melanoma", datasets):
        try:
            adata_sf = get_sade_feldman()
            if "log1p_tpm" not in adata_sf.layers and "tpm" in adata_sf.layers:
                adata_sf.layers["log1p_tpm"] = np.log1p(adata_sf.layers["tpm"])
            adata_sf = harmonize_response(adata_sf)
            design_sf = TrialDesign(
                participant_col="participant_id", visit_col="visit",
                arm_col="response_harmonized",
                arm_treated="Responder", arm_control="Non-responder",
            )
            res = load_or_run_gsea_did(
                adata_sf, design_sf, ("Pre", "Post"), "log1p_tpm", "Melanoma",
            )
            if res is not None:
                sheets["Melanoma"] = res
            del adata_sf
            gc.collect()
        except Exception as exc:
            raise RuntimeError("Supplementary Table 3: Melanoma GSEA failed") from exc

    # ── TNBC / Zhang (two-arm DiD: anti-PDL1+Chemo vs Chemo) ─────────
    if _selected("TNBC", datasets):
        try:
            adata_tnbc = get_tnbc_zhang()
            tnbc_design = TrialDesign(
                participant_col="participant_id",
                visit_col="visit",
                arm_col="arm",
                arm_treated="anti-PDL1+Chemo",
                arm_control="Chemo",
            )
            res = load_or_run_gsea_did(
                adata_tnbc, tnbc_design, ("Pre", "Post"), "log1p_norm", "TNBC",
            )
            if res is not None:
                sheets["TNBC"] = res
            del adata_tnbc
            gc.collect()
        except Exception as exc:
            raise RuntimeError("Supplementary Table 3: TNBC GSEA failed") from exc

    # ── COVID-19 (cross-sectional: severe vs mild) ───────────────────
    # Must match Figure 5: DFO_8-14 bin, log1p_cpm layer
    if _selected("COVID-19", datasets):
        try:
            adata_st = get_stephenson()
            if "log1p_cpm" not in adata_st.layers:
                from .._shared import add_log1p_cpm_layer
                adata_st = add_log1p_cpm_layer(
                    adata_st, counts_layer="counts", out_layer="log1p_cpm",
                )
            adata_st, _ = score_signatures(adata_st, layer="log1p_cpm")
            # Fixed DFO bin to match Figure 5 COVID analysis
            target_visit = _covid_target_visit(adata_st)
            covid_design = TrialDesign(
                participant_col="participant_id",
                visit_col="dfo_bin",
                arm_col="severity",
                arm_treated="Severe",
                arm_control="Mild",
            )
            res = load_or_run_gsea_cross_sectional(
                adata_st, covid_design, visit=target_visit,
                layer="log1p_cpm", dataset_name="COVID-19",
            )
            if res is not None:
                sheets["COVID-19"] = res
            del adata_st
            gc.collect()
        except Exception as exc:
            raise RuntimeError("Supplementary Table 3: COVID-19 GSEA failed") from exc

    # ── Single-arm datasets (within-arm Pre→Post) ──────────────────
    single_arm_gsea = [
        ("Vaccine", get_vaccine, ("Pre", "Post"), None, "Treated"),
        ("AML", get_aml, None, None, "Treatment"),
        ("CAR-T", get_cart, None, None, "CAR-T"),
    ]
    for ds_name, loader, visits_override, _layer_hint, arm_label in single_arm_gsea:
        if not _selected(ds_name, datasets):
            continue
        try:
            adata = loader()
            # Pick best available normalized layer
            layer = _layer_hint
            if layer is None:
                for _cand in ("log1p_tpm", "log1p_cpm", "log1p_norm"):
                    if _cand in adata.layers:
                        layer = _cand
                        break
            # layer=None means use .X (already normalized for Vaccine)
            if "arm" not in adata.obs.columns:
                adata.obs["arm"] = arm_label
            visits = visits_override or _detect_visits(adata)
            pid_col = (
                "participant_id"
                if "participant_id" in adata.obs.columns
                else "patient_id"
            )
            gsea_design = TrialDesign(
                participant_col=pid_col, visit_col="visit",
                arm_col="arm", arm_treated=arm_label, arm_control=arm_label,
            )
            res = load_or_run_gsea_within_arm(
                adata, gsea_design, arm=arm_label, visits=visits,
                layer=layer, dataset_name=ds_name,
            )
            if res is not None:
                sheets[ds_name] = res
            del adata
            gc.collect()
        except Exception as exc:
            raise RuntimeError(f"Supplementary Table 3: {ds_name} GSEA failed") from exc

    # Save as multi-sheet Excel
    if sheets:
        xlsx_path = SUPP_OUTPUT / "Supp_Table3_GSEA_results.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"    Saved {xlsx_path.name} ({len(sheets)} sheets)")
    else:
        print("    No GSEA results generated")

    return sheets


# ======================================================================
# Table 4 — Permutation test results
# ======================================================================


def _covid_target_visit(adata_st) -> str:
    """Days-from-onset bin used by every COVID-19 analysis (Figure 5, Supp Tables 2-4).

    DFO_8-14 when present; otherwise the first bin, in sorted order, holding both
    severity groups.
    """
    target_visit = "DFO_8-14"
    if "dfo_bin" in adata_st.obs.columns:
        available_bins = sorted(adata_st.obs["dfo_bin"].dropna().unique())
        if target_visit not in available_bins:
            for _bin in available_bins:
                _sub = adata_st[adata_st.obs["dfo_bin"] == _bin]
                if set(_sub.obs["severity"].unique()) >= {"Mild", "Severe"}:
                    return _bin
    return target_visit


def _permutation_rows(dataset: str, nulls: dict, observed: dict, exact: bool) -> list[dict]:
    """One Supplementary Table 4 row per feature with a finite observed statistic."""
    rows = []
    for feat, null in nulls.items():
        obs_beta = observed.get(feat, np.nan)
        if not np.isfinite(obs_beta):
            continue
        rows.append({"dataset": dataset, "feature": feat, "label": sig_display(feat),
                     "observed_beta": obs_beta,
                     **permutation_summary(null, obs_beta, exact=exact)})
    return rows


def table4_permutation_results(datasets: Collection[str] | None = None) -> pd.DataFrame:
    """Permutation test results for all signatures across all datasets.

    Each dataset uses the routine of ``manuscript_figures._relabeling``, the one behind
    Figure 5: group labels are reassigned across participants (melanoma, TNBC,
    COVID-19) or the two visits are swapped within participants (vaccine, AML,
    CAR-T). All assignments are enumerated when there are at most ``MAX_EXACT`` of
    them, and the p-value is then exact; otherwise ``N_DRAWS`` seeded draws are used.
    The column ``exact`` records which. The statistic is the standardized coefficient
    of the participant-level model of each design.

    Parameters
    ----------
    datasets : collection of str, optional
        Datasets to include (default all). A selected dataset that fails raises.
    """
    print("  Table 4: Permutation test results")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return pd.DataFrame()

    rows: list[dict] = []

    def _report(name: str, exact: bool, extra: str = "") -> None:
        mine = [r for r in rows if r["dataset"] == name]
        n_assign = mine[0]["n_permutations"] if mine else 0
        print(f"    {name}{extra}: {len(mine)} features, {n_assign:,} assignments "
              f"({'all enumerated' if exact else 'random draws'})")

    # ── Two-group DiD: reassign the group labels across participants ──
    two_group = [
        ("Melanoma", "melanoma", "log1p_tpm", TrialDesign(
            participant_col="participant_id", visit_col="visit",
            arm_col="response_harmonized", arm_treated="Responder",
            arm_control="Non-responder")),
        ("TNBC", "tnbc", "log1p_norm", TrialDesign(
            participant_col="participant_id", visit_col="visit", arm_col="arm",
            arm_treated="anti-PDL1+Chemo", arm_control="Chemo")),
    ]
    for ds_name, tag, layer, design in two_group:
        if not _selected(ds_name, datasets):
            continue
        try:
            if ds_name == "Melanoma":
                adata = get_sade_feldman()
                if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
                    adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
                adata, sig_cols = score_signatures(adata, layer=layer)
                adata = harmonize_response(adata)
            else:
                adata = get_tnbc_zhang()
                adata, sig_cols = score_signatures(adata, layer=layer)
            with _quiet():
                relabel = did_relabeling_cached(adata, sig_cols, design, ("Pre", "Post"),
                                                layer=layer, tag=tag)
            pairs = did_permutation_nulls(relabel)
            rows.extend(_permutation_rows(
                ds_name, {f: null for f, (null, _obs) in pairs.items()},
                {f: obs for f, (_null, obs) in pairs.items()}, bool(relabel.attrs["exact"])))
            _report(ds_name, bool(relabel.attrs["exact"]))
            del adata
            gc.collect()
        except Exception as exc:
            raise RuntimeError(f"Supplementary Table 4: {ds_name} permutation failed") from exc

    # ── Stephenson (cross-sectional: reassign Severe/Mild across participants) ────
    # The observed statistic and every null replicate are the same estimand: the
    # standardised between-arm OLS beta on participant-level means in the DFO_8-14
    # bin of the log1p_cpm layer (the bin and layer of Supp Tables 2-3). This beta is
    # not the Hedges' g that Supp Table 2 reports for COVID-19, so the two tables differ
    # slightly for this dataset.
    if _selected("COVID-19", datasets):
        try:
            adata_st = get_stephenson()
            if "log1p_cpm" not in adata_st.layers:
                adata_st = add_log1p_cpm_layer(
                    adata_st, counts_layer="counts", out_layer="log1p_cpm",
                )
            adata_st, sig_cols_st = score_signatures(adata_st, layer="log1p_cpm")
            target_visit = _covid_target_visit(adata_st)

            design_st = TrialDesign(
                participant_col="participant_id",
                visit_col="dfo_bin",
                arm_col="severity",
                arm_treated="Severe",
                arm_control="Mild",
            )
            df_cs = get_between_arm_aggregated_df(
                adata_st, sig_cols_st, design_st, target_visit, layer="log1p_cpm",
            )
            observed = {feat: between_arm_fit_beta(df_cs, feat, design_st) for feat in sig_cols_st}
            with _quiet():
                nulls, exact = between_arm_nulls(df_cs, sig_cols_st, design_st)
            rows.extend(_permutation_rows("COVID-19", nulls, observed, exact))
            _report("COVID-19", exact, f" ({target_visit}, n={df_cs['participant_id'].nunique()})")
            del adata_st, df_cs
            gc.collect()
        except Exception as exc:
            raise RuntimeError("Supplementary Table 4: COVID-19 permutation failed") from exc

    # ── Single-arm datasets: swap the two visits within participants ──
    single_arm = [
        ("Vaccine", get_vaccine, ("Pre", "Post")),
        ("AML", get_aml, None),
        ("CAR-T", get_cart, None),
    ]
    for ds_name, loader, visits_override in single_arm:
        if not _selected(ds_name, datasets):
            continue
        try:
            adata = loader()
            adata, sig_cols_imm = score_signatures(adata)
            visits = visits_override or _detect_visits(adata)
            design = _single_arm_design()

            obs_res = within_arm_comparison(
                adata, arm="All", features=sig_cols_imm,
                design=design, visits=visits, standardize=True,
            )
            observed = dict(zip(obs_res["feature"], obs_res["beta_time"]))

            with _quiet():
                df_use, unit = get_within_arm_aggregated_df(
                    adata, "All", sig_cols_imm, design, visits,
                    layer=None, aggregate="participant_visit",
                )
                nulls, exact = within_arm_nulls(df_use, sig_cols_imm, unit)
            rows.extend(_permutation_rows(ds_name, nulls, observed, exact))
            _report(ds_name, exact)
            del adata, df_use
            gc.collect()
        except Exception as exc:
            raise RuntimeError(f"Supplementary Table 4: {ds_name} permutation failed") from exc

    if not rows:
        print("    No permutation results generated")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # FDR within each dataset, over its signatures
    df["permutation_FDR"] = np.nan
    for ds in df["dataset"].unique():
        valid = df.loc[df["dataset"] == ds, "permutation_p"].dropna()
        if len(valid):
            df.loc[valid.index, "permutation_FDR"] = multipletests(valid, method="fdr_bh")[1]

    path = SUPP_OUTPUT / "Supp_Table4_permutation_results.csv"
    df.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(df)} rows)")
    return df


# ======================================================================
# Table 5 — Power analysis results
# ======================================================================


def table5_power_analysis() -> pd.DataFrame:
    """Power analysis for each dataset and signature, from the estimates of Table 2.

    Each row uses the formula of its design. Two-group rows (the DiD cohorts and the
    cross-sectional COVID-19 comparison) use the group sizes of the cohort, so an
    unequal allocation is not treated as balanced; single-arm rows use the paired
    formula. The standard deviation is the one that reproduces the observed standard
    error at the observed sizes. The minimum N is the total number of participants
    at the observed allocation.
    """
    print("  Table 5: Power analysis results")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return pd.DataFrame()

    from sctrial.stats.power import (
        power_paired,
        power_two_group,
        sample_size_paired,
        sample_size_two_group,
        sensitivity_paired,
        sensitivity_two_group,
    )

    # Read observed results from Table 2, regenerating it unless it was written by
    # the current estimator code (a WLS-era Table 2 would give stale DiD power rows).
    table2_path = SUPP_OUTPUT / "Supp_Table2_all_results.csv"
    stamp = _table2_fingerprint_path()
    fresh = (table2_path.exists() and stamp.exists()
             and stamp.read_text().strip() == STATS_FINGERPRINT)
    if fresh and not {"n_focal", "n_reference"} <= set(pd.read_csv(table2_path, nrows=0).columns):
        fresh = False
    if not fresh:
        print("    Table 2 missing or from other estimator code, generating first...")
        table2_all_results()
    t2 = pd.read_csv(table2_path)

    rows: list[dict] = []
    for _, row in t2.iterrows():
        effect = row.get("effect", np.nan)
        se = row.get("se", np.nan)
        n_units = row.get("n_units", np.nan)
        if not np.isfinite(effect) or not np.isfinite(se) or not np.isfinite(n_units):
            continue

        feature = row["feature"]
        n = int(n_units)
        n_focal, n_reference = row.get("n_focal", np.nan), row.get("n_reference", np.nan)
        two_group = bool(np.isfinite(n_focal) and np.isfinite(n_reference))
        abs_effect = abs(effect)

        if two_group:
            n1, n2 = int(n_focal), int(n_reference)
            if n1 + n2 != n:
                raise ValueError(f"{row['dataset']} {feature}: group sizes {n1} + {n2} != n_units {n}")
            # SE = sigma * sqrt(1/n1 + 1/n2)
            sigma_est = max(se / np.sqrt(1.0 / n1 + 1.0 / n2), 0.01)
            pwr = power_two_group(n1, n2, abs_effect, sigma=sigma_est)
            mde = sensitivity_two_group(n1, n2, sigma=sigma_est, power=0.80)
            min_n = (sum(sample_size_two_group(abs_effect, sigma=sigma_est, power=0.80,
                                               allocation_ratio=n1 / n2))
                     if abs_effect >= 1e-10 else None)
        else:
            n1 = n2 = None
            # SE = sigma * sqrt(2/n)
            sigma_est = max(se * np.sqrt(n / 2), 0.01)
            pwr = power_paired(n_participants=max(n, 2), effect_size=abs_effect, sigma=sigma_est)
            mde = sensitivity_paired(n_participants=max(n, 2), sigma=sigma_est, power=0.80)
            min_n = (sample_size_paired(effect_size=abs_effect, sigma=sigma_est, power=0.80)
                     if abs_effect >= 1e-10 else None)

        rows.append({
            "dataset": row["dataset"],
            "feature": feature,
            "label": row.get("label", sig_display(feature)),
            "estimand": row.get("estimand", ""),
            "observed_effect": effect,
            "observed_se": se,
            "n_participants": n,
            "n_focal": n1,
            "n_reference": n2,
            "sigma_estimated": sigma_est,
            "power_at_observed_N": pwr,
            "min_N_for_80pct_power": min_n,
            "min_detectable_effect_80pct": mde,
        })

    df = pd.DataFrame(rows)
    for col in ("n_focal", "n_reference"):
        df[col] = df[col].astype("Int64")
    path = SUPP_OUTPUT / "Supp_Table5_power_analysis.csv"
    df.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(df)} rows)")
    return df


# ======================================================================
# Table 7 — Gene-level DiD results (Sade-Feldman)
# ======================================================================


def table7_gene_level_results() -> pd.DataFrame:
    """Gene-level DiD results for the melanoma cohort.

    Runs DiD on every expressed gene (not just signatures) to provide
    the full genome-wide results for reproducibility.
    """
    print("  Table 7: Gene-level DiD results (Melanoma)")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return pd.DataFrame()

    try:
        adata_sf = get_sade_feldman()
        if "log1p_tpm" not in adata_sf.layers and "tpm" in adata_sf.layers:
            adata_sf.layers["log1p_tpm"] = np.log1p(adata_sf.layers["tpm"])
        adata_sf = harmonize_response(adata_sf)

        design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="response_harmonized",
            arm_treated="Responder",
            arm_control="Non-responder",
        )

        # Use all genes (var_names)
        all_genes = list(adata_sf.var_names)
        print(f"    Running DiD on {len(all_genes)} genes...")

        res = did_table(
            adata_sf, features=all_genes, design=design,
            visits=("Pre", "Post"), layer="log1p_tpm",
            standardize=True, aggregate="participant_visit",
        )
        # Add CIs from effect ± 1.96 × SE
        if "beta_DiD" in res.columns and "se_DiD" in res.columns:
            res["ci_lower"] = res["beta_DiD"] - 1.96 * res["se_DiD"]
            res["ci_upper"] = res["beta_DiD"] + 1.96 * res["se_DiD"]

        # Sort by p-value
        if "p_DiD" in res.columns:
            res = res.sort_values("p_DiD").reset_index(drop=True)

        # Put feature (gene name) as first column for readability
        if "feature" in res.columns:
            cols = ["feature"] + [c for c in res.columns if c != "feature"]
            res = res[cols]

        path = SUPP_OUTPUT / "Supp_Table7_gene_level_DiD_Melanoma.csv"
        res.to_csv(path, index=False)
        n_sig = (res["FDR_DiD"] < 0.05).sum() if "FDR_DiD" in res.columns else 0
        n_nom = (res["p_DiD"] < 0.05).sum() if "p_DiD" in res.columns else 0
        print(f"    Saved {path.name} ({len(res)} genes, {n_nom} nominally significant, {n_sig} FDR < 0.05)")

        del adata_sf
        gc.collect()
        return res
    except Exception as exc:
        raise RuntimeError("Supplementary Table 7: melanoma gene-level DiD failed") from exc


# ======================================================================
# Table 8 — Dataset metadata summary
# ======================================================================


def table8_dataset_metadata(datasets: Collection[str] | None = None) -> pd.DataFrame:
    """Per-participant metadata across all datasets.

    Columns: dataset, participant_id, arm/condition, visit, n_cells,
    response_status (where applicable).
    """
    print("  Table 8: Dataset metadata summary")

    rows: list[dict] = []

    datasets_info = [
        ("Melanoma", get_sade_feldman, "participant_id", "visit", "response_harmonized"),
        ("TNBC", get_tnbc_zhang, "participant_id", "visit", "arm"),
        ("COVID-19", get_stephenson, "participant_id", None, "severity"),
        ("Vaccine", get_vaccine, "participant_id", "visit", None),
        ("AML", get_aml, "participant_id", "visit", None),
        ("CAR-T", get_cart, "participant_id", "visit", None),
    ]

    for ds_name, loader, pid_col, visit_col, condition_col in datasets_info:
        if not _selected(ds_name, datasets):
            continue
        try:
            adata = loader()

            # Harmonize response for Melanoma (Sade-Feldman)
            if ds_name == "Melanoma" and condition_col == "response_harmonized":
                adata = harmonize_response(adata)

            obs = adata.obs.copy()

            # Build grouping columns
            group_cols = [pid_col]
            if visit_col and visit_col in obs.columns:
                group_cols.append(visit_col)

            # observed=True: participant and visit are Categorical, and without it the
            # groupby emits every participant x visit combination, including ones with
            # no cells (n_cells=0 rows for samples that do not exist).
            grouped = obs.groupby(group_cols, observed=True).size().reset_index(name="n_cells")
            grouped["dataset"] = ds_name

            # Add condition/response
            if condition_col and condition_col in obs.columns:
                cond_map = obs.groupby(pid_col, observed=True)[condition_col].first()
                grouped["condition"] = grouped[pid_col].map(cond_map)
            else:
                grouped["condition"] = None

            # Rename columns
            grouped = grouped.rename(columns={
                pid_col: "participant_id",
                visit_col: "visit" if visit_col else "visit",
            })
            if "visit" not in grouped.columns:
                grouped["visit"] = "cross-sectional"

            # Get cell type counts per participant-visit
            if "cell_type" in obs.columns:
                ct_col = "cell_type"
            elif "celltype" in obs.columns:
                ct_col = "celltype"
            else:
                ct_col = None

            if ct_col:
                n_celltypes = (
                    obs.groupby(group_cols, observed=True)[ct_col]
                    .nunique().reset_index(name="n_cell_types")
                )
                if visit_col and visit_col in n_celltypes.columns:
                    grouped = grouped.merge(n_celltypes, on=[pid_col if pid_col in n_celltypes.columns else "participant_id",
                                                              visit_col if visit_col in n_celltypes.columns else "visit"],
                                            how="left")
                else:
                    grouped = grouped.merge(n_celltypes, on=[pid_col if pid_col in n_celltypes.columns else "participant_id"],
                                            how="left")

            # Select final columns
            keep = ["dataset", "participant_id", "visit", "condition", "n_cells"]
            if "n_cell_types" in grouped.columns:
                keep.append("n_cell_types")
            avail = [c for c in keep if c in grouped.columns]
            rows.append(grouped[avail])

            n_parts = grouped["participant_id"].nunique()
            print(f"    {ds_name}: {n_parts} participants, {grouped['n_cells'].sum():,} cells")
            del adata
            gc.collect()
        except Exception as exc:
            raise RuntimeError(f"Supplementary Table 8: {ds_name} metadata failed") from exc

    if not rows:
        print("    No metadata generated")
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    path = SUPP_OUTPUT / "Supp_Table8_dataset_metadata.csv"
    df.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(df)} rows)")
    return df


# ======================================================================
# Table 6 — Gene-level DiD results (TNBC / Zhang)
# ======================================================================


def table6_gene_level_tnbc() -> pd.DataFrame:
    """Gene-level DiD results for the TNBC cohort (Zhang et al.).

    Runs DiD on every expressed gene (not just signatures) to provide
    the full genome-wide results for reproducibility.
    """
    print("  Table 6: Gene-level DiD results (TNBC)")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return pd.DataFrame()

    try:
        adata_tnbc = get_tnbc_zhang()

        tnbc_design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="anti-PDL1+Chemo",
            arm_control="Chemo",
        )

        all_genes = list(adata_tnbc.var_names)
        print(f"    Running DiD on {len(all_genes)} genes...")

        res = did_table(
            adata_tnbc, features=all_genes, design=tnbc_design,
            visits=("Pre", "Post"), layer="log1p_norm",
            standardize=True, aggregate="participant_visit",
        )
        if "beta_DiD" in res.columns and "se_DiD" in res.columns:
            res["ci_lower"] = res["beta_DiD"] - 1.96 * res["se_DiD"]
            res["ci_upper"] = res["beta_DiD"] + 1.96 * res["se_DiD"]

        if "p_DiD" in res.columns:
            res = res.sort_values("p_DiD").reset_index(drop=True)

        if "feature" in res.columns:
            cols = ["feature"] + [c for c in res.columns if c != "feature"]
            res = res[cols]

        path = SUPP_OUTPUT / "Supp_Table6_gene_level_DiD_TNBC.csv"
        res.to_csv(path, index=False)
        n_sig = (res["FDR_DiD"] < 0.05).sum() if "FDR_DiD" in res.columns else 0
        n_nom = (res["p_DiD"] < 0.05).sum() if "p_DiD" in res.columns else 0
        print(f"    Saved {path.name} ({len(res)} genes, {n_nom} nominally significant, {n_sig} FDR < 0.05)")

        del adata_tnbc
        gc.collect()
        return res
    except Exception as exc:
        raise RuntimeError("Supplementary Table 6: TNBC gene-level DiD failed") from exc


# ======================================================================
# Patch Table 1 — add "Genes Present" column per dataset
# ======================================================================


def patch_table1_genes_present(datasets: Collection[str] | None = None) -> pd.DataFrame:
    """Add a 'Genes Present' column to Supp Table 1 showing how many
    genes from each signature are actually found in each dataset."""
    print("  Patching Table 1: Adding Genes Present per dataset")

    t1_path = SUPP_OUTPUT / "Supp_Table1_gene_signatures.csv"
    if not t1_path.exists():
        print("    Table 1 not found, generating first...")
        table1_gene_signatures()
    t1 = pd.read_csv(t1_path)

    datasets_loaders = [
        ("Melanoma", get_sade_feldman),
        ("TNBC", get_tnbc_zhang),
        ("COVID-19", get_stephenson),
        ("Vaccine", get_vaccine),
        ("AML", get_aml),
        ("CAR-T", get_cart),
    ]

    for ds_name, loader in datasets_loaders:
        if not _selected(ds_name, datasets):
            continue
        try:
            adata = loader()
            var_names_set = set(adata.var_names)
            col_name = f"Genes_Present_{ds_name}"

            present_counts = []
            for _, row in t1.iterrows():
                genes = [g.strip() for g in row["Genes"].split(",")]
                present = [g for g in genes if g in var_names_set]
                present_counts.append(f"{len(present)}/{len(genes)}")
            t1[col_name] = present_counts
            del adata
            gc.collect()
        except Exception as exc:
            raise RuntimeError(f"Supplementary Table 1: {ds_name} genes present failed") from exc

    t1.to_csv(t1_path, index=False)

    # Save xlsx with Genes_Present columns explicitly formatted as Text
    # so Excel doesn't auto-convert fractions like "11/11" to dates.
    xlsx_path = t1_path.with_suffix(".xlsx")
    t1.to_excel(xlsx_path, index=False, engine="openpyxl")

    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path)
    ws = wb.active
    genes_present_cols = [
        cell.column for cell in ws[1]
        if cell.value and str(cell.value).startswith("Genes_Present_")
    ]
    for col_idx in genes_present_cols:
        for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
            for cell in row:
                cell.number_format = "@"
                cell.value = str(cell.value) if cell.value is not None else ""
    wb.save(xlsx_path)

    print(f"    Updated {t1_path.name} + {xlsx_path.name}")
    return t1


# ======================================================================
# Main entry point
# ======================================================================


def generate(tables: str = "all"):
    """Generate supplementary tables.

    Parameters
    ----------
    tables : str
        Which tables to generate: "all", "1-3", "4-6", "7-8",
        or a comma-separated list like "2,7,8".
    """
    print("=" * 60)
    print("Supplementary Tables")
    print("=" * 60)

    to_run = set()
    if tables == "all":
        to_run = {1, 2, 3, 4, 5, 6, 7, 8}
    elif tables == "1-3":
        to_run = {1, 2, 3}
    elif tables == "4-6":
        to_run = {4, 5, 6}
    elif tables == "7-8":
        to_run = {7, 8}
    else:
        to_run = {int(x.strip()) for x in tables.split(",")}

    if 1 in to_run:
        table1_gene_signatures()
    if 2 in to_run:
        table2_all_results()
    if 3 in to_run:
        table3_gsea_results()
    if 4 in to_run:
        table4_permutation_results()
    if 5 in to_run:
        table5_power_analysis()
    if 6 in to_run:
        table6_gene_level_tnbc()
    if 7 in to_run:
        table7_gene_level_results()
    if 8 in to_run:
        table8_dataset_metadata()

    # Patches
    if 1 in to_run:
        patch_table1_genes_present()

    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    import sys
    apply_style()
    tables_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    generate(tables=tables_arg)
