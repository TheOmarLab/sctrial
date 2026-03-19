"""
Supplementary Tables 1–7.
=========================

Table 1  Gene signature definitions (name, gene count, genes).
Table 2  Complete effect-size results across all signatures and datasets.
Table 3  GSEA pre-ranked results (one sheet per dataset).
Table 4  Permutation test results.
Table 5  Power analysis results.
Table 6  Gene-level DiD results (Sade-Feldman).
Table 7  Dataset metadata summary.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pickle as pkl
import warnings

from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests


from .._shared import (
    CLINICAL_SIGNATURES,
    GENE_SIGNATURES,
    SCTRIAL_AVAILABLE,
    SIGNATURE_DISPLAY_NAMES,
    SUPP_OUTPUT,
    TrialDesign,
    apply_style,
    clear_cache,
    did_table,
    within_arm_fit_beta,
    get_within_arm_aggregated_df,
    did_fit,
    get_did_aggregated_df,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    get_aml,
    get_cart,
    load_or_run_gsea_did,
    load_or_run_gsea_cross_sectional,
    load_or_run_gsea_prerank,
    score_clinical_signatures,
    score_signatures,
    sig_display,
    within_arm_comparison,
)

# ======================================================================
# Table 1 — Gene signature definitions
# ======================================================================


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
    "dataset", "estimand", "label", "feature",
    "effect", "se", "pvalue", "FDR", "ci_lower", "ci_upper",
    "n_units", "cov_type_used",
]

_COL_MAP = {
    "beta_time": "effect",
    "se_time": "se",
    "p_time": "pvalue",
    "FDR_time": "FDR",
    "ci_lo_time": "ci_lower",
    "ci_hi_time": "ci_upper",
    "beta_DiD": "effect",
    "se_DiD": "se",
    "p_DiD": "pvalue",
    "FDR_DiD": "FDR",
}


def _harmonise(df: pd.DataFrame, dataset: str, estimand: str) -> pd.DataFrame:
    """Rename columns to a common schema and tag with dataset/estimand."""
    out = df.rename(columns=_COL_MAP).copy()
    # Drop any duplicate columns created by rename
    out = out.loc[:, ~out.columns.duplicated()]
    out["dataset"] = dataset
    out["estimand"] = estimand
    if "label" not in out.columns:
        out["label"] = out["feature"].apply(sig_display)
    avail = [c for c in _KEEP_COLS if c in out.columns]
    result = out[avail].reset_index(drop=True)

    # Fill missing CIs from effect ± 1.96 × SE (normal approximation)
    if "ci_lower" in result.columns and "ci_upper" in result.columns:
        mask = result["ci_lower"].isna() & result["effect"].notna() & result["se"].notna()
        if mask.any():
            result.loc[mask, "ci_lower"] = result.loc[mask, "effect"] - 1.96 * result.loc[mask, "se"]
            result.loc[mask, "ci_upper"] = result.loc[mask, "effect"] + 1.96 * result.loc[mask, "se"]
    elif "effect" in result.columns and "se" in result.columns:
        result["ci_lower"] = result["effect"] - 1.96 * result["se"]
        result["ci_upper"] = result["effect"] + 1.96 * result["se"]

    return result


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


def table2_all_results() -> pd.DataFrame:
    """Complete results across all 5 datasets (immune + clinical sigs)."""
    print("  Table 2: Complete results across datasets")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return pd.DataFrame()

    all_results: list[pd.DataFrame] = []

    # ── Sade-Feldman (two-arm DiD) ────────────────────────────────────
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
        )
        res["label"] = res["feature"].apply(sig_display)
        all_results.append(_harmonise(res, "Melanoma", "DiD"))
        print(f"    Melanoma: {len(res)} features (DiD)")
        del adata_sf
        gc.collect()
    except Exception as exc:
        print(f"    Melanoma failed: {exc}")

    # ── Stephenson (cross-sectional Hedges' g) ────────────────────────
    try:
        from scipy import stats as sp_stats
        from statsmodels.stats.multitest import multipletests

        adata_st = get_stephenson()
        adata_st, sig_cols = score_signatures(adata_st)

        pid_col, sev_col = "participant_id", "severity"
        pb = adata_st.obs[[pid_col, sev_col]].copy()
        for c in sig_cols:
            if c in adata_st.obs.columns:
                pb[c] = adata_st.obs[c].values
        pb_mean = pb.groupby(pid_col).agg(
            {sev_col: "first", **{c: "mean" for c in sig_cols}}
        )

        sev_vals = pb_mean[sev_col].unique()
        severe = "severe" if "severe" in sev_vals else sev_vals[0]
        mild = "mild" if "mild" in sev_vals else sev_vals[-1]
        grp_s = pb_mean[pb_mean[sev_col] == severe]
        grp_m = pb_mean[pb_mean[sev_col] == mild]

        rows_st: list[dict] = []
        for feat in sig_cols:
            x_s = grp_s[feat].dropna().values
            x_m = grp_m[feat].dropna().values
            if len(x_s) < 2 or len(x_m) < 2:
                continue
            n_s, n_m = len(x_s), len(x_m)
            pooled_sd = np.sqrt(
                ((n_s - 1) * x_s.var(ddof=1) + (n_m - 1) * x_m.var(ddof=1))
                / (n_s + n_m - 2)
            )
            g = (x_s.mean() - x_m.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            g *= 1 - 3 / (4 * (n_s + n_m) - 9)
            se_g = np.sqrt((n_s + n_m) / (n_s * n_m) + g**2 / (2 * (n_s + n_m)))
            _, p = sp_stats.mannwhitneyu(x_s, x_m, alternative="two-sided")
            rows_st.append({
                "feature": feat, "effect": g, "se": se_g, "pvalue": p,
                "ci_lower": g - 1.96 * se_g, "ci_upper": g + 1.96 * se_g,
                "n_units": n_s + n_m, "cov_type_used": "hedges_g",
            })
        if rows_st:
            res_st = pd.DataFrame(rows_st)
            _, fdr, _, _ = multipletests(res_st["pvalue"], method="fdr_bh")
            res_st["FDR"] = fdr
            res_st["label"] = res_st["feature"].apply(sig_display)
            all_results.append(_harmonise(res_st, "COVID-19", "Hedges' g"))
            print(f"    COVID-19: {len(res_st)} features (Hedges' g)")
        del adata_st
        gc.collect()
    except Exception as exc:
        print(f"    COVID-19 failed: {exc}")

    # ── Single-arm datasets: Vaccine, AML, CAR-T ─────────────────────
    single_arm = [
        ("Vaccine", get_vaccine, ("Pre", "Post")),
        ("AML", get_aml, None),
        ("CAR-T", get_cart, None),
    ]
    for ds_name, loader, visits_override in single_arm:
        try:
            adata = loader()
            # Score both immune and clinical signatures
            adata, sig_cols_imm = score_signatures(adata)
            try:
                adata, sig_cols_clin = score_clinical_signatures(adata)
                all_sig_cols = list(dict.fromkeys(sig_cols_imm + sig_cols_clin))
            except Exception:
                all_sig_cols = sig_cols_imm

            visits = visits_override or _detect_visits(adata)
            design = _single_arm_design()

            res = within_arm_comparison(
                adata, arm="All", features=all_sig_cols,
                design=design, visits=visits, standardize=True,
            )
            res["label"] = res["feature"].apply(sig_display)
            all_results.append(_harmonise(res, ds_name, "within-arm Δ"))
            print(f"    {ds_name}: {len(res)} features (within-arm Δ)")
            del adata
            gc.collect()
        except Exception as exc:
            print(f"    {ds_name} failed: {exc}")

    if not all_results:
        print("    No results generated")
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)
    path = SUPP_OUTPUT / "Supp_Table2_all_results.csv"
    df.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(df)} rows)")
    return df


# ======================================================================
# Table 3 — GSEA results (per-dataset sheets)
# ======================================================================


def _stephenson_ranking() -> "pd.Series":
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


def table3_gsea_results() -> dict[str, pd.DataFrame]:
    """Run GSEA pre-ranked analysis for all 5 datasets, save as multi-sheet Excel.

    Uses shared ``load_or_run_gsea_did`` / ``load_or_run_gsea_prerank``
    helpers that cache results under ``manuscript/GSEA/{dataset}/{library}/``.
    If cached CSV files already exist they are loaded instantly; otherwise
    GSEA is run fresh and results are saved for next time.
    """
    print("  Table 3: GSEA results (per-dataset)")

    sheets: dict[str, pd.DataFrame] = {}

    # ── Sade-Feldman (two-arm DiD) ────────────────────────────────────
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
        print(f"    Melanoma GSEA failed: {exc}")

    # ── COVID-19 (cross-sectional: severe vs mild) ───────────────────
    try:
        adata_st = get_stephenson()
        adata_st, _ = score_signatures(adata_st, layer="counts")
        if "dfo_bin" in adata_st.obs.columns:
            top_bin = adata_st.obs["dfo_bin"].value_counts().idxmax()
        else:
            top_bin = "Pre"
        covid_design = TrialDesign(
            participant_col="participant_id",
            visit_col="dfo_bin",
            arm_col="severity",
            arm_treated="Severe",
            arm_control="Mild",
        )
        res = load_or_run_gsea_cross_sectional(
            adata_st, covid_design, visit=top_bin,
            layer="counts", dataset_name="COVID-19",
        )
        if res is not None:
            sheets["COVID-19"] = res
        del adata_st
        gc.collect()
    except Exception as exc:
        print(f"    COVID-19 GSEA failed: {exc}")

    # ── Single-arm datasets ───────────────────────────────────────────
    single_arm_gsea = [
        ("Vaccine", get_vaccine, ("Pre", "Post"), None),
        ("AML", get_aml, None, None),
        ("CAR-T", get_cart, None, None),
    ]
    for ds_name, loader, visits_override, layer in single_arm_gsea:
        try:
            adata = loader()
            # GSEA uses run_gsea_did which requires arm_col for DiD ranking.
            # For single-arm datasets we create a dummy arm so the DiD
            # ranking reduces to a within-arm pre→post contrast.
            if "arm" not in adata.obs.columns:
                adata.obs["arm"] = "Treated"
            visits = visits_override or _detect_visits(adata)
            gsea_design = TrialDesign(
                participant_col="participant_id", visit_col="visit",
                arm_col="arm", arm_treated="Treated", arm_control="Treated",
            )
            res = load_or_run_gsea_did(
                adata, gsea_design, visits, layer, ds_name,
            )
            if res is not None:
                sheets[ds_name] = res
            del adata
            gc.collect()
        except Exception as exc:
            print(f"    {ds_name} GSEA failed: {exc}")

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


def _run_one_did_permutation(args: tuple) -> float | None:
    """Worker for parallel DiD permutation: returns beta_DiD or None."""
    perm_idx, df_bytes, feat, unit, time, arm_bin, pid_arm_bytes, rng_seed = args

    df = pkl.loads(df_bytes)
    pid_arm = pkl.loads(pid_arm_bytes)
    rng = np.random.default_rng(rng_seed + perm_idx)
    shuffled = pid_arm.copy()
    shuffled["response_harmonized"] = rng.permutation(shuffled["response_harmonized"].values)
    pid_to_arm = dict(zip(shuffled["participant_id"], shuffled["response_harmonized"]))
    df_perm = df.copy()
    df_perm["arm_bin"] = df_perm["participant_id"].map(
        lambda x: 1 if pid_to_arm.get(x) == "Responder" else 0
    )
    try:
        out = did_fit(df_perm, y=feat, unit=unit, time=time, arm_bin=arm_bin, standardize=True)
        b = out.get("beta_DiD", np.nan)
        return float(b) if np.isfinite(b) else None
    except Exception:
        return None


def _run_one_within_arm_permutation(args: tuple) -> float | None:
    """Worker for parallel within-arm permutation: returns beta_time or None."""
    perm_idx, df_bytes, feat, unit, rng_seed = args

    df = pkl.loads(df_bytes)
    rng = np.random.default_rng(rng_seed + perm_idx)
    df_perm = df.copy()
    for pid in df_perm[unit].unique():
        mask = df_perm[unit] == pid
        visit_num = df_perm.loc[mask, "visit_num"].values
        df_perm.loc[mask, "visit_num"] = rng.permutation(visit_num)
    try:
        b = within_arm_fit_beta(df_perm, feat, unit, standardize=True)
        return float(b) if np.isfinite(b) else None
    except Exception:
        return None


def table4_permutation_results(
    n_permutations: int = 1000,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Permutation test results for all signatures across all datasets.

    Uses pre-aggregated data and parallelization for efficiency with 1000+
    permutations. For each dataset × signature, shuffles labels and re-runs
    DiD/within-arm on the small aggregated df (no full AnnData copies).

    Parameters
    ----------
    n_permutations : int
        Number of label shuffles (default 1000).
    n_jobs : int
        Parallel jobs for permutation loop (-1 = all cores).
    """
    print("  Table 4: Permutation test results")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    rng_seed = rng.integers(0, 2**31)
    rows: list[dict] = []

    # ── Sade-Feldman (two-arm DiD: shuffle response labels) ──────────
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
        obs_res = did_table(
            adata_sf, features=sig_cols, design=design,
            visits=("Pre", "Post"), layer="log1p_tpm",
            standardize=True, aggregate="participant_visit",
        )
        obs_betas = dict(zip(obs_res["feature"], obs_res["beta_DiD"]))

        with warnings.catch_warnings(action="ignore"):
            df_use, unit, time, arm_bin = get_did_aggregated_df(
                adata_sf, sig_cols, design, ("Pre", "Post"),
                layer="log1p_tpm", aggregate="participant_visit",
            )
        pid_arm = (
            adata_sf.obs[["participant_id", "response_harmonized"]]
            .drop_duplicates("participant_id")
        )
        df_bytes = pkl.dumps(df_use)
        pid_arm_bytes = pkl.dumps(pid_arm)

        for feat in sig_cols:
            obs_beta = obs_betas.get(feat, np.nan)
            if not np.isfinite(obs_beta):
                continue
            tasks = [
                (i, df_bytes, feat, unit, time, arm_bin, pid_arm_bytes, rng_seed)
                for i in range(n_permutations)
            ]
            with warnings.catch_warnings(action="ignore"):
                null_betas = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(_run_one_did_permutation)(t) for t in tasks
                )
            null_arr = np.array([b for b in null_betas if b is not None], dtype=float)
            if len(null_arr) > 0:
                perm_p = (np.sum(np.abs(null_arr) >= np.abs(obs_beta)) + 1) / (len(null_arr) + 1)
                rows.append({
                    "dataset": "Melanoma",
                    "feature": feat,
                    "label": sig_display(feat),
                    "observed_beta": obs_beta,
                    "permutation_p": perm_p,
                    "n_permutations": len(null_arr),
                    "null_mean": float(np.mean(null_arr)),
                    "null_sd": float(np.std(null_arr)),
                    "null_95_lo": float(np.percentile(null_arr, 2.5)),
                    "null_95_hi": float(np.percentile(null_arr, 97.5)),
                })
        print(f"    Melanoma: {len([r for r in rows if r['dataset'] == 'Melanoma'])} features")
        del adata_sf, df_use, df_bytes, pid_arm_bytes
        gc.collect()
    except Exception as exc:
        print(f"    Melanoma permutation failed: {exc}")

    # ── Stephenson (cross-sectional: permute Severe/Mild labels) ────
    try:
        from sctrial import between_arm_comparison as _bac
        adata_st = get_stephenson()
        if "log1p_cpm" not in adata_st.layers:
            from manuscript_figures._shared import add_log1p_cpm_layer
            adata_st = add_log1p_cpm_layer(
                adata_st, counts_layer="counts", out_layer="log1p_cpm",
            )
        adata_st, sig_cols_st = score_signatures(adata_st, layer="log1p_cpm")

        design_st = TrialDesign(
            participant_col="participant_id",
            visit_col="Collection_Day",
            arm_col="severity",
            arm_treated="Severe",
            arm_control="Mild",
        )
        obs_res_st = _bac(
            adata_st, visit="D0", features=sig_cols_st,
            design=design_st, aggregate="participant_visit",
            standardize=True,
        )
        beta_col_st = "beta_arm" if "beta_arm" in obs_res_st.columns else "beta_DiD"
        obs_betas_st = dict(zip(obs_res_st["feature"], obs_res_st[beta_col_st]))

        # Build participant-arm map for permutation
        pid_sev = (
            adata_st.obs[["participant_id", "severity"]]
            .drop_duplicates("participant_id")
        )

        with warnings.catch_warnings(action="ignore"):
            df_use_st, unit_st, time_st, arm_bin_st = get_did_aggregated_df(
                adata_st, sig_cols_st, design_st, ("D0", "D28"),
                layer="log1p_cpm", aggregate="participant_visit",
            )
        df_bytes_st = pkl.dumps(df_use_st)
        pid_sev_bytes = pkl.dumps(pid_sev)

        for feat in sig_cols_st:
            obs_beta = obs_betas_st.get(feat, np.nan)
            if not np.isfinite(obs_beta):
                continue
            tasks = [
                (i, df_bytes_st, feat, unit_st, time_st, arm_bin_st, pid_sev_bytes, rng_seed)
                for i in range(n_permutations)
            ]

            # Reuse DiD permutation worker — it permutes the arm column
            def _run_stephenson_perm(args: tuple) -> float | None:
                perm_idx, db, ft, un, tm, ab, pa_bytes, rs = args
                df = pkl.loads(db)
                pa = pkl.loads(pa_bytes)
                r = np.random.default_rng(rs + perm_idx)
                shuf = pa.copy()
                shuf["severity"] = r.permutation(shuf["severity"].values)
                pid_to_arm = dict(zip(shuf["participant_id"], shuf["severity"]))
                df_p = df.copy()
                df_p["arm_bin"] = df_p["participant_id"].map(
                    lambda x, m=pid_to_arm: 1 if m.get(x) == "Severe" else 0
                )
                try:
                    out = did_fit(df_p, y=ft, unit=un, time=tm, arm_bin=ab, standardize=True)
                    b = out.get("beta_DiD", np.nan)
                    return float(b) if np.isfinite(b) else None
                except Exception:
                    return None

            with warnings.catch_warnings(action="ignore"):
                null_betas_st = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(_run_stephenson_perm)(t) for t in tasks
                )
            null_arr = np.array([b for b in null_betas_st if b is not None], dtype=float)
            if len(null_arr) > 0:
                perm_p = (np.sum(np.abs(null_arr) >= np.abs(obs_beta)) + 1) / (len(null_arr) + 1)
                rows.append({
                    "dataset": "COVID-19",
                    "feature": feat,
                    "label": sig_display(feat),
                    "observed_beta": obs_beta,
                    "permutation_p": perm_p,
                    "n_permutations": len(null_arr),
                    "null_mean": float(np.mean(null_arr)),
                    "null_sd": float(np.std(null_arr)),
                    "null_95_lo": float(np.percentile(null_arr, 2.5)),
                    "null_95_hi": float(np.percentile(null_arr, 97.5)),
                })
        print(f"    COVID-19: {len([r for r in rows if r['dataset'] == 'COVID-19'])} features")
        del adata_st, df_use_st, df_bytes_st, pid_sev_bytes
        gc.collect()
    except Exception as exc:
        print(f"    COVID-19 permutation failed: {exc}")

    # ── Single-arm datasets: permute visit labels ────────────────────
    single_arm = [
        ("Vaccine", get_vaccine, ("Pre", "Post")),
        ("AML", get_aml, None),
        ("CAR-T", get_cart, None),
    ]
    for ds_name, loader, visits_override in single_arm:
        try:
            adata = loader()
            adata, sig_cols_imm = score_signatures(adata)
            visits = visits_override or _detect_visits(adata)
            design = _single_arm_design()

            obs_res = within_arm_comparison(
                adata, arm="All", features=sig_cols_imm,
                design=design, visits=visits, standardize=True,
            )
            obs_betas = dict(zip(obs_res["feature"], obs_res["beta_time"]))

            with warnings.catch_warnings(action="ignore"):
                df_use, unit = get_within_arm_aggregated_df(
                    adata, "All", sig_cols_imm, design, visits,
                    layer=None, aggregate="participant_visit",
                )
            df_bytes = pkl.dumps(df_use)

            for feat in sig_cols_imm:
                obs_beta = obs_betas.get(feat, np.nan)
                if not np.isfinite(obs_beta):
                    continue
                tasks = [(i, df_bytes, feat, unit, rng_seed) for i in range(n_permutations)]
                with warnings.catch_warnings(action="ignore"):
                    null_betas = Parallel(n_jobs=n_jobs, backend="loky")(
                        delayed(_run_one_within_arm_permutation)(t) for t in tasks
                    )
                null_arr = np.array([b for b in null_betas if b is not None], dtype=float)
                if len(null_arr) > 0:
                    perm_p = (np.sum(np.abs(null_arr) >= np.abs(obs_beta)) + 1) / (len(null_arr) + 1)
                    rows.append({
                        "dataset": ds_name,
                        "feature": feat,
                        "label": sig_display(feat),
                        "observed_beta": obs_beta,
                        "permutation_p": perm_p,
                        "n_permutations": len(null_arr),
                        "null_mean": float(np.mean(null_arr)),
                        "null_sd": float(np.std(null_arr)),
                        "null_95_lo": float(np.percentile(null_arr, 2.5)),
                        "null_95_hi": float(np.percentile(null_arr, 97.5)),
                    })
            print(f"    {ds_name}: {len([r for r in rows if r['dataset'] == ds_name])} features")
            del adata, df_use, df_bytes
            gc.collect()
        except Exception as exc:
            print(f"    {ds_name} permutation failed: {exc}")

    if not rows:
        print("    No permutation results generated")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Add FDR per dataset
    for ds in df["dataset"].unique():
        mask = df["dataset"] == ds
        _, fdr, _, _ = multipletests(df.loc[mask, "permutation_p"], method="fdr_bh")
        df.loc[mask, "permutation_FDR"] = fdr

    path = SUPP_OUTPUT / "Supp_Table4_permutation_results.csv"
    df.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(df)} rows)")
    return df


# ======================================================================
# Table 5 — Power analysis results
# ======================================================================


def table5_power_analysis() -> pd.DataFrame:
    """Power analysis for each dataset: observed power at current N,
    minimum N for 80% power, minimum detectable effect at current N."""
    print("  Table 5: Power analysis results")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return pd.DataFrame()

    from sctrial.stats.power import (
        power_did, sample_size_did, sensitivity_analysis,
        power_paired, sample_size_paired, sensitivity_paired,
    )

    # Read observed results from Table 2
    table2_path = SUPP_OUTPUT / "Supp_Table2_all_results.csv"
    if not table2_path.exists():
        print("    Table 2 not found, generating first...")
        table2_all_results()
    t2 = pd.read_csv(table2_path)

    rows: list[dict] = []
    for _, row in t2.iterrows():
        effect = row.get("effect", np.nan)
        se = row.get("se", np.nan)
        n_units = row.get("n_units", np.nan)
        if not np.isfinite(effect) or not np.isfinite(se) or not np.isfinite(n_units):
            continue

        dataset = row["dataset"]
        feature = row["feature"]
        label = row.get("label", sig_display(feature))
        estimand = row.get("estimand", "")
        is_did = estimand == "DiD"

        # Estimate sigma from SE.
        # DiD:    SE ≈ σ√(4/n)  → σ ≈ SE√(n/4), n = 2×n_per_group
        # Paired: SE ≈ σ√(2/n)  → σ ≈ SE√(n/2)
        n = int(n_units)
        if is_did:
            n_per_group = max(n // 2, 2)
            sigma_est = max(se * np.sqrt(n / 4), 0.01)
        else:
            n_per_group = max(n, 2)
            sigma_est = max(se * np.sqrt(n / 2), 0.01)

        # Power at observed N and effect
        abs_effect = abs(effect)
        if abs_effect < 1e-10:
            pwr = 0.05  # no effect → power = α
            min_n_val: float = np.inf
        else:
            try:
                if is_did:
                    pwr = power_did(n_per_group=n_per_group,
                                    effect_size=abs_effect, sigma=sigma_est)
                else:
                    pwr = power_paired(n_participants=n_per_group,
                                       effect_size=abs_effect, sigma=sigma_est)
            except Exception:
                pwr = np.nan
            try:
                if is_did:
                    min_n_val = float(sample_size_did(
                        effect_size=abs_effect, sigma=sigma_est, power=0.80))
                else:
                    min_n_val = float(sample_size_paired(
                        effect_size=abs_effect, sigma=sigma_est, power=0.80))
            except Exception:
                min_n_val = np.nan

        # Minimum detectable effect at current N
        try:
            if is_did:
                mde = sensitivity_analysis(n_per_group=n_per_group,
                                           sigma=sigma_est, power=0.80)
            else:
                mde = sensitivity_paired(n_participants=n_per_group,
                                         sigma=sigma_est, power=0.80)
        except Exception:
            mde = np.nan

        rows.append({
            "dataset": dataset,
            "feature": feature,
            "label": label,
            "estimand": estimand,
            "observed_effect": effect,
            "observed_se": se,
            "n_participants": n,
            "sigma_estimated": sigma_est,
            "power_at_observed_N": pwr,
            "min_N_for_80pct_power": int(min_n_val) if np.isfinite(min_n_val) else None,
            "min_detectable_effect_80pct": mde,
        })

    df = pd.DataFrame(rows)
    path = SUPP_OUTPUT / "Supp_Table5_power_analysis.csv"
    df.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(df)} rows)")
    return df


# ======================================================================
# Table 6 — Gene-level DiD results (Sade-Feldman)
# ======================================================================


def table6_gene_level_results() -> pd.DataFrame:
    """Gene-level DiD results for the Sade-Feldman melanoma cohort.

    Runs DiD on every expressed gene (not just signatures) to provide
    the full genome-wide results for reproducibility.
    """
    print("  Table 6: Gene-level DiD results (Sade-Feldman)")

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

        path = SUPP_OUTPUT / "Supp_Table6_gene_level_DiD_Sade_Feldman.csv"
        res.to_csv(path, index=False)
        n_sig = (res["FDR_DiD"] < 0.05).sum() if "FDR_DiD" in res.columns else 0
        n_nom = (res["p_DiD"] < 0.05).sum() if "p_DiD" in res.columns else 0
        print(f"    Saved {path.name} ({len(res)} genes, {n_nom} nominally significant, {n_sig} FDR < 0.05)")

        del adata_sf
        gc.collect()
        return res
    except Exception as exc:
        print(f"    Gene-level DiD failed: {exc}")
        return pd.DataFrame()


# ======================================================================
# Table 7 — Dataset metadata summary
# ======================================================================


def table7_dataset_metadata() -> pd.DataFrame:
    """Per-participant metadata across all 5 datasets.

    Columns: dataset, participant_id, arm/condition, visit, n_cells,
    response_status (where applicable).
    """
    print("  Table 7: Dataset metadata summary")

    rows: list[dict] = []

    datasets_info = [
        ("Melanoma", get_sade_feldman, "participant_id", "visit", "response_harmonized"),
        ("COVID-19", get_stephenson, "participant_id", None, "severity"),
        ("Vaccine", get_vaccine, "participant_id", "visit", None),
        ("AML", get_aml, "participant_id", "visit", None),
        ("CAR-T", get_cart, "participant_id", "visit", None),
    ]

    for ds_name, loader, pid_col, visit_col, condition_col in datasets_info:
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

            grouped = obs.groupby(group_cols).size().reset_index(name="n_cells")
            grouped["dataset"] = ds_name

            # Add condition/response
            if condition_col and condition_col in obs.columns:
                cond_map = obs.groupby(pid_col)[condition_col].first()
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
                n_celltypes = obs.groupby(group_cols)[ct_col].nunique().reset_index(name="n_cell_types")
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
            print(f"    {ds_name} metadata failed: {exc}")

    if not rows:
        print("    No metadata generated")
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    path = SUPP_OUTPUT / "Supp_Table7_dataset_metadata.csv"
    df.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(df)} rows)")
    return df


# ======================================================================
# Patch Table 1 — add "Genes Present" column per dataset
# ======================================================================


def patch_table1_genes_present() -> pd.DataFrame:
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
        ("COVID-19", get_stephenson),
        ("Vaccine", get_vaccine),
        ("AML", get_aml),
        ("CAR-T", get_cart),
    ]

    for ds_name, loader in datasets_loaders:
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
            print(f"    {ds_name} genes present failed: {exc}")

    t1.to_csv(t1_path, index=False)
    # Also save as xlsx to prevent Excel from auto-formatting
    # fraction strings like "11/11" as dates
    xlsx_path = t1_path.with_suffix(".xlsx")
    t1.to_excel(xlsx_path, index=False, engine="openpyxl")
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
        Which tables to generate: "all", "1-3" (original), "4-6" (new),
        or a comma-separated list like "4,5,6".
    """
    print("=" * 60)
    print("Supplementary Tables")
    print("=" * 60)

    to_run = set()
    if tables == "all":
        to_run = {1, 2, 3, 4, 5, 6}
    elif tables == "1-3":
        to_run = {1, 2, 3}
    elif tables == "4-6":
        to_run = {4, 5, 6}
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
        table6_gene_level_results()

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
