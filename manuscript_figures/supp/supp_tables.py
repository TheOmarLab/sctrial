"""
Supplementary Tables 1–3.
=========================

Table 1  Gene signature definitions (name, gene count, genes).
Table 2  Complete effect-size results across all signatures and datasets.
Table 3  GSEA pre-ranked results (one sheet per dataset).
"""

from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from .._shared import (
    CLINICAL_SIGNATURES,
    GENE_SIGNATURES,
    SCRIPT_DIR,
    SCTRIAL_AVAILABLE,
    SIGNATURE_DISPLAY_NAMES,
    SUPP_OUTPUT,
    TrialDesign,
    apply_style,
    clear_cache,
    did_table,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    get_aml,
    get_cart,
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
    return out[avail].reset_index(drop=True)


def _single_arm_design(pid_col: str = "participant_id") -> TrialDesign:
    """Design object for single-arm pre/post datasets."""
    return TrialDesign(
        participant_col=pid_col,
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Treated",
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
        all_results.append(_harmonise(res, "Sade-Feldman", "DiD"))
        print(f"    Sade-Feldman: {len(res)} features (DiD)")
        del adata_sf
        gc.collect()
    except Exception as exc:
        print(f"    Sade-Feldman failed: {exc}")

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
            all_results.append(_harmonise(res_st, "Stephenson", "Hedges' g"))
            print(f"    Stephenson: {len(res_st)} features (Hedges' g)")
        del adata_st
        gc.collect()
    except Exception as exc:
        print(f"    Stephenson failed: {exc}")

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

            if "arm" not in adata.obs.columns:
                adata.obs["arm"] = "Treated"
            arm_label = adata.obs["arm"].iloc[0]
            visits = visits_override or _detect_visits(adata)
            design = _single_arm_design()

            res = within_arm_comparison(
                adata, arm=arm_label, features=all_sig_cols,
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


def _run_gsea_for_dataset(
    adata,
    design: TrialDesign,
    visits: tuple[str, str],
    layer: str | None,
    dataset_name: str,
    gsea_root: str,
) -> pd.DataFrame | None:
    """Run pre-ranked GSEA for a single dataset, save CSV, return results."""
    try:
        from sctrial import run_gsea_did
    except ImportError:
        print(f"    {dataset_name}: sctrial GSEA not available")
        return None

    libraries = [
        ("MSigDB_Hallmark_2020", "Hallmark"),
        ("KEGG_2021_Human", "KEGG"),
        ("Reactome_2022", "Reactome"),
        ("GO_Biological_Process_2023", "GO_BP"),
        ("WikiPathways_2024_Human", "WikiPathways"),
    ]

    frames: list[pd.DataFrame] = []
    for lib_name, short_name in libraries:
        outdir = os.path.join(gsea_root, dataset_name, short_name)
        os.makedirs(outdir, exist_ok=True)
        try:
            res = run_gsea_did(
                adata, gene_sets=lib_name, design=design, visits=visits,
                layer=layer, rank_by="tstat",
                min_size=10, max_size=500, permutation_num=1000,
                outdir=outdir, no_plot=True,
            )
            if isinstance(res, pd.DataFrame) and len(res) > 0:
                res["Library"] = short_name
                frames.append(res)
                print(f"    {dataset_name}/{short_name}: {len(res)} pathways")
            else:
                print(f"    {dataset_name}/{short_name}: no results")
        except Exception as exc:
            print(f"    {dataset_name}/{short_name}: {exc}")

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "NES" in combined.columns:
        combined = combined.sort_values("NES", key=abs, ascending=False)
    return combined


def table3_gsea_results() -> dict[str, pd.DataFrame]:
    """Run GSEA pre-ranked analysis for all datasets, save as multi-sheet Excel."""
    print("  Table 3: GSEA results (per-dataset)")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return {}

    gsea_root = str(SCRIPT_DIR.parent.parent.parent / "manuscript" / "GSEA")
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
        res = _run_gsea_for_dataset(
            adata_sf, design_sf, ("Pre", "Post"), "log1p_tpm",
            "Sade_Feldman", gsea_root,
        )
        if res is not None:
            sheets["Sade-Feldman"] = res
        del adata_sf
        gc.collect()
    except Exception as exc:
        print(f"    Sade-Feldman GSEA failed: {exc}")

    # ── Stephenson (cross-sectional: severe vs mild) ─────────────────
    try:
        import gseapy as gp
        from scipy import stats as sp_stats

        adata_st = get_stephenson()
        pid_col, sev_col = "participant_id", "severity"

        # Pseudobulk to participant level, compute per-gene t-stat ranking
        layer_st = "log1p_cpm" if "log1p_cpm" in adata_st.layers else None
        expr = adata_st.layers[layer_st] if layer_st else adata_st.X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        obs = adata_st.obs[[pid_col, sev_col]].copy()
        obs.index = range(len(obs))

        # Mean expression per participant
        expr_df = pd.DataFrame(expr, columns=adata_st.var_names)
        expr_df[pid_col] = obs[pid_col].values
        expr_df[sev_col] = obs[sev_col].values
        pb_st = expr_df.groupby(pid_col).agg(
            {sev_col: "first", **{g: "mean" for g in adata_st.var_names}}
        )

        sev_vals = pb_st[sev_col].unique()
        severe = "severe" if "severe" in sev_vals else sev_vals[0]
        mild = "mild" if "mild" in sev_vals else sev_vals[-1]
        grp_s = pb_st[pb_st[sev_col] == severe]
        grp_m = pb_st[pb_st[sev_col] == mild]

        # Per-gene t-statistic ranking
        ranking = {}
        for gene in adata_st.var_names:
            xs = grp_s[gene].dropna().values
            xm = grp_m[gene].dropna().values
            if len(xs) < 3 or len(xm) < 3:
                continue
            t, _ = sp_stats.ttest_ind(xs, xm, equal_var=False)
            if np.isfinite(t):
                ranking[gene] = t

        if ranking:
            rnk = pd.Series(ranking).sort_values(ascending=False)
            libraries = [
                ("MSigDB_Hallmark_2020", "Hallmark"),
                ("KEGG_2021_Human", "KEGG"),
                ("Reactome_2022", "Reactome"),
                ("GO_Biological_Process_2023", "GO_BP"),
                ("WikiPathways_2024_Human", "WikiPathways"),
            ]
            frames_st: list[pd.DataFrame] = []
            for lib_name, short_name in libraries:
                outdir = os.path.join(gsea_root, "Stephenson", short_name)
                os.makedirs(outdir, exist_ok=True)
                try:
                    pre_res = gp.prerank(
                        rnk=rnk, gene_sets=lib_name,
                        min_size=10, max_size=500, permutation_num=1000,
                        outdir=outdir, no_plot=True,
                    )
                    res_df = pre_res.res2d if hasattr(pre_res, "res2d") else pre_res
                    if isinstance(res_df, pd.DataFrame) and len(res_df) > 0:
                        res_df["Library"] = short_name
                        frames_st.append(res_df)
                        print(f"    Stephenson/{short_name}: {len(res_df)} pathways")
                except Exception as exc:
                    print(f"    Stephenson/{short_name}: {exc}")
            if frames_st:
                combined_st = pd.concat(frames_st, ignore_index=True)
                if "NES" in combined_st.columns:
                    combined_st = combined_st.sort_values("NES", key=abs, ascending=False)
                sheets["Stephenson"] = combined_st
        del adata_st
        gc.collect()
    except Exception as exc:
        print(f"    Stephenson GSEA failed: {exc}")

    # ── Single-arm datasets ───────────────────────────────────────────
    single_arm_gsea = [
        ("Vaccine", get_vaccine, ("Pre", "Post"), None),
        ("AML", get_aml, None, None),
        ("CAR-T", get_cart, None, None),
    ]
    for ds_name, loader, visits_override, layer in single_arm_gsea:
        try:
            adata = loader()
            if "arm" not in adata.obs.columns:
                adata.obs["arm"] = "Treated"
            visits = visits_override or _detect_visits(adata)
            design = _single_arm_design()
            res = _run_gsea_for_dataset(
                adata, design, visits, layer, ds_name, gsea_root,
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
                # Excel sheet names max 31 chars
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"    Saved {xlsx_path.name} ({len(sheets)} sheets)")
    else:
        print("    No GSEA results generated")

    return sheets


# ======================================================================
# Main entry point
# ======================================================================


def generate():
    """Generate all supplementary tables."""
    print("=" * 60)
    print("Supplementary Tables")
    print("=" * 60)

    table1_gene_signatures()
    table2_all_results()
    table3_gsea_results()

    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
