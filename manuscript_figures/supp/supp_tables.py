"""
Supplementary Tables 1–4.
=========================

Table 1  Gene signature definitions (name, gene count, genes).
Table 2  Complete DiD results across all signatures and datasets.
Table 3  GSEA results from cached gseapy prerank reports.
Table 4  Clinical trial paired pre/post results (AML, CAR-T).
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .._shared import (
    GENE_SIGNATURES,
    CLINICAL_SIGNATURES,
    SIGNATURE_DISPLAY_NAMES,
    SUPP_OUTPUT,
    SCRIPT_DIR,
    TrialDesign,
    did_table,
    within_arm_comparison,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    load_clinical_trial_dataset,
    harmonize_response,
    score_signatures,
    score_clinical_signatures,
    sig_display,
    apply_style,
    clear_cache,
    dfo_sort_key,
    SCTRIAL_AVAILABLE,
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
# Table 2 — Complete DiD results across datasets
# ======================================================================

def _build_design_sf():
    """Sade-Feldman trial design."""
    return TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="response_harmonized",
        arm_treated="Responder",
        arm_control="Non-responder",
    )


def _build_design_stephenson():
    """Stephenson COVID-19 trial design."""
    return TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="severity",
        arm_treated="severe",
        arm_control="mild",
    )


def table2_did_results() -> pd.DataFrame:
    """Complete DiD results across all signatures and datasets."""
    print("  Table 2: DiD results across datasets")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return pd.DataFrame()

    all_results = []

    # ── Sade-Feldman ─────────────────────────────────────────────────
    try:
        adata_sf = get_sade_feldman()
        if "log1p_tpm" not in adata_sf.layers and "tpm" in adata_sf.layers:
            adata_sf.layers["log1p_tpm"] = np.log1p(adata_sf.layers["tpm"])
        adata_sf, sig_cols_sf = score_signatures(adata_sf, layer="log1p_tpm")
        adata_sf = harmonize_response(adata_sf)

        design_sf = _build_design_sf()
        res_sf = did_table(
            adata_sf,
            features=sig_cols_sf,
            design=design_sf,
            visits=("Pre", "Post"),
            layer="log1p_tpm",
            standardize=True,
            aggregate="participant_visit",
        )
        res_sf["dataset"] = "Sade-Feldman"
        res_sf["label"] = res_sf["feature"].apply(sig_display)
        all_results.append(res_sf)
        print(f"    Sade-Feldman: {len(res_sf)} features")
    except Exception as exc:
        print(f"    Sade-Feldman failed: {exc}")

    # ── Stephenson ───────────────────────────────────────────────────
    try:
        adata_st = get_stephenson()
        # Ensure visit column
        if "visit" not in adata_st.obs.columns:
            if "timepoint" in adata_st.obs.columns:
                adata_st.obs["visit"] = adata_st.obs["timepoint"]
            elif "days_from_onset" in adata_st.obs.columns:
                adata_st.obs["visit"] = adata_st.obs["days_from_onset"].apply(
                    lambda x: "Early" if x <= 7 else "Late"
                )
        # Ensure severity column
        if "severity" not in adata_st.obs.columns:
            for cand in ("Status", "status", "disease_severity", "severity_group"):
                if cand in adata_st.obs.columns:
                    adata_st.obs["severity"] = adata_st.obs[cand]
                    break

        adata_st, sig_cols_st = score_signatures(adata_st)

        # Determine visits
        visits_avail = sorted(adata_st.obs["visit"].unique())
        if len(visits_avail) >= 2:
            visits_st = (visits_avail[0], visits_avail[-1])

            # Determine arm labels
            sev_values = adata_st.obs["severity"].unique().tolist()
            arm_treated = "severe" if "severe" in sev_values else sev_values[0]
            arm_control = "mild" if "mild" in sev_values else sev_values[-1]

            design_st = TrialDesign(
                participant_col="participant_id",
                visit_col="visit",
                arm_col="severity",
                arm_treated=arm_treated,
                arm_control=arm_control,
            )
            res_st = did_table(
                adata_st,
                features=sig_cols_st,
                design=design_st,
                visits=visits_st,
                standardize=True,
                aggregate="participant_visit",
            )
            res_st["dataset"] = "Stephenson"
            res_st["label"] = res_st["feature"].apply(sig_display)
            all_results.append(res_st)
            print(f"    Stephenson: {len(res_st)} features")
        else:
            print("    Stephenson: insufficient visits for DiD")
    except Exception as exc:
        print(f"    Stephenson failed: {exc}")

    # ── Vaccine ──────────────────────────────────────────────────────
    try:
        adata_vax = get_vaccine()
        adata_vax, sig_cols_vax = score_signatures(adata_vax)

        # Single-arm: use within_arm_comparison
        adata_vax.obs["arm"] = "Vaccinated"
        vax_design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Vaccinated",
            arm_control="Vaccinated",
        )
        res_vax = within_arm_comparison(
            adata_vax,
            arm="Vaccinated",
            features=sig_cols_vax,
            design=vax_design,
            visits=("Pre", "Post"),
            standardize=True,
        )
        res_vax["dataset"] = "Vaccine"
        res_vax["label"] = res_vax["feature"].apply(sig_display)
        all_results.append(res_vax)
        print(f"    Vaccine: {len(res_vax)} features")
    except Exception as exc:
        print(f"    Vaccine failed: {exc}")

    if not all_results:
        print("    No DiD results generated")
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)
    path = SUPP_OUTPUT / "Supp_Table2_did_results.csv"
    df.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(df)} rows)")
    return df


# ======================================================================
# Table 3 — GSEA results
# ======================================================================

def table3_gsea_results() -> pd.DataFrame:
    """Collate GSEA results from cached gseapy CSV files."""
    print("  Table 3: GSEA results")

    # GSEA results live under manuscript/ at the repo root
    gsea_root = SCRIPT_DIR.parent.parent / "manuscript"
    db_dirs = {
        "Hallmark": gsea_root / "gsea_hallmark",
        "Reactome": gsea_root / "gsea_reactome",
        "GO_BP": gsea_root / "gsea_go_bp",
    }

    frames = []
    for db_name, db_dir in db_dirs.items():
        csv_path = db_dir / "gseapy.gene_set.prerank.report.csv"
        if not csv_path.exists():
            print(f"    {db_name}: not found at {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
            df.insert(0, "Database", db_name)
            frames.append(df)
            print(f"    {db_name}: {len(df)} pathways")
        except Exception as exc:
            print(f"    {db_name}: failed to read ({exc})")

    if not frames:
        print("    No GSEA results found")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Sort by absolute NES descending
    if "NES" in combined.columns:
        combined = combined.sort_values("NES", key=abs, ascending=False)

    path = SUPP_OUTPUT / "Supp_Table3_gsea_results.csv"
    combined.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(combined)} rows)")
    return combined


# ======================================================================
# Table 4 — Clinical trial paired pre/post results
# ======================================================================

def table4_clinical_results() -> pd.DataFrame:
    """Paired pre/post within-arm results for AML and CAR-T datasets."""
    print("  Table 4: Clinical trial results (AML, CAR-T)")

    if not SCTRIAL_AVAILABLE:
        print("    Skipped: sctrial not available")
        return pd.DataFrame()

    all_results = []

    for name in ("aml", "cart"):
        try:
            adata = load_clinical_trial_dataset(name)
            adata, sig_cols = score_clinical_signatures(adata)

            pid_col = (
                "participant_id"
                if "participant_id" in adata.obs.columns
                else "patient_id"
            )
            if "visit" not in adata.obs.columns:
                if "timepoint" in adata.obs.columns:
                    adata.obs["visit"] = adata.obs["timepoint"]
            visit_col = "visit"

            # Determine Pre/Post visits
            visits_avail = list(adata.obs[visit_col].unique())
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

            if "arm" not in adata.obs.columns:
                adata.obs["arm"] = "Treated"

            design = TrialDesign(
                participant_col=pid_col,
                visit_col=visit_col,
                arm_col="arm",
                arm_treated="Treated",
                arm_control="Treated",
            )

            res = within_arm_comparison(
                adata,
                arm="Treated",
                features=sig_cols,
                design=design,
                visits=(pre_v, post_v),
                layer=None,
                standardize=True,
            )
            res["dataset"] = name.upper()
            res["label"] = res["feature"].apply(sig_display)
            res["pre_visit"] = pre_v
            res["post_visit"] = post_v
            all_results.append(res)
            print(f"    {name.upper()}: {len(res)} features, "
                  f"visits=({pre_v}, {post_v})")

            del adata
            gc.collect()
        except Exception as exc:
            print(f"    {name.upper()} failed: {exc}")

    if not all_results:
        print("    No clinical results generated")
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)
    path = SUPP_OUTPUT / "Supp_Table4_clinical_results.csv"
    df.to_csv(path, index=False)
    print(f"    Saved {path.name} ({len(df)} rows)")
    return df


# ======================================================================
# Main entry point
# ======================================================================

def generate():
    """Generate all supplementary tables."""
    print("=" * 60)
    print("Supplementary Tables")
    print("=" * 60)

    table1_gene_signatures()
    table2_did_results()
    table3_gsea_results()
    table4_clinical_results()

    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
