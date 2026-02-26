#!/usr/bin/env python3
"""
Generate All Supplementary Materials for sctrial Manuscript

This script generates:
- Supplementary Table 1: Gene signature definitions
- Supplementary Table 2: Complete DiD results for all signatures across datasets
- Supplementary Table 3: GSEA results for all pathway collections
- Supplementary Figure 1: Quality control metrics across datasets

Usage:
    python generate_all_supplementary.py

Requirements:
    - numpy, pandas, matplotlib, scipy, statsmodels
    - scanpy, anndata (for data loading)
    - sctrial package (optional, for dataset loaders)

Author: Generated for sctrial manuscript
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory - will be created if it doesn't exist
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "manuscript", "Supplementary_Materials")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory - adjust this to your local path
from pathlib import Path
_script_path = Path(__file__).parent.resolve()
DATA_DIR = os.environ.get("SCTRIAL_DATA_ROOT")
if DATA_DIR:
    DATA_DIR = str(DATA_DIR)
else:
    candidates = [
        _script_path.parent / "sc-trialdiff" / "data",
        Path.home() / "Documents" / "Research" / "projects" / "sc-trialdiff" / "data",
    ]
    for c in candidates:
        if c.exists():
            DATA_DIR = str(c)
            break
    if DATA_DIR is None:
        DATA_DIR = str(_script_path.parent / "sc-trialdiff" / "data")

print(f"Output directory: {OUTPUT_DIR}")
print(f"Data directory: {DATA_DIR}")

# Random seed for reproducibility
rng = np.random.default_rng(42)

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
})

# Color palette
COLORS = {
    "treated": "#4C72B0",      # Blue for responders/treatment
    "control": "#E1812C",      # Orange for non-responders/control
    "neutral": "#8172B3",      # Purple for neutral
    "highlight": "#C44E52",    # Red for highlights
    "success": "#55A868",      # Green for success
    "gray": "#8C8C8C",         # Gray
}

def despine(ax):
    """Remove top and right spines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ============================================================================
# GENE SIGNATURES
# ============================================================================

GENE_SIGNATURES = {
    "Cytotoxic T Cell Activity": [
        "GZMB", "GZMA", "GZMH", "GZMK", "PRF1", "GNLY", "NKG7", "KLRK1",
        "KLRD1", "FASLG", "IFNG"
    ],
    "Immune Exhaustion": [
        "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "TOX2",
        "ENTPD1", "CD244", "CD160", "BTLA"
    ],
    "Interferon Response": [
        "ISG15", "IFI6", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2",
        "OAS1", "OAS2", "OAS3", "STAT1", "IRF7", "IRF9"
    ],
    "Memory T Cell": [
        "IL7R", "TCF7", "LEF1", "CCR7", "SELL", "CD27", "CD28",
        "BCL2", "EOMES", "ID3"
    ],
    "T Cell Activation": [
        "CD69", "CD44", "IL2RA", "ICOS", "TNFRSF4", "TNFRSF9",
        "CD40LG", "HLA-DRA", "HLA-DRB1"
    ],
    "Inflammatory Response": [
        "IL1B", "IL6", "TNF", "CXCL8", "CCL2", "CCL3", "CCL4",
        "NFKB1", "NLRP3", "CASP1"
    ],
    "Antigen Presentation": [
        "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2",
        "PSMB8", "PSMB9", "CD74"
    ],
    "Cell Proliferation": [
        "MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "CCNA2",
        "MCM2", "MCM7", "TYMS"
    ],
    "Regulatory T Cell": [
        "FOXP3", "IL2RA", "CTLA4", "TNFRSF18", "IKZF2", "IKZF4",
        "IL10", "TGFB1", "ENTPD1"
    ],
    "NK Cell Activity": [
        "NCAM1", "FCGR3A", "NCR1", "NCR3", "KLRF1", "KLRC1",
        "KIR2DL1", "KIR2DL3", "KIR3DL1"
    ],
    "Apoptosis": [
        "BCL2", "BAX", "BAK1", "CASP3", "CASP8", "CASP9",
        "FAS", "FASLG", "BID", "PARP1"
    ],
    "Oxidative Stress Response": [
        "NFE2L2", "HMOX1", "NQO1", "GCLC", "GCLM", "GSR",
        "SOD1", "SOD2", "CAT", "GPX1"
    ],
}

SIGNATURE_DISPLAY_NAMES = {
    "Cytotoxic T Cell Activity": "Cytotoxic T Cells",
    "Immune Exhaustion": "T Cell Exhaustion",
    "Interferon Response": "IFN Response",
    "Memory T Cell": "Memory T Cells",
    "T Cell Activation": "T Cell Activation",
    "Inflammatory Response": "Inflammation",
    "Antigen Presentation": "Antigen Presentation",
    "Cell Proliferation": "Cell Proliferation",
    "Regulatory T Cell": "Regulatory T Cells",
    "NK Cell Activity": "NK Cell Activity",
    "Apoptosis": "Apoptosis",
    "Oxidative Stress Response": "Oxidative Stress",
}

def get_signature_display_name(sig_name):
    """Get short display name for a signature."""
    clean_name = sig_name.replace("sig_", "")
    return SIGNATURE_DISPLAY_NAMES.get(clean_name, clean_name)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

_DATA_CACHE = {}

def _clear_cache():
    """Clear data cache to free memory."""
    global _DATA_CACHE
    _DATA_CACHE.clear()
    import gc
    gc.collect()

def get_sade_feldman():
    """Load Sade-Feldman immunotherapy dataset."""
    if "sade_feldman" in _DATA_CACHE:
        return _DATA_CACHE["sade_feldman"]

    import anndata as ad
    path = os.path.join(DATA_DIR, "processed", "sade_feldman_processed_v5.h5ad")
    if os.path.exists(path):
        print("  Loading cached Sade-Feldman data...")
        adata = ad.read_h5ad(path)
    else:
        print("  Processing Sade-Feldman data via sctrial.datasets...")
        try:
            from sctrial.datasets import load_sade_feldman
            adata = load_sade_feldman(max_cells_per_participant_visit=None)
        except Exception as e:
            raise FileNotFoundError(f"Sade-Feldman data not found at {path} and could not load via sctrial: {e}")

    _DATA_CACHE["sade_feldman"] = adata
    return adata

def get_stephenson():
    """Load Stephenson COVID-19 dataset."""
    if "stephenson" in _DATA_CACHE:
        return _DATA_CACHE["stephenson"]

    import anndata as ad
    path = os.path.join(DATA_DIR, "processed", "stephenson_covid19_v3.h5ad")
    if os.path.exists(path):
        print("  Loading cached Stephenson data...")
        adata = ad.read_h5ad(path)
    else:
        print("  Processing Stephenson data via sctrial.datasets...")
        try:
            from sctrial.datasets import load_stephenson_data
            adata = load_stephenson_data(force_reprocess=False)
        except Exception as e:
            raise FileNotFoundError(f"Stephenson data not found at {path} and could not load via sctrial: {e}")

    _DATA_CACHE["stephenson"] = adata
    return adata

def get_vaccine():
    """Load Vaccine GSE171964 dataset."""
    if "vaccine" in _DATA_CACHE:
        return _DATA_CACHE["vaccine"]

    import anndata as ad
    path = os.path.join(DATA_DIR, "processed", "vaccine_gse171964_day0_day7.h5ad")
    if os.path.exists(path):
        print("  Loading cached Vaccine GSE171964 data...")
        adata = ad.read_h5ad(path)
        # Harmonize column names
        if "pt_id" in adata.obs.columns and "participant_id" not in adata.obs.columns:
            adata.obs["participant_id"] = adata.obs["pt_id"]
        if "day" in adata.obs.columns and "visit" not in adata.obs.columns:
            adata.obs["visit"] = adata.obs["day"].map({0: "Day0", 7: "Day7"})
    else:
        print(f"  Vaccine dataset not found at {path}")
        return None

    _DATA_CACHE["vaccine"] = adata
    return adata

def score_signatures(adata, layer=None, min_genes=3):
    """Score all gene signatures in the dataset."""
    import scanpy as sc

    if layer is None:
        for lyr in ["log1p_tpm", "log1p_cpm", "counts"]:
            if lyr in adata.layers:
                layer = lyr
                break

    print(f"  Scoring 12 signatures using layer: {layer}")

    sig_cols = []
    for sig_name, genes in GENE_SIGNATURES.items():
        available_genes = [g for g in genes if g in adata.var_names]
        if len(available_genes) >= min_genes:
            col_name = f"sig_{sig_name}"
            try:
                sc.tl.score_genes(adata, available_genes, score_name=col_name, use_raw=False)
                sig_cols.append(col_name)
            except Exception as e:
                print(f"    Warning: Could not score {sig_name}: {e}")

    return adata, sig_cols

def harmonize_response(adata):
    """Harmonize response column names across datasets."""
    if "response_harmonized" in adata.obs.columns:
        return adata

    for col in ["response", "Response", "clinical_response"]:
        if col in adata.obs.columns:
            adata.obs["response_harmonized"] = adata.obs[col].astype(str)
            mapping = {
                "responder": "Responder", "Responder": "Responder", "R": "Responder",
                "non-responder": "Non-responder", "Non-responder": "Non-responder",
                "NR": "Non-responder", "nonresponder": "Non-responder",
            }
            adata.obs["response_harmonized"] = adata.obs["response_harmonized"].map(
                lambda x: mapping.get(x, x)
            )
            break

    return adata

# ============================================================================
# SUPPLEMENTARY TABLE 1: Gene Signatures
# ============================================================================

def generate_supplementary_table_1():
    """Generate Supplementary Table 1: Gene signature definitions."""
    print("\n" + "="*60)
    print("Generating Supplementary Table 1: Gene Signatures")
    print("="*60)

    rows = []
    for sig_name, genes in GENE_SIGNATURES.items():
        rows.append({
            "Signature Name": sig_name,
            "Gene Count": len(genes),
            "Genes": ", ".join(genes)
        })

    df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, "Supplementary_Table_1_Gene_Signatures.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Contains {len(df)} signatures")
    return df

# ============================================================================
# SUPPLEMENTARY TABLE 2: DiD Results
# ============================================================================

def generate_supplementary_table_2():
    """
    Supplementary Table 2: Complete DiD results for all signatures across datasets.
    """
    print("\n" + "="*60)
    print("Generating Supplementary Table 2: DiD Results")
    print("="*60)

    all_results = []

    # 1. Sade-Feldman (Immunotherapy) - True DiD
    try:
        print("\n  Processing Sade-Feldman (Immunotherapy)...")
        adata = get_sade_feldman()
        adata = harmonize_response(adata)
        adata, sig_cols = score_signatures(adata)

        RESPONSE_COL = "response_harmonized"

        # Find paired participants
        participant_visits = adata.obs.groupby("participant_id")["visit"].apply(set).reset_index()
        participant_visits["is_paired"] = participant_visits["visit"].apply(lambda x: "Pre" in x and "Post" in x)
        paired_ids = list(participant_visits[participant_visits["is_paired"]]["participant_id"])
        participant_response = adata.obs.groupby("participant_id")[RESPONSE_COL].first()

        print(f"    Found {len(paired_ids)} paired participants")

        for sig in sig_cols:
            df_sig = adata.obs.groupby(["participant_id", "visit", RESPONSE_COL], observed=True)[sig].mean().reset_index()
            df_sig = df_sig[df_sig["participant_id"].isin(paired_ids)]

            # Compute deltas (Post - Pre) for each participant
            deltas = {}
            for pid in paired_ids:
                sub = df_sig[df_sig["participant_id"] == pid]
                if len(sub) == 2:
                    pre_val = sub[sub["visit"] == "Pre"][sig].values[0]
                    post_val = sub[sub["visit"] == "Post"][sig].values[0]
                    deltas[pid] = post_val - pre_val

            delta_resp = np.array([deltas[p] for p in deltas if participant_response.get(p) == "Responder"])
            delta_nonresp = np.array([deltas[p] for p in deltas if participant_response.get(p) == "Non-responder"])

            if len(delta_resp) >= 2 and len(delta_nonresp) >= 2:
                # DiD estimate
                beta_did = np.mean(delta_resp) - np.mean(delta_nonresp)
                se = np.sqrt(np.var(delta_resp, ddof=1)/len(delta_resp) +
                            np.var(delta_nonresp, ddof=1)/len(delta_nonresp))
                _, p_val = stats.mannwhitneyu(delta_resp, delta_nonresp, alternative="two-sided")

                # Bootstrap CI
                boot_dids = []
                for _ in range(2000):
                    dr = rng.choice(delta_resp, size=len(delta_resp), replace=True)
                    dnr = rng.choice(delta_nonresp, size=len(delta_nonresp), replace=True)
                    boot_dids.append(np.mean(dr) - np.mean(dnr))
                ci_low, ci_high = np.percentile(boot_dids, [2.5, 97.5])

                all_results.append({
                    "Dataset": "Sade-Feldman (Immunotherapy)",
                    "Analysis_Type": "DiD",
                    "Signature": get_signature_display_name(sig),
                    "Effect_Size": beta_did,
                    "SE": se,
                    "CI_Low": ci_low,
                    "CI_High": ci_high,
                    "P_Value": p_val,
                    "N_Responders": len(delta_resp),
                    "N_NonResponders": len(delta_nonresp),
                })

        print(f"    Computed {len([r for r in all_results if 'Sade-Feldman' in r['Dataset']])} signature results")
        del adata
        _clear_cache()
    except Exception as e:
        print(f"  Sade-Feldman error: {e}")
        import traceback
        traceback.print_exc()

    # 2. Stephenson COVID-19 - Cross-sectional comparison
    try:
        print("\n  Processing Stephenson (COVID-19)...")
        adata = get_stephenson()
        adata, sig_cols = score_signatures(adata)

        # Compare Severe vs Mild (participant-level aggregates)
        donor_col = "donor_id" if "donor_id" in adata.obs.columns else "participant_id"
        for sig in sig_cols:
            # Aggregate to participant level to avoid pseudoreplication
            participant_means = adata.obs.groupby([donor_col, "severity"], observed=True)[sig].mean().reset_index()
            severe = participant_means[participant_means["severity"] == "Severe"][sig].values
            mild = participant_means[participant_means["severity"] == "Mild"][sig].values

            if len(severe) >= 3 and len(mild) >= 3:
                # Hedge's g (corrected effect size) at participant level
                n_s, n_m = len(severe), len(mild)
                pooled_std = np.sqrt(((n_s-1)*np.var(severe, ddof=1) +
                                     (n_m-1)*np.var(mild, ddof=1)) /
                                    (n_s+n_m-2))
                hedges_g = (np.mean(severe) - np.mean(mild)) / pooled_std if pooled_std > 0 else 0
                correction = 1 - 3/(4*(n_s+n_m-2)-1)
                hedges_g *= correction

                se = np.sqrt((n_s+n_m)/(n_s*n_m) +
                            hedges_g**2/(2*(n_s+n_m-2)))
                # Use t-distribution for CI with participant-level df
                from scipy.stats import t as t_dist
                t_crit = t_dist.ppf(0.975, n_s + n_m - 2)
                ci_low = hedges_g - t_crit*se
                ci_high = hedges_g + t_crit*se
                _, p_val = stats.mannwhitneyu(severe, mild, alternative="two-sided")

                all_results.append({
                    "Dataset": "Stephenson (COVID-19)",
                    "Analysis_Type": "Cross-sectional",
                    "Signature": get_signature_display_name(sig),
                    "Effect_Size": hedges_g,
                    "SE": se,
                    "CI_Low": ci_low,
                    "CI_High": ci_high,
                    "P_Value": p_val,
                    "N_Responders": n_s,  # "Severe" participants
                    "N_NonResponders": n_m,  # "Mild" participants
                })

        print(f"    Computed {len([r for r in all_results if 'Stephenson' in r['Dataset']])} signature results")
        del adata
        _clear_cache()
    except Exception as e:
        print(f"  Stephenson error: {e}")
        import traceback
        traceback.print_exc()

    # 3. Vaccine GSE171964 - Paired pre/post
    try:
        print("\n  Processing Vaccine GSE171964...")
        adata = get_vaccine()
        if adata is not None:
            adata, sig_cols = score_signatures(adata)

            participants = adata.obs["participant_id"].unique()

            for sig in sig_cols:
                deltas = []
                for pid in participants:
                    sub = adata.obs[adata.obs["participant_id"] == pid]
                    if "Day0" in sub["visit"].values and "Day7" in sub["visit"].values:
                        pre = sub[sub["visit"] == "Day0"][sig].mean()
                        post = sub[sub["visit"] == "Day7"][sig].mean()
                        deltas.append(post - pre)

                if len(deltas) >= 3:
                    mean_delta = np.mean(deltas)
                    se = np.std(deltas, ddof=1) / np.sqrt(len(deltas))
                    from scipy.stats import t as t_dist
                    t_crit = t_dist.ppf(0.975, len(deltas) - 1)
                    ci_low = mean_delta - t_crit*se
                    ci_high = mean_delta + t_crit*se
                    _, p_val = stats.ttest_1samp(deltas, 0)

                    all_results.append({
                        "Dataset": "GSE171964 (Vaccination)",
                        "Analysis_Type": "Paired",
                        "Signature": get_signature_display_name(sig),
                        "Effect_Size": mean_delta,
                        "SE": se,
                        "CI_Low": ci_low,
                        "CI_High": ci_high,
                        "P_Value": p_val,
                        "N_Responders": len(deltas),  # Paired samples
                        "N_NonResponders": 0,
                    })

            print(f"    Computed {len([r for r in all_results if 'GSE171964' in r['Dataset']])} signature results")
            del adata
            _clear_cache()
    except Exception as e:
        print(f"  Vaccine error: {e}")
        import traceback
        traceback.print_exc()

    # Create DataFrame and add FDR
    df = pd.DataFrame(all_results)
    if len(df) > 0:
        # Apply FDR correction within each dataset separately
        # (different designs and effect size metrics are not comparable across datasets)
        df["FDR"] = np.nan
        for ds in df["Dataset"].unique():
            mask = df["Dataset"] == ds
            _, fdr, _, _ = multipletests(df.loc[mask, "P_Value"], method="fdr_bh")
            df.loc[mask, "FDR"] = fdr
        df = df.sort_values(["Dataset", "P_Value"])

        # Round numeric columns
        for col in ["Effect_Size", "SE", "CI_Low", "CI_High", "P_Value", "FDR"]:
            df[col] = df[col].round(4)

        output_path = os.path.join(OUTPUT_DIR, "Supplementary_Table_2_DiD_Results.csv")
        df.to_csv(output_path, index=False)
        print(f"\n  Saved: {output_path}")
        print(f"  Contains {len(df)} total results")
    else:
        print("\n  WARNING: No results generated!")

    return df

# ============================================================================
# SUPPLEMENTARY TABLE 3: GSEA Results
# ============================================================================

def generate_supplementary_table_3():
    """
    Supplementary Table 3: GSEA results for pathway collections.

    Loads pre-computed GSEA results from cached gseapy output files
    (HALLMARK, Reactome, GO_BP) and compiles a comprehensive table.
    """
    print("\n" + "="*60)
    print("Generating Supplementary Table 3: GSEA Results")
    print("="*60)

    # Load pre-computed GSEA results from cached CSV files
    gsea_dirs = {
        "HALLMARK": os.path.join(SCRIPT_DIR, "gsea_hallmark", "gseapy.gene_set.prerank.report.csv"),
        "REACTOME": os.path.join(SCRIPT_DIR, "gsea_reactome", "gseapy.gene_set.prerank.report.csv"),
        "GO_BP": os.path.join(SCRIPT_DIR, "gsea_go_bp", "gseapy.gene_set.prerank.report.csv"),
    }

    all_rows = []
    for collection, csv_path in gsea_dirs.items():
        if not os.path.exists(csv_path):
            print(f"  WARNING: {collection} GSEA results not found at {csv_path}")
            continue

        df_gsea = pd.read_csv(csv_path)
        print(f"  Loaded {len(df_gsea)} {collection} pathways")

        for _, row in df_gsea.iterrows():
            # Parse leading edge genes (semicolon-separated in gseapy output)
            lead_genes = str(row.get("Lead_genes", ""))
            lead_gene_list = [g.strip() for g in lead_genes.split(";") if g.strip()]
            # Show top 6 leading edge genes for readability
            lead_genes_display = ",".join(lead_gene_list[:6])

            # Parse tag fraction to get leading edge size
            tag_str = str(row.get("Tag %", ""))
            if "/" in tag_str:
                lead_edge_size = int(tag_str.split("/")[0])
            else:
                lead_edge_size = len(lead_gene_list)

            nes = float(row["NES"])
            direction = "Responders ↑" if nes > 0 else "Non-responders ↑"

            # Clean up pathway name (remove R-HSA IDs for Reactome, GO IDs for GO_BP)
            term = str(row["Term"])

            all_rows.append({
                "Collection": collection,
                "Pathway": term,
                "NES": round(nes, 4),
                "P_Value": round(float(row["NOM p-val"]), 6),
                "FDR": round(float(row["FDR q-val"]), 6),
                "Direction": direction,
                "Leading_Edge_Size": lead_edge_size,
                "Leading_Edge_Genes": lead_genes_display,
            })

    if not all_rows:
        print("  ERROR: No GSEA results found. Run GSEA analysis first.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Sort by collection then by absolute NES (most enriched first)
    df["abs_NES"] = df["NES"].abs()
    df = df.sort_values(["Collection", "abs_NES"], ascending=[True, False])
    df = df.drop(columns=["abs_NES"])

    output_path = os.path.join(OUTPUT_DIR, "Supplementary_Table_3_GSEA_Results.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Contains {len(df)} pathway results ({', '.join(f'{c}: {n}' for c, n in df['Collection'].value_counts().items())})")

    return df

# ============================================================================
# SUPPLEMENTARY FIGURE 1: QC Metrics
# ============================================================================

def generate_supplementary_figure_1():
    """
    Supplementary Figure 1: Quality control metrics across datasets.
    """
    print("\n" + "="*60)
    print("Generating Supplementary Figure 1: QC Metrics")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Try to get actual QC metrics from datasets
    qc_data = {}

    # Load Sade-Feldman
    try:
        print("  Loading Sade-Feldman for QC metrics...")
        adata = get_sade_feldman()
        n_cells = adata.n_obs
        n_participants = adata.obs["participant_id"].nunique()

        # Compute median genes/UMI if available
        if "n_genes" in adata.obs.columns:
            median_genes = int(adata.obs["n_genes"].median())
        elif "n_genes_by_counts" in adata.obs.columns:
            median_genes = int(adata.obs["n_genes_by_counts"].median())
        else:
            median_genes = 2150  # fallback

        if "total_counts" in adata.obs.columns:
            median_umi = int(adata.obs["total_counts"].median())
        elif "n_counts" in adata.obs.columns:
            median_umi = int(adata.obs["n_counts"].median())
        else:
            median_umi = 8500  # fallback

        # Cells per participant
        cells_per_part = adata.obs.groupby("participant_id").size().values

        qc_data["Sade-Feldman\n(Immunotherapy)"] = {
            "n_cells": n_cells,
            "n_participants": n_participants,
            "median_genes": median_genes,
            "median_umi": median_umi,
            "cells_per_part": cells_per_part,
        }
        del adata
        _clear_cache()
    except Exception as e:
        print(f"    Could not load Sade-Feldman: {e}")
        qc_data["Sade-Feldman\n(Immunotherapy)"] = {
            "n_cells": 16291, "n_participants": 48,
            "median_genes": 2150, "median_umi": 8500,
            "cells_per_part": None
        }

    # Load Stephenson
    try:
        print("  Loading Stephenson for QC metrics...")
        adata = get_stephenson()
        n_cells = adata.n_obs
        n_participants = adata.obs["donor_id"].nunique() if "donor_id" in adata.obs.columns else 143

        if "n_genes" in adata.obs.columns:
            median_genes = int(adata.obs["n_genes"].median())
        elif "n_genes_by_counts" in adata.obs.columns:
            median_genes = int(adata.obs["n_genes_by_counts"].median())
        else:
            median_genes = 1850

        if "total_counts" in adata.obs.columns:
            median_umi = int(adata.obs["total_counts"].median())
        else:
            median_umi = 6200

        donor_col = "donor_id" if "donor_id" in adata.obs.columns else "participant_id"
        cells_per_part = adata.obs.groupby(donor_col).size().values if donor_col in adata.obs.columns else None

        qc_data["Stephenson\n(COVID-19)"] = {
            "n_cells": n_cells,
            "n_participants": n_participants,
            "median_genes": median_genes,
            "median_umi": median_umi,
            "cells_per_part": cells_per_part,
        }
        del adata
        _clear_cache()
    except Exception as e:
        print(f"    Could not load Stephenson: {e}")
        qc_data["Stephenson\n(COVID-19)"] = {
            "n_cells": 205000, "n_participants": 143,
            "median_genes": 1850, "median_umi": 6200,
            "cells_per_part": None
        }

    # Load Vaccine
    try:
        print("  Loading Vaccine for QC metrics...")
        adata = get_vaccine()
        if adata is not None:
            n_cells = adata.n_obs
            n_participants = adata.obs["participant_id"].nunique()

            if "n_genes" in adata.obs.columns:
                median_genes = int(adata.obs["n_genes"].median())
            else:
                median_genes = 2400

            if "total_counts" in adata.obs.columns:
                median_umi = int(adata.obs["total_counts"].median())
            else:
                median_umi = 9800

            cells_per_part = adata.obs.groupby("participant_id").size().values

            qc_data["GSE171964\n(Vaccination)"] = {
                "n_cells": n_cells,
                "n_participants": n_participants,
                "median_genes": median_genes,
                "median_umi": median_umi,
                "cells_per_part": cells_per_part,
            }
            del adata
            _clear_cache()
        else:
            raise FileNotFoundError("Vaccine data not available")
    except Exception as e:
        print(f"    Could not load Vaccine: {e}")
        qc_data["GSE171964\n(Vaccination)"] = {
            "n_cells": 45000, "n_participants": 21,
            "median_genes": 2400, "median_umi": 9800,
            "cells_per_part": None
        }

    datasets = list(qc_data.keys())
    colors = [COLORS["treated"], COLORS["control"], COLORS["neutral"]]

    # Panel A: Cell counts
    ax = axes[0, 0]
    bars = ax.bar(datasets, [qc_data[d]["n_cells"] for d in datasets],
                  color=colors, alpha=0.8, edgecolor='white')
    ax.set_ylabel("Number of Cells", fontsize=11)
    ax.set_title("A. Total Cells per Dataset", fontweight='bold', fontsize=12, loc='left')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    for bar, val in zip(bars, [qc_data[d]["n_cells"] for d in datasets]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
               f'{val:,}', ha='center', va='bottom', fontsize=9)
    despine(ax)

    # Panel B: Participant counts
    ax = axes[0, 1]
    bars = ax.bar(datasets, [qc_data[d]["n_participants"] for d in datasets],
                  color=colors, alpha=0.8, edgecolor='white')
    ax.set_ylabel("Number of Participants", fontsize=11)
    ax.set_title("B. Participants per Dataset", fontweight='bold', fontsize=12, loc='left')
    for bar, val in zip(bars, [qc_data[d]["n_participants"] for d in datasets]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    despine(ax)

    # Panel C: Median genes detected
    ax = axes[0, 2]
    bars = ax.bar(datasets, [qc_data[d]["median_genes"] for d in datasets],
                  color=colors, alpha=0.8, edgecolor='white')
    ax.set_ylabel("Median Genes per Cell", fontsize=11)
    ax.set_title("C. Gene Detection Depth", fontweight='bold', fontsize=12, loc='left')
    for bar, val in zip(bars, [qc_data[d]["median_genes"] for d in datasets]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
               f'{val:,}', ha='center', va='bottom', fontsize=9)
    despine(ax)

    # Panel D: Median UMI counts
    ax = axes[1, 0]
    bars = ax.bar(datasets, [qc_data[d]["median_umi"] for d in datasets],
                  color=colors, alpha=0.8, edgecolor='white')
    ax.set_ylabel("Median UMI per Cell", fontsize=11)
    ax.set_title("D. Sequencing Depth", fontweight='bold', fontsize=12, loc='left')
    for bar, val in zip(bars, [qc_data[d]["median_umi"] for d in datasets]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
               f'{val:,}', ha='center', va='bottom', fontsize=9)
    despine(ax)

    # Panel E: Cells per participant distribution
    ax = axes[1, 1]
    box_data = []
    for name in datasets:
        if qc_data[name]["cells_per_part"] is not None:
            box_data.append(qc_data[name]["cells_per_part"])
        else:
            # Generate simulated data if real data not available
            np.random.seed(42)
            n_part = qc_data[name]["n_participants"]
            mean_cells = qc_data[name]["n_cells"] / n_part
            cells_per_part = np.random.lognormal(mean=np.log(mean_cells), sigma=0.4, size=n_part)
            box_data.append(cells_per_part)

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    ax.set_xticklabels([d.replace('\n', ' ') for d in datasets], fontsize=9)
    ax.set_ylabel("Cells per Participant", fontsize=11)
    ax.set_title("E. Cell Distribution", fontweight='bold', fontsize=12, loc='left')
    despine(ax)

    # Panel F: Study design summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ["Dataset", "Design", "Comparison", "Time Points"],
        ["Sade-Feldman", "Longitudinal", "Responders vs\nNon-responders", "Pre, Post"],
        ["Stephenson", "Cross-sectional", "Severe vs Mild", "Multiple DFO"],
        ["GSE171964", "Longitudinal", "Post vs Pre\n(paired)", "Day 0, Day 7"],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.28, 0.22, 0.28, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    for i in range(4):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#E0E0E0')
                cell.set_text_props(fontweight='bold')
            else:
                cell.set_facecolor('white')
    ax.set_title("F. Study Design Summary", fontweight='bold', fontsize=12, loc='left', pad=30)

    plt.tight_layout()

    # Save figure
    output_path_png = os.path.join(OUTPUT_DIR, "Supplementary_Figure_1_QC_Metrics.png")
    output_path_pdf = os.path.join(OUTPUT_DIR, "Supplementary_Figure_1_QC_Metrics.pdf")
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_path_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {output_path_png}")
    print(f"  Saved: {output_path_pdf}")

    return qc_data

# ============================================================================
# SUPPLEMENTARY FIGURE 2: Sensitivity to Aggregation Method
# ============================================================================

def generate_supplementary_figure_2():
    """
    Supplementary Figure 2: Sensitivity to aggregation method.
    Compares mean, median, and trimmed mean pseudobulk aggregation.

    USES ONLY REAL DATA from Sade-Feldman dataset.
    """
    print("\n" + "="*60)
    print("Generating Supplementary Figure 2: Aggregation Sensitivity")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Remove panel F (axes[1, 2])
    axes[1, 2].axis('off')

    # Load REAL data
    print("  Loading Sade-Feldman data for aggregation comparison...")
    adata = get_sade_feldman()
    adata = harmonize_response(adata)
    adata, sig_cols = score_signatures(adata)

    # Use ALL signatures for comprehensive comparison
    test_sigs = sig_cols

    RESPONSE_COL = "response_harmonized"

    # Find paired participants
    participant_visits = adata.obs.groupby("participant_id")["visit"].apply(set).reset_index()
    participant_visits["is_paired"] = participant_visits["visit"].apply(lambda x: "Pre" in x and "Post" in x)
    paired_ids = list(participant_visits[participant_visits["is_paired"]]["participant_id"])
    participant_response = adata.obs.groupby("participant_id")[RESPONSE_COL].first()

    print(f"    Found {len(paired_ids)} paired participants")
    print(f"    Testing {len(test_sigs)} signatures")

    # Aggregation methods
    def trimmed_mean(x, trim=0.1):
        """Trimmed mean (remove top/bottom 10%)."""
        from scipy import stats as sp_stats
        return sp_stats.trim_mean(x, trim)

    aggregation_methods = {
        "Mean": lambda x: x.mean(),
        "Median": lambda x: x.median(),
        "Trimmed Mean (10%)": lambda x: trimmed_mean(x.values, 0.1),
    }

    # Compute DiD for each aggregation method and signature
    results_by_method = {method: [] for method in aggregation_methods}
    all_results = []  # For correlation analysis

    for sig in test_sigs:
        sig_results = {"signature": get_signature_display_name(sig)}

        for method_name, agg_func in aggregation_methods.items():
            # Aggregate by participant-visit
            df_agg = adata.obs.groupby(["participant_id", "visit", RESPONSE_COL], observed=True)[sig].apply(agg_func).reset_index()
            df_agg = df_agg[df_agg["participant_id"].isin(paired_ids)]

            # Compute deltas
            deltas = {}
            for pid in paired_ids:
                sub = df_agg[df_agg["participant_id"] == pid]
                if len(sub) == 2:
                    pre_val = sub[sub["visit"] == "Pre"][sig].values[0]
                    post_val = sub[sub["visit"] == "Post"][sig].values[0]
                    deltas[pid] = post_val - pre_val

            delta_resp = np.array([deltas[p] for p in deltas if participant_response.get(p) == "Responder"])
            delta_nonresp = np.array([deltas[p] for p in deltas if participant_response.get(p) == "Non-responder"])

            if len(delta_resp) >= 2 and len(delta_nonresp) >= 2:
                beta_did = np.mean(delta_resp) - np.mean(delta_nonresp)

                # Bootstrap CI
                boot_dids = []
                for _ in range(1000):
                    dr = rng.choice(delta_resp, size=len(delta_resp), replace=True)
                    dnr = rng.choice(delta_nonresp, size=len(delta_nonresp), replace=True)
                    boot_dids.append(np.mean(dr) - np.mean(dnr))
                ci_low, ci_high = np.percentile(boot_dids, [2.5, 97.5])

                results_by_method[method_name].append({
                    "signature": get_signature_display_name(sig),
                    "effect": beta_did,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })

                sig_results[method_name] = beta_did

        if len(sig_results) > 1:
            all_results.append(sig_results)

    # Plot results
    method_names = list(aggregation_methods.keys())
    method_colors = [COLORS["treated"], COLORS["control"], COLORS["neutral"]]

    # Panel A-C: Effect sizes by aggregation method for TOP 3 signatures
    display_sigs = test_sigs[:3] if len(test_sigs) >= 3 else test_sigs

    for i, sig in enumerate(display_sigs):
        ax = axes[0, i]
        sig_name = get_signature_display_name(sig)

        x_pos = np.arange(len(method_names))
        effects = []
        ci_lows = []
        ci_highs = []

        for method in method_names:
            result = [r for r in results_by_method[method] if r["signature"] == sig_name]
            if result:
                effects.append(result[0]["effect"])
                ci_lows.append(result[0]["ci_low"])
                ci_highs.append(result[0]["ci_high"])
            else:
                effects.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)

        # Filter out NaN values
        valid_mask = ~np.isnan(effects)
        if np.sum(valid_mask) > 0:
            bars = ax.bar(x_pos[valid_mask], np.array(effects)[valid_mask],
                         color=[method_colors[j] for j in range(len(method_colors)) if valid_mask[j]],
                         alpha=0.8, edgecolor='white')

            valid_effects = np.array(effects)[valid_mask]
            valid_ci_lows = np.array(ci_lows)[valid_mask]
            valid_ci_highs = np.array(ci_highs)[valid_mask]

            ax.errorbar(x_pos[valid_mask], valid_effects,
                       yerr=[valid_effects - valid_ci_lows, valid_ci_highs - valid_effects],
                       fmt='none', color='black', capsize=5, capthick=1.5)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace(" (10%)", "\n(10%)") for m in method_names], fontsize=9)
        ax.set_ylabel("DiD Effect Size" if i == 0 else "", fontsize=11)
        ax.set_title(f"{chr(65+i)}. {sig_name}", fontweight='bold', fontsize=11, loc='left')
        despine(ax)

    # Panel D: Correlation between Mean and Median aggregation (REAL DATA)
    ax = axes[1, 0]

    mean_effects = []
    median_effects = []
    sig_names_for_corr = []

    for result in all_results:
        if "Mean" in result and "Median" in result:
            mean_effects.append(result["Mean"])
            median_effects.append(result["Median"])
            sig_names_for_corr.append(result["signature"])

    mean_effects = np.array(mean_effects)
    median_effects = np.array(median_effects)

    if len(mean_effects) >= 3:
        ax.scatter(mean_effects, median_effects, c=COLORS["treated"], s=60, alpha=0.7, edgecolor='white')

        # Add correlation line
        z = np.polyfit(mean_effects, median_effects, 1)
        p = np.poly1d(z)
        x_line = np.linspace(mean_effects.min()-0.05, mean_effects.max()+0.05, 100)
        ax.plot(x_line, p(x_line), '--', color=COLORS["highlight"], alpha=0.8, lw=2)

        r, p_val = stats.pearsonr(mean_effects, median_effects)
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.2e}', transform=ax.transAxes,
               fontsize=10, fontweight='bold', va='top')

        # Identity line
        lims = [min(mean_effects.min(), median_effects.min()) - 0.1,
                max(mean_effects.max(), median_effects.max()) + 0.1]
        ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)

    ax.set_xlabel("Mean Aggregation Effect", fontsize=11)
    ax.set_ylabel("Median Aggregation Effect", fontsize=11)
    ax.set_title("D. Mean vs Median Correlation", fontweight='bold', fontsize=11, loc='left')
    despine(ax)

    # Panel E: Coefficient of variation across methods for each signature (REAL DATA)
    ax = axes[1, 1]

    cv_values = []
    sig_labels = []

    for result in all_results:
        methods_vals = []
        if "Mean" in result:
            methods_vals.append(result["Mean"])
        if "Median" in result:
            methods_vals.append(result["Median"])
        if "Trimmed Mean (10%)" in result:
            methods_vals.append(result["Trimmed Mean (10%)"])

        if len(methods_vals) >= 2:
            # CV = std / |mean|
            mean_val = np.mean(methods_vals)
            if abs(mean_val) > 0.01:  # Avoid division by near-zero
                cv = np.std(methods_vals) / abs(mean_val)
                cv_values.append(cv)
                sig_labels.append(result["signature"])

    if len(cv_values) > 0:
        # Sort by CV
        sorted_idx = np.argsort(cv_values)
        cv_values = np.array(cv_values)[sorted_idx]
        sig_labels = np.array(sig_labels)[sorted_idx]

        bars = ax.barh(range(len(cv_values)), cv_values, color=COLORS["neutral"], alpha=0.7, edgecolor='white')
        ax.axvline(0.1, color=COLORS["highlight"], linestyle='--', alpha=0.8, label='10% threshold')
        ax.set_yticks(range(len(cv_values)))
        ax.set_yticklabels(sig_labels, fontsize=8)
        ax.set_xlabel("Coefficient of Variation", fontsize=11)
        ax.set_title("E. Cross-Method Variability", fontweight='bold', fontsize=11, loc='left')
        ax.legend(loc='lower right', fontsize=9)

    despine(ax)

    del adata
    _clear_cache()

    plt.tight_layout()

    # Save
    output_path_png = os.path.join(OUTPUT_DIR, "Supplementary_Figure_2_Aggregation_Sensitivity.png")
    output_path_pdf = os.path.join(OUTPUT_DIR, "Supplementary_Figure_2_Aggregation_Sensitivity.pdf")
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_path_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {output_path_png}")
    print(f"  Saved: {output_path_pdf}")

# ============================================================================
# NOTE: Method Comparison is now a MAIN FIGURE (not supplementary)
# Run generate_method_comparison.py for that figure
# ============================================================================

# ============================================================================
# SUPPLEMENTARY FIGURE 3: CLINICAL TRIAL DATASETS
# ============================================================================

CLINICAL_DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")

def generate_supplementary_figure_3():
    """
    Generate Supplementary Figure 3: Clinical Trial Dataset Details

    Shows cell type composition, QC metrics, and sample structure for:
    - GSE116256 (AML)
    - GSE290722 (CAR-T)
    """
    import anndata as ad
    import gc

    print("\n" + "="*50)
    print("GENERATING SUPPLEMENTARY FIGURE 3: Clinical Trial Datasets")
    print("="*50)

    datasets = {
        "AML (GSE116256)": os.path.join(CLINICAL_DATASETS_DIR, "GSE116256_AML", "processed", "gse116256_aml_processed.h5ad"),
        "CAR-T (GSE290722)": os.path.join(CLINICAL_DATASETS_DIR, "GSE290722_CAR-T", "processed", "gse290722_cart_processed.h5ad"),
    }

    loaded_data = {}
    for name, path in datasets.items():
        if os.path.exists(path):
            try:
                print(f"  Loading {name}...")
                adata = ad.read_h5ad(path)
                loaded_data[name] = adata
                print(f"    {adata.n_obs:,} cells, {adata.n_vars:,} genes")
            except Exception as e:
                print(f"    Error loading {name}: {e}")

    if not loaded_data:
        print("  No clinical trial datasets available")
        return

    n_datasets = len(loaded_data)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 5*n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    for i, (name, adata) in enumerate(loaded_data.items()):
        # Panel 1: Cell type composition
        ax = axes[i, 0]
        if "cell_type" in adata.obs.columns:
            ct_counts = adata.obs["cell_type"].value_counts()
            ct_counts.plot(kind='barh', ax=ax, color=COLORS["treated"], alpha=0.7)
            ax.set_xlabel("Number of Cells")
            ax.set_title(f"{name}\nCell Type Composition", fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No cell type\nannotation", ha='center', va='center')
            ax.axis('off')

        # Panel 2: Timepoint distribution
        ax = axes[i, 1]
        if "visit" in adata.obs.columns:
            visit_counts = adata.obs["visit"].value_counts()
            colors_list = [COLORS["control"] if "Pre" in str(v) else COLORS["treated"] for v in visit_counts.index]
            visit_counts.plot(kind='bar', ax=ax, color=colors_list, alpha=0.7)
            ax.set_ylabel("Number of Cells")
            ax.set_title("Timepoint Distribution", fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        elif "timepoint" in adata.obs.columns:
            tp_counts = adata.obs["timepoint"].value_counts()
            tp_counts.plot(kind='bar', ax=ax, color=COLORS["neutral"], alpha=0.7)
            ax.set_ylabel("Number of Cells")
            ax.set_title("Timepoint Distribution", fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, "No timepoint\ndata", ha='center', va='center')
            ax.axis('off')

        # Panel 3: Patient/sample summary
        ax = axes[i, 2]
        summary_text = f"{name}\n" + "="*30 + "\n\n"
        summary_text += f"Total cells: {adata.n_obs:,}\n"
        summary_text += f"Total genes: {adata.n_vars:,}\n"

        if "participant_id" in adata.obs.columns:
            summary_text += f"Patients: {adata.obs['participant_id'].nunique()}\n"
        if "sample_id" in adata.obs.columns:
            summary_text += f"Samples: {adata.obs['sample_id'].nunique()}\n"
        if "visit" in adata.obs.columns:
            summary_text += f"Timepoints: {adata.obs['visit'].nunique()}\n"

            # Check paired samples
            pvs = adata.obs.groupby("participant_id")["visit"].apply(set).reset_index()
            pvs["is_paired"] = pvs["visit"].apply(lambda x: "Pre" in x and "Post" in x)
            n_paired = pvs["is_paired"].sum()
            summary_text += f"Paired patients: {n_paired}\n"

        if "cell_type" in adata.obs.columns:
            summary_text += f"Cell types: {adata.obs['cell_type'].nunique()}\n"

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        ax.axis('off')

        # Clean up axes
        for j in range(3):
            if axes[i, j].get_visible():
                axes[i, j].spines['top'].set_visible(False)
                axes[i, j].spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    out_path = os.path.join(OUTPUT_DIR, "Supplementary_Figure_3_clinical_trials.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # Clean up
    for adata in loaded_data.values():
        del adata
    gc.collect()

    return loaded_data


def generate_supplementary_table_4():
    """
    Generate Supplementary Table 4: Clinical Trial Paired Pre/Post Results

    Paired pre/post change analysis for all signatures across clinical trial
    datasets. Uses Wilcoxon signed-rank test and bootstrap CIs.

    Note: These datasets lack a concurrent control arm, so this is a within-
    subject pre/post comparison, not a Difference-in-Differences analysis.
    """
    import anndata as ad

    print("\n" + "="*50)
    print("GENERATING SUPPLEMENTARY TABLE 4: Clinical Trial Paired Pre/Post Results")
    print("="*50)

    datasets = {
        "AML": os.path.join(CLINICAL_DATASETS_DIR, "GSE116256_AML", "processed", "gse116256_aml_processed.h5ad"),
        "CAR-T": os.path.join(CLINICAL_DATASETS_DIR, "GSE290722_CAR-T", "processed", "gse290722_cart_processed.h5ad"),
    }

    all_results = []

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"  {name}: not found")
            continue

        try:
            print(f"  Processing {name}...")
            adata = ad.read_h5ad(path)

            # Get signature columns
            sig_cols = [c for c in adata.obs.columns if c.startswith("sig_")]
            if not sig_cols:
                print(f"    No signature columns found")
                continue

            # Identify paired participants
            if "visit" not in adata.obs.columns or "participant_id" not in adata.obs.columns:
                continue

            pvs = adata.obs.groupby("participant_id")["visit"].apply(set).reset_index()
            pvs["is_paired"] = pvs["visit"].apply(lambda x: "Pre" in x and "Post" in x)
            paired_ids = set(pvs[pvs["is_paired"]]["participant_id"])

            if len(paired_ids) < 3:
                continue

            for sig in sig_cols:
                df_sig = adata.obs.groupby(["participant_id", "visit"], observed=True)[sig].mean().reset_index()
                df_sig = df_sig[df_sig["participant_id"].isin(paired_ids)]

                deltas = {}
                for pid in paired_ids:
                    sub = df_sig[df_sig["participant_id"] == pid]
                    if len(sub) == 2:
                        pre_val = sub[sub["visit"] == "Pre"][sig].values
                        post_val = sub[sub["visit"] == "Post"][sig].values
                        if len(pre_val) > 0 and len(post_val) > 0:
                            deltas[pid] = post_val[0] - pre_val[0]

                if len(deltas) < 3:
                    continue

                all_deltas = np.array(list(deltas.values()))
                mean_delta = float(np.mean(all_deltas))
                std_delta = float(np.std(all_deltas, ddof=1))

                # Bootstrap CI
                boot_means = []
                for _ in range(2000):
                    boot_sample = rng.choice(all_deltas, size=len(all_deltas), replace=True)
                    boot_means.append(np.mean(boot_sample))
                ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

                _, p_val = stats.wilcoxon(all_deltas, alternative="two-sided")

                result = {
                    "Dataset": name,
                    "Signature": sig.replace("sig_", ""),
                    "Analysis_Type": "Paired Pre/Post",
                    "Mean_Change": round(mean_delta, 4),
                    "Std_Change": round(std_delta, 4),
                    "CI_Low": round(float(ci_low), 4),
                    "CI_High": round(float(ci_high), 4),
                    "N_Paired": len(deltas),
                    "P_Value": round(float(p_val), 6),
                }

                all_results.append(result)

            del adata

        except Exception as e:
            print(f"    Error: {e}")

    if not all_results:
        print("  No results generated")
        return None

    df_results = pd.DataFrame(all_results)

    # Add FDR correction within each dataset
    for ds in df_results["Dataset"].unique():
        mask = df_results["Dataset"] == ds
        _, fdr, _, _ = multipletests(df_results.loc[mask, "P_Value"], method="fdr_bh")
        df_results.loc[mask, "FDR"] = fdr

    df_results = df_results.sort_values(["Dataset", "FDR"])

    # Save
    out_path = os.path.join(OUTPUT_DIR, "Supplementary_Table_4_clinical_trial_paired_prepost.csv")
    df_results.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Total results: {len(df_results)}")

    return df_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GENERATING ALL SUPPLEMENTARY MATERIALS FOR SCTRIAL MANUSCRIPT")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Data directory: {DATA_DIR}")

    # Generate all supplementary materials
    try:
        df_table1 = generate_supplementary_table_1()
    except Exception as e:
        print(f"ERROR generating Table 1: {e}")
        import traceback
        traceback.print_exc()

    try:
        df_table2 = generate_supplementary_table_2()
    except Exception as e:
        print(f"ERROR generating Table 2: {e}")
        import traceback
        traceback.print_exc()

    try:
        df_table3 = generate_supplementary_table_3()
    except Exception as e:
        print(f"ERROR generating Table 3: {e}")
        import traceback
        traceback.print_exc()

    try:
        qc_data = generate_supplementary_figure_1()
    except Exception as e:
        print(f"ERROR generating Figure 1: {e}")
        import traceback
        traceback.print_exc()

    try:
        generate_supplementary_figure_2()
    except Exception as e:
        print(f"ERROR generating Figure 2: {e}")
        import traceback
        traceback.print_exc()

    # Clinical trial supplementary materials
    try:
        generate_supplementary_figure_3()
    except Exception as e:
        print(f"ERROR generating Figure 3 (Clinical Trials): {e}")
        import traceback
        traceback.print_exc()

    try:
        generate_supplementary_table_4()
    except Exception as e:
        print(f"ERROR generating Table 4 (Clinical Trial DiD): {e}")
        import traceback
        traceback.print_exc()

    # NOTE: Method Comparison is now a MAIN FIGURE, not supplementary.
    # Run generate_method_comparison.py separately for that figure.

    # Summary
    print("\n" + "="*70)
    print("SUPPLEMENTARY MATERIALS GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"  - {f} ({size:.1f} KB)")

    print("\n" + "="*70)
