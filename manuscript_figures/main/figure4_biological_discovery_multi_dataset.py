"""
Figure 4 — Biological Discovery & Multi-Dataset Generalization
===============================================================

Twelve-panel combined figure integrating gene/pathway-level biological
discovery from the melanoma cohort (panels A–E) with cross-dataset
generalization analyses (panels F–L).

Panels
------
A  Gene-level volcano plot (melanoma DiD).
B  Top genes ranked by effect size (waterfall).
C  GSEA enrichment bar chart (pathway enrichment).
D  Leading-edge gene overlap heatmap (transposed: genes on X, pathways on Y).
E  Cell-type-resolved DiD effect heatmap.
F  COVID-19 cross-sectional forest plot.
G  Vaccine paired forest plot.
H  AML within-arm forest plot.
I  CAR-T forest plot.
J  Melanoma DiD forest plot.
K  Cross-dataset effect-size heatmap.
L  Cross-dataset GSEA heatmap (replicated pathways).
"""

from __future__ import annotations

import gc
import hashlib
import pickle  # noqa: S403 — local dev cache of our own DataFrames
import re
import traceback
import warnings
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    load_or_run_gsea_cross_sectional,
    load_or_run_gsea_did,
    load_or_run_gsea_within_arm,
    save_panel,
    score_signatures,
    sig_display,
    within_arm_comparison,
)


FIGURE_NAME = "Figure4_biological_discovery_multi_dataset"


# ======================================================================
# FIGURE 4 — Biological Discovery (Melanoma)
# ======================================================================

# ── Cache directory for expensive computations ─────────────────────────
_CACHE_DIR = Path(__file__).resolve().parent.parent / "_cache"

# Pseudogene / non-coding RNA / mitochondrial / ribosomal patterns to
# EXCLUDE from volcano labels — only protein-coding genes get labelled.
_NONCODING_PATTERN = re.compile(
    r"^(RNU\d|RNA5SP|RNY\d|RN7S[LK]|SNOR[AD]|MIR\d|LINC\d|LOC\d|"
    r"AC\d{6}|AL\d{6}|AP\d{6}|"
    r"RP\d+-|RP[SL]\d+P|CT[ABCD]-|XXbac-|KB-|LA16c-|GS\d-|"
    r"HIGD1AP|MTCO\d|RMVSL|BCRP|NAMA$|SLMO|"
    r"IGH[VDJ]|IGKV|IGLV|IGKJ|IGLJ|"  # Ig V/D/J segments (NOT constant regions)
    r"TRB[VDJ]|TRA[VDJ]|TRG[VDJ]|TRD[VDJ]|"  # TCR V/D/J segments
    r"OR\d+[A-Z]|VN\d+R|"  # olfactory/vomeronasal receptors
    r"MT-|"  # mitochondrial genes
    r"RPS\d|RPL\d|"  # ribosomal protein genes
    r"HCG\d|SPRR\d|"  # HLA complex group pseudogenes, small proline-rich
    r"HLA-|"  # HLA genes (highly polymorphic, not discovery-informative)
    r"[A-Z]{1,2}\d{2}NC\d|"  # contig-derived lncRNAs (e.g., LL22NC03-75A1.9)
    r"[A-Z]\d{5}\.\d)",  # Ensembl-style identifiers (e.g., Z95704.4)
    re.IGNORECASE,
)
# Suffix patterns for processed pseudogenes (e.g. PRPS1P1, RPS2P1, HCG17)
_PSEUDOGENE_SUFFIX = re.compile(r"P\d+$", re.IGNORECASE)


def _is_likely_protein_coding(gene: str) -> bool:
    """Heuristic: return True if gene name looks protein-coding."""
    if _NONCODING_PATTERN.match(gene) is not None:
        return False
    # Catch processed pseudogenes ending with P + digits (e.g., PRPS1P1, RPS2P1)
    # but exclude genuine genes like TP53, APC, etc. (short suffix, common genes)
    if _PSEUDOGENE_SUFFIX.search(gene) and len(gene) > 4:
        # Only flag if the P-digit suffix is preceded by another digit
        # (e.g., PRPS1P1 → "1P1", but not APC → no match)
        base = _PSEUDOGENE_SUFFIX.sub("", gene)
        if base and base[-1].isdigit():
            return False
    return True


# ======================================================================
# Immune + Metabolic pathway inclusion filter
# ======================================================================
# Instead of manually blacklisting irrelevant pathways (analyst degrees
# of freedom), we INCLUDE only immune-related and metabolic pathways via
# keyword matching.  This is reproducible and transparent.

_IMMUNE_METABOLIC_KEYWORDS = [
    # ── Immune ──
    "immune", "inflamm", "cytokine", "interleukin", "interferon",
    "leukocyte", "lymphocyte", "T cell", "B cell", "NK cell",
    "macrophage", "monocyte", "dendritic", "neutrophil", "phagocyt",
    "antigen", "MHC", "complement", "toll-like", "chemokine",
    "apoptot", "autophagy", "NF-kB", "JAK-STAT", "TNF",
    "defense response", "innate immune", "adaptive immune",
    "myeloid", "granulocyte", "eosinophil", "basophil", "natural killer",
    "antibody", "immunoglobulin", "hematopoietic", "lymph", "thymus",
    "spleen", "response to virus", "response to bacter", "viral",
    "type I interferon", "cell killing",
    # ── Metabolic ──
    "oxidative phosphorylation", "glycolysis", "gluconeogenesis",
    "fatty acid", "lipid", "cholesterol", "bile acid",
    "amino acid", "glutamine", "glutamate", "tryptophan", "arginine",
    "citric acid", "TCA cycle", "krebs", "electron transport",
    "mitochondri", "respiratory chain", "OXPHOS",
    "pentose phosphate", "nucleotide", "purine", "pyrimidine",
    "xenobiotic", "drug metabolism", "metabolism",
    "mTOR", "AMPK", "PI3K", "Akt", "Wnt",
    "hypoxia", "HIF", "reactive oxygen", "ROS", "oxidative stress",
    "ferroptosis", "iron", "heme",
]
_IMMUNE_METABOLIC_RE = re.compile(
    "|".join(re.escape(kw) for kw in _IMMUNE_METABOLIC_KEYWORDS),
    re.IGNORECASE,
)


def _is_immune_or_metabolic(term: str) -> bool:
    """Return True if pathway name matches immune or metabolic keywords."""
    return _IMMUNE_METABOLIC_RE.search(str(term)) is not None


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_bio_discovery_data(*, use_cache: bool = True) -> dict:
    """Load datasets, run DiD, GSEA, and gene-level analysis.

    Results are cached to disk (pickle) because gene-level DiD across
    ~2000 genes takes several minutes.  Set ``use_cache=False`` to
    force recomputation.
    """
    _code_hash = hashlib.md5(  # noqa: S324 — not security, just cache tag
        Path(__file__).read_bytes()
    ).hexdigest()[:8]
    cache_key = f"figure4_sade_feldman_v4_{_code_hash}"
    cache_path = _CACHE_DIR / f"{cache_key}.pkl"

    if use_cache and cache_path.exists():
        print(f"  Loading cached data from {cache_path.name}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)  # noqa: S301 — trusted local cache
        # Reload adata (not cached — too large) for any panel that needs it
        adata = get_sade_feldman()
        if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
            adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
        adata, sig_cols = score_signatures(adata, layer="log1p_tpm")
        adata = harmonize_response(adata)
        cached["adata"] = adata
        cached["sig_cols"] = sig_cols
        return cached

    # ------------------------------------------------------------------
    # 1. Sade-Feldman: signature-level DiD + GSEA
    # ------------------------------------------------------------------
    adata = get_sade_feldman()

    if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])

    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")
    adata = harmonize_response(adata)

    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="response_harmonized",
        arm_treated="Responder",
        arm_control="Non-responder",
    )
    visits = ("Pre", "Post")

    # Signature-level DiD with bootstrap for small-sample CIs
    did_sig = did_table(
        adata,
        features=sig_cols,
        design=design,
        visits=visits,
        layer="log1p_tpm",
        standardize=True,
        aggregate="participant_visit",
        use_bootstrap=True,
        n_boot=999,
        seed=42,
    )

    # GSEA on multiple gene-set libraries for comprehensive pathway analysis
    # Uses shared helper that caches results under manuscript/GSEA/Melanoma/
    gsea_results = load_or_run_gsea_did(
        adata, design, visits, "log1p_tpm", "Melanoma",
    )

    if gsea_results is not None and len(gsea_results) > 0:
        # Detect term column for immune/metabolic filtering
        term_col = None
        for c in gsea_results.columns:
            cl = c.lower().strip()
            if cl == "term":
                term_col = c
                break
        if term_col is None:
            for c in gsea_results.columns:
                if c.lower() in ("name", "pathway"):
                    term_col = c
                    break
            if term_col is None:
                term_col = gsea_results.columns[0]

        # Filter to immune + metabolic pathways only
        n_before = len(gsea_results)
        gsea_results = gsea_results[
            gsea_results[term_col].apply(_is_immune_or_metabolic)
        ].reset_index(drop=True)
        n_after = len(gsea_results)
        print(
            f"  Immune/metabolic filter: {n_after}/{n_before} pathways retained"
        )

        # Handle duplicate pathways across libraries: average NES
        nes_col_name = None
        for c in gsea_results.columns:
            if c.lower().strip() == "nes":
                nes_col_name = c
                break
        if nes_col_name is None:
            for c in gsea_results.columns:
                if "nes" in c.lower():
                    nes_col_name = c
                    break

        if nes_col_name is not None:
            # Clean pathway names for dedup comparison
            gsea_results["_clean_term"] = gsea_results[term_col].apply(
                lambda s: _clean_pathway_name(s, max_len=200)
            )
            gsea_results[nes_col_name] = pd.to_numeric(
                gsea_results[nes_col_name], errors="coerce"
            )

            # For pathways that appear in multiple libraries, average their
            # numeric columns and concatenate leading-edge genes
            nom_col = None
            for c in gsea_results.columns:
                cl = c.lower().strip()
                if cl in ("nom p-val", "pval", "p-value", "nom_pval"):
                    nom_col = c
                    break

            fdr_col_detect = None
            for c in gsea_results.columns:
                if c.lower().strip() in ("fdr q-val", "fdr"):
                    fdr_col_detect = c
                    break
            if fdr_col_detect is None:
                for c in gsea_results.columns:
                    if "fdr" in c.lower():
                        fdr_col_detect = c
                        break

            lead_col_detect = None
            for c in gsea_results.columns:
                cl = c.lower().strip()
                if cl in ("lead_genes", "leading_edge"):
                    lead_col_detect = c
                    break
            if lead_col_detect is None:
                for c in gsea_results.columns:
                    if "lead" in c.lower() and "gene" in c.lower():
                        lead_col_detect = c
                        break

            # Check for duplicates
            dup_mask = gsea_results.duplicated(subset=["_clean_term"], keep=False)
            n_dup_pathways = gsea_results.loc[dup_mask, "_clean_term"].nunique()
            if n_dup_pathways > 0:
                print(f"  Averaging {n_dup_pathways} duplicate pathways "
                      f"across libraries")

                # For duplicates: group and average numeric cols, union lead genes
                deduped_rows = []
                for clean_name, group in gsea_results.groupby("_clean_term"):
                    if len(group) == 1:
                        deduped_rows.append(group.iloc[0].to_dict())
                        continue
                    row = group.iloc[0].to_dict()
                    # Average NES
                    row[nes_col_name] = group[nes_col_name].mean()
                    # Average NOM p-val if available
                    if nom_col is not None and nom_col in group.columns:
                        row[nom_col] = pd.to_numeric(
                            group[nom_col], errors="coerce"
                        ).mean()
                    # Average FDR
                    if fdr_col_detect is not None and fdr_col_detect in group.columns:
                        row[fdr_col_detect] = pd.to_numeric(
                            group[fdr_col_detect], errors="coerce"
                        ).mean()
                    # Union leading-edge genes
                    if lead_col_detect is not None and lead_col_detect in group.columns:
                        all_genes = set()
                        for _, r in group.iterrows():
                            gs = str(r[lead_col_detect])
                            all_genes.update(
                                g.strip()
                                for g in gs.replace(";", ",").split(",")
                                if g.strip()
                            )
                        row[lead_col_detect] = ";".join(sorted(all_genes))
                    # Mark as averaged
                    row["library"] = "averaged"
                    deduped_rows.append(row)

                gsea_results = pd.DataFrame(deduped_rows)

            gsea_results = gsea_results.drop(columns=["_clean_term"], errors="ignore")

        # Recompute FDR across the *pooled* pathway universe to control
        # multiplicity properly when combining libraries.
        fdr_col_name = None
        for c in gsea_results.columns:
            if c.lower().strip() in ("fdr q-val", "fdr"):
                fdr_col_name = c
                break
        if fdr_col_name is None:
            for c in gsea_results.columns:
                if "fdr" in c.lower():
                    fdr_col_name = c
                    break
        if fdr_col_name is not None:
            from statsmodels.stats.multitest import multipletests
            # Preserve per-library FDR, add pooled FDR
            gsea_results["FDR_per_library"] = gsea_results[fdr_col_name]
            nom_col = None
            for c in gsea_results.columns:
                cl = c.lower().strip()
                if cl in ("nom p-val", "pval", "p-value", "nom_pval"):
                    nom_col = c
                    break
            if nom_col is not None:
                pvals = pd.to_numeric(
                    gsea_results[nom_col], errors="coerce"
                ).fillna(1).values
            else:
                pvals = gsea_results[fdr_col_name].fillna(1).values
            _, fdr_pooled, _, _ = multipletests(
                pvals, method="fdr_bh",
            )
            gsea_results[fdr_col_name] = fdr_pooled
            n_sig_pooled = (fdr_pooled < 0.25).sum()
            print(
                f"  GSEA total: {len(gsea_results)} immune/metabolic pathways "
                f"({n_sig_pooled} FDR<0.25 after pooled correction)"
            )
        else:
            print(
                f"  GSEA total: {len(gsea_results)} immune/metabolic pathways"
            )
    else:
        gsea_results = None

    # ------------------------------------------------------------------
    # 2. Sade-Feldman gene-level DiD (top variable genes)
    # ------------------------------------------------------------------
    gene_results = None
    try:
        import scanpy as sc

        adata_genes = adata.copy()
        sc.pp.highly_variable_genes(
            adata_genes, n_top_genes=2000, layer="log1p_tpm", flavor="seurat",
        )
        top_genes = adata_genes.var_names[
            adata_genes.var["highly_variable"]
        ].tolist()
        print(f"  Sade-Feldman: {len(top_genes)} variable genes selected")

        # Use analytical (nonrobust) SEs for gene-level volcano.
        # Bootstrap is used for signature-level inference (Panel D) where
        # formal hypothesis tests matter; for the volcano (Panel E) we use
        # analytical SEs to provide a richer visualisation of effect-size
        # patterns across ~2000 genes.
        gene_results = did_table(
            adata_genes,
            features=top_genes,
            design=design,
            visits=visits,
            layer="log1p_tpm",
            standardize=True,
            aggregate="participant_visit",
        )

        # Fix #1: Handle degenerate gene-level fits.
        # With n=10 clusters (5 participants × 2 visits), cluster-robust
        # SEs often fail for some genes, producing NaN in se_DiD/p_DiD.
        # We drop these explicitly and log the count.
        n_total = len(gene_results)
        n_degenerate = gene_results["p_DiD"].isna().sum()
        if n_degenerate > 0:
            gene_results = gene_results.dropna(
                subset=["beta_DiD", "se_DiD", "p_DiD"]
            ).reset_index(drop=True)
            print(
                f"  Gene-level: dropped {n_degenerate}/{n_total} genes "
                f"with degenerate fits (NaN SE/p-value)"
            )

        n_sig = (gene_results["p_DiD"] < 0.05).sum()
        n_fdr = (gene_results["FDR_DiD"] < 0.1).sum()
        print(
            f"  Gene-level results: {len(gene_results)} genes, "
            f"{n_sig} nominal p<0.05, {n_fdr} FDR<0.1"
        )

        del adata_genes
        gc.collect()

    except Exception as exc:
        print(f"  Gene-level analysis unavailable: {exc}")
        traceback.print_exc()
        gene_results = None

    # ------------------------------------------------------------------
    # 3. Multi-dataset GSEA for pathway replication analysis
    # ------------------------------------------------------------------
    # Reuse Sade-Feldman GSEA results from above to avoid recomputation
    gsea_multi_dataset = _run_multi_dataset_gsea(sf_gsea_results=gsea_results)

    result = dict(
        adata=adata,
        sig_cols=sig_cols,
        design=design,
        visits=visits,
        did_sig=did_sig,
        gsea_results=gsea_results,
        gene_results=gene_results,
        gsea_multi_dataset=gsea_multi_dataset,
    )

    # Cache everything except adata (too large) and sig_cols (recomputed)
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        to_cache = {k: v for k, v in result.items()
                    if k not in ("adata", "sig_cols")}
        with open(cache_path, "wb") as f:
            pickle.dump(to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Cached results to {cache_path.name}")

    return result


# ======================================================================
# Multi-dataset GSEA for pathway replication
# ======================================================================

def _run_multi_dataset_gsea(sf_gsea_results: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Run GSEA on all 5 datasets and return results per dataset.
    
    Uses all 5 gene set libraries (Hallmark, KEGG, Reactome, GO BP, WikiPathways)
    for comprehensive pathway coverage, matching the single-dataset analysis.
    
    Parameters
    ----------
    sf_gsea_results : pd.DataFrame, optional
        Pre-computed GSEA results for Sade-Feldman from _prepare_data().
        If provided, these will be reused to avoid recomputation.
    
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping dataset name to GSEA results DataFrame.
    """
    print("  Running GSEA on all datasets for pathway replication...")
    gsea_multi = {}
    
    # 1. Melanoma - reuse results from _prepare_data() if available
    if sf_gsea_results is not None and len(sf_gsea_results) > 0:
        sf_results = sf_gsea_results.copy()
        if "dataset" not in sf_results.columns:
            sf_results["dataset"] = "Melanoma"
        else:
            sf_results["dataset"] = "Melanoma"
        if "Library" in sf_results.columns and "library" not in sf_results.columns:
            sf_results = sf_results.rename(columns={"Library": "library"})
        gsea_multi["Melanoma"] = sf_results
        print(f"    Melanoma: {len(sf_results)} pathways (reused from _prepare_data)")
    else:
        # Fallback: use load_or_run_gsea_did for caching
        try:
            sf = get_sade_feldman()
            sf = harmonize_response(sf)
            if "log1p_tpm" not in sf.layers and "tpm" in sf.layers:
                sf.layers["log1p_tpm"] = np.log1p(sf.layers["tpm"])
            sf_design = TrialDesign(
                participant_col="participant_id",
                visit_col="visit",
                arm_col="response_harmonized",
                arm_treated="Responder",
                arm_control="Non-responder",
            )
            sf_results = load_or_run_gsea_did(
                sf, sf_design, ("Pre", "Post"), "log1p_tpm", "Melanoma",
            )
            if sf_results is not None and len(sf_results) > 0:
                sf_results["dataset"] = "Melanoma"
                # Rename Library column to library for consistency
                if "Library" in sf_results.columns:
                    sf_results = sf_results.rename(columns={"Library": "library"})
                gsea_multi["Melanoma"] = sf_results
                print(f"    Melanoma: {len(sf_results)} pathways (cached or computed)")
        except Exception as exc:
            print(f"    Melanoma: FAILED ({exc})")
    
    # 2. Vaccine (use within_arm_comparison; .X is already normalized)
    try:
        vax = get_vaccine()
        vax, _ = score_signatures(vax, layer=None)
        vax.obs["arm_dummy"] = "Vaccinated"
        vax_design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm_dummy",
            arm_treated="Vaccinated",
            arm_control="Vaccinated",
        )
        vax_results = load_or_run_gsea_within_arm(
            vax, vax_design, arm="Vaccinated", visits=("Pre", "Post"),
            layer=None, dataset_name="Vaccine",
        )
        if vax_results is not None and len(vax_results) > 0:
            vax_results["dataset"] = "Vaccine"
            if "Library" in vax_results.columns and "library" not in vax_results.columns:
                vax_results = vax_results.rename(columns={"Library": "library"})
            gsea_multi["Vaccine"] = vax_results
            print(f"    Vaccine: {len(vax_results)} pathways")
    except Exception as exc:
        print(f"    Vaccine: FAILED ({exc})")
    
    # 3. AML (use within_arm_comparison; log1p_norm available)
    try:
        aml = get_aml()
        _aml_layer = "log1p_norm" if "log1p_norm" in aml.layers else None
        aml, _ = score_signatures(aml, layer=_aml_layer)
        aml.obs["arm_dummy"] = "Treatment"
        pid_col = ("participant_id" if "participant_id" in aml.obs.columns
                   else "patient_id")
        aml_design = TrialDesign(
            participant_col=pid_col,
            visit_col="visit",
            arm_col="arm_dummy",
            arm_treated="Treatment",
            arm_control="Treatment",
        )
        aml_results = load_or_run_gsea_within_arm(
            aml, aml_design, arm="Treatment", visits=("Pre", "Post"),
            layer=_aml_layer, dataset_name="AML",
        )
        if aml_results is not None and len(aml_results) > 0:
            aml_results["dataset"] = "AML"
            if "Library" in aml_results.columns and "library" not in aml_results.columns:
                aml_results = aml_results.rename(columns={"Library": "library"})
            gsea_multi["AML"] = aml_results
            print(f"    AML: {len(aml_results)} pathways")
    except Exception as exc:
        print(f"    AML: FAILED ({exc})")
    
    # 4. CAR-T (use within_arm_comparison; log1p_norm available)
    try:
        cart = get_cart()
        _cart_layer = "log1p_norm" if "log1p_norm" in cart.layers else None
        cart, _ = score_signatures(cart, layer=_cart_layer)
        cart.obs["arm_dummy"] = "CAR-T"
        pid_col = ("participant_id" if "participant_id" in cart.obs.columns
                   else "patient_id")
        cart_design = TrialDesign(
            participant_col=pid_col,
            visit_col="visit",
            arm_col="arm_dummy",
            arm_treated="CAR-T",
            arm_control="CAR-T",
        )
        cart_results = load_or_run_gsea_within_arm(
            cart, cart_design, arm="CAR-T", visits=("Pre", "Post"),
            layer=_cart_layer, dataset_name="CAR-T",
        )
        if cart_results is not None and len(cart_results) > 0:
            cart_results["dataset"] = "CAR-T"
            if "Library" in cart_results.columns and "library" not in cart_results.columns:
                cart_results = cart_results.rename(columns={"Library": "library"})
            gsea_multi["CAR-T"] = cart_results
            print(f"    CAR-T: {len(cart_results)} pathways")
    except Exception as exc:
        print(f"    CAR-T: FAILED ({exc})")
    
    # 5. COVID-19 (cross-sectional - use between_arm_comparison)
    try:
        covid = get_stephenson()
        # Add log1p_cpm layer if not present (consistent with Figure 5A / Supp Table 3)
        if "log1p_cpm" not in covid.layers and "counts" in covid.layers:
            from .._shared import add_log1p_cpm_layer
            covid = add_log1p_cpm_layer(
                covid, counts_layer="counts", out_layer="log1p_cpm",
            )
        covid, _ = score_signatures(covid, layer="log1p_cpm")
        # Fixed DFO bin to match Figure 5A and Supp Table 3
        target_bin = "DFO_8-14"
        if "dfo_bin" in covid.obs.columns:
            available_bins = sorted(covid.obs["dfo_bin"].dropna().unique())
            if target_bin not in available_bins:
                # Fallback: first bin (sorted) with both severity groups
                for _bin in available_bins:
                    _sub = covid[covid.obs["dfo_bin"] == _bin]
                    if set(_sub.obs["severity"].unique()) >= {"Mild", "Severe"}:
                        target_bin = _bin
                        break
        else:
            target_bin = "Pre"
        covid_design = TrialDesign(
            participant_col="participant_id",
            visit_col="dfo_bin",
            arm_col="severity",
            arm_treated="Severe",
            arm_control="Mild",
        )
        covid_results = load_or_run_gsea_cross_sectional(
            covid, covid_design, visit=target_bin,
            layer="log1p_cpm", dataset_name="COVID-19",
        )
        if covid_results is not None and len(covid_results) > 0:
            covid_results["dataset"] = "COVID-19"
            if "Library" in covid_results.columns and "library" not in covid_results.columns:
                covid_results = covid_results.rename(columns={"Library": "library"})
            gsea_multi["COVID-19"] = covid_results
            print(f"    COVID-19: {len(covid_results)} pathways")
    except Exception as exc:
        print(f"    COVID-19: FAILED ({exc})")
    
    print(f"  Multi-dataset GSEA: {len(gsea_multi)} datasets completed")
    return gsea_multi


# ======================================================================
# GSEA column detection helper
# ======================================================================

def _detect_gsea_columns(df: pd.DataFrame) -> dict:
    """Detect NES, FDR, Term, Tag, and Lead_genes columns from GSEA output."""
    cols = {"nes": None, "fdr": None, "term": None, "tag": None, "lead": None}

    # Exact match pass
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "nes":
            cols["nes"] = c
        elif cl in ("fdr q-val", "fdr"):
            cols["fdr"] = c
        elif cl == "term":
            cols["term"] = c
        elif cl.startswith("tag"):
            cols["tag"] = c
        elif cl in ("lead_genes", "leading_edge"):
            cols["lead"] = c

    # Fuzzy fallback
    if cols["nes"] is None:
        for c in df.columns:
            if c.lower() in ("nes", "normalized_enrichment_score"):
                cols["nes"] = c
                break
    if cols["fdr"] is None:
        for c in df.columns:
            if "fdr" in c.lower():
                cols["fdr"] = c
                break
    if cols["term"] is None:
        for c in df.columns:
            if c.lower() in ("name", "pathway"):
                cols["term"] = c
                break
        if cols["term"] is None:
            cols["term"] = df.columns[0]
    if cols["lead"] is None:
        for c in df.columns:
            if "lead" in c.lower() and "gene" in c.lower():
                cols["lead"] = c
                break

    return cols


def _clean_pathway_name(s: str, max_len: int = 55) -> str:
    """Clean pathway names for display.

    Strips GO IDs (GO:NNNNNNN), Reactome IDs (R-HSA-NNNNN), and
    trailing parenthetical accession numbers to save label space.
    """
    s = str(s).replace("_", " ").title()
    # Strip accession IDs that bloat labels
    s = re.sub(r"\s*\(Go:\d+\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*R-Hsa-\d+", "", s, flags=re.IGNORECASE)
    s = s.strip()
    return s[:max_len] + "…" if len(s) > max_len + 2 else s


# ======================================================================
# Panel A -- Gene waterfall plot (top genes by effect size)
# ======================================================================

def panel_A(ax, data: dict):
    """Top 30 protein-coding genes ranked by effect size (waterfall).

    Horizontal bar plot of the most extreme genes on each side,
    providing immediate biological interpretability.  Colour indicates
    direction *and* nominal significance (p < 0.05).
    """
    gene_results = data.get("gene_results")

    if gene_results is None or len(gene_results) == 0:
        # Fallback: signature DiD waterfall
        _panel_A_signature_waterfall(ax, data)
        return

    df = gene_results.copy()
    beta_col = "beta_DiD"
    p_col = "p_DiD"

    df = df.dropna(subset=[beta_col, p_col])

    # Filter to protein-coding genes
    df = df[df["feature"].apply(_is_likely_protein_coding)]

    p_thresh = 0.05
    n_per_side = 15

    # Top positive and negative
    top_pos = df.nlargest(n_per_side, beta_col)
    top_neg = df.nsmallest(n_per_side, beta_col)
    selected = pd.concat([top_pos, top_neg]).drop_duplicates()
    selected = selected.sort_values(beta_col, ascending=True).reset_index(
        drop=True
    )

    y_pos = np.arange(len(selected))
    colors = []
    for _, row in selected.iterrows():
        sig = row[p_col] < p_thresh
        if row[beta_col] > 0:
            colors.append(
                COLORS["treated"] if sig else COLORS["treated"] + "55"
            )
        else:
            colors.append(
                COLORS["control"] if sig else COLORS["control"] + "55"
            )

    ax.barh(y_pos, selected[beta_col].values, color=colors, alpha=0.9,
            edgecolor="white", linewidth=0.3, height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected["feature"].values, fontsize=4)

    ax.set_xlabel(r"Effect size ($\beta_{\mathrm{DiD}}$)")
    ax.set_title("Top Genes by Effect Size — Melanoma DiD", fontsize=11,
                 fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.9,
                       label="Responder ↑ (p < 0.05)"),
        mpatches.Patch(color=COLORS["treated"], alpha=0.35,
                       label="Responder ↑ (n.s.)"),
        mpatches.Patch(color=COLORS["control"], alpha=0.9,
                       label="Non-responder ↑ (p < 0.05)"),
        mpatches.Patch(color=COLORS["control"], alpha=0.35,
                       label="Non-responder ↑ (n.s.)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


def _panel_A_signature_waterfall(ax, data: dict):
    """Fallback: signature DiD waterfall plot."""
    did_sig = data["did_sig"]
    df = did_sig.copy()
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("beta_DiD", ascending=False).reset_index(drop=True)

    y_pos = np.arange(len(df))
    colors = [COLORS["treated"] if v > 0 else COLORS["control"]
              for v in df["beta_DiD"]]
    ax.barh(y_pos, df["beta_DiD"].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5, height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel(r"DiD coefficient ($\beta_{\mathrm{DiD}}$)")
    ax.set_title("Signature DiD Effects (Melanoma)", fontsize=11,
                 fontweight="bold")
    ax.invert_yaxis()
    despine(ax)


# ======================================================================
# Panel B -- GSEA Enrichment Bar Chart
# ======================================================================

def panel_B(ax, data: dict):
    """GSEA immune + metabolic pathway enrichment bar chart with balanced up/down."""
    gsea_results = data["gsea_results"]

    if gsea_results is None or len(gsea_results) == 0:
        _panel_B_signature_waterfall(ax, data)
        return

    df = gsea_results.copy()
    cols = _detect_gsea_columns(df)
    nes_col, fdr_col, term_col = cols["nes"], cols["fdr"], cols["term"]

    if nes_col is None:
        _panel_B_signature_waterfall(ax, data)
        return

    # Convert to numeric
    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Immune/metabolic filtering already done in _prepare_data.

    # Balanced selection: top N up + top N down by |NES|
    n_show = 15
    df_pos = df[df[nes_col] > 0].nlargest(n_show // 2 + 1, nes_col)
    df_neg = df[df[nes_col] < 0].nsmallest(n_show - len(df_pos), nes_col)
    # If one direction is sparse, fill from the other
    if len(df_pos) + len(df_neg) < n_show:
        remainder = n_show - len(df_pos) - len(df_neg)
        already = set(df_pos.index) | set(df_neg.index)
        extra = (
            df[~df.index.isin(already)]
            .assign(_abs=df[nes_col].abs())
            .nlargest(remainder, "_abs")
        )
        df_selected = pd.concat([df_pos, df_neg, extra.drop(columns="_abs")])
    else:
        df_selected = pd.concat([df_pos, df_neg])
    df_selected = df_selected.drop_duplicates().sort_values(nes_col, ascending=True)

    # Fix #4: Clean pathway names AND disambiguate duplicates
    df_selected["pathway"] = df_selected[term_col].apply(_clean_pathway_name)
    # Disambiguate duplicate display names
    _seen = {}
    new_labels = []
    for idx, row in df_selected.iterrows():
        label = row["pathway"]
        if label in _seen:
            _seen[label] += 1
            lib = str(row.get("library", ""))
            if lib and lib != "averaged":
                label = f"{label} [{lib[:8]}]"
            else:
                label = f"{label} ({_seen[label]})"
        else:
            _seen[label] = 1
        new_labels.append(label)
    df_selected["pathway"] = new_labels

    # Color by direction and significance — use project palette
    # treated (blue) = Responder ↑, control (orange) = Non-responder ↑
    clr_up_sig = COLORS["treated"]
    clr_up_ns = COLORS["treated"] + "66"  # 40% alpha hex
    clr_dn_sig = COLORS["control"]
    clr_dn_ns = COLORS["control"] + "66"
    colors = []
    for _, row in df_selected.iterrows():
        sig = (
            fdr_col is not None
            and pd.notna(row.get(fdr_col))
            and row[fdr_col] < 0.25
        )
        if row[nes_col] > 0:
            colors.append(clr_up_sig if sig else clr_up_ns)
        else:
            colors.append(clr_dn_sig if sig else clr_dn_ns)

    y_pos = np.arange(len(df_selected))
    ax.barh(y_pos, df_selected[nes_col].values, color=colors, alpha=0.9,
            edgecolor="white", linewidth=0.5, height=0.7)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_selected["pathway"].values, fontsize=8)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title("Pathway Enrichment", fontsize=11, fontweight="bold")

    # Build legend only for categories present
    def _is_sig(row):
        return (fdr_col and pd.notna(row.get(fdr_col))
                and row[fdr_col] < 0.25)

    has_up_sig = any(
        row[nes_col] > 0 and _is_sig(row)
        for _, row in df_selected.iterrows()
    )
    has_up_ns = any(
        row[nes_col] > 0 and not _is_sig(row)
        for _, row in df_selected.iterrows()
    )
    has_down_sig = any(
        row[nes_col] < 0 and _is_sig(row)
        for _, row in df_selected.iterrows()
    )
    has_down_ns = any(
        row[nes_col] < 0 and not _is_sig(row)
        for _, row in df_selected.iterrows()
    )
    legend_handles = []
    if has_up_sig:
        legend_handles.append(mpatches.Patch(
            color=COLORS["treated"], alpha=0.9,
            label="Responder ↑ (FDR < 0.25)",
        ))
    if has_up_ns:
        legend_handles.append(mpatches.Patch(
            color=COLORS["treated"], alpha=0.4,
            label="Responder ↑ (n.s.)",
        ))
    if has_down_sig:
        legend_handles.append(mpatches.Patch(
            color=COLORS["control"], alpha=0.9,
            label="Non-responder ↑ (FDR < 0.25)",
        ))
    if has_down_ns:
        legend_handles.append(mpatches.Patch(
            color=COLORS["control"], alpha=0.4,
            label="Non-responder ↑ (n.s.)",
        ))
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=9, loc="lower right",
                  frameon=True, framealpha=0.9)
    despine(ax)


def _panel_B_signature_waterfall(ax, data: dict):
    """Fallback: signature-level DiD effects bar chart."""
    did_sig = data["did_sig"]
    df = did_sig.copy()
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("beta_DiD", ascending=True)

    colors = [COLORS["treated"] if v > 0 else COLORS["control"]
              for v in df["beta_DiD"]]
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["beta_DiD"].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel(r"DiD coefficient ($\beta_{\mathrm{DiD}}$)")
    ax.set_title("DiD Signature Effects", fontsize=11, fontweight="bold")
    despine(ax)


# ======================================================================
# Panel C -- Leading-edge gene overlap heatmap
# ======================================================================

def panel_C(ax, data: dict, *, composite: bool = False):
    """Leading-edge gene overlap heatmap across top enriched pathways.

    Information-dense design:
    - Tight imshow grid coloured by NES direction
    - Pathway labels coloured by NES (blue=Responder↑, orange=Non-responder↑)
    - Top marginal bar showing gene recurrence count
    - Hierarchical column clustering for gene co-occurrence
    - Capped to 8 pathways × 20 genes for readability

    When *composite* is True, the marginal bar and tight_layout are
    skipped so the panel can be embedded in a composite GridSpec figure.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    gsea_results = data["gsea_results"]

    if gsea_results is None or len(gsea_results) == 0:
        _panel_C_did_summary(ax, data)
        return

    df = gsea_results.copy()
    cols = _detect_gsea_columns(df)
    nes_col, fdr_col, term_col, lead_col = (
        cols["nes"], cols["fdr"], cols["term"], cols["lead"]
    )

    if nes_col is None or lead_col is None or lead_col not in df.columns:
        _panel_C_did_summary(ax, data)
        return

    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Select top pathways balanced across NES directions — match count in
    # panel A (GSEA bar chart) so every bar in A has a row in this heatmap.
    MAX_PW = 15
    work_df = df.assign(_abs=df[nes_col].abs())
    pos_df = work_df[work_df[nes_col] > 0].nlargest(MAX_PW, "_abs")
    neg_df = work_df[work_df[nes_col] < 0].nlargest(MAX_PW, "_abs")

    n_pos = min(len(pos_df), MAX_PW // 2)
    n_neg = min(len(neg_df), MAX_PW // 2)
    # Fill remaining slots from whichever side has more
    remaining = MAX_PW - n_pos - n_neg
    if remaining > 0:
        if len(pos_df) > n_pos:
            extra_pos = min(remaining, len(pos_df) - n_pos)
            n_pos += extra_pos
            remaining -= extra_pos
        if remaining > 0 and len(neg_df) > n_neg:
            n_neg += min(remaining, len(neg_df) - n_neg)

    selected = pd.concat([
        pos_df.head(n_pos),
        neg_df.head(n_neg),
    ]).drop(columns="_abs", errors="ignore")
    selected = selected.sort_values(nes_col, ascending=True)

    # Parse leading-edge genes
    pathway_genes: dict[str, set[str]] = {}
    pathway_nes: dict[str, float] = {}
    pathway_fdr: dict[str, float] = {}
    all_genes: set[str] = set()
    _seen_names: set[str] = set()
    for _, row in selected.iterrows():
        pname = _clean_pathway_name(str(row[term_col]), max_len=32)
        if pname in _seen_names:
            lib = str(row.get("library", ""))
            pname = f"{pname} [{lib[:8]}]" if lib else f"{pname} (2)"
        _seen_names.add(pname)
        genes_str = str(row[lead_col])
        genes = [g.strip() for g in genes_str.replace(";", ",").split(",")
                 if g.strip()]
        genes = [g for g in genes if _is_likely_protein_coding(g)]
        pathway_genes[pname] = set(genes)
        pathway_nes[pname] = float(row[nes_col])
        if fdr_col is not None and pd.notna(row.get(fdr_col)):
            pathway_fdr[pname] = float(row[fdr_col])
        all_genes.update(genes)

    if not all_genes or not pathway_genes:
        _panel_C_did_summary(ax, data)
        return

    # Select informative genes — guarantee BOTH NES directions are
    # represented by selecting top genes PER direction then merging.
    # This avoids the problem where one direction's highly-overlapping
    # gene sets dominate a global top-N selection.
    pathways = list(pathway_genes.keys())
    pos_pathways = [p for p in pathways if pathway_nes.get(p, 0) > 0]
    neg_pathways = [p for p in pathways if pathway_nes.get(p, 0) <= 0]

    def _count_genes_in_group(pw_list):
        """Count gene occurrences within a group of pathways."""
        counts: dict[str, int] = {}
        for pw in pw_list:
            for g in pathway_genes.get(pw, set()):
                counts[g] = counts.get(g, 0) + 1
        return counts

    TOTAL_GENES = 20
    half = TOTAL_GENES // 2
    pos_counts = _count_genes_in_group(pos_pathways)
    neg_counts = _count_genes_in_group(neg_pathways)

    # Take top genes from each direction
    pos_genes = sorted(pos_counts.keys(),
                       key=lambda g: -pos_counts[g])[:half]
    neg_genes = sorted(neg_counts.keys(),
                       key=lambda g: -neg_counts[g])[:half]

    # Merge, removing duplicates (keep order)
    seen: set[str] = set()
    shared_genes: list[str] = []
    for g in pos_genes + neg_genes:
        if g not in seen:
            shared_genes.append(g)
            seen.add(g)

    # If one direction had fewer than half genes, fill from the other
    if len(shared_genes) < TOTAL_GENES:
        all_counts: dict[str, int] = {}
        for pw in pathways:
            for g in pathway_genes.get(pw, set()):
                all_counts[g] = all_counts.get(g, 0) + 1
        for g in sorted(all_counts.keys(), key=lambda g: -all_counts[g]):
            if g not in seen:
                shared_genes.append(g)
                seen.add(g)
            if len(shared_genes) >= TOTAL_GENES:
                break

    print(f"  Panel C: {len(pos_genes)} genes from NES>0 pathways, "
          f"{len(neg_genes)} genes from NES≤0 pathways")

    # Build binary matrix and prune zero rows/cols
    matrix = np.zeros((len(pathways), len(shared_genes)), dtype=int)
    for i, pw in enumerate(pathways):
        for j, g in enumerate(shared_genes):
            if g in pathway_genes[pw]:
                matrix[i, j] = 1
    # Prune zero rows (pathways with no genes in selection)
    row_ok = matrix.sum(axis=1) > 0
    matrix = matrix[row_ok]
    pathways = [p for p, k in zip(pathways, row_ok) if k]
    # Prune zero cols
    col_ok = matrix.sum(axis=0) > 0
    matrix = matrix[:, col_ok]
    shared_genes = [g for g, k in zip(shared_genes, col_ok) if k]

    n_pw_kept = sum(1 for p in pathways if pathway_nes.get(p, 0) > 0)
    n_neg_kept = sum(1 for p in pathways if pathway_nes.get(p, 0) <= 0)
    print(f"  Panel C: {n_pw_kept} NES>0 + {n_neg_kept} NES≤0 pathways "
          f"retained, {len(shared_genes)} genes")

    if matrix.size == 0 or not shared_genes:
        _panel_C_did_summary(ax, data)
        return

    n_pw, n_genes = matrix.shape

    # ── Hierarchical clustering of gene columns ──
    if n_genes >= 3:
        try:
            dist = pdist(matrix.T, metric="jaccard")
            dist = np.nan_to_num(dist, nan=1.0)
            Z = linkage(dist, method="average")
            gene_order = leaves_list(Z)
        except Exception:
            gene_order = np.arange(n_genes)
    else:
        gene_order = np.arange(n_genes)

    matrix = matrix[:, gene_order]
    shared_genes = [shared_genes[i] for i in gene_order]

    # Recompute column counts after clustering
    col_counts = matrix.sum(axis=0)

    # ── Sort pathways: NES>0 block on top, NES<0 on bottom ──
    pos_pws = [p for p in pathways if pathway_nes.get(p, 0) > 0]
    neg_pws = [p for p in pathways if pathway_nes.get(p, 0) <= 0]
    # Sort within each block by |NES|
    pos_pws.sort(key=lambda p: pathway_nes.get(p, 0))
    neg_pws.sort(key=lambda p: pathway_nes.get(p, 0))
    pathways_sorted = neg_pws + pos_pws
    row_idx = [pathways.index(p) for p in pathways_sorted]
    matrix = matrix[row_idx]
    pathways = pathways_sorted
    n_sep = len(neg_pws)  # separator position between blocks

    # Recompute column counts after reorder
    col_counts = matrix.sum(axis=0)

    # ── Colour constants ──
    BLUE = (0.122, 0.471, 0.706)   # steel blue (Responder ↑ / NES>0)
    ORANGE = (0.878, 0.478, 0.184)  # warm orange (Non-responder ↑ / NES<0)
    EMPTY_COLOR = (0.94, 0.94, 0.94)    # light gray for "not in leading edge"  # noqa: N806

    # ── Colour matrix: in leading edge (filled) vs not (empty) ──
    rgb = np.full((n_pw, n_genes, 3), 0.94)  # light gray for empty
    for i, pw in enumerate(pathways):
        nes_val = pathway_nes.get(pw, 0)
        fill = np.array(BLUE if nes_val > 0 else ORANGE)
        for j in range(n_genes):
            if matrix[i, j] == 1:
                rgb[i, j] = fill

    # ── Transpose: genes on Y-axis, pathways on X-axis ──
    rgb = np.transpose(rgb, (1, 0, 2))  # (n_genes, n_pw, 3)

    ax.imshow(rgb, aspect="auto", interpolation="nearest", origin="lower")

    # Thin white grid lines
    for i in range(n_genes + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.8, zorder=2)
    for j in range(n_pw + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.8, zorder=2)

    # Separator line between NES<0 and NES>0 blocks (now vertical)
    if n_sep > 0 and n_sep < n_pw:
        ax.axvline(n_sep - 0.5, color="black", linewidth=1.5, zorder=3)

    # X-axis: pathway labels, coloured by NES direction
    ax.set_xticks(range(n_pw))
    ax.set_xticklabels(pathways, rotation=35, ha="right", fontsize=5)
    for i, (pw, label) in enumerate(zip(pathways, ax.get_xticklabels())):
        label.set_color(BLUE if pathway_nes.get(pw, 0) > 0 else ORANGE)
        label.set_fontweight("bold")

    # Y-axis: gene labels
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(shared_genes, fontsize=6, style="italic")
    ax.tick_params(axis="both", length=0)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Gene recurrence: how many pathways each gene appears in
    # col_counts was computed from original (n_pw × n_genes) matrix
    gene_counts = col_counts

    if not composite:
        # ── Right marginal bar: gene recurrence count ──
        fig = ax.get_figure()
        fig.tight_layout(rect=[0, 0, 0.90, 1])
        ax_pos = ax.get_position()
        bar_width = 0.04
        bar_ax = fig.add_axes([
            ax_pos.x1 + 0.02, ax_pos.y0,
            bar_width, ax_pos.height,
        ])
        bar_colors = ["#555555"] * n_genes
        bar_ax.barh(range(n_genes), gene_counts, height=0.7,
                    color=bar_colors, edgecolor="none")
        bar_ax.set_ylim(-0.5, n_genes - 0.5)
        bar_ax.set_xlim(0, max(gene_counts) + 0.5)
        bar_ax.set_yticks([])
        bar_ax.set_xlabel("# paths", fontsize=5.5, labelpad=5)
        bar_ax.tick_params(axis="x", labelsize=5.5, length=2)
        bar_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=3))
        for spine in ["top", "right", "left"]:
            bar_ax.spines[spine].set_visible(False)
        bar_ax.spines["bottom"].set_linewidth(0.5)
        ax.set_title("Leading-Edge Gene Overlap", fontsize=11,
                     fontweight="bold", pad=8)
    else:
        ax.set_title("Leading-Edge Gene Overlap", fontsize=11,
                     fontweight="bold")

    # Legend — inside the heatmap lower-right (gray empty region)
    legend_handles = [
        mpatches.Patch(facecolor=BLUE, label="Resp. ↑"),
        mpatches.Patch(facecolor=ORANGE, label="Non-resp. ↑"),
        mpatches.Patch(facecolor=EMPTY_COLOR, edgecolor="#CCCCCC",
                       label="Not in leading edge"),
    ]
    ax.legend(
        handles=legend_handles, fontsize=7, loc="lower right",
        frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
        handlelength=1.0, handleheight=0.7,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def _panel_C_did_summary(ax, data: dict):
    """Fallback: signature-level DiD effects bar chart."""
    did_sig = data["did_sig"]
    df = did_sig.copy()
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("beta_DiD", ascending=True)

    colors = [COLORS["treated"] if v > 0 else COLORS["control"]
              for v in df["beta_DiD"]]
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["beta_DiD"].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel(r"DiD coefficient ($\beta_{\mathrm{DiD}}$)")
    ax.set_title("DiD Signature Effects", fontsize=11, fontweight="bold")
    despine(ax)


# ======================================================================
# Panel D -- Signature DiD forest plot with bootstrap CIs
# ======================================================================

def panel_D(ax, data: dict):
    """DiD effect size forest plot with bootstrap CIs.

    Coloring matches Figure 2 forest plot: blue = Responder ↑ (β > 0),
    orange = Non-responder ↑ (β < 0). All signatures are colored by
    direction regardless of significance.
    """
    did_sig = data["did_sig"]

    df = did_sig.copy()
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("beta_DiD", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))

    # Use bootstrap CIs if available; fall back to Wald
    has_boot_ci = "ci_lo_boot" in df.columns and "ci_hi_boot" in df.columns
    if has_boot_ci:
        ci_lo = df["ci_lo_boot"]
        ci_hi = df["ci_hi_boot"]
        ci_label = "95% Bootstrap CI"
    else:
        ci_lo = df["beta_DiD"] - 1.96 * df["se_DiD"]
        ci_hi = df["beta_DiD"] + 1.96 * df["se_DiD"]
        ci_label = "95% Wald CI"

    fdr_vals = df["FDR_DiD"].values

    sig_mask = pd.Series(fdr_vals < 0.1, index=df.index)

    # Color ALL signatures by direction (matching Figure 2 style)
    for i in df.index:
        clr = COLORS["treated"] if df.loc[i, "beta_DiD"] > 0 else COLORS["control"]
        lw = 2.0 if sig_mask.iloc[i] else 1.5
        ax.hlines(y[i], ci_lo.iloc[i], ci_hi.iloc[i],
                  color=clr, lw=lw, zorder=1)
        ax.scatter(df.loc[i, "beta_DiD"], y[i],
                   color=clr, s=50, zorder=2,
                   edgecolors="white", linewidths=0.5)

    ax.axvline(0, color="black", ls=":", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display"].values, fontsize=9)
    ax.set_xlabel(r"DiD coefficient ($\beta$, standardised)")
    ax.set_title(f"Signature DiD Effects ({ci_label})", fontsize=11,
                 fontweight="bold")

    # Legend matching Figure 2 style
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=COLORS["treated"], lw=1.5,
                   markersize=6, label="Responder ↑"),
        plt.Line2D([0], [0], marker="o", color=COLORS["control"], lw=1.5,
                   markersize=6, label="Non-responder ↑"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="lower right",
              frameon=True, framealpha=0.9)

    n_sig = sig_mask.sum()
    ax.text(0.97, 0.03, f"{n_sig}/{len(df)} FDR < 0.1",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic", color=COLORS["gray"])
    despine(ax)


# ======================================================================
# Panel E -- Gene-level volcano plot
# ======================================================================

def panel_E(ax, data: dict, *, composite: bool = False):
    """Volcano plot of gene-level DiD effects (Sade-Feldman).

    Labels prioritize protein-coding genes over pseudogenes/lncRNAs.
    When *composite* is True, fewer labels are drawn and adjustText
    is skipped to avoid cluttering the small composite axes.
    """
    gene_results = data["gene_results"]

    if gene_results is None or len(gene_results) == 0:
        ax.text(
            0.5, 0.5,
            "Gene-level results unavailable",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor=COLORS["gray"]),
        )
        ax.set_title("Gene-Level Volcano Plot", fontsize=11,
                     fontweight="bold")
        ax.axis("off")
        return

    df = gene_results.copy()
    beta_col = "beta_DiD"

    p_col = "p_DiD"

    df = df.dropna(subset=[beta_col, p_col])

    # Standard volcano: nominal p on y-axis, colour by nominal p < 0.05.
    # Analytical (nonrobust) SEs provide gene-level resolution; bootstrap
    # inference is reserved for signature-level tests (Panel D).
    p_thresh = 0.05
    df["nlog10"] = -np.log10(df[p_col].clip(lower=1e-300))
    sig_mask = df[p_col] < p_thresh

    df["category"] = "ns"
    df.loc[sig_mask & (df[beta_col] > 0), "category"] = "up"
    df.loc[sig_mask & (df[beta_col] < 0), "category"] = "down"

    color_map = {
        "ns": COLORS["gray"],
        "up": COLORS["treated"],
        "down": COLORS["control"],
    }
    alpha_map = {"ns": 0.3, "up": 0.8, "down": 0.8}
    if composite:
        size_map = {"ns": 2, "up": 6, "down": 6}
    else:
        size_map = {"ns": 8, "up": 20, "down": 20}

    for cat in ["ns", "up", "down"]:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[beta_col], sub["nlog10"],
            c=color_map[cat], alpha=alpha_map[cat],
            s=size_map[cat], edgecolors="none", rasterized=True,
        )

    N_LABELS = 10
    labelled_genes: list[str] = []  # ordered by score (highest first)

    for sign in ("pos", "neg"):
        sub = df[df[beta_col] > 0].copy() if sign == "pos" else df[df[beta_col] < 0].copy()
        if len(sub) == 0:
            continue
        # Restrict to protein-coding genes only
        sub = sub[sub["feature"].apply(_is_likely_protein_coding)]
        if len(sub) == 0:
            continue

        # Force-include top 3 genes by -log10(p) in each direction
        # (ensures the most significant genes are always labelled)
        force_genes = set(
            sub.nlargest(min(3, len(sub)), "nlog10")["feature"].tolist()
        )

        # Combined score: rank-normalised |β| + rank-normalised -log10(p)
        # This naturally selects genes at volcano tips (high on both axes).
        sub = sub.copy()
        sub["_rank_beta"] = sub[beta_col].abs().rank(pct=True)
        sub["_rank_sig"] = sub["nlog10"].rank(pct=True)
        sub["_score"] = sub["_rank_beta"] + sub["_rank_sig"]

        candidates = sub.nlargest(min(N_LABELS * 3, len(sub)), "_score")

        # Deduplicate: skip genes too close to an already-selected one
        # (prevents overlapping arrows pointing to the same spot).
        x_range = df[beta_col].max() - df[beta_col].min()
        y_range = df["nlog10"].max() - df["nlog10"].min()
        min_dx = x_range * 0.025  # ~2.5% of axis range
        min_dy = y_range * 0.025
        selected_coords: list[tuple[float, float]] = []
        picks: list[str] = []

        # Add forced genes first
        for _, cand in candidates.iterrows():
            if cand["feature"] in force_genes and cand["feature"] not in picks:
                picks.append(cand["feature"])
                selected_coords.append((cand[beta_col], cand["nlog10"]))

        # Fill remaining slots with score-ranked candidates
        for _, cand in candidates.iterrows():
            if cand["feature"] in picks:
                continue
            cx, cy = cand[beta_col], cand["nlog10"]
            too_close = False
            for sx, sy in selected_coords:
                if abs(cx - sx) < min_dx and abs(cy - sy) < min_dy:
                    too_close = True
                    break
            if not too_close:
                picks.append(cand["feature"])
                selected_coords.append((cx, cy))
                if len(picks) >= N_LABELS:
                    break

        labelled_genes.extend(picks)

    labelled_set = set(labelled_genes)
    labelled_rows = df[df["feature"].isin(labelled_set)].copy()

    from adjustText import adjust_text as _adjust_text

    _lbl_fs = 3.5 if composite else 6.5
    _arrow_lw = 0.25 if composite else 0.4

    texts = []
    for _, row in labelled_rows.iterrows():
        t = ax.text(
            row[beta_col], row["nlog10"], row["feature"],
            fontsize=_lbl_fs, fontweight="bold", color="#444444",
            ha="center", va="center", zorder=5,
        )
        texts.append(t)

    x_span = df[beta_col].max() - df[beta_col].min()
    y_span = df["nlog10"].max() - df["nlog10"].min()
    _adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#888888", lw=_arrow_lw,
                        shrinkA=5, shrinkB=3),
        force_text=(2.0, 2.0),
        force_points=(3.5, 3.5),
        expand=(1.8, 2.0),
        ensure_inside_axes=True,
        max_move=(x_span * 0.25, y_span * 0.25),
        only_move="xy",
    )

    # Threshold line
    thresh_y = -np.log10(p_thresh)
    ax.axhline(thresh_y, color=COLORS["gray"], ls="--", lw=0.8, zorder=0)
    ax.axvline(0, color="black", lw=0.6, zorder=0)

    ax.set_xlabel(r"Effect size ($\beta_{\mathrm{DiD}}$)")
    ax.set_ylabel(r"$-\log_{10}$(p)")
    ax.set_title("Gene-Level Volcano (Melanoma DiD)", fontsize=11,
                 fontweight="bold")

    # Legend — no footnotes, no summary boxes
    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.8,
                       label="Responder ↑"),
        mpatches.Patch(color=COLORS["control"], alpha=0.8,
                       label="Non-responder ↑"),
        mpatches.Patch(color=COLORS["gray"], alpha=0.3,
                       label="Not significant"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="lower left",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel C (new) -- Replicated pathways across cohorts
# ======================================================================

def panel_C_replicated(ax, data: dict):
    """Replicated pathways: heatmap showing top pathways enriched across datasets.
    
    Creates a heatmap with pathways as rows and datasets as columns.
    Shows the top 15 pathways with a balanced mix of high positive and high negative
    enrichment, prioritizing pathways that are highly enriched across datasets.
    Pathways must appear in at least 3 out of 5 datasets.
    
    Color intensity represents NES magnitude, with blue for positive NES
    (responder-enriched) and orange for negative NES (non-responder-enriched).
    Only shows pathways that are significant (FDR < 0.25) in at least one dataset.
    """
    gsea_multi = data.get("gsea_multi_dataset", {})
    
    if not gsea_multi:
        ax.text(0.5, 0.5, "Multi-dataset GSEA results unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.set_title("Replicated Pathways", fontsize=11, fontweight="bold")
        ax.axis("off")
        return

    # Combine all datasets into one DataFrame
    all_results = []
    for ds_name, df in gsea_multi.items():
        if df is None or len(df) == 0:
            continue
        cols = _detect_gsea_columns(df)
        nes_col_name, fdr_col_name, term_col_name = cols["nes"], cols["fdr"], cols["term"]
        
        if nes_col_name is None or fdr_col_name is None or term_col_name is None:
            continue
        
        df = df.copy()
        df[nes_col_name] = pd.to_numeric(df[nes_col_name], errors="coerce")
        df[fdr_col_name] = pd.to_numeric(df[fdr_col_name], errors="coerce")
        # Keep all pathways with valid NES - FDR will be used for significance annotation (stars)
        df = df.dropna(subset=[nes_col_name])
        
        if len(df) == 0:
            continue
        
        df["dataset"] = ds_name
        df["pathway"] = df[term_col_name].apply(
            lambda s: _clean_pathway_name(s, max_len=70)
        )
        # Rename columns to standard names for concatenation
        df_clean = df[[nes_col_name, fdr_col_name, "pathway", "dataset"]].copy()
        df_clean.columns = ["NES", "FDR", "pathway", "dataset"]
        all_results.append(df_clean)
    
    if not all_results:
        ax.text(0.5, 0.5, "No pathways found across datasets",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.set_title("Replicated Pathways", fontsize=11, fontweight="bold")
        ax.axis("off")
        return

    combined = pd.concat(all_results, ignore_index=True)
    
    # Get all unique pathways and datasets
    all_pathways = sorted(combined["pathway"].unique())
    all_datasets = sorted(combined["dataset"].unique())
    
    # Build NES matrix: pathways × datasets
    nes_matrix = np.full((len(all_pathways), len(all_datasets)), np.nan)
    fdr_matrix = np.full((len(all_pathways), len(all_datasets)), np.nan)
    
    for i, pathway in enumerate(all_pathways):
        for j, dataset in enumerate(all_datasets):
            subset = combined[(combined["pathway"] == pathway) & 
                             (combined["dataset"] == dataset)]
            if len(subset) > 0:
                # If multiple entries, take the one with highest |NES|
                row = subset.loc[subset["NES"].abs().idxmax()]
                nes_matrix[i, j] = row["NES"]
                fdr_matrix[i, j] = row["FDR"]
    
    # Filter pathways: prioritize ≥5 datasets, fallback to ≥4, then ≥3
    # Rank by absolute average NES across datasets
    pathway_counts = (~np.isnan(nes_matrix)).sum(axis=1)
    n_datasets = len(all_datasets)
    top_n = 15
    
    # Calculate average NES for all pathways
    avg_nes_all = np.nanmean(nes_matrix, axis=1)
    abs_avg_nes_all = np.abs(avg_nes_all)
    
    # First try: pathways in ≥5 datasets, ranked by absolute NES
    keep_5plus = pathway_counts >= min(5, n_datasets)
    
    if keep_5plus.sum() > 0:
        indices_5plus = np.where(keep_5plus)[0]
        abs_nes_5plus = abs_avg_nes_all[indices_5plus]
        order_5plus = indices_5plus[np.argsort(-abs_nes_5plus)]
        selected_5plus = order_5plus[:top_n].tolist()
    else:
        selected_5plus = []
    
    # Initialize all_selected with pathways from ≥5 datasets
    all_selected = selected_5plus
    
    # If we need more pathways, get from ≥4 datasets
    if len(all_selected) < top_n:
        keep_4plus = pathway_counts >= min(4, n_datasets)
        keep_4plus_only = keep_4plus & (~keep_5plus)
        
        if keep_4plus_only.sum() > 0:
            indices_4plus = np.where(keep_4plus_only)[0]
            abs_nes_4plus = abs_avg_nes_all[indices_4plus]
            order_4plus = indices_4plus[np.argsort(-abs_nes_4plus)]
            
            remaining = top_n - len(all_selected)
            selected_4plus = order_4plus[:remaining].tolist()
            
            all_selected = all_selected + selected_4plus
    
    # If we still need more pathways, get from ≥3 datasets
    if len(all_selected) < top_n:
        keep_3plus = pathway_counts >= min(3, n_datasets)
        keep_3plus_only = keep_3plus & (~keep_5plus) & (~keep_4plus)
        
        if keep_3plus_only.sum() > 0:
            indices_3plus = np.where(keep_3plus_only)[0]
            abs_nes_3plus = abs_avg_nes_all[indices_3plus]
            order_3plus = indices_3plus[np.argsort(-abs_nes_3plus)]
            
            remaining = top_n - len(all_selected)
            selected_3plus = order_3plus[:remaining].tolist()
            
            all_selected = all_selected + selected_3plus
    
    if len(all_selected) == 0:
        ax.text(0.5, 0.5, "No pathways replicated across ≥3 datasets",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.set_title("Replicated Pathways", fontsize=11, fontweight="bold")
        ax.axis("off")
        return
    
    # Extract selected pathways
    nes_matrix = nes_matrix[all_selected]
    fdr_matrix = fdr_matrix[all_selected]
    all_pathways = [all_pathways[i] for i in all_selected]
    
    # Calculate average NES for final sorting
    avg_nes = np.nanmean(nes_matrix, axis=1)
    final_order = np.argsort(avg_nes)
    
    nes_matrix = nes_matrix[final_order]
    fdr_matrix = fdr_matrix[final_order]
    all_pathways = [all_pathways[i] for i in final_order]
    
    # Custom colormap: blue (negative) to white/gray (zero) to red (positive)
    import matplotlib.colors as mcolors
    colors_neg = [(0.122, 0.471, 0.706), (0.95, 0.95, 0.95)]  # blue to light gray
    colors_pos = [(0.95, 0.95, 0.95), (0.839, 0.188, 0.192)]  # light gray to red
    n_bins = 100
    cmap_neg = mcolors.LinearSegmentedColormap.from_list('neg', colors_neg, N=n_bins//2)
    cmap_pos = mcolors.LinearSegmentedColormap.from_list('pos', colors_pos, N=n_bins//2)
    # Combine colormaps
    colors = np.vstack((cmap_neg(np.linspace(0, 1, n_bins//2)),
                       cmap_pos(np.linspace(0, 1, n_bins//2))))
    cmap = mcolors.LinearSegmentedColormap.from_list('diverging', colors, N=n_bins)
    
    # Set colorbar range to actual data min/max
    valid_nes = nes_matrix[~np.isnan(nes_matrix)]
    if len(valid_nes) > 0:
        vmin, vmax = float(np.nanmin(valid_nes)), float(np.nanmax(valid_nes))
        vrange = vmax - vmin
        vmin -= vrange * 0.05
        vmax += vrange * 0.05
    else:
        vmin, vmax = -2.0, 2.0
    
    # For NaN values set to white
    masked_nes = np.ma.masked_invalid(nes_matrix)
    
    # Plot heatmap 
    im = ax.imshow(masked_nes, aspect="auto", interpolation="nearest", origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_facecolor((0.95, 0.95, 0.95))
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NES", fontsize=9, rotation=0, labelpad=10)
    cbar.ax.tick_params(labelsize=8)
    
    for i in range(len(all_pathways) + 1):
        ax.axhline(i - 0.5, color="#E0E0E0", linewidth=0.8, zorder=1)
    for j in range(len(all_datasets) + 1):
        ax.axvline(j - 0.5, color="#E0E0E0", linewidth=0.8, zorder=1)
    
    # Add stars for significant pathways (FDR < 0.25)
    for i in range(len(all_pathways)):
        for j in range(len(all_datasets)):
            if not np.isnan(fdr_matrix[i, j]) and fdr_matrix[i, j] < 0.25:
                # Add star in the center of the cell
                ax.text(j, i, "*", ha="center", va="center", 
                       fontsize=10, fontweight="bold", color="white",
                       zorder=10)
    
    # Labels
    ax.set_yticks(range(len(all_pathways)))
    ax.set_yticklabels(all_pathways, fontsize=7.5)
    ax.set_xticks(range(len(all_datasets)))
    ax.set_xticklabels(all_datasets, fontsize=8, rotation=25, ha="right")
    
    ax.set_xlabel("Dataset", fontsize=9)
    ax.set_ylabel("Pathway", fontsize=9)
    ax.set_title("Replicated Pathways Across Datasets (* FDR < 0.25)",
                 fontsize=11, fontweight="bold")
    
    ax.tick_params(axis="both", length=0)
    despine(ax)


# ======================================================================
# Panel F -- Cell-type-resolved effect heatmap for top DiD genes
# ======================================================================

def panel_F(ax, data: dict):
    """Cell-type-resolved effect heatmap for top DiD genes.

    Rows = top genes by |β_DiD|, columns = cell types.
    Color = mean DiD-like effect per cell type (responder post-pre
    minus non-responder post-pre, using raw cell-level means).
    """
    gene_results = data.get("gene_results")
    adata = data.get("adata")

    if gene_results is None or adata is None or len(gene_results) == 0:
        ax.text(0.5, 0.5, "Gene-level results unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.set_title("Cell-Type DiD Effects", fontsize=11,
                     fontweight="bold")
        ax.axis("off")
        return

    df = gene_results.copy()
    beta_col = "beta_DiD"
    df = df.dropna(subset=[beta_col])

    # Restrict to protein-coding genes (skip RNU*, RNA5SP*, lncRNAs, etc.)
    df = df[df["feature"].apply(_is_likely_protein_coding)]

    # Select top 15 genes by |β_DiD| (balanced: top 8 pos + top 7 neg)
    n_per_dir = 8
    df_pos = df[df[beta_col] > 0].nlargest(n_per_dir, beta_col)
    df_neg = df[df[beta_col] < 0].nsmallest(n_per_dir - 1, beta_col)
    top_genes_df = pd.concat([df_pos, df_neg])
    top_genes = top_genes_df["feature"].tolist()

    # Restrict to genes present in adata
    available = [g for g in top_genes if g in adata.var_names]
    if len(available) == 0:
        ax.text(0.5, 0.5, "No top genes in adata",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    # Compute cell-type × gene DiD-like effects from raw cell-level data
    ct_col = "cell_type"
    if ct_col not in adata.obs.columns:
        ct_col = next(
            (c for c in adata.obs.columns if "cell" in c.lower()
             and "type" in c.lower()),
            None,
        )
    if ct_col is None:
        ax.text(0.5, 0.5, "No cell type column",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    layer_key = "log1p_tpm"
    cell_types = sorted(adata.obs[ct_col].dropna().unique())
    # Drop very rare cell types (<20 cells total)
    ct_counts = adata.obs[ct_col].value_counts()
    cell_types = [ct for ct in cell_types if ct_counts.get(ct, 0) >= 20]

    effect_mat = pd.DataFrame(
        np.nan, index=available, columns=cell_types
    )

    sub_adata = adata[:, available].copy()
    if layer_key in sub_adata.layers:
        X = sub_adata.layers[layer_key]
    else:
        X = sub_adata.X

    obs = sub_adata.obs.copy()
    obs["_visit"] = obs["visit"]
    obs["_arm"] = obs["response_harmonized"]
    obs["_ct"] = obs[ct_col]

    import scipy.sparse as sp
    if sp.issparse(X):
        X = X.toarray()
    expr_df = pd.DataFrame(X, index=obs.index, columns=available)
    expr_df["_visit"] = obs["_visit"].values
    expr_df["_arm"] = obs["_arm"].values
    expr_df["_ct"] = obs["_ct"].values

    for ct in cell_types:
        ct_mask = expr_df["_ct"] == ct
        ct_data = expr_df[ct_mask]
        for arm_label, visit_label in [
            ("Responder", "Pre"), ("Responder", "Post"),
            ("Non-responder", "Pre"), ("Non-responder", "Post"),
        ]:
            pass  # just checking groups exist

        for gene in available:
            try:
                means = {}
                for arm in ["Responder", "Non-responder"]:
                    for vis in ["Pre", "Post"]:
                        mask = (ct_data["_arm"] == arm) & (ct_data["_visit"] == vis)
                        vals = ct_data.loc[mask, gene]
                        means[(arm, vis)] = vals.mean() if len(vals) > 0 else np.nan

                # DiD = (R_post - R_pre) - (NR_post - NR_pre)
                r_delta = means[("Responder", "Post")] - means[("Responder", "Pre")]
                nr_delta = means[("Non-responder", "Post")] - means[("Non-responder", "Pre")]
                did_val = r_delta - nr_delta
                if np.isfinite(did_val):
                    effect_mat.loc[gene, ct] = did_val
            except Exception:
                pass

    # Sort genes by global β_DiD for consistent ordering
    gene_order = top_genes_df.set_index("feature").loc[available].sort_values(
        beta_col, ascending=True
    ).index.tolist()
    effect_mat = effect_mat.loc[gene_order]

    # Drop cell types with all NaN
    effect_mat = effect_mat.dropna(axis=1, how="all")

    if effect_mat.shape[1] == 0:
        ax.text(0.5, 0.5, "Insufficient cell-type data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    # Plot heatmap
    vmax = np.nanmax(np.abs(effect_mat.values))
    vmax = max(vmax, 0.01)  # avoid degenerate scale

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "did_div",
        [COLORS["control"], "#f0f0f0", COLORS["treated"]],
        N=256,
    )

    im = ax.imshow(
        effect_mat.values.astype(float),
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    # Mask NaN cells with hatching
    nan_mask = np.isnan(effect_mat.values.astype(float))
    if nan_mask.any():
        masked = np.ma.array(np.ones_like(effect_mat.values, dtype=float),
                             mask=~nan_mask)
        ax.pcolormesh(
            np.arange(effect_mat.shape[1] + 1) - 0.5,
            np.arange(effect_mat.shape[0] + 1) - 0.5,
            masked,
            cmap=mcolors.ListedColormap(["#e8e8e8"]),
            vmin=0, vmax=1, zorder=0,
        )

    ax.set_xticks(np.arange(effect_mat.shape[1]))
    ax.set_xticklabels(effect_mat.columns, rotation=30, ha="right",
                       fontsize=6.5)
    ax.set_yticks(np.arange(effect_mat.shape[0]))
    ax.set_yticklabels(effect_mat.index, fontsize=7)
    ax.set_title("Cell-Type DiD Effects (Top Genes)", fontsize=11,
                 fontweight="bold")

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta\Delta$ expression", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ======================================================================
# Composite figure


# ======================================================================
# FIGURE 5 — Multi-Dataset Generalization
# ======================================================================


def effect_size_ci(g: float, n1: int, n2: int, alpha: float = 0.05):
    """Approximate CI for Hedges' g (Hedges & Olkin 1985, normal approx)."""
    from scipy.stats import norm
    se = np.sqrt(1 / n1 + 1 / n2 + g ** 2 / (2 * (n1 + n2)))
    z = norm.ppf(1 - alpha / 2)
    return g - z * se, g + z * se


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

def _prepare_multi_dataset_data() -> dict[str, Any]:
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




# ======================================================================
# COMBINED COMPOSITE & GENERATION
# ======================================================================

# ── Helpers ───────────────────────────────────────────────────────────────

_mm = 1.0 / 25.4


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


# ── Panel aliases for composite ───────────────────────────────────────────
fig4_volcano = panel_E
fig4_waterfall = panel_A
fig4_gsea_bars = panel_B
fig4_leading_edge = panel_C
fig4_celltype_hm = panel_F
fig4_gsea_cross = panel_C_replicated


# ── Individual panel saving ───────────────────────────────────────────────

def _save_individual_panels(data4: dict, data5: dict) -> None:
    """Save each panel A–L as a standalone figure."""
    print("  Saving individual panels...")

    # Fig4 panels (A–E)
    fig4_panels = [
        ("A", fig4_volcano, data4, (8, 6), dict(composite=False)),
        ("B", fig4_waterfall, data4, (8, 6), {}),
        ("C", fig4_gsea_bars, data4, (8, 6), {}),
        ("D", fig4_leading_edge, data4, (10, 7), {}),
        ("E", fig4_celltype_hm, data4, (8, 6), {}),
    ]
    for label, fn, data, fsize, kwargs in fig4_panels:
        fig_p, ax_p = plt.subplots(figsize=fsize)
        fn(ax_p, data, **kwargs)
        if label != "D":
            fig_p.tight_layout()
        save_panel(fig_p, f"panel_{label}", FIGURE_NAME, MAIN_OUTPUT)

    # Fig5 panels (F–K)
    fig5_panels = [
        (panel_a_covid,    "F_covid_severity",   data5, 6.5),
        (panel_b_vaccine,  "G_vaccine_paired",   data5, 6.5),
        (panel_c_aml,      "H_aml_clinical",     data5, 6.5),
        (panel_d_cart,      "I_cart_clinical",    data5, 6.5),
        (panel_e_melanoma, "J_melanoma_did",      data5, 6.5),
        (panel_f_heatmap,  "K_heatmap",           data5, 6.5),
    ]
    for fn, name, data, w in fig5_panels:
        n_feat = 8
        h = max(2.8, 0.38 * n_feat + 1.1)
        fig_p, ax_p = plt.subplots(figsize=(w, h))
        fn(ax_p, data)
        fig_p.tight_layout(pad=0.6)
        save_panel(fig_p, f"panel_{name}", FIGURE_NAME, MAIN_OUTPUT)

    # Panel L — cross-dataset GSEA heatmap
    fig_p, ax_p = plt.subplots(figsize=(9.5, max(2.8, 0.38 * 10 + 1.1)))
    fig4_gsea_cross(ax_p, data5)
    fig_p.tight_layout(pad=0.6)
    save_panel(fig_p, "panel_L_cross_dataset_gsea", FIGURE_NAME, MAIN_OUTPUT)

    print("    Individual panels saved (A–L)")


# ── Composite artboard ────────────────────────────────────────────────────

def _build_composite(data4: dict, data5: dict) -> None:
    """Build and save the combined 12-panel artboard (180 × 215 mm)."""
    print("  Building composite artboard...")

    _SMALL_RC = {
        "font.size": 4.5,
        "axes.titlesize": 5,
        "axes.labelsize": 4.5,
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "legend.fontsize": 3.5,
        "legend.title_fontsize": 3.5,
    }
    _MAX_FONT = 6

    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    fig_c = plt.figure(figsize=(180 * _mm, 215 * _mm))

    # Layout (9 rows: 5 content + 4 spacers):
    #   Row 0:  A (volcano) | B (waterfall)
    #   Row 2:  C (GSEA bars, narrow) | D (leading-edge, transposed)
    #   Row 4:  E (cell-type hm, narrow) | F (COVID) | G (Vaccine)
    #   Row 6:  H (AML) | I (CAR-T) | J (Melanoma)
    #   Row 8:  K (heatmap, ~square) | L (GSEA cross, narrow)
    outer = fig_c.add_gridspec(
        9, 1,
        height_ratios=[
            1.2,    # row 0: A | B
            0.55,   # spacer
            1.2,    # row 2: C | D
            0.55,   # spacer
            1.0,    # row 4: E | F | G
            0.65,   # spacer (extra between rows 3–4)
            1.0,    # row 6: H | I | J
            0.55,   # spacer
            1.2,    # row 8: K | L
        ],
        hspace=0.0,
        left=0.08, right=0.96, top=0.98, bottom=0.03,
    )

    # ── Row 0: A | B ────────────────────────────────────────────────
    gs0 = outer[0].subgridspec(1, 2, wspace=0.35)
    ax_a = fig_c.add_subplot(gs0[0])
    ax_b = fig_c.add_subplot(gs0[1])

    # ── Row 2: C (narrower, shifted right) | D ───────────────────────
    gs2 = outer[2].subgridspec(1, 3, wspace=0.55, width_ratios=[0.01, 0.10, 0.15])
    ax_c = fig_c.add_subplot(gs2[1])
    ax_d = fig_c.add_subplot(gs2[2])

    # ── Row 4: E (narrow) | F | G ───────────────────────────────────
    gs4 = outer[4].subgridspec(
        1, 3, wspace=0.90, width_ratios=[0.8, 1, 1],
    )
    ax_e = fig_c.add_subplot(gs4[0])
    ax_f = fig_c.add_subplot(gs4[1])
    ax_g = fig_c.add_subplot(gs4[2])

    # ── Row 6: H | I | J ────────────────────────────────────────────
    gs6 = outer[6].subgridspec(1, 3, wspace=1.05)
    ax_h = fig_c.add_subplot(gs6[0])
    ax_i = fig_c.add_subplot(gs6[1])
    ax_j = fig_c.add_subplot(gs6[2])

    # ── Row 8: K | L ────────────────────────────────────────────────
    gs8 = outer[8].subgridspec(1, 2, wspace=0.90, width_ratios=[1.2, 0.9])
    ax_k = fig_c.add_subplot(gs8[0])
    ax_l = fig_c.add_subplot(gs8[1])

    # ── Draw Fig4 panels (A–E) ──────────────────────────────────────
    fig4_volcano(ax_a, data4, composite=True)
    fig4_waterfall(ax_b, data4)
    ax_b.tick_params(axis='y', labelsize=3.5)
    if ax_b.get_ylabel():
        ax_b.yaxis.label.set_fontsize(4)

    fig4_gsea_bars(ax_c, data4)
    ax_c.tick_params(axis='y', labelsize=3.5)
    if ax_c.get_ylabel():
        ax_c.yaxis.label.set_fontsize(4)

    fig4_leading_edge(ax_d, data4, composite=True)

    # Swap x/y axes of D: pathways → Y, genes → X
    _d_imgs = ax_d.get_images()
    if _d_imgs:
        _d_arr = np.array(_d_imgs[0].get_array())
        _d_arr_t = np.transpose(
            _d_arr, (1, 0) + tuple(range(2, _d_arr.ndim)),
        )
        _d_xticks = [t.get_text() for t in ax_d.get_xticklabels()]
        _d_yticks = [t.get_text() for t in ax_d.get_yticklabels()]
        _d_xtick_colors = [t.get_color() for t in ax_d.get_xticklabels()]
        _d_title = ax_d.get_title()
        _d_leg = ax_d.get_legend()
        _d_leg_handles = _d_leg.legend_handles if _d_leg else []
        _d_leg_labels = (
            [t.get_text() for t in _d_leg.get_texts()] if _d_leg else []
        )

        ax_d.clear()
        ax_d.imshow(
            _d_arr_t, aspect="auto", interpolation="nearest", origin="lower",
        )
        _n_y, _n_x = _d_arr_t.shape[:2]
        for _gi in range(_n_y + 1):
            ax_d.axhline(_gi - 0.5, color="white", linewidth=0.8, zorder=2)
        for _gj in range(_n_x + 1):
            ax_d.axvline(_gj - 0.5, color="white", linewidth=0.8, zorder=2)
        ax_d.set_xticks(range(len(_d_yticks)))
        ax_d.set_xticklabels(
            _d_yticks, rotation=35, ha="right", fontsize=4, style="italic",
        )
        ax_d.set_yticks(range(len(_d_xticks)))
        ax_d.set_yticklabels(_d_xticks, fontsize=3.5)
        for _lbl, _clr in zip(ax_d.get_yticklabels(), _d_xtick_colors):
            _lbl.set_color(_clr)
            _lbl.set_fontweight("bold")
        ax_d.tick_params(axis="both", length=0)
        ax_d.set_title(_d_title, fontweight="bold")
        for _sp in ax_d.spines.values():
            _sp.set_visible(False)
        if _d_leg_handles:
            ax_d.legend(
                handles=_d_leg_handles, labels=_d_leg_labels,
                fontsize=3.5, loc="upper left",
                frameon=True, framealpha=0.9,
                handlelength=1.0, handleheight=0.7,
            )

    # E colorbar font
    _axes_before_e = set(fig_c.get_axes())
    fig4_celltype_hm(ax_e, data4)
    _axes_after_e = set(fig_c.get_axes())
    for _cb_ax in _axes_after_e - _axes_before_e:
        _cb_ax.tick_params(labelsize=4)
        if _cb_ax.get_ylabel():
            _cb_ax.set_ylabel(_cb_ax.get_ylabel(), fontsize=4)
        if _cb_ax.get_xlabel():
            _cb_ax.set_xlabel(_cb_ax.get_xlabel(), fontsize=4)
    ax_e.tick_params(axis='x', labelsize=3.5)
    ax_e.tick_params(axis='y', labelsize=4.0)

    # ── Draw Fig5 panels (F–L) ──────────────────────────────────────
    _fj_tick = 4.0
    _fj_lbl = 4.0
    panel_a_covid(ax_f, data5)
    panel_b_vaccine(ax_g, data5)
    panel_c_aml(ax_h, data5)
    panel_d_cart(ax_i, data5)
    panel_e_melanoma(ax_j, data5)
    for _ax_fj in [ax_f, ax_g, ax_h, ax_i, ax_j]:
        _ax_fj.tick_params(axis='both', labelsize=_fj_tick)
        if _ax_fj.get_xlabel():
            _ax_fj.xaxis.label.set_fontsize(_fj_lbl)
        if _ax_fj.get_ylabel():
            _ax_fj.yaxis.label.set_fontsize(_fj_lbl)

    # K — heatmap with colorbar
    _axes_before_k = set(fig_c.get_axes())
    panel_f_heatmap(ax_k, data5)
    _axes_after_k = set(fig_c.get_axes())
    _cb_k_size = 4.5
    for _cb_ax in _axes_after_k - _axes_before_k:
        _cb_ax.tick_params(labelsize=_cb_k_size)
        if _cb_ax.get_ylabel():
            _cb_ax.set_ylabel(_cb_ax.get_ylabel(), fontsize=_cb_k_size)
        if _cb_ax.get_xlabel():
            _cb_ax.set_xlabel(_cb_ax.get_xlabel(), fontsize=_cb_k_size)
    if ax_k.get_ylabel():
        ax_k.yaxis.label.set_fontsize(5.5)

    # L — cross-dataset GSEA with colorbar
    _axes_before_l = set(fig_c.get_axes())
    fig4_gsea_cross(ax_l, data5)
    _axes_after_l = set(fig_c.get_axes())
    for _cb_ax in _axes_after_l - _axes_before_l:
        _cb_ax.tick_params(labelsize=_cb_k_size)
        if _cb_ax.get_ylabel():
            _cb_ax.set_ylabel(_cb_ax.get_ylabel(), fontsize=_cb_k_size)
        if _cb_ax.get_xlabel():
            _cb_ax.set_xlabel(_cb_ax.get_xlabel(), fontsize=_cb_k_size)
    ax_l.set_ylabel("")

    # ── Post-processing ──────────────────────────────────────────────

    # Remove single-letter panel labels added by panel functions
    for _ax in [ax_f, ax_g, ax_h, ax_i, ax_j, ax_k, ax_l]:
        to_remove = [
            t for t in _ax.texts
            if len(t.get_text()) == 1 and t.get_text().isupper()
        ]
        for t in to_remove:
            t.remove()

    # Legends — Fig4: A/B default, C/D top-left
    for ax_target, loc in {
        ax_a: "upper right", ax_b: "lower right",
        ax_c: "upper left", ax_d: "upper left",
    }.items():
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_target.legend(
                handles=handles, labels=labels,
                fontsize=3.5, loc=loc,
                frameon=True, framealpha=0.85,
                handlelength=1, handletextpad=0.3,
                borderpad=0.3, labelspacing=0.2,
            )

    # Legends — Fig5 forest panels
    for ax_target in [ax_f, ax_g, ax_h, ax_i, ax_j]:
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_target.legend(
                handles=handles, labels=labels,
                fontsize=3.5, loc="lower right",
                frameon=True, framealpha=0.85,
                edgecolor="#CCCCCC", borderpad=0.3,
                handlelength=1, handletextpad=0.3,
                labelspacing=0.2,
            )

    # Shrink annotation text in volcano/waterfall
    for _ax in [ax_a, ax_b]:
        for txt in _ax.texts:
            if txt.get_fontsize() > 5:
                txt.set_fontsize(max(txt.get_fontsize() * 0.55, 3.0))

    # Reduce heatmap annotation font in K
    for txt in ax_k.texts:
        txt.set_fontsize(max(txt.get_fontsize() * 0.55, 2.5))
    ax_k.tick_params(axis='both', labelsize=4.5)

    # L: center-align stars in each cell, black color
    for txt in ax_l.texts:
        _tstr = txt.get_text().strip()
        if _tstr and all(c in "*†✱★" for c in _tstr):
            _tx, _ty = txt.get_position()
            txt.set_position((round(_tx), round(_ty)))
            txt.set_ha("center")
            txt.set_va("center_baseline")
            txt.set_color("black")
    ax_l.tick_params(axis='x', labelsize=4)
    ax_l.tick_params(axis='y', labelsize=4.0)

    # Standardize all titles: center-aligned, uniform font size
    _title_fs = 5
    for _ax in [ax_a, ax_b, ax_c, ax_d, ax_e,
                ax_f, ax_g, ax_h, ax_i, ax_j,
                ax_k, ax_l]:
        _ax.set_title(_ax.get_title(), fontsize=_title_fs, fontweight="bold",
                      loc="center")

    _cap_fontsize(fig_c, _MAX_FONT)

    # Bold panel labels A–L (E–J further left & higher; E, I extra left)
    _lbl_fs = 7
    _lbl_x, _lbl_y = -0.12, 1.12
    _lbl_x_far, _lbl_y_far = -0.25, 1.16
    _lbl_x_extra = -0.35
    for _ax, lbl in [
        (ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D"),
        (ax_k, "K"),
    ]:
        _ax.text(_lbl_x, _lbl_y, lbl, transform=_ax.transAxes,
                 fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_l.text(-0.55, _lbl_y, "L", transform=ax_l.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    for _ax, lbl in [
        (ax_f, "F"), (ax_g, "G"),
        (ax_h, "H"), (ax_j, "J"),
    ]:
        _ax.text(_lbl_x_far, _lbl_y_far, lbl, transform=_ax.transAxes,
                 fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    for _ax, lbl in [(ax_e, "E"), (ax_i, "I")]:
        _ax.text(_lbl_x_extra, _lbl_y_far, lbl, transform=_ax.transAxes,
                 fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")

    plt.rcParams.update(_prev_rc)

    # Save composite
    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, MAIN_OUTPUT, close=False)
    pdf_path = MAIN_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print("    Composite artboard saved (PNG + PDF)")


# ── Public entry point ────────────────────────────────────────────────────

def generate() -> None:
    """Generate individual panels A–L and the combined composite."""
    print("=" * 60)
    print("Figure 4: Biological Discovery & Multi-Dataset Generalization")
    print("=" * 60)

    print("  Preparing Figure 4 data...")
    data4 = _prepare_bio_discovery_data()
    print("  Preparing Figure 5 data...")
    data5 = _prepare_multi_dataset_data()

    # Reuse multi-dataset GSEA from Fig4 data prep (already cached)
    gsea_multi = data4.get("gsea_multi_dataset")
    if gsea_multi is None:
        gsea_multi = _run_multi_dataset_gsea(
            sf_gsea_results=data4.get("gsea_results"),
        )
    data5["gsea_multi_dataset"] = gsea_multi

    _save_individual_panels(data4, data5)
    _build_composite(data4, data5)

    if "adata" in data4:
        del data4["adata"]
    del data4, data5
    gc.collect()
    print("  Done.\n")


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    generate()
