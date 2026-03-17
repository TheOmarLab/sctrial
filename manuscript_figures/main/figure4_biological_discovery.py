"""
Figure 4 — Biological Discovery: Pathways & Genes
===================================================

Six-panel figure combining GSEA pathway enrichment, leading-edge gene
analysis, cross-dataset pathway replication, gene-level volcano and
waterfall plots, and effect-size distribution.

Panels
------
A  GSEA enrichment bar chart (immune + metabolic pathways, 5 libraries).
B  Leading-edge gene overlap heatmap across top enriched pathways.
C  Replicated pathways across cohorts (cross-dataset consistency).
D  Gene-level volcano plot (Sade-Feldman DiD, protein-coding gene labels).
E  Top genes ranked by effect size (waterfall plot, protein-coding only).
F  Cell-type-resolved DiD effect heatmap for top genes.
"""

from __future__ import annotations

import gc
import hashlib
import pickle  # noqa: S403 — local dev cache of our own DataFrames
import re
import traceback
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._shared import (
    COLORS,
    GSEA_LIBRARIES,
    MAIN_OUTPUT,
    TrialDesign,
    apply_style,
    despine,
    did_table,
    get_aml,
    get_cart,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    load_or_run_gsea_cross_sectional,
    load_or_run_gsea_did,
    load_or_run_gsea_within_arm,
    run_gsea_did,
    save_panel,
    score_signatures,
    sig_display,
)

# ── Cache directory for expensive computations ─────────────────────────
_CACHE_DIR = Path(__file__).resolve().parent.parent / "_cache"

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "Figure4_biological_discovery"

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

def _prepare_data(*, use_cache: bool = True) -> dict:
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
    # Uses shared helper that caches results under manuscript/GSEA/Sade_Feldman/
    gsea_results = load_or_run_gsea_did(
        adata, design, visits, "log1p_tpm", "Sade_Feldman",
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
    
    # 1. Sade-Feldman - reuse results from _prepare_data() if available
    if sf_gsea_results is not None and len(sf_gsea_results) > 0:
        # Add dataset column if not present
        sf_results = sf_gsea_results.copy()
        if "dataset" not in sf_results.columns:
            sf_results["dataset"] = "Sade-Feldman"
        # Ensure library column name is consistent
        if "Library" in sf_results.columns and "library" not in sf_results.columns:
            sf_results = sf_results.rename(columns={"Library": "library"})
        gsea_multi["Sade-Feldman"] = sf_results
        print(f"    Sade-Feldman: {len(sf_results)} pathways (reused from _prepare_data)")
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
                sf, sf_design, ("Pre", "Post"), "log1p_tpm", "Sade_Feldman",
            )
            if sf_results is not None and len(sf_results) > 0:
                sf_results["dataset"] = "Sade-Feldman"
                # Rename Library column to library for consistency
                if "Library" in sf_results.columns:
                    sf_results = sf_results.rename(columns={"Library": "library"})
                gsea_multi["Sade-Feldman"] = sf_results
                print(f"    Sade-Feldman: {len(sf_results)} pathways (cached or computed)")
        except Exception as exc:
            print(f"    Sade-Feldman: FAILED ({exc})")
    
    # 2. Vaccine (use within_arm_comparison)
    try:
        vax = get_vaccine()
        vax, _ = score_signatures(vax, layer="counts")
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
            layer="counts", dataset_name="Vaccine",
        )
        if vax_results is not None and len(vax_results) > 0:
            vax_results["dataset"] = "Vaccine"
            if "Library" in vax_results.columns and "library" not in vax_results.columns:
                vax_results = vax_results.rename(columns={"Library": "library"})
            gsea_multi["Vaccine"] = vax_results
            print(f"    Vaccine: {len(vax_results)} pathways")
    except Exception as exc:
        print(f"    Vaccine: FAILED ({exc})")
    
    # 3. AML (use within_arm_comparison)
    try:
        aml = get_aml()
        aml, _ = score_signatures(aml, layer="counts")
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
            layer="counts", dataset_name="AML",
        )
        if aml_results is not None and len(aml_results) > 0:
            aml_results["dataset"] = "AML"
            if "Library" in aml_results.columns and "library" not in aml_results.columns:
                aml_results = aml_results.rename(columns={"Library": "library"})
            gsea_multi["AML"] = aml_results
            print(f"    AML: {len(aml_results)} pathways")
    except Exception as exc:
        print(f"    AML: FAILED ({exc})")
    
    # 4. CAR-T (use within_arm_comparison)
    try:
        cart = get_cart()
        cart, _ = score_signatures(cart, layer="counts")
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
            layer="counts", dataset_name="CAR-T",
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
        covid, _ = score_signatures(covid, layer="counts")
        if "dfo_bin" in covid.obs.columns:
            top_bin = covid.obs["dfo_bin"].value_counts().idxmax()
        else:
            top_bin = "Pre"
        covid_design = TrialDesign(
            participant_col="participant_id",
            visit_col="dfo_bin",
            arm_col="severity",
            arm_treated="Severe",
            arm_control="Mild",
        )
        covid_results = load_or_run_gsea_cross_sectional(
            covid, covid_design, visit=top_bin,
            layer="counts", dataset_name="COVID-19",
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
    ax.set_yticklabels(selected["feature"].values, fontsize=7)

    ax.set_xlabel(r"Effect size ($\beta_{\mathrm{DiD}}$)")
    ax.set_title("Top Genes by Effect Size — Sade-Feldman DiD", fontsize=11)

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
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
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
    ax.set_title("Signature DiD Effects (Sade-Feldman)", fontsize=11)
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
    ax.set_title("Pathway Enrichment", fontsize=11)

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
        ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
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
    ax.set_title("DiD Signature Effects", fontsize=11)
    despine(ax)


# ======================================================================
# Panel C -- Leading-edge gene overlap heatmap
# ======================================================================

def panel_C(ax, data: dict):
    """Leading-edge gene overlap heatmap across top enriched pathways.

    Information-dense design:
    - Tight imshow grid coloured by NES direction
    - Pathway labels coloured by NES (blue=Responder↑, orange=Non-responder↑)
    - Top marginal bar showing gene recurrence count
    - Hierarchical column clustering for gene co-occurrence
    - Capped to 8 pathways × 20 genes for readability
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

    # ── Plot with imshow (tight, no whitespace) ──
    ax.imshow(rgb, aspect="auto", interpolation="nearest", origin="lower")

    # Thin white grid lines
    for i in range(n_pw + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.8, zorder=2)
    for j in range(n_genes + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.8, zorder=2)

    # Separator line between NES<0 and NES>0 blocks
    if n_sep > 0 and n_sep < n_pw:
        ax.axhline(n_sep - 0.5, color="black", linewidth=1.5, zorder=3)

    # X-axis: gene labels
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(shared_genes, rotation=55, ha="right", fontsize=6,
                       style="italic")

    # Y-axis: pathway labels (no FDR annotation), coloured by NES direction
    ax.set_yticks(range(n_pw))
    ax.set_yticklabels(pathways, fontsize=6.5)
    for i, (pw, label) in enumerate(zip(pathways, ax.get_yticklabels())):
        label.set_color(BLUE if pathway_nes.get(pw, 0) > 0 else ORANGE)
        label.set_fontweight("bold")
    ax.tick_params(axis="both", length=0)

    # ── Top marginal bar: gene recurrence count ──
    # Run tight_layout FIRST so ax position accounts for tick labels,
    # then position the marginal bar relative to the adjusted axes.
    fig = ax.get_figure()
    fig.tight_layout(rect=[0, 0, 1, 0.90])  # leave top 10% for bar+title
    ax_pos = ax.get_position()
    bar_height = 0.04  # fraction of figure height
    bar_ax = fig.add_axes([
        ax_pos.x0, ax_pos.y1 + 0.02,
        ax_pos.width, bar_height,
    ])
    bar_colors = ["#555555"] * n_genes
    bar_ax.bar(range(n_genes), col_counts, width=0.7, color=bar_colors,
               edgecolor="none")
    bar_ax.set_xlim(-0.5, n_genes - 0.5)
    bar_ax.set_ylim(0, max(col_counts) + 0.5)
    bar_ax.set_xticks([])
    bar_ax.set_ylabel("# paths", fontsize=5.5, rotation=0, labelpad=25,
                      va="center")
    bar_ax.tick_params(axis="y", labelsize=5.5, length=2)
    bar_ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=3))
    for spine in ["top", "right", "bottom"]:
        bar_ax.spines[spine].set_visible(False)
    bar_ax.spines["left"].set_linewidth(0.5)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Title on the bar axes — sits above bars, won't overlap heatmap
    bar_ax.set_title("Leading-Edge Gene Overlap", fontsize=11, pad=8)

    # Legend — inside the heatmap lower-right (gray empty region)
    legend_handles = [
        mpatches.Patch(facecolor=BLUE, label="Resp. ↑"),
        mpatches.Patch(facecolor=ORANGE, label="Non-resp. ↑"),
        mpatches.Patch(facecolor=EMPTY_COLOR, edgecolor="#CCCCCC",
                       label="Not in leading edge"),
    ]
    ax.legend(
        handles=legend_handles, fontsize=5.5, loc="lower right",
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
    ax.set_title("DiD Signature Effects", fontsize=11)
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
    ax.set_title(f"Signature DiD Effects ({ci_label})", fontsize=11)

    # Legend matching Figure 2 style
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=COLORS["treated"], lw=1.5,
                   markersize=6, label="Responder ↑"),
        plt.Line2D([0], [0], marker="o", color=COLORS["control"], lw=1.5,
                   markersize=6, label="Non-responder ↑"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              frameon=True, framealpha=0.9)

    n_sig = sig_mask.sum()
    ax.text(0.97, 0.03, f"{n_sig}/{len(df)} FDR < 0.1",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic", color=COLORS["gray"])
    despine(ax)


# ======================================================================
# Panel E -- Gene-level volcano plot
# ======================================================================

def panel_E(ax, data: dict):
    """Volcano plot of gene-level DiD effects (Sade-Feldman).

    Labels prioritize protein-coding genes over pseudogenes/lncRNAs.
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
        ax.set_title("Gene-Level Volcano Plot", fontsize=11)
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
    size_map = {"ns": 8, "up": 20, "down": 20}

    # Plot non-significant first, then significant on top
    for cat in ["ns", "up", "down"]:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[beta_col], sub["nlog10"],
            c=color_map[cat], alpha=alpha_map[cat],
            s=size_map[cat], edgecolors="none", rasterized=True,
        )

    # Label top PROTEIN-CODING genes using a combined score that weights
    # both statistical significance and effect size.  This ensures genes
    # at the "tips" of the volcano (high |β| AND high -log10(p)) are
    # always labelled — the exact genes a reader's eye is drawn to.
    N_LABELS = 10  # per direction
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

    # --- Render labels using adjustText for professional placement ---
    from adjustText import adjust_text as _adjust_text

    labelled_rows = df[df["feature"].isin(labelled_set)].copy()

    texts = []
    for _, row in labelled_rows.iterrows():
        dir_clr = (COLORS["treated"] if row[beta_col] > 0
                   else COLORS["control"])
        t = ax.text(
            row[beta_col], row["nlog10"], row["feature"],
            fontsize=6.5, fontweight="bold", color=dir_clr,
            ha="center", va="center", zorder=5,
        )
        texts.append(t)

    # Constrain label movement so arrows stay short and professional.
    x_span = df[beta_col].max() - df[beta_col].min()
    y_span = df["nlog10"].max() - df["nlog10"].min()
    _adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4,
                        shrinkA=5, shrinkB=3),
        force_text=(1.5, 1.5),
        force_points=(3.0, 3.0),
        expand=(1.5, 1.8),
        ensure_inside_axes=True,
        max_move=(x_span * 0.15, y_span * 0.15),
        only_move="xy",
    )

    # Threshold line
    thresh_y = -np.log10(p_thresh)
    ax.axhline(thresh_y, color=COLORS["gray"], ls="--", lw=0.8, zorder=0)
    ax.axvline(0, color="black", lw=0.6, zorder=0)

    ax.set_xlabel(r"Effect size ($\beta_{\mathrm{DiD}}$)")
    ax.set_ylabel(r"$-\log_{10}$(p)")
    ax.set_title("Gene-Level Volcano (Sade-Feldman DiD)", fontsize=11)

    # Legend — no footnotes, no summary boxes
    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.8,
                       label="Responder ↑"),
        mpatches.Patch(color=COLORS["control"], alpha=0.8,
                       label="Non-responder ↑"),
        mpatches.Patch(color=COLORS["gray"], alpha=0.3,
                       label="Not significant"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower left",
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
        ax.set_title("Replicated Pathways", fontsize=11)
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
            lambda s: _clean_pathway_name(s, max_len=50)
        )
        # Rename columns to standard names for concatenation
        df_clean = df[[nes_col_name, fdr_col_name, "pathway", "dataset"]].copy()
        df_clean.columns = ["NES", "FDR", "pathway", "dataset"]
        all_results.append(df_clean)
    
    if not all_results:
        ax.text(0.5, 0.5, "No pathways found across datasets",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.set_title("Replicated Pathways", fontsize=11)
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
        ax.set_title("Replicated Pathways", fontsize=11)
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
    ax.set_yticklabels(all_pathways, fontsize=7)
    ax.set_xticks(range(len(all_datasets)))
    ax.set_xticklabels(all_datasets, fontsize=8, rotation=45, ha="right")
    
    ax.set_xlabel("Dataset", fontsize=9)
    ax.set_ylabel("Pathway", fontsize=9)
    ax.set_title("Replicated Pathways Across Datasets (* FDR < 0.25)", fontsize=11)
    
    ax.tick_params(axis="both", length=0)
    despine(ax)


# ======================================================================
# Panel F (new) -- Gene-level effect-size distribution
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
        ax.set_title("Cell-Type DiD Effects", fontsize=11)
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
    ax.set_xticklabels(effect_mat.columns, rotation=45, ha="right",
                       fontsize=6.5)
    ax.set_yticks(np.arange(effect_mat.shape[0]))
    ax.set_yticklabels(effect_mat.index, fontsize=7)
    ax.set_title("Cell-Type DiD Effects (Top Genes)", fontsize=11)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta\Delta$ expression", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Figure 4 individual panels.

    Panel mapping (new → old):
      A  GSEA enrichment bar chart         (was Panel B)
      B  Leading-edge gene overlap heatmap  (was Panel C)
      C  Replicated pathways                (NEW)
      D  Gene-level volcano                 (was Panel E)
      E  Top genes waterfall                (was Panel A)
      F  Gene-level effect distribution     (NEW)
    """
    print("Figure 4: Biological Discovery — Pathways & Genes")
    data = _prepare_data()

    # ── Save individual panels ────────────────────────────────────────
    # Panel B (heatmap + marginal bar) needs more space for pathway labels
    panel_sizes = {
        "B": (10, 7),
    }
    for panel_label, panel_func in [
        ("A", panel_B),              # GSEA bar chart
        ("B", panel_C),              # Leading-edge heatmap
        ("C", panel_C_replicated),   # Replicated pathways (NEW)
        ("D", panel_E),              # Volcano
        ("E", panel_A),              # Waterfall
        ("F", panel_F),              # Effect distribution (NEW)
    ]:
        fsize = panel_sizes.get(panel_label, (8, 6))
        fig_p, ax_p = plt.subplots(figsize=fsize)
        panel_func(ax_p, data)
        # For Panel B (leading-edge heatmap), tight_layout must run
        # BEFORE the marginal bar is positioned (handled internally).
        if panel_label != "B":
            fig_p.tight_layout()
        save_panel(fig_p, f"panel_{panel_label}", FIGURE_NAME, MAIN_OUTPUT)

    # ── Cleanup ───────────────────────────────────────────────────────
    del data["adata"]
    del data
    gc.collect()
    print("  Done.\n")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    apply_style()
    generate()
