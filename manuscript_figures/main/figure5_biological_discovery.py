"""
Figure 5 -- Biological Discovery
=================================

Four-panel figure (2x2) combining GSEA pathway enrichment, leading-edge
gene analysis, signature-level DiD effects, and gene-level volcano plots.

Panels
------
A  GSEA enrichment bar chart (Hallmark pathways, balanced up/down selection).
B  Leading-edge gene overlap heatmap across top enriched pathways.
C  Signature DiD effects with bootstrap CIs (forest plot).
D  Gene-level volcano plot (Sade-Feldman DiD, protein-coding gene labels).
"""

from __future__ import annotations

import gc
import re
import traceback

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .._shared import *  # noqa: F401,F403

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "Figure5_biological_discovery"
FIGSIZE = (18, 14)

# Pseudogene / non-coding RNA patterns to deprioritize in volcano labels
_NONCODING_PATTERN = re.compile(
    r"^(RNU\d|RNA5SP|RNY\d|RN7SL|SNOR[AD]|MIR\d|LINC\d|LOC\d|"
    r"AC\d{6}|AL\d{6}|AP\d{6}|"
    r"RP\d+-|RP[SL]\d+P|CT[ABCD]-|XXbac-|KB-|LA16c-|GS\d-|"
    r"HIGD1AP|MTCO\d|RMVSL|BCRP|NAMA$|SLMO|"
    r"IGH[VDJGM]|IGK[VJC]|IGL[VJC]|"  # Ig variable/joining/constant regions
    r"TRB[VDJ]|TRA[VDJ]|TRG[VDJ]|TRD[VDJ]|"  # TCR regions
    r"OR\d+[A-Z])",  # olfactory receptors
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
# Data preparation
# ======================================================================

def _prepare_data() -> dict:
    """Load datasets, run DiD, GSEA, and gene-level analysis."""
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
    gsea_libraries = [
        "MSigDB_Hallmark_2020",
        "KEGG_2021_Human",
        "Reactome_2022",
        "GO_Biological_Process_2023",
    ]
    gsea_results = None
    gsea_all = {}  # per-library results for Panel B
    for lib in gsea_libraries:
        try:
            # Use tstat ranking to reduce ties in the preranked list.
            # signed_confidence produces many ties when n_units is small
            # because p-values cluster (23%+ duplicate rate).  The
            # t-statistic (beta/SE) varies continuously and breaks ties.
            res = run_gsea_did(
                adata,
                gene_sets=lib,
                design=design,
                visits=visits,
                layer="log1p_tpm",
                rank_by="tstat",
                min_size=10,
                max_size=500,
                permutation_num=1000,
                outdir=None,
                no_plot=True,
            )
            if isinstance(res, pd.DataFrame) and len(res) > 0:
                res["library"] = lib
                gsea_all[lib] = res
                print(f"  GSEA {lib}: {len(res)} pathways tested")
        except Exception as exc:
            print(f"  GSEA {lib} unavailable: {exc}")

    if gsea_all:
        gsea_results = pd.concat(gsea_all.values(), ignore_index=True)
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
            pvals = gsea_results[fdr_col_name].fillna(1).values
            # Use BH on the raw NES p-values (FDR q-val from gseapy is
            # already per-library BH; we re-correct across the pool).
            # gseapy's FDR q-val is the corrected p-value, so we use
            # the NOM p-val column if available for proper re-correction.
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
            _, fdr_pooled, _, _ = multipletests(
                pvals, method="fdr_bh",
            )
            gsea_results[fdr_col_name] = fdr_pooled
            n_sig_pooled = (fdr_pooled < 0.25).sum()
            print(
                f"  GSEA total: {len(gsea_results)} pathways across "
                f"{len(gsea_all)} libraries "
                f"({n_sig_pooled} FDR<0.25 after pooled correction)"
            )
        else:
            print(
                f"  GSEA total: {len(gsea_results)} pathways across "
                f"{len(gsea_all)} libraries"
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
        # Bootstrap is used for signature-level inference (Panel C) where
        # formal hypothesis tests matter; for the volcano (Panel D) we use
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

    return dict(
        adata=adata,
        sig_cols=sig_cols,
        design=design,
        visits=visits,
        did_sig=did_sig,
        gsea_results=gsea_results,
        gene_results=gene_results,
    )


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


# Irrelevant pathway terms for an immunotherapy/immune context.
# These appear when broad GO libraries surface tissue-specific or
# developmental terms that are biologically implausible for melanoma
# scRNA-seq.  Matching is case-insensitive substring.
_IRRELEVANT_PATHWAY_TERMS = [
    "sperm", "spermat", "flagell", "cilium assembly",
    "odontogenesis", "amelogenesis", "enamel",
    "cardiac chamber", "heart jogging",
    "embryonic digit", "limb morphogenesis",
    "sensory perception of smell", "olfactory",
    "lens fiber", "lens development",
    "photoreceptor", "retinal",
    "keratinocyte", "cornification",
    "skeletal muscle contraction",
    "flight", "insemination",
]
_IRRELEVANT_RE = re.compile(
    "|".join(re.escape(t) for t in _IRRELEVANT_PATHWAY_TERMS),
    re.IGNORECASE,
)


def _is_relevant_pathway(term: str) -> bool:
    """Return False for pathways implausible in immunotherapy context."""
    return _IRRELEVANT_RE.search(str(term)) is None


# ======================================================================
# Panel A -- GSEA Enrichment Bar Chart
# ======================================================================

def panel_A(ax, data: dict):
    """GSEA Hallmark pathway enrichment bar chart with balanced up/down."""
    gsea_results = data["gsea_results"]

    if gsea_results is None or len(gsea_results) == 0:
        _panel_A_signature_waterfall(ax, data)
        return

    df = gsea_results.copy()
    cols = _detect_gsea_columns(df)
    nes_col, fdr_col, term_col = cols["nes"], cols["fdr"], cols["term"]

    if nes_col is None:
        _panel_A_signature_waterfall(ax, data)
        return

    # Convert to numeric
    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Filter out biologically implausible pathways for this context
    if term_col is not None:
        n_before = len(df)
        df = df[df[term_col].apply(_is_relevant_pathway)]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  Panel A: dropped {n_dropped} irrelevant pathways")

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

    # Clean pathway names
    df_selected["pathway"] = df_selected[term_col].apply(_clean_pathway_name)

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

    # Significance stars — position outside bar end
    if fdr_col is not None:
        for i, (_, row) in enumerate(df_selected.iterrows()):
            fdr_val = row[fdr_col]
            if pd.notna(fdr_val) and fdr_val < 0.25:
                if fdr_val < 0.001:
                    star = "***"
                elif fdr_val < 0.01:
                    star = "**"
                elif fdr_val < 0.05:
                    star = "*"
                else:
                    star = "†"
                x_pos = row[nes_col]
                # Place star on the side away from zero (outside the bar)
                if x_pos > 0:
                    ax.text(x_pos + 0.08, i, star, ha="left", va="center",
                            fontsize=8, fontweight="bold", color="#333333")
                else:
                    ax.text(x_pos - 0.08, i, star, ha="right", va="center",
                            fontsize=8, fontweight="bold", color="#333333")

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_selected["pathway"].values, fontsize=8)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title("GSEA Pathway Enrichment (Pooled FDR)", fontsize=11)

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
# Panel B -- Leading-edge gene overlap heatmap
# ======================================================================

def panel_B(ax, data: dict):
    """Leading-edge gene overlap heatmap across top enriched pathways."""
    gsea_results = data["gsea_results"]

    if gsea_results is None or len(gsea_results) == 0:
        _panel_B_did_summary(ax, data)
        return

    df = gsea_results.copy()
    cols = _detect_gsea_columns(df)
    nes_col, fdr_col, term_col, lead_col = (
        cols["nes"], cols["fdr"], cols["term"], cols["lead"]
    )

    if nes_col is None or lead_col is None or lead_col not in df.columns:
        _panel_B_did_summary(ax, data)
        return

    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Filter irrelevant pathways (same as Panel A)
    if term_col is not None:
        df = df[df[term_col].apply(_is_relevant_pathway)]

    # Select top 8 pathways by |NES| (FDR < 0.25 preferred)
    if fdr_col is not None:
        sig_df = df[df[fdr_col] < 0.25]
    else:
        sig_df = df
    if len(sig_df) < 4:
        sig_df = df  # fall back to all

    selected = sig_df.assign(
        _abs=sig_df[nes_col].abs(),
    ).nlargest(8, "_abs").drop(columns="_abs")
    selected = selected.sort_values(nes_col, ascending=False)

    # Parse leading-edge genes — use unique display names to avoid
    # key collisions when two pathways truncate to the same label.
    pathway_genes: dict[str, set[str]] = {}
    all_genes: set[str] = set()
    _seen_names: set[str] = set()
    for _, row in selected.iterrows():
        pname = _clean_pathway_name(str(row[term_col]), max_len=30)
        # Disambiguate duplicate display names
        if pname in _seen_names:
            lib = str(row.get("library", ""))
            pname = f"{pname} [{lib[:8]}]" if lib else f"{pname} (2)"
        _seen_names.add(pname)
        genes_str = str(row[lead_col])
        genes = [
            g.strip()
            for g in genes_str.replace(";", ",").split(",")
            if g.strip()
        ]
        # Keep only protein-coding-looking genes
        genes = [g for g in genes if _is_likely_protein_coding(g)]
        pathway_genes[pname] = set(genes)
        all_genes.update(genes)

    if not all_genes or not pathway_genes:
        _panel_B_did_summary(ax, data)
        return

    # Select genes appearing in ≥2 pathways for the heatmap (most informative)
    gene_counts = {}
    for genes in pathway_genes.values():
        for g in genes:
            gene_counts[g] = gene_counts.get(g, 0) + 1

    shared_genes = sorted([g for g, c in gene_counts.items() if c >= 2],
                          key=lambda g: -gene_counts[g])
    if len(shared_genes) < 5:
        # Too few shared genes; show top genes by frequency
        shared_genes = sorted(gene_counts.keys(), key=lambda g: -gene_counts[g])[:25]
    shared_genes = shared_genes[:25]  # cap at 25 for readability

    if not shared_genes:
        _panel_B_did_summary(ax, data)
        return

    # Build binary matrix
    pathways = list(pathway_genes.keys())
    matrix = np.zeros((len(pathways), len(shared_genes)), dtype=int)
    for i, pw in enumerate(pathways):
        for j, gene in enumerate(shared_genes):
            if gene in pathway_genes[pw]:
                matrix[i, j] = 1

    # Plot heatmap with explicit legend for binary values
    from matplotlib.colors import ListedColormap
    binary_cmap = ListedColormap(["#FFF8DC", "#8B0000"])  # cream / dark red
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=shared_genes,
        yticklabels=pathways,
        cmap=binary_cmap,
        vmin=0, vmax=1,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        square=False,
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=7,
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title("Leading-Edge Gene Overlap", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Binary legend (■ present / □ absent)
    legend_handles = [
        mpatches.Patch(facecolor="#8B0000", edgecolor="grey",
                       label="In leading edge"),
        mpatches.Patch(facecolor="#FFF8DC", edgecolor="grey",
                       label="Absent"),
    ]
    ax.legend(
        handles=legend_handles, fontsize=7, loc="lower right",
        frameon=True, framealpha=0.9,
    )


def _panel_B_did_summary(ax, data: dict):
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
# Panel C -- Signature DiD forest plot with bootstrap CIs
# ======================================================================

def panel_C(ax, data: dict):
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
# Panel D -- Gene-level volcano plot
# ======================================================================

def panel_D(ax, data: dict):
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
    # inference is reserved for signature-level tests (Panel C).
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

    # Label top genes by |effect size|, regardless of significance.
    # With n=10 clusters and bootstrap p-values, very few genes pass
    # p<0.05 — but extreme effect sizes are still informative.
    # Protein-coding genes are prioritised; non-coding backfills.
    texts = []
    for sign, n_label in [("pos", 8), ("neg", 8)]:
        if sign == "pos":
            sub = df[df[beta_col] > 0].copy()
            top_func = "nlargest"
        else:
            sub = df[df[beta_col] < 0].copy()
            top_func = "nsmallest"
        if len(sub) == 0:
            continue

        sub["_pc"] = sub["feature"].apply(_is_likely_protein_coding)
        pc = sub[sub["_pc"]]
        nc = sub[~sub["_pc"]]

        top_pc = getattr(pc, top_func)(
            min(n_label, len(pc)), beta_col,
        )
        remaining = n_label - len(top_pc)
        if remaining > 0 and len(nc) > 0:
            top_nc = getattr(nc, top_func)(
                min(remaining, len(nc)), beta_col,
            )
            top = pd.concat([top_pc, top_nc])
        else:
            top = top_pc

        for _, row in top.iterrows():
            is_pc = _is_likely_protein_coding(row["feature"])
            # Color by direction regardless of significance
            dir_clr = (COLORS["treated"] if row[beta_col] > 0
                       else COLORS["control"])
            t = ax.text(
                row[beta_col], row["nlog10"], row["feature"],
                fontsize=6.5,
                fontweight="bold" if is_pc else "normal",
                fontstyle="normal" if is_pc else "italic",
                color=dir_clr,
            )
            texts.append(t)

    # Use adjustText with small arrows pointing from labels to dots
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(
                arrowstyle="-|>", color="#555555",
                lw=0.6, mutation_scale=6,
            ),
            force_text=(1.5, 2.0),
            force_points=(0.5, 0.5),
            expand_text=(1.5, 1.8),
            min_arrow_len=5,
        )
    except ImportError:
        pass

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
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Figure 5 individual panels."""
    print("Figure 5: Biological Discovery")
    data = _prepare_data()

    # ── Save individual panels ────────────────────────────────────────
    for panel_label, panel_func in [
        ("A", panel_A),
        ("B", panel_B),
        ("C", panel_C),
        ("D", panel_D),
    ]:
        fig_p, ax_p = plt.subplots(figsize=(8, 6))
        panel_func(ax_p, data)
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
