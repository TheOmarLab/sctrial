"""
Figure 5 v2 -- Biological Discovery (CAR-T Dataset)
====================================================

Analogous to Figure 5 but using the CAR-T dataset (GSE290722, n=32
participants, single-arm, Pre/Post paired design).

Statistical method: within-arm paired comparison (participant fixed-effects
model, equivalent to a paired t-test) via ``sctrial.within_arm_comparison()``.

Panels
------
A  Top genes ranked by effect size (waterfall plot, protein-coding only).
B  GSEA enrichment bar chart (immune + metabolic pathways, 5 libraries).
C  Leading-edge gene overlap heatmap across top enriched pathways.
D  Signature within-arm effects with CIs (forest plot).
E  Gene-level volcano plot (paired Pre→Post, protein-coding labels).
"""

from __future__ import annotations

import gc
import re
import traceback
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .._shared import *  # noqa: F401,F403

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "Figure5v2_cart_biological_discovery"
FIGSIZE = (18, 14)

# Reuse helpers from Figure 5
from .figure5_biological_discovery import (
    _is_likely_protein_coding,
    _detect_gsea_columns,
    _clean_pathway_name,
    _is_immune_or_metabolic,
)


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_data() -> dict:
    """Load CAR-T dataset, run within-arm comparison, GSEA, gene-level."""
    from sctrial import (
        TrialDesign,
        within_arm_comparison,
    )

    # ------------------------------------------------------------------
    # 1. Load CAR-T dataset
    # ------------------------------------------------------------------
    adata = load_clinical_trial_dataset("cart")

    # Score gene signatures
    adata, sig_cols = score_signatures(adata, layer="log1p_norm")

    # For within-arm comparison we still need a TrialDesign.
    # All participants are in the same arm ("CAR-T").
    # We create a dummy arm column so the design object works.
    adata.obs["arm_dummy"] = "CAR-T"
    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm_dummy",
        arm_treated="CAR-T",
        arm_control="CAR-T",  # same arm — within-arm only
    )
    visits = ("Pre", "Post")

    # ------------------------------------------------------------------
    # 2. Signature-level within-arm comparison
    # ------------------------------------------------------------------
    sig_results = within_arm_comparison(
        adata,
        arm="CAR-T",
        features=sig_cols,
        design=design,
        visits=visits,
        layer="log1p_norm",
        standardize=True,
        aggregate="participant_visit",
    )
    print(f"  CAR-T signature results: {len(sig_results)} signatures, "
          f"{(sig_results['FDR_time'] < 0.1).sum()} FDR<0.1")

    # ------------------------------------------------------------------
    # 3. GSEA on Pre→Post ranking
    # ------------------------------------------------------------------
    # run_gsea_pseudobulk uses DiD internally (two-arm), so for single-arm
    # CAR-T we rank genes by their within-arm t-statistic and run
    # gseapy.prerank directly.
    import gseapy as gp

    # Run within-arm on ALL genes to get a genome-wide ranking
    all_genes_res = within_arm_comparison(
        adata,
        arm="CAR-T",
        features=adata.var_names.tolist(),
        design=design,
        visits=visits,
        layer="log1p_norm",
        standardize=True,
        aggregate="participant_visit",
    )
    # Rank by t-statistic = beta_time / se_time (continuous, few ties)
    all_genes_res["tstat"] = all_genes_res["beta_time"] / all_genes_res["se_time"]
    ranking = (
        all_genes_res.dropna(subset=["tstat"])
        .set_index("feature")["tstat"]
        .sort_values(ascending=False)
    )
    # Remove infinite values
    ranking = ranking.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  Gene ranking: {len(ranking)} genes (for GSEA)")

    # 5 libraries including WikiPathways
    gsea_libraries = [
        "MSigDB_Hallmark_2020",
        "KEGG_2021_Human",
        "Reactome_2022",
        "GO_Biological_Process_2023",
        "WikiPathways_2024_Human",
    ]
    gsea_all = {}
    for lib in gsea_libraries:
        try:
            pre = gp.prerank(
                rnk=ranking,
                gene_sets=lib,
                min_size=10,
                max_size=500,
                permutation_num=1000,
                outdir=None,
                no_plot=True,
                seed=42,
            )
            res = pre.res2d if hasattr(pre, "res2d") else pd.DataFrame(pre)
            if isinstance(res, pd.DataFrame) and len(res) > 0:
                res["library"] = lib
                gsea_all[lib] = res
                print(f"  GSEA {lib}: {len(res)} pathways tested")
        except Exception as exc:
            print(f"  GSEA {lib} unavailable: {exc}")

    gsea_results = None
    if gsea_all:
        gsea_results = pd.concat(gsea_all.values(), ignore_index=True)

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
            gsea_results["_clean_term"] = gsea_results[term_col].apply(
                lambda s: _clean_pathway_name(s, max_len=200)
            )
            gsea_results[nes_col_name] = pd.to_numeric(
                gsea_results[nes_col_name], errors="coerce"
            )

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

            dup_mask = gsea_results.duplicated(subset=["_clean_term"], keep=False)
            n_dup_pathways = gsea_results.loc[dup_mask, "_clean_term"].nunique()
            if n_dup_pathways > 0:
                print(f"  Averaging {n_dup_pathways} duplicate pathways "
                      f"across libraries")

                deduped_rows = []
                for clean_name, group in gsea_results.groupby("_clean_term"):
                    if len(group) == 1:
                        deduped_rows.append(group.iloc[0].to_dict())
                        continue
                    row = group.iloc[0].to_dict()
                    row[nes_col_name] = group[nes_col_name].mean()
                    if nom_col is not None and nom_col in group.columns:
                        row[nom_col] = pd.to_numeric(
                            group[nom_col], errors="coerce"
                        ).mean()
                    if fdr_col_detect is not None and fdr_col_detect in group.columns:
                        row[fdr_col_detect] = pd.to_numeric(
                            group[fdr_col_detect], errors="coerce"
                        ).mean()
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
                    row["library"] = "averaged"
                    deduped_rows.append(row)

                gsea_results = pd.DataFrame(deduped_rows)

            gsea_results = gsea_results.drop(columns=["_clean_term"], errors="ignore")

        # Pool FDR across libraries
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
            gsea_results["FDR_per_library"] = gsea_results[fdr_col_name]
            nom_col = None
            for c in gsea_results.columns:
                cl = c.lower().strip()
                if cl in ("nom p-val", "pval", "p-value", "nom_pval"):
                    nom_col = c
                    break
            pvals = (
                pd.to_numeric(gsea_results[nom_col], errors="coerce").fillna(1).values
                if nom_col is not None
                else gsea_results[fdr_col_name].fillna(1).values
            )
            _, fdr_pooled, _, _ = multipletests(pvals, method="fdr_bh")
            gsea_results[fdr_col_name] = fdr_pooled
            n_sig = (fdr_pooled < 0.25).sum()
            print(f"  GSEA total: {len(gsea_results)} immune/metabolic pathways, "
                  f"{n_sig} FDR<0.25 after pooled correction")

    # ------------------------------------------------------------------
    # 4. Gene-level within-arm comparison (top 2000 variable genes)
    # ------------------------------------------------------------------
    gene_results = None
    try:
        import scanpy as sc

        adata_genes = adata.copy()
        sc.pp.highly_variable_genes(
            adata_genes, n_top_genes=2000, layer="log1p_norm", flavor="seurat",
        )
        top_genes = adata_genes.var_names[
            adata_genes.var["highly_variable"]
        ].tolist()
        print(f"  CAR-T: {len(top_genes)} variable genes selected")

        gene_results = within_arm_comparison(
            adata_genes,
            arm="CAR-T",
            features=top_genes,
            design=design,
            visits=visits,
            layer="log1p_norm",
            standardize=True,
            aggregate="participant_visit",
        )

        # Handle degenerate fits (NaN SE/p-value)
        n_total = len(gene_results)
        n_degenerate = gene_results["p_time"].isna().sum()
        if n_degenerate > 0:
            gene_results = gene_results.dropna(
                subset=["beta_time", "se_time", "p_time"]
            ).reset_index(drop=True)
            print(
                f"  Gene-level: dropped {n_degenerate}/{n_total} genes "
                f"with degenerate fits (NaN SE/p-value)"
            )

        n_sig = (gene_results["p_time"] < 0.05).sum()
        n_fdr = (gene_results["FDR_time"] < 0.1).sum()
        print(f"  Gene-level results: {len(gene_results)} genes, "
              f"{n_sig} nominal p<0.05, {n_fdr} FDR<0.1")

        del adata_genes
        gc.collect()

    except Exception as exc:
        print(f"  Gene-level analysis unavailable: {exc}")
        traceback.print_exc()

    return dict(
        adata=adata,
        sig_cols=sig_cols,
        design=design,
        visits=visits,
        sig_results=sig_results,
        gsea_results=gsea_results,
        gene_results=gene_results,
    )


# ======================================================================
# Panel A -- Gene waterfall plot (top genes by effect size)
# ======================================================================

def panel_A(ax, data: dict):
    """Top 30 protein-coding genes ranked by effect size (CAR-T)."""
    gene_results = data.get("gene_results")

    if gene_results is None or len(gene_results) == 0:
        ax.text(0.5, 0.5, "Gene-level results unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df = gene_results.copy()
    beta_col = "beta_time"
    p_col = "p_time"

    df = df.dropna(subset=[beta_col, p_col])
    df = df[df["feature"].apply(_is_likely_protein_coding)]

    p_thresh = 0.05
    n_per_side = 15

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

    ax.set_xlabel(r"Effect size ($\beta_{\mathrm{time}}$)")
    ax.set_title("Top Genes by Effect Size — CAR-T Pre→Post", fontsize=11)

    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.9,
                       label="Post ↑ (p < 0.05)"),
        mpatches.Patch(color=COLORS["treated"], alpha=0.35,
                       label="Post ↑ (n.s.)"),
        mpatches.Patch(color=COLORS["control"], alpha=0.9,
                       label="Pre ↑ (p < 0.05)"),
        mpatches.Patch(color=COLORS["control"], alpha=0.35,
                       label="Pre ↑ (n.s.)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel B -- GSEA Enrichment Bar Chart
# ======================================================================

def panel_B(ax, data: dict):
    """GSEA immune + metabolic pathway enrichment bar chart (CAR-T Pre→Post)."""
    gsea_results = data["gsea_results"]

    if gsea_results is None or len(gsea_results) == 0:
        ax.text(0.5, 0.5, "GSEA results unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df = gsea_results.copy()
    cols = _detect_gsea_columns(df)
    nes_col, fdr_col, term_col = cols["nes"], cols["fdr"], cols["term"]

    if nes_col is None:
        ax.text(0.5, 0.5, "NES column not found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Immune/metabolic filtering already done in _prepare_data.

    # Balanced selection: top N up + top N down
    n_show = 15
    df_pos = df[df[nes_col] > 0].nlargest(n_show // 2 + 1, nes_col)
    df_neg = df[df[nes_col] < 0].nsmallest(n_show - len(df_pos), nes_col)
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

    # Clean and disambiguate pathway names
    df_selected["pathway"] = df_selected[term_col].apply(_clean_pathway_name)
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

    # Color: blue = Post ↑ (NES > 0), orange = Pre ↑ (NES < 0)
    clr_up_sig = COLORS["treated"]
    clr_up_ns = COLORS["treated"] + "66"
    clr_dn_sig = COLORS["control"]
    clr_dn_ns = COLORS["control"] + "66"
    colors = []
    for _, row in df_selected.iterrows():
        sig = (fdr_col is not None and pd.notna(row.get(fdr_col))
               and row[fdr_col] < 0.25)
        if row[nes_col] > 0:
            colors.append(clr_up_sig if sig else clr_up_ns)
        else:
            colors.append(clr_dn_sig if sig else clr_dn_ns)

    y_pos = np.arange(len(df_selected))
    ax.barh(y_pos, df_selected[nes_col].values, color=colors, alpha=0.9,
            edgecolor="white", linewidth=0.5, height=0.7)

    # Significance stars
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
    ax.set_title("GSEA Pathway Enrichment — CAR-T Pre→Post "
                 "(Immune + Metabolic)", fontsize=11)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.9,
                       label="Post ↑ (FDR < 0.25)"),
        mpatches.Patch(color=COLORS["treated"], alpha=0.4,
                       label="Post ↑ (n.s.)"),
        mpatches.Patch(color=COLORS["control"], alpha=0.9,
                       label="Pre ↑ (FDR < 0.25)"),
        mpatches.Patch(color=COLORS["control"], alpha=0.4,
                       label="Pre ↑ (n.s.)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel C -- Leading-edge gene overlap heatmap
# ======================================================================

def panel_C(ax, data: dict):
    """Leading-edge gene overlap heatmap (CAR-T).

    Information-dense design:
    - Tight imshow grid coloured by NES direction
    - Pathway labels coloured by NES (blue=Post↑, orange=Pre↑)
    - Top marginal bar showing gene recurrence count
    - Hierarchical column clustering for gene co-occurrence
    - Capped to 8 pathways × 20 genes for readability
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    gsea_results = data["gsea_results"]

    if gsea_results is None or len(gsea_results) == 0:
        ax.text(0.5, 0.5, "GSEA results unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df = gsea_results.copy()
    cols = _detect_gsea_columns(df)
    nes_col, fdr_col, term_col, lead_col = (
        cols["nes"], cols["fdr"], cols["term"], cols["lead"]
    )

    if nes_col is None or lead_col is None or lead_col not in df.columns:
        ax.text(0.5, 0.5, "Leading-edge data unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Select top 8 pathways by |NES|
    if fdr_col is not None:
        sig_df = df[df[fdr_col] < 0.25]
    else:
        sig_df = df
    if len(sig_df) < 4:
        sig_df = df

    selected = sig_df.assign(
        _abs=sig_df[nes_col].abs(),
    ).nlargest(8, "_abs").drop(columns="_abs")
    selected = selected.sort_values(nes_col, ascending=True)

    pathway_genes: dict[str, set[str]] = {}
    pathway_nes: dict[str, float] = {}
    all_genes: set[str] = set()
    _seen_names: set[str] = set()
    for _, row in selected.iterrows():
        pname = _clean_pathway_name(str(row[term_col]), max_len=40)
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
        all_genes.update(genes)

    if not all_genes or not pathway_genes:
        ax.text(0.5, 0.5, "No leading-edge genes found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    gene_counts: dict[str, int] = {}
    for genes in pathway_genes.values():
        for g in genes:
            gene_counts[g] = gene_counts.get(g, 0) + 1

    n_pw_total = len(pathway_genes)

    def _discrim_score(gene: str) -> float:
        c = gene_counts[gene]
        if c < 2:
            return -999
        return -abs(c - n_pw_total / 2)

    shared_genes = sorted(
        [g for g, c in gene_counts.items() if c >= 2],
        key=lambda g: (_discrim_score(g), -gene_counts[g]),
        reverse=True,
    )
    if len(shared_genes) < 5:
        shared_genes = sorted(gene_counts.keys(),
                              key=lambda g: -gene_counts[g])[:20]
    shared_genes = shared_genes[:20]  # cap at 20

    if not shared_genes:
        ax.text(0.5, 0.5, "No shared leading-edge genes",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    pathways = list(pathway_genes.keys())
    matrix = np.zeros((len(pathways), len(shared_genes)), dtype=int)
    for i, pw in enumerate(pathways):
        for j, gene in enumerate(shared_genes):
            if gene in pathway_genes[pw]:
                matrix[i, j] = 1

    # Prune zero-information rows and columns
    row_sums = matrix.sum(axis=1)
    keep_rows = row_sums > 0
    if keep_rows.sum() < len(pathways):
        n_pruned = len(pathways) - keep_rows.sum()
        print(f"  Panel C: pruned {n_pruned} pathways with no shared genes")
        mask_list = keep_rows.tolist()
        matrix = matrix[keep_rows]
        pathways = [p for p, k in zip(pathways, mask_list) if k]

    col_sums = matrix.sum(axis=0)
    keep_cols = col_sums > 0
    if keep_cols.sum() < len(shared_genes):
        mask_list = keep_cols.tolist()
        matrix = matrix[:, keep_cols]
        shared_genes = [g for g, k in zip(shared_genes, mask_list) if k]

    if matrix.size == 0:
        ax.text(0.5, 0.5, "No shared leading-edge genes",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
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

    col_counts = matrix.sum(axis=0)

    # ── Colour constants ──
    BLUE = (0.122, 0.471, 0.706)
    ORANGE = (0.878, 0.478, 0.184)

    # ── Colour matrix ──
    rgb = np.full((n_pw, n_genes, 3), 0.94)
    for i, pw in enumerate(pathways):
        nes_val = pathway_nes.get(pw, 0)
        fill = np.array(BLUE if nes_val > 0 else ORANGE)
        for j in range(n_genes):
            if matrix[i, j] == 1:
                rgb[i, j] = fill

    # ── Plot ──
    ax.imshow(rgb, aspect="auto", interpolation="nearest", origin="lower")

    for i in range(n_pw + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.8, zorder=2)
    for j in range(n_genes + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.8, zorder=2)

    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(shared_genes, rotation=55, ha="right", fontsize=6,
                       style="italic")

    ax.set_yticks(range(n_pw))
    ax.set_yticklabels(pathways, fontsize=6.5)
    for i, (pw, label) in enumerate(zip(pathways, ax.get_yticklabels())):
        label.set_color(BLUE if pathway_nes.get(pw, 0) > 0 else ORANGE)
        label.set_fontweight("bold")
    ax.tick_params(axis="both", length=0)

    # ── Top marginal bar ──
    fig = ax.get_figure()
    ax_pos = ax.get_position()
    bar_height = 0.04
    bar_ax = fig.add_axes([
        ax_pos.x0, ax_pos.y1 + 0.005,
        ax_pos.width, bar_height,
    ])
    bar_ax.bar(range(n_genes), col_counts, width=0.7, color="#555555",
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

    ax.set_title("Leading-Edge Gene Overlap (CAR-T)", fontsize=11, pad=28)
    ax.set_xlabel("")
    ax.set_ylabel("")

    legend_handles = [
        mpatches.Patch(facecolor=BLUE, label="Post ↑"),
        mpatches.Patch(facecolor=ORANGE, label="Pre ↑"),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="upper right",
              frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
              handlelength=1.0, handleheight=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ======================================================================
# Panel D -- Signature within-arm forest plot
# ======================================================================

def panel_D(ax, data: dict):
    """Within-arm Pre→Post effect size forest plot (CAR-T).

    Blue = Post ↑ (β_time > 0), orange = Pre ↑ (β_time < 0).
    """
    sig_results = data["sig_results"]
    df = sig_results.copy()
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("beta_time", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))

    # CIs
    has_boot_ci = "ci_lo_boot" in df.columns and "ci_hi_boot" in df.columns
    if has_boot_ci:
        ci_lo = df["ci_lo_boot"]
        ci_hi = df["ci_hi_boot"]
        ci_label = "95% Bootstrap CI"
    else:
        ci_lo = df["ci_lo_time"]
        ci_hi = df["ci_hi_time"]
        ci_label = "95% CI"

    fdr_vals = df["FDR_time"].values
    sig_mask = pd.Series(fdr_vals < 0.1, index=df.index)

    for i in df.index:
        clr = COLORS["treated"] if df.loc[i, "beta_time"] > 0 else COLORS["control"]
        lw = 2.0 if sig_mask.iloc[i] else 1.5
        ax.hlines(y[i], ci_lo.iloc[i], ci_hi.iloc[i],
                  color=clr, lw=lw, zorder=1)
        ax.scatter(df.loc[i, "beta_time"], y[i],
                   color=clr, s=50, zorder=2,
                   edgecolors="white", linewidths=0.5)

    ax.axvline(0, color="black", ls=":", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display"].values, fontsize=9)
    ax.set_xlabel(r"Pre→Post change ($\beta_{\mathrm{time}}$, standardised)")
    ax.set_title(f"Signature Effects — CAR-T ({ci_label})", fontsize=11)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=COLORS["treated"], lw=1.5,
                   markersize=6, label="Post ↑"),
        plt.Line2D([0], [0], marker="o", color=COLORS["control"], lw=1.5,
                   markersize=6, label="Pre ↑"),
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
    """Volcano plot of gene-level within-arm effects (CAR-T Pre→Post).

    Labels prioritize protein-coding genes.
    """
    gene_results = data["gene_results"]

    if gene_results is None or len(gene_results) == 0:
        ax.text(0.5, 0.5, "Gene-level results unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                          edgecolor=COLORS["gray"]))
        ax.set_title("Gene-Level Volcano Plot", fontsize=11)
        ax.axis("off")
        return

    df = gene_results.copy()
    beta_col = "beta_time"
    p_col = "p_time"

    df = df.dropna(subset=[beta_col, p_col])

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

    for cat in ["ns", "up", "down"]:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[beta_col], sub["nlog10"],
            c=color_map[cat], alpha=alpha_map[cat],
            s=size_map[cat], edgecolors="none", rasterized=True,
        )

    # Label top PROTEIN-CODING genes per direction
    N_LABELS = 8
    texts = []
    labelled_genes: set[str] = set()

    for sign in ("pos", "neg"):
        sub = df[df[beta_col] > 0].copy() if sign == "pos" else df[df[beta_col] < 0].copy()
        if len(sub) == 0:
            continue
        sub = sub[sub["feature"].apply(_is_likely_protein_coding)]
        if len(sub) == 0:
            continue

        picks: list[str] = []

        sig = sub[sub[p_col] < p_thresh]
        picks.extend(
            sig.nsmallest(min(N_LABELS, len(sig)), p_col)["feature"].tolist()
        )

        remaining = N_LABELS - len(picks)
        if remaining > 0:
            pool = sub[~sub["feature"].isin(picks)]
            top_func = "nlargest" if sign == "pos" else "nsmallest"
            picks.extend(
                getattr(pool, top_func)(
                    min(remaining, len(pool)), beta_col
                )["feature"].tolist()
            )

        labelled_genes.update(picks)

    for _, row in df[df["feature"].isin(labelled_genes)].iterrows():
        dir_clr = (COLORS["treated"] if row[beta_col] > 0
                   else COLORS["control"])
        t = ax.text(
            row[beta_col], row["nlog10"], row["feature"],
            fontsize=8, fontweight="bold", color=dir_clr,
        )
        texts.append(t)

    # Suppress adjustText FancyArrowPatch transform warning
    try:
        from adjustText import adjust_text
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*transform.*",
                category=UserWarning,
            )
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(
                    arrowstyle="-|>", color="#444444",
                    lw=0.7, mutation_scale=7,
                ),
                force_text=(2.0, 2.5),
                force_points=(0.8, 0.8),
                expand_text=(1.8, 2.0),
                min_arrow_len=5,
            )
    except ImportError:
        pass

    thresh_y = -np.log10(p_thresh)
    ax.axhline(thresh_y, color=COLORS["gray"], ls="--", lw=0.8, zorder=0)
    ax.axvline(0, color="black", lw=0.6, zorder=0)

    ax.set_xlabel(r"Effect size ($\beta_{\mathrm{time}}$)")
    ax.set_ylabel(r"$-\log_{10}$(p)")
    ax.set_title("Gene-Level Volcano — CAR-T Pre→Post", fontsize=11)

    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.8,
                       label="Post ↑"),
        mpatches.Patch(color=COLORS["control"], alpha=0.8,
                       label="Pre ↑"),
        mpatches.Patch(color=COLORS["gray"], alpha=0.3,
                       label="Not significant"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower left",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Figure 5 v2 (CAR-T) individual panels."""
    print("Figure 5 v2: Biological Discovery (CAR-T)")
    data = _prepare_data()

    for panel_label, panel_func in [
        ("A", panel_A),
        ("B", panel_B),
        ("C", panel_C),
        ("D", panel_D),
        ("E", panel_E),
    ]:
        fig_p, ax_p = plt.subplots(figsize=(8, 6))
        panel_func(ax_p, data)
        fig_p.tight_layout()
        save_panel(fig_p, f"panel_{panel_label}", FIGURE_NAME, MAIN_OUTPUT)

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
