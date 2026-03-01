"""
Figure 5 v2 -- Biological Discovery (CAR-T Dataset)
====================================================

Analogous to Figure 5 but using the CAR-T dataset (GSE290722, n=32
participants, single-arm, Pre/Post paired design).

Statistical method: within-arm paired comparison (participant fixed-effects
model, equivalent to a paired t-test) via ``sctrial.within_arm_comparison()``.

Panels
------
A  GSEA enrichment bar chart (Hallmark + KEGG + Reactome + GO).
B  Leading-edge gene overlap heatmap across top enriched pathways.
C  Signature within-arm effects with CIs (forest plot).
D  Gene-level volcano plot (paired Pre→Post, protein-coding labels).
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
FIGURE_NAME = "Figure5v2_cart_biological_discovery"
FIGSIZE = (18, 14)

# Reuse the protein-coding filter from Figure 5
from .figure5_biological_discovery import (
    _is_likely_protein_coding,
    _detect_gsea_columns,
    _clean_pathway_name,
    _is_relevant_pathway,
)

# CAR-T specific: these pathway terms are irrelevant for a PBMC/CAR-T
# context but might surface from broad GO libraries
_CART_IRRELEVANT_TERMS = [
    "sperm", "spermat", "flagell", "cilium assembly",
    "odontogenesis", "amelogenesis", "enamel",
    "cardiac chamber", "heart jogging",
    "embryonic digit", "limb morphogenesis",
    "sensory perception of smell", "olfactory",
    "lens fiber", "lens development",
    "photoreceptor", "retinal",
    "skeletal muscle contraction",
    "flight", "insemination",
    "melanocyte",  # not relevant for CAR-T (no melanoma context)
]
_CART_IRRELEVANT_RE = re.compile(
    "|".join(re.escape(t) for t in _CART_IRRELEVANT_TERMS),
    re.IGNORECASE,
)


def _is_relevant_pathway_cart(term: str) -> bool:
    """Return False for pathways implausible in CAR-T PBMC context."""
    return _CART_IRRELEVANT_RE.search(str(term)) is None


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

    gsea_libraries = [
        "MSigDB_Hallmark_2020",
        "KEGG_2021_Human",
        "Reactome_2022",
        "GO_Biological_Process_2023",
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
            print(f"  GSEA total: {len(gsea_results)} pathways, "
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
# Panel A -- GSEA Enrichment Bar Chart
# ======================================================================

def panel_A(ax, data: dict):
    """GSEA pathway enrichment bar chart (CAR-T Pre→Post)."""
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

    # Filter irrelevant pathways
    if term_col is not None:
        n_before = len(df)
        df = df[df[term_col].apply(_is_relevant_pathway_cart)]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  Panel A: dropped {n_dropped} irrelevant pathways")

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

    df_selected["pathway"] = df_selected[term_col].apply(_clean_pathway_name)

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
    ax.set_title("GSEA Pathway Enrichment — CAR-T Pre→Post", fontsize=11)

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
# Panel B -- Leading-edge gene overlap heatmap
# ======================================================================

def panel_B(ax, data: dict):
    """Leading-edge gene overlap heatmap (CAR-T)."""
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

    if term_col is not None:
        df = df[df[term_col].apply(_is_relevant_pathway_cart)]

    # Select top 8 pathways by |NES| (agnostic, no hand-picking)
    if fdr_col is not None:
        sig_df = df[df[fdr_col] < 0.25]
    else:
        sig_df = df
    if len(sig_df) < 4:
        sig_df = df

    selected = sig_df.assign(
        _abs=sig_df[nes_col].abs(),
    ).nlargest(8, "_abs").drop(columns="_abs")
    selected = selected.sort_values(nes_col, ascending=False)

    pathway_genes: dict[str, set[str]] = {}
    all_genes: set[str] = set()
    _seen_names: set[str] = set()
    for _, row in selected.iterrows():
        pname = _clean_pathway_name(str(row[term_col]), max_len=30)
        if pname in _seen_names:
            lib = str(row.get("library", ""))
            pname = f"{pname} [{lib[:8]}]" if lib else f"{pname} (2)"
        _seen_names.add(pname)
        genes_str = str(row[lead_col])
        genes = [g.strip() for g in genes_str.replace(";", ",").split(",")
                 if g.strip()]
        genes = [g for g in genes if _is_likely_protein_coding(g)]
        pathway_genes[pname] = set(genes)
        all_genes.update(genes)

    if not all_genes or not pathway_genes:
        ax.text(0.5, 0.5, "No leading-edge genes found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    gene_counts = {}
    for genes in pathway_genes.values():
        for g in genes:
            gene_counts[g] = gene_counts.get(g, 0) + 1

    n_pw = len(pathway_genes)

    # Prefer genes that DISCRIMINATE between pathways (present in some,
    # absent in others) over genes shared by all.  Score each gene by
    # how close its count is to n_pw/2 (maximum discrimination).
    def _discrim_score(gene: str) -> float:
        c = gene_counts[gene]
        # Must appear in ≥2 pathways to be "shared"
        if c < 2:
            return -999
        # Maximise discrimination: distance from n/2 is BAD
        return -abs(c - n_pw / 2)

    shared_genes = sorted(
        [g for g, c in gene_counts.items() if c >= 2],
        key=lambda g: (_discrim_score(g), -gene_counts[g]),
        reverse=True,
    )
    if len(shared_genes) < 5:
        # Fallback: just take the most frequent genes
        shared_genes = sorted(gene_counts.keys(),
                              key=lambda g: -gene_counts[g])[:25]
    shared_genes = shared_genes[:25]

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

    from matplotlib.colors import ListedColormap
    binary_cmap = ListedColormap(["#FFF8DC", "#8B0000"])
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=shared_genes, yticklabels=pathways,
        cmap=binary_cmap, vmin=0, vmax=1,
        cbar=False, linewidths=0.5, linecolor="white", square=False,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                       fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title("Leading-Edge Gene Overlap (CAR-T)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")

    legend_handles = [
        mpatches.Patch(facecolor="#8B0000", edgecolor="grey",
                       label="In leading edge"),
        mpatches.Patch(facecolor="#FFF8DC", edgecolor="grey",
                       label="Absent"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9)


# ======================================================================
# Panel C -- Signature within-arm forest plot
# ======================================================================

def panel_C(ax, data: dict):
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
# Panel D -- Gene-level volcano plot
# ======================================================================

def panel_D(ax, data: dict):
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

    try:
        from adjustText import adjust_text
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
