"""
Alternative Panel B implementations for Figures 5 and 5v2.
==========================================================

Five options for the B panel in the Biological Discovery figures:

B1  Signature score ridgeplot (Pre vs Post distributions)
B2  Gene-level MA plot (mean expression vs fold change)
B3  GSEA running enrichment score plot (mountain plot)
B4  Pathway–pathway network (shared leading-edge genes)
B5  Ranked gene waterfall (top genes by t-statistic / effect size)

Each function takes (ax, data) where data is the dict returned by
``_prepare_data()`` from figure5 or figure5v2.

Parameters
----------
mode : str
    "did" for Sade-Feldman two-arm DiD (beta_DiD, p_DiD columns)
    "within" for CAR-T single-arm within-arm (beta_time, p_time columns)
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .._shared import (
    COLORS,
    despine,
    sig_display,
)
from .figure5_biological_discovery import (
    _is_likely_protein_coding,
    _detect_gsea_columns,
    _clean_pathway_name,
)


# ======================================================================
# B1 — Signature score ridgeplot (Pre vs Post)
# ======================================================================

def panel_B1(ax, data: dict, *, mode: str = "did"):
    """Signature score ridgeplot showing Pre vs Post distributions.

    Shows the distribution of participant-level mean signature scores
    for Pre and Post visits, revealing the magnitude and consistency
    of treatment-induced shifts across signatures.
    """
    adata = data["adata"]
    sig_cols = data["sig_cols"]

    # Aggregate to participant-visit level
    obs = adata.obs.copy()
    visit_col = data["design"].visit_col
    pid_col = data["design"].participant_col

    # Build participant-visit means for each signature
    rows = []
    for pid in obs[pid_col].unique():
        for visit in obs[visit_col].unique():
            mask = (obs[pid_col] == pid) & (obs[visit_col] == visit)
            if mask.sum() == 0:
                continue
            row = {"participant": pid, "visit": visit}
            for sc_ in sig_cols:
                row[sc_] = obs.loc[mask, sc_].mean()
            rows.append(row)
    agg = pd.DataFrame(rows)

    # Melt to long format
    melted = agg.melt(
        id_vars=["participant", "visit"],
        value_vars=sig_cols,
        var_name="signature",
        value_name="score",
    )
    melted["display"] = melted["signature"].apply(sig_display)

    # Sort signatures by mean Pre→Post change (largest shift on top)
    shift = (
        melted.groupby(["display", "visit"])["score"]
        .mean()
        .unstack(fill_value=0)
    )
    if "Post" in shift.columns and "Pre" in shift.columns:
        shift["delta"] = shift["Post"] - shift["Pre"]
        order = shift.sort_values("delta", ascending=True).index.tolist()
    else:
        order = sorted(melted["display"].unique())

    # Plot paired violins
    n_sigs = len(order)
    y_positions = np.arange(n_sigs)

    for i, sig_name in enumerate(order):
        sub = melted[melted["display"] == sig_name]
        for visit, color, offset in [
            ("Pre", COLORS["control"], -0.15),
            ("Post", COLORS["treated"], 0.15),
        ]:
            vals = sub[sub["visit"] == visit]["score"].dropna().values
            if len(vals) < 2:
                continue
            # Horizontal violin: use boxplot-style
            parts = ax.violinplot(
                vals,
                positions=[i + offset],
                vert=False,
                widths=0.25,
                showmeans=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
                pc.set_edgecolor("none")
            parts["cmeans"].set_color(color)
            parts["cmeans"].set_linewidth(1.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Signature Score (participant mean)")

    title_suffix = "Sade-Feldman" if mode == "did" else "CAR-T"
    ax.set_title(f"Signature Distributions Pre vs Post ({title_suffix})",
                 fontsize=11)

    legend_handles = [
        mpatches.Patch(color=COLORS["control"], alpha=0.6, label="Pre"),
        mpatches.Patch(color=COLORS["treated"], alpha=0.6, label="Post"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# B2 — Gene-level MA plot
# ======================================================================

def panel_B2(ax, data: dict, *, mode: str = "did"):
    """MA plot: mean expression (x) vs effect size (y).

    Shows the relationship between baseline expression level and
    the magnitude of treatment-induced change for each gene.
    """
    gene_results = data.get("gene_results")
    adata = data["adata"]

    if gene_results is None or len(gene_results) == 0:
        ax.text(0.5, 0.5, "Gene-level results unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df = gene_results.copy()
    beta_col = "beta_DiD" if mode == "did" else "beta_time"
    p_col = "p_DiD" if mode == "did" else "p_time"

    # Compute mean expression per gene across all cells
    # Use the appropriate layer
    layer_name = None
    for candidate in ("log1p_tpm", "log1p_norm", "log1p_cpm"):
        if candidate in adata.layers:
            layer_name = candidate
            break

    if layer_name:
        import scipy.sparse as sp
        X = adata[:, df["feature"].values].layers[layer_name]
        if sp.issparse(X):
            mean_expr = np.asarray(X.mean(axis=0)).ravel()
        else:
            mean_expr = X.mean(axis=0).ravel()
    else:
        mean_expr = np.zeros(len(df))

    df["mean_expr"] = mean_expr
    df = df.dropna(subset=[beta_col, p_col])

    p_thresh = 0.05
    sig_mask = df[p_col] < p_thresh

    # Non-significant
    ns = df[~sig_mask]
    ax.scatter(ns["mean_expr"], ns[beta_col],
               c=COLORS["gray"], alpha=0.2, s=6, edgecolors="none",
               rasterized=True)

    # Significant up
    up = df[sig_mask & (df[beta_col] > 0)]
    ax.scatter(up["mean_expr"], up[beta_col],
               c=COLORS["treated"], alpha=0.7, s=15, edgecolors="none",
               rasterized=True)

    # Significant down
    dn = df[sig_mask & (df[beta_col] < 0)]
    ax.scatter(dn["mean_expr"], dn[beta_col],
               c=COLORS["control"], alpha=0.7, s=15, edgecolors="none",
               rasterized=True)

    ax.axhline(0, color="black", lw=0.6, zorder=0)

    # Label top genes
    for sub_df, color in [(up, COLORS["treated"]), (dn, COLORS["control"])]:
        if len(sub_df) == 0:
            continue
        pc = sub_df[sub_df["feature"].apply(_is_likely_protein_coding)]
        top = pc.nsmallest(min(5, len(pc)), p_col)
        for _, row in top.iterrows():
            ax.annotate(
                row["feature"],
                (row["mean_expr"], row[beta_col]),
                fontsize=7, fontweight="bold", color=color,
                xytext=(5, 5), textcoords="offset points",
            )

    ax.set_xlabel("Mean expression (log1p)")
    beta_label = r"$\beta_{\mathrm{DiD}}$" if mode == "did" else r"$\beta_{\mathrm{time}}$"
    ax.set_ylabel(f"Effect size ({beta_label})")

    title_suffix = "Sade-Feldman DiD" if mode == "did" else "CAR-T Pre→Post"
    ax.set_title(f"MA Plot — {title_suffix}", fontsize=11)

    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.7,
                       label="Post ↑" if mode == "within" else "Responder ↑"),
        mpatches.Patch(color=COLORS["control"], alpha=0.7,
                       label="Pre ↑" if mode == "within" else "Non-responder ↑"),
        mpatches.Patch(color=COLORS["gray"], alpha=0.3,
                       label="Not significant"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# B3 — GSEA running enrichment score (mountain plot)
# ======================================================================

def panel_B3(ax, data: dict, *, mode: str = "did"):
    """GSEA running enrichment score for top 2 pathways (up + down).

    Shows the classic 'mountain plot' for the single most enriched
    pathway in each direction, illustrating where the leading edge
    falls in the ranked gene list.
    """
    gsea_results = data.get("gsea_results")

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
    df = df.dropna(subset=[nes_col])

    # Pick top up and top down pathway
    top_up = df[df[nes_col] > 0].nlargest(1, nes_col)
    top_dn = df[df[nes_col] < 0].nsmallest(1, nes_col)

    # Check if we have the ranking data to build the running ES
    # We need the gene-level results for the ranked list
    gene_results = data.get("gene_results")
    if gene_results is None:
        ax.text(0.5, 0.5, "Gene ranking unavailable for ES plot",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    beta_col = "beta_DiD" if mode == "did" else "beta_time"
    se_col = "se_DiD" if mode == "did" else "se_time"

    # Build full ranking from gene results (t-statistic)
    gr = gene_results.copy()
    gr["tstat"] = gr[beta_col] / gr[se_col]
    gr = gr.dropna(subset=["tstat"]).sort_values("tstat", ascending=False)
    ranked_genes = gr["feature"].tolist()

    lead_col = cols["lead"]
    if lead_col is None or lead_col not in df.columns:
        ax.text(0.5, 0.5, "Leading-edge data unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    # Compute running enrichment score for selected pathways
    for pathway_df, color, label_prefix in [
        (top_up, COLORS["treated"], "Post ↑" if mode == "within" else "Resp ↑"),
        (top_dn, COLORS["control"], "Pre ↑" if mode == "within" else "Non-resp ↑"),
    ]:
        if len(pathway_df) == 0:
            continue
        row = pathway_df.iloc[0]
        genes_str = str(row[lead_col])
        pathway_genes = set(
            g.strip() for g in genes_str.replace(";", ",").split(",")
            if g.strip()
        )

        # Running ES computation
        n_genes = len(ranked_genes)
        n_hits = sum(1 for g in ranked_genes if g in pathway_genes)
        if n_hits == 0:
            continue

        p_hit = 1.0 / n_hits
        p_miss = 1.0 / (n_genes - n_hits) if n_genes > n_hits else 0

        running_es = []
        es = 0.0
        for gene in ranked_genes:
            if gene in pathway_genes:
                es += p_hit
            else:
                es -= p_miss
            running_es.append(es)

        x = np.arange(len(running_es)) / len(running_es)
        pname = _clean_pathway_name(str(row[term_col]), max_len=35)
        ax.plot(x, running_es, color=color, lw=1.5,
                label=f"{label_prefix}: {pname}")

    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Gene rank (by t-statistic)")
    ax.set_ylabel("Running Enrichment Score")

    title_suffix = "Sade-Feldman" if mode == "did" else "CAR-T"
    ax.set_title(f"GSEA Running ES — {title_suffix}", fontsize=11)
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# B4 — Pathway–pathway network
# ======================================================================

def panel_B4(ax, data: dict, *, mode: str = "did"):
    """Pathway–pathway network connected by shared leading-edge genes.

    Nodes = top enriched pathways, edges = Jaccard similarity of
    leading-edge gene sets.  Shows functional clustering without
    the heatmap redundancy problem.
    """
    gsea_results = data.get("gsea_results")

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
        ax.text(0.5, 0.5, "Required GSEA columns unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Select top 12 by |NES|
    top = df.assign(_abs=df[nes_col].abs()).nlargest(12, "_abs").drop(columns="_abs")

    # Parse leading-edge genes
    pathway_names = []
    pathway_genes_list = []
    pathway_nes = []
    for _, row in top.iterrows():
        pname = _clean_pathway_name(str(row[term_col]), max_len=25)
        # Disambiguate
        count = pathway_names.count(pname)
        if count > 0:
            pname = f"{pname} ({count + 1})"
        pathway_names.append(pname)
        genes = set(
            g.strip() for g in str(row[lead_col]).replace(";", ",").split(",")
            if g.strip()
        )
        pathway_genes_list.append(genes)
        pathway_nes.append(float(row[nes_col]))

    n = len(pathway_names)
    if n < 3:
        ax.text(0.5, 0.5, "Too few pathways for network",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    # Compute Jaccard similarity matrix
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            union = len(pathway_genes_list[i] | pathway_genes_list[j])
            if union > 0:
                jaccard[i, j] = jaccard[j, i] = (
                    len(pathway_genes_list[i] & pathway_genes_list[j]) / union
                )

    # Simple force-directed layout using spring embedding
    # Seed with circular layout, then iterate
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)])

    # Spring iteration
    for _ in range(200):
        forces = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[j] - pos[i]
                dist = np.linalg.norm(diff) + 1e-6
                # Repulsion (all pairs)
                repulsion = -0.05 / (dist ** 2) * (diff / dist)
                forces[i] += repulsion
                forces[j] -= repulsion
                # Attraction (connected pairs)
                if jaccard[i, j] > 0.05:
                    attraction = 0.3 * jaccard[i, j] * diff
                    forces[i] += attraction
                    forces[j] -= attraction
        pos += forces * 0.1
        # Center
        pos -= pos.mean(axis=0)

    # Normalize positions to [0, 1]
    pos -= pos.min(axis=0)
    rng = pos.max(axis=0) - pos.min(axis=0)
    rng[rng == 0] = 1
    pos /= rng

    # Draw edges
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard[i, j] > 0.05:
                lw = jaccard[i, j] * 4
                ax.plot(
                    [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color="#CCCCCC", lw=lw, zorder=0,
                )

    # Draw nodes
    node_colors = [
        COLORS["treated"] if nes > 0 else COLORS["control"]
        for nes in pathway_nes
    ]
    node_sizes = [abs(nes) * 120 + 80 for nes in pathway_nes]
    ax.scatter(pos[:, 0], pos[:, 1], c=node_colors, s=node_sizes,
               edgecolors="white", linewidths=1.0, zorder=2)

    # Labels
    for i, name in enumerate(pathway_names):
        ax.annotate(
            name, (pos[i, 0], pos[i, 1]),
            fontsize=6, ha="center", va="bottom",
            xytext=(0, 8), textcoords="offset points",
        )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.15)
    ax.axis("off")

    title_suffix = "Sade-Feldman" if mode == "did" else "CAR-T"
    ax.set_title(f"Pathway Network — {title_suffix}", fontsize=11)

    up_label = "Responder ↑" if mode == "did" else "Post ↑"
    dn_label = "Non-responder ↑" if mode == "did" else "Pre ↑"
    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], label=up_label),
        mpatches.Patch(color=COLORS["control"], label=dn_label),
        plt.Line2D([0], [0], color="#CCCCCC", lw=2,
                   label="Shared genes (Jaccard)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower left",
              frameon=True, framealpha=0.9)


# ======================================================================
# B5 — Ranked gene waterfall
# ======================================================================

def panel_B5(ax, data: dict, *, mode: str = "did"):
    """Top 30 genes ranked by effect size, colored by significance.

    Horizontal bar plot of the most extreme genes on each side,
    providing immediate biological interpretability.
    """
    gene_results = data.get("gene_results")

    if gene_results is None or len(gene_results) == 0:
        ax.text(0.5, 0.5, "Gene-level results unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    df = gene_results.copy()
    beta_col = "beta_DiD" if mode == "did" else "beta_time"
    p_col = "p_DiD" if mode == "did" else "p_time"

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
            colors.append(COLORS["treated"] if sig else COLORS["treated"] + "55")
        else:
            colors.append(COLORS["control"] if sig else COLORS["control"] + "55")

    ax.barh(y_pos, selected[beta_col].values, color=colors, alpha=0.9,
            edgecolor="white", linewidth=0.3, height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected["feature"].values, fontsize=7)

    beta_label = r"$\beta_{\mathrm{DiD}}$" if mode == "did" else r"$\beta_{\mathrm{time}}$"
    ax.set_xlabel(f"Effect size ({beta_label})")

    title_suffix = "Sade-Feldman DiD" if mode == "did" else "CAR-T Pre→Post"
    ax.set_title(f"Top Genes by Effect Size — {title_suffix}", fontsize=11)

    up_label = "Responder ↑" if mode == "did" else "Post ↑"
    dn_label = "Non-responder ↑" if mode == "did" else "Pre ↑"
    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.9,
                       label=f"{up_label} (p < 0.05)"),
        mpatches.Patch(color=COLORS["treated"], alpha=0.35,
                       label=f"{up_label} (n.s.)"),
        mpatches.Patch(color=COLORS["control"], alpha=0.9,
                       label=f"{dn_label} (p < 0.05)"),
        mpatches.Patch(color=COLORS["control"], alpha=0.35,
                       label=f"{dn_label} (n.s.)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)
