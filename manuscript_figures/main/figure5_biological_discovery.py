"""
Figure 5 -- Biological Discovery
=================================

Four-panel figure (2x2) combining GSEA pathway enrichment, leading-edge
gene analysis, cell-type abundance changes, and gene-level volcano plots.

Panels
------
A  GSEA enrichment heatmap (Hallmark pathways).
B  Leading-edge gene overlap / top-pathway summary.
C  Cell-type abundance changes (Responders vs Non-responders, Pre vs Post).
D  Gene-level volcano plot from within-arm CAR-T analysis.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .._shared import *  # noqa: F401,F403

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "Figure5_biological_discovery"
FIGSIZE = (18, 14)


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_data() -> dict:
    """Load datasets, run DiD, GSEA, within-arm gene analysis, and compute
    cell-type proportions.
    """
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

    # Signature-level DiD
    did_sig = did_table(
        adata,
        features=sig_cols,
        design=design,
        visits=visits,
        layer="log1p_tpm",
        standardize=True,
        aggregate="participant_visit",
    )

    # GSEA on Hallmark gene sets
    gsea_results = None
    try:
        gsea_results = run_gsea_did(
            adata,
            gene_sets="MSigDB_Hallmark_2020",
            design=design,
            visits=visits,
            layer="log1p_tpm",
            min_size=10,
            max_size=500,
            permutation_num=1000,
            outdir=None,
            no_plot=True,
        )
        if isinstance(gsea_results, pd.DataFrame) and len(gsea_results) > 0:
            print(f"  GSEA: {len(gsea_results)} pathways tested")
        else:
            gsea_results = None
    except Exception as exc:
        print(f"  GSEA unavailable: {exc}")
        gsea_results = None

    # ------------------------------------------------------------------
    # 2. DiD signature ranking for panel C (dot-plot style)
    # ------------------------------------------------------------------
    # (Cell-type proportions are not informative for Sade-Feldman
    #  since all cells are annotated as "Immune" without subtypes.
    #  We use the DiD results as a ranked effect-size dot plot instead.)
    ct_props = None
    celltype_col = None

    # ------------------------------------------------------------------
    # 3. Sade-Feldman gene-level DiD (top variable genes)
    # ------------------------------------------------------------------
    gene_results = None
    try:
        import scanpy as sc

        # Select top variable genes from Sade-Feldman
        adata_genes = adata.copy()
        sc.pp.highly_variable_genes(
            adata_genes, n_top_genes=2000, layer="log1p_tpm", flavor="seurat",
        )
        top_genes = adata_genes.var_names[
            adata_genes.var["highly_variable"]
        ].tolist()
        print(f"  Sade-Feldman: {len(top_genes)} variable genes selected")

        gene_results = did_table(
            adata_genes,
            features=top_genes,
            design=design,
            visits=visits,
            layer="log1p_tpm",
            standardize=True,
            aggregate="participant_visit",
        )

        # Normalize column names for panel_D
        rename_map = {}
        if "beta_DiD" not in gene_results.columns and "beta_time" in gene_results.columns:
            rename_map["beta_time"] = "beta_DiD"
        if "FDR_DiD" in gene_results.columns:
            rename_map["FDR_DiD"] = "fdr"
        if "p_DiD" in gene_results.columns:
            rename_map["p_DiD"] = "p_value"
        if rename_map:
            gene_results = gene_results.rename(columns=rename_map)

        n_sig = (gene_results["fdr"] < 0.1).sum() if "fdr" in gene_results.columns else 0
        print(f"  Gene-level results: {len(gene_results)} genes, {n_sig} FDR<0.1")

        del adata_genes
        gc.collect()

    except Exception as exc:
        print(f"  Gene-level analysis unavailable: {exc}")
        import traceback; traceback.print_exc()
        gene_results = None

    return dict(
        adata=adata,
        sig_cols=sig_cols,
        design=design,
        visits=visits,
        did_sig=did_sig,
        gsea_results=gsea_results,
        ct_props=ct_props,
        celltype_col=celltype_col,
        gene_results=gene_results,
    )


# ======================================================================
# Panel A -- GSEA Enrichment Heatmap
# ======================================================================

def panel_A(ax, data: dict):
    """GSEA Hallmark pathway enrichment bar chart."""
    gsea_results = data["gsea_results"]

    if gsea_results is None or len(gsea_results) == 0:
        # Fallback: show signature DiD waterfall
        _panel_A_signature_waterfall(ax, data)
        return

    df = gsea_results.copy()

    # Identify columns — exact match first, then fuzzy
    nes_col = fdr_col = term_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "nes":
            nes_col = c
        elif cl == "fdr q-val" or cl == "fdr":
            fdr_col = c
        elif cl == "term":
            term_col = c

    # Fuzzy fallback (but avoid Lead_genes matching "nes")
    if nes_col is None:
        for c in df.columns:
            if c.lower() in ("nes", "normalized_enrichment_score"):
                nes_col = c
                break
    if fdr_col is None:
        for c in df.columns:
            if "fdr" in c.lower():
                fdr_col = c
                break
    if term_col is None:
        for c in df.columns:
            cl = c.lower()
            if cl == "name" or cl == "pathway":
                term_col = c
                break
        if term_col is None:
            term_col = df.columns[0]

    if nes_col is None:
        _panel_A_signature_waterfall(ax, data)
        return

    # CRITICAL: convert NES to numeric (gseapy sometimes returns object dtype)
    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")

    df = df.dropna(subset=[nes_col])

    # Top 15 pathways by absolute NES
    df["abs_nes"] = df[nes_col].abs()
    df = df.sort_values("abs_nes", ascending=False).head(15)
    df = df.sort_values(nes_col, ascending=True)

    # Clean pathway names
    df["pathway"] = (
        df[term_col]
        .str.replace("_", " ")
        .str.title()
        .apply(lambda s: s[:38] + "…" if len(str(s)) > 40 else s)
    )

    # Color by direction — strong, vivid colors for significant; muted for n.s.
    colors = []
    for _, row in df.iterrows():
        sig = fdr_col is not None and row[fdr_col] < 0.25
        if row[nes_col] > 0:
            colors.append("#C0392B" if sig else "#E6B0AA")  # vivid red / muted red
        else:
            colors.append("#2471A3" if sig else "#AED6F1")  # vivid blue / muted blue

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df[nes_col].values, color=colors, alpha=0.9,
            edgecolor="white", linewidth=0.5, height=0.7)

    # Significance stars
    if fdr_col is not None:
        for i, (_, row) in enumerate(df.iterrows()):
            fdr_val = row[fdr_col]
            if pd.notna(fdr_val) and fdr_val < 0.25:
                star = "***" if fdr_val < 0.001 else "**" if fdr_val < 0.01 else "*" if fdr_val < 0.05 else "†"
                x_pos = row[nes_col]
                ha = "left" if x_pos > 0 else "right"
                offset = 0.05 if x_pos > 0 else -0.05
                ax.text(x_pos + offset, i, star, ha=ha, va="center",
                        fontsize=9, fontweight="bold", color="black")

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["pathway"].values, fontsize=8)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title("GSEA Hallmark Pathway Enrichment", fontsize=11)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#C0392B", alpha=0.9, label="Up (FDR < 0.25)"),
        mpatches.Patch(color="#E6B0AA", alpha=0.9, label="Up (n.s.)"),
        mpatches.Patch(color="#2471A3", alpha=0.9, label="Down (FDR < 0.25)"),
        mpatches.Patch(color="#AED6F1", alpha=0.9, label="Down (n.s.)"),
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
# Panel B -- Leading-edge / pathway summary
# ======================================================================

def panel_B(ax, data: dict):
    """GSEA dot plot: top enriched pathways (up and down) with dot size by gene-set overlap."""
    gsea_results = data["gsea_results"]

    if gsea_results is None or len(gsea_results) == 0:
        _panel_B_did_summary(ax, data)
        return

    df = gsea_results.copy()

    # Identify columns — exact match to avoid Lead_genes matching "nes"
    nes_col = fdr_col = term_col = tag_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "nes":
            nes_col = c
        elif cl == "fdr q-val" or cl == "fdr":
            fdr_col = c
        elif cl == "term":
            term_col = c
        elif cl.startswith("tag"):
            tag_col = c

    # Fuzzy fallback
    if nes_col is None:
        for c in df.columns:
            if c.lower() in ("nes", "normalized_enrichment_score"):
                nes_col = c
                break
    if term_col is None:
        for c in df.columns:
            if c.lower() in ("name", "pathway"):
                term_col = c
                break

    if nes_col is None or term_col is None:
        _panel_B_did_summary(ax, data)
        return

    # Convert to numeric
    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Top 5 up and top 5 down (by NES, excluding near-zero)
    df_pos = df[df[nes_col] > 0.1]
    df_neg = df[df[nes_col] < -0.1]
    df_up = df_pos.nlargest(min(5, len(df_pos)), nes_col) if len(df_pos) else pd.DataFrame()
    df_down = df_neg.nsmallest(min(5, len(df_neg)), nes_col) if len(df_neg) else pd.DataFrame()
    selected = pd.concat([df_up, df_down]).drop_duplicates()

    if len(selected) == 0:
        # Fallback: just take top 10 by absolute NES
        selected = df.assign(_abs=df[nes_col].abs()).nlargest(10, "_abs").drop(columns="_abs")

    selected = selected.sort_values(nes_col, ascending=True)

    # Clean names
    selected["pathway"] = (
        selected[term_col]
        .str.replace("_", " ")
        .str.title()
        .apply(lambda s: s[:35] + "…" if len(str(s)) > 37 else s)
    )

    # Extract gene set overlap sizes from Tag % (format: "12/45")
    sizes = np.full(len(selected), 100.0)
    if tag_col is not None:
        for i, (_, row) in enumerate(selected.iterrows()):
            try:
                parts = str(row[tag_col]).split("/")
                sizes[i] = int(parts[1]) if len(parts) == 2 else 100
            except (ValueError, IndexError):
                sizes[i] = 100
        sizes = np.clip(sizes / 2, 30, 250)

    y_pos = np.arange(len(selected))

    # Color by -log10(FDR)
    if fdr_col is not None and fdr_col in selected.columns:
        fdr_vals = pd.to_numeric(selected[fdr_col], errors="coerce").clip(lower=1e-10).values
        nlog_fdr = -np.log10(fdr_vals)
    else:
        nlog_fdr = np.ones(len(selected))

    vmax_val = max(3.0, float(np.nanmax(nlog_fdr))) if len(nlog_fdr) > 0 else 3.0

    scatter = ax.scatter(
        selected[nes_col].values, y_pos,
        s=sizes, c=nlog_fdr, cmap="RdYlBu_r",
        edgecolor="black", linewidth=0.6, alpha=0.9,
        vmin=0, vmax=vmax_val,
        zorder=3,
    )

    ax.axvline(0, color="black", lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected["pathway"].values, fontsize=8)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title("Top Enriched Pathways (Up & Down)", fontsize=11)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("−log₁₀(FDR)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    despine(ax)


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
# Panel C -- Cell-type abundance changes
# ======================================================================

def panel_C(ax, data: dict):
    """DiD effect size dot plot with FDR annotation."""
    did_sig = data["did_sig"]

    df = did_sig.copy()
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("beta_DiD", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    ci_lo = df["beta_DiD"] - 1.96 * df["se_DiD"]
    ci_hi = df["beta_DiD"] + 1.96 * df["se_DiD"]

    # Significance colouring
    sig_mask = df["FDR_DiD"] < 0.1

    # Non-significant
    if (~sig_mask).any():
        ns = df.index[~sig_mask]
        ax.hlines(y[ns], ci_lo.iloc[ns], ci_hi.iloc[ns],
                  color=COLORS["gray"], lw=1.5, zorder=1)
        ax.scatter(df.loc[ns, "beta_DiD"], y[ns],
                   color=COLORS["gray"], s=40, zorder=2,
                   edgecolors="white", linewidths=0.5)

    # Significant
    if sig_mask.any():
        s = df.index[sig_mask]
        for i in s:
            clr = COLORS["treated"] if df.loc[i, "beta_DiD"] > 0 else COLORS["control"]
            ax.hlines(y[i], ci_lo.iloc[i], ci_hi.iloc[i],
                      color=clr, lw=2, zorder=1)
            ax.scatter(df.loc[i, "beta_DiD"], y[i],
                       color=clr, s=55, zorder=2,
                       edgecolors="white", linewidths=0.5)

    ax.axvline(0, color="black", ls="--", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display"].values, fontsize=9)
    ax.set_xlabel(r"DiD coefficient ($\beta$, standardised)")
    ax.set_title("Signature DiD Effects (95% CI)", fontsize=11)

    n_sig = sig_mask.sum()
    ax.text(0.97, 0.03, f"{n_sig}/{len(df)} FDR < 0.1",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic", color=COLORS["gray"])
    despine(ax)


# ======================================================================
# Panel D -- Gene-level volcano plot
# ======================================================================

def panel_D(ax, data: dict):
    """Volcano plot of gene-level DiD effects (Sade-Feldman)."""
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

    # Determine significance column
    if "fdr" in df.columns:
        sig_col = "fdr"
    elif "FDR_DiD" in df.columns:
        sig_col = "FDR_DiD"
    else:
        sig_col = "p_value"

    # Determine beta column
    if "beta_DiD" in df.columns:
        beta_col = "beta_DiD"
    elif "beta_time" in df.columns:
        beta_col = "beta_time"
    else:
        ax.text(0.5, 0.5, "No beta column found",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.axis("off")
        return

    df = df.dropna(subset=[beta_col, sig_col])
    df["nlog10"] = -np.log10(df[sig_col].clip(lower=1e-300))

    # Determine significance threshold
    n_fdr_sig = (df[sig_col] < 0.1).sum()
    if n_fdr_sig >= 3 and sig_col == "fdr":
        threshold = 0.1
        thresh_label = "FDR < 0.1"
    else:
        # Fall back to nominal p-value
        if "p_value" in df.columns:
            sig_col = "p_value"
            df["nlog10"] = -np.log10(df[sig_col].clip(lower=1e-300))
        threshold = 0.05
        thresh_label = "p < 0.05"

    # Classify genes
    df["category"] = "ns"
    sig_mask = df[sig_col] < threshold
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

    # Label top genes — more labels, use adjustText to avoid overlaps
    texts = []
    for direction, n_label in [("up", 8), ("down", 8)]:
        sub = df[df["category"] == direction]
        if len(sub) == 0:
            continue
        if direction == "up":
            top = sub.nlargest(n_label, beta_col)
        else:
            top = sub.nsmallest(n_label, beta_col)

        for _, row in top.iterrows():
            t = ax.text(
                row[beta_col], row["nlog10"], row["feature"],
                fontsize=6.5, fontweight="bold",
                color=color_map[direction],
            )
            texts.append(t)

    # Attempt adjustText for non-overlapping labels
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=COLORS["gray"], lw=0.4))
    except ImportError:
        pass  # fall back to raw placement

    # Threshold line
    thresh_y = -np.log10(threshold)
    ax.axhline(thresh_y, color=COLORS["gray"], ls="--", lw=0.8, zorder=0)
    ax.axvline(0, color="black", lw=0.6, zorder=0)

    ax.set_xlabel(r"Effect size ($\beta$)")
    ax.set_ylabel(r"$-\log_{10}$(" + ("FDR" if "fdr" in sig_col.lower() else "p") + ")")
    ax.set_title("Gene-Level Volcano (Sade-Feldman DiD)", fontsize=11)

    # Summary annotation
    n_up = (df["category"] == "up").sum()
    n_down = (df["category"] == "down").sum()
    ax.text(
        0.97, 0.97,
        f"{thresh_label}:\n"
        f"  {n_up} up, {n_down} down\n"
        f"  {len(df)} total genes",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.8),
        family="monospace",
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.8, label="Upregulated"),
        mpatches.Patch(color=COLORS["control"], alpha=0.8, label="Downregulated"),
        mpatches.Patch(color=COLORS["gray"], alpha=0.3, label="Not significant"),
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
