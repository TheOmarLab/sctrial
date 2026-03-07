"""
Supplementary Figure 7 — Biological Discovery Details
======================================================

Four-panel (2x2) figure expanding on cell-type-specific treatment effects
and gene-level analyses from the CAR-T within-arm comparison.

Panels
------
A  Exhaustion signature forest plot across cell types.
B  Effect heterogeneity (SD) across cell types per signature.
C  Top differentially affected genes (horizontal bar chart).
D  Gene-level effect distribution histogram.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    TrialDesign,
    add_log1p_cpm_layer,
    apply_style,
    clear_cache,
    despine,
    load_clinical_trial_dataset,
    save_panel,
    score_clinical_signatures,
    sig_display,
    within_arm_comparison,
)

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "SuppFig7_biological_details"
FIGSIZE = (16, 12)


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_data() -> dict:
    """Load CAR-T dataset, run cell-type-specific and gene-level analyses."""

    adata = load_clinical_trial_dataset("cart")

    # ── Ensure log1p layer ────────────────────────────────────────────
    add_log1p_cpm_layer(adata)
    layer = "log1p_cpm" if "log1p_cpm" in adata.layers else None
    if layer is None:
        for candidate in ("log1p", "counts"):
            if candidate in adata.layers:
                layer = candidate
                break
    if layer is None:
        import scipy.sparse as sp
        X = adata.X
        adata.layers["log1p"] = np.log1p(
            X.toarray() if sp.issparse(X) else np.array(X)
        )
        layer = "log1p"

    # ── Score signatures ──────────────────────────────────────────────
    adata, sig_cols = score_clinical_signatures(adata, layer=layer)
    print(f"  Scored {len(sig_cols)} clinical signatures")

    # ── Detect columns ────────────────────────────────────────────────
    pid_col = (
        "patient_id" if "patient_id" in adata.obs.columns
        else "participant_id" if "participant_id" in adata.obs.columns
        else None
    )
    visit_col = (
        "visit" if "visit" in adata.obs.columns
        else "timepoint" if "timepoint" in adata.obs.columns
        else None
    )
    celltype_col = None
    for c in ("cell_type", "celltype", "cluster", "CellType"):
        if c in adata.obs.columns:
            celltype_col = c
            break

    if pid_col is None or visit_col is None:
        raise RuntimeError("Cannot identify participant/visit columns in CAR-T data")

    # Ensure arm column
    adata.obs["arm"] = "Treated"
    design = TrialDesign(
        participant_col=pid_col,
        visit_col=visit_col,
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )

    visit_vals = list(adata.obs[visit_col].unique())
    # Sort numerically if labels contain digits, otherwise keep original order
    import re as _re
    has_digits = all(_re.search(r"\d", str(v)) for v in visit_vals)
    if has_digits:
        visit_vals.sort(key=lambda v: int(_re.findall(r"\d+", str(v))[0]))
    visits = (visit_vals[0], visit_vals[-1]) if len(visit_vals) >= 2 else ("Pre", "Post")

    # ------------------------------------------------------------------
    # Panels A & B: Cell-type-specific within-arm comparison
    # ------------------------------------------------------------------
    ct_results = []
    if celltype_col is not None:
        ct_counts = adata.obs[celltype_col].value_counts()
        major_cts = ct_counts[ct_counts >= 500].index.tolist()
        print(f"  Major cell types (>=500 cells): {len(major_cts)}")

        for ct in major_cts:
            mask = adata.obs[celltype_col] == ct
            ad_ct = adata[mask].copy()
            try:
                res = within_arm_comparison(
                    ad_ct,
                    arm="Treated",
                    features=sig_cols,
                    design=design,
                    visits=visits,
                    layer=layer,
                    standardize=True,
                )
                # Normalize column names
                rename_map = {}
                if "beta_time" in res.columns:
                    rename_map["beta_time"] = "beta_DiD"
                if "FDR_time" in res.columns:
                    rename_map["FDR_time"] = "fdr"
                if "p_time" in res.columns:
                    rename_map["p_time"] = "p_value"
                if "se_time" in res.columns:
                    rename_map["se_time"] = "se"
                if rename_map:
                    res = res.rename(columns=rename_map)

                res["cell_type"] = ct
                ct_results.append(res)
            except Exception as exc:
                print(f"    Skipping {ct}: {exc}")

    ct_df = pd.concat(ct_results, ignore_index=True) if ct_results else pd.DataFrame()
    print(f"  Cell-type results: {len(ct_df)} rows across {ct_df['cell_type'].nunique() if len(ct_df) else 0} types")

    # ------------------------------------------------------------------
    # Panels C & D: Gene-level within-arm comparison
    # ------------------------------------------------------------------
    gene_results = None
    try:
        import scanpy as sc

        sc.pp.highly_variable_genes(
            adata, n_top_genes=2000, layer=layer, flavor="seurat"
        )
        top_genes = adata.var_names[adata.var["highly_variable"]].tolist()
        print(f"  Selected {len(top_genes)} HVGs for gene-level analysis")

        gene_results = within_arm_comparison(
            adata,
            arm="Treated",
            features=top_genes,
            design=design,
            visits=visits,
            layer=layer,
            standardize=True,
        )

        # Normalize column names
        rename_map = {}
        if "beta_time" in gene_results.columns:
            rename_map["beta_time"] = "beta_DiD"
        if "FDR_time" in gene_results.columns:
            rename_map["FDR_time"] = "fdr"
        if "p_time" in gene_results.columns:
            rename_map["p_time"] = "p_value"
        if "se_time" in gene_results.columns:
            rename_map["se_time"] = "se"
        if rename_map:
            gene_results = gene_results.rename(columns=rename_map)

        n_sig = (gene_results["fdr"] < 0.1).sum() if "fdr" in gene_results.columns else 0
        print(f"  Gene-level: {len(gene_results)} genes, {n_sig} FDR<0.1")

    except Exception as exc:
        print(f"  Gene-level analysis failed: {exc}")

    return dict(
        adata=adata,
        sig_cols=sig_cols,
        ct_df=ct_df,
        gene_results=gene_results,
        layer=layer,
    )


# ======================================================================
# Panel A — Exhaustion signature across cell types (forest plot)
# ======================================================================

def panel_A(ax, data: dict):
    """Forest plot of exhaustion signature beta +/- 95% CI across cell types."""
    ct_df = data["ct_df"]

    if ct_df is None or len(ct_df) == 0:
        ax.text(
            0.5, 0.5,
            "Cell-type-specific results unavailable",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor=COLORS["gray"]),
        )
        ax.set_title("Exhaustion Signature Across Cell Types", fontsize=11)
        ax.axis("off")
        return

    # Find exhaustion signature column
    exh_col = None
    for col in ("sig_Exhaustion", "sig_Immune Exhaustion"):
        if col in ct_df["feature"].values:
            exh_col = col
            break
    if exh_col is None:
        # Fall back to first available signature
        exh_col = ct_df["feature"].iloc[0]

    df = ct_df[ct_df["feature"] == exh_col].copy()
    if len(df) == 0:
        ax.text(0.5, 0.5, "No exhaustion data available",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.axis("off")
        return

    df = df.sort_values("beta_DiD").reset_index(drop=True)
    y_pos = np.arange(len(df))

    # Compute 95% CI
    if "se" in df.columns:
        ci_lo = df["beta_DiD"] - 1.96 * df["se"]
        ci_hi = df["beta_DiD"] + 1.96 * df["se"]
    else:
        ci_lo = df["beta_DiD"]
        ci_hi = df["beta_DiD"]

    # Significance coloring
    sig_col = "fdr" if "fdr" in df.columns else "p_value"
    sig_mask = df[sig_col] < 0.1 if sig_col in df.columns else pd.Series(False, index=df.index)

    # Non-significant
    if (~sig_mask).any():
        ax.hlines(y_pos[~sig_mask], ci_lo[~sig_mask], ci_hi[~sig_mask],
                  color=COLORS["gray"], linewidth=1.5, zorder=1)
        ax.scatter(df.loc[~sig_mask, "beta_DiD"], y_pos[~sig_mask],
                   color=COLORS["gray"], s=45, zorder=2, edgecolors="white",
                   linewidths=0.5)

    # Significant
    if sig_mask.any():
        ax.hlines(y_pos[sig_mask], ci_lo[sig_mask], ci_hi[sig_mask],
                  color=COLORS["highlight"], linewidth=1.8, zorder=1)
        ax.scatter(df.loc[sig_mask, "beta_DiD"], y_pos[sig_mask],
                   color=COLORS["highlight"], s=55, zorder=2, edgecolors="white",
                   linewidths=0.5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["cell_type"].values, fontsize=9)
    ax.set_xlabel(r"Within-arm effect ($\beta$, standardised)")
    ax.set_title(
        f"{sig_display(exh_col)} Across Cell Types",
        fontsize=11, fontweight="bold",
    )

    legend_handles = [
        mpatches.Patch(color=COLORS["highlight"], label="FDR < 0.1"),
        mpatches.Patch(color=COLORS["gray"], label="Not significant"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel B — Effect heterogeneity across cell types
# ======================================================================

def panel_B(ax, data: dict):
    """Horizontal bar chart of effect SD across cell types per signature."""
    ct_df = data["ct_df"]

    if ct_df is None or len(ct_df) == 0:
        ax.text(
            0.5, 0.5,
            "Cell-type heterogeneity data unavailable",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor=COLORS["gray"]),
        )
        ax.set_title("Effect Heterogeneity Across Cell Types", fontsize=11)
        ax.axis("off")
        return

    # Compute SD of beta across cell types for each signature
    het_df = (
        ct_df.groupby("feature", observed=True)["beta_DiD"]
        .agg(["std", "mean"])
        .reset_index()
    )
    het_df = het_df.sort_values("std", ascending=True).reset_index(drop=True)

    # Clean display names
    het_df["display"] = het_df["feature"].map(sig_display)

    y_pos = np.arange(len(het_df))
    median_sd = het_df["std"].median()

    # Color by mean direction
    colors = [
        COLORS["treated"] if m > 0 else COLORS["control"]
        for m in het_df["mean"]
    ]

    ax.barh(y_pos, het_df["std"].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(median_sd, color="black", linewidth=0.8, linestyle="--",
               zorder=0, label=f"Median SD = {median_sd:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(het_df["display"].values, fontsize=9)
    ax.set_xlabel("SD of effect across cell types")
    ax.set_title("Effect Heterogeneity Across Cell Types", fontsize=11,
                 fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.85, label="Mean effect > 0"),
        mpatches.Patch(color=COLORS["control"], alpha=0.85, label="Mean effect < 0"),
        plt.Line2D([0], [0], color="black", linewidth=0.8, linestyle="--",
                   label=f"Median SD = {median_sd:.3f}"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel C — Top differentially affected genes
# ======================================================================

def panel_C(ax, data: dict):
    """Horizontal bar chart of top 10 up + top 10 down genes by FDR."""
    gene_results = data["gene_results"]

    if gene_results is None or len(gene_results) == 0:
        ax.text(
            0.5, 0.5,
            "Gene-level results unavailable\n(CAR-T dataset or analysis failed)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor=COLORS["gray"]),
        )
        ax.set_title("Top Differentially Affected Genes", fontsize=11)
        ax.axis("off")
        return

    df = gene_results.copy()
    sig_col = "fdr" if "fdr" in df.columns else "p_value"

    # Select top 10 up and top 10 down by effect size among significant genes
    df_sig = df[df[sig_col] < 0.1] if (df[sig_col] < 0.1).sum() >= 5 else df.nsmallest(20, sig_col)

    top_up = df_sig.nlargest(10, "beta_DiD")
    top_down = df_sig.nsmallest(10, "beta_DiD")
    selected = pd.concat([top_down, top_up]).drop_duplicates(subset="feature")
    selected = selected.sort_values("beta_DiD", ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(selected))
    colors = [
        COLORS["treated"] if v > 0 else COLORS["control"]
        for v in selected["beta_DiD"]
    ]

    ax.barh(y_pos, selected["beta_DiD"].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)

    # Significance markers
    for i, (_, row) in enumerate(selected.iterrows()):
        fdr_val = row.get(sig_col, 1.0)
        if pd.notna(fdr_val):
            if fdr_val < 0.01:
                marker = "**"
            elif fdr_val < 0.1:
                marker = "*"
            else:
                marker = ""
            if marker:
                x_pos = row["beta_DiD"]
                ha = "left" if x_pos > 0 else "right"
                offset = abs(x_pos) * 0.05 if x_pos != 0 else 0.01
                x_text = x_pos + offset if x_pos > 0 else x_pos - offset
                ax.text(x_text, i, marker, ha=ha, va="center",
                        fontsize=9, fontweight="bold", color="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected["feature"].values, fontsize=8)
    ax.set_xlabel(r"Within-arm effect ($\beta$, standardised)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title("Top Differentially Affected Genes (CAR-T)", fontsize=11,
                 fontweight="bold")

    ax.text(
        0.97, 0.03,
        "** FDR < 0.01    * FDR < 0.1",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, style="italic", color=COLORS["gray"],
    )

    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.85, label="Upregulated"),
        mpatches.Patch(color=COLORS["control"], alpha=0.85, label="Downregulated"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left" if selected["beta_DiD"].mean() < 0 else "upper right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel D — Gene effect distribution
# ======================================================================

def panel_D(ax, data: dict):
    """Histogram of all gene-level beta values with significant genes overlay."""
    gene_results = data["gene_results"]

    if gene_results is None or len(gene_results) == 0:
        ax.text(
            0.5, 0.5,
            "Gene-level results unavailable",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor=COLORS["gray"]),
        )
        ax.set_title("Gene Effect Distribution", fontsize=11)
        ax.axis("off")
        return

    df = gene_results.copy()
    sig_col = "fdr" if "fdr" in df.columns else "p_value"
    betas = df["beta_DiD"].dropna()
    sig_mask = df[sig_col] < 0.1 if sig_col in df.columns else pd.Series(False, index=df.index)
    sig_betas = df.loc[sig_mask, "beta_DiD"].dropna()

    # Histogram of all genes
    bins = np.linspace(betas.min(), betas.max(), 50)
    ax.hist(betas, bins=bins, color=COLORS["gray"], alpha=0.5,
            edgecolor="white", linewidth=0.3, label="All genes", zorder=1)

    # Overlay significant genes
    if len(sig_betas) > 0:
        ax.hist(sig_betas, bins=bins, color=COLORS["highlight"], alpha=0.7,
                edgecolor="white", linewidth=0.3, label="FDR < 0.1", zorder=2)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=3)

    # Summary text box
    n_total = len(betas)
    n_sig = len(sig_betas)
    n_up = (sig_betas > 0).sum() if len(sig_betas) > 0 else 0
    n_down = (sig_betas < 0).sum() if len(sig_betas) > 0 else 0

    summary = (
        f"Total genes: {n_total:,}\n"
        f"FDR < 0.1: {n_sig:,}\n"
        f"  Up: {n_up:,}\n"
        f"  Down: {n_down:,}"
    )
    ax.text(
        0.97, 0.97, summary,
        transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.85),
        family="monospace",
    )

    ax.set_xlabel(r"Within-arm effect ($\beta$, standardised)")
    ax.set_ylabel("Number of genes")
    ax.set_title("Gene-Level Effect Distribution (CAR-T)", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=9, frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Supplementary Figure 7 individual panels."""
    print("Supplementary Figure 7: Biological Discovery Details")
    apply_style()

    try:
        data = _prepare_data()
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return

    # ── Save individual panels ────────────────────────────────────────
    for panel_label, panel_func in [
        ("A_exhaustion_forest", panel_A),
        ("B_effect_heterogeneity", panel_B),
        ("C_top_genes", panel_C),
        ("D_gene_distribution", panel_D),
    ]:
        fig_p, ax_p = plt.subplots(figsize=(8, 6))
        panel_func(ax_p, data)
        fig_p.tight_layout()
        save_panel(fig_p, f"panel_{panel_label}", FIGURE_NAME, SUPP_OUTPUT)

    # ── Cleanup ───────────────────────────────────────────────────────
    if "adata" in data:
        del data["adata"]
    del data
    clear_cache()
    gc.collect()
    print("  Done.\n")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    apply_style()
    generate()
