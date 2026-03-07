"""
Supplementary Figure 6 -- Additional Robustness Details
========================================================

Three-panel figure (1x3) providing extended robustness analyses that
supplement the main statistical robustness figure.  These panels were
originally Figure 8C, 8D, and Figure 5A in the 12-panel manuscript.

Panels
------
A  Cell-level vs participant-level effect sizes (beta_DiD) side-by-side.
B  Effect sizes with 95% CIs (model SE for cell, bootstrap SE for
   participant level).
C  HALLMARK GSEA pathway enrichment from cached prerank results.
"""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    TrialDesign,
    apply_style,
    clear_cache,
    despine,
    did_table,
    get_sade_feldman,
    harmonize_response,
    save_panel,
    score_signatures,
    sig_display,
)

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "SuppFig6_robustness_details"
FIGSIZE = (18, 6)
SCRIPT_DIR = Path(__file__).resolve().parent       # supp/
N_BOOT = 200


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_did_data() -> dict:
    """Run DiD at both cell and participant_visit levels on Sade-Feldman.

    Uses sctrial's built-in wild-cluster bootstrap for participant-level
    SE estimation (``use_bootstrap=True``).
    """
    adata = get_sade_feldman()
    adata = harmonize_response(adata)

    if "log1p_tpm" not in adata.layers and "tpm" in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])

    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="response_harmonized",
        arm_treated="Responder",
        arm_control="Non-responder",
    )
    visits = ("Pre", "Post")

    common_kw = dict(
        features=sig_cols,
        design=design,
        visits=visits,
        layer="log1p_tpm",
        standardize=True,
    )

    # Cell-level DiD
    print("  Running cell-level DiD ...")
    df_cell = did_table(adata, aggregate="cell", **common_kw)

    # Participant-level DiD (analytical SE)
    print("  Running participant-level DiD ...")
    df_part = did_table(adata, aggregate="participant_visit", **common_kw)

    # Participant-level DiD with bootstrap SE via sctrial API
    print(f"  Running bootstrap DiD (n={N_BOOT}) via sctrial ...")
    df_boot = did_table(
        adata, aggregate="participant_visit",
        use_bootstrap=True, n_boot=N_BOOT, seed=42,
        **common_kw,
    )

    # Extract bootstrap SE from the bootstrap results
    boot_se = dict(zip(df_boot["feature"], df_boot["se_DiD"]))

    return dict(
        df_cell=df_cell,
        df_part=df_part,
        boot_se=boot_se,
        sig_cols=sig_cols,
        adata=adata,
    )


def _load_gsea_cache() -> pd.DataFrame | None:
    """Load cached HALLMARK GSEA prerank results.

    Searches several candidate locations for the CSV.
    """
    candidates = [
        # From supp/ -> manuscript_figures/ -> sc_trial_inference/ ->
        # sctrial/ -> sc-trialdiff/ -> manuscript/gsea_hallmark/
        SCRIPT_DIR.parent.parent.parent.parent / "manuscript"
        / "gsea_hallmark" / "gseapy.gene_set.prerank.report.csv",
        # SUPP_OUTPUT is manuscript/supp; go up to manuscript/
        SUPP_OUTPUT.parent / "gsea_hallmark"
        / "gseapy.gene_set.prerank.report.csv",
        # From _shared.MANUSCRIPT_DIR (sctrial/manuscript)
        SCRIPT_DIR.parent.parent.parent / "manuscript"
        / "gsea_hallmark" / "gseapy.gene_set.prerank.report.csv",
    ]
    for path in candidates:
        path = path.resolve()
        if path.exists():
            df = pd.read_csv(path)
            print(f"  Loaded GSEA cache: {path.name} ({len(df)} pathways)")
            return df

    print("  WARNING: GSEA cache not found in any candidate location")
    return None


# ======================================================================
# Panel A -- Cell-level vs participant-level beta side-by-side
# ======================================================================

def panel_A(ax, data: dict):
    """Side-by-side horizontal bars comparing cell-level vs participant-level
    beta_DiD for each signature."""
    df_cell = data["df_cell"]
    df_part = data["df_part"]

    if df_cell is None or df_part is None:
        ax.text(0.5, 0.5, "No data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    # Merge on feature
    cell = df_cell.set_index("feature")[["beta_DiD"]].rename(
        columns={"beta_DiD": "beta_cell"})
    part = df_part.set_index("feature")[["beta_DiD"]].rename(
        columns={"beta_DiD": "beta_part"})
    df = cell.join(part, how="inner").reset_index()
    df["display"] = df["feature"].map(sig_display)

    # Sort by participant-level beta
    df = df.sort_values("beta_part", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    bar_h = 0.35

    ax.barh(y - bar_h / 2, df["beta_cell"], height=bar_h,
            color=COLORS["highlight"], alpha=0.8, edgecolor="white",
            linewidth=0.5, label="Cell-level")
    ax.barh(y + bar_h / 2, df["beta_part"], height=bar_h,
            color=COLORS["treated"], alpha=0.8, edgecolor="white",
            linewidth=0.5, label="Participant-level")

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display"], fontsize=9)
    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (standardized)", fontsize=10)
    ax.set_title("Effect Sizes by Aggregation Level", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel B -- Effect sizes with 95% CI (forest plot)
# ======================================================================

def panel_B(ax, data: dict):
    """Forest-style plot with error bars.  Cell-level uses model SE;
    participant-level uses bootstrap SE."""
    df_cell = data["df_cell"]
    df_part = data["df_part"]
    boot_se = data["boot_se"]

    if df_cell is None or df_part is None:
        ax.text(0.5, 0.5, "No data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    # Build merged DataFrame
    cell = df_cell.set_index("feature")[["beta_DiD", "se_DiD"]].rename(
        columns={"beta_DiD": "beta_cell", "se_DiD": "se_cell"})
    part = df_part.set_index("feature")[["beta_DiD"]].rename(
        columns={"beta_DiD": "beta_part"})
    df = cell.join(part, how="inner").reset_index()
    df["display"] = df["feature"].map(sig_display)
    df["se_boot"] = df["feature"].map(boot_se)

    # Sort by participant-level beta
    df = df.sort_values("beta_part", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    offset = 0.18

    # Cell-level (model SE)
    ci95_cell = 1.96 * df["se_cell"]
    ax.errorbar(df["beta_cell"], y - offset,
                xerr=ci95_cell, fmt="s", markersize=5,
                color=COLORS["highlight"], alpha=0.8,
                ecolor=COLORS["highlight"], elinewidth=1.0, capsize=3,
                label="Cell-level (model SE)", zorder=3)

    # Participant-level (bootstrap SE)
    ci95_boot = 1.96 * df["se_boot"]
    ax.errorbar(df["beta_part"], y + offset,
                xerr=ci95_boot, fmt="o", markersize=5,
                color=COLORS["treated"], alpha=0.8,
                ecolor=COLORS["treated"], elinewidth=1.0, capsize=3,
                label="Participant-level (bootstrap SE)", zorder=3)

    ax.axvline(0, color="black", lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display"], fontsize=9)
    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ with 95% CI", fontsize=10)
    ax.set_title("Effect Sizes with Confidence Intervals", fontsize=11)

    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)

    # Annotation
    ax.text(
        0.03, 0.97,
        f"Bootstrap: n={N_BOOT}\n(participant resampling)",
        transform=ax.transAxes, fontsize=7, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.8),
    )
    despine(ax)


# ======================================================================
# Panel C -- HALLMARK GSEA pathway enrichment (cached)
# ======================================================================

def panel_C(ax, data: dict):
    """Bidirectional bar chart of top 15 HALLMARK pathways by NES,
    loaded from cached GSEA prerank results."""
    gsea_df = data.get("gsea_df")

    if gsea_df is None or len(gsea_df) == 0:
        ax.text(0.5, 0.5,
                "GSEA results unavailable\n(cached file not found)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                          edgecolor=COLORS["gray"]))
        ax.set_title("HALLMARK Pathway Enrichment", fontsize=11)
        ax.axis("off")
        return

    df = gsea_df.copy()

    # Identify columns (gseapy output format)
    nes_col = "NES"
    fdr_col = "FDR q-val"
    term_col = "Term"

    # Ensure numeric
    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Sort by absolute NES, take top 15
    df["abs_nes"] = df[nes_col].abs()
    df = df.sort_values("abs_nes", ascending=False).head(15)
    df = df.sort_values(nes_col, ascending=True).reset_index(drop=True)

    # Clean pathway names: remove HALLMARK_ prefix, replace underscores
    df["pathway"] = (
        df[term_col]
        .str.replace(r"^HALLMARK[_ ]", "", regex=True)
        .str.replace("_", " ")
        .str.title()
    )
    # Truncate long names
    df["pathway"] = df["pathway"].apply(
        lambda s: s[:38] + "..." if len(str(s)) > 41 else s
    )

    y = np.arange(len(df))

    # Colour by direction
    colors = [
        COLORS["treated"] if v > 0 else COLORS["control"]
        for v in df[nes_col]
    ]

    ax.barh(y, df[nes_col].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5, height=0.7)

    # FDR star markers
    for i, (_, row) in enumerate(df.iterrows()):
        fdr_val = row[fdr_col]
        if pd.notna(fdr_val) and fdr_val < 0.25:
            star = "***" if fdr_val < 0.001 else "**" if fdr_val < 0.01 else "*"
            x_pos = row[nes_col]
            ha = "left" if x_pos > 0 else "right"
            x_off = 0.03 if x_pos > 0 else -0.03
            ax.text(x_pos + x_off, i, star, ha=ha, va="center",
                    fontsize=9, fontweight="bold", color="black")

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["pathway"].values, fontsize=8)
    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=10)
    ax.set_title("HALLMARK Pathway Enrichment (GSEA)", fontsize=11)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.85,
                       label="Upregulated in Responders"),
        mpatches.Patch(color=COLORS["control"], alpha=0.85,
                       label="Downregulated in Responders"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9)

    ax.text(0.97, 0.03,
            "* FDR < 0.25  ** < 0.01  *** < 0.001",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, fontstyle="italic", color=COLORS["gray"])
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate():
    """Create and save Supplementary Figure 6."""
    print("Supplementary Figure 6: Additional Robustness Details")

    # ── Prepare data ──────────────────────────────────────────────────
    try:
        data = _prepare_did_data()
    except Exception as exc:
        print(f"  ERROR preparing DiD data: {exc}")
        data = dict(df_cell=None, df_part=None, boot_se={},
                    sig_cols=[], adata=None)

    try:
        gsea_df = _load_gsea_cache()
    except Exception as exc:
        print(f"  ERROR loading GSEA cache: {exc}")
        gsea_df = None
    data["gsea_df"] = gsea_df

    # ── Save individual panels ────────────────────────────────────────
    for panel_label, panel_func in [("A", panel_A), ("B", panel_B),
                                     ("C", panel_C)]:
        fig_p, ax_p = plt.subplots(figsize=(7, 6))
        panel_func(ax_p, data)
        fig_p.tight_layout()
        save_panel(fig_p, f"panel_{panel_label}", FIGURE_NAME, SUPP_OUTPUT)

    # ── Cleanup ───────────────────────────────────────────────────────
    if data.get("adata") is not None:
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
