#!/usr/bin/env python
"""Generate README Quick Start demo assets from real data.

Runs the exact Quick Start workflow and produces:
  - readme_quickstart.png  (composite: table + forest plot)
  - results_table.csv

All outputs go to docs/source/_static/media/
"""
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MEDIA_DIR = REPO_ROOT / "docs" / "source" / "_static" / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))
import sctrial as st

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Sample size too small")


def main():
    # ── 1. Load real dataset ───────────────────────────────────────────
    print("Loading Sade-Feldman dataset...")
    adata = st.load_sade_feldman(max_cells_per_participant_visit=None)
    adata = st.harmonize_response(adata)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # ── 2. Define trial design ─────────────────────────────────────────
    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="response_harmonized",
        arm_treated="Responder",
        arm_control="Non-responder",
        celltype_col="cell_type",
    )

    # ── 3. Score gene sets ─────────────────────────────────────────────
    gene_sets = {
        "Cytotoxicity": ["GZMA", "GZMB", "PRF1", "GNLY", "NKG7"],
        "Exhaustion":   ["PDCD1", "CTLA4", "HAVCR2", "LAG3", "TIGIT"],
    }
    adata = st.score_gene_sets(
        adata, gene_sets, layer="log1p_tpm", method="zmean", prefix="ms_"
    )

    # ── 4. Run DiD ─────────────────────────────────────────────────────
    features = [c for c in adata.obs.columns if c.startswith("ms_")]
    results = st.did_table(
        adata, features, design, visits=("Pre", "Post"), celltype="CD8 T cell"
    )
    print(results[["feature", "beta_DiD", "se_DiD", "p_DiD", "FDR_DiD"]])

    # ── Save CSV ───────────────────────────────────────────────────────
    results.to_csv(MEDIA_DIR / "results_table.csv", index=False)

    # ── 5. Build composite figure ──────────────────────────────────────
    fig = plt.figure(figsize=(12, 4), facecolor="white", dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.35)

    # Left panel: results table
    ax_tab = fig.add_subplot(gs[0])
    ax_tab.set_axis_off()
    ax_tab.set_title("DiD Results — CD8 T cells", fontsize=11, fontweight="bold",
                     loc="left", pad=8)

    display_cols = ["feature", "beta_DiD", "se_DiD", "p_DiD", "FDR_DiD"]
    tab_data = results[display_cols].copy()
    tab_data["feature"] = tab_data["feature"].str.replace("ms_", "", regex=False)
    for c in ["beta_DiD", "se_DiD"]:
        tab_data[c] = tab_data[c].map(lambda x: f"{x:.3f}")
    for c in ["p_DiD", "FDR_DiD"]:
        tab_data[c] = tab_data[c].map(lambda x: f"{x:.2e}" if x < 0.01 else f"{x:.3f}")
    tab_data.columns = ["Signature", "β_DiD", "SE", "p-value", "FDR"]

    table = ax_tab.table(
        cellText=tab_data.values,
        colLabels=tab_data.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(tab_data.columns)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(len(tab_data)):
        for j in range(len(tab_data.columns)):
            cell = table[i + 1, j]
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")
            else:
                cell.set_facecolor("white")

    # Right panel: forest plot
    ax_forest = fig.add_subplot(gs[1])
    st.plot_did_forest(results, ax=ax_forest)
    ax_forest.set_title("Forest Plot", fontsize=11, fontweight="bold", loc="left", pad=8)

    fig.savefig(
        MEDIA_DIR / "readme_quickstart.png",
        bbox_inches="tight", dpi=150, facecolor="white"
    )
    plt.close(fig)
    print(f"\nSaved: {MEDIA_DIR / 'readme_quickstart.png'}")
    print(f"Saved: {MEDIA_DIR / 'results_table.csv'}")


if __name__ == "__main__":
    main()
