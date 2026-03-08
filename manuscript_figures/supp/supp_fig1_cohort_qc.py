"""
Supplementary Figure 1 — Cohort Integrity and QC Readiness.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt

from .._shared import SUPP_OUTPUT, apply_style, clear_cache, save_panel
from . import supp_fig1_qc as qc
from . import supp_fig3_clinical as cohort

FIGURE_NAME = "SuppFig1_cohort_qc_readiness"


def generate():
    """Create and save Supplementary Figure 1 panels (A–H)."""
    print("Supplementary Figure 1: Cohort Integrity and QC Readiness")
    loaded_qc = qc._load_all()
    loaded_cohort = cohort._load_all()

    if not loaded_qc or not loaded_cohort:
        print("  Missing data; skipping.")
        return

    # A: dataset census / design table
    fig, ax = plt.subplots(figsize=(10, 4))
    cohort._panel_design_table(ax, loaded_cohort)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # B: pairing structure
    fig, ax = plt.subplots(figsize=(8, 4))
    cohort._panel_pairing(ax, loaded_cohort)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # C: cells per participant by arm
    fig, ax = plt.subplots(figsize=(9, 5))
    cohort._panel_cells_per_pid_arm(ax, loaded_cohort)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # D: genes per cell distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    qc._panel_ngenes_dist(ax, loaded_qc)
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # E: total counts distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    qc._panel_counts_dist(ax, loaded_qc)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # F: mito / ribo QC distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    qc._panel_mito_ribo(ax, loaded_qc)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # G: inequality (Lorenz + Gini)
    fig, ax = plt.subplots(figsize=(6, 6))
    qc._panel_lorenz_gini(ax, loaded_qc)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # H: attrition + visit completeness summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cohort._panel_dropout(axes[0], loaded_cohort)
    cohort._panel_completeness_detailed(axes[1], loaded_cohort)
    fig.tight_layout()
    save_panel(fig, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    for data in loaded_qc.values():
        if "adata" in data:
            del data["adata"]
    for data in loaded_cohort.values():
        if "adata" in data:
            del data["adata"]
    loaded_qc.clear()
    loaded_cohort.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
