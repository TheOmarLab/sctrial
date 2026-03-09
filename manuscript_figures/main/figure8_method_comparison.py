"""
Figure 8 — Method Comparison and Benchmarking
===============================================

Four-panel figure comparing cell-level versus participant-level DiD
inference on the Sade-Feldman melanoma immunotherapy cohort.

Panels
------
A : Scatter of cell-level vs participant-level effect sizes (Pearson r).
B : Side-by-side −log10(p) comparison (cell vs participant level).
C : Paired bar chart of effect sizes by signature, each method side by side.
D : Standard-error comparison showing inflated precision at cell level.
"""

from __future__ import annotations

import gc
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .._shared import (
    COLORS,
    MAIN_OUTPUT,
    TrialDesign,
    apply_style,
    despine,
    did_table,
    get_sade_feldman,
    harmonize_response,
    save_panel,
    score_signatures,
    sig_display,
)

warnings.filterwarnings("ignore")

FIGURE_NAME = "Figure8_method_comparison"
VISITS: tuple[str, str] = ("Pre", "Post")
N_BOOT = 999

DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)


# ── data preparation ────────────────────────────────────────────────────

def _prepare_data() -> dict:
    """Run cell-level and participant-level DiD; return both DataFrames."""
    adata = get_sade_feldman()
    if "log1p_tpm" not in adata.layers:
        raise RuntimeError("log1p_tpm layer missing from Sade-Feldman dataset.")
    adata = harmonize_response(adata)
    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

    common_kw = dict(
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer="log1p_tpm",
        standardize=True,
    )

    # Cell-level DiD
    print("  Running cell-level DiD ...")
    df_cell = did_table(adata, aggregate="cell", **common_kw)

    # Participant-level DiD (analytical SE)
    print("  Running participant-level DiD ...")
    df_part = did_table(adata, aggregate="participant_visit", **common_kw)

    # Participant-level DiD (bootstrap SE + CI)
    print("  Running bootstrap DiD ...")
    df_boot = did_table(
        adata, aggregate="participant_visit",
        use_bootstrap=True, n_boot=N_BOOT, seed=42,
        **common_kw,
    )

    return {
        "df_cell": df_cell,
        "df_part": df_part,
        "df_boot": df_boot,
        "sig_cols": sig_cols,
        "adata": adata,
    }


# ── Panel A: Effect-size correlation scatter ────────────────────────────

def _panel_a(ax, data: dict) -> None:
    """Scatter of cell-level vs participant-level β_DiD with Pearson r."""
    df_cell = data["df_cell"]
    df_part = data["df_part"]

    merged = df_cell[["feature", "beta_DiD"]].merge(
        df_part[["feature", "beta_DiD"]],
        on="feature", suffixes=("_cell", "_part"),
    )
    merged["display"] = merged["feature"].apply(sig_display)

    x = merged["beta_DiD_cell"].values
    y = merged["beta_DiD_part"].values

    # Identity line
    lo = min(x.min(), y.min()) * 1.15
    hi = max(x.max(), y.max()) * 1.15
    ax.plot([lo, hi], [lo, hi], ls="--", color=COLORS["gray"], lw=1, zorder=1)

    ax.scatter(x, y, s=55, color=COLORS["treated"],
               edgecolor="white", linewidth=0.5, zorder=3)

    # Labels
    for _, row in merged.iterrows():
        ax.annotate(
            row["display"],
            (row["beta_DiD_cell"], row["beta_DiD_part"]),
            fontsize=6, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
        )

    r, p = stats.pearsonr(x, y)
    ax.text(
        0.05, 0.95, f"r = {r:.3f}\np = {p:.2e}",
        transform=ax.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
    )

    ax.set_xlabel(r"Cell-level $\beta_{\mathrm{DiD}}$")
    ax.set_ylabel(r"Participant-level $\beta_{\mathrm{DiD}}$")
    ax.set_title("Effect Size Correlation", fontsize=10)
    despine(ax)


# ── Panel B: P-value comparison ─────────────────────────────────────────

def _panel_b(ax, data: dict) -> None:
    """Horizontal grouped bars of −log10(p) at cell vs participant level."""
    df_cell = data["df_cell"].copy()
    df_part = data["df_part"].copy()

    merged = df_cell[["feature", "p_DiD"]].merge(
        df_part[["feature", "p_DiD"]],
        on="feature", suffixes=("_cell", "_part"),
    )
    merged["display"] = merged["feature"].apply(sig_display)
    merged["nlog10_cell"] = -np.log10(merged["p_DiD_cell"].clip(lower=1e-300))
    merged["nlog10_part"] = -np.log10(merged["p_DiD_part"].clip(lower=1e-300))
    merged = merged.sort_values("nlog10_part", ascending=True)

    y_pos = np.arange(len(merged))
    bar_h = 0.35

    ax.barh(
        y_pos - bar_h / 2, merged["nlog10_cell"].values,
        height=bar_h, color=COLORS["highlight"], alpha=0.8,
        label=f"Cell-level (n = {13183:,})", edgecolor="none",
    )
    ax.barh(
        y_pos + bar_h / 2, merged["nlog10_part"].values,
        height=bar_h, color=COLORS["treated"], alpha=0.8,
        label="Participant-level (n = 25)", edgecolor="none",
    )

    ax.axvline(-np.log10(0.05), ls="--", color=COLORS["gray"], lw=0.8,
               label="p = 0.05")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["display"].values, fontsize=8)
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.set_title("Cell vs Participant Inference", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel C: Side-by-side effect sizes ──────────────────────────────────

def _panel_c(ax, data: dict) -> None:
    """Paired horizontal bars comparing effect magnitudes."""
    df_cell = data["df_cell"]
    df_part = data["df_part"]

    merged = df_cell[["feature", "beta_DiD"]].merge(
        df_part[["feature", "beta_DiD"]],
        on="feature", suffixes=("_cell", "_part"),
    )
    merged["display"] = merged["feature"].apply(sig_display)
    merged = merged.sort_values("beta_DiD_part", ascending=True)

    y_pos = np.arange(len(merged))
    bar_h = 0.35

    ax.barh(
        y_pos - bar_h / 2, merged["beta_DiD_cell"].values,
        height=bar_h, color=COLORS["highlight"], alpha=0.8,
        label="Cell-level", edgecolor="none",
    )
    ax.barh(
        y_pos + bar_h / 2, merged["beta_DiD_part"].values,
        height=bar_h, color=COLORS["treated"], alpha=0.8,
        label="Participant-level", edgecolor="none",
    )

    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["display"].values, fontsize=8)
    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (standardized)")
    ax.set_title("Effect Size Comparison", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel D: Standard-error comparison ──────────────────────────────────

def _panel_d(ax, data: dict) -> None:
    """Paired horizontal bars of SE at cell vs participant level."""
    df_cell = data["df_cell"]
    df_boot = data["df_boot"]

    # Cell-level SE
    se_cell = df_cell[["feature", "se_DiD"]].copy()
    se_cell = se_cell.rename(columns={"se_DiD": "se_cell"})

    # Participant-level bootstrap SE
    se_part = df_boot[["feature"]].copy()
    se_part["se_part"] = df_boot.get("se_DiD_boot", df_boot["se_DiD"])

    merged = se_cell.merge(se_part, on="feature")
    merged["display"] = merged["feature"].apply(sig_display)
    merged = merged.sort_values("se_part", ascending=True)

    y_pos = np.arange(len(merged))
    bar_h = 0.35

    ax.barh(
        y_pos - bar_h / 2, merged["se_cell"].values,
        height=bar_h, color=COLORS["highlight"], alpha=0.8,
        label="Cell-level SE", edgecolor="none",
    )
    ax.barh(
        y_pos + bar_h / 2, merged["se_part"].values,
        height=bar_h, color=COLORS["treated"], alpha=0.8,
        label="Participant-level SE (bootstrap)", edgecolor="none",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["display"].values, fontsize=8)
    ax.set_xlabel("Standard Error")
    ax.set_title("Precision: Cell vs Participant Level", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Composite generation ────────────────────────────────────────────────

def generate() -> None:
    """Create and save all Figure 8 panels."""
    apply_style()
    print("Figure 8: Method Comparison and Benchmarking")
    data = _prepare_data()

    panel_funcs = [
        ("panel_A_effect_correlation", _panel_a),
        ("panel_B_pvalue_comparison", _panel_b),
        ("panel_C_effect_size_comparison", _panel_c),
        ("panel_D_standard_error_comparison", _panel_d),
    ]
    for panel_name, func in panel_funcs:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        func(ax, data)
        save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    adata = data.get("adata")
    if adata is not None:
        del adata
    del data
    gc.collect()

    print(f"  Figure 8 complete: {FIGURE_NAME}")
