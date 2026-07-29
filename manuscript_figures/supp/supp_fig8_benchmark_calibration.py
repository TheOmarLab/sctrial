"""
Supplementary Figure 8 — Full NatMeth benchmark: calibration, power and robustness.
====================================================================================

The complete benchmark grids that Figure 3 condenses to single representative
facets. Every panel is drawn from the frozen run (manifest-addressed) using the
SHARED benchmark toolkit ``manuscript_figures._benchmark`` — this figure never
imports another figure's panel code, and there is exactly one copy of each
renderer. Plotting methods, design and parameters match Figure 3 and the other
supplements (broken NEBULA axis, scenario-level 95% Monte Carlo CIs, consistent
method order and colours).

Panels
------
  A  Full mixed-signal null-gene FPR (balanced + one-directional; 50/200/500/2000
     tested genes; 2/4/10/20% signal; separate NEBULA scale; scenario CIs).
  B  Complete pure-null calibration: Type I error vs participants (two-arm and
     single-arm) and vs tested-set size (anchor design); separate NEBULA scale.
  C  Representative QQ plots (200 genes, 10% signal, balanced; separate NEBULA
     y-scale; 95% pointwise beta envelope).
  D  Beta-envelope calibration heatmaps.
  E  NEBULA hierarchy validation (from the provenance-stamped diagnostic).
  F  Full marginal-detection curves (both designs; beta = 0.2/0.5/1.0).
  G  FDR-controlled discovery sensitivity (end-to-end BH TPR).
  H  Bias and RMSE (per method's own oracle; architecture explicit).
  I  Null-gene FPR across robustness families.
  J  Signal detection (end-to-end BH TPR) across signal-bearing robustness families.
  K  Gene evaluability across robustness families.
  L  Convergence among attempted fits.
  M  End-to-end vs tested-only detection (cell-yield families).
"""
from __future__ import annotations

import gc
import json

import matplotlib.pyplot as plt
import numpy as np

from .._benchmark import (
    _load_benchmark_data,
    _load_core_benchmark_data,
    _panel_bench_discovery_sensitivity,
    _panel_bench_endtoend_vs_tested,
    _panel_bench_family_tpr,
    _panel_bench_mixed_fpr,
    _panel_bench_power_vs_n,
    _panel_bench_pure_null_fpr,
    _panel_bench_qq_heatmap,
    _panel_bench_qq_single,
    _panel_bench_quality,
    _panel_bench_scenario_families,
    _panel_bench_signal_rmse,
    _panel_bench_typeI_main,
)
from .._shared import MANUSCRIPT_DIR, SUPP_OUTPUT, apply_style, save_panel

FIGURE_NAME = "SuppFig8_benchmark_calibration"
_DIAG_DIR = MANUSCRIPT_DIR / "benchmark" / "validation" / "nebula_diagnostic"


def _panel_nebula_hierarchy(fig, *, composite: bool = False):
    """NEBULA hierarchy validation from the provenance-stamped diagnostic
    (offset control, matched-model positive control, sigma_u ablation, workaround).

    Reads the diagnostic outputs written by scripts/verify_nebula_offset.py under
    the separate diagnostic tag; renders a 'pending' note if they are absent so the
    figure never fabricates the experiment.
    """
    log = _DIAG_DIR / "nebula_hierarchy_diagnostic.log"
    prov = _DIAG_DIR / "provenance.json"
    ax = fig.add_subplot(1, 1, 1)
    if not log.exists():
        ax.text(0.5, 0.5, "NEBULA hierarchy diagnostic pending\n(run scripts/verify_nebula_offset.py)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#666")
        ax.set_axis_off()
        return
    # Parse the sigma_u ablation table from the diagnostic log.
    rows = []
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                su = float(parts[0])
                mb = float(parts[1])
                fpr = float(parts[3])
                if 0.0 <= su <= 1.0 and -5 < mb < 5 and 0 <= fpr <= 1:
                    rows.append((su, mb, fpr))
            except ValueError:
                continue
    if not rows:
        ax.text(0.5, 0.5, "NEBULA diagnostic present but unparsed", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="#666")
        ax.set_axis_off()
        return
    su = np.array([r[0] for r in rows])
    fpr = np.array([r[2] for r in rows])
    ax.plot(su, fpr, "-o", color="#d62728", markersize=6)
    ax.axhline(0.05, color="#888", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"Participant$\times$visit variance $\sigma_u$", fontsize=11)
    ax.set_ylabel("NEBULA Type I error (p < 0.05)", fontsize=11)
    ax.set_title("NEBULA hierarchy validation (σ$_u$ ablation)", fontsize=12, fontweight="bold")
    if prov.exists():
        tag = json.loads(prov.read_text()).get("tag", "")
        ax.text(0.02, 0.98, tag, transform=ax.transAxes, fontsize=6, va="top", color="#999")


def generate() -> None:
    apply_style()
    print(f"{FIGURE_NAME}: full benchmark calibration/power/robustness")
    bench = _load_benchmark_data()
    core = _load_core_benchmark_data()
    print(f"  sensitivity rows {len(bench):,}  core rows {len(core):,}")


    # A: mixed-signal FPR, balanced + one-directional
    for arch, sfx in (("balanced", "balanced"), ("one_directional", "onedir")):
        fig = plt.figure(figsize=(13, 4.6))
        _panel_bench_mixed_fpr(fig, bench, architecture=arch)
        save_panel(fig, f"panel_A_mixedFPR_{sfx}", FIGURE_NAME, SUPP_OUTPUT)

    # B: pure-null Type I — vs participants (two designs) and vs tested-set
    fig = plt.figure(figsize=(11, 4.6))
    _panel_bench_typeI_main(fig, core)
    fig.suptitle("Pure-null Type I error vs biological sample size", fontweight="bold", y=0.99)
    save_panel(fig, "panel_B_pureNull_vs_participants", FIGURE_NAME, SUPP_OUTPUT)
    fig = plt.figure(figsize=(7.5, 5.0))
    _panel_bench_pure_null_fpr(fig, bench)
    save_panel(fig, "panel_B_pureNull_vs_testedset", FIGURE_NAME, SUPP_OUTPUT)

    # C: QQ
    fig = plt.figure(figsize=(11, 5.5))
    _panel_bench_qq_single(fig, bench, n_genes=200, signal_pct=10)
    save_panel(fig, "panel_C_QQ", FIGURE_NAME, SUPP_OUTPUT)

    # D: beta-envelope heatmap
    fig = plt.figure(figsize=(11, 5.5))
    _panel_bench_qq_heatmap(fig, bench)
    save_panel(fig, "panel_D_beta_envelope", FIGURE_NAME, SUPP_OUTPUT)

    # E: NEBULA hierarchy validation
    fig = plt.figure(figsize=(7, 5))
    _panel_nebula_hierarchy(fig)
    save_panel(fig, "panel_E_nebula_hierarchy", FIGURE_NAME, SUPP_OUTPUT)

    # F: full marginal-detection curves
    fig = plt.figure(figsize=(12, 7.0))
    _panel_bench_power_vs_n(fig, core)
    save_panel(fig, "panel_F_marginal_curves", FIGURE_NAME, SUPP_OUTPUT)

    # G: FDR-controlled discovery sensitivity
    fig = plt.figure(figsize=(13, 4.6))
    _panel_bench_discovery_sensitivity(fig, bench)
    save_panel(fig, "panel_G_discovery_sensitivity", FIGURE_NAME, SUPP_OUTPUT)

    # H: bias / RMSE
    fig = plt.figure(figsize=(12, 5.0))
    _panel_bench_signal_rmse(fig, bench)
    save_panel(fig, "panel_H_bias_rmse", FIGURE_NAME, SUPP_OUTPUT)

    # I: family null FPR
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _panel_bench_scenario_families(ax, core)
    ax.set_title("Null-gene FPR across robustness families", fontweight="bold")
    save_panel(fig, "panel_I_family_FPR", FIGURE_NAME, SUPP_OUTPUT)

    # J: family end-to-end TPR
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _panel_bench_family_tpr(ax, core)
    save_panel(fig, "panel_J_family_TPR", FIGURE_NAME, SUPP_OUTPUT)

    # K: evaluability
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _panel_bench_quality(ax, core, kind="evaluability")
    save_panel(fig, "panel_K_evaluability", FIGURE_NAME, SUPP_OUTPUT)

    # L: convergence
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _panel_bench_quality(ax, core, kind="convergence")
    save_panel(fig, "panel_L_convergence", FIGURE_NAME, SUPP_OUTPUT)

    # M: end-to-end vs tested-only
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _panel_bench_endtoend_vs_tested(ax, core)
    save_panel(fig, "panel_M_endtoend_vs_tested", FIGURE_NAME, SUPP_OUTPUT)

    print(f"  {FIGURE_NAME} complete: individual panels A-M saved")
    gc.collect()
