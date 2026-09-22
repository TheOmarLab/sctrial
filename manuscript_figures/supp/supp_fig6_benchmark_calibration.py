"""
Supplementary Figure 6 — Full NatMeth benchmark: calibration, power and robustness.
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
  A  Full mixed-signal null-gene FPR (balanced; 50/200/500/2000 tested genes;
     2/4/10/20% signal; separate NEBULA scale; scenario CIs).
  B  Full mixed-signal null-gene FPR (one-directional; same conditions as A).
  C  Pure-null Type I error vs tested-set size (separate NEBULA scale).
  D  NEBULA hierarchy validation (sigma_u ablation; provenance-stamped diagnostic).
  E  End-to-end vs tested-only detection (cell-yield families).
  F  Representative QQ plots (200 genes, 10% signal, balanced; separate NEBULA
     y-scale; 95% pointwise beta envelope).
  G  Null-gene calibration heatmaps: % of null p-values outside 95% CI.
  H  FDR-controlled discovery sensitivity (end-to-end BH TPR).
  I  Full marginal-detection curves (both designs; beta = 0.2/0.5/1.0).
  J  Bias and RMSE (per method's own oracle; balanced architecture).
  K  Null-gene FPR across robustness families.
  L  Signal detection (end-to-end BH TPR) across signal-bearing robustness families.
  M  Gene evaluability across robustness families.
  N  Convergence among attempted fits.
"""
from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np

from .._benchmark import (
    _bench_legend_below,
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
from .._shared import MANUSCRIPT_DIR, SUPP_OUTPUT, apply_style, despine, save_panel

FIGURE_NAME = "SuppFig6_benchmark_calibration"
_DIAG_DIR = MANUSCRIPT_DIR / "benchmark" / "validation" / "nebula_diagnostic"


def _parse_nebula_diagnostic(log_text: str) -> dict:
    """Section-aware parse of the NEBULA diagnostic log.

    Returns the sigma_u ablation table plus the model-compatible control and the
    workaround, each pulled from its OWN section so numeric lines elsewhere in the
    log cannot leak into the ablation curve.
    """
    import re

    lines = log_text.splitlines()

    def _section(start_key, end_key=None):
        s = next((i for i, ln in enumerate(lines) if start_key in ln), None)
        if s is None:
            return []
        e = len(lines)
        if end_key is not None:
            e = next((j for j in range(s + 1, len(lines)) if end_key in lines[j]), len(lines))
        return lines[s:e]

    # sigma_u ablation table (between its header and the WORKAROUND section)
    abl = _section("SIGMA_U ABLATION", "WORKAROUND DIAGNOSTIC")
    ablation = []
    for ln in abl:
        p = ln.split()
        if len(p) >= 6:
            try:
                su, mb, fpr = float(p[0]), float(p[1]), float(p[3])
            except ValueError:
                continue
            if 0.0 <= su <= 1.0 and 0.0 <= fpr <= 1.0:
                ablation.append((su, mb, fpr))

    ctrl = _section("MODEL-COMPATIBLE NULL", "SIGMA_U ABLATION")
    ctrl_txt = "\n".join(ctrl)
    def _grab(pat, txt):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None
    control = {
        "fpr": _grab(r"FPR at 0\.05\s*:\s*([\d.]+)", ctrl_txt),
        "mean_beta": _grab(r"mean beta\s*:\s*([+-]?[\d.]+)", ctrl_txt),
    }
    wk = _section("WORKAROUND DIAGNOSTIC")
    wk_txt = "\n".join(wk)
    workaround = {
        "standard_fpr": _grab(r"standard.*?FPR\s+([\d.]+)", wk_txt),
        "workaround_fpr": _grab(r"workaround.*?FPR\s+([\d.]+)", wk_txt),
    }
    return {"ablation": ablation, "control": control, "workaround": workaround}


def _panel_nebula_hierarchy(fig, *, ax=None, composite: bool = False):
    """NEBULA hierarchy validation from the provenance-stamped diagnostic.

    The sigma_u ablation isolates the omitted participant-by-visit variance as the
    cause of NEBULA's Type I inflation: rejection rate rises monotonically from
    nominal at sigma_u = 0 to the calibrated value. The model-compatible positive
    control (NEBULA's own DGP) is annotated to show the offset contract is sound
    and calibration is restored when the assumed hierarchy holds.

    Reads scripts/verify_nebula_offset.py outputs under the separate diagnostic
    tag; renders a 'pending' note if absent so the figure never fabricates it.
    """
    log = _DIAG_DIR / "nebula_hierarchy_diagnostic.log"
    prov = _DIAG_DIR / "provenance.json"
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)
    _lbl = 5.05 if composite else 11
    _ttl = 6.0 if composite else 12
    _ann = 4.2 if composite else 8
    if not log.exists():
        ax.text(0.5, 0.5, "NEBULA hierarchy diagnostic pending\n(run scripts/verify_nebula_offset.py)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#666")
        ax.set_axis_off()
        return
    parsed = _parse_nebula_diagnostic(log.read_text())
    rows = parsed["ablation"]
    if not rows:
        ax.text(0.5, 0.5, "NEBULA diagnostic present but unparsed", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="#666")
        ax.set_axis_off()
        return
    su = np.array([r[0] for r in rows])
    fpr = np.array([r[2] for r in rows])
    neb = "#ff7f0e"  # NEBULA colour (matches the benchmark panels)
    ax.plot(su, fpr, "-o", color=neb, markersize=(4.2 if composite else 7),
            markeredgecolor="white", markeredgewidth=0.6, linewidth=(1.4 if composite else 2.2),
            zorder=5)
    ax.axhline(0.05, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
    # Mark the calibrated sigma_u (the frozen simulator's value = the largest x).
    su_cal = float(su.max())
    ax.axvline(su_cal, color="#888", linestyle=":", linewidth=0.9, alpha=0.7, zorder=1)
    ax.annotate(rf"calibrated $\sigma_u$={su_cal:.3f}", (su_cal + 0.015, 0.02),
                fontsize=(_ann + 1.2 if composite else _ann), ha="left",
                va="bottom", color="#555", rotation=90)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.03, su_cal + 0.06)
    ax.set_xlabel(r"Participant$\times$visit s.d. $\sigma_u$ (omitted by NEBULA)", fontsize=_lbl)
    ax.set_ylabel("NEBULA Type I error\n(p < 0.05)", fontsize=5.2 if composite else _lbl)
    ax.set_title(r"NEBULA hierarchy validation ($\sigma_u$ ablation)",
                 fontsize=_ttl, fontweight="bold")
    ax.tick_params(labelsize=_lbl)
    # Positive control: under NEBULA's own DGP the offset is unbiased and Type I is
    # nominal, so the inflation above is the omitted hierarchy, not a wiring error.
    # Placed in the lower-right empty region (the curve is at ~0.8+ for x>0.3), so
    # the rising segment never crosses the box.
    c = parsed["control"]
    if c.get("fpr") is not None and c.get("mean_beta") is not None:
        mb = c["mean_beta"]
        mb_str = f"{mb:+.3f}" if abs(mb) >= 5e-4 else "0.000"
        ax.annotate(
            f"Model-compatible control\n(NEBULA's own DGP):\nType I = {c['fpr']:.3f}, "
            fr"mean $\beta$ = {mb_str}""\n→ offset validated",
            (0.22, 0.30), xycoords="axes fraction", fontsize=_ann, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.95))
    despine(ax)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
    # No provenance stamp on the figure itself: provenance lives in provenance.json
    # under the diagnostic tag; a stamp inside the axes read as a stray watermark.
    del prov


def _composite(bench, core) -> None:
    """Assemble the condensed A-O benchmark artboard at 180×215 mm.

    Rows merged vs the original: A+B share one row; D+G+H share one row;
    L+M share one row; N+O share one row. Panel C (pure-null vs participants,
    identical to Fig3C) is dropped. All renderers are unchanged.
    """

    _mm = 1.0 / 25.4
    fig = plt.figure(figsize=(180 * _mm, 215 * _mm))

    # Rows: top (legend only), AB, DGH, FG (QQ | beta-envelope),
    # HI (marginal 2-row | discovery 2-row aligned side-by-side), J, LM, NO.
    # Panel letters run A-N sequentially by visual order.
    layout = [
        ("AB",  0.82), ("s", 0.72),
        ("DGH", 0.78), ("s", 0.90),
        ("FG",  1.20), ("s", 0.62),
        ("HI",  1.25), ("s", 0.62),
        ("J",   1.20), ("s", 0.62),
        ("LM",  0.65), ("s", 0.62),
        ("NO",  0.65),
    ]
    hr = [h for _, h in layout]
    outer = fig.add_gridspec(len(layout), 1, height_ratios=hr,
                             left=0.085, right=0.965, top=0.992, bottom=0.008, hspace=0.0)
    cell = {name: outer[i] for i, (name, _) in enumerate(layout) if name != "s"}

    # A | B — mixed-signal FPR (balanced | one-directional)
    sub_ab = cell["AB"].subgridspec(1, 2, wspace=0.22)
    _panel_bench_mixed_fpr(fig, bench, composite=True, gs_parent=sub_ab[0], architecture="balanced")
    _panel_bench_mixed_fpr(fig, bench, composite=True, gs_parent=sub_ab[1], architecture="one_directional")

    # C | D | E — pure-null vs tested-set | NEBULA hierarchy | end-to-end vs tested
    sub_dgh = cell["DGH"].subgridspec(1, 3, width_ratios=[1.4, 1.0, 1.0], wspace=0.32)
    _panel_bench_pure_null_fpr(fig, bench, composite=True, gs_parent=sub_dgh[0])
    _panel_nebula_hierarchy(fig, ax=fig.add_subplot(sub_dgh[1]), composite=True)
    _panel_bench_endtoend_vs_tested(fig.add_subplot(sub_dgh[2]), core, composite=True)

    # F | G — QQ plots (common ylabel) | beta-envelope heatmaps
    sub_fg = cell["FG"].subgridspec(1, 2, wspace=0.30, width_ratios=[1, 1.18])
    _panel_bench_qq_single(fig, bench, n_genes=200, signal_pct=10, composite=True,
                           gs_parent=sub_fg[0], suppress_ylabel=True)
    _panel_bench_qq_heatmap(fig, bench, composite=True, gs_parent=sub_fg[1])

    # H | I — discovery sensitivity (2×2 sizes) | marginal-detection curves (2 designs)
    # Use a shared 2-row × 2-col inner grid so H's size rows align with I's design rows.
    # hspace=0.55 matches power_vs_n's internal hspace so the row boundary aligns.
    sub_hi = cell["HI"].subgridspec(2, 2, hspace=0.95, wspace=0.30)
    _panel_bench_discovery_sensitivity(fig, bench, composite=True, gs_parent=sub_hi[0, 0],
                                       panel_sizes=[50, 200], suppress_ylabel=True,
                                       marker_scale=0.7)
    _panel_bench_discovery_sensitivity(fig, bench, composite=True, gs_parent=sub_hi[1, 0],
                                       panel_sizes=[500, 2000], suppress_ylabel=True,
                                       marker_scale=0.7)
    _panel_bench_power_vs_n(fig, core, composite=True, gs_parent=sub_hi[0:2, 1],
                            marker_scale=0.7)

    # J — bias / RMSE
    _panel_bench_signal_rmse(fig, bench, composite=True, gs_parent=cell["J"])

    # K | L — null FPR families | TPR families
    sub_lm = cell["LM"].subgridspec(1, 2, wspace=0.38)
    ax_l = fig.add_subplot(sub_lm[0])
    _panel_bench_scenario_families(ax_l, core, composite=True)
    ax_l.set_title("Null-gene FPR across robustness families", fontsize=6.0, fontweight="bold", pad=4)
    _panel_bench_family_tpr(fig.add_subplot(sub_lm[1]), core, composite=True)

    # M | N — evaluability | convergence
    sub_no = cell["NO"].subgridspec(1, 2, wspace=0.38)
    _panel_bench_quality(fig.add_subplot(sub_no[0]), core, kind="evaluability", composite=True)
    _panel_bench_quality(fig.add_subplot(sub_no[1]), core, kind="convergence", composite=True)

    fig.canvas.draw()

    # Legend in the spacer below A and B — use tight-bbox so it clears xtick labels.
    _bench_legend_below(fig, cell["AB"], fontsize=4.6, short=True, markersize=4,
                        y_pad=0.007)

    # No figure-level legend: A and B carry no per-panel legend in composite mode.

    # Common ylabels for F (QQ, both rows) and I (discovery sensitivity, both rows).
    _ylfs = 5.2
    pos_f = sub_fg[0].get_position(fig)
    fig.text(pos_f.x0 - 0.045, 0.5 * (pos_f.y0 + pos_f.y1),
             r"Observed $-\log_{10}(p)$", fontsize=_ylfs,
             ha="right", va="center", rotation=90)
    pos_i0 = sub_hi[0, 0].get_position(fig)
    pos_i1 = sub_hi[1, 0].get_position(fig)
    fig.text(pos_i0.x0 - 0.045, 0.5 * (pos_i1.y0 + pos_i0.y1),
             "FDR-controlled\ndiscovery sensitivity", fontsize=_ylfs,
             ha="right", va="center", rotation=90)

    # Panel titles centred above each cell (composite renderers suppress their own titles).
    def _title(sp, text, y_offset=0.005):
        pos = sp.get_position(fig)
        fig.text(0.5 * (pos.x0 + pos.x1), min(pos.y1 + y_offset, 0.995), text,
                 fontsize=6.0, fontweight="bold", va="bottom", ha="center")

    _title(sub_ab[0], "Mixed-signal null-gene FPR (balanced)", y_offset=0.062)
    _title(sub_ab[1], "Mixed-signal null-gene FPR (one-directional)", y_offset=0.062)
    _title(sub_fg[0], "Null-gene p-value QQ (200 genes, 10% signal)", y_offset=0.018)
    _title(sub_fg[1], "Null-gene calibration: % of p-values outside 95% CI", y_offset=0.018)
    _title(sub_hi[0, 0], "FDR-controlled discovery sensitivity (end-to-end TPR)", y_offset=0.010)
    _title(sub_hi[0:2, 1], "Marginal detection probability", y_offset=0.010)

    # Panel letters A-N in sequential visual order.
    for lab, sp in [
        ("A", sub_ab[0]),  ("B", sub_ab[1]),
        ("C", sub_dgh[0]), ("D", sub_dgh[1]), ("E", sub_dgh[2]),
        ("F", sub_fg[0]),  ("G", sub_fg[1]),
        ("H", sub_hi[0, 0]), ("I", sub_hi[0:2, 1]),
        ("J", cell["J"]),
        ("K", sub_lm[0]),  ("L", sub_lm[1]),
        ("M", sub_no[0]),  ("N", sub_no[1]),
    ]:
        pos = sp.get_position(fig)
        # Further left + higher than the axes corner so the letter clears the
        # rotated y-axis labels of the bar panels (L-O).
        fig.text(max(pos.x0 - 0.064, 0.001), min(pos.y1 + 0.010, 0.998), lab,
                 fontsize=8.5, fontweight="bold", va="bottom", ha="left")

    save_panel(fig, FIGURE_NAME, FIGURE_NAME, SUPP_OUTPUT, close=False)
    pdf_path = SUPP_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {FIGURE_NAME} composite (A-O) saved")


def generate() -> None:
    apply_style()
    print(f"{FIGURE_NAME}: full benchmark calibration/power/robustness")
    # Include the one-directional composition-stress architecture: the mixed-FPR
    # panel shows balanced AND one-directional side by side. Every other panel
    # filters architecture internally (balanced only), so this only feeds panel A.
    bench = _load_benchmark_data(architectures=("balanced", "heterogeneous", "one_directional"))
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

    print(f"  {FIGURE_NAME}: individual panels A-M saved; assembling composite")
    _composite(bench, core)
    print(f"  {FIGURE_NAME} complete")
    gc.collect()
