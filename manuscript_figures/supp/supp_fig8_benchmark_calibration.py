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
from .._shared import MANUSCRIPT_DIR, SUPP_OUTPUT, apply_style, despine, save_panel

FIGURE_NAME = "SuppFig8_benchmark_calibration"
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
    _lbl = 5.4 if composite else 11
    _ttl = 6.0 if composite else 12
    _ann = 4.6 if composite else 8
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
    ax.annotate(rf"calibrated $\sigma_u$={su_cal:.3f}", (su_cal, 0.02),
                fontsize=_ann, ha="right", va="bottom", color="#555", rotation=90)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.03, su_cal + 0.06)
    ax.set_xlabel(r"Participant$\times$visit s.d. $\sigma_u$ (omitted by NEBULA)", fontsize=_lbl)
    ax.set_ylabel("NEBULA Type I error (p < 0.05)", fontsize=_lbl)
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
            (0.40, 0.30), xycoords="axes fraction", fontsize=_ann, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.95))
    despine(ax)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
    # No provenance stamp on the figure itself: provenance lives in provenance.json
    # under the diagnostic tag; a stamp inside the axes read as a stray watermark.
    del prov


def _composite(bench, core) -> None:
    """Assemble the full A-O benchmark artboard (one figure, shared toolkit).

    Every panel is embedded via the same renderers used for the standalone panels
    (gs_parent / ax + composite=True), so no panel code is duplicated. A single
    method key sits at the top and applies to every panel (the colours are
    identical throughout); the few panels with a NON-method key of their own (QQ
    beta-envelope, tested-only vs end-to-end, bias/RMSE) keep it.
    """
    from .._benchmark import _bench_legend_handles
    _mm = 1.0 / 25.4
    fig = plt.figure(figsize=(180 * _mm, 548 * _mm))

    # Content rows separated by generous spacer rows so each panel's title clears
    # the content above and the faceted strip titles below, and nothing overlaps.
    layout = [
        ("top", 0.42), ("s", 0.34),
        ("A", 1.30), ("s", 0.74),
        ("B", 1.30), ("s", 0.78),
        ("CD", 1.30), ("s", 0.80),
        ("E", 1.75), ("s", 0.68),
        ("F", 1.55), ("s", 0.78),
        ("GH", 1.42), ("s", 0.78),
        ("I", 1.85), ("s", 0.74),
        ("J", 1.35), ("s", 0.80),
        ("K", 1.62), ("s", 0.70),
        ("L", 1.08), ("s", 0.74),
        ("M", 1.08), ("s", 0.74),
        ("N", 1.08), ("s", 0.74),
        ("O", 1.08),
    ]
    hr = [h for _, h in layout]
    outer = fig.add_gridspec(len(layout), 1, height_ratios=hr,
                             left=0.085, right=0.965, top=0.992, bottom=0.008, hspace=0.0)
    cell = {name: outer[i] for i, (name, _) in enumerate(layout) if name != "s"}

    # --- calibration ---
    _panel_bench_mixed_fpr(fig, bench, composite=True, gs_parent=cell["A"], architecture="balanced")
    _panel_bench_mixed_fpr(fig, bench, composite=True, gs_parent=cell["B"], architecture="one_directional")
    sub_cd = cell["CD"].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.36)
    _panel_bench_typeI_main(fig, core, composite=True, gs_parent=sub_cd[0])
    _panel_bench_pure_null_fpr(fig, bench, composite=True, gs_parent=sub_cd[1])
    # --- distributional ---
    _panel_bench_qq_single(fig, bench, n_genes=200, signal_pct=10, composite=True, gs_parent=cell["E"])
    _panel_bench_qq_heatmap(fig, bench, composite=True, gs_parent=cell["F"])
    # --- NEBULA mechanism | end-to-end vs tested ---
    sub_gh = cell["GH"].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.34)
    _panel_nebula_hierarchy(fig, ax=fig.add_subplot(sub_gh[0]), composite=True)
    _panel_bench_endtoend_vs_tested(fig.add_subplot(sub_gh[1]), core, composite=True)
    # --- power / discovery / estimation ---
    _panel_bench_power_vs_n(fig, core, composite=True, gs_parent=cell["I"])
    _panel_bench_discovery_sensitivity(fig, bench, composite=True, gs_parent=cell["J"])
    _panel_bench_signal_rmse(fig, bench, composite=True, gs_parent=cell["K"])
    # --- robustness families (bars) ---
    ax_l = fig.add_subplot(cell["L"])
    _panel_bench_scenario_families(ax_l, core, composite=True)
    ax_l.set_title("Null-gene FPR across robustness families", fontsize=6.0, fontweight="bold", pad=4)
    _panel_bench_family_tpr(fig.add_subplot(cell["M"]), core, composite=True)
    _panel_bench_quality(fig.add_subplot(cell["N"]), core, kind="evaluability", composite=True)
    _panel_bench_quality(fig.add_subplot(cell["O"]), core, kind="convergence", composite=True)

    fig.canvas.draw()

    # Figure title + ONE shared method key in the top band (applies to every
    # panel; per-panel method legends would collide in a stack this dense).
    p_top = cell["top"].get_position(fig)
    fig.text(0.5, p_top.y1, "Supplementary Figure 8  |  NatMeth benchmark: calibration, power and robustness",
             ha="center", va="top", fontsize=8.5, fontweight="bold")
    fig.legend(handles=_bench_legend_handles(), loc="center",
               bbox_to_anchor=(0.5, 0.5 * (p_top.y0 + p_top.y1) - 0.004),
               ncol=5, frameon=True, framealpha=0.95, edgecolor="#cccccc",
               fontsize=6.2, columnspacing=1.3, handlelength=1.6)

    # Panel titles for the panels whose renderer draws none in composite mode,
    # centred above the cell (clear of the faceted strip titles).
    def _title(sp, text):
        pos = sp.get_position(fig)
        fig.text(0.5 * (pos.x0 + pos.x1), min(pos.y1 + 0.005, 0.995), text,
                 fontsize=6.0, fontweight="bold", va="bottom", ha="center")

    _title(cell["A"], "Mixed-signal null-gene FPR (balanced)")
    _title(cell["B"], "Mixed-signal null-gene FPR (one-directional)")
    _title(sub_cd[0], "Pure-null Type I error vs participants")
    _title(cell["E"], "Null-gene p-value QQ (200 genes, 10% signal)")
    _title(cell["F"], "Null-gene calibration: % of p-values outside 95% CI")
    _title(cell["I"], "Marginal detection probability")
    _title(cell["J"], "FDR-controlled discovery sensitivity (end-to-end TPR)")

    # Panel letters A-O at each cell's top-left.
    for lab, sp in [
        ("A", cell["A"]), ("B", cell["B"]), ("C", sub_cd[0]), ("D", sub_cd[1]),
        ("E", cell["E"]), ("F", cell["F"]), ("G", sub_gh[0]), ("H", sub_gh[1]),
        ("I", cell["I"]), ("J", cell["J"]), ("K", cell["K"]), ("L", cell["L"]),
        ("M", cell["M"]), ("N", cell["N"]), ("O", cell["O"]),
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
