"""Shared NatMeth benchmark plotting toolkit.

Manifest-addressed loaders, aggregation helpers, and the benchmark panel
renderers shared by Figure 3 and the benchmark supplementary figure. No figure
module imports another figure's code; both import the renderers from here, so
there is exactly one copy of each renderer.
"""
from __future__ import annotations

import warnings
from pathlib import Path  # noqa: F401

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib.patches import Patch  # noqa: F401
from matplotlib.ticker import MaxNLocator, MultipleLocator  # noqa: F401
from scipy import stats

from ._shared import MANUSCRIPT_DIR, despine

warnings.filterwarnings("ignore")

# ======================================================================
# NatMeth signal-fraction benchmark (panels C–E)
# ======================================================================

# Derive from MANUSCRIPT_DIR (which honours SCTRIAL_MANUSCRIPT_DIR) rather than a
# hardcoded parents[4]: the checkout depth differs between the local tree and the
# HPC, where parents[4] pointed outside the project and silently blanked the
# benchmark panels.
# Results are addressed by the MANIFEST that produced them, never by "latest" or
# by newest-file. The frozen configuration names the manifest; the loader reads
# that directory and no other. A glob would happily pick up a development run
# that finished after the definitive one, which is how stale results reached
# figures before.
_RESULTS_ROOT = MANUSCRIPT_DIR / "benchmark" / "results"
_FROZEN_CONFIG = MANUSCRIPT_DIR / "benchmark" / "validation" / "frozen_simulator_config.json"


def _frozen_manifest_sha() -> str:
    """The manifest hash of the frozen configuration."""
    import json as _json

    if not _FROZEN_CONFIG.exists():
        raise FileNotFoundError(
            f"no frozen configuration at {_FROZEN_CONFIG}. Benchmark figures are "
            "drawn only from a frozen run; run scripts/calibrate_simulator.py freeze."
        )
    m = (_json.loads(_FROZEN_CONFIG.read_text()).get("manifest") or {})
    sha = m.get("manifest_sha256") or m.get("config_sha256")
    if not sha:
        raise ValueError(f"{_FROZEN_CONFIG} carries no manifest hash")
    return str(sha)

# Derived from the benchmark package rather than restated here. These four dicts
# were previously hand-maintained copies, and a method added to CORE_METHODS
# silently vanished from every panel until all of them were updated. The focal
# method must come LAST: the ordering controls both draw order and the legend,
# and sctrial has to render on top of the others.
_BENCH_METHOD_LABELS = {
    "sctrial_did": "sctrial (DiD)",
    "dreamlet": "dreamlet",
    "nebula": "NEBULA",
    "wilcoxon_paired": "Wilcoxon (Δ scores)",
    "limma_voom": "limma-voom",
    "edger_qlf": "edgeR-QLF",
}
_BENCH_METHOD_COLORS = {
    "sctrial_did": "#1f77b4", "dreamlet": "#d62728", "nebula": "#ff7f0e",
    "wilcoxon_paired": "#2ca02c", "limma_voom": "#9467bd", "edger_qlf": "#8c564b",
}
_BENCH_METHOD_MARKERS = {
    "sctrial_did": "o", "dreamlet": "D", "nebula": "s",
    "wilcoxon_paired": "^", "limma_voom": "v", "edger_qlf": "P",
}


def _bench_methods() -> list[str]:
    """Reported methods, in draw order, taken from the benchmark package.

    Raises rather than silently dropping a method that has no style defined, so
    adding one to CORE_METHODS cannot make it disappear from the figures.
    """
    from sctrial.benchmark.orchestrator import CORE_METHODS

    missing = [m for m in CORE_METHODS if m not in _BENCH_METHOD_LABELS]
    if missing:
        raise ValueError(
            f"no plotting style defined for benchmark method(s) {missing}; add them "
            "to _BENCH_METHOD_LABELS/_COLORS/_MARKERS rather than letting them be "
            "dropped from every panel"
        )
    others = [m for m in CORE_METHODS if m != "sctrial_did"]
    return others + (["sctrial_did"] if "sctrial_did" in CORE_METHODS else [])


_BENCH_METHODS = _bench_methods()

_PANEL_SIZES = [50, 200, 500, 2000]
# Exactly realisable at every panel size, so the grid is a complete factorial.
# 1% and 5% are not (50 x 1% = 0.5 genes), and labelling one gene out of 50 as
# "1%" is how an apparent panel-size dependence was manufactured before.
_SIGNAL_FRACTIONS = [2, 4, 10, 20]


def _load_benchmark_data(architectures=("balanced", "heterogeneous")) -> pd.DataFrame:
    from sctrial.benchmark.paths import require_layout

    layout = require_layout(_RESULTS_ROOT, _frozen_manifest_sha())
    csv = layout.combined_csv("sensitivity_combined.csv")
    if not csv.exists():
        raise FileNotFoundError(
            f"Benchmark results not found at {csv}.\n"
            "Run the signal-fraction sensitivity benchmark on HPC first."
        )
    # The completion record is written ONLY by the aggregator, and only after it
    # has verified that the shards form exactly the expected scenario set under
    # one manifest, with a valid completion record per scenario. Without it, this
    # file may hold a single shard -- which is what a partial grid looks like:
    # plausible, and quietly missing 75% of the benchmark.
    # The WHOLE-BENCHMARK marker, not this grid's. A grid marker only says the
    # sensitivity aggregation succeeded; the core grid could have failed entirely
    # and these panels would still be drawn. The finalizer writes the publication
    # marker only after checking that the union of both grids equals the frozen
    # expected scenario set exactly, with no overlap and one manifest.
    _pub = layout.publication_marker()
    if not _pub.exists():
        raise FileNotFoundError(
            f"no publication completion marker ({_pub.name}) for manifest "
            f"{layout.manifest_sha[:12]}. Figures are drawn only from a benchmark "
            "verified complete across ALL grids by "
            "scripts/finalize_benchmark.py; a grid-level marker is not sufficient, "
            "because one grid can succeed while another never finishes."
        )
    _complete = layout.completion_marker("sensitivity")
    if not _complete.exists():
        raise FileNotFoundError(
            f"{csv} has no completion record ({_complete.name}). It was "
            "not produced by scripts/aggregate_benchmark.py and may be a single "
            "shard of the grid. Re-run the aggregator."
        )

    df = pd.read_csv(csv, low_memory=False)

    from sctrial.benchmark.manifest import assert_single_manifest

    assert_single_manifest(df, "benchmark results")

    # Read the DATA, not the scenario NAME. Parsing `_g(\d+)` and `_f(\d+)` out of
    # the name has two failure modes that have both already occurred:
    #   1. the nominal label is not the realised fraction -- at 50 genes,
    #      round(50 * 0.01) is 1 gene, i.e. 2%, so the "1%" column was really 2%
    #      and this manufactured an apparent panel-size dependence;
    #   2. any new scenario whose name happens to contain `_f<N>` is silently
    #      swept into these panels. The one-directional composition-stress arm is
    #      named `sens_g200_f20_onedir` and `_f(\d+)` matches it.
    # The runner now records panel_size, signal_fraction_realised and
    # architecture as columns, so nothing needs to be inferred from a string.
    required = {"panel_size", "signal_fraction_realised", "architecture"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv} is missing {sorted(missing)}. It predates the "
            "current runner and its scenario labels cannot be trusted; re-run the "
            "benchmark rather than parsing the scenario name."
        )
    df["n_genes"] = df["panel_size"].astype(int)
    df["signal_pct"] = (df["signal_fraction_realised"] * 100).round().astype(int)
    df["is_null_scenario"] = df["signal_fraction_realised"] == 0

    # Which signal architectures to keep. Figure 3's condensed panels take the
    # PRIMARY (balanced) architecture only; the benchmark supplement additionally
    # requests the one-directional composition-stress arm for its side-by-side
    # panel, so the caller names the architectures explicitly rather than the
    # loader hard-coding a single set (which silently blanked the one-directional
    # supplementary panel).
    df = df[df["architecture"].isin(architectures)].copy()
    return df


def _load_core_benchmark_data() -> pd.DataFrame:
    """The CORE grid: sample-size sweep, cell yield, missing, imbalance, families.

    Distinct from the sensitivity grid (panel size x signal fraction). Same
    manifest, same publication marker: a core file without the whole-benchmark
    completion record is refused, exactly as for the sensitivity file.

    Derived columns are taken from DATA, never a scenario name, wherever a column
    exists: design from n_control (0 => single-arm), sample size from
    n_participants. Only `family` and `beta`, which have no column, come from the
    scenario id -- and that is legitimate, because the scenario id IS the family
    label; the rule forbids inferring panel_size/signal_fraction from names, not
    grouping by the scenario itself.
    """
    from sctrial.benchmark.paths import require_layout

    layout = require_layout(_RESULTS_ROOT, _frozen_manifest_sha())
    csv = layout.combined_csv("benchmark_combined.csv")
    if not csv.exists():
        raise FileNotFoundError(
            f"Core benchmark results not found at {csv}. Run the core grid on HPC."
        )
    if not layout.publication_marker().exists():
        raise FileNotFoundError(
            f"no publication completion marker for manifest {layout.manifest_sha[:12]}; "
            "the core grid panels are drawn only from a benchmark verified complete "
            "across ALL grids by scripts/finalize_benchmark.py."
        )
    if not layout.completion_marker("core").exists():
        raise FileNotFoundError(
            f"{csv} has no core completion record; re-run scripts/aggregate_benchmark.py."
        )
    df = pd.read_csv(csv, low_memory=False)

    from sctrial.benchmark.manifest import assert_single_manifest

    assert_single_manifest(df, "core benchmark results")

    for col in ("n_participants", "n_control", "n_signal_realised"):
        if col not in df.columns:
            raise ValueError(f"{csv} lacks {col}; re-run the benchmark.")
    df["design"] = np.where(df["n_control"] == 0, "single_arm", "two_arm")
    df["total_n"] = df["n_participants"].astype(int)
    df["is_null_gene"] = df["true_beta"] == 0.0
    # family: scenario id with the design prefix and the trailing _n{n}[_b{beta}]
    # stripped. beta: the effect magnitude where present.
    base = df["scenario"].str.replace(r"^(two_arm|single_arm)__", "", regex=True)
    df["family"] = base.str.replace(r"_n\d+.*$", "", regex=True)
    df["beta"] = base.str.extract(r"_b([\d.]+)$")[0].astype(float)
    return df


def _per_scenario_rate(df: pd.DataFrame, *, on_signal: bool, alpha: float = 0.05):
    """Per-(scenario, method) rejection rate with the CORRECT hierarchy.

    Endpoint per replicate (fraction of the relevant genes with p<alpha), then
    mean and Monte Carlo SE across replicates. Never pools gene rows across
    replicates or scenarios. Returns mean, mcse, n_rep per (scenario, method).
    """
    sub = df[df["is_signal"] == on_signal].copy()
    ok = sub["pvalue"].notna()
    sub = sub[ok]
    sub["hit"] = (sub["pvalue"] < alpha).astype(float)
    per_rep = (sub.groupby(["scenario", "method", "iteration"])["hit"]
               .mean().reset_index())
    agg = (per_rep.groupby(["scenario", "method"])["hit"]
           .agg(mean="mean", sd="std", n_rep="count").reset_index())
    agg["mcse"] = agg["sd"] / np.sqrt(agg["n_rep"].clip(lower=1))
    return agg


def _plot_offscale(ax, x, y, *, method, ymax, style, label=None, annotate=True):
    """Plot a method's series, clipping values above ymax to the axis top and
    annotating the true value, so NEBULA (which is off-scale on any axis scaled
    for the calibrated methods) is shown as extreme rather than allowed to blow
    out the shared axis or vanish above it.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    clipped = np.minimum(y, ymax)
    ax.plot(x, clipped, label=label, **style)
    if annotate:
        over = y > ymax
        for xi, yi, c in zip(x, y, over):
            if c:
                ax.annotate(f"{yi:.2f}", (xi, ymax), fontsize=5.0,
                            ha="center", va="bottom",
                            color=style.get("color", "#333"), rotation=0,
                            xytext=(0, 1), textcoords="offset points", clip_on=False)


def _method_style(method: str, is_focal: bool = False, alpha: float = 1.0, *, composite: bool = False):
    if composite:
        ms_hi, ms_lo = 5.6, 4.3
        lw_hi, lw_lo = 1.45, 1.1
        mew = 0.48
    else:
        ms_hi, ms_lo = 9, 7
        lw_hi, lw_lo = 2.5, 1.8
        mew = 0.6
    return {
        "color": _BENCH_METHOD_COLORS[method],
        "marker": _BENCH_METHOD_MARKERS[method],
        "markersize": ms_hi if is_focal else ms_lo,
        "markeredgecolor": "white",
        "markeredgewidth": mew,
        "linewidth": lw_hi if is_focal else lw_lo,
        "alpha": alpha,
    }


def _add_nominal_band(ax, level: float = 0.05, low: float = 0.03, high: float = 0.07, color: str = "#555555"):
    # A single neutral-gray dashed reference at the nominal level. The shaded
    # 0.03-0.07 band is intentionally dropped: it collided with dreamlet's red
    # series near 0.05, and only some panels carried it, so every calibration
    # panel now shows the same gray line-only reference (low/high kept for the
    # call signature).
    del low, high
    ax.axhline(level, color=color, linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)


def _style_axis(ax) -> None:
    ax.grid(axis="y", linestyle=":", color="#b0b0b0", alpha=0.45, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", color="#333333", width=0.8, length=4)


def _compute_signal_bias_rmse_table(bench_df: pd.DataFrame) -> pd.DataFrame:
    # Balanced architecture only: the bias/RMSE panel describes the primary
    # architecture, and the supplement now loads one-directional in the same frame
    # (for the mixed-FPR panel), which must not leak into this estimand summary.
    if "architecture" in bench_df.columns:
        bench_df = bench_df[bench_df["architecture"] == "balanced"]
    mixed = bench_df[~bench_df["is_null_scenario"]].copy()
    sig = mixed[mixed["true_beta"] != 0.0].dropna(subset=["estimated_beta"]).copy()
    sig["err"] = sig["estimated_beta"] - sig["true_beta"]
    sig["sq_err"] = sig["err"] ** 2
    rows = []
    for (method, n_g, frac), grp in sig.groupby(["method", "n_genes", "signal_pct"]):
        if grp.empty:
            continue
        rows.append({"method": method, "n_genes": int(n_g), "signal_pct": int(frac),
                     "bias": float(grp["err"].mean()), "rmse": float(np.sqrt(grp["sq_err"].mean())),
                     "n_tests": int(len(grp))})
    return pd.DataFrame(rows)


def _panel_bench_lambda_gc(ax, bench_df: pd.DataFrame, *, composite: bool = False) -> None:
    null_scenarios = bench_df[bench_df["is_null_scenario"]]
    pvals_pure = null_scenarios[null_scenarios["true_beta"] == 0.0]
    rows = []
    for (method, n_g), grp in pvals_pure.groupby(["method", "n_genes"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) < 50:
            continue
        chi2_obs = stats.chi2.isf(pvals, df=1)
        chi2_obs = chi2_obs[np.isfinite(chi2_obs)]
        if len(chi2_obs) < 50:
            continue
        lam = float(np.median(chi2_obs) / stats.chi2.ppf(0.5, df=1))
        rows.append({"method": method, "n_genes": int(n_g), "lambda_gc": lam})
    lam_df = pd.DataFrame(rows)
    x_positions = np.arange(len(_PANEL_SIZES), dtype=float)
    n_to_x = dict(zip(_PANEL_SIZES, x_positions))

    _lbl_fs = 5.15 if composite else 11
    _ttl_fs = 6.0 if composite else 12
    _ttl_pad = 5 if composite else 10
    _leg_fs = 5.2 if composite else 9

    _lam_top = 1.17  # axis scaled for the CALIBRATED methods
    for method in _BENCH_METHODS:
        sub = lam_df[lam_df["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_focal = method == "sctrial_did"
        style = _method_style(method, is_focal=is_focal, composite=composite)
        xs = np.array([n_to_x[int(n)] for n in sub["n_genes"].values], float)
        lam = sub["lambda_gc"].to_numpy(float)
        # NEBULA's lambda_GC is far off any axis scaled for the calibrated
        # methods; clip to the top and annotate the true value rather than let it
        # set the range and flatten everyone else to a line.
        ax.plot(xs, np.minimum(lam, _lam_top), label=_BENCH_METHOD_LABELS[method],
                zorder=10 if is_focal else 3, **style)
        for xi, li in zip(xs, lam):
            if li > _lam_top:
                ax.annotate(f"{li:.1f}", (xi, _lam_top), fontsize=(4.6 if composite else 7),
                            ha="center", va="bottom", color=style["color"],
                            xytext=(0, 1), textcoords="offset points", clip_on=False)

    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.65, zorder=1)
    ax.axhspan(0.95, 1.05, color="#d62728", alpha=0.06, zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p:,}" for p in _PANEL_SIZES], fontsize=_lbl_fs)
    ax.set_xlim(-0.35, len(_PANEL_SIZES) - 0.65)
    ax.set_xlabel("Panel size (genes)", fontsize=_lbl_fs)
    ax.set_ylabel(r"Genomic inflation factor ($\lambda_{\mathrm{GC}}$)", fontsize=_lbl_fs)
    ax.set_title("Pure-null calibration across panel sizes", fontsize=_ttl_fs, fontweight="bold", pad=_ttl_pad)
    ax.set_ylim(0.88, 1.18)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.tick_params(axis="y", labelsize=_lbl_fs)
    if composite:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=True, framealpha=0.95,
                  edgecolor="#cccccc", fontsize=_leg_fs, markerscale=0.52, handlelength=1.0, ncol=2)
    else:
        ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cccccc",
                  fontsize=_leg_fs, markerscale=1.0, handlelength=1.5)
    _style_axis(ax)


def _panel_bench_runtime(ax, bench_df: pd.DataFrame, *, composite: bool = False) -> None:
    """Per-iteration runtime by method × panel size (log y).

    X-axis uses evenly-spaced categorical positions so the 4 panel sizes
    are ticked at equal intervals, independent of their raw values.
    """
    # Aggregation, as specified: median runtime over replicates WITHIN each
    # scenario, then an equal-weight summary (median + IQR) of the scenario
    # medians within a tested-set size, so scenarios are weighted equally rather
    # than by their replicate count. Individual scenario medians are shown as
    # points, the equal-weight median as the line, and the IQR as a band.
    per_iter = (bench_df.groupby(["method", "scenario", "n_genes", "iteration"])
                ["runtime_seconds"].first().reset_index())
    scen_med = (per_iter.groupby(["method", "scenario", "n_genes"])["runtime_seconds"]
                .median().reset_index())
    summary = (scen_med.groupby(["method", "n_genes"])["runtime_seconds"]
               .agg(med="median", q25=lambda s: s.quantile(0.25),
                    q75=lambda s: s.quantile(0.75)).reset_index())
    x_positions = np.arange(len(_PANEL_SIZES), dtype=float)
    n_to_x = dict(zip(_PANEL_SIZES, x_positions))

    _lbl_fs = 6.2 if composite else 11
    _ttl_fs = 7.0 if composite else 12
    _ttl_pad = 5 if composite else 10
    _leg_fs = 6.0 if composite else 9

    for method in _BENCH_METHODS:
        sub = summary[summary["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_focal = method == "sctrial_did"
        style = _method_style(method, is_focal=is_focal, composite=composite)
        xs = np.array([n_to_x[int(n)] for n in sub["n_genes"].values], float)
        # scenario-median points (small, jittered slightly) behind the summary
        pts = scen_med[scen_med["method"] == method]
        jit = (hash(method) % 5 - 2) * 0.03
        ax.scatter([n_to_x[int(n)] + jit for n in pts["n_genes"]],
                   pts["runtime_seconds"], s=(4 if composite else 14),
                   color=style["color"], alpha=0.38, edgecolors="none",
                   rasterized=True, zorder=2)
        ax.errorbar(xs, sub["med"],
                    yerr=[sub["med"] - sub["q25"], sub["q75"] - sub["med"]],
                    fmt="none", ecolor=style["color"], elinewidth=0.8,
                    capsize=1.5, alpha=0.6, zorder=4)
        ax.plot(xs, sub["med"], label=_BENCH_METHOD_LABELS[method],
                zorder=10 if is_focal else 5, **style)

    # Endpoint speed ratios (relative to sctrial) at the largest tested-set size,
    # as a single annotation rather than per-point clutter. Implementation- and
    # hardware-specific; the caption states the timing boundary and configuration.
    big = summary[summary["n_genes"] == max(_PANEL_SIZES)].set_index("method")["med"]
    if "sctrial_did" in big.index and big["sctrial_did"] > 0:
        base = big["sctrial_did"]
        # Every other method's fold-change vs sctrial, slowest first. NEBULA is
        # INCLUDED (it is the slowest, not the fastest, so there is no risk of
        # implying a favourable runtime for a miscalibrated method); omitting it
        # read as selectively hiding the worst-case cost.
        lines = [f"at {max(_PANEL_SIZES):,} genes vs sctrial:"]
        ratios = sorted(((m, big[m] / base) for m in big.index if m != "sctrial_did"),
                        key=lambda kv: kv[1], reverse=True)
        for mth, r in ratios:
            lines.append(f"  {_BENCH_METHOD_LABELS[mth]}: {r:.0f}x")
        # Lower-right corner: empty at the largest tested-set size (all lines are
        # high there), so the ratio box never collides with the upper-left legend.
        ax.text(0.97, 0.03, "\n".join(lines), transform=ax.transAxes,
                fontsize=(6.0 if composite else 7.5), va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.9))

    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p:,}" for p in _PANEL_SIZES], fontsize=_lbl_fs)
    ax.set_xlim(-0.35, len(_PANEL_SIZES) - 0.65)
    ax.set_xlabel("Number of tested genes", fontsize=_lbl_fs)
    ax.set_ylabel("Wall-clock seconds per simulated dataset", fontsize=_lbl_fs)
    ax.set_title("Runtime scaling", fontsize=_ttl_fs, fontweight="bold", pad=_ttl_pad)
    ax.tick_params(axis="y", labelsize=_lbl_fs)
    # Canonical legend order (sctrial, Wilcoxon, limma-voom, dreamlet, NEBULA),
    # not the draw order, so every panel's legend reads the same.
    if composite:
        ax.legend(
            handles=_bench_legend_handles(), loc="upper left",
            bbox_to_anchor=(0.02, 1.04),
            frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=_leg_fs,
            markerscale=0.52, handlelength=1.0,
        )
    else:
        ax.legend(
            handles=_bench_legend_handles(), loc="upper left", frameon=True,
            framealpha=0.95, edgecolor="#cccccc", fontsize=_leg_fs,
            markerscale=1.0, handlelength=1.5,
        )
    _style_axis(ax)


def _panel_bench_signal_rmse(fig, bench_df: pd.DataFrame, *, composite: bool = False,
                             gs_parent=None) -> None:
    df = _compute_signal_bias_rmse_table(bench_df)
    if hasattr(fig, "set_constrained_layout"):
        fig.set_constrained_layout(False)
    # Lower the panel top so the method legend sits ABOVE the column titles
    # instead of colliding with them. When embedded in a composite cell, build a
    # subgridspec inside that cell rather than a figure-wide gridspec.
    if gs_parent is not None:
        gs = gs_parent.subgridspec(2, 4, hspace=0.52, wspace=0.18)
    elif composite:
        gs = fig.add_gridspec(2, 4, hspace=0.52, wspace=0.18, left=0.07, right=0.99, top=0.78, bottom=0.16)
    else:
        gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.22, left=0.08, right=0.985, top=0.80, bottom=0.11)

    _ttl_fs = 6.35 if composite else 12
    _yl_fs = 5.0 if composite else 11
    _axis_fs = 4.2 if composite else 10
    _xlab_fs = 5.45 if composite else 10
    # All FIVE methods (the previous list dropped limma-voom). Each method's bias
    # is against its OWN oracle (METHOD_ESTIMAND); this is NOT a cross-method
    # bias ranking, and the caption must say so.
    bar_width = 0.16
    # Canonical bar/legend order (sctrial first); bars have no z-order concern, so
    # this is the display order used everywhere else.
    method_order = list(_LEGEND_ORDER)
    x_positions = np.arange(len(_SIGNAL_FRACTIONS))
    bias_lo = min(df["bias"].min(), 0) - 0.02
    bias_hi = max(df["bias"].max(), 0.02) * 1.12
    # Adaptive bias tick step: a fixed 0.05 spacing left only the "0.00" tick when
    # every method's bias is within +-0.02, making the vertical scale unreadable.
    _bias_span = max(abs(bias_lo), abs(bias_hi))
    _bias_step = 0.05 if _bias_span > 0.12 else (0.02 if _bias_span > 0.045 else 0.01)
    rmse_hi = df["rmse"].max() * 1.18
    bias_axes = []
    rmse_axes = []
    for col, n_g in enumerate(_PANEL_SIZES):
        ax_bias = fig.add_subplot(gs[0, col])
        ax_rmse = fig.add_subplot(gs[1, col])
        bias_axes.append(ax_bias)
        rmse_axes.append(ax_rmse)
        sub = df[df["n_genes"] == n_g]
        for mi, method in enumerate(method_order):
            bias_vals = []
            rmse_vals = []
            for frac in _SIGNAL_FRACTIONS:
                cell = sub[(sub["method"] == method) & (sub["signal_pct"] == frac)]
                if len(cell):
                    bias_vals.append(float(cell["bias"].iloc[0]))
                    rmse_vals.append(float(cell["rmse"].iloc[0]))
                else:
                    bias_vals.append(np.nan)
                    rmse_vals.append(np.nan)
            offset = (mi - (len(method_order) - 1) / 2) * bar_width
            ax_bias.bar(x_positions + offset, bias_vals, bar_width,
                        color=_BENCH_METHOD_COLORS[method], edgecolor="white", linewidth=0.6, zorder=3)
            ax_rmse.bar(x_positions + offset, rmse_vals, bar_width,
                        color=_BENCH_METHOD_COLORS[method], edgecolor="white", linewidth=0.6, zorder=3)
        ax_bias.axhline(0.0, color="#222222", linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
        ax_bias.set_xticks(x_positions)
        ax_bias.set_xticklabels([])
        ax_bias.set_ylim(-0.03, 0.01)
        ax_bias.yaxis.set_major_locator(MultipleLocator(_bias_step))
        ax_bias.set_title(f"{n_g:,} genes", fontsize=_ttl_fs, fontweight="bold",
                          color="#1a1a1a", pad=(-7 if composite else 8))
        _style_axis(ax_bias)
        ax_rmse.set_xticks(x_positions)
        ax_rmse.set_xticklabels([f"{f}%" for f in _SIGNAL_FRACTIONS], fontsize=_axis_fs)
        ax_rmse.set_xlabel("Signal fraction", fontsize=_xlab_fs)
        ax_rmse.set_ylim(0, rmse_hi)
        ax_rmse.yaxis.set_major_locator(MultipleLocator(0.05))
        ax_rmse.tick_params(axis="both", labelsize=_axis_fs)
        _style_axis(ax_rmse)
        ax_bias.tick_params(axis="y", labelsize=_axis_fs)
        if col > 0:
            ax_bias.set_yticklabels([])
            ax_rmse.set_yticklabels([])
    bias_axes[0].set_ylabel(r"Mean bias ($\hat{\beta} - \beta$)", fontsize=_yl_fs)
    rmse_axes[0].set_ylabel(r"RMSE of $\hat{\beta}$", fontsize=_yl_fs)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_BENCH_METHOD_COLORS[m], edgecolor="white",
                       linewidth=0.6, label=_BENCH_METHOD_LABELS[m])
        for m in method_order
    ]
    _leg_fs = 5.2 if composite else 10
    if composite:
        _anchor = bias_axes[0]
        _cx = 2.0
        _anchor.text(_cx, 1.25, "Effect-size estimation accuracy", transform=_anchor.transAxes,
                     ha="center", va="bottom", fontsize=5.9, fontweight="bold")
        _anchor.legend(handles=legend_handles, loc="upper center", ncol=len(method_order),
                       bbox_to_anchor=(_cx, -0.15), bbox_transform=_anchor.transAxes,
                       frameon=True, framealpha=0.93, edgecolor="#cccccc", fontsize=_leg_fs,
                       handlelength=0.85, handleheight=0.42, handletextpad=0.35,
                       columnspacing=0.55, borderpad=0.35)
    else:
        fig.suptitle("Estimator bias and RMSE (per method's own oracle)",
                     fontsize=13, fontweight="bold", y=0.998)
        fig.text(0.5, 0.965, "Balanced signal architecture", ha="center", va="top",
                 fontsize=10, fontstyle="italic", color="#444")
        fig.legend(handles=legend_handles, loc="upper center", ncol=len(method_order),
                   bbox_to_anchor=(0.53, 0.915), frameon=True, framealpha=0.95,
                   edgecolor="#cccccc", fontsize=_leg_fs)


# ======================================================================
# QQ calibration helpers (from Supp Fig 5 panel I)
# ======================================================================

def _compute_frac_outside_ci(pvals):
    """Fraction of null p-value order statistics outside the 95% Beta CI band."""
    pvals = np.sort(np.asarray(pvals, dtype=float))
    n = len(pvals)
    ranks = np.arange(1, n + 1)
    lo = stats.beta.ppf(0.025, ranks, n - ranks + 1)
    hi = stats.beta.ppf(0.975, ranks, n - ranks + 1)
    return float(np.mean((pvals < lo) | (pvals > hi)))


def _panel_bench_qq_single(
    fig, bench_df,
    n_genes: int = 200,
    signal_pct: int = 10,
    *,
    composite: bool = False,
    gs_parent=None,
    suppress_ylabel: bool = False,
):
    """2×2 QQ plots for one (n_genes, signal_pct) condition."""
    # Select by COLUMNS, never by a reconstructed scenario name. The grid names
    # scenarios sens_g{panel}_n{count}; panel size and realised signal fraction
    # are recorded as the n_genes and signal_pct columns precisely so panels do
    # not parse names. Reconstructing "sens_g{n}_f{pct}" matched the OLD naming
    # and would silently return an empty panel on the current data.
    sub_all = bench_df[
        (bench_df["n_genes"] == n_genes)
        & (bench_df["signal_pct"] == signal_pct)
        & (bench_df["architecture"] == "balanced")
    ]
    if sub_all.empty:
        print(f"    WARNING: no balanced scenario for n_genes={n_genes} "
              f"signal_pct={signal_pct}% in the QQ panel")
        return []
    null = sub_all[sub_all["true_beta"] == 0.0]

    # 2x3 so all FIVE methods fit (a 2x2 silently dropped the 5th, sctrial, the
    # focal method). Independent y-axes: a QQ is read against its own diagonal, and
    # NEBULA's inflation reaches -log10(p)~300, which on a shared axis crushes the
    # calibrated methods into an invisible blob. Each method on its own scale shows
    # NEBULA's extremity AND the others' calibration.
    n_methods = len(_BENCH_METHODS)
    if gs_parent is not None:
        gs_inner = gs_parent.subgridspec(2, 3, hspace=1.30, wspace=0.38)
        axes = [fig.add_subplot(gs_inner[r, c]) for r in range(2) for c in range(3)]
    else:
        ax_grid = fig.subplots(2, 3, sharex=False, sharey=False,
                               gridspec_kw={"hspace": 0.52, "wspace": 0.38})
        axes = list(ax_grid.flatten())
    for extra in axes[n_methods:]:
        extra.set_visible(False)

    _sct      = 2.5 if composite else 8
    _ttl_fs   = 5.2 if composite else 12
    _axlbl_fs = 4.0 if composite else 10
    _tick_fs  = 4.0 if composite else 10

    for mi, (ax, method) in enumerate(zip(axes, _LEGEND_ORDER)):
        pvals = (
            null.loc[null["method"] == method, "pvalue"]
            .dropna()
            .sort_values()
            .values
        )
        if len(pvals) == 0:
            continue
        n = len(pvals)
        ranks = np.arange(1, n + 1)
        expected = (ranks - 0.5) / n
        obs_log = -np.log10(pvals + 1e-300)
        exp_log = -np.log10(expected + 1e-300)
        lo_env = -np.log10(stats.beta.ppf(0.975, ranks, n - ranks + 1) + 1e-300)
        hi_env = -np.log10(stats.beta.ppf(0.025, ranks, n - ranks + 1) + 1e-300)
        ax.fill_between(exp_log, lo_env, hi_env, color="#9a9a9a",
                        alpha=0.32, zorder=1)
        ax.scatter(exp_log, obs_log, s=_sct, alpha=0.55,
                   color=_BENCH_METHOD_COLORS[method], edgecolors="none",
                   rasterized=True, zorder=3)
        lim = max(exp_log.max(), obs_log.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#333333", linestyle="--",
                linewidth=0.8, alpha=0.7, zorder=2)
        ax.set_title(_BENCH_METHOD_LABELS[method], fontsize=_ttl_fs,
                     fontweight="bold", color=_BENCH_METHOD_COLORS[method],
                     pad=4, y=1.0)
        ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=_axlbl_fs,
                      labelpad=(1 if composite else 4))
        if mi % 3 == 0 and not suppress_ylabel:
            ax.set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=_axlbl_fs)
        _style_axis(ax)
        ax.tick_params(axis="both", which="major", labelsize=_tick_fs)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

    if axes:
        _leg_fs = 4.8 if composite else 8
        _leg_kw = dict(
            handles=[Patch(facecolor="#9a9a9a", alpha=0.32, edgecolor="none",
                           label="95% Beta envelope")],
            frameon=True, framealpha=0.9, edgecolor="#cccccc",
            fontsize=_leg_fs, handleheight=0.6, handlelength=1.0,
            handletextpad=0.35, borderpad=0.3, borderaxespad=0.0,
        )
        if composite:
            leg = axes[-1].legend(loc="lower right", **_leg_kw)
        else:
            leg = axes[0].legend(loc="upper left", bbox_to_anchor=(0.03, 0.98), **_leg_kw)
        leg.get_frame().set_linewidth(0.55)

    return list(axes)


def _panel_bench_qq_heatmap(fig, bench_df, *, composite: bool = False, gs_parent=None):
    """Calibration summary heatmap: % outside 95% CI per (method, n_genes, signal_pct)."""
    import matplotlib.colors as mcolors

    n_genes_vals = _PANEL_SIZES
    signal_pct_vals = _SIGNAL_FRACTIONS

    _iter_col = "iteration" if "iteration" in bench_df.columns else None
    frac_data = {}
    for method in _LEGEND_ORDER:
        mat = np.full((len(n_genes_vals), len(signal_pct_vals)), np.nan)
        for ri, ng in enumerate(n_genes_vals):
            for ci, sf in enumerate(signal_pct_vals):
                # Column selection, not a reconstructed scenario name (see
                # _panel_bench_qq_single). Balanced architecture, null genes only.
                sub = bench_df[
                    (bench_df["n_genes"] == ng)
                    & (bench_df["signal_pct"] == sf)
                    & (bench_df["architecture"] == "balanced")
                    & (bench_df["method"] == method)
                    & (bench_df["true_beta"] == 0.0)
                ]
                sub = sub.dropna(subset=["pvalue"])
                if len(sub) < 10:
                    continue
                if _iter_col is not None:
                    fracs = []
                    for _, grp in sub.groupby(_iter_col):
                        pv = grp["pvalue"].values
                        if len(pv) >= 5:
                            fracs.append(_compute_frac_outside_ci(pv))
                    if fracs:
                        mat[ri, ci] = float(np.mean(fracs))
                else:
                    mat[ri, ci] = _compute_frac_outside_ci(sub["pvalue"].values)
        frac_data[method] = mat

    all_vals = np.concatenate([v[~np.isnan(v)] for v in frac_data.values()])
    if len(all_vals) == 0:
        print("    WARNING: no calibration data found for panel F heatmap")
        return []
    # The statistic is a proportion in [0, 100%]: the calibrated methods cluster
    # near the nominal 5%, NEBULA sits at 90-100%, and nothing lands in between.
    # A diverging map centred at 0% wasted its whole lower (negative) half and made
    # a well-calibrated 5% read as "mildly miscalibrated". Instead a SEQUENTIAL map
    # on a SPLIT scale that expands 0-12% and 90-100% (compressing the empty
    # 12-90% middle) resolves the small differences among the calibrated methods
    # while still placing NEBULA off in the dark extreme -- the same broken-scale
    # convention the line panels use for NEBULA.
    _brk_d = [0.0, 0.12, 0.90, 1.0]   # data-value break-points
    _brk_c = [0.0, 0.45, 0.55, 1.0]   # colour-map positions

    def _fwd(v):
        return np.interp(np.asarray(v, dtype=float), _brk_d, _brk_c)

    def _inv(t):
        return np.interp(np.asarray(t, dtype=float), _brk_c, _brk_d)

    norm = mcolors.FuncNorm((_fwd, _inv), vmin=0.0, vmax=1.0)
    cmap = "YlOrRd"

    _ttl_fs   = 5.2 if composite else 11
    _axlbl_fs = 4.0 if composite else 9
    _cblbl_fs = 4.6 if composite else 9
    _ann_fs   = 4.2 if composite else 8
    _tick_fs  = 4.0 if composite else 8
    _ytick_fs = 3.2 if composite else 8

    # 2 rows x 3 method columns + a colorbar column, so all FIVE methods appear
    # (a 2x2 grid silently dropped sctrial, the focal method). The 6th cell is
    # hidden.
    n_methods = len(_BENCH_METHODS)
    if gs_parent is not None:
        gs_inner = gs_parent.subgridspec(2, 4, hspace=1.05, wspace=0.45,
                                         width_ratios=[1, 1, 1, 0.06])
        axes = [fig.add_subplot(gs_inner[r, c]) for r in range(2) for c in range(3)]
        cbar_ax = fig.add_subplot(gs_inner[:, 3])
    else:
        gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.32,
                              width_ratios=[1, 1, 1, 0.05],
                              left=0.08, right=0.93, top=0.90, bottom=0.12)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
        cbar_ax = fig.add_subplot(gs[:, 3])
    for extra in axes[n_methods:]:
        extra.set_visible(False)

    col_labels = [f"{sf}%" for sf in signal_pct_vals]
    row_labels = [f"{ng:,}" for ng in n_genes_vals]

    im_last = None
    for mi, (ax, method) in enumerate(zip(axes, _LEGEND_ORDER)):
        mat = frac_data[method]
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto",
                       interpolation="nearest")
        im_last = im

        for ri in range(len(n_genes_vals)):
            for ci in range(len(signal_pct_vals)):
                val = mat[ri, ci]
                if np.isnan(val):
                    txt = "n/a"
                    fc = "#777777"
                else:
                    txt = f"{val * 100:.1f}%"
                    # White text only on the dark (high-%) cells.
                    fc = "white" if val > 0.35 else "#222222"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=_ann_fs, color=fc)

        ax.set_xticks(range(len(signal_pct_vals)))
        ax.set_yticks(range(len(n_genes_vals)))
        ax.set_xticklabels(col_labels, fontsize=_tick_fs)
        ax.set_yticklabels(row_labels, fontsize=_ytick_fs)
        ax.tick_params(length=2, pad=1)

        ax.set_xlabel("Signal fraction", fontsize=_axlbl_fs,
                      labelpad=(1 if composite else 4))
        if mi == 0 and not composite:
            ax.set_ylabel("Genes", fontsize=_axlbl_fs)

        ax.set_title(
            _BENCH_METHOD_LABELS[method],
            fontsize=_ttl_fs, fontweight="bold",
            color=_BENCH_METHOD_COLORS[method], pad=4,
            y=1.0,
        )
        _style_axis(ax)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cb = fig.colorbar(im_last, cax=cbar_ax, orientation="vertical")
    cb.set_label("% outside 95% CI", fontsize=_cblbl_fs, labelpad=2)
    # Ticks mark the split: the expanded low band (0/5/12%) and the NEBULA band
    # (90/100%); the nominal 5% target is drawn as the dashed reference.
    cb.set_ticks([0.0, 0.05, 0.12, 0.90, 1.0])
    cb.ax.tick_params(labelsize=_tick_fs, length=2, pad=1)
    cb.formatter = plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
    cb.update_ticks()
    cb.ax.axhline(0.05, color="#111111", linewidth=0.8, linestyle="--")

    if composite and gs_parent is not None:
        pos = gs_parent.get_position(fig)
        fig.text(pos.x0 - 0.030, 0.5 * (pos.y0 + pos.y1),
                 "Genes tested", fontsize=_axlbl_fs,
                 ha="right", va="center", rotation=90)
    if not composite:
        fig.suptitle(
            "Null-gene p-value calibration: % of null p-values outside 95% CI",
            fontsize=12, fontweight="bold",
        )

    return list(axes)


# ======================================================================
# CORE-GRID panels: sample size, power, scenario families
# ======================================================================

_DESIGN_LABEL = {"two_arm": "Two-arm (DiD)", "single_arm": "Single-arm (paired)"}
# Calibrated methods share the main plotting region; NEBULA is shown on its own
# scale (a strip or separate axis) because its rejection rate is ~0.75, an order
# of magnitude off the nominal band. Clipping it into the calibrated region would
# either flatten it to a line or crush the methods that matter.
_CALIBRATED = [m for m in _BENCH_METHODS if m != "nebula"]

# Legend order is fixed across every benchmark panel: sctrial, Wilcoxon,
# limma-voom, dreamlet, NEBULA (NEBULA omitted where it is not shown). This is
# independent of PLOT order, which keeps the focal method drawn last / on top.
_LEGEND_ORDER = ["sctrial_did", "wilcoxon_paired", "limma_voom", "dreamlet", "nebula"]


# Short one-word method labels for the compact single-row legends beneath the
# half-width composite panels, where the full "(DiD)" / "(Δ scores)" forms would
# not fit on one line.
_BENCH_METHOD_LABELS_SHORT = {
    "sctrial_did": "sctrial",
    "dreamlet": "dreamlet",
    "nebula": "NEBULA",
    "wilcoxon_paired": "Wilcoxon",
    "limma_voom": "limma-voom",
    "edger_qlf": "edgeR-QLF",
}


def _bench_legend_handles(methods=None, *, short=False, markersize=7):
    from matplotlib.lines import Line2D

    methods = methods if methods is not None else _LEGEND_ORDER
    labels = _BENCH_METHOD_LABELS_SHORT if short else _BENCH_METHOD_LABELS
    return [
        Line2D([0], [0], color=_BENCH_METHOD_COLORS[m], marker=_BENCH_METHOD_MARKERS[m],
               linestyle="-", markersize=markersize, markeredgecolor="white",
               markeredgewidth=0.6, label=labels[m])
        for m in methods
    ]


def _bench_figlegend(fig, methods=None, *, y=1.0, fontsize=8):
    """One global legend above the panel, in the standard method order, never
    over the plotted data."""
    fig.legend(handles=_bench_legend_handles(methods), loc="upper center",
               ncol=len(methods if methods is not None else _LEGEND_ORDER),
               bbox_to_anchor=(0.5, y), frameon=True, framealpha=0.95,
               edgecolor="#cccccc", fontsize=fontsize, columnspacing=1.1,
               handlelength=1.6)


def _axes_in_cell(fig, cell):
    """Axes whose centre falls inside a subplotspec cell's bbox."""
    pos = cell.get_position(fig)
    out = []
    for ax in fig.axes:
        bb = ax.get_position()
        cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
        if pos.x0 <= cx <= pos.x1 and pos.y0 <= cy <= pos.y1:
            out.append(ax)
    return out


def _bench_legend_below(fig, cell, *, methods=None, fontsize=4.6, y_pad=0.004,
                        ncol=None, short=False, axes=None, markersize=7,
                        y_anchor_override=None):
    """A horizontal method legend centred just below a composite cell, matching
    the standalone panels (which each carry their own legend below the axes).
    Defaults to a SINGLE ROW (ncol = number of methods); pass `ncol` to wrap.
    `short` uses the one-word method labels so five entries fit one line.

    When `axes` is given (or discoverable in the cell), the legend is placed just
    below the LOWEST x-axis label of those axes -- computed from the rendered
    tight bbox -- so it never overlaps the tick or axis labels regardless of how
    deep matplotlib places them. Pass `y_anchor_override` (figure fraction) to
    pin the anchor explicitly."""
    methods = methods if methods is not None else _LEGEND_ORDER
    pos = cell.get_position(fig)
    cx = 0.5 * (pos.x0 + pos.x1)
    if y_anchor_override is not None:
        y_anchor = y_anchor_override
    else:
        if axes is None:
            axes = _axes_in_cell(fig, cell)
        y_anchor = pos.y0 - y_pad
        if axes:
            try:
                r = fig.canvas.get_renderer()
                y_disp = min(ax.get_tightbbox(r).y0 for ax in axes)
                y_fig = fig.transFigure.inverted().transform((0, y_disp))[1]
                y_anchor = min(y_anchor, y_fig - y_pad)
            except Exception:
                pass
    fig.legend(
        handles=_bench_legend_handles(methods, short=short, markersize=markersize),
        loc="upper center",
        bbox_to_anchor=(cx, y_anchor), ncol=ncol if ncol is not None else len(methods),
        frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=fontsize,
        columnspacing=0.6, handlelength=1.0, handletextpad=0.25, borderpad=0.3,
    )


def _broken_pair(fig, gs_cell, *, main_ylim, strip_ylim, height_ratios=(1, 3.4)):
    """A broken y-axis: a thin NEBULA strip above a main calibrated-method axis.

    Returns (ax_main, ax_strip) sharing x, with the conventional diagonal break
    marks. NEBULA goes in the strip at its true ~0.75 range; the calibrated
    methods sit in the main axis at the nominal scale.
    """
    inner = gs_cell.subgridspec(2, 1, height_ratios=height_ratios, hspace=0.08)
    ax_strip = fig.add_subplot(inner[0])
    ax_main = fig.add_subplot(inner[1], sharex=ax_strip)
    ax_strip.set_ylim(*strip_ylim)
    ax_main.set_ylim(*main_ylim)
    ax_strip.spines["bottom"].set_visible(False)
    ax_main.spines["top"].set_visible(False)
    ax_strip.tick_params(labelbottom=False, bottom=False)
    d = 0.012
    kw = dict(transform=ax_strip.transAxes, color="#333", clip_on=False, lw=0.9)
    ax_strip.plot((-d, +d), (-d * 3, +d * 3), **kw)
    ax_strip.plot((1 - d, 1 + d), (-d * 3, +d * 3), **kw)
    kw["transform"] = ax_main.transAxes
    ax_main.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_main.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    return ax_main, ax_strip


def _panel_bench_typeI_main(fig, core_df, *, composite: bool = False, gs_parent=None):
    """3C. Pure-null Type I error vs sample size, two aligned designs.

    x is participants per arm (two-arm) / paired participants (single-arm), which
    is n_treated in both cases. Calibrated methods sit at the nominal band with
    scenario-level 95% MC CIs and a dashed 0.05 reference (no shaded tolerance
    band). NEBULA is in a narrow upper strip on its own 0.70-0.80 scale.
    """
    null = core_df[(core_df["family"] == "null") & (~core_df["is_signal"])]
    rate = _per_scenario_rate(null, on_signal=False)
    meta = (null.groupby("scenario")
            .agg(design=("design", "first"), per_arm=("n_treated", "first"))
            .reset_index())
    rate = rate.merge(meta, on="scenario")

    # In a composite, build inside the parent subplotspec (subgridspec) rather
    # than a new top-level gridspec, so the broken-axis pairs stay within the
    # allotted cell instead of overflowing into neighbouring panels.
    gs = (gs_parent.subgridspec(1, 2, wspace=0.28) if gs_parent is not None
          else fig.add_gridspec(1, 2, wspace=0.28))
    _ttl = 5.5 if composite else 12
    _ax = 5.2 if composite else 11
    _tk = 4.7 if composite else 10
    for ci, design in enumerate(("two_arm", "single_arm")):
        sub = rate[rate["design"] == design]
        n_vals = sorted(sub["per_arm"].unique())
        pos = {n: i for i, n in enumerate(n_vals)}
        ax_main, ax_strip = _broken_pair(fig, gs[ci], main_ylim=(0.0, 0.10),
                                         strip_ylim=(0.68, 0.82))
        for method in _CALIBRATED:
            m = sub[sub["method"] == method].sort_values("per_arm")
            if m.empty:
                continue
            style = _method_style(method, is_focal=(method == "sctrial_did"),
                                  composite=composite)
            xs = [pos[n] for n in m["per_arm"]]
            ax_main.plot(xs, m["mean"],
                         label=_BENCH_METHOD_LABELS[method] if ci == 0 else None, **style)
            ax_main.errorbar(xs, m["mean"], yerr=1.96 * m["mcse"], fmt="none",
                             ecolor=style["color"], elinewidth=0.8, capsize=1.5, alpha=0.6)
        neb = sub[sub["method"] == "nebula"].sort_values("per_arm")
        if not neb.empty:
            ns = _method_style("nebula", composite=composite)
            xs = [pos[n] for n in neb["per_arm"]]
            ax_strip.plot(xs, neb["mean"],
                          label=_BENCH_METHOD_LABELS["nebula"] if ci == 0 else None, **ns)
            ax_strip.errorbar(xs, neb["mean"], yerr=1.96 * neb["mcse"], fmt="none",
                              ecolor=ns["color"], elinewidth=0.8, capsize=1.5, alpha=0.6)
        ax_main.axhline(0.05, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
        ax_main.set_xticks(range(len(n_vals)))
        ax_main.set_xticklabels(n_vals)
        ax_main.set_xlim(-0.4, len(n_vals) - 0.6)
        ax_main.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
        ax_strip.set_yticks([0.7, 0.8])
        xlabel = "Participants per arm" if design == "two_arm" else "Paired participants"
        ax_main.set_xlabel(xlabel, fontsize=_ax)
        ax_strip.set_title(_DESIGN_LABEL[design], fontsize=_ttl, fontweight="bold", pad=4)
        for a in (ax_main, ax_strip):
            a.tick_params(labelsize=_tk)
            _style_axis(a)
        if ci == 0:
            ax_main.set_ylabel("Type I error (p < 0.05)", fontsize=_ax)
    if not composite:
        # One global legend below the panels, never over the plotted data.
        fig.legend(handles=_bench_legend_handles(), loc="lower center",
                   bbox_to_anchor=(0.5, -0.04), ncol=len(_LEGEND_ORDER),
                   frameon=True, framealpha=0.95, edgecolor="#cccccc",
                   fontsize=8, columnspacing=1.1, handlelength=1.6)


def _panel_bench_typeI_vs_n(axes, core_df, *, composite: bool = False):
    """Pure-null Type I error vs sample size, one facet per design.

    The central calibration result: the calibrated methods sit at the nominal
    band across the whole sample-size range while NEBULA is flat and far above
    it, so its miscalibration is structural, not a small-sample artefact.
    NEBULA is clipped to the axis top and annotated rather than allowed to
    compress the calibrated methods.
    """
    null = core_df[(core_df["family"] == "null") & (~core_df["is_signal"])]
    rate = _per_scenario_rate(null, on_signal=False)
    meta = (null.groupby("scenario")
            .agg(design=("design", "first"), total_n=("total_n", "first"))
            .reset_index())
    rate = rate.merge(meta, on="scenario")

    _ttl = 5.8 if composite else 12
    _ax = 5.2 if composite else 11
    _tk = 4.7 if composite else 10
    ymax = 0.15
    for ax, design in zip(axes, ("two_arm", "single_arm")):
        sub = rate[rate["design"] == design]
        # Evenly spaced categorical positions, labelled with the actual
        # participant counts. A log axis with non-decade values (16, 24, 40, ...)
        # collides matplotlib's decade minor-tick labels with the real ticks.
        n_vals = sorted(sub["total_n"].unique())
        pos = {n: i for i, n in enumerate(n_vals)}
        for method in _BENCH_METHODS:
            m = sub[sub["method"] == method].sort_values("total_n")
            if m.empty:
                continue
            style = _method_style(method, is_focal=(method == "sctrial_did"),
                                  composite=composite)
            xs = [pos[n] for n in m["total_n"]]
            _plot_offscale(ax, xs, m["mean"], method=method, ymax=ymax, style=style,
                           label=_BENCH_METHOD_LABELS[method] if design == "two_arm" else None)
            vis = m[m["mean"] <= ymax]
            ax.errorbar([pos[n] for n in vis["total_n"]], vis["mean"],
                        yerr=1.96 * vis["mcse"], fmt="none", ecolor=style["color"],
                        elinewidth=0.8, capsize=1.5, alpha=0.6)
        _add_nominal_band(ax)
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(range(len(n_vals)))
        ax.set_xticklabels(n_vals)
        ax.set_xlim(-0.4, len(n_vals) - 0.6)
        ax.set_title(_DESIGN_LABEL[design], fontsize=_ttl, fontweight="bold", pad=4)
        ax.set_xlabel("Participants", fontsize=_ax)
        ax.tick_params(labelsize=_tk)
        _style_axis(ax)
    axes[0].set_ylabel("Type I error (p < 0.05)", fontsize=_ax)
    if not composite:
        axes[0].legend(loc="upper right", fontsize=8, frameon=True,
                       framealpha=0.95, edgecolor="#cccccc")


def _bh_reject(p, q=0.05):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    rej = np.zeros(len(p), bool)
    if not ok.any():
        return rej
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    if passed.any():
        rej[order[: np.max(np.where(passed)[0]) + 1]] = True
    return rej


def _per_scenario_fdr(df, q=0.05):
    """Per-(scenario, method) BH FDR with the correct hierarchy.

    Realised FDR per replicate (false rejections / total rejections after BH at
    q), then mean and MCSE across replicates. The realised replicate-level FDR is
    the manuscript's stated endpoint.
    """
    rows = []
    for (sc, m, it), g in df.groupby(["scenario", "method", "iteration"], sort=False):
        rej = _bh_reject(g["pvalue"].to_numpy(), q)
        sig = g["is_signal"].to_numpy(bool)
        n_rej = int(rej.sum())
        rows.append((sc, m, it, float((rej & ~sig).sum() / n_rej) if n_rej else 0.0))
    per_rep = pd.DataFrame(rows, columns=["scenario", "method", "iteration", "fdr"])
    agg = (per_rep.groupby(["scenario", "method"])["fdr"]
           .agg(mean="mean", sd="std", n_rep="count").reset_index())
    agg["mcse"] = agg["sd"] / np.sqrt(agg["n_rep"].clip(lower=1))
    return agg


def _per_scenario_tpr(df, q=0.05, mode="end_to_end"):
    """Per-(scenario, method) BH-controlled true-positive rate (per replicate,
    then mean + MCSE across replicates).

    mode='end_to_end': FDR-controlled discovery sensitivity =
        #{prespecified signal genes with BH q<0.05} / #{all prespecified signal genes};
        signal genes filtered before testing count as NON-DETECTIONS.
    mode='tested': denominator restricted to signal genes that were evaluable.
    """
    rows = []
    for (sc, m, it), g in df.groupby(["scenario", "method", "iteration"], sort=False):
        sig = g["is_signal"].to_numpy(bool)
        n_sig = int(sig.sum())
        if n_sig == 0:
            continue
        rej = _bh_reject(g["pvalue"].to_numpy(), q)
        ev = g["evaluable"].to_numpy(bool)
        n_hit = int((rej & sig).sum())
        denom = int((sig & ev).sum()) if mode == "tested" else n_sig
        rows.append((sc, m, it, float(n_hit / denom) if denom else np.nan))
    per_rep = pd.DataFrame(rows, columns=["scenario", "method", "iteration", "tpr"])
    agg = (per_rep.groupby(["scenario", "method"])["tpr"]
           .agg(mean="mean", sd="std", n_rep="count").reset_index())
    agg["mcse"] = agg["sd"] / np.sqrt(agg["n_rep"].clip(lower=1))
    return agg


def _per_scenario_quality(df, kind="evaluability"):
    """Per-(scenario, method) evaluability or convergence (per replicate, then
    mean + MCSE).

    evaluability = #{genes with valid inference} / #{prespecified genes}.
    convergence  = #{converged fits} / #{attempted (evaluable) fits}.
    """
    rows = []
    for (sc, m, it), g in df.groupby(["scenario", "method", "iteration"], sort=False):
        ev = g["evaluable"].to_numpy(bool)
        conv = g["converged"].to_numpy(bool)
        if kind == "evaluability":
            val = float(ev.mean()) if len(ev) else np.nan
        else:  # convergence among attempted (evaluable) fits
            val = float(conv[ev].mean()) if ev.any() else np.nan
        rows.append((sc, m, it, val))
    per_rep = pd.DataFrame(rows, columns=["scenario", "method", "iteration", "value"])
    agg = (per_rep.groupby(["scenario", "method"])["value"]
           .agg(mean="mean", sd="std", n_rep="count").reset_index())
    agg["mcse"] = agg["sd"] / np.sqrt(agg["n_rep"].clip(lower=1))
    return agg


def _faceted_broken_by_fraction(fig, rate, *, ylabel, main_ylim, strip_ylim,
                                title, arch="Balanced signal architecture",
                                nominal=True, composite=False, panel_sizes=None,
                                gs_parent=None, suppress_ylabel=False,
                                title_y_composite=0.80,
                                common_xlabel_composite=False,
                                common_xlabel_pad=0.005,
                                ytick_main=None, ytick_strip=None,
                                marker_scale=1.0):
    """Shared 3D/3E body: four tested-set-size facets, x = signal fraction,
    calibrated methods in the main region and NEBULA in an upper strip.

    `rate` has columns scenario, method, mean, mcse plus n_genes and signal_pct.
    `panel_sizes` restricts the facets shown (the lean main figure shows one
    representative tested-set size; the full grid lives in the supplement).
    """
    _ttl = 4.6 if composite else 11
    _ax = 5.2 if composite else 10
    _tk = 4.5 if composite else 9
    sizes = panel_sizes if panel_sizes is not None else _PANEL_SIZES
    fracs = _SIGNAL_FRACTIONS
    xpos = {f: i for i, f in enumerate(fracs)}
    # Wider per-method x-offsets so the calibrated methods (all near the nominal
    # band) stay separable; at 0.03 limma-voom's marker sat under sctrial's.
    off = {"dreamlet": 0.11, "limma_voom": 0.055, "sctrial_did": -0.055,
           "wilcoxon_paired": -0.11, "nebula": 0.0}
    gs = (gs_parent.subgridspec(1, len(sizes), wspace=0.30) if gs_parent is not None
          else fig.add_gridspec(1, len(sizes), wspace=0.30))
    for ci, ng in enumerate(sizes):
        ax_main, ax_strip = _broken_pair(fig, gs[ci], main_ylim=main_ylim,
                                         strip_ylim=strip_ylim)
        sub = rate[rate["n_genes"] == ng]
        for method in _CALIBRATED:
            m = sub[sub["method"] == method].sort_values("signal_pct")
            if m.empty:
                continue
            style = _method_style(method, is_focal=(method == "sctrial_did"),
                                  composite=composite)
            style["markersize"] *= marker_scale
            xs = [xpos[int(f)] + off[method] for f in m["signal_pct"]]
            ax_main.plot(xs, m["mean"],
                         label=_BENCH_METHOD_LABELS[method] if ci == 0 else None, **style)
            ax_main.errorbar(xs, m["mean"], yerr=1.96 * m["mcse"], fmt="none",
                             ecolor=style["color"], elinewidth=0.7, capsize=1.2, alpha=0.55)
        neb = sub[sub["method"] == "nebula"].sort_values("signal_pct")
        if not neb.empty:
            ns = _method_style("nebula", composite=composite)
            ns["markersize"] *= marker_scale
            xs = [xpos[int(f)] for f in neb["signal_pct"]]
            ax_strip.plot(xs, neb["mean"],
                          label=_BENCH_METHOD_LABELS["nebula"] if ci == 0 else None, **ns)
            ax_strip.errorbar(xs, neb["mean"], yerr=1.96 * neb["mcse"], fmt="none",
                              ecolor=ns["color"], elinewidth=0.7, capsize=1.2, alpha=0.55)
        if nominal:
            ax_main.axhline(0.05, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
        ax_main.set_xticks(range(len(fracs)))
        ax_main.set_xticklabels([f"{f}%" for f in fracs])
        ax_main.set_xlim(-0.5, len(fracs) - 0.5)
        ax_strip.set_title(f"{ng:,} tested genes", fontsize=_ttl, fontweight="bold",
                           pad=(1 if composite else 3),
                           y=(title_y_composite if composite else 1.0))
        if not (composite and common_xlabel_composite):
            ax_main.set_xlabel("Signal fraction", fontsize=_ax,
                               labelpad=(1 if composite else 4))
        for a in (ax_main, ax_strip):
            a.tick_params(labelsize=_tk)
            _style_axis(a)
        if ytick_main is not None:
            ax_main.set_yticks(ytick_main)
        if ytick_strip is not None:
            ax_strip.set_yticks(ytick_strip)
        if ci > 0:
            ax_main.set_yticklabels([])
            ax_strip.set_yticklabels([])
        elif not suppress_ylabel:
            ax_main.set_ylabel(ylabel, fontsize=_ax)
    if composite and common_xlabel_composite and gs_parent is not None:
        pos = gs_parent.get_position(fig)
        fig.text(0.5 * (pos.x0 + pos.x1), pos.y0 - common_xlabel_pad,
                 "Signal fraction", fontsize=_ax, ha="center", va="top")
    if not composite:
        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
        if arch:
            fig.text(0.5, 0.955, arch, ha="center", va="top", fontsize=10,
                     fontstyle="italic", color="#444")
        # One global legend below the facets, standard order, off the data.
        fig.legend(handles=_bench_legend_handles(), loc="lower center",
                   bbox_to_anchor=(0.5, -0.04), ncol=len(_LEGEND_ORDER),
                   frameon=True, framealpha=0.95, edgecolor="#cccccc",
                   fontsize=8, columnspacing=1.1, handlelength=1.6)


def _panel_bench_mixed_fpr(fig, bench_df, *, composite: bool = False, panel_sizes=None,
                           gs_parent=None, architecture="balanced"):
    """Mixed-signal null-gene FPR, four tested-set-size facets, for one signal
    architecture ('balanced' or 'one_directional')."""
    _arch_lab = {"balanced": "Balanced signal architecture",
                 "one_directional": "One-directional signal architecture"}.get(
                     architecture, architecture)
    bal = bench_df[(bench_df["architecture"] == architecture)
                   & (bench_df["signal_fraction_realised"] > 0)
                   & (~bench_df["is_signal"])]
    rate = _per_scenario_rate(bal, on_signal=False)
    meta = bal.groupby("scenario").agg(n_genes=("n_genes", "first"),
                                       signal_pct=("signal_pct", "first")).reset_index()
    rate = rate.merge(meta, on="scenario")
    _faceted_broken_by_fraction(
        fig, rate, ylabel="Null-gene FPR\n(p < 0.05)",
        main_ylim=(0.025, 0.075), strip_ylim=(0.68, 0.90), arch=_arch_lab,
        title="Mixed-signal null-gene false-positive rate", composite=composite,
        panel_sizes=panel_sizes, gs_parent=gs_parent, title_y_composite=0.75,
        common_xlabel_composite=True, common_xlabel_pad=0.018, marker_scale=0.7)


def _panel_bench_bh_fdr(fig, bench_df, *, composite: bool = False, panel_sizes=None,
                        gs_parent=None):
    """3E. BH-controlled realised FDR, balanced, four tested-set-size facets."""
    bal = bench_df[(bench_df["architecture"] == "balanced")
                   & (bench_df["signal_fraction_realised"] > 0)]
    rate = _per_scenario_fdr(bal)
    meta = bal.groupby("scenario").agg(n_genes=("n_genes", "first"),
                                       signal_pct=("signal_pct", "first")).reset_index()
    rate = rate.merge(meta, on="scenario")
    _faceted_broken_by_fraction(
        fig, rate, ylabel=r"Realized FDR at BH $q<0.05$",
        main_ylim=(0.0, 0.12), strip_ylim=(0.65, 1.02),
        title="False discovery rate after Benjamini-Hochberg", composite=composite,
        panel_sizes=panel_sizes, gs_parent=gs_parent, title_y_composite=0.95)


def _panel_bench_power_vs_n(fig, core_df, *, composite: bool = False, only_beta=None,
                            gs_parent=None, marker_scale=1.0):
    """Marginal detection power vs sample size, faceted design x effect size.

    Separates single-arm from two-arm -- pooling them onto one participant axis
    is what produced the spurious non-monotonic curve. Within each facet power
    rises monotonically with participants. `only_beta` restricts to a single
    representative effect size (the lean main figure); the full effect-size grid
    lives in the supplement.
    """
    de = core_df[(core_df["family"] == "de_balanced") & (core_df["is_signal"])]
    rate = _per_scenario_rate(de, on_signal=True)
    meta = (de.groupby("scenario")
            .agg(design=("design", "first"), per_arm=("n_treated", "first"),
                 beta=("beta", "first")).reset_index())
    rate = rate.merge(meta, on="scenario")
    betas = sorted(rate["beta"].dropna().unique())
    if only_beta is not None:
        betas = [b for b in betas if abs(b - only_beta) < 1e-9] or betas[:1]
    designs = ("two_arm", "single_arm")
    row_header = {"two_arm": "Two-arm DiD", "single_arm": "Single-arm paired change"}
    # Tighter rows, room on the left for row headers and one shared y-label, room
    # at the top for the title + legend.
    # Roomy hspace: the two-arm (top) and single-arm (bottom) rows are INDEPENDENT
    # designs, each with its own x ticks and x-axis title, so they need vertical
    # separation or the two-arm ticks/label collide with the single-arm axis.
    gs = (gs_parent.subgridspec(len(designs), len(betas), hspace=1.40, wspace=0.26)
          if gs_parent is not None
          else fig.add_gridspec(len(designs), len(betas), hspace=1.40, wspace=0.26,
                                left=0.13, right=0.98, top=0.84, bottom=0.10))
    _ttl = 7.0 if composite else 11
    _ax = 5.2 if composite else 10
    _tk = 4.6 if composite else 9
    # Deterministic x-offsets expose methods that otherwise overlap almost
    # exactly. NEBULA is EXCLUDED from marginal-detection comparisons: its Type I
    # error is ~0.75, so its "detections" are mostly false positives; its full
    # rejection rates and matched-model validation are in the supplement.
    off = {"wilcoxon_paired": -0.15, "limma_voom": -0.05,
           "sctrial_did": 0.05, "dreamlet": 0.15}
    axes_by_row = {}
    for ri, design in enumerate(designs):
        for ci, beta in enumerate(betas):
            ax = fig.add_subplot(gs[ri, ci])
            axes_by_row.setdefault(ri, []).append(ax)
            sub = rate[(rate["design"] == design) & (rate["beta"] == beta)]
            n_vals = sorted(sub["per_arm"].unique())
            pos = {n: i for i, n in enumerate(n_vals)}
            for method in _CALIBRATED:
                m = sub[sub["method"] == method].sort_values("per_arm")
                if m.empty:
                    continue
                style = _method_style(method, is_focal=(method == "sctrial_did"),
                                      composite=composite)
                style["markersize"] *= marker_scale
                xs = [pos[n] + off[method] for n in m["per_arm"]]
                ys = m["mean"].to_numpy(float)
                half = 1.96 * m["mcse"].to_numpy(float)
                # 95% Monte Carlo CIs, clipped to the valid probability range so a
                # near-0 or near-1 estimate cannot draw a whisker outside [0, 1].
                lo = np.clip(ys - half, 0.0, 1.0)
                hi = np.clip(ys + half, 0.0, 1.0)
                zbase = 10 if method == "sctrial_did" else 5
                # Error bars first (behind), then the line + marker on top, so the
                # marker is always drawn above its interval even where the CI is
                # very narrow (e.g. beta = 1.0, where it may be hidden entirely).
                ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="none",
                            ecolor=style["color"], elinewidth=0.9,
                            capsize=(1.6 if composite else 2.3), capthick=0.9,
                            alpha=0.9, zorder=zbase - 1)
                ax.plot(xs, ys, **style, zorder=zbase)
            ax.set_ylim(0, 1.2)
            ax.set_xticks(range(len(n_vals)))
            ax.set_xticklabels(n_vals)
            ax.set_xlim(-0.5, len(n_vals) - 0.5)
            # In a composite the effect size is folded into the panel's corner
            # title, so the per-axis beta title (which would collide with it) is
            # suppressed; standalone keeps it as the column header.
            # Effect-size column header on the top row, in BOTH standalone and
            # composite, but ONLY when there is more than one beta column: a
            # single-column crop (Figure 3 F, only_beta=0.5) already names the beta
            # in its panel title, so a column header would duplicate it.
            if ri == 0 and len(betas) > 1:
                ax.set_title(rf"$\beta$ = {beta}",
                             fontsize=(5.4 if composite else _ttl), fontweight="bold",
                             pad=(2 if composite else 6),
                             y=(0.82 if composite else 1.0))
            if ci > 0:
                ax.set_yticklabels([])
            xl = "Participants per arm" if design == "two_arm" else "Paired participants"
            ax.set_xlabel(xl, fontsize=_ax, labelpad=(4 if composite else 4))
            ax.tick_params(labelsize=_tk)
            _style_axis(ax)
        if composite:
            # Compact design label as the leftmost y-axis label, so the embedded
            # panel keeps its two-arm / single-arm rows without external text math.
            # Short forms fit the narrow composite column (design detail is in the
            # caption).
            short = {"two_arm": "Two-arm", "single_arm": "Single-arm"}[design]
            axes_by_row[ri][0].set_ylabel(short, fontsize=_ax)
    if not composite:
        # Row headers on the left, ONE shared y-label, title, and a global legend
        # below the title (four calibrated methods; NEBULA omitted here).
        for ri, design in enumerate(designs):
            axs = axes_by_row[ri]
            p0, p1 = axs[0].get_position(), axs[-1].get_position()
            ycen = (p0.y0 + p0.y1) / 2
            fig.text(0.02, ycen, row_header[design], rotation=90, ha="left",
                     va="center", fontsize=11, fontweight="bold")
            del p1
        fig.text(0.065, 0.5, "Marginal detection probability", rotation=90,
                 ha="center", va="center", fontsize=11)
        fig.suptitle("Marginal detection probability by effect size and sample size",
                     fontsize=13, fontweight="bold", y=0.99)
        # Canonical legend order (sctrial first), independent of the sctrial-last
        # draw order; NEBULA is legitimately absent from the power comparison.
        _cal_legend = [m for m in _LEGEND_ORDER if m != "nebula"]
        fig.legend(handles=_bench_legend_handles(_cal_legend), loc="upper center",
                   bbox_to_anchor=(0.5, 0.925), ncol=len(_cal_legend), frameon=True,
                   framealpha=0.95, edgecolor="#cccccc", fontsize=8,
                   columnspacing=1.1, handlelength=1.6)


def _panel_bench_scenario_families(ax, core_df, *, composite: bool = False):
    """Null-gene FPR across robustness families, per method.

    Cell yield, missing visits, arm imbalance and composition stress. sctrial and
    Wilcoxon hold nominal everywhere; dreamlet and limma inflate under
    composition stress; NEBULA is off-scale throughout (clipped + annotated).
    """
    fam_order = [
        ("cells_50", "50 cells/PV"), ("cells_250", "250 cells/PV"), ("cells_1000", "1,000 cells/PV"),
        ("missing_10pct", "10% miss"), ("missing_20pct", "20% miss"),
        ("imbal_3v7", "3:7 arm imbalance"), ("imbal_5v10", "5:10 arm imbalance"), ("imbal_10v20", "10:20 arm imbalance"),
        ("compstress_onedir", "comp. stress"),
    ]
    present = {f for f in core_df["family"].unique()}
    fam_order = [(f, lab) for f, lab in fam_order if f in present]
    null_genes = core_df[~core_df["is_signal"]]
    rate = _per_scenario_rate(null_genes, on_signal=False)
    meta = core_df.groupby("scenario").agg(family=("family", "first")).reset_index()
    rate = rate.merge(meta, on="scenario")
    # equal-weight mean across scenarios within a family (some families have 2)
    fam_rate = (rate.groupby(["family", "method"])["mean"]
                .mean().reset_index())

    x = np.arange(len(fam_order))
    width = 0.16
    ymax = 0.15
    # Canonical bar/legend order (sctrial first), matching every other panel.
    order = [m for m in _LEGEND_ORDER if m in set(core_df["method"].unique())]
    for mi, method in enumerate(order):
        vals = []
        for f, _ in fam_order:
            r = fam_rate[(fam_rate["family"] == f) & (fam_rate["method"] == method)]
            vals.append(float(r["mean"].iloc[0]) if not r.empty else np.nan)
        vals = np.array(vals)
        style = _method_style(method, is_focal=(method == "sctrial_did"), composite=composite)
        offset = (mi - (len(order) - 1) / 2) * width
        clipped = np.minimum(vals, ymax)
        ax.bar(x + offset, clipped, width, color=style["color"],
               label=_BENCH_METHOD_LABELS[method], edgecolor="white", linewidth=0.4)
        # Off-scale value as vertical text INSIDE the top of the clipped bar.
        # NEBULA (orange) uses black text for legibility; others use white.
        _txt_col = "#111111" if method == "nebula" else "white"
        for xi, v in zip(x + offset, vals):
            if v > ymax:
                ax.text(xi, ymax - 0.004, f"{v:.2f}", fontsize=(4.4 if composite else 6),
                        ha="center", va="top", color=_txt_col, rotation=90,
                        fontweight="bold")
    _add_nominal_band(ax)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in fam_order],
                       rotation=20, ha="right", fontsize=(3.8 if composite else 8))
    ax.set_ylabel("Null-gene FPR\n(p < 0.05)", fontsize=(5.0 if composite else 10))
    ax.tick_params(axis="y", labelsize=(4.6 if composite else 8))
    if not composite:
        # Below the plot: NEBULA's clipped bars fill the top across the whole
        # width, so an in-axes legend would sit on top of them (and hid the
        # cell-yield family annotations in the first render).
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=len(_BENCH_METHODS), fontsize=8, frameon=True,
                  framealpha=0.95, edgecolor="#cccccc", columnspacing=1.0)
    _style_axis(ax)




def _panel_bench_discovery_sensitivity(fig, bench_df, *, composite=False,
                                       gs_parent=None, mode="end_to_end",
                                       panel_sizes=None, suppress_ylabel=False,
                                       marker_scale=1.0):
    """FDR-controlled discovery sensitivity: BH-controlled end-to-end true-positive
    rate vs signal fraction, one facet per tested-set size (balanced architecture).

    Uses the shared broken-axis renderer so the calibrated methods (low sensitivity
    under strict FDR control) stay legible while NEBULA sits on the upper strip.
    NEBULA's high value is NOT FDR-controlled (its realised FDR is severe, see the
    FDR panel) and must be read alongside it, not as a favourable result.
    """
    bal = bench_df[(bench_df["architecture"] == "balanced")
                   & (bench_df["signal_fraction_realised"] > 0)]
    rate = _per_scenario_tpr(bal, mode=mode)
    meta = bal.groupby("scenario").agg(n_genes=("n_genes", "first"),
                                       signal_pct=("signal_pct", "first")).reset_index()
    rate = rate.merge(meta, on="scenario")
    _faceted_broken_by_fraction(
        fig, rate,
        ylabel="FDR-controlled discovery sensitivity",
        main_ylim=(0.0, 0.30), strip_ylim=(0.85, 1.20),
        title="FDR-controlled discovery sensitivity (end-to-end TPR)",
        nominal=False, composite=composite, gs_parent=gs_parent,
        panel_sizes=panel_sizes, suppress_ylabel=suppress_ylabel,
        ytick_main=[0.0, 0.1, 0.2, 0.3], ytick_strip=[1.0, 1.2],
        marker_scale=marker_scale)


def _panel_bench_quality(ax, core_df, *, kind="evaluability", composite=False):
    """Evaluability or convergence per method across the robustness families
    (core grid). Evaluability exposes gene filtering (dreamlet drops only in the
    lowest cell-yield / empirical-heterogeneous-yield families); convergence is
    computed among ATTEMPTED (evaluable) fits and stays at 1.0, so the two are
    reported separately and the reduced retention is shown to be filtering, not a
    convergence failure."""
    fam_order = [
        ("cells_50", "50 cells/PV"), ("cells_250", "250 cells/PV"), ("cells_1000", "1,000 cells/PV"),
        ("null_hetero", "emp. het. yield"),
        ("missing_10pct", "10% miss"), ("missing_20pct", "20% miss"),
        ("imbal_3v7", "3:7 arm imbalance"), ("imbal_5v10", "5:10 arm imbalance"), ("imbal_10v20", "10:20 arm imbalance"),
        ("de_hetero", "het. effect"), ("compstress_onedir", "comp. stress"),
    ]
    q = _per_scenario_quality(core_df, kind=kind)
    meta = core_df.groupby("scenario").agg(family=("family", "first")).reset_index()
    q = q.merge(meta, on="scenario")
    fam_rate = q.groupby(["family", "method"])["mean"].mean().reset_index()
    present = set(core_df["family"].unique())
    fam_order = [(f, lab) for f, lab in fam_order if f in present]
    order = [m for m in _LEGEND_ORDER if m in set(core_df["method"].unique())]
    x = np.arange(len(fam_order))
    width = 0.16
    _tk = 3.8 if composite else 8
    _ax = 5.0 if composite else 10
    _ttl = 6.0 if composite else 12
    for mi, method in enumerate(order):
        vals = []
        for f, _ in fam_order:
            r = fam_rate[(fam_rate["family"] == f) & (fam_rate["method"] == method)]
            vals.append(float(r["mean"].iloc[0]) if not r.empty else np.nan)
        offset = (mi - (len(order) - 1) / 2) * width
        ax.bar(x + offset, vals, width, color=_BENCH_METHOD_COLORS[method],
               label=_BENCH_METHOD_LABELS[method], edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="#888", linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)
    # Focus the y-range on the informative band so the small dips are visible.
    ax.set_ylim(0.80, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in fam_order], rotation=20, ha="right",
                       fontsize=_tk)
    # Short y-label in the composite so the rotated label's top does not reach the
    # panel letter at the cell's upper-left (the parenthetical is in the caption).
    if composite:
        ylab = "Evaluability" if kind == "evaluability" else "Convergence"
    else:
        ylab = ("Evaluability (valid / prespecified)" if kind == "evaluability"
                else "Convergence (converged / attempted)")
    ax.set_ylabel(ylab, fontsize=_ax)
    ax.set_title("Gene evaluability across robustness families" if kind == "evaluability"
                 else "Convergence among attempted fits",
                 fontsize=_ttl, fontweight="bold", pad=(4 if composite else 8))
    ax.tick_params(axis="y", labelsize=_tk)
    if not composite:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
                  ncol=len(order), fontsize=8, frameon=True, framealpha=0.95,
                  edgecolor="#cccccc", columnspacing=1.0)
    _style_axis(ax)


# Same family order as _panel_bench_quality's fam_order (minus the null-only
# families) so the stacked bar panels L/M/N/O share one left-to-right x-order and
# their family columns line up vertically.
_ROBUST_FAMILIES = [
    ("cells_50", "50 cells/PV"), ("cells_250", "250 cells/PV"), ("cells_1000", "1,000 cells/PV"),
    ("missing_10pct", "10% miss"), ("missing_20pct", "20% miss"),
    ("imbal_3v7", "3:7 arm imbalance"), ("imbal_5v10", "5:10 arm imbalance"), ("imbal_10v20", "10:20 arm imbalance"),
    ("de_hetero", "het. effect"),
    ("compstress_onedir", "comp. stress"),
]


def _panel_bench_family_tpr(ax, core_df, *, composite=False):
    """End-to-end BH TPR across the robustness families that CONTAIN signal-bearing
    scenarios (the manifest availability table excludes null/null_hetero). NEBULA's
    high TPR is not FDR-controlled (see the FDR/discovery panels) and is read
    alongside them."""
    sig = core_df[core_df["is_signal"].groupby(core_df["scenario"]).transform("any")]
    tpr = _per_scenario_tpr(sig, mode="end_to_end")
    meta = sig.groupby("scenario").agg(family=("family", "first")).reset_index()
    tpr = tpr.merge(meta, on="scenario")
    fam_rate = tpr.groupby(["family", "method"])["mean"].mean().reset_index()
    present = set(core_df["family"].unique())
    fams = [(f, lab) for f, lab in _ROBUST_FAMILIES if f in present]
    order = [m for m in _LEGEND_ORDER if m in set(core_df["method"].unique())]
    x = np.arange(len(fams))
    width = 0.16
    _tk = 3.8 if composite else 8
    _ax = 5.0 if composite else 10
    _ttl = 6.0 if composite else 12
    for mi, method in enumerate(order):
        vals = []
        for f, _ in fams:
            r = fam_rate[(fam_rate["family"] == f) & (fam_rate["method"] == method)]
            vals.append(float(r["mean"].iloc[0]) if not r.empty else np.nan)
        offset = (mi - (len(order) - 1) / 2) * width
        ax.bar(x + offset, vals, width, color=_BENCH_METHOD_COLORS[method],
               label=_BENCH_METHOD_LABELS[method], edgecolor="white", linewidth=0.4)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in fams], rotation=20, ha="right", fontsize=_tk)
    ax.set_ylabel("End-to-end\nBH TPR", fontsize=5.2 if composite else _ax)
    ax.set_title("Signal detection across robustness families",
                 fontsize=_ttl, fontweight="bold", pad=(4 if composite else 8))
    ax.tick_params(axis="y", labelsize=_tk)
    if not composite:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=len(order),
                  fontsize=8, frameon=True, framealpha=0.95, edgecolor="#cccccc",
                  columnspacing=1.0)
    _style_axis(ax)


def _panel_bench_endtoend_vs_tested(ax, core_df, *, composite=False):
    """End-to-end vs tested-only BH TPR, per method, focused on the cell-yield
    families where filtering occurs. The gap (end-to-end below tested) is the
    detection lost to gene filtering; it is non-zero only for dreamlet in the
    lowest cell-yield / empirical-heterogeneous-yield conditions."""
    fams = ["cells_50", "cells_250", "cells_1000"]
    sub = core_df[core_df["family"].isin(fams)
                  & core_df["is_signal"].groupby(core_df["scenario"]).transform("any")]
    e2e = _per_scenario_tpr(sub, mode="end_to_end").rename(columns={"mean": "e2e"})
    tst = _per_scenario_tpr(sub, mode="tested").rename(columns={"mean": "tst"})
    m = e2e.merge(tst[["scenario", "method", "tst"]], on=["scenario", "method"])
    meta = sub.groupby("scenario").agg(family=("family", "first")).reset_index()
    m = m.merge(meta, on="scenario")
    summ = m.groupby("method")[["e2e", "tst"]].mean().reset_index()
    order = [mm for mm in _LEGEND_ORDER if mm in set(summ["method"])]
    x = np.arange(len(order))
    width = 0.36
    _tk = 4.6 if composite else 9
    _ax = 5.0 if composite else 10
    _ttl = 6.0 if composite else 12
    for i, mth in enumerate(order):
        r = summ[summ["method"] == mth].iloc[0]
        col = _BENCH_METHOD_COLORS[mth]
        ax.bar(i - width / 2, r["tst"], width, color=col, alpha=0.45,
               edgecolor="white", linewidth=0.4, label="Tested-only" if i == 0 else None)
        ax.bar(i + width / 2, r["e2e"], width, color=col, alpha=0.95,
               edgecolor="white", linewidth=0.4, label="End-to-end" if i == 0 else None)
        # NEBULA's high TPR is not FDR-controlled (its realised FDR is severe, see
        # the FDR/discovery panels); mark it so the tall bars are not misread as
        # superior power.
        if mth == "nebula":
            ax.annotate("*", (i, max(r["tst"], r["e2e"]) + 0.02), ha="center",
                        va="bottom", fontsize=11, color=col, fontweight="bold")
    ax.set_xticks(x)
    # Full canonical labels, matching the evaluability/convergence panels.
    ax.set_xticklabels([_BENCH_METHOD_LABELS[mm] for mm in order],
                       rotation=20, ha="right", fontsize=_tk)
    # Headroom so the legend (upper-left, over the empty region) and the NEBULA
    # asterisk clear the tallest bars.
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("BH TPR\n(cell-yield families)", fontsize=_ax)
    ax.set_title("End-to-end vs tested-only detection",
                 fontsize=_ttl, fontweight="bold", pad=(4 if composite else 8))
    ax.tick_params(axis="y", labelsize=_tk)
    # The NEBULA "*" marks that its high TPR is not FDR-controlled; the sentence
    # explaining it lives in the caption (lab convention keeps notes off figures).
    # The tested-only vs end-to-end key IS the panel; always draw it (small in the
    # composite), in the empty upper-left region.
    ax.legend(loc="upper left", fontsize=(4.6 if composite else 8), frameon=True,
              framealpha=0.95, edgecolor="#cccccc", handlelength=1.0,
              handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
    _style_axis(ax)


def _panel_bench_pure_null_fpr(fig, bench_df, *, composite: bool = False,
                               gs_parent=None):
    """Pure-null Type I error vs gene panel size, with NEBULA on a broken upper
    axis (matches the Figure 3 display convention) so its ~0.75 inflation is
    shown without compressing the calibrated methods. Returns (ax_main, ax_strip)."""
    null = bench_df[(bench_df["is_null_scenario"]) & (bench_df["true_beta"] == 0.0)]
    rows = []
    for (method, n_g), grp in null.groupby(["method", "n_genes"]):
        pvals = grp["pvalue"].dropna().values
        if len(pvals) == 0:
            continue
        k = int((pvals < 0.05).sum())
        n = len(pvals)
        p = k / n
        ci = stats.binomtest(k, n, p=0.05).proportion_ci(confidence_level=0.95, method="wilson")
        rows.append({
            "method": method, "n_genes": int(n_g),
            "fpr": p, "ci_lo": ci.low, "ci_hi": ci.high,
        })
    df = pd.DataFrame(rows)

    if gs_parent is None:
        gs_parent = fig.add_gridspec(1, 1)[0]
    # Same main y-range, ticks and gray line-only reference as the vs-participants
    # Type I panel, so the two calibration sub-panels read as one system.
    ax_main, ax_strip = _broken_pair(fig, gs_parent, main_ylim=(0.0, 0.10),
                                     strip_ylim=(0.68, 0.82))
    ax_main.axhline(0.05, color="#555555", linestyle="--", linewidth=1.0,
                    alpha=0.8, zorder=1)

    panel_sizes = sorted(_PANEL_SIZES)
    x_positions = np.arange(len(panel_sizes), dtype=float)
    n_to_x = dict(zip(panel_sizes, x_positions))
    method_offsets = {
        "wilcoxon_paired": -0.10,
        "nebula": -0.05,
        "limma_voom": 0.0,
        "sctrial_did": +0.05,
        "dreamlet": +0.09,
    }
    _ms_hi, _ms_lo = (5.0, 3.85) if composite else (9.0, 7.2)
    _lw_hi, _lw_lo = (1.25, 0.95) if composite else (2.0, 1.4)
    _cap_w = (2.0, 0.85) if composite else (4, 1.2)
    for method in _BENCH_METHODS:
        sub = df[df["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        # NEBULA (~0.75) goes on the upper strip; calibrated methods in the main axis.
        target = ax_strip if method == "nebula" else ax_main
        xs = np.array([n_to_x[int(n)] for n in sub["n_genes"].values]) + method_offsets[method]
        ys = sub["fpr"].values
        lo = sub["ci_lo"].values
        hi = sub["ci_hi"].values
        is_focal = method == "sctrial_did"
        target.errorbar(
            xs, ys, yerr=[ys - lo, hi - ys], fmt=_BENCH_METHOD_MARKERS[method],
            markersize=_ms_hi if is_focal else _ms_lo,
            color=_BENCH_METHOD_COLORS[method],
            markerfacecolor=_BENCH_METHOD_COLORS[method], markeredgecolor="white",
            markeredgewidth=0.6 if composite else 0.8,
            ecolor=_BENCH_METHOD_COLORS[method],
            elinewidth=1.0 if composite else 1.4,
            capsize=_cap_w[0], capthick=_cap_w[1],
            linestyle="-",
            linewidth=_lw_hi if is_focal else _lw_lo,
            alpha=0.92, zorder=10 if is_focal else 4,
        )
    _tk_fs = 5.05 if composite else 11
    _ttl_fs = 6.0 if composite else 12
    _leg_fs = 5.2 if composite else 8

    ax_main.set_xticks(x_positions)
    ax_main.set_xticklabels([f"{p:,}" for p in panel_sizes], fontsize=_tk_fs, rotation=0)
    ax_main.set_xlim(-0.35, len(panel_sizes) - 0.65)
    ax_main.set_xlabel("Panel size (genes)", fontsize=_tk_fs)
    ax_main.set_ylabel("Pure-null Type I error\n(p < 0.05)", fontsize=5.2 if composite else _tk_fs)
    ax_main.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax_strip.set_yticks([0.7, 0.8])
    for a in (ax_main, ax_strip):
        a.tick_params(labelsize=_tk_fs)
        _style_axis(a)
    # Standalone panel carries its own method key; in a composite the shared
    # top legend covers every panel, so the in-panel legend is omitted (it would
    # otherwise sit on the ~0.05 calibrated cluster).
    if not composite:
        ax_main.legend(
            handles=_bench_legend_handles(), loc="upper left", frameon=True,
            framealpha=0.95, edgecolor="#cccccc", fontsize=_leg_fs, ncol=2,
        )
    ax_strip.set_title(
        "Pure-null Type I error", fontsize=_ttl_fs, fontweight="bold",
        pad=(4 if composite else 8),
    )
    return ax_main, ax_strip
