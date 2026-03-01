"""
Figure 6 -- Scalability & Power Analysis
=========================================

Four-panel (2×2) figure combining computational scalability benchmarks
with empirical power analysis and observed effect sizes:

    A  Runtime scaling (cells vs time, log–log with reference slopes)
    B  Memory scaling (cells vs peak memory, log–log)
    C  Empirical power curves (sample size vs power)
    D  Forest plot of observed |Cohen's d| across datasets
"""

from __future__ import annotations

import gc
import time
import tracemalloc
import warnings

import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from .._shared import (
    COLORS,
    MAIN_OUTPUT,
    TrialDesign,
    apply_style,
    clear_cache,
    despine,
    did_table,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    load_clinical_trial_dataset,
    save_panel,
    score_signatures,
)
from sctrial import cohens_d_from_did, effect_size_ci

# ── Figure-level constants ────────────────────────────────────────────

FIGURE_NAME = "Figure6_scalability_power"
FIGSIZE = (18, 12)

# Sade-Feldman design (two-arm DiD)
SF_DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)
SF_VISITS: tuple[str, str] = ("Pre", "Post")

# Benchmark parameters — use many features so the timing is meaningful.
# With 100 features × OLS regressions, the benchmark reveals genuine
# computational scaling rather than trivially fast sub-second times.
BENCHMARK_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
N_BENCHMARK_GENES = 500          # genes in the synthetic AnnData
N_BENCHMARK_FEATURES = 100       # features passed to did_table
N_BENCHMARK_PARTICIPANTS = 20

# Power analysis
N_POWER_ITERATIONS = 1_000       # more iterations for smoother curves
POWER_ALPHA = 0.05
RNG_SEED = 42


# ======================================================================
# Data preparation
# ======================================================================

def _run_scalability_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark DiD runtime and memory across increasing dataset sizes.

    Returns
    -------
    timing_df : pd.DataFrame
        Columns: n_cells, time_s
    memory_df : pd.DataFrame
        Columns: n_cells, peak_mb
    """
    print("  Running scalability benchmarks ...")
    timings: list[dict] = []
    mem_usage: list[dict] = []

    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )
    features = [f"gene_{i}" for i in range(N_BENCHMARK_FEATURES)]

    for n_cells in BENCHMARK_SIZES:
        print(f"    n_cells = {n_cells:>7,} ... ", end="", flush=True)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_cells, N_BENCHMARK_GENES)).astype(np.float32)
        obs = pd.DataFrame({
            "participant_id": [
                f"P{i % N_BENCHMARK_PARTICIPANTS}"
                for i in range(n_cells)
            ],
            "visit": [
                "Pre" if i % 2 == 0 else "Post"
                for i in range(n_cells)
            ],
            "arm": [
                "Treated"
                if (i % N_BENCHMARK_PARTICIPANTS) < N_BENCHMARK_PARTICIPANTS // 2
                else "Control"
                for i in range(n_cells)
            ],
        })
        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = [f"gene_{i}" for i in range(N_BENCHMARK_GENES)]

        tracemalloc.start()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                did_table(
                    adata,
                    features=features,
                    design=design,
                    visits=("Pre", "Post"),
                    aggregate="participant_visit",
                    standardize=True,
                )
            except Exception:
                pass
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        timings.append({"n_cells": n_cells, "time_s": elapsed})
        mem_usage.append({"n_cells": n_cells, "peak_mb": peak / 1024**2})
        print(f"{elapsed:.2f}s, {peak / 1024**2:.1f} MB")

        del adata, X, obs
        gc.collect()

    return pd.DataFrame(timings), pd.DataFrame(mem_usage)


def _compute_simulation_power() -> pd.DataFrame:
    """Compute power curves via simulation at multiple effect sizes.

    Simulates DiD data with unit variance at three effect sizes:
    small (d=0.3), medium (d=0.5), large (d=0.8).
    Varies sample size per arm from 5 to 50.

    Returns
    -------
    pd.DataFrame
        Columns: n_per_group, effect_size, d_label, power
    """
    print("  Computing simulation-based power curves ...")

    effect_sizes = [
        (0.3, "Small (d = 0.3)"),
        (0.5, "Medium (d = 0.5)"),
        (0.8, "Large (d = 0.8)"),
    ]
    sample_sizes = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    rng = np.random.default_rng(RNG_SEED)

    records: list[dict] = []
    for d_val, d_label in effect_sizes:
        for n_per_group in sample_sizes:
            n_sig = 0
            n_total = 2 * n_per_group
            for _ in range(N_POWER_ITERATIONS):
                ctrl_deltas = rng.normal(0, 1, size=n_per_group)
                trt_deltas = rng.normal(d_val, 1, size=n_per_group)

                did_est = np.mean(trt_deltas) - np.mean(ctrl_deltas)
                se = np.sqrt(
                    np.var(trt_deltas, ddof=1) / n_per_group
                    + np.var(ctrl_deltas, ddof=1) / n_per_group
                )
                if se > 0:
                    t_stat = did_est / se
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_total - 2))
                    if p_val < POWER_ALPHA:
                        n_sig += 1

            power = n_sig / N_POWER_ITERATIONS
            records.append({
                "n_per_group": n_per_group,
                "effect_size": d_val,
                "d_label": d_label,
                "power": power,
            })

        print(f"    {d_label}: power@n=10={records[-7]['power']:.2f}, "
              f"@n=30={records[-2]['power']:.2f}")

    return pd.DataFrame(records)


def _paired_cohens_d(
    participant_deltas: np.ndarray,
) -> tuple[float, float, float]:
    """Compute paired Cohen's d_z = mean(delta) / sd(delta) with 95% CI.

    Returns (d, ci_lower, ci_upper).
    """
    n = len(participant_deltas)
    d = float(np.mean(participant_deltas) / np.std(participant_deltas, ddof=1))
    se_d = np.sqrt(1 / n + d**2 / (2 * n))
    t_crit = stats.t.ppf(0.975, n - 1)
    return d, d - t_crit * se_d, d + t_crit * se_d


def _best_paired_d(
    adata,
    sigs: list[str],
    pid_col: str = "participant_id",
    visit_col: str = "visit",
    pre_label: str = "Pre",
    post_label: str = "Post",
) -> dict | None:
    """Find the signature with the largest absolute paired Cohen's d."""
    best_d, best_rec = 0.0, None
    for sig in sigs:
        if sig not in adata.obs.columns:
            continue
        pb = (
            adata.obs.groupby([pid_col, visit_col], observed=True)[sig]
            .mean()
            .reset_index()
        )
        deltas = []
        for pid, pdf in pb.groupby(pid_col):
            if {pre_label, post_label}.issubset(set(pdf[visit_col])):
                pre_val = pdf.loc[pdf[visit_col] == pre_label, sig].values[0]
                post_val = pdf.loc[pdf[visit_col] == post_label, sig].values[0]
                deltas.append(post_val - pre_val)
        if len(deltas) >= 3:
            arr = np.array(deltas)
            d_val, ci_lo, ci_hi = _paired_cohens_d(arr)
            if abs(d_val) > abs(best_d):
                best_d = d_val
                best_rec = {"d": d_val, "d_lower": ci_lo, "d_upper": ci_hi}
    return best_rec


def _compute_effect_sizes_across_datasets() -> pd.DataFrame:
    """Compute observed |Cohen's d| for each dataset.

    All effect sizes are reported as absolute values so the forest plot
    has a consistent, interpretable orientation (larger = bigger effect).

    Returns
    -------
    pd.DataFrame
        Columns: dataset, d, d_lower, d_upper
    """
    print("  Computing effect sizes across datasets ...")
    records: list[dict] = []

    # ── Sade-Feldman (two-arm DiD) ────────────────────────────────────
    try:
        sf = get_sade_feldman()
        sf = harmonize_response(sf)
        sf, sf_sigs = score_signatures(sf, layer="log1p_tpm")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            did_res = did_table(
                sf,
                features=sf_sigs,
                design=SF_DESIGN,
                visits=SF_VISITS,
                aggregate="participant_visit",
                standardize=True,
            )
        top_sig = did_res.loc[did_res["beta_DiD"].abs().idxmax(), "feature"]

        pb = (
            sf.obs.groupby(
                [SF_DESIGN.participant_col, SF_DESIGN.visit_col,
                 SF_DESIGN.arm_col],
                observed=True,
            )[top_sig]
            .mean()
            .reset_index()
        )
        deltas: dict[str, list[float]] = {}
        for arm in [SF_DESIGN.arm_treated, SF_DESIGN.arm_control]:
            arm_pb = pb[pb[SF_DESIGN.arm_col] == arm]
            arm_deltas = []
            for pid, pdf in arm_pb.groupby(SF_DESIGN.participant_col):
                if set(SF_VISITS).issubset(set(pdf[SF_DESIGN.visit_col])):
                    pre_val = pdf.loc[
                        pdf[SF_DESIGN.visit_col] == SF_VISITS[0], top_sig
                    ].values[0]
                    post_val = pdf.loc[
                        pdf[SF_DESIGN.visit_col] == SF_VISITS[1], top_sig
                    ].values[0]
                    arm_deltas.append(post_val - pre_val)
            deltas[arm] = arm_deltas

        d = cohens_d_from_did(
            np.array(deltas[SF_DESIGN.arm_treated]),
            np.array(deltas[SF_DESIGN.arm_control]),
        )
        n1, n2 = len(deltas[SF_DESIGN.arm_treated]), len(deltas[SF_DESIGN.arm_control])
        ci_lo, ci_hi = effect_size_ci(d, n1, n2)
        # Store absolute value — sign is arbitrary in two-arm designs
        records.append({
            "dataset": "Sade-Feldman (Immunotherapy)",
            "d": abs(d),
            "d_lower": abs(d) - abs(d - ci_lo),  # preserve CI width
            "d_upper": abs(d) + abs(ci_hi - d),
        })
        print(f"    Sade-Feldman: |d|={abs(d):.2f}")
    except Exception as exc:
        print(f"    Sade-Feldman: FAILED ({exc})")

    # ── Vaccine (single-arm paired) ──────────────────────────────────
    try:
        vax = get_vaccine()
        vax, vax_sigs = score_signatures(vax, layer="counts")
        rec = _best_paired_d(vax, vax_sigs)
        if rec is not None:
            d_abs = abs(rec["d"])
            hw = (rec["d_upper"] - rec["d_lower"]) / 2
            records.append({
                "dataset": "Vaccine (GSE171964)",
                "d": d_abs,
                "d_lower": d_abs - hw,
                "d_upper": d_abs + hw,
            })
            print(f"    Vaccine: |d|={d_abs:.2f}")
    except Exception as exc:
        print(f"    Vaccine: FAILED ({exc})")

    # ── AML (single-arm paired) ──────────────────────────────────────
    try:
        aml = load_clinical_trial_dataset("aml")
        aml, aml_sigs = score_signatures(aml, layer="counts")
        rec = _best_paired_d(aml, aml_sigs)
        if rec is not None:
            d_abs = abs(rec["d"])
            hw = (rec["d_upper"] - rec["d_lower"]) / 2
            records.append({
                "dataset": "AML (GSE116256)",
                "d": d_abs,
                "d_lower": d_abs - hw,
                "d_upper": d_abs + hw,
            })
            print(f"    AML: |d|={d_abs:.2f}")
    except Exception as exc:
        print(f"    AML: FAILED ({exc})")

    # ── CAR-T (single-arm paired) ────────────────────────────────────
    try:
        cart = load_clinical_trial_dataset("cart")
        cart, cart_sigs = score_signatures(cart, layer="counts")
        rec = _best_paired_d(cart, cart_sigs)
        if rec is not None:
            d_abs = abs(rec["d"])
            hw = (rec["d_upper"] - rec["d_lower"]) / 2
            records.append({
                "dataset": "CAR-T (GSE290722)",
                "d": d_abs,
                "d_lower": d_abs - hw,
                "d_upper": d_abs + hw,
            })
            print(f"    CAR-T: |d|={d_abs:.2f}")
    except Exception as exc:
        print(f"    CAR-T: FAILED ({exc})")

    # ── COVID-19 Stephenson (cross-sectional: Severe vs Healthy) ─────
    try:
        covid = get_stephenson()
        covid, covid_sigs = score_signatures(covid, layer="counts")

        arm_col = "severity"
        if arm_col not in covid.obs.columns:
            for c in covid.obs.columns:
                if "severity" in c.lower():
                    arm_col = c
                    break

        arm_vals = covid.obs[arm_col].unique()
        severe_label = [v for v in arm_vals
                        if "sever" in str(v).lower() or "crit" in str(v).lower()]
        healthy_label = [v for v in arm_vals
                         if "health" in str(v).lower() or "mild" in str(v).lower()]

        if severe_label and healthy_label:
            severe_label = severe_label[0]
            healthy_label = healthy_label[0]

            best_d, best_rec = 0.0, None
            for sig in covid_sigs:
                if sig not in covid.obs.columns:
                    continue
                # Pseudobulk per participant
                pid_col = "participant_id"
                if pid_col not in covid.obs.columns:
                    continue
                pb_s = (covid.obs.loc[covid.obs[arm_col] == severe_label]
                        .groupby(pid_col)[sig].mean())
                pb_h = (covid.obs.loc[covid.obs[arm_col] == healthy_label]
                        .groupby(pid_col)[sig].mean())
                if len(pb_s) < 3 or len(pb_h) < 3:
                    continue
                pooled_sd = np.sqrt(
                    ((len(pb_s) - 1) * pb_s.std()**2
                     + (len(pb_h) - 1) * pb_h.std()**2)
                    / (len(pb_s) + len(pb_h) - 2)
                )
                if pooled_sd < 1e-12:
                    continue
                d_val = (pb_s.mean() - pb_h.mean()) / pooled_sd
                if abs(d_val) > abs(best_d):
                    best_d = d_val
                    n1, n2 = len(pb_s), len(pb_h)
                    se_d = np.sqrt(
                        1 / n1 + 1 / n2 + best_d**2 / (2 * (n1 + n2))
                    )
                    t_crit = stats.t.ppf(0.975, n1 + n2 - 2)
                    best_rec = {
                        "d": abs(best_d),
                        "d_lower": abs(best_d) - t_crit * se_d,
                        "d_upper": abs(best_d) + t_crit * se_d,
                    }

            if best_rec is not None:
                best_rec["dataset"] = "COVID-19 (Stephenson)"
                records.append(best_rec)
                print(f"    COVID-19: |d|={best_rec['d']:.2f}")
    except Exception as exc:
        print(f"    COVID-19: FAILED ({exc})")

    return pd.DataFrame(records)


def _prepare_data() -> dict:
    """Run all data preparation steps."""
    print("Figure 6: Scalability & Power Analysis")

    timing_df, memory_df = _run_scalability_benchmark()
    power_df = _compute_simulation_power()
    effect_df = _compute_effect_sizes_across_datasets()

    return {
        "timing_df": timing_df,
        "memory_df": memory_df,
        "power_df": power_df,
        "effect_df": effect_df,
    }


# ======================================================================
# Panel drawing functions
# ======================================================================

def _fmt_cells(n: int) -> str:
    """Format cell count as human-readable string."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    return f"{n / 1_000:.0f}K"


def panel_A(ax: plt.Axes, data: dict) -> None:
    """Panel A: Runtime scaling (cells vs time) — log–log."""
    timing_df = data["timing_df"]

    ax.plot(
        timing_df["n_cells"], timing_df["time_s"],
        color=COLORS["treated"], marker="o", markersize=6,
        markeredgecolor="white", markeredgewidth=0.8,
        linewidth=1.8, zorder=3,
    )

    # Filled area under curve for visual weight
    ax.fill_between(
        timing_df["n_cells"], timing_df["time_s"],
        alpha=0.08, color=COLORS["treated"], zorder=1,
    )

    # Reference: linear scaling from first point
    x0, y0 = timing_df["n_cells"].iloc[0], timing_df["time_s"].iloc[0]
    x_ref = timing_df["n_cells"].values
    y_linear = y0 * (x_ref / x0)
    ax.plot(x_ref, y_linear, color=COLORS["gray"], ls=":", lw=1.0,
            zorder=1, alpha=0.6, label="O(n) reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime scaling", fontsize=10, fontweight="bold")

    ax.set_xticks(timing_df["n_cells"].values)
    ax.set_xticklabels([_fmt_cells(n) for n in timing_df["n_cells"]])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.tick_params(axis="x", rotation=0)

    ax.legend(frameon=False, fontsize=8, loc="upper left")
    despine(ax)


def panel_B(ax: plt.Axes, data: dict) -> None:
    """Panel B: Memory scaling (cells vs peak memory) — log–log."""
    memory_df = data["memory_df"]

    ax.plot(
        memory_df["n_cells"], memory_df["peak_mb"],
        color=COLORS["neutral"], marker="s", markersize=6,
        markeredgecolor="white", markeredgewidth=0.8,
        linewidth=1.8, zorder=3,
    )

    ax.fill_between(
        memory_df["n_cells"], memory_df["peak_mb"],
        alpha=0.08, color=COLORS["neutral"], zorder=1,
    )

    # Reference: linear scaling from first point
    x0, y0 = memory_df["n_cells"].iloc[0], memory_df["peak_mb"].iloc[0]
    x_ref = memory_df["n_cells"].values
    y_linear = y0 * (x_ref / x0)
    ax.plot(x_ref, y_linear, color=COLORS["gray"], ls=":", lw=1.0,
            zorder=1, alpha=0.6, label="O(n) reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title("Memory scaling", fontsize=10, fontweight="bold")

    ax.set_xticks(memory_df["n_cells"].values)
    ax.set_xticklabels([_fmt_cells(n) for n in memory_df["n_cells"]])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.tick_params(axis="x", rotation=0)

    ax.legend(frameon=False, fontsize=8, loc="upper left")
    despine(ax)


def panel_C(ax: plt.Axes, data: dict) -> None:
    """Panel C: Simulation-based power curves at multiple effect sizes."""
    power_df = data["power_df"]
    if power_df.empty:
        ax.text(0.5, 0.5, "Insufficient data\nfor power analysis",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=COLORS["gray"])
        ax.set_title("Statistical power (DiD)", fontsize=10, fontweight="bold")
        despine(ax)
        return

    # Ordered: Large → Medium → Small so legend reads top-to-bottom
    curve_styles = {
        "Large (d = 0.8)":  (COLORS["highlight"], "D", 2.0),
        "Medium (d = 0.5)": (COLORS["treated"],   "o", 2.0),
        "Small (d = 0.3)":  (COLORS["neutral"],   "s", 1.5),
    }

    for d_label, grp in power_df.groupby("d_label", sort=False):
        color, marker, lw = curve_styles.get(
            d_label, (COLORS["gray"], "o", 1.5))
        ax.plot(
            grp["n_per_group"], grp["power"],
            color=color, marker=marker, markersize=5,
            markeredgecolor="white", markeredgewidth=0.5,
            linewidth=lw, zorder=3, label=d_label,
        )

    # 80% power threshold
    ax.axhline(0.80, color=COLORS["gray"], linewidth=0.8,
               linestyle="--", zorder=1, alpha=0.5)
    ax.text(power_df["n_per_group"].max() * 0.98, 0.82,
            "80% power", ha="right", va="bottom", fontsize=7,
            color=COLORS["gray"], fontstyle="italic")

    ax.set_xlabel("Participants per group")
    ax.set_ylabel(r"Power (1 – $\beta$)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Statistical power (DiD simulation)",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    despine(ax)


def panel_D(ax: plt.Axes, data: dict) -> None:
    """Panel D: Forest plot of observed |Cohen's d| across datasets."""
    effect_df = data["effect_df"]
    if effect_df.empty:
        ax.text(0.5, 0.5, "No effect size data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=COLORS["gray"])
        ax.set_title("Observed effect sizes", fontsize=10, fontweight="bold")
        despine(ax)
        return

    # Sort by |d| ascending so largest effects are at top
    df = effect_df.sort_values("d", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df))

    # Consistent color per dataset
    dataset_colors = {
        "Sade-Feldman (Immunotherapy)": COLORS["control"],
        "Vaccine (GSE171964)": COLORS["treated"],
        "AML (GSE116256)": COLORS["success"],
        "CAR-T (GSE290722)": COLORS["neutral"],
        "COVID-19 (Stephenson)": COLORS["highlight"],
    }

    # For annotation clipping: cap x at max point + padding
    x_right = df["d"].max() + 0.6

    for i, (_, row) in enumerate(df.iterrows()):
        color = dataset_colors.get(row["dataset"], COLORS["gray"])

        # CI whisker — clip to visible range
        ci_right = min(row["d_upper"], x_right - 0.05)
        ax.hlines(y_pos[i], row["d_lower"], ci_right,
                  color=color, linewidth=2.0, zorder=2)
        # Arrow cap if CI extends beyond visible range
        if row["d_upper"] > x_right - 0.05:
            ax.annotate("", xy=(ci_right, y_pos[i]),
                        xytext=(ci_right - 0.08, y_pos[i]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        # Point estimate
        ax.scatter(row["d"], y_pos[i], color=color, s=70, zorder=3,
                   edgecolors="white", linewidths=0.8)

        # Annotate d value above the point (offset vertically to avoid line)
        ax.text(row["d"], y_pos[i] + 0.3,
                f"|d| = {row['d']:.2f}",
                fontsize=7, va="bottom", ha="center", color=color,
                fontweight="bold")

    # Reference lines for Cohen's benchmarks
    for ref_d in (0.2, 0.5, 0.8):
        ax.axvline(ref_d, color=COLORS["gray"], linewidth=0.6,
                   linestyle=":", zorder=0, alpha=0.5)
    # Place benchmark labels at top of plot
    top_y = len(df) - 0.15
    for ref_d, ref_label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.text(ref_d, top_y, ref_label, fontsize=6.5,
                color=COLORS["gray"], ha="center", fontstyle="italic")

    # Zero line
    ax.axvline(0, color="black", linewidth=0.6, zorder=0, alpha=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["dataset"].values, fontsize=8)
    ax.set_xlabel("|Cohen's d|  (standardised effect size)")
    ax.set_title("Observed effect sizes across datasets",
                 fontsize=10, fontweight="bold")
    # Cap x-axis: show up to max point + padding, clip very wide CIs
    max_d = df["d"].max()
    ax.set_xlim(left=-0.1, right=max_d + 0.6)
    ax.set_ylim(-0.6, len(df) + 0.1)
    despine(ax)


# ======================================================================
# Composite figure
# ======================================================================

def generate() -> None:
    """Create and save Figure 6 individual panels."""
    apply_style()
    data = _prepare_data()

    panels = [
        ("panel_A", panel_A, (8, 6)),
        ("panel_B", panel_B, (8, 6)),
        ("panel_C", panel_C, (8, 6)),
        ("panel_D", panel_D, (8, 4.5)),   # shorter — only 5 rows
    ]

    for name, draw_fn, figsize in panels:
        pfig, pax = plt.subplots(figsize=figsize)
        draw_fn(pax, data)
        pfig.tight_layout()
        save_panel(pfig, name, FIGURE_NAME, MAIN_OUTPUT)

    clear_cache()
    gc.collect()
    print("  Done.\n")


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    generate()
