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
import hashlib
import time
import tracemalloc
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sctrial import cohens_d_from_did, effect_size_ci

from .._shared import (
    COLORS,
    MAIN_OUTPUT,
    TrialDesign,
    apply_style,
    between_arm_comparison,
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

# Power analysis (subsampling)
N_POWER_ITERATIONS = 50          # resamples per sample size (cached after first run)
POWER_ALPHA = 0.05
RNG_SEED = 42

# Dataset info tuple fields:
#   (name, adata, design, visits, sig_cols, design_type)
# design_type: "two_arm_did" | "paired" | "cross_sectional"
DatasetInfo = tuple[str, object, object, tuple, list[str], str]


# ======================================================================
# Disk cache for expensive computations
# ======================================================================

_CACHE_DIR = MAIN_OUTPUT / FIGURE_NAME.replace("Figure", "Figure") / ".cache"


def _cache_key(*args: str) -> str:
    """Deterministic cache key from string components."""
    return hashlib.md5("|".join(args).encode()).hexdigest()[:12]


def _load_cache(tag: str) -> pd.DataFrame | None:
    """Try to load a cached DataFrame from JSON."""
    path = _CACHE_DIR / f"{tag}.json"
    if path.exists():
        try:
            return pd.read_json(path, orient="records")
        except Exception:
            return None
    return None


def _save_cache(tag: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame to JSON cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(_CACHE_DIR / f"{tag}.json", orient="records", indent=2)


# ======================================================================
# Data preparation
# ======================================================================

def _load_all_datasets() -> list[DatasetInfo]:
    """Load and score all 5 real datasets.

    Returns a list of ``DatasetInfo`` tuples sorted by n_obs ascending
    (for plotting on a monotonic x-axis).  Each tuple is
    ``(name, adata, design, visits, sig_cols, design_type)``.
    """
    datasets: list[DatasetInfo] = []

    # ── Sade-Feldman (two-arm DiD) ────────────────────────────────────
    try:
        sf = get_sade_feldman()
        sf = harmonize_response(sf)
        sf, sf_sigs = score_signatures(sf, layer="log1p_tpm")
        datasets.append(
            ("Sade-Feldman", sf, SF_DESIGN, SF_VISITS, sf_sigs, "two_arm_did")
        )
    except Exception as exc:
        print(f"    Sade-Feldman: FAILED to load ({exc})")

    # ── Vaccine (single-arm paired) ───────────────────────────────────
    try:
        vax = get_vaccine()
        vax, vax_sigs = score_signatures(vax, layer="counts")
        vax.obs["arm_dummy"] = "Vaccinated"
        vax_design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm_dummy",
            arm_treated="Vaccinated",
            arm_control="Vaccinated",
        )
        datasets.append(
            ("Vaccine", vax, vax_design, ("Pre", "Post"), vax_sigs, "paired")
        )
    except Exception as exc:
        print(f"    Vaccine: FAILED to load ({exc})")

    # ── AML (single-arm paired) ───────────────────────────────────────
    try:
        aml = load_clinical_trial_dataset("aml")
        aml, aml_sigs = score_signatures(aml, layer="counts")
        aml.obs["arm_dummy"] = "Treatment"
        pid_col = ("participant_id" if "participant_id" in aml.obs.columns
                   else "patient_id")
        aml_design = TrialDesign(
            participant_col=pid_col,
            visit_col="visit",
            arm_col="arm_dummy",
            arm_treated="Treatment",
            arm_control="Treatment",
        )
        datasets.append(
            ("AML", aml, aml_design, ("Pre", "Post"), aml_sigs, "paired")
        )
    except Exception as exc:
        print(f"    AML: FAILED to load ({exc})")

    # ── CAR-T (single-arm paired) ─────────────────────────────────────
    try:
        cart = load_clinical_trial_dataset("cart")
        cart, cart_sigs = score_signatures(cart, layer="counts")
        cart.obs["arm_dummy"] = "CAR-T"
        pid_col = ("participant_id" if "participant_id" in cart.obs.columns
                   else "patient_id")
        cart_design = TrialDesign(
            participant_col=pid_col,
            visit_col="visit",
            arm_col="arm_dummy",
            arm_treated="CAR-T",
            arm_control="CAR-T",
        )
        datasets.append(
            ("CAR-T", cart, cart_design, ("Pre", "Post"), cart_sigs, "paired")
        )
    except Exception as exc:
        print(f"    CAR-T: FAILED to load ({exc})")

    # ── COVID-19 Stephenson (cross-sectional: Severe vs Mild) ─────────
    try:
        covid = get_stephenson()
        covid, covid_sigs = score_signatures(covid, layer="counts")
        # Cross-sectional: participants appear in only one dfo_bin.
        # Use most populated bin for benchmark visit, severity as arm.
        if "dfo_bin" in covid.obs.columns:
            top_bin = covid.obs["dfo_bin"].value_counts().idxmax()
        else:
            top_bin = "Pre"
        covid_design = TrialDesign(
            participant_col="participant_id",
            visit_col="dfo_bin",
            arm_col="severity",
            arm_treated="Severe",
            arm_control="Mild",
        )
        # Store the single cross-sectional visit (used for benchmark)
        datasets.append(
            ("COVID-19", covid, covid_design, (top_bin,), covid_sigs,
             "cross_sectional")
        )
    except Exception as exc:
        print(f"    COVID-19: FAILED to load ({exc})")

    # Sort by cell count for a clean x-axis
    datasets.sort(key=lambda t: t[1].n_obs)
    return datasets


def _run_scalability_benchmark(
    datasets: list[DatasetInfo],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark sctrial runtime and memory on each real dataset.

    Uses ``did_table`` for longitudinal datasets and
    ``between_arm_comparison`` for cross-sectional ones.

    Returns
    -------
    timing_df : pd.DataFrame
        Columns: n_cells, time_s, dataset
    memory_df : pd.DataFrame
        Columns: n_cells, peak_mb, dataset
    """
    # Check cache
    cache_tag = "benchmark_" + _cache_key(
        *[f"{n}:{a.n_obs}" for n, a, *_ in datasets]
    )
    cached_t = _load_cache(cache_tag + "_time")
    cached_m = _load_cache(cache_tag + "_mem")
    if cached_t is not None and cached_m is not None:
        print("  Scalability benchmarks (cached)")
        return cached_t, cached_m

    print("  Running scalability benchmarks on real datasets ...")
    timings: list[dict] = []
    mem_usage: list[dict] = []

    for name, adata, design, visits, sigs, dtype in datasets:
        n_cells = adata.n_obs
        print(f"    {name} ({n_cells:,} cells, {dtype}) ... ",
              end="", flush=True)

        # Force GC before measurement to reduce noise
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()          # reset peak so we only measure did_table
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if dtype == "cross_sectional":
                # Cross-sectional: use between_arm_comparison at one visit
                between_arm_comparison(
                    adata,
                    visit=visits[0],
                    features=sigs,
                    design=design,
                    aggregate="participant_visit",
                    standardize=True,
                )
            else:
                # Longitudinal (two_arm_did or paired): use did_table
                did_table(
                    adata,
                    features=sigs,
                    design=design,
                    visits=visits,
                    aggregate="participant_visit",
                    standardize=True,
                )
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        timings.append({
            "n_cells": n_cells, "time_s": elapsed, "dataset": name,
            "n_genes": adata.n_vars, "design_type": dtype,
        })
        mem_usage.append({
            "n_cells": n_cells, "peak_mb": peak / 1024**2, "dataset": name,
            "n_genes": adata.n_vars, "design_type": dtype,
        })
        print(f"{elapsed:.2f}s, {peak / 1024**2:.1f} MB")

        gc.collect()

    timing_df = pd.DataFrame(timings)
    memory_df = pd.DataFrame(mem_usage)
    _save_cache(cache_tag + "_time", timing_df)
    _save_cache(cache_tag + "_mem", memory_df)
    return timing_df, memory_df


def _identify_best_feature(
    adata, sigs: list[str], design, visits: tuple, dtype: str,
) -> tuple[str, float] | None:
    """Run a full analysis on all data and return the most significant feature.

    This identifies the "oracle" feature for power analysis — simulating a
    scenario where a researcher pre-registers the best signature.
    Always uses the raw (unadjusted) p-value to pick the best feature.

    Returns (feature_name, p_value) or None if no feature found.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dtype == "cross_sectional":
            res = between_arm_comparison(
                adata, visit=visits[0], features=sigs,
                design=design, aggregate="participant_visit",
                standardize=True,
            )
        else:
            res = did_table(
                adata, features=sigs, design=design,
                visits=visits, aggregate="participant_visit",
                standardize=True,
            )
    if res.empty:
        return None
    # Use raw p-value column (p_DiD for did_table, p_arm for between_arm)
    p_col = next(
        (c for c in ("p_DiD", "p_arm", "p_value") if c in res.columns), None
    )
    if p_col is None:
        return None
    best_idx = res[p_col].idxmin()
    return res.loc[best_idx, "feature"], float(res.loc[best_idx, p_col])


def _compute_subsampling_power(
    datasets: list[DatasetInfo],
) -> pd.DataFrame:
    """Compute empirical power via participant subsampling on real datasets.

    Strategy: First identify the best feature from the full sample, then
    test only that single feature at each subsample size (unadjusted
    p-value).  This simulates pre-registered power for a single endpoint.

    Returns
    -------
    pd.DataFrame
        Columns: n_participants, dataset, power
    """
    # Check cache
    cache_tag = "power_v2_" + _cache_key(
        *[f"{n}:{a.n_obs}" for n, a, *_ in datasets],
        str(N_POWER_ITERATIONS),
    )
    cached = _load_cache(cache_tag)
    if cached is not None:
        print("  Empirical power curves (cached)")
        return cached

    print("  Computing empirical power curves (participant subsampling) ...")
    rng = np.random.default_rng(RNG_SEED)
    records: list[dict] = []

    # Cap: skip datasets with >100K cells (memory-intensive to subsample)
    MAX_CELLS_FOR_POWER = 100_000

    for name, adata, design, visits, sigs, dtype in datasets:
        pid_col = design.participant_col
        all_pids = adata.obs[pid_col].unique()
        n_total = len(all_pids)
        if n_total < 6:
            print(f"    {name}: too few participants ({n_total}), skipping")
            continue
        if adata.n_obs > MAX_CELLS_FOR_POWER:
            print(f"    {name}: too many cells ({adata.n_obs:,}) for "
                  "subsampling, skipping")
            continue

        # Identify best feature from full sample
        result = _identify_best_feature(adata, sigs, design, visits, dtype)
        if result is None:
            print(f"    {name}: no feature found, skipping")
            continue
        best_feat, best_p = result

        # Skip if even the full sample doesn't reach significance
        if best_p >= 0.05:
            print(f"    {name}: best feature not significant at full sample "
                  f"(p={best_p:.3f}), skipping power analysis")
            continue

        # Choose subsample sizes: from 4 up to n_total
        sub_sizes = sorted(set(
            [4, 6, 8] +
            list(range(5, min(n_total, 30) + 1, 5)) +
            [n_total]
        ))
        sub_sizes = [s for s in sub_sizes if s <= n_total]

        print(f"    {name} ({n_total} ppts, {dtype}, feat={best_feat}): ",
              end="", flush=True)

        for n_sub in sub_sizes:
            n_sig = 0
            for _ in range(N_POWER_ITERATIONS):
                sampled_pids = rng.choice(all_pids, size=n_sub, replace=False)
                mask = adata.obs[pid_col].isin(sampled_pids)
                sub_adata = adata[mask].copy()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        if dtype == "cross_sectional":
                            res = between_arm_comparison(
                                sub_adata,
                                visit=visits[0],
                                features=[best_feat],
                                design=design,
                                aggregate="participant_visit",
                                standardize=True,
                            )
                        else:
                            res = did_table(
                                sub_adata,
                                features=[best_feat],
                                design=design,
                                visits=visits,
                                aggregate="participant_visit",
                                standardize=True,
                            )
                        # Use raw p-value for single pre-specified test
                        p_col = next(
                            (c for c in ("p_DiD", "p_arm", "p_value")
                             if c in res.columns),
                            None,
                        )
                        if p_col is None:
                            continue
                        if res[p_col].iloc[0] < POWER_ALPHA:
                            n_sig += 1
                    except Exception:
                        pass  # subsample may be too small for model

            power = n_sig / N_POWER_ITERATIONS
            records.append({
                "n_participants": n_sub,
                "dataset": name,
                "power": power,
            })

        powers = [r["power"] for r in records if r["dataset"] == name]
        print(f"power range [{min(powers):.2f}, {max(powers):.2f}]")

    power_df = pd.DataFrame(records)
    _save_cache(cache_tag, power_df)
    return power_df


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


def _abs_d_with_ci(d: float, ci_lo: float, ci_hi: float) -> dict:
    """Convert a signed Cohen's d with CI to absolute |d| with valid CI.

    Properly handles the folding so lower bound is clamped to ≥ 0.
    """
    d_abs = abs(d)
    # Preserve CI half-width, clamp lower to 0
    hw = (ci_hi - ci_lo) / 2.0
    return {
        "d": d_abs,
        "d_lower": max(0.0, d_abs - hw),
        "d_upper": d_abs + hw,
    }


def _compute_effect_sizes_across_datasets(
    datasets: list[DatasetInfo],
) -> pd.DataFrame:
    """Compute signed Cohen's d for the best signature in each dataset.

    For each dataset, the signature with the largest absolute effect is
    selected.  Effect sizes are **signed** so the forest plot shows
    direction of change.  Each row records the signature name, design
    type, and participant count for annotation.

    Returns
    -------
    pd.DataFrame
        Columns: dataset, signature, d, d_lower, d_upper,
                 design_type, n_participants
    """
    # Check cache
    cache_tag = "effects_v3_" + _cache_key(
        *[f"{n}:{a.n_obs}" for n, a, *_ in datasets],
    )
    cached = _load_cache(cache_tag)
    if cached is not None:
        print("  Effect sizes (cached)")
        return cached

    print("  Computing signed effect sizes (best signature per dataset) ...")
    records: list[dict] = []

    for name, adata, design, visits, sigs, dtype in datasets:
        pid_col = design.participant_col

        try:
            best_d, best_rec = 0.0, None

            for sig in sigs:
                if sig not in adata.obs.columns:
                    continue

                if dtype == "two_arm_did":
                    pb = (
                        adata.obs.groupby(
                            [pid_col, design.visit_col, design.arm_col],
                            observed=True,
                        )[sig].mean().reset_index()
                    )
                    deltas: dict[str, list[float]] = {}
                    for arm in [design.arm_treated, design.arm_control]:
                        arm_pb = pb[pb[design.arm_col] == arm]
                        arm_d: list[float] = []
                        for _, pdf in arm_pb.groupby(pid_col):
                            if set(visits).issubset(
                                set(pdf[design.visit_col])
                            ):
                                pre = pdf.loc[
                                    pdf[design.visit_col] == visits[0], sig
                                ].values[0]
                                post = pdf.loc[
                                    pdf[design.visit_col] == visits[1], sig
                                ].values[0]
                                arm_d.append(post - pre)
                        deltas[arm] = arm_d

                    n1 = len(deltas[design.arm_treated])
                    n2 = len(deltas[design.arm_control])
                    if n1 < 2 or n2 < 2:
                        continue
                    d_val = cohens_d_from_did(
                        np.array(deltas[design.arm_treated]),
                        np.array(deltas[design.arm_control]),
                    )
                    ci_lo, ci_hi = effect_size_ci(d_val, n1, n2)
                    n_ppt = n1 + n2

                elif dtype == "paired":
                    pb = (
                        adata.obs.groupby(
                            [pid_col, design.visit_col], observed=True,
                        )[sig].mean().reset_index()
                    )
                    ds: list[float] = []
                    for _, pdf in pb.groupby(pid_col):
                        if set(visits).issubset(
                            set(pdf[design.visit_col])
                        ):
                            pre = pdf.loc[
                                pdf[design.visit_col] == visits[0], sig
                            ].values[0]
                            post = pdf.loc[
                                pdf[design.visit_col] == visits[1], sig
                            ].values[0]
                            ds.append(post - pre)
                    if len(ds) < 3:
                        continue
                    d_val, ci_lo, ci_hi = _paired_cohens_d(np.array(ds))
                    n_ppt = len(ds)

                elif dtype == "cross_sectional":
                    pb_t = (adata.obs.loc[
                        adata.obs[design.arm_col] == design.arm_treated
                    ].groupby(pid_col)[sig].mean())
                    pb_c = (adata.obs.loc[
                        adata.obs[design.arm_col] == design.arm_control
                    ].groupby(pid_col)[sig].mean())
                    n1, n2 = len(pb_t), len(pb_c)
                    if n1 < 3 or n2 < 3:
                        continue
                    pooled_sd = np.sqrt(
                        ((n1 - 1) * pb_t.std()**2
                         + (n2 - 1) * pb_c.std()**2)
                        / (n1 + n2 - 2)
                    )
                    if pooled_sd < 1e-12:
                        continue
                    d_val = float((pb_t.mean() - pb_c.mean()) / pooled_sd)
                    ci_lo, ci_hi = effect_size_ci(d_val, n1, n2)
                    n_ppt = n1 + n2
                else:
                    continue

                if abs(d_val) > abs(best_d):
                    best_d = d_val
                    best_rec = {
                        "dataset": name,
                        "signature": sig.replace("sig_", ""),
                        "d": d_val,
                        "d_lower": ci_lo,
                        "d_upper": ci_hi,
                        "design_type": dtype,
                        "n_participants": n_ppt,
                    }

            if best_rec is not None:
                records.append(best_rec)
                print(f"    {name}: d={best_rec['d']:+.2f} "
                      f"[{best_rec['d_lower']:+.2f}, "
                      f"{best_rec['d_upper']:+.2f}] "
                      f"({best_rec['signature']}, n={best_rec['n_participants']})")
            else:
                print(f"    {name}: no valid signatures")

        except Exception as exc:
            print(f"    {name}: FAILED ({exc})")

    effect_df = pd.DataFrame(records)
    _save_cache(cache_tag, effect_df)
    return effect_df


def _prepare_data() -> dict:
    """Run all data preparation steps using real datasets only."""
    print("Figure 6: Scalability & Power Analysis")

    # Load all real datasets once — reused across panels
    datasets = _load_all_datasets()

    timing_df, memory_df = _run_scalability_benchmark(datasets)
    power_df = _compute_subsampling_power(datasets)
    effect_df = _compute_effect_sizes_across_datasets(datasets)

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


def _scaling_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    y_ticks: list[float] | None = None,
    y_tick_labels: list[str] | None = None,
) -> None:
    """Shared helper: log-log scatter of (n_cells × n_genes) vs metric."""
    from matplotlib.ticker import FixedLocator, NullLocator

    dtype_colors = {
        "two_arm_did": COLORS["control"],
        "paired": COLORS["treated"],
        "cross_sectional": COLORS["highlight"],
    }
    dtype_labels = {
        "two_arm_did": "Two-arm DiD",
        "paired": "Paired pre/post",
        "cross_sectional": "Cross-sectional",
    }

    # x-axis = n_cells × n_genes (proportional to matrix size)
    matrix_size = (df["n_cells"] * df["n_genes"]).values.astype(float)
    y_vals = df[y_col].values.astype(float)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # ── Subtle grid ──
    ax.grid(True, which="major", axis="both", color="#e0e0e0",
            linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ── Fit power-law trend:  y = a * (cells×genes)^b ──
    log_x = np.log10(matrix_size)
    log_y = np.log10(np.clip(y_vals, 1e-6, None))
    b, log_a = np.polyfit(log_x, log_y, 1)

    x_trend = np.logspace(
        np.log10(matrix_size.min() * 0.5),
        np.log10(matrix_size.max() * 2.0),
        100,
    )
    y_trend = 10 ** (log_a + b * np.log10(x_trend))
    ax.plot(x_trend, y_trend, "--", color="#999", alpha=0.6,
            linewidth=1.2, zorder=1,
            label=f"$\\propto n^{{{b:.2f}}}$")

    # ── Scatter points: color = design type ──
    plotted_dtypes: set = set()
    for i, (_, row) in enumerate(df.iterrows()):
        dt = row.get("design_type", "paired")
        c = dtype_colors.get(dt, COLORS["treated"])
        lbl = dtype_labels.get(dt) if dt not in plotted_dtypes else None
        plotted_dtypes.add(dt)
        ax.scatter(
            matrix_size[i], y_vals[i],
            s=80, color=c, edgecolors="white", linewidths=0.8,
            zorder=4, label=lbl, alpha=0.92,
        )

    # ── Dataset labels with adjustText ──
    from adjustText import adjust_text  # noqa: E402
    texts = []
    for i, (_, row) in enumerate(df.iterrows()):
        n_c = row["n_cells"]
        n_g = row["n_genes"]
        cell_str = (f"{n_c / 1_000_000:.0f}M" if n_c >= 1_000_000
                    else f"{n_c / 1_000:.0f}K")
        gene_str = f"{n_g // 1000}K"
        lbl = f"{row['dataset']}  ({cell_str} × {gene_str})"
        texts.append(
            ax.text(
                matrix_size[i], y_vals[i], lbl,
                fontsize=7, color="#444", fontstyle="italic",
            )
        )
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6),
        ensure_inside_axes=True,
        min_arrow_len=5,
    )

    # ── Explicit x-axis ticks (scientific notation via LaTeX) ──
    x_ticks = [3e8, 5e8, 1e9, 2e9, 5e9]
    x_labels = [
        r"$3{\times}10^8$", r"$5{\times}10^8$", r"$10^9$",
        r"$2{\times}10^9$", r"$5{\times}10^9$",
    ]
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.set_xticklabels(x_labels)
    ax.xaxis.set_minor_locator(NullLocator())

    # ── Explicit y-axis ticks ──
    if y_ticks is not None and y_tick_labels is not None:
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.set_yticklabels(y_tick_labels)
        ax.yaxis.set_minor_locator(NullLocator())

    ax.set_xlabel("Matrix size  (cells × genes)")
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8, frameon=True, fancybox=True,
              framealpha=0.85, edgecolor="#ddd", loc="lower right")
    despine(ax)


def panel_A(ax: plt.Axes, data: dict) -> None:
    """Panel A: Runtime scaling — log-log scatter with trend line."""
    _scaling_scatter(
        ax, data["timing_df"],
        y_col="time_s", y_label="Runtime (seconds)",
        title="Computational scaling",
        y_ticks=[0.1, 0.2, 0.5, 1.0, 2.0],
        y_tick_labels=["0.1", "0.2", "0.5", "1.0", "2.0"],
    )


def panel_B(ax: plt.Axes, data: dict) -> None:
    """Panel B: Memory scaling — log-log scatter with trend line."""
    _scaling_scatter(
        ax, data["memory_df"],
        y_col="peak_mb", y_label="Peak memory",
        title="Memory scaling",
        y_ticks=[512, 1024, 2048, 4096, 8192, 12288],
        y_tick_labels=["0.5 GB", "1 GB", "2 GB", "4 GB", "8 GB", "12 GB"],
    )


def panel_C(ax: plt.Axes, data: dict) -> None:
    """Panel C: Empirical power via participant subsampling."""
    power_df = data["power_df"]
    if power_df.empty:
        ax.text(0.5, 0.5, "Insufficient data\nfor power analysis",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=COLORS["gray"])
        ax.set_title("Empirical power", fontsize=11, fontweight="bold")
        despine(ax)
        return

    # Color palette consistent with Panel D
    dataset_colors = {
        "Sade-Feldman": COLORS["control"],
        "Vaccine":      COLORS["treated"],
        "AML":          COLORS["success"],
        "CAR-T":        COLORS["neutral"],
        "COVID-19":     COLORS["highlight"],
    }
    markers = {
        "Sade-Feldman": "o",
        "Vaccine":      "s",
        "AML":          "D",
        "CAR-T":        "^",
        "COVID-19":     "v",
    }

    # Subtle grid
    ax.grid(True, which="major", axis="y", color="#e8e8e8",
            linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    for ds_name, grp in power_df.groupby("dataset", sort=False):
        color = dataset_colors.get(ds_name, COLORS["gray"])
        marker = markers.get(ds_name, "o")
        ax.plot(
            grp["n_participants"], grp["power"],
            color=color, marker=marker, markersize=6,
            markeredgecolor="white", markeredgewidth=0.6,
            linewidth=2.0, zorder=3, label=ds_name,
        )

    # 80% power threshold
    ax.axhline(0.80, color=COLORS["gray"], linewidth=0.8,
               linestyle="--", zorder=1, alpha=0.5)
    x_max = power_df["n_participants"].max()
    ax.text(x_max * 0.98, 0.82,
            "80% power", ha="right", va="bottom", fontsize=7.5,
            color=COLORS["gray"], fontstyle="italic")

    ax.set_xlabel("Number of participants")
    ax.set_ylabel(r"Power (1 − $\beta$)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Empirical power", fontsize=11, fontweight="bold", pad=10)
    ax.legend(frameon=True, fancybox=True, framealpha=0.85,
              edgecolor="#ddd", fontsize=8, loc="lower right")
    despine(ax)


def panel_D(ax: plt.Axes, data: dict) -> None:
    """Panel D: Forest plot of signed Cohen's d, grouped by design type."""
    effect_df = data["effect_df"]
    if effect_df.empty:
        ax.text(0.5, 0.5, "No effect size data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=COLORS["gray"])
        ax.set_title("Observed effect sizes", fontsize=10, fontweight="bold")
        despine(ax)
        return

    # Group by design type and sort within group by |d|
    type_order = ["two_arm_did", "paired", "cross_sectional"]
    type_labels = {
        "two_arm_did": "Two-arm DiD",
        "paired": "Paired pre/post",
        "cross_sectional": "Cross-sectional",
    }
    dataset_colors = {
        "Sade-Feldman": COLORS["control"],
        "Vaccine":      COLORS["treated"],
        "AML":          COLORS["success"],
        "CAR-T":        COLORS["neutral"],
        "COVID-19":     COLORS["highlight"],
    }

    # Build ordered row list with group separators
    rows: list[dict] = []
    for dtype in type_order:
        grp = effect_df[effect_df["design_type"] == dtype].sort_values(
            "d", key=abs, ascending=True,
        )
        if grp.empty:
            continue
        # Add a group label row (rendered as text, no data point)
        rows.append({"_group_label": type_labels[dtype]})
        for _, row in grp.iterrows():
            rows.append(row.to_dict())

    if not rows:
        return

    # Assign y-positions (group labels get a position but no point)
    y = 0
    y_positions: list[float] = []
    y_labels: list[str] = []
    data_rows: list[tuple[int, dict]] = []

    for r in reversed(rows):  # top-to-bottom
        if "_group_label" in r:
            y_positions.append(y)
            y_labels.append(r["_group_label"])
            y += 1
        else:
            y_positions.append(y)
            n = r["n_participants"]
            sig_short = r.get("signature", "")
            y_labels.append(
                f"{r['dataset']}  (n={n}, {sig_short})"
            )
            data_rows.append((len(y_positions) - 1, r))
            y += 1

    # Subtle grid
    ax.grid(True, which="major", axis="x", color="#e8e8e8",
            linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Zero line
    ax.axvline(0, color="#666", linewidth=0.9, linestyle="-",
               zorder=1, alpha=0.5)

    # Plot data points and CIs
    for idx, row in data_rows:
        yp = y_positions[idx]
        color = dataset_colors.get(row["dataset"], COLORS["gray"])

        # CI whisker
        ax.hlines(yp, row["d_lower"], row["d_upper"],
                  color=color, linewidth=2.0, zorder=2, alpha=0.7)
        # Point estimate
        ax.scatter(row["d"], yp, color=color, s=70, zorder=3,
                   edgecolors="white", linewidths=0.8)

        # Annotate signed d value to the right
        x_annot = row["d_upper"] + 0.08
        ax.text(x_annot, yp,
                f"d = {row['d']:+.2f}",
                fontsize=7, va="center", ha="left", color=color,
                fontweight="bold")

    # Style group labels (bold, slightly different color)
    for i, lbl in enumerate(y_labels):
        is_group = any(
            r.get("_group_label") == lbl for r in rows if "_group_label" in r
        )
        if is_group:
            # Draw subtle horizontal separator
            yp = y_positions[i]
            ax.axhline(yp - 0.5, color=COLORS["gray"], linewidth=0.4,
                       linestyle="-", alpha=0.3, xmin=0.0, xmax=1.0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7.5)
    # Bold the group labels
    for tick_label in ax.get_yticklabels():
        text = tick_label.get_text()
        if text in type_labels.values():
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(8)
            tick_label.set_color(COLORS["gray"])

    # Cohen's d reference lines
    for ref_d in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        ax.axvline(ref_d, color=COLORS["gray"], linewidth=0.4,
                   linestyle=":", zorder=0, alpha=0.3)

    ax.set_xlabel("Cohen's d  (signed effect size)")
    ax.set_title("Observed effect sizes",
                 fontsize=11, fontweight="bold", pad=10)

    # Symmetric x-limits around data range
    all_vals = effect_df[["d_lower", "d_upper", "d"]].values.flatten()
    x_margin = max(abs(all_vals.min()), abs(all_vals.max())) + 0.5
    ax.set_xlim(-x_margin, x_margin)
    ax.set_ylim(-0.6, max(y_positions) + 0.6)
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
        ("panel_D", panel_D, (9, 5.5)),   # taller — group labels + 5 rows
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
