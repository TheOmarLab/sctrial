"""
Figure 6 -- Scalability & Power Analysis
=========================================

Four-panel (2×2) figure combining computational scalability benchmarks
with empirical power analysis and observed effect sizes:

    A  Runtime scaling (n_participants × n_features vs wall time, log–log)
    B  Memory scaling (n_participants × n_features vs tracemalloc peak, log–log)
    C  Empirical power curves (sample size vs power, with Wilson CI bands)
    D  Forest plot of signed Cohen's d for all signatures across datasets
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
N_POWER_ITERATIONS = 200         # resamples per sample size (cached after first run)
POWER_ALPHA = 0.05
RNG_SEED = 42

# Code version tag — bump when analysis logic changes to invalidate caches
_CODE_VERSION = "v6"

# Dataset info tuple fields:
#   (name, adata, design, visits, sig_cols, design_type)
# design_type: "two_arm_did" | "paired" | "cross_sectional"
DatasetInfo = tuple[str, object, object, tuple, list[str], str]


# ======================================================================
# Disk cache for expensive computations
# ======================================================================

_CACHE_DIR = MAIN_OUTPUT / FIGURE_NAME.replace("Figure", "Figure") / ".cache"


def _cache_key(*args: str) -> str:
    """Deterministic cache key from string components.

    Includes ``_CODE_VERSION`` so that logic changes automatically
    invalidate stale caches.
    """
    payload = "|".join([_CODE_VERSION] + list(args))
    return hashlib.md5(payload.encode()).hexdigest()[:12]


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

    The x-axis metric is ``n_participants × n_features`` — the actual
    computational input to the statistical model — rather than full
    matrix size (n_cells × n_genes), which is not what the analysis
    operates on.

    Memory is measured via ``tracemalloc`` (Python heap allocations),
    not OS-level RSS.

    Returns
    -------
    timing_df, memory_df : pd.DataFrame
        Columns: n_participants, n_features, n_cells, time_s/peak_mb,
                 dataset, design_type
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
        n_features = len(sigs)
        pid_col = design.participant_col
        n_pids = adata.obs[pid_col].nunique()
        print(f"    {name} ({n_pids} ppts × {n_features} features, "
              f"{n_cells:,} cells, {dtype}) ... ", end="", flush=True)

        # Force GC before measurement to reduce noise
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if dtype == "cross_sectional":
                between_arm_comparison(
                    adata,
                    visit=visits[0],
                    features=sigs,
                    design=design,
                    aggregate="participant_visit",
                    standardize=True,
                )
            else:
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

        shared = {
            "n_cells": n_cells,
            "n_participants": n_pids,
            "n_features": n_features,
            "dataset": name,
            "design_type": dtype,
        }
        timings.append({**shared, "time_s": elapsed})
        mem_usage.append({**shared, "peak_mb": peak / 1024**2})
        print(f"{elapsed:.2f}s, {peak / 1024**2:.1f} MB")

        gc.collect()

    timing_df = pd.DataFrame(timings)
    memory_df = pd.DataFrame(mem_usage)
    _save_cache(cache_tag + "_time", timing_df)
    _save_cache(cache_tag + "_mem", memory_df)
    return timing_df, memory_df


# Pre-specified endpoints for power analysis.
# These are biologically motivated signatures chosen a priori (not data-driven),
# avoiding double-dipping / winner's-curse bias.  We test each pre-specified
# signature and report power for whichever exists in the dataset.
# Column names follow the ``sig_<name_with_underscores>`` convention from
# ``score_signatures()``.
_PRESPECIFIED_ENDPOINTS = [
    "sig_Cytotoxic T Cell Activity",
    "sig_Interferon Response",
    "sig_Immune Exhaustion",
    "sig_T Cell Activation",
]


def _select_prespecified_feature(
    sigs: list[str],
) -> str | None:
    """Return the first pre-specified endpoint present in *sigs*.

    This is independent of the data — no p-values are computed — so there
    is no selection bias or double-dipping.
    """
    for ep in _PRESPECIFIED_ENDPOINTS:
        if ep in sigs:
            return ep
    return None


def _stratified_subsample_pids(
    adata,
    design: TrialDesign,
    dtype: str,
    n_sub: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample participants preserving arm/visit balance where possible.

    For two-arm designs, draws proportionally from each arm.
    For single-arm designs, draws uniformly.
    """
    pid_col = design.participant_col
    all_pids = adata.obs[pid_col].unique()

    if dtype == "two_arm_did" and n_sub >= 4:
        # Stratify by arm
        arm_pids: dict[str, np.ndarray] = {}
        for arm in [design.arm_treated, design.arm_control]:
            arm_pids[arm] = adata.obs.loc[
                adata.obs[design.arm_col] == arm, pid_col
            ].unique()
        n_t = len(arm_pids[design.arm_treated])
        n_c = len(arm_pids[design.arm_control])
        frac_t = n_t / (n_t + n_c)
        n_sub_t = max(2, round(n_sub * frac_t))
        n_sub_c = n_sub - n_sub_t
        if n_sub_c < 2:
            n_sub_c = 2
            n_sub_t = n_sub - n_sub_c
        n_sub_t = min(n_sub_t, n_t)
        n_sub_c = min(n_sub_c, n_c)
        sampled = np.concatenate([
            rng.choice(arm_pids[design.arm_treated], n_sub_t, replace=False),
            rng.choice(arm_pids[design.arm_control], n_sub_c, replace=False),
        ])
        return sampled

    return rng.choice(all_pids, size=min(n_sub, len(all_pids)), replace=False)


def _compute_subsampling_power(
    datasets: list[DatasetInfo],
) -> pd.DataFrame:
    """Compute empirical power via participant subsampling on real datasets.

    Strategy: Use a **pre-specified endpoint** (chosen a priori, not
    data-driven) to avoid double-dipping / winner's-curse bias.
    All datasets are included regardless of full-sample significance.

    Subsampling is stratified by arm for two-arm designs.

    Failures are tracked: power = n_significant / n_valid_fits, and
    ``n_failures`` is recorded per point.

    Returns
    -------
    pd.DataFrame
        Columns: n_participants, dataset, power, n_valid, n_failures,
                 feature, design_type
    """
    # Check cache
    cache_tag = "power_" + _cache_key(
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

    for name, adata, design, visits, sigs, dtype in datasets:
        pid_col = design.participant_col
        all_pids = adata.obs[pid_col].unique()
        n_total = len(all_pids)
        if n_total < 6:
            print(f"    {name}: too few participants ({n_total}), skipping")
            continue

        # Pre-specified endpoint (no data peeking)
        feat = _select_prespecified_feature(sigs)
        if feat is None:
            print(f"    {name}: no pre-specified endpoint available, skipping")
            continue

        # Choose subsample sizes: from 4 up to n_total
        sub_sizes = sorted(set(
            [4, 6, 8] +
            list(range(5, min(n_total, 30) + 1, 5)) +
            [n_total]
        ))
        sub_sizes = [s for s in sub_sizes if s <= n_total]

        # Use fewer iterations for large datasets to avoid OOM
        if adata.n_obs > 100_000:
            n_iter = 30  # ~205K cells: expensive to copy per subsample
        elif adata.n_obs > 50_000:
            n_iter = N_POWER_ITERATIONS // 2
        else:
            n_iter = N_POWER_ITERATIONS

        print(f"    {name} ({n_total} ppts, {dtype}, feat={feat}, "
              f"n_iter={n_iter}): ", end="", flush=True)

        for n_sub in sub_sizes:
            n_sig = 0
            n_valid = 0
            n_fail = 0
            for _ in range(n_iter):
                sampled_pids = _stratified_subsample_pids(
                    adata, design, dtype, n_sub, rng,
                )
                mask = adata.obs[pid_col].isin(sampled_pids)
                # Use view (not copy) to avoid OOM on large datasets
                sub_adata = adata[mask]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        if dtype == "cross_sectional":
                            res = between_arm_comparison(
                                sub_adata,
                                visit=visits[0],
                                features=[feat],
                                design=design,
                                aggregate="participant_visit",
                                standardize=True,
                            )
                        else:
                            res = did_table(
                                sub_adata,
                                features=[feat],
                                design=design,
                                visits=visits,
                                aggregate="participant_visit",
                                standardize=True,
                            )
                        p_col = next(
                            (c for c in ("p_DiD", "p_arm", "p_value")
                             if c in res.columns),
                            None,
                        )
                        if p_col is None or res.empty:
                            n_fail += 1
                            continue
                        n_valid += 1
                        if res[p_col].iloc[0] < POWER_ALPHA:
                            n_sig += 1
                    except Exception:
                        n_fail += 1

            # Free memory between subsample sizes
            gc.collect()

            # Power = significant / valid (not total iterations)
            power = n_sig / n_valid if n_valid > 0 else np.nan
            records.append({
                "n_participants": n_sub,
                "dataset": name,
                "power": power,
                "n_valid": n_valid,
                "n_failures": n_fail,
                "feature": feat,
                "design_type": dtype,
            })

        ds_records = [r for r in records if r["dataset"] == name]
        powers = [r["power"] for r in ds_records if not np.isnan(r["power"])]
        total_fail = sum(r["n_failures"] for r in ds_records)
        if powers:
            print(f"range [{min(powers):.2f}, {max(powers):.2f}], "
                  f"{total_fail} fit failures")
        else:
            print("no valid results")

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




def _compute_effect_sizes_across_datasets(
    datasets: list[DatasetInfo],
) -> pd.DataFrame:
    """Compute signed Cohen's d for **pre-specified** signatures across datasets.

    Uses ``_PRESPECIFIED_ENDPOINTS`` to avoid winner's-curse / post-selection
    bias.  Reports the effect size for each pre-specified signature that
    exists in each dataset — not just the "best" one.

    Returns
    -------
    pd.DataFrame
        Columns: dataset, signature, d, d_lower, d_upper,
                 design_type, n_participants
    """
    # Check cache
    cache_tag = "effects_" + _cache_key(
        *[f"{n}:{a.n_obs}" for n, a, *_ in datasets],
    )
    cached = _load_cache(cache_tag)
    if cached is not None:
        print("  Effect sizes (cached)")
        return cached

    print("  Computing signed effect sizes (pre-specified endpoints) ...")
    records: list[dict] = []

    for name, adata, design, visits, sigs, dtype in datasets:
        pid_col = design.participant_col
        # Only compute for pre-specified endpoints
        target_sigs = [s for s in _PRESPECIFIED_ENDPOINTS if s in sigs
                       and s in adata.obs.columns]

        for sig in target_sigs:
            try:
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

                records.append({
                    "dataset": name,
                    "signature": sig.replace("sig_", ""),
                    "d": d_val,
                    "d_lower": ci_lo,
                    "d_upper": ci_hi,
                    "design_type": dtype,
                    "n_participants": n_ppt,
                })
                print(f"    {name}/{sig.replace('sig_', '')}: "
                      f"d={d_val:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}] "
                      f"(n={n_ppt})")

            except Exception as exc:
                print(f"    {name}/{sig}: FAILED ({exc})")

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


def _scaling_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
) -> None:
    """Shared helper: log-log scatter of n_cells vs metric.

    X-axis is ``n_cells`` — the dominant cost driver — since
    pseudobulk aggregation over all cells dominates both runtime and
    memory, while the downstream OLS/Wilcoxon on the pseudobulked
    (n_participants × n_features) table is trivially fast.

    No pooled trend line is fitted across heterogeneous methods;
    observed points are shown directly with dataset labels.
    """
    from matplotlib.ticker import LogLocator, NullLocator

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

    x_vals = df["n_cells"].values.astype(float)
    y_vals = df[y_col].values.astype(float)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # ── Subtle grid ──
    ax.grid(True, which="major", axis="both", color="#e0e0e0",
            linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ── Scatter points: color = design type ──
    plotted_dtypes: set = set()
    for i, (_, row) in enumerate(df.iterrows()):
        dt = row.get("design_type", "paired")
        c = dtype_colors.get(dt, COLORS["treated"])
        lbl = dtype_labels.get(dt) if dt not in plotted_dtypes else None
        plotted_dtypes.add(dt)
        ax.scatter(
            x_vals[i], y_vals[i],
            s=90, color=c, edgecolors="white", linewidths=0.8,
            zorder=4, label=lbl, alpha=0.92,
        )

    # ── Dataset labels with adjustText ──
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    def _fmt_cells(n: float) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.0f}M"
        return f"{n / 1_000:.0f}K"

    texts = []
    for i, (_, row) in enumerate(df.iterrows()):
        n_c = int(row["n_cells"])
        n_p = int(row["n_participants"])
        lbl = f"{row['dataset']}  ({_fmt_cells(n_c)} cells, {n_p} ppts)"
        texts.append(
            ax.text(
                x_vals[i], y_vals[i], lbl,
                fontsize=7.5, color="#333",
            )
        )
    if adjust_text is not None and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6),
            ensure_inside_axes=True,
            min_arrow_len=5,
            max_move=80,
        )

    # ── Let matplotlib choose uniform log ticks ──
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_locator(NullLocator())

    # ── Add right-side padding so labels near max-x don't clip ──
    x_lo, x_hi = ax.get_xlim()
    ax.set_xlim(x_lo, x_hi * 1.6)

    ax.set_xlabel("Number of cells")
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8, frameon=True, fancybox=True,
              framealpha=0.85, edgecolor="#ddd", loc="lower right")
    despine(ax)


def panel_A(ax: plt.Axes, data: dict) -> None:
    """Panel A: Runtime scaling — log-log scatter."""
    _scaling_scatter(
        ax, data["timing_df"],
        y_col="time_s", y_label="Wall time (seconds)",
        title="Runtime scaling",
    )


def panel_B(ax: plt.Axes, data: dict) -> None:
    """Panel B: Memory scaling — log-log scatter (tracemalloc)."""
    _scaling_scatter(
        ax, data["memory_df"],
        y_col="peak_mb", y_label="Python allocation peak (MB)",
        title="Memory scaling",
    )


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def panel_C(ax: plt.Axes, data: dict) -> None:
    """Panel C: Empirical power via participant subsampling with Wilson CIs."""
    power_df = data["power_df"]
    if power_df.empty:
        ax.text(0.5, 0.5, "Insufficient data\nfor power analysis",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=COLORS["gray"])
        ax.set_title("Empirical power", fontsize=11, fontweight="bold")
        despine(ax)
        return

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
        grp = grp.sort_values("n_participants")

        # Compute Wilson CIs from n_valid and power
        ci_lo_vals, ci_hi_vals = [], []
        for _, row in grp.iterrows():
            n_v = int(row.get("n_valid", N_POWER_ITERATIONS))
            k = round(row["power"] * n_v) if not np.isnan(row["power"]) else 0
            lo, hi = _wilson_ci(k, n_v)
            ci_lo_vals.append(lo)
            ci_hi_vals.append(hi)

        x = grp["n_participants"].values
        y = grp["power"].values

        # CI band
        ax.fill_between(x, ci_lo_vals, ci_hi_vals,
                        color=color, alpha=0.12, zorder=1)
        # Line
        ax.plot(
            x, y,
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

    # Note pre-specified endpoint
    feat_name = power_df["feature"].iloc[0] if "feature" in power_df.columns else ""
    feat_short = feat_name.replace("sig_", "").replace("_", " ")

    ax.set_xlabel("Number of participants")
    ax.set_ylabel(r"Power (1 − $\beta$)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Empirical power — {feat_short}",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(frameon=True, fancybox=True, framealpha=0.85,
              edgecolor="#ddd", fontsize=8, loc="lower right")
    despine(ax)


def panel_D(ax: plt.Axes, data: dict) -> None:
    """Panel D: Forest plot of signed Cohen's d for pre-specified endpoints.

    Shows all pre-specified signatures across all datasets (no post-selection),
    grouped by dataset.  Y-axis labels show dataset name only; signature and
    sample size are annotated to the right of each point.
    """
    effect_df = data["effect_df"]
    if effect_df.empty:
        ax.text(0.5, 0.5, "No effect size data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=COLORS["gray"])
        ax.set_title("Effect sizes (pre-specified endpoints)",
                     fontsize=10, fontweight="bold")
        despine(ax)
        return

    dataset_colors = {
        "Sade-Feldman": COLORS["control"],
        "Vaccine":      COLORS["treated"],
        "AML":          COLORS["success"],
        "CAR-T":        COLORS["neutral"],
        "COVID-19":     COLORS["highlight"],
    }

    # Group by dataset (preserve order from data loading)
    ds_order = list(dict.fromkeys(effect_df["dataset"]))

    # Build ordered row list with dataset separators
    rows: list[dict] = []
    for ds in ds_order:
        grp = effect_df[effect_df["dataset"] == ds].sort_values(
            "d", key=abs, ascending=True,
        )
        if grp.empty:
            continue
        rows.append({"_group_label": ds})
        for _, row in grp.iterrows():
            rows.append(row.to_dict())

    if not rows:
        return

    # Assign y-positions
    y = 0
    y_positions: list[float] = []
    y_labels: list[str] = []
    data_rows: list[tuple[int, dict]] = []

    for r in reversed(rows):
        if "_group_label" in r:
            y_positions.append(y)
            y_labels.append(r["_group_label"])
            y += 1
        else:
            y_positions.append(y)
            sig_short = r.get("signature", "").replace("_", " ")
            y_labels.append(f"  {sig_short}")
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

        ax.hlines(yp, row["d_lower"], row["d_upper"],
                  color=color, linewidth=2.0, zorder=2, alpha=0.7)
        ax.scatter(row["d"], yp, color=color, s=70, zorder=3,
                   edgecolors="white", linewidths=0.8)

        # Right-side annotation: d value and n
        x_annot = row["d_upper"] + 0.06
        ax.text(x_annot, yp,
                f"{row['d']:+.2f}  (n={row['n_participants']})",
                fontsize=6.5, va="center", ha="left", color="#555")

    # Style dataset group labels
    for i, lbl in enumerate(y_labels):
        is_group = lbl in ds_order
        if is_group:
            yp = y_positions[i]
            ax.axhline(yp - 0.5, color=COLORS["gray"], linewidth=0.4,
                       linestyle="-", alpha=0.3, xmin=0.0, xmax=1.0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7.5)
    for tick_label in ax.get_yticklabels():
        text = tick_label.get_text()
        if text in ds_order:
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(8)
            color = dataset_colors.get(text, COLORS["gray"])
            tick_label.set_color(color)

    # Cohen's d reference lines
    for ref_d in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        ax.axvline(ref_d, color=COLORS["gray"], linewidth=0.4,
                   linestyle=":", zorder=0, alpha=0.3)

    ax.set_xlabel("Cohen's d  (signed effect size)")
    ax.set_title("Effect sizes (pre-specified endpoints)",
                 fontsize=11, fontweight="bold", pad=10)

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
