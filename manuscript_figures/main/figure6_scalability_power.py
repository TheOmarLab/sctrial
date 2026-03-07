"""
Figure 6 -- Scalability & Power Analysis
=========================================

Four-panel (2×2) figure combining computational scalability benchmarks
with empirical power analysis and observed effect sizes:

    A  Runtime scaling (n_cells vs wall time, log–log)
    B  Memory scaling (n_cells vs peak memory allocation, log–log)
    C  Empirical power curves (n_participants vs power, Wilson CI bands)
    D  Forest plot of signed Cohen's d for pre-specified signatures
"""

from __future__ import annotations

import gc
import hashlib
import time
import warnings
from pathlib import Path

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
_CODE_VERSION = "v12"

# Dataset info tuple fields:
#   (name, adata, design, visits, sig_cols, design_type)
# design_type: "two_arm_did" | "paired" | "cross_sectional"
DatasetInfo = tuple[str, object, object, tuple, list[str], str]


# ======================================================================
# Disk cache for expensive computations
# ======================================================================

_CACHE_DIR = Path(__file__).resolve().parent.parent / "_cache"


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


N_BENCHMARK_REPLICATES = 5  # replicate runs per dataset for median + IQR

# Short tags for publication-quality labels
_DATASET_TAGS: dict[str, str] = {
    "Sade-Feldman": "SF",
    "AML": "AML",
    "CAR-T": "CAR-T",
    "Vaccine": "VAX",
    "COVID-19": "COVID",
}

# Method family mapping for marker shape encoding
_METHOD_FAMILY: dict[str, str] = {
    "two_arm_did": "did_table",
    "paired": "did_table",
    "cross_sectional": "between_arm",
}


def _run_scalability_benchmark(
    datasets: list[DatasetInfo],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark sctrial runtime and memory on each real dataset.

    Runs ``N_BENCHMARK_REPLICATES`` replicate measurements per dataset
    so that Panels A/B can show median ± IQR.  Uses ``did_table`` for
    longitudinal datasets and ``between_arm_comparison`` for
    cross-sectional ones.

    Memory is measured via ``psutil`` RSS (resident set size), which
    captures both Python-managed and C-level allocations (numpy, scipy).

    Returns
    -------
    timing_df, memory_df : pd.DataFrame
        Each has one row per replicate.
        Columns: n_participants, n_features, n_cells, time_s/peak_mb,
                 dataset, design_type, method, replicate
    """
    # Check cache
    cache_tag = "benchmark_" + _cache_key(
        *[f"{n}:{a.n_obs}" for n, a, *_ in datasets],
        str(N_BENCHMARK_REPLICATES),
    )
    cached_t = _load_cache(cache_tag + "_time")
    cached_m = _load_cache(cache_tag + "_mem")
    if cached_t is not None and cached_m is not None:
        print("  Scalability benchmarks (cached)")
        return cached_t, cached_m

    print(f"  Running scalability benchmarks ({N_BENCHMARK_REPLICATES} "
          f"replicates per dataset) ...")
    timings: list[dict] = []
    mem_usage: list[dict] = []

    for name, adata, design, visits, sigs, dtype in datasets:
        n_cells = adata.n_obs
        n_features = len(sigs)
        pid_col = design.participant_col
        n_pids = adata.obs[pid_col].nunique()
        method = _METHOD_FAMILY.get(dtype, "did_table")
        print(f"    {name} ({n_pids} ppts × {n_features} features, "
              f"{n_cells:,} cells, {dtype}) ... ", end="", flush=True)

        run_times = []
        run_peaks = []
        for rep in range(N_BENCHMARK_REPLICATES):
            gc.collect()
            import tracemalloc
            tracemalloc.start()
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
            run_times.append(elapsed)
            run_peaks.append(peak / 1024**2)

            shared = {
                "n_cells": n_cells,
                "n_participants": n_pids,
                "n_features": n_features,
                "dataset": name,
                "design_type": dtype,
                "method": method,
                "replicate": rep,
            }
            timings.append({**shared, "time_s": elapsed})
            mem_usage.append({**shared, "peak_mb": peak / 1024**2})

            gc.collect()

        med_t = np.median(run_times)
        med_m = np.median(run_peaks)
        print(f"median {med_t:.2f}s, {med_m:.1f} MB")

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
    "sig_Inflammatory Response",
]

# Per-dataset biologically motivated primary endpoint.
# Each choice is justified by the treatment mechanism and published
# biology — *not* by looking at which signature gives the lowest p.
#
# • Sade-Feldman (anti-PD-1 melanoma): Anti-PD-1 reinvigorates
#   exhausted T cells via the IFN-γ axis (Sade-Feldman et al., Cell 2018).
#   Primary endpoint: Interferon Response.
# • AML (chemo / targeted): Cytotoxic T-cell reconstitution after
#   induction chemotherapy is the hallmark response biomarker
#   (Daver et al., Blood 2019).
# • CAR-T (CD19 CAR-T): Engineered cytotoxic killing is the mechanism;
#   Cytotoxic T Cell Activity directly measures effector function.
# • Vaccine: Vaccination primes de-novo cytotoxic T-cell expansion
#   (Arunachalam et al., Nature 2021).
# • COVID-19 Stephenson (Severe vs Mild, cross-sectional): Severity
#   is associated with dysregulated inflammatory response
#   (Stephenson et al., Nat Med 2021).
_DATASET_PRIMARY_ENDPOINT: dict[str, str] = {
    "Sade-Feldman": "sig_Interferon Response",
    "AML":          "sig_Cytotoxic T Cell Activity",
    "CAR-T":        "sig_Cytotoxic T Cell Activity",
    "Vaccine":      "sig_Cytotoxic T Cell Activity",
    "COVID-19":     "sig_Inflammatory Response",
}


def _select_prespecified_feature(
    sigs: list[str],
    dataset_name: str = "",
) -> str | None:
    """Return the biologically motivated primary endpoint for *dataset_name*.

    Falls back to the first available pre-specified endpoint if the
    dataset has no explicit mapping.  This is independent of the data —
    no p-values are computed — so there is no selection bias.
    """
    # Try dataset-specific endpoint first
    primary = _DATASET_PRIMARY_ENDPOINT.get(dataset_name)
    if primary and primary in sigs:
        return primary
    # Fallback: first pre-specified endpoint present
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

    if dtype in ("two_arm_did", "cross_sectional") and n_sub >= 4:
        # Stratify by arm to preserve balance in both two-arm DiD
        # and cross-sectional (Severe vs Mild, etc.) designs.
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
        feat = _select_prespecified_feature(sigs, dataset_name=name)
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

        # Use fewer iterations for large datasets to manage runtime.
        # Views (not copies) are used, so OOM is not a concern; the
        # bottleneck is wall-clock time for pseudobulk on 200K+ cells.
        if adata.n_obs > 100_000:
            n_iter = 100
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
    sd = float(np.std(participant_deltas, ddof=1))
    if sd < 1e-12:
        return 0.0, 0.0, 0.0
    d = float(np.mean(participant_deltas) / sd)
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
                    # Filter to the same visit used in Panel C / benchmark
                    # so the estimand is consistent across panels.
                    visit_mask = adata.obs[design.visit_col] == visits[0]
                    obs_visit = adata.obs.loc[visit_mask]
                    pb_t = (obs_visit.loc[
                        obs_visit[design.arm_col] == design.arm_treated
                    ].groupby(pid_col)[sig].mean())
                    pb_c = (obs_visit.loc[
                        obs_visit[design.arm_col] == design.arm_control
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
    shared_xlim: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Shared helper: log-log scatter with replicate median + IQR.

    Returns the (x_lo, x_hi) used, for cross-panel alignment.
    """
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullLocator

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
    # Method family → marker shape
    method_markers = {
        "did_table": "o",
        "between_arm": "D",
    }
    ax.set_xscale("log")
    ax.set_yscale("log")

    # ── Subtle grid ──
    ax.grid(True, which="major", axis="both", color="#f0f0f0",
            linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)

    # ── Aggregate replicates → median + IQR per dataset ──
    has_replicates = "replicate" in df.columns and df["replicate"].nunique() > 1
    if has_replicates:
        agg = df.groupby("dataset", sort=False).agg(
            n_cells=("n_cells", "first"),
            n_participants=("n_participants", "first"),
            n_features=("n_features", "first"),
            design_type=("design_type", "first"),
            method=("method", "first") if "method" in df.columns else ("design_type", "first"),
            y_med=(y_col, "median"),
            y_q25=(y_col, lambda x: x.quantile(0.25)),
            y_q75=(y_col, lambda x: x.quantile(0.75)),
        ).reset_index()
    else:
        agg = df.copy()
        agg["y_med"] = agg[y_col]
        agg["y_q25"] = agg[y_col]
        agg["y_q75"] = agg[y_col]
        if "method" not in agg.columns:
            agg["method"] = agg["design_type"].map(
                lambda d: _METHOD_FAMILY.get(d, "did_table"))

    # ── Plot points and IQR error bars ──
    plotted_dtypes: set = set()
    for _, row in agg.iterrows():
        x = float(row["n_cells"])
        y = float(row["y_med"])
        dt = row.get("design_type", "paired")
        method = row.get("method", "did_table")
        c = dtype_colors.get(dt, COLORS["treated"])
        marker = method_markers.get(method, "o")

        # Design type legend entry
        dt_lbl = dtype_labels.get(dt) if dt not in plotted_dtypes else None
        plotted_dtypes.add(dt)

        # IQR error bar (only if replicates)
        if has_replicates:
            y_lo = float(row["y_q25"])
            y_hi = float(row["y_q75"])
            ax.plot([x, x], [y_lo, y_hi], color=c, linewidth=2.5,
                    alpha=0.35, zorder=2, solid_capstyle="round")

        ax.scatter(
            x, y, s=140, color=c, edgecolors="white", linewidths=1.2,
            zorder=4, label=dt_lbl, alpha=0.95, marker=marker,
        )

    # ── Short dataset tags via ax.annotate (log-safe) ──
    def _fmt_cells(n: float) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        return f"{n / 1_000:.0f}K"

    for _, row in agg.iterrows():
        x = float(row["n_cells"])
        y = float(row["y_med"])
        tag = _DATASET_TAGS.get(row["dataset"], row["dataset"][:5])
        n_c = int(row["n_cells"])
        lbl = f"{tag}  ({_fmt_cells(n_c)}, {int(row['n_participants'])}p)"
        # Use offset in *points* (display coords) so arrows always connect
        ax.annotate(
            lbl, xy=(x, y), xycoords="data",
            xytext=(14, 10), textcoords="offset points",
            fontsize=8.5, color="#333", fontweight="medium",
            arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6,
                            shrinkA=0, shrinkB=4),
        )

    # ── Tick formatting ──
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5), numticks=12))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    # For y-axis: use explicit ticks at 1-2-5 sub-decades for uniform visual spacing
    y_lo, y_hi = ax.get_ylim()
    import math
    decade_lo = math.floor(math.log10(max(y_lo, 1e-10)))
    decade_hi = math.ceil(math.log10(max(y_hi, 1e-10)))
    y_ticks = []
    for exp in range(decade_lo - 1, decade_hi + 2):
        for sub in (1, 2, 5):
            val = sub * 10**exp
            if y_lo * 0.8 <= val <= y_hi * 1.2:
                y_ticks.append(val)
    if y_ticks:
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: f"{v:g}"
        ))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.tick_params(which="minor", length=3, width=0.6)
    ax.tick_params(which="major", length=5, width=1.0)

    # ── Axis limits ──
    if shared_xlim is not None:
        ax.set_xlim(shared_xlim)
    else:
        x_lo, x_hi = ax.get_xlim()
        ax.set_xlim(x_lo * 0.6, x_hi * 3.0)
    xlim_out = ax.get_xlim()

    # ── Labels, title ──
    ax.set_xlabel("Number of cells", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(axis="both", which="major", labelsize=9.5)

    # ── Legend: compact, lower-right to avoid label collisions ──
    ax.legend(fontsize=9, frameon=True, fancybox=False,
              framealpha=0.95, edgecolor="#ccc", loc="lower right",
              borderaxespad=0.8, handletextpad=0.6)

    despine(ax)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    return xlim_out


def panel_A(ax: plt.Axes, data: dict) -> None:
    """Panel A: Runtime scaling — log-log scatter with replicate IQR."""
    xlim = _scaling_scatter(
        ax, data["timing_df"],
        y_col="time_s",
        y_label="Wall time (s, median)",
        title="Runtime scaling",
    )
    # Store xlim for panel B alignment
    data["_shared_xlim"] = xlim


def panel_B(ax: plt.Axes, data: dict) -> None:
    """Panel B: Memory scaling — log-log scatter (RSS delta) with IQR."""
    _scaling_scatter(
        ax, data["memory_df"],
        y_col="peak_mb",
        y_label="Peak memory allocation (MB)",
        title="Memory scaling",
        shared_xlim=data.get("_shared_xlim"),
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


def panel_C(fig_or_ax, data: dict) -> plt.Figure | None:
    """Panel C: Small-multiple power panels — one subplot per dataset.

    Each subplot shows the empirical power curve (line + Wilson CI band)
    for that dataset's primary endpoint, sharing the same y-scale [0, 1].
    Returns a new Figure when called from generate(); ignores fig_or_ax.
    """
    power_df = data["power_df"]
    if power_df.empty:
        return None

    dataset_colors = {
        "Sade-Feldman": COLORS["control"],
        "Vaccine":      COLORS["treated"],
        "AML":          COLORS["success"],
        "CAR-T":        COLORS["neutral"],
        "COVID-19":     COLORS["highlight"],
    }

    # Ordered list of datasets (as they appear in data)
    ds_names = list(dict.fromkeys(power_df["dataset"]))
    n_ds = len(ds_names)

    fig, axes = plt.subplots(1, n_ds, figsize=(3.2 * n_ds, 4.0),
                             sharey=True, squeeze=False)
    axes = axes.ravel()

    for i, ds_name in enumerate(ds_names):
        ax = axes[i]
        grp = power_df[power_df["dataset"] == ds_name].sort_values(
            "n_participants"
        )
        color = dataset_colors.get(ds_name, COLORS["gray"])

        # Wilson CIs
        ci_lo, ci_hi = [], []
        for _, row in grp.iterrows():
            n_v = int(row.get("n_valid", N_POWER_ITERATIONS))
            k = round(row["power"] * n_v) if not np.isnan(row["power"]) else 0
            lo, hi = _wilson_ci(k, n_v)
            ci_lo.append(lo)
            ci_hi.append(hi)

        x = grp["n_participants"].values
        y = grp["power"].values

        # Subtle grid
        ax.grid(True, which="major", axis="y", color="#f0f0f0",
                linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)

        # CI band
        ax.fill_between(x, ci_lo, ci_hi,
                        color=color, alpha=0.15, zorder=1, linewidth=0)
        # Line + markers
        ax.plot(x, y, color=color, marker="o", markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.8,
                linewidth=2.2, zorder=3, solid_capstyle="round")

        # 80% power threshold
        ax.axhline(0.80, color="#bbb", linewidth=0.7,
                   linestyle="--", zorder=1, alpha=0.5)

        # Dataset-specific x-range
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_ylim(-0.02, 1.05)

        # Title = dataset name + endpoint
        if "feature" in grp.columns and not grp.empty:
            feat = grp["feature"].iloc[0].replace("sig_", "").replace("_", " ")
        else:
            feat = ""
        ax.set_title(ds_name, fontsize=10.5, fontweight="bold",
                     color=color, pad=6)
        # Endpoint subtitle
        if feat:
            ax.text(0.5, 0.96, feat, transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.5, color="#666",
                    fontstyle="italic")

        ax.set_xlabel("Participants", fontsize=9.5)
        if i == 0:
            ax.set_ylabel(r"Power (1 − $\beta$)", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8.5)

        despine(ax)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.spines["left"].set_linewidth(1.0)

    fig.suptitle("Empirical power — pre-specified endpoints",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


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

    # Fixed biologically meaningful signature order (consistent across datasets)
    _SIG_ORDER = [
        "Cytotoxic T Cell Activity",
        "Immune Exhaustion",
        "Inflammatory Response",
        "Interferon Response",
        "T Cell Activation",
    ]

    # Build ordered row list with dataset separators
    rows: list[dict] = []
    for ds in ds_order:
        grp = effect_df[effect_df["dataset"] == ds]
        if grp.empty:
            continue
        # Sort by fixed biological order
        sig_order_map = {s: i for i, s in enumerate(_SIG_ORDER)}
        grp = grp.copy()
        grp["_sort_key"] = grp["signature"].map(
            lambda s: sig_order_map.get(s, len(_SIG_ORDER))
        )
        grp = grp.sort_values("_sort_key")
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
    ax.grid(True, which="major", axis="x", color="#f0f0f0",
            linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Zero line
    ax.axvline(0, color="#444", linewidth=1.0, linestyle="-",
               zorder=1, alpha=0.5)

    # Plot data points and CIs
    for idx, row in data_rows:
        yp = y_positions[idx]
        color = dataset_colors.get(row["dataset"], COLORS["gray"])

        ax.hlines(yp, row["d_lower"], row["d_upper"],
                  color=color, linewidth=2.2, zorder=2, alpha=0.7)
        ax.scatter(row["d"], yp, color=color, s=80, zorder=3,
                   edgecolors="white", linewidths=1.0)

        # Right-side annotation: d value and n
        x_annot = row["d_upper"] + 0.08
        ax.text(x_annot, yp,
                f"{row['d']:+.2f}  (n={row['n_participants']})",
                fontsize=7, va="center", ha="left", color="#444")

    # Style dataset group labels
    for i, lbl in enumerate(y_labels):
        is_group = lbl in ds_order
        if is_group:
            yp = y_positions[i]
            ax.axhline(yp - 0.5, color=COLORS["gray"], linewidth=0.4,
                       linestyle="-", alpha=0.3, xmin=0.0, xmax=1.0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8.5)
    for tick_label in ax.get_yticklabels():
        text = tick_label.get_text()
        if text in ds_order:
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(9.5)
            color = dataset_colors.get(text, COLORS["gray"])
            tick_label.set_color(color)

    # Cohen's d reference lines
    for ref_d in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        ax.axvline(ref_d, color=COLORS["gray"], linewidth=0.4,
                   linestyle=":", zorder=0, alpha=0.25)

    ax.set_xlabel("Cohen's d  (signed effect size)", fontsize=11)
    ax.set_title("Effect sizes — pre-specified endpoints",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", which="major", labelsize=9.5)

    all_vals = effect_df[["d_lower", "d_upper", "d"]].values.flatten()
    x_margin = max(abs(all_vals.min()), abs(all_vals.max())) + 0.5
    ax.set_xlim(-x_margin, x_margin)
    ax.set_ylim(-0.6, max(y_positions) + 0.6)
    despine(ax)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)


# ======================================================================
# Composite figure
# ======================================================================

def generate() -> None:
    """Create and save Figure 6 individual panels (PNG + PDF)."""
    apply_style()
    data = _prepare_data()

    # Panel A must run before B so shared_xlim propagates
    single_panels = [
        ("panel_A", panel_A, (8, 5.5)),
        ("panel_B", panel_B, (8, 5.5)),
        ("panel_D", panel_D, (10, 7)),
    ]

    for name, draw_fn, figsize in single_panels:
        pfig, pax = plt.subplots(figsize=figsize)
        draw_fn(pax, data)
        pfig.tight_layout()
        save_panel(pfig, name, FIGURE_NAME, MAIN_OUTPUT)
        pdf_path = MAIN_OUTPUT / f"{FIGURE_NAME}_panels" / f"{name}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pfig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(pfig)

    # Panel C: small-multiple figure (creates its own subplots)
    cfig = panel_C(None, data)
    if cfig is not None:
        save_panel(cfig, "panel_C", FIGURE_NAME, MAIN_OUTPUT, close=False)
        pdf_path = MAIN_OUTPUT / f"{FIGURE_NAME}_panels" / "panel_C.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        cfig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(cfig)

    clear_cache()
    gc.collect()
    print("  Done.\n")


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    generate()
