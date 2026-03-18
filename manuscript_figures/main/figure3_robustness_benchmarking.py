"""
Figure 3 — Statistical Robustness & Method Benchmarking
========================================================

Eight-panel figure combining bootstrap validation, leave-one-out
sensitivity, power analysis, cell-vs-participant method comparison,
and cross-dataset effect sizes.

Panels
------
A   Bootstrap distribution histograms for top signatures.
B   Leave-one-out participant sensitivity analysis.
C   Empirical power curves (participant subsampling).
C2  Power heatmap (datasets × participant-count bins).
D   Cell vs participant effect-size scatter (Pearson r).
E   Cell vs participant −log₁₀(p) comparison.
F   Standard-error comparison (cell vs participant level).
G   Cross-dataset signed Cohen's d forest (pre-specified endpoints).

Runtime scaling (Cleveland dot plot) moved to Supplementary Figure 3
panel J.
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
from matplotlib.ticker import LogLocator, NullLocator
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
    get_aml,
    get_cart,
    save_panel,
    score_signatures,
    sig_display,
    within_arm_comparison,
)

warnings.filterwarnings("ignore")

FIGURE_NAME = "Figure3_robustness_benchmarking"
VISITS: tuple[str, str] = ("Pre", "Post")
N_BOOT = 999

DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)

# ── Scalability / power constants ────────────────────────────────────────

N_BENCHMARK_REPLICATES = 5
N_POWER_ITERATIONS = 200
POWER_ALPHA = 0.05
RNG_SEED = 42
_CODE_VERSION = "v12"

DatasetInfo = tuple[str, object, object, tuple, list[str], str]

_DATASET_TAGS: dict[str, str] = {
    "Sade-Feldman": "SF",
    "AML": "AML",
    "CAR-T": "CAR-T",
    "Vaccine": "VAX",
    "COVID-19": "COVID",
}

_METHOD_FAMILY: dict[str, str] = {
    "two_arm_did": "did_table",
    "paired": "did_table",
    "cross_sectional": "between_arm",
}

_PRESPECIFIED_ENDPOINTS = [
    "sig_Cytotoxic T Cell Activity",
    "sig_Interferon Response",
    "sig_Immune Exhaustion",
    "sig_T Cell Activation",
    "sig_Inflammatory Response",
]

_DATASET_PRIMARY_ENDPOINT: dict[str, str] = {
    "Sade-Feldman": "sig_Interferon Response",
    "AML":          "sig_Cytotoxic T Cell Activity",
    "CAR-T":        "sig_Cytotoxic T Cell Activity",
    "Vaccine":      "sig_Cytotoxic T Cell Activity",
    "COVID-19":     "sig_Inflammatory Response",
}

DATASET_COLORS = {
    "Sade-Feldman": COLORS["control"],
    "Vaccine":      COLORS["treated"],
    "AML":          COLORS["success"],
    "CAR-T":        COLORS["neutral"],
    "COVID-19":     COLORS["highlight"],
}


# ======================================================================
# Disk cache helpers
# ======================================================================

_CACHE_DIR = Path(__file__).resolve().parent.parent / "_cache"


def _cache_key(*args: str) -> str:
    payload = "|".join([_CODE_VERSION] + list(args))
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _load_cache(tag: str) -> pd.DataFrame | None:
    path = _CACHE_DIR / f"{tag}.json"
    if path.exists():
        try:
            return pd.read_json(path, orient="records")
        except Exception:
            return None
    return None


def _save_cache(tag: str, df: pd.DataFrame) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(_CACHE_DIR / f"{tag}.json", orient="records", indent=2)


# ======================================================================
# Sade-Feldman data preparation (panels A, B, E-G)
# ======================================================================

def _prepare_sf_data() -> dict:
    """Load Sade-Feldman and run analyses for panels A, B, E-G."""
    adata = get_sade_feldman()
    if "log1p_tpm" not in adata.layers:
        if "tpm" in adata.layers:
            adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
        else:
            raise RuntimeError("No tpm layer for log1p_tpm creation.")
    adata = harmonize_response(adata)
    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

    common_kw = dict(
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer="log1p_tpm",
        standardize=True,
    )

    print("  Running cell-level DiD ...")
    df_cell = did_table(adata, aggregate="cell", **common_kw)

    print("  Running participant-level DiD ...")
    df_part = did_table(adata, aggregate="participant_visit", **common_kw)

    print("  Running bootstrap DiD ...")
    df_boot = did_table(
        adata, aggregate="participant_visit",
        use_bootstrap=True, n_boot=N_BOOT, seed=42,
        **common_kw,
    )

    print("  Running leave-one-out analysis ...")
    loo_records = _run_loo(adata, sig_cols, common_kw)

    return {
        "df_cell": df_cell,
        "df_part": df_part,
        "df_boot": df_boot,
        "loo_records": loo_records,
        "sig_cols": sig_cols,
        "adata": adata,
    }


def _run_loo(adata, sig_cols: list[str], common_kw: dict) -> pd.DataFrame:
    """Drop each participant one at a time and re-run DiD."""
    pid_col = DESIGN.participant_col
    all_pids = adata.obs[pid_col].unique()
    records = []

    for i, drop_pid in enumerate(all_pids):
        mask = adata.obs[pid_col] != drop_pid
        sub = adata[mask]
        try:
            res = did_table(sub, aggregate="participant_visit", **common_kw)
            for _, row in res.iterrows():
                records.append({
                    "dropped_pid": drop_pid,
                    "feature": row["feature"],
                    "beta_DiD": row["beta_DiD"],
                    "se_DiD": row["se_DiD"],
                    "p_DiD": row["p_DiD"],
                })
        except Exception:
            pass
        if (i + 1) % 5 == 0:
            print(f"    LOO {i + 1}/{len(all_pids)}")

    print(f"  LOO complete: {len(all_pids)} participants")
    return pd.DataFrame(records)


# ======================================================================
# Multi-dataset loading (panels C, D, H)
# ======================================================================

SF_DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)
SF_VISITS: tuple[str, str] = ("Pre", "Post")


def _load_all_datasets() -> list[DatasetInfo]:
    """Load and score all 5 real datasets, sorted by n_obs ascending."""
    datasets: list[DatasetInfo] = []

    try:
        sf = get_sade_feldman()
        sf = harmonize_response(sf)
        sf, sf_sigs = score_signatures(sf, layer="log1p_tpm")
        datasets.append(
            ("Sade-Feldman", sf, SF_DESIGN, SF_VISITS, sf_sigs, "two_arm_did")
        )
    except Exception as exc:
        print(f"    Sade-Feldman: FAILED to load ({exc})")

    try:
        vax = get_vaccine()
        vax, vax_sigs = score_signatures(vax, layer="counts")
        vax_design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col=None,
        )
        datasets.append(
            ("Vaccine", vax, vax_design, ("Pre", "Post"), vax_sigs, "paired")
        )
    except Exception as exc:
        print(f"    Vaccine: FAILED to load ({exc})")

    try:
        aml = get_aml()
        aml, aml_sigs = score_signatures(aml, layer="counts")
        pid_col = ("participant_id" if "participant_id" in aml.obs.columns
                   else "patient_id")
        aml_design = TrialDesign(
            participant_col=pid_col,
            visit_col="visit",
            arm_col=None,
        )
        datasets.append(
            ("AML", aml, aml_design, ("Pre", "Post"), aml_sigs, "paired")
        )
    except Exception as exc:
        print(f"    AML: FAILED to load ({exc})")

    try:
        cart = get_cart()
        cart, cart_sigs = score_signatures(cart, layer="counts")
        pid_col = ("participant_id" if "participant_id" in cart.obs.columns
                   else "patient_id")
        cart_design = TrialDesign(
            participant_col=pid_col,
            visit_col="visit",
            arm_col=None,
        )
        datasets.append(
            ("CAR-T", cart, cart_design, ("Pre", "Post"), cart_sigs, "paired")
        )
    except Exception as exc:
        print(f"    CAR-T: FAILED to load ({exc})")

    try:
        covid = get_stephenson()
        covid, covid_sigs = score_signatures(covid, layer="counts")
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
        datasets.append(
            ("COVID-19", covid, covid_design, (top_bin,), covid_sigs,
             "cross_sectional")
        )
    except Exception as exc:
        print(f"    COVID-19: FAILED to load ({exc})")

    datasets.sort(key=lambda t: t[1].n_obs)
    return datasets


# ======================================================================
# Scalability benchmark
# ======================================================================

def _run_scalability_benchmark(
    datasets: list[DatasetInfo],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark sctrial runtime and memory on each real dataset."""
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


# ======================================================================
# Power analysis
# ======================================================================

def _select_prespecified_feature(
    sigs: list[str],
    dataset_name: str = "",
) -> str | None:
    """Return the biologically motivated primary endpoint for *dataset_name*."""
    primary = _DATASET_PRIMARY_ENDPOINT.get(dataset_name)
    if primary and primary in sigs:
        return primary
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
    """Subsample participants preserving arm/visit balance where possible."""
    pid_col = design.participant_col
    all_pids = adata.obs[pid_col].unique()

    if dtype in ("two_arm_did", "cross_sectional") and n_sub >= 4:
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
    """Compute empirical power via participant subsampling on real datasets."""
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

        feat = _select_prespecified_feature(sigs, dataset_name=name)
        if feat is None:
            print(f"    {name}: no pre-specified endpoint available, skipping")
            continue

        # Features are module scores in .obs, not genes in .X.
        # Slim adata to 1 gene to free ~99.99% of .X memory.
        adata = adata[:, adata.var_names[:1]].copy()
        gc.collect()

        sub_sizes = sorted(set(
            [4, 6, 8]
            + list(range(5, min(n_total, 30) + 1, 5))
            + [n_total]
        ))
        sub_sizes = [s for s in sub_sizes if s <= n_total]

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
                        elif dtype == "paired":
                            res = within_arm_comparison(
                                sub_adata,
                                arm="All",
                                features=[feat],
                                design=design,
                                visits=visits,
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
                            (c for c in ("p_time", "p_DiD", "p_arm", "p_value")
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

            gc.collect()
            # Use n_iter (total attempts) as denominator so fit failures
            # count as non-significant — avoids inflating power.
            power = n_sig / n_iter
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


# ======================================================================
# Cross-dataset effect sizes
# ======================================================================

def _paired_cohens_d(
    participant_deltas: np.ndarray,
) -> tuple[float, float, float]:
    """Compute paired Cohen's d_z = mean(delta) / sd(delta) with 95% CI."""
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
    """Compute signed Cohen's d for pre-specified signatures across datasets."""
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


def _prepare_scalability_data() -> dict:
    """Run all multi-dataset preparation steps for panels C, D, H.

    Processes datasets one at a time for power/effect to avoid OOM
    when all 5 datasets (~750K cells total) are in memory simultaneously.
    """
    print("  Loading all datasets for scalability / power / effect panels ...")
    datasets = _load_all_datasets()

    # Scalability benchmarks (lightweight — just timing, not memory-intensive)
    timing_df, memory_df = _run_scalability_benchmark(datasets)

    # Power: process one dataset at a time to avoid OOM
    power_frames = []
    for ds in datasets:
        pdf = _compute_subsampling_power([ds])
        if pdf is not None and not pdf.empty:
            power_frames.append(pdf)
        gc.collect()
    power_df = pd.concat(power_frames, ignore_index=True) if power_frames else pd.DataFrame()

    # Effect sizes: also one at a time
    effect_frames = []
    for ds in datasets:
        edf = _compute_effect_sizes_across_datasets([ds])
        if edf is not None and not edf.empty:
            effect_frames.append(edf)
        gc.collect()
    effect_df = pd.concat(effect_frames, ignore_index=True) if effect_frames else pd.DataFrame()

    # Free all datasets
    del datasets
    gc.collect()

    return {
        "timing_df": timing_df,
        "memory_df": memory_df,
        "power_df": power_df,
        "effect_df": effect_df,
    }


# ======================================================================
# Panel A: Bootstrap vs Analytical SE
# ======================================================================

def _panel_a(ax, data: dict) -> None:
    """Bootstrap SE scatter (analytical vs bootstrap)."""
    df_boot = data["df_boot"]

    feats, analytical, bootstrap = [], [], []
    for _, row in df_boot.iterrows():
        se_an = row["se_DiD"]
        se_bt = row.get("se_DiD_boot", np.nan)
        if np.isfinite(se_an) and np.isfinite(se_bt):
            feats.append(row["feature"])
            analytical.append(se_an)
            bootstrap.append(se_bt)

    analytical = np.array(analytical)
    bootstrap = np.array(bootstrap)

    if len(analytical) == 0:
        ax.text(0.5, 0.5, "Insufficient bootstrap data",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    lo = min(analytical.min(), bootstrap.min()) * 0.85
    hi = max(analytical.max(), bootstrap.max()) * 1.15
    ax.plot([lo, hi], [lo, hi], ls="--", color=COLORS["gray"], lw=1, zorder=1)

    ax.scatter(analytical, bootstrap, s=50, color=COLORS["treated"],
               edgecolor="white", linewidth=0.5, zorder=3)

    for feat, x, y in zip(feats, analytical, bootstrap):
        ax.annotate(
            sig_display(feat), (x, y),
            fontsize=6, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
        )

    r, p = stats.pearsonr(analytical, bootstrap)
    ax.text(
        0.05, 0.95, f"r = {r:.2f}\np = {p:.1e}",
        transform=ax.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
    )

    ax.set_xlabel("Analytical SE (cluster-robust)")
    ax.set_ylabel("Bootstrap SE (wild cluster)")
    ax.set_title("Bootstrap vs Analytical SE", fontsize=10)
    despine(ax)


# ======================================================================
# Panel B: Leave-one-out sensitivity
# ======================================================================

def _panel_b(ax, data: dict) -> None:
    """LOO influence plot: effect of dropping each participant."""
    loo_df = data["loo_records"]
    df_part = data["df_part"]

    if loo_df is None or len(loo_df) == 0:
        ax.text(0.5, 0.5, "LOO results unavailable",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    top_sigs = (
        df_part.assign(_abs=df_part["beta_DiD"].abs())
        .nlargest(4, "_abs")["feature"].tolist()
    )

    palette = [COLORS["treated"], COLORS["control"],
               COLORS["highlight"], COLORS["neutral"]]

    for idx, feat in enumerate(top_sigs):
        full_beta = df_part.loc[df_part["feature"] == feat, "beta_DiD"].values[0]
        loo_betas = loo_df.loc[loo_df["feature"] == feat, "beta_DiD"].values

        if len(loo_betas) == 0:
            continue

        color = palette[idx % len(palette)]

        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(loo_betas))
        ax.scatter(
            loo_betas, np.full_like(loo_betas, idx) + jitter,
            s=20, color=color, alpha=0.6, edgecolor="none", zorder=2,
        )
        ax.scatter(
            full_beta, idx, s=80, color=color, marker="D",
            edgecolor="black", linewidth=0.8, zorder=4,
        )
        ax.hlines(
            idx, loo_betas.min(), loo_betas.max(),
            colors=color, lw=1.5, alpha=0.4, zorder=1,
        )

    ax.set_yticks(range(len(top_sigs)))
    ax.set_yticklabels([sig_display(s) for s in top_sigs], fontsize=8)
    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (standardized)")
    ax.set_title("Leave-One-Out Sensitivity", fontsize=10)

    from matplotlib.lines import Line2D
    # Signature-color legend entries
    handles = []
    for idx, feat in enumerate(top_sigs):
        color = palette[idx % len(palette)]
        handles.append(
            Line2D([0], [0], marker="D", color="w", markerfacecolor=color,
                   markeredgecolor="black", markersize=6,
                   label=sig_display(feat)),
        )
    # Marker-type legend entries
    handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["gray"],
               markersize=5, label="LOO estimate"),
    )
    handles.append(
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["gray"],
               markeredgecolor="black", markersize=6, label="Full sample"),
    )
    ax.legend(handles=handles, fontsize=6.5, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel C: Runtime scaling (log-log scatter)
# ======================================================================

def _scaling_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    shared_xlim: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Shared helper: log-log scatter with replicate median + IQR."""
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
    method_markers = {
        "did_table": "o",
        "between_arm": "D",
    }
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(True, which="major", axis="both", color="#f0f0f0",
            linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)

    has_replicates = "replicate" in df.columns and df["replicate"].nunique() > 1
    if has_replicates:
        agg = df.groupby("dataset", sort=False).agg(
            n_cells=("n_cells", "first"),
            n_participants=("n_participants", "first"),
            n_features=("n_features", "first"),
            design_type=("design_type", "first"),
            method=("method", "first") if "method" in df.columns
            else ("design_type", "first"),
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

    plotted_dtypes: set = set()
    for _, row in agg.iterrows():
        x = float(row["n_cells"])
        y = float(row["y_med"])
        dt = row.get("design_type", "paired")
        method = row.get("method", "did_table")
        c = dtype_colors.get(dt, COLORS["treated"])
        marker = method_markers.get(method, "o")

        dt_lbl = dtype_labels.get(dt) if dt not in plotted_dtypes else None
        plotted_dtypes.add(dt)

        if has_replicates:
            y_lo = float(row["y_q25"])
            y_hi = float(row["y_q75"])
            ax.plot([x, x], [y_lo, y_hi], color=c, linewidth=2.5,
                    alpha=0.35, zorder=2, solid_capstyle="round")

        ax.scatter(
            x, y, s=140, color=c, edgecolors="white", linewidths=1.2,
            zorder=4, label=dt_lbl, alpha=0.95, marker=marker,
        )

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
        ax.annotate(
            lbl, xy=(x, y), xycoords="data",
            xytext=(14, 10), textcoords="offset points",
            fontsize=8.5, color="#333", fontweight="medium",
            arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6,
                            shrinkA=0, shrinkB=4),
        )

    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(which="major", length=5, width=1.0)

    if shared_xlim is not None:
        ax.set_xlim(shared_xlim)
    else:
        x_lo, x_hi = ax.get_xlim()
        ax.set_xlim(x_lo * 0.6, x_hi * 3.0)
    xlim_out = ax.get_xlim()

    ax.set_xlabel("Number of cells", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(axis="both", which="major", labelsize=9.5)

    ax.legend(fontsize=9, frameon=True, fancybox=False,
              framealpha=0.95, edgecolor="#ccc", loc="lower right",
              borderaxespad=0.8, handletextpad=0.6)

    despine(ax)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    return xlim_out


def _panel_c(ax, data: dict) -> None:
    """Panel C: Runtime scaling — Cleveland dot plot (lollipop)."""
    scale_data = data.get("scale_data")
    if scale_data is None:
        ax.text(0.5, 0.5, "Runtime data unavailable",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        despine(ax)
        return

    df = scale_data["timing_df"]
    dtype_colors = {
        "two_arm_did": COLORS["control"],
        "paired": COLORS["treated"],
        "cross_sectional": COLORS["highlight"],
    }

    def _fmt_cells(n: float) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        return f"{n / 1_000:.0f}K"

    # Aggregate replicates
    has_replicates = "replicate" in df.columns and df["replicate"].nunique() > 1
    if has_replicates:
        agg = df.groupby("dataset", sort=False).agg(
            n_cells=("n_cells", "first"),
            n_participants=("n_participants", "first"),
            design_type=("design_type", "first"),
            time_med=("time_s", "median"),
            time_q25=("time_s", lambda x: x.quantile(0.25)),
            time_q75=("time_s", lambda x: x.quantile(0.75)),
        ).reset_index()
    else:
        agg = df.copy()
        agg["time_med"] = agg["time_s"]
        agg["time_q25"] = agg["time_s"]
        agg["time_q75"] = agg["time_s"]

    agg = agg.sort_values("time_med", ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(agg))
    for i, row in agg.iterrows():
        c = dtype_colors.get(row.get("design_type", "paired"), COLORS["treated"])
        t = float(row["time_med"])
        # Lollipop stem
        ax.hlines(i, 0, t, colors=c, lw=1.5, alpha=0.6, zorder=1)
        # Dot
        ax.scatter(t, i, s=100, color=c, edgecolors="white", linewidths=1.0,
                   zorder=3)
        # IQR whisker if replicates
        if has_replicates:
            q25, q75 = float(row["time_q25"]), float(row["time_q75"])
            ax.plot([q25, q75], [i, i], color=c, lw=3, alpha=0.3,
                    solid_capstyle="round", zorder=2)

    # Y-axis labels: dataset name + cell count
    labels = []
    for _, row in agg.iterrows():
        tag = _DATASET_TAGS.get(row["dataset"], row["dataset"][:8])
        labels.append(f"{tag} ({_fmt_cells(int(row['n_cells']))})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Wall time (seconds)", fontsize=10)
    ax.set_title("Runtime Scaling", fontsize=11)
    ax.set_xlim(left=0)

    # Design-type legend
    from matplotlib.lines import Line2D
    dtype_labels_map = {
        "two_arm_did": "Two-arm DiD",
        "paired": "Paired pre/post",
        "cross_sectional": "Cross-sectional",
    }
    seen = set()
    leg_handles = []
    for _, row in agg.iterrows():
        dt = row.get("design_type", "paired")
        if dt not in seen:
            seen.add(dt)
            leg_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=dtype_colors.get(dt, COLORS["treated"]),
                       markersize=8, label=dtype_labels_map.get(dt, dt)),
            )
    if leg_handles:
        ax.legend(handles=leg_handles, fontsize=7, loc="lower right",
                  frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel D: Power curves (small-multiples)
# ======================================================================

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half_width = (
        z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    )
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def _panel_d_power_curves(data: dict) -> plt.Figure | None:
    """Panel D: Isotonic power curves with Wilson CI ribbon per dataset."""
    from sklearn.isotonic import IsotonicRegression

    scale_data = data.get("scale_data")
    if scale_data is None:
        return None
    power_df = scale_data["power_df"]
    if power_df.empty:
        return None

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
        color = DATASET_COLORS.get(ds_name, COLORS["gray"])

        x = grp["n_participants"].values.astype(float)
        y = grp["power"].values.astype(float)

        ci_lo, ci_hi = [], []
        for _, row in grp.iterrows():
            n_v = int(row.get("n_valid", N_POWER_ITERATIONS))
            k = round(row["power"] * n_v) if not np.isnan(row["power"]) else 0
            lo, hi = _wilson_ci(k, n_v)
            ci_lo.append(lo)
            ci_hi.append(hi)
        ci_lo = np.asarray(ci_lo)
        ci_hi = np.asarray(ci_hi)

        iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0,
                                 out_of_bounds="clip")
        y_iso = iso.fit_transform(x, y)

        ci_lo_iso = IsotonicRegression(
            increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit_transform(x, ci_lo)
        ci_hi_iso = IsotonicRegression(
            increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit_transform(x, ci_hi)

        ax.grid(True, which="major", axis="y", color="#f0f0f0",
                linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)

        ax.fill_between(x, ci_lo_iso, ci_hi_iso,
                        color=color, alpha=0.15, zorder=1, linewidth=0)
        ax.plot(x, y_iso, color=color, linewidth=2.5, zorder=3,
                solid_capstyle="round")

        ax.axhline(0.80, color="#bbb", linewidth=0.7,
                   linestyle="--", zorder=1, alpha=0.5)

        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_ylim(-0.02, 1.05)

        if "feature" in grp.columns and not grp.empty:
            feat = grp["feature"].iloc[0].replace("sig_", "").replace("_", " ")
        else:
            feat = ""
        ax.set_title(ds_name, fontsize=10.5, fontweight="bold",
                     color=color, pad=6)
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


# ======================================================================
# Panel D2: Power heatmap — datasets × participant-count bins
# ======================================================================

def _panel_d2_power_heatmap(data: dict) -> plt.Figure | None:
    """Power heatmap — datasets × participant-count bins.

    Rows = datasets, columns = participant count (uniform width), fill = power.
    Cells ≥ 0.80 get a bold border.  N/A cells are light gray with "—".
    Returns a new Figure.
    """
    sd = data.get("scale_data")
    if sd is None:
        return None
    power_df = sd["power_df"]
    if power_df.empty:
        return None

    dataset_colors = {
        "Sade-Feldman": COLORS["control"],
        "Vaccine":      COLORS["treated"],
        "AML":          COLORS["success"],
        "CAR-T":        COLORS["neutral"],
        "COVID-19":     COLORS["highlight"],
    }

    ds_names = list(dict.fromkeys(power_df["dataset"]))
    all_n = sorted(power_df["n_participants"].unique())
    n_cols = len(all_n)
    n_rows = len(ds_names)

    # Build power + reliability matrices
    power_matrix = np.full((n_rows, n_cols), np.nan)
    nvalid_matrix = np.full((n_rows, n_cols), np.nan)
    niter_matrix = np.full((n_rows, n_cols), np.nan)
    for i, ds in enumerate(ds_names):
        grp = power_df[power_df["dataset"] == ds]
        for _, row in grp.iterrows():
            j = all_n.index(int(row["n_participants"]))
            power_matrix[i, j] = row["power"]
            n_v = int(row.get("n_valid", 0))
            n_f = int(row.get("n_failures", 0))
            nvalid_matrix[i, j] = n_v
            niter_matrix[i, j] = n_v + n_f

    # Build feature labels for y-axis
    y_labels = []
    for ds in ds_names:
        grp = power_df[power_df["dataset"] == ds]
        if "feature" in grp.columns and not grp.empty:
            feat = grp["feature"].iloc[0].replace("sig_", "").replace("_", " ")
            y_labels.append(f"{ds}\n({feat})")
        else:
            y_labels.append(ds)

    # ── Figure with uniform-width columns via imshow ──
    fig, ax = plt.subplots(
        figsize=(max(8, 0.7 * n_cols + 3), max(3.5, 0.7 * n_rows + 1)),
    )

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#EDEDED")

    im = ax.imshow(
        np.ma.masked_invalid(power_matrix),
        cmap=cmap, vmin=0, vmax=1, aspect="auto",
        interpolation="nearest",
    )

    # Grid lines
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color="white", linewidth=1.2, zorder=2)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color="white", linewidth=1.2, zorder=2)

    # Text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = power_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="#AAA")
                continue
            text_color = "white" if val > 0.65 else "#333"
            pwr_str = "<.01" if 0 < val < 0.01 else f"{val:.2f}"
            n_v = int(nvalid_matrix[i, j])
            n_i = int(niter_matrix[i, j])
            has_failures = n_v < n_i
            if has_failures:
                ax.text(j, i - 0.12, pwr_str,
                        ha="center", va="center", fontsize=8,
                        color=text_color,
                        fontweight="bold" if val >= 0.80 else "medium")
                ax.text(j, i + 0.24, f"{n_v}/{n_i}",
                        ha="center", va="center", fontsize=5,
                        color="#CC4444" if n_v / n_i < 0.85 else text_color,
                        alpha=0.7)
            else:
                ax.text(j, i, pwr_str,
                        ha="center", va="center", fontsize=8,
                        color=text_color,
                        fontweight="bold" if val >= 0.80 else "medium")

    # Hatch tiles where < 85% of iterations produced valid fits
    for i in range(n_rows):
        for j in range(n_cols):
            n_v = nvalid_matrix[i, j]
            n_i = niter_matrix[i, j]
            if np.isnan(n_v) or np.isnan(n_i):
                continue
            if n_v / n_i < 0.85:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=0, edgecolor="none",
                    facecolor="none", hatch="//", zorder=3, alpha=0.3,
                )
                ax.add_patch(rect)

    # Bold border on cells ≥ 0.80
    for i in range(n_rows):
        for j in range(n_cols):
            val = power_matrix[i, j]
            if not np.isnan(val) and val >= 0.80:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=2.0, edgecolor="#222", facecolor="none",
                    zorder=5,
                )
                ax.add_patch(rect)

    # Axes
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(int(n)) for n in all_n], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    for i, lbl_text in enumerate(ds_names):
        ax.get_yticklabels()[i].set_color(
            dataset_colors.get(lbl_text, COLORS["gray"])
        )
        ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("Number of participants", fontsize=11, labelpad=6)
    ax.set_title("Power heatmap — pre-specified endpoints",
                 fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(axis="both", which="both", length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85, aspect=25)
    cbar.set_label(r"Power (1 − $\beta$)", fontsize=10)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 1.0])
    cbar.set_ticklabels(["0", ".2", ".4", ".6", "1"])
    cbar.ax.axhline(0.80, color="#222", linewidth=2, linestyle="-")
    cbar.ax.annotate(
        "0.80", xy=(1, 0.80), xycoords=("axes fraction", "data"),
        xytext=(6, 0), textcoords="offset points",
        fontsize=7.5, fontweight="bold", color="#222",
        va="center", ha="left",
    )

    fig.tight_layout()
    return fig


# ======================================================================
# Panel E: Cell vs participant effect-size scatter
# ======================================================================

def _panel_e(ax, data: dict) -> None:
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

    span = max(x.max(), y.max()) - min(x.min(), y.min())
    margin = span * 0.15 if span > 0 else 0.5
    lo = min(x.min(), y.min()) - margin
    hi = max(x.max(), y.max()) + margin
    ax.plot([lo, hi], [lo, hi], ls="--", color=COLORS["gray"], lw=1, zorder=1)

    ax.scatter(x, y, s=55, color=COLORS["treated"],
               edgecolor="white", linewidth=0.5, zorder=3)

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


# ======================================================================
# Panel F: P-value comparison
# ======================================================================

def _panel_f(ax, data: dict) -> None:
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
        label="Cell-level", edgecolor="none",
    )
    ax.barh(
        y_pos + bar_h / 2, merged["nlog10_part"].values,
        height=bar_h, color=COLORS["treated"], alpha=0.8,
        label="Participant-level", edgecolor="none",
    )

    ax.axvline(-np.log10(0.05), ls="--", color=COLORS["gray"], lw=0.8,
               label="p = 0.05")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["display"].values, fontsize=8)
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.set_title("Cell vs Participant Inference", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel G: Standard-error comparison
# ======================================================================

def _panel_g(ax, data: dict) -> None:
    """Paired horizontal bars of SE at cell vs participant level."""
    df_cell = data["df_cell"]
    df_boot = data["df_boot"]

    se_cell = df_cell[["feature", "se_DiD"]].copy()
    se_cell = se_cell.rename(columns={"se_DiD": "se_cell"})

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


# ======================================================================
# Panel H: Cross-dataset Cohen's d forest
# ======================================================================

def _panel_h(ax, data: dict) -> None:
    """Forest plot of signed Cohen's d for pre-specified endpoints."""
    scale_data = data.get("scale_data")
    if scale_data is None:
        ax.text(0.5, 0.5, "Effect-size data unavailable",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        despine(ax)
        return

    effect_df = scale_data["effect_df"]
    if effect_df.empty:
        ax.text(0.5, 0.5, "No effect size data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=COLORS["gray"])
        ax.set_title("Effect sizes (pre-specified endpoints)",
                     fontsize=10, fontweight="bold")
        despine(ax)
        return

    ds_order = list(dict.fromkeys(effect_df["dataset"]))

    _SIG_ORDER = [
        "Cytotoxic T Cell Activity",
        "Immune Exhaustion",
        "Inflammatory Response",
        "Interferon Response",
        "T Cell Activation",
    ]

    rows: list[dict] = []
    for ds in ds_order:
        grp = effect_df[effect_df["dataset"] == ds]
        if grp.empty:
            continue
        sig_order_map = {s: i for i, s in enumerate(_SIG_ORDER)}
        grp = grp.copy()
        grp["_sort_key"] = grp["signature"].map(
            lambda s, m=sig_order_map: m.get(s, len(_SIG_ORDER))
        )
        grp = grp.sort_values("_sort_key")
        rows.append({"_group_label": ds})
        for _, row in grp.iterrows():
            rows.append(row.to_dict())

    if not rows:
        return

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

    ax.grid(True, which="major", axis="x", color="#f0f0f0",
            linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)

    ax.axvline(0, color="#444", linewidth=1.0, linestyle="-",
               zorder=1, alpha=0.5)

    for idx, row in data_rows:
        yp = y_positions[idx]
        color = DATASET_COLORS.get(row["dataset"], COLORS["gray"])

        ax.hlines(yp, row["d_lower"], row["d_upper"],
                  color=color, linewidth=2.2, zorder=2, alpha=0.7)
        ax.scatter(row["d"], yp, color=color, s=80, zorder=3,
                   edgecolors="white", linewidths=1.0)

        x_annot = row["d_upper"] + 0.08
        ax.text(x_annot, yp,
                f"{row['d']:+.2f}  (n\u2090={row['n_participants']})",
                fontsize=7, va="center", ha="left", color="#444")

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
            color = DATASET_COLORS.get(text, COLORS["gray"])
            tick_label.set_color(color)

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
# Composite generation
# ======================================================================

def generate() -> None:
    """Create and save all Figure 3 panels.

    Panel mapping:
      A  Bootstrap vs analytical SE
      B  Leave-one-out sensitivity
      C  Power curves (small-multiples)
      C2 Power heatmap (datasets × participant bins)
      D  Cell vs participant β scatter
      E  Cell vs participant p-value bars
      F  Cell vs participant SE bars
      G  Cross-dataset Cohen's d forest

    Runtime scaling moved to Supplementary Figure 3 panel J.
    """
    apply_style()
    print("Figure 3: Robustness & Benchmarking")

    # Sade-Feldman data (panels A, B, D, E, F)
    data = _prepare_sf_data()

    # Multi-dataset scalability / power / effect (panels C, G)
    try:
        data["scale_data"] = _prepare_scalability_data()
    except Exception as exc:
        print(f"  Warning: Could not load scalability data: {exc}")
        data["scale_data"] = None

    # Single-axes panels
    panel_funcs = [
        ("panel_A_bootstrap_validation", _panel_a, (6.5, 5)),
        ("panel_B_loo_sensitivity", _panel_b, (6.5, 5)),
    ]
    for panel_name, func, size in panel_funcs:
        fig, ax = plt.subplots(figsize=size)
        func(ax, data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    # Panel C (power curves — creates its own figure)
    cfig = _panel_d_power_curves(data)
    if cfig is not None:
        save_panel(cfig, "panel_C_power_curves", FIGURE_NAME, MAIN_OUTPUT)

    # Panel C2 (power heatmap — creates its own figure)
    hfig = _panel_d2_power_heatmap(data)
    if hfig is not None:
        save_panel(hfig, "panel_C2_power_heatmap", FIGURE_NAME, MAIN_OUTPUT)

    # Panels D/E/F (cell vs participant comparisons)
    def_panels = [
        ("panel_D_effect_correlation", _panel_e, (6.5, 5)),
        ("panel_E_pvalue_comparison", _panel_f, (6.5, 5)),
        ("panel_F_se_comparison", _panel_g, (6.5, 5)),
    ]
    for panel_name, func, size in def_panels:
        fig, ax = plt.subplots(figsize=size)
        func(ax, data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    # Panel G (cross-dataset effect sizes)
    fig, ax = plt.subplots(figsize=(10, 7))
    _panel_h(ax, data)
    fig.tight_layout()
    save_panel(fig, "panel_G_cross_dataset_effects", FIGURE_NAME, MAIN_OUTPUT)

    # Cleanup
    adata = data.get("adata")
    if adata is not None:
        del adata
    del data
    clear_cache()
    gc.collect()

    print(f"  Figure 3 complete: {FIGURE_NAME}")


# ── CLI entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    generate()
