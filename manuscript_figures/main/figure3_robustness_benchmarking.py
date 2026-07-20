"""
Figure 3 — Statistical Robustness & Method Benchmarking
========================================================

Seven-panel figure combining bootstrap validation, leave-one-out
sensitivity, NatMeth signal-fraction benchmark summaries (null-gene FPR
curves, signal-gene effect-size RMSE, genomic inflation λ_GC), and
real-data precision / effect-size panels.

Panels
------
A   Bootstrap vs analytical SE (Sade–Feldman, top) and TNBC (bottom).
B   Leave-one-out participant sensitivity (Sade–Feldman, top) and TNBC (bottom).
C   Effect-size bias and RMSE on signal genes (simulator benchmark). Combined artboard:
    left column, benchmark top row (faceted bias/RMSE grid).
D   Null-gene p-value calibration: QQ plots (top) + % outside 95% CI heatmap (bottom);
    right column spanning benchmark rows 3–4 (from Supp Fig 5 panel I).
E   Genomic inflation λ_GC under pure null (simulator benchmark). Combined artboard:
    left column, benchmark bottom row.
F   Computational cost: median runtime per iteration by method and panel size
    (simulator benchmark). Combined artboard: left column, benchmark bottom row.
G   Standard-error comparison (cell vs participant level): Sade–Feldman (top)
    and TNBC (bottom).
H   Cross-dataset signed Cohen's d forest (pre-specified endpoints), now
    including TNBC as a sixth dataset group.

Null-gene FPR vs signal fraction (former panel C) moved to Supp Fig 5 panel H.
Cell-vs-participant effect-size scatter and p-value comparison moved
to Figure 2 panels B–C.
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
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
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
    get_tnbc_zhang,
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

TNBC_DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="arm",
    arm_treated="anti-PDL1+Chemo",
    arm_control="Chemo",
    celltype_col="cell_type",
)
TNBC_VISITS: tuple[str, str] = ("Pre", "Post")

N_BENCHMARK_REPLICATES = 5
_CODE_VERSION = "v14"

_METHOD_FAMILY: dict[str, str] = {
    "two_arm_did": "did_table",
    "paired": "did_table",
    "cross_sectional": "between_arm",
}

DatasetInfo = tuple[str, object, object, tuple, list[str], str]

_DATASET_TAGS: dict[str, str] = {
    "Sade-Feldman": "SF",
    "AML": "AML",
    "CAR-T": "CAR-T",
    "Vaccine": "VAX",
    "COVID-19": "COVID",
    "TNBC": "TNBC",
}

_PRESPECIFIED_ENDPOINTS = [
    "sig_Cytotoxic T Cell Activity",
    "sig_Interferon Response",
    "sig_Immune Exhaustion",
    "sig_T Cell Activation",
    "sig_Inflammatory Response",
]

DATASET_COLORS = {
    "Sade-Feldman": COLORS["control"],
    "Vaccine":      COLORS["treated"],
    "AML":          COLORS["success"],
    "CAR-T":        COLORS["neutral"],
    "COVID-19":     COLORS["highlight"],
    "TNBC":         "#996633",
}

_DATASET_DISPLAY_NAMES: dict[str, str] = {
    "Sade-Feldman": "Melanoma",
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
# Data preparation
# ======================================================================

def _prepare_sf_data() -> dict:
    adata = get_sade_feldman()
    if "log1p_tpm" not in adata.layers:
        if "tpm" in adata.layers:
            adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
        else:
            raise RuntimeError("No tpm layer for log1p_tpm creation.")
    adata = harmonize_response(adata)
    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

    common_kw = dict(
        features=sig_cols, design=DESIGN, visits=VISITS,
        layer="log1p_tpm", standardize=True,
    )

    print("  Running cell-level DiD ...")
    df_cell = did_table(adata, aggregate="cell", **common_kw)
    print("  Running participant-level DiD ...")
    df_part = did_table(adata, aggregate="participant_visit", **common_kw)
    print("  Running bootstrap DiD ...")
    df_boot = did_table(
        adata, aggregate="participant_visit",
        use_bootstrap=True, n_boot=N_BOOT, seed=42, **common_kw,
    )
    print("  Running leave-one-out analysis ...")
    loo_records = _run_loo(adata, sig_cols, common_kw)

    return {
        "df_cell": df_cell, "df_part": df_part, "df_boot": df_boot,
        "loo_records": loo_records, "sig_cols": sig_cols, "adata": adata,
    }


def _prepare_tnbc_data() -> dict:
    adata = get_tnbc_zhang()
    if "log1p_norm" not in adata.layers:
        raise RuntimeError("No log1p_norm layer found for TNBC dataset.")
    adata, sig_cols = score_signatures(adata, layer="log1p_norm")

    common_kw = dict(
        features=sig_cols, design=TNBC_DESIGN, visits=TNBC_VISITS,
        layer="log1p_norm", standardize=True,
    )

    print("  [TNBC] Running cell-level DiD ...")
    df_cell = did_table(adata, aggregate="cell", **common_kw)
    print("  [TNBC] Running participant-level DiD ...")
    df_part = did_table(adata, aggregate="participant_visit", **common_kw)
    print("  [TNBC] Running bootstrap DiD ...")
    df_boot = did_table(
        adata, aggregate="participant_visit",
        use_bootstrap=True, n_boot=N_BOOT, seed=42, **common_kw,
    )
    print("  [TNBC] Running leave-one-out analysis ...")
    loo_records = _run_loo(adata, sig_cols, common_kw)

    return {
        "df_cell": df_cell, "df_part": df_part, "df_boot": df_boot,
        "loo_records": loo_records, "sig_cols": sig_cols, "adata": adata,
    }


def _run_loo(adata, sig_cols: list[str], common_kw: dict) -> pd.DataFrame:
    pid_col = common_kw["design"].participant_col
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


SF_DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)
SF_VISITS: tuple[str, str] = ("Pre", "Post")


def _run_scalability_benchmark(datasets: list[DatasetInfo]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_tag = "benchmark_" + _cache_key(
        *[f"{n}:{a.n_obs}" for n, a, *_ in datasets],
        str(N_BENCHMARK_REPLICATES),
    )
    cached_t = _load_cache(cache_tag + "_time")
    cached_m = _load_cache(cache_tag + "_mem")
    if cached_t is not None and cached_m is not None:
        print("  Scalability benchmarks (cached)")
        return cached_t, cached_m

    print(f"  Running scalability benchmarks ({N_BENCHMARK_REPLICATES} replicates per dataset) ...")
    timings: list[dict] = []
    mem_usage: list[dict] = []

    for name, adata, design, visits, sigs, dtype in datasets:
        n_cells = adata.n_obs
        n_features = len(sigs)
        pid_col = design.participant_col
        n_pids = adata.obs[pid_col].nunique()
        method = _METHOD_FAMILY.get(dtype, "did_table")
        print(f"    {name} ({n_pids} ppts × {n_features} features, {n_cells:,} cells, {dtype}) ... ", end="", flush=True)

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
                        adata, visit=visits[0], features=sigs, design=design,
                        aggregate="participant_visit", standardize=True,
                    )
                elif dtype == "paired":
                    within_arm_comparison(
                        adata, arm="All", features=sigs, design=design,
                        visits=visits, aggregate="participant_visit", standardize=True,
                    )
                else:
                    did_table(
                        adata, features=sigs, design=design, visits=visits,
                        aggregate="participant_visit", standardize=True,
                    )
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            run_times.append(elapsed)
            run_peaks.append(peak / 1024**2)

            shared = {
                "n_cells": n_cells, "n_participants": n_pids, "n_features": n_features,
                "dataset": name, "design_type": dtype, "method": method, "replicate": rep,
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


def _paired_cohens_d(participant_deltas: np.ndarray) -> tuple[float, float, float]:
    n = len(participant_deltas)
    sd = float(np.std(participant_deltas, ddof=1))
    if sd < 1e-12:
        return 0.0, 0.0, 0.0
    d = float(np.mean(participant_deltas) / sd)
    se_d = np.sqrt(1 / n + d**2 / (2 * n))
    t_crit = stats.t.ppf(0.975, n - 1)
    return d, d - t_crit * se_d, d + t_crit * se_d


def _compute_effect_sizes_across_datasets(datasets: list[DatasetInfo]) -> pd.DataFrame:
    cache_tag = "effects_" + _cache_key(*[f"{n}:{a.n_obs}" for n, a, *_ in datasets])
    cached = _load_cache(cache_tag)
    if cached is not None:
        print("  Effect sizes (cached)")
        return cached

    print("  Computing signed effect sizes (pre-specified endpoints) ...")
    records: list[dict] = []

    for name, adata, design, visits, sigs, dtype in datasets:
        pid_col = design.participant_col
        target_sigs = [s for s in _PRESPECIFIED_ENDPOINTS if s in sigs and s in adata.obs.columns]

        for sig in target_sigs:
            try:
                if dtype == "two_arm_did":
                    pb = (
                        adata.obs.groupby([pid_col, design.visit_col, design.arm_col], observed=True)
                        [sig].mean().reset_index()
                    )
                    deltas: dict[str, list[float]] = {}
                    for arm in [design.arm_treated, design.arm_control]:
                        arm_pb = pb[pb[design.arm_col] == arm]
                        arm_d: list[float] = []
                        for _, pdf in arm_pb.groupby(pid_col):
                            if set(visits).issubset(set(pdf[design.visit_col])):
                                pre = pdf.loc[pdf[design.visit_col] == visits[0], sig].values[0]
                                post = pdf.loc[pdf[design.visit_col] == visits[1], sig].values[0]
                                arm_d.append(post - pre)
                        deltas[arm] = arm_d
                    n1 = len(deltas[design.arm_treated])
                    n2 = len(deltas[design.arm_control])
                    if n1 < 2 or n2 < 2:
                        continue
                    d_val = cohens_d_from_did(np.array(deltas[design.arm_treated]), np.array(deltas[design.arm_control]))
                    ci_lo, ci_hi = effect_size_ci(d_val, n1, n2)
                    n_ppt = n1 + n2
                elif dtype == "paired":
                    pb = (
                        adata.obs.groupby([pid_col, design.visit_col], observed=True)
                        [sig].mean().reset_index()
                    )
                    ds: list[float] = []
                    for _, pdf in pb.groupby(pid_col):
                        if set(visits).issubset(set(pdf[design.visit_col])):
                            pre = pdf.loc[pdf[design.visit_col] == visits[0], sig].values[0]
                            post = pdf.loc[pdf[design.visit_col] == visits[1], sig].values[0]
                            ds.append(post - pre)
                    if len(ds) < 3:
                        continue
                    d_val, ci_lo, ci_hi = _paired_cohens_d(np.array(ds))
                    n_ppt = len(ds)
                elif dtype == "cross_sectional":
                    visit_mask = adata.obs[design.visit_col] == visits[0]
                    obs_visit = adata.obs.loc[visit_mask]
                    pb_t = obs_visit.loc[obs_visit[design.arm_col] == design.arm_treated].groupby(pid_col)[sig].mean()
                    pb_c = obs_visit.loc[obs_visit[design.arm_col] == design.arm_control].groupby(pid_col)[sig].mean()
                    n1, n2 = len(pb_t), len(pb_c)
                    if n1 < 3 or n2 < 3:
                        continue
                    pooled_sd = np.sqrt(((n1 - 1) * pb_t.std()**2 + (n2 - 1) * pb_c.std()**2) / (n1 + n2 - 2))
                    if pooled_sd < 1e-12:
                        continue
                    d_val = float((pb_t.mean() - pb_c.mean()) / pooled_sd)
                    ci_lo, ci_hi = effect_size_ci(d_val, n1, n2)
                    n_ppt = n1 + n2
                else:
                    continue

                records.append({
                    "dataset": name, "signature": sig.replace("sig_", ""),
                    "d": d_val, "d_lower": ci_lo, "d_upper": ci_hi,
                    "design_type": dtype, "n_participants": n_ppt,
                })
                print(f"    {name}/{sig.replace('sig_', '')}: d={d_val:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}] (n={n_ppt})")
            except Exception as exc:
                print(f"    {name}/{sig}: FAILED ({exc})")

    effect_df = pd.DataFrame(records)
    _save_cache(cache_tag, effect_df)
    return effect_df


def _load_dataset_by_index(idx: int) -> DatasetInfo | None:
    loaders = [
        lambda: _load_sf(),
        lambda: _load_tnbc(),
        lambda: _load_vaccine(),
        lambda: _load_aml(),
        lambda: _load_cart(),
        lambda: _load_covid(),
    ]
    if idx >= len(loaders):
        return None
    try:
        return loaders[idx]()
    except Exception as exc:
        print(f"    Dataset {idx}: FAILED to load ({exc})")
        return None


def _load_sf() -> DatasetInfo:
    sf = get_sade_feldman()
    sf = harmonize_response(sf)
    sf, sf_sigs = score_signatures(sf, layer="log1p_tpm")
    return ("Sade-Feldman", sf, SF_DESIGN, SF_VISITS, sf_sigs, "two_arm_did")

def _load_vaccine() -> DatasetInfo:
    vacc = get_vaccine()
    vacc, vacc_sigs = score_signatures(vacc, layer="counts")
    vacc_design = TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None)
    return ("Vaccine", vacc, vacc_design, ("Pre", "Post"), vacc_sigs, "paired")

def _load_aml() -> DatasetInfo:
    aml = get_aml()
    aml, aml_sigs = score_signatures(aml, layer="counts")
    pid_col = "participant_id" if "participant_id" in aml.obs.columns else "patient_id"
    aml_design = TrialDesign(participant_col=pid_col, visit_col="visit", arm_col=None)
    return ("AML", aml, aml_design, ("Pre", "Post"), aml_sigs, "paired")

def _load_cart() -> DatasetInfo:
    cart = get_cart()
    cart, cart_sigs = score_signatures(cart, layer="counts")
    pid_col = "participant_id" if "participant_id" in cart.obs.columns else "patient_id"
    cart_design = TrialDesign(participant_col=pid_col, visit_col="visit", arm_col=None)
    return ("CAR-T", cart, cart_design, ("Pre", "Post"), cart_sigs, "paired")

def _load_covid() -> DatasetInfo:
    covid = get_stephenson()
    covid, covid_sigs = score_signatures(covid, layer="counts")
    top_bin = covid.obs["dfo_bin"].value_counts().idxmax() if "dfo_bin" in covid.obs.columns else "Pre"
    covid_design = TrialDesign(
        participant_col="participant_id", visit_col="dfo_bin",
        arm_col="severity", arm_treated="Severe", arm_control="Mild",
    )
    return ("COVID-19", covid, covid_design, (top_bin,), covid_sigs, "cross_sectional")

def _load_tnbc() -> DatasetInfo:
    tnbc = get_tnbc_zhang()
    tnbc, tnbc_sigs = score_signatures(tnbc, layer="log1p_norm")
    return ("TNBC", tnbc, TNBC_DESIGN, TNBC_VISITS, tnbc_sigs, "two_arm_did")


def _prepare_scalability_data() -> dict:
    print("  Loading and processing datasets one at a time ...")
    timing_frames: list[pd.DataFrame] = []
    memory_frames: list[pd.DataFrame] = []
    effect_frames: list[pd.DataFrame] = []

    for idx in range(6):
        ds = _load_dataset_by_index(idx)
        if ds is None:
            continue
        name = ds[0]
        print(f"  {name}: {ds[1].n_obs:,} cells, {ds[1].n_vars:,} genes")
        try:
            t_df, m_df = _run_scalability_benchmark([ds])
            if t_df is not None and not t_df.empty:
                timing_frames.append(t_df)
            if m_df is not None and not m_df.empty:
                memory_frames.append(m_df)
        except Exception as exc:
            print(f"    Runtime benchmark failed for {name}: {exc}")
        edf = _compute_effect_sizes_across_datasets([ds])
        if edf is not None and not edf.empty:
            effect_frames.append(edf)
        del ds
        gc.collect()

    return {
        "timing_df": pd.concat(timing_frames, ignore_index=True) if timing_frames else pd.DataFrame(),
        "memory_df": pd.concat(memory_frames, ignore_index=True) if memory_frames else pd.DataFrame(),
        "effect_df": pd.concat(effect_frames, ignore_index=True) if effect_frames else pd.DataFrame(),
    }


# ======================================================================
# NatMeth signal-fraction benchmark (panels C–E)
# ======================================================================

_BENCHMARK_CSV = (
    Path(__file__).resolve().parents[4]
    / "manuscript" / "benchmark" / "sensitivity" / "sensitivity_combined.csv"
)

_BENCH_METHODS = ["wilcoxon_paired", "nebula", "dreamlet", "sctrial_did"]
_BENCH_METHOD_LABELS = {
    "sctrial_did": "sctrial (DiD)", "dreamlet": "dreamlet",
    "nebula": "NEBULA", "wilcoxon_paired": "Wilcoxon (Δ scores)",
}
_BENCH_METHOD_COLORS = {
    "sctrial_did": "#1f77b4", "dreamlet": "#d62728",
    "nebula": "#ff7f0e", "wilcoxon_paired": "#2ca02c",
}
_BENCH_METHOD_MARKERS = {
    "sctrial_did": "o", "dreamlet": "D", "nebula": "s", "wilcoxon_paired": "^",
}

_PANEL_SIZES = [50, 200, 500, 2000]
_SIGNAL_FRACTIONS = [1, 5, 10, 20]


def _load_benchmark_data() -> pd.DataFrame:
    if not _BENCHMARK_CSV.exists():
        raise FileNotFoundError(
            f"Benchmark results not found at {_BENCHMARK_CSV}.\n"
            "Run the signal-fraction sensitivity benchmark on HPC first."
        )
    df = pd.read_csv(_BENCHMARK_CSV, low_memory=False)
    df["n_genes"] = df["scenario"].str.extract(r"_g(\d+)")[0].astype(int)
    frac = df["scenario"].str.extract(r"_f(\d+)")
    df["signal_pct"] = pd.to_numeric(frac[0], errors="coerce").fillna(0).astype(int)
    df["is_null_scenario"] = df["scenario"].str.contains("sens_null")
    return df


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


def _add_nominal_band(ax, level: float = 0.05, low: float = 0.03, high: float = 0.07, color: str = "#d62728"):
    ax.axhspan(low, high, color=color, alpha=0.06, zorder=0)
    ax.axhline(level, color=color, linestyle="--", linewidth=1.0, alpha=0.65, zorder=1)


def _style_axis(ax) -> None:
    ax.grid(axis="y", linestyle=":", color="#b0b0b0", alpha=0.45, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", color="#333333", width=0.8, length=4)


def _compute_signal_bias_rmse_table(bench_df: pd.DataFrame) -> pd.DataFrame:
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

    for method in _BENCH_METHODS:
        sub = lam_df[lam_df["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_focal = method == "sctrial_did"
        style = _method_style(method, is_focal=is_focal, composite=composite)
        xs = [n_to_x[int(n)] for n in sub["n_genes"].values]
        ax.plot(xs, sub["lambda_gc"], label=_BENCH_METHOD_LABELS[method],
                zorder=10 if is_focal else 3, **style)

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
    rt = (
        bench_df.groupby(["method", "scenario", "n_genes", "iteration"])[
            "runtime_seconds"
        ]
        .first()
        .reset_index()
    )
    summary = (
        rt.groupby(["method", "n_genes"])["runtime_seconds"]
        .median()
        .reset_index()
    )
    x_positions = np.arange(len(_PANEL_SIZES), dtype=float)
    n_to_x = dict(zip(_PANEL_SIZES, x_positions))

    _lbl_fs = 5.05 if composite else 11
    _ttl_fs = 6.0 if composite else 12
    _ttl_pad = 5 if composite else 10
    _leg_fs = 4.5 if composite else 9

    for method in _BENCH_METHODS:
        sub = summary[summary["method"] == method].sort_values("n_genes")
        if sub.empty:
            continue
        is_focal = method == "sctrial_did"
        style = _method_style(method, is_focal=is_focal, composite=composite)
        xs = [n_to_x[int(n)] for n in sub["n_genes"].values]
        ax.plot(
            xs, sub["runtime_seconds"],
            label=_BENCH_METHOD_LABELS[method],
            zorder=10 if is_focal else 3,
            **style,
        )

    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p:,}" for p in _PANEL_SIZES], fontsize=_lbl_fs)
    ax.set_xlim(-0.35, len(_PANEL_SIZES) - 0.65)
    ax.set_xlabel("Panel size (genes)", fontsize=_lbl_fs)
    ax.set_ylabel("Median runtime per iteration (s)", fontsize=_lbl_fs)
    ax.set_title("Computational cost", fontsize=_ttl_fs, fontweight="bold", pad=_ttl_pad)
    ax.tick_params(axis="y", labelsize=_lbl_fs)
    if composite:
        ax.legend(
            loc="upper left", bbox_to_anchor=(0.02, 0.58),
            frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=_leg_fs,
            markerscale=0.52, handlelength=1.0,
        )
    else:
        ax.legend(
            loc="upper left", frameon=True, framealpha=0.95,
            edgecolor="#cccccc", fontsize=_leg_fs,
            markerscale=1.0, handlelength=1.5,
        )
    _style_axis(ax)


def _panel_bench_signal_rmse(fig, bench_df: pd.DataFrame, *, composite: bool = False) -> None:
    df = _compute_signal_bias_rmse_table(bench_df)
    if hasattr(fig, "set_constrained_layout"):
        fig.set_constrained_layout(False)
    if composite:
        gs = fig.add_gridspec(2, 4, hspace=0.52, wspace=0.18, left=0.07, right=0.99, top=0.82, bottom=0.16)
    else:
        gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.22, left=0.08, right=0.985, top=0.84, bottom=0.11)

    _ttl_fs = 6.35 if composite else 12
    _yl_fs = 6.2 if composite else 11
    _axis_fs = 5.35 if composite else 10
    _xlab_fs = 5.45 if composite else 10
    bar_width = 0.20
    method_order = ["sctrial_did", "wilcoxon_paired", "nebula", "dreamlet"]
    x_positions = np.arange(len(_SIGNAL_FRACTIONS))
    bias_lo = min(df["bias"].min(), 0) - 0.02
    bias_hi = max(df["bias"].max(), 0.02) * 1.12
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
        ax_bias.set_ylim(bias_lo, bias_hi)
        ax_bias.yaxis.set_major_locator(MultipleLocator(0.05))
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
        _anchor.legend(handles=legend_handles, loc="upper center", ncol=4,
                       bbox_to_anchor=(_cx, -0.15), bbox_transform=_anchor.transAxes,
                       frameon=True, framealpha=0.93, edgecolor="#cccccc", fontsize=_leg_fs,
                       handlelength=0.85, handleheight=0.42, handletextpad=0.35,
                       columnspacing=0.55, borderpad=0.35)
    else:
        fig.legend(handles=legend_handles, loc="upper center", ncol=4,
                   bbox_to_anchor=(0.53, 0.94), frameon=True, framealpha=0.95,
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
):
    """2×2 QQ plots for one (n_genes, signal_pct) condition."""
    scenario_name = f"two_arm__sens_g{n_genes}_f{signal_pct}"
    sub_all = bench_df[bench_df["scenario"] == scenario_name]
    if sub_all.empty:
        print(f"    WARNING: scenario {scenario_name} not found for QQ panel")
        return []
    null = sub_all[sub_all["true_beta"] == 0.0]

    if gs_parent is not None:
        gs_inner = gs_parent.subgridspec(2, 2, hspace=0.65, wspace=0.28)
        axes, ref_ax = [], None
        for r in range(2):
            for c in range(2):
                kw = {} if ref_ax is None else {"sharex": ref_ax, "sharey": ref_ax}
                ax = fig.add_subplot(gs_inner[r, c], **kw)
                if ref_ax is None:
                    ref_ax = ax
                axes.append(ax)
    else:
        ax_grid = fig.subplots(2, 2, sharex=True, sharey=True,
                               gridspec_kw={"hspace": 0.52, "wspace": 0.28})
        axes = list(ax_grid.flatten())

    _sct      = 2.5 if composite else 8
    _ttl_fs   = 5.2 if composite else 12
    _axlbl_fs = 5.1 if composite else 10
    _tick_fs  = 5.1 if composite else 10

    for mi, (ax, method) in enumerate(zip(axes, _BENCH_METHODS)):
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
                     pad=1, y=0.88 if composite else 1.0)
        if mi >= 2:
            ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=_axlbl_fs)
        if mi == 0 and not composite:
            ax.set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=_axlbl_fs)
        _style_axis(ax)
        ax.tick_params(axis="both", which="major", labelsize=_tick_fs)
        ax.tick_params(axis="x", labelbottom=(mi >= 2))
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
    for method in _BENCH_METHODS:
        mat = np.full((len(n_genes_vals), len(signal_pct_vals)), np.nan)
        for ri, ng in enumerate(n_genes_vals):
            for ci, sf in enumerate(signal_pct_vals):
                scenario = f"two_arm__sens_g{ng}_f{sf}"
                sub = bench_df[
                    (bench_df["scenario"] == scenario)
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
    vcenter = 0.05
    _abs_dev = max(np.nanmax(np.abs(all_vals - vcenter)) * 1.1, 0.03)
    vmin, vmax = vcenter - _abs_dev, vcenter + _abs_dev
    norm = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    cmap = "RdBu_r"

    _ttl_fs   = 5.2 if composite else 11
    _axlbl_fs = 5.1 if composite else 9
    _cblbl_fs = 5.6 if composite else 9
    _ann_fs   = 5.3 if composite else 8
    _tick_fs  = 5.0 if composite else 8

    _wr = [0.58, 0.58, 0.032]
    if gs_parent is not None:
        # 4-column: left_hm | spacer | right_hm | cbar  — colorbar flush to right heatmap
        _wr_c = [0.72, 0.12, 0.72, 0.030]
        gs_inner = gs_parent.subgridspec(2, 4, hspace=0.65, wspace=0.18,
                                         width_ratios=_wr_c)
        axes = [
            fig.add_subplot(gs_inner[0, 0]),
            fig.add_subplot(gs_inner[0, 2]),
            fig.add_subplot(gs_inner[1, 0]),
            fig.add_subplot(gs_inner[1, 2]),
        ]
        cbar_ax = fig.add_subplot(gs_inner[:, 3])
    else:
        gs = fig.add_gridspec(2, 3, hspace=0.65, wspace=0.35,
                              width_ratios=_wr,
                              left=0.10, right=0.95, top=0.92, bottom=0.10)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ]
        cbar_ax = fig.add_subplot(gs[:, 2])

    col_labels = [f"{sf}%" for sf in signal_pct_vals]
    row_labels = [f"{ng:,}" for ng in n_genes_vals]

    im_last = None
    for mi, (ax, method) in enumerate(zip(axes, _BENCH_METHODS)):
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
                    fc = "white" if abs(val - vcenter) > _abs_dev * 0.5 else "#222222"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=_ann_fs, color=fc)

        ax.set_xticks(range(len(signal_pct_vals)))
        ax.set_yticks(range(len(n_genes_vals)))
        ax.set_xticklabels(col_labels, fontsize=_tick_fs)
        ax.set_yticklabels(row_labels, fontsize=_tick_fs)
        ax.tick_params(length=2, pad=1)

        if mi >= 2:
            ax.set_xlabel("Signal fraction", fontsize=_axlbl_fs)
        if mi == 0 and not composite:
            ax.set_ylabel("Genes", fontsize=_axlbl_fs)

        ax.set_title(
            _BENCH_METHOD_LABELS[method],
            fontsize=_ttl_fs, fontweight="bold",
            color=_BENCH_METHOD_COLORS[method], pad=2,
        )
        _style_axis(ax)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cb = fig.colorbar(im_last, cax=cbar_ax, orientation="vertical")
    cb.set_label("% outside 95% CI", fontsize=_cblbl_fs, labelpad=2)
    cb.ax.tick_params(labelsize=_tick_fs, length=2, pad=1)
    cb.formatter = plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
    cb.update_ticks()
    cb.ax.axhline(vcenter, color="#333333", linewidth=0.7, linestyle="--")

    if not composite:
        fig.suptitle(
            "Null-gene p-value calibration: % of null p-values outside 95% CI",
            fontsize=12, fontweight="bold",
        )

    return list(axes)


# ======================================================================
# Panel A: Bootstrap vs Analytical SE
# ======================================================================

def _panel_a_single(
    ax, data: dict, *, title: str = "Bootstrap vs Analytical SE",
    composite: bool = False, dataset: str = "melanoma",
) -> None:
    """Bootstrap SE scatter (analytical vs bootstrap) with leader lines."""
    from adjustText import adjust_text

    df_boot = data["df_boot"]

    if df_boot is None or len(df_boot) == 0:
        ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight="bold")
        despine(ax)
        return

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
        ax.text(0.5, 0.5, "Insufficient bootstrap data", ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    raw_lo = min(analytical.min(), bootstrap.min())
    raw_hi = max(analytical.max(), bootstrap.max())
    data_range = raw_hi - raw_lo
    lo = raw_lo - 0.12 * data_range
    hi = raw_hi + 0.12 * data_range
    x_lo = lo - 0.08 * (hi - lo)
    ax.plot([x_lo, hi], [x_lo, hi], ls="--", color=COLORS["gray"], lw=1, zorder=1)

    ax.scatter(analytical, bootstrap, s=22, color=COLORS["treated"],
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xlabel("Analytical SE (cluster-robust)")
    ax.set_ylabel("Bootstrap SE (wild cluster)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(x_lo, hi)
    ax.set_ylim(lo, hi)
    despine(ax)

    # ── Labels with leader lines via adjust_text ──────────────────────
    fontsize_pt = 5.2 if composite else 6.5
    texts = []
    for feat, xa, ya in zip(feats, analytical, bootstrap):
        label = sig_display(feat)
        texts.append(ax.text(xa, ya, label, fontsize=fontsize_pt, ha="left", va="center"))

    adjust_text(
        texts, x=analytical, y=bootstrap, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
        expand=(1.4, 1.6),
        force_text=(0.8, 0.8),
        force_points=(0.5, 0.5),
    )

    r, p = stats.pearsonr(analytical, bootstrap)
    ax.text(
        0.05, 0.95, f"r = {r:.2f}\np = {p:.1e}",
        transform=ax.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
    )


def _panel_a(ax_top, ax_bottom, data_sf: dict, data_tnbc: dict, *, composite: bool = False) -> None:
    _panel_a_single(ax_top, data_sf, title="Bootstrap vs Analytical SE (Melanoma)", composite=composite, dataset="melanoma")
    _panel_a_single(ax_bottom, data_tnbc, title="Bootstrap vs Analytical SE (TNBC)", composite=composite, dataset="tnbc")


# ======================================================================
# Panel B: Leave-one-out sensitivity
# ======================================================================

_PANEL_B_PALETTE = [
    COLORS["treated"], COLORS["control"], COLORS["highlight"], COLORS["neutral"],
    COLORS["success"], "#996633", "#17becf", "#bcbd22",
]
_panel_b_color_map: dict[str, str] = {}


def _panel_b_color(feat: str) -> str:
    if feat not in _panel_b_color_map:
        _panel_b_color_map[feat] = _PANEL_B_PALETTE[len(_panel_b_color_map) % len(_PANEL_B_PALETTE)]
    return _panel_b_color_map[feat]


def _panel_b_single(ax, data: dict, *, title: str = "Leave-One-Out Sensitivity") -> None:
    loo_df = data["loo_records"]
    df_part = data["df_part"]

    if loo_df is None or len(loo_df) == 0 or df_part is None or len(df_part) == 0:
        ax.text(0.5, 0.5, "LOO results unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight="bold")
        despine(ax)
        return

    top_sigs = (
        df_part.assign(_abs=df_part["beta_DiD"].abs())
        .nlargest(4, "_abs")["feature"].tolist()
    )
    _all_vals: list[float] = []

    for idx, feat in enumerate(top_sigs):
        full_beta = df_part.loc[df_part["feature"] == feat, "beta_DiD"].values[0]
        loo_betas = loo_df.loc[loo_df["feature"] == feat, "beta_DiD"].values
        if len(loo_betas) == 0:
            continue
        color = _panel_b_color(feat)
        _all_vals.extend(loo_betas.tolist())
        _all_vals.append(float(full_beta))
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(loo_betas))
        ax.scatter(loo_betas, np.full_like(loo_betas, idx) + jitter,
                   s=20, color=color, alpha=0.6, edgecolor="none", zorder=2)
        ax.scatter(full_beta, idx, s=38, color=color, marker="D",
                   edgecolor="black", linewidth=0.8, zorder=4)
        ax.hlines(idx, loo_betas.min(), loo_betas.max(), colors=color, lw=1.5, alpha=0.4, zorder=1)

    if _all_vals:
        _v_lo, _v_hi = min(_all_vals), max(_all_vals)
        _v_range = max(_v_hi - _v_lo, 1e-6)
        ax.set_xlim(_v_lo - 0.15 * _v_range, _v_hi + 0.15 * _v_range)

    ax.set_yticks(range(len(top_sigs)))
    ax.set_yticklabels([sig_display(s) for s in top_sigs], fontsize=8)
    if top_sigs:
        ax.set_ylim(-0.6, len(top_sigs) - 1 + 0.6)
    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (standardized)")
    ax.set_title(title, fontsize=10, fontweight="bold")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = []
    for idx, feat in enumerate(top_sigs):
        color = _panel_b_color(feat)
        handles.append(Patch(facecolor=color, edgecolor="#333333", linewidth=0.5, label=sig_display(feat)))
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["gray"], markersize=4.0, label="LOO estimate"))
    handles.append(Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["gray"], markeredgecolor="black", markersize=4.8, label="Full sample"))
    ax.legend(handles=handles, fontsize=7.6, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


def _panel_b(ax_top, ax_bottom, data_sf: dict, data_tnbc: dict) -> None:
    _panel_b_single(ax_top, data_sf, title="Leave-One-Out Sensitivity (Melanoma)")
    _panel_b_single(ax_bottom, data_tnbc, title="Leave-One-Out Sensitivity (TNBC)")


# ======================================================================
# Runtime scaling (exported for Supp Fig 3)
# ======================================================================

def _panel_c(ax, data: dict) -> None:
    scale_data = data.get("scale_data")
    if scale_data is None:
        ax.text(0.5, 0.5, "Runtime data unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=10)
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

    has_replicates = "replicate" in df.columns and df["replicate"].nunique() > 1
    if has_replicates:
        agg = df.groupby("dataset", sort=False).agg(
            n_cells=("n_cells", "first"), n_participants=("n_participants", "first"),
            design_type=("design_type", "first"), time_med=("time_s", "median"),
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
        ax.hlines(i, 0, t, colors=c, lw=1.5, alpha=0.6, zorder=1)
        ax.scatter(t, i, s=100, color=c, edgecolors="white", linewidths=1.0, zorder=3)
        if has_replicates:
            q25, q75 = float(row["time_q25"]), float(row["time_q75"])
            ax.plot([q25, q75], [i, i], color=c, lw=3, alpha=0.3, solid_capstyle="round", zorder=2)

    labels = []
    for _, row in agg.iterrows():
        tag = _DATASET_TAGS.get(row["dataset"], row["dataset"][:8])
        labels.append(f"{tag} ({_fmt_cells(int(row['n_cells']))})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Wall time (seconds)", fontsize=10)
    ax.set_title("Runtime Scaling", fontsize=11)
    ax.set_xlim(left=0)

    from matplotlib.lines import Line2D
    dtype_labels_map = {"two_arm_did": "Two-arm DiD", "paired": "Paired pre/post", "cross_sectional": "Cross-sectional"}
    seen = set()
    leg_handles = []
    for _, row in agg.iterrows():
        dt = row.get("design_type", "paired")
        if dt not in seen:
            seen.add(dt)
            leg_handles.append(Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor=dtype_colors.get(dt, COLORS["treated"]),
                                      markersize=8, label=dtype_labels_map.get(dt, dt)))
    if leg_handles:
        ax.legend(handles=leg_handles, fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel F: Standard-error comparison
# ======================================================================

def _panel_d_se_comparison_single(ax, data: dict, *, title: str = "Precision: Cell vs Participant Level") -> None:
    df_cell = data.get("df_cell")
    df_boot = data.get("df_boot")

    if (df_cell is None or len(df_cell) == 0 or df_boot is None or len(df_boot) == 0
            or "feature" not in df_cell.columns or "se_DiD" not in df_cell.columns
            or "feature" not in df_boot.columns):
        ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight="bold")
        despine(ax)
        return

    se_cell = df_cell[["feature", "se_DiD"]].copy().rename(columns={"se_DiD": "se_cell"})
    se_part = df_boot[["feature"]].copy()
    se_part["se_part"] = df_boot.get("se_DiD_boot", df_boot["se_DiD"])
    merged = se_cell.merge(se_part, on="feature")

    if merged.empty:
        ax.text(0.5, 0.5, "No overlapping features", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight="bold")
        despine(ax)
        return

    merged["display"] = merged["feature"].apply(sig_display)
    merged = merged.sort_values("se_part", ascending=True)
    y_pos = np.arange(len(merged))
    bar_h = 0.35

    ax.barh(y_pos - bar_h / 2, merged["se_cell"].values, height=bar_h,
            color=COLORS["highlight"], alpha=0.8, label="Cell-level SE", edgecolor="none")
    ax.barh(y_pos + bar_h / 2, merged["se_part"].values, height=bar_h,
            color=COLORS["treated"], alpha=0.8, label="Participant-level SE (bootstrap)", edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["display"].values, fontsize=8)
    ax.set_xlabel("Standard Error")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


def _panel_d_se_comparison(ax_top, ax_bottom, data_sf: dict, data_tnbc: dict) -> None:
    _panel_d_se_comparison_single(ax_top, data_sf, title="Precision: Cell vs Participant Level (Melanoma)")
    _panel_d_se_comparison_single(ax_bottom, data_tnbc, title="Precision: Cell vs Participant Level (TNBC)")


# ======================================================================
# Panel G: Cross-dataset Cohen's d forest
# ======================================================================

def _panel_e_cross_dataset(ax, data: dict, *, composite: bool = False) -> None:
    scale_data = data.get("scale_data")
    if scale_data is None:
        ax.text(0.5, 0.5, "Effect-size data unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        despine(ax)
        return

    effect_df = scale_data["effect_df"]
    if effect_df.empty:
        ax.text(0.5, 0.5, "No effect size data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=COLORS["gray"])
        ax.set_title("Effect sizes (pre-specified endpoints)", fontsize=10, fontweight="bold")
        despine(ax)
        return

    _natural_order = list(dict.fromkeys(effect_df["dataset"]))
    _priority = ["TNBC", "Sade-Feldman"]
    ds_order = [d for d in _priority if d in _natural_order] + [d for d in _natural_order if d not in _priority]

    _SIG_ORDER = [
        "Cytotoxic T Cell Activity", "Immune Exhaustion", "Inflammatory Response",
        "Interferon Response", "T Cell Activation",
    ]

    rows: list[dict] = []
    for ds in ds_order:
        grp = effect_df[effect_df["dataset"] == ds]
        if grp.empty:
            continue
        sig_order_map = {s: i for i, s in enumerate(_SIG_ORDER)}
        grp = grp.copy()
        grp["_sort_key"] = grp["signature"].map(lambda s, m=sig_order_map: m.get(s, len(_SIG_ORDER)))
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

    ax.grid(True, which="major", axis="x", color="#f0f0f0", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#444", linewidth=1.0, linestyle="-", zorder=1, alpha=0.5)

    _pt_size = 24 if composite else 50
    _ci_lw = 1.7 if composite else 2.2
    _ann_fs = 5.0 if composite else 7
    _yt_fs = 6.4 if composite else 8.5
    _yt_grp_fs = 7.2 if composite else 9.5
    _xl_fs = 7.6 if composite else 11
    _ttl_fs = 8.6 if composite else 13
    _xt_fs = 6.4 if composite else 9.5

    for idx, row in data_rows:
        yp = y_positions[idx]
        color = DATASET_COLORS.get(row["dataset"], COLORS["gray"])
        ax.hlines(yp, row["d_lower"], row["d_upper"], color=color, linewidth=_ci_lw, zorder=2, alpha=0.7)
        ax.scatter(row["d"], yp, color=color, s=_pt_size, zorder=3, edgecolors="white", linewidths=1.0)
        x_annot = row["d_upper"] + 0.08
        ax.text(x_annot, yp, f"{row['d']:+.2f}  (n\u2090={row['n_participants']})",
                fontsize=_ann_fs, va="center", ha="left", color="#444")

    for i, lbl in enumerate(y_labels):
        if lbl in ds_order:
            yp = y_positions[i]
            ax.axhline(yp - 0.5, color=COLORS["gray"], linewidth=0.4, linestyle="-", alpha=0.3, xmin=0.0, xmax=1.0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=_yt_fs)
    for tick_label in ax.get_yticklabels():
        text = tick_label.get_text()
        if text in ds_order:
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(_yt_grp_fs)
            tick_label.set_color(DATASET_COLORS.get(text, COLORS["gray"]))

    for ref_d in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        ax.axvline(ref_d, color=COLORS["gray"], linewidth=0.4, linestyle=":", zorder=0, alpha=0.25)

    ax.set_xlabel("Cohen's d  (signed effect size)", fontsize=_xl_fs)
    ax.set_title("Effect sizes — pre-specified endpoints", fontsize=_ttl_fs, fontweight="bold",
                 pad=12 if not composite else 8)
    ax.tick_params(axis="x", which="major", labelsize=_xt_fs)
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
    apply_style()
    print("Figure 3: Robustness & Benchmarking")

    data = _prepare_sf_data()

    try:
        data_tnbc = _prepare_tnbc_data()
    except Exception as exc:
        print(f"  Warning: Could not prepare TNBC data for panels A/B/F: {exc}")
        data_tnbc = {"df_cell": pd.DataFrame(), "df_part": pd.DataFrame(),
                     "df_boot": pd.DataFrame(), "loo_records": pd.DataFrame(),
                     "sig_cols": [], "adata": None}

    try:
        data["scale_data"] = _prepare_scalability_data()
    except Exception as exc:
        print(f"  Warning: Could not load multi-dataset effect sizes: {exc}")
        data["scale_data"] = None

    bench_df: pd.DataFrame | None = None
    try:
        bench_df = _load_benchmark_data()
        print(f"  Benchmark CSV: {len(bench_df):,} rows, {bench_df.scenario.nunique()} scenarios")
    except FileNotFoundError as exc:
        print(f"  Warning: {exc}")

    # ── Individual panels ──────────────────────────────────────────────
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6.5, 9.5))
    _panel_a(ax_top, ax_bottom, data, data_tnbc)
    fig.tight_layout()
    save_panel(fig, "panel_A_bootstrap_validation", FIGURE_NAME, MAIN_OUTPUT)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6.5, 9.5))
    _panel_b(ax_top, ax_bottom, data, data_tnbc)
    fig.tight_layout()
    save_panel(fig, "panel_B_loo_sensitivity", FIGURE_NAME, MAIN_OUTPUT)

    if bench_df is not None:
        fig_c_ind = plt.figure(figsize=(14, 6.8))
        _panel_bench_signal_rmse(fig_c_ind, bench_df)
        fig_c_ind.suptitle("Effect-size estimation accuracy on signal genes",
                            fontsize=13, fontweight="bold", y=0.995)
        save_panel(fig_c_ind, "panel_C_benchmark_signal_rmse", FIGURE_NAME, MAIN_OUTPUT)

        fig_e_ind, ax_e_ind = plt.subplots(figsize=(7.2, 5.0))
        _panel_bench_lambda_gc(ax_e_ind, bench_df)
        fig_e_ind.tight_layout()
        save_panel(fig_e_ind, "panel_E_benchmark_lambda_gc", FIGURE_NAME, MAIN_OUTPUT)

        fig_f_ind, ax_f_ind = plt.subplots(figsize=(7.2, 5.0))
        _panel_bench_runtime(ax_f_ind, bench_df)
        fig_f_ind.tight_layout()
        save_panel(fig_f_ind, "panel_F_benchmark_runtime", FIGURE_NAME, MAIN_OUTPUT)

        fig_d_ind = plt.figure(figsize=(10, 11.0))
        gs_d_ind = fig_d_ind.add_gridspec(2, 1, hspace=0.40,
                                           left=0.07, right=0.97, top=0.90, bottom=0.08)
        _panel_bench_qq_single(fig_d_ind, bench_df, n_genes=200, signal_pct=10,
                                gs_parent=gs_d_ind[0])
        _panel_bench_qq_heatmap(fig_d_ind, bench_df, gs_parent=gs_d_ind[1])
        fig_d_ind.suptitle("Null-gene p-value calibration", fontsize=13, fontweight="bold")
        save_panel(fig_d_ind, "panel_D_benchmark_qq", FIGURE_NAME, MAIN_OUTPUT)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6.5, 9.5))
    _panel_d_se_comparison(ax_top, ax_bottom, data, data_tnbc)
    fig.tight_layout()
    save_panel(fig, "panel_G_se_comparison", FIGURE_NAME, MAIN_OUTPUT)

    fig, ax = plt.subplots(figsize=(10, 9))
    _panel_e_cross_dataset(ax, data)
    fig.tight_layout()
    save_panel(fig, "panel_H_cross_dataset_effects", FIGURE_NAME, MAIN_OUTPUT)

    # ── Combined artboard (180 × 215 mm) ──────────────────────────────
    _SMALL_RC = {
        "font.size": 5, "axes.titlesize": 5.5, "axes.labelsize": 5,
        "xtick.labelsize": 4.5, "ytick.labelsize": 4.5,
        "legend.fontsize": 4, "legend.title_fontsize": 4,
    }
    _MAX_FONT_COMPOSITE = 6

    def _cap_fontsize(fig, maximum):
        for ax in fig.get_axes():
            for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                        + ax.get_xticklabels() + ax.get_yticklabels() + ax.texts):
                if txt.get_fontsize() > maximum:
                    txt.set_fontsize(maximum)
            if ax.get_legend():
                for txt in ax.get_legend().get_texts():
                    if txt.get_fontsize() > maximum:
                        txt.set_fontsize(maximum)
        for txt in fig.texts:
            if txt.get_fontsize() > maximum:
                txt.set_fontsize(maximum)

    def _match_subfig_axes_height_to_ref(ref_ax, subfig, *, height_frac: float = 1.0):
        axes = [ax for ax in subfig.get_axes() if ax.get_visible()]
        if not axes:
            return
        ref_bb = ref_ax.get_position()
        target_h = max(ref_bb.height * height_frac, 1e-6)
        block_y0 = min(ax.get_position().y0 for ax in axes)
        block_y1 = max(ax.get_position().y1 for ax in axes)
        block_h = max(block_y1 - block_y0, 1e-6)
        scale = target_h / block_h
        target_y0 = ref_bb.y0
        for ax in axes:
            bb = ax.get_position()
            new_y0 = target_y0 + (bb.y0 - block_y0) * scale
            ax.set_position([bb.x0, new_y0, bb.width, bb.height * scale])

    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm = 1.0 / 25.4
    # ── Fixed at 180 × 215 mm ──
    fig_c = plt.figure(figsize=(180 * _mm, 215 * _mm))

    # Layout (180 × 215 mm):
    #   Row 0: A — Melanoma | TNBC side by side                           — medium
    #   Row 1: spacer (A–B gap)
    #   Row 2: B — Melanoma | TNBC side by side                           — medium
    #   Row 3: spacer (B–bench gap)
    #   Row 4: C (bias/RMSE, left) | D top = QQ plots (right)             — medium
    #   Row 5: spacer (bench_top–bench_bottom gap)
    #   Row 6: E (λ_GC) + F (runtime) on left | D bottom = heatmap (right)
    #   Row 7: spacer (bench–G/H gap)
    #   Row 8: G (left, stacked) | H (right, forest plot)                  — tall
    outer = fig_c.add_gridspec(
        9, 1,
        height_ratios=[0.70, 0.08, 0.60, 0.08, 0.78, 0.10, 0.82, 0.06, 1.85],
        hspace=0.32,
        left=0.07, right=0.97, top=0.975, bottom=0.04,
    )

    # ── Row 0: A — both subpanels side by side ───────────────────────
    gs_a = outer[0].subgridspec(1, 2, wspace=0.35)
    ax_a_top = fig_c.add_subplot(gs_a[0])
    ax_a_bot = fig_c.add_subplot(gs_a[1])

    _ax_sp_ab = fig_c.add_subplot(outer[1])
    _ax_sp_ab.set_axis_off()

    # ── Row 2: B — both subpanels side by side ───────────────────────
    gs_b = outer[2].subgridspec(1, 2, wspace=0.35)
    ax_b_top = fig_c.add_subplot(gs_b[0])
    ax_b_bot = fig_c.add_subplot(gs_b[1])

    _ax_sp_top = fig_c.add_subplot(outer[3])
    _ax_sp_top.set_axis_off()

    # ── Rows 4, 6: C | D-top / E+F | D-bottom (row 5 = spacer) ──────
    # Rows 4 and 6 share identical width_ratios so the left (C / E+F) and
    # right (D top / D bottom) columns align vertically.
    _BENCH_W = [1.35, 1.10]  # left (C / E+F) : right (new D = QQ+heatmap)

    # Row 4: C (left) | D top = QQ plots (right)
    gs_bench_top = outer[4].subgridspec(1, 2, wspace=0.38, width_ratios=_BENCH_W)
    sub_c = fig_c.add_subfigure(gs_bench_top[0])

    _ax_sp_34 = fig_c.add_subplot(outer[5])
    _ax_sp_34.set_axis_off()

    # Row 6: E (λ_GC) + F (runtime) on left | D bottom = heatmap on right
    gs_bench_bot = outer[6].subgridspec(1, 2, wspace=0.22, width_ratios=_BENCH_W)
    gs_ef = gs_bench_bot[0].subgridspec(1, 2, wspace=0.65, width_ratios=[0.88, 0.82])
    ax_e = fig_c.add_subplot(gs_ef[0])   # E = λ_GC (formerly D)
    ax_f = fig_c.add_subplot(gs_ef[1])   # F = runtime (formerly E)

    if bench_df is not None:
        _panel_bench_signal_rmse(sub_c, bench_df, composite=True)
        _panel_bench_lambda_gc(ax_e, bench_df, composite=True)
        _panel_bench_runtime(ax_f, bench_df, composite=True)
        qq_single_axes = _panel_bench_qq_single(
            fig_c, bench_df, n_genes=200, signal_pct=10, composite=True,
            gs_parent=gs_bench_top[1],
        )
        qq_axes = _panel_bench_qq_heatmap(
            fig_c, bench_df, composite=True, gs_parent=gs_bench_bot[1],
        )
    else:
        ax_e.text(0.5, 0.5, "—", ha="center", va="center", transform=ax_e.transAxes, fontsize=6)
        ax_e.set_axis_off()
        ax_f.text(0.5, 0.5, "—", ha="center", va="center", transform=ax_f.transAxes, fontsize=6)
        ax_f.set_axis_off()
        ax_sc = sub_c.subplots(1, 1)
        ax_sc.set_axis_off()
        qq_single_axes, qq_axes = [], []

    _ax_sp_mid = fig_c.add_subplot(outer[7])
    _ax_sp_mid.set_axis_off()

    # ── Row 8: G (left) | H (right) ──────────────────────────────────
    gs_gh = outer[8].subgridspec(1, 2, wspace=0.38, width_ratios=[1.0, 1.0])
    gs_g = gs_gh[0].subgridspec(2, 1, hspace=0.55)
    ax_g_top = fig_c.add_subplot(gs_g[0])
    ax_g_bot = fig_c.add_subplot(gs_g[1])
    ax_h = fig_c.add_subplot(gs_gh[1])

    # ── Draw all panels ───────────────────────────────────────────────
    _panel_a(ax_a_bot, ax_a_top, data, data_tnbc, composite=True)
    _panel_b(ax_b_bot, ax_b_top, data, data_tnbc)
    _panel_d_se_comparison(ax_g_bot, ax_g_top, data, data_tnbc)
    _panel_e_cross_dataset(ax_h, data, composite=True)

    fig_c.canvas.draw()
    # Scale C's axes block to match D-top (QQ plots) vertical extent so the
    # subfigure's internal margins don't leave C taller than its neighbour.
    if qq_single_axes:
        _qs_y0 = min(ax.get_position().y0 for ax in qq_single_axes)
        _qs_y1 = max(ax.get_position().y1 for ax in qq_single_axes)
        _qs_h = max(_qs_y1 - _qs_y0, 1e-6)
        _c_axes = [ax for ax in sub_c.get_axes() if ax.get_visible()]
        if _c_axes:
            _cb_y0 = min(ax.get_position().y0 for ax in _c_axes)
            _cb_y1 = max(ax.get_position().y1 for ax in _c_axes)
            _cb_h = max(_cb_y1 - _cb_y0, 1e-6)
            _cscale = _qs_h / _cb_h
            for ax in _c_axes:
                bb = ax.get_position()
                new_y0 = _qs_y0 + (bb.y0 - _cb_y0) * _cscale
                ax.set_position([bb.x0, new_y0, bb.width, bb.height * _cscale])

    # ── Centered ylabels for D-top and D-bottom ──────────────────────
    _d_ylbl_fs = 5.1
    if qq_single_axes:
        _qs_y0 = min(ax.get_position().y0 for ax in qq_single_axes)
        _qs_y1 = max(ax.get_position().y1 for ax in qq_single_axes)
        _qs_x0 = min(ax.get_position().x0 for ax in qq_single_axes)
        fig_c.text(
            _qs_x0 - 0.040, 0.5 * (_qs_y0 + _qs_y1),
            r"Observed $-\log_{10}(p)$",
            ha="center", va="center", fontsize=_d_ylbl_fs, rotation=90,
            transform=fig_c.transFigure,
        )
    if qq_axes:
        _hm_y0 = min(ax.get_position().y0 for ax in qq_axes)
        _hm_y1 = max(ax.get_position().y1 for ax in qq_axes)
        _hm_x0 = min(ax.get_position().x0 for ax in qq_axes)
        fig_c.text(
            _hm_x0 - 0.040, 0.5 * (_hm_y0 + _hm_y1),
            "Genes",
            ha="center", va="center", fontsize=_d_ylbl_fs, rotation=90,
            transform=fig_c.transFigure,
        )

    # ── Legend overrides ──────────────────────────────────────────────
    _inside = {
        ax_a_top: "upper right", ax_a_bot: "upper right",
        ax_b_top: "lower right", ax_b_bot: "lower right",
        ax_g_top: "lower right", ax_g_bot: "lower right",
        ax_h: "lower right",
    }
    _leg_fs_inside = {ax_a_top: 5.2, ax_a_bot: 5.2, ax_b_top: 5.2, ax_b_bot: 5.2, ax_g_top: 5.2, ax_g_bot: 5.2}
    for ax_target, loc in _inside.items():
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            _fs = _leg_fs_inside.get(ax_target, 4.6)
            ax_target.legend(handles=handles, labels=labels, fontsize=_fs, loc=loc,
                             frameon=True, framealpha=0.85, handlelength=1,
                             handletextpad=0.3, borderpad=0.3, labelspacing=0.2)

    # Shrink annotation text in composite panels A/B
    for ax_ in (ax_a_top, ax_a_bot, ax_b_top, ax_b_bot):
        for txt in ax_.texts:
            if txt.get_fontsize() > 5:
                txt.set_fontsize(max(txt.get_fontsize() * 0.55, 3.0))

    _cap_fontsize(fig_c, _MAX_FONT_COMPOSITE)

    def _raise_axis_fonts(ax, *, title_fs, label_fs, tick_fs, legend_fs, text_fs):
        ax.title.set_fontsize(max(ax.title.get_fontsize(), title_fs))
        ax.xaxis.label.set_fontsize(max(ax.xaxis.label.get_fontsize(), label_fs))
        ax.yaxis.label.set_fontsize(max(ax.yaxis.label.get_fontsize(), label_fs))
        ax.tick_params(axis="both", labelsize=tick_fs)
        for txt in ax.texts:
            txt.set_fontsize(max(txt.get_fontsize(), text_fs))
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(max(txt.get_fontsize(), legend_fs))

    _raise_axis_fonts(ax_a_top, title_fs=6.0, label_fs=6.0, tick_fs=5.4, legend_fs=5.2, text_fs=5.2)
    _raise_axis_fonts(ax_a_bot, title_fs=6.0, label_fs=6.0, tick_fs=5.4, legend_fs=5.2, text_fs=5.2)
    _raise_axis_fonts(ax_e, title_fs=6.0, label_fs=6.0, tick_fs=5.4, legend_fs=5.2, text_fs=4.6)
    _raise_axis_fonts(ax_f, title_fs=6.0, label_fs=6.0, tick_fs=5.4, legend_fs=5.2, text_fs=4.6)

    # Match E (λ_GC) axis fonts to F (runtime)
    ax_e.xaxis.label.set_fontsize(6.0)
    ax_e.yaxis.label.set_fontsize(6.0)
    ax_e.tick_params(axis="both", labelsize=5.4)
    for _tl in ax_e.get_xticklabels() + ax_e.get_yticklabels():
        _tl.set_fontsize(5.4)
    # Split E (λ_GC) legend: first 2 entries upper-center, last 2 lower-center
    _e_leg_orig = ax_e.get_legend()
    if _e_leg_orig:
        _e_handles = _e_leg_orig.legend_handles
        _e_labels = [_t.get_text() for _t in _e_leg_orig.get_texts()]
        _e_leg_orig.remove()
        _leg_kw_e = dict(
            ncol=2, frameon=True, framealpha=0.92, edgecolor="#cccccc",
            fontsize=5.2, markerscale=0.52, handlelength=0.9,
            handletextpad=0.3, columnspacing=0.45, borderpad=0.28,
        )
        _e_leg_top = ax_e.legend(
            handles=_e_handles[:2], labels=_e_labels[:2],
            loc="upper center", **_leg_kw_e,
        )
        ax_e.add_artist(_e_leg_top)
        ax_e.legend(
            handles=_e_handles[2:], labels=_e_labels[2:],
            loc="lower center", **_leg_kw_e,
        )

    # ── Reduce ytick label sizes for G and H ──────────────────────────
    for ax_ in (ax_g_top, ax_g_bot):
        for tl in ax_.get_yticklabels():
            tl.set_fontsize(5.2)
    for tl in ax_h.get_yticklabels():
        tl.set_fontsize(min(tl.get_fontsize(), 5.2))

    # ── Panel labels ──────────────────────────────────────────────────
    _lbl_fs = 7
    ax_a_top.text(-0.22, 1.22, "A", transform=ax_a_top.transAxes,
                  fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    for ax, lbl in [(ax_b_top, "B"), (ax_g_top, "G")]:
        ax.text(-0.15, 1.12, lbl, transform=ax.transAxes,
                fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_f.text(-0.28, 1.22, "F", transform=ax_f.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_e.text(-0.38, 1.22, "E", transform=ax_e.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_h.text(-0.07, 1.05, "H", transform=ax_h.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_c_list = sub_c.get_axes()
    if ax_c_list:
        ax_c_list[0].text(-0.32, 1.42, "C", transform=ax_c_list[0].transAxes,
                          fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    if qq_single_axes:
        qq_single_axes[0].text(-0.42, 1.62, "D", transform=qq_single_axes[0].transAxes,
                               fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")

    # ── QQ panel section titles ───────────────────────────────────────
    _f_all_axes = list(qq_axes or []) + list(qq_single_axes or [])
    _f_ttl_fs = 6.35
    _f_sub_fs = 4.8
    if _f_all_axes:
        _f_top_y = max(ax.get_position().y1 for ax in _f_all_axes)
        _f_xs = [ax.get_position().x0 for ax in _f_all_axes] + [
            ax.get_position().x1 for ax in _f_all_axes
        ]
        fig_c.text(
            0.5 * (min(_f_xs) + max(_f_xs)), _f_top_y + 0.018,
            "Null-gene p-value calibration",
            ha="center", va="bottom", fontsize=_f_ttl_fs, fontweight="bold",
            transform=fig_c.transFigure, clip_on=False,
        )
    if qq_single_axes:
        _qs_top_y = max(ax.get_position().y1 for ax in qq_single_axes)
        _qs_xs = [ax.get_position().x0 for ax in qq_single_axes] + [
            ax.get_position().x1 for ax in qq_single_axes
        ]
        fig_c.text(
            0.5 * (min(_qs_xs) + max(_qs_xs)), _qs_top_y + 0.002,
            "QQ plots (200 genes, 10% signal)",
            ha="center", va="bottom", fontsize=_f_sub_fs, fontweight="bold",
            transform=fig_c.transFigure, clip_on=False,
        )
    if qq_axes:
        _hm_top_y = max(ax.get_position().y1 for ax in qq_axes)
        _hm_xs = [ax.get_position().x0 for ax in qq_axes] + [
            ax.get_position().x1 for ax in qq_axes
        ]
        fig_c.text(
            0.5 * (min(_hm_xs) + max(_hm_xs)), _hm_top_y + 0.012,
            "% of null p-values outside 95% CI",
            ha="center", va="bottom", fontsize=_f_sub_fs, fontweight="bold",
            transform=fig_c.transFigure, clip_on=False,
        )

    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, MAIN_OUTPUT, close=False)
    pdf_path = MAIN_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")

    # ── Cleanup ───────────────────────────────────────────────────────
    adata = data.get("adata")
    if adata is not None:
        del adata
    adata_tnbc = data_tnbc.get("adata")
    if adata_tnbc is not None:
        del adata_tnbc
    del data
    del data_tnbc
    clear_cache()
    gc.collect()
    print("  Figure 3 complete: 8 individual panels + combined (A–H); D=QQ(top+bottom), E=λ_GC, F=runtime")


if __name__ == "__main__":
    apply_style()
    generate()