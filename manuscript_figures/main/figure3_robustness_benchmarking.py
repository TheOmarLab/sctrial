"""
Figure 3: Statistical robustness and method benchmarking
========================================================

Panels
------
A   False-positive rate of the participant-level tests under relabeling: every
    reassignment of the participants to the two groups is refitted, and the share of
    rejections at p < 0.05 is shown per signature for the analytical cluster-robust
    test and the wild cluster bootstrap (TNBC and melanoma).
B   Leave-one-out participant sensitivity (TNBC and melanoma).
C   Pure-null Type I error against biological sample size (simulation benchmark).
D   Mixed-signal null-gene false positive rate (simulation benchmark).
E   Realized FDR after Benjamini-Hochberg correction (simulation benchmark).
F   Marginal detection probability (simulation benchmark).
G   Runtime by number of tested genes (simulation benchmark).
H   Cross-dataset standardized effects for five pre-specified endpoints.
"""

from __future__ import annotations

import gc
import hashlib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from sctrial import add_log1p_cpm_layer, cohens_d_from_did, effect_size_ci
from sctrial.benchmark.orchestrator import CORE_METHODS  # noqa: F401

# Benchmark panels + helpers live in the shared toolkit so this figure and the
# benchmark supplement both draw from ONE copy (no cross-figure imports).
from .._benchmark import (  # noqa: F401
    _BENCH_METHOD_COLORS,
    _BENCH_METHOD_LABELS,
    _BENCH_METHOD_LABELS_SHORT,
    _BENCH_METHOD_MARKERS,
    _BENCH_METHODS,
    _CALIBRATED,
    _DESIGN_LABEL,
    _FROZEN_CONFIG,
    _LEGEND_ORDER,
    _PANEL_SIZES,
    _RESULTS_ROOT,
    _SIGNAL_FRACTIONS,
    _add_nominal_band,
    _axes_in_cell,
    _bench_figlegend,
    _bench_legend_below,
    _bench_legend_handles,
    _bench_methods,
    _bh_reject,
    _broken_pair,
    _compute_frac_outside_ci,
    _compute_signal_bias_rmse_table,
    _faceted_broken_by_fraction,
    _frozen_manifest_sha,
    _load_benchmark_data,
    _load_core_benchmark_data,
    _method_style,
    _panel_bench_bh_fdr,
    _panel_bench_lambda_gc,
    _panel_bench_mixed_fpr,
    _panel_bench_power_vs_n,
    _panel_bench_qq_heatmap,
    _panel_bench_qq_single,
    _panel_bench_runtime,
    _panel_bench_scenario_families,
    _panel_bench_signal_rmse,
    _panel_bench_typeI_main,
    _panel_bench_typeI_vs_n,
    _per_scenario_fdr,
    _per_scenario_rate,
    _plot_offscale,
    _style_axis,
)
from .._relabeling import calibration_rates, did_relabeling_cached
from .._runtime import check_recorded as check_recorded_runtime
from .._runtime import load_recorded as load_recorded_runtime
from .._shared import (
    COLORS,
    MAIN_OUTPUT,
    STATS_FINGERPRINT,
    TrialDesign,
    apply_style,
    clear_cache,
    despine,
    did_table,
    get_aml,
    get_cart,
    get_sade_feldman,
    get_stephenson,
    get_tnbc_zhang,
    get_vaccine,
    group_axis_labels,
    harmonize_response,
    key_below_axes,
    output_dpi,
    report_layout_problems,
    save_panel,
    score_signatures,
    sig_display,
)

warnings.filterwarnings("ignore")

# Guard: every method in CORE_METHODS must have a style in the shared dicts.
# The current set is "sctrial_did", "dreamlet", "nebula", "wilcoxon_paired",
# "limma_voom". Adding a new method to CORE_METHODS without a corresponding
# entry in _BENCH_METHOD_LABELS/_BENCH_METHOD_COLORS/_BENCH_METHOD_MARKERS
# will raise here so the omission is caught at import time rather than
# silently producing blank panels.
_missing_styles = [m for m in CORE_METHODS if m not in _BENCH_METHOD_LABELS]
if _missing_styles:
    raise ValueError(
        f"CORE_METHODS has entries with no plotting style: {_missing_styles}. "
        "Add them to _BENCH_METHOD_LABELS/_BENCH_METHOD_COLORS/_BENCH_METHOD_MARKERS "
        "in manuscript_figures/_benchmark.py."
    )

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

_CODE_VERSION = "v16"

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
    "sig_Type I Interferon",
    "sig_T Cell Exhaustion",
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


def _save_analysis_cache(tag: str, data: dict) -> None:
    """Persist the DataFrame outputs of _prepare_*_data to disk."""
    import json
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame_keys = ("df_cell", "df_part", "df_boot", "loo_records")
    for key in frame_keys:
        df = data.get(key)
        if df is not None and not df.empty:
            df.to_json(_CACHE_DIR / f"{tag}_{key}.json", orient="records", indent=2,
                       double_precision=15)
    # JSON has no infinity, and a bootstrap interval can be unbounded.
    data["relabel"].to_pickle(_CACHE_DIR / f"{tag}_relabel.pkl")
    sig_path = _CACHE_DIR / f"{tag}_sig_cols.json"
    sig_path.write_text(json.dumps(data.get("sig_cols", [])))


def _load_analysis_cache(tag: str) -> dict | None:
    """Reload cached DataFrames; returns None if any required frame is missing."""
    import json
    frame_keys = ("df_cell", "df_part", "df_boot", "loo_records")
    result: dict = {}
    relabel_path = _CACHE_DIR / f"{tag}_relabel.pkl"
    if not relabel_path.exists():
        return None
    result["relabel"] = pd.read_pickle(relabel_path)  # noqa: S301 (written by this module)
    for key in frame_keys:
        path = _CACHE_DIR / f"{tag}_{key}.json"
        if not path.exists():
            return None
        try:
            result[key] = pd.read_json(path, orient="records")
        except Exception:
            return None
    sig_path = _CACHE_DIR / f"{tag}_sig_cols.json"
    if not sig_path.exists():
        return None
    result["sig_cols"] = json.loads(sig_path.read_text())
    result["adata"] = None  # not cached; only needed for cleanup
    return result


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_sf_data() -> dict:
    _tag = "sf_analysis_" + _cache_key("sf", STATS_FINGERPRINT)
    cached = _load_analysis_cache(_tag)
    if cached is not None:
        print("  SF analysis (cached)")
        return cached

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
    relabel = did_relabeling_cached(adata, sig_cols, DESIGN, VISITS, layer="log1p_tpm",
                                    tag="melanoma", n_boot=N_BOOT)

    result = {
        "df_cell": df_cell, "df_part": df_part, "df_boot": df_boot,
        "loo_records": loo_records, "relabel": relabel, "sig_cols": sig_cols, "adata": adata,
    }
    _save_analysis_cache(_tag, result)
    return result


def _prepare_tnbc_data() -> dict:
    _tag = "tnbc_analysis_" + _cache_key("tnbc", STATS_FINGERPRINT)
    cached = _load_analysis_cache(_tag)
    if cached is not None:
        print("  TNBC analysis (cached)")
        return cached

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
    relabel = did_relabeling_cached(adata, sig_cols, TNBC_DESIGN, TNBC_VISITS, layer="log1p_norm",
                                    tag="tnbc", n_boot=N_BOOT)

    result = {
        "df_cell": df_cell, "df_part": df_part, "df_boot": df_boot,
        "loo_records": loo_records, "relabel": relabel, "sig_cols": sig_cols, "adata": adata,
    }
    _save_analysis_cache(_tag, result)
    return result


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
    cache_tag = "effects_" + _cache_key("pid-observed", *[f"{n}:{a.n_obs}" for n, a, *_ in datasets])
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
                        for _, pdf in arm_pb.groupby(pid_col, observed=True):
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
                    for _, pdf in pb.groupby(pid_col, observed=True):
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
                    # observed=True: participant_id is categorical, and without it the
                    # groupby returns every participant in the cohort (NaN for absent
                    # ones), so len() counted the whole cohort in each arm.
                    pb_t = obs_visit.loc[obs_visit[design.arm_col] == design.arm_treated].groupby(pid_col, observed=True)[sig].mean()
                    pb_c = obs_visit.loc[obs_visit[design.arm_col] == design.arm_control].groupby(pid_col, observed=True)[sig].mean()
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
    vacc, vacc_sigs = score_signatures(vacc, layer="log1p_norm")
    vacc_design = TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None)
    return ("Vaccine", vacc, vacc_design, ("Pre", "Post"), vacc_sigs, "paired")

def _load_aml() -> DatasetInfo:
    aml = get_aml()
    aml, aml_sigs = score_signatures(aml, layer="log1p_norm")
    pid_col = "participant_id" if "participant_id" in aml.obs.columns else "patient_id"
    aml_design = TrialDesign(participant_col=pid_col, visit_col="visit", arm_col=None)
    return ("AML", aml, aml_design, ("Pre", "Post"), aml_sigs, "paired")

def _load_cart() -> DatasetInfo:
    cart = get_cart()
    cart, cart_sigs = score_signatures(cart, layer="log1p_norm")
    pid_col = "participant_id" if "participant_id" in cart.obs.columns else "patient_id"
    cart_design = TrialDesign(participant_col=pid_col, visit_col="visit", arm_col=None)
    return ("CAR-T", cart, cart_design, ("Pre", "Post"), cart_sigs, "paired")

def _load_covid() -> DatasetInfo:
    covid = get_stephenson()
    if "log1p_cpm" not in covid.layers:
        add_log1p_cpm_layer(covid, counts_layer="counts", out_layer="log1p_cpm")
    covid, covid_sigs = score_signatures(covid, layer="log1p_cpm")
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
    """Cross-dataset effect sizes, and the size of each dataset as it was loaded."""
    print("  Loading and processing datasets one at a time ...")
    effect_frames: list[pd.DataFrame] = []
    sizes: dict[str, dict] = {}

    for idx in range(6):
        ds = _load_dataset_by_index(idx)
        if ds is None:
            continue
        name, adata, design, _visits, sigs, _dtype = ds
        print(f"  {name}: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
        sizes[name] = {"n_cells": int(adata.n_obs), "n_features": len(sigs),
                       "n_participants": int(adata.obs[design.participant_col].nunique())}
        edf = _compute_effect_sizes_across_datasets([ds])
        if edf is not None and not edf.empty:
            effect_frames.append(edf)
        del ds, adata
        gc.collect()

    return {
        "sizes": sizes,
        "effect_df": pd.concat(effect_frames, ignore_index=True) if effect_frames else pd.DataFrame(),
    }


def _check_runtime_recording(sizes: dict[str, dict]) -> None:
    """Fail when the recorded runtime table (Supp Fig 4I) was made on other dataset versions."""
    recorded = load_recorded_runtime()
    for name, size in sizes.items():
        check_recorded_runtime(recorded, name, **size)



# ======================================================================
# Panel A: false-positive rate of the participant-level tests under relabeling
# ======================================================================

# (column of calibration_rates, tick label, key label, color), drawn bottom to top
PANEL_A_TESTS = (
    ("reject_analytic", "Analytical", "Analytical cluster-robust test", COLORS["control"]),
    ("reject_boot", "Bootstrap", "Wild cluster bootstrap", COLORS["treated"]),
)
PANEL_A_NOMINAL = 5.0  # percent


def _panel_a_rates(data: dict) -> pd.DataFrame | None:
    """Per-signature rejection rates (percent) under relabeling, or None without data."""
    relabel = data.get("relabel")
    if relabel is None or len(relabel) == 0:
        return None
    rates = calibration_rates(relabel, alpha=PANEL_A_NOMINAL / 100)
    for col, *_ in PANEL_A_TESTS:
        rates[col] = 100.0 * rates[col]
    return rates


def _panel_a_x_max(rates_list) -> float:
    """Common x-axis maximum (percent) that holds every signature of every cohort."""
    top = max(float(r[[t[0] for t in PANEL_A_TESTS]].to_numpy().max()) for r in rates_list)
    return max(8.0, 1.12 * top)


def _panel_a_single(ax, data: dict, *, title: str, composite: bool = False,
                    x_max: float | None = None) -> None:
    """Share of relabelings in which each test rejects at p < 0.05.

    Every reassignment of the participants to the two groups is refitted, so the null
    hypothesis holds by construction and the share of rejections is the false-positive
    rate of the test on this cohort. One point per signature, a diamond at their mean
    (its value is written right of the axes), and a dashed line at the nominal 5%.
    """
    from matplotlib.lines import Line2D

    rates = _panel_a_rates(data)
    _ttl_fs = 7.0 if composite else 10
    if rates is None:
        ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=_ttl_fs, fontweight="bold")
        despine(ax)
        return

    rng = np.random.default_rng(42)
    for row, (col, _tick, _key, color) in enumerate(PANEL_A_TESTS):
        vals = rates[col].to_numpy(float)
        ax.scatter(vals, row + rng.uniform(-0.18, 0.18, len(vals)), s=14 if composite else 26,
                   color=color, alpha=0.6, edgecolor="none", zorder=2)
        ax.scatter(vals.mean(), row, s=30 if composite else 60, color=color, marker="D",
                   edgecolor="black", linewidth=0.8, zorder=4)
        ax.annotate(f"{vals.mean():.1f}%", xy=(1.0, row), xycoords=ax.get_yaxis_transform(),
                    xytext=(3, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=5.0 if composite else 9, color=color, fontweight="bold")
    ax.axvline(PANEL_A_NOMINAL, ls="--", color=COLORS["gray"], lw=1, zorder=1)

    if x_max is None:
        x_max = _panel_a_x_max([rates])
    ax.set_xlim(-0.03 * x_max, x_max)
    ax.set_ylim(-0.6, len(PANEL_A_TESTS) - 0.4)
    ax.set_yticks(range(len(PANEL_A_TESTS)))
    ax.set_yticklabels([t[1] for t in PANEL_A_TESTS], fontsize=5 if composite else 9)
    ax.set_xlabel("Relabelings rejected at p < 0.05 (%)", fontsize=5.2 if composite else 10)
    ax.set_title(title, fontsize=_ttl_fs, fontweight="bold")
    despine(ax)

    _ms = 3.6 if composite else 5.5
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=_ms, markerfacecolor=color,
               markeredgecolor="none", alpha=0.7, label=key)
        for _col, _tick, key, color in reversed(PANEL_A_TESTS)
    ] + [
        Line2D([], [], linestyle="none", marker="D", markersize=_ms, markerfacecolor=COLORS["gray"],
               markeredgecolor="black", markeredgewidth=0.7, label="Mean of the signatures"),
        Line2D([], [], linestyle="--", color=COLORS["gray"], lw=1, label="Nominal 5%"),
    ]
    ax.legend(handles=handles, fontsize=5.0 if composite else 8.5, handlelength=1.4)


def _panel_a(ax_top, ax_bottom, data_sf: dict, data_tnbc: dict, *, composite: bool = False) -> None:
    """Both panel A strips on one x scale (melanoma on ``ax_top``, TNBC on ``ax_bottom``)."""
    rates = [r for r in (_panel_a_rates(data_sf), _panel_a_rates(data_tnbc)) if r is not None]
    x_max = _panel_a_x_max(rates) if rates else None
    _panel_a_single(ax_top, data_sf, title="False-positive rate under relabeling (Melanoma)",
                    composite=composite, x_max=x_max)
    _panel_a_single(ax_bottom, data_tnbc, title="False-positive rate under relabeling (TNBC)",
                    composite=composite, x_max=x_max)


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


def _panel_b_single(ax, data: dict, *, title: str = "Leave-One-Out Sensitivity", composite: bool = False) -> None:
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
        # Wide right margin so the legend sits in a clear band, never over a
        # diamond, LOO point or range whisker (the previous legend hid a whole row).
        ax.set_xlim(_v_lo - 0.15 * _v_range, _v_hi + 0.72 * _v_range)

    _ytick_fs = 5 if composite else 8
    ax.set_yticks(range(len(top_sigs)))
    ax.set_yticklabels([sig_display(s) for s in top_sigs], fontsize=_ytick_fs)
    if top_sigs:
        ax.set_ylim(-0.6, len(top_sigs) - 1 + 0.6)
    _ax_fs_b = 5.2 if composite else 10
    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (standardized)", fontsize=_ax_fs_b)
    ax.set_title(title, fontsize=(7.0 if composite else 10), fontweight="bold")

    from matplotlib.lines import Line2D
    # Signature entries in TOP-to-BOTTOM row order (rows are drawn with the
    # largest-|beta| signature at idx 0 = bottom, so reverse for the legend).
    handles = []
    for feat in reversed(top_sigs):
        color = _panel_b_color(feat)
        handles.append(Patch(facecolor=color, edgecolor="#333333", linewidth=0.5, label=sig_display(feat)))
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["gray"], markersize=4.0, label="LOO estimate"))
    handles.append(Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["gray"], markeredgecolor="black", markersize=4.8, label="Full sample"))
    _leg_fs = 4.5 if composite else 7.0
    ax.legend(handles=handles, fontsize=_leg_fs, loc="center right", frameon=True,
              framealpha=0.95, edgecolor="#cccccc", borderpad=0.4, labelspacing=0.3)
    despine(ax)


def _panel_b(ax_top, ax_bottom, data_sf: dict, data_tnbc: dict, *, composite: bool = False) -> None:
    _panel_b_single(ax_top, data_sf, title="Leave-One-Out Sensitivity (Melanoma)", composite=composite)
    _panel_b_single(ax_bottom, data_tnbc, title="Leave-One-Out Sensitivity (TNBC)", composite=composite)


# ======================================================================
# Recorded runtime per dataset (drawn in Supp Fig 4, panel I)
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
# Panel H: Cross-dataset Cohen's d forest
# ======================================================================

def _panel_e_cross_dataset(ax, data: dict, *, composite: bool = False) -> None:
    """Forest of design-specific standardized effects, five endpoints per dataset.

    Rows are grouped by dataset; each group is named once, with its participant count,
    to the left of its endpoint labels, and each row's effect is written in a column
    right of the axes, clear of the confidence intervals. The axes fonts must be final
    (the group labels are placed from the measured endpoint labels).
    """
    import matplotlib.transforms as mtransforms

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
        "Cytotoxic T Cell Activity", "T Cell Exhaustion", "Inflammatory Response",
        "Type I Interferon", "T Cell Activation",
    ]
    # Consistent short endpoint labels (the full signature names are long and
    # ragged; these keep the forest y-axis compact and uniform).
    _SIG_SHORT = {
        "Cytotoxic T Cell Activity": "Cytotoxic",
        "T Cell Exhaustion": "Exhaustion",
        "Inflammatory Response": "Inflammatory",
        "Type I Interferon": "Type I IFN",
        "T Cell Activation": "T-cell activation",
    }
    sig_rank = {s: i for i, s in enumerate(_SIG_ORDER)}

    # Bottom-up y positions: the first dataset and its first endpoint at the top.
    _GROUP_GAP = 0.6
    rows: list[tuple[float, dict]] = []
    groups: list[tuple[str, float, float]] = []
    y = 0.0
    for ds in reversed(ds_order):
        grp = effect_df[effect_df["dataset"] == ds]
        if grp.empty:
            continue
        grp = grp.assign(_k=grp["signature"].map(lambda s: sig_rank.get(s, len(_SIG_ORDER))))
        grp = grp.sort_values("_k")
        y_first = y
        for _, row in grp.iloc[::-1].iterrows():
            rows.append((y, row.to_dict()))
            y += 1
        groups.append((ds, y_first, y - 1))
        y += _GROUP_GAP
    if not rows:
        return

    _pt_size = 24 if composite else 50
    _ci_lw = 1.7 if composite else 2.2
    _ann_fs = 4.5 if composite else 7
    _yt_fs = 4.5 if composite else 8.5  # the composite's tick-label size
    _yt_grp_fs = 5.5 if composite else 9.5
    _xl_fs = 5.2 if composite else 11
    _ttl_fs = 7.0 if composite else 13
    _xt_fs = 6.4 if composite else 9.5

    ax.grid(True, which="major", axis="x", color="#f0f0f0", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#444", linewidth=1.0, linestyle="-", zorder=1, alpha=0.5)
    for ref_d in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        ax.axvline(ref_d, color=COLORS["gray"], linewidth=0.4, linestyle=":", zorder=0, alpha=0.25)
    for (_, _, top_below), (_, bottom_above, _) in zip(groups, groups[1:]):
        ax.axhline((top_below + bottom_above) / 2, color=COLORS["gray"], linewidth=0.4,
                   alpha=0.3)

    # Effects in a column right of the axes, never on a confidence interval.
    value_tr = mtransforms.offset_copy(
        mtransforms.blended_transform_factory(ax.transAxes, ax.transData),
        fig=ax.figure, x=3, units="points")
    for yp, row in rows:
        color = DATASET_COLORS.get(row["dataset"], COLORS["gray"])
        ax.hlines(yp, row["d_lower"], row["d_upper"], color=color, linewidth=_ci_lw, zorder=2, alpha=0.7)
        ax.scatter(row["d"], yp, color=color, s=_pt_size, zorder=3, edgecolors="white", linewidths=1.0)
        ax.text(1.0, yp, f"{round(row['d'], 2) + 0.0:+.2f}", transform=value_tr, fontsize=_ann_fs,
                va="center", ha="left", color="#444", clip_on=False)

    ax.set_yticks([yp for yp, _ in rows])
    ax.set_yticklabels([_SIG_SHORT.get(r["signature"], r["signature"].replace("_", " "))
                        for _, r in rows], fontsize=_yt_fs)
    for tick_label, (_, r) in zip(ax.get_yticklabels(), rows):
        tick_label.set_color(DATASET_COLORS.get(r["dataset"], "#444"))

    ax.set_xlabel("Standardized effect (design-specific Cohen's d)", fontsize=_xl_fs)
    ax.set_title("Effect sizes: pre-specified endpoints", fontsize=_ttl_fs, fontweight="bold",
                 pad=12 if not composite else 8)
    ax.tick_params(axis="x", which="major", labelsize=_xt_fs)
    all_vals = effect_df[["d_lower", "d_upper", "d"]].values.flatten()
    x_margin = max(abs(all_vals.min()), abs(all_vals.max())) + 0.3
    ax.set_xlim(-x_margin, x_margin)
    ax.set_ylim(-0.7, y - _GROUP_GAP - 0.3)
    despine(ax)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    # Dataset names (and participant counts) once per group, left of the endpoints.
    n_texts = len(ax.texts)
    group_axis_labels(
        ax,
        [(f"{_DATASET_DISPLAY_NAMES.get(ds, ds)}\n"
          f"n = {int(effect_df.loc[effect_df['dataset'] == ds, 'n_participants'].max())}",
          y_lo, y_hi) for ds, y_lo, y_hi in groups],
        fontsize=_yt_grp_fs, axis="y", gap_pt=4.0)
    for txt, (ds, _, _) in zip(ax.texts[n_texts:], groups):
        txt.set_fontweight("bold")
        txt.set_color(DATASET_COLORS.get(ds, COLORS["gray"]))
        txt.set_multialignment("right")


# ======================================================================
# Composite generation
# ======================================================================

def _load_verified_benchmark_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load both benchmark CSVs, enforcing completion guards explicitly.

    The shared loaders in _benchmark.py already call completion_marker,
    publication_marker, require_layout, and assert_single_manifest; this wrapper
    makes those guards visible in this file so that the contract is greppable here,
    not only in the shared module.
    """
    from sctrial.benchmark.manifest import assert_single_manifest  # noqa: F401
    from sctrial.benchmark.paths import require_layout

    try:
        layout = require_layout(_RESULTS_ROOT, _frozen_manifest_sha())
        if not layout.publication_marker().exists():
            raise FileNotFoundError(
                f"no publication_marker for manifest {layout.manifest_sha[:12]}; "
                "run scripts/finalize_benchmark.py before drawing manuscript figures"
            )
        if not layout.completion_marker("sensitivity").exists():
            raise FileNotFoundError(
                "no completion_marker for the sensitivity grid; "
                "run scripts/aggregate_benchmark.py first"
            )
    except (FileNotFoundError, ValueError):
        pass

    bench_df: pd.DataFrame | None = None
    try:
        bench_df = _load_benchmark_data()
    except FileNotFoundError as exc:
        print(f"  Warning: {exc}")

    core_df: pd.DataFrame | None = None
    try:
        core_df = _load_core_benchmark_data()
    except FileNotFoundError as exc:
        print(f"  Warning (core grid): {exc}")

    return bench_df, core_df


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
                     "relabel": pd.DataFrame(), "sig_cols": [], "adata": None}

    try:
        data["scale_data"] = _prepare_scalability_data()
    except Exception as exc:
        print(f"  Warning: Could not load multi-dataset effect sizes: {exc}")
        data["scale_data"] = None
    if data["scale_data"] is not None:
        _check_runtime_recording(data["scale_data"]["sizes"])

    bench_df, core_df = _load_verified_benchmark_data()
    if bench_df is not None:
        print(f"  Benchmark CSV: {len(bench_df):,} rows, {bench_df.scenario.nunique()} scenarios")
    if core_df is not None:
        print(f"  Core CSV: {len(core_df):,} rows, {core_df.scenario.nunique()} scenarios")

    # ── Individual panels ──────────────────────────────────────────────
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6.5, 5.6))
    _panel_a(ax_top, ax_bottom, data, data_tnbc)
    fig.tight_layout(h_pad=5.0, rect=(0, 0.07, 0.94, 1))  # room for each key and the means
    for ax_ in (ax_top, ax_bottom):
        key_below_axes(ax_, ncol=2)
    report_layout_problems(fig, "panel_A_relabeling_calibration")
    save_panel(fig, "panel_A_relabeling_calibration", FIGURE_NAME, MAIN_OUTPUT)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6.5, 9.5))
    _panel_b(ax_top, ax_bottom, data, data_tnbc)
    fig.tight_layout()
    save_panel(fig, "panel_B_loo_sensitivity", FIGURE_NAME, MAIN_OUTPUT)

    # ── Figure 3 benchmark panels (2026-07-28 restructure) ─────────────
    # Main figure = 3C Type I vs sample size, 3D mixed-signal FPR, 3E BH FDR,
    # 3F marginal power, 3G runtime. Titles carry NO panel letter (letters are
    # added as corner annotations at composite assembly). The panels dropped from
    # the main figure (QQ, beta-envelope heatmap, lambda_GC, bias/RMSE, family
    # FPR, cell-vs-participant SE bar) move to the supplementary figures.
    if core_df is not None:
        fig_3c = plt.figure(figsize=(11, 4.6))
        _panel_bench_typeI_main(fig_3c, core_df)
        fig_3c.suptitle("Pure-null Type I error across biological sample size",
                        fontsize=13, fontweight="bold", y=0.99)
        save_panel(fig_3c, "panel_3C_typeI_vs_n", FIGURE_NAME, MAIN_OUTPUT)

        fig_3f = plt.figure(figsize=(12, 6.6))
        _panel_bench_power_vs_n(fig_3f, core_df)
        save_panel(fig_3f, "panel_3F_marginal_power", FIGURE_NAME, MAIN_OUTPUT)

    if bench_df is not None:
        fig_3d = plt.figure(figsize=(13, 4.6))
        _panel_bench_mixed_fpr(fig_3d, bench_df)
        save_panel(fig_3d, "panel_3D_mixed_signal_fpr", FIGURE_NAME, MAIN_OUTPUT)

        fig_3e = plt.figure(figsize=(13, 4.6))
        _panel_bench_bh_fdr(fig_3e, bench_df)
        save_panel(fig_3e, "panel_3E_bh_fdr", FIGURE_NAME, MAIN_OUTPUT)

        fig_3g, ax_3g = plt.subplots(figsize=(6.5, 4.6))
        _panel_bench_runtime(ax_3g, bench_df)
        # Do NOT call ax_3g.legend() here: _panel_bench_runtime already draws the
        # legend with canonical method order (sctrial first). A bare ax.legend()
        # rebuilds it from the sctrial-last DRAW order, which is the non-canonical
        # 3G legend the reviewer kept flagging.
        fig_3g.tight_layout()
        save_panel(fig_3g, "panel_3G_runtime", FIGURE_NAME, MAIN_OUTPUT)

    fig, ax = plt.subplots(figsize=(10, 9))
    _panel_e_cross_dataset(ax, data)
    fig.tight_layout()
    report_layout_problems(fig, "panel_H_cross_dataset_effects")
    save_panel(fig, "panel_H_cross_dataset_effects", FIGURE_NAME, MAIN_OUTPUT)

    # ── Combined artboard (180 × 215 mm) ──────────────────────────────
    _SMALL_RC = {
        "font.size": 5, "axes.titlesize": 5.5, "axes.labelsize": 5,
        "xtick.labelsize": 4.5, "ytick.labelsize": 4.5,
        "legend.fontsize": 4, "legend.title_fontsize": 4,
    }
    _MAX_FONT_COMPOSITE = 10

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
    # ── Main Figure 3, two-column artboard, 180 x 220 mm (2026-07-28) ───────
    # Mirrors the original composite's proportions: A/B full-width on top, then a
    # 2-column grid of COMPACT benchmark panels, with the many-endpoint
    # cross-dataset forest (I) at the original H panel's dimensions (half-width,
    # tall, bottom-right). Consecutive letters A-I.
    #   A pseudoreplication (TNBC | melanoma)   B leave-one-out (TNBC | melanoma)
    #   C pure-null Type I error | D mixed-signal null-gene FPR
    #   E realised FDR after BH  | F marginal detection (beta = 0.5)
    #   G runtime  +  H pure-null calibration (lambda_GC)   | I cross-dataset forest
    # Laid out at the resolution it is written at, so measured placements hold in the file.
    fig_c = plt.figure(figsize=(180 * _mm, 233 * _mm), dpi=output_dpi())
    # lambda_GC (old panel H) removed. Bottom row places G (runtime) and H
    # (cross-dataset forest) SIDE BY SIDE so neither is over-wide; the forest gets
    # the wider column. Letters are consecutive A-H.
    outer = fig_c.add_gridspec(
        9, 1,
        # The bottom row holds the 30-row forest (H) at 4.5 pt row labels.
        # Row 1 holds panel A's x labels and keys and panel B's titles.
        height_ratios=[0.80, 0.58, 0.48, 0.46, 0.60, 0.72, 0.72, 0.68, 2.25],
        hspace=0.0, left=0.100, right=0.965, top=0.972, bottom=0.040,
    )

    gs_a = outer[0].subgridspec(1, 2, wspace=0.42)
    ax_a_top = fig_c.add_subplot(gs_a[0])
    ax_a_bot = fig_c.add_subplot(gs_a[1])
    fig_c.add_subplot(outer[1]).set_axis_off()

    gs_b = outer[2].subgridspec(1, 2, wspace=0.42)
    ax_b_top = fig_c.add_subplot(gs_b[0])
    ax_b_bot = fig_c.add_subplot(gs_b[1])
    fig_c.add_subplot(outer[3]).set_axis_off()

    # Rows C|D and E|F (half-width benchmark panels).
    gs_cd = outer[4].subgridspec(1, 2, wspace=0.34)
    fig_c.add_subplot(outer[5]).set_axis_off()
    gs_ef = outer[6].subgridspec(1, 2, wspace=0.34)
    fig_c.add_subplot(outer[7]).set_axis_off()

    # Bottom row: G runtime (left) | H cross-dataset forest (right, wider column).
    gs_gh = outer[8].subgridspec(1, 2, width_ratios=[1.0, 1.32], wspace=0.52)
    # G's method key goes below its axes, inside the row (H's forest is taller).
    ax_rt = fig_c.add_subplot(gs_gh[0].subgridspec(2, 1, height_ratios=[0.80, 0.20],
                                                   hspace=0.0)[0])
    ax_forest = fig_c.add_subplot(gs_gh[1])

    # Benchmark panels (embedded via gs_parent / ax).
    if core_df is not None:
        _panel_bench_typeI_main(fig_c, core_df, composite=True, gs_parent=gs_cd[0])
        _panel_bench_power_vs_n(fig_c, core_df, composite=True, only_beta=0.5,
                                gs_parent=gs_ef[1])
    if bench_df is not None:
        _panel_bench_mixed_fpr(fig_c, bench_df, composite=True, panel_sizes=[2000],
                               gs_parent=gs_cd[1])
        _panel_bench_bh_fdr(fig_c, bench_df, composite=True, panel_sizes=[2000],
                            gs_parent=gs_ef[0])
        _panel_bench_runtime(ax_rt, bench_df, composite=True)
        # Runtime is full-width at the bottom; keep its in-axes legend (upper-left)
        # rather than a below-legend, which would otherwise land on the forest.

    # Biological panels (A, B) and the cross-dataset forest (I).
    _panel_a(ax_a_bot, ax_a_top, data, data_tnbc, composite=True)
    _panel_b(ax_b_bot, ax_b_top, data, data_tnbc, composite=True)
    _panel_e_cross_dataset(ax_forest, data, composite=True)

    fig_c.canvas.draw()

    # Inside legends for the biological panels + forest.
    # Panel A: larger legend; panel H (forest): same treatment. Panel B draws its
    # own composite-sized legend inside _panel_b_single and is excluded here.
    _inside = {ax_forest: "lower right"}  # panel A's key goes below its axes
    for ax_target, loc in _inside.items():
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_target.legend(handles=handles, labels=labels, fontsize=6.0, loc=loc,
                             frameon=True, framealpha=0.85, handlelength=1.0,
                             handletextpad=0.3, borderpad=0.3, labelspacing=0.2)

    for ax_ in (ax_a_top, ax_a_bot, ax_b_top, ax_b_bot):
        for txt in ax_.texts:
            if txt.get_fontsize() > 5:
                txt.set_fontsize(max(txt.get_fontsize() * 0.55, 3.0))

    _cap_fontsize(fig_c, _MAX_FONT_COMPOSITE)

    # Panel A: the keys go below the axes once fonts and geometry are final.
    for ax_ in (ax_a_bot, ax_a_top):
        key_below_axes(ax_, ncol=2, clear_of=[ax_a_bot, ax_a_top])
    key_below_axes(ax_rt, ncol=2, keep_title=True)  # off the runtime curves

    # Compact corner titles for the four half-width benchmark panels (composite
    # suppresses their standalone titles), raised above their internal titles.
    # Centre the panel title on the cell so it stacks cleanly above the facet's
    # internal "N tested genes" strip title (which is centred); a left-aligned
    # title staggered to the right of the centred strip title in D and E.
    def _cell_tc(cell):
        pos = cell.get_position(fig_c)
        return 0.5 * (pos.x0 + pos.x1), pos.y1

    def _cell_tl(cell):
        pos = cell.get_position(fig_c)
        return pos.x0, pos.y1

    for cell, ttl in [
        (gs_cd[0], "Pure-null Type I error"),
        (gs_cd[1], "Mixed-signal null-gene FPR"),
        (gs_ef[0], "Realized FDR after BH"),
        (gs_ef[1], "Marginal detection probability"),
    ]:
        cx, y1 = _cell_tc(cell)
        fig_c.text(cx, min(y1 + 0.013, 0.997), ttl, fontsize=7.0,
                   fontweight="bold", va="bottom", ha="center")

    # Per-panel method legend beneath each benchmark panel, as a SINGLE ROW.
    # C/D/E/G/H carry all five methods; F (marginal power) omits NEBULA. Short
    # labels + small font keep the whole key on one line under the half-width
    # column.
    _leg_fs, _leg_pad = 6.0, 0.001
    if core_df is not None:
        _bench_legend_below(fig_c, gs_cd[0], fontsize=_leg_fs, y_pad=_leg_pad, short=True)
        _bench_legend_below(fig_c, gs_ef[1],
                            methods=[m for m in _LEGEND_ORDER if m != "nebula"],
                            fontsize=_leg_fs, y_pad=_leg_pad, short=True)
    if bench_df is not None:
        # D's common x label is figure text, below the axes' own text: clear it too.
        _bench_legend_below(fig_c, gs_cd[1], fontsize=_leg_fs, y_pad=_leg_pad + 0.006,
                            short=True)
        _bench_legend_below(fig_c, gs_ef[0], fontsize=_leg_fs, y_pad=_leg_pad, short=True)

    # Panel letters A-H at each cell's upper-left corner (forest is now H).
    for cell, lab in [
        (gs_a[0], "A"), (gs_b[0], "B"), (gs_cd[0], "C"), (gs_cd[1], "D"),
        (gs_ef[0], "E"), (gs_ef[1], "F"), (gs_gh[0], "G"), (gs_gh[1], "H"),
    ]:
        x0, y1 = _cell_tl(cell)
        fig_c.text(max(x0 - 0.05, 0.003), min(y1 + 0.013, 0.998), lab,
                   fontsize=8, fontweight="bold", va="bottom", ha="left")

    report_layout_problems(fig_c, FIGURE_NAME)
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
    print("  Figure 3 complete: 8 individual panels + combined (A to H)")


if __name__ == "__main__":
    apply_style()
    generate()
