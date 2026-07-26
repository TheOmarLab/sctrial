"""Monte Carlo calibration gates for the benchmark simulator.

A calibration judged on a single simulated realisation cannot distinguish "the
simulator is miscalibrated" from "this draw was unlucky". Every gate here is
therefore an **envelope test**: the statistic is recomputed on ``n_mc``
independent simulated datasets, and the real-data value is required to fall
inside the central 95% of the simulated distribution. A statistic whose real
value sits at the 3rd percentile of 200 replicates is a genuine mismatch; the
same value against one replicate is noise.

The gates
---------
=====  ==================================================================
Gate   Question
=====  ==================================================================
A      Does the transcriptome have the right occupancy? (genes detected
       per cell, zero fraction, gene-mean distribution)
B      Do depth and cell yield match? (UMI per cell across the whole
       distribution, cells per participant-visit)
C      Does the conditional mean-dispersion relationship match?
D      Is effect recovery driven by normalisation scope rather than by
       signal direction? (ablation, not an envelope test -- see
       :func:`composition_ablation`)
E      Does the longitudinal structure match **as a distribution**?
       (gene-wise pre/post correlation across participants)
F      Does each method receive the input its estimand assumes?
       (contract tests, in ``tests/test_benchmark_contracts.py``)
G      Is a conventional pseudobulk comparator included?
=====  ==================================================================

Gates A, B, C and E are envelope tests and live here. D is an ablation whose
answer is a comparison, not a calibration. F and G are contracts, enforced by
unit tests rather than by simulation.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "GATE_STATISTICS",
    "PINNED_STATISTICS",
    "GateResult",
    "run_gates",
    "composition_ablation",
]

# PINNED statistics are near-deterministic readbacks of an empirical pool the
# simulator resamples. Their agreement is true by construction, so a failure
# indicates an IMPLEMENTATION defect, not a fidelity defect -- and their
# agreement must never be presented as independent evidence that the simulator
# "recovered" a property of the data.
#
# Everything else is DERIVED: it emerges from the generative model after counts
# are drawn, and only those carry a fidelity verdict.
PINNED_STATISTICS = frozenset({
    # library-size pool, resampled directly
    "umi_per_cell_q05", "umi_per_cell_q25", "umi_per_cell_median",
    "umi_per_cell_q75", "umi_per_cell_q95",
    # cells-per-participant-visit pool, resampled directly
    "cells_per_pv_median", "cells_per_pv_cv",
    # gene-rate pool, permuted directly
    "gene_mean_log_mean", "gene_mean_log_sd",
    # per-gene dispersion, resampled paired with the gene rate
    "cond_alpha_median", "cond_alpha_q25", "cond_alpha_q75",
    "cond_alpha_log_sd", "cond_alpha_slope", "cond_alpha_n_genes",
})

GATE_STATISTICS: dict[str, list[str]] = {
    "A_transcriptome": [
        # zero_fraction is algebraically the MEAN of the same per-cell curve whose
        # quantiles follow, so the mean and the quantiles are not independent
        # evidence. mean/median is the shape discriminator between them.
        "genes_detected_per_cell_mean",
        "genes_detected_per_cell_meanmedian",
        "genes_detected_per_cell_median",
        "genes_detected_per_cell_q25",
        "genes_detected_per_cell_q75",
        "zero_fraction",
        "gene_mean_log_mean",
        "gene_mean_log_sd",
    ],
    "B_depth_yield": [
        "umi_per_cell_q05",
        "umi_per_cell_q25",
        "umi_per_cell_median",
        "umi_per_cell_q75",
        "umi_per_cell_q95",
        "cells_per_pv_median",
        "cells_per_pv_cv",
    ],
    "C_dispersion": [
        # Support size on BOTH arms. If these differ materially the quantile
        # comparisons are between differently supported distributions.
        "cond_alpha_n_genes",
        "cond_alpha_median",
        "cond_alpha_q25",
        "cond_alpha_q75",
        "cond_alpha_log_sd",
        "cond_alpha_slope",
    ],
    "E_longitudinal": [
        "prepost_corr_pooled",
        "prepost_corr_genewise_median",
        "prepost_corr_genewise_mean",
        "prepost_corr_genewise_sd",
        "prepost_corr_genewise_q10",
        "prepost_corr_genewise_q25",
        "prepost_corr_genewise_q75",
        "prepost_corr_genewise_q90",
        "between_participant_sd",
        "delta_sd_median",
    ],
}


@dataclass
class GateResult:
    """Envelope verdict for one statistic."""

    gate: str
    statistic: str
    observed: float
    sim_median: float
    sim_lo95: float
    sim_hi95: float
    percentile: float
    verdict: str

    def as_row(self) -> dict:
        # `ratio` is meaningless for a statistic already on a log scale and
        # undefined for a signed one. gene_mean_log_mean reported ratio 0.995 for a
        # true 2.20% discrepancy (understated 4.5x), and prepost_corr_genewise_q25
        # printed "n/a" because observed and simulated have opposite signs -- the
        # formula announcing its own breakdown. `z` is scale-free and always
        # defined, and is the discrepancy the verdict is actually based on.
        sigma_mc = (self.sim_hi95 - self.sim_lo95) / 3.92 if np.isfinite(self.sim_hi95) else np.nan
        log_scale = "_log_" in self.statistic or self.statistic.endswith("_log_sd")
        if log_scale:
            discrepancy = self.observed - self.sim_median  # already logs: difference IS the effect
            kind = "log_difference"
        elif "corr" in self.statistic or self.statistic.endswith("_slope"):
            discrepancy = self.observed - self.sim_median  # signed, bounded: ratio is invalid
            kind = "signed_difference"
        else:
            discrepancy = (
                self.observed / self.sim_median - 1.0
                if np.isfinite(self.sim_median) and self.sim_median != 0.0
                else np.nan
            )
            kind = "relative"
        return {
            "gate": self.gate,
            "statistic": self.statistic,
            "kind": kind,
            "observed": self.observed,
            "sim_median": self.sim_median,
            "sim_lo95": self.sim_lo95,
            "sim_hi95": self.sim_hi95,
            "discrepancy": discrepancy,
            "sigma_mc": sigma_mc,
            "z": (
                (self.observed - self.sim_median) / sigma_mc
                if np.isfinite(sigma_mc) and sigma_mc > 0
                else np.nan
            ),
            "percentile": self.percentile,
            "verdict": self.verdict,
        }


def _verdict(
    observed: float,
    sims: np.ndarray,
    boot: np.ndarray | None = None,
) -> tuple[float, str, float, float, float]:
    """Verdict for one statistic.

    Two regimes, and the difference matters:

    * With a participant BOOTSTRAP of the reference cohort (``boot``), the
      tolerance is the cohort's own sampling uncertainty. PASS when the
      simulator's discrepancy from the observed value is no larger than the 95th
      percentile of bootstrap discrepancies; INCONCLUSIVE to the 99th; FAIL
      beyond. A simulator closer to TNBC than two draws of TNBC are to each other
      cannot be distinguished from a second run of the same study.
    * Without it, fall back to the Monte Carlo envelope. That envelope is ~0.4%
      wide at 141,553 cells and tests exact equality, so it is retained only as a
      diagnostic and its failures must not be read as fidelity failures.

    INCONCLUSIVE is a real state, not a courtesy. Previously that condition could
    not be expressed and was silently coded as FAIL, which is how a resolution
    limit gets reported as a simulator defect.
    """
    sims = sims[np.isfinite(sims)]
    if sims.size < 10 or not np.isfinite(observed):
        return np.nan, "INSUFFICIENT", np.nan, np.nan, np.nan
    lo95, hi95 = np.percentile(sims, [2.5, 97.5])
    pct = float((sims < observed).mean() * 100.0)
    sim_med = float(np.median(sims))

    if boot is not None:
        boot = boot[np.isfinite(boot)]
        if boot.size >= 20:
            # Discrepancies of bootstrap realisations from the observed value.
            ref = np.abs(boot - observed)
            d = abs(observed - sim_med)
            t95, t99 = np.percentile(ref, [95, 99])
            v = "PASS" if d <= t95 else ("INCONCLUSIVE" if d <= t99 else "FAIL")
            return pct, v, sim_med, float(lo95), float(hi95)

    lo99, hi99 = np.percentile(sims, [0.5, 99.5])
    if lo95 <= observed <= hi95:
        v = "PASS"
    elif lo99 <= observed <= hi99:
        v = "INCONCLUSIVE"
    else:
        v = "FAIL"
    return pct, v, sim_med, float(lo95), float(hi95)


def _one_replicate(args: tuple) -> dict:
    """Summarise one simulated replicate. Module-level for spawn picklability."""
    cfg_kwargs, seed = args
    from .calibration import summarize_simulation
    from .simulator_v2 import TranscriptomeSimConfig

    cfg = TranscriptomeSimConfig(**{**cfg_kwargs, "seed": seed})
    stats = summarize_simulation(cfg).statistics()
    # The gene-wise correlation vector is kept for the distributional gate but is
    # far too large to return per replicate at full scale; summarise to a fixed
    # quantile grid, which is what the distribution test consumes.
    r = stats.pop("_prepost_corr_genewise", None)
    if r is not None and len(r):
        stats["_corr_quantiles"] = np.percentile(r, np.linspace(1, 99, 99)).tolist()
    return stats


def run_gates(
    cfg,
    observed: dict,
    n_mc: int = 200,
    n_jobs: int = 8,
    seed0: int = 100_000,
    out_dir: str | Path | None = None,
    verbose: bool = True,
    bootstrap: dict | None = None,
) -> pd.DataFrame:
    """Run every envelope gate.

    Parameters
    ----------
    cfg
        A :class:`~sctrial.benchmark.simulator_v2.TranscriptomeSimConfig`. It is
        used verbatim except for ``seed``, which is varied across replicates.
    observed
        Real-data statistics from
        :meth:`sctrial.benchmark.calibration.SummaryAccumulator.statistics`.
    n_mc
        Number of simulated replicates. 200 gives a 2.5th-percentile bound with
        about 5 order statistics below it, which is the practical minimum for a
        95% envelope; 500 is preferable when the compute allows.
    """
    from dataclasses import asdict

    cfg_kwargs = {k: v for k, v in asdict(cfg).items()}
    cfg_kwargs.pop("seed", None)
    args = [(cfg_kwargs, seed0 + i) for i in range(n_mc)]

    if verbose:
        print(f"running {n_mc} Monte Carlo replicates on {n_jobs} workers", flush=True)
    if n_jobs > 1:
        ctx = mp.get_context("spawn")  # fork corrupts inherited BLAS/R state
        with ctx.Pool(n_jobs) as pool:
            sims = []
            for i, s in enumerate(pool.imap_unordered(_one_replicate, args), 1):
                sims.append(s)
                if verbose and i % 10 == 0:
                    print(f"  {i}/{n_mc} replicates", flush=True)
    else:
        sims = [_one_replicate(a) for a in args]

    rows: list[dict] = []
    for gate, keys in GATE_STATISTICS.items():
        for key in keys:
            arr = np.array([s.get(key, np.nan) for s in sims], dtype=float)
            ref = (
                np.array([b.get(key, np.nan) for b in bootstrap], dtype=float)
                if bootstrap
                else None
            )
            pct, v, med, lo, hi = _verdict(observed.get(key, np.nan), arr, ref)
            row = GateResult(
                gate, key, float(observed.get(key, np.nan)), med, lo, hi, pct, v
            ).as_row()
            # A PINNED statistic is a readback of a pool the simulator resamples;
            # its verdict reports wiring, not fidelity.
            row["class"] = "PINNED" if key in PINNED_STATISTICS else "DERIVED"
            if ref is not None and np.isfinite(ref).sum() >= 20:
                r = ref[np.isfinite(ref)]
                row["boot_tol95"] = float(np.percentile(np.abs(r - row["observed"]), 95))
                row["boot_tol99"] = float(np.percentile(np.abs(r - row["observed"]), 99))
                row["discrepancy_abs"] = abs(row["observed"] - med)
            rows.append(row)

    rows.extend(_distribution_gate(observed, sims, bootstrap))
    df = pd.DataFrame(rows)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "gate_envelopes.csv", index=False)
        with open(out_dir / "gate_summary.json", "w") as fh:
            json.dump(
                {
                    "n_mc": n_mc,
                    "n_statistics": len(df),
                    "n_pass": int((df["verdict"] == "PASS").sum()),
                    "n_inconclusive": int((df["verdict"] == "INCONCLUSIVE").sum()),
                    "n_fail": int((df["verdict"] == "FAIL").sum()),
                    # Only DERIVED failures are fidelity failures; a PINNED failure
                    # is an implementation defect and is reported separately.
                    "n_fail_derived": int(
                        ((df["verdict"] == "FAIL") & (df["class"] == "DERIVED")).sum()
                    ),
                    "n_fail_pinned": int(
                        ((df["verdict"] == "FAIL") & (df["class"] == "PINNED")).sum()
                    ),
                    "failures": df.loc[df["verdict"] == "FAIL", "statistic"].tolist(),
                    "bootstrap_used": bool(bootstrap),
                },
                fh,
                indent=2,
            )
        if verbose:
            print(f"wrote {out_dir / 'gate_envelopes.csv'}", flush=True)
    return df


def _distribution_gate(observed: dict, sims: list[dict], bootstrap: list | None = None) -> list[dict]:
    """Gate E as a DISTRIBUTION test with an empirically calibrated noise floor.

    Scalar summaries can agree while the distribution does not: the pooled
    correlation passed at ratio 0.89 in a run where five distribution statistics
    failed. So the comparison is between whole quantile profiles.

    The reference is the distance between two PARTICIPANT-BOOTSTRAP realisations
    of the reference cohort. If the real-versus-simulated distance is no larger
    than the distance routinely seen between two draws of the study itself, the
    simulator is as close as a second run of that study would be, which is the
    strongest fidelity claim the data can support. Comparing the distance to zero
    would fail for any finite sample and demanding every quantile match to 0.5% is
    not a meaningful requirement.

    Falls back to a simulation-versus-simulation reference when no bootstrap is
    supplied; that answers a weaker question (is the discrepancy larger than the
    simulator's own run-to-run variation) and is labelled accordingly.
    """
    # `corr_quantiles` is what measure_targets writes for real data;
    # `_corr_quantiles` is what a simulated replicate carries in memory. Accept
    # either so the gate works from a targets file and from a live accumulator.
    obs_grid = observed.get("corr_quantiles", observed.get("_corr_quantiles"))
    grids = [np.asarray(s["_corr_quantiles"]) for s in sims if "_corr_quantiles" in s]
    if obs_grid is None or len(grids) < 10:
        return [{
            "gate": "E_longitudinal", "statistic": "prepost_corr_distribution",
            "class": "DERIVED", "kind": "wasserstein", "observed": np.nan,
            "sim_median": np.nan, "sim_lo95": np.nan, "sim_hi95": np.nan,
            "discrepancy": np.nan, "sigma_mc": np.nan, "z": np.nan,
            "percentile": np.nan, "verdict": "INSUFFICIENT",
        }]
    obs_grid = np.asarray(obs_grid)
    stack = np.vstack(grids)

    # 1-Wasserstein between two distributions on a shared quantile grid is the
    # mean absolute difference of their quantile functions.
    def _w1(a, b):
        return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

    def _pairwise(grids_list, seed, n_max=2000):
        n = len(grids_list)
        rng = np.random.default_rng(seed)
        pairs = rng.integers(0, n, size=(min(n_max, n * n), 2))
        return np.array([_w1(grids_list[i], grids_list[j]) for i, j in pairs if i != j])

    # BOTH SIDES PAIRWISE. Every realisation contributes to a centroid, so a
    # distance-to-centroid is systematically smaller than a distance between two
    # independent draws -- by roughly sqrt(2) for symmetric noise. Comparing an
    # observed distance-to-centroid against a pairwise reference therefore biased
    # the verdict toward PASS, which is the dangerous direction; and using
    # centroid distances on both sides made it too strict, so two independent
    # samples from the IDENTICAL distribution scored INCONCLUSIVE (W1 0.0099
    # against a 95th percentile of 0.0084).
    #
    # The observed statistic is now the median distance from the real data to each
    # INDEPENDENT simulated realisation, and the reference is the distance between
    # two independent realisations of the reference process. Like against like.
    d_obs = float(np.median([_w1(obs_grid, g) for g in stack]))

    # Accept either key. The guard previously tested for BOTH spellings but the
    # body read only one, so a bootstrap carrying `_corr_quantiles` alone would
    # have raised KeyError instead of being used.
    boot_grids = []
    for b in bootstrap or []:
        if not isinstance(b, dict):
            continue
        g = b.get("corr_quantiles", b.get("_corr_quantiles"))
        if g is not None and len(g):
            boot_grids.append(np.asarray(g))
    if len(boot_grids) >= 20:
        # Preferred: carries the real cohort's participant-level sampling
        # variability, which is the variation a second run of this study would see.
        ref = _pairwise(boot_grids, seed=7)
        basis = "participant_bootstrap"
    else:
        # Fallback: reflects only the simulator's Monte Carlo noise, a tighter and
        # therefore more conservative reference. Labelled so it is never mistaken
        # for the bootstrap-calibrated version.
        ref = _pairwise([np.asarray(g) for g in stack], seed=11)
        basis = "simulation_only"

    hi95, hi99 = float(np.percentile(ref, 95)), float(np.percentile(ref, 99))
    v = "PASS" if d_obs <= hi95 else ("INCONCLUSIVE" if d_obs <= hi99 else "FAIL")
    return [{
        "gate": "E_longitudinal",
        "statistic": f"prepost_corr_distribution[{basis}]",
        "class": "DERIVED",
        "kind": "wasserstein",
        "observed": d_obs,
        "sim_median": float(np.median(ref)),
        "sim_lo95": 0.0,
        "sim_hi95": hi95,
        "discrepancy": d_obs,
        "sigma_mc": float(np.std(ref)),
        "z": np.nan,
        "boot_tol95": hi95,
        "boot_tol99": hi99,
        "percentile": float((ref < d_obs).mean() * 100.0),
        "verdict": v,
    }]


# ---------------------------------------------------------------------------
# Gate D -- composition ablation (a comparison, not an envelope test)
# ---------------------------------------------------------------------------


def composition_ablation(
    cfg,
    architectures=("balanced", "heterogeneous", "one_directional"),
    signal_fraction: float = 0.2,
    magnitude: float = 0.5,
    panel_size: int = 200,
    n_rep: int = 5,
    seed0: int = 900_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Isolate normalisation scope from signal direction.

    For each signal architecture the same generated cells are analysed three
    ways: normalised on the full transcriptome, normalised on the tested panel
    only, and against the noiseless oracle estimand. If the recovery gap tracks
    normalisation scope rather than signal direction, the compositional
    explanation is demonstrated rather than merely consistent with the data.
    """
    from dataclasses import replace

    from .simulator_v2 import (
        build_params,
        iter_pv_blocks,
        make_signal,
        nested_panels,
        oracle_estimands,
    )

    rows = []
    for arch in architectures:
        for rep in range(n_rep):
            seed = seed0 + rep
            panels = nested_panels(replace(cfg, seed=seed), rng=np.random.default_rng(seed + 1))
            panel = [f"gene_{i}" for i in panels[panel_size]]
            effects = make_signal(
                panel, signal_fraction, arch, magnitude, rng=np.random.default_rng(seed + 2)
            )
            c = replace(cfg, seed=seed, effects=effects)
            params = build_params(c)
            oracle = oracle_estimands(params)

            gidx = params["gene_index"]
            pv: dict[tuple, np.ndarray] = {}
            meta: dict[tuple, str] = {}
            for blk in iter_pv_blocks(c, params=params):
                key = (blk["participant"], blk["visit"])
                s = blk["counts"].sum(axis=0).astype(np.float64)
                pv[key] = pv.get(key, 0) + s
                meta[key] = blk["arm"]

            sig = [g for g in effects]
            sig_idx = [gidx[g] for g in sig]
            panel_idx = [gidx[g] for g in panel]

            for scope in ("full_transcriptome", "panel_only"):
                est = _did_log1p_cpm(pv, meta, sig_idx, panel_idx, scope)
                truth = np.array([oracle["log1p_cpm"][i] for i in sig_idx])
                inj = np.array([oracle["count_link"][i] for i in sig_idx])
                rows.append(
                    {
                        "architecture": arch,
                        "rep": rep,
                        "scope": scope,
                        "mean_estimate": float(np.mean(est)),
                        "mean_oracle_log1p_cpm": float(np.mean(truth)),
                        "mean_injected_beta": float(np.mean(inj)),
                        "bias_vs_oracle": float(np.mean(est - truth)),
                        "bias_vs_injected": float(np.mean(est - inj)),
                    }
                )
            if verbose:
                print(f"  ablation {arch} rep {rep + 1}/{n_rep}", flush=True)
    return pd.DataFrame(rows)


def _did_log1p_cpm(pv, meta, sig_idx, panel_idx, scope):
    """Participant-level DiD of log(1+CPM) under a given normalisation scope."""
    keys = sorted(pv)
    participants = sorted({k[0] for k in keys})
    deltas, treated = [], []
    for p in participants:
        vals = {}
        for visit in ("Pre", "Post"):
            v = pv.get((p, visit))
            if v is None:
                break
            denom = v.sum() if scope == "full_transcriptome" else v[panel_idx].sum()
            denom = denom if denom > 0 else 1.0
            vals[visit] = np.log1p(v[sig_idx] / denom * 1e6)
        if len(vals) != 2:
            continue
        deltas.append(vals["Post"] - vals["Pre"])
        treated.append(meta[(p, "Pre")] == "Treated")
    deltas = np.array(deltas)
    treated = np.array(treated)
    if (~treated).any():
        return deltas[treated].mean(axis=0) - deltas[~treated].mean(axis=0)
    return deltas.mean(axis=0)
