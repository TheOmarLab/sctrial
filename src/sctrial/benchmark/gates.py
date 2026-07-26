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
    "GateResult",
    "run_gates",
    "composition_ablation",
]

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


def _verdict(observed: float, sims: np.ndarray) -> tuple[float, str, float, float, float]:
    sims = sims[np.isfinite(sims)]
    if sims.size < 10 or not np.isfinite(observed):
        return np.nan, "INSUFFICIENT", np.nan, np.nan, np.nan
    lo95, hi95 = np.percentile(sims, [2.5, 97.5])
    lo99, hi99 = np.percentile(sims, [0.5, 99.5])
    pct = float((sims < observed).mean() * 100.0)
    if lo95 <= observed <= hi95:
        v = "PASS"
    elif lo99 <= observed <= hi99:
        v = "MARGINAL"
    else:
        v = "FAIL"
    return pct, v, float(np.median(sims)), float(lo95), float(hi95)


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
            pct, v, med, lo, hi = _verdict(observed.get(key, np.nan), arr)
            rows.append(
                GateResult(gate, key, float(observed.get(key, np.nan)), med, lo, hi, pct, v).as_row()
            )

    rows.extend(_distribution_gate(observed, sims))
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
                    "n_marginal": int((df["verdict"] == "MARGINAL").sum()),
                    "n_fail": int((df["verdict"] == "FAIL").sum()),
                    "failures": df.loc[df["verdict"] == "FAIL", "statistic"].tolist(),
                },
                fh,
                indent=2,
            )
        if verbose:
            print(f"wrote {out_dir / 'gate_envelopes.csv'}", flush=True)
    return df


def _distribution_gate(observed: dict, sims: list[dict]) -> list[dict]:
    """Gate E as a distribution test, not a comparison of summaries.

    The reference distribution is built from simulation-versus-simulation
    distances, so the test asks the only well-posed question available: is the
    real-versus-simulated discrepancy larger than the discrepancy between two
    draws of the simulator itself? Comparing a real-versus-simulated distance to
    zero would fail for any finite sample and tell us nothing.
    """
    obs_r = observed.get("_prepost_corr_genewise")
    grids = [np.asarray(s["_corr_quantiles"]) for s in sims if "_corr_quantiles" in s]
    if obs_r is None or len(grids) < 10:
        return [
            {
                "gate": "E_longitudinal",
                "statistic": "prepost_corr_distribution",
                "observed": np.nan,
                "sim_median": np.nan,
                "sim_lo95": np.nan,
                "sim_hi95": np.nan,
                "ratio": np.nan,
                "percentile": np.nan,
                "verdict": "INSUFFICIENT",
            }
        ]
    obs_grid = np.percentile(np.asarray(obs_r), np.linspace(1, 99, 99))
    stack = np.vstack(grids)
    centre = np.median(stack, axis=0)

    # Distance = max absolute quantile discrepancy (a KS-type statistic computed
    # on the shared quantile grid).
    d_obs = float(np.max(np.abs(obs_grid - centre)))
    d_sim = np.max(np.abs(stack - centre[None, :]), axis=1)
    pct, v, med, lo, hi = _verdict(d_obs, d_sim)
    # One-sided: only an unusually LARGE discrepancy is evidence against.
    hi95 = float(np.percentile(d_sim, 95))
    hi99 = float(np.percentile(d_sim, 99))
    v = "PASS" if d_obs <= hi95 else ("MARGINAL" if d_obs <= hi99 else "FAIL")
    return [
        {
            "gate": "E_longitudinal",
            "statistic": "prepost_corr_distribution",
            "observed": d_obs,
            "sim_median": med,
            "sim_lo95": 0.0,
            "sim_hi95": hi95,
            "ratio": d_obs / med if med else np.nan,
            "percentile": pct,
            "verdict": v,
        }
    ]


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
