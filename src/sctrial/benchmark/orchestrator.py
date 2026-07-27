"""Benchmark orchestrator — runs the scenario grid on the transcriptome simulator.

Every method is given the input its estimand assumes (see
:mod:`sctrial.benchmark.contracts`) and is scored against that estimand, not
against one shared "true beta". Both of those were wrong in the previous version
and each produced a published conclusion that did not survive checking.

Usage::

    from sctrial.benchmark.orchestrator import run_benchmark
    results = run_benchmark(n_jobs=16, output_dir="benchmark_results")
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import METHOD_ESTIMAND, prepare_inputs
from .scenario_contract import (
    check_scenario_results,
    check_simulation,
    completion_record,
)
from .simulator_v2 import TranscriptomeSimConfig, make_signal, nested_panels, simulate_trial_v2

logger = logging.getLogger(__name__)

# --- adaptive Monte Carlo stopping -------------------------------------------
# At 200 replicates the Monte Carlo SE of a 5% false-positive rate is
# sqrt(.05*.95/200) = 0.015, so 5% and 7% are indistinguishable -- which is the
# comparison the paper turns on. Rather than pay for a blanket 1,000 everywhere,
# add replicates only where the estimate is still imprecise.
# `n_iterations` is the BASE batch; these govern the extension only.
_MC_BATCH = 100
_MC_MAX = 1000

# A HIGHER cap where the estimand itself is discrete.
#
# Per-replicate power is a mean over the scenario's signal genes, so with one
# signal gene it is Bernoulli and its worst-case replicate SD is 0.5. Meeting the
# 0.01 power-MCSE target then needs sqrt(0.25)/0.01 squared = 2,500 replicates,
# and at 1,000 the best achievable is 0.0158. Four scenarios (50-gene panels at
# 2% and 4% signal) are in this position.
#
# The alternative was to relax the target where it is inconvenient, which would
# mean the prespecified precision requirement did not actually hold across the
# grid. These are the cheapest scenarios in the benchmark, so the common target
# is kept and the cap is raised instead. The rule is stated in terms of the
# DISCRETENESS OF THE ESTIMAND and is fixed before any method result is seen.
#
# Two signal genes nominally needs 1,250, but per-replicate gene outcomes are
# correlated through shared participants and random effects, so the theoretical
# figure is not a guaranteed bound. Both cases therefore get 2,500 and, if that
# still does not satisfy the empirical replicate-level MCSE, the scenario stops
# at its cap and reports the precision achieved.
_MC_MAX_SPARSE_SIGNAL = 2500
_SPARSE_SIGNAL_GENES = 2


def mc_max_for(scenario: dict) -> int:
    """The replicate cap for one scenario.

    Keyed on the SIGNAL GENE COUNT, not on the number of genes tested. A
    2,000-gene scenario does not carry 40x more independent information than a
    50-gene one: genes within a replicate share participants, libraries, random
    effects, normalisation and signal architecture. The independent unit is the
    simulated dataset, which is the same pseudoreplication argument the
    manuscript itself makes.
    """
    n_signal = int(scenario.get("n_signal", 0) or 0)
    if 0 < n_signal <= _SPARSE_SIGNAL_GENES:
        return _MC_MAX_SPARSE_SIGNAL
    return _MC_MAX
_MCSE_TARGET_FPR = 0.005
_MCSE_TARGET_POWER = 0.01


def _replicate_rates(rows: list) -> tuple[np.ndarray, np.ndarray]:
    """Per-REPLICATE null-FPR and power.

    The independent unit is the simulated dataset, not the gene. Genes within one
    replicate share its participants, its library sizes and its random effects, so
    pooling them would understate the Monte Carlo error -- pseudoreplication
    inside a benchmark whose subject is pseudoreplication.
    """
    df = pd.DataFrame(rows)
    if df.empty or "iteration" not in df:
        return np.array([]), np.array([])
    ok = df["pvalue"].notna()
    null = df[ok & ~df["is_signal"]]
    sig = df[ok & df["is_signal"]]
    fpr = (
        null.assign(hit=null["pvalue"] < 0.05).groupby("iteration")["hit"].mean().to_numpy()
        if len(null) else np.array([])
    )
    pw = (
        sig.assign(hit=sig["pvalue"] < 0.05).groupby("iteration")["hit"].mean().to_numpy()
        if len(sig) else np.array([])
    )
    return fpr, pw


def _mcse(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.inf


def _needs_more_replicates(rows: list, n_done: int, mc_max: int = _MC_MAX) -> tuple[bool, str]:
    """Whether this SCENARIO has reached its Monte Carlo precision target.

    Stopping is decided at the scenario level and applies to every method at once,
    so all methods always see exactly the same R simulated datasets. Stopping each
    method independently would break the pairing that makes the comparison
    powerful -- methods would then be compared on partly different data.

    Requires the criterion to hold for the WORST method: the run continues until
    every reported method is precise enough, or the cap is reached.
    """
    if n_done >= mc_max:
        return False, f"at the {mc_max}-replicate cap; report achieved precision"
    df = pd.DataFrame(rows)
    if df.empty or "method" not in df:
        return False, "no rows"
    worst_f, worst_p, need = 0.0, 0.0, False
    for method, grp in df.groupby("method", observed=True):
        fpr, pw = _replicate_rates(grp.to_dict("records"))
        se_f, se_p = _mcse(fpr), _mcse(pw)
        if len(fpr):
            worst_f = max(worst_f, se_f)
            need |= se_f > _MCSE_TARGET_FPR
        if len(pw):
            worst_p = max(worst_p, se_p)
            need |= se_p > _MCSE_TARGET_POWER
        del method
    return bool(need), f"worst-method MCSE fpr={worst_f:.4f} power={worst_p:.4f}"

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

_PANEL = 50
_SIGNAL_FRACTION = 0.20

# Exactly realisable at 50, 200, 500 and 2000 genes, so the panel-size x
# signal-burden grid is a complete factorial with no missing cells.
_SIGNAL_FRACTIONS = (0.02, 0.04, 0.10, 0.20)
_N_ITERATIONS = 200
_CELLS = 500

# Methods reported in the paper.
#
# ``limma_voom`` is the Gate G conventional pseudobulk comparator: a reviewer is
# entitled to ask how the standard tool behaves, and "we excluded it" is not an
# answer. It carries the repeated measure through ``duplicateCorrelation`` and is
# permitted to FAIL rather than silently fall back to an unpaired model; the
# failure rate is itself a reported result.
#
# ``edger_qlf`` stays excluded: edgeR-QL has no repeated-measures route for a
# two-arm design that is not rank-deficient, so including it would compare a
# different estimand under the same label.
CORE_METHODS = [
    "sctrial_did",
    "dreamlet",
    "nebula",
    "wilcoxon_paired",
    "limma_voom",
]

INTERNAL_METHODS = ["sctrial_mixed"]

# Primary signal architecture. ``one_directional`` is retained but named as a
# composition stress test: a coordinated shift moves the library-size reference,
# and about two thirds of the inflation previously attributed to empirical-Bayes
# moderation is that artifact.
_PRIMARY_ARCH = "balanced"


def scenario_seed(master_seed: int, scenario_name: str, replicate: int) -> int:
    """A seed addressed by (scenario, replicate index), not by draw order.

    With a single sequential RNG stream, adding a batch of replicates under
    adaptive stopping would renumber every later dataset, so replicate 437 would
    not be the same data across a resume, a different worker count, or a
    re-partitioned SLURM job. Hashing the address makes replicate 437 always
    replicate 437.

    hashlib rather than hash(): Python salts str hashing per process, so hash()
    is not reproducible across runs at all.
    """
    import hashlib

    key = f"{master_seed}|{scenario_name}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _scenario(
    name: str,
    description: str,
    *,
    design: str,
    panel_size: int = _PANEL,
    n_signal: int = 0,
    architecture: str = _PRIMARY_ARCH,
    magnitude: float = 0.5,
    arm_ratio: tuple[int, int] | None = None,
    **cfg,
) -> dict:
    # arm_ratio is emitted ALWAYS, including as None, so the scenario's design
    # always wins over anything inherited from the frozen calibration. An absent
    # key inherits; an explicit None overrides.
    cfg["arm_ratio"] = arm_ratio
    return {
        "name": name,
        "description": description,
        "panel_size": panel_size,
        "n_signal": int(n_signal),
        # The realised fraction, which is the only one that exists. There is no
        # separate "nominal" fraction to disagree with it.
        "signal_fraction": (int(n_signal) / panel_size) if panel_size else 0.0,
        "architecture": architecture,
        "magnitude": magnitude,
        "config_kwargs": {"design": design, **cfg},
    }


def build_scenario_grid(design: str = "two_arm") -> list[dict]:
    """Core scenario grid for one design family."""
    s: list[dict] = []

    # 1. Complete null across sample size
    for n in [8, 12, 20, 40, 60]:
        s.append(
            _scenario(
                f"null_n{n}",
                f"Complete null, n={n} per arm",
                design=design,
                n_per_arm=n,
                cells_per_pv_fixed=_CELLS,
            )
        )

    # 2. Null with nuisance heterogeneity: larger participant variance and a wide
    #    cell-yield distribution (resampled rather than fixed).
    for n in [20, 40]:
        s.append(
            _scenario(
                f"null_hetero_n{n}",
                f"Null, high participant SD + imbalanced cell yield, n={n}",
                design=design,
                n_per_arm=n,
                between_participant_sd=1.5,
                # The point of this scenario is IMBALANCED yield, so it uses the
                # anchor's own empirical cells-per-visit distribution (median 243,
                # CV 0.785) rather than a fixed count. `cells_scale=0.1` was a
                # relative factor inherited from a different anchor; against Treg
                # it silently meant ~30 cells.
                cells_per_pv_fixed=None,
                cells_scale=1.0,
            )
        )

    # 3. Sparse DE, primary (balanced) architecture
    for n in [20, 40, 60]:
        for beta in [0.2, 0.5, 1.0]:
            s.append(
                _scenario(
                    f"de_balanced_n{n}_b{beta}",
                    f"Sparse DE, balanced signs, n={n}, beta={beta}",
                    design=design,
                    n_per_arm=n,
                    n_signal=int(round(_PANEL * _SIGNAL_FRACTION)),
                    architecture="balanced",
                    magnitude=beta,
                    cells_per_pv_fixed=_CELLS,
                )
            )

    # 4. Heterogeneous effect magnitudes — the most realistic architecture
    for n in [20, 40]:
        for beta in [0.5, 1.0]:
            s.append(
                _scenario(
                    f"de_hetero_n{n}_b{beta}",
                    f"Sparse DE, heterogeneous magnitudes, n={n}, beta={beta}",
                    design=design,
                    n_per_arm=n,
                    n_signal=int(round(_PANEL * _SIGNAL_FRACTION)),
                    architecture="heterogeneous",
                    magnitude=beta,
                    cells_per_pv_fixed=_CELLS,
                )
            )

    # 5. One-directional COMPOSITION STRESS. Named so no regex on the scenario
    #    name can pool it with the primary DE scenarios.
    for n in [20, 40]:
        s.append(
            _scenario(
                f"compstress_onedir_n{n}",
                f"Composition stress: all signal genes +0.5, n={n}",
                design=design,
                n_per_arm=n,
                n_signal=int(round(_PANEL * _SIGNAL_FRACTION)),
                architecture="one_directional",
                cells_per_pv_fixed=_CELLS,
            )
        )

    # 6. Varying cell yield, spanning the anchor's own realistic range.
    #    Treg gives a median of 243 cells per participant-visit (min 13, max ~769),
    #    so the grid is sparse / empirical-median / rich. 5,000 was inherited from
    #    a whole-sample anchor and is unreachable for any single cell type.
    #    CHOSEN FROM THE POPULATION, deliberately before seeing any method
    #    ranking: a low-yield condition must not be raised because a comparator
    #    struggles there. Evaluability is reported alongside power for exactly
    #    this reason.
    for n_cells in [50, 250, 1000]:
        s.append(
            _scenario(
                f"cells_{n_cells}_n40",
                f"Cell yield {n_cells}/visit, n=40",
                design=design,
                n_per_arm=40,
                n_signal=int(round(_PANEL * _SIGNAL_FRACTION)),
                cells_per_pv_fixed=n_cells,
            )
        )

    # 7. Unequal arms (two-arm only)
    if design == "two_arm":
        for ratio in [(3, 7), (5, 10), (10, 20)]:
            s.append(
                _scenario(
                    f"imbal_{ratio[0]}v{ratio[1]}",
                    f"Unequal arms {ratio[0]}:{ratio[1]}",
                    design=design,
                    n_per_arm=sum(ratio),
                    arm_ratio=ratio,  # the only scenarios that set it
                    n_signal=int(round(_PANEL * _SIGNAL_FRACTION)),
                    cells_per_pv_fixed=_CELLS,
                )
            )

    # 8. Missing post visits
    for rate in [0.1, 0.2]:
        s.append(
            _scenario(
                f"missing_{int(rate * 100)}pct_n40",
                f"{int(rate * 100)}% of Post visits missing, n=40",
                design=design,
                n_per_arm=40,
                missing_rate=rate,
                n_signal=int(round(_PANEL * _SIGNAL_FRACTION)),
                cells_per_pv_fixed=_CELLS,
            )
        )

    return s


def build_sensitivity_grid(design: str = "two_arm", panels=None) -> list[dict]:
    """Panel size x signal BURDEN sensitivity, declared as integer gene counts.

    Panels are NESTED subsets of one simulated transcriptome, so a panel-size
    effect is separable from a gene-identity effect.

    SIGNAL IS AN INTEGER COUNT, and the fractions are chosen to be EXACTLY
    realisable at every panel size, so the grid is a complete factorial:

        fraction     50   200   500  2000
            2%        1     4    10    40
            4%        2     8    20    80
           10%        5    20    50   200
           20%       10    40   100   400

    The earlier 1% and 5% are not realisable at 50 genes (0.5 and 2.5 genes).
    Rounding them would mislabel the lowest-signal condition -- one gene out of 50
    is 2%, not 1% -- and skipping them instead leaves holes in the very panel-size
    comparison the analysis exists to make. Nothing is scientifically privileged
    about 1% and 5%; they were sensitivity values. 2% and 4% are more rigorous
    because the experimental factor is then exactly defined everywhere.
    """
    s: list[dict] = []
    n_per_arm = 40
    skipped: list[str] = []
    for panel in list(panels) if panels else [50, 200, 500, 2000]:
        s.append(
            _scenario(
                f"sens_null_g{panel}",
                f"Sensitivity null, {panel}-gene panel, n={n_per_arm}/arm",
                design=design,
                n_per_arm=n_per_arm,
                panel_size=panel,
                n_signal=0,
                cells_per_pv_fixed=_CELLS,
            )
        )
        for frac in _SIGNAL_FRACTIONS:
            exact = panel * frac
            if abs(exact - round(exact)) > 1e-9 or exact < 1:
                # Should never happen with the chosen fractions; kept so a future
                # edit that breaks exactness is announced rather than rounded.
                skipped.append(f"{panel}g x {frac:.0%} ({exact:g} genes)")
                continue
            n_sig = int(round(exact))
            for arch in ("balanced", "one_directional"):
                tag = "" if arch == "balanced" else "_onedir"
                s.append(
                    _scenario(
                        f"sens_g{panel}_n{n_sig}{tag}",
                        f"{panel} genes, {n_sig} signal ({n_sig / panel:.1%}), {arch}",
                        design=design,
                        n_per_arm=n_per_arm,
                        panel_size=panel,
                        n_signal=n_sig,
                        architecture=arch,
                        cells_per_pv_fixed=_CELLS,
                    )
                )
    if skipped:
        # Announced, never silent: a dropped cell that nobody is told about is
        # indistinguishable from one that was never designed.
        print(f"  sensitivity grid: skipped unrealisable cells -> {'; '.join(skipped)}")
    return s


# ---------------------------------------------------------------------------
# Single-iteration worker
# ---------------------------------------------------------------------------


def _run_single_iteration(args: tuple) -> list[dict]:
    """Run every method on one simulated dataset. Executed in worker processes."""
    import warnings

    warnings.filterwarnings("ignore")

    scenario_name, iteration, seed, scenario, methods, base_config = args

    # The frozen calibration is the FLOOR; a scenario may only override the knobs
    # it is explicitly varying. Building the config from scenario kwargs alone is
    # how the previous benchmark ran on dataclass defaults (2.3e7 UMIs per cell)
    # while the Methods described calibrated parameters -- the calibration existed,
    # it was simply never threaded through.
    kw = dict(base_config or {})
    kw.update(scenario["config_kwargs"])

    # A single-arm design tests Delta versus 0, so a common time effect is NOT
    # removed by the contrast the way it is in a two-arm DiD. Forcing it to zero
    # keeps a "null" scenario actually null.
    if kw.get("design") == "single_arm":
        kw.setdefault("time_effect", 0.0)

    panel_size = scenario["panel_size"]
    probe = TranscriptomeSimConfig(seed=seed, **kw)

    # Draw the panel first so the signal is defined on the tested genes, then
    # build the config with those effects. The panel is a nested subset of the
    # transcriptome and is reproducible from the seed alone.
    panels = nested_panels(probe, rng=np.random.default_rng(seed + 1))
    if panel_size not in panels:
        raise ValueError(f"panel size {panel_size} is not one of {sorted(panels)}")
    panel_genes = [f"gene_{i}" for i in panels[panel_size]]

    effects = (
        make_signal(
            panel_genes,
            scenario["n_signal"],
            scenario["architecture"],
            scenario["magnitude"],
            rng=np.random.default_rng(seed + 2),
        )
        if scenario["n_signal"] > 0
        else {}
    )
    if scenario["n_signal"] > 0 and len(effects) != scenario["n_signal"]:
        raise RuntimeError(
            f"{scenario_name}: requested {scenario['n_signal']} signal genes, "
            f"realised {len(effects)}"
        )
    cfg = TranscriptomeSimConfig(seed=seed, effects=effects, **kw)

    sim = simulate_trial_v2(cfg)

    # The design that was actually SIMULATED. Asserted against the request so a
    # config leak cannot silently flatten the sample-size axis again.
    realised_arms = list(sim["latent"]["arms"])
    n_treated = sum(a == "Treated" for a in realised_arms)
    n_control = len(realised_arms) - n_treated
    # Realised CELL yield too. Recording only the requested design is how a
    # collapsed grid stayed invisible; cell yield is the other axis a leaked
    # calibration field can silently capture.
    _cells = sim["pseudobulk_counts"]["n_cells"].to_numpy(dtype=float)
    cells_median = float(np.median(_cells))
    cells_min, cells_max = float(_cells.min()), float(_cells.max())
    # FULL requested-versus-realised contract, not just the arm count. Checks
    # design, visits, cell yield, panel size and signal count against what the
    # simulator actually produced. See benchmark/scenario_contract.py for why
    # some fields are asserted exactly and others only by generating mode.
    inputs = prepare_inputs(sim, panel_genes)
    oracle = inputs["oracle"]
    design_type = cfg.design
    signal_genes = set(effects)

    check_simulation(scenario, cfg, sim, panel_genes, signal_genes).raise_if_violated()

    rows = []
    for method in methods:
        t0 = time.time()
        try:
            results = _dispatch_method(method, inputs, design_type=design_type)
        except Exception as exc:
            logger.warning(
                "Method %s failed on %s iter %d: %s", method, scenario_name, iteration, exc
            )
            results = {
                g: {
                    "beta": np.nan,
                    "pvalue": np.nan,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "converged": False,
                    "failure_mode": "numerical",
                }
                for g in panel_genes
            }
        elapsed = time.time() - t0

        estimand = METHOD_ESTIMAND.get(method, "count_link")
        truth_table = oracle.get(estimand, {})
        for gene in panel_genes:
            r = results.get(gene, {})
            injected = effects.get(gene, 0.0)
            rows.append(
                {
                    "scenario": scenario_name,
                    "iteration": iteration,
                    "method": method,
                    "gene": gene,
                    # The value THIS method's estimand implies. Scoring every
                    # method against `injected_beta` compares different
                    # functionals and penalises whichever differs most from it.
                    "true_beta": float(truth_table.get(gene, injected)),
                    "injected_beta": float(injected),
                    "estimand": estimand,
                    "is_signal": gene in signal_genes,
                    "estimated_beta": r.get("beta", np.nan),
                    "pvalue": r.get("pvalue", np.nan),
                    "ci_lo": r.get("ci_lo", np.nan),
                    "ci_hi": r.get("ci_hi", np.nan),
                    "converged": r.get("converged", False),
                    # Every method receives the SAME panel. If a method's own
                    # internal filtering then drops a gene, that is reported here
                    # as an evaluability rate rather than silently shrinking the
                    # denominator of its calibration and power estimates.
                    "evaluable": bool(np.isfinite(r.get("pvalue", np.nan))),
                    "failure_mode": r.get("failure_mode", "numerical"),
                    # Wall time for the WHOLE iteration (all genes), matching the
                    # figure axis "Median runtime per iteration (s)". Storing
                    # elapsed/n_genes while plotting it as per-iteration
                    # understated the cost by the panel size and manufactured the
                    # "flat runtime scaling" claim.
                    "runtime_seconds": elapsed,
                    "runtime_scope": "per_iteration",
                    # REALISED, not requested. The two diverged once already.
                    "n_participants": len(realised_arms),
                    "n_treated": n_treated,
                    "n_control": n_control,
                    "cells_per_pv_median": cells_median,
                    "cells_per_pv_min": cells_min,
                    "cells_per_pv_max": cells_max,
                    "n_per_arm": cfg.n_per_arm,
                    "panel_size": panel_size,
                    "n_signal_realised": len(signal_genes),
                    "signal_fraction_realised": len(signal_genes) / max(panel_size, 1),
                    "architecture": scenario["architecture"],
                }
            )

    return rows


def _dispatch_method(method: str, inputs: dict, design_type: str = "two_arm") -> dict:
    """Route to a runner with its CONTRACTED input representation.

    See :mod:`sctrial.benchmark.contracts` for the contract table. Nothing is
    inferred here: an unknown method raises rather than getting a guessed
    representation.
    """
    panel = inputs["panel_genes"]

    if method in ("sctrial_did", "sctrial_mixed"):
        from .runners import sctrial_did

        return sctrial_did.run(
            inputs["participant_log1p_cpm"],
            panel,
            from_pseudobulk=True,
            design_type=design_type,
        )
    if method == "wilcoxon_paired":
        from .runners import wilcoxon_paired

        return wilcoxon_paired.run(
            inputs["participant_log1p_cpm"], panel, design_type=design_type
        )
    if method == "dreamlet":
        from .runners import dreamlet_runner

        return dreamlet_runner.run(
            inputs["pseudobulk_counts"],
            panel,
            design_type=design_type,
            lib_size=inputs["lib_size"],
        )
    if method == "limma_voom":
        from .runners import limma_voom

        return limma_voom.run(
            inputs["pseudobulk_counts"],
            panel,
            design_type=design_type,
            lib_size=inputs["lib_size"],
        )
    if method == "edger_qlf":
        from .runners import edger_qlf

        return edger_qlf.run(
            inputs["pseudobulk_counts"],
            panel,
            design_type=design_type,
            lib_size=inputs["lib_size"],
        )
    if method == "nebula":
        from .runners import nebula_runner

        return nebula_runner.run(
            inputs["cell_counts"],
            panel,
            design_type=design_type,
            lib_size=inputs["cell_lib_size"],
        )
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Grid driver
# ---------------------------------------------------------------------------


def _run_grid(
    grid_fn,
    designs: list[str],
    methods: list[str],
    n_iterations: int,
    n_jobs: int,
    output_dir: Path,
    resume: bool,
    combined_name: str,
    seed: int,
    base_config: dict | None = None,
    manifest: dict | None = None,
    adaptive: bool = True,
    completion_dir: Path | None = None,
) -> pd.DataFrame:
    """One driver for both grids.

    The core and sensitivity drivers were previously near-identical copies, and a
    resume defect fixed in one was not fixed in the other.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # No sequential RNG: every replicate seed is addressed by
    # (scenario, index) via scenario_seed, so adaptive extension and resume
    # cannot renumber datasets.
    all_results = []

    for design in designs:
        scenarios = grid_fn(design)
        print(f"\n{'=' * 60}")
        print(f"Design: {design} — {len(scenarios)} scenarios x {n_iterations} iterations")
        print(f"Methods: {methods}")
        print(f"Parallel workers: {n_jobs}")
        print(f"{'=' * 60}")

        for si, scenario in enumerate(scenarios):
            name = f"{design}__{scenario['name']}"
            csv_path = output_dir / f"{name}.csv"
            # Addressed by (scenario, index) so adaptive extension and resume
            # cannot renumber datasets. `rng` is retained only for reproducible
            # scenario ordering.
            def _seed(i: int, _n=name) -> int:
                return scenario_seed(seed, _n, i)

            if resume and csv_path.exists():
                existing = pd.read_csv(csv_path)
                # A partial file is worse than no file: it silently shrinks the
                # Monte Carlo sample for one cell of the grid while every summary
                # reports the nominal iteration count.
                n_done = existing["iteration"].nunique() if "iteration" in existing else 0
                # >= not ==: adaptive stopping may have run PAST the base batch,
                # and a complete deeper run must not be treated as partial.
                if n_done >= n_iterations:
                    print(f"  [{si + 1}/{len(scenarios)}] {name} — CACHED, skipping")
                    all_results.append(existing)
                    continue
                print(
                    f"  [{si + 1}/{len(scenarios)}] {name} — PARTIAL "
                    f"({n_done}/{n_iterations} iterations), re-running"
                )

            print(f"  [{si + 1}/{len(scenarios)}] {name}: {scenario['description']}")

            t0 = time.time()
            all_rows: list = []
            done = 0
            target = n_iterations
            # Adaptive stopping makes the replicate count a per-scenario OUTCOME.
            # Recording WHY a scenario stopped is what lets the aggregator tell a
            # legitimate early stop from a job that was killed mid-extension --
            # both leave more rows than the base batch, so a count cannot.
            stop_reason = "max_replicates_reached"
            mc_max = mc_max_for(scenario)

            def _flush(_rows=None, _path=csv_path) -> None:
                pd.DataFrame(_rows if _rows is not None else all_rows).to_csv(
                    _path, index=False
                )

            # ADAPTIVE MONTE CARLO. Run the base batch, then extend only while the
            # replicate-level Monte Carlo error is above target. Scenarios with an
            # obvious answer stop early; near-nominal ones, where the paper's
            # claims actually live, get more effort.
            while done < target:
                batch = [
                    (name, it, _seed(it), scenario, methods, base_config)
                    for it in range(done, target)
                ]
                if not batch:
                    break
                if n_jobs == 1:
                    for a in batch:
                        all_rows.extend(_run_single_iteration(a))
                else:
                    # 'spawn', never 'fork': a forked worker inherits R/BLAS state
                    # and produced demonstrably wrong results (edgeR FPR 0.002
                    # in-pipeline versus 0.05 standalone).
                    ctx = mp.get_context("spawn")
                    with ctx.Pool(n_jobs) as pool:
                        for k, out in enumerate(pool.imap(_run_single_iteration, batch)):
                            all_rows.extend(out)
                            if (k + 1) % 20 == 0:
                                elapsed = time.time() - t0
                                print(
                                    f"    {done + k + 1}/{target} iterations "
                                    f"({elapsed:.0f}s elapsed)", flush=True
                                )
                                _flush()
                done = target
                _flush()

                if not adaptive:
                    # NOT "precision_reached": adaptation was switched off, so no
                    # precision target was ever evaluated. Claiming it would make
                    # the stop reason unfalsifiable, and the aggregator refuses
                    # non-adaptive shards in a definitive run for exactly this
                    # reason -- a debug run must not be able to become a result.
                    stop_reason = "adaptive_disabled"
                    break
                more, why = _needs_more_replicates(all_rows, done, mc_max)
                if not more:
                    stop_reason = (
                        "max_replicates_reached" if done >= mc_max else "precision_reached"
                    )
                    print(f"    stopping at {done} replicates ({why})", flush=True)
                    break
                extra = min(_MC_BATCH, mc_max - done)
                if extra <= 0:
                    stop_reason = "max_replicates_reached"
                    break
                print(f"    extending +{extra} replicates ({why})", flush=True)
                target = done + extra

            df = pd.DataFrame(all_rows)
            # Stamp provenance on EVERY row. Figure loaders refuse to combine
            # rows from different manifests, which is what stops a corrected run
            # being silently averaged with the run it replaced.
            if manifest is not None:
                df["manifest_sha256"] = manifest["manifest_sha256"]
                df["git_sha"] = manifest.get("git_sha", "unknown")
            # SECOND GATE, on the recorded columns rather than the simulator
            # object, so a defect in the RECORDING path is caught too. The
            # redundancy is deliberate: the original leak survived because only
            # one layer was ever checked.
            report = check_scenario_results(
                name, scenario, df,
                manifest_sha=(manifest or {}).get("manifest_sha256"),
            )
            report.raise_if_violated()

            df.to_csv(csv_path, index=False)

            # The completion record is written LAST, so its absence marks a
            # truncated scenario. Nothing downstream treats a scenario as
            # complete without it.
            rec = completion_record(
                name, scenario, df,
                stop_reason=stop_reason,
                max_replicates=mc_max,
                manifest_sha=(manifest or {}).get("manifest_sha256"),
                mcse_target_fpr=_MCSE_TARGET_FPR,
                mcse_target_power=_MCSE_TARGET_POWER,
            )
            rec["contract_checks"] = report.checked
            # Passed in, never derived from the output path's NAME. Inferring
            # directory structure from a string is how a layout change silently
            # relocates records: the previous version keyed off
            # `output_dir.name == "scenarios"`, which stopped being true the
            # moment the grids were given their own subdirectories.
            comp_dir = completion_dir if completion_dir is not None else (
                output_dir.parent / "completion"
            )
            comp_dir.mkdir(parents=True, exist_ok=True)
            (comp_dir / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str))

            all_results.append(df)
            print(
                f"    Done in {time.time() - t0:.0f}s -> {csv_path.name} "
                f"[{rec['n_replicates_completed']} reps, {stop_reason}, "
                f"fpr MCSE {rec['fpr_mcse']:.4f}]"
            )

    # PRODUCERS DO NOT WRITE THE COMBINED FILE.
    #
    # The run is split across several SLURM jobs sharing one output directory.
    # Any producer that writes a combined file races the others, and the last to
    # finish wins -- silently, on the only file the figures read. Globbing inside
    # the producer does not fix it either: a shard that fails, is delayed, is
    # resumed, or leaves a stale file behind still yields a plausible result.
    #
    # scripts/aggregate_benchmark.py runs ONCE, under afterok on every producer,
    # and refuses to write unless the shards form exactly the expected scenario
    # set under one manifest. See its module docstring.
    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    print(f"\nWrote {len(all_results)} scenario file(s) to {output_dir}")
    print("Combined results are written by scripts/aggregate_benchmark.py, "
          "which runs after all shards succeed.")
    return combined


def run_benchmark(
    designs: list[str] | None = None,
    methods: list[str] | None = None,
    n_iterations: int = _N_ITERATIONS,
    n_jobs: int = 1,
    output_dir: str | Path = "benchmark_results",
    resume: bool = True,
    base_config: dict | None = None,
    manifest: dict | None = None,
    completion_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run the core scenario grid."""
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    return _run_grid(
        build_scenario_grid,
        designs or ["two_arm", "single_arm"],
        methods or CORE_METHODS,
        n_iterations,
        n_jobs,
        Path(output_dir),
        resume,
        "benchmark_combined.csv",
        seed=2024,
        base_config=base_config,
        manifest=manifest,
        completion_dir=Path(completion_dir) if completion_dir else None,
    )


def run_sensitivity_benchmark(
    designs: list[str] | None = None,
    methods: list[str] | None = None,
    n_iterations: int = _N_ITERATIONS,
    n_jobs: int = 1,
    output_dir: str | Path = "benchmark_results/sensitivity",
    resume: bool = True,
    base_config: dict | None = None,
    manifest: dict | None = None,
    panels: list[int] | None = None,
    completion_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run the panel-size x signal-fraction sensitivity grid."""
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    grid_fn = (
        (lambda design: build_sensitivity_grid(design, panels=panels))
        if panels
        else build_sensitivity_grid
    )
    return _run_grid(
        grid_fn,
        designs or ["two_arm"],
        methods or CORE_METHODS,
        n_iterations,
        n_jobs,
        Path(output_dir),
        resume,
        "sensitivity_combined.csv",
        seed=90210,
        base_config=base_config,
        manifest=manifest,
        completion_dir=Path(completion_dir) if completion_dir else None,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NatMeth benchmark: simulation grid")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers (-1 = all cores)")
    parser.add_argument("--n-iterations", type=int, default=_N_ITERATIONS)
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    parser.add_argument("--designs", nargs="+", default=["two_arm", "single_arm"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--sensitivity", action="store_true", help="run the sensitivity grid")
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    fn = run_sensitivity_benchmark if args.sensitivity else run_benchmark
    fn(
        designs=args.designs,
        methods=args.methods,
        n_iterations=args.n_iterations,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
