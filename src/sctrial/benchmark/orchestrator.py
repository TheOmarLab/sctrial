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

import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import METHOD_ESTIMAND, prepare_inputs
from .simulator_v2 import TranscriptomeSimConfig, make_signal, nested_panels, simulate_trial_v2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

_PANEL = 50
_SIGNAL_FRACTION = 0.20
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


def _scenario(
    name: str,
    description: str,
    *,
    design: str,
    panel_size: int = _PANEL,
    signal_fraction: float = 0.0,
    architecture: str = _PRIMARY_ARCH,
    magnitude: float = 0.5,
    **cfg,
) -> dict:
    return {
        "name": name,
        "description": description,
        "panel_size": panel_size,
        "signal_fraction": signal_fraction,
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
                cells_scale=0.1,
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
                    signal_fraction=_SIGNAL_FRACTION,
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
                    signal_fraction=_SIGNAL_FRACTION,
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
                signal_fraction=_SIGNAL_FRACTION,
                architecture="one_directional",
                cells_per_pv_fixed=_CELLS,
            )
        )

    # 6. Varying cell yield
    for n_cells in [200, 1000, 5000]:
        s.append(
            _scenario(
                f"cells_{n_cells}_n40",
                f"Cell yield {n_cells}/visit, n=40",
                design=design,
                n_per_arm=40,
                signal_fraction=_SIGNAL_FRACTION,
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
                    arm_ratio=ratio,
                    signal_fraction=_SIGNAL_FRACTION,
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
                signal_fraction=_SIGNAL_FRACTION,
                cells_per_pv_fixed=_CELLS,
            )
        )

    return s


def build_sensitivity_grid(design: str = "two_arm", panels=None) -> list[dict]:
    """Panel size x signal fraction sensitivity.

    Panels are NESTED subsets of one simulated transcriptome, so a panel-size
    effect is separable from a gene-identity effect. With independently drawn
    panels the two are confounded, and "progressive miscalibration with panel
    size" could not be distinguished from a change in which genes were tested.

    Signal counts are reported as REALISED fractions rather than nominal labels:
    at 50 genes ``round(50 * 0.01)`` is 1 gene, i.e. 2%, and the previous loaders
    parsed the scenario name rather than the data, manufacturing an apparent
    panel-size dependence.
    """
    s: list[dict] = []
    n_per_arm = 40
    # Panel sizes are selectable so the 2000-gene cells can run as their own job:
    # one 2000-gene iteration measured 495 s against 44 s at 50 genes, so mixing
    # them puts the whole grid at the mercy of the slowest cells.
    for panel in list(panels) if panels else [50, 200, 500, 2000]:
        s.append(
            _scenario(
                f"sens_null_g{panel}",
                f"Sensitivity null, {panel}-gene panel, n={n_per_arm}",
                design=design,
                n_per_arm=n_per_arm,
                panel_size=panel,
                signal_fraction=0.0,
                cells_per_pv_fixed=_CELLS,
            )
        )
        for frac in [0.01, 0.05, 0.10, 0.20]:
            realised = max(1, int(round(panel * frac))) / panel
            for arch in ("balanced", "one_directional"):
                tag = "" if arch == "balanced" else "_onedir"
                s.append(
                    _scenario(
                        f"sens_g{panel}_f{int(frac * 100)}{tag}",
                        f"{panel} genes, nominal {frac:.0%} / realised {realised:.1%} "
                        f"signal, {arch}",
                        design=design,
                        n_per_arm=n_per_arm,
                        panel_size=panel,
                        signal_fraction=frac,
                        architecture=arch,
                        cells_per_pv_fixed=_CELLS,
                    )
                )
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
            scenario["signal_fraction"],
            scenario["architecture"],
            scenario["magnitude"],
            rng=np.random.default_rng(seed + 2),
        )
        if scenario["signal_fraction"] > 0
        else {}
    )
    cfg = TranscriptomeSimConfig(seed=seed, effects=effects, **kw)

    sim = simulate_trial_v2(cfg)
    inputs = prepare_inputs(sim, panel_genes)
    oracle = inputs["oracle"]
    design_type = cfg.design
    signal_genes = set(effects)

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
                    "failure_mode": r.get("failure_mode", "numerical"),
                    # Wall time for the WHOLE iteration (all genes), matching the
                    # figure axis "Median runtime per iteration (s)". Storing
                    # elapsed/n_genes while plotting it as per-iteration
                    # understated the cost by the panel size and manufactured the
                    # "flat runtime scaling" claim.
                    "runtime_seconds": elapsed,
                    "runtime_scope": "per_iteration",
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
) -> pd.DataFrame:
    """One driver for both grids.

    The core and sensitivity drivers were previously near-identical copies, and a
    resume defect fixed in one was not fixed in the other.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
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
            seeds = [int(rng.integers(0, 2**31)) for _ in range(n_iterations)]

            if resume and csv_path.exists():
                existing = pd.read_csv(csv_path)
                # A partial file is worse than no file: it silently shrinks the
                # Monte Carlo sample for one cell of the grid while every summary
                # reports the nominal iteration count.
                n_done = existing["iteration"].nunique() if "iteration" in existing else 0
                if n_done == n_iterations:
                    print(f"  [{si + 1}/{len(scenarios)}] {name} — CACHED, skipping")
                    all_results.append(existing)
                    continue
                print(
                    f"  [{si + 1}/{len(scenarios)}] {name} — PARTIAL "
                    f"({n_done}/{n_iterations} iterations), re-running"
                )

            print(f"  [{si + 1}/{len(scenarios)}] {name}: {scenario['description']}")
            task_args = [
                (name, it, seeds[it], scenario, methods, base_config)
                for it in range(n_iterations)
            ]

            t0 = time.time()
            all_rows: list = []
            flush_interval = 20

            def _process(i: int, batch: list, _t0=t0, _rows=all_rows, _path=csv_path) -> None:
                _rows.extend(batch)
                if (i + 1) % flush_interval == 0:
                    elapsed = time.time() - _t0
                    eta = elapsed / (i + 1) * (n_iterations - i - 1)
                    print(
                        f"    {i + 1}/{n_iterations} iterations "
                        f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                        flush=True,
                    )
                    pd.DataFrame(_rows).to_csv(_path, index=False)

            if n_jobs == 1:
                for i, a in enumerate(task_args):
                    _process(i, _run_single_iteration(a))
            else:
                # 'spawn', never 'fork': a forked worker inherits R/BLAS state and
                # produced demonstrably wrong results (edgeR FPR 0.002 in-pipeline
                # versus 0.05 standalone).
                ctx = mp.get_context("spawn")
                with ctx.Pool(n_jobs) as pool:
                    for i, batch in enumerate(pool.imap(_run_single_iteration, task_args)):
                        _process(i, batch)

            df = pd.DataFrame(all_rows)
            df.to_csv(csv_path, index=False)
            all_results.append(df)
            print(f"    Done in {time.time() - t0:.0f}s -> {csv_path.name}")

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = output_dir / combined_name
    combined.to_csv(combined_path, index=False)
    print(f"\nAll results saved -> {combined_path}")
    print(f"Total rows: {len(combined):,}")
    return combined


def run_benchmark(
    designs: list[str] | None = None,
    methods: list[str] | None = None,
    n_iterations: int = _N_ITERATIONS,
    n_jobs: int = 1,
    output_dir: str | Path = "benchmark_results",
    resume: bool = True,
    base_config: dict | None = None,
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
    )


def run_sensitivity_benchmark(
    designs: list[str] | None = None,
    methods: list[str] | None = None,
    n_iterations: int = _N_ITERATIONS,
    n_jobs: int = 1,
    output_dir: str | Path = "benchmark_results/sensitivity",
    resume: bool = True,
    base_config: dict | None = None,
    panels: list[int] | None = None,
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
