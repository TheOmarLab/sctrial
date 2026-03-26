"""Benchmark orchestrator — runs the full simulation grid with parallelization.

Usage::

    from sctrial.benchmark.orchestrator import run_benchmark
    results = run_benchmark(n_jobs=25, output_dir="benchmark_results")

Or from command line::

    python -m sctrial.benchmark.orchestrator --n-jobs 25 --output-dir benchmark_results
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .simulator import SimulationConfig, simulate_trial

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario definitions (from the locked NatMeth benchmark plan)
# ---------------------------------------------------------------------------

_N_GENES = 50
_N_SIGNAL = 10
_N_ITERATIONS = 200
_MEAN_CELLS = 500

# Core methods
CORE_METHODS = [
    "sctrial_fe",
    "edger_qlf",
    "limma_voom",
    "dreamlet",
    "nebula",
    "wilcoxon_paired",
]

# Internal sensitivity (not headline)
INTERNAL_METHODS = ["sctrial_mixed"]


def _make_effects(n_signal: int, beta: float, mixed_sign: bool = False) -> dict:
    """Create effect dict for n_signal genes."""
    effects = {}
    for i in range(n_signal):
        if mixed_sign and i % 2 == 1:
            effects[f"gene_{i}"] = -beta
        else:
            effects[f"gene_{i}"] = beta
    return effects


def build_scenario_grid(design: str = "two_arm") -> list[dict]:
    """Build the full scenario grid for one design family.

    Returns list of dicts, each with keys: name, config_kwargs, description.
    """
    scenarios = []

    # 1. Complete null
    for n in [8, 12, 20, 40, 60]:
        scenarios.append(
            {
                "name": f"null_n{n}",
                "description": f"Complete null, n={n} per arm",
                "config_kwargs": {
                    "design": design,
                    "n_per_arm": n,
                    "n_genes": _N_GENES,
                    "effects": {},
                    "mean_cells_per_visit": _MEAN_CELLS,
                },
            }
        )

    # 2. Null + nuisance heterogeneity
    for n in [20, 40]:
        scenarios.append(
            {
                "name": f"null_hetero_n{n}",
                "description": f"Null + high participant SD + imbalanced cells, n={n}",
                "config_kwargs": {
                    "design": design,
                    "n_per_arm": n,
                    "n_genes": _N_GENES,
                    "effects": {},
                    "mean_cells_per_visit": _MEAN_CELLS,
                    "participant_sd": 0.8,
                    "cell_count_mode": "lognormal",
                    "cell_count_cv": 0.8,
                },
            }
        )

    # 3. Sparse DE (positive)
    for n in [20, 40, 60]:
        for beta in [0.2, 0.5, 1.0]:
            scenarios.append(
                {
                    "name": f"de_pos_n{n}_b{beta}",
                    "description": f"Sparse DE (positive), n={n}, beta={beta}",
                    "config_kwargs": {
                        "design": design,
                        "n_per_arm": n,
                        "n_genes": _N_GENES,
                        "effects": _make_effects(_N_SIGNAL, beta, mixed_sign=False),
                        "mean_cells_per_visit": _MEAN_CELLS,
                    },
                }
            )

    # 4. Sparse DE (mixed sign)
    for n in [20, 40]:
        for beta in [0.5, 1.0]:
            scenarios.append(
                {
                    "name": f"de_mixed_n{n}_b{beta}",
                    "description": f"Sparse DE (mixed sign), n={n}, beta={beta}",
                    "config_kwargs": {
                        "design": design,
                        "n_per_arm": n,
                        "n_genes": _N_GENES,
                        "effects": _make_effects(_N_SIGNAL, beta, mixed_sign=True),
                        "mean_cells_per_visit": _MEAN_CELLS,
                    },
                }
            )

    # 5. Varying cells
    for n_cells in [200, 1000, 5000]:
        scenarios.append(
            {
                "name": f"cells_{n_cells}_n40",
                "description": f"Varying cells ({n_cells}/visit), n=40, beta=0.5",
                "config_kwargs": {
                    "design": design,
                    "n_per_arm": 40,
                    "n_genes": _N_GENES,
                    "effects": _make_effects(_N_SIGNAL, 0.5),
                    "mean_cells_per_visit": n_cells,
                },
            }
        )

    # 6. Unequal arms (two-arm only)
    if design == "two_arm":
        for ratio in [(3, 7), (5, 10), (10, 20)]:
            scenarios.append(
                {
                    "name": f"imbal_{ratio[0]}v{ratio[1]}",
                    "description": f"Unequal arms {ratio[0]}:{ratio[1]}, beta=0.5",
                    "config_kwargs": {
                        "design": design,
                        "n_per_arm": sum(ratio),
                        "n_genes": _N_GENES,
                        "effects": _make_effects(_N_SIGNAL, 0.5),
                        "mean_cells_per_visit": _MEAN_CELLS,
                        "arm_ratio": ratio,
                    },
                }
            )

    # 7. Missing visits
    for rate in [0.1, 0.2]:
        scenarios.append(
            {
                "name": f"missing_{int(rate * 100)}pct_n40",
                "description": f"Missing {int(rate * 100)}% post visits, n=40, beta=0.5",
                "config_kwargs": {
                    "design": design,
                    "n_per_arm": 40,
                    "n_genes": _N_GENES,
                    "effects": _make_effects(_N_SIGNAL, 0.5),
                    "mean_cells_per_visit": _MEAN_CELLS,
                    "missing_rate": rate,
                },
            }
        )

    return scenarios


# ---------------------------------------------------------------------------
# Single-iteration worker (for multiprocessing)
# ---------------------------------------------------------------------------


def _run_single_iteration(args: tuple) -> list[dict]:
    """Run all methods on one simulated dataset. Called in worker processes."""
    import warnings

    warnings.filterwarnings("ignore")

    scenario_name, iteration, seed, config_kwargs, methods = args

    cfg = SimulationConfig(seed=seed, **config_kwargs)
    design_type = cfg.design
    sim = simulate_trial(cfg)
    gene_cols = [f"gene_{i}" for i in range(cfg.n_genes)]
    signal_genes = set(cfg.effects.keys())

    rows = []
    for method in methods:
        t0 = time.time()
        try:
            results = _dispatch_method(method, sim, gene_cols, design_type=design_type)
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
                for g in gene_cols
            }
        elapsed = time.time() - t0

        for gene in gene_cols:
            r = results.get(gene, {})
            rows.append(
                {
                    "scenario": scenario_name,
                    "iteration": iteration,
                    "method": method,
                    "gene": gene,
                    "true_beta": cfg.effects.get(gene, 0.0),
                    "is_signal": gene in signal_genes,
                    "estimated_beta": r.get("beta", np.nan),
                    "pvalue": r.get("pvalue", np.nan),
                    "ci_lo": r.get("ci_lo", np.nan),
                    "ci_hi": r.get("ci_hi", np.nan),
                    "converged": r.get("converged", False),
                    "failure_mode": r.get("failure_mode", "numerical"),
                    "runtime_seconds": elapsed / len(gene_cols),
                    "n_per_arm": cfg.n_per_arm,
                    "mean_cells": cfg.mean_cells_per_visit,
                }
            )

    return rows


def _dispatch_method(
    method: str,
    sim: dict,
    gene_cols: list[str],
    design_type: str = "two_arm",
) -> dict:
    """Route to the correct runner with the appropriate data representation.

    - sctrial, NEBULA: cell-level AnnData
    - edgeR, limma-voom, dreamlet: summed-count pseudobulk (true counts)
    - Wilcoxon (paired delta): mean-expression pseudobulk

    Parameters
    ----------
    design_type : str
        "two_arm" or "single_arm". Determines the statistical model used
        by R-based runners (interaction vs paired visit).
    """
    # Backwards compat: old sim dicts may have "pseudobulk" instead of split keys
    pb_counts = sim.get("pseudobulk_counts", sim.get("pseudobulk"))
    pb_means = sim.get("pseudobulk_means", sim.get("pseudobulk"))

    # Log-transformed pseudobulk means for methods that need expression-scale
    # input (sctrial_fe, wilcoxon_paired). This ensures all methods report
    # betas on a comparable log-fold-change scale:
    #   - edgeR/limma/dreamlet internally log-transform raw counts
    #   - NEBULA fits a log-link NB model on raw counts
    #   - sctrial_fe and wilcoxon_paired need log-transformed input explicitly
    if pb_means is not None:
        pb_log = pb_means.copy()
        gene_mask = [c for c in gene_cols if c in pb_log.columns]
        pb_log[gene_mask] = np.log1p(pb_log[gene_mask])
    else:
        pb_log = None

    if method == "sctrial_fe":
        from .runners.sctrial_fe import run

        # Pass log-pseudobulk DataFrame instead of raw cell-level AnnData
        # so that DiD betas are on log-expression scale, comparable to
        # edgeR/limma/dreamlet log-fold-changes
        return run(pb_log, gene_cols, from_pseudobulk=True)
    elif method == "edger_qlf":
        from .runners.edger_qlf import run

        return run(pb_counts, gene_cols, design_type=design_type)
    elif method == "limma_voom":
        from .runners.limma_voom import run

        return run(pb_counts, gene_cols, design_type=design_type)
    elif method == "dreamlet":
        from .runners.dreamlet_runner import run

        return run(pb_counts, gene_cols, design_type=design_type)
    elif method == "nebula":
        from .runners.nebula_runner import run

        return run(sim["adata"], gene_cols, design_type=design_type)
    elif method == "wilcoxon_paired":
        from .runners.wilcoxon_paired import run

        return run(pb_log, gene_cols)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_benchmark(
    designs: list[str] | None = None,
    methods: list[str] | None = None,
    n_iterations: int = _N_ITERATIONS,
    n_jobs: int = 1,
    output_dir: str | Path = "benchmark_results",
    resume: bool = True,
) -> pd.DataFrame:
    """Run the full NatMeth benchmark grid.

    Parameters
    ----------
    designs : list of str
        Design families to run. Default: ["two_arm", "single_arm"].
    methods : list of str
        Methods to benchmark. Default: CORE_METHODS.
    n_iterations : int
        Monte Carlo iterations per scenario.
    n_jobs : int
        Parallel workers. Use -1 for all cores.
    output_dir : str or Path
        Directory for output CSVs and figures.
    resume : bool
        If True, skip scenarios that already have results in output_dir.

    Returns
    -------
    DataFrame with all results concatenated.
    """
    if designs is None:
        designs = ["two_arm", "single_arm"]
    if methods is None:
        methods = CORE_METHODS
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2024)
    all_results = []

    for design in designs:
        scenarios = build_scenario_grid(design)
        print(f"\n{'=' * 60}")
        print(f"Design: {design} — {len(scenarios)} scenarios × {n_iterations} iterations")
        print(f"Methods: {methods}")
        print(f"Parallel workers: {n_jobs}")
        print(f"{'=' * 60}")

        for si, scenario in enumerate(scenarios):
            name = f"{design}__{scenario['name']}"
            csv_path = output_dir / f"{name}.csv"

            # Resume support
            if resume and csv_path.exists():
                print(f"  [{si + 1}/{len(scenarios)}] {name} — CACHED, skipping")
                existing = pd.read_csv(csv_path)
                all_results.append(existing)
                continue

            print(f"  [{si + 1}/{len(scenarios)}] {name}: {scenario['description']}")

            # Pre-generate seeds
            seeds = [int(rng.integers(0, 2**31)) for _ in range(n_iterations)]

            # Build task args
            task_args = [
                (name, it, seeds[it], scenario["config_kwargs"], methods)
                for it in range(n_iterations)
            ]

            t0 = time.time()

            if n_jobs == 1:
                all_rows = []
                for i, args in enumerate(task_args):
                    all_rows.extend(_run_single_iteration(args))
                    if (i + 1) % 20 == 0:
                        print(
                            f"    {i + 1}/{n_iterations} iterations done ({time.time() - t0:.0f}s)"
                        )
            else:
                all_rows = []
                with mp.Pool(n_jobs) as pool:
                    for i, batch in enumerate(pool.imap(_run_single_iteration, task_args)):
                        all_rows.extend(batch)
                        if (i + 1) % 20 == 0:
                            elapsed = time.time() - t0
                            eta = elapsed / (i + 1) * (n_iterations - i - 1)
                            print(
                                f"    {i + 1}/{n_iterations} iterations "
                                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                            )

            elapsed = time.time() - t0
            df = pd.DataFrame(all_rows)
            df.to_csv(csv_path, index=False)
            all_results.append(df)
            print(f"    Done in {elapsed:.0f}s → {csv_path.name}")

    # Combine all
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = output_dir / "benchmark_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nAll results saved → {combined_path}")
    print(f"Total rows: {len(combined):,}")

    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NatMeth benchmark: simulation grid")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers (-1 = all cores)")
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=_N_ITERATIONS,
        help="Monte Carlo iterations per scenario",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results", help="Output directory"
    )
    parser.add_argument(
        "--designs", nargs="+", default=["two_arm", "single_arm"], help="Design families"
    )
    parser.add_argument(
        "--methods", nargs="+", default=None, help="Methods to run (default: all core)"
    )
    parser.add_argument("--no-resume", action="store_true", help="Don't skip existing results")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_benchmark(
        designs=args.designs,
        methods=args.methods,
        n_iterations=args.n_iterations,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
