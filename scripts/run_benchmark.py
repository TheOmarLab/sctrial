#!/usr/bin/env python
"""
NatMeth Benchmark Runner
========================

Full benchmark execution script. Run phases independently or all together.

Usage examples::

    # Phase 0: Simulator validation gate (MUST pass before Phase 2)
    python scripts/run_benchmark.py --phase validate --n-jobs 1

    # Phase 2: Full simulation grid (heavy — days on 25 cores)
    python scripts/run_benchmark.py --phase simulate --n-jobs 25

    # Phase 3: Real-data permutation + subsampling
    python scripts/run_benchmark.py --phase realdata --n-jobs 25

    # Phase 4: Ablation (runs on simulation + real data)
    python scripts/run_benchmark.py --phase ablation --n-jobs 4

    # All phases sequentially
    python scripts/run_benchmark.py --phase all --n-jobs 25
"""
import argparse
import sys
import warnings
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Output under project root, not relative to file path (which can resolve
# incorrectly on HPC). Use script's parent.parent = project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "benchmark"


def phase_validate(n_jobs: int):
    """Phase 0: simulator calibration and validation.

    Moved out of this script. Calibration is now a separate, canonical entry
    point because it must run ONCE, on a compute node, and produce the frozen
    configuration this benchmark then consumes. Keeping a second calibration
    path here is exactly how the shipped benchmark ended up running on
    uncalibrated defaults while the Methods described calibrated ones.
    """
    raise SystemExit(
        "Phase `validate` has moved to scripts/calibrate_simulator.py:\n"
        "  python scripts/calibrate_simulator.py targets --dataset tnbc\n"
        "  python scripts/calibrate_simulator.py gates --n-mc 200 --n-jobs 16\n"
        "  python scripts/calibrate_simulator.py ablate\n"
        "  python scripts/calibrate_simulator.py freeze\n"
        "Run those under sbatch, never on a login node."
    )



def _load_frozen_config() -> dict:
    """The one configuration every benchmark phase must use.

    Refuses to proceed without it. A missing calibration used to fall through to
    ``SimulationConfig`` defaults, which produced 2.3e7 UMIs per cell against a
    TNBC median of 2,113 while the Methods claimed calibrated parameters.
    """
    import json

    path = OUTPUT_DIR / "validation" / "frozen_simulator_config.json"
    if not path.exists():
        raise SystemExit(
            f"No frozen simulator configuration at {path}.\n"
            "Run scripts/calibrate_simulator.py (targets -> gates -> freeze) first. "
            "There is deliberately no default fallback."
        )
    with open(path) as fh:
        frozen = json.load(fh)["config"]
    frozen.pop("seed", None)
    frozen.pop("effects", None)
    return frozen


def phase_simulate(n_jobs: int, n_iterations: int, designs=None):
    """Phase 2: Full simulation benchmark grid.

    ``designs`` is exposed so the two design families can run as separate jobs.
    Each iteration now simulates a full 20,284-gene transcriptome rather than
    only the tested panel, which is far more expensive; both families in one job
    does not fit a 72-hour wall clock. Splitting also means a timeout costs one
    family rather than the whole grid, and `resume` picks up complete scenarios.
    """
    from sctrial.benchmark.orchestrator import run_benchmark

    designs = designs or ["two_arm", "single_arm"]
    out_dir = OUTPUT_DIR / "simulation"
    print("=" * 60)
    print("PHASE 2: Simulation Benchmark")
    print(f"  {n_iterations} iterations × 2 designs × ~30 scenarios × 6 methods")
    print(f"  Workers: {n_jobs}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    run_benchmark(
        designs=designs,
        n_iterations=n_iterations,
        n_jobs=n_jobs,
        output_dir=out_dir,
        resume=True,
    )


def phase_realdata(n_jobs: int):
    """Phase 3: Real-data permutation + subsampling."""
    print("=" * 60)
    print("PHASE 3: Real-Data Benchmark")
    print("=" * 60)

    out_dir = OUTPUT_DIR / "realdata"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Permutation tests
    from sctrial.benchmark.permutation import run_permutation_test

    print("\n--- Melanoma permutation (1000×) ---")
    from sctrial.datasets import load_sade_feldman
    sf = load_sade_feldman(
        processed_name="sade_feldman_processed_v6.h5ad",
    )
    # Use a subset of genes for tractability
    gene_cols = sf.var_names[:50].tolist()
    run_permutation_test(
        sf, gene_cols,
        design_type="two_arm",
        n_permutations=1000,
        n_jobs=n_jobs,
        participant_col="participant_id",
        arm_col="response",
        visit_col="visit",
        output_path=out_dir / "permutation_melanoma.csv",
    )

    # Subsampling
    from sctrial.benchmark.subsample import run_subsampling

    print("\n--- Melanoma subsampling (100×) ---")
    run_subsampling(
        sf, gene_cols,
        n_resamples=100,
        participant_col="participant_id",
        arm_col="response",
        visit_col="visit",
        output_path=out_dir / "subsampling_melanoma.csv",
    )


def phase_sensitivity(n_jobs: int, n_iterations: int, designs=None):
    """Phase 5: Signal-fraction sensitivity benchmark.

    Tests how null-gene FPR depends on gene-panel size (50-2000) and
    signal fraction (1-20%). Answers the key reviewer question: does
    dreamlet inflation attenuate with larger, more realistic panels?

    Grid: 4 panel sizes × (4 signal fractions + 1 null) = 20 scenarios
    per design, × 200 iterations × 4 methods.
    """
    from sctrial.benchmark.orchestrator import run_sensitivity_benchmark

    out_dir = OUTPUT_DIR / "sensitivity"
    print("=" * 60)
    print("PHASE 5: Signal-Fraction Sensitivity Benchmark")
    print(f"  {n_iterations} iterations × 20 scenarios × 4 methods")
    print("  Panel sizes: 50, 200, 500, 2000 genes")
    print("  Signal fractions: 1%, 5%, 10%, 20% + pure null")
    print(f"  Workers: {n_jobs}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    run_sensitivity_benchmark(
        designs=designs or ["two_arm"],
        n_iterations=n_iterations,
        n_jobs=n_jobs,
        output_dir=out_dir,
        resume=True,
    )


def phase_ablation(n_jobs: int):
    """Phase 4: progressive-component ablation (the pseudoreplication ladder).

    Runs on the SAME frozen simulator configuration as the main grid. Reading
    calibration from a file that may or may not exist, with a hardcoded fallback,
    is how the published ablation silently ran on different parameters from the
    benchmark it was supposed to support; there is no fallback here.
    """
    import numpy as np
    import pandas as pd

    from sctrial.benchmark.ablation import run_ablation
    from sctrial.benchmark.contracts import prepare_inputs
    from sctrial.benchmark.metrics import summarize_iteration
    from sctrial.benchmark.simulator_v2 import (
        TranscriptomeSimConfig,
        make_signal,
        nested_panels,
        simulate_trial_v2,
    )

    print("=" * 60)
    print("PHASE 4: Ablation Study")
    print("=" * 60)

    out_dir = OUTPUT_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = _load_frozen_config()

    all_rows = []
    for scenario, frac in [("null", 0.0), ("signal", 0.20)]:
        print(f"\n--- Ablation: {scenario} (signal fraction {frac:.0%}) ---")
        for it in range(100):
            seed = 42 + it
            kw = dict(frozen)
            kw.update(n_per_arm=40, cells_per_pv_fixed=500, seed=seed)
            probe = TranscriptomeSimConfig(**kw)
            panels = nested_panels(probe, rng=np.random.default_rng(seed + 1))
            panel = [f"gene_{i}" for i in panels[50]]
            effects = (
                make_signal(panel, frac, "balanced", 0.5, rng=np.random.default_rng(seed + 2))
                if frac > 0
                else {}
            )
            sim = simulate_trial_v2(TranscriptomeSimConfig(effects=effects, **kw))
            inputs = prepare_inputs(sim, panel)

            # Every ablation rung analyses log(1+CPM), so the truth it is scored
            # against is the log1p_cpm oracle, not the injected beta. Those differ
            # for low-expression genes at realistic depth.
            truth = sim["oracle"]["log1p_cpm"]
            results = run_ablation(inputs, panel)
            for var_name, gene_results in results.items():
                metrics = summarize_iteration(gene_results, truth, set(effects))
                metrics["variant"] = var_name
                metrics["scenario"] = scenario
                metrics["iteration"] = it
                all_rows.append(metrics)

            if (it + 1) % 20 == 0:
                print(f"  {it + 1}/100 iterations done")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    print(f"  Saved -> {out_dir / 'ablation_results.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="NatMeth Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["validate", "simulate", "sensitivity", "realdata", "ablation", "all"],
        required=True,
        help="Which phase to run",
    )
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel workers (-1 = all cores)")
    parser.add_argument("--n-iterations", type=int, default=200,
                        help="Monte Carlo iterations per scenario")
    parser.add_argument("--designs", nargs="+", default=None,
                        help="Design families to run; split across jobs for wall-clock")

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ("validate", "all"):
        phase_validate(args.n_jobs)

    if args.phase in ("simulate", "all"):
        phase_simulate(args.n_jobs, args.n_iterations, args.designs)

    if args.phase in ("sensitivity", "all"):
        phase_sensitivity(args.n_jobs, args.n_iterations, args.designs)

    if args.phase in ("realdata", "all"):
        phase_realdata(args.n_jobs)

    if args.phase in ("ablation", "all"):
        phase_ablation(args.n_jobs)


if __name__ == "__main__":
    main()
