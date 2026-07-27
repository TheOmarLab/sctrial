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
    """Phase 0: Simulator validation gate.

    Calibrates the simulator from TNBC (raw counts), validates on
    vaccine (holdout), and generates descriptive comparison for melanoma
    (no raw counts). MUST pass before Phase 2.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from sctrial.benchmark.simulator import (
        SimulationConfig,
        calibrate_from_real_data,
        validate_simulator,
    )

    out_dir = OUTPUT_DIR / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load real datasets
    print("=" * 60)
    print("PHASE 0: Simulator Validation Gate")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Calibration / Validation / Descriptive split (locked rule):
    #
    #   TNBC (GSE169246)     = CALIBRATION source (raw UMI counts)
    #   Vaccine (GSE171964)  = HOLDOUT validation (raw counts, NOT used for param estimation)
    #   Melanoma (Sade-Feld) = DESCRIPTIVE only (no raw counts, transformed-scale check)
    #
    # Simulator parameters are estimated from TNBC ONLY.
    # Vaccine tests generalization. Melanoma is real-data benchmarks only.
    # ---------------------------------------------------------------
    datasets = {}

    # 1. CALIBRATION SOURCE: TNBC (two-arm, raw UMI counts)
    print("\nLoading TNBC (CALIBRATION SOURCE, raw counts)...")
    from sctrial.datasets import load_tnbc_zhang
    tnbc = load_tnbc_zhang()
    datasets["tnbc"] = {
        "adata": tnbc,
        "layer": None,                # .X is normalized
        "count_layer": "counts",       # raw integer UMI counts
        "participant_col": "participant_id", "visit_col": "visit",
        "arm_col": "arm",
        "design": "two_arm",
        "role": "CALIBRATION",
    }

    # 2. HOLDOUT VALIDATION: Vaccine (single-arm, raw counts)
    print("Loading Vaccine (HOLDOUT VALIDATION, raw counts)...")
    from sctrial.datasets import load_vaccine_gse171964
    vax = load_vaccine_gse171964()
    if "pt_id" in vax.obs.columns:
        vax.obs["participant_id"] = vax.obs["pt_id"]
    if "day" in vax.obs.columns:
        vax.obs["visit"] = vax.obs["day"].map({0: "Pre", 7: "Post"})
    datasets["vaccine"] = {
        "adata": vax,
        "layer": None,                # .X is raw counts
        "count_layer": None,           # auto-detect (.X is integer counts)
        "participant_col": "participant_id", "visit_col": "visit",
        "arm_col": "arm" if "arm" in vax.obs.columns else None,
        "design": "single_arm",
        "role": "HOLDOUT VALIDATION",
    }

    # 3. DESCRIPTIVE ONLY: Melanoma (no raw counts)
    print("Loading Sade-Feldman (DESCRIPTIVE ONLY, no raw counts)...")
    from sctrial.datasets import load_sade_feldman
    sf = load_sade_feldman(
        max_cells_per_participant_visit=None,
        processed_name="sade_feldman_processed_v6.h5ad",
    )
    datasets["melanoma"] = {
        "adata": sf,
        "layer": "log1p_tpm",
        "count_layer": None,          # NO raw counts available
        "participant_col": "participant_id", "visit_col": "visit",
        "arm_col": "response",
        "design": "two_arm",
        "role": "DESCRIPTIVE (transformed-scale, no raw counts)",
    }

    import json

    # ---- Step 1: Calibrate from TNBC ONLY ----
    print("\n" + "=" * 60)
    print("Step 1: Calibrate simulator from TNBC (calibration source)")
    print("=" * 60)
    d_cal = datasets["tnbc"]
    calibration_params = calibrate_from_real_data(
        d_cal["adata"],
        layer=d_cal["layer"],
        count_layer=d_cal.get("count_layer"),
        participant_col=d_cal["participant_col"],
        visit_col=d_cal["visit_col"],
    )
    for k, v in calibration_params.items():
        if isinstance(v, bool):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.3f}")
    with open(out_dir / "calibration_tnbc.json", "w") as f:
        json.dump(calibration_params, f, indent=2)

    # ---- Step 2: Validate against ALL datasets with correct labels ----
    print("\n" + "=" * 60)
    print("Step 2: Validate simulator against all datasets")
    print("=" * 60)

    for name, d in datasets.items():
        role = d["role"]
        print(f"\n--- {name} [{role}] ---")

        # For non-calibration datasets, still compute their params for comparison
        params = calibrate_from_real_data(
            d["adata"],
            layer=d["layer"],
            count_layer=d.get("count_layer"),
            participant_col=d["participant_col"],
            visit_col=d["visit_col"],
        )
        for k, v in params.items():
            if isinstance(v, bool):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.3f}")

        # Save per-dataset descriptive params (for reference, not for calibration)
        with open(out_dir / f"params_{name}.json", "w") as f:
            json.dump(params, f, indent=2)

        # Generate simulated data using TNBC CALIBRATION PARAMS (not per-dataset)
        # Only adjust n_per_arm and design to match the real dataset's structure
        is_two_arm = d["design"] == "two_arm"
        n_participants = d["adata"].obs[d["participant_col"]].nunique()
        n_per_arm = n_participants // 2 if is_two_arm else n_participants
        cfg = SimulationConfig(
            design=d["design"],
            n_per_arm=n_per_arm,
            n_genes=min(50, d["adata"].n_vars),
            # ALL distributional params from TNBC calibration (not per-dataset)
            mean_cells_per_visit=int(calibration_params["mean_cells_per_visit"]),
            participant_sd=max(0.1, np.sqrt(calibration_params["participant_icc"])),
            baseline_mean=calibration_params["baseline_mean"],
            baseline_sd=calibration_params["baseline_sd"],
            target_library_size=int(np.exp(calibration_params["library_size_mean"])),
            library_size_sd=calibration_params["library_size_sd"],
            seed=42,
        )

        validation = validate_simulator(
            cfg, d["adata"], layer=d["layer"],
            participant_col=d["participant_col"],
            visit_col=d["visit_col"],
        )

        # Plot validation figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        scale_label = "count-scale" if params["has_raw_counts"] else "transformed-scale"
        fig.suptitle(
            f"Simulator Validation: {name}\n"
            f"[{role} — {scale_label} comparison — calibrated from TNBC]",
            fontsize=12, fontweight="bold",
        )

        cl = validation["cell_level"]
        pl = validation["pseudobulk_level"]

        # 1. Mean-variance
        ax = axes[0, 0]
        ax.scatter(np.log10(cl["real_gene_means"] + 1),
                   np.log10(cl["real_gene_vars"] + 1),
                   s=10, alpha=0.5, label="Real", color="C0")
        ax.scatter(np.log10(cl["sim_gene_means"] + 1),
                   np.log10(cl["sim_gene_vars"] + 1),
                   s=10, alpha=0.5, label="Simulated", color="C1")
        ax.set_xlabel("log10(mean + 1)")
        ax.set_ylabel("log10(variance + 1)")
        ax.set_title("Mean-Variance")
        ax.legend(fontsize=8)

        # 2. Zero fraction
        ax = axes[0, 1]
        ax.hist(cl["real_zero_fraction"], bins=30, alpha=0.5, label="Real", color="C0")
        ax.hist(cl["sim_zero_fraction"], bins=30, alpha=0.5, label="Simulated", color="C1")
        ax.set_xlabel("Zero fraction per gene")
        ax.set_title("Zero Fraction")
        ax.legend(fontsize=8)

        # 3. Library size
        ax = axes[0, 2]
        ax.hist(np.log10(cl["real_library_sizes"] + 1), bins=50,
                alpha=0.5, label="Real", color="C0", density=True)
        ax.hist(np.log10(cl["sim_library_sizes"] + 1), bins=50,
                alpha=0.5, label="Simulated", color="C1", density=True)
        ax.set_xlabel("log10(library size)")
        ax.set_title("Library Size Distribution")
        ax.legend(fontsize=8)

        # 4. Cell counts per participant-visit
        ax = axes[1, 0]
        ax.hist(pl["real_cell_counts"], bins=30, alpha=0.5,
                label="Real", color="C0", density=True)
        ax.hist(pl["sim_cell_counts"], bins=30, alpha=0.5,
                label="Simulated", color="C1", density=True)
        ax.set_xlabel("Cells per participant-visit")
        ax.set_title("Cell Count Distribution")
        ax.legend(fontsize=8)

        # 5. Pseudobulk means
        ax = axes[1, 1]
        ax.hist(np.log10(pl["sim_pb_means"] + 1), bins=50,
                alpha=0.7, color="C1", density=True)
        ax.set_xlabel("log10(pseudobulk mean + 1)")
        ax.set_title("Pseudobulk Mean Distribution (sim)")

        # 6. Summary text
        ax = axes[1, 2]
        ax.axis("off")
        text = (
            f"Dataset: {name}\n"
            f"Role: {role}\n"
            f"Raw counts: {'YES' if params['has_raw_counts'] else 'NO'}\n"
            f"Real cells: {d['adata'].n_obs:,}\n"
            f"Real genes: {d['adata'].n_vars:,}\n"
            f"Participants: {d['adata'].obs[d['participant_col']].nunique()}\n"
            f"\nSimulator params (from TNBC):\n"
            f"  mean_cells: {calibration_params['mean_cells_per_visit']:.0f}\n"
            f"  ICC: {calibration_params['participant_icc']:.3f}\n"
            f"  dispersion: {calibration_params['dispersion_median']:.1f}\n"
            f"  baseline: {calibration_params['baseline_mean']:.2f}\n"
            f"  lib_size: {calibration_params['library_size_mean']:.2f}\n"
        )
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace")

        fig.tight_layout()
        fig_path = out_dir / f"validation_{name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fig_path}")

    print("\n" + "=" * 60)
    print("VALIDATION GATE — Review figures in:")
    print(f"  {out_dir}/")
    print()
    print("  validation_tnbc.png    → CALIBRATION source (must match well)")
    print("  validation_vaccine.png → HOLDOUT validation (tests generalization)")
    print("  validation_melanoma.png → DESCRIPTIVE only (transformed-scale)")
    print()
    print("Key check: TNBC should match closely (calibrated on it).")
    print("Vaccine should match reasonably (holdout — the real test).")
    print("Melanoma is descriptive — no count-scale claims.")
    print()
    print("If vaccine validation looks reasonable, proceed to Phase 2.")
    print("=" * 60)


def _ensure_calibration(n_jobs: int) -> None:
    """Run phase_validate if calibration_tnbc.json is missing."""
    cal_path = OUTPUT_DIR / "validation" / "calibration_tnbc.json"
    if not cal_path.exists():
        print("calibration_tnbc.json not found — running phase_validate first...")
        try:
            phase_validate(n_jobs)
        except Exception as exc:
            print(f"  WARNING: phase_validate failed ({exc}) — falling back to hardcoded params")


def phase_simulate(n_jobs: int, n_iterations: int):
    """Phase 2: Full simulation benchmark grid."""
    _ensure_calibration(n_jobs)
    from sctrial.benchmark.orchestrator import run_benchmark

    out_dir = OUTPUT_DIR / "simulation"
    print("=" * 60)
    print("PHASE 2: Simulation Benchmark")
    print(f"  {n_iterations} iterations × 2 designs × ~30 scenarios × 6 methods")
    print(f"  Workers: {n_jobs}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    run_benchmark(
        designs=["two_arm", "single_arm"],
        n_iterations=n_iterations,
        n_jobs=n_jobs,
        output_dir=out_dir,
        resume=True,
        calibration_json=OUTPUT_DIR / "validation" / "calibration_tnbc.json",
    )


def phase_realdata(n_jobs: int):
    """Phase 3: Real-data permutation + subsampling on TNBC.

    TNBC (Zhang et al.) is used because it has raw counts in
    adata.layers["counts"], which all four core methods require.
    Melanoma (Sade-Feldman) is excluded: it has no raw counts
    (adata.X is log1p-TPM), so dreamlet and NEBULA cannot run on it.
    """
    print("=" * 60)
    print("PHASE 3: Real-Data Benchmark (TNBC)")
    print("=" * 60)

    out_dir = OUTPUT_DIR / "realdata"
    out_dir.mkdir(parents=True, exist_ok=True)

    from sctrial.benchmark.permutation import run_permutation_test
    from sctrial.benchmark.subsample import run_subsampling
    from sctrial.datasets import load_tnbc_zhang

    print("\n--- TNBC (Zhang et al.) ---")
    tnbc = load_tnbc_zhang()
    gene_cols_tnbc = tnbc.var_names[:50].tolist()

    print(f"  Permutation (1000×, {len(gene_cols_tnbc)} genes) ...")
    run_permutation_test(
        tnbc, gene_cols_tnbc,
        design_type="two_arm",
        n_permutations=1000,
        n_jobs=n_jobs,
        participant_col="participant_id",
        arm_col="arm",
        visit_col="visit",
        output_path=out_dir / "permutation_tnbc.csv",
    )

    print(f"  Subsampling (100×) ...")
    run_subsampling(
        tnbc, gene_cols_tnbc,
        n_resamples=100,
        n_jobs=n_jobs,
        participant_col="participant_id",
        arm_col="arm",
        visit_col="visit",
        output_path=out_dir / "subsampling_tnbc.csv",
    )
    del tnbc


def phase_sensitivity(n_jobs: int, n_iterations: int):
    """Phase 5: Signal-fraction sensitivity benchmark.

    Tests how null-gene FPR depends on gene-panel size (50-2000) and
    signal fraction (1-20%). Answers the key reviewer question: does
    dreamlet inflation attenuate with larger, more realistic panels?

    Grid: 4 panel sizes × (4 signal fractions + 1 null) = 20 scenarios
    per design, × 200 iterations × 4 methods.
    """
    _ensure_calibration(n_jobs)
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
        designs=["two_arm"],
        n_iterations=n_iterations,
        n_jobs=n_jobs,
        output_dir=out_dir,
        resume=True,
        calibration_json=OUTPUT_DIR / "validation" / "calibration_tnbc.json",
    )


def phase_ablation(n_jobs: int):
    """Phase 4: Ablation study.

    Uses TNBC-calibrated simulator params (same family as Phase 2)
    to ensure ablation results support the same manuscript claims.
    """

    _ensure_calibration(n_jobs)
    import pandas as pd

    from sctrial.benchmark.ablation import run_ablation
    from sctrial.benchmark.metrics import summarize_iteration
    from sctrial.benchmark.simulator import SimulationConfig, simulate_trial

    print("=" * 60)
    print("PHASE 4: Ablation Study")
    print("=" * 60)

    out_dir = OUTPUT_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    from sctrial.benchmark.orchestrator import _load_calibration

    cal = _load_calibration(OUTPUT_DIR / "validation" / "calibration_tnbc.json")

    # Run ablation on 100 simulated null + 100 simulated signal datasets
    all_rows = []
    for scenario, beta in [("null", 0.0), ("signal", 0.5)]:
        print(f"\n--- Ablation: {scenario} (beta={beta}) ---")
        for it in range(100):
            cfg = SimulationConfig(
                n_per_arm=40, n_genes=50,
                effects={f"gene_{i}": beta for i in range(10)} if beta > 0 else {},
                **cal,
                seed=42 + it,
            )
            sim = simulate_trial(cfg)
            gene_cols = [f"gene_{i}" for i in range(50)]
            signal_genes = set(cfg.effects.keys())

            results = run_ablation(sim, gene_cols)

            for var_name, gene_results in results.items():
                metrics = summarize_iteration(gene_results, sim["truth"], signal_genes)
                metrics["variant"] = var_name
                metrics["scenario"] = scenario
                metrics["iteration"] = it
                all_rows.append(metrics)

            if (it + 1) % 20 == 0:
                print(f"  {it+1}/100 iterations done")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    print(f"  Saved → {out_dir / 'ablation_results.csv'}")


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
                        help="Monte Carlo iterations (Phase 2 only)")

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ("validate", "all"):
        phase_validate(args.n_jobs)

    if args.phase in ("simulate", "all"):
        phase_simulate(args.n_jobs, args.n_iterations)

    if args.phase in ("sensitivity", "all"):
        phase_sensitivity(args.n_jobs, args.n_iterations)

    if args.phase in ("realdata", "all"):
        phase_realdata(args.n_jobs)

    if args.phase in ("ablation", "all"):
        phase_ablation(args.n_jobs)


if __name__ == "__main__":
    main()
