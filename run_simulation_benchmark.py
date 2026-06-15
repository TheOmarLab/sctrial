from pathlib import Path
if __name__ == '__main__':
    OUTPUT_DIR = Path("/common/omarmlab/members/itzel/sctrial_bench/sctrial/temp/simulation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ── 1. Main simulation benchmark ──────────────
    print("==========================================")
    print("STEP 1: Main simulation benchmark")
    print("==========================================")
    from sctrial.benchmark.orchestrator import run_benchmark
    main_df = run_benchmark(
        n_jobs=25,
        output_dir=OUTPUT_DIR,
        resume=True,
    )
    print(f"Main benchmark done: {len(main_df):,} rows")
    # ── 2. Sensitivity benchmark ──────────────────
    print("\n==========================================")
    print("STEP 2: Sensitivity benchmark")
    print("==========================================")
    from sctrial.benchmark.orchestrator import run_sensitivity_benchmark
    sens_df = run_sensitivity_benchmark(
        n_jobs=8,
        output_dir=OUTPUT_DIR,
        resume=True,
    )
    print(f"Sensitivity benchmark done: {len(sens_df):,} rows")
    print("\n==========================================")
    print("ALL DONE")
    print(f"  Main CSV        → {OUTPUT_DIR}/main/benchmark_combined.csv")
    print(f"  Sensitivity CSV → {OUTPUT_DIR}/sensitivity/sensitivity_combined.csv")
    print("==========================================")
