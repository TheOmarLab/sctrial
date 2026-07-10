"""Rerun failed sensitivity scenarios: two_arm__sens_g2000_f10 and
two_arm__sens_g2000_f20.

This recreates the scenario configs independently (does not touch the
main orchestrator run) and saves to a separate folder so results can be
merged in later without disrupting the currently running job.
"""
from pathlib import Path

if __name__ == '__main__':
    from sctrial.benchmark.orchestrator import (
        build_sensitivity_grid,
        _run_scenario,
    )
    import numpy as np
    import pandas as pd

    OUTPUT_DIR = Path("/common/omarmlab/members/itzel/sctrial_bench/sctrial/temp/simulation/sensitivity_rerun")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = build_sensitivity_grid("two_arm")
    target_names = ["sens_g2000_f10", "sens_g2000_f20"]

    n_iterations = 200
    methods = ["sctrial_did", "dreamlet", "nebula", "wilcoxon_paired"]

    for target_name in target_names:
        scenario = next((s for s in scenarios if s["name"] == target_name), None)
        if scenario is None:
            print(f"WARNING: Scenario '{target_name}' not found in grid, skipping")
            continue

        print(f"\nRerunning scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")

        name = f"two_arm__{scenario['name']}"
        csv_path = OUTPUT_DIR / f"{name}.csv"

        rng = np.random.default_rng(2025)  # same seed base as run_sensitivity_benchmark
        seeds = [int(rng.integers(0, 2**31)) for _ in range(n_iterations)]
        task_args = [
            (name, it, seeds[it], scenario["config_kwargs"], methods)
            for it in range(n_iterations)
        ]

        all_rows = _run_scenario(task_args, n_iterations, n_jobs=25, csv_path=csv_path)

        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        print(f"Done → {csv_path}")
        print(f"Total rows: {len(df):,}")