#!/usr/bin/env python
"""Assert that the WHOLE manuscript benchmark is complete. The last job to run.

    sbatch --dependency=afterok:$A1:$A2 scripts/slurm_finalize_benchmark.sh

    5 producers -> 2 grid aggregators -> 1 publication finalizer

WHY A THIRD LAYER
-----------------
The grid aggregators each answer "is THIS grid complete". Neither answers "is the
manuscript benchmark complete", and the gap between those is a real failure mode:
the core aggregation succeeds, the sensitivity grid fails or never finishes, and a
figure script finds a perfectly valid `benchmark_complete_core.json` and
regenerates manuscript outputs from half the benchmark. Nothing would report an
error; the figures would simply be missing three quarters of the panel-size axis.

This is the third member of a class this project has already met twice:

* several producer jobs writing one combined file, last writer winning;
* two grids globbing one scenario directory, each seeing the other's shards;
* one valid grid being mistaken for the complete benchmark.

The first two were fixed structurally. This closes the third the same way, rather
than by remembering to check.

WHAT IT ASSERTS
---------------
Union of grid scenario sets equals the frozen expected set exactly; the grids are
disjoint; every grid has a valid completion marker; all rows and all markers carry
ONE manifest hash, matching the directory they live in. Only then is
`publication_complete.json` written, and the figure entry point requires that file
rather than any grid-level marker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

GRIDS = ("core", "sensitivity")
_COMBINED = {"core": "benchmark_combined.csv", "sensitivity": "sensitivity_combined.csv"}


def _expected(grid: str) -> set[str]:
    from sctrial.benchmark.orchestrator import build_scenario_grid, build_sensitivity_grid

    if grid == "core":
        return {
            f"{d}__{s['name']}"
            for d in ("two_arm", "single_arm")
            for s in build_scenario_grid(d)
        }
    return {f"two_arm__{s['name']}" for s in build_sensitivity_grid("two_arm")}


def finalize(layout, strict: bool = True) -> dict:
    problems: list[str] = []
    per_grid: dict[str, dict] = {}
    seen: dict[str, str] = {}
    manifests: set[str] = set()

    for grid in GRIDS:
        marker = layout.completion_marker(grid)
        if not marker.exists():
            problems.append(f"{grid}: no completion marker at {marker.name}")
            continue
        rec = json.loads(marker.read_text())
        per_grid[grid] = rec
        manifests.add(str(rec.get("manifest_sha256")))

        combined = layout.combined_csv(_COMBINED[grid])
        if not combined.exists():
            problems.append(f"{grid}: completion marker present but {combined.name} missing")
            continue
        df = pd.read_csv(combined, usecols=["scenario", "manifest_sha256"], low_memory=False)
        present = set(df["scenario"].unique())
        manifests |= set(df["manifest_sha256"].dropna().astype(str))

        want = _expected(grid)
        if present - want:
            problems.append(f"{grid}: UNEXPECTED scenarios {sorted(present - want)[:6]}")
        if want - present:
            problems.append(f"{grid}: MISSING scenarios {sorted(want - present)[:6]}")

        # Disjointness, checked across grids rather than assumed from the split.
        for name in present:
            if name in seen:
                problems.append(f"{name}: appears in both {seen[name]} and {grid}")
            seen[name] = grid

        if set(rec.get("scenarios") or []) != present:
            problems.append(
                f"{grid}: completion marker lists {len(rec.get('scenarios') or [])} scenarios "
                f"but the combined file holds {len(present)}"
            )

    # The union must be exactly the frozen expected set -- not a superset, not a
    # subset. Checking each grid separately does not establish this: two grids can
    # each be internally complete while the pair is not what the manuscript needs.
    expected_all = set().union(*(_expected(g) for g in GRIDS))
    if seen and set(seen) != expected_all:
        missing, extra = expected_all - set(seen), set(seen) - expected_all
        if missing:
            problems.append(f"union MISSING {len(missing)} scenario(s): {sorted(missing)[:8]}")
        if extra:
            problems.append(f"union has {len(extra)} UNEXPECTED: {sorted(extra)[:8]}")

    manifests.discard("None")
    manifests.discard("nan")
    if len(manifests) > 1:
        problems.append(f"grids carry {len(manifests)} manifests: {[m[:12] for m in manifests]}")
    elif manifests and not next(iter(manifests)).startswith(layout.manifest_sha[:12]):
        problems.append(
            f"rows carry manifest {next(iter(manifests))[:12]} under {layout.manifest_sha[:12]}"
        )

    if problems and strict:
        raise SystemExit(
            "BENCHMARK IS NOT COMPLETE -- refusing to write the publication marker:\n  - "
            + "\n  - ".join(problems)
        )

    reps = {}
    for grid, rec in per_grid.items():
        reps[grid] = {
            "n_scenarios": rec.get("n_scenarios"),
            "n_rows": rec.get("n_rows"),
            "replicates_min": rec.get("replicates_min"),
            "replicates_max": rec.get("replicates_max"),
            "stop_reasons": rec.get("stop_reasons"),
        }

    record = {
        "grids": list(per_grid),
        "n_scenarios_total": len(seen),
        "expected_total": len(expected_all),
        "manifest_sha256": next(iter(manifests)) if manifests else None,
        "per_grid": reps,
        "combined_files": {g: _COMBINED[g] for g in per_grid},
        "problems": problems,
    }
    layout.publication_marker().write_text(json.dumps(record, indent=2, default=str))
    return record


def main() -> None:
    from sctrial.benchmark.paths import ResultLayout

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest_sha", nargs="?", default=None)
    ap.add_argument("--results-root", default=None)
    args = ap.parse_args()

    base = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
    root = Path(base) if base else REPO.parent.parent / "manuscript"

    sha = args.manifest_sha
    if not sha:
        frozen = root / "benchmark" / "validation" / "frozen_simulator_config.json"
        m = json.loads(frozen.read_text()).get("manifest") or {}
        sha = m.get("manifest_sha256") or m.get("config_sha256")
        if not sha:
            raise SystemExit(f"{frozen} carries no manifest hash")

    results_root = (
        Path(args.results_root) if args.results_root else root / "benchmark" / "results"
    )
    layout = ResultLayout(results_root, sha)
    rec = finalize(layout)
    print("BENCHMARK COMPLETE")
    print(f"  manifest   {rec['manifest_sha256'][:16]}")
    print(f"  scenarios  {rec['n_scenarios_total']} of {rec['expected_total']} expected")
    for grid, r in rec["per_grid"].items():
        print(f"  {grid:12s} {r['n_scenarios']} scenarios, {r['n_rows']:,} rows, "
              f"{r['replicates_min']}-{r['replicates_max']} replicates, {r['stop_reasons']}")
    print(f"  wrote {layout.publication_marker()}")


if __name__ == "__main__":
    main()
