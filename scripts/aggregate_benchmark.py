#!/usr/bin/env python
"""Combine the benchmark shards. The ONLY writer of the combined results file.

The definitive run is split across several SLURM jobs by design family and panel
size. Letting each of them write the combined file meant the last to finish
overwrote the rest, and the combined file is the only one the figures read -- so
the pure-null calibration panel would have been drawn from a quarter of the grid,
with no error and no missing-data marker.

Globbing inside the producers is not enough either. A shard that fails, is
delayed, is resumed, or leaves a stale file behind still yields a plausible
combined file. One dependent aggregator removes the entire concurrency class:
producers write only their own scenario files, this runs once after all of them
succeed, and it refuses to write unless the shards form exactly the expected set.

    sbatch --dependency=afterok:J1:J2:J3:J4 scripts/slurm_aggregate.sh core

Writes `benchmark_complete.json` on success. Figure generation refuses to run
without it, so a partial grid cannot reach a manuscript figure.
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


def _expected_scenarios(grid: str) -> set[str]:
    """Every scenario the frozen grid declares, across all shards."""
    from sctrial.benchmark.orchestrator import build_scenario_grid, build_sensitivity_grid

    out: set[str] = set()
    if grid == "core":
        for design in ("two_arm", "single_arm"):
            out |= {f"{design}__{s['name']}" for s in build_scenario_grid(design)}
    else:
        out |= {f"two_arm__{s['name']}" for s in build_sensitivity_grid("two_arm")}
    return out


def aggregate(out_dir: Path, grid: str, combined_name: str, expect_iterations: int) -> None:
    expected = _expected_scenarios(grid)
    shards = sorted(f for f in out_dir.glob("*.csv") if f.name != combined_name)
    if not shards:
        raise SystemExit(f"no scenario files in {out_dir}")

    frames, seen, problems = [], {}, []
    for f in shards:
        df = pd.read_csv(f, low_memory=False)
        if "scenario" not in df.columns:
            problems.append(f"{f.name}: no scenario column")
            continue
        for name, grp in df.groupby("scenario", observed=True):
            if name in seen:
                problems.append(f"{name}: appears in {seen[name]} and {f.name}")
            seen[name] = f.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    present = set(combined["scenario"].unique())

    # 1. exact scenario set -- nothing missing, nothing unexpected
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing:
        problems.append(f"{len(missing)} scenario(s) MISSING: {missing[:8]}")
    if unexpected:
        problems.append(f"{len(unexpected)} UNEXPECTED scenario(s): {unexpected[:8]}")

    # 2. one manifest across every shard
    if "manifest_sha256" not in combined.columns:
        problems.append("results carry no manifest_sha256")
    else:
        manifests = sorted(set(combined["manifest_sha256"].dropna().astype(str)))
        if len(manifests) != 1:
            problems.append(f"shards mix {len(manifests)} manifests: {[m[:12] for m in manifests]}")
        if combined["manifest_sha256"].isna().any():
            problems.append("some rows are unstamped")

    # 3. every scenario reached its iteration count
    if "iteration" in combined.columns:
        counts = combined.groupby("scenario", observed=True)["iteration"].nunique()
        short = counts[counts < expect_iterations]
        if len(short):
            problems.append(
                f"{len(short)} scenario(s) under {expect_iterations} iterations: "
                f"{short.head(8).to_dict()}"
            )

    # 4. requested and realised design agree
    if {"n_participants", "n_treated", "n_control"} <= set(combined.columns):
        bad = combined[combined["n_participants"] != combined["n_treated"] + combined["n_control"]]
        if len(bad):
            problems.append(f"{len(bad)} rows where treated + control != total participants")
        flat = combined.groupby("scenario", observed=True)["n_participants"].nunique()
        if (flat > 1).any():
            problems.append(f"scenarios with inconsistent participant counts: {flat[flat>1].to_dict()}")
    else:
        problems.append("results carry no realised-design columns")

    if problems:
        raise SystemExit(
            "REFUSING to write the combined results:\n  - " + "\n  - ".join(problems)
        )

    tmp = out_dir / (combined_name + ".tmp")
    combined.to_csv(tmp, index=False)
    os.replace(tmp, out_dir / combined_name)

    record = {
        "grid": grid,
        "combined": combined_name,
        "n_shards": len(shards),
        "n_scenarios": len(present),
        "n_rows": int(len(combined)),
        "iterations_per_scenario": expect_iterations,
        "manifest_sha256": str(combined["manifest_sha256"].iloc[0]),
        "scenarios": sorted(present),
    }
    with open(out_dir / "benchmark_complete.json", "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"combined {len(shards)} shards -> {out_dir / combined_name}")
    print(f"  scenarios {len(present)}  rows {len(combined):,}  "
          f"manifest {record['manifest_sha256'][:12]}")
    print(f"  wrote completion record -> {out_dir / 'benchmark_complete.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("grid", choices=["core", "sensitivity"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--iterations", type=int, default=200)
    args = ap.parse_args()

    base = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
    root = Path(base) if base else REPO.parent.parent / "manuscript"
    out_dir = Path(args.out_dir) if args.out_dir else (
        root / "benchmark" / ("simulation" if args.grid == "core" else "sensitivity")
    )
    combined = "benchmark_combined.csv" if args.grid == "core" else "sensitivity_combined.csv"
    aggregate(out_dir, args.grid, combined, args.iterations)


if __name__ == "__main__":
    main()
