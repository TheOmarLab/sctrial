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

    sbatch --dependency=afterok:J1:J2:J3:J4 scripts/slurm_aggregate.sh core <sha>

COMPLETION IS A RECORD, NOT A ROW COUNT
---------------------------------------
Monte Carlo stopping is adaptive and decided per scenario, so the replicate count
is an OUTCOME rather than a constant. A threshold cannot validate it: extension
only ever adds replicates, so "at least the base batch" is satisfied by every
scenario including one killed part-way through an extension, which is exactly the
case that must be caught. Each scenario therefore writes a completion record as
its last action, and this checks the record rather than counting rows -- that the
record exists, that its replicate count matches the CSV, and that its stated stop
reason is actually justified by the achieved Monte Carlo error.

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


def _check_completion(
    records: dict,
    present: set[str],
    combined: pd.DataFrame,
    allow_non_adaptive: bool = False,
) -> list[str]:
    """Validate adaptive-Monte-Carlo completion semantics.

    Three distinct failures are separated here:

    * a scenario with results but NO record -- killed before it finished;
    * a record whose replicate count disagrees with its CSV -- a partial write,
      or a resume that adopted rows from a different run;
    * a record claiming `precision_reached` whose achieved MCSE does not meet the
      target it names. That last one is the check that makes the stop reason
      meaningful rather than decorative: without it, a truncated scenario could
      simply be labelled as having stopped early.
    """
    from sctrial.benchmark.scenario_contract import (
        DEFINITIVE_STOP_REASONS,
        VALID_STOP_REASONS,
    )

    problems: list[str] = []

    no_record = sorted(present - set(records))
    if no_record:
        problems.append(
            f"{len(no_record)} scenario(s) have results but NO completion record "
            f"(killed mid-run?): {no_record[:8]}"
        )

    actual = (
        combined.groupby("scenario", observed=True)["iteration"].nunique().to_dict()
        if "iteration" in combined.columns else {}
    )

    for name in sorted(set(records) & present):
        rec = records[name]
        reason = rec.get("stop_reason")
        n_rec = rec.get("n_replicates_completed")
        n_csv = actual.get(name)

        if reason not in VALID_STOP_REASONS:
            problems.append(f"{name}: invalid stop_reason {reason!r}")

        if n_csv is not None and n_rec != n_csv:
            problems.append(
                f"{name}: record says {n_rec} replicates, CSV holds {n_csv}"
            )

        if reason == "precision_reached":
            # The scenario claims it stopped because it was precise enough. Verify
            # that against the achieved error, or the reason is unfalsifiable.
            f_ok = rec.get("fpr_mcse", float("inf")) <= rec.get("mcse_target_fpr", 0)
            p_ok = rec.get("power_mcse", float("inf")) <= rec.get("mcse_target_power", 0)
            if not (f_ok and p_ok):
                problems.append(
                    f"{name}: claims precision_reached but achieved fpr MCSE "
                    f"{rec.get('fpr_mcse'):.4f} (target {rec.get('mcse_target_fpr')}) "
                    f"power MCSE {rec.get('power_mcse'):.4f} "
                    f"(target {rec.get('mcse_target_power')})"
                )
        elif reason == "max_replicates_reached":
            if n_rec is not None and n_rec < rec.get("max_replicates", 0):
                problems.append(
                    f"{name}: claims max_replicates_reached at {n_rec} of "
                    f"{rec.get('max_replicates')}"
                )

        if reason not in DEFINITIVE_STOP_REASONS and not allow_non_adaptive:
            problems.append(
                f"{name}: stopped with {reason!r} -- adaptive Monte Carlo was off, "
                "so this shard has no precision guarantee and cannot form part of a "
                "definitive run (pass --allow-non-adaptive for a debug aggregation)"
            )

    return problems


def aggregate(
    layout,
    grid: str,
    combined_name: str,
    min_iterations: int,
    allow_non_adaptive: bool = False,
) -> None:
    expected = _expected_scenarios(grid)
    shards = sorted(layout.scenarios.glob("*.csv"))
    if not shards:
        raise SystemExit(f"no scenario files in {layout.scenarios}")

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
            del grp
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

    # 2. one manifest across every shard, matching the directory it lives in
    if "manifest_sha256" not in combined.columns:
        problems.append("results carry no manifest_sha256")
    else:
        manifests = sorted(set(combined["manifest_sha256"].dropna().astype(str)))
        if len(manifests) != 1:
            problems.append(
                f"shards mix {len(manifests)} manifests: {[m[:12] for m in manifests]}"
            )
        elif not manifests[0].startswith(layout.manifest_sha[:12]):
            problems.append(
                f"rows carry manifest {manifests[0][:12]} but sit under "
                f"{layout.manifest_sha[:12]}"
            )
        if combined["manifest_sha256"].isna().any():
            problems.append("some rows are unstamped")

    # 3. ADAPTIVE completion -- records, not a fixed row count
    records = layout.completed_scenarios()
    problems += _check_completion(records, present, combined, allow_non_adaptive)

    # A floor still applies: adaptive stopping extends beyond the base batch and
    # never below it, so anything under the base batch is a truncation the
    # completion record alone might not reveal.
    if "iteration" in combined.columns:
        counts = combined.groupby("scenario", observed=True)["iteration"].nunique()
        short = counts[counts < min_iterations]
        if len(short):
            problems.append(
                f"{len(short)} scenario(s) below the {min_iterations}-replicate base "
                f"batch: {short.head(8).to_dict()}"
            )

    # 4. requested and realised design agree
    if {"n_participants", "n_treated", "n_control"} <= set(combined.columns):
        bad = combined[
            combined["n_participants"] != combined["n_treated"] + combined["n_control"]
        ]
        if len(bad):
            problems.append(f"{len(bad)} rows where treated + control != total participants")
        flat = combined.groupby("scenario", observed=True)["n_participants"].nunique()
        if (flat > 1).any():
            problems.append(
                f"scenarios with inconsistent participant counts: {flat[flat > 1].to_dict()}"
            )
    else:
        problems.append("results carry no realised-design columns")

    # 5. nothing left behind by a killed job
    orphans = layout.orphan_scenarios()
    if orphans:
        problems.append(f"{len(orphans)} scenario CSV(s) with no completion record: {orphans[:8]}")

    if problems:
        raise SystemExit(
            "REFUSING to write the combined results:\n  - " + "\n  - ".join(problems)
        )

    layout.combined.mkdir(parents=True, exist_ok=True)
    target = layout.combined_csv(combined_name)
    tmp = target.with_suffix(".tmp")
    combined.to_csv(tmp, index=False)
    os.replace(tmp, target)

    reps = {n: r["n_replicates_completed"] for n, r in records.items()}
    stops = {}
    for r in records.values():
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1

    record = {
        "grid": grid,
        "combined": combined_name,
        "n_shards": len(shards),
        "n_scenarios": len(present),
        "n_rows": int(len(combined)),
        "replicates_per_scenario": reps,
        "replicates_min": min(reps.values()) if reps else None,
        "replicates_max": max(reps.values()) if reps else None,
        "stop_reasons": stops,
        "base_batch": min_iterations,
        "adaptive": not allow_non_adaptive,
        "manifest_sha256": str(combined["manifest_sha256"].iloc[0]),
        "scenarios": sorted(present),
    }
    with open(layout.completion_marker(grid), "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"combined {len(shards)} shards -> {target}")
    print(f"  scenarios {len(present)}  rows {len(combined):,}  "
          f"manifest {record['manifest_sha256'][:12]}")
    print(f"  replicates {record['replicates_min']}-{record['replicates_max']}  "
          f"stop reasons {stops}")
    print(f"  wrote completion record -> {layout.completion_marker(grid)}")


def main() -> None:
    from sctrial.benchmark.paths import ResultLayout

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("grid", choices=["core", "sensitivity"])
    ap.add_argument("manifest_sha", help="manifest hash naming the results directory")
    ap.add_argument("--results-root", default=None)
    ap.add_argument(
        "--min-iterations", type=int, default=200,
        help="base Monte Carlo batch; adaptive stopping may exceed it, never fall below",
    )
    ap.add_argument(
        "--allow-non-adaptive", action="store_true",
        help="accept shards run with adaptive stopping OFF (debug aggregations only)",
    )
    args = ap.parse_args()

    base = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
    root = Path(base) if base else REPO.parent.parent / "manuscript"
    results_root = (
        Path(args.results_root) if args.results_root
        else root / "benchmark" / "results"
    )
    combined = "benchmark_combined.csv" if args.grid == "core" else "sensitivity_combined.csv"
    aggregate(
        ResultLayout(results_root, args.manifest_sha),
        args.grid,
        combined,
        args.min_iterations,
        allow_non_adaptive=args.allow_non_adaptive,
    )


if __name__ == "__main__":
    main()
