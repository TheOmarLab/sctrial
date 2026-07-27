#!/usr/bin/env python
"""Measure the largest scheduled scenarios before sizing the definitive run.

    sbatch scripts/timing_probe.py          # via slurm_timing_probe.sh
    NEVER on a login node.

WHY THIS EXISTS
---------------
The previous wall-clock projection was measured while a leaked calibration field
capped every two-arm scenario at eleven participants. It therefore under-projects
the sample-size scenarios by roughly the ratio of their real size to eleven, and
sizing a 72-hour allocation from it would truncate the grid.

WHY TWO SCENARIOS AND NOT ONE
-----------------------------
"Largest" is not a single point in this grid. Simulation and memory scale with
participants x visits x cells-per-visit x the 20,284-gene transcriptome, while
model fitting scales with the number of TESTED genes. The grid's maxima fall in
different cells:

    cells_1000_n40    80 participants x 1,000 cells/visit = 160,000 cells
                      -- the memory and simulation maximum
    sens_null_g2000   80 participants x 500 cells/visit, 2,000 tested genes
                      -- the fitting maximum

No scenario combines the largest sample size, the largest cell yield and the
largest panel, so probing an invented worst case would measure something that is
never scheduled. Both real maxima are measured instead.

MEMORY IS REPORTED, NOT INFERRED
--------------------------------
With a 20,284-gene background transcriptome and up to 160,000 cells, a dense
intermediate is a more dangerous failure than a slow model fit, and it would
appear as an OOM kill hours into the run rather than as an error.

Three numbers are reported because they answer different questions, and
conflating them gave a wrong answer once already:

* the Python HEAP peak (tracemalloc), reset per scenario -- this is the
  simulator's own footprint, and it is what a dense intermediate would inflate;
* the PROCESS TREE peak, sampled while the scenario runs -- this includes the R
  subprocesses, which hold the memory that matters for `--mem` and appear in
  neither tracemalloc nor this process's own `ru_maxrss`;
* the `ru_maxrss` HIGH-WATER mark, which is CUMULATIVE since process start and
  therefore not per-scenario at all.

The first version of this probe reported only the third and labelled it
per-scenario. Two consecutive scenarios then reported an identical 5.04 GB to
two decimal places -- the second was the first one carried forward, not a
measurement. `ru_maxrss` cannot be reset, so a per-scenario peak has to be
sampled instead.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# `ru_maxrss` is KILOBYTES on Linux and BYTES on macOS. The probe runs on Linux;
# the platform check keeps a local dry run from reporting figures 1,024x wrong.
_RSS_UNIT = 1 if sys.platform == "linux" else 1024


def _rusage_high_water_gb() -> tuple[float, float]:
    """CUMULATIVE peak RSS since process start, for self and children, in GB.

    `ru_maxrss` is a high-water mark that never decreases, so this is NOT a
    per-scenario figure -- after the first scenario it reports the largest value
    seen so far. That is the right quantity for sizing `--mem` (a worker must
    survive its worst moment) and the wrong one for attributing memory to a
    scenario. Per-scenario peaks come from `_RSSSampler` below.

    The distinction is not academic: the first two scenarios reported an
    identical 5.04 GB to two decimal places, which is what revealed that the
    second figure was the first one carried forward rather than a measurement.
    """
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_UNIT
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * _RSS_UNIT
    return me / 1024**2, kids / 1024**2


def _proc_tree_rss_bytes() -> int:
    """Current RSS of this process and every descendant, from /proc.

    Deliberately NOT psutil: it is absent from the analysis environment, and
    adding a package to size a run would change the very package manifest the
    frozen run records. /proc is already the source psutil reads on Linux.

    Returns 0 off Linux, which makes the sampler report unavailable rather than
    silently reporting a wrong number.
    """
    proc = Path("/proc")
    if not proc.exists():
        return 0
    page = os.sysconf("SC_PAGE_SIZE")
    me = os.getpid()

    # ppid for every live process, so descendants can be found without psutil.
    parent: dict[int, int] = {}
    rss: dict[int, int] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            # Field 4 of /proc/<pid>/stat is ppid, but comm (field 2) may contain
            # spaces or parentheses, so split after the final ')'.
            stat = (entry / "stat").read_text()
            parent[pid] = int(stat[stat.rindex(")") + 2:].split()[1])
            rss[pid] = int((entry / "statm").read_text().split()[1]) * page
        except (OSError, ValueError, IndexError):
            continue

    total = rss.get(me, 0)
    for pid in rss:
        walk, hops = pid, 0
        while walk in parent and hops < 64:
            walk = parent[walk]
            hops += 1
            if walk == me:
                total += rss[pid]
                break
            if walk <= 1:
                break
    return total


class _RSSSampler:
    """Sample this process tree's RSS on a background thread.

    Needed because `ru_maxrss` cannot be reset, so a per-scenario peak cannot be
    derived from it. Samples the whole tree, since the R subprocesses hold the
    memory that matters and appear in neither this process's `ru_maxrss` nor in
    tracemalloc.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = _proc_tree_rss_bytes() > 0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_gb = max(self.peak_gb, _proc_tree_rss_bytes() / 1024**3)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return False


def _scenario_by_name(name: str) -> tuple[str, dict]:
    from sctrial.benchmark.orchestrator import build_scenario_grid, build_sensitivity_grid

    for design in ("two_arm", "single_arm"):
        for sc in build_scenario_grid(design):
            if sc["name"] == name:
                return design, sc
    for sc in build_sensitivity_grid("two_arm"):
        if sc["name"] == name:
            return "two_arm", sc
    raise SystemExit(f"no scenario named {name!r} in the frozen grid")


def _frozen_config() -> dict:
    """The frozen calibration, with scenario-owned fields stripped.

    Exercising dataclass defaults would measure a configuration the definitive
    run never uses -- and running on defaults is the specific defect that let the
    shipped benchmark run at 2.3e7 UMIs per cell.
    """
    from sctrial.benchmark.simulator_v2 import SCENARIO_OWNED_FIELDS

    base = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
    if not base:
        raise SystemExit("SCTRIAL_MANUSCRIPT_DIR is not set")
    path = Path(base) / "benchmark" / "validation" / "frozen_simulator_config.json"
    if not path.exists():
        raise SystemExit(
            f"no frozen configuration at {path}. The probe must measure the "
            "configuration the definitive run will use, not dataclass defaults."
        )
    blob = json.loads(path.read_text())
    cfg = dict(blob.get("config") or {})
    manifest = dict(blob.get("manifest") or {})
    cfg.pop("seed", None)
    cfg.pop("effects", None)
    if isinstance(cfg.get("panel_sizes"), list):
        cfg["panel_sizes"] = tuple(cfg["panel_sizes"])
    if isinstance(cfg.get("arm_ratio"), list):
        cfg["arm_ratio"] = tuple(cfg["arm_ratio"])
    stripped = {k: v for k, v in cfg.items() if k not in SCENARIO_OWNED_FIELDS}
    return {"config": stripped, "manifest": manifest}


def probe(name: str, methods: list[str], n_jobs: int) -> dict:
    from sctrial.benchmark.orchestrator import _run_single_iteration

    design, scenario = _scenario_by_name(name)
    frozen = _frozen_config()
    full_name = f"{design}__{name}"

    k = scenario["config_kwargs"]
    ratio = k.get("arm_ratio")
    n_part = sum(ratio) if ratio else (
        k["n_per_arm"] if design == "single_arm" else 2 * k["n_per_arm"]
    )
    print(f"\n{'=' * 72}")
    print(f"{full_name}: {scenario['description']}")
    print(f"  participants {n_part}  cells/visit {k.get('cells_per_pv_fixed', 'empirical')}"
          f"  tested genes {scenario['panel_size']}  signal {scenario['n_signal']}")
    print(f"{'=' * 72}", flush=True)

    tracemalloc.start()
    t0 = time.time()
    with _RSSSampler() as sampler:
        rows = _run_single_iteration(
            (full_name, 0, 12345, scenario, methods, frozen["config"])
        )
    wall = time.time() - t0
    _, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_self, rss_kids = _rusage_high_water_gb()
    scenario_peak = sampler.peak_gb if sampler.available else float("nan")

    df = pd.DataFrame(rows)
    per_method = {}
    for method, grp in df.groupby("method", observed=True):
        per_method[str(method)] = {
            "runtime_seconds": float(grp["runtime_seconds"].iloc[0]),
            "evaluable_fraction": float(grp["evaluable"].mean()),
            "converged_fraction": float(grp["converged"].mean()),
            "finite_pvalue_fraction": float(np.isfinite(grp["pvalue"]).mean()),
            "n_genes": int(len(grp)),
        }

    # REALISED, not requested -- the whole point of the exercise.
    realised = {
        "n_participants": int(df["n_participants"].iloc[0]),
        "n_treated": int(df["n_treated"].iloc[0]),
        "n_control": int(df["n_control"].iloc[0]),
        "cells_per_pv_median": float(df["cells_per_pv_median"].iloc[0]),
        "cells_per_pv_max": float(df["cells_per_pv_max"].iloc[0]),
        "panel_size": int(df["panel_size"].iloc[0]),
    }
    total_cells = realised["cells_per_pv_median"] * realised["n_participants"] * 2

    print(f"  wall {wall:8.1f} s   python heap peak {heap_peak / 1024**3:6.2f} GB")
    if sampler.available:
        print(f"  peak RSS this scenario (whole tree) {scenario_peak:6.2f} GB")
    else:
        print("  peak RSS this scenario: UNAVAILABLE (/proc absent)")
    print(f"  rusage high-water since start: self {rss_self:6.2f} GB  "
          f"children {rss_kids:6.2f} GB  [CUMULATIVE, not per-scenario]")
    print(f"  realised: {realised['n_treated']}T/{realised['n_control']}C, "
          f"{realised['cells_per_pv_median']:.0f} cells/visit "
          f"(~{total_cells:,.0f} cells total)")
    for m, s in sorted(per_method.items(), key=lambda kv: -kv[1]["runtime_seconds"]):
        flag = "" if s["finite_pvalue_fraction"] > 0.5 else "   <-- FAILING"
        print(f"    {m:16s} {s['runtime_seconds']:8.1f} s  "
              f"evaluable {s['evaluable_fraction']:6.1%}  "
              f"converged {s['converged_fraction']:6.1%}{flag}", flush=True)

    return {
        "scenario": full_name,
        "description": scenario["description"],
        "wall_seconds": wall,
        "python_heap_peak_gb": heap_peak / 1024**3,
        "peak_rss_scenario_gb": scenario_peak,
        "rusage_high_water_self_gb": rss_self,
        "rusage_high_water_children_gb": rss_kids,
        "requested": {
            "n_participants": n_part,
            "cells_per_pv_fixed": k.get("cells_per_pv_fixed"),
            "panel_size": scenario["panel_size"],
            "n_signal": scenario["n_signal"],
        },
        "realised": realised,
        "total_cells": total_cells,
        "methods": per_method,
        "n_jobs": n_jobs,
        "manifest_sha256": frozen["manifest"].get("manifest_sha256"),
    }


def main() -> None:
    import argparse

    from sctrial.benchmark.orchestrator import CORE_METHODS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenarios", nargs="+",
        default=["cells_1000_n40", "sens_null_g2000", "null_n60"],
        help="the grid's true maxima: memory, fitting, and sample size",
    )
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    methods = args.methods or list(CORE_METHODS)
    print(f"methods: {methods}")

    results = [probe(name, methods, args.n_jobs) for name in args.scenarios]

    print(f"\n{'=' * 72}")
    print("SIZING THE DEFINITIVE RUN")
    print(f"{'=' * 72}")
    slowest = max(results, key=lambda r: r["wall_seconds"])
    def _mem(r):
        v = r.get("peak_rss_scenario_gb")
        # Fall back to the cumulative mark only if sampling was unavailable.
        return v if v == v else max(r["rusage_high_water_self_gb"],
                                    r["rusage_high_water_children_gb"])

    hungriest = max(results, key=_mem)
    print(f"  slowest single replicate : {slowest['scenario']} "
          f"{slowest['wall_seconds']:.0f} s")
    print(f"  largest memory footprint : {hungriest['scenario']} "
          f"{_mem(hungriest):.1f} GB (process tree peak)")
    print("\n  Per-worker memory, not per-node, drives --mem: N_JOBS concurrent")
    print("  workers each hold their own copy. Size --mem from the largest")
    print("  footprint x N_JOBS, with margin.")

    out = Path(args.out) if args.out else (
        Path(os.environ.get("SCTRIAL_MANUSCRIPT_DIR", "."))
        / "benchmark" / "results" / "_preflight" / "timing_probe.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
