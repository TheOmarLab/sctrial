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
appear as an OOM kill hours into the run rather than as an error. Peak RSS is
recorded for this process AND its children -- the R subprocesses do the heavy
fitting and are invisible to tracemalloc, which sees only the Python heap. Both
are reported so a memory problem can be attributed to the right layer.
"""
from __future__ import annotations

import json
import os
import resource
import sys
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


def _peak_rss_gb() -> tuple[float, float]:
    """Peak RSS of this process and of its children, in GB.

    Children matter more than self here: dreamlet, limma-voom and NEBULA run in R
    subprocesses, and their memory does not appear in this process's own usage or
    in tracemalloc.
    """
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_UNIT
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * _RSS_UNIT
    return me / 1024**2, kids / 1024**2


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
    rows = _run_single_iteration((full_name, 0, 12345, scenario, methods, frozen["config"]))
    wall = time.time() - t0
    _, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_self, rss_kids = _peak_rss_gb()

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
    print(f"  peak RSS self {rss_self:6.2f} GB   children (R) {rss_kids:6.2f} GB")
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
        "peak_rss_self_gb": rss_self,
        "peak_rss_children_gb": rss_kids,
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
    hungriest = max(results, key=lambda r: r["peak_rss_children_gb"] + r["peak_rss_self_gb"])
    print(f"  slowest single replicate : {slowest['scenario']} "
          f"{slowest['wall_seconds']:.0f} s")
    print(f"  largest memory footprint : {hungriest['scenario']} "
          f"{hungriest['peak_rss_self_gb'] + hungriest['peak_rss_children_gb']:.1f} GB "
          f"(self {hungriest['peak_rss_self_gb']:.1f} + R {hungriest['peak_rss_children_gb']:.1f})")
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
