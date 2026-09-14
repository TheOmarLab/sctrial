#!/usr/bin/env python
"""Pre-flight check for the definitive benchmark run.

Runs a small number of iterations of one two-arm and one single-arm scenario with
every reported method, and refuses to pass unless each method returns finite
results for both designs. It also reports measured per-iteration cost by method,
which is what the full run should be sized from -- the transcriptome-scale
simulator is far more expensive per iteration than the panel-only one it
replaced, and sizing a 72-hour job from the old timings would silently truncate
the grid.

    sbatch scripts/slurm_smoke.sh

Never run on a login node.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _environment() -> dict:
    """Package versions that can change a benchmark result."""
    import subprocess
    import sys

    env = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "scipy", "anndata", "sctrial"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "unavailable"
    try:
        out = subprocess.run(
            ["Rscript", "-e",
             'ip <- installed.packages()[,"Version"]; '
             'cat(paste0("R=", getRversion(), " bioc=", '
             'tryCatch(as.character(BiocManager::version()), error=function(e) "NA"), " ", '
             'paste(sapply(c("dreamlet","limma","edgeR","nebula"), function(p) '
             'paste0(p,"=", if (p %in% rownames(installed.packages())) ip[[p]] else "absent")), '
             'collapse=" ")))'],
            capture_output=True, text=True, timeout=180,
        ).stdout.strip()
        env["r"] = out or "unavailable"
    except Exception:
        env["r"] = "unavailable"
    return env


def _parse_versions(blob) -> dict:
    """Parse a 'name=version name=version' string into a dict.

    Compared as PARSED PAIRS, not as raw strings. The same environment renders
    differently depending on ordering and on which packages are listed, so a
    string comparison reports drift when nothing has changed -- and a parity check
    that cries wolf is one people learn to ignore, which costs more than having no
    check at all.
    """
    if isinstance(blob, dict):
        return {str(k): str(v) for k, v in blob.items()}
    out = {}
    for tok in str(blob).replace(",", " ").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _environment_drift(frozen: dict, current: dict) -> dict:
    """Version differences that actually matter, ignoring formatting."""
    drift = {}
    for key, now in current.items():
        was = frozen.get(key)
        if was is None:
            continue
        if key == "r":
            f_pkgs, c_pkgs = _parse_versions(was), _parse_versions(now)
            for pkg, cv in c_pkgs.items():
                fv = f_pkgs.get(pkg)
                # A package absent from the frozen record is not drift; it is
                # something newly reported, such as the Bioconductor release.
                if fv is not None and fv != cv:
                    drift[f"r:{pkg}"] = (fv, cv)
        elif str(was) != str(now):
            drift[key] = (was, now)
    return drift


def _frozen_config() -> dict:
    """The frozen simulator configuration, if one exists."""
    import json
    import os

    base = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
    if not base:
        return {}
    path = Path(base) / "benchmark" / "validation" / "frozen_simulator_config.json"
    if not path.exists():
        return {}
    cfg = dict(json.load(open(path)).get("config") or {})
    cfg.pop("seed", None)
    cfg.pop("effects", None)
    if isinstance(cfg.get("panel_sizes"), list):
        cfg["panel_sizes"] = tuple(cfg["panel_sizes"])
    if isinstance(cfg.get("arm_ratio"), list):
        cfg["arm_ratio"] = tuple(cfg["arm_ratio"])
    return cfg


def _frozen_environment() -> dict:
    """Versions recorded when the benchmark was frozen, if a freeze exists."""
    import json
    import os

    base = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
    if not base:
        return {}
    path = Path(base) / "benchmark" / "validation" / "frozen_simulator_config.json"
    if not path.exists():
        return {}
    try:
        m = json.load(open(path)).get("manifest") or {}
    except Exception:
        return {}
    out = {k: v for k, v in (m.get("python_packages") or {}).items()}
    if m.get("python"):
        out["python"] = m["python"]
    if m.get("r_versions"):
        out["r"] = m["r_versions"]
    return out


def main() -> None:
    import argparse

    from sctrial.benchmark.orchestrator import (
        CORE_METHODS,
        _run_single_iteration,
        build_scenario_grid,
        build_sensitivity_grid,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iterations", type=int, default=2)
    ap.add_argument("--panel-probe", action="store_true",
                    help="also time the largest sensitivity panel (2000 genes)")
    args = ap.parse_args()

    cases = []
    for design in ("two_arm", "single_arm"):
        grid = build_scenario_grid(design)
        # One null and one signal-bearing scenario per design.
        null = next(s for s in grid if s["name"].startswith("null_n20"))
        de = next(s for s in grid if s["name"].startswith("de_balanced_n20"))
        cases += [(design, null), (design, de)]
    if args.panel_probe:
        # The 2,000-gene NULL as well as the 2,000-gene signal case: a filtering
        # collapse or an evaluability problem at scale shows up under the null,
        # where there is no signal to mask it.
        sens = build_sensitivity_grid("two_arm")
        # Selected BY PROPERTY, not by name. Names encode signal as a gene count
        # now, so the old literal `sens_g2000_f20` matched nothing and was skipped
        # in silence -- the probe reported success having never run the 2,000-gene
        # signal case it exists to bound.
        wanted = [
            ("2000-gene null", lambda s: s["panel_size"] == 2000 and s["n_signal"] == 0),
            ("2000-gene max signal", lambda s: (
                s["panel_size"] == 2000
                and abs(s["signal_fraction"] - 0.20) < 1e-9
                and s["architecture"] == "balanced"
            )),
        ]
        for label, pred in wanted:
            match = next((s for s in sens if pred(s)), None)
            if match is None:
                raise SystemExit(
                    f"no {label} scenario in the sensitivity grid. The probe cannot "
                    "bound the largest panel, and skipping it silently is how an "
                    "unmeasured cell reaches a 72-hour allocation."
                )
            cases.append(("two_arm", match))

    # ENVIRONMENT PARITY. The definitive run must execute under the same package
    # versions the smoke test validated, so they are recorded here and compared
    # against the frozen manifest rather than assumed to match.
    env = _environment()
    print("=== environment ===")
    for k, v in env.items():
        print(f"  {k}: {v}")
    frozen_env = _frozen_environment()
    if frozen_env:
        drift = _environment_drift(frozen_env, env)
        if drift:
            print("\n  WARNING: environment differs from the frozen manifest:")
            for k, (was, now) in sorted(drift.items()):
                print(f"    {k}: frozen {was!r} -> now {now!r}")
        else:
            print("  matches the frozen manifest")
    print()

    # Exercise the ACTUAL frozen configuration, not dataclass defaults. A smoke
    # test that runs on defaults validates a code path the definitive run will
    # never take -- and running on defaults is precisely the defect that let the
    # previous benchmark ship at 2.3e7 UMIs per cell.
    base_config = _frozen_config()
    print(f"base config: {'FROZEN' if base_config else 'DATACLASS DEFAULTS (no freeze found)'}")
    if base_config:
        print(f"  anchor dispersion_median={base_config.get('dispersion_median')}, "
              f"prepost_corr={base_config.get('prepost_corr')}, "
              f"n_per_arm={base_config.get('n_per_arm')}, "
              f"arm_ratio={base_config.get('arm_ratio')}")
    print()

    rows = []
    ok = True
    for design, scenario in cases:
        name = f"{design}__{scenario['name']}"
        print(f"\n=== {name} ({scenario['description']}) ===", flush=True)
        for it in range(args.n_iterations):
            t0 = time.time()
            out = _run_single_iteration(
                (name, it, 1000 + it, scenario, CORE_METHODS, base_config)
            )
            wall = time.time() - t0
            df = pd.DataFrame(out)
            for method, grp in df.groupby("method"):
                finite = np.isfinite(grp["pvalue"]).mean()
                rows.append(
                    {
                        "design": design,
                        "scenario": scenario["name"],
                        "panel_size": scenario["panel_size"],
                        "method": method,
                        "iteration": it,
                        "finite_pvalue_frac": float(finite),
                        "method_seconds": float(grp["runtime_seconds"].iloc[0]),
                        "iteration_seconds": wall,
                    }
                )
                flag = "" if finite > 0.5 else "   <-- FAILING"
                print(
                    f"  it{it} {method:16s} finite p {finite:6.1%}  "
                    f"{grp['runtime_seconds'].iloc[0]:8.1f} s{flag}",
                    flush=True,
                )
                if finite <= 0.5:
                    ok = False
            print(f"  it{it} TOTAL {wall:.1f} s", flush=True)

    res = pd.DataFrame(rows)
    print("\n=== per-method median seconds (by panel size) ===")
    piv = res.pivot_table(
        index="method", columns="panel_size", values="method_seconds", aggfunc="median"
    )
    print(piv.to_string(float_format=lambda v: f"{v:8.1f}"))

    print("\n=== projected wall-clock for the definitive run ===")
    per_iter = res.groupby(["design", "scenario"])["iteration_seconds"].median()
    med = float(per_iter.median())
    n_scen_two = len(build_scenario_grid("two_arm"))
    n_scen_one = len(build_scenario_grid("single_arm"))
    n_sens = len(build_sensitivity_grid("two_arm"))
    for label, n_scen, n_iter, workers in (
        ("core two_arm", n_scen_two, 200, 10),
        ("core single_arm", n_scen_one, 200, 10),
        ("sensitivity", n_sens, 200, 10),
    ):
        hours = n_scen * n_iter * med / workers / 3600
        print(
            f"  {label:18s} {n_scen:3d} scenarios x 200 it / {workers} workers "
            f"~= {hours:6.1f} h   (at the median 50-gene iteration; larger panels cost more)"
        )
    print(
        "\nNOTE: the projection uses the median iteration measured here. The 2000-gene "
        "sensitivity cells are substantially slower; run --panel-probe to bound them."
    )

    out_dir = Path("manuscript/benchmark/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "smoke_benchmark.csv", index=False)
    print(f"\nwrote {out_dir / 'smoke_benchmark.csv'}")

    if not ok:
        raise SystemExit("SMOKE TEST FAILED: a method returned mostly non-finite p-values")
    print("\nSMOKE TEST PASSED: every method returned finite results for both designs")


if __name__ == "__main__":
    main()
