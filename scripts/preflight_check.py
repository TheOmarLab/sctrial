#!/usr/bin/env python
"""One command, run on the cluster, that must pass before any job is scheduled.

    micromamba run -n sctrial python scripts/preflight_check.py --expect-tag sctrial-benchmark-v1.0.0

Every check here corresponds to a failure this project has actually had, and each
one was silent at the time:

* a calibration read from a deleted scratch file;
* a stale .npz analysed after a partial sync;
* HEAD pinned at an old commit while newer code ran, with 109 dirty files;
* `git_commit` recorded as the literal string "unavailable" because git is absent
  from compute nodes;
* the frozen configuration never reaching the scenario generator;
* a benchmark shipped at 2.3e7 UMIs per cell against a TNBC median of 2,113.

None of them would have been caught by a test suite, because none of them is a
defect in the code. They are mismatches between the code, the configuration and
the machine. This checks that triple.

Exit status is the whole point: non-zero means DO NOT LAUNCH.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

_OK = "  PASS  "
_NO = "  FAIL  "


class Preflight:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        print(f"{_OK if ok else _NO}{label}" + (f"  --  {detail}" if detail else ""))
        if not ok:
            self.failures.append(f"{label}: {detail}" if detail else label)
        return ok

    def note(self, label: str, detail: str) -> None:
        print(f"  ....  {label}  --  {detail}")
        self.notes.append(f"{label}: {detail}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expect-tag", default=None,
                    help="annotated tag the deployed commit must correspond to")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--allow-existing-results", action="store_true",
                    help="permit results already present for this manifest (resume)")
    args = ap.parse_args()

    pf = Preflight()
    print("=" * 74)
    print("PRE-LAUNCH VERIFICATION")
    print("=" * 74)

    base = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
    if not base:
        raise SystemExit("SCTRIAL_MANUSCRIPT_DIR is not set")
    manuscript = Path(base)
    frozen_path = manuscript / "benchmark" / "validation" / "frozen_simulator_config.json"

    # ---- 1. a frozen configuration exists and parses -------------------
    if not pf.check("frozen configuration present", frozen_path.exists(), str(frozen_path)):
        raise SystemExit("\nDO NOT LAUNCH: no frozen configuration")
    blob = json.loads(frozen_path.read_text())
    manifest = blob.get("manifest") or {}
    config = blob.get("config") or {}

    # ---- 2. the manifest hash is present, canonical and self-consistent
    from sctrial.benchmark.manifest import manifest_hash, source_tree_sha256, verify_manifest

    sha = manifest.get("manifest_sha256")
    pf.check("manifest carries a hash", bool(sha), str(sha)[:16] if sha else "MISSING")
    if sha:
        import re

        pf.check("hash is a hex digest", bool(re.fullmatch(r"[0-9a-f]{64}", sha)))
        pf.check("hash is self-consistent", manifest_hash(manifest) == sha,
                 "recomputed digest matches the stored one")
        try:
            verify_manifest(manifest)
            pf.check("verify_manifest", True)
        except Exception as exc:  # noqa: BLE001
            pf.check("verify_manifest", False, str(exc))

    # ---- 3. THE SOURCE ON DISK IS THE SOURCE THAT WAS FROZEN -----------
    # This is the check that catches "HEAD says one thing, the running code is
    # another". It reads the files, not git.
    recorded = manifest.get("source_tree_sha256")
    actual = source_tree_sha256()
    pf.check(
        "source tree matches the frozen manifest",
        bool(recorded) and recorded == actual,
        f"frozen {str(recorded)[:16]} vs on-disk {actual[:16]}",
    )

    # ---- 4. git provenance is real, not the string "unavailable" -------
    commit = str(manifest.get("git_commit") or "")
    pf.check("git commit recorded", len(commit) == 40 and commit != "unavailable" * 4,
             commit[:12] or "MISSING")
    pf.check("frozen from a clean tree", manifest.get("git_dirty") is False,
             f"git_dirty={manifest.get('git_dirty')}")
    if args.expect_commit:
        pf.check("commit matches --expect-commit", commit.startswith(args.expect_commit),
                 f"{commit[:12]} vs {args.expect_commit[:12]}")

    if args.expect_tag:
        import subprocess

        try:
            tagged = subprocess.run(
                ["git", "-C", str(REPO), "rev-list", "-n", "1", args.expect_tag],
                capture_output=True, text=True, timeout=60,
            ).stdout.strip()
        except Exception:
            tagged = ""
        if tagged:
            pf.check(f"deployed commit is {args.expect_tag}", tagged == commit,
                     f"tag {tagged[:12]} vs frozen {commit[:12]}")
        else:
            # git is absent on compute nodes; the deploy captured provenance on
            # the login node instead. Say so rather than passing silently.
            dep = REPO / ".deploy_provenance.json"
            info = json.loads(dep.read_text()) if dep.exists() else {}
            pf.note("tag check skipped", f"git unavailable here; deploy recorded "
                                         f"{str(info.get('git_describe', '?'))}")

    # ---- 5. no experimental design inside the calibration --------------
    from sctrial.benchmark.simulator_v2 import SCENARIO_OWNED_FIELDS

    leaked = [k for k in SCENARIO_OWNED_FIELDS if k in config]
    pf.check("no scenario-owned fields in the calibration", not leaked, str(leaked))

    # ---- 6. gates passed -----------------------------------------------
    gs = manifest.get("gate_summary") or blob.get("gate_summary") or {}
    pf.check("all gates pass", gs.get("n_fail", 1) == 0 and gs.get("n_pass", 0) > 0,
             f"{gs.get('n_pass')} pass / {gs.get('n_fail')} fail")
    pf.check("no pinned-statistic failures", gs.get("n_fail_pinned", 0) == 0)

    # ---- 7. calibration artifacts still hash as recorded ---------------
    for key, fname in (("targets_sha256", f"{blob.get('dataset', 'tnbc')}_sim_targets.json"),
                       ("empirical_sha256", f"{blob.get('dataset', 'tnbc')}_empirical.npz")):
        rec = manifest.get(key)
        path = manuscript / "benchmark" / "validation" / fname
        if not rec:
            continue
        import hashlib

        if path.exists():
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            pf.check(f"{fname} unchanged", h == rec, f"{h[:12]} vs {rec[:12]}")
        else:
            pf.check(f"{fname} present", False, f"missing at {path}")

    # ---- 8. the scenario grid is the one that was reasoned about -------
    from sctrial.benchmark.orchestrator import (
        build_scenario_grid,
        build_sensitivity_grid,
        mc_max_for,
    )

    core = [s for d in ("two_arm", "single_arm") for s in build_scenario_grid(d)]
    sens = build_sensitivity_grid("two_arm")
    pf.check("core grid size", len(core) == 57, f"{len(core)} scenarios")
    pf.check("sensitivity grid size", len(sens) == 36, f"{len(sens)} scenarios")

    by_panel: dict[int, set] = {}
    for s in sens:
        by_panel.setdefault(s["panel_size"], set()).add(round(s["signal_fraction"], 4))
    expected = {0.0, 0.02, 0.04, 0.10, 0.20}
    pf.check("signal grid is a complete factorial",
             set(by_panel) == {50, 200, 500, 2000}
             and all(expected <= f for f in by_panel.values()),
             f"panels {sorted(by_panel)}")
    raised = [s["name"] for s in sens if mc_max_for(s) > 1000]
    pf.note("raised replicate cap", f"{len(raised)} scenarios at 2500: {sorted(raised)}")

    # ---- 9. no results already sitting under this manifest -------------
    from sctrial.benchmark.paths import ResultLayout

    if sha:
        layout = ResultLayout(manuscript / "benchmark" / "results", sha)
        existing = []
        for grid in ("core", "sensitivity"):
            d = layout.scenarios_for(grid)
            if d.exists():
                existing += [p.name for p in d.glob("*.csv")]
        if args.allow_existing_results:
            pf.note("existing results", f"{len(existing)} scenario file(s) (resume permitted)")
        else:
            pf.check("no pre-existing results for this manifest", not existing,
                     f"{len(existing)} file(s) would be resumed into")

    # ---- 10. every method actually dispatches --------------------------
    from sctrial.benchmark.orchestrator import CORE_METHODS

    pf.check("method set", len(CORE_METHODS) == 5, str(CORE_METHODS))
    try:
        import subprocess

        r = subprocess.run(
            ["Rscript", "-e",
             'cat(all(c("dreamlet","limma","edgeR","nebula") %in% rownames(installed.packages())))'],
            capture_output=True, text=True, timeout=300,
        ).stdout.strip()
        pf.check("R packages installed", r.upper() == "TRUE", r)
    except Exception as exc:  # noqa: BLE001
        pf.check("R packages installed", False, str(exc))

    print("=" * 74)
    if pf.failures:
        print(f"DO NOT LAUNCH -- {len(pf.failures)} check(s) failed:")
        for f in pf.failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED -- cleared to launch")
    print(f"  manifest {sha[:16]}   commit {commit[:12]}")
    print("=" * 74)


if __name__ == "__main__":
    main()
