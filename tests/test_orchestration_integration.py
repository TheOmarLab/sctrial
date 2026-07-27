"""End-to-end test of the DISTRIBUTED workflow, not of the statistics.

The statistical smoke test established that every method can fit 50-gene and
2,000-gene data. It caught neither of the two defects that made the pre-launch
audit return NO-GO, because neither was statistical:

* the frozen calibration's 6-versus-5 design leaked into every scenario, so
  ``null_n60`` simulated 11 participants while recording 60;
* the split producer jobs each wrote the combined results file, so the last to
  finish silently discarded three quarters of the grid.

Both live in the layer that turns a frozen simulator into scenario files and a
manuscript table. This test exercises that layer: real scenario construction ->
real simulation -> per-shard files -> the real aggregator -> the real figure
loader. It asserts what was REQUESTED against what was REALISED at every step.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sctrial.benchmark.orchestrator import _run_single_iteration, _scenario  # noqa: E402
from sctrial.benchmark.simulator_v2 import SCENARIO_OWNED_FIELDS  # noqa: E402

_GENES = 2500

# A frozen calibration carrying the anchor's design, exactly as the freeze writes
# it. If any of this leaks into a scenario the assertions below fail.
_FROZEN = {
    "arm_ratio": [6, 5],
    "n_per_arm": 11,
    "cells_per_pv_fixed": 999,
    "n_genes_transcriptome": _GENES,
    "panel_sizes": [50, 200, 500],
    "use_empirical_library": False,
    "use_empirical_cells_per_pv": False,
    "use_empirical_gene_rates": False,
    "use_empirical_dispersion": False,
    "dispersion_median": 0.3404,
    "prepost_corr": 0.0312,
}

# Six cheap scenarios, one per override the grid actually relies on.
_CASES = [
    ("anchor_6v5", dict(n_per_arm=11, arm_ratio=(6, 5), n_signal=0, cells_per_pv_fixed=25)),
    ("balanced_8v8", dict(n_per_arm=8, n_signal=0, cells_per_pv_fixed=25)),
    ("balanced_20v20", dict(n_per_arm=20, n_signal=0, cells_per_pv_fixed=25)),
    ("imbalance_4v9", dict(n_per_arm=13, arm_ratio=(4, 9), n_signal=0, cells_per_pv_fixed=25)),
    ("low_cell", dict(n_per_arm=8, n_signal=0, cells_per_pv_fixed=12)),
    ("mixed_signal", dict(n_per_arm=8, n_signal=10, cells_per_pv_fixed=25)),
    # OMITS cells_per_pv_fixed on purpose, mirroring the real null_hetero
    # scenarios. A scenario that specifies every field is protected by its own
    # explicit values; only an omitted field can be captured by a leaked frozen
    # one. This is the case that actually exercises the loader's field stripping
    # end to end -- without it the test passed even with the leak reintroduced,
    # because a second defence happened to cover for the first.
    ("inherits_cells", dict(n_per_arm=8, n_signal=0, cells_scale=1.0)),
]


@pytest.fixture(scope="module")
def distributed_run(tmp_path_factory):
    """Run the six scenarios as TWO shards, then the real aggregator."""
    out = tmp_path_factory.mktemp("grid")
    calibration_only = {k: v for k, v in _FROZEN.items() if k not in SCENARIO_OWNED_FIELDS}
    manifest = {"manifest_sha256": "a" * 64, "git_sha": "b" * 40}

    scenarios = [
        _scenario(
            name, name, design="two_arm", panel_size=50,
            use_empirical_library=False, use_empirical_cells_per_pv=False,
            use_empirical_gene_rates=False, use_empirical_dispersion=False,
            n_genes_transcriptome=_GENES, panel_sizes=(50, 200, 500), **kw,
        )
        for name, kw in _CASES
    ]

    # TWO shards writing into ONE directory -- the arrangement that broke.
    for shard in (scenarios[:3], scenarios[3:]):
        for sc in shard:
            rows = []
            for it in range(2):
                rows += _run_single_iteration(
                    (f"two_arm__{sc['name']}", it, 7000 + it, sc,
                     ["sctrial_did", "wilcoxon_paired"], calibration_only)
                )
            df = pd.DataFrame(rows)
            df["manifest_sha256"] = manifest["manifest_sha256"]
            df["git_sha"] = manifest["git_sha"]
            df.to_csv(out / f"two_arm__{sc['name']}.csv", index=False)
    return out, scenarios


def test_realised_design_matches_every_request(distributed_run):
    """The design simulated must equal the design asked for, per scenario.

    This is the assertion that would have caught B1. The results previously
    recorded the REQUESTED sample size, so a collapsed grid was undetectable
    downstream.
    """
    out, scenarios = distributed_run
    for sc, (name, kw) in zip(scenarios, _CASES):
        df = pd.read_csv(out / f"two_arm__{name}.csv")
        realised = int(df["n_participants"].iloc[0])
        nt, nc = int(df["n_treated"].iloc[0]), int(df["n_control"].iloc[0])
        expected = sum(kw["arm_ratio"]) if kw.get("arm_ratio") else 2 * kw["n_per_arm"]
        assert realised == expected, (
            f"{name}: requested {expected} participants, simulated {realised} — "
            "a design field leaked from the frozen calibration"
        )
        assert nt + nc == realised
        if kw.get("arm_ratio"):
            assert (nt, nc) == kw["arm_ratio"], f"{name}: arm sizes {(nt, nc)}"
        else:
            assert nt == nc == kw["n_per_arm"], f"{name}: arms {(nt, nc)} not balanced"


def test_realised_signal_matches_every_request(distributed_run):
    out, scenarios = distributed_run
    for sc, (name, kw) in zip(scenarios, _CASES):
        df = pd.read_csv(out / f"two_arm__{name}.csv")
        per_iter = df[df["method"] == "sctrial_did"].groupby("iteration")["is_signal"].sum()
        assert set(per_iter) == {kw["n_signal"]}, (
            f"{name}: requested {kw['n_signal']} signal genes, realised {set(per_iter)}"
        )
        assert set(df["panel_size"]) == {50}


def test_omitted_scenario_fields_are_not_captured_by_the_calibration(distributed_run):
    """A field a scenario does NOT set must not be supplied by the frozen config.

    `inherits_cells` deliberately omits `cells_per_pv_fixed`, exactly as the real
    `null_hetero` scenarios do, and the frozen configuration carries
    `cells_per_pv_fixed=999`. If the loader failed to strip scenario-owned fields
    that 999 would win and the scenario's intended empirical cell yield would
    vanish.

    This is the only case here that can detect a broken loader: a scenario
    specifying every field is protected by its own explicit values regardless.
    """
    out, _ = distributed_run
    df = pd.read_csv(out / "two_arm__inherits_cells.csv")
    assert int(df["n_participants"].iloc[0]) == 16

    # The frozen 999 cells/visit must not be what was simulated. With
    # cells_scale=1.0 and no empirical pool the parametric fallback is used, whose
    # mean is far from 999.
    assert "cells_per_pv_median" in df.columns, (
        "results carry no realised cell yield, so a leaked calibration value "
        "cannot be detected — and an assertion guarded by `if column in df` "
        "silently skips, which is worse than no assertion"
    )
    med = float(df["cells_per_pv_median"].iloc[0])
    assert abs(med - 999) > 1, (
        f"the frozen cells_per_pv_fixed=999 reached a scenario that omitted it "
        f"(realised median {med} cells/visit)"
    )


def test_aggregator_refuses_an_incomplete_grid(distributed_run, monkeypatch):
    """The aggregator must reject a missing shard rather than combining it."""
    out, scenarios = distributed_run
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_agg", ROOT / "scripts" / "aggregate_benchmark.py"
    )
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)
    expected = {f"two_arm__{n}" for n, _ in _CASES}
    monkeypatch.setattr(agg, "_expected_scenarios", lambda grid: expected)

    # Complete -> succeeds and writes the completion record.
    agg.aggregate(out, "core", "benchmark_combined.csv", expect_iterations=2)
    assert (out / "benchmark_combined.csv").exists()
    record = json.loads((out / "benchmark_complete.json").read_text())
    assert record["n_scenarios"] == len(_CASES)
    combined = pd.read_csv(out / "benchmark_combined.csv")
    assert set(combined["scenario"].unique()) == expected, (
        "the combined file does not contain every shard — this is B2"
    )

    # Remove one shard -> must REFUSE.
    hidden = out / "two_arm__low_cell.csv"
    hidden.rename(out / "hidden.bak")
    (out / "benchmark_combined.csv").unlink()
    try:
        with pytest.raises(SystemExit, match="MISSING"):
            agg.aggregate(out, "core", "benchmark_combined.csv", expect_iterations=2)
        assert not (out / "benchmark_combined.csv").exists(), (
            "the aggregator wrote a combined file for an incomplete grid"
        )
    finally:
        (out / "hidden.bak").rename(hidden)


def test_aggregator_refuses_mixed_manifests(distributed_run, monkeypatch):
    """Two benchmark runs must never be averaged together."""
    import importlib.util

    out, _ = distributed_run
    spec = importlib.util.spec_from_file_location(
        "_agg2", ROOT / "scripts" / "aggregate_benchmark.py"
    )
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)
    monkeypatch.setattr(
        agg, "_expected_scenarios", lambda grid: {f"two_arm__{n}" for n, _ in _CASES}
    )

    target = out / "two_arm__mixed_signal.csv"
    original = target.read_text()
    df = pd.read_csv(target)
    df["manifest_sha256"] = "c" * 64          # a different run
    df.to_csv(target, index=False)
    try:
        with pytest.raises(SystemExit, match="manifest"):
            agg.aggregate(out, "core", "benchmark_combined.csv", expect_iterations=2)
    finally:
        target.write_text(original)


def test_the_whole_chain_runs_under_the_real_scripts():
    """The aggregator must be invocable exactly as the SLURM job invokes it."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aggregate_benchmark.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "core" in r.stdout and "sensitivity" in r.stdout
