"""End-to-end test of the DISTRIBUTED workflow, not of the statistics.

The statistical smoke test established that every method can fit 50-gene and
2,000-gene data. It caught neither of the two defects that made the pre-launch
audit return NO-GO, because neither was statistical:

* the frozen calibration's 6-versus-5 design leaked into every scenario, so
  ``null_n60`` simulated 11 participants while recording 60;
* the split producer jobs each wrote the combined results file, so the last to
  finish silently discarded three quarters of the grid.

Both live in the layer that turns a frozen simulator into scenario files and a
manuscript table. This test exercises that layer through the REAL production
driver -- `_run_grid`, not a hand-rolled loop -- so scenario construction,
simulation, the contract validator, per-scenario completion records, the
aggregator and the figure loader are all the code that will run on HPC.

The first version of this test PASSED with the design leak reintroduced. Every
scenario specified every field, so each was protected by its own explicit values
and only an OMITTED field could be captured by a leaked one. That is why
`inherits_cells` exists, and why the assertions below read realised columns
rather than the configuration they were handed.
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

from sctrial.benchmark.orchestrator import _run_grid, _scenario  # noqa: E402
from sctrial.benchmark.paths import ResultLayout  # noqa: E402
from sctrial.benchmark.simulator_v2 import SCENARIO_OWNED_FIELDS  # noqa: E402

_GENES = 2500
_SHA = "a" * 64

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

# The coverage the pre-launch review asked for: the TNBC-matched anchor design,
# the small and large ends of the sample-size axis, an imbalanced allocation, the
# low-cell stress condition, heterogeneous cell yield, and a mixed-signal cell.
_CASES = [
    ("anchor_6v5", dict(n_per_arm=11, arm_ratio=(6, 5), n_signal=0, cells_per_pv_fixed=20)),
    ("balanced_8v8", dict(n_per_arm=8, n_signal=0, cells_per_pv_fixed=20)),
    ("balanced_60v60", dict(n_per_arm=60, n_signal=0, cells_per_pv_fixed=8)),
    ("imbalance_10v20", dict(n_per_arm=30, arm_ratio=(10, 20), n_signal=0, cells_per_pv_fixed=8)),
    ("low_cell_50", dict(n_per_arm=8, n_signal=0, cells_per_pv_fixed=50)),
    ("mixed_signal", dict(n_per_arm=8, n_signal=10, cells_per_pv_fixed=20)),
    # OMITS cells_per_pv_fixed on purpose, mirroring the real null_hetero
    # scenarios. A scenario that specifies every field is protected by its own
    # explicit values; only an omitted field can be captured by a leaked frozen
    # one. This is the only case here that can detect a broken loader.
    ("inherits_cells", dict(n_per_arm=8, n_signal=0, cells_scale=1.0)),
]

_EXPECTED = {f"two_arm__{n}" for n, _ in _CASES}


def _build(kw: dict) -> dict:
    name = kw.pop("_name")
    return _scenario(
        name, name, design="two_arm", panel_size=50,
        use_empirical_library=False, use_empirical_cells_per_pv=False,
        use_empirical_gene_rates=False, use_empirical_dispersion=False,
        n_genes_transcriptome=_GENES, panel_sizes=(50, 200, 500), **kw,
    )


@pytest.fixture(scope="module")
def distributed_run(tmp_path_factory):
    """Run the seven scenarios as TWO shards through the real grid driver."""
    root = tmp_path_factory.mktemp("results")
    layout = ResultLayout(root, _SHA).create()
    calibration_only = {k: v for k, v in _FROZEN.items() if k not in SCENARIO_OWNED_FIELDS}
    manifest = {"manifest_sha256": _SHA, "git_sha": "b" * 40}

    scenarios = [_build({"_name": n, **kw}) for n, kw in _CASES]

    # TWO shards writing into ONE directory -- the arrangement that broke.
    for shard in (scenarios[:3], scenarios[3:]):
        _run_grid(
            grid_fn=lambda _design, _s=shard: _s,
            designs=["two_arm"],
            methods=["sctrial_did", "wilcoxon_paired"],
            n_iterations=2,
            n_jobs=1,
            output_dir=layout.scenarios,
            resume=False,
            combined_name="unused.csv",
            seed=7000,
            base_config=calibration_only,
            manifest=manifest,
            adaptive=False,
        )
    return layout, scenarios


def _agg():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_agg", ROOT / "scripts" / "aggregate_benchmark.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_realised_design_matches_every_request(distributed_run):
    """The design simulated must equal the design asked for, per scenario.

    This is the assertion that would have caught B1. The results previously
    recorded the REQUESTED sample size, so a collapsed grid was undetectable
    downstream.
    """
    layout, _ = distributed_run
    for name, kw in _CASES:
        df = pd.read_csv(layout.scenario_csv(f"two_arm__{name}"))
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
    layout, _ = distributed_run
    for name, kw in _CASES:
        df = pd.read_csv(layout.scenario_csv(f"two_arm__{name}"))
        per_iter = df[df["method"] == "sctrial_did"].groupby("iteration")["is_signal"].sum()
        assert set(per_iter) == {kw["n_signal"]}, (
            f"{name}: requested {kw['n_signal']} signal genes, realised {set(per_iter)}"
        )
        assert set(df["panel_size"]) == {50}


def test_realised_cell_yield_matches_every_request(distributed_run):
    """Fixed yields exactly; an omitted yield must NOT be a constant.

    The second half is the design leak seen from the other direction: a scenario
    that asked for a distribution and received one repeated value did not get
    what it asked for.
    """
    layout, _ = distributed_run
    for name, kw in _CASES:
        df = pd.read_csv(layout.scenario_csv(f"two_arm__{name}"))
        lo = float(df["cells_per_pv_min"].min())
        hi = float(df["cells_per_pv_max"].max())
        want = kw.get("cells_per_pv_fixed")
        if want is not None:
            assert lo == hi == want, f"{name}: requested {want} cells/visit, got [{lo}, {hi}]"
        else:
            assert lo < hi, (
                f"{name}: requested the empirical distribution but every "
                f"participant-visit has {lo} cells — a fixed value leaked in"
            )


def test_omitted_scenario_fields_are_not_captured_by_the_calibration(distributed_run):
    """A field a scenario does NOT set must not be supplied by the frozen config.

    `inherits_cells` deliberately omits `cells_per_pv_fixed`, exactly as the real
    `null_hetero` scenarios do, and the frozen configuration carries
    `cells_per_pv_fixed=999`. If the loader failed to strip scenario-owned fields
    that 999 would win and the scenario's intended cell yield would vanish.
    """
    layout, _ = distributed_run
    df = pd.read_csv(layout.scenario_csv("two_arm__inherits_cells"))
    assert int(df["n_participants"].iloc[0]) == 16

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


def test_every_scenario_writes_a_completion_record(distributed_run):
    """Completion is a record written last, not a row count.

    Adaptive stopping only ever ADDS replicates, so a count threshold is
    satisfied by a scenario killed part-way through an extension. The record is
    what distinguishes them.
    """
    layout, _ = distributed_run
    records = layout.completed_scenarios()
    assert set(records) == _EXPECTED, f"missing completion records: {_EXPECTED - set(records)}"
    assert layout.orphan_scenarios() == []
    for name, rec in records.items():
        assert rec["n_replicates_completed"] == 2, name
        assert rec["manifest_sha256"] == _SHA, name
        assert rec["stop_reason"] == "adaptive_disabled", (
            f"{name}: adaptation was off, so any other stop reason is a claim the "
            "run cannot support"
        )
        assert "evaluability" in rec and rec["evaluability"], name


def test_aggregator_refuses_an_incomplete_grid(distributed_run, monkeypatch):
    """The aggregator must reject a missing shard rather than combining it."""
    layout, _ = distributed_run
    agg = _agg()
    monkeypatch.setattr(agg, "_expected_scenarios", lambda grid: _EXPECTED)

    agg.aggregate(layout, "core", "benchmark_combined.csv", 2, allow_non_adaptive=True)
    target = layout.combined_csv("benchmark_combined.csv")
    assert target.exists()
    record = json.loads(layout.completion_marker("core").read_text())
    assert record["n_scenarios"] == len(_CASES)
    combined = pd.read_csv(target)
    assert set(combined["scenario"].unique()) == _EXPECTED, (
        "the combined file does not contain every shard — this is B2"
    )

    hidden = layout.scenario_csv("two_arm__low_cell_50")
    bak = hidden.with_suffix(".bak")
    hidden.rename(bak)
    target.unlink()
    try:
        with pytest.raises(SystemExit, match="MISSING"):
            agg.aggregate(layout, "core", "benchmark_combined.csv", 2, allow_non_adaptive=True)
        assert not target.exists(), (
            "the aggregator wrote a combined file for an incomplete grid"
        )
    finally:
        bak.rename(hidden)


def test_aggregator_refuses_a_scenario_with_no_completion_record(distributed_run, monkeypatch):
    """A CSV without a record is a truncated scenario, not a complete one."""
    layout, _ = distributed_run
    agg = _agg()
    monkeypatch.setattr(agg, "_expected_scenarios", lambda grid: _EXPECTED)

    rec = layout.scenario_completion("two_arm__mixed_signal")
    original = rec.read_text()
    rec.unlink()
    try:
        with pytest.raises(SystemExit, match="NO completion record"):
            agg.aggregate(layout, "core", "benchmark_combined.csv", 2, allow_non_adaptive=True)
    finally:
        rec.write_text(original)


def test_aggregator_refuses_non_adaptive_shards_by_default(distributed_run, monkeypatch):
    """A debug run must not be able to become a definitive result."""
    layout, _ = distributed_run
    agg = _agg()
    monkeypatch.setattr(agg, "_expected_scenarios", lambda grid: _EXPECTED)

    with pytest.raises(SystemExit, match="adaptive Monte Carlo was off"):
        agg.aggregate(layout, "core", "benchmark_combined.csv", 2)


def test_aggregator_refuses_an_unjustified_precision_claim(distributed_run, monkeypatch):
    """`precision_reached` is verified against achieved MCSE, not taken on trust.

    Without this the stop reason is unfalsifiable: a scenario truncated at any
    point could simply be labelled as having stopped early.
    """
    layout, _ = distributed_run
    agg = _agg()
    monkeypatch.setattr(agg, "_expected_scenarios", lambda grid: _EXPECTED)

    rec_path = layout.scenario_completion("two_arm__mixed_signal")
    original = rec_path.read_text()
    rec = json.loads(original)
    rec["stop_reason"] = "precision_reached"
    rec["fpr_mcse"] = 0.25            # nowhere near the target
    rec["mcse_target_fpr"] = 0.005
    rec_path.write_text(json.dumps(rec))
    try:
        with pytest.raises(SystemExit, match="claims precision_reached"):
            agg.aggregate(layout, "core", "benchmark_combined.csv", 2, allow_non_adaptive=True)
    finally:
        rec_path.write_text(original)


def test_aggregator_refuses_a_record_disagreeing_with_its_csv(distributed_run, monkeypatch):
    """A record and its CSV must agree on how many replicates exist."""
    layout, _ = distributed_run
    agg = _agg()
    monkeypatch.setattr(agg, "_expected_scenarios", lambda grid: _EXPECTED)

    rec_path = layout.scenario_completion("two_arm__balanced_8v8")
    original = rec_path.read_text()
    rec = json.loads(original)
    rec["n_replicates_completed"] = 999
    rec_path.write_text(json.dumps(rec))
    try:
        with pytest.raises(SystemExit, match="record says 999"):
            agg.aggregate(layout, "core", "benchmark_combined.csv", 2, allow_non_adaptive=True)
    finally:
        rec_path.write_text(original)


def test_aggregator_refuses_mixed_manifests(distributed_run, monkeypatch):
    """Two benchmark runs must never be averaged together."""
    layout, _ = distributed_run
    agg = _agg()
    monkeypatch.setattr(agg, "_expected_scenarios", lambda grid: _EXPECTED)

    target = layout.scenario_csv("two_arm__mixed_signal")
    original = target.read_text()
    df = pd.read_csv(target)
    df["manifest_sha256"] = "c" * 64          # a different run
    df.to_csv(target, index=False)
    try:
        with pytest.raises(SystemExit, match="manifest"):
            agg.aggregate(layout, "core", "benchmark_combined.csv", 2, allow_non_adaptive=True)
    finally:
        target.write_text(original)


def test_results_are_addressed_by_manifest_not_by_latest(tmp_path):
    """No 'latest', no newest-file, no search across manifests."""
    from sctrial.benchmark.paths import require_layout

    root = tmp_path / "results"
    ResultLayout(root, "d" * 64).create()
    with pytest.raises(FileNotFoundError, match="no results for manifest"):
        require_layout(root, "e" * 64)
    # The error must name what IS available rather than silently picking one.
    try:
        require_layout(root, "e" * 64)
    except FileNotFoundError as exc:
        assert "d" * 64 in str(exc)

    with pytest.raises(ValueError, match="hex digest"):
        ResultLayout(root, "latest")


def test_the_whole_chain_runs_under_the_real_scripts():
    """The aggregator must be invocable exactly as the SLURM job invokes it."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aggregate_benchmark.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "core" in r.stdout and "sensitivity" in r.stdout
    assert "manifest_sha" in r.stdout, (
        "the aggregator must take the manifest explicitly; resolving it by "
        "convention is how stale results were read"
    )


def test_two_grids_do_not_overwrite_each_others_completion_marker(tmp_path):
    """Core and sensitivity aggregate into ONE manifest directory.

    They run as separate jobs, so a single shared completion marker would be
    written twice and the survivor would attest to whichever finished last --
    the same last-writer-wins defect as the combined file itself, one level up.
    A figure checking that marker would then see a complete record while half the
    benchmark was missing.
    """
    layout = ResultLayout(tmp_path / "results", "9" * 64).create()
    core = layout.completion_marker("core")
    sens = layout.completion_marker("sensitivity")
    assert core != sens, (
        "both grids write the same completion marker; the second silently "
        "replaces the first"
    )
    core.write_text('{"grid": "core"}')
    sens.write_text('{"grid": "sensitivity"}')
    assert json.loads(core.read_text())["grid"] == "core"
    assert json.loads(sens.read_text())["grid"] == "sensitivity"

    # Their combined CSVs must not collide either.
    assert layout.combined_csv("benchmark_combined.csv") != layout.combined_csv(
        "sensitivity_combined.csv"
    )
