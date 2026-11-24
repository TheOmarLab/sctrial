"""The benchmark driver runs the phases its ``--phase`` argument names."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
PHASES = ["simulate", "sensitivity", "realdata", "ablation"]


def _driver(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("run_benchmark_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called = []
    for name in PHASES:
        monkeypatch.setattr(
            module, f"phase_{name}", lambda *a, _name=name, **k: called.append(_name)
        )
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "benchmark")
    return module, called


def test_phase_all_runs_the_four_benchmark_phases(monkeypatch, tmp_path):
    module, called = _driver(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--phase", "all"])
    module.main()
    assert called == PHASES


@pytest.mark.parametrize("phase", PHASES)
def test_one_phase_runs_alone(monkeypatch, tmp_path, phase):
    module, called = _driver(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--phase", phase])
    module.main()
    assert called == [phase]


def test_phase_validate_points_to_the_calibration_script(monkeypatch, tmp_path):
    module, called = _driver(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--phase", "validate"])
    with pytest.raises(SystemExit, match="calibrate_simulator.py"):
        module.main()
    assert called == []
