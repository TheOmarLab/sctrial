"""The runtime panel reads recorded measurements, and the influence panel is seeded."""

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from anndata import AnnData  # noqa: E402

import sctrial as st  # noqa: E402

pytest.importorskip("scanpy")


@pytest.fixture
def runtime(load_figure_module):
    load_figure_module("main/figure3_robustness_benchmarking")
    return sys.modules["manuscript_figures._runtime"]


def _paired_cells(n=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"participant_id": f"P{i}", "visit": visit, "sig_a": rng.normal(), "sig_b": rng.normal()}
            for i in range(n) for visit in ("Pre", "Post") for _ in range(3)]
    obs = pd.DataFrame(rows)
    for col in ("participant_id", "visit"):
        obs[col] = pd.Categorical(obs[col])
    obs.index = obs.index.astype(str)
    return AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def test_measure_times_every_replicate_and_the_table_round_trips(runtime, tmp_path):
    adata = _paired_cells()
    design = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None,
                            celltype_col=None)
    table = runtime.measure([("Toy", adata, design, ("Pre", "Post"), ["sig_a", "sig_b"], "paired")])
    assert list(table.columns) == runtime.COLUMNS
    assert len(table) == runtime.N_REPLICATES and (table["time_s"] > 0).all()
    assert table["method"].unique().tolist() == ["within_arm_comparison"]

    path = tmp_path / "dataset_runtime.csv"
    with open(path, "w") as fh:
        fh.write("# provenance line\n")
        table.to_csv(fh, index=False)
    loaded = runtime.load_recorded(path)
    pd.testing.assert_frame_equal(loaded, table, check_exact=False)

    runtime.check_recorded(loaded, "Toy", n_cells=adata.n_obs, n_participants=5, n_features=2)
    with pytest.raises(ValueError, match="Record again"):
        runtime.check_recorded(loaded, "Toy", n_cells=adata.n_obs + 1, n_participants=5, n_features=2)
    with pytest.raises(ValueError, match="no recorded runtime"):
        runtime.check_recorded(loaded, "Other", n_cells=1, n_participants=1, n_features=1)
    with pytest.raises(ValueError, match="lacks columns"):
        table.drop(columns="time_s").to_csv(path, index=False)
        runtime.load_recorded(path)


def test_influence_points_do_not_depend_on_the_global_random_state(load_figure_module):
    sf4 = load_figure_module("supp/supp_fig4_model_diagnostics")
    rng = np.random.default_rng(3)
    results = {
        name: {
            "residuals": pd.DataFrame({"cooks_d": rng.uniform(0, 0.3, 24),
                                       "pid": np.repeat([f"P{i}" for i in range(12)], 2)}),
            "effects": pd.DataFrame({"n_obs": [24.0, 24.0]}),
            "features": ["f1", "f2"],
            "cfg": {"participant_col": "pid"},
        }
        for name in ("TNBC", "Melanoma")
    }

    def offsets(global_seed):
        np.random.seed(global_seed)
        fig, ax = plt.subplots()
        sf4._panel_influence(ax, results)
        out = [c.get_offsets().copy() for c in ax.collections if len(c.get_offsets()) == 24]
        plt.close(fig)
        return out

    first, second = offsets(1), offsets(2)
    assert len(first) == 2
    assert all((a == b).all() for a, b in zip(first, second))
    for i, pts in enumerate(first):
        assert np.all(np.abs(pts[:, 0] - i) <= 0.2)


def test_a_missing_or_stale_recording_is_an_error_with_instructions(load_figure_module, tmp_path,
                                                                    monkeypatch):
    fig3 = load_figure_module("main/figure3_robustness_benchmarking")
    runtime = sys.modules["manuscript_figures._runtime"]
    with pytest.raises(FileNotFoundError, match="python -m manuscript_figures._runtime"):
        runtime.load_recorded(tmp_path / "absent.csv")

    recorded = pd.DataFrame([{"dataset": "TNBC", "design_type": "two_arm_did", "method": "did_table",
                              "n_cells": 100, "n_participants": 12, "n_features": 19,
                              "replicate": r, "time_s": 1.0 + r, "peak_mb": 5.0} for r in range(2)])
    monkeypatch.setattr(fig3, "load_recorded_runtime", lambda: recorded)
    fig3._check_runtime_recording({"TNBC": {"n_cells": 100, "n_participants": 12, "n_features": 19}})
    with pytest.raises(ValueError, match="Record again"):
        fig3._check_runtime_recording({"TNBC": {"n_cells": 101, "n_participants": 12, "n_features": 19}})
