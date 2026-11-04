"""Regression test for the Figure 3 pre-specified endpoint effect sizes.

The cross-sectional branch grouped a categorical participant column without
``observed=True``, so every arm was sized by the whole cohort's category list.
In the COVID-19 cohort that reported n = 68 for a comparison of 16 participants.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st

ROOT = Path(__file__).resolve().parent.parent
SIG = "sig_T Cell Activation"


@pytest.fixture
def figure3(tmp_path, monkeypatch):
    monkeypatch.setenv("SCTRIAL_MANUSCRIPT_DIR", str(tmp_path))
    saved = {k: v for k, v in sys.modules.items() if k.startswith("manuscript_figures")}
    for k in saved:
        del sys.modules[k]
    packages = [
        ("manuscript_figures", "manuscript_figures/__init__.py", ["manuscript_figures"]),
        ("manuscript_figures.main", "manuscript_figures/main/__init__.py",
         ["manuscript_figures/main"]),
    ]
    for name, rel, subs in packages:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / rel, submodule_search_locations=[str(ROOT / s) for s in subs]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    name = "manuscript_figures.main.figure3_robustness_benchmarking"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "manuscript_figures/main/figure3_robustness_benchmarking.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_load_cache", lambda tag: None)
    monkeypatch.setattr(mod, "_save_cache", lambda tag, obj: None)
    yield mod
    for k in [k for k in sys.modules if k.startswith("manuscript_figures")]:
        del sys.modules[k]
    sys.modules.update(saved)


def _cross_sectional_cohort():
    """Ten enrolled participants, of whom 3 Severe and 4 Mild were sampled in the bin."""
    rng = np.random.default_rng(0)
    present = {"P0": "Severe", "P1": "Severe", "P2": "Severe",
               "P3": "Mild", "P4": "Mild", "P5": "Mild", "P6": "Mild"}
    rows = []
    for pid, arm in present.items():
        shift = 0.5 if arm == "Severe" else 0.0
        for value in rng.normal(shift, 1.0, size=int(rng.integers(5, 40))):
            rows.append({"participant_id": pid, "arm": arm, "dfo_bin": "DFO_8-14", SIG: value})
    obs = pd.DataFrame(rows)
    obs["participant_id"] = pd.Categorical(
        obs["participant_id"], categories=[f"P{i}" for i in range(10)]
    )
    obs.index = obs.index.astype(str)
    design = st.TrialDesign(
        participant_col="participant_id", visit_col="dfo_bin", arm_col="arm",
        arm_treated="Severe", arm_control="Mild", celltype_col=None,
    )
    return AnnData(X=np.zeros((len(obs), 1)), obs=obs), design


def test_cross_sectional_effect_counts_only_sampled_participants(figure3):
    adata, design = _cross_sectional_cohort()
    out = figure3._compute_effect_sizes_across_datasets(
        [("COVID-19", adata, design, ("DFO_8-14",), [SIG], "cross_sectional")]
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_participants"] == 7

    pb = adata.obs.groupby("participant_id", observed=True).agg(
        arm=("arm", "first"), score=(SIG, "mean")
    )
    t, c = pb.loc[pb["arm"] == "Severe", "score"], pb.loc[pb["arm"] == "Mild", "score"]
    pooled = np.sqrt((2 * t.var() + 3 * c.var()) / 5)
    assert row["d"] == pytest.approx((t.mean() - c.mean()) / pooled, abs=1e-12)
