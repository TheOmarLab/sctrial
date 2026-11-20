"""The frame of a feature's model does not grow with the number of features in the call.

Each of these functions fits one model per feature. The frame of every feature held a column
for all features and was copied once per feature, so the time per feature grew with the
number of features.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from anndata import AnnData

from sctrial import TrialDesign
from sctrial.stats import heterogeneity, mixed_effects, timeseries

VISITS = ["V1", "V2", "V3"]
DESIGN = TrialDesign(celltype_col=None)


def _adata(n_genes: int) -> AnnData:
    rng = np.random.default_rng(5)
    rows = [
        (f"{arm[0]}{i}", arm, visit)
        for arm in ("Treated", "Control")
        for i in range(8)
        for visit in VISITS
        for _ in range(3)
    ]
    obs = pd.DataFrame(rows, columns=["participant_id", "arm", "visit"])
    obs.index = obs.index.astype(str)
    participants = sorted(obs["participant_id"].unique())
    obs["biomarker"] = obs["participant_id"].map(dict(zip(participants, rng.normal(size=16))))
    adata = AnnData(X=rng.normal(size=(len(obs), n_genes)), obs=obs)
    adata.var_names = [f"g{i:02d}" for i in range(n_genes)]
    return adata


def _run_heterogeneity(adata):
    heterogeneity.test_treatment_heterogeneity(
        adata, adata.var_names.tolist(), DESIGN, ("V1", "V3"), "biomarker"
    )


def _run_mixed(adata):
    mixed_effects.did_table_mixed(adata, adata.var_names.tolist(), DESIGN, ("V1", "V3"))


def _run_trend(adata):
    timeseries.trend_interaction(adata, adata.var_names.tolist(), DESIGN, VISITS)


def _record_frame_width(monkeypatch, owner, name, widths):
    """Replace a function that is given a model frame by one that records the frame width."""
    real = getattr(owner, name)

    def recording(*args, **kwargs):
        frame = next(v for v in (*args, *kwargs.values()) if isinstance(v, pd.DataFrame))
        widths.append(frame.shape[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, recording)


@pytest.mark.parametrize(
    "run,owner,name",
    [
        (_run_heterogeneity, smf, "ols"),
        (_run_mixed, mixed_effects, "did_mixed"),
        (_run_trend, smf, "ols"),
    ],
    ids=["heterogeneity", "mixed_effects", "trend_interaction"],
)
def test_model_frame_width_is_the_same_for_few_and_many_features(monkeypatch, run, owner, name):
    widths: dict[int, list[int]] = {10: [], 40: []}
    for n_genes, seen in widths.items():
        with monkeypatch.context() as patch:
            _record_frame_width(patch, owner, name, seen)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run(_adata(n_genes))
    assert len(widths[10]) == 10 and len(widths[40]) == 40
    assert set(widths[40]) == set(widths[10]) and len(set(widths[10])) == 1
