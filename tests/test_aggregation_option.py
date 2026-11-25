"""Every table that takes ``agg`` applies it: mean, median, or the share of positive cells."""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sctrial import TrialDesign
from sctrial.stats.comparisons import (
    between_arm_comparison,
    get_between_arm_aggregated_df,
    get_within_arm_aggregated_df,
    within_arm_comparison,
)
from sctrial.stats.did import did_table
from sctrial.stats.heterogeneity import test_treatment_heterogeneity as heterogeneity
from sctrial.stats.mixed_effects import did_table_mixed

GENES = ["G0", "G1", "G2"]
VISITS = ("V1", "V2")


def _counts(n_per_arm=8, n_cells=9, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for arm in ("Treated", "Control"):
        for p in range(n_per_arm):
            for visit in VISITS:
                rows.extend((f"{arm[0]}{p:02d}", arm, visit) for _ in range(n_cells))
    obs = pd.DataFrame(rows, columns=["participant_id", "arm", "visit"])
    obs.index = [f"c{i}" for i in range(len(obs))]
    pids = sorted(obs["participant_id"].unique())
    codes = obs["participant_id"].map({p: i for i, p in enumerate(pids)}).to_numpy()
    rate = rng.uniform(0.3, 1.5, (len(pids), len(GENES)))[codes]
    rate = rate * np.where(((obs["arm"] == "Treated") & (obs["visit"] == "V2")).to_numpy()[:, None], 1.8, 1.0)
    X = rng.poisson(rate).astype(float)
    X[5, 1] = np.nan  # a cell without a value is not a cell of the summary
    obs["biomarker"] = obs["participant_id"].map({p: float(i % 2) for i, p in enumerate(pids)})
    obs["score"] = X[:, 2] + rng.poisson(0.5, len(obs))  # a feature kept in obs
    obs["load"] = rng.normal(size=len(obs))  # a covariate that varies from cell to cell
    obs["celltype"] = np.where(np.arange(len(obs)) % 3 == 0, "A", "B")
    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col=None,
    )
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES)), design


def _summarised_by_hand(adata, how, by=("participant_id", "visit")):
    """The same cells, each holding a value whose mean over the cells of ``by`` is ``how``."""
    X = np.column_stack([adata.X, adata.obs["score"].to_numpy()])
    keys = adata.obs[list(by)].astype(str).agg("|".join, axis=1)
    if how == "pct_pos":
        X = np.where(np.isnan(X), np.nan, (X > 0) * 100.0)
    else:
        missing = np.isnan(X)
        X = pd.DataFrame(X).groupby(keys.to_numpy()).transform("median").to_numpy().copy()
        X[missing] = np.nan
    obs = adata.obs.copy()
    obs["score"] = X[:, -1]
    if how == "median":
        # a numeric covariate follows the median; under pct_pos it keeps its mean
        obs["load"] = obs.groupby(keys.to_numpy())["load"].transform("median")
    return ad.AnnData(X=X[:, :-1], obs=obs, var=adata.var.copy())


def _tables(adata, design, agg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "within": within_arm_comparison(adata, arm="Treated", features=GENES, design=design,
                                            visits=VISITS, agg=agg),
            "between_ols": between_arm_comparison(adata, visit="V2", features=GENES, design=design, agg=agg),
            "between_rank": between_arm_comparison(adata, visit="V2", features=GENES, design=design,
                                                   method="wilcoxon", agg=agg),
            "did": did_table(adata, GENES, design, VISITS, agg=agg).sort_values("feature").reset_index(drop=True),
            "heterogeneity": heterogeneity(adata, [*GENES, "score"], design, VISITS, "biomarker", agg=agg),
            "did_obs": did_table(adata, ["score"], design, VISITS, agg=agg),
            "within_obs": within_arm_comparison(adata, arm="Control", features=["score"], design=design,
                                                visits=VISITS, agg=agg),
            "mixed": did_table_mixed(adata, GENES, design, VISITS, agg=agg),
            "did_cov": did_table(adata, GENES, design, VISITS, agg=agg, covariates=["load"])
            .sort_values("feature")
            .reset_index(drop=True),
            "within_cov": within_arm_comparison(adata, arm="Treated", features=GENES, design=design,
                                                visits=VISITS, agg=agg, covariates=["load"]),
            "between_cov": between_arm_comparison(adata, visit="V2", features=GENES, design=design, agg=agg,
                                                  covariates=["load"]),
            "heterogeneity_cov": heterogeneity(adata, GENES, design, VISITS, "biomarker", agg=agg,
                                               covariates=["load"]),
            "mixed_cov": did_table_mixed(adata, GENES, design, VISITS, agg=agg, covariates=["load"]),
            "within_frame": get_within_arm_aggregated_df(adata, arm="Treated", features=GENES, design=design,
                                                         visits=VISITS, agg=agg)[0],
            "between_frame": get_between_arm_aggregated_df(adata, visit="V2", features=GENES, design=design,
                                                           agg=agg),
        }


@pytest.mark.parametrize("agg", ["pct_pos", "median"])
def test_each_table_applies_the_requested_aggregation(agg):
    adata, design = _counts()
    got = _tables(adata, design, agg)
    want = _tables(_summarised_by_hand(adata, agg), design, "mean")
    plain = _tables(adata, design, "mean")

    for name in got:
        # the mixed model is fitted by numerical optimisation, which stops within its own tolerance
        rtol = 1e-5 if name.startswith("mixed") else 1e-10
        pd.testing.assert_frame_equal(got[name], want[name], check_exact=False, rtol=rtol, obj=name)
        numeric = got[name].select_dtypes("number").columns
        assert not np.allclose(
            got[name][numeric].to_numpy(dtype=float), plain[name][numeric].to_numpy(dtype=float), equal_nan=True
        ), f"{name} ignores agg={agg!r}"


@pytest.mark.parametrize("agg", ["pct_pos", "median"])
def test_aggregation_within_cell_types_applies_the_requested_aggregation(agg):
    adata, design = _counts()
    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col="celltype",
    )
    by_hand = _summarised_by_hand(adata, agg, by=("participant_id", "visit", "celltype"))
    kw = dict(features=GENES, design=design, visits=VISITS, aggregate="participant_visit_celltype")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = did_table(adata, agg=agg, **kw).sort_values("feature").reset_index(drop=True)
        want = did_table(by_hand, agg="mean", **kw).sort_values("feature").reset_index(drop=True)
        plain = did_table(adata, agg="mean", **kw).sort_values("feature").reset_index(drop=True)

    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-10)
    assert not np.allclose(got["beta_DiD"], plain["beta_DiD"])


def test_share_of_positive_cells_is_taken_over_the_cells_with_a_value():
    adata, design = _counts()
    frame = get_between_arm_aggregated_df(adata, visit="V1", features=["G1"], design=design, agg="pct_pos")
    first = frame.set_index("participant_id").loc["T00", "G1"]
    cells = adata.X[((adata.obs["participant_id"] == "T00") & (adata.obs["visit"] == "V1")).to_numpy(), 1]

    assert np.isnan(cells).sum() == 1
    assert first == pytest.approx(100.0 * np.mean(cells[~np.isnan(cells)] > 0))


def test_an_unknown_aggregation_is_refused():
    adata, design = _counts()
    with pytest.raises(ValueError, match="Unsupported agg"):
        within_arm_comparison(adata, arm="Treated", features=GENES, design=design, visits=VISITS, agg="max")
