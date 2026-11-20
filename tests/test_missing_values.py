"""A row without a value for a variable of a model is not in that model.

Each function below fits one model per feature on a frame that can hold missing values. The
rule is that of ``did_fit``: the row leaves the model, a participant needs every visit of the
design for the feature, and the count reported is the number of participants fitted. Every
test compares the result with the one obtained after removing the rows or the participant.
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sctrial import TrialDesign
from sctrial.stats.abundance import abundance_did
from sctrial.stats.comparisons import (
    between_arm_comparison,
    compare_gene_in_celltype,
    get_within_arm_aggregated_df,
    within_arm_comparison,
    within_arm_fit_beta,
)
from sctrial.stats.cv import loo_cv_did
from sctrial.stats.timeseries import test_parallel_trends as parallel_trends
from sctrial.stats.timeseries import trend_interaction

GENES = ["G0", "G1", "G2"]


def _cohort(n_per_arm=6, n_cells=4, visits=("V1", "V2"), seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for arm in ("Treated", "Control"):
        for p in range(n_per_arm):
            for visit in visits:
                rows.extend((f"{arm[0]}{p:02d}", arm, visit) for _ in range(n_cells))
    obs = pd.DataFrame(rows, columns=["participant_id", "arm", "visit"])
    obs.index = [f"c{i}" for i in range(len(obs))]
    pids = sorted(obs["participant_id"].unique())
    codes = obs["participant_id"].map({p: i for i, p in enumerate(pids)}).to_numpy()
    step = obs["visit"].map({v: i for i, v in enumerate(visits)}).to_numpy(dtype=float)
    X = rng.normal(size=(len(obs), len(GENES))) + rng.normal(0, 0.5, (len(pids), len(GENES)))[codes]
    X += np.outer(step * (obs["arm"] == "Treated").to_numpy(), [0.8, 0.4, 0.0])
    # one value per participant and visit
    load = rng.normal(size=(len(pids), len(visits)))
    obs["load"] = load[codes, step.astype(int)]
    obs["celltype"] = np.where(rng.random(len(obs)) < 0.5, "A", "B")
    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col="celltype",
    )
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES)), design


def _with_values(adata, column, rows, value=np.nan):
    """Copy of ``adata`` with a gene (by position) or an obs column set to ``value`` in ``rows``."""
    X, obs = adata.X.copy(), adata.obs.copy()
    if isinstance(column, int):
        X[rows, column] = value
    else:
        obs.loc[obs.index[rows], column] = value
    return ad.AnnData(X=X, obs=obs, var=adata.var.copy())


def _rows(adata, participant, visit=None):
    obs = adata.obs
    mask = (obs["participant_id"] == participant).to_numpy()
    if visit is not None:
        mask &= (obs["visit"] == visit).to_numpy()
    return mask


def _same(row, reference, keys):
    for key in keys:
        assert row[key] == pytest.approx(reference[key], rel=1e-9, abs=1e-12, nan_ok=True), key


def _feature(table, name="G1"):
    return table.set_index("feature").loc[name]


WITHIN_KEYS = ("beta_time", "se_time", "p_time", "ci_lo_time", "ci_hi_time", "resid_sd", "n_units")


@pytest.mark.parametrize("use_bootstrap", [False, True])
@pytest.mark.parametrize("standardize", [True, False])
def test_within_arm_leaves_out_a_participant_without_the_feature_at_one_visit(standardize, use_bootstrap):
    # The other visit stayed in the model: it was counted as a participant and as a cluster,
    # and its value entered the scaling of the outcome.
    adata, design = _cohort()
    holed = _with_values(adata, 1, _rows(adata, "T00", "V2"))
    kw = dict(arm="Treated", features=GENES, design=design, visits=("V1", "V2"),
              standardize=standardize, use_bootstrap=use_bootstrap, n_boot=199)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = within_arm_comparison(holed, **kw)
        ref = within_arm_comparison(adata[~_rows(adata, "T00")].copy(), **kw)
        complete = within_arm_comparison(adata, **kw)

    keys = WITHIN_KEYS + (("p_time_boot", "ci_lo_boot", "ci_hi_boot") if use_bootstrap else ())
    assert _feature(res)["n_units"] == 5
    _same(_feature(res), _feature(ref), keys)
    assert _feature(res, "G0")["n_units"] == 6
    _same(_feature(res, "G0"), _feature(complete, "G0"), keys)


def test_within_arm_leaves_out_a_participant_without_a_covariate_at_one_visit():
    adata, design = _cohort()
    holed = _with_values(adata, "load", _rows(adata, "T00", "V2"))
    kw = dict(arm="Treated", features=GENES, design=design, visits=("V1", "V2"), covariates=["load"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = within_arm_comparison(holed, **kw)
        ref = within_arm_comparison(adata[~_rows(adata, "T00")].copy(), **kw)

    assert _feature(res)["n_units"] == 5
    _same(_feature(res), _feature(ref), WITHIN_KEYS)


def test_between_arm_rank_test_uses_the_participants_that_have_the_feature():
    # The missing value went into the rank test: every statistic missing, the participant counted.
    adata, design = _cohort()
    holed = _with_values(adata, 1, _rows(adata, "T00", "V2"))
    kw = dict(visit="V2", features=GENES, design=design, method="wilcoxon")

    res = between_arm_comparison(holed, **kw)
    ref = between_arm_comparison(adata[~_rows(adata, "T00")].copy(), **kw)

    assert _feature(res)["n_units"] == 11
    assert np.isfinite(_feature(res)["p_arm"])
    _same(_feature(res), _feature(ref), ("beta_arm", "se_arm", "ci_lo_arm", "ci_hi_arm", "p_arm", "n_units"))
    assert _feature(res, "G0")["n_units"] == 12


def test_between_arm_model_scales_the_outcome_on_the_rows_it_fits():
    adata, design = _cohort()
    holed = _with_values(adata, "load", _rows(adata, "T00", "V2"))
    kw = dict(visit="V2", features=GENES, design=design, covariates=["load"])

    res = between_arm_comparison(holed, **kw)
    ref = between_arm_comparison(adata[~_rows(adata, "T00")].copy(), **kw)

    assert _feature(res)["n_units"] == 11
    _same(_feature(res), _feature(ref), ("beta_arm", "se_arm", "ci_lo_arm", "ci_hi_arm", "p_arm", "n_units"))


def test_within_arm_estimator_for_permutations_is_that_of_the_table():
    adata, design = _cohort()
    holed = _with_values(adata, 1, _rows(adata, "T00", "V2"))
    visits = ("V1", "V2")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = within_arm_comparison(holed, arm="Treated", features=["G1"], design=design, visits=visits)
    frame, unit = get_within_arm_aggregated_df(
        holed, arm="Treated", features=["G1"], design=design, visits=visits
    )

    assert within_arm_fit_beta(frame, "G1", unit) == pytest.approx(table["beta_time"].iloc[0], rel=1e-10)


def test_gene_comparison_in_a_cell_type_drops_the_cells_without_a_value():
    adata, _ = _cohort(n_cells=12)
    cell = np.flatnonzero(_rows(adata, "T00", "V1"))[0]
    kw = dict(gene="G1", celltypes=["A", "B"], group_col="arm", group1="Treated", group2="Control",
              layer=None, log1p=False, min_cells_per_patient=5)

    res, per_participant = compare_gene_in_celltype(_with_values(adata, 1, [cell]), **kw)
    keep = np.ones(adata.n_obs, dtype=bool)
    keep[cell] = False
    ref, ref_participant = compare_gene_in_celltype(adata[keep].copy(), **kw)

    assert np.isfinite(res["p_value"])
    _same(res, ref, ("p_value", "mean_group1", "mean_group2", "delta", "n_group1", "n_group2"))
    pd.testing.assert_frame_equal(per_participant, ref_participant)


TREND_KEYS = ("beta_trend", "beta_treat_trend", "se_treat_trend", "p_treat_trend", "n_units")


@pytest.mark.parametrize("model", ["linear", "quadratic"])
def test_trend_leaves_out_a_participant_without_the_feature_at_one_visit(model):
    # The formula dropped the row and the cluster labels kept it, so the feature had no estimate.
    visits = ["V1", "V2", "V3", "V4"]
    adata, design = _cohort(visits=tuple(visits))
    holed = _with_values(adata, 1, _rows(adata, "T00", "V3"))
    kw = dict(features=GENES, design=design, visits=visits, model=model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = trend_interaction(holed, **kw)
        ref = trend_interaction(adata[~_rows(adata, "T00")].copy(), **kw)
        complete = trend_interaction(adata, **kw)

    assert "error" not in res.columns
    assert _feature(res)["n_units"] == 11
    _same(_feature(res), _feature(ref), TREND_KEYS)
    assert _feature(res, "G0")["n_units"] == 12
    _same(_feature(res, "G0"), _feature(complete, "G0"), TREND_KEYS)


def test_trend_gives_a_feature_without_any_value_no_estimate_and_no_error():
    visits = ["V1", "V2", "V3"]
    adata, design = _cohort(visits=tuple(visits))
    holed = _with_values(adata, 1, np.ones(adata.n_obs, dtype=bool))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = trend_interaction(holed, GENES, design, visits)

    assert "error" not in res.columns
    assert _feature(res)["n_units"] == 0 and np.isnan(_feature(res)["p_treat_trend"])
    assert np.isfinite(_feature(res, "G0")["p_treat_trend"])


def test_trend_table_has_its_columns_when_no_feature_has_an_estimate():
    visits = ["V1", "V2", "V3"]
    adata, design = _cohort(visits=tuple(visits))
    flat = ad.AnnData(X=np.ones_like(adata.X), obs=adata.obs.copy(), var=adata.var.copy())

    res = trend_interaction(flat, GENES, design, visits)
    assert {"beta_treat_trend", "p_treat_trend", "FDR_treat_trend", "n_units"} <= set(res.columns)
    assert res["p_treat_trend"].isna().all()

    pre = parallel_trends(flat, GENES, design, pre_visits=visits)
    assert list(pre["feature"]) == GENES
    assert pre["p_pretrend"].isna().all() and (pre["warning"] == "OK").all()


ABUNDANCE_KEYS = ("beta_DiD", "se_DiD", "p_DiD", "beta_time", "p_time", "n_participants")


@pytest.mark.parametrize("use_bootstrap", [False, True])
def test_abundance_leaves_out_a_participant_without_a_covariate_at_one_visit(use_bootstrap):
    # The length mismatch was caught and another model fitted in silence: change scores with
    # the baseline covariate and a homoskedastic standard error, and no bootstrap.
    adata, design = _cohort(n_cells=30)
    holed = _with_values(adata, "load", _rows(adata, "T00", "V2"))
    kw = dict(design=design, visits=("V1", "V2"), covariates=["load"], min_units=4,
              use_bootstrap=use_bootstrap, n_boot=199)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = abundance_did(holed, **kw).set_index("celltype")
        ref = abundance_did(adata[~_rows(adata, "T00")].copy(), **kw).set_index("celltype")

    keys = ABUNDANCE_KEYS + (("p_DiD_boot", "se_DiD_boot", "ci_lo_boot", "ci_hi_boot") if use_bootstrap else ())
    assert list(res.index) == list(ref.index) == ["A", "B"]
    for celltype in res.index:
        assert res.loc[celltype, "n_participants"] == 11
        _same(res.loc[celltype], ref.loc[celltype], keys)
        if use_bootstrap:
            assert np.isfinite(res.loc[celltype, "p_DiD_boot"])


@pytest.mark.parametrize("use_bootstrap", [False, True])
def test_abundance_of_two_participants_reports_no_standard_error(use_bootstrap):
    # Four rows and four parameters: the cluster correction divided by zero.
    adata, design = _cohort(n_per_arm=1, n_cells=40)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = abundance_did(adata, design, ("V1", "V2"), min_units=2, use_bootstrap=use_bootstrap, n_boot=99)

    assert len(res) == 2
    assert np.isfinite(res["beta_DiD"]).all()
    assert not np.isfinite(res["se_DiD"]).any() and res["p_DiD"].isna().all()


def test_leave_one_out_does_not_leave_out_a_missing_label():
    adata, design = _cohort()
    holed = _with_values(adata, "participant_id", np.flatnonzero(_rows(adata, "T00"))[:2])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = loo_cv_did(holed, ["G0"], design, ("V1", "V2"))

    assert sorted(loo["excluded"].unique()) == sorted(adata.obs["participant_id"].unique())
