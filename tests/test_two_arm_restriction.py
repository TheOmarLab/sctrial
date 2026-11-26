"""Two-arm analyses use the two arms named by the design and no other row.

``arm_bin = (arm == arm_treated)`` codes every other row as control, so a participant
of a third arm, or one with a missing arm label, entered the control group. Every
two-arm model frame is now built through ``sctrial.subset_arms``. Each test compares
a function on data that hold such participants with the same function on the data
after removing them by hand.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st

FEATS = ["sig_A", "sig_B"]
GENES = ["g1", "g2"]
EXCLUDED = "excluded from the two-arm analysis"
SHIFT = {"Treated": 2.0, "Control": 0.0, "Other": -5.0, "Placebo": 0.0}
TWO_ARMS = [(f"T{i}", "Treated") for i in range(5)] + [(f"C{i}", "Control") for i in range(5)]
DESIGN = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                        arm_treated="Treated", arm_control="Control", celltype_col="cell_type")


N_CELLS = 12


def _cells(people, *, visits=("Pre", "Post"), seed=1, arm_dtype="category") -> AnnData:
    """Twelve cells per participant-visit; ``people`` holds (participant, arm or None)."""
    rng = np.random.default_rng(seed)
    rows, counts = [], []
    for pid, arm in people:
        level = rng.normal()
        shift = SHIFT.get(arm, -3.0)
        for step, visit in enumerate(visits):
            share_t = float(np.clip(0.4 + 0.08 * shift * step + rng.normal(scale=0.05), 0.1, 0.8))
            types = rng.choice(["T", "B", "Mono"], size=N_CELLS, p=[share_t, (1 - share_t) / 2,
                                                                     (1 - share_t) / 2])
            for cell in range(N_CELLS):
                rows.append({"participant_id": pid, "arm": arm, "visit": visit,
                             "cell_type": types[cell],
                             "bmk": float(int(pid[1:]) % 2),
                             **{f: level + shift * step + rng.normal(scale=0.5) for f in FEATS}})
                counts.append(rng.poisson(np.exp(1.5 + 0.3 * shift * step), size=len(GENES)))
    obs = pd.DataFrame(rows)
    for col in ("participant_id", "visit", "cell_type"):
        obs[col] = pd.Categorical(obs[col])
    obs["arm"] = pd.Categorical(obs["arm"]) if arm_dtype == "category" else obs["arm"].astype(object)
    obs.index = obs.index.astype(str)
    x = np.asarray(counts, dtype=float)
    adata = AnnData(X=np.log1p(x), obs=obs)
    adata.var_names = GENES
    adata.layers["counts"] = x
    return adata


def _by_hand(adata: AnnData) -> AnnData:
    keep = adata.obs["arm"].isin(["Treated", "Control"]).to_numpy()
    return adata[keep].copy()


EXTRA = {
    "third arm": dict(people=TWO_ARMS + [(f"O{i}", "Other") for i in range(4)]),
    "missing label, categorical": dict(people=TWO_ARMS + [(f"N{i}", None) for i in range(4)]),
    "missing label, object": dict(people=TWO_ARMS + [(f"N{i}", None) for i in range(4)],
                                  arm_dtype="object"),
    "third arm and missing label": dict(
        people=TWO_ARMS + [(f"O{i}", "Other") for i in range(3)] + [(f"N{i}", None) for i in range(2)]),
}
PAIR = ("Pre", "Post")
SERIES = ("V1", "V2", "V3")


def _did(aggregate):
    return lambda ad, d: st.did_table(ad, features=FEATS, design=d, visits=PAIR,
                                      aggregate=aggregate, use_bootstrap=True, n_boot=99, seed=3)


def _did_frame(ad, d):
    return st.get_did_aggregated_df(ad, FEATS, d, PAIR)[0]


def _module_scores(ad, d):
    pb = st.module_score_pseudobulk(ad, FEATS, d, PAIR, pool_col="cell_type",
                                    min_cells_per_group=1)
    return st.module_score_did_by_pool(pb, d, PAIR, n_perm=50, seed=2)


def _polynomial(ad, d):
    out = st.polynomial_trend(ad, feature=FEATS[0], design=d, visits=SERIES)
    return pd.DataFrame({"coefficient": pd.Series(out["coefficients"], dtype=float)}).reset_index()


# name -> (call, visits of the fixture)
FUNCTIONS = {
    "did_table, participant": (_did("participant_visit"), PAIR),
    "did_table, cell": (_did("cell"), PAIR),
    "did_table_parallel": (lambda ad, d: st.did_table_parallel(
        ad, features=FEATS, design=d, visits=PAIR, n_jobs=1), PAIR),
    "did_table_by_celltype": (lambda ad, d: st.did_table_by_celltype(
        ad, features=FEATS, design=d, visits=PAIR), PAIR),
    "get_did_aggregated_df": (_did_frame, PAIR),
    "between_arm_comparison": (lambda ad, d: st.between_arm_comparison(
        ad, visit="Post", features=FEATS, design=d), PAIR),
    "between_arm_comparison, wilcoxon": (lambda ad, d: st.between_arm_comparison(
        ad, visit="Post", features=FEATS, design=d, method="wilcoxon"), PAIR),
    "get_between_arm_aggregated_df": (lambda ad, d: st.get_between_arm_aggregated_df(
        ad, FEATS, d, "Post"), PAIR),
    "abundance_did": (lambda ad, d: st.abundance_did(ad, design=d, visits=PAIR, min_units=3), PAIR),
    "pseudobulk_did": (lambda ad, d: st.pseudobulk_did(
        ad, genes=GENES, design=d, visits=PAIR, counts_layer="counts"), PAIR),
    "test_treatment_heterogeneity": (lambda ad, d: st.test_treatment_heterogeneity(
        ad, features=FEATS, design=d, visits=PAIR, biomarker_col="bmk"), PAIR),
    "did_table_mixed": (lambda ad, d: st.did_table_mixed(
        ad, features=FEATS, design=d, visits=PAIR), PAIR),
    "compare_fixed_vs_mixed": (lambda ad, d: st.compare_fixed_vs_mixed(
        ad, features=FEATS, design=d, visits=PAIR), PAIR),
    "trend_interaction": (lambda ad, d: st.trend_interaction(
        ad, features=FEATS, design=d, visits=SERIES), SERIES),
    "test_parallel_trends": (lambda ad, d: st.test_parallel_trends(
        ad, features=FEATS, design=d, pre_visits=SERIES), SERIES),
    "polynomial_trend": (_polynomial, SERIES),
    "event_study_did": (lambda ad, d: st.event_study_did(
        ad, features=FEATS, design=d, visits=SERIES), SERIES),
    "loo_cv_did": (lambda ad, d: st.loo_cv_did(ad, features=FEATS, design=d, visits=PAIR), PAIR),
    "kfold_cv_did": (lambda ad, d: st.kfold_cv_did(
        ad, features=FEATS, design=d, visits=PAIR, k=5, n_repeats=2, seed=4), PAIR),
    "check_covariate_balance": (lambda ad, d: st.check_covariate_balance(
        ad, d, covariates=["bmk", "sig_A"], visit="Pre"), PAIR),
    "module_score_did_by_pool": (_module_scores, PAIR),
}


def _excluded_warnings(caught) -> list[str]:
    return [str(w.message) for w in caught
            if issubclass(w.category, UserWarning) and EXCLUDED in str(w.message)]


def _run(call, adata):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call(adata, DESIGN)
    return result, _excluded_warnings(caught)


def _assert_same(result: pd.DataFrame, reference: pd.DataFrame) -> None:
    assert list(result.columns) == list(reference.columns)
    assert len(result) == len(reference) and len(reference) > 0
    for col in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[col]):
            np.testing.assert_allclose(result[col].to_numpy(float), reference[col].to_numpy(float),
                                       rtol=0, atol=1e-10, equal_nan=True, err_msg=col)
        else:
            assert result[col].astype(str).tolist() == reference[col].astype(str).tolist(), col


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("extra", EXTRA)
def test_rows_outside_the_two_arms_do_not_enter_the_analysis(function, extra):
    call, visits = FUNCTIONS[function]
    adata = _cells(visits=visits, **EXTRA[extra])
    result, messages = _run(call, adata)
    reference, none = _run(call, _by_hand(adata))
    assert none == []
    _assert_same(result, reference)

    assert len(messages) == 1
    if "third arm" in extra:
        assert "'Other'" in messages[0]
    if "missing" in extra:
        assert "missing label" in messages[0]


def test_the_estimate_differs_from_the_one_that_pooled_the_third_arm_into_control():
    """The fixture is able to show the defect: pooling moves the estimate."""
    adata = _cells(**EXTRA["third arm"])
    pooled = adata.copy()
    pooled.obs["arm"] = pd.Categorical(pooled.obs["arm"].astype(str).replace({"Other": "Control"}))
    kw = dict(features=FEATS, design=DESIGN, visits=PAIR, aggregate="participant_visit")
    with pytest.warns(UserWarning, match=EXCLUDED):
        restricted = st.did_table(adata, **kw).set_index("feature")["beta_DiD"]
    as_before = st.did_table(pooled, **kw).set_index("feature")["beta_DiD"]
    assert (restricted - as_before).abs().min() > 0.1
    assert (st.did_table(adata, **kw)["n_units"] == 10).all()
    assert (st.did_table(pooled, **kw)["n_units"] == 14).all()


@pytest.mark.parametrize("function", FUNCTIONS)
def test_two_named_arms_only_raise_no_warning_about_excluded_rows(function):
    call, visits = FUNCTIONS[function]
    _result, messages = _run(call, _cells(TWO_ARMS, visits=visits))
    assert messages == []


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("people,absent,present", [
    ([(f"T{i}", "Treated") for i in range(5)] + [(f"P{i}", "Placebo") for i in range(5)],
     "Control", "Placebo"),
    ([(f"C{i}", "Control") for i in range(5)] + [(f"O{i}", "Other") for i in range(5)],
     "Treated", "Other"),
])
def test_a_design_that_does_not_match_the_arm_labels_is_an_error(function, people, absent, present):
    """Before, 'Placebo' stood in for the control arm and an absent treated arm gave beta 0."""
    call, visits = FUNCTIONS[function]
    with pytest.raises(ValueError, match=f"No rows of arm '{absent}'") as error:
        call(_cells(people, visits=visits), DESIGN)
    assert present in str(error.value)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_equal_arm_labels_are_an_error_in_two_arm_functions(function):
    call, visits = FUNCTIONS[function]
    same = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                          arm_treated="Treated", arm_control="Treated", celltype_col="cell_type")
    with pytest.raises(ValueError, match="two distinct arm labels"):
        call(_cells(TWO_ARMS, visits=visits), same)


def test_within_arm_functions_accept_equal_arm_labels():
    """A design with one label for both arms describes one arm; within-arm analysis is unchanged."""
    adata = _cells([(f"T{i}", "Treated") for i in range(6)])
    same = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                          arm_treated="Treated", arm_control="Treated", celltype_col=None)
    no_arm = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None,
                            celltype_col=None)
    kw = dict(features=FEATS, visits=PAIR, standardize=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        named = st.within_arm_comparison(adata, arm="Treated", design=same, **kw)
    assert _excluded_warnings(caught) == []
    reference = st.within_arm_comparison(adata, arm="All", design=no_arm, **kw)
    np.testing.assert_allclose(named["beta_time"], reference["beta_time"], atol=1e-10)


def test_did_table_bayes_uses_the_two_named_arms():
    pytest.importorskip("pymc")
    kw = dict(features=FEATS[:1], design=DESIGN, visits=PAIR, draws=40, tune=40, chains=1, seed=5)
    adata = _cells(**EXTRA["third arm"])
    with pytest.warns(UserWarning, match=EXCLUDED):
        result = st.did_table_bayes(adata, **kw)
    reference = st.did_table_bayes(_by_hand(adata), **kw)
    assert int(result["n_units"].iloc[0]) == 10
    np.testing.assert_allclose(result["beta_DiD"], reference["beta_DiD"], atol=1e-10)


def test_module_score_did_compares_arm_labels_after_stripping_whitespace():
    adata = _cells(**EXTRA["third arm"])
    padded = adata.copy()
    padded.obs["arm"] = pd.Categorical(" " + padded.obs["arm"].astype(str) + " ")
    with pytest.warns(UserWarning, match=EXCLUDED):
        result = _module_scores(padded, DESIGN)
    _assert_same(result, _module_scores(_by_hand(adata), DESIGN))


def test_module_score_did_names_the_within_arm_function_for_a_single_arm_design():
    adata = _cells(TWO_ARMS)
    pb = st.module_score_pseudobulk(adata, FEATS, DESIGN, PAIR, pool_col="cell_type",
                                    min_cells_per_group=1)
    one_arm = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None,
                             celltype_col=None)
    with pytest.raises(ValueError, match="module_score_within_arm_by_pool"):
        st.module_score_did_by_pool(pb, one_arm, PAIR)


def test_gsea_wrappers_restrict_once_before_they_loop():
    from sctrial.stats.gsea import _two_arm_rows

    adata = _cells(TWO_ARMS + [(f"O{i}", "Other") for i in range(4)], visits=SERIES)
    with pytest.warns(UserWarning, match=EXCLUDED) as caught:
        rows = _two_arm_rows(adata, DESIGN, ("V1", "V3"))
    # the restriction warns once; warnings of other libraries are not counted
    assert sum(EXCLUDED in str(w.message) for w in caught) == 1
    assert set(rows.obs["visit"].astype(str)) == {"V1", "V3"}
    assert set(rows.obs["arm"].astype(str)) == {"Treated", "Control"}
    with warnings.catch_warnings():
        # the restriction must not warn a second time; warnings of other libraries are theirs
        warnings.filterwarnings("error", message=f".*{EXCLUDED}")
        again = _two_arm_rows(rows, DESIGN, ("V1", "V3"))
    assert again.n_obs == rows.n_obs


# ── an arm that is absent only inside a subset is not an error ──


def test_an_arm_absent_in_one_cell_type_gives_missing_estimates_not_an_error():
    adata = _cells(TWO_ARMS)
    only_treated = (adata.obs["cell_type"] == "B") & (adata.obs["arm"] == "Control")
    adata = adata[~only_treated.to_numpy()].copy()
    res = st.did_table(adata, features=FEATS, design=DESIGN, visits=PAIR, celltype="B")
    assert res["beta_DiD"].isna().all() and res["p_DiD"].isna().all()
    both = st.did_table(adata, features=FEATS, design=DESIGN, visits=PAIR, celltype="T")
    assert both["beta_DiD"].notna().all()


def test_did_fit_gives_a_missing_estimate_when_one_arm_is_not_in_the_model():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"unit": np.repeat([f"p{i}" for i in range(6)], 2),
                       "time": np.tile([0, 1], 6), "arm_bin": 0, "y": rng.normal(size=12)})
    out = st.did_fit(df, y="y", unit="unit", time="time", arm_bin="arm_bin")
    assert np.isnan(out["beta_DiD"]) and np.isnan(out["p_DiD"]) and out["n_units"] == 6
    df["arm_bin"] = np.repeat([1, 1, 1, 0, 0, 0], 2)
    assert np.isfinite(st.did_fit(df, y="y", unit="unit", time="time", arm_bin="arm_bin")["beta_DiD"])


def test_between_arm_fit_gives_a_missing_estimate_when_a_feature_leaves_one_arm_empty():
    adata = _cells(TWO_ARMS)
    adata.obs["sig_partial"] = adata.obs["sig_A"].where(adata.obs["arm"] == "Treated")
    res = st.between_arm_comparison(adata, visit="Post", features=["sig_A", "sig_partial"],
                                    design=DESIGN, standardize=False).set_index("feature")
    assert np.isfinite(res.loc["sig_A", "beta_arm"])
    assert np.isnan(res.loc["sig_partial", "beta_arm"]) and np.isnan(res.loc["sig_partial", "p_arm"])


# ── subset_arms ──


def test_subset_arms_returns_the_input_when_every_row_is_in_a_named_arm():
    adata = _cells(TWO_ARMS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert st.subset_arms(adata, DESIGN) is adata
        assert st.subset_arms(adata.obs, DESIGN) is adata.obs


def test_subset_arms_removes_other_rows_and_counts_their_participants():
    adata = _cells(**EXTRA["third arm and missing label"])
    with pytest.warns(UserWarning) as caught:
        sub = st.subset_arms(adata, DESIGN)
    message = str(caught[0].message)
    assert "Participants per excluded label: 'Other' (3), missing label (2)" in message
    assert "'Treated' and 'Control'" in message
    assert set(sub.obs["arm"].astype(str)) == {"Treated", "Control"}
    assert sub.n_obs == 10 * 2 * N_CELLS and adata.n_obs == 15 * 2 * N_CELLS

    with pytest.warns(UserWarning, match=EXCLUDED):
        frame = st.subset_arms(adata.obs, DESIGN)
    assert isinstance(frame, pd.DataFrame) and len(frame) == sub.n_obs
    assert frame.index.equals(sub.obs.index)


def test_subset_arms_counts_rows_when_the_participant_column_is_absent():
    frame = pd.DataFrame({"arm": ["Treated", "Control", "Other", "Other", None]})
    with pytest.warns(UserWarning, match=r"Rows per excluded label: 'Other' \(2\), missing label \(1\)"):
        out = st.subset_arms(frame, DESIGN)
    assert out["arm"].tolist() == ["Treated", "Control"]


def test_subset_arms_lists_at_most_ten_labels():
    frame = pd.DataFrame({"arm": ["Treated", "Control", *[f"arm{i:02d}" for i in range(13)]]})
    with pytest.warns(UserWarning, match="3 more labels") as caught:
        st.subset_arms(frame, DESIGN)
    assert "'arm09'" in str(caught[0].message) and "'arm10'" not in str(caught[0].message)


def test_subset_arms_errors():
    adata = _cells(TWO_ARMS)
    with pytest.raises(ValueError, match="requires design.arm_col"):
        st.subset_arms(adata, st.TrialDesign(arm_col=None))
    with pytest.raises(KeyError, match="group"):
        st.subset_arms(adata, st.TrialDesign(arm_col="group"))
    with pytest.raises(ValueError, match="labels present: \\['Control', 'Treated'\\]"):
        st.subset_arms(adata, st.TrialDesign(arm_treated="Drug"))
    empty = adata.obs.iloc[:0]
    assert st.subset_arms(empty, DESIGN) is empty


def test_data_with_one_named_arm_only_give_missing_estimates_not_an_error():
    """A subset without the other arm is not a mislabeled design; nothing is estimable."""
    one_arm = _cells([(f"T{i}", "Treated") for i in range(6)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert st.subset_arms(one_arm, DESIGN) is one_arm
    did = st.did_table(one_arm, features=FEATS, design=DESIGN, visits=PAIR)
    assert did["beta_DiD"].isna().all() and did["p_DiD"].isna().all()
    assert (did["n_units"] == 6).all()
    between = st.between_arm_comparison(one_arm, visit="Post", features=FEATS, design=DESIGN)
    assert between["beta_arm"].isna().all() and between["p_arm"].isna().all()
    assert st.abundance_did(one_arm, design=DESIGN, visits=PAIR, min_units=3).empty


def test_did_table_bayes_gives_a_missing_estimate_for_one_arm():
    pytest.importorskip("pymc")
    one_arm = _cells([(f"T{i}", "Treated") for i in range(6)])
    res = st.did_table_bayes(one_arm, features=FEATS[:1], design=DESIGN, visits=PAIR,
                             draws=20, tune=20, chains=1, seed=1)
    assert np.isnan(res["beta_DiD"].iloc[0]) and int(res["n_units"].iloc[0]) == 6


# ── TrialDesign ──


def test_arm_bin_is_defined_for_the_two_named_arms_only():
    adata = _cells(TWO_ARMS)
    coded = DESIGN.arm_bin(adata.obs)
    assert coded.tolist() == (adata.obs["arm"] == "Treated").astype(int).tolist()
    with pytest.raises(ValueError, match=r"labels \['Other'\]"):
        DESIGN.arm_bin(_cells(**EXTRA["third arm"]).obs)
    for extra in ("missing label, categorical", "missing label, object"):
        with pytest.raises(ValueError, match=f"{4 * 2 * N_CELLS} missing labels"):
            DESIGN.arm_bin(_cells(**EXTRA[extra]).obs)


def test_validate_reports_other_arms_and_missing_labels_as_excluded():
    with pytest.warns(UserWarning, match="exclude these rows") as caught:
        DESIGN.validate(_cells(**EXTRA["third arm and missing label"]))
    message = str(caught[0].message)
    assert "['Other']" in message and f"{2 * 2 * N_CELLS} rows with a missing arm label" in message
    with pytest.warns(UserWarning, match=f"{4 * 2 * N_CELLS} rows with a missing arm label"):
        DESIGN.validate(_cells(**EXTRA["missing label, object"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DESIGN.validate(_cells(TWO_ARMS))


# ── did_mixed on a table ──


def _participant_table(adata: AnnData) -> pd.DataFrame:
    table = adata.obs.groupby(["participant_id", "visit", "arm"], observed=True, dropna=False)[
        "sig_A"].mean().reset_index()
    return table.assign(arm=table["arm"].astype(object))


def test_did_mixed_restricts_to_two_arms_when_the_control_arm_is_named():
    table = _participant_table(_cells(**EXTRA["third arm and missing label"]))
    kw = dict(outcome="sig_A", participant_col="participant_id", time_col="visit", arm_col="arm",
              arm_treated="Treated", visits=PAIR)
    with pytest.warns(UserWarning, match=EXCLUDED):
        named = st.did_mixed(table, arm_control="Control", **kw)
    by_hand = table[table["arm"].isin(["Treated", "Control"])]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        reference = st.did_mixed(by_hand, **kw)
    assert named["beta_DiD"] == pytest.approx(reference["beta_DiD"], abs=1e-10)

    with pytest.warns(UserWarning, match="pools every row that is not 'Treated'") as caught:
        pooled = st.did_mixed(table, **kw)
    assert "['Control', 'Other']" in str(caught[0].message)
    assert "2 rows with a missing label" not in str(caught[0].message)
    assert "4 rows with a missing label" in str(caught[0].message)
    assert abs(pooled["beta_DiD"] - reference["beta_DiD"]) > 0.1

    with pytest.raises(ValueError, match="No rows of arm 'Placebo'"):
        st.did_mixed(table, arm_control="Placebo", **kw)
