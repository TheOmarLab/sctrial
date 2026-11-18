"""Participant relabeling: enumeration, exact permutation p-values, and the fits behind them.

Every statistic of ``manuscript_figures._relabeling`` is checked against an independent
formula: with two visits per participant and a standardized outcome, the DiD
coefficient is the difference in mean change between the groups, the within-arm
coefficient is the mean change, and the between-arm coefficient is the difference
in means.
"""

import itertools
import sys
from math import comb

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st

pytest.importorskip("scanpy")

FEATS = ["sig_A", "sig_B"]
TWO_ARM = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                         arm_treated="Treated", arm_control="Control", celltype_col=None)
ONE_ARM = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None,
                         celltype_col=None)


@pytest.fixture
def relabeling(load_figure_module, monkeypatch, tmp_path):
    load_figure_module("supp/supp_tables")
    module = sys.modules["manuscript_figures._relabeling"]
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setattr(module, "_CACHE_DIR", tmp_path / "cache")
    return module


def _cells(n_treated: int, n_control: int, *, seed: int = 0, n_unpaired: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    rows = []
    people = [(f"T{i}", "Treated") for i in range(n_treated)] + [
        (f"C{i}", "Control") for i in range(n_control)]
    for pid, arm in people:
        shift = rng.normal()
        for visit in ("Pre", "Post"):
            for _ in range(3):
                rows.append({"participant_id": pid, "arm": arm, "visit": visit,
                             **{f: shift + rng.normal() for f in FEATS}})
    for i in range(n_unpaired):
        rows.append({"participant_id": f"U{i}", "arm": "Control", "visit": "Pre",
                     **{f: rng.normal() for f in FEATS}})
    obs = pd.DataFrame(rows)
    for col in ("participant_id", "arm", "visit"):
        obs[col] = pd.Categorical(obs[col])
    obs.index = obs.index.astype(str)
    return AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _standardized_changes(adata: AnnData, feat: str) -> pd.Series:
    """Post minus Pre of the participant-visit means, after z-scoring those means."""
    means = adata.obs.groupby(["participant_id", "visit"], observed=True)[feat].mean().unstack()
    means = means.dropna()
    values = means.to_numpy()
    z = (means - values.mean()) / values.ravel().std(ddof=1)
    return z["Post"] - z["Pre"]


def test_group_assignments_are_all_the_assignments_when_few(relabeling):
    rows, exact = relabeling.group_assignments(12, 6)
    assert exact and rows.shape == (comb(12, 6), 12) == (924, 12)
    assert (rows.sum(axis=1) == 6).all()
    assert len({tuple(r) for r in rows}) == 924

    draws, exact = relabeling.group_assignments(12, 6, max_exact=100, n_draws=50, seed=3)
    assert not exact and draws.shape == (50, 12) and (draws.sum(axis=1) == 6).all()
    again, _ = relabeling.group_assignments(12, 6, max_exact=100, n_draws=50, seed=3)
    assert (draws == again).all()
    with pytest.raises(ValueError):
        relabeling.group_assignments(5, 5)


def test_swap_assignments_are_all_the_sign_patterns_when_few(relabeling):
    rows, exact = relabeling.swap_assignments(6)
    assert exact and rows.shape == (64, 6)
    assert len({tuple(r) for r in rows}) == 64
    assert not rows[0].any()  # the observed data, no swap, is the first assignment
    draws, exact = relabeling.swap_assignments(30)
    assert not exact and draws.shape == (relabeling.N_DRAWS, 30)


def test_permutation_summary_exact_counts_the_observed_assignment(relabeling):
    null = np.array([-2.0, -1.0, 0.5, 1.0, 2.0, -0.2])
    exact = relabeling.permutation_summary(null, 1.0, exact=True)
    assert exact["permutation_p"] == pytest.approx(4 / 6)   # |-2|, |-1|, |1|, |2|
    assert exact["n_permutations"] == 6 and exact["exact"] is True
    drawn = relabeling.permutation_summary(null, 1.0, exact=False)
    assert drawn["permutation_p"] == pytest.approx((4 + 1) / (6 + 1))
    assert relabeling.permutation_summary(null, 2.0, exact=True)["permutation_p"] == pytest.approx(2 / 6)
    assert np.isnan(relabeling.permutation_summary(np.array([]), 1.0, exact=True)["permutation_p"])


def test_did_relabeling_matches_did_table_and_the_difference_in_mean_change(relabeling):
    adata = _cells(4, 5, n_unpaired=3)
    table = relabeling.did_relabeling_cached(adata, FEATS, TWO_ARM, ("Pre", "Post"), layer=None,
                                             tag="unit", n_boot=99)
    assert table.attrs["exact"] and table.attrs["n_assignments"] == comb(9, 4) == 126
    assert len(table) == 126 * len(FEATS)
    assert table.groupby("feature")["observed"].sum().eq(1).all()
    assert (table["cov_type_used"] == "cluster").all()

    reference = st.did_table(adata, features=FEATS, design=TWO_ARM, visits=("Pre", "Post"),
                             standardize=True, aggregate="participant_visit",
                             use_bootstrap=True, n_boot=99, seed=42).set_index("feature")
    observed = table[table["observed"]].set_index("feature")
    for mine, theirs in (("beta_DiD", "beta_DiD"), ("p_boot", "p_DiD_boot"),
                         ("ci_lo_boot", "ci_lo_boot"), ("ci_hi_boot", "ci_hi_boot")):
        np.testing.assert_allclose(observed.loc[FEATS, mine].to_numpy(float),
                                   reference.loc[FEATS, theirs].to_numpy(float), atol=1e-10)

    # every assignment, against the difference in mean standardized change
    for feat in FEATS:
        change = _standardized_changes(adata, feat)
        ids = sorted(change.index.astype(str))
        change = change.rename(index=str).loc[ids].to_numpy()
        expected = []
        for members in itertools.combinations(range(len(ids)), 4):
            first = np.zeros(len(ids), bool)
            first[list(members)] = True
            expected.append(change[first].mean() - change[~first].mean())
        got = table.loc[table["feature"] == feat].sort_values("assignment")["beta_DiD"].to_numpy()
        np.testing.assert_allclose(got, expected, atol=1e-10)

    # a second call reads the cache and returns the same table
    again = relabeling.did_relabeling_cached(adata, FEATS, TWO_ARM, ("Pre", "Post"), layer=None,
                                             tag="unit", n_boot=99)
    pd.testing.assert_frame_equal(table, again)
    assert again.attrs["exact"] and len(list((relabeling._CACHE_DIR).glob("relabeling_unit_*.pkl"))) == 1


def test_check_observed_assignment_rejects_a_table_that_differs(relabeling):
    adata = _cells(3, 4)
    df_use, unit, time, arm_bin = st.get_did_aggregated_df(adata, FEATS, TWO_ARM, ("Pre", "Post"))
    table = relabeling.did_relabeling(df_use, FEATS, unit, time, arm_bin, inference=False)
    reference = st.did_table(adata, features=FEATS, design=TWO_ARM, visits=("Pre", "Post"),
                             standardize=True, aggregate="participant_visit")
    relabeling.check_observed_assignment(table, reference, inference=False)
    wrong = reference.copy()
    wrong["beta_DiD"] = wrong["beta_DiD"] + 1e-6
    with pytest.raises(AssertionError, match="does not reproduce did_table"):
        relabeling.check_observed_assignment(table, wrong, inference=False)


def test_exact_permutation_p_is_the_share_of_assignments_at_least_as_extreme(relabeling):
    adata = _cells(4, 5, seed=5)
    df_use, unit, time, arm_bin = st.get_did_aggregated_df(adata, FEATS, TWO_ARM, ("Pre", "Post"))
    table = relabeling.did_relabeling(df_use, FEATS, unit, time, arm_bin, inference=False)
    for feat, (null, observed) in relabeling.did_permutation_nulls(table).items():
        summary = relabeling.permutation_summary(null, observed, exact=table.attrs["exact"])
        assert summary["n_permutations"] == 126
        assert summary["permutation_p"] == pytest.approx(np.mean(np.abs(null) >= abs(observed) - 1e-12))
        assert summary["permutation_p"] >= 1 / 126


def test_calibration_rates_are_shares_of_assignments(relabeling):
    table = pd.DataFrame({
        "feature": ["a"] * 4 + ["b"] * 4,
        "p_analytic": [0.01, 0.2, 0.3, 0.9, 0.5, 0.6, 0.7, 0.8],
        "p_boot": [0.01, 0.04, 0.3, 0.9, 0.049, 0.05, 0.7, 0.8],
        "ci_lo_boot": [0.1, 0.2, -1, -1, 0.1, -1, -1, -1],
        "ci_hi_boot": [1, 1, 1, 1, 1, 1, 1, 1],
    })
    rates = relabeling.calibration_rates(table).set_index("feature")
    assert rates.loc["a", ["reject_analytic", "reject_boot", "zero_in_interval"]].tolist() == [0.25, 0.5, 0.5]
    assert rates.loc["b", ["reject_analytic", "reject_boot", "zero_in_interval"]].tolist() == [0.0, 0.25, 0.75]
    with pytest.raises(ValueError, match="missing values"):
        relabeling.calibration_rates(table.assign(p_boot=np.nan))


def test_within_arm_nulls_are_the_sign_flips_of_the_changes(relabeling):
    adata = _cells(6, 0, seed=2)
    df_use, unit = st.get_within_arm_aggregated_df(adata, "All", FEATS, ONE_ARM, ("Pre", "Post"))
    nulls, exact = relabeling.within_arm_nulls(df_use, FEATS, unit)
    assert exact
    observed = st.within_arm_comparison(adata, arm="All", features=FEATS, design=ONE_ARM,
                                        visits=("Pre", "Post"), standardize=True).set_index("feature")
    for feat in FEATS:
        change = _standardized_changes(adata, feat)
        ids = sorted(change.index.astype(str))
        change = change.rename(index=str).loc[ids].to_numpy()
        signs = 1 - 2 * ((np.arange(64)[:, None] >> np.arange(6)) & 1)
        np.testing.assert_allclose(nulls[feat], (signs * change).mean(axis=1), atol=1e-10)
        assert nulls[feat][0] == pytest.approx(observed.loc[feat, "beta_time"], abs=1e-10)


def test_between_arm_nulls_are_the_differences_in_means(relabeling):
    adata = _cells(4, 5, seed=4)
    adata.obs["dfo_bin"] = pd.Categorical(["bin"] * adata.n_obs)
    design = st.TrialDesign(participant_col="participant_id", visit_col="dfo_bin", arm_col="arm",
                            arm_treated="Treated", arm_control="Control", celltype_col=None)
    df_cs = st.get_between_arm_aggregated_df(adata, FEATS, design, "bin")
    nulls, exact = relabeling.between_arm_nulls(df_cs, FEATS, design)
    assert exact and len(nulls[FEATS[0]]) == 126
    for feat in FEATS:
        y = df_cs[feat].to_numpy(float)
        z = (y - y.mean()) / y.std(ddof=1)
        expected = []
        for members in itertools.combinations(range(9), 4):
            first = np.zeros(9, bool)
            first[list(members)] = True
            expected.append(z[first].mean() - z[~first].mean())
        np.testing.assert_allclose(nulls[feat], expected, atol=1e-10)
        observed = st.between_arm_fit_beta(df_cs, feat, design)
        assert np.isclose(nulls[feat], observed, atol=1e-10).sum() >= 1
