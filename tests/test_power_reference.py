"""The t reference of the power functions and the residual SD the power table reads."""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import stats

import sctrial as st
from sctrial.stats.power import (
    power_paired,
    power_two_group,
    sample_size_paired,
    sample_size_two_group,
    sensitivity_paired,
    sensitivity_two_group,
)


def _nct_power(ncp, df, alpha=0.05):
    crit = stats.t.ppf(1 - alpha / 2, df)
    return stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)


def test_default_reference_is_the_normal_approximation():
    assert power_two_group(5, 11, 1.0) == pytest.approx(0.4579, abs=1e-4)
    assert sample_size_two_group(1.0, allocation_ratio=5 / 11) == (12, 26)
    assert sensitivity_two_group(3, 8) == pytest.approx(1.8967, abs=1e-4)
    assert power_paired(6, 0.8) == power_paired(6, 0.8, reference="normal")
    assert sample_size_paired(0.5) == 63


@pytest.mark.parametrize("n1,n2,effect,sigma", [(6, 6, 1.4, 1.1), (3, 8, 1.2, 0.9), (20, 35, 0.4, 1.0)])
def test_two_group_t_power_is_the_two_sample_t_test(n1, n2, effect, sigma):
    ncp = effect / (sigma * np.sqrt(1 / n1 + 1 / n2))
    power = power_two_group(n1, n2, effect, sigma, reference="t")
    assert power == pytest.approx(_nct_power(ncp, n1 + n2 - 2))
    assert power < power_two_group(n1, n2, effect, sigma)
    one_sided = power_two_group(n1, n2, effect, sigma, two_sided=False, reference="t")
    assert one_sided == pytest.approx(stats.nct.sf(stats.t.ppf(0.95, n1 + n2 - 2), n1 + n2 - 2, ncp))


@pytest.mark.parametrize("n,effect,sigma", [(6, 0.8, 1.0), (11, 1.0, 0.6), (30, 0.3, 0.7)])
def test_paired_t_power_is_the_paired_t_test(n, effect, sigma):
    sd_change = np.sqrt(2) * sigma
    power = power_paired(n, effect, sigma, reference="t")
    assert power == pytest.approx(_nct_power(effect / (sd_change / np.sqrt(n)), n - 1))
    assert power < power_paired(n, effect, sigma)


def test_t_sample_sizes_are_the_smallest_that_reach_the_power():
    n = sample_size_paired(0.9, sigma=1.0, reference="t")
    assert power_paired(n, 0.9, reference="t") >= 0.80 > power_paired(n - 1, 0.9, reference="t")
    assert n >= sample_size_paired(0.9, sigma=1.0)

    n_t, n_c = sample_size_two_group(1.2, sigma=1.0, allocation_ratio=3 / 8, reference="t")
    assert power_two_group(n_t, n_c, 1.2, reference="t") >= 0.80
    assert n_t == int(np.ceil(3 / 8 * n_c - 1e-9))
    assert power_two_group(int(np.ceil(3 / 8 * (n_c - 1) - 1e-9)), n_c - 1, 1.2, reference="t") < 0.80
    assert n_t + n_c >= sum(sample_size_two_group(1.2, sigma=1.0, allocation_ratio=3 / 8))


def test_t_sensitivity_inverts_the_power():
    mde = sensitivity_two_group(3, 8, sigma=1.3, reference="t")
    assert power_two_group(3, 8, mde, sigma=1.3, reference="t") == pytest.approx(0.80)
    mde = sensitivity_paired(6, sigma=0.7, power=0.9, reference="t")
    assert power_paired(6, mde, sigma=0.7, reference="t") == pytest.approx(0.90)
    assert sensitivity_paired(6, reference="t") > sensitivity_paired(6)


def test_unknown_reference_is_refused():
    for call in (lambda: power_two_group(5, 5, 1.0, reference="z"),
                 lambda: power_paired(5, 1.0, reference="z"),
                 lambda: sample_size_paired(1.0, reference="z"),
                 lambda: sample_size_two_group(1.0, reference="z"),
                 lambda: sensitivity_paired(5, reference="z"),
                 lambda: sensitivity_two_group(5, 5, reference="z")):
        with pytest.raises(ValueError, match="reference"):
            call()


def _cohort(n_treated, n_control, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_treated + n_control):
        arm = "T" if i < n_treated else "C"
        for visit in ("Pre", "Post"):
            for _ in range(2):
                rows.append({"participant_id": f"P{i:02d}", "arm": arm, "visit": visit,
                             "sig": float(rng.normal())})
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    return AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _changes(adata):
    means = adata.obs.groupby(["participant_id", "arm", "visit"], observed=True)["sig"].mean().unstack("visit")
    scale = means[["Pre", "Post"]].to_numpy().ravel().std(ddof=1)
    return ((means["Post"] - means["Pre"]) / scale).reset_index()


def test_did_resid_sd_times_sqrt2_is_the_pooled_sd_of_the_changes():
    adata = _cohort(3, 8, seed=11)
    design = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                            arm_treated="T", arm_control="C", celltype_col=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.did_table(adata, features=["sig"], design=design, visits=("Pre", "Post"),
                           aggregate="participant_visit", standardize=True)
    change = _changes(adata)
    a, b = change.loc[change["arm"] == "T", 0], change.loc[change["arm"] == "C", 0]
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    assert np.sqrt(2) * res["resid_sd"].iloc[0] == pytest.approx(pooled, rel=1e-10)


def test_within_arm_resid_sd_times_sqrt2_is_the_sd_of_the_changes():
    adata = _cohort(7, 0, seed=12)
    design = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None,
                            celltype_col=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.within_arm_comparison(adata, arm="All", features=["sig"], design=design,
                                       visits=("Pre", "Post"), standardize=True)
    change = _changes(adata)[0]
    assert np.sqrt(2) * res["resid_sd"].iloc[0] == pytest.approx(change.std(ddof=1), rel=1e-10)
