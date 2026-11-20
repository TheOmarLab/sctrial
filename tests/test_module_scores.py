import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st


def test_module_score_pseudobulk_and_did():
    participants = [f"P{i}" for i in range(6)]
    arms = ["Treated"] * 3 + ["Control"] * 3
    visits = ["V1", "V2"]

    obs_rows = []
    scores = []
    for pid, arm in zip(participants, arms):
        for visit in visits:
            for _ in range(5):
                obs_rows.append(
                    {
                        "participant_id": pid,
                        "arm": arm,
                        "visit": visit,
                        "celltype": "CT1",
                        "pool": "Immune",
                    }
                )
                # Module score increases only in Treated at V2
                if arm == "Treated" and visit == "V2":
                    scores.append([2.0, 1.0])
                else:
                    scores.append([1.0, 1.0])

    obs = pd.DataFrame(obs_rows)
    X = np.asarray(scores, dtype=float)
    adata = AnnData(X=X, obs=obs)
    adata.obs["ms_A"] = X[:, 0]
    adata.obs["ms_B"] = X[:, 1]

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col="celltype",
    )

    pb = st.module_score_pseudobulk(
        adata,
        module_cols=["ms_A", "ms_B"],
        design=design,
        visits=("V1", "V2"),
        pool_col="pool",
        min_cells_per_group=1,
    )
    assert not pb.empty

    res = st.module_score_did_by_pool(
        pb,
        design,
        visits=("V1", "V2"),
        min_paired=2,
        n_perm=200,
        seed=0,
    )
    assert not res.empty
    # Expect positive DiD for ms_A
    res_a = res[res["module"] == "ms_A"]
    assert (res_a["beta_DiD"] > 0).all()

    # Within-arm pseudobulk (treated only)
    pb_treated = pb[pb["arm"] == "Treated"].copy()
    res_within = st.module_score_within_arm_by_pool(
        pb_treated,
        design,
        visits=("V1", "V2"),
        min_paired=2,
        fdr_within=None,
    )
    assert not res_within.empty
    res_within_a = res_within[res_within["module"] == "ms_A"]
    assert (res_within_a["mean_delta"] > 0).all()


def test_within_arm_by_pool_gives_no_p_value_for_constant_module():
    rng = np.random.default_rng(5)
    rows = []
    for pid in range(6):
        for visit in ("V1", "V2"):
            rows.append({"participant_id": f"P{pid}", "visit": visit, "pool": "T", "module": "flat",
                         "module_score": 0.25})
            rows.append({"participant_id": f"P{pid}", "visit": visit, "pool": "T", "module": "moving",
                         "module_score": float(rng.normal()) + (visit == "V2")})
    pb = pd.DataFrame(rows)
    design = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = st.module_score_within_arm_by_pool(pb, design, visits=("V1", "V2"), min_paired=3)

    by_module = res.set_index("module")
    assert np.isnan(by_module.loc["flat", "p_time"])
    assert by_module.loc["flat", "mean_delta"] == 0
    assert 0 < by_module.loc["moving", "p_time"] <= 1


def test_module_score_permutation_does_not_depend_on_the_origin():
    from sctrial.stats.module_scores import _perm_test_diff

    delta = pd.Series([0.0, 1, 2, 3, 4, 5, 6, 7])
    arms = pd.Series(["T"] * 4 + ["C"] * 4)
    reference = _perm_test_diff(delta, arms, 3000, 42, treated_label="T")
    assert reference == pytest.approx(2 / 70, abs=0.01)
    for shift in (2.0**32, -2.0**32):
        assert _perm_test_diff(delta + shift, arms, 3000, 42, treated_label="T") == reference


def test_within_arm_by_pool_tests_small_changes_that_are_not_zero():
    # Changes of 1e-9 are small, not absent. An absolute tolerance treated them as all zero.
    rows = []
    for pid in range(8):
        for visit, value in (("V1", 1.0), ("V2", 1.0 + 1e-9 * (pid + 1))):
            rows.append({"participant_id": f"P{pid}", "visit": visit, "pool": "T", "module": "tiny",
                         "module_score": value})
    pb = pd.DataFrame(rows)
    design = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None)
    res = st.module_score_within_arm_by_pool(pb, design, visits=("V1", "V2"), min_paired=3)
    assert res["p_time"].iloc[0] == pytest.approx(2 / 2**8)
