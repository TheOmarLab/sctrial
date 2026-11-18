"""The covariance of the analytical DiD test is chosen by the caller."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import stats

import sctrial as st
import sctrial.stats.gsea as gsea
from sctrial.stats.did import DiDConfig, did_fit, get_did_aggregated_df

GENES = ["G1", "G2", "G3", "G4"]
VISITS = ("V1", "V2")


def _unequal_arms(n_treated: int = 3, n_control: int = 8, seed: int = 0) -> AnnData:
    """Two visits, arms of unequal size, four cells per participant-visit."""
    rng = np.random.default_rng(seed)
    rows, values = [], []
    people = [(f"T{i}", "Treated") for i in range(n_treated)] + [
        (f"C{i}", "Control") for i in range(n_control)
    ]
    for pid, arm in people:
        level = rng.normal(size=len(GENES))
        shift = rng.normal(scale=2.0 if arm == "Treated" else 0.5, size=len(GENES))
        for visit in VISITS:
            for _ in range(4):
                rows.append({"participant_id": pid, "arm": arm, "visit": visit})
                values.append(level + (shift if visit == "V2" else 0.0) + rng.normal(scale=0.3, size=len(GENES)))
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    adata = AnnData(X=np.asarray(values), obs=obs)
    adata.var_names = GENES
    return adata


def _design() -> st.TrialDesign:
    return st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col=None,
    )


def _participant_changes(adata: AnnData, gene: str) -> tuple[np.ndarray, np.ndarray]:
    frame = adata.obs.assign(y=np.asarray(adata[:, gene].X).ravel())
    means = frame.groupby(["participant_id", "arm", "visit"], observed=True)["y"].mean().unstack("visit")
    change = (means["V2"] - means["V1"]).reset_index(name="change")
    return (
        change.loc[change["arm"] == "Treated", "change"].to_numpy(),
        change.loc[change["arm"] == "Control", "change"].to_numpy(),
    )


@pytest.mark.filterwarnings("ignore:Only .* clusters:UserWarning")
def test_nonrobust_is_the_pooled_two_sample_t_test_on_participant_changes():
    adata = _unequal_arms()
    res = st.did_table(adata, GENES, _design(), VISITS, cov_type="nonrobust").set_index("feature")
    assert set(res["cov_type_used"]) == {"nonrobust"}
    for gene in GENES:
        treated, control = _participant_changes(adata, gene)
        expected = stats.ttest_ind(treated, control, equal_var=True)
        assert res.loc[gene, "p_DiD"] == pytest.approx(expected.pvalue, rel=1e-9)
        assert res.loc[gene, "beta_DiD"] / res.loc[gene, "se_DiD"] == pytest.approx(expected.statistic, rel=1e-9)


@pytest.mark.filterwarnings("ignore:Only .* clusters:UserWarning")
def test_default_is_cluster_robust_and_differs_from_the_pooled_test():
    adata = _unequal_arms()
    default = st.did_table(adata, GENES, _design(), VISITS).set_index("feature").loc[GENES]
    cluster = st.did_table(adata, GENES, _design(), VISITS, cov_type="cluster").set_index("feature").loc[GENES]
    pooled = st.did_table(adata, GENES, _design(), VISITS, cov_type="nonrobust").set_index("feature").loc[GENES]
    pd.testing.assert_frame_equal(default, cluster)
    assert set(default["cov_type_used"]) == {"cluster"}
    np.testing.assert_allclose(default["beta_DiD"], pooled["beta_DiD"], rtol=1e-12)
    assert not np.allclose(default["se_DiD"], pooled["se_DiD"], rtol=1e-3)


@pytest.mark.filterwarnings("ignore:Only .* clusters:UserWarning")
def test_config_and_parallel_table_carry_the_covariance():
    adata = _unequal_arms()
    pooled = st.did_table(adata, GENES, _design(), VISITS, cov_type="nonrobust").set_index("feature").loc[GENES]
    by_config = st.did_table(
        adata, GENES, _design(), VISITS, config=DiDConfig(cov_type="nonrobust")
    ).set_index("feature").loc[GENES]
    pd.testing.assert_frame_equal(by_config, pooled)
    for n_jobs in (1, 2):
        parallel = st.did_table_parallel(
            adata, GENES, _design(), VISITS, cov_type="nonrobust", n_jobs=n_jobs, backend="threading"
        ).set_index("feature").loc[GENES]
        pd.testing.assert_frame_equal(parallel, pooled)
    assert DiDConfig().cov_type == "cluster"


@pytest.mark.filterwarnings("ignore:Only .* clusters:UserWarning")
def test_bootstrap_studentizes_with_the_requested_covariance():
    adata = _unequal_arms()
    table = st.did_table(
        adata, GENES, _design(), VISITS, cov_type="nonrobust", use_bootstrap=True, n_boot=199, seed=3
    ).set_index("feature")
    frame, unit, time, arm_bin = get_did_aggregated_df(adata, GENES, _design(), VISITS)
    p_by_cov = {}
    for cov in ("nonrobust", "cluster"):
        p_by_cov[cov] = [
            did_fit(frame, y=g, unit=unit, time=time, arm_bin=arm_bin, cov_type=cov,
                    use_bootstrap=True, n_boot=199, seed=3)["p_DiD_boot"]
            for g in GENES
        ]
    np.testing.assert_allclose(table.loc[GENES, "p_DiD_boot"], p_by_cov["nonrobust"], rtol=1e-12)
    assert set(table["cov_type_used"]) == {"nonrobust"}
    assert not np.allclose(p_by_cov["nonrobust"], p_by_cov["cluster"])


def test_gsea_wrappers_pass_the_covariance_to_the_gene_table(monkeypatch):
    received = []

    def fake_did_table(adata, *args, **kwargs):
        received.append(kwargs.get("cov_type"))
        return pd.DataFrame({
            "feature": GENES, "beta_DiD": [1.0, -1.0, 0.5, -0.5], "se_DiD": [0.5] * 4,
            "p_DiD": [0.01, 0.2, 0.3, 0.4], "n_units": [11] * 4,
        })

    def fake_prerank(rnk, gene_sets, **kwargs):
        return types.SimpleNamespace(res2d=pd.DataFrame({"Term": list(gene_sets), "NES": [1.0]}))

    monkeypatch.setattr(gsea, "did_table", fake_did_table)
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=fake_prerank))
    adata, sets = _unequal_arms(), {"S1": ["G1", "G2"]}

    gsea.run_gsea_did(adata, sets, _design(), VISITS, rank_by="tstat")
    gsea.run_gsea_did(adata, sets, _design(), VISITS, rank_by="tstat", cov_type="nonrobust")
    gsea.run_gsea_did_multi(adata, {"A": sets}, _design(), VISITS, rank_by="tstat", cov_type="nonrobust")
    assert received == ["cluster", "nonrobust", "nonrobust"]

    received.clear()
    adata.obs["celltype"] = "T"
    by_type = st.TrialDesign(
        participant_col="participant_id", visit_col="visit", arm_col="arm",
        arm_treated="Treated", arm_control="Control", celltype_col="celltype",
    )
    gsea.run_gsea_did_by_celltype(adata, sets, by_type, VISITS, rank_by="tstat", cov_type="nonrobust")
    assert received and set(received) == {"nonrobust"}
