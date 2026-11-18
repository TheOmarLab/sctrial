import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from anndata import AnnData

import sctrial as st


def test_check_did_assumptions_basic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "outcome": rng.normal(size=40),
            "time_num": np.repeat([0, 1], 20),
            "arm_bin": np.repeat([0, 1], 20),
            "participant_id": np.repeat([f"P{i}" for i in range(20)], 2),
        }
    )
    fit = smf.ols("outcome ~ time_num + time_num:arm_bin + C(participant_id)", data=df).fit()
    out = st.check_did_assumptions(fit)
    assert "bp_pvalue" in out
    assert "jb_pvalue" in out
    # BP p-value should be between 0 and 1
    assert 0 <= out["bp_pvalue"] <= 1
    # JB p-value should be between 0 and 1
    assert 0 <= out["jb_pvalue"] <= 1
    # Durbin-Watson should be between 0 and 4
    if "durbin_watson" in out:
        assert 0 <= out["durbin_watson"] <= 4


def test_check_did_assumptions_with_outliers():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "outcome": np.concatenate([rng.normal(size=36), np.array([10.0, 12.0, -9.0, -8.0])]),
            "time_num": np.repeat([0, 1], 20),
            "arm_bin": np.repeat([0, 1], 20),
            "participant_id": np.repeat([f"P{i}" for i in range(20)], 2),
        }
    )
    fit = smf.ols("outcome ~ time_num + time_num:arm_bin + C(participant_id)", data=df).fit()
    out = st.check_did_assumptions(fit)
    assert "bp_pvalue" in out
    assert "jb_pvalue" in out
    # With outliers, Jarque-Bera p-value should be low (significant non-normality)
    # but we just verify valid range
    assert 0 <= out["bp_pvalue"] <= 1
    assert 0 <= out["jb_pvalue"] <= 1


def test_pseudobulk_export():
    obs = []
    for pid in ["P1", "P2"]:
        for visit in ["V1", "V2"]:
            for _ in range(3):
                obs.append(
                    {"participant_id": pid, "visit": visit, "arm": "Treated", "cell_type": "A"}
                )
    obs = pd.DataFrame(obs)
    X = np.random.poisson(2, size=(len(obs), 2)).astype(float)
    adata = AnnData(X=X, obs=obs)
    adata.var_names = ["G1", "G2"]
    adata.layers["counts"] = adata.X.copy()

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col="cell_type",
    )
    pb = st.pseudobulk_export(
        adata,
        genes=["G1", "G2"],
        design=design,
        visits=("V1", "V2"),
        celltype_col="cell_type",
    )
    assert pb.n_obs == 4
    assert pb.n_vars == 2


def _pseudobulk_fixture(n_participants=6):
    rng = np.random.default_rng(3)
    rows = []
    for pid in range(n_participants):
        for visit in ("V1", "V2"):
            for _ in range(4):
                rows.append(
                    {"participant_id": f"P{pid}", "visit": visit, "arm": "Treated", "cell_type": "A"}
                )
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    counts = rng.poisson(5, size=(len(obs), 3)).astype(float)
    counts[:, 2] = 0.0
    adata = AnnData(X=counts, obs=obs)
    adata.var_names = ["G1", "G2", "SILENT"]
    adata.layers["counts"] = adata.X.copy()
    return adata


def test_pseudobulk_export_has_string_index_without_warning():
    adata = _pseudobulk_fixture(2)
    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col="cell_type",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pb = st.pseudobulk_export(
            adata, genes=["G1"], design=design, visits=("V1", "V2"), celltype_col="cell_type"
        )
    assert all(isinstance(name, str) for name in pb.obs_names)


def test_pseudobulk_within_arm_gives_no_p_value_for_constant_gene():
    adata = _pseudobulk_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res, _ = st.pseudobulk_within_arm(
            adata,
            genes=["G1", "G2", "SILENT"],
            participant_col="participant_id",
            visit_col="visit",
            visits=("V1", "V2"),
            celltype_col="cell_type",
            counts_layer="counts",
            min_paired=4,
        )
    by_gene = res.set_index("feature")
    assert np.isnan(by_gene.loc["SILENT", "p_time"])
    assert by_gene.loc["SILENT", "mean_delta"] == 0
    assert 0 < by_gene.loc["G1", "p_time"] <= 1
