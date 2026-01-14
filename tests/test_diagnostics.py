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


def test_pseudobulk_export():
    obs = []
    for pid in ["P1", "P2"]:
        for visit in ["V1", "V2"]:
            for _ in range(3):
                obs.append({"participant_id": pid, "visit": visit, "arm": "Treated", "cell_type": "A"})
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
