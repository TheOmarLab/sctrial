import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def _toy_adata() -> AnnData:
    obs = []
    for pid in ["P1", "P2", "P3", "P4"]:
        arm = "Treated" if pid in ["P1", "P2"] else "Control"
        for visit in ["V1", "V2"]:
            obs.append(
                {
                    "participant_id": pid,
                    "visit": visit,
                    "arm": arm,
                }
            )
    obs = pd.DataFrame(obs)
    X = np.random.normal(size=(len(obs), 1))
    adata = AnnData(X=X, obs=obs)
    adata.obs["sig1"] = np.random.normal(size=len(obs))
    return adata


def test_did_config_overrides():
    adata = _toy_adata()
    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )
    cfg = st.DiDConfig(standardize=False, use_bootstrap=False, seed=7)
    res = st.did_table(
        adata,
        features=["sig1"],
        design=design,
        visits=("V1", "V2"),
        config=cfg,
    )
    assert not res.empty
    assert "beta_DiD" in res.columns


def test_did_analyzer():
    adata = _toy_adata()
    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )
    analyzer = st.DiDAnalyzer(adata, design)
    res = analyzer.fit(features=["sig1"], visits=("V1", "V2"))
    assert not res.empty
    summary = analyzer.summarize()
    assert isinstance(summary, str)


def test_workflow_chain():
    adata = _toy_adata()
    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )
    wf = st.workflow(adata).did_table(
        features=["sig1"],
        design=design,
        visits=("V1", "V2"),
    )
    res = wf.result()
    assert res is not None
    assert "beta_DiD" in res.columns
