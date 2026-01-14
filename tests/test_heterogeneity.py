import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def test_treatment_heterogeneity_basic():
    rng = np.random.default_rng(0)
    obs = []
    for pid in range(12):
        arm = "Treated" if pid < 6 else "Control"
        biomarker = 1 if pid % 2 == 0 else 0
        for visit in ["V1", "V2"]:
            obs.append(
                {
                    "participant_id": f"P{pid}",
                    "visit": visit,
                    "arm": arm,
                    "biomarker": biomarker,
                }
            )
    obs = pd.DataFrame(obs)
    X = rng.normal(size=(len(obs), 1))
    adata = AnnData(X=X, obs=obs)
    adata.obs["sig1"] = rng.normal(size=len(obs))

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )

    res = st.test_treatment_heterogeneity(
        adata,
        features=["sig1"],
        design=design,
        visits=("V1", "V2"),
        biomarker_col="biomarker",
    )
    assert not res.empty
    assert "beta_heterogeneity" in res.columns


def test_treatment_heterogeneity_numeric_biomarker():
    rng = np.random.default_rng(1)
    obs = []
    for pid in range(10):
        arm = "Treated" if pid < 5 else "Control"
        biomarker = rng.normal()
        for visit in ["V1", "V2"]:
            obs.append(
                {
                    "participant_id": f"P{pid}",
                    "visit": visit,
                    "arm": arm,
                    "biomarker_num": biomarker,
                }
            )
    obs = pd.DataFrame(obs)
    X = rng.normal(size=(len(obs), 1))
    adata = AnnData(X=X, obs=obs)
    adata.obs["sig1"] = rng.normal(size=len(obs))

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )

    res = st.test_treatment_heterogeneity(
        adata,
        features=["sig1"],
        design=design,
        visits=("V1", "V2"),
        biomarker_col="biomarker_num",
        threshold=0.0,
    )
    assert "beta_heterogeneity" in res.columns
