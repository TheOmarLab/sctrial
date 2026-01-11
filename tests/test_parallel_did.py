import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def _make_simple_adata() -> AnnData:
    # 4 participants, 2 visits, 2 arms
    obs = []
    for pid in ["P1", "P2", "P3", "P4"]:
        arm = "Treated" if pid in ["P1", "P2"] else "Control"
        for visit in ["V1", "V2"]:
            obs.append({"participant_id": pid, "visit": visit, "arm": arm})
    obs = pd.DataFrame(obs)
    # Single gene with small treatment effect at V2
    X = np.array([
        [1.0], [1.5],  # P1
        [1.2], [1.6],  # P2
        [1.1], [1.2],  # P3
        [1.0], [1.1],  # P4
    ])
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=["G1"]))
    return adata


def test_did_table_parallel_matches_serial():
    adata = _make_simple_adata()
    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )

    res_serial = st.did_table(
        adata,
        features=["G1"],
        design=design,
        visits=("V1", "V2"),
        aggregate="participant_visit",
        standardize=False,
        use_bootstrap=False,
    )
    res_parallel = st.did_table_parallel(
        adata,
        features=["G1"],
        design=design,
        visits=("V1", "V2"),
        aggregate="participant_visit",
        standardize=False,
        use_bootstrap=False,
        n_jobs=2,
        backend="threading",
    )

    assert res_parallel.shape[0] == res_serial.shape[0]
    assert np.allclose(res_parallel["beta_DiD"], res_serial["beta_DiD"])
    assert np.allclose(res_parallel["p_DiD"], res_serial["p_DiD"])
