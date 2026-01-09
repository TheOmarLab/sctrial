import numpy as np
import pandas as pd
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
                obs_rows.append({
                    "participant_id": pid,
                    "arm": arm,
                    "visit": visit,
                    "celltype": "CT1",
                    "pool": "Immune",
                })
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
        n_perm=200,
        seed=0,
    )
    assert not res.empty
    # Expect positive DiD for ms_A
    res_a = res[res["module"] == "ms_A"]
    assert (res_a["beta_DiD"] > 0).all()
