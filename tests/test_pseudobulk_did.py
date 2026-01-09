import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def test_pseudobulk_did_basic():
    participants = [f"P{i}" for i in range(8)]
    arms = ["Treated"] * 4 + ["Control"] * 4
    visits = ["V1", "V2"]

    obs_rows = []
    expr = []
    for pid, arm in zip(participants, arms):
        for visit in visits:
            for _ in range(10):
                obs_rows.append({
                    "participant_id": pid,
                    "arm": arm,
                    "visit": visit,
                    "celltype": "CT1",
                })
                if arm == "Treated" and visit == "V2":
                    expr.append([20.0, 10.0])
                else:
                    expr.append([10.0, 10.0])

    obs = pd.DataFrame(obs_rows)
    X = np.asarray(expr, dtype=float)
    adata = AnnData(X=X, obs=obs)
    adata.var_names = ["GENE1", "GENE2"]
    adata.layers["counts"] = adata.X.copy()

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col="celltype",
    )

    res = st.pseudobulk_did(
        adata,
        genes=["GENE1"],
        design=design,
        visits=("V1", "V2"),
        celltype_col="celltype",
        min_cells_per_group=1,
        min_paired=2,
        use_bootstrap=False,
    )

    assert not res.empty
    assert res.loc[0, "beta_DiD"] > 0
    assert np.isfinite(res.loc[0, "p_DiD"])
