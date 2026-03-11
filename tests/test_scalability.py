import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def test_did_parallel_small():
    # Small synthetic dataset just to ensure parallel wrapper runs
    n_p = 12
    visits = ["V1", "V2"]
    rows = []
    for i in range(n_p):
        arm = "Treated" if i < 6 else "Control"
        for v in visits:
            rows.append(
                {
                    "participant_id": f"P{i}",
                    "visit": v,
                    "arm": arm,
                }
            )
    obs = pd.DataFrame(rows)
    X = np.random.RandomState(1).poisson(1.0, size=(len(obs), 10)).astype(float)
    adata = AnnData(X=X, obs=obs)
    adata.var_names = [f"G{i}" for i in range(10)]

    design = st.TrialDesign(arm_treated="Treated", arm_control="Control")
    res = st.did_table_parallel(
        adata,
        features=["G0", "G1", "G2"],
        design=design,
        visits=("V1", "V2"),
        n_jobs=1,
    )
    assert not res.empty
