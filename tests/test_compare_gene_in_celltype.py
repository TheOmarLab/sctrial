import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def test_compare_gene_in_celltype_basic():
    n_cells_per_participant = 20
    participants = [f"P{i}" for i in range(6)]
    groups = ["RECeT"] * 3 + ["Sham"] * 3
    obs_rows = []
    expr = []
    for pid, grp in zip(participants, groups):
        for _ in range(n_cells_per_participant):
            obs_rows.append(
                {
                    "participant_id": pid,
                    "group": grp,
                    "celltype": "T",
                }
            )
            expr.append(10.0 if grp == "RECeT" else 1.0)

    obs = pd.DataFrame(obs_rows)
    X = np.asarray(expr, dtype=float).reshape(-1, 1)
    adata = AnnData(X=X, obs=obs)
    adata.var_names = ["GENE1"]
    adata.layers["counts"] = adata.X.copy()

    result, df_patient = st.compare_gene_in_celltype(
        adata,
        gene="GENE1",
        celltypes="T",
        group_col="group",
        group1="RECeT",
        group2="Sham",
        participant_col="participant_id",
        celltype_col="celltype",
        min_cells_per_patient=5,
        min_patients_per_group=3,
    )

    assert result["n_group1"] == 3
    assert result["n_group2"] == 3
    assert np.isfinite(result["p_value"])
    assert result["delta"] > 0
    assert len(df_patient) == 6
