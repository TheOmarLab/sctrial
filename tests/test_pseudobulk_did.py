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


def test_pseudobulk_expression_matches_groupby():
    obs = pd.DataFrame({
        "participant_id": ["P1", "P1", "P2", "P2"],
        "visit": ["V1", "V2", "V1", "V2"],
    })
    X = np.asarray([
        [10.0, 5.0],
        [20.0, 5.0],
        [15.0, 5.0],
        [25.0, 5.0],
    ])
    adata = AnnData(X=X, obs=obs)
    adata.var_names = ["G1", "G2"]
    adata.layers["counts"] = adata.X.copy()

    pb = st.pseudobulk_expression(
        adata,
        genes=["G1", "G2"],
        groupby=["participant_id", "visit"],
        counts_layer="counts",
        log1p=False,
        include_n_cells=True,
    )

    # Manual groupby for comparison (pure numpy)
    manual_rows = []
    for (pid, visit), idx in obs.groupby(["participant_id", "visit"], observed=True).indices.items():
        sub = X[idx]
        sums = sub.sum(axis=0)
        total = sums.sum()
        manual_rows.append([pid, visit, sums[0] / (total + 1e-12) * 1e6, sums[1] / (total + 1e-12) * 1e6])
    manual = pd.DataFrame(manual_rows, columns=["participant_id", "visit", "G1", "G2"])
    manual = manual.sort_values(["participant_id", "visit"]).reset_index(drop=True)
    pb_sorted = pb.sort_values(["participant_id", "visit"]).reset_index(drop=True)

    assert np.allclose(pb_sorted[["G1", "G2"]].values, manual[["G1", "G2"]].values, atol=1e-6)
