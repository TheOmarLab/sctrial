import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def test_abundance_did_basic():
    """Verify that abundance_did detects engineered shifts in cell proportions."""
    # Increase cohort size to provide sufficient variance for the FE model
    n_p = 20
    visits = ["V1", "V2"]

    rows = []
    for i in range(n_p):
        pid = f"P{i}"
        arm = "Treated" if i < 10 else "Control"
        for v in visits:
            # Shift: Treated group goes from 50/100 to 100/150
            # This means celltype A proportion increases from 50% to 67% in Treated
            # Add small random jitter to cell counts to ensure unique proportions
            jitter = i % 3
            n_cells_a = (100 if (arm == "Treated" and v == "V2") else 50) + jitter
            for _ in range(n_cells_a):
                rows.append({"participant_id": pid, "visit": v, "arm": arm, "celltype": "A"})
            for _ in range(50):
                rows.append({"participant_id": pid, "visit": v, "arm": arm, "celltype": "B"})

    obs = pd.DataFrame(rows)
    adata = AnnData(X=np.zeros((len(obs), 1)), obs=obs)

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        celltype_col="celltype",
        arm_treated="Treated",
        arm_control="Control"
    )
    res = st.abundance_did(adata, design, visits=("V1", "V2"), min_units=2)

    # Should detect the engineered effect
    assert len(res) >= 1, "Abundance DiD should return results"

    # Celltype A should show positive DiD (proportion increases more in Treated)
    res_a = res[res["celltype"] == "A"]
    if len(res_a) > 0:
        assert res_a.iloc[0]["beta_DiD"] > 0, \
            "Celltype A should show positive beta_DiD (increased proportion in Treated)"
        # The effect is strong, so p-value should be significant
        assert res_a.iloc[0]["p_DiD"] < 0.1, \
            "Engineered effect should be statistically detectable"
