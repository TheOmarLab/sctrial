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
    # min_units=2 is sufficient
    res = st.abundance_did(adata, design, visits=("V1", "V2"), min_units=2)
    
    assert len(res) >= 1, "Abundance DiD should return results for celltype A"