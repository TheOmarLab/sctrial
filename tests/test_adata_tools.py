import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def test_subset_primary_crossover():
    """Test that subset_primary correctly filters by visit and excludes crossovers."""
    obs = pd.DataFrame({
        "visit": ["V1", "V2", "V1", "V2", "V3"],
        "is_crossover": [False, False, False, True, False]
    })
    adata = AnnData(X=np.zeros((5, 1)), obs=obs)

    # Configure design with the crossover column
    design = st.TrialDesign(visit_col="visit", crossover_col="is_crossover")

    # 1. Test subsetting only by visits (V1, V2) - should have 4 cells initially
    # 2. Test crossover exclusion - cell at index 3 is V2 but also a crossover
    ad_sub = st.subset_primary(adata, design, visits=("V1", "V2"), exclude_crossovers=True)

    assert len(ad_sub) == 3
    assert "V3" not in ad_sub.obs["visit"].values
    assert not ad_sub.obs["is_crossover"].any()

def test_subset_cells_general():
    """Test the general-purpose subset_cells helper."""
    obs = pd.DataFrame({
        "arm": ["T", "T", "C", "C"],
        "visit": ["V1", "V2", "V1", "V2"],
        "celltype": ["A", "A", "B", "B"]
    })
    adata = AnnData(X=np.zeros((4, 1)), obs=obs)
    design = st.TrialDesign(arm_col="arm", visit_col="visit", celltype_col="celltype")

    # Subset by arm and celltype
    ad_sub = st.subset_cells(adata, design, arm="T", celltype="A")

    assert len(ad_sub) == 2
    assert (ad_sub.obs["arm"] == "T").all()
    assert (ad_sub.obs["celltype"] == "A").all()

def test_did_table_celltype_agg(sample_adata, trial_design):
    """Test DiD table with celltype-level aggregation."""
    # Ensure celltype column is set in design
    design = st.TrialDesign(
        participant_col=trial_design.participant_col,
        visit_col=trial_design.visit_col,
        arm_col=trial_design.arm_col,
        celltype_col="celltype"
    )

    res = st.did_table(
        sample_adata,
        features=["G0"],
        design=design,
        visits=("V1", "V2"),
        aggregate="participant_visit_celltype"
    )
    assert not res.empty
    assert "beta_DiD" in res.columns
