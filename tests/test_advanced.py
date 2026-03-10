import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def test_did_table_with_covariates(sample_adata, trial_design):
    # Add a covariate to obs
    sample_adata.obs["age"] = np.random.randint(20, 60, size=sample_adata.n_obs)

    # Run DiD with covariate
    res = st.did_table(
        sample_adata, features=["G0"], design=trial_design, visits=("V1", "V2"), covariates=["age"]
    )

    assert "beta_DiD" in res.columns
    assert "p_DiD" in res.columns
    assert len(res) == 1


def test_abundance_did_with_covariates(sample_adata, trial_design):
    # Add a covariate to obs
    # Provide some variation in age and proportions
    sample_adata.obs["age"] = np.random.randint(20, 60, size=sample_adata.n_obs).astype(float)

    # In sample_adata, each participant has only ONE cell type (TypeA or TypeB)
    # abundance_did tests shifts in cell proportions WITHIN a participant's cell pool.
    # If each participant has only 1 cell per participant-visit, prop is always 1.0 (no variation).

    rows = []
    for i in range(10):  # 10 participants
        pid = f"PX{i}"
        arm = "Treated" if i < 5 else "Control"
        for v in ["V1", "V2"]:
            # Add different counts of TypeA and TypeB to create variation in 'y'
            n_a = 10 + i if (arm == "Treated" and v == "V2") else 10
            for _ in range(n_a):
                rows.append(
                    {
                        "participant_id": pid,
                        "visit": v,
                        "arm": arm,
                        "celltype": "TypeA",
                        "age": 40.0 + i,
                    }
                )
            for _ in range(10):
                rows.append(
                    {
                        "participant_id": pid,
                        "visit": v,
                        "arm": arm,
                        "celltype": "TypeB",
                        "age": 40.0 + i,
                    }
                )

    obs = pd.DataFrame(rows)
    adata_new = AnnData(X=np.zeros((len(obs), 1)), obs=obs)

    res = st.abundance_did(
        adata_new, design=trial_design, visits=("V1", "V2"), covariates=["age"], min_units=2
    )

    assert "beta_DiD" in res.columns
    assert len(res) > 0


def test_summarize_did_results():
    df = pd.DataFrame(
        {
            "feature": ["G1", "G2"],
            "beta_DiD": [1.5, -2.0],
            "p_DiD": [0.001, 0.04],
            "FDR_DiD": [0.01, 0.1],
        }
    )

    summary = st.summarize_did_results(df)
    assert "Trial-Aware DiD Summary" in summary
    assert "G1" in summary
    assert "G2" in summary
    assert "beta=1.500" in summary
    assert "beta=-2.000" in summary
