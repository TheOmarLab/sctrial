import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sctrial.design import TrialDesign


@pytest.fixture
def sample_adata():
    """Fixture to provide a more realistic trial AnnData object."""
    from scipy import sparse
    rng = np.random.default_rng(42)
    # Increase size: 20 participants, 100 genes
    n_p, n_g = 20, 100
    visits = ["V1", "V2"]

    obs_list = []
    for i in range(n_p):
        arm = "Treated" if i < 10 else "Control"
        for v in visits:
            obs_list.append({
                "participant_id": f"P{i}",
                "visit": v,
                "arm": arm,
                "celltype": "TypeA" if i % 2 == 0 else "TypeB",
                "is_crossover": False,
                "batch": f"B{i % 3}"
            })

    obs = pd.DataFrame(obs_list)
    # Poisson counts with some dropout and treatment effect for G0-G4 in Treated-V2
    X = rng.poisson(1.5, size=(len(obs), n_g)).astype(float)

    # Add a synthetic treatment effect
    # find indices of Treated arm at V2
    idx_v2_treated = obs[(obs["arm"] == "Treated") & (obs["visit"] == "V2")].index
    X[idx_v2_treated, :5] += 2.0  # Upregulate first 5 genes

    # Convert to sparse as in real world single-cell data
    X_sparse = sparse.csr_matrix(X)

    adata = AnnData(X=X_sparse, obs=obs)
    adata.var_names = [f"G{i}" for i in range(n_g)]
    # Store raw counts in a layer
    adata.layers["counts"] = adata.X.copy()
    return adata

@pytest.fixture
def trial_design():
    return TrialDesign(arm_treated="Treated", arm_control="Control")
