import importlib.util
import sys
from pathlib import Path

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
            obs_list.append(
                {
                    "participant_id": f"P{i}",
                    "visit": v,
                    "arm": arm,
                    "celltype": "TypeA" if i % 2 == 0 else "TypeB",
                    "is_crossover": False,
                    "batch": f"B{i % 3}",
                }
            )

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


@pytest.fixture
def load_figure_module(tmp_path, monkeypatch):
    """Import a manuscript figure module (``"main/figure2_tnbc_analysis"``) in isolation.

    Figure output goes to a temporary directory, and every ``manuscript_figures``
    module loaded here is removed from ``sys.modules`` afterwards.
    """
    root = Path(__file__).resolve().parent.parent
    monkeypatch.setenv("SCTRIAL_MANUSCRIPT_DIR", str(tmp_path))
    saved = {k: v for k, v in sys.modules.items() if k.startswith("manuscript_figures")}
    for k in saved:
        del sys.modules[k]

    def _exec(name, path, search=None):
        kw = {"submodule_search_locations": [str(root / search)]} if search else {}
        spec = importlib.util.spec_from_file_location(name, root / path, **kw)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def load(relpath):
        package = relpath.split("/")[0]
        _exec("manuscript_figures", "manuscript_figures/__init__.py", "manuscript_figures")
        _exec(f"manuscript_figures.{package}", f"manuscript_figures/{package}/__init__.py",
              f"manuscript_figures/{package}")
        name = "manuscript_figures." + relpath.replace("/", ".")
        return _exec(name, f"manuscript_figures/{relpath}.py")

    yield load
    for k in [k for k in sys.modules if k.startswith("manuscript_figures")]:
        del sys.modules[k]
    sys.modules.update(saved)
