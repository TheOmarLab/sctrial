import types

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial.stats.gsea as gsea
from sctrial import TrialDesign


def _dummy_prerank(rnk, gene_sets, **kwargs):
    res = pd.DataFrame(
        {
            "gene_set": list(gene_sets.keys()) if isinstance(gene_sets, dict) else ["GS1"],
            "NES": [1.0],
            "FDR q-val": [0.1],
        }
    )
    return types.SimpleNamespace(res2d=res)


def _three_arm_adata() -> AnnData:
    """Six cells: two visits by three arms, so the wrappers have rows to leave out."""
    adata = AnnData(X=np.random.default_rng(0).random((6, 2)))
    adata.var_names = ["G1", "G2"]
    adata.obs["visit"] = ["V1", "V2"] * 3
    adata.obs["arm"] = ["Treated", "Treated", "Control", "Control", "Other", "Other"]
    adata.obs["participant_id"] = ["p1", "p1", "p2", "p2", "p3", "p3"]
    adata.obs["celltype"] = ["A", "B", "A", "B", "A", "B"]
    return adata


def test_run_gsea_did_multi(monkeypatch):
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))
    received = []

    def fake_did_table(adata, *args, **kwargs):
        received.append(adata)
        return pd.DataFrame(
            {
                "feature": ["G1", "G2"],
                "beta_DiD": [1.0, -1.0],
                "se_DiD": [0.5, 0.5],
                "p_DiD": [0.01, 0.2],
                "n_units": [5, 5],
            }
        )

    monkeypatch.setattr(gsea, "did_table", fake_did_table)

    with pytest.warns(UserWarning, match="excluded from the two-arm analysis") as caught:
        res = gsea.run_gsea_did_multi(
            _three_arm_adata(),
            gene_sets={"A": {"S1": ["G1"]}, "B": {"S2": ["G2"]}},
            design=TrialDesign(celltype_col="celltype"),
            visits=("V1", "V2"),
        )
    assert set(res.keys()) == {"A", "B"}
    # one restriction before the loop: one warning, and every ranking sees two arms
    assert len([w for w in caught if "excluded" in str(w.message)]) == 1
    assert len(received) == 2
    assert all(set(a.obs["arm"]) == {"Treated", "Control"} and a.n_obs == 4 for a in received)


def test_run_gsea_did_by_celltype(monkeypatch):
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))
    received = []

    def fake_run(adata, *args, **kwargs):
        received.append(adata)
        return pd.DataFrame({"gene_set": ["S1"], "NES": [1.0], "FDR q-val": [0.1]})

    monkeypatch.setattr(gsea, "run_gsea_did", fake_run)

    with pytest.warns(UserWarning, match="excluded from the two-arm analysis") as caught:
        res = gsea.run_gsea_did_by_celltype(
            _three_arm_adata(),
            gene_sets={"S1": ["G1"]},
            design=TrialDesign(celltype_col="celltype"),
            visits=("V1", "V2"),
            celltypes=["A", "B"],
        )
    assert set(res.keys()) == {"A", "B"}
    assert len([w for w in caught if "excluded" in str(w.message)]) == 1
    assert all(set(a.obs["arm"]) == {"Treated", "Control"} and a.n_obs == 4 for a in received)


def test_run_gsea_pseudobulk(monkeypatch):
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))
    monkeypatch.setattr(
        gsea,
        "pseudobulk_did",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "feature": ["G1", "G2"],
                "beta_DiD": [1.0, -1.0],
                "se_DiD": [0.5, 0.5],
                "p_DiD": [0.01, 0.2],
                "n_units": [5, 5],
            }
        ),
    )

    adata = AnnData(X=np.random.rand(4, 2))
    adata.var_names = ["G1", "G2"]
    design = types.SimpleNamespace()

    res = gsea.run_gsea_pseudobulk(
        adata,
        gene_sets={"S1": ["G1"]},
        design=design,
        visits=("V1", "V2"),
    )
    assert isinstance(res, pd.DataFrame)
