import types

import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial.stats.gsea as gsea


def _dummy_prerank(rnk, gene_sets, **kwargs):
    res = pd.DataFrame(
        {
            "gene_set": list(gene_sets.keys()) if isinstance(gene_sets, dict) else ["GS1"],
            "NES": [1.0],
            "FDR q-val": [0.1],
        }
    )
    return types.SimpleNamespace(res2d=res)


def test_run_gsea_did_multi(monkeypatch):
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))
    monkeypatch.setattr(
        gsea,
        "did_table",
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

    res = gsea.run_gsea_did_multi(
        adata,
        gene_sets={"A": {"S1": ["G1"]}, "B": {"S2": ["G2"]}},
        design=types.SimpleNamespace(),
        visits=("V1", "V2"),
    )
    assert set(res.keys()) == {"A", "B"}


def test_run_gsea_did_by_celltype(monkeypatch):
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))

    def fake_run(*args, **kwargs):
        return pd.DataFrame({"gene_set": ["S1"], "NES": [1.0], "FDR q-val": [0.1]})

    monkeypatch.setattr(gsea, "run_gsea_did", fake_run)

    adata = AnnData(X=np.random.rand(4, 2))
    adata.obs["celltype"] = ["A", "B", "A", "B"]

    design = types.SimpleNamespace(celltype_col="celltype")
    res = gsea.run_gsea_did_by_celltype(
        adata,
        gene_sets={"S1": ["G1"]},
        design=design,
        visits=("V1", "V2"),
        celltypes=["A", "B"],
    )
    assert set(res.keys()) == {"A", "B"}


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
