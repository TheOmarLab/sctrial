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
            _celltype_adata(),
            gene_sets={"S1": ["G1"]},
            design=TrialDesign(celltype_col="celltype"),
            visits=("V1", "V2"),
            celltypes=["A", "B"],
        )
    assert set(res.keys()) == {"A", "B"}
    assert len([w for w in caught if "excluded" in str(w.message)]) == 1
    assert not [w for w in caught if "skipped" in str(w.message)]
    assert all(set(a.obs["arm"]) == {"Treated", "Control"} for a in received)


def _celltype_adata() -> AnnData:
    """Three arms. A and B are sampled at both visits in both analysis arms; the other cell
    types are in the third arm only, in one analysis arm only, or at one visit only."""
    rows = []

    def add(celltype, arm, participants, visits=("V1", "V2")):
        for pid in participants:
            rows.extend({"celltype": celltype, "arm": arm, "participant_id": pid, "visit": v}
                        for v in visits)

    for celltype in ("A", "B"):
        add(celltype, "Treated", ["t1", "t2"])
        add(celltype, "Control", ["c1", "c2"])
        add(celltype, "Other", ["o1"])
    add("OnlyOther", "Other", ["o1", "o2"])
    add("OnlyTreated", "Treated", ["t1", "t2"])
    add("OneVisit", "Treated", ["t1", "t2"], visits=("V1",))
    add("OneVisit", "Control", ["c1", "c2"], visits=("V1",))
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    adata = AnnData(X=np.random.default_rng(0).random((len(obs), 2)), obs=obs)
    adata.var_names = ["G1", "G2"]
    return adata


@pytest.mark.parametrize("celltypes", [None, ["A", "OnlyOther", "OnlyTreated", "OneVisit", "B"]])
def test_run_gsea_did_by_celltype_skips_cell_types_without_a_did(monkeypatch, celltypes):
    # A cell type found only in the excluded arm used to stop the whole call with a KeyError.
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))
    seen = []

    def fake_run(adata, *args, celltype, **kwargs):
        seen.append(celltype)
        return pd.DataFrame({"gene_set": ["S1"], "NES": [1.0], "FDR q-val": [0.1]})

    monkeypatch.setattr(gsea, "run_gsea_did", fake_run)

    with pytest.warns(UserWarning) as caught:
        res = gsea.run_gsea_did_by_celltype(
            _celltype_adata(),
            gene_sets={"S1": ["G1"]},
            design=TrialDesign(celltype_col="celltype"),
            visits=("V1", "V2"),
            celltypes=celltypes,
        )
    assert list(res) == seen == ["A", "B"]
    skipped = [str(w.message) for w in caught if "skipped" in str(w.message)]
    assert len(skipped) == 1 and "OnlyTreated" in skipped[0] and "OneVisit" in skipped[0]
    # a cell type that is not in the two arms is skipped only when it was asked for
    assert ("OnlyOther" in skipped[0]) == (celltypes is not None)


def test_run_gsea_did_by_celltype_skips_a_cell_type_with_too_few_participants(monkeypatch):
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))

    def fake_did_table(adata, *args, celltype, **kwargs):
        return pd.DataFrame({"feature": ["G1", "G2"], "beta_DiD": [1.0, -1.0], "se_DiD": [0.5, 0.5],
                             "p_DiD": [0.01, 0.2], "n_units": [2 if celltype == "B" else 4] * 2})

    monkeypatch.setattr(gsea, "did_table", fake_did_table)

    with pytest.warns(UserWarning) as caught:
        res = gsea.run_gsea_did_by_celltype(
            _celltype_adata(),
            gene_sets={"S1": ["G1"]},
            design=TrialDesign(celltype_col="celltype"),
            visits=("V1", "V2"),
            celltypes=["A", "B"],
            min_units=4,
        )
    assert list(res) == ["A"]
    assert any("1 cell types were skipped" in str(w.message) and "B" in str(w.message) for w in caught)


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
