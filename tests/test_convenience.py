import numpy as np
import pytest

import sctrial as st


def test_quick_did_runs(sample_adata):
    gene_sets = {"Sig": ["G0", "G1", "G2"]}
    res = st.quick_did(
        sample_adata,
        module_scores=gene_sets,
        visits=("V1", "V2"),
        counts_layer="counts",
    )
    assert not res.empty
    assert "feature" in res.columns


def test_auto_detect_design(sample_adata):
    design = st.auto_detect_design(sample_adata)
    assert isinstance(design, st.TrialDesign)
    assert design.participant_col in sample_adata.obs.columns


def test_quick_did_requires_two_visits(sample_adata):
    ad = sample_adata.copy()
    ad = ad[ad.obs["visit"] == "V1"].copy()
    with pytest.raises(ValueError):
        st.quick_did(ad, module_scores={"Sig": ["G0", "G1"]})
