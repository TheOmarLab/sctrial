from __future__ import annotations

import sctrial as st


def test_basic_workflow(sample_adata, trial_design):
    # preprocessing: add log1p(CPM)
    ad = st.add_log1p_cpm_layer(sample_adata, out_layer="log1p_cpm")

    # score a gene set using the normalized layer
    gene_sets = {"SET": ["G0", "G1", "G2"]}
    ad = st.score_gene_sets(ad, gene_sets, layer="log1p_cpm", method="mean", prefix="ms_")

    # DiD on the scored feature
    res = st.did_table(
        ad,
        features=["ms_SET"],
        design=trial_design,
        visits=("V1", "V2"),
        exclude_crossovers=False,
        agg="mean",
    )

    assert "beta_DiD" in res.columns
    assert "p_DiD" in res.columns
    assert res.shape[0] == 1
