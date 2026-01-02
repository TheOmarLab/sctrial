import numpy as np
import pytest
import sctrial as st

def test_log1p_cpm_and_scoring(sample_adata):
    # Test normalization
    st.add_log1p_cpm_layer(sample_adata, counts_layer="counts", out_layer="norm")
    assert "norm" in sample_adata.layers
    
    # Test scoring with a zero-variance gene
    sample_adata.X[:, 0] = 5.0  # Constant gene
    gene_sets = {"test": ["G0", "G1"]}
    
    # Pass min_genes=1 to allow scoring the 2-gene set
    st.score_gene_sets(sample_adata, gene_sets, layer="norm", method="zmean", min_genes=1)
    
    assert "test" in sample_adata.obs.columns
    res_scores = sample_adata.obs["test"].values
    assert not np.any(np.isnan(res_scores)), "Z-score mean contains NaNs"