import pandas as pd
from unittest.mock import MagicMock, patch
import sctrial as st
import gseapy as gp

def test_run_gsea_did_real_logic(sample_adata, trial_design):
    """Test run_gsea_did with real gseapy call using a dummy local gene set."""
    
    # We only use a few genes
    genes = sample_adata.var_names[:10].tolist()
    adata_sub = sample_adata[:, genes].copy()
    
    # Create a dummy gene set
    gene_sets = {
        "PATHWAY_UP": ["G0", "G1", "G2"],
        "PATHWAY_DOWN": ["G8", "G9"]
    }
    
    # Run GSEA Prerank
    # We use a real call to gp.prerank but with small data
    res = st.run_gsea_did(
        adata_sub,
        gene_sets=gene_sets,
        design=trial_design,
        visits=("V1", "V2"),
        rank_by="signed_confidence",
        permutation_num=10, # small for speed
        min_size=1,
        max_size=100
    )
    
    assert isinstance(res, gp.Prerank)
    assert "PATHWAY_UP" in res.res2d.index or "PATHWAY_UP" in res.results
    
def test_run_gsea_did_mock(sample_adata, trial_design):
    """Test run_gsea_did with a mocked gseapy call to ensure ranking logic is correct."""
    
    # Mock gseapy to verify internal ranking behavior
    with patch("sctrial.stats.gsea.gp.prerank") as mock_prerank:
        mock_prerank.return_value = MagicMock()
        
        genes = sample_adata.var_names[:5].tolist()
        adata_sub = sample_adata[:, genes].copy()
        
        st.run_gsea_did(
            adata_sub,
            gene_sets="KEGG_2021_Human",
            design=trial_design,
            visits=("V1", "V2"),
            rank_by="signed_confidence"
        )
        
        assert mock_prerank.called
        args, kwargs = mock_prerank.call_args
        rnk = kwargs["rnk"]
        assert isinstance(rnk, pd.DataFrame)
        # Check that it's sorted by rank descending
        assert rnk["rank"].iloc[0] >= rnk["rank"].iloc[-1]
