from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
from anndata import AnnData
from .did import did_table
from ..design import TrialDesign

if TYPE_CHECKING:
    import gseapy as gp

try:
    import gseapy as gp
except ImportError:
    gp = None

def run_gsea_did(
    adata: AnnData,
    gene_sets: Union[str, Dict[str, List[str]]],
    design: TrialDesign,
    visits: Tuple[str, str],
    layer: Optional[str] = None,
    exclude_crossovers: bool = True,
    rank_by: str = "signed_confidence",
    use_bootstrap: bool = False,
    n_boot: int = 999,
    return_obj: bool = False,
    **kwargs
) -> Union[pd.DataFrame, "gp.Prerank"]:
    """Perform Gene Set Enrichment Analysis (GSEA) on trial-aware rankings.
    
    This function calculates Difference-in-Differences (DiD) effect sizes for 
    all genes, ranks them, and performs GSEA using `gseapy.prerank`. This approach 
    ensures that enriched pathways represent treatment effects rather than 
    baseline differences.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
    gene_sets
        A library name (e.g. 'KEGG_2021_Human') or a dictionary mapping 
        pathway names to gene lists.
    design
        A `TrialDesign` object.
    visits
        A tuple of (baseline, followup) visit labels.
    layer
        Layer to extract gene expression from. Recommended to use a 
        normalized layer (e.g., 'log1p_cpm').
    exclude_crossovers
        Whether to exclude crossover cells from the DiD ranking.
    rank_by
        Metric for ranking genes:

        - 'signed_confidence': sign(beta_DiD) * -log10(p_DiD). Highlights
          genes with high effect and high significance.
        - 'beta': ranks genes solely by the DiD effect size.
        - 'tstat': ranks genes by the t-statistic (beta_DiD / se_DiD).
    use_bootstrap
        Whether to use Wild Cluster Bootstrap for DiD p-values (used if 
        rank_by is 'signed_confidence').
    n_boot
        Number of bootstrap permutations.
    return_obj
        Whether to return the full gseapy object. If False (default), 
        returns the results DataFrame (`res2d`).
    **kwargs
        Additional parameters passed to `gseapy.prerank` (e.g., `permutation_num`, 
        `outdir`, `min_size`, `max_size`).

    Returns
    -------
    pd.DataFrame or gseapy.Prerank
        A DataFrame of enrichment results (if return_obj=False) or the 
        gseapy result object.

    Examples
    --------
    >>> res = run_gsea_did(adata, gene_sets="KEGG_2021_Human", design=design, visits=("V1", "V2"))
    >>> print(res.head())
    """
    if gp is None:
        raise ImportError("gseapy is required for run_gsea_did. Install with 'pip install gseapy'.")

    # 1. Run DiD for all genes
    genes = adata.var_names.tolist()
    res = did_table(
        adata,
        features=genes,
        design=design,
        visits=visits,
        exclude_crossovers=exclude_crossovers,
        layer=layer,
        aggregate="participant_visit",
        use_bootstrap=use_bootstrap,
        n_boot=n_boot
    )
    
    # 2. Rank genes
    if rank_by == "signed_confidence":
        res["rank"] = np.sign(res["beta_DiD"].fillna(0)) * -np.log10(res["p_DiD"].fillna(1) + 1e-12)
    elif rank_by == "beta":
        res["rank"] = res["beta_DiD"].fillna(0)
    elif rank_by == "tstat":
        res["rank"] = res["beta_DiD"].fillna(0) / (res["se_DiD"].fillna(1) + 1e-12)
    else:
        raise ValueError(f"Unknown rank_by: {rank_by}")
        
    ranking = res[["feature", "rank"]].dropna().sort_values("rank", ascending=False)
    
    # 3. Run GSEA Prerank
    pre_res = gp.prerank(
        rnk=ranking,
        gene_sets=gene_sets,
        **kwargs
    )
    
    if return_obj:
        return pre_res
    
    # Handle both gseapy >= 1.0 (has .res2d) and potentially older versions
    if hasattr(pre_res, "res2d"):
        return pre_res.res2d
    return pre_res
