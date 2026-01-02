GSEA and Pathway Analysis
=========================

This tutorial explains how to perform Gene Set Enrichment Analysis (GSEA) using rankings derived from trial-aware Difference-in-Differences (DiD) models.

1. Background
-------------

Standard differential expression (DE) often fails to capture the longitudinal treatment effect in a trial setting. A gene might be highly expressed in both groups at baseline, or its expression might change over time due to natural variation.

By using DiD effect sizes and their associated statistical significance, we can rank genes based on how much their change over time **differs** between treatment and control arms. This ensures that the enriched pathways reflect the treatment effect, not baseline differences.

2. Ranking Genes by DiD
-----------------------

The `run_gsea_did` function automates this process. It performs the following steps:
1. Runs a DiD model for **every gene** in your dataset (aggregated at the participant-visit level).
2. Calculates a rank for each gene.
3. Passes the ranked list to `gseapy.prerank`.

.. code-block:: python

   import sctrial as st
   import gseapy as gp
   import matplotlib.pyplot as plt
   
   # Assuming 'adata' and 'design' are already set up as in the Basic Workflow tutorial
   
   # Run GSEA using a standard library (e.g., KEGG)
   # 'rank_by' can be:
   # - "signed_confidence": sign(beta) * -log10(p). Highlights genes that are both 
   #   highly significant AND have a large effect.
   # - "beta": uses raw beta_DiD values.
   res_gsea = st.run_gsea_did(
       adata,
       gene_sets="KEGG_2021_Human",
       design=design,
       visits=("V1", "V2"),
       rank_by="signed_confidence",
       permutation_num=100,  # Number of permutations
       outdir="gsea_results",
       min_size=15,
       max_size=500
   )
   
   # The result is a gseapy Prerank object
   print(res_gsea.res2d.head())

3. Interpreting Results
-----------------------

The rankings reflect the "Trial Effect". A high positive Normalized Enrichment Score (NES) means the pathway is upregulated by the treatment over time relative to the control.

.. code-block:: python

   # View top enriched pathways
   top_pathways = res_gsea.res2d.sort_values("NES", ascending=False).head(10)
   print(top_pathways)

   # Plot the enrichment plot for the top pathway
   from gseapy import gseaplot
   terms = res_gsea.res2d.index
   gseaplot(rank_metric=res_gsea.ranking, term=terms[0], **res_gsea.results[terms[0]])
   plt.show()

4. Custom Gene Sets
-------------------

You can also pass a dictionary of custom gene sets, which is useful for testing specific biological hypotheses.

.. code-block:: python

   custom_sets = {
       "T_Cell_Activation": ["CD3D", "CD3E", "CD28", "LCK"],
       "IFN_Response": ["STAT1", "IFIT1", "MX1", "ISG15"]
   }
   
   res_custom = st.run_gsea_did(
       adata,
       gene_sets=custom_sets,
       design=design,
       visits=("V1", "V2"),
       rank_by="signed_confidence"
   )

5. Visualization of Rankings
----------------------------

It is often useful to visualize the distribution of DiD effect sizes (Volcano plot) alongside GSEA results.

.. code-block:: python

   # Get the underlying DiD table used for ranking
   # We can run it manually for all genes
   did_res = st.did_table(
       adata, 
       features=adata.var_names, 
       design=design, 
       visits=("V1", "V2")
   )

   # Add -log10(p) for plotting
   did_res = st.plotting.did_volcano_frame(did_res)
   
   # Plot volcano
   plt.scatter(did_res["beta_DiD"], did_res["neglog10p"], alpha=0.5)
   plt.xlabel("Effect Size (beta_DiD)")
   plt.ylabel("-log10(p_DiD)")
   plt.title("Trial-Aware Volcano Plot")
   plt.show()
