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

   # By default, run_gsea_did returns a DataFrame (res2d) directly
   # Use return_obj=True to get the full gseapy Prerank object instead
   print(res_gsea.head())

3. Interpreting Results
-----------------------

The rankings reflect the "Trial Effect". A high positive Normalized Enrichment Score (NES) means the pathway is upregulated by the treatment over time relative to the control.

.. code-block:: python

   # View top enriched pathways (res_gsea is already a DataFrame)
   top_pathways = res_gsea.sort_values("NES", ascending=False).head(10)
   print(top_pathways)

   # To access the full gseapy object for plotting, use return_obj=True:
   gsea_obj = st.run_gsea_did(
       adata, gene_sets="KEGG_2021_Human", design=design,
       visits=("V1", "V2"), return_obj=True
   )

   # Plot the enrichment plot for the top pathway
   from gseapy import gseaplot
   terms = gsea_obj.res2d.index
   gseaplot(rank_metric=gsea_obj.ranking, term=terms[0], **gsea_obj.results[terms[0]])
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

6. Multi-Celltype GSEA Heatmap
------------------------------

When analyzing multiple cell types or cell pools, a heatmap is the best way to compare pathway enrichments across different lineages. The `plot_gsea_heatmap` function allows you to visualize multiple GSEA results in a single, comprehensive view.

.. code-block:: python

   # Run GSEA for multiple cell types (e.g., T cells, B cells, Monocytes)
   # and combine the results into a single table
   import pandas as pd
   
   celltypes = ["CD4 T cells", "B cells", "Monocytes"]
   all_gsea_results = []
   
   for ct in celltypes:
       ct_adata = adata[adata.obs["celltype"] == ct].copy()
       res = st.run_gsea_did(
           ct_adata,
           gene_sets=custom_sets,
           design=design,
           visits=("V1", "V2")
       )
       res["pool"] = ct
       all_gsea_results.append(res)
       
   combined_res = pd.concat(all_gsea_results)
   
   # Plot a comprehensive heatmap across all cell types
   # Shows the Normalized Enrichment Score (NES)
   st.plot_gsea_heatmap(
       combined_res, 
       fdr_thresh=0.25, 
       figsize=(10, 8),
       title="Pathway Enrichment Across Cell Types"
   )
   plt.show()
