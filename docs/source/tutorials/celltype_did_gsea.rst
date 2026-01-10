Celltype‑specific DiD + GSEA
============================

Here, we run DiD stratified by cell type and then perform GSEA per cell type.

.. code-block:: python

   # DiD per cell type
   res_ct = st.did_table_by_celltype(
       adata, features=genes, design=design, visits=("V1", "V2"),
       aggregate="participant_visit", use_bootstrap=True
   )

   # GSEA per cell type using DiD rankings
   gsea_ct = st.run_gsea_did_by_celltype(
       adata,
       gene_sets="KEGG_2021_Human",
       design=design,
       visits=("V1", "V2"),
       rank_by="tstat",
   )

   # Inspect a single cell type
   display(gsea_ct["CD8_T"].head())

Visualization
-------------

.. code-block:: python

   # Heatmap of pathway enrichment across cell types
   merged = st.run_gsea_did_multi(gsea_ct)
   st.plot_gsea_heatmap(
       merged,
       pool_col="celltype",
       fdr_thresh=0.25,
       top_n=20,
       title="Celltype‑Specific GSEA Heatmap"
   )

   # Forest plot for a chosen cell type
   st.plot_did_forest(
       res_ct[res_ct["celltype"] == "CD8_T"],
       title="DiD Effects in CD8 T"
   )
