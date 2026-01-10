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
