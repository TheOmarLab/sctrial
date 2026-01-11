Quickstart
==========

`sctrial` provides a streamlined, trial‑aware API for single‑cell clinical studies.

Minimal end‑to‑end example
--------------------------

.. code-block:: python

   import sctrial as st

   # 1. Define Design
   design = st.TrialDesign(
       participant_col="pid",
       visit_col="visit",
       arm_col="arm",
       arm_treated="Treated",
       arm_control="Control",
       celltype_col="cell_type",
   )

   # 2. Preprocess (add log1p‑CPM layer)
   adata = st.add_log1p_cpm_layer(adata, counts_layer="counts")

   # 3. Score Gene Sets (module scores)
   gene_sets = {"Signature1": ["GENE1", "GENE2", "GENE3"]}
   adata = st.score_gene_sets(
       adata,
       gene_sets,
       layer="log1p_cpm",
       method="zmean",
       prefix="ms_",
   )

   # 4. Run DiD
   res = st.did_table(
       adata,
       features=["ms_Signature1"],
       design=design,
       visits=("V1", "V2"),
   )

   # 5. Summarize & visualize
   print(st.summarize_did_results(res))
   st.plot_did_forest(res)

Common add‑ons
-------------

**Fluent workflow API**

.. code-block:: python

   # Chain common steps in a single workflow
   res = (
       st.workflow(adata)
       .add_log1p_cpm_layer(counts_layer="counts")
       .score_gene_sets(gene_sets, layer="log1p_cpm", prefix="ms_")
       .did_table(features=["ms_Signature1"], design=design, visits=("V1", "V2"))
       .result()
   )

**Pairing diagnostics**

.. code-block:: python

   # Count paired participants before longitudinal analysis
   paired = st.count_paired(adata, design, visits=("V1", "V2"))
   print(paired)

**AUCell scoring (optional)**

.. code-block:: python

   # Requires: pip install pyscenic
   adata = st.score_gene_sets_aucell(
       adata,
       gene_sets,
       layer="log1p_cpm",
       prefix="aucell_",
   )

**Cross‑sectional comparisons**

.. code-block:: python

   cross = st.between_arm_comparison(
       adata,
       visit="V1",
       features=["ms_Signature1"],
       design=design,
       aggregate="participant_visit",
   )

**GSEA on DiD results**

.. code-block:: python

   gsea = st.run_gsea_did(
       adata,
       gene_sets="KEGG_2021_Human",
       design=design,
       visits=("V1", "V2"),
   )
   st.plot_gsea_heatmap(gsea, fdr_thresh=0.25, top_n=20)

**Pseudobulk export**

.. code-block:: python

   pb = st.pseudobulk_export(
       adata,
       genes=["IFNG", "GZMB", "NKG7"],
       design=design,
       visits=("V1", "V2"),
       celltype_col="cell_type",
   )

Next steps
----------

- :doc:`tutorials/basic_workflow` for a full pipeline.
- :doc:`tutorials/celltype_did_gsea` for cell‑type‑specific DiD + GSEA.
- :doc:`tutorials/module_score_did_pipeline` for module‑score workflows.
