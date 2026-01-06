Quickstart
==========

`sctrial` provides a streamlined API for trial-aware single-cell analysis.

.. code-block:: python

   import sctrial as st

   # 1. Define Design
   design = st.TrialDesign(
       participant_col="pid", visit_col="visit", arm_col="arm",
       arm_treated="Treated", arm_control="Control"
   )

   # 2. Preprocess (add log1p-CPM layer)
   adata = st.add_log1p_cpm_layer(adata, counts_layer="counts")

   # 3. Score Gene Sets (use prefix to avoid name collisions)
   gene_sets = {"Signature1": ["GENE1", "GENE2", "GENE3"]}
   adata = st.score_gene_sets(adata, gene_sets, layer="log1p_cpm",
                               method="zmean", prefix="ms_")

   # 4. Run DiD
   res = st.did_table(adata, features=["ms_Signature1"], design=design,
                       visits=("V1", "V2"))

   # 5. Summarize
   print(st.summarize_did_results(res))

For more details, see the :doc:`tutorials/basic_workflow`.
