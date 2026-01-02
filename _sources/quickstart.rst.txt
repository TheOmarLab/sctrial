Quickstart
==========

`sctrial` provides a streamlined API for trial-aware single-cell analysis.

.. code-block:: python

   import sctrial as st

   # 1. Define Design
   design = st.TrialDesign(
       participant_col="pid", visit_col="visit", arm_col="arm"
   )

   # 2. Score Gene Sets
   adata = st.score_gene_sets(adata, {"SET": ["G1", "G2"]}, method="zmean")

   # 3. Run DiD
   res = st.did_table(adata, features=["SET"], design=design, visits=("V1", "V2"))

   # 4. Summarize
   print(st.summarize_did_results(res))

For more details, see the :doc:`tutorials/basic_workflow`.
