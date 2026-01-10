Module Score DiD Pipeline
=========================

This tutorial demonstrates a module‑score pseudobulk DiD workflow.

.. code-block:: python

   pb = st.module_score_pseudobulk(
       adata,
       module_cols=module_cols,
       design=design,
       visits=("V1", "V2"),
       pool_col="celltype",
       min_cells_per_group=5,
   )

   res = st.module_score_did_by_pool(
       pb,
       design=design,
       visits=("V1", "V2"),
       n_perm=1000,
       fdr_within="module",
   )

   display(res.sort_values("p_DiD").head())
