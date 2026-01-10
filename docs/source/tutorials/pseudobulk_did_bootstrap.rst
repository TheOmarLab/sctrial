Pseudobulk DiD + Bootstrap
==========================

This tutorial shows pseudobulk DiD with wild‑cluster bootstrap p‑values.

.. code-block:: python

   res_pb = st.pseudobulk_did(
       adata,
       genes=panel_genes,
       design=design,
       visits=("V1", "V2"),
       celltype_col="celltype",
       min_cells_per_group=5,
       use_bootstrap=True,
       n_boot=999,
   )

   display(res_pb.sort_values("p_DiD").head())
