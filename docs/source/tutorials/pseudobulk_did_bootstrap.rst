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

Visualization
-------------

.. code-block:: python

   # Heatmap of pseudobulk DiD effects (genes × cell types)
   pivot = res_pb.pivot(index="feature", columns="celltype", values="beta_DiD")
   plt.figure(figsize=(10, max(4, 0.35 * len(pivot))))
   sns.heatmap(pivot, cmap="RdBu_r", center=0)
   plt.title("Pseudobulk DiD Effects")
   plt.tight_layout()
   plt.show()

   # Forest plot for a selected gene
   st.plot_did_forest(
       res_pb[res_pb["feature"] == "IL7R"],
       title="Pseudobulk DiD: IL7R"
   )
