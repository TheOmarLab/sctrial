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
       fdr_within="module",   # Per-module FDR (exploratory)
       fdr_global=True,       # Also compute global FDR across all tests
   )

   # Use FDR_DiD_global for properly controlled multiple testing
   display(res.sort_values("p_DiD").head())

Visualization
-------------

.. code-block:: python

   # Heatmap of module‑score DiD by cell type
   pivot = res.pivot(index="module", columns="pool", values="beta_DiD")
   plt.figure(figsize=(10, max(4, 0.35 * len(pivot))))
   sns.heatmap(pivot, cmap="RdBu_r", center=0)
   plt.title("Module‑Score DiD by Cell Type")
   plt.tight_layout()
   plt.show()

   # Module‑score UMAP panel for a selected signature
   st.plot_module_umap_panel(
       adata,
       module_cols=[module_cols[0]],
       celltype_col=design.celltype_col,
       umap_key=\"X_umap\",
       n_cols=1,
       figsize=(6, 5),
   )
   plt.show()
