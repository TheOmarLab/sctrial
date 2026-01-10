Advanced Trial Analysis
=======================

This tutorial covers complex trial scenarios, including crossover designs, robust statistical methods like bootstrapping, and advanced visualizations for trial data.

1. Crossover Designs
--------------------

In many clinical trials, participants in the control arm may "cross over" to the treatment arm after a certain period, or vice versa. Handling these correctly is essential to avoid conflating different types of effects.

`sctrial` uses a `crossover_col` in the `TrialDesign` to identify these cells.

.. code-block:: python

   import sctrial as st
   import pandas as pd
   
   # Setup design with crossover column
   design = st.TrialDesign(
       participant_col="participant_id",
       visit_col="visit",
       arm_col="arm",
       crossover_col="is_crossover"
   )
   
   # By default, functions like did_table have exclude_crossovers=True.
   # This ensures that only the 'primary' randomized phase is analyzed.
   res_primary = st.did_table(
       adata, 
       features=["ms_OXPHOS"], 
       design=design, 
       visits=("V1", "V2"),
       exclude_crossovers=True
   )

If you want to analyze the crossover phase specifically, you can use the `subset_cells` helper:

.. code-block:: python

   # Subset to ONLY crossover cells
   adata_cross = adata[adata.obs["is_crossover"] == True].copy()
   
   # Now perform a within-arm comparison for these participants
   res_cross = st.within_arm_comparison(
       adata_cross,
       arm="Control", # They were in Control but now crossed over
       features=["ms_OXPHOS"],
       design=design,
       visits=("V2", "V3") # Assuming V3 is the post-crossover visit
   )

2. Robust Inference with Wild Cluster Bootstrap
-----------------------------------------------

For clinical trials with a small number of participants (e.g., fewer than 15 per group), standard asymptotic p-values may be unreliable. `sctrial` provides support for the **Wild Cluster Bootstrap (Rademacher)**, which is more robust in these settings.

.. code-block:: python

   # Run DiD with Wild Cluster Bootstrap
   res_boot = st.did_table(
       adata,
       features=["ms_OXPHOS"],
       design=design,
       visits=("V1", "V2"),
       use_bootstrap=True,
       n_boot=999
   )
   # res_boot['p_DiD'] now contains the bootstrap-derived p-value.

3. Weighted Pseudobulk Analysis
-------------------------------

When aggregating cells to the participant level (`aggregate="participant_visit"`), it is often beneficial to weight the participants by the number of cells they contributed. `sctrial` automatically performs **Weighted OLS** using the square root of the cell counts if `n_cells` is available (which it is by default in aggregation modes).

.. code-block:: python

   # Weighted pseudobulk DiD is the default when using 'participant_visit'
   res_weighted = st.did_table(
       adata,
       features=["ms_OXPHOS"],
       design=design,
       visits=("V1", "V2"),
       aggregate="participant_visit"
   )

4. Advanced Visualizations
--------------------------

Trial data requires specific visualizations to communicate interactions and effect sizes effectively.

Forest Plots
~~~~~~~~~~~~

Visualize the effect sizes and 95% confidence intervals for multiple features or cell types.

.. code-block:: python

   # Plot abundance DiD results
   ab_res = st.abundance_did(adata, design, visits=("V1", "V2"))
   st.plot_did_forest(ab_res, feature_col="celltype", title="Cell Type Composition Shifts")

Paired Within-Arm Plots
~~~~~~~~~~~~~~~~~~~~~~~

Visualize how individual participants change from baseline to follow-up (spaghetti plots).

.. code-block:: python

   st.plot_within_arm_comparison(
       adata, 
       arm="Treated", 
       feature="ms_OXPHOS", 
       design=design, 
       visits=("V1", "V2"),
       plot_type="paired"
   )

Trial-Aware UMAPs
~~~~~~~~~~~~~~~~~

Visualize the expression of a feature on UMAP, stratified by trial arm and visit.

.. code-block:: python

   st.plot_trial_umap(
       adata, 
       feature="ms_OXPHOS", 
       design=design, 
       visits=("V1", "V2")
   )

Summary
-------

By combining these advanced statistical and visualization tools, you can perform rigorous, publication-ready inference on complex single-cell clinical trials.

5. Stratified Analysis
----------------------

Often you want to run the same DiD analysis across all cell types in your study. `sctrial` provides `did_table_by_celltype` to automate this loop.

.. code-block:: python

   # Run DiD for all cell types
   full_res = st.did_table_by_celltype(
       adata, 
       features=["ms_OXPHOS"], 
       design=design, 
       visits=("V1", "V2")
   )
   # Results include a 'celltype' column and global 'FDR_DiD_stratified'.

6. Permutation Tests
--------------------

For non-parametric inference, you can use permutation tests for both two-sample and paired comparisons.

.. code-block:: python

   from sctrial.utils import permutation_pvalue, permutation_pvalue_paired
   
   # Two-sample permutation test
   p = permutation_pvalue(group1_vals, group2_vals, n_perm=10000)
   
   # Paired permutation test (e.g. for pre/post changes)
   p_paired = permutation_pvalue_paired(baseline_vals, followup_vals, n_perm=10000)

7. Feature Profiling
--------------------

Before running inference, you might want to see how marker genes or module scores are distributed across cell types or clusters.

.. code-block:: python

   profile = st.profile_features(
       adata, 
       features=["MUC1", "MUC13", "EPCAM"], 
       groupby="celltype"
   )
   # Returns a DataFrame of mean expression per celltype.

8. Specialized Heatmaps and Radar Plots
---------------------------------------

When analyzing many cell types or gene sets, high-dimensional visualizations are essential.

GSEA Radar Plots
~~~~~~~~~~~~~~~~

Visualize GSEA Normalized Enrichment Scores (NES) for a specific term across multiple cell pools or types.

.. code-block:: python

   # Assuming you have a merged GSEA results DataFrame
   st.plot_gsea_radar(merged_res, term="OXPHOS", pool_col="celltype")

GSEA Heatmaps
~~~~~~~~~~~~~

Visualize NES values for top terms across multiple cell types.

.. code-block:: python

   # Heatmap of top significant pathways across cell types
   # The function automatically selects pathways with FDR < fdr_thresh
   st.plot_gsea_heatmap(
       merged_res,
       fdr_thresh=0.25,  # Include pathways significant in at least one pool
       top_n=20,         # Show top 20 pathways by minimum FDR
       pool_col="celltype"
   )

9. Trial UMAP Panels
--------------------

A high-impact visualization that shows a feature's expression on UMAP, stratified 2x2 by treatment arm and visit.

.. code-block:: python

   # Multi-panel UMAP for a specific feature
   st.plot_trial_umap_panel(
       adata, 
       feature="ms_OXPHOS", 
       design=design, 
       visits=("V1", "V2")
   )

Additional visualizations
-------------------------

.. code-block:: python

   # DiD forest plot for top signatures
   st.plot_did_forest(res, title="Top DiD Effects")

   # GSEA radar chart (requires gseapy)
   st.plot_gsea_radar(
       gsea_results=res_gsea,
       top_n=8,
       title="Top Enriched Pathways"
   )

Conclusion
----------

`sctrial` provides a comprehensive toolkit—from robust bootstrapping and weighted aggregation to stratified analysis and specialized trial visualizations—ensuring that your clinical single-cell analysis is statistically rigorous and easily interpretable.
