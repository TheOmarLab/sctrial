Basic Workflow Tutorial
=======================

This tutorial guides you through the core functionalities of `sctrial` using synthetic data. We will cover data preparation, trial design specification, preprocessing, and the primary Difference-in-Differences (DiD) inference for both gene expression and cell-type abundance.

1. Setup and Data Generation
----------------------------

In clinical trials, data is often hierarchical: cells are nested within participants, who are measured across multiple visits. First, let's create a representative `AnnData` object with these properties.

.. code-block:: python

   import sctrial as st
   import numpy as np
   import pandas as pd
   from anndata import AnnData
   from scipy import sparse

   # Create 20 participants, 2 visits (V1, V2), 2 arms (Treated, Control)
   n_p, n_g = 20, 100
   obs_list = []
   for i in range(n_p):
       arm = "Treated" if i < 10 else "Control"
       for v in ["V1", "V2"]:
           # We generate 50 cells per participant per visit
           for _ in range(50):
               obs_list.append({
                   "participant_id": f"P{i}",
                   "visit": v,
                   "arm": arm,
                   "celltype": "TypeA" if i % 2 == 0 else "TypeB",
                   "age": 30 + (i % 5) * 10,  # A participant-level covariate
                   "batch": f"B{i % 3}"       # A batch covariate
               })
   obs = pd.DataFrame(obs_list)
   
   # Generate Poisson counts
   X = np.random.poisson(1.5, size=(len(obs), n_g)).astype(float)
   
   # Add a treatment effect: Genes G0-G4 increase in V2 for the Treated group
   idx_v2_treated = obs[(obs["arm"] == "Treated") & (obs["visit"] == "V2")].index
   X[idx_v2_treated, :5] += 3.0
   
   adata = AnnData(X=sparse.csr_matrix(X), obs=obs)
   adata.var_names = [f"G{i}" for i in range(n_g)]
   adata.layers["counts"] = adata.X.copy()

2. Defining the Trial Design
----------------------------

The `TrialDesign` object is a central configuration that tells `sctrial` which columns in `adata.obs` correspond to the trial's structure. This avoids passing column names to every function.

.. code-block:: python

   design = st.TrialDesign(
       participant_col="participant_id",
       visit_col="visit",
       arm_col="arm",
       arm_treated="Treated",
       arm_control="Control",
       celltype_col="celltype",
       baseline_visit="V1",
       followup_visit="V2"
   )

   # Validate that the design matches the AnnData object
   design.validate(adata)

3. Preprocessing and Scoring
----------------------------

Normalization is crucial for single-cell data. `sctrial` provides a helper to add a `log1p(CPM)` layer, which is a standard format for many downstream statistical models.

.. code-block:: python

   # Add log1p(CPM) normalization
   # This computes (counts / lib_size * 1e6).log1p()
   adata = st.add_log1p_cpm_layer(adata, counts_layer="counts", out_layer="log1p_cpm")

Gene set scoring (or module scoring) allows you to aggregate signals from multiple genes. This often increases statistical power and improves interpretability.

.. code-block:: python

   # Define a gene set
   gene_sets = {"Antigen_Presentation": ["G0", "G1", "G2"]}
   
   # Score the gene set using the 'zmean' method
   # This z-scores each gene across all cells before averaging
   adata = st.score_gene_sets(
       adata, 
       gene_sets, 
       layer="log1p_cpm", 
       method="zmean", 
       prefix="ms_"
   )

4. Difference-in-Differences (DiD) Analysis
-------------------------------------------

The core of `sctrial` is the Difference-in-Differences (DiD) model. It tests whether the change over time (baseline to follow-up) differs between the treatment and control arms.

Statistical highlights:
- **Participant Fixed Effects**: By including `C(participant_id)`, we control for baseline heterogeneity between individuals.
- **Cluster-Robust Standard Errors**: We cluster by participant to account for the non-independence of cells from the same person.
- **Covariate Support**: You can include additional variables (like `age` or `batch`) to control for potential confounders.

.. code-block:: python

   # Run DiD on the module score
   res = st.did_table(
       adata,
       features=["ms_Antigen_Presentation"],
       design=design,
       visits=("V1", "V2"),
       covariates=["age", "batch"],
       aggregate="participant_visit"  # Average cells per participant-visit before fitting
   )
   print(res)

   # Generate a human-readable summary
   summary_text = st.summarize_did_results(res)
   print(summary_text)

5. Cell-Type Abundance Analysis
-------------------------------

Treatment effects can also manifest as changes in cell-type composition. `abundance_did` uses the same DiD logic but operates on cell-type proportions (e.g., via `arcsin-sqrt` transform).

.. code-block:: python

   ab_res = st.abundance_did(
       adata,
       design=design,
       visits=("V1", "V2"),
       covariates=["age"]
   )
   print(ab_res)

6. Visualization
----------------

`sctrial` provides plotting helpers to visualize these interactions.

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot interaction for a module score
   # This shows the "divergence" between arms over time
   st.plot_trial_interaction(
       adata, 
       feature="ms_Antigen_Presentation", 
       design=design, 
       visits=("V1", "V2")
   )
   plt.title("Treatment Interaction: Antigen Presentation")
   plt.show()

   # Plot abundance interaction for TypeA cells
   st.plot_abundance_interaction(
       adata, 
       celltype="TypeA", 
       design=design, 
       visits=("V1", "V2")
   )
   plt.show()

Next Steps
----------

For more advanced analysis, see the :doc:`gsea_analysis` tutorial or learn how to handle :doc:`advanced_designs`.
