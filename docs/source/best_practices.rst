Best Practices Guide
====================

This guide provides recommendations for robust and reproducible trial analysis with ``sctrial``.

.. contents:: Table of Contents
   :local:
   :depth: 2

Study Design
------------

Data Format Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``AnnData`` object with:

- Raw counts in ``adata.layers["counts"]`` (recommended)
- Trial metadata in ``adata.obs``:

  - Participant identifiers (e.g., "participant_id")
  - Visit labels (e.g., "visit")
  - Treatment arm (e.g., "arm")
  - Cell-type annotations (optional, e.g., "celltype")


Sample Size Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum Requirements**:

- **Two-arm DiD**: At least **4 paired participants per arm** (8 total)
- **Single-arm paired**: At least **4 paired participants**
- Recommend **10-15 paired participants** for reliable inference
- For **< 15 participants**: Always use bootstrap (``use_bootstrap=True``)

**Power Considerations**:

.. code-block:: python

   # For adequate power to detect moderate effects (d=0.5)
   # Recommend n=20-30 paired participants per arm

   # Check your sample size
   n_paired = st.count_paired(adata.obs, "visit", ["baseline", "followup"])
   print(f"Paired participants: {n_paired}")

   if n_paired < 15:
       print("⚠️  Consider using bootstrap inference")

Visit Selection
~~~~~~~~~~~~~~~

.. code-block:: python

   # Best practice: Clearly define baseline and primary outcome visits
   design = st.TrialDesign(
       participant_col="participant_id",
       visit_col="visit",
       arm_col="arm",
       arm_treated="Treatment",
       arm_control="Control"
   )

   # Use consistent visit labels
   baseline_visit = "Day_0"
   outcome_visit = "Week_12"

   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=(baseline_visit, outcome_visit)
   )

Data Preprocessing
------------------

Quality Control
~~~~~~~~~~~~~~~

Perform QC **before** using ``sctrial``:

.. code-block:: python

   import scanpy as sc

   # 1. Remove low-quality cells
   sc.pp.filter_cells(adata, min_genes=200)
   sc.pp.filter_genes(adata, min_cells=3)

   # 2. Calculate QC metrics
   adata.var['mt'] = adata.var_names.str.startswith('MT-')
   sc.pp.calculate_qc_metrics(
       adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True
   )

   # 3. Filter outliers
   adata = adata[adata.obs.pct_counts_mt < 20, :]

   # 4. THEN use sctrial for trial-aware analysis
   adata = st.add_log1p_cpm_layer(adata, counts_layer="counts")

Normalization
~~~~~~~~~~~~~

**Recommended approach**:

.. code-block:: python

   # Use log1p-CPM normalization (built-in)
   adata = st.add_log1p_cpm_layer(
       adata,
       counts_layer="counts",
       out_layer="log1p_cpm"
   )

   # Use this layer for scoring and DiD
   adata = st.score_gene_sets(adata, gene_sets, layer="log1p_cpm")

.. note::

   ``add_log1p_cpm_layer()`` performs input validation before normalizing:

   - **Negative counts** raise a ``ValueError`` — fix upstream QC before calling.
   - **NaN or inf values** emit a warning and may produce unexpected results.
   - **All-zero cells** emit a warning (library size = 0 yields ``log1p(0) = 0`` for every gene).

   The function also records provenance metadata in ``adata.uns["log1p_cpm_info"]``
   (source layer, target layer, timestamp) for reproducibility tracking.

**Alternative for batch correction**:

.. code-block:: python

   # If you need batch correction, do it before sctrial
   import scanpy as sc

   # Standard Scanpy workflow
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)
   sc.pp.combat(adata, key='batch')  # Batch correction

   # Then store for sctrial
   adata.layers["log1p_normalized"] = adata.X.copy()

   # Use in sctrial
   adata = st.score_gene_sets(adata, gene_sets, layer="log1p_normalized")

Statistical Analysis
--------------------

Aggregation Strategies
~~~~~~~~~~~~~~~~~~~~~~~

**Recommendation**: Use ``aggregate="participant_visit"`` (default)

.. code-block:: python

   # ✓ RECOMMENDED: Pseudobulk aggregation
   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2"),
       aggregate="participant_visit"  # Default
   )

**Why?**

- Respects statistical independence (participants, not cells, are sampling units)
- Matches clinical trial analysis conventions
- Provides valid p-values and standard errors
- Much faster for large datasets

**Avoid** ``aggregate="cell"`` for inferential statistics:

.. code-block:: python

   # ✗ NOT RECOMMENDED for p-values
   # (cells from same participant are not independent)
   res = st.did_table(
       adata,
       features=features,
       design=design,
       aggregate="cell"  # Violates independence assumption
   )

Bootstrap for Small Samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **n < 15 paired participants**, use Wild Cluster Bootstrap:

.. code-block:: python

   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2"),
       use_bootstrap=True,  # More robust for small N
       n_boot=999,  # 999 is usually sufficient; use 9999 for final results
       seed=42  # For reproducibility
   )

   # Bootstrap adds extra columns to the result DataFrame:
   #   p_DiD_boot   — bootstrap p-value
   #   se_DiD_boot  — bootstrap standard error
   #   ci_lo_boot   — lower bound of bootstrap-t 95% CI
   #   ci_hi_boot   — upper bound of bootstrap-t 95% CI

Multiple Testing Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Always use FDR correction** for multiple features:

.. code-block:: python

   res = st.did_table(adata, features=features, design=design, visits=("V1", "V2"))

   # Use FDR-corrected p-values (automatically added)
   significant = res[res["FDR_DiD"] < 0.05]
   print(f"Significant features: {len(significant)}")

**FDR threshold recommendations**:

- **Exploratory analysis**: FDR < 0.10 or 0.25
- **Confirmatory analysis**: FDR < 0.05
- **Discovery**: Report all results with FDR values; let readers judge

.. code-block:: python

   # Report at multiple thresholds
   print(f"FDR < 0.05: {sum(res['FDR_DiD'] < 0.05)}")
   print(f"FDR < 0.10: {sum(res['FDR_DiD'] < 0.10)}")
   print(f"FDR < 0.25: {sum(res['FDR_DiD'] < 0.25)}")

**Grouped FDR correction**: When using ``module_score_did_by_pool()`` with
``fdr_within="module"``, FDR is corrected within each group separately. This does
**not** control the overall false discovery rate across all tests. Use the
``FDR_DiD_global`` column (enabled by default via ``fdr_global=True``) for a
properly controlled global correction:

.. code-block:: python

   res = st.module_score_did_by_pool(
       pb, design=design, visits=("V1", "V2"),
       fdr_within="module",   # Per-module FDR (exploratory)
       fdr_global=True,       # Also compute global FDR (recommended)
   )

   # Use FDR_DiD_global for properly controlled inference
   significant = res[res["FDR_DiD_global"] < 0.05]

Covariates
~~~~~~~~~~

**When to use covariates**:

- To adjust for known confounders (age, sex, batch)
- To increase statistical power

**Requirements**:

- Must be **numeric** or constant within participant-visit
- Should be **pre-specified**, not data-driven

.. code-block:: python

   # ✓ GOOD: Baseline covariates
   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2"),
       covariates=["age", "sex_numeric", "batch"]
   )

   # ✗ BAD: Time-varying covariates that differ within participant-visit
   # Will raise an error if they vary

Bayesian DiD
~~~~~~~~~~~~~

When using ``did_table_bayes()``, prior specification matters for small samples.
Use ``prior_predictive_check()`` to verify that your priors produce plausible
outcome values:

.. code-block:: python

   # Check priors before fitting
   check = st.prior_predictive_check(
       adata, features=features, design=design,
       visits=("V1", "V2"),
       prior_scale=1.0,
       sigma_scale=1.0,
   )

   if not check["prior_covers_data"]:
       print("Priors may be too narrow; consider increasing prior_scale")

   # Fit with custom priors
   res = st.did_table_bayes(
       adata, features=features, design=design,
       visits=("V1", "V2"),
       prior_scale=2.0,    # Wider priors for more uncertainty
       sigma_scale=2.0,
   )

**Prior sensitivity**: If results change substantially with different
``prior_scale`` values (e.g., 0.5 vs 2.0), the data is insufficient to
overwhelm the prior. Report results from multiple prior specifications.

Feature Selection
-----------------

Gene Sets vs. Individual Genes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommendation**: Use biologically meaningful gene sets

.. code-block:: python

   # ✓ RECOMMENDED: Test gene sets/modules
   gene_sets = {
       "OXPHOS": ["COX7A1", "ATP5F1A", "NDUFA1", ...],
       "Glycolysis": ["PKM", "LDHA", "HK2", ...],
       "IFN_Response": ["ISG15", "MX1", "OAS1", ...],
   }

   adata = st.score_gene_sets(adata, gene_sets, layer="log1p_cpm", method="zmean", prefix="ms_")
   features = [f"ms_{k}" for k in gene_sets.keys()]

   res = st.did_table(adata, features=features, design=design, visits=("V1", "V2"))

**Why gene sets?**

- More robust to technical noise
- Biologically interpretable
- Reduces multiple testing burden
- Better statistical power

**For genome-wide analysis**:

.. code-block:: python

   # If testing all genes, use stringent FDR threshold
   all_genes = adata.var_names.tolist()
   res = st.did_table(adata, features=all_genes, design=design, visits=("V1", "V2"))

   # Very stringent threshold due to ~20,000 tests
   significant = res[res["FDR_DiD"] < 0.01]

Scoring Method Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For general use: z-normalized mean (recommended)
   adata = st.score_gene_sets(adata, gene_sets, method="zmean")

   # For simple averaging: mean
   adata = st.score_gene_sets(adata, gene_sets, method="mean")

**``zmean`` advantages**:

- Normalizes for gene expression level
- More robust across datasets
- Recommended for cross-cohort comparisons

Cell-Type Analysis
------------------

When to Stratify
~~~~~~~~~~~~~~~~

Stratify by cell type when:

1. You expect heterogeneous treatment effects across cell types
2. You want cell-type-specific insights

.. code-block:: python

   # Stratified analysis
   res = st.did_table_by_celltype(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2"),
       celltypes=["CD4_T", "CD8_T", "Monocytes", "B_cells"]
   )

   # Results include celltype column
   print(res.groupby("celltype")["FDR_DiD"].apply(lambda x: sum(x < 0.05)))

**Note**: Stratification increases multiple testing burden; use ``FDR_DiD_stratified`` column.

Abundance Analysis
~~~~~~~~~~~~~~~~~~

Test for changes in cell-type proportions:

.. code-block:: python

   # Cell-type abundance DiD
   res_abundance = st.abundance_did(
       adata,
       design=design,
       visits=("V1", "V2")
   )

   # Significant shifts in composition
   print(res_abundance[res_abundance["FDR_DiD"] < 0.05])

Reproducibility
---------------

Set Random Seeds
~~~~~~~~~~~~~~~~

.. code-block:: python

   # For bootstrap
   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2"),
       use_bootstrap=True,
       seed=42  # Reproducible bootstrap
   )

   # For gene set scoring (if using random features)
   import numpy as np
   np.random.seed(42)

Document Versions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import sctrial as st
   import scanpy as sc
   import anndata as ad

   # Log versions
   print(f"sctrial: {st.__version__}")
   print(f"scanpy: {sc.__version__}")
   print(f"anndata: {ad.__version__}")

Save Intermediate Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save processed AnnData
   adata.write_h5ad("processed_data.h5ad")

   # Save DiD results
   res.to_csv("did_results.csv", index=False)

   # Save design
   import json
   design_dict = {
       "participant_col": design.participant_col,
       "visit_col": design.visit_col,
       "arm_col": design.arm_col,
       "arm_treated": design.arm_treated,
       "arm_control": design.arm_control,
   }
   with open("design.json", "w") as f:
       json.dump(design_dict, f, indent=2)

Reporting Results
-----------------

Minimum Reporting Standards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always report:

1. **Sample size**: Number of paired participants
2. **Effect size**: beta_DiD (standardized effect)
3. **Uncertainty**: Standard errors or confidence intervals
4. **Multiple testing**: FDR-corrected p-values
5. **Statistical method**: "Difference-in-differences with participant fixed effects"

.. code-block:: python

   # Generate summary report
   summary = st.summarize_did_results(res, alpha=0.05)
   print(summary)

Visualization
~~~~~~~~~~~~~

Create publication-quality figures:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Interaction plot for top feature
   top_feature = res.sort_values("p_DiD").iloc[0]["feature"]

   fig = st.plot_trial_interaction(
       adata,
       feature=top_feature,
       design=design,
       visits=("V1", "V2")
   )
   plt.savefig(f"interaction_{top_feature}.pdf", dpi=300, bbox_inches="tight")

   # Forest plot for all significant features
   sig_res = res[res["FDR_DiD"] < 0.05]
   fig = st.plot_did_forest(sig_res)
   plt.savefig("forest_plot.pdf", dpi=300, bbox_inches="tight")

Common Pitfalls to Avoid
-------------------------

1. **Testing at cell level for inference** → Use ``aggregate="participant_visit"``
2. **Ignoring multiple testing** → Always check FDR, not raw p-values
3. **Cherry-picking significant results** → Pre-specify hypotheses when possible
4. **Using wrong visits** → Double-check baseline vs. outcome timepoints
5. **Ignoring insufficient power** → Check sample size, use bootstrap for small N
6. **Not validating data** → Use ``st.diagnose_trial_data()`` before analysis

Pre-Analysis Checklist
-----------------------

Before running DiD analysis:

.. code-block:: python

   # ✓ Run diagnostics
   diag = st.diagnose_trial_data(adata, design, verbose=True)

   # ✓ Check for warnings
   if diag["warnings"]:
       print("Address these issues before proceeding:")
       for w in diag["warnings"]:
           print(f"  - {w}")

   # ✓ Verify design
   print(design)
   print(f"Treated arm: {design.arm_treated}")
   print(f"Control arm: {design.arm_control}")

   # ✓ Check visits
   visits_to_compare = ("baseline", "week_12")
   n_paired = st.count_paired(adata.obs, design.visit_col, visits_to_compare)
   print(f"Paired participants: {n_paired}")

   # ✓ Validate features
   valid, missing = st.validate_features(adata, features, allow_missing=False)
   print(f"Valid features: {len(valid)}")

   # ✓ Check normalization
   if "log1p_cpm" not in adata.layers:
       adata = st.add_log1p_cpm_layer(adata)

   # NOW run analysis
   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=visits_to_compare,
       use_bootstrap=(n_paired < 15)  # Auto-enable for small samples
   )
