Troubleshooting Guide
=====================

This guide covers common issues and their solutions when using ``sctrial``.

.. contents:: Table of Contents
   :local:
   :depth: 2

Installation Issues
-------------------

Import Errors
~~~~~~~~~~~~~

**Problem**: ``ImportError: No module named 'sctrial'``

**Solution**:

.. code-block:: bash

   # Ensure sctrial is installed
   pip install sctrial

   # For development installation
   pip install -e .

**Problem**: ``ImportError: matplotlib is required for plotting``

**Solution**:

.. code-block:: bash

   # Install with plotting dependencies
   pip install 'sctrial[plots]'

   # Or install all optional dependencies
   pip install 'sctrial[plots,gsea]'

Data Preparation Issues
-----------------------

Missing Features Error
~~~~~~~~~~~~~~~~~~~~~~

**Error**: ``KeyError: Features not found in obs or var_names: ['ms_OXPHOS', 'ms_Glycolysis']``

**Cause**: Module scores haven't been calculated yet, or feature names have typos.

**Solution**:

1. Check that you've run ``score_gene_sets()`` before DiD analysis:

.. code-block:: python

   import sctrial as st

   # First score gene sets
   gene_sets = {"OXPHOS": ["COX7A1", "ATP5F1A"]}
   adata = st.score_gene_sets(adata, gene_sets, prefix="ms_")

   # Then run DiD
   res = st.did_table(adata, features=["ms_OXPHOS"], design=design, visits=("V1", "V2"))

2. Check for typos in feature names:

.. code-block:: python

   # List available features
   print("Obs columns:", adata.obs.columns.tolist()[:10])
   print("Var names:", adata.var_names.tolist()[:10])

Missing Required Columns
~~~~~~~~~~~~~~~~~~~~~~~~~

**Error**: ``KeyError: Required column 'participant_id' not found in adata.obs``

**Cause**: Column names in your data don't match the ``TrialDesign`` specification.

**Solution**:

1. Use ``auto_detect_design()`` to automatically detect column names:

.. code-block:: python

   # Auto-detect design
   design = st.auto_detect_design(adata)

   # Verify and adjust if needed
   print(design)

2. Or manually specify the correct column names:

.. code-block:: python

   design = st.TrialDesign(
       participant_col="donor_id",  # Your actual column name
       visit_col="timepoint",
       arm_col="treatment",
       arm_treated="Drug_A",
       arm_control="Placebo"
   )

3. Check your column names:

.. code-block:: python

   print(adata.obs.columns.tolist())

Statistical Issues
------------------

Insufficient Participants Warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Warning**: ``n_units < 4, returning NaN``

**Cause**: DiD requires at least 4 paired participants for statistical estimation.

**Solution**:

1. Check how many paired participants you have:

.. code-block:: python

   n_paired = st.count_paired(adata.obs, "visit", ["V1", "V2"])
   print(f"Paired participants: {n_paired}")

2. Verify visit filtering:

.. code-block:: python

   # Check visit distribution
   print(adata.obs.groupby(["participant_id", "visit"]).size())

3. If you truly have < 4 participants:

   - Consider pooling data from multiple similar cohorts
   - Use within-arm analysis instead: ``st.within_arm_comparison()``
   - Consult a statistician about alternative approaches

All Results are NaN
~~~~~~~~~~~~~~~~~~~~

**Problem**: DiD results show all NaN values

**Possible Causes & Solutions**:

1. **Zero variance in features**:

.. code-block:: python

   # Check feature variance
   import numpy as np
   feature_var = adata.obs["ms_OXPHOS"].var()
   print(f"Feature variance: {feature_var}")

   # If variance is very low (< 1e-8), feature is constant
   # Solution: Remove or investigate why feature has no variation

2. **Insufficient paired participants** (see above)

3. **All values in one visit**:

.. code-block:: python

   # Check visit distribution
   print(adata.obs[design.visit_col].value_counts())

   # Ensure both visits have data

Non-Significant Results
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: No features reach statistical significance

**Not necessarily a problem!** This could indicate:

1. **No true treatment effect** - This is a valid scientific finding
2. **Insufficient power** - Try:

.. code-block:: python

   # Use bootstrap for small samples
   res = st.did_table(
       adata, features=features, design=design,
       visits=("V1", "V2"),
       use_bootstrap=True,  # More robust for small N
       n_boot=999  # 999 is usually sufficient; use 9999 for final results
   )

3. **High variability** - Check effect sizes (beta_DiD) even if p-values are high
4. **Need cell-type stratification**:

.. code-block:: python

   # Try cell-type-specific analysis
   res = st.did_table_by_celltype(
       adata, features=features, design=design,
       visits=("V1", "V2")
   )

Memory Issues
-------------

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

**Error**: ``MemoryError`` or kernel crashes

**Solutions**:

1. **Subsample cells** (stratified):

.. code-block:: python

   # Downsample while maintaining structure
   import scanpy as sc
   sc.pp.subsample(adata, n_obs=50000, random_state=42)

2. **Use sparse layers**:

.. code-block:: python

   import scipy.sparse as sp

   # Ensure counts are sparse
   if not sp.issparse(adata.layers["counts"]):
       adata.layers["counts"] = sp.csr_matrix(adata.layers["counts"])

3. **Analyze fewer features at once**:

.. code-block:: python

   # Process in batches
   features = list(adata.var_names)
   batch_size = 1000

   results = []
   for i in range(0, len(features), batch_size):
       batch = features[i:i+batch_size]
       res = st.did_table(adata, features=batch, design=design, visits=("V1", "V2"))
       results.append(res)

   full_results = pd.concat(results, ignore_index=True)

Performance Issues
------------------

Very Slow Analysis
~~~~~~~~~~~~~~~~~~

**Problem**: Analysis takes too long

**Solutions**:

1. **Use pseudobulk aggregation** (recommended):

.. code-block:: python

   # Aggregate to participant-visit level (much faster)
   res = st.did_table(
       adata, features=features, design=design,
       visits=("V1", "V2"),
       aggregate="participant_visit"  # This is default and fastest
   )

2. **Reduce number of features**:

.. code-block:: python

   # Analyze only module scores, not all genes
   gene_sets = {...}
   adata = st.score_gene_sets(adata, gene_sets)
   features = [f"ms_{k}" for k in gene_sets.keys()]

3. **For GSEA, reduce gene universe**:

.. code-block:: python

   # Pre-filter genes with low expression
   sc.pp.filter_genes(adata, min_counts=10)

Plotting Issues
---------------

Matplotlib Backend Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error**: ``RuntimeError: Invalid DISPLAY variable``

**Solution**:

.. code-block:: python

   # Use non-interactive backend
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt

Missing Plot Elements
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Plots appear but are empty or missing data

**Solution**:

1. **Check data availability**:

.. code-block:: python

   # Verify feature exists
   if "ms_OXPHOS" not in adata.obs.columns:
       print("Feature not found!")

2. **Check visit filtering**:

.. code-block:: python

   # Ensure visits are present
   subset = adata[adata.obs[design.visit_col].isin(visits)]
   print(f"Cells after filtering: {subset.n_obs}")

Data Validation
---------------

Using the Diagnostic Tool
~~~~~~~~~~~~~~~~~~~~~~~~~

Before running analysis, diagnose your data:

.. code-block:: python

   # Run diagnostics
   diagnostics = st.diagnose_trial_data(adata, design, verbose=True)

   # Check warnings
   if diagnostics['warnings']:
       print("Issues found:")
       for w in diagnostics['warnings']:
           print(f"  - {w}")

   # Check recommendations
   for r in diagnostics['recommendations']:
       print(f"  💡 {r}")

Validate Before Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Validate adata structure
   issues = st.validate_adata(adata, design, strict=False)

   if issues:
       print("Validation issues:")
       for issue in issues:
           print(f"  ⚠️  {issue}")
   else:
       print("✓ Data validated successfully!")

Common Error Messages
---------------------

"No numeric features found to analyze"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: All requested features are either missing or non-numeric.

**Solution**: Check that features are in ``adata.var_names`` or ``adata.obs.columns`` and are numeric.

"Covariate varies within participant-visit"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: You're using a covariate that changes within participant-visits (e.g., age that increments).

**Solution**: Use only covariates that are constant within participant-visit groups, or aggregate them appropriately.

Getting Help
------------

If you're still stuck:

1. **Check the tutorials**:

   - Browse the :doc:`tutorials/index` for end-to-end notebook walkthroughs

2. **Search issues**: Check `GitHub Issues <https://github.com/TheOmarLab/sctrial/issues>`_

3. **Report a bug**:

   .. code-block:: python

      # Include this information
      import sctrial as st
      print(f"sctrial version: {st.__version__}")
      print(f"AnnData shape: {adata.shape}")
      print(f"Design: {design}")

      # Copy the full error traceback

4. **Ask for help**: Open a new issue on GitHub with:

   - Minimal reproducible example
   - Error message and full traceback
   - Environment information (OS, Python version, package versions)
