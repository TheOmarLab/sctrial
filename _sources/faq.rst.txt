Frequently Asked Questions
==========================

General Questions
-----------------

What is sctrial?
~~~~~~~~~~~~~~~~

``sctrial`` is a Python package for trial-aware statistical inference on single-cell RNA-seq data from clinical trials. It implements Difference-in-Differences (DiD) analysis with participant fixed effects, designed specifically for longitudinal randomized trials at single-cell resolution.

What makes sctrial different from standard scRNA-seq tools?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Traditional tools (like Scanpy's rank_genes_groups or Seurat's FindMarkers) don't account for:

- Longitudinal paired data structure
- Treatment-by-time interactions (the core of DiD)
- Participant-level clustering for valid inference
- Clinical trial design elements (crossovers, arms, visits)

``sctrial`` is purpose-built for these experimental designs.

Can I use sctrial with Seurat objects?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Convert Seurat to AnnData:

.. code-block:: r

   # In R
   library(Seurat)
   library(SeuratDisk)

   # Save as h5Seurat
   SaveH5Seurat(seurat_obj, filename = "data.h5Seurat")
   Convert("data.h5Seurat", dest = "h5ad")

.. code-block:: python

   # In Python
   import scanpy as sc
   adata = sc.read_h5ad("data.h5ad")

   # Continue with sctrial
   import sctrial as st
   design = st.auto_detect_design(adata)

Data Requirements
-----------------

What data format does sctrial require?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``AnnData`` object with:

- Raw counts in ``adata.layers["counts"]`` (recommended)
- Trial metadata in ``adata.obs``:

  - Participant identifiers (e.g., "participant_id")
  - Visit labels (e.g., "visit")
  - Treatment arm (e.g., "arm")
  - Cell-type annotations (optional, e.g., "celltype")

How many participants do I need?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum**: 4 paired participants per arm (statistical requirement)

**Recommended**: 10-15 paired participants for robust inference

**Ideal**: 20-30 paired participants for adequate power to detect moderate effects

For smaller samples, use bootstrap: ``use_bootstrap=True``

Do I need the same number of cells per participant?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No! DiD aggregates to the participant level, so unequal cell counts are fine. The method uses weighted least squares to account for this.

Can I use sctrial with cross-sectional data (no longitudinal measurements)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, DiD requires longitudinal (paired) data. For cross-sectional comparisons, use:

.. code-block:: python

   res = st.between_arm_comparison(
       adata,
       visit="endpoint",
       features=features,
       design=design
   )

Statistical Questions
---------------------

What is Difference-in-Differences (DiD)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DiD tests whether the *change* from baseline to follow-up differs between treatment arms. This accounts for:

- Baseline differences between arms
- Secular trends affecting both arms equally

Formula: ``(Treated_Post - Treated_Pre) - (Control_Post - Control_Pre)``

The DiD coefficient (``beta_DiD``) estimates the treatment effect.

Should I use cell-level or participant-level aggregation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Always use participant-level** (``aggregate="participant_visit"``, which is the default) for valid statistical inference.

Why? Cells from the same participant are not independent, violating regression assumptions. Participant-level aggregation (pseudobulk) respects the experimental design.

What are Wild Cluster Bootstrap p-values?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A resampling method that provides more accurate p-values when you have few participants (< 15). Enable with ``use_bootstrap=True``.

It resamples at the participant level (not cell level) to respect clustering.

How do I interpret beta_DiD?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``standardize=True`` (default), ``beta_DiD`` is in standard deviation units (like Cohen's d).

- |beta_DiD| > 0.2: Small effect
- |beta_DiD| > 0.5: Medium effect
- |beta_DiD| > 0.8: Large effect

Positive beta_DiD = feature increased more (or decreased less) in treated vs. control.

What FDR threshold should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Exploratory**: FDR < 0.10 or 0.25
- **Standard**: FDR < 0.05
- **Stringent**: FDR < 0.01

Always report FDR-corrected p-values, not raw p-values, when testing multiple features.

Analysis Questions
------------------

Can I test for batch effects?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Include batch as a covariate:

.. code-block:: python

   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2"),
       covariates=["batch"]
   )

Or use ComBat for batch correction before analysis:

.. code-block:: python

   import scanpy as sc
   sc.pp.combat(adata, key='batch')

How do I handle multiple time points?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DiD is designed for 2 time points. For more:

1. **Primary analysis**: Compare baseline to primary endpoint

.. code-block:: python

   res = st.did_table(adata, features=features, design=design, visits=("Day_0", "Week_12"))

2. **Secondary analyses**: Test other visit pairs

.. code-block:: python

   # Early response
   res_early = st.did_table(adata, features=features, design=design, visits=("Day_0", "Week_2"))

   # Late response
   res_late = st.did_table(adata, features=features, design=design, visits=("Week_2", "Week_12"))

Can I do dose-response analysis?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not directly with DiD. Options:

1. **Pairwise comparisons**: Compare each dose to control separately
2. **Treat dose as numeric covariate**: Include in ``covariates`` parameter
3. **Post-hoc trend test**: Use external tools on DiD estimates

What if I have crossover design?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify the crossover column in ``TrialDesign``:

.. code-block:: python

   design = st.TrialDesign(
       participant_col="participant_id",
       visit_col="visit",
       arm_col="arm",
       arm_treated="Treatment",
       arm_control="Control",
       crossover_col="crossed_over"  # Boolean column
   )

   # Exclude crossovers from primary analysis
   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2"),
       exclude_crossovers=True  # Recommended
   )

How do I test for cell-type-specific effects?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   res = st.did_table_by_celltype(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2")
   )

   # Results stratified by cell type
   print(res.groupby("celltype")["FDR_DiD"].min())

Feature and Gene Set Questions
-------------------------------

Can I test individual genes or do I need gene sets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both work:

**Gene sets** (recommended for robustness):

.. code-block:: python

   gene_sets = {"OXPHOS": ["COX7A1", "ATP5F1A", ...]}
   adata = st.score_gene_sets(adata, gene_sets)
   features = ["ms_OXPHOS"]

**Individual genes**:

.. code-block:: python

   features = ["CD3D", "CD8A", "IL2RA"]

For genome-wide tests, use stringent FDR thresholds (e.g., 0.01).

What if some genes in my gene set are missing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``score_gene_sets()`` automatically handles this:

- Uses genes present in ``adata.var_names``
- Skips missing genes with a warning
- Requires at least 1 gene to calculate a score

.. code-block:: python

   gene_sets = {"MySet": ["GENE1", "GENE2_MISSING", "GENE3"]}
   # Will score using GENE1 and GENE3, skip GENE2_MISSING

Should I use mean or zmean for gene set scoring?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**zmean** (recommended):

- Z-normalizes each gene before averaging
- Robust to genes with different expression levels
- Better for cross-dataset comparisons

**mean**:

- Simple average
- Use when all genes have similar dynamic range

.. code-block:: python

   # Recommended
   adata = st.score_gene_sets(adata, gene_sets, method="zmean")

How do I incorporate pathway databases?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use with GSEA integration:

.. code-block:: python

   # Run DiD-aware GSEA
   res = st.run_gsea_did(
       adata,
       gene_sets="KEGG_2021_Human",  # Or MSigDB, GO, etc.
       design=design,
       visits=("V1", "V2"),
       rank_by="signed_confidence"
   )

   # Or custom gene sets
   from gseapy import get_library_name
   print(get_library_name())  # See available databases

Technical Questions
-------------------

Why am I getting NaN results?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common causes:

1. **Too few participants** (< 4 paired)
2. **Zero variance** in feature
3. **Missing data** in required columns
4. **Wrong visit labels**

Run diagnostics:

.. code-block:: python

   diag = st.diagnose_trial_data(adata, design, verbose=True)

Can I run sctrial in parallel?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not currently built-in. For large gene sets, process in batches:

.. code-block:: python

   import numpy as np

   features = list(adata.var_names)
   batch_size = 1000

   results = []
   for i in range(0, len(features), batch_size):
       batch = features[i:i+batch_size]
       res = st.did_table(adata, features=batch, design=design, visits=("V1", "V2"))
       results.append(res)

   full_res = pd.concat(results)

How do I save and reload results?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save results
   res.to_csv("did_results.csv", index=False)
   adata.write_h5ad("processed_adata.h5ad")

   # Reload
   import pandas as pd
   import scanpy as sc

   res = pd.read_csv("did_results.csv")
   adata = sc.read_h5ad("processed_adata.h5ad")

Can I use sctrial with other omics (ATAC-seq, proteomics)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! As long as your data is in ``AnnData`` format with appropriate trial metadata.

For ATAC-seq: Use peak accessibility as features
For proteomics: Use protein abundance as features

The statistical framework is the same.

Integration Questions
---------------------

Can I combine sctrial with Scanpy?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Use Scanpy for QC, normalization, and clustering:

.. code-block:: python

   import scanpy as sc
   import sctrial as st

   # Scanpy workflow
   sc.pp.filter_cells(adata, min_genes=200)
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)
   sc.tl.pca(adata)
   sc.pp.neighbors(adata)
   sc.tl.leiden(adata)

   # sctrial workflow
   design = st.TrialDesign(...)
   res = st.did_table(adata, features=..., design=design, visits=("V1", "V2"))

How do I visualize results in UMAP space?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import scanpy as sc

   # Compute UMAP (if not already done)
   sc.tl.umap(adata)

   # Visualize by trial design
   st.plot_trial_umap_panel(
       adata,
       feature="ms_OXPHOS",
       design=design,
       visits=("V1", "V2")
   )

Still have questions?
---------------------

- Check the :doc:`tutorials/index` for step-by-step examples
- Browse :doc:`troubleshooting` for common issues
- Open an issue on `GitHub <https://github.com/TheOmarLab/sctrial/issues>`_
