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

Can I automatically download example datasets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! The built-in data loaders can download directly from GEO or EBI:

.. code-block:: python

   import sctrial as st

   # Sade-Feldman melanoma immunotherapy (GSE120575)
   # Requires scanpy for cell-type annotation: pip install sctrial[plots]
   adata = st.load_sade_feldman(allow_download=True)

   # Stephenson COVID-19 (E-MTAB-10026)
   adata = st.load_stephenson_data(allow_download=True)

   # Vaccine PBMC time course (GSE171964)
   adata = st.load_vaccine_gse171964(allow_download=True)

   # AML clinical trial (GSE116256)
   # Requires scanpy: pip install sctrial[plots]
   adata = st.load_aml(allow_download=True)

   # CAR-T clinical trial (GSE290722)
   # Requires scanpy: pip install sctrial[plots]
   adata = st.load_cart(allow_download=True)

Downloads are disabled by default (``allow_download=False``) so nothing is
fetched without explicit consent. Only missing files are downloaded; files
already on disk are reused.

.. note::

   The Sade-Feldman, AML, and CAR-T loaders perform marker-based cell-type
   annotation using scanpy (Leiden clustering + Wilcoxon marker scoring).
   Install the ``plots`` extra to enable this: ``pip install sctrial[plots]``.

Data Requirements
-----------------

What data format does sctrial require?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``AnnData`` object with:

- Raw counts in ``adata.layers["counts"]`` (recommended)
- Trial metadata in ``adata.obs``:

  - Participant identifiers (e.g., "participant_id")
  - Visit labels (e.g., "visit")
  - Treatment arm (e.g., "arm") — set ``arm_col=None`` for single-arm studies
  - Cell-type annotations (optional, e.g., "celltype")

How many participants do I need?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a rough guide:

- **~4 paired participants per arm** is a practical lower bound for estimation, though inference will be imprecise.
- **~10–15 paired participants** per arm generally provides more reliable inference.
- **~20–30 paired participants** per arm is typically needed for adequate power to detect moderate effects (d ≈ 0.5).

For smaller samples, bootstrap inference (``use_bootstrap=True``) is recommended as a safeguard against anti-conservative cluster-robust standard errors.

Do I need the same number of cells per participant?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No! DiD aggregates to the participant level, so unequal cell counts are fine. The method uses OLS with cluster-robust standard errors to account for within-participant correlation.

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

When should I use bootstrap vs. standard errors?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cluster-robust standard errors (default)** are often adequate when the
number of participants per arm is reasonably large (e.g. ~15 or more) and
sample sizes are roughly balanced.

**Wild Cluster Bootstrap** (``use_bootstrap=True``) is recommended as a
safeguard when:

- The number of participants per arm is modest (e.g. fewer than ~15)
- Sample sizes are unbalanced across arms
- More accurate p-values and confidence intervals are needed for publication

.. code-block:: python

   # Default: Cluster-robust SEs (fast, adequate for larger samples)
   res = st.did_table(adata, features=features, design=design, visits=visits)

   # Bootstrap: More accurate for small samples (slower)
   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=visits,
       use_bootstrap=True,
       n_boot=999,  # Use 999 or 1999 for publication
       seed=42      # For reproducibility
   )

**Mathematical note**: Bootstrap resamples at the participant level:

.. math::

   \hat{t}_b^* = \frac{\hat{\beta}^* - \hat{\beta}}{\text{SE}(\hat{\beta}^*)}

where :math:`\hat{\beta}^*` is the bootstrap estimate.

.. _faq-sample-size:

What are the minimum sample size recommendations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Approximate guidance (two-arm DiD)**:

=========================  ==============  =======================
Scenario                   Per Arm         Total Participants
=========================  ==============  =======================
Practical lower bound      ~4 paired       ~8
Adequate for bootstrap     ~6–8 paired     ~12–16
Generally reliable         ~10–15 paired   ~20–30
Well-powered (d ≈ 0.5)     ~20+ paired     ~40+
=========================  ==============  =======================

**Single-arm paired designs** (e.g. vaccination, single-agent therapy):

=========================  =======================
Scenario                   Paired Participants
=========================  =======================
Practical lower bound      ~4
Adequate for bootstrap     ~6–8
Generally reliable         ~10–15
Well-powered (d ≈ 0.5)     ~14+
=========================  =======================

These are rules of thumb. Actual requirements depend on effect size,
variance structure, and the degree of within-participant correlation.

**Power calculation example**:

.. code-block:: python

   import sctrial as st

   # How many participants for 80% power to detect d=0.5?
   n_needed = st.sample_size_did(effect_size=0.5, power=0.80)
   print(f"Need {n_needed} per arm ({2*n_needed} total)")

   # What power do I have with 15 per arm?
   power = st.power_did(n_per_group=15, effect_size=0.5)
   print(f"Power with 15/arm: {power:.1%}")

   # Generate power curve
   ns, powers = st.power_curve(n_range=(5, 50), effect_size=0.5)

   # For single-arm paired designs (e.g. vaccination studies):
   n_paired = st.sample_size_paired(effect_size=0.5, power=0.80)
   print(f"Need {n_paired} paired participants")

   power = st.power_paired(n_participants=6, effect_size=0.8)
   print(f"Power with 6 participants: {power:.1%}")

   mde = st.sensitivity_paired(n_participants=6, power=0.80)
   print(f"Min detectable effect: {mde:.2f} SD")

**Practical considerations**:

- Single-cell data doesn't increase statistical power (participants do)
- Cell counts help reduce noise, but n_participants drives inference
- Small samples → use bootstrap and report effect sizes with CIs
- Consider mixed effects models for partial pooling with small n

How should I handle missing data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Types of missingness in trials**:

1. **Missing visits** (participant dropout)
2. **Missing cells** (technical failure)
3. **Missing annotations** (cell type not assigned)

**For missing visits (most critical)**:

DiD requires data at BOTH visits. Participants with missing visits are dropped:

.. code-block:: python

   # Check pairing before analysis
   n_paired = st.count_paired(adata.obs, "visit", ["V1", "V2"])
   print(f"Paired participants: {n_paired}")

   # Diagnose issues
   diag = st.diagnose_trial_data(adata, design, verbose=True)

**Options for missing data**:

1. **Complete case analysis** (default, recommended for trials)

   - Uses only participants with data at both visits
   - Valid if missingness is independent of treatment effect (MCAR/MAR)

2. **Multiple imputation** (advanced)

   - Not built into sctrial; use external packages
   - Only if missingness mechanism is well-understood

3. **Last observation carried forward** (NOT recommended)

   - Biases results; avoid in clinical trials

**For missing cell-type annotations**:

.. code-block:: python

   # Option 1: Exclude unassigned cells
   adata = adata[adata.obs["celltype"] != "Unknown"].copy()

   # Option 2: Include as separate category
   # (if "Unknown" cells are biologically meaningful)

**Diagnosing missingness patterns**:

.. code-block:: python

   # Check for systematic missingness
   import pandas as pd

   missing = (
       adata.obs
       .groupby(["arm", "visit"])["participant_id"]
       .nunique()
   )
   print("Participants per arm × visit:")
   print(missing.unstack())

   # Check if dropout differs by arm (potential bias!)
   dropout_by_arm = adata.obs.groupby("arm")["participant_id"].nunique()

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

When aggregated, if an ``n_cells`` column is present, Weighted Least Squares (WLS) is
used with inverse-variance weights proportional to the number of cells per group.
This gives more weight to participant-visit groups with more cells (lower variance means).

What are Wild Cluster Bootstrap p-values?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A resampling method that provides more accurate p-values **and confidence
intervals** when you have few participants (< 15). Enable with
``use_bootstrap=True``.

It resamples at the participant level (not cell level) to respect clustering.
When enabled, the output DataFrame includes additional columns:

- ``p_DiD_boot``: bootstrap p-value
- ``se_DiD_boot``: bootstrap standard error
- ``ci_lo_boot`` / ``ci_hi_boot``: bootstrap-t confidence interval

The CI uses the bootstrap-t method (Hall, 1992), which applies quantiles of the
bootstrap *t*-distribution to the observed point estimate and standard error.

How do I interpret beta_DiD?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``standardize=True`` (default), ``beta_DiD`` is in standard deviation units (like Cohen's d).

- ``|beta_DiD|`` > 0.2: Small effect
- ``|beta_DiD|`` > 0.5: Medium effect
- ``|beta_DiD|`` > 0.8: Large effect

Positive beta_DiD = feature increased more (or decreased less) in treated vs. control.

For standardized effect sizes (Cohen's d or Hedge's g), use ``add_effect_sizes_to_did()``:

.. code-block:: python

   res = st.did_table(adata, features=features, design=design, visits=("V1", "V2"))
   res = st.add_effect_sizes_to_did(res)

   # Columns added: effect_size, effect_size_lower, effect_size_upper,
   #                effect_size_interpretation
   print(res[["feature", "beta_DiD", "effect_size", "effect_size_interpretation"]])

What FDR threshold should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Exploratory**: FDR < 0.10 or 0.25
- **Standard**: FDR < 0.05
- **Stringent**: FDR < 0.01

Always report FDR-corrected p-values, not raw p-values, when testing multiple features.

How do I use Bayesian DiD?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Bayesian DiD provides posterior distributions and credible intervals.
Requires PyMC (``pip install sctrial[bayes]``):

.. code-block:: python

   res = st.did_table_bayes(
       adata, features=features, design=design,
       visits=("V1", "V2"),
       prior_scale=1.0,    # Scale for coefficient priors
       sigma_scale=1.0,    # Scale for residual SD prior
       draws=2000,
       chains=4,
   )

Run a prior predictive check first to ensure priors are reasonable:

.. code-block:: python

   check = st.prior_predictive_check(
       adata, features=features, design=design,
       visits=("V1", "V2"),
   )
   print(f"Priors cover data: {check['prior_covers_data']}")

What is the difference between FDR_DiD and FDR_DiD_global?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``module_score_did_by_pool()``, when ``fdr_within`` is set:

- **FDR_DiD**: Corrected within each group (e.g., per module). Useful for
  exploratory analysis but does not control the overall false discovery rate.
- **FDR_DiD_global**: Corrected across all tests globally. Use this for
  properly controlled inference.

.. code-block:: python

   res = st.module_score_did_by_pool(
       pb, design=design, visits=("V1", "V2"),
       fdr_within="module",
       fdr_global=True,  # default
   )

   # For publication: use global FDR
   sig = res[res["FDR_DiD_global"] < 0.05]

Statistical Assumptions & Limitations
--------------------------------------

What assumptions does DiD require?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DiD relies on several key assumptions for causal interpretation:

1. **Parallel trends**: In the absence of treatment, both arms would have
   followed the same trajectory. This cannot be tested directly with two time
   points, but can be assessed with pre-treatment data using
   ``test_parallel_trends()`` in multi-timepoint designs.

2. **No anticipation**: Participants do not change behaviour before treatment
   begins. Violated if, e.g., treated participants alter expression profiles
   at the baseline visit in anticipation of therapy.

3. **SUTVA (Stable Unit Treatment Value Assumption)**: No interference or
   spillover between participants. Each participant's outcome depends only on
   their own treatment assignment.

4. **No post-treatment confounding**: Covariates included in the model should
   be pre-treatment (baseline) variables. Conditioning on post-treatment
   variables can introduce bias (see Angrist & Pischke, 2009).

These assumptions are inherent to the DiD framework and cannot be verified by
the software alone. Users should assess their plausibility for each study
design. Sensitivity analysis (``e_value_rr()``) can quantify how strong an
unmeasured confounder would need to be to explain away the observed effect.

Are there limitations to abundance analysis?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``abundance_did()`` uses an arcsin-square-root transform on cell-type
proportions, which is a standard variance-stabilising transformation for
compositional data (Shao, 1999). This approach:

- Is simple, interpretable, and well-suited to the DiD framework
- Handles zero proportions gracefully (arcsin(0) = 0)
- Has been used in single-cell clinical trial analyses (Squair et al., 2021)

However, proportions are inherently compositional: an increase in one cell type
forces decreases in others. Fully compositional approaches (e.g., Bayesian
multinomial-logit models, scCODA; Büttner et al., 2021) explicitly model this
constraint. Consider these alternatives when compositional effects are a
primary concern and sample sizes are sufficient.

What are the limitations of survival analysis?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hazard_regression_with_features()`` fits per-feature Cox proportional hazards
models. Key caveats:

- **Proportional hazards assumption**: The hazard ratio for each feature is
  assumed constant over time. Violated when treatment effects emerge or wane.
  Assess with Schoenfeld residuals (external to sctrial).

- **Separation**: When a feature perfectly predicts survival, the Cox model can
  fail to converge or produce infinite hazard ratios. sctrial detects these
  cases and returns NaN rather than misleading estimates.

- **Multiple testing**: Testing many features inflates false positives. Always
  use the FDR-corrected column (``FDR``) rather than raw p-values.

- **Not a causal model**: Unlike DiD, Cox regression on observational features
  estimates associations, not causal effects, unless combined with an
  appropriate causal framework.

Can I use categorical covariates in Bayesian DiD?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. ``did_table_bayes()`` automatically dummy-encodes categorical columns
(e.g., sex, batch, site) using one-hot encoding with the first category
dropped to avoid multicollinearity with the intercept. You can pass
categorical columns directly in the ``covariates`` parameter:

.. code-block:: python

   res = st.did_table_bayes(
       adata, features, design, visits=("V1", "V2"),
       covariates=["sex", "site"],  # categorical columns are auto-encoded
   )

This follows standard Bayesian practice for including indicator variables
in linear models (Gelman & Hill, 2006).

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

How do I harmonize response labels across datasets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``harmonize_response()`` to standardize different response label conventions:

.. code-block:: python

   import sctrial as st

   adata = st.harmonize_response(adata)
   # Maps R/NR, Responder/Non-responder, etc. → "Responder"/"Non-responder"
   print(adata.obs["response_harmonized"].value_counts())

Feature and Gene Set Questions
-------------------------------

Can I test individual genes or do I need gene sets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both work:

**Gene sets** (recommended for robustness):

.. code-block:: python

   gene_sets = {"OXPHOS": ["COX7A1", "ATP5F1A", ...]}
   adata = st.score_gene_sets(adata, gene_sets, prefix="ms_")
   features = ["ms_OXPHOS"]

**Individual genes**:

.. code-block:: python

   features = ["CD3D", "CD8A", "IL2RA"]

For genome-wide tests, use stringent FDR thresholds (e.g., 0.01).

What if some genes in my gene set are missing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``score_gene_sets()`` automatically handles this:

- Uses only genes present in ``adata.var_names``; missing genes are silently
  dropped from the set.
- A warning is logged when any genes are missing (at ``INFO`` level) or when
  the overlap falls below ``min_genes`` (at ``WARNING`` level).
- By default, ``min_genes=3`` — at least 3 genes must overlap to compute a
  score.  If fewer are found, the score is set to ``NaN``.
- Duplicate gene names within a set are automatically removed.

.. code-block:: python

   gene_sets = {"MySet": ["GENE1", "GENE2_MISSING", "GENE3"]}
   # Will score using GENE1 and GENE3, skip GENE2_MISSING
   # (provided min_genes <= 2; default min_genes=3 would yield NaN here)

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

Yes! Use ``did_table_parallel`` for parallel DiD across many features:

.. code-block:: python

   res = st.did_table_parallel(
       adata,
       features=list(adata.var_names),
       design=design,
       visits=("V1", "V2"),
       n_jobs=4,  # Number of parallel workers
   )

For smaller gene sets, the standard ``did_table`` is often fast enough.
You can also process in batches manually:

.. code-block:: python

   import pandas as pd

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
