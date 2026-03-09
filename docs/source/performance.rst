Performance Guide
=================

This guide covers performance optimization strategies for large-scale single-cell trial analysis.

.. contents:: Table of Contents
   :local:
   :depth: 2

Performance Overview
--------------------

Typical Performance
~~~~~~~~~~~~~~~~~~~

On a standard laptop (16GB RAM):

- **10K cells, 50 features**: < 5 seconds
- **100K cells, 500 features**: ~30-60 seconds
- **500K cells, 2000 features**: ~5-10 minutes

Memory requirements:

- **100K cells**: ~2-4 GB
- **500K cells**: ~8-12 GB
- **1M+ cells**: Requires optimization strategies below

Optimization Strategies
-----------------------

1. Use Pseudobulk Aggregation (Most Important!)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Always use** ``aggregate="participant_visit"`` (the default):

.. code-block:: python

   # ✓ FAST: Aggregates to ~100-1000 participant-visit combinations
   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2"),
       aggregate="participant_visit"  # Default
   )

**Avoid** cell-level analysis for large datasets:

.. code-block:: python

   # ✗ SLOW: Fits regression on millions of cells
   res = st.did_table(
       adata,
       features=features,
       design=design,
       aggregate="cell"  # Not recommended
   )

**Performance impact**: 10-100x faster with pseudobulk

2. Analyze Gene Sets, Not All Genes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ✓ FAST: Test 10-50 gene sets
   gene_sets = {
       "OXPHOS": ["COX7A1", "ATP5F1A", ...],
       "Glycolysis": ["PKM", "LDHA", ...],
       # ... 50 total sets
   }
   adata = st.score_gene_sets(adata, gene_sets, prefix="ms_")
   features = [f"ms_{k}" for k in gene_sets.keys()]

   # Fast analysis
   res = st.did_table(adata, features=features, design=design, visits=("V1", "V2"))

vs.

.. code-block:: python

   # ✗ SLOWER: Test all ~20,000 genes
   features = adata.var_names.tolist()
   res = st.did_table(adata, features=features, design=design, visits=("V1", "V2"))

**Performance impact**: 100-400x faster with gene sets

3. Use Sparse Matrices
~~~~~~~~~~~~~~~~~~~~~~~

Ensure counts are stored as sparse matrices:

.. code-block:: python

   import scipy.sparse as sp

   # Check if sparse
   print(f"Is sparse: {sp.issparse(adata.X)}")

   # Convert to sparse if needed
   if not sp.issparse(adata.layers["counts"]):
       adata.layers["counts"] = sp.csr_matrix(adata.layers["counts"])

**Memory savings**: 10-50x reduction for typical scRNA-seq data

4. Use Built-in Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For many features, use ``did_table_parallel()`` to distribute work across CPU cores:

.. code-block:: python

   res = st.did_table_parallel(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2"),
       n_jobs=4  # Number of CPU cores
   )

**Performance impact**: Near-linear speedup with core count

.. note::

   **Do not subsample or downsample cells** to work around memory limits.
   Subsampling distorts pseudobulk means, alters cell-type composition within
   participant groups, and changes WLS weights — especially for rare cell types
   and low-abundance signals. Instead, use sparse storage (strategy 3), feature
   batching (see below), per-cell-type decomposition, or run on hardware with
   sufficient memory.

For Large-Scale Analysis
-------------------------

Workflow for 1M+ Cells
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import sctrial as st
   import scanpy as sc
   import scipy.sparse as sp

   # Step 1: Load data (backed mode to avoid loading X into memory)
   adata = sc.read_h5ad("large_dataset.h5ad", backed='r')
   print(f"Starting with {adata.n_obs:,} cells")

   # Step 2: Bring into memory with sparse storage
   adata = adata.to_memory()
   if not sp.issparse(adata.X):
       adata.X = sp.csr_matrix(adata.X)

   # Step 3: Filter low-expression genes to reduce feature space
   sc.pp.filter_genes(adata, min_counts=10)
   print(f"After gene filtering: {adata.n_vars:,} genes")

   # Step 4: Normalize
   adata = st.add_log1p_cpm_layer(adata, counts_layer="counts")

   # Step 5: Score gene sets (reduces thousands of genes → tens of features)
   gene_sets = {...}  # Your gene sets
   adata = st.score_gene_sets(adata, gene_sets, layer="log1p_cpm", prefix="ms_")

   # Step 6: DiD analysis with parallelization
   features = [f"ms_{k}" for k in gene_sets.keys()]
   res = st.did_table_parallel(
       adata,
       features=features,
       design=design,
       visits=("baseline", "followup"),
       n_jobs=4
   )

   print(f"Analysis complete! Tested {len(features)} features.")

Batch Processing
~~~~~~~~~~~~~~~~

For genome-wide analysis, process genes in batches:

.. code-block:: python

   import pandas as pd

   def process_in_batches(adata, all_features, design, visits, batch_size=1000):
       """Process features in batches to limit memory usage."""
       results = []

       n_batches = int(np.ceil(len(all_features) / batch_size))
       print(f"Processing {len(all_features)} features in {n_batches} batches...")

       for i in range(0, len(all_features), batch_size):
           batch_features = all_features[i:i+batch_size]
           print(f"Batch {i//batch_size + 1}/{n_batches}: {len(batch_features)} features")

           res_batch = st.did_table(
               adata,
               features=batch_features,
               design=design,
               visits=visits
           )
           results.append(res_batch)

           # Optional: Free memory
           import gc
           gc.collect()

       # Combine results
       full_res = pd.concat(results, ignore_index=True)

       # Recalculate FDR across all features
       from statsmodels.stats.multitest import multipletests
       mask = full_res["p_DiD"].notna()
       full_res["FDR_DiD"] = np.nan
       if mask.sum() > 0:
           full_res.loc[mask, "FDR_DiD"] = multipletests(
               full_res.loc[mask, "p_DiD"],
               method="fdr_bh"
           )[1]

       return full_res

   # Use it
   all_genes = adata.var_names.tolist()
   res = process_in_batches(
       adata,
       all_genes,
       design,
       visits=("V1", "V2"),
       batch_size=1000
   )

Memory Management
-----------------

Monitor Memory Usage
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import psutil
   import os

   def print_memory_usage():
       """Print current memory usage."""
       process = psutil.Process(os.getpid())
       mem = process.memory_info().rss / 1024 ** 3  # GB
       print(f"Memory usage: {mem:.2f} GB")

   print_memory_usage()  # Before analysis
   res = st.did_table(...)
   print_memory_usage()  # After analysis

Reduce Memory Footprint
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Use float32 instead of float64
   adata.X = adata.X.astype(np.float32)

   # 2. Remove unnecessary layers
   del adata.layers["unnecessary_layer"]

   # 3. Subset to genes of interest before analysis
   genes_of_interest = [...]  # Your gene list
   adata = adata[:, genes_of_interest].copy()

   # 4. Free memory explicitly
   import gc
   del large_object
   gc.collect()

Out-of-Core Processing
~~~~~~~~~~~~~~~~~~~~~~

For datasets too large to fit in memory, use backed mode:

.. code-block:: python

   import scanpy as sc

   # Read in backed mode (doesn't load X into memory)
   adata = sc.read_h5ad("huge_dataset.h5ad", backed='r')

   # Extract only what you need
   genes_to_analyze = ["Gene1", "Gene2", ...]
   adata_subset = adata[:, genes_to_analyze].to_memory()

   # Now analyze the subset
   res = st.did_table(adata_subset, features=genes_to_analyze, design=design, visits=("V1", "V2"))

GSEA Performance
----------------

Optimize GSEA Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Pre-filter low-expression genes
   import scanpy as sc
   sc.pp.filter_genes(adata, min_counts=50)

   # 2. Reduce permutations for speed (at cost of precision)
   res = st.run_gsea_did(
       adata,
       gene_sets="KEGG_2021_Human",
       design=design,
       visits=("V1", "V2"),
       permutation_num=1000,  # Default is 1000; could use 100 for quick test
       min_size=15,  # Skip very small gene sets
       max_size=500  # Skip very large gene sets
   )

   # 3. Use specific gene set library (faster than "all")
   # Instead of all MSigDB, choose specific collection
   res = st.run_gsea_did(
       adata,
       gene_sets="KEGG_2021_Human",  # Specific library
       design=design,
       visits=("V1", "V2")
   )

Parallel Processing (Advanced)
-------------------------------

In addition to ``did_table_parallel()`` (see strategy 4 above), you can manually
parallelize across cell types:

Parallelize Across Cell Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from joblib import Parallel, delayed

   def run_did_for_celltype(adata, celltype, features, design, visits):
       """Run DiD for a single cell type."""
       try:
           res = st.did_table(
               adata,
               features=features,
               design=design,
               celltype=celltype,
               visits=visits
           )
           res["celltype"] = celltype
           return res
       except Exception as e:
           print(f"Failed for {celltype}: {e}")
           return None

   # Parallel execution
   celltypes = ["CD4_T", "CD8_T", "Monocytes", "B_cells", "NK"]
   results = Parallel(n_jobs=4)(
       delayed(run_did_for_celltype)(adata, ct, features, design, ("V1", "V2"))
       for ct in celltypes
   )

   # Combine results
   results = [r for r in results if r is not None]
   full_res = pd.concat(results, ignore_index=True)

**Warning**: Ensure your system has enough memory for parallel processes.

Benchmarking
------------

Benchmark Your Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time

   # Time your analysis
   start = time.time()

   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2")
   )

   elapsed = time.time() - start
   print(f"Analysis completed in {elapsed:.2f} seconds")
   print(f"Time per feature: {elapsed/len(features):.3f} seconds")

Profiling
~~~~~~~~~

For detailed performance analysis:

.. code-block:: python

   import cProfile
   import pstats

   # Profile the analysis
   profiler = cProfile.Profile()
   profiler.enable()

   res = st.did_table(adata, features=features, design=design, visits=("V1", "V2"))

   profiler.disable()

   # Print stats
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Top 20 slowest functions

Hardware Recommendations
------------------------

For Different Dataset Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Small datasets (< 100K cells)**:

- CPU: Any modern laptop
- RAM: 8 GB
- Storage: Local SSD

**Medium datasets (100K - 500K cells)**:

- CPU: Desktop with 4+ cores
- RAM: 16-32 GB
- Storage: SSD recommended

**Large datasets (500K - 2M cells)**:

- CPU: Workstation with 8+ cores
- RAM: 64-128 GB
- Storage: Fast NVMe SSD

**Very large datasets (> 2M cells)**:

- Use cluster/cloud computing with sufficient RAM
- Process features in batches
- Use backed mode + sparse storage

Quick Performance Checklist
----------------------------

Before running large-scale analysis:

.. code-block:: python

   # ✓ 1. Check data format
   import scipy.sparse as sp
   assert sp.issparse(adata.X), "Convert to sparse!"

   # ✓ 2. Use pseudobulk
   aggregate = "participant_visit"  # Not "cell"!

   # ✓ 3. Analyze gene sets, not all genes
   n_features = len(features)
   assert n_features < 1000, f"Too many features ({n_features}), use gene sets!"

   # ✓ 4. Use parallelization for many features
   # res = st.did_table_parallel(adata, features, design, visits, n_jobs=4)

   # ✓ 5. Monitor memory
   print_memory_usage()

   # NOW run analysis
   res = st.did_table(
       adata,
       features=features,
       design=design,
       visits=("V1", "V2"),
       aggregate="participant_visit"
   )
