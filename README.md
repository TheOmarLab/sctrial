# sctrial

[![Test Status](https://github.com/TheOmarLab/sctrial/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/TheOmarLab/sctrial/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/sctrial.svg)](https://pypi.org/project/sctrial/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://TheOmarLab.github.io/sctrial/)

`sctrial` provides *trial-aware inference* helpers for **single-cell** data represented as `AnnData`,
focusing on analyses common in longitudinal / randomized / crossover designs:

- Difference-in-Differences (DiD) with participant fixed effects
- Paired within-arm pre→post contrasts
- Between-arm contrasts at a fixed visit
- Cell-type abundance change testing (proportion DiD)
- Module / pathway scoring and downstream statistics

This package intentionally **does not** implement QC, integration, clustering, or annotation workflows.
It assumes you already have an `AnnData` object with the metadata needed to describe the study design.

## Core assumptions

You have an `AnnData` where:
- `adata.obs` includes participant identifiers, visit labels, treatment arm, and (optionally) cell type labels
- raw counts are available in `adata.layers["counts"]` (recommended)

See `sctrial.design.TrialDesign` for how you tell the package which columns to use.

## Documentation

Comprehensive documentation, including API references and detailed tutorials, is available in the `docs/` directory. To build the documentation locally:

```bash
pip install sphinx sphinx_rtd_theme nbsphinx
cd docs
sphinx-build -b html source build/html
```

## Quick start

```python
import sctrial as st
from sctrial.design import TrialDesign
from sctrial.preprocessing import add_log1p_cpm_layer
from sctrial.scoring import score_gene_sets
from sctrial.stats.did import did_table
from sctrial.stats.abundance import abundance_did

design = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="recet",
    arm_treated="RECeT",
    arm_control="Sham",
    celltype_col="celltype",
    crossover_col="sham_crossover",   # optional
)

adata = add_log1p_cpm_layer(adata, counts_layer="counts", out_layer="log1p_cpm")

gene_sets = {"OXPHOS": ["G1", "G2", "G3"], "Myeloid_APC": ["G4", "G5", "G6"]}
adata = score_gene_sets(adata, gene_sets, layer="log1p_cpm", method="zmean", prefix="ms_")
```

## Comparisons and GSEA

The package supports standard trial comparisons and visualizations:

```python
from sctrial.stats.comparisons import within_arm_comparison, between_arm_comparison
from sctrial.stats.gsea import run_gsea_did
from sctrial.plotting import plot_trial_interaction, plot_abundance_interaction

# 1. Interaction plot for a feature
plot_trial_interaction(adata, feature="ms_OXPHOS", design=design, visits=("V1", "V2"))

# 2. Within-arm longitudinal change (e.g. baseline to followup in Treated)
res_within = within_arm_comparison(
    adata, arm="Treated", features=["ms_OXPHOS"], design=design, visits=("V1", "V2")
)

# 3. Between-arm comparison at a fixed visit
res_between = between_arm_comparison(
    adata, visit="V2", features=["ms_OXPHOS"], design=design
)

# 4. GSEA on DiD-derived gene rankings
# Requires gseapy
res_gsea = run_gsea_did(
    adata, 
    gene_sets="KEGG_2021_Human", 
    design=design, 
    visits=("V1", "V2"),
    rank_by="signed_confidence"
)
```

## Real-World Examples

We provide interactive Jupyter Notebooks applying `sctrial` to well-known public clinical datasets in the `examples/` directory:

- **COVID-19 PBMC**: `examples/example_covid19_stephenson.ipynb` (Stephenson et al., Nature 2021)
- **Immunotherapy Response**: `examples/example_immunotherapy_sade_feldman.ipynb` (Sade-Feldman et al., Cell 2018)
- **Vaccine Studies**: `examples/example_vaccine_immport.ipynb` (ImmPort database template)
- **Scalability Stress Test**: `examples/stress_test_real_scale.ipynb`

For a step-by-step tutorial with synthetic data, see the [Basic Workflow Tutorial](docs/source/tutorials/basic_workflow.rst).

## Primary phase DiD on module scores (participant-level means)

```python
res = did_table(
    adata=adata,
    features=[f"ms_{k}" for k in gene_sets],
    design=design,
    visits=("3/T0","6/T12w"),
    exclude_crossovers=True,
    aggregate="participant_visit",
)
```

## License
MIT
