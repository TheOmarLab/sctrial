<p align="center">
  <img src="docs/source/_static/logo.svg" alt="sctrial" width="280">
</p>
<p align="center"><strong>Design-Aware Participant-Level Inference for Longitudinal Single-Cell Studies</strong></p>

<p align="center">
  <a href="https://github.com/TheOmarLab/sctrial/actions/workflows/test.yml">
    <img src="https://github.com/TheOmarLab/sctrial/actions/workflows/test.yml/badge.svg?branch=main" alt="Test Status">
  </a>
  <a href="https://github.com/TheOmarLab/sctrial/releases">
    <img src="https://img.shields.io/github/v/release/TheOmarLab/sctrial?label=version" alt="Release Version">
  </a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python Versions">
  <a href="https://pypi.org/project/sctrial/">
    <img src="https://img.shields.io/pypi/v/sctrial.svg" alt="PyPI">
  </a>
  <a href="https://pepy.tech/project/sctrial">
    <img src="https://static.pepy.tech/badge/sctrial" alt="Downloads">
  </a>
  <a href="https://www.omar-lab.com/sctrial/">
    <img src="https://readthedocs.org/projects/sctrial/badge/?version=latest" alt="Documentation">
  </a>
</p>

---

## Overview

**sctrial** is a Python package for participant-level differential analysis of longitudinal single-cell RNA-seq data from clinical trials and translational studies. Built on [AnnData](https://anndata.readthedocs.io/) and the [scverse](https://scverse.org/) ecosystem, it provides:

- **Difference-in-Differences (DiD)** analysis with participant fixed effects
- **Paired within-arm** pre→post contrasts
- **Between-group** comparisons at fixed time points
- **Cell-type abundance** change testing
- **Gene set enrichment** analysis (GSEA) on DiD rankings
- **Power analysis** and sample size calculations
- **Effect size** estimation with confidence intervals

<p align="center">
  <img src="docs/source/_static/overview_figurev2.png" alt="sctrial overview: from scRNA-seq input through trial-aware analysis to statistical outputs" width="100%">
</p>

## Study Designs

sctrial matches the analysis to the design of the study. A two-arm longitudinal study is analyzed with a difference-in-differences, a single-arm longitudinal study with the within-participant change, and a cross-sectional study with a comparison of two groups at one time point. In every design the unit of inference is the participant.

<p align="center">
  <img src="docs/source/_static/clips/reel.gif" alt="Schematic animation of how sctrial handles study designs: one design definition, participant-level inference, two-arm and single-arm longitudinal designs, cross-sectional comparison, repeated visits, small cohorts, incomplete and unbalanced data, cell-type-specific analysis, and power for two independent groups" width="100%">
</p>

## Key Features

| Feature | Description |
|---------|-------------|
| **Trial-Aware Design** | Define participant, visit, arm, and cell type columns once |
| **Participant-Level Inference** | Participant-level aggregation, cluster-robust inference, optional wild cluster bootstrap |
| **Multiple Comparisons** | Built-in false discovery rate (FDR) correction across features and cell types |
| **Power Analysis** | Power and sample size for two-arm DiD, single-arm paired and two-group designs |
| **Single-Arm Support** | `arm_col=None` for studies without a control arm |
| **Publication-Ready Plots** | Forest plots, interaction plots, GSEA heatmaps |
| **Large Datasets** | Processes full single-cell datasets without subsampling |

## Installation

```bash
pip install sctrial
```

For development:
```bash
git clone https://github.com/TheOmarLab/sctrial.git
cd sctrial
pip install -e ".[dev]"
```

## Quick Start

```bash
pip install "sctrial[plots]"   # includes dataset loaders and visualization
```

```python
import sctrial as st

# 1. Load a real immunotherapy trial dataset
#    (allow_download=True fetches the GEO files on first use; the default raises
#    FileNotFoundError with the download link when they are missing)
adata = st.load_sade_feldman(allow_download=True)
adata = st.harmonize_response(adata)  # majority-vote response labels

# 2. Define trial design
design = st.TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
    celltype_col="cell_type",
)

# 3. Score gene sets (dataset ships pre-normalized with log1p_tpm layer)
gene_sets = {
    "Cytotoxicity": ["GZMA", "GZMB", "PRF1", "GNLY", "NKG7"],
    "Exhaustion":   ["PDCD1", "CTLA4", "HAVCR2", "LAG3", "TIGIT"],
}
adata = st.score_gene_sets(adata, gene_sets, layer="log1p_tpm", method="zmean", prefix="ms_")

# 4. Run Difference-in-Differences on CD8 T cells
features = [c for c in adata.obs.columns if c.startswith("ms_")]
results = st.did_table(adata, features, design, visits=("Pre", "Post"), celltype="CD8 T cell")
print(results[["feature", "beta_DiD", "se_DiD", "p_DiD", "FDR_DiD"]])
```

Or use the `quick_did` convenience wrapper, which combines gene-set scoring and the DiD analysis in a single call:

```python
results = st.quick_did(
    adata,
    module_scores=gene_sets,
    visits=("Pre", "Post"),
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
    celltype_col="cell_type",
)
```

## Supported Study Designs & Datasets

sctrial includes loaders (`st.load_*()`) for six real datasets from clinical trials and translational studies. They cover the supported study designs:

| Design | Description | Dataset | Source | Tutorial |
|--------|-------------|---------|--------|----------|
| **Two-group paired DiD** | Pre/post × responder/non-responder group interaction | Sade-Feldman et al., *Cell* 2018, melanoma immunotherapy | [GSE120575](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120575) | [Immunotherapy](tutorials/example_immunotherapy_sade_feldman.ipynb) |
| **Two-arm paired DiD** | Pre/post × treatment/control interaction | Zhang et al., *Cell* 2021, TNBC immunotherapy (anti-PD-L1+Chemo vs Chemo) | [GSE169246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169246) | [TNBC](tutorials/example_tnbc_zhang.ipynb) |
| **Single-arm pre/post** | Paired within-arm contrasts over time | ImmPort GSE171964, PBMC vaccine response | [GSE171964](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171964) | [Vaccine](tutorials/example_vaccine_immport.ipynb) |
| **Single-arm pre/post** | Paired within-arm contrasts across several time points | van Galen et al., *Cell* 2019, AML chemotherapy | [GSE116256](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256) | Not available |
| **Single-arm, several time points** | Longitudinal tracking across 4 visits | GSE290722, CAR-T cell therapy (ZUMA-1) | [GSE290722](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290722) | Not available |
| **Cross-sectional between-group** | Between-group comparison at one time point | Stephenson et al., *Nature Medicine* 2021, COVID-19 severity | [E-MTAB-10026](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10026) | [COVID-19](tutorials/example_covid19_stephenson.ipynb) |

Additional tutorial: [Scalability Benchmark](tutorials/stress_test_real_scale.ipynb), performance testing on the Sade-Feldman dataset.

## Documentation

Full documentation: [https://www.omar-lab.com/sctrial/](https://www.omar-lab.com/sctrial/)

- [API Reference](https://www.omar-lab.com/sctrial/api/index.html)
- [Tutorials](https://www.omar-lab.com/sctrial/tutorials/index.html)
- [FAQ](https://www.omar-lab.com/sctrial/faq.html)

## Citation

If you use **sctrial** in your research, please cite:

> Vasanthakumari P, Valencia I, Aghmiouni MR, Magana B, Omar M. **sctrial: Participant-Level Differential Analysis for Longitudinal Single-Cell Experiments.** *bioRxiv* 2026. [doi:10.64898/2026.04.02.716219](https://www.biorxiv.org/content/10.64898/2026.04.02.716219v1).

```bibtex
@article{vasanthakumari2026sctrial,
  title = {sctrial: Participant-Level Differential Analysis for Longitudinal Single-Cell Experiments},
  author = {Vasanthakumari, Priyanka and Valencia, Itzel and Aghmiouni, Maryam R. and Magana, Bryan and Omar, Mohamed},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.04.02.716219},
  url = {https://www.biorxiv.org/content/10.64898/2026.04.02.716219v1}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
