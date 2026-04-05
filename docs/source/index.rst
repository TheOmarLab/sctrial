sctrial — Participant-Level Differential Analysis for Longitudinal Single-Cell Experiments
=========================================================================================

.. image:: https://github.com/TheOmarLab/sctrial/actions/workflows/test.yml/badge.svg?branch=main
   :target: https://github.com/TheOmarLab/sctrial/actions/workflows/test.yml
   :alt: Tests

.. image:: https://img.shields.io/badge/python-3.9%2B-blue
   :alt: Python

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

**sctrial** is a Python package for participant-level differential analysis of longitudinal
single-cell RNA-seq data. Built on `AnnData <https://anndata.readthedocs.io>`_, it
aggregates cell-level measurements to participant-level replicates and applies
design-specific statistical methods — difference-in-differences, paired contrasts, and
cross-sectional comparisons — so that inference reflects the actual number of independent
biological units in the study.

.. image:: _static/overview_figure.png
   :alt: sctrial overview — from scRNA-seq input through trial-aware analysis to statistical outputs
   :width: 100%
   :align: center

Key Applications
----------------

- **Difference-in-Differences (DiD)** — Participant fixed-effects regression with wild cluster
  bootstrap inference for treatment effect estimation across arms and timepoints.

- **Paired Contrasts** — Within-arm pre→post comparisons with paired statistical tests (Wilcoxon,
  t-test) and effect sizes (Cohen's d, Hedges' g, log₂ fold-change).

- **Between-Arm Tests** — Cross-sectional comparisons between treatment and control arms at
  fixed timepoints with proper pseudobulk replication.

- **Abundance Analysis** — Cell-type composition changes across conditions using proportion-based
  statistics and participant-level aggregation.

- **Gene Set Enrichment Analysis** — GSEA on DiD-ranked gene lists with support for Hallmark,
  KEGG, Reactome, and custom gene set collections.

- **Power Analysis** — Sample size calculations, power curves, and minimum detectable effect
  sizes for planning single-cell clinical studies.

Getting Started
---------------

Install sctrial and run your first analysis:

.. code-block:: bash

   pip install sctrial

See the :doc:`installation` guide for optional extras (plotting, GSEA, Bayesian modules),
then follow the :doc:`quickstart` for a complete walkthrough.
For end-to-end analyses on real clinical datasets, explore the :doc:`tutorials/index`.

Citing sctrial
--------------

If you use **sctrial** in your research, please cite:

  Vasanthakumari P, Valencia I, Aghmiouni MR, Magana B, Omar MN.
  **sctrial: Participant-Level Differential Analysis for Longitudinal Single-Cell Experiments.**
  *bioRxiv* (2026).

.. code-block:: bibtex

   @article{vasanthakumari2026sctrial,
     title = {sctrial: Participant-Level Differential Analysis for Longitudinal Single-Cell Experiments},
     author = {Vasanthakumari, Priyanka and Valencia, Itzel and Aghmiouni, Maryam R. and Magana, Bryan and Omar, Mohamed N.},
     journal = {bioRxiv},
     year = {2026},
     url = {https://github.com/TheOmarLab/sctrial}
   }

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Guides

   best_practices
   troubleshooting
   faq
   performance
