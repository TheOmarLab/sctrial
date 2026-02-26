:html_theme.sidebar_secondary.remove:

.. raw:: html

   <div class="hero-section">
     <h1>sctrial</h1>
     <p class="hero-tagline">Trial-Aware Statistical Inference for Single-Cell Data</p>
     <p class="hero-badges">
       <a href="https://github.com/TheOmarLab/sctrial/actions/workflows/test.yml">
         <img src="https://github.com/TheOmarLab/sctrial/actions/workflows/test.yml/badge.svg?branch=main" alt="Tests">
       </a>
       <a href="https://github.com/TheOmarLab/sctrial/releases">
         <img src="https://img.shields.io/github/v/release/TheOmarLab/sctrial?label=version" alt="Version">
       </a>
       <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python">
       <a href="https://opensource.org/licenses/MIT">
         <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
       </a>
     </p>
     <div class="hero-buttons">
       <a class="btn-hero-primary" href="installation.html">Get Started</a>
       <a class="btn-hero-secondary" href="tutorials/index.html">Tutorials</a>
     </div>
   </div>


Key Features
------------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Difference-in-Differences
      :text-align: center

      Rigorous DiD analysis with participant fixed effects and wild cluster bootstrap inference.

   .. grid-item-card:: Paired Contrasts
      :text-align: center

      Within-arm pre→post comparisons with paired statistical tests and effect sizes.

   .. grid-item-card:: Between-Arm Tests
      :text-align: center

      Cross-sectional comparisons between treatment and control arms at fixed timepoints.

   .. grid-item-card:: Abundance Analysis
      :text-align: center

      Cell-type composition changes across conditions with proportion-based statistics.

   .. grid-item-card:: Gene Set Enrichment
      :text-align: center

      GSEA on DiD-ranked gene lists with multiple gene set collections.

   .. grid-item-card:: Power Analysis
      :text-align: center

      Sample size calculations and power curves for planning single-cell clinical studies.


.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Quick Start
      :link: quickstart
      :link-type: doc

      Install sctrial and run your first analysis in minutes.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Complete reference for all public functions and classes.


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
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Guides

   best_practices
   troubleshooting
   faq
   performance
