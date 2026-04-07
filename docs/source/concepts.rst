Concepts and Methods
====================

This page describes the statistical framework underlying **sctrial**.
For the full derivation and empirical evaluation, see the preprint:
Vasanthakumari et al., *bioRxiv* 2026, doi:
`10.64898/2026.04.02.716219 <https://doi.org/10.64898/2026.04.02.716219>`_.

The pseudoreplication problem
-----------------------------

In multi-participant scRNA-seq experiments, each participant contributes
hundreds to thousands of cells.  Treating these cells as independent
observations inflates the effective sample size and produces
anti-conservative p-values.  Multiple studies have demonstrated this
effect:

- Squair et al. (2021) showed that cell-level differential expression
  tests yield false-discovery rates far above nominal levels when
  participants are ignored
  (`Nature Communications <https://doi.org/10.1038/s41467-021-25960-2>`__).

- Zimmerman et al. (2021) confirmed that pseudobulk aggregation to the
  sample level restores correct Type I error control
  (`Nature Communications <https://doi.org/10.1038/s41467-021-21038-1>`__).

- Murphy & Skene (2022) benchmarked pseudobulk against mixed-model and
  cell-level approaches and found that pseudobulk methods offer the best
  balance of calibration and power
  (`Nature Communications <https://doi.org/10.1038/s41467-022-35519-4>`__).

By default, **sctrial** aggregates cell-level expression to
participant-level summaries before inference, typically at the
participant-visit level.  Supported aggregation functions include mean,
median, and percent-positive summaries.  The participant — not the
cell — is the recommended unit of inference throughout the package.

Supported study designs
-----------------------

sctrial supports three study architectures, each targeting a different
estimand:

**Two-arm longitudinal (Difference-in-Differences)**
  Participants in two arms (e.g. treatment vs control, responder vs
  non-responder) are measured at two or more time points.  For the
  canonical two-visit setting, sctrial estimates a standard pre/post DiD
  parameter.  For studies with more than two visits, sctrial additionally
  provides event-study
  (:func:`~sctrial.stats.timeseries.event_study_did`) and
  trend-interaction
  (:func:`~sctrial.stats.timeseries.trend_interaction`) analyses to
  characterize time-varying differential effects.

**Single-arm paired**
  All participants receive the same intervention and are measured pre
  and post.  The estimand is the *average within-participant change*.

**Cross-sectional**
  Two groups are compared at a single time point.  The estimand is the
  *between-group difference* in participant-level means at that visit.

Each design is specified through the :class:`~sctrial.design.TrialDesign`
dataclass, which encodes the participant, visit, and arm columns so that
the correct estimand is applied automatically.

Difference-in-Differences estimation
-------------------------------------

For the two-arm, two-visit longitudinal case, sctrial estimates a
Difference-in-Differences (DiD) parameter.  For participant *i* in
arm *a* at time *t*, the model is:

.. math::

   Y_{iat} = \alpha_i + \gamma \cdot \text{Post}_t
             + \beta_{\text{DiD}} \cdot (\text{Treated}_a \times \text{Post}_t)
             + \varepsilon_{iat}

where:

- :math:`\alpha_i` is a participant fixed effect that absorbs all
  time-invariant confounders,
- :math:`\gamma` captures the common time trend,
- :math:`\beta_{\text{DiD}}` is the parameter of interest — the
  *additional* change in the treated arm relative to control.

With two time periods, estimating this model is algebraically equivalent
to first-differencing:

.. math::

   \Delta Y_i = Y_{i,\text{post}} - Y_{i,\text{pre}}

and then comparing :math:`\Delta Y` between arms via a two-sample test.
This equivalence is standard in econometrics
(Wooldridge, *Introductory Econometrics: A Modern Approach*,
Cengage Learning, 7th ed., 2020;
Angrist & Pischke, *Mostly Harmless Econometrics*,
Princeton University Press, 2009, ISBN 978-0-691-12035-5).

The identifying assumption
~~~~~~~~~~~~~~~~~~~~~~~~~~

The key assumption for a causal interpretation of :math:`\beta_{\text{DiD}}`
is **parallel trends**: in the absence of treatment, both arms would have
followed the same trajectory.  With only two time points, this assumption
cannot be tested directly.  When at least two pre-treatment visits are
available, sctrial provides
:func:`~sctrial.stats.timeseries.test_parallel_trends` to assess
evidence of differential pre-treatment trends; interpretation is more
informative when multiple pre-treatment intervals are observed.

For a formal treatment of parallel-trends testing and sensitivity
analysis, see Rambachan & Roth (2023),
`Review of Economic Studies <https://doi.org/10.1093/restud/rdad018>`__.

When parallel trends is implausible (e.g. observational severity
comparisons), sctrial results should be interpreted as descriptive
associations, not causal effects.

Inference in small samples
--------------------------

Single-cell clinical studies typically have 5–30 participants.
Cluster-robust (Huber–White) standard errors are asymptotically valid
but can be anti-conservative when the number of clusters is small
(Cameron & Miller, 2015,
`Journal of Human Resources <https://doi.org/10.3368/jhr.50.2.317>`__).

sctrial provides two approaches for finite-sample inference:

**Wild cluster bootstrap** (``use_bootstrap=True``)
  Resamples Rademacher weights at the participant level to construct
  a bootstrap-*t* distribution.  Recommended as a small-sample
  safeguard, especially when the number of participants per arm is
  modest (e.g. fewer than ~15 per arm).  Based on the procedure
  described in Cameron, Gelbach & Miller (2008),
  `Review of Economics and Statistics <https://doi.org/10.1162/rest.90.3.414>`__.

**Permutation testing**
  sctrial also supports permutation-based inference on participant-level
  summaries, which randomly reassigns treatment labels across
  participants and recomputes the test statistic under the null.
  Distribution-free but computationally more expensive.

For larger and better-balanced studies, cluster-robust standard errors
are often adequate, though diagnostics remain important.

Within-arm and cross-sectional comparisons
------------------------------------------

For single-arm paired designs,
:func:`~sctrial.stats.comparisons.within_arm_comparison` fits an OLS
model with participant fixed effects on pseudobulk means, regressing the
outcome on a visit indicator (``outcome ~ visit_num + C(participant)``).
The coefficient on the visit term estimates the average
within-participant change, and inference uses cluster-robust standard
errors (or bootstrap when specified).

For cross-sectional between-arm comparisons at a fixed visit,
:func:`~sctrial.stats.comparisons.between_arm_comparison` aggregates to
participant-level means and applies either OLS or a Mann–Whitney *U*
test between groups.

In both cases, inference operates on participant-level summaries, not
individual cells.

Effect sizes
------------

sctrial reports standardized effect sizes alongside p-values:

- **Cohen's d** and **Hedges' g** (bias-corrected for small *n*) for
  between-group contrasts.
- **Bootstrap confidence intervals** for effect sizes when analytical
  intervals rely on normality assumptions that may not hold in small
  samples.

Effect sizes are computed from participant-level aggregates via
:func:`~sctrial.stats.effect_size.cohens_d` and
:func:`~sctrial.stats.effect_size.hedges_g`.

Power analysis
--------------

sctrial includes prospective power calculations for both study designs:

- :func:`~sctrial.stats.power.power_did` /
  :func:`~sctrial.stats.power.sample_size_did` for two-arm DiD.
- :func:`~sctrial.stats.power.power_paired` /
  :func:`~sctrial.stats.power.sample_size_paired` for single-arm
  paired designs.

These use a two-sample *t*-test power formula (two-arm) or a
one-sample *t*-test formula (single-arm) as the basis, since the
DiD estimator under balanced allocation reduces to a comparison of
participant-level change scores.

Gene set enrichment analysis
----------------------------

For pathway-level analysis, sctrial ranks genes by a signed
confidence metric:

.. math::

   r_g = \text{sign}(\hat{\beta}_g) \cdot (-\log_{10}(p_g))

and passes this ranking to pre-ranked GSEA via
`gseapy <https://gseapy.readthedocs.io>`_.  This preserves both
direction and statistical strength in the ranking without requiring
an arbitrary fold-change cutoff.

Relationship to other tools
---------------------------

sctrial is complementary to existing pseudobulk differential expression
tools:

- **muscat** (Crowell et al., 2020,
  `Nature Communications <https://doi.org/10.1038/s41467-020-19894-4>`__)
  provides flexible differential-state analysis for multi-sample,
  multi-condition scRNA-seq data using pseudobulk and mixed-model
  approaches.

- **dreamlet** (Hoffman et al., 2023,
  `bioRxiv <https://doi.org/10.1101/2023.03.17.533005>`__)
  fits precision-weighted linear mixed models on voom-transformed
  pseudobulk counts with empirical Bayes variance moderation.

- **NEBULA** (He et al., 2021,
  `Communications Biology <https://doi.org/10.1038/s42003-021-02146-6>`__)
  fits negative binomial mixed models directly on cell-level counts.

sctrial differs from these tools in that it centers participant-level
longitudinal estimands (DiD, paired change scores) and associated
diagnostics as the primary analysis target.  It estimates effects
gene-by-gene without cross-gene variance borrowing; in our benchmark
settings, this improved calibration in mixed-signal panels with a high
fraction of affected genes.  It also provides a unified interface across
two-arm, single-arm, and cross-sectional designs within a single
package.
