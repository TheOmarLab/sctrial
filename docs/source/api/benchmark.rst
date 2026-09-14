Benchmark Simulation
====================

Transcriptome-scale hierarchical gamma-Poisson simulator for generating
realistic single-cell clinical-trial data with known ground truth, plus the
calibration, validation and orchestration tools used to benchmark statistical
methods against it.

The generative model, per cell ``c`` of participant ``i`` at visit ``t``,
gene ``g``:

.. math::

   \log \mu_{icgt} = \log L_{ic} + \alpha_g + b_{ig} + u_{igt}
                     + \gamma_g \mathrm{Post}_t + \beta_g (T_i \times \mathrm{Post}_t)

with :math:`Y \sim \mathrm{NB}(\mu, \phi_g)` and
:math:`\mathrm{Var} = \mu + \phi_g \mu^2`. The full transcriptome is simulated;
analysis panels are drawn from it as nested subsets of the detectable genes, so
normalisation and library-size offsets see the whole transcriptome exactly as a
real workflow would.

Simulator
---------

.. autoclass:: sctrial.benchmark.TranscriptomeSimConfig
   :members:
   :show-inheritance:

.. autofunction:: sctrial.benchmark.simulate_trial_v2

.. autofunction:: sctrial.benchmark.make_signal

.. autofunction:: sctrial.benchmark.nested_panels

.. autofunction:: sctrial.benchmark.oracle_estimands

.. autofunction:: sctrial.benchmark.simulator_v2.load_tnbc_targets

.. autofunction:: sctrial.benchmark.simulator_v2.load_empirical

.. autofunction:: sctrial.benchmark.simulator_v2.build_params

.. autofunction:: sctrial.benchmark.simulator_v2.gene_baseline_rates

.. autofunction:: sctrial.benchmark.simulator_v2.expected_counts_per_cell

.. autofunction:: sctrial.benchmark.simulator_v2.eligible_panel_genes

.. autofunction:: sctrial.benchmark.simulator_v2.iter_pv_blocks

Method contracts
----------------

Different methods do not estimate the same functional, and they require
different input representations. Both are declared explicitly rather than
inferred at call time.

.. autofunction:: sctrial.benchmark.prepare_inputs

.. autofunction:: sctrial.benchmark.contracts.prepare_inputs_from_adata

.. autofunction:: sctrial.benchmark.contracts.participant_log1p_cpm

.. py:data:: METHOD_INPUT
   :module: sctrial.benchmark
   :type: dict[str, str]

   The input representation each method expects.

   - ``participant_log1p_cpm`` — sctrial_did, sctrial_mixed, wilcoxon_paired
   - ``pseudobulk_counts`` — dreamlet, limma_voom
   - ``cell_counts`` — nebula

.. py:data:: METHOD_ESTIMAND
   :module: sctrial.benchmark
   :type: dict[str, str]

   The estimand scale each method is scored against.

   - ``log1p_cpm`` — sctrial_did, sctrial_mixed, wilcoxon_paired
   - ``count_link`` — dreamlet, limma_voom, nebula

Calibration
-----------

Estimators that measure empirical properties of a real dataset (library-size
distribution, per-gene dispersion, per-gene baseline rates) used to calibrate
the simulator so it reproduces observed nuisance statistics.

.. autoclass:: sctrial.benchmark.calibration.SummaryAccumulator
   :members:
   :show-inheritance:

.. autofunction:: sctrial.benchmark.calibration.summarize_blocks

.. autofunction:: sctrial.benchmark.calibration.summarize_adata

.. autofunction:: sctrial.benchmark.calibration.summarize_simulation

.. autofunction:: sctrial.benchmark.calibration.conditional_dispersion

.. autofunction:: sctrial.benchmark.calibration.measure_targets

Validation gates
----------------

Monte Carlo envelope tests (Gates A–E) that verify the simulator faithfully
reproduces the empirical nuisance statistics it was calibrated against.

.. py:data:: GATE_STATISTICS
   :module: sctrial.benchmark.gates
   :type: dict[str, list[str]]

   Maps each gate label (``"A"``, ``"B"``, …) to the list of statistics it
   checks.

.. py:data:: PINNED_STATISTICS
   :module: sctrial.benchmark.gates
   :type: frozenset[str]

   Statistics that are near-deterministic readbacks of an empirical pool the
   simulator resamples. Agreement is true by construction; failure indicates an
   implementation defect, not a fidelity defect.

.. autoclass:: sctrial.benchmark.gates.GateResult
   :members:
   :show-inheritance:

.. autofunction:: sctrial.benchmark.gates.run_gates

.. autofunction:: sctrial.benchmark.gates.composition_ablation

Metrics
-------

Per-replicate and per-scenario metric functions used to evaluate statistical
methods.

.. autofunction:: sctrial.benchmark.metrics.compute_fpr

.. autofunction:: sctrial.benchmark.metrics.compute_fdr_tpr

.. autofunction:: sctrial.benchmark.metrics.compute_bias_rmse

.. autofunction:: sctrial.benchmark.metrics.compute_ci_coverage

.. autofunction:: sctrial.benchmark.metrics.compute_sign_recovery

.. autofunction:: sctrial.benchmark.metrics.compute_lambda_gc

.. autofunction:: sctrial.benchmark.metrics.compute_topk_jaccard

.. autofunction:: sctrial.benchmark.metrics.compute_failure_rates

.. autofunction:: sctrial.benchmark.metrics.summarize_iteration

Endpoints
---------

Pre-specified statistical endpoints computed from replicate-level results.

.. py:data:: ALPHA
   :module: sctrial.benchmark.endpoints
   :type: float
   :value: 0.05

   Locked nominal significance threshold.

.. py:data:: Q_FDR
   :module: sctrial.benchmark.endpoints
   :type: float
   :value: 0.05

   Locked Benjamini–Hochberg FDR threshold.

.. autofunction:: sctrial.benchmark.endpoints.replicate_endpoints

.. autofunction:: sctrial.benchmark.endpoints.scenario_endpoints

Orchestrator
------------

Monte Carlo orchestration, scenario grid construction, and the top-level entry
points for running and sensitivity-testing the benchmark.

.. py:data:: CORE_METHODS
   :module: sctrial.benchmark
   :type: list[str]

   The methods included in every standard benchmark run:
   ``sctrial_did``, ``wilcoxon_paired``, ``dreamlet``, ``limma_voom``,
   ``nebula``.

.. autofunction:: sctrial.benchmark.run_benchmark

.. autofunction:: sctrial.benchmark.run_sensitivity_benchmark

.. autofunction:: sctrial.benchmark.orchestrator.build_scenario_grid

.. autofunction:: sctrial.benchmark.orchestrator.build_sensitivity_grid

.. autofunction:: sctrial.benchmark.orchestrator.mc_max_for

.. autofunction:: sctrial.benchmark.orchestrator.scenario_seed

Permutation & subsampling
--------------------------

Real-data hypothesis tests and sensitivity analyses that operate on observed
AnnData objects rather than simulated data.

.. autofunction:: sctrial.benchmark.permutation.run_permutation_test

.. autofunction:: sctrial.benchmark.subsample.run_subsampling

Ablation
--------

Estimator variants used to isolate the contribution of individual design
choices (normalisation scope, aggregation level, fixed effects).

.. py:data:: ABLATION_VARIANTS
   :module: sctrial.benchmark.ablation
   :type: dict[str, tuple]

   Registry mapping variant name to ``(input_type, runner, label)``.

.. autofunction:: sctrial.benchmark.ablation.run_ablation

Infrastructure
--------------

Reproducibility and result-layout utilities shared across the benchmark
pipeline.

**Result layout**

.. autoclass:: sctrial.benchmark.paths.ResultLayout
   :members:
   :show-inheritance:

.. autofunction:: sctrial.benchmark.paths.preflight_layout

.. autofunction:: sctrial.benchmark.paths.require_layout

**Scenario contracts**

.. autoclass:: sctrial.benchmark.scenario_contract.Violation
   :members:
   :show-inheritance:

.. autoclass:: sctrial.benchmark.scenario_contract.ContractReport
   :members:
   :show-inheritance:

.. autofunction:: sctrial.benchmark.scenario_contract.check_simulation

.. autofunction:: sctrial.benchmark.scenario_contract.check_scenario_results

.. autofunction:: sctrial.benchmark.scenario_contract.evaluability_by_method

.. autofunction:: sctrial.benchmark.scenario_contract.completion_record

**Manifest & reproducibility**

.. py:data:: SOURCE_TREE_PATHS
   :module: sctrial.benchmark.manifest
   :type: tuple[str, ...]

   Paths hashed when computing the source-tree digest:
   ``src``, ``scripts``, ``pyproject.toml``.

.. autofunction:: sctrial.benchmark.manifest.manifest_hash

.. autofunction:: sctrial.benchmark.manifest.verify_manifest

.. autofunction:: sctrial.benchmark.manifest.assert_single_manifest

.. autofunction:: sctrial.benchmark.manifest.source_tree_sha256

**Seeds**

.. py:data:: SEP
   :module: sctrial.benchmark.seeds
   :type: str

   Field separator used when constructing seed strings (ASCII unit separator,
   ``\x1f``).

.. autofunction:: sctrial.benchmark.seeds.stable_seed
