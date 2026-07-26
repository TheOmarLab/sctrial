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

Method contracts
----------------
Different methods do not estimate the same functional, and they require
different input representations. Both are declared explicitly rather than
inferred at call time.

.. autofunction:: sctrial.benchmark.prepare_inputs

.. autodata:: sctrial.benchmark.METHOD_INPUT

.. autodata:: sctrial.benchmark.METHOD_ESTIMAND

Orchestrator
------------
.. autofunction:: sctrial.benchmark.run_benchmark

.. autofunction:: sctrial.benchmark.run_sensitivity_benchmark

.. autodata:: sctrial.benchmark.CORE_METHODS
