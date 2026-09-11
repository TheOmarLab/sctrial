Benchmark Simulation
====================

Hierarchical gamma-Poisson simulator for generating realistic scRNA-seq
clinical trial data with known ground truth, and orchestration tools for
running controlled benchmarks across multiple statistical methods.

Simulator
---------
.. autoclass:: sctrial.benchmark.SimulationConfig
   :members:
   :show-inheritance:

.. autofunction:: sctrial.benchmark.simulate_trial

.. autofunction:: sctrial.benchmark.calibrate_from_real_data

.. autofunction:: sctrial.benchmark.validate_simulator

Orchestrator
------------
.. autofunction:: sctrial.benchmark.run_benchmark

.. autodata:: sctrial.benchmark.CORE_METHODS
