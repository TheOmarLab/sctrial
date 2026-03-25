"""Benchmarking suite for NatMeth submission.

Provides:
- Hierarchical gamma-Poisson simulator for realistic scRNA-seq trial data
- Standardized method runners (sctrial, edgeR, limma-voom, dreamlet, NEBULA)
- Participant-label permutation and subsampling reproducibility
- Metrics: FPR, FDR, TPR, bias, RMSE, coverage, sign recovery, λ_GC, Jaccard top-k

Public API
----------
.. autosummary::

    SimulationConfig
    simulate_trial
    calibrate_from_real_data
    validate_simulator
    run_benchmark
    CORE_METHODS
"""

from .orchestrator import CORE_METHODS, run_benchmark
from .simulator import (
    SimulationConfig,
    calibrate_from_real_data,
    simulate_trial,
    validate_simulator,
)

__all__ = [
    "SimulationConfig",
    "simulate_trial",
    "calibrate_from_real_data",
    "validate_simulator",
    "run_benchmark",
    "CORE_METHODS",
]
