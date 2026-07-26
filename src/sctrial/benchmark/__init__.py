"""Benchmarking suite for the methods paper.

Provides:

- a transcriptome-scale, three-level hierarchical gamma-Poisson simulator
  calibrated to real data (:mod:`~sctrial.benchmark.simulator_v2`);
- the calibration estimators and validation statistics
  (:mod:`~sctrial.benchmark.calibration`);
- Monte Carlo calibration gates (:mod:`~sctrial.benchmark.gates`);
- explicit per-method input and estimand contracts
  (:mod:`~sctrial.benchmark.contracts`);
- standardised method runners (sctrial, dreamlet, NEBULA, Wilcoxon, limma-voom);
- participant-label permutation and subsampling on real data;
- metrics: FPR, FDR, TPR, bias, RMSE, coverage, sign recovery, lambda_GC, Jaccard.

There is exactly ONE simulator. The previous panel-only model
(``benchmark.simulator``) was deleted rather than deprecated: it had no
transcriptome, no participant x visit level, and defaults that silently produced
2.3e7 UMIs per cell, and leaving it importable would mean a caller could still
reach it. Nothing in the repository generates benchmark data any other way.

Public API
----------
.. autosummary::

    TranscriptomeSimConfig
    simulate_trial_v2
    run_benchmark
    run_sensitivity_benchmark
    CORE_METHODS
    METHOD_ESTIMAND
"""

from .contracts import METHOD_ESTIMAND, METHOD_INPUT, prepare_inputs
from .orchestrator import CORE_METHODS, run_benchmark, run_sensitivity_benchmark
from .simulator_v2 import (
    TranscriptomeSimConfig,
    make_signal,
    nested_panels,
    oracle_estimands,
    simulate_trial_v2,
)

__all__ = [
    "TranscriptomeSimConfig",
    "simulate_trial_v2",
    "make_signal",
    "nested_panels",
    "oracle_estimands",
    "prepare_inputs",
    "METHOD_ESTIMAND",
    "METHOD_INPUT",
    "run_benchmark",
    "run_sensitivity_benchmark",
    "CORE_METHODS",
]
