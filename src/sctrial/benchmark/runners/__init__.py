"""Method runners for benchmarking.

Each runner takes standardized input and returns a dict:
    {gene_name: {"beta": float, "pvalue": float, "ci_lo": float, "ci_hi": float,
                 "converged": bool, "failure_mode": str | None}}

failure_mode is one of:
    None           success: the gene has a p-value
    "convergence"  the optimizer did not converge but returned estimates
    "numerical"    no test: the fit raised, or returned no finite p-value
    "timeout"      exceeded the wall-clock limit

A gene without a finite p-value was not tested, whatever its runner recorded:
:func:`finalize_results` marks it as not converged, with a failure mode. The method
dispatcher and the ablation pass the rows of every method through it, so the rule is
written once and not in each runner.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["finalize_results"]


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finalize_results(results: dict[str, dict]) -> dict[str, dict]:
    """Record every gene without a finite p-value as not tested.

    A method can return a row for a gene it could not test without raising: a t-test on
    changes that have no variance returns a missing p-value, and so does a missing value
    in the result table of an R method. Such a row is not a converged fit. It gets
    ``converged = False`` and, when its runner named no failure, ``failure_mode =
    "numerical"``, so that the failure rates count it and the convergence rate does not.
    Its interval is missing too: a standard error of zero gives the interval of width
    zero around the estimate, which would enter the coverage of the method. The estimate
    is kept. A row with a p-value keeps what its runner returned, and gets ``converged =
    True`` and ``failure_mode = None`` when it carries no flags (the model rungs of the
    ablation).

    Returns new rows; ``results`` is left as it is.
    """
    out = {}
    for gene, result in results.items():
        row = dict(result)
        if _is_finite(row.get("pvalue")):
            row.setdefault("converged", True)
            row.setdefault("failure_mode", None)
        else:
            row["pvalue"] = row["ci_lo"] = row["ci_hi"] = float("nan")
            row["converged"] = False
            if row.get("failure_mode") is None:
                row["failure_mode"] = "numerical"
        out[gene] = row
    return out
