"""Method runners for benchmarking.

Each runner takes standardized input and returns a dict:
    {gene_name: {"beta": float, "pvalue": float, "ci_lo": float, "ci_hi": float,
                 "converged": bool, "failure_mode": str | None}}

failure_mode is one of:
    None          — success
    "convergence" — optimizer did not converge but returned estimates
    "numerical"   — fit threw an exception, no estimates
    "timeout"     — exceeded wall-clock limit
"""
