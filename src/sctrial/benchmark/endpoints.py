"""The benchmark's endpoint definitions, fixed before the definitive run.

Written down and frozen so that no number reported in the paper depends on a
choice made after seeing results. Each definition states its DENOMINATOR
explicitly, because that is where the ambiguity lives.

Null scenarios
    ``fpr``                 fraction of evaluable null genes with p < alpha
    ``pvalue_ks``           KS distance of null p-values from Uniform(0,1)
    ``lambda_gc``           genomic inflation factor, secondary diagnostic
    ``evaluability``        see below

Mixed-signal scenarios
    ``fdr``                 after Benjamini-Hochberg at q, FP / max(R, 1)
    ``tpr``                 after BH at q, TP / (all true signal genes)
    ``power_tested``        p < alpha among signal genes the method EVALUATED
    ``power_end_to_end``    p < alpha among ALL signal genes in the panel
    ``null_fpr``            null-gene false positives in the presence of signal
    ``direction_accuracy``  sign agreement on genes the method called
    ``evaluability``        see below

Computational
    ``runtime_seconds``     per iteration, under the evaluated implementation
    ``convergence_rate``    fraction of genes with a converged fit

Why power has two denominators
------------------------------
If a method filters 30% of the panel and power is computed only among the genes
it kept, it is rewarded for discarding the difficult ones. ``power_tested`` is
the conditional quantity a user sees; ``power_end_to_end`` counts a filtered
signal gene as undetected, which is what actually happened to that gene. Both are
reported, always together, and ``evaluability`` is reported beside them so the
gap is attributable rather than mysterious.

For p-value calibration only evaluable p-values can be used -- a missing p-value
has no position in a QQ plot -- so the evaluability rate must be printed next to
every calibration figure.

The unit of replication
-----------------------
Every rate is computed WITHIN a replicate first and then averaged across
replicates, and the Monte Carlo error is the standard error over replicates.
Genes within one replicate share its participants, library sizes and random
effects, so pooling them as independent Bernoulli draws understates the error --
pseudoreplication inside a benchmark whose subject is pseudoreplication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ALPHA", "Q_FDR", "scenario_endpoints", "replicate_endpoints"]

ALPHA = 0.05
Q_FDR = 0.05


def _bh_reject(p: np.ndarray, q: float = Q_FDR) -> np.ndarray:
    """Benjamini-Hochberg rejections among non-missing p-values."""
    out = np.zeros(p.shape, dtype=bool)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = q * (np.arange(1, m + 1) / m)
    passing = p[order] <= thresh
    if passing.any():
        out[order[: np.flatnonzero(passing)[-1] + 1]] = True
    return out


def replicate_endpoints(g: pd.DataFrame, alpha: float = ALPHA, q: float = Q_FDR) -> dict:
    """Endpoints for ONE method within ONE replicate.

    ``g`` must contain every gene ASSIGNED to the panel, including those the
    method could not evaluate; that is what makes the end-to-end denominator
    meaningful.
    """
    p = g["pvalue"].to_numpy(dtype=float)
    is_sig = g["is_signal"].to_numpy(dtype=bool)
    evaluable = np.isfinite(p)

    n_panel = len(g)
    n_signal_panel = int(is_sig.sum())
    n_null_panel = int((~is_sig).sum())

    hit = evaluable & (p < alpha)
    rej = _bh_reject(p, q)

    out = {
        "n_panel": n_panel,
        "n_evaluable": int(evaluable.sum()),
        # Denominator is the ASSIGNED panel, not what the method chose to keep.
        "evaluability": float(evaluable.sum() / n_panel) if n_panel else np.nan,
        "n_signal_panel": n_signal_panel,
        "n_null_panel": n_null_panel,
    }

    # --- null behaviour ---
    if n_null_panel:
        ev_null = evaluable & ~is_sig
        out["fpr"] = float(hit[ev_null].mean()) if ev_null.any() else np.nan
        out["n_evaluable_null"] = int(ev_null.sum())
        pv = np.sort(p[ev_null])
        if pv.size > 1:
            ecdf = np.arange(1, pv.size + 1) / pv.size
            out["pvalue_ks"] = float(np.max(np.abs(ecdf - pv)))
        else:
            out["pvalue_ks"] = np.nan
    else:
        out["fpr"] = np.nan
        out["pvalue_ks"] = np.nan

    # --- signal behaviour ---
    if n_signal_panel:
        ev_sig = evaluable & is_sig
        # Conditional: among genes this method actually tested.
        out["power_tested"] = float(hit[ev_sig].mean()) if ev_sig.any() else np.nan
        # End-to-end: a filtered signal gene is an undetected signal gene.
        out["power_end_to_end"] = float(hit[is_sig].sum() / n_signal_panel)
        n_rej = int(rej.sum())
        out["fdr"] = float((rej & ~is_sig).sum() / max(n_rej, 1))
        out["tpr"] = float((rej & is_sig).sum() / n_signal_panel)
        out["n_rejected"] = n_rej
        called = hit & is_sig
        if called.any() and "estimated_beta" in g and "true_beta" in g:
            b = g["estimated_beta"].to_numpy(dtype=float)[called]
            t = g["true_beta"].to_numpy(dtype=float)[called]
            keep = np.isfinite(b) & np.isfinite(t) & (np.abs(t) > 1e-12)
            out["direction_accuracy"] = (
                float((np.sign(b[keep]) == np.sign(t[keep])).mean()) if keep.any() else np.nan
            )
        else:
            out["direction_accuracy"] = np.nan
    else:
        for k in ("power_tested", "power_end_to_end", "fdr", "tpr", "direction_accuracy"):
            out[k] = np.nan

    if "converged" in g:
        out["convergence_rate"] = float(g["converged"].astype(bool).mean())
    if "runtime_seconds" in g and len(g):
        out["runtime_seconds"] = float(g["runtime_seconds"].iloc[0])
    return out


def scenario_endpoints(
    df: pd.DataFrame, alpha: float = ALPHA, q: float = Q_FDR
) -> pd.DataFrame:
    """Endpoints per scenario and method, with replicate-level Monte Carlo error.

    Each rate is computed within a replicate and then averaged; the reported
    ``*_mcse`` is the standard error over replicates. Averaging genes across
    replicates instead would understate it, because genes in one replicate are
    not independent of each other.
    """
    rows = []
    for (scenario, method), grp in df.groupby(["scenario", "method"], observed=True):
        per_rep = [
            replicate_endpoints(r, alpha=alpha, q=q)
            for _it, r in grp.groupby("iteration", observed=True)
        ]
        if not per_rep:
            continue
        rep = pd.DataFrame(per_rep)
        row = {"scenario": scenario, "method": method, "n_replicates": len(rep)}
        for col in rep.columns:
            v = rep[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            row[col] = float(np.mean(v))
            row[f"{col}_mcse"] = (
                float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
