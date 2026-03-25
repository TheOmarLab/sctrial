"""Benchmark metrics for method comparison.

All metrics follow the locked rules from the NatMeth benchmark plan:

1. Top-k stability computed separately for simulation and real-data subsampling
2. CI coverage only reported where intervals are defined (N/A for Wilcoxon)
3. λ_GC is a secondary null diagnostic, not a headline metric
4. Non-convergence split into: convergence / numerical / timeout
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests


def compute_fpr(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Per-test false positive rate under the null.

    Parameters
    ----------
    pvalues : array
        P-values from null genes only.
    alpha : float
        Significance threshold.

    Returns
    -------
    dict with "fpr", "n_tests", "wilson_ci_lo", "wilson_ci_hi"
    """
    pvalues = pvalues[~np.isnan(pvalues)]
    n = len(pvalues)
    if n == 0:
        return {"fpr": np.nan, "n_tests": 0,
                "wilson_ci_lo": np.nan, "wilson_ci_hi": np.nan}

    k = (pvalues < alpha).sum()
    fpr = k / n

    # Wilson confidence interval
    z = 1.96
    denom = 1 + z**2 / n
    center = (fpr + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(fpr * (1 - fpr) / n + z**2 / (4 * n**2)) / denom

    return {
        "fpr": fpr,
        "n_tests": n,
        "wilson_ci_lo": max(0, center - spread),
        "wilson_ci_hi": min(1, center + spread),
    }


def compute_fdr_tpr(
    pvalues: np.ndarray,
    is_signal: np.ndarray,
    q: float = 0.05,
) -> dict:
    """FDR and TPR (power) at a given BH threshold.

    Parameters
    ----------
    pvalues : array
        P-values for all genes (null + signal).
    is_signal : array of bool
        True for genes with non-zero effect.
    q : float
        BH-FDR threshold.

    Returns
    -------
    dict with "fdr", "tpr", "n_discoveries", "n_true_positives", "n_signal"
    """
    mask = ~np.isnan(pvalues)
    pvalues = pvalues[mask]
    is_signal = is_signal[mask]

    if len(pvalues) == 0:
        return {"fdr": np.nan, "tpr": np.nan, "n_discoveries": 0,
                "n_true_positives": 0, "n_signal": 0}

    reject = multipletests(pvalues, alpha=q, method="fdr_bh")[0]

    n_disc = reject.sum()
    n_tp = (reject & is_signal).sum()
    n_fp = (reject & ~is_signal).sum()
    n_signal = is_signal.sum()

    fdr = n_fp / n_disc if n_disc > 0 else 0.0
    tpr = n_tp / n_signal if n_signal > 0 else np.nan

    return {
        "fdr": fdr,
        "tpr": tpr,
        "n_discoveries": int(n_disc),
        "n_true_positives": int(n_tp),
        "n_signal": int(n_signal),
    }


def compute_bias_rmse(
    estimated: np.ndarray,
    true: np.ndarray,
) -> dict:
    """Bias and RMSE of effect-size estimates.

    Parameters
    ----------
    estimated : array
        Estimated β values.
    true : array
        True β values.

    Returns
    -------
    dict with "bias", "rmse", "n"
    """
    mask = ~(np.isnan(estimated) | np.isnan(true))
    est = estimated[mask]
    tru = true[mask]
    n = len(est)

    if n == 0:
        return {"bias": np.nan, "rmse": np.nan, "n": 0}

    errors = est - tru
    return {
        "bias": float(errors.mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "n": n,
    }


def compute_ci_coverage(
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    true: np.ndarray,
) -> dict | None:
    """CI coverage rate (95% nominal).

    Returns None if no valid intervals are available (e.g., Wilcoxon).
    This follows locked rule 2: coverage only where intervals are defined.
    """
    mask = ~(np.isnan(ci_lo) | np.isnan(ci_hi) | np.isnan(true))
    if mask.sum() == 0:
        return None  # N/A — method does not produce intervals

    lo = ci_lo[mask]
    hi = ci_hi[mask]
    tru = true[mask]

    covered = (tru >= lo) & (tru <= hi)
    return {
        "coverage": float(covered.mean()),
        "n": int(mask.sum()),
    }


def compute_sign_recovery(
    estimated: np.ndarray,
    true: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """Fraction of correct effect-size signs for |β| > threshold."""
    mask = (np.abs(true) > threshold) & ~np.isnan(estimated)
    if mask.sum() == 0:
        return {"sign_recovery": np.nan, "n": 0}

    correct = np.sign(estimated[mask]) == np.sign(true[mask])
    return {
        "sign_recovery": float(correct.mean()),
        "n": int(mask.sum()),
    }


def compute_lambda_gc(pvalues: np.ndarray) -> float:
    """Genomic inflation factor from null p-values.

    Secondary diagnostic (locked rule 3): use alongside FPR and QQ,
    not as a headline metric.
    """
    pvalues = pvalues[~np.isnan(pvalues)]
    if len(pvalues) == 0:
        return np.nan

    chi2_obs = sp_stats.chi2.ppf(1 - np.clip(pvalues, 1e-300, 1), df=1)
    return float(np.median(chi2_obs) / sp_stats.chi2.ppf(0.5, df=1))


def compute_topk_jaccard(
    ranking_a: pd.Series,
    ranking_b: pd.Series,
    k: int = 20,
) -> float:
    """Jaccard overlap of top-k genes between two rankings.

    Rankings are Series indexed by gene name, values are p-values
    (lower = more significant).
    """
    top_a = set(ranking_a.nsmallest(k).index)
    top_b = set(ranking_b.nsmallest(k).index)

    if len(top_a) == 0 or len(top_b) == 0:
        return 0.0

    intersection = top_a & top_b
    union = top_a | top_b
    return len(intersection) / len(union)


def compute_failure_rates(results: list[dict]) -> dict:
    """Compute failure rates split by mode (locked rule 4).

    Parameters
    ----------
    results : list of dict
        Each dict has "failure_mode": None | "convergence" | "numerical" | "timeout"

    Returns
    -------
    dict with keys: "convergence_rate", "numerical_rate", "timeout_rate", "total_failure_rate", "n"
    """
    n = len(results)
    if n == 0:
        return {
            "convergence_rate": 0.0, "numerical_rate": 0.0,
            "timeout_rate": 0.0, "total_failure_rate": 0.0, "n": 0,
        }

    modes = [r.get("failure_mode") for r in results]
    return {
        "convergence_rate": sum(m == "convergence" for m in modes) / n,
        "numerical_rate": sum(m == "numerical" for m in modes) / n,
        "timeout_rate": sum(m == "timeout" for m in modes) / n,
        "total_failure_rate": sum(m is not None for m in modes) / n,
        "n": n,
    }


def summarize_iteration(
    results: dict[str, dict],
    truth: dict[str, float],
    signal_genes: set[str],
) -> dict:
    """Compute all metrics for a single method on a single iteration.

    Parameters
    ----------
    results : dict
        gene_name → {"beta", "pvalue", "ci_lo", "ci_hi", "converged", "failure_mode"}
    truth : dict
        gene_name → true effect size
    signal_genes : set
        Gene names with non-zero true effects

    Returns
    -------
    dict of all metrics
    """
    genes = sorted(results.keys())
    pvals = np.array([results[g]["pvalue"] for g in genes])
    betas = np.array([results[g]["beta"] for g in genes])
    true_betas = np.array([truth.get(g, 0.0) for g in genes])
    is_sig = np.array([g in signal_genes for g in genes])

    ci_lo = np.array([results[g].get("ci_lo", np.nan) for g in genes])
    ci_hi = np.array([results[g].get("ci_hi", np.nan) for g in genes])

    # Null genes only
    null_mask = ~is_sig
    null_pvals = pvals[null_mask]

    out = {}

    # FPR (null genes)
    out.update(compute_fpr(null_pvals))

    # FDR + TPR (all genes, if there are signal genes)
    if is_sig.any():
        fdr_tpr = compute_fdr_tpr(pvals, is_sig)
        out["fdr"] = fdr_tpr["fdr"]
        out["tpr"] = fdr_tpr["tpr"]
    else:
        out["fdr"] = np.nan
        out["tpr"] = np.nan

    # Bias + RMSE
    bias_rmse = compute_bias_rmse(betas, true_betas)
    out["bias"] = bias_rmse["bias"]
    out["rmse"] = bias_rmse["rmse"]

    # CI coverage (may be None for methods without CIs)
    coverage = compute_ci_coverage(ci_lo, ci_hi, true_betas)
    out["coverage"] = coverage["coverage"] if coverage else np.nan
    out["coverage_applicable"] = coverage is not None

    # Sign recovery
    sign = compute_sign_recovery(betas, true_betas)
    out["sign_recovery"] = sign["sign_recovery"]

    # λ_GC (null genes)
    out["lambda_gc"] = compute_lambda_gc(null_pvals)

    # Failure rates
    failures = compute_failure_rates(list(results.values()))
    out["convergence_failure_rate"] = failures["convergence_rate"]
    out["numerical_failure_rate"] = failures["numerical_rate"]
    out["timeout_failure_rate"] = failures["timeout_rate"]
    out["total_failure_rate"] = failures["total_failure_rate"]

    return out
