"""Subsampling reproducibility analysis.

For each dataset, subsample participants at different fractions,
run all methods, and measure ranking stability via Spearman ρ
and top-k Jaccard overlap.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _run_subsample_iteration(args: tuple) -> dict:
    """Worker: run all methods on one participant subsample.

    The fourth element may be an in-memory AnnData (n_jobs=1) or an h5ad path.
    Returns a dict with keys: frac, resample, n_sub, pvals (method → gene → pvalue).
    """
    import warnings

    warnings.filterwarnings("ignore")

    (frac, r, sub_pids, adata_or_path, gene_cols, methods,
     participant_col, arm_col, visit_col) = args

    from .orchestrator import _dispatch_method, _pseudobulk_counts_from_adata
    from sctrial.stats.pseudobulk import pseudobulk_expression

    if isinstance(adata_or_path, (str, Path)):
        import anndata as ad

        adata = ad.read_h5ad(adata_or_path)
    else:
        adata = adata_or_path

    mask = adata.obs[participant_col].isin(sub_pids)
    adata_sub = adata[mask].copy()

    pb_means = pseudobulk_expression(
        adata_sub, gene_cols,
        groupby=[participant_col, visit_col, arm_col],
        log1p=False,
    )
    pb_counts = _pseudobulk_counts_from_adata(
        adata_sub, gene_cols, [participant_col, visit_col, arm_col]
    )
    sim_sub = {
        "adata": adata_sub,
        "pseudobulk_means": pb_means,
        "pseudobulk_counts": pb_counts,
    }

    import time as _time

    pvals_by_method = {}
    method_times = []
    for method in methods:
        t0 = _time.time()
        try:
            res = _dispatch_method(method, sim_sub, gene_cols)
            pvals_by_method[method] = {g: res[g].get("pvalue", np.nan) for g in gene_cols}
        except Exception as exc:
            logger.warning("Subsample frac=%s r=%d method=%s failed: %s", frac, r, method, exc)
            pvals_by_method[method] = {g: np.nan for g in gene_cols}
        method_times.append(f"{method}={_time.time()-t0:.0f}s")

    print(
        f"  [sub frac={frac} r={r:03d}] done — {', '.join(method_times)}",
        flush=True,
    )
    return {"frac": frac, "resample": r, "n_sub": len(sub_pids), "pvals": pvals_by_method}


def run_subsampling(
    adata,
    gene_cols: list[str],
    methods: list[str] | None = None,
    fractions: list[float] | None = None,
    n_resamples: int = 100,
    n_jobs: int = 1,
    participant_col: str = "participant",
    arm_col: str = "arm",
    visit_col: str = "visit",
    output_path: str | Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run subsampling reproducibility on a single dataset.

    Parameters
    ----------
    adata : AnnData
        Full dataset.
    gene_cols : list[str]
        Genes to test.
    methods : list[str]
        Methods to benchmark.
    fractions : list[float]
        Participant fractions to subsample. Default: [0.5, 0.7, 0.9].
    n_resamples : int
        Number of random subsamples per fraction.
    n_jobs : int
        Parallel workers. -1 = all CPUs. >1 serializes adata to h5ad.

    Returns
    -------
    DataFrame with: fraction, resample, method, spearman_rho, jaccard_top20
    """
    if methods is None:
        from .orchestrator import CORE_METHODS

        methods = CORE_METHODS
    if fractions is None:
        fractions = [0.5, 0.7, 0.9]

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    from scipy.stats import spearmanr

    from .metrics import compute_topk_jaccard
    from .orchestrator import _dispatch_method, _pseudobulk_counts_from_adata

    rng = np.random.default_rng(seed)
    participants = adata.obs[participant_col].unique()
    n_total = len(participants)

    import datetime

    def _ts() -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")

    # Full-data reference (sequential — done once)
    print(
        f"[{_ts()}] Running full-data reference ({len(methods)} methods, {len(gene_cols)} genes)...",
        flush=True,
    )
    from sctrial.stats.pseudobulk import pseudobulk_expression

    pb_means_full = pseudobulk_expression(
        adata,
        gene_cols,
        groupby=[participant_col, visit_col, arm_col],
        log1p=False,
    )
    pb_counts_full = _pseudobulk_counts_from_adata(
        adata, gene_cols, [participant_col, visit_col, arm_col]
    )
    sim_full = {
        "adata": adata,
        "pseudobulk_means": pb_means_full,
        "pseudobulk_counts": pb_counts_full,
    }

    full_pvals: dict[str, pd.Series] = {}
    for method in methods:
        try:
            res = _dispatch_method(method, sim_full, gene_cols)
            full_pvals[method] = pd.Series({g: res[g]["pvalue"] for g in gene_cols})
        except Exception as exc:
            logger.warning("Full-data %s failed: %s", method, exc)
            full_pvals[method] = pd.Series({g: np.nan for g in gene_cols})

    # Build all (fraction, resample) argument tuples
    all_args = []
    for frac in fractions:
        n_sub = max(4, int(n_total * frac))
        for r in range(n_resamples):
            sub_pids = rng.choice(participants, size=n_sub, replace=False).tolist()
            all_args.append((frac, r, sub_pids, None, gene_cols, methods,
                             participant_col, arm_col, visit_col))

    total_runs = len(all_args)
    print(
        f"[{_ts()}] Running {total_runs} subsamples "
        f"({len(fractions)} fractions × {n_resamples} resamples, {n_jobs} workers)...",
        flush=True,
    )
    t0 = time.time()

    subsample_results = []

    if n_jobs == 1:
        for i, args in enumerate(all_args):
            args = args[:3] + (adata,) + args[4:]
            subsample_results.append(_run_subsample_iteration(args))
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (total_runs - i - 1)
                print(
                    f"[{_ts()}] {i+1}/{total_runs} subsamples "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                    flush=True,
                )
    else:
        import tempfile
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            tmp_path = f.name
        try:
            print(f"[{_ts()}] Serializing adata to disk for parallel workers...", flush=True)
            adata.write_h5ad(tmp_path)
            print(f"[{_ts()}] Serialization done. Dispatching {total_runs} jobs...", flush=True)

            path_args = [a[:3] + (tmp_path,) + a[4:] for a in all_args]

            n_done = 0
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(_run_subsample_iteration, a): (a[0], a[1])
                    for a in path_args
                }
                for future in as_completed(futures):
                    try:
                        subsample_results.append(future.result())
                    except Exception as exc:
                        frac, r = futures[future]
                        logger.warning("Subsample frac=%s r=%d raised: %s", frac, r, exc)
                    n_done += 1
                    if n_done % 20 == 0:
                        elapsed = time.time() - t0
                        eta = elapsed / n_done * (total_runs - n_done)
                        print(
                            f"[{_ts()}] {n_done}/{total_runs} subsamples complete "
                            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                            flush=True,
                        )
        finally:
            os.unlink(tmp_path)

    # Compute Spearman ρ and Jaccard against full-data reference
    rows = []
    for item in subsample_results:
        if item is None:
            continue
        frac = item["frac"]
        r = item["resample"]
        n_sub = item["n_sub"]
        for method in methods:
            sub_pvals = pd.Series(item["pvals"].get(method, {}))
            full = full_pvals[method]

            common = sub_pvals.dropna().index.intersection(full.dropna().index)
            rho = np.nan
            if len(common) > 5:
                rho, _ = spearmanr(full[common], sub_pvals[common])

            jaccard = compute_topk_jaccard(full, sub_pvals, k=20)

            rows.append({
                "fraction": frac,
                "resample": r,
                "method": method,
                "spearman_rho": rho,
                "jaccard_top20": jaccard,
                "n_participants": n_sub,
            })

    df = pd.DataFrame(rows)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved → {output_path}")

    return df
