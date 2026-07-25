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
    import signal
    import warnings

    warnings.filterwarnings("ignore")

    (frac, r, sub_pids, adata_or_path, gene_cols, methods,
     participant_col, arm_col, visit_col) = args

    _WALL_CLOCK_SECS = 3600

    def _wall_timeout(signum, frame):
        raise TimeoutError(
            f"Subsample frac={frac} r={r} exceeded {_WALL_CLOCK_SECS}s wall-clock limit"
        )

    old_handler = signal.signal(signal.SIGALRM, _wall_timeout)
    signal.alarm(_WALL_CLOCK_SECS)

    try:
        from .orchestrator import _dispatch_method, _pseudobulk_counts_from_adata
        from sctrial.stats.pseudobulk import pseudobulk_expression

        if isinstance(adata_or_path, (str, Path)):
            import anndata as ad

            adata = ad.read_h5ad(adata_or_path)
        else:
            adata = adata_or_path

        mask = adata.obs[participant_col].isin(sub_pids)
        adata_sub = adata[mask].copy()

        # Standardize obs column names so runners work with defaults
        col_rename = {
            c: std for c, std in [
                (participant_col, "participant"),
                (arm_col, "arm"),
                (visit_col, "visit"),
            ]
            if c != std and c in adata_sub.obs.columns
        }
        if col_rename:
            adata_sub.obs = adata_sub.obs.rename(columns=col_rename)

        pb_means = pseudobulk_expression(
            adata_sub, gene_cols,
            groupby=["participant", "visit", "arm"],
            log1p=False,
        )
        pb_counts = _pseudobulk_counts_from_adata(
            adata_sub, gene_cols, ["participant", "visit", "arm"]
        )
        sim_sub = {
            "adata": adata_sub,
            "pseudobulk_means": pb_means,
            "pseudobulk_counts": pb_counts,
        }

        import time as _time

        pvals_by_method = {}
        runtimes_by_method = {}
        method_times = []
        for method in methods:
            t0 = _time.time()
            try:
                res = _dispatch_method(method, sim_sub, gene_cols)
                pvals_by_method[method] = {g: res[g].get("pvalue", np.nan) for g in gene_cols}
            except TimeoutError:
                raise  # let the wall-clock alarm propagate
            except Exception as exc:
                logger.warning("Subsample frac=%s r=%d method=%s failed: %s", frac, r, method, exc)
                pvals_by_method[method] = {g: np.nan for g in gene_cols}
            elapsed = _time.time() - t0
            runtimes_by_method[method] = elapsed
            method_times.append(f"{method}={elapsed:.0f}s")

        print(
            f"  [sub frac={frac} r={r:03d}] done — {', '.join(method_times)}",
            flush=True,
        )
        return {
            "frac": frac,
            "resample": r,
            "n_sub": len(sub_pids),
            "pvals": pvals_by_method,
            "runtimes": runtimes_by_method,
        }

    except TimeoutError as exc:
        print(f"  [sub frac={frac} r={r:03d}] WALL-CLOCK TIMEOUT — {exc}", flush=True)
        # Return NaN results so this (frac, resample) appears in completed_pairs
        # on resume and is not retried.
        return {
            "frac": frac,
            "resample": r,
            "n_sub": len(sub_pids),
            "pvals": {method: {g: np.nan for g in gene_cols} for method in methods},
            "runtimes": {},
        }

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


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

    # Standardize adata obs column names so runners work with defaults
    col_rename = {
        c: std for c, std in [
            (participant_col, "participant"),
            (arm_col, "arm"),
            (visit_col, "visit"),
        ]
        if c != std and c in adata.obs.columns
    }
    if col_rename:
        adata = adata.copy()
        adata.obs = adata.obs.rename(columns=col_rename)

    pb_means_full = pseudobulk_expression(
        adata,
        gene_cols,
        groupby=["participant", "visit", "arm"],
        log1p=False,
    )
    pb_counts_full = _pseudobulk_counts_from_adata(
        adata, gene_cols, ["participant", "visit", "arm"]
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

    def _item_to_rows(item):
        """Compute Spearman ρ and Jaccard for one subsample result dict."""
        out = []
        for method in methods:
            sub_pvals = pd.Series(item["pvals"].get(method, {}))
            full = full_pvals[method]
            common = sub_pvals.dropna().index.intersection(full.dropna().index)
            rho = np.nan
            if len(common) > 5:
                rho, _ = spearmanr(full[common], sub_pvals[common])
            jaccard = compute_topk_jaccard(full, sub_pvals, k=20)
            n_valid = int(sub_pvals.notna().sum())
            out.append({
                "fraction": item["frac"],
                "resample": item["resample"],
                "method": method,
                "spearman_rho": rho,
                "jaccard_top20": jaccard,
                "n_participants": item["n_sub"],
                "n_valid_genes": n_valid,
                "runtime_seconds": item.get("runtimes", {}).get(method, np.nan),
            })
        return out

    def _periodic_save(rows, path):
        if path is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    # Resume: load existing partial results and skip completed (frac, resample) pairs
    all_rows: list[dict] = []
    completed_pairs: set[tuple] = set()
    if output_path and Path(output_path).exists():
        try:
            existing = pd.read_csv(output_path)
            # Each (fraction, resample) pair has one row per method — deduplicate
            completed_pairs = {
                (float(row["fraction"]), int(row["resample"]))
                for _, row in existing.iterrows()
            }
            all_rows = existing.to_dict("records")
            print(
                f"[{_ts()}] Resuming: {len(completed_pairs)} (frac, resample) pairs "
                f"already done — skipping those.",
                flush=True,
            )
        except Exception as exc:
            logger.warning("Could not load existing results for resume: %s", exc)

    # Build all (fraction, resample) argument tuples.
    # Use standardized column names ("participant"/"arm"/"visit") because the
    # adata passed to workers has already been renamed.
    all_args = []
    for frac in fractions:
        n_sub = max(4, int(n_total * frac))
        for r in range(n_resamples):
            if (float(frac), int(r)) in completed_pairs:
                continue
            sub_pids = rng.choice(participants, size=n_sub, replace=False).tolist()
            all_args.append((frac, r, sub_pids, None, gene_cols, methods,
                             "participant", "arm", "visit"))

    total_remaining = len(all_args)
    total_runs = len(fractions) * n_resamples
    print(
        f"[{_ts()}] Running {total_remaining} remaining subsamples "
        f"(of {total_runs} total, {n_jobs} workers)...",
        flush=True,
    )

    if not all_args:
        print(f"[{_ts()}] All subsamples already complete.", flush=True)
        return pd.DataFrame(all_rows)

    t0 = time.time()

    if n_jobs == 1:
        for i, args in enumerate(all_args):
            args = args[:3] + (adata,) + args[4:]
            item = _run_subsample_iteration(args)
            if item is not None:
                all_rows.extend(_item_to_rows(item))
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (total_remaining - i - 1)
                print(
                    f"[{_ts()}] {len(completed_pairs) + i + 1}/{total_runs} subsamples "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                    flush=True,
                )
                _periodic_save(all_rows, output_path)
    else:
        import tempfile
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            tmp_path = f.name
        try:
            print(f"[{_ts()}] Serializing adata to disk for parallel workers...", flush=True)
            adata.write_h5ad(tmp_path)
            print(
                f"[{_ts()}] Serialization done. Dispatching {total_remaining} jobs...",
                flush=True,
            )

            path_args = [a[:3] + (tmp_path,) + a[4:] for a in all_args]

            n_done = 0
            ctx = mp.get_context("spawn")
            # No max_tasks_per_child: workers live for the job lifetime.
            # Recycling all workers simultaneously caused every 4×N tasks to
            # deadlock on NFS when new processes loaded R libraries concurrently.
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
                futures = {
                    executor.submit(_run_subsample_iteration, a): (a[0], a[1])
                    for a in path_args
                }
                for future in as_completed(futures):
                    try:
                        item = future.result()
                        if item is not None:
                            all_rows.extend(_item_to_rows(item))
                    except Exception as exc:
                        frac, r = futures[future]
                        logger.warning("Subsample frac=%s r=%d raised: %s", frac, r, exc)
                    n_done += 1
                    if n_done % 20 == 0:
                        elapsed = time.time() - t0
                        eta = elapsed / n_done * (total_remaining - n_done)
                        total_done = len(completed_pairs) + n_done
                        print(
                            f"[{_ts()}] {total_done}/{total_runs} subsamples complete "
                            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                            flush=True,
                        )
                        _periodic_save(all_rows, output_path)
        finally:
            os.unlink(tmp_path)

    df = pd.DataFrame(all_rows)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[{_ts()}] Saved → {output_path}", flush=True)

    return df
