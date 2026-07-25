"""Participant-label permutation for real-data null calibration.

Shuffles arm labels (two-arm) or visit labels (single-arm) while
preserving the participant structure. All core methods are run on
each permuted dataset.
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


def _permute_arms(adata, participant_col: str, arm_col: str, rng):
    """Shuffle arm labels across participants (preserving within-participant structure)."""
    adata = adata.copy()
    pids = adata.obs[participant_col].unique()
    arms = adata.obs.groupby(participant_col)[arm_col].first()
    shuffled_arms = rng.permutation(arms.values)
    arm_map = dict(zip(pids, shuffled_arms))
    adata.obs[arm_col] = adata.obs[participant_col].map(arm_map)
    return adata


def _permute_visits(adata, participant_col: str, visit_col: str, rng):
    """Shuffle visit labels within each participant."""
    adata = adata.copy()
    for pid in adata.obs[participant_col].unique():
        mask = adata.obs[participant_col] == pid
        visits = adata.obs.loc[mask, visit_col].values.copy()
        adata.obs.loc[mask, visit_col] = rng.permutation(visits)
    return adata


def _run_permutation_iteration(args: tuple) -> list[dict]:
    """Run all methods on one permuted dataset.

    The third element of args may be either an in-memory AnnData (n_jobs=1)
    or a path to an h5ad file (parallel workers, avoids pickling).
    """
    import signal
    import time
    import warnings

    warnings.filterwarnings("ignore")

    (
        perm_idx,
        seed,
        adata_or_path,
        design_type,
        gene_cols,
        methods,
        participant_col,
        arm_col,
        visit_col,
    ) = args

    # Hard wall-clock limit per permutation. If any step (pseudobulk, R session,
    # etc.) hangs beyond this, SIGALRM fires and propagates out to the executor,
    # which logs a warning and moves on. Runners re-raise TimeoutError so it
    # cannot be silently swallowed by their broad except clauses.
    _WALL_CLOCK_SECS = 3600

    def _wall_timeout(signum, frame):
        raise TimeoutError(
            f"Permutation {perm_idx} exceeded {_WALL_CLOCK_SECS}s wall-clock limit"
        )

    old_handler = signal.signal(signal.SIGALRM, _wall_timeout)
    signal.alarm(_WALL_CLOCK_SECS)

    try:
        from .orchestrator import _dispatch_method, _pseudobulk_counts_from_adata

        if isinstance(adata_or_path, (str, Path)):
            import anndata as ad

            adata = ad.read_h5ad(adata_or_path)
        else:
            adata = adata_or_path

        rng = np.random.default_rng(seed)

        if design_type == "two_arm":
            adata_perm = _permute_arms(adata, participant_col, arm_col, rng)
        else:
            adata_perm = _permute_visits(adata, participant_col, visit_col, rng)

        # Standardize obs column names so _dispatch_method runners work with defaults
        col_rename = {
            c: std for c, std in [
                (participant_col, "participant"),
                (arm_col, "arm"),
                (visit_col, "visit"),
            ]
            if c != std and c in adata_perm.obs.columns
        }
        if col_rename:
            adata_perm.obs = adata_perm.obs.rename(columns=col_rename)

        from sctrial.stats.pseudobulk import pseudobulk_expression

        pb_means = pseudobulk_expression(
            adata_perm,
            gene_cols,
            groupby=["participant", "visit", "arm"],
            log1p=False,
        )
        pb_counts = _pseudobulk_counts_from_adata(
            adata_perm, gene_cols, ["participant", "visit", "arm"]
        )
        sim = {"adata": adata_perm, "pseudobulk_means": pb_means, "pseudobulk_counts": pb_counts}

        rows = []
        method_times = []
        for method in methods:
            t0 = time.time()
            try:
                results = _dispatch_method(method, sim, gene_cols)
            except TimeoutError:
                raise  # let the wall-clock alarm propagate
            except Exception as exc:
                logger.warning("Permutation %d, method %s failed: %s", perm_idx, method, exc)
                results = {g: {"pvalue": np.nan} for g in gene_cols}
            elapsed = time.time() - t0
            method_times.append(f"{method}={elapsed:.0f}s")

            for gene in gene_cols:
                r = results.get(gene, {})
                rows.append(
                    {
                        "permutation": perm_idx,
                        "method": method,
                        "gene": gene,
                        "pvalue": r.get("pvalue", np.nan),
                        "beta": r.get("beta", np.nan),
                        "converged": r.get("converged", np.nan),
                        "failure_mode": r.get("failure_mode", None),
                        "runtime_seconds": elapsed,
                    }
                )

        print(
            f"  [perm {perm_idx:04d}] done — {', '.join(method_times)}",
            flush=True,
        )
        return rows

    except TimeoutError as exc:
        print(f"  [perm {perm_idx:04d}] WALL-CLOCK TIMEOUT — {exc}", flush=True)
        # Return NaN rows so this permutation appears in completed_perms on resume
        # and is not retried (same seed → same degenerate dataset every time).
        return [
            {
                "permutation": perm_idx,
                "method": method,
                "gene": gene,
                "pvalue": np.nan,
                "beta": np.nan,
                "converged": np.nan,
                "failure_mode": "timeout",
                "runtime_seconds": np.nan,
            }
            for method in methods
            for gene in gene_cols
        ]

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def run_permutation_test(
    adata,
    gene_cols: list[str],
    design_type: str = "two_arm",
    methods: list[str] | None = None,
    n_permutations: int = 1000,
    n_jobs: int = 1,
    participant_col: str = "participant",
    arm_col: str = "arm",
    visit_col: str = "visit",
    output_path: str | Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run participant-label permutation test on real data.

    Parameters
    ----------
    adata : AnnData
        Real dataset.
    gene_cols : list[str]
        Genes to test.
    design_type : {"two_arm", "single_arm"}
    methods : list[str]
        Methods to run. Default: all core methods.
    n_permutations : int
        Number of permutations.
    n_jobs : int
        Parallel workers. -1 = all CPUs. >1 serializes adata to a
        temporary h5ad file so subprocesses can load it without pickling.

    Returns
    -------
    DataFrame with columns: permutation, method, gene, pvalue
    """
    if methods is None:
        from .orchestrator import CORE_METHODS

        methods = CORE_METHODS

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_permutations)]

    import datetime

    def _ts() -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")

    # Resume: load any existing partial results and skip completed permutations
    all_rows: list[dict] = []
    completed_perms: set[int] = set()
    if output_path and Path(output_path).exists():
        try:
            existing = pd.read_csv(output_path)
            completed_perms = set(existing["permutation"].astype(int).unique())
            all_rows = existing.to_dict("records")
            print(
                f"[{_ts()}] Resuming: {len(completed_perms)}/{n_permutations} "
                f"permutations already done — skipping those.",
                flush=True,
            )
        except Exception as exc:
            logger.warning("Could not load existing results for resume: %s", exc)

    remaining = [i for i in range(n_permutations) if i not in completed_perms]
    print(
        f"[{_ts()}] Running {len(remaining)} remaining permutations × {len(methods)} methods "
        f"on {len(gene_cols)} genes ({n_jobs} workers)...",
        flush=True,
    )

    if not remaining:
        print(f"[{_ts()}] All permutations already complete.", flush=True)
        return pd.DataFrame(all_rows)

    def _periodic_save(rows, path):
        if path is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    t0 = time.time()

    if n_jobs == 1:
        for count, i in enumerate(remaining):
            args = (
                i, seeds[i], adata, design_type, gene_cols, methods,
                participant_col, arm_col, visit_col,
            )
            all_rows.extend(_run_permutation_iteration(args))
            if (count + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (count + 1) * (len(remaining) - count - 1)
                print(
                    f"[{_ts()}] {len(completed_perms) + count + 1}/{n_permutations} permutations "
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
                f"[{_ts()}] Serialization done. Dispatching {len(remaining)} jobs...",
                flush=True,
            )

            arg_list = [
                (i, seeds[i], tmp_path, design_type, gene_cols, methods,
                 participant_col, arm_col, visit_col)
                for i in remaining
            ]

            n_done = 0
            ctx = mp.get_context("spawn")
            # No max_tasks_per_child: workers live for the job lifetime.
            # Recycling all workers simultaneously caused every 4×N tasks to
            # deadlock on NFS when new processes loaded R libraries concurrently.
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
                futures = {
                    executor.submit(_run_permutation_iteration, a): a[0]
                    for a in arg_list
                }
                for future in as_completed(futures):
                    try:
                        all_rows.extend(future.result())
                    except Exception as exc:
                        perm_idx = futures[future]
                        logger.warning("Permutation %d raised: %s", perm_idx, exc)
                    n_done += 1
                    if n_done % 50 == 0:
                        elapsed = time.time() - t0
                        eta = elapsed / n_done * (len(remaining) - n_done)
                        total_done = len(completed_perms) + n_done
                        print(
                            f"[{_ts()}] {total_done}/{n_permutations} permutations complete "
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
