"""Participant-label permutation for real-data null calibration.

Shuffles arm labels (two-arm) or visit labels (single-arm) while
preserving the participant structure. All core methods are run on
each permuted dataset.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
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
    """Run all methods on one permuted dataset."""
    import warnings

    warnings.filterwarnings("ignore")

    (
        perm_idx,
        seed,
        adata,
        design_type,
        gene_cols,
        methods,
        participant_col,
        arm_col,
        visit_col,
    ) = args

    from .contracts import prepare_inputs_from_adata
    from .orchestrator import _dispatch_method

    rng = np.random.default_rng(seed)

    # Permute
    if design_type == "two_arm":
        adata_perm = _permute_arms(adata, participant_col, arm_col, rng)
    else:
        adata_perm = _permute_visits(adata, participant_col, visit_col, rng)

    # The SAME contracts as the simulated path: full-transcriptome normalisation,
    # full-transcriptome library sizes, panel selected afterwards. This analysis
    # previously built its own pseudobulk and normalised inside the tested panel,
    # so the real-data results characterised these methods under a different
    # normalisation scope from the simulation that was used to characterise them.
    inputs = prepare_inputs_from_adata(
        adata_perm,
        gene_cols,
        participant_col=participant_col,
        visit_col=visit_col,
        arm_col=arm_col,
    )

    rows = []
    for method in methods:
        try:
            results = _dispatch_method(method, inputs, design_type=design_type)
        except Exception as exc:
            logger.debug("Permutation %d, method %s failed: %s", perm_idx, method, exc)
            results = {g: {"pvalue": np.nan} for g in gene_cols}

        for gene in gene_cols:
            r = results.get(gene, {})
            rows.append(
                {
                    "permutation": perm_idx,
                    "method": method,
                    "gene": gene,
                    "pvalue": r.get("pvalue", np.nan),
                }
            )

    return rows


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
        Parallel workers.

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

    print(
        f"Running {n_permutations} permutations × {len(methods)} methods "
        f"on {len(gene_cols)} genes ({n_jobs} workers)..."
    )

    # NOTE: For multiprocessing with AnnData, we need to serialize carefully.
    # For now, use sequential or thread-based parallelism.
    # Full multiprocessing would require saving adata to disk and reloading.
    all_rows = []
    t0 = time.time()

    for i in range(n_permutations):
        args = (
            i,
            seeds[i],
            adata,
            design_type,
            gene_cols,
            methods,
            participant_col,
            arm_col,
            visit_col,
        )
        rows = _run_permutation_iteration(args)
        all_rows.extend(rows)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_permutations - i - 1)
            print(
                f"  {i + 1}/{n_permutations} permutations "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )

    df = pd.DataFrame(all_rows)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved → {output_path}")

    return df
