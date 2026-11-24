"""Participant-label permutation for real-data null calibration.

Shuffles arm labels across participants (two-arm) or exchanges the two visit
labels within participants (single-arm). Samples stay intact. All core methods
are run on each permuted dataset.

The cohort is mapped once to the vocabulary of the runners
(:func:`sctrial.benchmark.contracts.canonical_cohort`) before the permutations.
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
    # observed=True: on a Categorical participant column with unused levels the
    # groupby would add NaN-arm rows, and the shuffled labels would no longer be a
    # permutation of the real participants' arms.
    arms = adata.obs.groupby(participant_col, observed=True)[arm_col].first()
    arm_map = dict(zip(arms.index, rng.permutation(arms.values)))
    adata.obs[arm_col] = adata.obs[participant_col].map(arm_map)
    return adata


def _permute_visits(adata, participant_col: str, visit_col: str, rng):
    """Exchange the two visit labels of a random subset of the participants.

    Each participant's pair of samples is kept or exchanged as a whole, so every cell of a
    sample carries the same label and the permuted dataset has the samples of the cohort.
    """
    adata = adata.copy()
    visit = adata.obs[visit_col]
    labels = list(pd.unique(visit.dropna()))
    if len(labels) != 2:
        raise ValueError(f"visit permutation needs two visit labels, found {labels}.")
    participants = pd.unique(adata.obs[participant_col])
    exchanged = participants[rng.integers(0, 2, size=len(participants)).astype(bool)]
    values = np.asarray(visit, dtype=object).copy()
    first, second = values == labels[0], values == labels[1]
    rows = adata.obs[participant_col].isin(exchanged).to_numpy()
    values[rows & first] = labels[1]
    values[rows & second] = labels[0]
    adata.obs[visit_col] = pd.Series(values, index=adata.obs.index).astype(visit.dtype)
    return adata


def _run_permutation_iteration(args: tuple) -> list[dict]:
    """Run all methods on one permuted dataset."""
    import warnings

    warnings.filterwarnings("ignore")

    perm_idx, seed, cohort, design_type, gene_cols, methods = args

    from .contracts import prepare_inputs_from_adata
    from .orchestrator import _dispatch_method

    rng = np.random.default_rng(seed)

    # Permute. ``cohort`` carries the canonical columns and labels.
    if design_type == "two_arm":
        adata_perm = _permute_arms(cohort, "participant", "arm", rng)
    else:
        adata_perm = _permute_visits(cohort, "participant", "visit", rng)

    # The SAME contracts as the simulated path: full-transcriptome normalisation,
    # full-transcriptome library sizes, panel selected afterwards. This analysis
    # previously built its own pseudobulk and normalised inside the tested panel,
    # so the real-data results characterised these methods under a different
    # normalisation scope from the simulation that was used to characterise them.
    inputs = prepare_inputs_from_adata(adata_perm, gene_cols, counts_layer=None)

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
    *,
    visits: tuple[str, str],
    arm_treated: str | None = None,
    arm_control: str | None = None,
    counts_layer: str | None = "counts",
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
        Not used for the computation: the permutations run one after another. The
        value appears in the progress message only.
    visits : tuple[str, str]
        (baseline, follow-up) labels of ``visit_col`` in this cohort.
    arm_treated, arm_control : str
        The two labels of ``arm_col`` in this cohort. Required for a two-arm
        design; rows of other arms are left out with one warning.
    counts_layer : str or None
        Layer that holds raw counts (``None`` for ``adata.X``). The count models
        refuse a matrix with non-integer values.

    Returns
    -------
    DataFrame with columns: permutation, method, gene, pvalue
    """
    from .contracts import canonical_cohort, require_raw_counts

    if methods is None:
        from .orchestrator import CORE_METHODS

        methods = CORE_METHODS
    if design_type not in ("two_arm", "single_arm"):
        raise ValueError(f"design_type must be 'two_arm' or 'single_arm', got {design_type!r}")

    cohort = canonical_cohort(
        adata,
        participant_col=participant_col,
        visit_col=visit_col,
        visits=visits,
        arm_col=arm_col if design_type == "two_arm" else None,
        arm_treated=arm_treated,
        arm_control=arm_control,
        counts_layer=counts_layer,
    )
    require_raw_counts(methods, cohort)

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
        args = (i, seeds[i], cohort, design_type, gene_cols, methods)
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
