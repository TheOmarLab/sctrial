"""Subsampling reproducibility analysis.

For each dataset, subsample participants at different fractions,
run all methods, and measure ranking stability via Spearman ρ
and top-k Jaccard overlap.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_subsampling(
    adata,
    gene_cols: list[str],
    methods: list[str] | None = None,
    fractions: list[float] | None = None,
    n_resamples: int = 100,
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

    Returns
    -------
    DataFrame with: fraction, resample, method, spearman_rho, jaccard_top20
    """
    if methods is None:
        from .orchestrator import CORE_METHODS

        methods = CORE_METHODS
    if fractions is None:
        fractions = [0.5, 0.7, 0.9]

    from scipy.stats import spearmanr

    from .contracts import prepare_inputs_from_adata
    from .metrics import compute_topk_jaccard
    from .orchestrator import _dispatch_method

    rng = np.random.default_rng(seed)
    participants = adata.obs[participant_col].unique()
    n_total = len(participants)

    # First: run full-data reference for each method
    print(f"Running full-data reference ({len(methods)} methods, {len(gene_cols)} genes)...")
    # Same contracts as the simulated path; see the note in permutation.py.
    inputs_full = prepare_inputs_from_adata(
        adata,
        gene_cols,
        participant_col=participant_col,
        visit_col=visit_col,
        arm_col=arm_col,
    )

    full_pvals = {}
    for method in methods:
        try:
            res = _dispatch_method(method, inputs_full)
            full_pvals[method] = pd.Series({g: res[g]["pvalue"] for g in gene_cols})
        except Exception as exc:
            logger.warning("Full-data %s failed: %s", method, exc)
            full_pvals[method] = pd.Series({g: np.nan for g in gene_cols})

    # Run subsamples
    rows = []
    t0 = time.time()
    total_runs = len(fractions) * n_resamples

    run_count = 0
    for frac in fractions:
        n_sub = max(4, int(n_total * frac))
        print(f"\nFraction {frac}: {n_sub}/{n_total} participants, {n_resamples} resamples...")

        for r in range(n_resamples):
            sub_pids = rng.choice(participants, size=n_sub, replace=False)
            mask = adata.obs[participant_col].isin(sub_pids)
            adata_sub = adata[mask].copy()

            inputs_sub = prepare_inputs_from_adata(
                adata_sub,
                gene_cols,
                participant_col=participant_col,
                visit_col=visit_col,
                arm_col=arm_col,
            )

            for method in methods:
                try:
                    res = _dispatch_method(method, inputs_sub)
                    sub_pvals = pd.Series({g: res[g]["pvalue"] for g in gene_cols})

                    # Spearman ρ
                    common = sub_pvals.dropna().index.intersection(
                        full_pvals[method].dropna().index
                    )
                    if len(common) > 5:
                        rho, _ = spearmanr(
                            full_pvals[method][common],
                            sub_pvals[common],
                        )
                    else:
                        rho = np.nan

                    # Top-k Jaccard
                    jaccard = compute_topk_jaccard(
                        full_pvals[method],
                        sub_pvals,
                        k=20,
                    )

                    rows.append(
                        {
                            "fraction": frac,
                            "resample": r,
                            "method": method,
                            "spearman_rho": rho,
                            "jaccard_top20": jaccard,
                            "n_participants": n_sub,
                        }
                    )
                except Exception as exc:
                    logger.debug("Subsample %s frac=%s r=%d failed: %s", method, frac, r, exc)
                    rows.append(
                        {
                            "fraction": frac,
                            "resample": r,
                            "method": method,
                            "spearman_rho": np.nan,
                            "jaccard_top20": np.nan,
                            "n_participants": n_sub,
                        }
                    )

            run_count += 1
            if run_count % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / run_count * (total_runs - run_count)
                print(f"  {run_count}/{total_runs} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    df = pd.DataFrame(rows)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved → {output_path}")

    return df
