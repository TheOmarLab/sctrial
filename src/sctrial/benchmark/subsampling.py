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
    treated_label: str = "Treated",
    control_label: str = "Control",
) -> pd.DataFrame:
    """Run subsampling reproducibility on a single dataset."""
    if methods is None:
        from .orchestrator import CORE_METHODS

        methods = CORE_METHODS
    if fractions is None:
        fractions = [0.5, 0.7, 0.9]

    from scipy.stats import spearmanr

    from .metrics import compute_topk_jaccard
    from .orchestrator import _dispatch_method

    rng = np.random.default_rng(seed)
    participants = adata.obs[participant_col].unique()
    n_total = len(participants)

    # FIX 3: build both pseudobulk keys for the full-data reference so that
    # _dispatch_method routes edgeR/dreamlet/limma to counts, not means.
    print(f"Running full-data reference ({len(methods)} methods, {len(gene_cols)} genes)...")
    from sctrial.stats.pseudobulk import pseudobulk_expression

    pb_means_full = pseudobulk_expression(
        adata,
        gene_cols,
        groupby=[participant_col, visit_col, arm_col],
    )
    pb_counts_full = (
        adata.obs[[participant_col, visit_col, arm_col]]
        .join(pd.DataFrame(
            adata.layers["counts"][
                :, [adata.var_names.get_loc(g) for g in gene_cols]
            ].toarray(),
            columns=gene_cols,
            index=adata.obs.index,
        ))
        .groupby([participant_col, visit_col, arm_col])[gene_cols]
        .sum()
        .reset_index()
    )
    sim_full = {
        "adata": adata,
        "pseudobulk_means": pb_means_full,
        "pseudobulk_counts": pb_counts_full,
    }

    full_pvals = {}
    for method in methods:
        try:
            res = _dispatch_method(method, sim_full, gene_cols,
                                   treated_label=treated_label,
                                   control_label=control_label,
                                   participant_col=participant_col)
            full_pvals[method] = pd.Series({g: res[g]["pvalue"] for g in gene_cols})
        except Exception as exc:
            logger.warning("Full-data %s failed: %s", method, exc)
            full_pvals[method] = pd.Series({g: np.nan for g in gene_cols})

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

            # FIX 3 (cont): same pattern for each subsample — both keys built.
            pb_means_sub = pseudobulk_expression(
                adata_sub,
                gene_cols,
                groupby=[participant_col, visit_col, arm_col],
            )
            pb_counts_sub = (
                adata_sub.obs[[participant_col, visit_col, arm_col]]
                .join(pd.DataFrame(
                    adata_sub.layers["counts"][
                        :, [adata_sub.var_names.get_loc(g) for g in gene_cols]
                    ].toarray(),
                    columns=gene_cols,
                    index=adata_sub.obs.index,
                ))
                .groupby([participant_col, visit_col, arm_col])[gene_cols]
                .sum()
                .reset_index()
            )
            sim_sub = {
                "adata": adata_sub,
                "pseudobulk_means": pb_means_sub,
                "pseudobulk_counts": pb_counts_sub,
            }

            for method in methods:
                try:
                    res = _dispatch_method(method, sim_sub, gene_cols,
                       treated_label=treated_label,
                       control_label=control_label,
                       participant_col=participant_col)
                    sub_pvals = pd.Series({g: res[g]["pvalue"] for g in gene_cols})

                    common = sub_pvals.dropna().index.intersection(
                        full_pvals[method].dropna().index
                    )
                    rho = (
                        spearmanr(full_pvals[method][common], sub_pvals[common])[0]
                        if len(common) > 5
                        else np.nan
                    )

                    jaccard = compute_topk_jaccard(full_pvals[method], sub_pvals, k=20)

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
                print(
                    f"  {run_count}/{total_runs} "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )

    df = pd.DataFrame(rows)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved → {output_path}")

    return df
