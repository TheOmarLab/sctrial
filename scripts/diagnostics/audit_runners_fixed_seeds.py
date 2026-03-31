#!/usr/bin/env python
"""Diagnostic 3: Audit edgeR and limma on fixed seeds.

Records per-seed: number of genes retained after filtering,
median p-value, FPR, and raw beta distributions.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sctrial.benchmark.simulator import SimulationConfig, simulate_trial
from sctrial.benchmark.runners import edger_qlf, limma_voom


OUTDIR = Path("manuscript/benchmark/diagnostics/runner_audit")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(10))
CONFIG_KWARGS = dict(
    design="two_arm",
    n_per_arm=8,
    n_genes=50,
    effects={},
    mean_cells_per_visit=500,
    baseline_mean=-12.86,
    baseline_sd=2.67,
    target_library_size=2981,
    library_size_sd=0.76,
    participant_sd=0.05,
)

RUNNERS = {
    "edger_qlf": edger_qlf,
    "limma_voom": limma_voom,
}


def summarize_result(
    res: dict, method: str, seed: int, n_genes: int
) -> dict:
    valid = {g: d for g, d in res.items() if d.get("converged", False)}
    pvals = [d["pvalue"] for d in valid.values() if d["pvalue"] == d["pvalue"]]
    betas = [d["beta"] for d in valid.values() if d["beta"] == d["beta"]]

    return {
        "seed": seed,
        "method": method,
        "n_input_genes": n_genes,
        "n_returned_genes": len(res),
        "n_converged": len(valid),
        "n_valid_pvalues": len(pvals),
        "n_filtered_out": n_genes - len(res),
        "median_pvalue": float(np.median(pvals)) if pvals else np.nan,
        "fpr_005": float(np.mean(np.array(pvals) < 0.05)) if pvals else np.nan,
        "mean_abs_beta": float(np.mean(np.abs(betas))) if betas else np.nan,
    }


def main() -> None:
    all_summaries = []
    all_detail = []

    for seed in SEEDS:
        cfg = SimulationConfig(seed=seed, **CONFIG_KWARGS)
        sim = simulate_trial(cfg)
        pb = sim["pseudobulk_counts"]
        genes = [c for c in pb.columns if c.startswith("gene_")]

        for method_name, runner in RUNNERS.items():
            res = runner.run(pb, genes, design_type="two_arm")
            summary = summarize_result(res, method_name, seed, len(genes))
            all_summaries.append(summary)

            for gene, d in res.items():
                all_detail.append({
                    "seed": seed,
                    "method": method_name,
                    "gene": gene,
                    "beta": d.get("beta"),
                    "pvalue": d.get("pvalue"),
                    "converged": d.get("converged"),
                    "failure_mode": d.get("failure_mode"),
                })

        print(f"Seed {seed} done")

    sdf = pd.DataFrame(all_summaries)
    sdf.to_csv(OUTDIR / "runner_audit_summary.csv", index=False)

    ddf = pd.DataFrame(all_detail)
    ddf.to_csv(OUTDIR / "runner_audit_detail.csv", index=False)

    print("\n=== Per-method summary across seeds ===")
    print(sdf.groupby("method")[
        ["n_valid_pvalues", "n_filtered_out", "median_pvalue", "fpr_005"]
    ].agg(["mean", "std"]).round(4))

    print("\n=== Per-seed detail ===")
    print(sdf[["seed", "method", "n_valid_pvalues", "n_filtered_out",
               "median_pvalue", "fpr_005"]].to_string(index=False))


if __name__ == "__main__":
    main()
