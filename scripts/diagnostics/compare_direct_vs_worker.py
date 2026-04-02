#!/usr/bin/env python
"""Diagnostic 4: Compare direct runner calls vs worker path for same seeds.

This isolates whether corruption happens inside _run_single_iteration
or in the multiprocessing/batching/output layer.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sctrial.benchmark.orchestrator import _run_single_iteration
from sctrial.benchmark.simulator import SimulationConfig, simulate_trial
from sctrial.benchmark.runners import edger_qlf


OUTDIR = Path("manuscript/benchmark/diagnostics/direct_vs_worker")
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

ALL_METHODS = [
    "sctrial_did",
    "edger_qlf",
    "limma_voom",
    "dreamlet",
    "nebula",
    "wilcoxon_paired",
]


def direct_call(seed: int) -> pd.DataFrame:
    """Call edgeR runner directly, bypassing _run_single_iteration."""
    cfg = SimulationConfig(seed=seed, **CONFIG_KWARGS)
    sim = simulate_trial(cfg)
    pb = sim["pseudobulk_counts"]
    genes = [c for c in pb.columns if c.startswith("gene_")]
    res = edger_qlf.run(pb, genes, design_type="two_arm")
    rows = []
    for g, d in res.items():
        rows.append({
            "gene": g,
            "beta": d.get("beta", np.nan),
            "pvalue": d.get("pvalue", np.nan),
        })
    return pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)


def worker_call(seed: int) -> pd.DataFrame:
    """Call through _run_single_iteration (same as pool workers)."""
    args = ("diag_null_n8", seed, seed, CONFIG_KWARGS, ["edger_qlf"])
    rows = _run_single_iteration(args)
    df = pd.DataFrame(rows)
    return df[["gene", "estimated_beta", "pvalue"]].rename(
        columns={"estimated_beta": "beta"}
    ).sort_values("gene").reset_index(drop=True)


def worker_call_wrapper(seed: int) -> tuple[int, pd.DataFrame]:
    return seed, worker_call(seed)


def main() -> None:
    print("=== Direct calls (main process) ===")
    direct = {}
    for seed in SEEDS:
        direct[seed] = direct_call(seed)
        pvals = direct[seed]["pvalue"].dropna()
        print(f"  Seed {seed}: median_p={pvals.median():.3f}, "
              f"FPR={float((pvals < 0.05).mean()):.3f}, n={len(pvals)}")

    print("\n=== Worker calls (main process, sequential) ===")
    worker_seq = {}
    for seed in SEEDS:
        worker_seq[seed] = worker_call(seed)
        pvals = worker_seq[seed]["pvalue"].dropna()
        print(f"  Seed {seed}: median_p={pvals.median():.3f}, "
              f"FPR={float((pvals < 0.05).mean()):.3f}, n={len(pvals)}")

    print("\n=== Worker calls (spawn pool, 4 workers) ===")
    ctx = mp.get_context("spawn")
    worker_par = {}
    with ctx.Pool(4) as pool:
        for seed, df in pool.imap(worker_call_wrapper, SEEDS):
            worker_par[seed] = df
            pvals = df["pvalue"].dropna()
            print(f"  Seed {seed}: median_p={pvals.median():.3f}, "
                  f"FPR={float((pvals < 0.05).mean()):.3f}, n={len(pvals)}")

    print("\n=== Comparison ===")
    rows = []
    for seed in SEEDS:
        d = direct[seed]
        ws = worker_seq[seed]
        wp = worker_par[seed]

        # Direct vs worker-sequential
        m1 = d.merge(ws, on="gene", suffixes=("_direct", "_worker_seq"))
        # Direct vs worker-parallel
        m2 = d.merge(wp, on="gene", suffixes=("_direct", "_worker_par"))

        rows.append({
            "seed": seed,
            "direct_median_p": float(d["pvalue"].median()),
            "worker_seq_median_p": float(ws["pvalue"].median()),
            "worker_par_median_p": float(wp["pvalue"].median()),
            "direct_fpr": float((d["pvalue"] < 0.05).mean()),
            "worker_seq_fpr": float((ws["pvalue"] < 0.05).mean()),
            "worker_par_fpr": float((wp["pvalue"] < 0.05).mean()),
            "max_p_diff_seq": float(
                np.nanmax(np.abs(m1["pvalue_direct"] - m1["pvalue_worker_seq"]))
            ),
            "max_p_diff_par": float(
                np.nanmax(np.abs(m2["pvalue_direct"] - m2["pvalue_worker_par"]))
            ),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "direct_vs_worker.csv", index=False)
    print(out.to_string(index=False))

    # Summary
    print(f"\nMean FPR - direct: {out['direct_fpr'].mean():.4f}")
    print(f"Mean FPR - worker_seq: {out['worker_seq_fpr'].mean():.4f}")
    print(f"Mean FPR - worker_par: {out['worker_par_fpr'].mean():.4f}")

    max_diff = max(out["max_p_diff_seq"].max(), out["max_p_diff_par"].max())
    if max_diff < 1e-6:
        print(f"\nPASS: All execution modes produce identical results (max diff={max_diff:.2e})")
    else:
        print(f"\nWARNING: Execution modes differ (max diff={max_diff:.2e})")


if __name__ == "__main__":
    main()
