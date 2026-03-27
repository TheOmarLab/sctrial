#!/usr/bin/env python
"""Diagnostic 1: Check whether n_jobs=1 and pooled execution produce same results."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sctrial.benchmark.orchestrator import _run_single_iteration


OUTDIR = Path("manuscript/benchmark/diagnostics/parallel_equivalence")
OUTDIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    "sctrial_did",
    "edger_qlf",
    "limma_voom",
    "dreamlet",
    "nebula",
    "wilcoxon_paired",
]

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

SEEDS = list(range(10))


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(["method", "gene"]).reset_index(drop=True)
    keep = [
        "method", "gene", "estimated_beta", "pvalue", "ci_lo", "ci_hi",
        "converged", "failure_mode",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def df_hash(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def one_seed(seed: int) -> tuple[int, pd.DataFrame]:
    args = ("diag_null_n8", seed, seed, CONFIG_KWARGS, METHODS)
    rows = _run_single_iteration(args)
    return seed, rows_to_df(rows)


def compare_frames(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    merged = df1.merge(
        df2,
        on=["method", "gene"],
        suffixes=("_serial", "_parallel"),
        how="outer",
        indicator=True,
    )

    beta_diff = np.abs(
        merged["estimated_beta_serial"] - merged["estimated_beta_parallel"]
    )
    pval_diff = np.abs(merged["pvalue_serial"] - merged["pvalue_parallel"])

    return {
        "row_match": bool((merged["_merge"] == "both").all()),
        "max_abs_beta_diff": float(np.nanmax(beta_diff)),
        "max_abs_pvalue_diff": float(np.nanmax(pval_diff)),
        "n_beta_diff_gt_1e-8": int(np.sum(beta_diff > 1e-8)),
        "n_pvalue_diff_gt_1e-8": int(np.sum(pval_diff > 1e-8)),
    }


def main() -> None:
    print("=== Running serial (n_jobs=1) ===")
    serial = {}
    for seed in SEEDS:
        s, df = one_seed(seed)
        serial[s] = df
        df.to_csv(OUTDIR / f"serial_seed{s}.csv", index=False)
        print(f"  Seed {s} done ({len(df)} rows)")

    print("\n=== Running parallel (spawn, n_workers=4) ===")
    ctx = mp.get_context("spawn")
    args_list = [
        ("diag_null_n8", s, s, CONFIG_KWARGS, METHODS) for s in SEEDS
    ]
    parallel = {}
    with ctx.Pool(4) as pool:
        for s, df in pool.imap(one_seed, SEEDS):
            parallel[s] = df
            df.to_csv(OUTDIR / f"parallel_seed{s}.csv", index=False)
            print(f"  Seed {s} done ({len(df)} rows)")

    print("\n=== Comparison ===")
    summary = []
    for seed in SEEDS:
        comp = compare_frames(serial[seed], parallel[seed])
        comp["seed"] = seed
        comp["serial_hash"] = df_hash(serial[seed])
        comp["parallel_hash"] = df_hash(parallel[seed])
        comp["exact_hash_match"] = comp["serial_hash"] == comp["parallel_hash"]
        summary.append(comp)

    sdf = pd.DataFrame(summary).sort_values("seed")
    sdf.to_csv(OUTDIR / "summary.csv", index=False)
    print(sdf.to_string(index=False))

    # Summary verdict
    all_match = all(s["exact_hash_match"] for s in summary)
    max_p_diff = max(s["max_abs_pvalue_diff"] for s in summary)
    print(f"\nAll hashes match: {all_match}")
    print(f"Max p-value diff across all seeds: {max_p_diff:.2e}")
    if all_match:
        print("PASS: Serial and parallel produce identical results.")
    elif max_p_diff < 1e-6:
        print("PASS (approx): Tiny numerical differences only.")
    else:
        print("FAIL: Serial and parallel produce different results.")


if __name__ == "__main__":
    main()
