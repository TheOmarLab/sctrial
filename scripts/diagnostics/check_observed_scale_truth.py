#!/usr/bin/env python
"""Diagnostic 2: Check whether 'null' genes are truly null on the observed log-pseudobulk scale.

This is the most important diagnostic. If signal genes distort the observed-scale
estimand for null genes (e.g., via library-size normalization or compositional effects),
then FPR on 'null genes' in DE scenarios is meaningless.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sctrial.benchmark.simulator import SimulationConfig, simulate_trial


OUTDIR = Path("manuscript/benchmark/diagnostics/observed_scale_truth")
OUTDIR.mkdir(parents=True, exist_ok=True)


def observed_scale_did(
    pb_means: pd.DataFrame, gene_cols: list[str]
) -> pd.DataFrame:
    """Compute empirical participant-level DiD on log1p(pseudobulk means)."""
    df = pb_means.copy()
    for g in gene_cols:
        df[g] = np.log1p(df[g])

    pre = df[df["visit"] == "Pre"].set_index("participant")
    post = df[df["visit"] == "Post"].set_index("participant")
    common = pre.index.intersection(post.index)

    pre = pre.loc[common]
    post = post.loc[common]

    arm = pre["arm"]
    treated = arm == "Treated"
    control = arm == "Control"

    rows = []
    for g in gene_cols:
        delta = post[g] - pre[g]
        did = float(delta[treated].mean() - delta[control].mean())
        rows.append(
            {
                "gene": g,
                "observed_scale_did": did,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    # --- Pure null scenario ---
    print("=" * 60)
    print("SCENARIO A: Pure null (no DE genes)")
    print("=" * 60)
    cfg_null = SimulationConfig(
        design="two_arm",
        n_per_arm=20,
        n_genes=50,
        effects={},
        mean_cells_per_visit=500,
        baseline_mean=-12.86,
        baseline_sd=2.67,
        target_library_size=2981,
        library_size_sd=0.76,
        participant_sd=0.05,
        seed=0,
    )
    sim_null = simulate_trial(cfg_null)
    pb_null = sim_null["pseudobulk_means"]
    genes = [c for c in pb_null.columns if c.startswith("gene_")]
    truth_null = observed_scale_did(pb_null, genes)
    truth_null["latent_true_beta"] = 0.0
    truth_null.to_csv(OUTDIR / "null_scenario_observed_truth.csv", index=False)

    print(f"Max abs observed-scale DiD: {truth_null['observed_scale_did'].abs().max():.6f}")
    print(f"Mean abs observed-scale DiD: {truth_null['observed_scale_did'].abs().mean():.6f}")
    print()

    # --- DE scenario (10/50 genes with effect) ---
    print("=" * 60)
    print("SCENARIO B: DE scenario (10/50 genes with beta=0.5)")
    print("=" * 60)
    effects = {f"gene_{i}": 0.5 for i in range(10)}
    cfg_de = SimulationConfig(
        design="two_arm",
        n_per_arm=20,
        n_genes=50,
        effects=effects,
        mean_cells_per_visit=500,
        baseline_mean=-12.86,
        baseline_sd=2.67,
        target_library_size=2981,
        library_size_sd=0.76,
        participant_sd=0.05,
        seed=0,
    )
    sim_de = simulate_trial(cfg_de)
    pb_de = sim_de["pseudobulk_means"]
    truth_de = observed_scale_did(pb_de, genes)
    truth_de["latent_true_beta"] = [effects.get(g, 0.0) for g in genes]
    truth_de.to_csv(OUTDIR / "de_scenario_observed_truth.csv", index=False)

    # Split by latent truth
    latent_null = truth_de[truth_de["latent_true_beta"] == 0.0]
    latent_signal = truth_de[truth_de["latent_true_beta"] != 0.0]

    print("\nSignal genes (latent beta=0.5):")
    print(f"  Mean observed DiD: {latent_signal['observed_scale_did'].mean():.4f}")
    print(f"  Min observed DiD:  {latent_signal['observed_scale_did'].min():.4f}")
    print(f"  Max observed DiD:  {latent_signal['observed_scale_did'].max():.4f}")

    print("\nNull genes (latent beta=0.0):")
    print(f"  Mean observed DiD: {latent_null['observed_scale_did'].mean():.4f}")
    print(f"  Mean abs:          {latent_null['observed_scale_did'].abs().mean():.4f}")
    print(f"  Max abs:           {latent_null['observed_scale_did'].abs().max():.4f}")

    # Key test: are null genes truly near-zero on the observed scale?
    print(f"\n  Fraction with |DiD| < 0.01: {(latent_null['observed_scale_did'].abs() < 0.01).mean():.2f}")
    print(f"  Fraction with |DiD| < 0.05: {(latent_null['observed_scale_did'].abs() < 0.05).mean():.2f}")
    print(f"  Fraction with |DiD| < 0.10: {(latent_null['observed_scale_did'].abs() < 0.10).mean():.2f}")

    # --- Run across multiple seeds for stability ---
    print("\n" + "=" * 60)
    print("SCENARIO C: DE scenario across 20 seeds")
    print("=" * 60)
    null_dids_all = []
    for seed in range(20):
        cfg = SimulationConfig(
            design="two_arm",
            n_per_arm=20,
            n_genes=50,
            effects=effects,
            mean_cells_per_visit=500,
            baseline_mean=-12.86,
            baseline_sd=2.67,
            target_library_size=2981,
            library_size_sd=0.76,
            participant_sd=0.05,
            seed=seed,
        )
        sim = simulate_trial(cfg)
        pb = sim["pseudobulk_means"]
        truth = observed_scale_did(pb, genes)
        truth["latent_true_beta"] = [effects.get(g, 0.0) for g in genes]
        null_rows = truth[truth["latent_true_beta"] == 0.0]
        null_dids_all.extend(null_rows["observed_scale_did"].tolist())

    null_dids = np.array(null_dids_all)
    print(f"Total null gene observations: {len(null_dids)}")
    print(f"Mean observed DiD: {null_dids.mean():.6f}")
    print(f"Mean abs:          {np.abs(null_dids).mean():.4f}")
    print(f"Max abs:           {np.abs(null_dids).max():.4f}")
    print(f"SD:                {null_dids.std():.4f}")
    print(f"\nFraction with |DiD| < 0.01: {(np.abs(null_dids) < 0.01).mean():.2f}")
    print(f"Fraction with |DiD| < 0.05: {(np.abs(null_dids) < 0.05).mean():.2f}")
    print(f"Fraction with |DiD| < 0.10: {(np.abs(null_dids) < 0.10).mean():.2f}")

    if np.abs(null_dids).mean() < 0.05:
        print("\nVERDICT: Null genes are near-null on observed scale. Truth definition is valid.")
    else:
        print("\nWARNING: Null genes show substantial observed-scale DiD effects.")
        print("FPR calculations in DE scenarios may be misleading.")


if __name__ == "__main__":
    main()
