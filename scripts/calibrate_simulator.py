#!/usr/bin/env python
"""Canonical calibration and validation entry point for the benchmark simulator.

This is the ONLY supported way to measure simulator targets or run the
calibration gates. It replaces a set of ad-hoc scripts that had drifted into
mutually inconsistent versions; if a number in the manuscript describes the
simulator, it came from here.

    # 1. measure every target from the real data (writes the canonical files)
    python scripts/calibrate_simulator.py targets --dataset tnbc

    # 2. Monte Carlo envelope gates A/B/C/E
    python scripts/calibrate_simulator.py gates --n-mc 200 --n-jobs 16

    # 3. Gate D: normalisation-scope ablation
    python scripts/calibrate_simulator.py ablate --n-rep 5

    # 4. freeze the configuration that the definitive benchmark will use
    python scripts/calibrate_simulator.py freeze

Never run any phase on an HPC login node. Submit with sbatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

VALIDATION_DIR = REPO.parent.parent / "manuscript" / "benchmark" / "validation"


def _manuscript_dir() -> Path:
    import os

    env = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
    if env:
        return Path(env) / "benchmark" / "validation"
    return VALIDATION_DIR


def _load_dataset(name: str):
    from sctrial import datasets

    loaders = {
        "tnbc": datasets.load_tnbc_zhang,
        "vaccine": datasets.load_vaccine,
        "aml": datasets.load_aml,
    }
    if name not in loaders:
        raise SystemExit(f"unknown dataset {name!r}; choose from {sorted(loaders)}")
    return loaders[name]()


def cmd_targets(args) -> None:
    from sctrial.benchmark.calibration import measure_targets

    out = _manuscript_dir()
    adata = _load_dataset(args.dataset)
    print(f"{args.dataset}: {adata.n_obs:,} cells x {adata.n_vars:,} genes", flush=True)
    stats = measure_targets(
        adata,
        participant_col=args.participant_col,
        visit_col=args.visit_col,
        arm_col=args.arm_col,
        celltype_col=args.celltype_col,
        layer=args.layer,
        out_json=out / f"{args.dataset}_sim_targets.json",
        out_npz=out / f"{args.dataset}_empirical.npz",
    )
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith("_")}, indent=2))


def _config_from_targets(args):
    from sctrial.benchmark.simulator_v2 import TranscriptomeSimConfig

    out = _manuscript_dir()
    with open(out / f"{args.dataset}_sim_targets.json") as fh:
        t = json.load(fh)
    cfg = TranscriptomeSimConfig(
        n_per_arm=t.get("n_participants", 12) // 2,
        n_genes_transcriptome=t["n_genes_transcriptome"],
        cells_per_pv_mean=t["cells_per_pv_mean"],
        cells_per_pv_cv=t["cells_per_pv_cv"],
        cells_per_pv_min=t["cells_per_pv_min"],
        cells_per_pv_max=t["cells_per_pv_max"],
        cells_scale=args.cells_scale,
        empirical_path=str(out / f"{args.dataset}_empirical.npz"),
        lib_log_mean=t["lib_log_mean"],
        lib_log_sd=t["lib_log_sd"],
        gene_rate_log_mean=t["gene_mean_log_mean"],
        gene_rate_log_sd=t["gene_mean_log_sd"],
        dispersion_median=t["dispersion_median"],
        dispersion_mean_slope=t["dispersion_mean_slope"],
        dispersion_log_sd=t["dispersion_log_sd"],
        # LATENT variance components, not the observable correlation. The
        # observable is attenuated by pseudobulk sampling noise (sigma_e), so
        # using it as the generating parameter under-disperses the hierarchy --
        # the same conditional-versus-marginal error as calibrating dispersion on
        # the pooled curve. The observables are what the gates test.
        between_participant_sd=t["between_participant_sd_latent"],
        prepost_corr=t["prepost_corr_latent"],
    )
    return cfg, t


def cmd_gates(args) -> None:
    from sctrial.benchmark.gates import run_gates

    cfg, targets = _config_from_targets(args)
    out = _manuscript_dir()
    df = run_gates(
        cfg,
        observed=targets,
        n_mc=args.n_mc,
        n_jobs=args.n_jobs,
        out_dir=out / "gates",
    )
    import pandas as pd

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))
    n_fail = int((df["verdict"] == "FAIL").sum())
    print(f"\n{len(df)} statistics: {(df['verdict'] == 'PASS').sum()} PASS, "
          f"{(df['verdict'] == 'MARGINAL').sum()} MARGINAL, {n_fail} FAIL")
    if n_fail:
        print("FAILING:", df.loc[df["verdict"] == "FAIL", "statistic"].tolist())


def cmd_ablate(args) -> None:
    from sctrial.benchmark.gates import composition_ablation

    cfg, _ = _config_from_targets(args)
    out = _manuscript_dir() / "gates"
    out.mkdir(parents=True, exist_ok=True)
    df = composition_ablation(cfg, n_rep=args.n_rep, panel_size=args.panel_size)
    df.to_csv(out / "gate_d_composition_ablation.csv", index=False)
    agg = df.groupby(["architecture", "scope"])[
        ["mean_estimate", "mean_oracle_log1p_cpm", "bias_vs_oracle"]
    ].mean()
    print(agg.to_string())
    print(f"\nwrote {out / 'gate_d_composition_ablation.csv'}")


def cmd_freeze(args) -> None:
    """Write the frozen configuration the definitive benchmark run must use."""
    cfg, targets = _config_from_targets(args)
    out = _manuscript_dir()
    gate_summary = out / "gates" / "gate_summary.json"
    if not gate_summary.exists():
        raise SystemExit(
            "refusing to freeze: no gate results at "
            f"{gate_summary}. Run `gates` first — freezing an unvalidated "
            "configuration is how the previous benchmark shipped uncalibrated."
        )
    with open(gate_summary) as fh:
        summary = json.load(fh)
    if summary.get("n_fail", 1) > 0 and not args.force:
        raise SystemExit(
            f"refusing to freeze: {summary['n_fail']} gate(s) FAIL "
            f"({summary.get('failures')}). Fix the calibration, or pass --force "
            "and record the justification in tasks/MASTER_PLAN.md."
        )
    frozen = {
        "config": asdict(cfg),
        "targets_source": str(out / f"{args.dataset}_sim_targets.json"),
        "gate_summary": summary,
        "dataset": args.dataset,
    }
    path = out / "frozen_simulator_config.json"
    with open(path, "w") as fh:
        json.dump(frozen, fh, indent=2, default=str)
    print(f"froze configuration -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="tnbc")
    p.add_argument("--participant-col", default="participant")
    p.add_argument("--visit-col", default="visit")
    p.add_argument("--arm-col", default="arm")
    p.add_argument("--celltype-col", default="cell_type")
    p.add_argument("--layer", default=None)
    p.add_argument("--cells-scale", type=float, default=1.0)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("targets", help="measure simulator targets from real data").set_defaults(
        func=cmd_targets
    )

    g = sub.add_parser("gates", help="Monte Carlo envelope gates A/B/C/E")
    g.add_argument("--n-mc", type=int, default=200)
    g.add_argument("--n-jobs", type=int, default=8)
    g.set_defaults(func=cmd_gates)

    a = sub.add_parser("ablate", help="Gate D normalisation-scope ablation")
    a.add_argument("--n-rep", type=int, default=5)
    a.add_argument("--panel-size", type=int, default=200)
    a.set_defaults(func=cmd_ablate)

    f = sub.add_parser("freeze", help="freeze the validated configuration")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_freeze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
