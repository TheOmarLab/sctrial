"""Phase-0 sensitivity check for the NEBULA cell-inclusion filter.

The frozen benchmark filtered NEBULA cells with
``colSums(counts) > 0 & meta$lib_size > 0`` (``cell_filter="panel_and_lib"``).
The ``colSums(counts) > 0`` term drops cells with no reads in the TESTED panel,
which is outcome- and panel-size-dependent. The correct filter is
``meta$lib_size > 0`` only (``cell_filter="lib_only"``).

This runs BOTH filters on the SAME seeded simulated data, through the canonical
path (simulate_trial_v2 -> contracts.prepare_inputs -> nebula_runner.run), and
reports how much the NEBULA null FPR, mean |beta| and retained-cell count move.
It does NOT modify the frozen runner behaviour: the default filter is unchanged,
so the frozen v1.0.0 results remain reproducible; this only measures the delta.

Run on HPC via sbatch (NEBULA at 2000 genes is minutes per fit).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "manuscript" / "benchmark" / "validation" / "nebula_cell_filter"

# Representative small two-arm trial; the filter effect does not depend on the
# exact size, but these values reproduce the benchmark's grossly-inflated NEBULA
# null FPR so the corrected number is comparable.
N_PER_ARM = 6
CELLS_PER_PV = 500
N_REP = 20
ALPHA = 0.05
SIGNAL_FRAC = 0.10  # for the mixed-signal scenarios

# (label, panel_size, architecture, n_signal). n_signal 0 => pure null.
SCENARIOS = [
    ("pure_null_g50", 50, "balanced", 0),
    ("pure_null_g2000", 2000, "balanced", 0),
    ("mixed_balanced_g200", 200, "balanced", None),
    ("mixed_onedir_g200", 200, "one_directional", None),
]

FILTERS = ["panel_and_lib", "lib_only"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _fpr_and_beta(res: dict, null_genes: list[str]):
    pvals, betas = [], []
    for g in null_genes:
        r = res.get(g)
        if r is None:
            continue
        p, b = r.get("pvalue"), r.get("beta")
        if p is not None and np.isfinite(p):
            pvals.append(float(p))
        if b is not None and np.isfinite(b):
            betas.append(abs(float(b)))
    fpr = float(np.mean(np.asarray(pvals) < ALPHA)) if pvals else np.nan
    mean_abs_beta = float(np.mean(betas)) if betas else np.nan
    return fpr, mean_abs_beta, len(pvals)


def main() -> None:
    # Reuse the ONE canonical frozen-config loader (manifest-verified) instead of a
    # second copy, so this diagnostic sees exactly the population the benchmark froze.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import run_benchmark

    from sctrial.benchmark.contracts import prepare_inputs
    from sctrial.benchmark.runners import nebula_runner
    from sctrial.benchmark.simulator_v2 import (
        TranscriptomeSimConfig,
        make_signal,
        nested_panels,
        simulate_trial_v2,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = run_benchmark._load_frozen_config()
    log_lines: list[str] = []

    def emit(s=""):
        print(s, flush=True)
        log_lines.append(s)

    emit("=" * 78)
    emit("NEBULA cell-filter sensitivity: panel_and_lib (frozen) vs lib_only (fix)")
    emit("=" * 78)
    emit(f"  n_per_arm={N_PER_ARM}  cells_per_pv={CELLS_PER_PV}  n_rep={N_REP}  alpha={ALPHA}")
    emit(f"  git={_git_sha()[:12]}")

    summary = []
    for label, panel_size, arch, n_sig_spec in SCENARIOS:
        emit("")
        emit(f"--- {label}: panel={panel_size} arch={arch} ---")
        acc = {cf: {"fpr": [], "beta": [], "ncell": []} for cf in FILTERS}
        for rep in range(N_REP):
            seed = 20_260_729 + rep
            kw = dict(frozen)
            kw.update(n_per_arm=N_PER_ARM, cells_per_pv_fixed=CELLS_PER_PV, seed=seed)
            probe = TranscriptomeSimConfig(**kw)
            panels = nested_panels(probe, rng=np.random.default_rng(seed + 1))
            panel = [f"gene_{i}" for i in panels[panel_size]]
            n_signal = 0 if n_sig_spec == 0 else round(SIGNAL_FRAC * len(panel))
            effects = (
                make_signal(panel, n_signal, arch, 0.5, rng=np.random.default_rng(seed + 2))
                if n_signal > 0
                else {}
            )
            sim = simulate_trial_v2(TranscriptomeSimConfig(effects=effects, **kw))
            inputs = prepare_inputs(sim, panel)
            null_genes = [g for g in inputs["panel_genes"] if g not in effects]
            for cf in FILTERS:
                res = nebula_runner.run(
                    inputs["cell_counts"], panel, design_type="two_arm",
                    lib_size=inputs["cell_lib_size"], cell_filter=cf,
                )
                fpr, mab, ncell = _fpr_and_beta(res, null_genes)
                acc[cf]["fpr"].append(fpr)
                acc[cf]["beta"].append(mab)
                acc[cf]["ncell"].append(ncell)

        def _ms(v):
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return np.nan, np.nan
            return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0

        emit(f"  {'filter':<16}{'null-FPR':>12}{'MCSE':>9}{'mean|beta|':>12}{'n_null_p':>10}")
        row = {"scenario": label, "panel_size": panel_size, "architecture": arch}
        for cf in FILTERS:
            fpr_m, fpr_se = _ms(acc[cf]["fpr"])
            beta_m, _ = _ms(acc[cf]["beta"])
            ncell_m, _ = _ms(acc[cf]["ncell"])
            emit(f"  {cf:<16}{fpr_m:>12.4f}{fpr_se:>9.4f}{beta_m:>12.4f}{ncell_m:>10.0f}")
            row[f"{cf}_fpr"] = fpr_m
            row[f"{cf}_fpr_mcse"] = fpr_se
            row[f"{cf}_mean_abs_beta"] = beta_m
        d = row["lib_only_fpr"] - row["panel_and_lib_fpr"]
        row["delta_fpr"] = d
        emit(f"  delta FPR (lib_only - panel_and_lib): {d:+.4f}")
        summary.append(row)

    emit("")
    emit("=" * 78)
    emit("VERDICT (computed): NEBULA null FPR stays far above alpha under both filters")
    emit("means the inflation is not the filter; a large per-panel delta means the")
    emit("frozen NEBULA numbers should be regenerated with lib_only (Phase 1).")

    (OUT_DIR / "nebula_cell_filter.log").write_text("\n".join(log_lines) + "\n")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "provenance.json").write_text(json.dumps({
        "tag": "nebula-cell-filter-phase0",
        "git_sha": _git_sha(),
        "n_per_arm": N_PER_ARM, "cells_per_pv": CELLS_PER_PV, "n_rep": N_REP,
        "created_unix": int(time.time()),
    }, indent=2))
    emit(f"\nwrote {OUT_DIR}/nebula_cell_filter.log, summary.json, provenance.json")


if __name__ == "__main__":
    main()
