"""Benchmark orchestrator — runs the scenario grid on the transcriptome simulator.

Every method is given the input its estimand assumes (see
:mod:`sctrial.benchmark.contracts`) and is scored against that estimand, not
against one shared "true beta". Both of those were wrong in the previous version
and each produced a published conclusion that did not survive checking.

Usage::

    from sctrial.benchmark.orchestrator import run_benchmark
    results = run_benchmark(n_jobs=16, output_dir="benchmark_results")
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import METHOD_ESTIMAND, prepare_inputs
from .simulator_v2 import TranscriptomeSimConfig, make_signal, nested_panels, simulate_trial_v2

logger = logging.getLogger(__name__)

# --- adaptive Monte Carlo stopping -------------------------------------------
# At 200 replicates the Monte Carlo SE of a 5% false-positive rate is
# sqrt(.05*.95/200) = 0.015, so 5% and 7% are indistinguishable -- which is the
# comparison the paper turns on. Rather than pay for a blanket 1,000 everywhere,
# add replicates only where the estimate is still imprecise.
# `n_iterations` is the BASE batch; these govern the extension only.
_MC_BATCH = 100
_MC_MAX = 1000
_MCSE_TARGET_FPR = 0.005
_MCSE_TARGET_POWER = 0.01


def _replicate_rates(rows: list) -> tuple[np.ndarray, np.ndarray]:
    """Per-REPLICATE null-FPR and power.

    The independent unit is the simulated dataset, not the gene. Genes within one
    replicate share its participants, its library sizes and its random effects, so
    pooling them would understate the Monte Carlo error -- pseudoreplication
    inside a benchmark whose subject is pseudoreplication.
    """
    df = pd.DataFrame(rows)
    if df.empty or "iteration" not in df:
        return np.array([]), np.array([])
    ok = df["pvalue"].notna()
    null = df[ok & ~df["is_signal"]]
    sig = df[ok & df["is_signal"]]
    fpr = (
        null.assign(hit=null["pvalue"] < 0.05).groupby("iteration")["hit"].mean().to_numpy()
        if len(null) else np.array([])
    )
    pw = (
        sig.assign(hit=sig["pvalue"] < 0.05).groupby("iteration")["hit"].mean().to_numpy()
        if len(sig) else np.array([])
    )
    return fpr, pw


def _mcse(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.inf


def _needs_more_replicates(rows: list, n_done: int) -> tuple[bool, str]:
    """Whether this SCENARIO has reached its Monte Carlo precision target.

    Stopping is decided at the scenario level and applies to every method at once,
    so all methods always see exactly the same R simulated datasets. Stopping each
    method independently would break the pairing that makes the comparison
    powerful -- methods would then be compared on partly different data.

    Requires the criterion to hold for the WORST method: the run continues until
    every reported method is precise enough, or the cap is reached.
    """
    if n_done >= _MC_MAX:
        return False, f"at the {_MC_MAX}-replicate cap; report achieved precision"
    df = pd.DataFrame(rows)
    if df.empty or "method" not in df:
        return False, "no rows"
    worst_f, worst_p, need = 0.0, 0.0, False
    for method, grp in df.groupby("method", observed=True):
        fpr, pw = _replicate_rates(grp.to_dict("records"))
        se_f, se_p = _mcse(fpr), _mcse(pw)
        if len(fpr):
            worst_f = max(worst_f, se_f)
            need |= se_f > _MCSE_TARGET_FPR
        if len(pw):
            worst_p = max(worst_p, se_p)
            need |= se_p > _MCSE_TARGET_POWER
        del method
    return bool(need), f"worst-method MCSE fpr={worst_f:.4f} power={worst_p:.4f}"

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

_PANEL = 50
_SIGNAL_FRACTION = 0.20
_N_ITERATIONS = 200
_CELLS = 500

# Methods reported in the paper.
#
# ``limma_voom`` is the Gate G conventional pseudobulk comparator: a reviewer is
# entitled to ask how the standard tool behaves, and "we excluded it" is not an
# answer. It carries the repeated measure through ``duplicateCorrelation`` and is
# permitted to FAIL rather than silently fall back to an unpaired model; the
# failure rate is itself a reported result.
#
# ``edger_qlf`` stays excluded: edgeR-QL has no repeated-measures route for a
# two-arm design that is not rank-deficient, so including it would compare a
# different estimand under the same label.
CORE_METHODS = [
    "sctrial_did",
    "dreamlet",
    "nebula",
    "wilcoxon_paired",
    "limma_voom",
]

INTERNAL_METHODS = ["sctrial_mixed"]

# Primary signal architecture. ``one_directional`` is retained but named as a
# composition stress test: a coordinated shift moves the library-size reference,
# and about two thirds of the inflation previously attributed to empirical-Bayes
# moderation is that artifact.
_PRIMARY_ARCH = "balanced"


def scenario_seed(master_seed: int, scenario_name: str, replicate: int) -> int:
    """A seed addressed by (scenario, replicate index), not by draw order.

    With a single sequential RNG stream, adding a batch of replicates under
    adaptive stopping would renumber every later dataset, so replicate 437 would
    not be the same data across a resume, a different worker count, or a
    re-partitioned SLURM job. Hashing the address makes replicate 437 always
    replicate 437.

    hashlib rather than hash(): Python salts str hashing per process, so hash()
    is not reproducible across runs at all.
    """
    import hashlib

    key = f"{master_seed}|{scenario_name}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _scenario(
    name: str,
    description: str,
    *,
    design: str,
    panel_size: int = _PANEL,
    signal_fraction: float = 0.0,
    architecture: str = _PRIMARY_ARCH,
    magnitude: float = 0.5,
    arm_ratio: tuple[int, int] | None = None,
    **cfg,
) -> dict:
    # arm_ratio is emitted ALWAYS, including as None, so the scenario's design
    # always wins over anything inherited from the frozen calibration. An absent
    # key inherits; an explicit None overrides.
    cfg["arm_ratio"] = arm_ratio
    return {
        "name": name,
        "description": description,
        "panel_size": panel_size,
        "signal_fraction": signal_fraction,
        "architecture": architecture,
        "magnitude": magnitude,
        "config_kwargs": {"design": design, **cfg},
    }


def build_scenario_grid(design: str = "two_arm") -> list[dict]:
    """Core scenario grid for one design family."""
    s: list[dict] = []

    # 1. Complete null across sample size
    for n in [8, 12, 20, 40, 60]:
        s.append(
            _scenario(
                f"null_n{n}",
                f"Complete null, n={n} per arm",
                design=design,
                n_per_arm=n,
                cells_per_pv_fixed=_CELLS,
            )
        )

    # 2. Null with nuisance heterogeneity: larger participant variance and a wide
    #    cell-yield distribution (resampled rather than fixed).
    for n in [20, 40]:
        s.append(
            _scenario(
                f"null_hetero_n{n}",
                f"Null, high participant SD + imbalanced cell yield, n={n}",
                design=design,
                n_per_arm=n,
                between_participant_sd=1.5,
                # Explicit cell count, NOT a scale factor. `cells_scale=0.1`
                # multiplied whatever pool the anchor supplies, and when the
                # anchor became Treg (~301 cells/visit) that silently became ~30
                # cells -- at which dreamlet returns finite p-values for about 10%
                # of a 50-gene panel instead of 96%, so its Type I error would
                # have been read off ~5 genes and the collapse would have looked
                # like a property of dreamlet.
                cells_per_pv_fixed=150,
            )
        )

    # 3. Sparse DE, primary (balanced) architecture
    for n in [20, 40, 60]:
        for beta in [0.2, 0.5, 1.0]:
            s.append(
                _scenario(
                    f"de_balanced_n{n}_b{beta}",
                    f"Sparse DE, balanced signs, n={n}, beta={beta}",
                    design=design,
                    n_per_arm=n,
                    signal_fraction=_SIGNAL_FRACTION,
                    architecture="balanced",
                    magnitude=beta,
                    cells_per_pv_fixed=_CELLS,
                )
            )

    # 4. Heterogeneous effect magnitudes — the most realistic architecture
    for n in [20, 40]:
        for beta in [0.5, 1.0]:
            s.append(
                _scenario(
                    f"de_hetero_n{n}_b{beta}",
                    f"Sparse DE, heterogeneous magnitudes, n={n}, beta={beta}",
                    design=design,
                    n_per_arm=n,
                    signal_fraction=_SIGNAL_FRACTION,
                    architecture="heterogeneous",
                    magnitude=beta,
                    cells_per_pv_fixed=_CELLS,
                )
            )

    # 5. One-directional COMPOSITION STRESS. Named so no regex on the scenario
    #    name can pool it with the primary DE scenarios.
    for n in [20, 40]:
        s.append(
            _scenario(
                f"compstress_onedir_n{n}",
                f"Composition stress: all signal genes +0.5, n={n}",
                design=design,
                n_per_arm=n,
                signal_fraction=_SIGNAL_FRACTION,
                architecture="one_directional",
                cells_per_pv_fixed=_CELLS,
            )
        )

    # 6. Varying cell yield
    for n_cells in [200, 1000, 5000]:
        s.append(
            _scenario(
                f"cells_{n_cells}_n40",
                f"Cell yield {n_cells}/visit, n=40",
                design=design,
                n_per_arm=40,
                signal_fraction=_SIGNAL_FRACTION,
                cells_per_pv_fixed=n_cells,
            )
        )

    # 7. Unequal arms (two-arm only)
    if design == "two_arm":
        for ratio in [(3, 7), (5, 10), (10, 20)]:
            s.append(
                _scenario(
                    f"imbal_{ratio[0]}v{ratio[1]}",
                    f"Unequal arms {ratio[0]}:{ratio[1]}",
                    design=design,
                    n_per_arm=sum(ratio),
                    arm_ratio=ratio,  # the only scenarios that set it
                    signal_fraction=_SIGNAL_FRACTION,
                    cells_per_pv_fixed=_CELLS,
                )
            )

    # 8. Missing post visits
    for rate in [0.1, 0.2]:
        s.append(
            _scenario(
                f"missing_{int(rate * 100)}pct_n40",
                f"{int(rate * 100)}% of Post visits missing, n=40",
                design=design,
                n_per_arm=40,
                missing_rate=rate,
                signal_fraction=_SIGNAL_FRACTION,
                cells_per_pv_fixed=_CELLS,
            )
        )

    return s


def build_sensitivity_grid(design: str = "two_arm", panels=None) -> list[dict]:
    """Panel size x signal fraction sensitivity.

    Panels are NESTED subsets of one simulated transcriptome, so a panel-size
    effect is separable from a gene-identity effect. With independently drawn
    panels the two are confounded, and "progressive miscalibration with panel
    size" could not be distinguished from a change in which genes were tested.

    Signal counts are reported as REALISED fractions rather than nominal labels:
    at 50 genes ``round(50 * 0.01)`` is 1 gene, i.e. 2%, and the previous loaders
    parsed the scenario name rather than the data, manufacturing an apparent
    panel-size dependence.
    """
    s: list[dict] = []
    n_per_arm = 40
    # Panel sizes are selectable so the 2000-gene cells can run as their own job:
    # one 2000-gene iteration measured 495 s against 44 s at 50 genes, so mixing
    # them puts the whole grid at the mercy of the slowest cells.
    for panel in list(panels) if panels else [50, 200, 500, 2000]:
        s.append(
            _scenario(
                f"sens_null_g{panel}",
                f"Sensitivity null, {panel}-gene panel, n={n_per_arm}",
                design=design,
                n_per_arm=n_per_arm,
                panel_size=panel,
                signal_fraction=0.0,
                cells_per_pv_fixed=_CELLS,
            )
        )
        for frac in [0.01, 0.05, 0.10, 0.20]:
            realised = max(1, int(round(panel * frac))) / panel
            for arch in ("balanced", "one_directional"):
                tag = "" if arch == "balanced" else "_onedir"
                s.append(
                    _scenario(
                        f"sens_g{panel}_f{int(frac * 100)}{tag}",
                        f"{panel} genes, nominal {frac:.0%} / realised {realised:.1%} "
                        f"signal, {arch}",
                        design=design,
                        n_per_arm=n_per_arm,
                        panel_size=panel,
                        signal_fraction=frac,
                        architecture=arch,
                        cells_per_pv_fixed=_CELLS,
                    )
                )
    return s


# ---------------------------------------------------------------------------
# Single-iteration worker
# ---------------------------------------------------------------------------


def _run_single_iteration(args: tuple) -> list[dict]:
    """Run every method on one simulated dataset. Executed in worker processes."""
    import warnings

    warnings.filterwarnings("ignore")

    scenario_name, iteration, seed, scenario, methods, base_config = args

    # The frozen calibration is the FLOOR; a scenario may only override the knobs
    # it is explicitly varying. Building the config from scenario kwargs alone is
    # how the previous benchmark ran on dataclass defaults (2.3e7 UMIs per cell)
    # while the Methods described calibrated parameters -- the calibration existed,
    # it was simply never threaded through.
    kw = dict(base_config or {})
    kw.update(scenario["config_kwargs"])

    # A single-arm design tests Delta versus 0, so a common time effect is NOT
    # removed by the contrast the way it is in a two-arm DiD. Forcing it to zero
    # keeps a "null" scenario actually null.
    if kw.get("design") == "single_arm":
        kw.setdefault("time_effect", 0.0)

    panel_size = scenario["panel_size"]
    probe = TranscriptomeSimConfig(seed=seed, **kw)

    # Draw the panel first so the signal is defined on the tested genes, then
    # build the config with those effects. The panel is a nested subset of the
    # transcriptome and is reproducible from the seed alone.
    panels = nested_panels(probe, rng=np.random.default_rng(seed + 1))
    if panel_size not in panels:
        raise ValueError(f"panel size {panel_size} is not one of {sorted(panels)}")
    panel_genes = [f"gene_{i}" for i in panels[panel_size]]

    effects = (
        make_signal(
            panel_genes,
            scenario["signal_fraction"],
            scenario["architecture"],
            scenario["magnitude"],
            rng=np.random.default_rng(seed + 2),
        )
        if scenario["signal_fraction"] > 0
        else {}
    )
    cfg = TranscriptomeSimConfig(seed=seed, effects=effects, **kw)

    sim = simulate_trial_v2(cfg)

    # The design that was actually SIMULATED. Asserted against the request so a
    # config leak cannot silently flatten the sample-size axis again.
    realised_arms = list(sim["latent"]["arms"])
    n_treated = sum(a == "Treated" for a in realised_arms)
    n_control = len(realised_arms) - n_treated
    if cfg.arm_ratio is None and cfg.design == "two_arm":
        expected = 2 * cfg.n_per_arm
        if len(realised_arms) != expected:
            raise RuntimeError(
                f"{scenario_name}: requested n_per_arm={cfg.n_per_arm} "
                f"({expected} participants) but simulated {len(realised_arms)}. "
                "A design field has leaked in from the frozen calibration."
            )

    inputs = prepare_inputs(sim, panel_genes)
    oracle = inputs["oracle"]
    design_type = cfg.design
    signal_genes = set(effects)

    rows = []
    for method in methods:
        t0 = time.time()
        try:
            results = _dispatch_method(method, inputs, design_type=design_type)
        except Exception as exc:
            logger.warning(
                "Method %s failed on %s iter %d: %s", method, scenario_name, iteration, exc
            )
            results = {
                g: {
                    "beta": np.nan,
                    "pvalue": np.nan,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "converged": False,
                    "failure_mode": "numerical",
                }
                for g in panel_genes
            }
        elapsed = time.time() - t0

        estimand = METHOD_ESTIMAND.get(method, "count_link")
        truth_table = oracle.get(estimand, {})
        for gene in panel_genes:
            r = results.get(gene, {})
            injected = effects.get(gene, 0.0)
            rows.append(
                {
                    "scenario": scenario_name,
                    "iteration": iteration,
                    "method": method,
                    "gene": gene,
                    # The value THIS method's estimand implies. Scoring every
                    # method against `injected_beta` compares different
                    # functionals and penalises whichever differs most from it.
                    "true_beta": float(truth_table.get(gene, injected)),
                    "injected_beta": float(injected),
                    "estimand": estimand,
                    "is_signal": gene in signal_genes,
                    "estimated_beta": r.get("beta", np.nan),
                    "pvalue": r.get("pvalue", np.nan),
                    "ci_lo": r.get("ci_lo", np.nan),
                    "ci_hi": r.get("ci_hi", np.nan),
                    "converged": r.get("converged", False),
                    # Every method receives the SAME panel. If a method's own
                    # internal filtering then drops a gene, that is reported here
                    # as an evaluability rate rather than silently shrinking the
                    # denominator of its calibration and power estimates.
                    "evaluable": bool(np.isfinite(r.get("pvalue", np.nan))),
                    "failure_mode": r.get("failure_mode", "numerical"),
                    # Wall time for the WHOLE iteration (all genes), matching the
                    # figure axis "Median runtime per iteration (s)". Storing
                    # elapsed/n_genes while plotting it as per-iteration
                    # understated the cost by the panel size and manufactured the
                    # "flat runtime scaling" claim.
                    "runtime_seconds": elapsed,
                    "runtime_scope": "per_iteration",
                    # REALISED, not requested. The two diverged once already.
                    "n_participants": len(realised_arms),
                    "n_treated": n_treated,
                    "n_control": n_control,
                    "n_per_arm": cfg.n_per_arm,
                    "panel_size": panel_size,
                    "n_signal_realised": len(signal_genes),
                    "signal_fraction_realised": len(signal_genes) / max(panel_size, 1),
                    "architecture": scenario["architecture"],
                }
            )

    return rows


def _dispatch_method(method: str, inputs: dict, design_type: str = "two_arm") -> dict:
    """Route to a runner with its CONTRACTED input representation.

    See :mod:`sctrial.benchmark.contracts` for the contract table. Nothing is
    inferred here: an unknown method raises rather than getting a guessed
    representation.
    """
    panel = inputs["panel_genes"]

    if method in ("sctrial_did", "sctrial_mixed"):
        from .runners import sctrial_did

        return sctrial_did.run(
            inputs["participant_log1p_cpm"],
            panel,
            from_pseudobulk=True,
            design_type=design_type,
        )
    if method == "wilcoxon_paired":
        from .runners import wilcoxon_paired

        return wilcoxon_paired.run(
            inputs["participant_log1p_cpm"], panel, design_type=design_type
        )
    if method == "dreamlet":
        from .runners import dreamlet_runner

        return dreamlet_runner.run(
            inputs["pseudobulk_counts"],
            panel,
            design_type=design_type,
            lib_size=inputs["lib_size"],
        )
    if method == "limma_voom":
        from .runners import limma_voom

        return limma_voom.run(
            inputs["pseudobulk_counts"],
            panel,
            design_type=design_type,
            lib_size=inputs["lib_size"],
        )
    if method == "edger_qlf":
        from .runners import edger_qlf

        return edger_qlf.run(
            inputs["pseudobulk_counts"],
            panel,
            design_type=design_type,
            lib_size=inputs["lib_size"],
        )
    if method == "nebula":
        from .runners import nebula_runner

        return nebula_runner.run(
            inputs["cell_counts"],
            panel,
            design_type=design_type,
            lib_size=inputs["cell_lib_size"],
        )
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Grid driver
# ---------------------------------------------------------------------------


def _run_grid(
    grid_fn,
    designs: list[str],
    methods: list[str],
    n_iterations: int,
    n_jobs: int,
    output_dir: Path,
    resume: bool,
    combined_name: str,
    seed: int,
    base_config: dict | None = None,
    manifest: dict | None = None,
    adaptive: bool = True,
) -> pd.DataFrame:
    """One driver for both grids.

    The core and sensitivity drivers were previously near-identical copies, and a
    resume defect fixed in one was not fixed in the other.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # No sequential RNG: every replicate seed is addressed by
    # (scenario, index) via scenario_seed, so adaptive extension and resume
    # cannot renumber datasets.
    all_results = []

    for design in designs:
        scenarios = grid_fn(design)
        print(f"\n{'=' * 60}")
        print(f"Design: {design} — {len(scenarios)} scenarios x {n_iterations} iterations")
        print(f"Methods: {methods}")
        print(f"Parallel workers: {n_jobs}")
        print(f"{'=' * 60}")

        for si, scenario in enumerate(scenarios):
            name = f"{design}__{scenario['name']}"
            csv_path = output_dir / f"{name}.csv"
            # Addressed by (scenario, index) so adaptive extension and resume
            # cannot renumber datasets. `rng` is retained only for reproducible
            # scenario ordering.
            def _seed(i: int, _n=name) -> int:
                return scenario_seed(seed, _n, i)

            if resume and csv_path.exists():
                existing = pd.read_csv(csv_path)
                # A partial file is worse than no file: it silently shrinks the
                # Monte Carlo sample for one cell of the grid while every summary
                # reports the nominal iteration count.
                n_done = existing["iteration"].nunique() if "iteration" in existing else 0
                # >= not ==: adaptive stopping may have run PAST the base batch,
                # and a complete deeper run must not be treated as partial.
                if n_done >= n_iterations:
                    print(f"  [{si + 1}/{len(scenarios)}] {name} — CACHED, skipping")
                    all_results.append(existing)
                    continue
                print(
                    f"  [{si + 1}/{len(scenarios)}] {name} — PARTIAL "
                    f"({n_done}/{n_iterations} iterations), re-running"
                )

            print(f"  [{si + 1}/{len(scenarios)}] {name}: {scenario['description']}")

            t0 = time.time()
            all_rows: list = []
            done = 0
            target = n_iterations

            def _flush(_rows=None, _path=csv_path) -> None:
                pd.DataFrame(_rows if _rows is not None else all_rows).to_csv(
                    _path, index=False
                )

            # ADAPTIVE MONTE CARLO. Run the base batch, then extend only while the
            # replicate-level Monte Carlo error is above target. Scenarios with an
            # obvious answer stop early; near-nominal ones, where the paper's
            # claims actually live, get more effort.
            while done < target:
                batch = [
                    (name, it, _seed(it), scenario, methods, base_config)
                    for it in range(done, target)
                ]
                if not batch:
                    break
                if n_jobs == 1:
                    for a in batch:
                        all_rows.extend(_run_single_iteration(a))
                else:
                    # 'spawn', never 'fork': a forked worker inherits R/BLAS state
                    # and produced demonstrably wrong results (edgeR FPR 0.002
                    # in-pipeline versus 0.05 standalone).
                    ctx = mp.get_context("spawn")
                    with ctx.Pool(n_jobs) as pool:
                        for k, out in enumerate(pool.imap(_run_single_iteration, batch)):
                            all_rows.extend(out)
                            if (k + 1) % 20 == 0:
                                elapsed = time.time() - t0
                                print(
                                    f"    {done + k + 1}/{target} iterations "
                                    f"({elapsed:.0f}s elapsed)", flush=True
                                )
                                _flush()
                done = target
                _flush()

                if not adaptive:
                    break
                more, why = _needs_more_replicates(all_rows, done)
                if not more:
                    print(f"    stopping at {done} replicates ({why})", flush=True)
                    break
                extra = min(_MC_BATCH, _MC_MAX - done)
                if extra <= 0:
                    break
                print(f"    extending +{extra} replicates ({why})", flush=True)
                target = done + extra

            df = pd.DataFrame(all_rows)
            # Stamp provenance on EVERY row. Figure loaders refuse to combine
            # rows from different manifests, which is what stops a corrected run
            # being silently averaged with the run it replaced.
            if manifest is not None:
                df["manifest_sha256"] = manifest["manifest_sha256"]
                df["git_sha"] = manifest.get("git_sha", "unknown")
            df.to_csv(csv_path, index=False)
            all_results.append(df)
            print(f"    Done in {time.time() - t0:.0f}s -> {csv_path.name}")

    # REBUILD FROM DISK, not from this invocation's results.
    #
    # The definitive run is deliberately split across four SLURM jobs by design
    # family and panel size, all writing into one directory under one combined
    # name. Concatenating only `all_results` meant the last job to finish
    # overwrote the others, so the combined file -- the ONLY file the figures
    # read -- would have held one design family, or one panel size, with no
    # error. The pure-null calibration panel would have been drawn from a quarter
    # of the grid and looked complete.
    combined_path = output_dir / combined_name
    parts = []
    for f in sorted(output_dir.glob("*.csv")):
        if f.name == combined_name:
            continue
        try:
            parts.append(pd.read_csv(f, low_memory=False))
        except Exception as exc:  # pragma: no cover - unreadable partial file
            raise RuntimeError(f"cannot read {f}: {exc}") from exc
    if not parts:
        raise RuntimeError(f"no per-scenario results found in {output_dir}")
    combined = pd.concat(parts, ignore_index=True)

    # Atomic replace: two jobs finishing together would otherwise interleave two
    # non-atomic writes into a corrupt file.
    tmp = combined_path.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False)
    os.replace(tmp, combined_path)

    print(f"\nCombined {len(parts)} scenario files -> {combined_path}")
    print(f"Total rows: {len(combined):,}")
    if "scenario" in combined:
        print(f"Scenarios present: {combined['scenario'].nunique()}")
    return combined


def run_benchmark(
    designs: list[str] | None = None,
    methods: list[str] | None = None,
    n_iterations: int = _N_ITERATIONS,
    n_jobs: int = 1,
    output_dir: str | Path = "benchmark_results",
    resume: bool = True,
    base_config: dict | None = None,
    manifest: dict | None = None,
) -> pd.DataFrame:
    """Run the core scenario grid."""
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    return _run_grid(
        build_scenario_grid,
        designs or ["two_arm", "single_arm"],
        methods or CORE_METHODS,
        n_iterations,
        n_jobs,
        Path(output_dir),
        resume,
        "benchmark_combined.csv",
        seed=2024,
        base_config=base_config,
        manifest=manifest,
    )


def run_sensitivity_benchmark(
    designs: list[str] | None = None,
    methods: list[str] | None = None,
    n_iterations: int = _N_ITERATIONS,
    n_jobs: int = 1,
    output_dir: str | Path = "benchmark_results/sensitivity",
    resume: bool = True,
    base_config: dict | None = None,
    manifest: dict | None = None,
    panels: list[int] | None = None,
) -> pd.DataFrame:
    """Run the panel-size x signal-fraction sensitivity grid."""
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    grid_fn = (
        (lambda design: build_sensitivity_grid(design, panels=panels))
        if panels
        else build_sensitivity_grid
    )
    return _run_grid(
        grid_fn,
        designs or ["two_arm"],
        methods or CORE_METHODS,
        n_iterations,
        n_jobs,
        Path(output_dir),
        resume,
        "sensitivity_combined.csv",
        seed=90210,
        base_config=base_config,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NatMeth benchmark: simulation grid")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers (-1 = all cores)")
    parser.add_argument("--n-iterations", type=int, default=_N_ITERATIONS)
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    parser.add_argument("--designs", nargs="+", default=["two_arm", "single_arm"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--sensitivity", action="store_true", help="run the sensitivity grid")
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    fn = run_sensitivity_benchmark if args.sensitivity else run_benchmark
    fn(
        designs=args.designs,
        methods=args.methods,
        n_iterations=args.n_iterations,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
