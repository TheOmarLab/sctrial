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

def _manuscript_dir() -> Path:
    """Resolve the manuscript tree without guessing a parent depth.

    A hardcoded ``parents[N]`` is what silently resolved outside the project on
    HPC and blanked four figure panels while every script exited 0. The layouts
    genuinely differ -- ``manuscript/`` sits inside the project root on the
    cluster and beside the repo locally -- so this checks rather than assumes,
    and fails loudly when neither exists.
    """
    import os

    env = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
    if env:
        return Path(env) / "benchmark" / "validation"
    for base in (REPO / "manuscript", REPO.parent.parent / "manuscript"):
        if base.is_dir():
            return base / "benchmark" / "validation"
    raise SystemExit(
        "Cannot locate the manuscript tree. Checked "
        f"{REPO / 'manuscript'} and {REPO.parent.parent / 'manuscript'}. "
        "Set SCTRIAL_MANUSCRIPT_DIR explicitly rather than letting a path guess "
        "resolve somewhere unintended."
    )


def _load_dataset(name: str):
    from sctrial import datasets

    # Names must match `sctrial.datasets` exactly. Resolved by getattr with an
    # explicit check rather than by attribute access at import time, so a rename
    # fails here with a clear message instead of at the top of a 20-hour job.
    loaders = {
        "tnbc": "load_tnbc_zhang",
        "vaccine": "load_vaccine_gse171964",
        "aml": "load_aml",
        "melanoma": "load_sade_feldman",
        "covid": "load_stephenson_data",
        "cart": "load_cart",
    }
    if name not in loaders:
        raise SystemExit(f"unknown dataset {name!r}; choose from {sorted(loaders)}")
    fn = getattr(datasets, loaders[name], None)
    if fn is None:
        raise SystemExit(
            f"sctrial.datasets has no {loaders[name]!r}; the loader was renamed and "
            "this table was not updated"
        )
    return fn()




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
    # Match the anchor population's ACTUAL retained design, not the nominal one.
    # In TNBC one participant contributes no Treg cells at Post, so the anchor has
    # 11 fully paired participants split 6/5 by arm, not 12 split 6/6. Every
    # longitudinal target was estimated on those 11 pairs, so simulating a
    # balanced 6/6 would compare a balanced simulation against an unbalanced
    # measurement.
    _by_arm = t.get("paired_participants_by_arm") or {}
    _n_paired = int(t.get("n_participants_paired", t.get("n_participants", 12)))
    if len(_by_arm) == 2:
        _a, _b = (int(v) for v in sorted(_by_arm.values(), reverse=True))
        _arm_ratio = (_a, _b)
        _n_per_arm = _a + _b
    else:
        _arm_ratio = None
        _n_per_arm = max(_n_paired // 2, 1)
    cfg = TranscriptomeSimConfig(
        n_per_arm=_n_per_arm,
        arm_ratio=_arm_ratio,
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
        # Anchor and RESIDUAL sd travel with the median. They are only used when
        # the empirical per-gene dispersion pool is unavailable; the default path
        # resamples that pool paired with the gene rate.
        dispersion_median=t["dispersion_median"],
        dispersion_mean_slope=t["dispersion_mean_slope"],
        dispersion_anchor=t["dispersion_anchor"],
        dispersion_residual_sd=t["dispersion_residual_sd"],
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
    import pandas as pd

    from sctrial.benchmark.calibration import (
        participant_bootstrap_statistics,
        summarize_adata_per_celltype,
    )
    from sctrial.benchmark.gates import run_gates

    cfg, targets = _config_from_targets(args)
    out = _manuscript_dir()

    boot = None
    if args.n_boot > 0:
        # The acceptance tolerance is the REFERENCE COHORT's own sampling
        # uncertainty, obtained by resampling participants (both visits, all cell
        # types, together). Not the spread across unrelated datasets, which mixes
        # in disease, tissue, protocol and composition differences, and not the
        # fixed-parameter Monte Carlo envelope, which at 141,553 cells is ~0.4%
        # wide and tests exact equality.
        print(f"participant bootstrap of the reference cohort: {args.n_boot} draws",
              flush=True)
        adata = _load_dataset(args.dataset)
        accs = summarize_adata_per_celltype(
            adata,
            participant_col=args.participant_col,
            visit_col=args.visit_col,
            arm_col=args.arm_col,
            celltype_col=args.celltype_col,
            layer=args.layer,
        )
        boot = participant_bootstrap_statistics(accs, n_boot=args.n_boot, verbose=True)
        del adata, accs

    df = run_gates(
        cfg,
        observed=targets,
        n_mc=args.n_mc,
        n_jobs=args.n_jobs,
        out_dir=out / "gates",
        bootstrap=boot,
    )

    with pd.option_context("display.width", 240, "display.max_columns", 24):
        print(df.to_string(index=False))
    counts = df["verdict"].value_counts().to_dict()
    derived_fail = df[(df["verdict"] == "FAIL") & (df["class"] == "DERIVED")]
    pinned_fail = df[(df["verdict"] == "FAIL") & (df["class"] == "PINNED")]
    print(f"\n{len(df)} statistics: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    print(f"  DERIVED failures (fidelity): {len(derived_fail)} "
          f"{derived_fail['statistic'].tolist()}")
    print(f"  PINNED failures (implementation): {len(pinned_fail)} "
          f"{pinned_fail['statistic'].tolist()}")


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


def cmd_diagnose(args) -> None:
    """Measure the hierarchy variance components WITH and WITHOUT cell-type conditioning.

    The dispersion estimate was already shown to be inflated 0.774 -> 0.442 by pooling cell
    types. The variance components are computed from participant x visit pseudobulk SUMMED over
    cell types, so between-participant differences in cell-type COMPOSITION are counted as
    gene-expression variance -- the same error one level up the hierarchy. Composition is largely
    a participant-level property stable across visits, so it should load onto sigma_b rather than
    sigma_u and inflate the implied pre/post correlation.

    This measures both and reports the ratio. It is a measurement, not a tuning knob.
    """
    import numpy as np

    from sctrial.benchmark.calibration import summarize_adata

    adata = _load_dataset(args.dataset)
    print(f"{args.dataset}: {adata.n_obs:,} cells x {adata.n_vars:,} genes", flush=True)
    acc = summarize_adata(
        adata,
        participant_col=args.participant_col,
        visit_col=args.visit_col,
        arm_col=args.arm_col,
        celltype_col=args.celltype_col,
        layer=args.layer,
    )
    pooled = acc.variance_components()
    within = acc.variance_components(within_stratum=True)

    print("\n=== hierarchy variance components ===")
    print(f"{'quantity':32s} {'pooled over cell types':>24s} {'within cell type':>18s}")
    pairs = [
        ("between_participant_sd_latent", "between_participant_sd_latent_within_ct"),
        ("prepost_corr_latent", "prepost_corr_latent_within_ct"),
        ("sigma_b_latent", "sigma_b_latent_within_ct"),
        ("sigma_u_latent", "sigma_u_latent_within_ct"),
        ("sigma_e_pseudobulk", "sigma_e_pseudobulk_within_ct"),
    ]
    for a, b in pairs:
        va, vb = pooled.get(a, np.nan), within.get(b, np.nan)
        ratio = vb / va if va else np.nan
        print(f"{a:32s} {va:24.4f} {vb:18.4f}   ratio {ratio:.3f}")
    print(f"\ncell types used: {within.get('variance_components_n_celltypes')}")

    # The decisive measurement for Gate E: is TNBC's gene-wise correlation
    # heterogeneity intrinsic, or is it cell-type composition leaking in?
    gw = acc.genewise_corr_within_ct = acc.genewise_corr_within_stratum()
    per_ct = gw.pop("_per_celltype", {})
    print("\n=== gene-wise pre/post correlation: POOLED vs WITHIN cell type ===")
    st = acc.statistics()
    rows = [
        ("median", "prepost_corr_genewise_median", "genewise_corr_within_ct_median"),
        ("mean", "prepost_corr_genewise_mean", "genewise_corr_within_ct_mean"),
        ("sd", "prepost_corr_genewise_sd", "genewise_corr_within_ct_sd"),
        ("q10", "prepost_corr_genewise_q10", "genewise_corr_within_ct_q10"),
        ("q25", "prepost_corr_genewise_q25", "genewise_corr_within_ct_q25"),
        ("q75", "prepost_corr_genewise_q75", "genewise_corr_within_ct_q75"),
        ("q90", "prepost_corr_genewise_q90", "genewise_corr_within_ct_q90"),
    ]
    print(f"{'stat':8s} {'pooled':>10s} {'within CT':>10s}")
    for label, a, b in rows:
        print(f"{label:8s} {st.get(a, float('nan')):10.4f} {gw.get(b, float('nan')):10.4f}")
    print(f"\nn gene-celltype pairs: {gw.get('genewise_corr_within_ct_n'):,} "
          f"across {gw.get('genewise_corr_within_ct_n_celltypes')} cell types")
    print("\nper cell type (median / sd / n genes):")
    for ct, v in sorted(per_ct.items()):
        print(f"  {ct:28s} {v['median']:7.4f} {v['sd']:7.4f} {v['n_genes']:7,d}")
    print(f"genes used (pooled): {pooled.get('variance_components_n_genes')}")

    out = _manuscript_dir() / "gates"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "variance_component_conditioning.json", "w") as fh:
        json.dump(
            {"pooled": pooled, "within_celltype": within, "genewise_corr": gw,
             "genewise_corr_per_celltype": per_ct},
            fh, indent=2, default=float,
        )
    print(f"\nwrote {out / 'variance_component_conditioning.json'}")


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(cfg, out: Path, dataset: str, summary: dict) -> dict:
    """Everything needed to reproduce, or to detect drift in, the definitive run.

    A config file alone is not enough provenance. This project has already had a
    calibration read from a deleted scratch file, a stale .npz analysed after a
    partial sync, a calibration that never reached the scenario generator, and a
    push to the wrong branch. Every one was silent. The manifest is hashed and
    re-verified by each benchmark job before it does any work.
    """
    import hashlib
    import platform
    import shutil
    import subprocess
    from dataclasses import asdict

    import numpy as np

    from sctrial.benchmark.simulator_v2 import TranscriptomeSimConfig, eligible_panel_genes

    # Provenance captured at DEPLOY time, on a node where git exists.
    #
    # git is NOT on PATH on this cluster's compute nodes. Every git call therefore
    # returned the string "unavailable", which meant git_commit recorded
    # "unavailable" -- a manifest with no provenance at all, written silently --
    # and bool("unavailable") is True, so the dirty guard refused for a reason
    # that had nothing to do with the tree being dirty. A manifest that cannot
    # identify the code is worse than no manifest, because it looks like one.
    _deployed: dict = {}
    _dep_file = REPO / ".deploy_provenance.json"
    if _dep_file.exists():
        try:
            _deployed = json.loads(_dep_file.read_text())
        except Exception:
            _deployed = {}

    _git_available = subprocess.run(
        ["git", "--version"], capture_output=True, text=True
    ).returncode == 0 if shutil.which("git") else False

    def _git(*a):
        if not _git_available:
            return ""
        try:
            return subprocess.run(
                ["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=60
            ).stdout.strip()
        except Exception:
            return ""

    if not _git_available and not _deployed:
        raise SystemExit(
            "cannot establish provenance: git is unavailable here and no "
            f"{_dep_file.name} was written by `scripts/sync_hpc.sh deploy`. "
            "Refusing to record a manifest that cannot identify the code it "
            "describes. Deploy first."
        )

    def _code_dirty() -> str:  # noqa: D401
        """Uncommitted changes to CODE, which is what reproducibility depends on.

        A bare `git status --porcelain` also reports untracked scratch at the repo
        root -- logs, ad-hoc sbatch files, working output. Those do not affect
        whether the run can be reproduced from the commit, and treating them as
        blocking either stops the freeze for the wrong reason or, worse, invites
        someone to pass --force and skip the check that matters.
        """
        return _git("status", "--porcelain", "--", "src", "scripts", "tests",
                    "manuscript_figures", "pyproject.toml")

    probe = TranscriptomeSimConfig(**{**asdict(cfg), "seed": 0, "effects": {}})
    eligible = eligible_panel_genes(probe)
    elig_hash = hashlib.sha256(np.asarray(eligible, dtype=np.int64).tobytes()).hexdigest()

    versions = {}
    for mod in ("numpy", "pandas", "scipy", "anndata", "sctrial"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "unavailable"
    try:
        r = subprocess.run(
            ["Rscript", "-e",
             'ip <- installed.packages()[,"Version"]; '
             'cat(paste(c("R", names(ip)[names(ip) %in% c("limma","edgeR","dreamlet","nebula","variancePartition")]), '
             'c(as.character(getRversion()), ip[names(ip) %in% c("limma","edgeR","dreamlet","nebula","variancePartition")]), '
             'sep="=", collapse=" "))'],
            capture_output=True, text=True, timeout=300,
        )
        r_versions = r.stdout.strip() or "unavailable"
    except Exception:
        r_versions = "unavailable"

    from sctrial.benchmark.manifest import source_tree_sha256 as _source_tree_sha256

    def _bioc_version() -> str:
        """Recorded explicitly, not inferred from package versions."""
        try:
            return subprocess.run(
                ["Rscript", "-e",
                 'cat(as.character(BiocManager::version()))'],
                capture_output=True, text=True, timeout=120,
            ).stdout.strip() or "unavailable"
        except Exception:
            return "unavailable"

    targets = out / f"{dataset}_sim_targets.json"
    npz = out / f"{dataset}_empirical.npz"
    return {
        "git_commit": _git("rev-parse", "HEAD") or _deployed.get("git_commit", ""),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD")
        or _deployed.get("git_branch", ""),
        "git_dirty": bool(_code_dirty()) if _git_available
        else bool(_deployed.get("git_dirty", True)),
        "git_dirty_files": _code_dirty().splitlines()[:20],
        "provenance_source": "local_git" if _git_available else "deploy_capture",
        "git_untracked_noncode": len(
            [ln for ln in _git("status", "--porcelain").splitlines() if ln.startswith("??")]
        ),
        "git_describe": _git("describe", "--tags", "--always")
        or _deployed.get("git_describe", ""),
        # What is ACTUALLY on disk, independent of git. Every benchmark job
        # recomputes this and refuses to run if it differs.
        "source_tree_sha256": _source_tree_sha256(),
        "dataset": dataset,
        "calibration_level": "within_cell_type",
        "targets_sha256": _sha256(targets) if targets.exists() else None,
        "empirical_sha256": _sha256(npz) if npz.exists() else None,
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
        ).hexdigest(),
        "eligible_genes_sha256": elig_hash,
        "n_eligible_genes": int(len(eligible)),
        "panel_sizes": list(cfg.panel_sizes),
        "seeds": {"panel": "seed+1", "signal": "seed+2", "gene_rates": "seed",
                  "other_draws": "seed+7919", "background_dispersion": "seed+104729"},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": versions,
        "r_versions": r_versions,
        "bioconductor": _bioc_version(),
        "gate_summary": summary,
    }


def cmd_freeze(args) -> None:
    """Write the frozen configuration the definitive benchmark run must use."""
    from sctrial.benchmark.simulator_v2 import SCENARIO_OWNED_FIELDS

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
    n_derived = summary.get("n_fail_derived", summary.get("n_fail", 1))
    n_pinned = summary.get("n_fail_pinned", 0)
    if n_pinned and not args.force:
        raise SystemExit(
            f"refusing to freeze: {n_pinned} PINNED statistic(s) FAIL. A PINNED "
            "statistic is a readback of a pool the simulator resamples, so this is "
            "an IMPLEMENTATION defect, not a fidelity one. Fix the wiring."
        )
    if n_derived > 0 and not args.force:
        raise SystemExit(
            f"refusing to freeze: {n_derived} DERIVED gate(s) FAIL "
            f"({summary.get('failures')}). Fix the calibration, or pass --force "
            "and record the justification in tasks/MASTER_PLAN.md."
        )
    manifest = _build_manifest(cfg, out, args.dataset, summary)

    # The manifest's own hash, computed over every other field. Results are
    # addressed by it on disk (results/<manifest_sha>/), stamped onto every row,
    # and re-verified before a figure reads them, so it must exist and must be a
    # hex digest -- `config_sha256` alone covers the calibration but not the
    # source tree, the gate verdicts or the package versions, which are exactly
    # the things that changed between the runs this is meant to keep apart.
    from sctrial.benchmark.manifest import manifest_hash

    manifest["manifest_sha256"] = manifest_hash(manifest)

    # A dirty tree cannot be reproduced from any commit, so a result produced
    # from it is unverifiable however carefully it is hashed.
    if manifest["git_dirty"] and not args.force:
        raise SystemExit(
            "refusing to freeze: the working tree has uncommitted changes. A run "
            "produced from it cannot be reproduced from any commit. Commit or "
            "stash first."
        )

    # The distributional gate must have used its PRESPECIFIED participant-bootstrap
    # reference. The simulation-only fallback is useful during development and is
    # a different, tighter test; accepting it here would freeze on a test that was
    # not the one specified.
    import pandas as _pd

    _led = out / "gates" / "gate_envelopes.csv"
    if _led.exists():
        _df = _pd.read_csv(_led)
        _fb = _df["statistic"].astype(str).str.contains("simulation_only")
        if _fb.any() and not args.force:
            raise SystemExit(
                "refusing to freeze: the distributional gate fell back to a "
                "simulation-only reference "
                f"({_df.loc[_fb, 'statistic'].tolist()}). The prespecified test uses "
                "the participant bootstrap. Re-run the gates with --n-boot."
            )
        if (_df["verdict"] == "INSUFFICIENT").any() and not args.force:
            raise SystemExit(
                "refusing to freeze: "
                f"{_df.loc[_df.verdict == 'INSUFFICIENT', 'statistic'].tolist()} "
                "returned INSUFFICIENT. A gate that never ran is not a gate that "
                "passed."
            )

    # THE FROZEN OBJECT CARRIES NO EXPERIMENTAL DESIGN.
    #
    # The calibration describes the reference POPULATION: expression rates,
    # dispersion, the three variance components, library-size and cell-yield
    # pools. The scenario grid describes the EXPERIMENT: arm sizes, visits,
    # signal, cell-yield condition. Conflating them collapsed the whole two-arm
    # sample-size axis, because the anchor's retained 6-versus-5 design leaked in
    # as `arm_ratio` and was read ahead of each scenario's `n_per_arm`.
    #
    # Consumers already strip these fields. Not WRITING them is the stronger fix:
    # a consumer that forgets to strip, or a new consumer that never knew it had
    # to, then cannot leak a design it was never given. The anchor's own design is
    # kept alongside as provenance -- it is a real property of the TNBC Treg
    # cohort and every longitudinal target was estimated on it -- under a key
    # nothing loads as configuration.
    full = asdict(cfg)
    calibration = {k: v for k, v in full.items() if k not in SCENARIO_OWNED_FIELDS}
    anchor_design = {k: v for k, v in full.items() if k in SCENARIO_OWNED_FIELDS}

    frozen = {
        "config": calibration,
        "anchor_design": anchor_design,
        "scenario_owned_fields": list(SCENARIO_OWNED_FIELDS),
        "targets_source": str(out / f"{args.dataset}_sim_targets.json"),
        "gate_summary": summary,
        "dataset": args.dataset,
        "manifest": manifest,
    }
    path = out / "frozen_simulator_config.json"
    with open(path, "w") as fh:
        json.dump(frozen, fh, indent=2, default=str)
    print(f"froze configuration -> {path}")
    print("\n=== RUN MANIFEST ===")
    for k in ("git_commit", "git_branch", "git_dirty", "calibration_level",
              "targets_sha256", "empirical_sha256", "config_sha256",
              "eligible_genes_sha256", "n_eligible_genes", "python", "r_versions"):
        print(f"  {k}: {manifest.get(k)}")
    if manifest["git_dirty"]:
        print("\nWARNING: the working tree is DIRTY. The frozen commit does not "
              "describe what will run. Commit before the definitive run.")


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
    g.add_argument("--n-boot", type=int, default=500,
                   help="participant bootstrap draws defining the acceptance tolerance; "
                        "0 falls back to the Monte Carlo envelope")
    g.set_defaults(func=cmd_gates)

    a = sub.add_parser("ablate", help="Gate D normalisation-scope ablation")
    a.add_argument("--n-rep", type=int, default=5)
    a.add_argument("--panel-size", type=int, default=200)
    a.set_defaults(func=cmd_ablate)

    sub.add_parser(
        "diagnose", help="variance components with vs without cell-type conditioning"
    ).set_defaults(func=cmd_diagnose)

    f = sub.add_parser("freeze", help="freeze the validated configuration")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_freeze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
