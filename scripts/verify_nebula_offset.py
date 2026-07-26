#!/usr/bin/env python
"""Empirically settle NEBULA's offset scale, rather than trusting a reading of the docs.

The benchmark passed ``offset = log(colSums(counts))``. If nebula logs the offset
internally, that is a double log and library-size adjustment is almost entirely
lost. The audit asserted this from the package documentation; documentation has
already been misread once in this project (the convergence codes), so it is
settled here by experiment.

Design: the offset only matters when library size is CONFOUNDED with the design.
With depth drawn independently of arm and visit, both conventions look fine and
the comparison is uninformative -- a first version of this script made exactly
that mistake and produced a 2x bias difference that was within noise.

So: multiply the library size of treated-post cells by a known factor, and inject
NO effect at all. Every gene is null. A working offset absorbs the depth shift and
leaves the interaction coefficients centred on zero; a double-logged offset
absorbs only a fraction of it, and every null gene picks up a spurious positive
interaction. The size of that spurious shift is the diagnostic.

    sbatch scripts/slurm_calibrate.sh   # or run under micromamba on a compute node
    python scripts/verify_nebula_offset.py

Never run on a login node.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

_R_TEMPLATE = """\
suppressPackageStartupMessages({{library(nebula); library(Matrix)}})
counts <- readMM("{mtx}")
rownames(counts) <- readLines("{genes}")
meta <- read.csv("{meta}", stringsAsFactors=TRUE)
meta$arm   <- factor(meta$arm, levels=c("Control","Treated"))
meta$visit <- factor(meta$visit, levels=c("Pre","Post"))
design <- model.matrix(~arm * visit, data=meta)
res <- nebula(counts, id=meta$participant, pred=design,
              offset={offset_expr}, method="LN", ncore=1, verbose=FALSE)
cn <- colnames(design); k <- length(cn)
out <- data.frame(gene=res$summary$gene,
                  logFC=res$summary[[paste0("logFC_", cn[k])]],
                  pvalue=res$summary[[paste0("p_", cn[k])]],
                  convergence_code=res$convergence)
write.csv(out, "{out}", row.names=FALSE)
"""


def _run(counts, meta, genes, offset_expr, tmp: Path, tag: str) -> pd.DataFrame:
    import subprocess

    from scipy.io import mmwrite

    mtx = tmp / f"counts_{tag}.mtx"
    mmwrite(str(mtx), counts)
    gpath = tmp / f"genes_{tag}.txt"
    gpath.write_text("\n".join(genes) + "\n")
    mpath = tmp / f"meta_{tag}.csv"
    meta.to_csv(mpath, index=False)
    opath = tmp / f"out_{tag}.csv"
    script = tmp / f"run_{tag}.R"
    script.write_text(
        _R_TEMPLATE.format(
            mtx=mtx, genes=gpath, meta=mpath, out=opath, offset_expr=offset_expr
        )
    )
    proc = subprocess.run(
        ["Rscript", str(script)], capture_output=True, text=True, timeout=7200
    )
    if not opath.exists():
        raise RuntimeError(f"nebula ({tag}) produced no output:\n{proc.stderr[-3000:]}")
    return pd.read_csv(opath)


def main() -> None:
    from sctrial.benchmark.simulator_v2 import (
        TranscriptomeSimConfig,
        build_params,
        iter_pv_blocks,
    )

    DEPTH_FACTOR = 3.0  # treated-post cells sequenced 3x deeper
    panel_n = 150

    cfg = TranscriptomeSimConfig(
        n_per_arm=6,
        n_genes_transcriptome=2500,
        cells_per_pv_fixed=250,
        use_empirical_library=False,
        use_empirical_cells_per_pv=False,
        lib_log_mean=7.5,
        lib_log_sd=0.8,
        effects={},  # NO true effect anywhere: every gene is null
        seed=11,
    )

    # Generate, then scale the depth of treated-post cells only. Scaling counts
    # after generation (binomial thinning would be the alternative) keeps the
    # confounding exactly known and independent of the generative parameters.
    params = build_params(cfg)
    rng = np.random.default_rng(999)
    blocks, obs = [], []
    for blk in iter_pv_blocks(cfg, params=params):
        counts = blk["counts"]
        if blk["arm"] == "Treated" and blk["visit"] == "Post":
            counts = rng.poisson(counts * DEPTH_FACTOR).astype(np.int32)
        blocks.append(counts)
        obs.append(
            pd.DataFrame(
                {
                    "participant": blk["participant"],
                    "visit": blk["visit"],
                    "arm": blk["arm"],
                },
                index=range(counts.shape[0]),
            )
        )
    X = np.vstack(blocks)
    meta = pd.concat(obs, ignore_index=True)
    meta["lib_size"] = X.sum(axis=1).astype(float)

    panel = [f"gene_{i}" for i in range(panel_n)]
    idx = list(range(panel_n))

    tp = (meta["arm"] == "Treated") & (meta["visit"] == "Post")
    print(f"cells={len(meta):,}  panel={panel_n}  TRUE EFFECT = 0 for every gene")
    print(f"depth confounding: treated-post median {meta.loc[tp, 'lib_size'].median():,.0f} "
          f"vs others {meta.loc[~tp, 'lib_size'].median():,.0f} "
          f"({meta.loc[tp, 'lib_size'].median() / meta.loc[~tp, 'lib_size'].median():.2f}x)")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        import scipy.sparse as sp

        keep = meta["lib_size"].to_numpy() > 0
        c = sp.csr_matrix(X[keep][:, idx].T)  # genes x cells
        m = meta[keep].reset_index(drop=True)

        results = {}
        for tag, expr in (("linear", "meta$lib_size"), ("logged", "log(meta$lib_size)")):
            print(f"\n--- offset = {expr} ---", flush=True)
            res = _run(c, m, panel, expr, tmp, tag).set_index("gene")
            b = res["logFC"].to_numpy(dtype=float)
            pv = res["pvalue"].to_numpy(dtype=float)
            results[tag] = {
                "mean_beta": float(np.nanmean(b)),
                "median_beta": float(np.nanmedian(b)),
                "fpr_0.05": float(np.nanmean(pv < 0.05)),
                "converged_frac": float(np.mean(res["convergence_code"] > -20)),
            }
            for k, v in results[tag].items():
                print(f"  {k}: {v:.4f}")

    print("\n=== VERDICT ===")
    lin, log = results["linear"], results["logged"]
    print("TRUTH: every interaction coefficient should be 0.")
    print(f"  linear offset : mean beta {lin['mean_beta']:+.4f}")
    print(f"  logged offset : mean beta {log['mean_beta']:+.4f}")
    better = "linear" if abs(lin["mean_beta"]) < abs(log["mean_beta"]) else "logged"
    print(f"-> the offset that absorbs a {DEPTH_FACTOR}x depth shift is the "
          f"{better.upper()} scale.")
    print(
        f"   (ratio of residual depth artifact: "
        f"{abs(log['mean_beta']) / max(abs(lin['mean_beta']), 1e-9):.1f}x worse when logged)"
    )
    if better != "linear":
        print(
            "\nWARNING: this contradicts the contract in "
            "src/sctrial/benchmark/runners/nebula_runner.py. Do NOT run the "
            "benchmark until the contract is corrected."
        )

    print(
        "\nNOTE: the false-positive rate under BOTH conventions is reported above "
        "and is expected to exceed 0.05 here. nebula carries a subject-level "
        "random intercept but no participant-by-visit term, so within-participant "
        "longitudinal variation is left in the cell-level residual. That is a "
        "property of the model, is measured properly by the benchmark grid, and "
        "is not what this script is testing."
    )

    _model_compatible_null(DEPTH_FACTOR, panel_n)
    sigma_u_ablation(DEPTH_FACTOR)


def _model_compatible_null(depth_factor: float, panel_n: int, n_rep: int = 40) -> None:
    """Is the residual bias the OFFSET, or the missing hierarchy level?

    The experiment above leaves mean beta at -0.21 under the correct linear
    offset. That is small next to the +0.83 of the double-logged version, but it
    is not zero and the difference matters: if the offset itself were still
    mis-specified, every NEBULA number in the benchmark would inherit a bias.

    The competing explanation is model misspecification rather than a wiring
    error. The simulator's generative model carries a participant-BY-VISIT random
    effect u_igt; NEBULA's NBLMM carries a subject-level random intercept and
    nothing at the visit level, so that variance lands in the cell-level residual
    and the treatment-by-time contrast absorbs part of it.

    This isolates the two. The data-generating process here is EXACTLY the model
    NEBULA assumes -- subject random intercept only, no participant-by-visit
    term, NB counts, known dispersion, zero true effect, and the same 3x depth
    confound. If mean beta returns to ~0 the offset contract is fully validated
    and the -0.21 is attributable to the missing hierarchy level, which is a
    finding about the method rather than a defect in the harness.
    """
    import tempfile as _tf

    import scipy.sparse as sp

    print("\n" + "=" * 72)
    print("MODEL-COMPATIBLE NULL: DGP matches NEBULA's assumed model exactly")
    print("=" * 72)

    rng = np.random.default_rng(4242)
    n_per_arm, n_cells, alpha = 6, 250, 0.4
    participants = [f"P{i:02d}" for i in range(2 * n_per_arm)]
    arms = ["Treated"] * n_per_arm + ["Control"] * n_per_arm
    # Well-expressed genes only: dispersion is barely identified at low counts and
    # would add noise to the very quantity being measured.
    rate = rng.lognormal(np.log(2e-3), 0.5, size=panel_n)
    rate = rate / rate.sum() * 0.35
    sigma_b = 0.45  # subject random intercept, the ONLY random effect

    betas, pvals, codes = [], [], []
    for rep in range(n_rep):
        blocks, meta = [], []
        for pid, arm in zip(participants, arms):
            b = rng.normal(0.0, sigma_b, size=panel_n)
            for visit in ("Pre", "Post"):
                lib = rng.lognormal(np.log(3000), 0.5, size=n_cells)
                if arm == "Treated" and visit == "Post":
                    lib = lib * depth_factor  # depth confounded with arm x visit
                mu = np.exp(np.log(rate)[None, :] + b[None, :]) * lib[:, None]
                lam = rng.gamma(1.0 / alpha, mu * alpha)
                blocks.append(rng.poisson(lam).astype(np.int32))
                meta.append(
                    pd.DataFrame(
                        {"participant": pid, "arm": arm, "visit": visit},
                        index=range(n_cells),
                    )
                )
        X = np.vstack(blocks)
        m = pd.concat(meta, ignore_index=True)
        m["lib_size"] = X.sum(axis=1).astype(float)
        keep = m["lib_size"].to_numpy() > 0
        genes = [f"gene_{i}" for i in range(panel_n)]
        with _tf.TemporaryDirectory() as td:
            res = _run(
                sp.csr_matrix(X[keep].T), m[keep].reset_index(drop=True), genes,
                "meta$lib_size", Path(td), f"mc{rep}",
            )
        betas.append(res["logFC"].to_numpy(dtype=float))
        pvals.append(res["pvalue"].to_numpy(dtype=float))
        codes.append(res["convergence_code"].to_numpy(dtype=float))
        if (rep + 1) % 10 == 0:
            print(f"  {rep + 1}/{n_rep} replicates", flush=True)

    b = np.concatenate(betas)
    pv = np.concatenate(pvals)
    cd = np.concatenate(codes)
    mcse = float(np.nanstd([np.nanmean(x) for x in betas]) / np.sqrt(len(betas)))
    print(f"\n  replicates: {n_rep}, genes/replicate: {panel_n}, true beta = 0")
    print(f"  mean beta      : {np.nanmean(b):+.4f}  (MCSE {mcse:.4f})")
    print(f"  median beta    : {np.nanmedian(b):+.4f}")
    print(f"  FPR at 0.05    : {np.nanmean(pv < 0.05):.4f}")
    print(f"  converged      : {np.mean(cd > -20):.4f}")
    print("\n  VERDICT:", end=" ")
    if abs(np.nanmean(b)) < max(3 * mcse, 0.03):
        print("mean beta is ~0 under the model-compatible DGP.")
        print("  => the linear-offset CONTRACT IS VALIDATED: correct raw counts, correct")
        print("     raw positive scaling factor, logged internally by nebula.")
        print("  => the -0.21 seen under the three-level simulation is therefore NOT a")
        print("     wiring error. Attributing it specifically to the participant-by-visit")
        print("     variance requires the ablation below; a matched-DGP comparison alone")
        print("     does not isolate which omitted structure is responsible.")
    else:
        print("mean beta is STILL biased under a DGP matching NEBULA's own model.")
        print("  => the offset contract is NOT fully validated. Investigate contrast")
        print("     orientation, predictor centring, gene filtering, count/offset")
        print("     alignment, and the LN versus HL approximation before the run.")


def sigma_u_ablation(depth_factor: float = 3.0, panel_n: int = 120, n_rep: int = 25) -> None:
    """Does the participant-BY-VISIT variance specifically drive the deterioration?

    The matched-DGP test shows NEBULA is well behaved under its own two-level
    model and misbehaves under the three-level simulation. That is consistent with
    the missing biosample level being responsible, but it does not establish it:
    an omitted mean-zero random effect reliably corrupts dependence and standard
    errors, while whether it induces systematic COEFFICIENT bias depends on the
    nonlinear link, the covariance structure, the approximation and the depth
    design.

    So vary only sigma_u, holding everything else fixed. A monotone deterioration
    in mean beta and Type I error with sigma_u is direct mechanistic evidence; a
    flat response would refute the attribution and send the search elsewhere.

    Supplementary evidence, not a headline result.
    """
    import tempfile as _tf

    import scipy.sparse as sp

    print("\n" + "=" * 72)
    print("SIGMA_U ABLATION: is the participant-by-visit level responsible?")
    print("=" * 72)

    n_per_arm, n_cells, alpha, sigma_b = 6, 200, 0.4, 0.45
    participants = [f"P{i:02d}" for i in range(2 * n_per_arm)]
    arms = ["Treated"] * n_per_arm + ["Control"] * n_per_arm

    rows: list[dict] = []
    print(f"\n  {'sigma_u':>8s} {'mean beta':>11s} {'MCSE':>8s} {'FPR@0.05':>9s} {'converged':>10s}")
    for sigma_u in (0.0, 0.25, 0.50, 0.766):  # last = the Treg-calibrated value
        rng = np.random.default_rng(31337)
        rate = rng.lognormal(np.log(2e-3), 0.5, size=panel_n)
        rate = rate / rate.sum() * 0.35
        per_rep_mean, all_p, all_c = [], [], []
        for _rep in range(n_rep):
            blocks, meta = [], []
            for pid, arm in zip(participants, arms):
                b_i = rng.normal(0.0, sigma_b, size=panel_n)
                for visit in ("Pre", "Post"):
                    u = (
                        rng.normal(0.0, sigma_u, size=panel_n)
                        if sigma_u > 0
                        else np.zeros(panel_n)
                    )
                    lib = rng.lognormal(np.log(3000), 0.5, size=n_cells)
                    if arm == "Treated" and visit == "Post":
                        lib = lib * depth_factor
                    mu = np.exp(np.log(rate)[None, :] + (b_i + u)[None, :]) * lib[:, None]
                    blocks.append(rng.poisson(rng.gamma(1.0 / alpha, mu * alpha)).astype(np.int32))
                    meta.append(
                        pd.DataFrame(
                            {"participant": pid, "arm": arm, "visit": visit},
                            index=range(n_cells),
                        )
                    )
            X = np.vstack(blocks)
            m = pd.concat(meta, ignore_index=True)
            m["lib_size"] = X.sum(axis=1).astype(float)
            keep = m["lib_size"].to_numpy() > 0
            with _tf.TemporaryDirectory() as td:
                res = _run(
                    sp.csr_matrix(X[keep].T), m[keep].reset_index(drop=True),
                    [f"gene_{i}" for i in range(panel_n)],
                    "meta$lib_size", Path(td), f"su{sigma_u}_{_rep}",
                )
            per_rep_mean.append(float(np.nanmean(res["logFC"].to_numpy(dtype=float))))
            all_p.append(res["pvalue"].to_numpy(dtype=float))
            all_c.append(res["convergence_code"].to_numpy(dtype=float))
        mb = float(np.nanmean(per_rep_mean))
        mcse = float(np.nanstd(per_rep_mean) / np.sqrt(len(per_rep_mean)))
        fpr = float(np.nanmean(np.concatenate(all_p) < 0.05))
        conv = float(np.mean(np.concatenate(all_c) > -20))
        rows.append({"sigma_u": sigma_u, "mean_beta": mb, "mcse": mcse, "fpr": fpr})
        print(f"  {sigma_u:8.3f} {mb:+11.4f} {mcse:8.4f} {fpr:9.4f} {conv:10.4f}", flush=True)

    # COMPUTE the verdict from the numbers. The previous version printed a fixed
    # interpretive sentence whatever the data showed, which is precisely how a
    # diagnostic stops being one -- it would have announced "monotone
    # deterioration" even for a flat response.
    fprs = np.array([r["fpr"] for r in rows], dtype=float)
    biases = np.array([r["mean_beta"] for r in rows], dtype=float)
    mcses = np.array([r["mcse"] for r in rows], dtype=float)
    fpr_monotone = bool(np.all(np.diff(fprs) > -0.02)) and (fprs[-1] - fprs[0] > 0.10)
    bias_monotone = bool(np.all(np.diff(np.abs(biases)) > -1e-3)) and (
        abs(biases[-1]) > 3 * mcses[-1] and abs(biases[-1]) - abs(biases[0]) > 0.05
    )

    print("\n  VERDICT (computed, not asserted):")
    print(
        f"    Type I error  : {'MONOTONE in sigma_u' if fpr_monotone else 'NOT monotone'}"
        f"  ({fprs[0]:.4f} -> {fprs[-1]:.4f})"
    )
    print(
        f"    coefficient   : {'MONOTONE in sigma_u' if bias_monotone else 'NOT explained by sigma_u'}"
        f"  ({biases[0]:+.4f} -> {biases[-1]:+.4f}, MCSE {mcses[-1]:.4f})"
    )
    if fpr_monotone and not bias_monotone:
        print("    => the omitted participant-by-visit level explains the CALIBRATION")
        print("       failure but NOT any coefficient bias. An omitted mean-zero random")
        print("       effect corrupts dependence and standard errors; it need not shift")
        print("       the point estimate, and here it does not.")
        print("    => any larger coefficient bias seen under the full simulator has a")
        print("       DIFFERENT cause and must not be attributed to sigma_u.")
    elif fpr_monotone and bias_monotone:
        print("    => sigma_u drives both. The attribution is supported on both counts.")
    else:
        print("    => sigma_u is NOT the responsible omission. Look elsewhere.")
    print("\n  NOTE: NEBULA is designed for multi-subject CELL-LEVEL inference; a")
    print("  treatment-by-visit contrast over repeated biosamples is not the")
    print("  cross-sectional subject-level use case it was principally developed for.")
    print("  The finding is about fit to THIS hierarchy, not general calibration.")


if __name__ == "__main__":
    main()
