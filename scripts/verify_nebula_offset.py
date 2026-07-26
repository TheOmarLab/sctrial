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
                }
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


if __name__ == "__main__":
    main()
