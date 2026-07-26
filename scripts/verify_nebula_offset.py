#!/usr/bin/env python
"""Empirically settle NEBULA's offset scale, rather than trusting a reading of the docs.

The benchmark passed ``offset = log(colSums(counts))``. If nebula logs the offset
internally, that is a double log and library-size adjustment is almost entirely
lost. The audit asserted this from the package documentation; documentation has
already been misread once in this project (the convergence codes), so it is
settled here by experiment.

Design: simulate a two-arm paired trial with a KNOWN interaction effect and
deliberately wide per-cell library variation, then fit the same model twice --
once with the offset on the linear scale, once logged. Whichever recovers the
injected effect, and whose null genes stay centred at zero, is the correct
convention.

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
    from sctrial.benchmark.simulator_v2 import TranscriptomeSimConfig, simulate_trial_v2

    beta = 0.5
    n_signal = 20
    panel_n = 100
    cfg = TranscriptomeSimConfig(
        n_per_arm=6,
        n_genes_transcriptome=2500,
        cells_per_pv_fixed=300,
        use_empirical_library=False,
        use_empirical_cells_per_pv=False,
        # Wide library variation: if the offset is mis-scaled, this is what makes
        # it visible. With a narrow depth distribution both conventions look fine.
        lib_log_mean=7.5,
        lib_log_sd=1.2,
        seed=11,
    )
    panel = [f"gene_{i}" for i in range(panel_n)]
    cfg = TranscriptomeSimConfig(
        **{**cfg.__dict__, "effects": {g: beta for g in panel[:n_signal]}}
    )
    sim = simulate_trial_v2(cfg)

    adata = sim["adata"]
    lib = np.asarray(adata.X.sum(axis=1)).ravel().astype(float)
    sub = adata[:, panel]
    counts = sub.X.T.tocoo()  # genes x cells
    meta = adata.obs[["participant", "arm", "visit"]].copy()
    meta["lib_size"] = lib
    keep = lib > 0

    print(f"cells={adata.n_obs:,}  panel={panel_n}  signal={n_signal}  beta={beta}")
    print(f"library size: median={np.median(lib):,.0f}  IQR="
          f"{np.percentile(lib, 25):,.0f}-{np.percentile(lib, 75):,.0f}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        import scipy.sparse as sp

        c = sp.csr_matrix(counts)[:, keep]
        m = meta[keep].reset_index(drop=True)
        results = {}
        for tag, expr in (("linear", "meta$lib_size"), ("logged", "log(meta$lib_size)")):
            print(f"\n--- offset = {expr} ---", flush=True)
            res = _run(c, m, panel, expr, tmp, tag)
            res = res.set_index("gene")
            sig = res.loc[[g for g in panel[:n_signal] if g in res.index], "logFC"]
            null = res.loc[[g for g in panel[n_signal:] if g in res.index], "logFC"]
            nullp = res.loc[[g for g in panel[n_signal:] if g in res.index], "pvalue"]
            results[tag] = {
                "signal_mean_beta": float(np.nanmean(sig)),
                "signal_bias": float(np.nanmean(sig) - beta),
                "null_mean_beta": float(np.nanmean(null)),
                "null_fpr_0.05": float(np.nanmean(nullp < 0.05)),
                "converged_frac": float(np.mean(res["convergence_code"] > -20)),
                "convergence_codes": res["convergence_code"].value_counts().to_dict(),
            }
            for k, v in results[tag].items():
                print(f"  {k}: {v}")

    print("\n=== VERDICT ===")
    lin, log = results["linear"], results["logged"]
    better = "linear" if abs(lin["signal_bias"]) < abs(log["signal_bias"]) else "logged"
    print(f"signal-gene bias: linear {lin['signal_bias']:+.4f}  vs  "
          f"logged {log['signal_bias']:+.4f}")
    print(f"null FPR:         linear {lin['null_fpr_0.05']:.4f}  vs  "
          f"logged {log['null_fpr_0.05']:.4f}   (nominal 0.05)")
    print(f"-> nebula's offset argument is on the {better.upper()} scale.")
    if better != "linear":
        print(
            "WARNING: this contradicts the contract in "
            "sctrial/benchmark/runners/nebula_runner.py. Do not run the benchmark "
            "until the contract is corrected."
        )


if __name__ == "__main__":
    main()
