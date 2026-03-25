"""NEBULA runner for cell-level benchmarking.

NEBULA is a subject-aware negative binomial mixed model (NBLMM)
that operates at the cell level — it does NOT aggregate to pseudobulk.
This makes it a distinct methodological class from edgeR/limma/dreamlet.

Requires: install.packages("nebula") in R.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)

_R_SCRIPT_TWO_ARM = """\
library(nebula)
library(Matrix)

counts <- readMM("{mtx_path}")
genes  <- readLines("{genes_path}")
rownames(counts) <- genes

meta <- read.csv("{meta_csv}", stringsAsFactors=TRUE)
meta$arm   <- factor(meta$arm, levels=c("{control}", "{treated}"))
meta$visit <- factor(meta$visit, levels=c("{pre}", "{post}"))

# Two-arm: interaction model
design <- model.matrix(~arm * visit, data=meta)

res <- nebula(
  counts,
  id = meta$participant,
  pred = design,
  offset = log(colSums(counts)),
  method = "LN",
  verbose = FALSE
)

coef_names <- colnames(design)
interaction_idx <- length(coef_names)

out <- data.frame(
  gene = res$summary$gene,
  logFC = res$summary[[paste0("logFC_", coef_names[interaction_idx])]],
  pvalue = res$summary[[paste0("p_", coef_names[interaction_idx])]],
  converged = res$convergence,
  stringsAsFactors = FALSE
)
rownames(out) <- out$gene
write.csv(out, "{output_csv}")
"""

_R_SCRIPT_SINGLE_ARM = """\
library(nebula)
library(Matrix)

counts <- readMM("{mtx_path}")
genes  <- readLines("{genes_path}")
rownames(counts) <- genes

meta <- read.csv("{meta_csv}", stringsAsFactors=TRUE)
meta$visit <- factor(meta$visit, levels=c("{pre}", "{post}"))

# Single-arm: visit effect only
design <- model.matrix(~visit, data=meta)

res <- nebula(
  counts,
  id = meta$participant,
  pred = design,
  offset = log(colSums(counts)),
  method = "LN",
  verbose = FALSE
)

coef_names <- colnames(design)
visit_idx <- length(coef_names)

out <- data.frame(
  gene = res$summary$gene,
  logFC = res$summary[[paste0("logFC_", coef_names[visit_idx])]],
  pvalue = res$summary[[paste0("p_", coef_names[visit_idx])]],
  converged = res$convergence,
  stringsAsFactors = FALSE
)
rownames(out) <- out$gene
write.csv(out, "{output_csv}")
"""


def run(
    adata,
    gene_cols: list[str],
    arm_col: str = "arm",
    visit_col: str = "visit",
    participant_col: str = "participant",
    treated_label: str = "Treated",
    control_label: str = "Control",
    visits: tuple[str, str] = ("Pre", "Post"),
    design_type: str = "two_arm",
) -> dict[str, dict]:
    """Run NEBULA NBLMM on cell-level counts.

    Unlike other runners, NEBULA takes the full cell-level AnnData,
    NOT pseudobulk.
    """
    try:
        from rpy2.robjects import r as R
    except ImportError:
        logger.error("rpy2 not installed — cannot run NEBULA")
        return {g: _fail_result("numerical") for g in gene_cols}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Subset to requested genes
        adata_sub = adata[:, gene_cols].copy()
        X = adata_sub.X
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        # Export as MatrixMarket (genes × cells = transposed)
        from scipy.io import mmwrite

        mtx_path = tmpdir / "counts.mtx"
        mmwrite(str(mtx_path), X.T)  # genes × cells

        genes_path = tmpdir / "genes.txt"
        with open(genes_path, "w") as f:
            for g in gene_cols:
                f.write(g + "\n")

        # Export metadata
        meta_df = adata_sub.obs[[participant_col, arm_col, visit_col]].copy()
        meta_df.columns = ["participant", "arm", "visit"]
        meta_csv = tmpdir / "meta.csv"
        meta_df.to_csv(meta_csv, index=False)

        output_csv = tmpdir / "results.csv"

        template = _R_SCRIPT_TWO_ARM if design_type == "two_arm" else _R_SCRIPT_SINGLE_ARM
        script = template.format(
            mtx_path=str(mtx_path),
            genes_path=str(genes_path),
            meta_csv=str(meta_csv),
            output_csv=str(output_csv),
            treated=treated_label,
            control=control_label,
            pre=visits[0],
            post=visits[1],
        )

        try:
            R(script)
            res = pd.read_csv(output_csv, index_col=0)
        except Exception as exc:
            logger.warning("NEBULA failed: %s", exc)
            return {g: _fail_result("numerical") for g in gene_cols}

    out = {}
    for gene in gene_cols:
        if gene in res.index:
            row = res.loc[gene]
            converged = bool(row.get("converged", True))
            failure = None if converged else "convergence"
            out[gene] = {
                "beta": float(row.get("logFC", np.nan)),
                "pvalue": float(row.get("pvalue", np.nan)),
                "ci_lo": np.nan,  # NEBULA doesn't return CIs by default
                "ci_hi": np.nan,
                "converged": converged,
                "failure_mode": failure,
            }
        else:
            out[gene] = _fail_result("numerical")

    return out


def _fail_result(mode: str) -> dict:
    return {
        "beta": np.nan,
        "pvalue": np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "converged": False,
        "failure_mode": mode,
    }
