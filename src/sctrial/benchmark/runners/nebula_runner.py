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

from . import _r_session

logger = logging.getLogger(__name__)

# Loaded once per worker process when the session is first created.
_NEBULA_INIT_R = """\
suppressPackageStartupMessages({
  library(nebula)
  library(Matrix)
})
"""

_R_SCRIPT_TWO_ARM = """\
library(nebula)
library(Matrix)

counts <- readMM("{mtx_path}")
genes  <- readLines("{genes_path}")
rownames(counts) <- genes

meta <- read.csv("{meta_csv}", stringsAsFactors=TRUE)
arm_in_data   <- as.character(meta$arm)
visit_in_data <- as.character(meta$visit)
arm_levels   <- if (all(c("Control","Treated") %in% arm_in_data))   c("Control","Treated")   else sort(unique(arm_in_data))
visit_levels <- if (all(c("Pre","Post")         %in% visit_in_data)) c("Pre","Post")           else sort(unique(visit_in_data))
if (length(arm_levels)   < 2) stop(paste("Need >=2 arm levels, got:",   paste(arm_levels,   collapse=",")))
if (length(visit_levels) < 2) stop(paste("Need >=2 visit levels, got:", paste(visit_levels, collapse=",")))
meta$arm   <- factor(meta$arm,   levels=arm_levels)
meta$visit <- factor(meta$visit, levels=visit_levels)

# Drop zero-count cells — log(0) = -Inf offset causes NA in objective function
keep <- colSums(counts) > 0
if (sum(keep) < 2) stop("Too few cells with non-zero counts after filtering")
counts <- counts[, keep]
meta   <- meta[keep, ]

# NEBULA requires cells of the same subject to be contiguous
ord    <- order(meta$participant)
counts <- counts[, ord]
meta   <- meta[ord, ]

# Two-arm: interaction model
design <- model.matrix(~arm * visit, data=meta)

res <- nebula(
  counts,
  id = meta$participant,
  pred = design,
  offset = log(colSums(counts)),
  method = "LN",
  ncore = 1,
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
visit_in_data <- as.character(meta$visit)
visit_levels  <- if (all(c("Pre","Post") %in% visit_in_data)) c("Pre","Post") else sort(unique(visit_in_data))
if (length(visit_levels) < 2) stop(paste("Need >=2 visit levels, got:", paste(visit_levels, collapse=",")))
meta$visit <- factor(meta$visit, levels=visit_levels)

# Drop zero-count cells — log(0) = -Inf offset causes NA in objective function
keep <- colSums(counts) > 0
if (sum(keep) < 2) stop("Too few cells with non-zero counts after filtering")
counts <- counts[, keep]
meta   <- meta[keep, ]

# NEBULA requires cells of the same subject to be contiguous
ord    <- order(meta$participant)
counts <- counts[, ord]
meta   <- meta[ord, ]

# Single-arm: visit effect only
design <- model.matrix(~visit, data=meta)

res <- nebula(
  counts,
  id = meta$participant,
  pred = design,
  offset = log(colSums(counts)),
  method = "LN",
  ncore = 1,
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

# Real-data templates: offset = meta$lib_size (full-transcriptome, linear scale).
#
# NEBULA takes the scaling factor on the LINEAR scale and logs it internally.
# Passing log(lib) therefore logs it twice (log(log(lib))), which compresses
# the offset and attenuates library-size adjustment almost entirely.
# The scaling factor must also be the FULL-TRANSCRIPTOME library size supplied
# by the caller, not colSums() of the tested panel: a panel sum moves with the
# signal, so normalising by it partly divides out the effect being estimated.
_R_SCRIPT_TWO_ARM_REAL = """\
library(nebula)
library(Matrix)

counts <- readMM("{mtx_path}")
genes  <- readLines("{genes_path}")
rownames(counts) <- genes

meta <- read.csv("{meta_csv}", stringsAsFactors=TRUE)
meta$arm   <- factor(meta$arm,   levels=c("{control}", "{treated}"))
meta$visit <- factor(meta$visit, levels=c("{pre}", "{post}"))

# Drop cells with no library or NA lib_size (NA would propagate into keep and
# cause "missing value where TRUE/FALSE needed" in the if() below)
keep <- !is.na(meta$lib_size) & colSums(counts) > 0 & meta$lib_size > 0
if (sum(keep) < 2) stop("Too few cells with non-zero counts after filtering")
counts <- counts[, keep]
meta   <- meta[keep, ]

# Two-arm: interaction model
design <- model.matrix(~arm * visit, data=meta)

res <- nebula(
  counts,
  id = meta$participant,
  pred = design,
  offset = meta$lib_size,
  method = "LN",
  ncore = 1,
  verbose = FALSE
)

coef_names <- colnames(design)
interaction_idx <- length(coef_names)

out <- data.frame(
  gene = res$summary$gene,
  logFC = res$summary[[paste0("logFC_", coef_names[interaction_idx])]],
  pvalue = res$summary[[paste0("p_", coef_names[interaction_idx])]],
  convergence_code = res$convergence,
  stringsAsFactors = FALSE
)
rownames(out) <- out$gene
write.csv(out, "{output_csv}")
"""

_R_SCRIPT_SINGLE_ARM_REAL = """\
library(nebula)
library(Matrix)

counts <- readMM("{mtx_path}")
genes  <- readLines("{genes_path}")
rownames(counts) <- genes

meta <- read.csv("{meta_csv}", stringsAsFactors=TRUE)
meta$visit <- factor(meta$visit, levels=c("{pre}", "{post}"))

# Drop cells with no library or NA lib_size (see two-arm note above)
keep <- !is.na(meta$lib_size) & colSums(counts) > 0 & meta$lib_size > 0
if (sum(keep) < 2) stop("Too few cells with non-zero counts after filtering")
counts <- counts[, keep]
meta   <- meta[keep, ]

# Single-arm: visit effect only
design <- model.matrix(~visit, data=meta)

res <- nebula(
  counts,
  id = meta$participant,
  pred = design,
  offset = meta$lib_size,
  method = "LN",
  ncore = 1,
  verbose = FALSE
)

coef_names <- colnames(design)
visit_idx <- length(coef_names)

out <- data.frame(
  gene = res$summary$gene,
  logFC = res$summary[[paste0("logFC_", coef_names[visit_idx])]],
  pvalue = res$summary[[paste0("p_", coef_names[visit_idx])]],
  convergence_code = res$convergence,
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
    lib_size=None,
) -> dict[str, dict]:
    """Run NEBULA NBLMM on cell-level counts.

    Unlike other runners, NEBULA takes the full cell-level AnnData, NOT
    pseudobulk.

    ``lib_size`` is the FULL-TRANSCRIPTOME library size per cell on the LINEAR
    scale. When provided, it is passed directly to NEBULA as the offset (NEBULA
    logs it internally). When None, falls back to the panel-scoped
    ``log(colSums(counts))`` used by the simulation path.
    """
    with tempfile.TemporaryDirectory() as _tmpdir:
        td = Path(_tmpdir)

        # Subset to requested genes
        adata_sub = adata[:, gene_cols].copy()
        X = adata_sub.layers["counts"] if "counts" in adata_sub.layers else adata_sub.X
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        # Export as MatrixMarket (genes × cells = transposed)
        from scipy.io import mmwrite

        mtx_path = td / "counts.mtx"
        mmwrite(str(mtx_path), X.T)  # genes × cells

        genes_path = td / "genes.txt"
        with open(genes_path, "w") as f:
            for g in gene_cols:
                f.write(g + "\n")

        # Export metadata
        meta_df = adata_sub.obs[[participant_col, arm_col, visit_col]].copy()
        meta_df.columns = ["participant", "arm", "visit"]
        if lib_size is not None:
            meta_df["lib_size"] = np.asarray(lib_size, dtype=float)
        meta_csv = td / "meta.csv"
        meta_df.to_csv(meta_csv, index=False)

        output_csv = td / "results.csv"

        if lib_size is not None:
            # Real-data path: full-transcriptome lib_size supplied by caller.
            # Use templates that pass it as the NEBULA offset on the linear scale.
            if design_type == "two_arm":
                template = _R_SCRIPT_TWO_ARM_REAL
            else:
                template = _R_SCRIPT_SINGLE_ARM_REAL
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
        else:
            # Simulation path: keep existing offset = log(colSums(counts)).
            template = _R_SCRIPT_TWO_ARM if design_type == "two_arm" else _R_SCRIPT_SINGLE_ARM
            script = template.format(
                mtx_path=str(mtx_path),
                genes_path=str(genes_path),
                meta_csv=str(meta_csv),
                output_csv=str(output_csv),
            )

        script_file = td / "run_nebula.R"
        script_file.write_text(script)
        try:
            session = _r_session.get_session("nebula", _NEBULA_INIT_R)
            session.run(str(script_file), timeout=1800)
            res = pd.read_csv(output_csv, index_col=0)
        except TimeoutError:
            raise  # propagate so the worker's wall-clock alarm can fire
        except Exception as exc:
            logger.warning("NEBULA failed: %s", exc)
            return {g: _fail_result("numerical") for g in gene_cols}

    out = {}
    for gene in gene_cols:
        if gene in res.index:
            row = res.loc[gene]
            if "convergence_code" in res.columns:
                code = float(row.get("convergence_code", np.nan))
                converged = bool(code > -20) if np.isfinite(code) else False
            else:
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
