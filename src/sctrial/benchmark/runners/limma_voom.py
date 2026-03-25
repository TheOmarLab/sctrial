"""limma-voom runner for pseudobulk benchmarking.

Uses limma-voom with the interaction term (arm × visit) to estimate
the DiD effect, with participant as a blocking factor.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_R_SCRIPT = """\
library(limma)
library(edgeR)

counts <- read.csv("{counts_csv}", row.names=1, check.names=FALSE)
meta   <- read.csv("{meta_csv}", stringsAsFactors=TRUE)

meta$arm   <- factor(meta$arm, levels=c("{control}", "{treated}"))
meta$visit <- factor(meta$visit, levels=c("{pre}", "{post}"))

# Design with interaction
design <- model.matrix(~arm * visit, data=meta)

# DGEList + voom
y <- DGEList(counts=t(counts))
keep <- filterByExpr(y, design, min.count=1)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- calcNormFactors(y)

v <- voom(y, design)

# Fit with blocking on participant (paired design)
corfit <- duplicateCorrelation(v, design, block=meta$participant)
fit <- lmFit(v, design, block=meta$participant,
             correlation=corfit$consensus.correlation)
fit <- eBayes(fit)

# Test interaction term
coef_idx <- ncol(design)
res <- topTable(fit, coef=coef_idx, number=Inf, sort.by="none",
                confint=TRUE)

write.csv(res, "{output_csv}")
"""


def run(
    pseudobulk: pd.DataFrame,
    gene_cols: list[str],
    arm_col: str = "arm",
    visit_col: str = "visit",
    participant_col: str = "participant",
    treated_label: str = "Treated",
    control_label: str = "Control",
    visits: tuple[str, str] = ("Pre", "Post"),
) -> dict[str, dict]:
    """Run limma-voom on pseudobulk counts with participant blocking."""
    try:
        from rpy2.robjects import r as R
    except ImportError:
        logger.error("rpy2 not installed — cannot run limma-voom")
        return {g: _fail_result("numerical") for g in gene_cols}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        counts_df = pseudobulk[gene_cols].copy().clip(lower=0)
        counts_csv = tmpdir / "counts.csv"
        counts_df.to_csv(counts_csv)

        meta_df = pseudobulk[[participant_col, arm_col, visit_col]].copy()
        meta_df.columns = ["participant", "arm", "visit"]
        meta_csv = tmpdir / "meta.csv"
        meta_df.to_csv(meta_csv, index=False)

        assert len(counts_df) == len(meta_df)

        output_csv = tmpdir / "results.csv"

        script = _R_SCRIPT.format(
            counts_csv=str(counts_csv),
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
            logger.warning("limma-voom failed: %s", exc)
            return {g: _fail_result("numerical") for g in gene_cols}

    out = {}
    for gene in gene_cols:
        if gene in res.index:
            row = res.loc[gene]
            out[gene] = {
                "beta": float(row.get("logFC", np.nan)),
                "pvalue": float(row.get("P.Value", np.nan)),
                "ci_lo": float(row.get("CI.L", np.nan)),
                "ci_hi": float(row.get("CI.R", np.nan)),
                "converged": True,
                "failure_mode": None,
            }
        else:
            out[gene] = _fail_result("convergence")

    return out


def _fail_result(mode: str) -> dict:
    return {
        "beta": np.nan, "pvalue": np.nan,
        "ci_lo": np.nan, "ci_hi": np.nan,
        "converged": False, "failure_mode": mode,
    }
