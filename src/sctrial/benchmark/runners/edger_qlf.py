"""edgeR quasi-likelihood F-test runner for pseudobulk benchmarking.

Calls edgeR via rpy2. The pseudobulk matrix is exported as CSV,
processed in R, and results are read back.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_R_SCRIPT = """\
library(edgeR)

# Read data
counts <- read.csv("{counts_csv}", row.names=1, check.names=FALSE)
meta   <- read.csv("{meta_csv}", stringsAsFactors=TRUE)

# Ensure factor levels
meta$arm   <- factor(meta$arm, levels=c("{control}", "{treated}"))
meta$visit <- factor(meta$visit, levels=c("{pre}", "{post}"))

# Design matrix: ~arm * visit (interaction = DiD effect)
design <- model.matrix(~arm * visit, data=meta)

# Create DGEList and filter
y <- DGEList(counts=t(counts))
keep <- filterByExpr(y, design, min.count=1)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- calcNormFactors(y)

# Estimate dispersion and fit
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design)

# Test the interaction term (last coefficient = arm:visit)
coef_idx <- ncol(design)
qlf <- glmQLFTest(fit, coef=coef_idx)
res <- topTags(qlf, n=Inf, sort.by="none")$table

# Output
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
    """Run edgeR-QLF on pseudobulk counts.

    Parameters
    ----------
    pseudobulk : DataFrame
        Participant-visit level expression. Columns include gene_cols + metadata.
    gene_cols : list[str]
        Gene names.

    Returns
    -------
    dict : gene → {"beta", "pvalue", "ci_lo", "ci_hi", "converged", "failure_mode"}
    """
    try:
        from rpy2.robjects import r as R
    except ImportError:
        logger.error("rpy2 not installed — cannot run edgeR")
        return {
            g: _fail_result("numerical") for g in gene_cols
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Export counts (samples × genes) — rows must match meta rows
        # pseudobulk_counts already contains summed integer counts
        counts_df = pseudobulk[gene_cols].copy().clip(lower=0)
        counts_csv = tmpdir / "counts.csv"
        counts_df.to_csv(counts_csv)

        # Export metadata
        meta_df = pseudobulk[[participant_col, arm_col, visit_col]].copy()
        meta_df.columns = ["participant", "arm", "visit"]
        meta_csv = tmpdir / "meta.csv"
        meta_df.to_csv(meta_csv, index=False)

        # Verify row alignment
        assert len(counts_df) == len(meta_df), (
            f"Row mismatch: counts={len(counts_df)}, meta={len(meta_df)}"
        )

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
            logger.warning("edgeR-QLF failed: %s", exc)
            return {g: _fail_result("numerical") for g in gene_cols}

    # Parse results
    out = {}
    for gene in gene_cols:
        if gene in res.index:
            row = res.loc[gene]
            out[gene] = {
                "beta": float(row.get("logFC", np.nan)),
                "pvalue": float(row.get("PValue", np.nan)),
                "ci_lo": np.nan,  # edgeR-QLF doesn't return CIs by default
                "ci_hi": np.nan,
                "converged": True,
                "failure_mode": None,
            }
        else:
            # Gene filtered out by filterByExpr
            out[gene] = _fail_result("convergence")

    return out


def _fail_result(mode: str) -> dict:
    return {
        "beta": np.nan, "pvalue": np.nan,
        "ci_lo": np.nan, "ci_hi": np.nan,
        "converged": False, "failure_mode": mode,
    }
