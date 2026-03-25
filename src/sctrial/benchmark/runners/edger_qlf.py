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

_R_SCRIPT_TWO_ARM = """\
library(edgeR)

counts <- read.csv("{counts_csv}", row.names=1, check.names=FALSE)
meta   <- read.csv("{meta_csv}", row.names=1, stringsAsFactors=TRUE)

meta$arm   <- factor(meta$arm, levels=c("{control}", "{treated}"))
meta$visit <- factor(meta$visit, levels=c("{pre}", "{post}"))

# Two-arm: interaction model ~arm * visit (DiD effect)
design <- model.matrix(~arm * visit, data=meta)

y <- DGEList(counts=t(counts))
keep <- filterByExpr(y, design, min.count=1)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- calcNormFactors(y)
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design)

# Test the interaction term (last coefficient)
coef_idx <- ncol(design)
qlf <- glmQLFTest(fit, coef=coef_idx)
res <- topTags(qlf, n=Inf, sort.by="none")$table
write.csv(res, "{output_csv}")
"""

_R_SCRIPT_SINGLE_ARM = """\
library(edgeR)

counts <- read.csv("{counts_csv}", row.names=1, check.names=FALSE)
meta   <- read.csv("{meta_csv}", row.names=1, stringsAsFactors=TRUE)

meta$visit       <- factor(meta$visit, levels=c("{pre}", "{post}"))
meta$participant <- factor(meta$participant)

# Single-arm paired: ~participant + visit (block on participant, test visit)
design <- model.matrix(~participant + visit, data=meta)

y <- DGEList(counts=t(counts))
keep <- filterByExpr(y, design, min.count=1)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- calcNormFactors(y)
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design)

# Test the visit coefficient (Post vs Pre)
coef_idx <- ncol(design)
qlf <- glmQLFTest(fit, coef=coef_idx)
res <- topTags(qlf, n=Inf, sort.by="none")$table
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
    design_type: str = "two_arm",
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
        return {g: _fail_result("numerical") for g in gene_cols}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create consistent sample IDs for row alignment
        sample_ids = [f"S{i}" for i in range(len(pseudobulk))]

        counts_df = pseudobulk[gene_cols].copy().clip(lower=0)
        counts_df.index = sample_ids
        counts_csv = tmpdir / "counts.csv"
        counts_df.to_csv(counts_csv)

        meta_df = pseudobulk[[participant_col, arm_col, visit_col]].copy()
        meta_df.columns = ["participant", "arm", "visit"]
        meta_df.index = sample_ids
        meta_csv = tmpdir / "meta.csv"
        meta_df.to_csv(meta_csv)

        # Verify row alignment
        assert len(counts_df) == len(meta_df), (
            f"Row mismatch: counts={len(counts_df)}, meta={len(meta_df)}"
        )

        output_csv = tmpdir / "results.csv"

        template = _R_SCRIPT_TWO_ARM if design_type == "two_arm" else _R_SCRIPT_SINGLE_ARM
        script = template.format(
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
        "beta": np.nan,
        "pvalue": np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "converged": False,
        "failure_mode": mode,
    }
