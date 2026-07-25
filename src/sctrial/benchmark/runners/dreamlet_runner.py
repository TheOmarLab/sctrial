"""dreamlet runner for pseudobulk benchmarking.

dreamlet provides precision-weighted linear mixed models for
repeated-measures pseudobulk analysis. It is the closest existing
competitor to sctrial's design-aware approach.

Requires: BiocManager::install("dreamlet") in R.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import _r_session

logger = logging.getLogger(__name__)

# Loaded once per worker process when the session is first created.
_DREAMLET_INIT_R = """\
suppressPackageStartupMessages({
  library(dreamlet)
  library(edgeR)
})
"""

_R_SCRIPT_TWO_ARM = """\
suppressPackageStartupMessages({{
  library(dreamlet)
  library(edgeR)
}})

counts <- read.csv("{counts_csv}", row.names=1, check.names=FALSE)
meta   <- read.csv("{meta_csv}", row.names=1, stringsAsFactors=TRUE)

arm_in_data   <- as.character(meta$arm)
visit_in_data <- as.character(meta$visit)
arm_levels   <- if (all(c("Control","Treated") %in% arm_in_data))   c("Control","Treated")   else sort(unique(arm_in_data))
visit_levels <- if (all(c("Pre","Post")         %in% visit_in_data)) c("Pre","Post")           else sort(unique(visit_in_data))
if (length(arm_levels)   < 2) stop(paste("Need >=2 arm levels, got:",   paste(arm_levels,   collapse=",")))
if (length(visit_levels) < 2) stop(paste("Need >=2 visit levels, got:", paste(visit_levels, collapse=",")))
meta$arm   <- factor(meta$arm,   levels=arm_levels)
meta$visit <- factor(meta$visit, levels=visit_levels)

y <- DGEList(counts=t(counts))
keep <- filterByExpr(y, min.count=1)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- calcNormFactors(y)

# Two-arm: interaction + random intercept for participant
form <- ~ arm * visit + (1|participant)
vobjDream <- voomWithDreamWeights(y, form, meta)
fitmm <- dream(vobjDream, form, meta)
fitmm <- eBayes(fitmm)

# Test the interaction coefficient (label-agnostic grep)
coef_name <- grep("arm.*visit|visit.*arm", colnames(coef(fitmm)), value=TRUE)
if (length(coef_name) == 0) {{
  coef_name <- colnames(coef(fitmm))[ncol(coef(fitmm))]
}}
res <- topTable(fitmm, coef=coef_name[1], number=Inf, sort.by="none",
                confint=TRUE)
write.csv(res, "{output_csv}")
"""

_R_SCRIPT_SINGLE_ARM = """\
suppressPackageStartupMessages({{
  library(dreamlet)
  library(edgeR)
}})

counts <- read.csv("{counts_csv}", row.names=1, check.names=FALSE)
meta   <- read.csv("{meta_csv}", row.names=1, stringsAsFactors=TRUE)

visit_in_data <- as.character(meta$visit)
visit_levels  <- if (all(c("Pre","Post") %in% visit_in_data)) c("Pre","Post") else sort(unique(visit_in_data))
if (length(visit_levels) < 2) stop(paste("Need >=2 visit levels, got:", paste(visit_levels, collapse=",")))
meta$visit       <- factor(meta$visit, levels=visit_levels)
meta$participant <- factor(meta$participant)

y <- DGEList(counts=t(counts))
keep <- filterByExpr(y, min.count=1)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- calcNormFactors(y)

# Single-arm: visit effect + random intercept for participant
form <- ~ visit + (1|participant)
vobjDream <- voomWithDreamWeights(y, form, meta)
fitmm <- dream(vobjDream, form, meta)
fitmm <- eBayes(fitmm)

# Test the visit coefficient (label-agnostic: second visit level)
coef_name <- grep("^visit", colnames(coef(fitmm)), value=TRUE)
if (length(coef_name) == 0) {{
  coef_name <- colnames(coef(fitmm))[ncol(coef(fitmm))]
}} else {{
  coef_name <- coef_name[length(coef_name)]
}}
res <- topTable(fitmm, coef=coef_name, number=Inf, sort.by="none",
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
    design_type: str = "two_arm",
) -> dict[str, dict]:
    """Run dreamlet repeated-measures pseudobulk analysis."""
    with tempfile.TemporaryDirectory() as _tmpdir:
        td = Path(_tmpdir)

        # Create consistent sample IDs for row alignment
        sample_ids = [f"S{i}" for i in range(len(pseudobulk))]

        counts_df = pseudobulk[gene_cols].copy().clip(lower=0)
        counts_df.index = sample_ids
        counts_csv = td / "counts.csv"
        counts_df.to_csv(counts_csv)

        meta_df = pseudobulk[[participant_col, arm_col, visit_col]].copy()
        meta_df.columns = ["participant", "arm", "visit"]
        meta_df.index = sample_ids
        meta_csv = td / "meta.csv"
        meta_df.to_csv(meta_csv)

        assert len(counts_df) == len(meta_df)
        output_csv = td / "results.csv"

        template = _R_SCRIPT_TWO_ARM if design_type == "two_arm" else _R_SCRIPT_SINGLE_ARM
        script = template.format(
            counts_csv=str(counts_csv),
            meta_csv=str(meta_csv),
            output_csv=str(output_csv),
        )

        script_file = td / "run_dreamlet.R"
        script_file.write_text(script)
        try:
            session = _r_session.get_session("dreamlet", _DREAMLET_INIT_R)
            session.run(str(script_file), timeout=1800)
            res = pd.read_csv(output_csv, index_col=0)
        except TimeoutError:
            raise  # propagate so the worker's wall-clock alarm can fire
        except Exception as exc:
            logger.warning("dreamlet failed: %s", exc)
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
        "beta": np.nan,
        "pvalue": np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "converged": False,
        "failure_mode": mode,
    }
