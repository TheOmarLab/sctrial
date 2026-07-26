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

meta$arm   <- factor(meta$arm, levels=c("{control}", "{treated}"))
meta$visit <- factor(meta$visit, levels=c("{pre}", "{post}"))

y <- DGEList(counts=t(counts))
keep <- filterByExpr(y, min.count=1)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- calcNormFactors(y)

# Two-arm: interaction + random intercept for participant
form <- ~ arm * visit + (1|participant)
vobjDream <- voomWithDreamWeights(y, form, meta)
fitmm <- dream(vobjDream, form, meta)
fitmm <- eBayes(fitmm)

# Test the interaction coefficient
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

meta$visit       <- factor(meta$visit, levels=c("{pre}", "{post}"))
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

# Test the visit coefficient
coef_name <- "visitPost"
if (!(coef_name %in% colnames(coef(fitmm)))) {{
  coef_name <- colnames(coef(fitmm))[ncol(coef(fitmm))]
}}
res <- topTable(fitmm, coef=coef_name, number=Inf, sort.by="none",
                confint=TRUE)
write.csv(res, "{output_csv}")
"""

# limma/voom/dreamlet/edgeR report log2 fold-changes; the simulator injects the
# effect on the NATURAL log scale (simulator.py: log_mu += effect) and NEBULA,
# sctrial_did and wilcoxon_paired all report natural-log betas. Harvesting logFC
# unconverted put log2 values into the same `estimated_beta` column as natural-log
# truth, inflating every dreamlet effect by 1/ln2 = 1.4427 and manufacturing the
# "substantial effect-size bias" finding: measured dreamlet signal-gene beta was
# 0.7157 vs 0.5/ln2 = 0.7213, while every natural-log method sat at 0.498-0.504.
_LN2 = float(np.log(2.0))  # log2 -> natural log


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
            treated=treated_label,
            control=control_label,
            pre=visits[0],
            post=visits[1],
        )

        script_file = td / "run_dreamlet.R"
        script_file.write_text(script)
        try:
            session = _r_session.get_session("dreamlet", _DREAMLET_INIT_R)
            session.run(str(script_file), timeout=1800)
            res = pd.read_csv(output_csv, index_col=0)
        except Exception as exc:
            logger.warning("dreamlet failed: %s", exc)
            return {g: _fail_result("numerical") for g in gene_cols}

    out = {}
    for gene in gene_cols:
        if gene in res.index:
            row = res.loc[gene]
            out[gene] = {
                "beta": float(row.get("logFC", np.nan)) * _LN2,
                "pvalue": float(row.get("P.Value", np.nan)),
                "ci_lo": float(row.get("CI.L", np.nan)) * _LN2,
                "ci_hi": float(row.get("CI.R", np.nan)) * _LN2,
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
