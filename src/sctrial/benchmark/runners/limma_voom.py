"""limma-voom runner for pseudobulk benchmarking.

Uses limma-voom with design-appropriate models:
- Two-arm: interaction term (arm × visit) for DiD effect
- Single-arm: paired visit effect with participant blocking
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_R_SCRIPT_TWO_ARM = """\
library(limma)
library(edgeR)

counts <- read.csv("{counts_csv}", row.names=1, check.names=FALSE)
meta   <- read.csv("{meta_csv}", row.names=1, stringsAsFactors=TRUE)

meta$arm   <- factor(meta$arm, levels=c("{control}", "{treated}"))
meta$visit <- factor(meta$visit, levels=c("{pre}", "{post}"))

# Two-arm: ~arm * visit interaction, with the repeated measure carried by
# duplicateCorrelation below.
design <- model.matrix(~arm * visit, data=meta)

# NORMALISATION-SCOPE CONTRACT. lib.size is the FULL-TRANSCRIPTOME total supplied
# by the caller, not colSums of the tested panel. With a panel denominator a
# coordinated signal moves the reference every null gene is measured against, and
# each null gene acquires an offsetting apparent effect. Roughly two thirds of the
# "empirical-Bayes inflation" previously reported for dreamlet is that artifact.
# keep.lib.sizes=TRUE prevents filterByExpr from recomputing it from the panel.
y <- DGEList(counts=t(counts))
y$samples$lib.size <- meta$lib_size
keep <- filterByExpr(y, design, min.count=1)
y <- y[keep, , keep.lib.sizes=TRUE]
y <- calcNormFactors(y)

v0 <- voom(y, design)

# GATE G: the conventional pseudobulk comparator. A two-arm paired design needs
# the repeated measure modelled; ~arm*visit alone treats the two visits of a
# participant as independent. Participant fixed effects are rank-deficient
# (participant is nested within arm), so the conventional limma route is
# duplicateCorrelation with participant as the block, iterated once as the limma
# user guide prescribes. It is allowed to FAIL rather than fall back silently:
# a fallback would report a different model under the same method name.
corfit <- duplicateCorrelation(v0, design, block=meta$participant)
v <- voom(y, design, block=meta$participant, correlation=corfit$consensus)
corfit <- duplicateCorrelation(v, design, block=meta$participant)
fit <- lmFit(v, design, block=meta$participant, correlation=corfit$consensus)
fit <- eBayes(fit)

coef_idx <- ncol(design)
res <- topTable(fit, coef=coef_idx, number=Inf, sort.by="none",
                confint=TRUE)
write.csv(res, "{output_csv}")
"""

_R_SCRIPT_SINGLE_ARM = """\
library(limma)
library(edgeR)

counts <- read.csv("{counts_csv}", row.names=1, check.names=FALSE)
meta   <- read.csv("{meta_csv}", row.names=1, stringsAsFactors=TRUE)

meta$visit       <- factor(meta$visit, levels=c("{pre}", "{post}"))
meta$participant <- factor(meta$participant)

design <- model.matrix(~participant + visit, data=meta)

# NORMALISATION-SCOPE CONTRACT. lib.size is the FULL-TRANSCRIPTOME total supplied
# by the caller, not colSums of the tested panel. With a panel denominator a
# coordinated signal moves the reference every null gene is measured against, and
# each null gene acquires an offsetting apparent effect. Roughly two thirds of the
# "empirical-Bayes inflation" previously reported for dreamlet is that artifact.
# keep.lib.sizes=TRUE prevents filterByExpr from recomputing it from the panel.
y <- DGEList(counts=t(counts))
y$samples$lib.size <- meta$lib_size
keep <- filterByExpr(y, design, min.count=1)
y <- y[keep, , keep.lib.sizes=TRUE]
y <- calcNormFactors(y)

v <- voom(y, design)
fit <- lmFit(v, design)
fit <- eBayes(fit)

# Test the visit coefficient (Post vs Pre)
coef_idx <- ncol(design)
res <- topTable(fit, coef=coef_idx, number=Inf, sort.by="none",
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
    lib_size=None,
) -> dict[str, dict]:
    """Run limma-voom on pseudobulk counts with participant blocking."""
    with tempfile.TemporaryDirectory() as _tmpdir:
        td = Path(_tmpdir)

        sample_ids = [f"S{i}" for i in range(len(pseudobulk))]

        counts_df = pseudobulk[gene_cols].copy().clip(lower=0)
        counts_df.index = sample_ids
        counts_csv = td / "counts.csv"
        counts_df.to_csv(counts_csv)

        meta_df = pseudobulk[[participant_col, arm_col, visit_col]].copy()
        meta_df.columns = ["participant", "arm", "visit"]
        # Full-transcriptome library size. Required: without it the model
        # normalises against the tested panel, whose total moves with the
        # signal. See sctrial.benchmark.contracts.
        if lib_size is None:
            raise ValueError(
                "an explicit full-transcriptome lib_size is required; a panel-derived "
                "library size makes the normalisation reference move with the signal"
            )
        meta_df["lib_size"] = np.asarray(lib_size, dtype=float)
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

        script_file = td / "run_limma.R"
        script_file.write_text(script)
        try:
            proc = subprocess.run(
                ["Rscript", str(script_file)],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode != 0:
                logger.warning("limma-voom R error: %s", proc.stderr[-500:])
                return {g: _fail_result("numerical") for g in gene_cols}
            res = pd.read_csv(output_csv, index_col=0)
        except Exception as exc:
            logger.warning("limma-voom failed: %s", exc)
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
