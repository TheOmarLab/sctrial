"""Canonical calibration and summary statistics for the benchmark simulator.

This module is the **single** implementation of every quantity the simulator is
calibrated to and validated against. It replaces a set of one-off scripts that
had drifted into mutually inconsistent versions; a calibration measured by one
script and validated by another is how a simulator stops describing the data it
claims to describe.

Two distinct jobs live here, and conflating them is the trap this module exists
to prevent:

**1. Estimating the GENERATING parameters** (:func:`conditional_dispersion`).
Run once, on the real data. Uses a Cox-Reid adjusted profile-likelihood
Gamma-Poisson MLE with empirical-Bayes shrinkage toward a mean-dependent trend --
the same class of estimator as ``glmGamPoi``/``edgeR``, and *not* a raw
method-of-moments estimate. The strata are participant x visit x cell type and
are therefore small (tens of cells), where MoM is badly biased and the Cox-Reid
term is exactly what removes the bias from having estimated one mean per stratum.

**2. Measuring VALIDATION statistics** (:class:`SummaryAccumulator`).
Run on the real data *and* on every Monte Carlo replicate, through the same code
path. These use fast moment estimators. That is deliberate: for an envelope test
only *consistency* between the real and simulated arm matters, because any bias
in the statistic is shared by both. Using the expensive MLE here would make
Monte Carlo calibration unaffordable and would buy nothing.

Homogeneity convention
----------------------
The simulator generates one homogeneous cell population, so its conditional
statistics are computed within participant x visit. Real data must be conditioned
on participant x visit x **cell type** to be comparable, because pooling cell
types loads between-population mean differences onto the dispersion: on TNBC the
median conditional alpha falls from 0.774 to 0.275 (0.35x) when cell type is
added. The cell-type-pooled *marginal* curve is reported for context but is not
an acceptance criterion -- the simulation has no cell types to pool, and matching
it by tuning a latent parameter would be unidentifiable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "SummaryAccumulator",
    "summarize_blocks",
    "summarize_adata",
    "summarize_simulation",
    "conditional_dispersion",
    "measure_targets",
]

# Genes carrying essentially no counts cannot inform a dispersion estimate. This
# is an estimability filter on a PARAMETER, not a subsample of the data: every
# cell is used, and the excluded genes are reported.
_MIN_MEAN_COUNT = 0.02
_MIN_STRATA_DETECTED = 5

# Genes per reduction pass. Bounds the float64 transient; see ``add_block``.
_REDUCE_CHUNK = 2000


# ---------------------------------------------------------------------------
# Streaming summary statistics -- one implementation, two consumers
# ---------------------------------------------------------------------------


@dataclass
class SummaryAccumulator:
    """Accumulates every validation statistic block by block.

    A block is one homogeneous stratum's cell x gene count matrix. Nothing here
    requires the full matrix to exist, which is what makes full-scale Monte Carlo
    calibration (141k cells x 20k genes x hundreds of replicates) affordable.

    Conditional dispersion is accumulated as pooled within-stratum moments:
    within stratum ``k`` the fitted mean is ``mu_c = s_c * lambda_k`` with
    ``lambda_k = sum(y) / sum(s)``, so

        sum (y - mu)^2 = sum y^2 - 2 lambda sum(y s) + lambda^2 sum(s^2)

    and every term is a per-gene reduction over cells. The NB2 moment estimate is
    then ``alpha = (sum (y-mu)^2 - sum mu) / sum mu^2`` pooled over strata.
    """

    n_genes: int
    gene_names: list[str] | None = None
    # Estimability thresholds. Exposed so tests can construct minimal cases;
    # production callers must leave them at the module defaults.
    min_mean_count: float = _MIN_MEAN_COUNT
    min_strata_detected: int = _MIN_STRATA_DETECTED
    # Minimum CPM for a gene to enter the gene-wise correlation distribution.
    min_cpm_genewise: float = 0.0

    # per-gene accumulators
    total_counts: np.ndarray = field(init=False)
    n_detected_cells: np.ndarray = field(init=False)
    resid_ss: np.ndarray = field(init=False)
    mu_sum: np.ndarray = field(init=False)
    mu2_sum: np.ndarray = field(init=False)
    n_strata_used: np.ndarray = field(init=False)

    # per-cell accumulators (small: one scalar per cell)
    cell_umi: list = field(default_factory=list)
    cell_genes: list = field(default_factory=list)

    # per participant-visit
    pv_rows: list = field(default_factory=list)

    n_cells_total: int = 0

    def __post_init__(self) -> None:
        z = lambda: np.zeros(self.n_genes, dtype=np.float64)  # noqa: E731
        self.total_counts = z()
        self.n_detected_cells = z()
        self.resid_ss = z()
        self.mu_sum = z()
        self.mu2_sum = z()
        self.n_strata_used = z()

    def add_block(
        self,
        counts: np.ndarray,
        participant: str,
        visit: str,
        arm: str = "NA",
        stratum: str | None = None,
    ) -> None:
        """Add one homogeneous stratum of cells.

        ``counts`` is ``n_cells x n_genes``. ``stratum`` labels the homogeneity
        unit (participant x visit for simulated data; participant x visit x cell
        type for real data) and is used only for provenance -- the caller is
        responsible for calling once per stratum.
        """
        counts = np.asarray(counts)
        if counts.ndim != 2 or counts.shape[1] != self.n_genes:
            raise ValueError(
                f"block has shape {counts.shape}, expected (n_cells, {self.n_genes})"
            )
        n_cells = counts.shape[0]
        if n_cells == 0:
            return

        # Reduce over GENE CHUNKS. A full TNBC-scale block is up to 27,653 cells
        # x 20,284 genes; upcasting it to float64 in one piece peaks near 11 GB,
        # and 16 Monte Carlo workers doing that at once will OOM a 400 GB node.
        # Chunking bounds the transient without changing any result.
        G = self.n_genes
        lib = np.zeros(n_cells, dtype=np.float64)
        genes_per_cell = np.zeros(n_cells, dtype=np.int32)
        gene_sum = np.zeros(G)
        y2_sum = np.zeros(G)
        ys_sum = np.zeros(G)
        detected = np.zeros(G)

        for c0 in range(0, G, _REDUCE_CHUNK):
            sl = slice(c0, min(c0 + _REDUCE_CHUNK, G))
            y = counts[:, sl].astype(np.float64)
            lib += y.sum(axis=1)
            nz = y > 0
            genes_per_cell += nz.sum(axis=1).astype(np.int32)
            gene_sum[sl] = y.sum(axis=0)
            detected[sl] = nz.sum(axis=0)

        self.cell_umi.append(lib.astype(np.float32))
        self.cell_genes.append(genes_per_cell)
        self.n_cells_total += n_cells
        self.total_counts += gene_sum
        self.n_detected_cells += detected

        # Split-half pseudobulk. This is what makes the participant and
        # participant x visit variance components IDENTIFIABLE: with one
        # pseudobulk value per participant-visit, the participant x visit
        # deviation and the pseudobulk sampling error are perfectly confounded.
        # Two independent halves of the same cells share the biological terms and
        # differ only by sampling noise, so their difference estimates that noise
        # directly and it can be subtracted off.
        half = np.zeros(n_cells, dtype=bool)
        half[: n_cells // 2] = True
        rs = np.random.default_rng(abs(hash((participant, visit, stratum))) % (2**32))
        rs.shuffle(half)
        sum_a = counts[half].sum(axis=0).astype(np.float64)
        sum_b = counts[~half].sum(axis=0).astype(np.float64)

        # Participant-visit pseudobulk (summed counts) for the longitudinal gate.
        self.pv_rows.append(
            {
                "participant": participant,
                "visit": visit,
                "arm": arm,
                "stratum": stratum if stratum is not None else f"{participant}|{visit}",
                "n_cells": n_cells,
                "counts": gene_sum,
                "counts_a": sum_a,
                "counts_b": sum_b,
                "n_cells_a": int(half.sum()),
                "n_cells_b": int((~half).sum()),
                # Per-cell values and per-gene moment contributions, kept so a
                # PARTICIPANT-level bootstrap can re-pool them without re-reading
                # the count matrix. Cells are not independent calibration units;
                # participants are, and with 141k cells a fixed-participant
                # envelope is ~0.4% wide and rejects any tractable simulator.
                "umi": lib.astype(np.float32),
                "genes_per_cell": genes_per_cell,
                "detected": detected,
            }
        )

        # Conditional dispersion moments need >=2 cells to have any residual df.
        if n_cells < 2 or lib.sum() <= 0:
            return
        s = lib / lib.mean()  # offset, mean 1 within the stratum
        s_sum = float(s.sum())
        s2_sum = float((s * s).sum())
        if s_sum <= 0:
            return
        lam = gene_sum / s_sum  # Poisson/ML mean rate given the offsets
        for c0 in range(0, G, _REDUCE_CHUNK):
            sl = slice(c0, min(c0 + _REDUCE_CHUNK, G))
            y = counts[:, sl].astype(np.float64)
            ys_sum[sl] = y.T @ s
            y2_sum[sl] = (y * y).sum(axis=0)

        resid = y2_sum - 2.0 * lam * ys_sum + (lam**2) * s2_sum
        self.resid_ss += resid
        self.mu_sum += lam * s_sum
        self.mu2_sum += (lam**2) * s2_sum
        self.n_strata_used += (gene_sum > 0).astype(np.float64)
        self.pv_rows[-1].update(
            resid_ss=resid, mu_sum=lam * s_sum, mu2_sum=(lam**2) * s2_sum
        )

    # -- finalisation -------------------------------------------------

    def conditional_alpha(self) -> tuple[np.ndarray, np.ndarray]:
        """Pooled within-stratum NB2 moment dispersion and the per-gene mean.

        Returns ``(alpha, mean_count_per_cell)``; ``alpha`` is NaN where the gene
        is not estimable.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            alpha = (self.resid_ss - self.mu_sum) / self.mu2_sum
            mean_count = self.total_counts / max(self.n_cells_total, 1)
        estimable = (
            (self.mu2_sum > 0)
            & (mean_count >= self.min_mean_count)
            & (self.n_strata_used >= self.min_strata_detected)
        )
        alpha = np.where(estimable, alpha, np.nan)
        return alpha, mean_count

    def pv_frame(self) -> pd.DataFrame:
        """Participant x visit pseudobulk counts, summed over strata."""
        if not self.pv_rows:
            return pd.DataFrame()
        agg: dict[tuple, dict] = {}
        for r in self.pv_rows:
            key = (r["participant"], r["visit"])
            if key not in agg:
                agg[key] = {
                    "participant": r["participant"],
                    "visit": r["visit"],
                    "arm": r["arm"],
                    "n_cells": 0,
                    "counts": np.zeros(self.n_genes),
                    "counts_a": np.zeros(self.n_genes),
                    "counts_b": np.zeros(self.n_genes),
                }
            agg[key]["n_cells"] += r["n_cells"]
            for c in ("counts", "counts_a", "counts_b"):
                agg[key][c] = agg[key][c] + r[c]
        return pd.DataFrame(list(agg.values()))

    def variance_components(self, min_cpm: float = 10.0, within_stratum: bool = False) -> dict:
        """LATENT participant and participant x visit SDs on the log-rate scale.

        These are the quantities the simulator is parameterised by, and they are
        NOT the observable correlation. Conflating the two is the same error as
        calibrating dispersion on the marginal curve: the observable pre/post
        correlation of ``log(1+CPM)`` is attenuated by pseudobulk sampling noise,
        so setting the generating parameter equal to it under-disperses the
        hierarchy (measured: observable 0.348 for a generating 0.466).

        Identification, with two visits per participant and one pseudobulk value
        each, comes from the split halves::

            Var(y_A - y_B) = 4 * sigma_e^2          (halves share b and u)
            Var(post - pre) = 2 sigma_u^2 + 2 sigma_e^2
            Var((post + pre)/2) = sigma_b^2 + sigma_u^2/2 + sigma_e^2/2

        Restricted to genes above ``min_cpm`` because ``log(1+x) ~ log(x)`` only
        there; below it the transform itself shrinks the variance and the
        components would be biased low.
        """
        if within_stratum:
            return self._variance_components_within_stratum(min_cpm)
        pv = self.pv_frame()
        if len(pv) == 0:
            return {}
        pre = pv[pv["visit"] == "Pre"].set_index("participant")
        post = pv[pv["visit"] == "Post"].set_index("participant")
        common = [p for p in pre.index if p in post.index]
        if len(common) < 3:
            return {}
        pre, post = pre.loc[common], post.loc[common]

        def _log_cpm(frame, col, ncol=None):
            m = np.vstack(frame[col].to_numpy())
            tot = m.sum(axis=1, keepdims=True)
            tot[tot == 0] = 1.0
            return np.log1p(m / tot * 1e6)

        y_pre, y_post = _log_cpm(pre, "counts"), _log_cpm(post, "counts")
        mean_cpm = 0.5 * (np.expm1(y_pre).mean(axis=0) + np.expm1(y_post).mean(axis=0))
        keep = mean_cpm >= min_cpm
        if keep.sum() < 20:
            keep = mean_cpm >= np.percentile(mean_cpm, 95)
        y_pre, y_post = y_pre[:, keep], y_post[:, keep]

        # Sampling variance from the split halves, at FULL depth: each half has
        # half the cells, so Var(half) ~ 2 Var(full) and Var(A - B) ~ 4 Var(full).
        d_half = []
        for frame in (pre, post):
            a = _log_cpm(frame, "counts_a")[:, keep]
            b = _log_cpm(frame, "counts_b")[:, keep]
            d_half.append(a - b)
        sigma_e2 = float(np.mean(np.concatenate(d_half) ** 2) / 4.0)

        # Remove the arm x visit mean so a real treatment effect is not counted
        # as participant variance.
        arms = pre["arm"].to_numpy()
        d = y_post - y_pre
        m = 0.5 * (y_post + y_pre)
        for a in np.unique(arms):
            sel = arms == a
            if sel.sum() > 1:
                d[sel] -= d[sel].mean(axis=0, keepdims=True)
                m[sel] -= m[sel].mean(axis=0, keepdims=True)

        n_eff = max(len(common) - len(np.unique(arms)), 1)
        var_d = float((d**2).sum() / (n_eff * d.shape[1]))
        var_m = float((m**2).sum() / (n_eff * m.shape[1]))

        sigma_u2 = max((var_d - 2.0 * sigma_e2) / 2.0, 0.0)
        sigma_b2 = max(var_m - sigma_u2 / 2.0 - sigma_e2 / 2.0, 0.0)
        total = sigma_b2 + sigma_u2
        return {
            "between_participant_sd_latent": float(np.sqrt(total)),
            "prepost_corr_latent": float(sigma_b2 / total) if total > 0 else np.nan,
            "sigma_b_latent": float(np.sqrt(sigma_b2)),
            "sigma_u_latent": float(np.sqrt(sigma_u2)),
            "sigma_e_pseudobulk": float(np.sqrt(sigma_e2)),
            "variance_components_n_genes": int(keep.sum()),
        }

    def _variance_components_within_stratum(self, min_cpm: float) -> dict:
        """Variance components estimated WITHIN each homogeneous stratum.

        The pooled version aggregates over cell types before differencing
        participants, so between-participant differences in cell-type COMPOSITION
        are counted as gene-expression variance. That is the same conditioning
        error that inflated the dispersion estimate (0.774 pooled versus 0.442
        within cell type), one level up the hierarchy.
        """
        import collections

        by_ct: dict[str, list] = collections.defaultdict(list)
        for r in self.pv_rows:
            parts = str(r["stratum"]).split("|")  # participant|visit|celltype
            by_ct[parts[2] if len(parts) > 2 else "ALL"].append(r)

        keys = [
            "between_participant_sd_latent",
            "prepost_corr_latent",
            "sigma_b_latent",
            "sigma_u_latent",
            "sigma_e_pseudobulk",
        ]
        collected: dict[str, list] = {k: [] for k in keys}
        weights: list[float] = []
        for _ct, rows in by_ct.items():
            sub = SummaryAccumulator(n_genes=self.n_genes)
            sub.pv_rows = rows
            res = SummaryAccumulator.variance_components(sub, min_cpm=min_cpm)
            if not res or not np.isfinite(res.get("prepost_corr_latent", np.nan)):
                continue
            for k in keys:
                collected[k].append(res[k])
            weights.append(float(res.get("variance_components_n_genes", 1)))
        if not weights:
            return {}
        w = np.asarray(weights) / np.sum(weights)
        agg = {f"{k}_within_ct": float(np.sum(w * np.asarray(collected[k]))) for k in keys}
        agg["variance_components_n_celltypes"] = int(len(weights))
        return agg

    def genewise_corr_within_stratum(self, min_cpm: float = 1.0) -> dict:
        """Gene-wise pre/post correlation computed WITHIN each cell type.

        The pooled version differences participants after summing over cell types, so a gene
        restricted to one cell type inherits that cell type's ABUNDANCE variation across
        participants. That alone creates gene-to-gene heterogeneity in participant-level
        correlation, with no gene-intrinsic biology behind it.

        The simulator contains one homogeneous population and no composition variation, so if
        TNBC's heterogeneity is compositional then the pooled statistic is not a quantity the
        simulator could or should reproduce -- the same conditional-versus-marginal distinction
        already applied to dispersion. Measuring within cell type is what makes the two arms of
        the gate comparable.
        """
        import collections

        by_ct: dict[str, list] = collections.defaultdict(list)
        for r in self.pv_rows:
            parts = str(r["stratum"]).split("|")
            by_ct[parts[2] if len(parts) > 2 else "ALL"].append(r)

        allr: list[np.ndarray] = []
        per_ct = {}
        for ct, rows in by_ct.items():
            sub = SummaryAccumulator(n_genes=self.n_genes)
            sub.pv_rows = rows
            # `min_cpm` was accepted and never applied, so unexpressed
            # gene-by-cell-type pairs entered the distribution unfiltered.
            sub.min_cpm_genewise = min_cpm
            st = sub._longitudinal_statistics(sub.pv_frame())
            r = st.get("_prepost_corr_genewise")
            if r is None or len(r) < 50:
                continue
            allr.append(np.asarray(r))
            per_ct[ct] = {
                "n_genes": int(len(r)),
                "median": float(np.median(r)),
                "sd": float(np.std(r)),
            }
        if not allr:
            return {}
        pooled_r = np.concatenate(allr)
        out = {
            "genewise_corr_within_ct_median": float(np.median(pooled_r)),
            "genewise_corr_within_ct_mean": float(np.mean(pooled_r)),
            "genewise_corr_within_ct_sd": float(np.std(pooled_r)),
            "genewise_corr_within_ct_n": int(pooled_r.size),
            "genewise_corr_within_ct_n_celltypes": int(len(per_ct)),
        }
        for q in (10, 25, 75, 90):
            out[f"genewise_corr_within_ct_q{q}"] = float(np.percentile(pooled_r, q))
        out["_per_celltype"] = per_ct
        return out

    def statistics(self) -> dict:
        """Every gate statistic, as a flat dict of scalars plus a few vectors."""
        umi = np.concatenate(self.cell_umi) if self.cell_umi else np.array([0.0])
        gdet = np.concatenate(self.cell_genes) if self.cell_genes else np.array([0])
        alpha, mean_count = self.conditional_alpha()
        ok = np.isfinite(alpha) & (alpha > 0)

        pv = self.pv_frame()
        cells_pv = pv["n_cells"].to_numpy(dtype=float) if len(pv) else np.array([1.0])

        stats: dict = {
            # --- Gate A: transcriptome occupancy ---
            "genes_detected_per_cell_mean": float(np.mean(gdet)),
            # Shape discriminator. zero_fraction is the MEAN of this same curve, so
            # a matching mean with mismatched quantiles is a SHAPE error, not a
            # level error: measured 1.22 simulated against 1.13 for TNBC.
            "genes_detected_per_cell_meanmedian": float(
                np.mean(gdet) / max(np.median(gdet), 1e-12)
            ),
            "genes_detected_per_cell_median": float(np.median(gdet)),
            "genes_detected_per_cell_q25": float(np.percentile(gdet, 25)),
            "genes_detected_per_cell_q75": float(np.percentile(gdet, 75)),
            "zero_fraction": float(1.0 - self.n_detected_cells.sum()
                                   / max(self.n_cells_total * self.n_genes, 1)),
            "gene_mean_log_mean": float(
                np.mean(np.log(mean_count[mean_count > 0]))
            ),
            "gene_mean_log_sd": float(np.std(np.log(mean_count[mean_count > 0]))),
            # --- Gate B: depth and yield ---
            "umi_per_cell_median": float(np.median(umi)),
            "umi_per_cell_q25": float(np.percentile(umi, 25)),
            "umi_per_cell_q75": float(np.percentile(umi, 75)),
            "umi_per_cell_q05": float(np.percentile(umi, 5)),
            "umi_per_cell_q95": float(np.percentile(umi, 95)),
            "cells_per_pv_median": float(np.median(cells_pv)),
            "cells_per_pv_cv": float(np.std(cells_pv) / max(np.mean(cells_pv), 1e-12)),
            "n_cells_total": int(self.n_cells_total),
            # --- Gate C: conditional dispersion ---
            "cond_alpha_median": float(np.median(alpha[ok])) if ok.any() else np.nan,
            "cond_alpha_q25": float(np.percentile(alpha[ok], 25)) if ok.any() else np.nan,
            "cond_alpha_q75": float(np.percentile(alpha[ok], 75)) if ok.any() else np.nan,
            "cond_alpha_log_sd": (
                float(np.std(np.log(alpha[ok]))) if ok.sum() > 2 else np.nan
            ),
            "cond_alpha_n_genes": int(ok.sum()),
        }

        # Mean-dependence slope of the conditional curve (log alpha ~ log mean).
        if ok.sum() > 20:
            x = np.log(mean_count[ok])
            yv = np.log(alpha[ok])
            slope, intercept = np.polyfit(x, yv, 1)
            stats["cond_alpha_slope"] = float(slope)
            stats["cond_alpha_intercept"] = float(intercept)
            # Decile curve, for the distributional figure.
            qs = np.quantile(x, np.linspace(0.05, 0.95, 10))
            binned = []
            for lo, hi in zip(qs[:-1], qs[1:]):
                m = (x >= lo) & (x < hi)
                binned.append(float(np.median(alpha[ok][m])) if m.any() else np.nan)
            stats["cond_alpha_curve_x"] = [float(v) for v in qs[:-1]]
            stats["cond_alpha_curve_y"] = binned
        else:
            stats["cond_alpha_slope"] = np.nan
            stats["cond_alpha_intercept"] = np.nan

        # --- Gate E: longitudinal structure, gene-wise ---
        stats.update(self._longitudinal_statistics(pv))
        # LATENT components, for parameterising the simulator. Reported
        # alongside the observables precisely so the two are never confused.
        stats.update(self.variance_components())
        return stats

    def _longitudinal_statistics(self, pv: pd.DataFrame) -> dict:
        """Per-gene pre/post correlation ACROSS participants, as a distribution.

        A single pooled scalar correlation can be right while the distribution is
        wrong, which is why this reports the whole distribution: the gate is
        whether the real and simulated *distributions* agree, not their medians.
        The outcome is ``log(1 + CPM)`` at the participant x visit level, i.e. the
        analysis scale, not the raw count scale.
        """
        out: dict = {}
        if len(pv) == 0 or "Pre" not in set(pv["visit"]) or "Post" not in set(pv["visit"]):
            return {
                "prepost_corr_pooled": np.nan,
                "prepost_corr_genewise_median": np.nan,
                "between_participant_sd": np.nan,
            }
        mat = np.vstack(pv["counts"].to_numpy())
        totals = mat.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        y = np.log1p(mat / totals * 1e6)

        pre_mask = (pv["visit"] == "Pre").to_numpy()
        post_mask = (pv["visit"] == "Post").to_numpy()
        pre_ids = pv.loc[pre_mask, "participant"].tolist()
        post_ids = pv.loc[post_mask, "participant"].tolist()
        common = [p for p in pre_ids if p in set(post_ids)]
        if len(common) < 3:
            return {
                "prepost_corr_pooled": np.nan,
                "prepost_corr_genewise_median": np.nan,
                "between_participant_sd": np.nan,
            }
        pre_idx = [pre_ids.index(p) for p in common]
        post_idx = [post_ids.index(p) for p in common]
        pre = y[pre_mask][pre_idx]  # n_participants x n_genes
        post = y[post_mask][post_idx]

        # Pooled: one correlation over all (participant, gene) pairs, after
        # removing the gene mean so it measures participant structure only.
        pre_c = pre - pre.mean(axis=0, keepdims=True)
        post_c = post - post.mean(axis=0, keepdims=True)
        denom = np.sqrt((pre_c**2).sum() * (post_c**2).sum())
        out["prepost_corr_pooled"] = float((pre_c * post_c).sum() / denom) if denom > 0 else np.nan

        # Gene-wise: one correlation per gene across participants.
        sd_pre = pre_c.std(axis=0)
        sd_post = post_c.std(axis=0)
        good = (sd_pre > 1e-8) & (sd_post > 1e-8)
        if self.min_cpm_genewise > 0:
            mean_cpm = 0.5 * (np.expm1(pre).mean(axis=0) + np.expm1(post).mean(axis=0))
            good &= mean_cpm >= self.min_cpm_genewise
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (pre_c * post_c).mean(axis=0) / (sd_pre * sd_post)
        r = r[good & np.isfinite(r)]
        if r.size:
            out["prepost_corr_genewise_median"] = float(np.median(r))
            out["prepost_corr_genewise_mean"] = float(np.mean(r))
            out["prepost_corr_genewise_sd"] = float(np.std(r))
            for q in (10, 25, 75, 90):
                out[f"prepost_corr_genewise_q{q}"] = float(np.percentile(r, q))
            out["prepost_corr_genewise_n"] = int(r.size)  # support size, both arms
            out["_prepost_corr_genewise"] = r  # full vector, for distribution tests
        else:
            out["prepost_corr_genewise_median"] = np.nan

        # Between-participant SD of the gene-level outcome, averaged over genes.
        both = np.concatenate([pre, post], axis=0)
        out["between_participant_sd"] = float(np.mean(both.std(axis=0)))
        # Participant-level change score SD -- the quantity the DiD actually uses.
        delta = post - pre
        out["delta_sd_median"] = float(np.median(delta.std(axis=0)))
        return out


def summarize_blocks(blocks, n_genes: int) -> SummaryAccumulator:
    """Feed an iterable of ``(counts, participant, visit, arm, stratum)`` blocks."""
    acc = SummaryAccumulator(n_genes=n_genes)
    for counts, participant, visit, arm, stratum in blocks:
        acc.add_block(counts, participant, visit, arm=arm, stratum=stratum)
    return acc


def summarize_adata(
    adata,
    participant_col: str = "participant",
    visit_col: str = "visit",
    arm_col: str | None = "arm",
    celltype_col: str | None = "cell_type",
    layer: str | None = None,
) -> SummaryAccumulator:
    """Summarise real data, conditioning on participant x visit x cell type.

    ``celltype_col=None`` conditions on participant x visit only. That is correct
    for an already-homogeneous population and WRONG for a real mixed sample: on
    TNBC it inflates the conditional dispersion 2.8x by loading between-cell-type
    mean differences onto it.
    """
    obs = adata.obs
    keys = [participant_col, visit_col]
    if celltype_col is not None and celltype_col in obs.columns:
        keys.append(celltype_col)
    acc = SummaryAccumulator(n_genes=adata.n_vars, gene_names=list(adata.var_names))

    X = adata.layers[layer] if layer is not None else adata.X
    grouped = obs.groupby(keys, observed=True).indices
    for key, idx in grouped.items():
        key = key if isinstance(key, tuple) else (key,)
        sub = X[idx]
        counts = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
        arm = str(obs[arm_col].iloc[idx[0]]) if arm_col and arm_col in obs.columns else "NA"
        acc.add_block(
            counts,
            participant=str(key[0]),
            visit=str(key[1]),
            arm=arm,
            stratum="|".join(str(k) for k in key),
        )
    return acc


def _accumulator_from_rows(rows: list, n_genes: int) -> SummaryAccumulator:
    """Rebuild an accumulator from stored per-stratum contributions."""
    acc = SummaryAccumulator(n_genes=n_genes)
    acc.pv_rows = rows
    for r in rows:
        acc.n_cells_total += int(r["n_cells"])
        acc.total_counts += r["counts"]
        acc.n_detected_cells += r["detected"]
        acc.cell_umi.append(r["umi"])
        acc.cell_genes.append(r["genes_per_cell"])
        if "resid_ss" in r:
            acc.resid_ss += r["resid_ss"]
            acc.mu_sum += r["mu_sum"]
            acc.mu2_sum += r["mu2_sum"]
            acc.n_strata_used += (r["counts"] > 0).astype(np.float64)
    return acc


def participant_bootstrap_statistics(
    accs: dict[str, SummaryAccumulator] | SummaryAccumulator,
    n_boot: int = 500,
    seed: int = 20260726,
    verbose: bool = False,
) -> list[dict]:
    """Statistics under a PARTICIPANT-level bootstrap of the reference cohort.

    This is what makes an acceptance tolerance defensible. A fixed-parameter Monte
    Carlo envelope over 141,553 cells is ~0.4% wide, so it rejects a simulator
    matching the data to 0.5% -- its null ("the simulator IS the data-generating
    process") is false by construction for anything tractable. But cells are not
    the independent calibration units. Participants are, and there are twelve.

    Resampling participants with replacement, carrying BOTH visits and all cell
    types with each, gives the sampling uncertainty of the reference cohort
    itself. A simulator whose discrepancy from TNBC is no larger than the
    discrepancy between two bootstrap realisations of TNBC cannot be
    distinguished from a second draw of the same study.
    """
    if isinstance(accs, SummaryAccumulator):
        accs = {"ALL": accs}
    rng = np.random.default_rng(seed)
    by_ct_pid: dict[str, dict[str, list]] = {}
    arm_of: dict[str, str] = {}
    for ct, acc in accs.items():
        d: dict[str, list] = {}
        for r in acc.pv_rows:
            d.setdefault(r["participant"], []).append(r)
            arm_of.setdefault(r["participant"], str(r.get("arm", "NA")))
        by_ct_pid[ct] = d
    pids = sorted({p for d in by_ct_pid.values() for p in d})

    # STRATIFY BY ARM. Resampling 12 participants indiscriminately can produce a
    # 9/3 split from a 6/6 design, so the bootstrap would carry allocation
    # imbalance that the real design does not have and the tolerance would be
    # inflated by a source of variation the study never faced. Drawing within arm
    # is the cluster-stratified bootstrap this design calls for.
    by_arm: dict[str, list[str]] = {}
    for pid in pids:
        by_arm.setdefault(arm_of.get(pid, "NA"), []).append(pid)

    out = []
    for b in range(n_boot):
        # A participant enters or leaves the cohort as a WHOLE -- both visits, all
        # cell types, all their cells -- preserving longitudinal pairing and
        # cross-cell-type dependence within an individual.
        draw = np.concatenate([
            rng.choice(members, size=len(members), replace=True)
            for _arm, members in sorted(by_arm.items())
        ])
        per_ct = []
        for ct, d in by_ct_pid.items():
            rows = []
            for j, pid in enumerate(draw):
                for r in d.get(pid, []):
                    # Relabel so a participant drawn twice is two participants.
                    rows.append({**r, "participant": f"{pid}#{j}"})
            if rows:
                per_ct.append(_accumulator_from_rows(rows, accs[ct].n_genes))
        if not per_ct:
            continue
        st = typical_celltype_targets({str(i): a for i, a in enumerate(per_ct)})
        # Carry the quantile grid, not just the scalars. Without it the
        # distributional gate silently falls back to a simulation-versus-simulation
        # reference, which reflects only Monte Carlo noise rather than the
        # reference cohort's participant-level sampling variability -- the very
        # tolerance the bootstrap exists to provide.
        _r = st.pop("_prepost_corr_genewise", None)
        if _r is not None and len(_r):
            st["corr_quantiles"] = np.percentile(
                np.asarray(_r), np.linspace(1, 99, 99)
            ).tolist()
        out.append(st)
        if verbose and (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
    return out


def summarize_adata_per_celltype(
    adata,
    participant_col: str = "participant",
    visit_col: str = "visit",
    arm_col: str | None = "arm",
    celltype_col: str = "cell_type",
    layer: str | None = None,
    min_cells: int = 2000,
    exclude: tuple[str, ...] = ("Unassigned", "unassigned", "Unknown", "Doublet", "Mixed"),
) -> dict[str, SummaryAccumulator]:
    """One accumulator per cell type, each stratified by participant x visit only.

    THE PRIMARY SIMULATOR REPRESENTS ONE HOMOGENEOUS CELL POPULATION, so every
    target must be measured at that level. Measuring on the cell-type-pooled
    sample and generating a single population is the incoherence the gates kept
    detecting.

    It also makes the estimators comparable. ``summarize_adata`` pools 242 strata
    (participant x visit x cell type) while ``summarize_simulation`` has 24, and
    the moment estimator consumes one degree of freedom per stratum, so the two
    arms carried different biases. Here each cell type has exactly 24 strata, the
    same as the simulator.
    """
    obs = adata.obs
    X = adata.layers[layer] if layer is not None else adata.X
    out: dict[str, SummaryAccumulator] = {}
    for ct, ct_idx in obs.groupby(celltype_col, observed=True).indices.items():
        # "Unassigned" is, by construction, whatever did not resolve to a marker
        # profile: a heterogeneous leftover, not a cell population. Calibrating a
        # HOMOGENEOUS-population simulator on it would reintroduce exactly the
        # mixture the conditioning exists to remove. Cell types below `min_cells`
        # are dropped for the opposite reason: too few cells per participant-visit
        # to estimate anything (Mast cell gave a correlation SD of 0.53 on 7,045
        # gene-cell-type pairs).
        if str(ct) in exclude or len(ct_idx) < min_cells:
            continue
        acc = SummaryAccumulator(n_genes=adata.n_vars, gene_names=list(adata.var_names))
        sub_obs = obs.iloc[ct_idx]
        for key, rel in sub_obs.groupby([participant_col, visit_col], observed=True).indices.items():
            key = key if isinstance(key, tuple) else (key,)
            rows = ct_idx[rel]
            block = X[rows]
            block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
            arm = (
                str(sub_obs[arm_col].iloc[rel[0]])
                if arm_col and arm_col in sub_obs.columns
                else "NA"
            )
            acc.add_block(
                block,
                participant=str(key[0]),
                visit=str(key[1]),
                arm=arm,
                stratum=f"{key[0]}|{key[1]}",
            )
        out[str(ct)] = acc
    return out


# Features defining "a typical cell population". Chosen because each one drives a
# distinct part of the generative model -- yield, depth, complexity, sparsity,
# cell-level noise, and the two hierarchy levels -- so the medoid is central in
# the space that actually matters, not in an arbitrary one.
# MINIMAL NON-REDUNDANT feature set. Each entry drives a distinct part of the
# generative model: yield, depth, complexity, cell-level noise, and the two
# hierarchy levels.
#
# `zero_fraction` was REMOVED as redundant. It is exactly
# `1 - mean(genes_detected_per_cell) / n_genes` (see `statistics`), i.e. the mean
# of the same per-cell curve whose median is already here. Including both
# double-weights transcriptome complexity against yield, depth and the hierarchy.
#
# No derived quantity is included either: `prepost_corr_latent` is a function of
# sigma_b and sigma_u, so adding it would weight the covariance structure twice.
_MEDOID_FEATURES = (
    "cells_per_pv_median",
    "umi_per_cell_median",
    "genes_detected_per_cell_median",
    "cond_alpha_median",
    "sigma_b_latent",
    "sigma_u_latent",
)


def select_medoid_celltype(accs: dict[str, SummaryAccumulator]) -> tuple[str, pd.DataFrame]:
    """Pick the REAL cell type closest to the multivariate centre.

    Taking the median of each parameter independently across cell types produces a
    vector that need correspond to no actual population: a yield from one cell
    type, a dispersion from another, a participant variance from a third. These
    quantities interact -- Gate E showed that cell count, sigma_b, sigma_u and
    measurement noise jointly determine the observed longitudinal correlation --
    so combining their marginals destroys the empirical joint structure and can
    reintroduce exactly the incoherence the within-cell-type conditioning removed.

    Reference-based simulators (muscat, scDesign3) avoid this by estimating within
    actual subpopulations rather than reducing a mixture to independently chosen
    marginals. The medoid is an ACTUAL cell type, so its parameter vector is
    guaranteed to be one nature produced.

    Returns the chosen cell type and the standardised feature table, so the choice
    is auditable rather than asserted.
    """
    per_ct = {ct: acc.statistics() for ct, acc in accs.items()}
    feat = pd.DataFrame(
        {ct: {f: st.get(f, np.nan) for f in _MEDOID_FEATURES} for ct, st in per_ct.items()}
    ).T
    usable = feat.dropna(axis=1, how="any")
    if usable.shape[1] == 0 or len(usable) == 1:
        return sorted(per_ct)[0], feat
    # ROBUST scaling: median and IQR, not mean and SD. One cell type sequenced far
    # deeper than the rest (Plasma cell, 12,905 UMI against a ~2,100 median)
    # inflates the SD of that feature and compresses every other cell type's
    # contribution toward zero, so the feature silently stops discriminating.
    med = usable.median()
    iqr = (usable.quantile(0.75) - usable.quantile(0.25)).replace(0, np.nan)
    iqr = iqr.fillna(usable.std(ddof=0)).replace(0, 1.0)
    z = (usable - med) / iqr

    def _medoid_of(frame: pd.DataFrame) -> tuple[str, np.ndarray]:
        a = frame.to_numpy()
        dist = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(axis=2)).sum(axis=1)
        return str(frame.index[int(np.argmin(dist))]), dist

    medoid, total = _medoid_of(z)

    # LEAVE-ONE-FEATURE-OUT stability. A medoid that only holds while every
    # feature is present is an artefact of the feature list, not a property of the
    # data. Recording each variant's winner makes the claim checkable.
    lofo: dict[str, str] = {}
    if z.shape[1] > 2:
        for f in z.columns:
            lofo[f"without_{f}"] = _medoid_of(z.drop(columns=[f]))[0]

    feat = feat.copy()
    feat["_total_distance"] = np.nan
    feat.loc[z.index, "_total_distance"] = total
    feat.attrs["lofo_medoid"] = lofo
    feat.attrs["lofo_stability"] = (
        float(np.mean([v == medoid for v in lofo.values()])) if lofo else np.nan
    )
    feat.attrs["features_used"] = list(z.columns)
    return medoid, feat


def celltype_range(accs: dict[str, SummaryAccumulator]) -> dict:
    """Inter-cell-type range of every statistic, for reporting only.

    NOT a calibration target. It documents how much cell types differ, which is
    the honest statement to accompany a single-population simulator.
    """
    per_ct = {ct: acc.statistics() for ct, acc in accs.items()}
    keys = sorted({k for v in per_ct.values() for k in v if not k.startswith("_")})
    out: dict = {}
    for k in keys:
        vals = np.array(
            [v[k] for v in per_ct.values() if isinstance(v.get(k), (int, float))],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f"{k}__ct_lo"] = float(np.min(vals))
            out[f"{k}__ct_hi"] = float(np.max(vals))
            out[f"{k}__ct_median"] = float(np.median(vals))
    return out


def typical_celltype_targets(accs: dict[str, SummaryAccumulator]) -> dict:
    """Statistics of the MEDOID cell type, plus the inter-cell-type range.

    The returned scalars are one real population's parameter vector, not a
    componentwise median; see :func:`select_medoid_celltype`.
    """
    medoid, feat = select_medoid_celltype(accs)
    out = dict(accs[medoid].statistics())
    out.update(celltype_range(accs))
    out["anchor_celltype"] = medoid
    out["n_celltypes_available"] = len(accs)
    out["celltypes_available"] = sorted(accs)
    out["medoid_feature_table"] = feat.to_dict()
    out["medoid_features_used"] = feat.attrs.get("features_used")
    out["medoid_lofo"] = feat.attrs.get("lofo_medoid")
    out["medoid_lofo_stability"] = feat.attrs.get("lofo_stability")
    return out


def summarize_simulation(cfg) -> SummaryAccumulator:
    """Summarise one simulated replicate through the identical statistic code.

    Consumes :func:`sctrial.benchmark.simulator_v2.iter_pv_blocks`, so no
    replicate is ever materialised in full and the simulated arm of every gate is
    computed by exactly the same code as the real arm.
    """
    from .simulator_v2 import iter_pv_blocks

    acc = SummaryAccumulator(n_genes=cfg.n_genes_transcriptome)
    for blk in iter_pv_blocks(cfg):
        acc.add_block(
            blk["counts"],
            participant=blk["participant"],
            visit=blk["visit"],
            arm=blk["arm"],
            stratum=f"{blk['participant']}|{blk['visit']}",
        )
    return acc


# ---------------------------------------------------------------------------
# Generating-parameter estimation: Cox-Reid adjusted profile-likelihood GP MLE
# ---------------------------------------------------------------------------


def _fit_group_means(y, s, starts, ends, alpha, n_iter=4):
    """ML group rates ``lambda_k`` for fixed ``alpha`` (Newton, vectorised over genes).

    Solves ``sum_c (y_c - s_c lam) / (1 + alpha s_c lam) = 0`` per (gene, group).
    The Poisson estimate ``sum(y)/sum(s)`` is the alpha -> 0 solution and is used
    as the start, so a handful of iterations suffice.
    """
    n_groups = len(starts)
    n_genes = y.shape[1]
    lam = np.empty((n_groups, n_genes))
    for k, (a, b) in enumerate(zip(starts, ends)):
        ys, ss = y[a:b], s[a:b, None]
        s_sum = ss.sum()
        lam_k = ys.sum(axis=0) / max(s_sum, 1e-12)
        if alpha > 0:
            for _ in range(n_iter):
                denom = 1.0 + alpha * ss * lam_k[None, :]
                f = ((ys - ss * lam_k[None, :]) / denom).sum(axis=0)
                fp = -(ss * (1.0 + alpha * ys) / denom**2).sum(axis=0)
                step = np.where(np.abs(fp) > 1e-12, f / fp, 0.0)
                lam_k = np.maximum(lam_k - step, 1e-12)
        lam[k] = lam_k
    return lam


def _apl(y, s, starts, ends, lam, alpha):
    """Cox-Reid adjusted profile log-likelihood, per gene.

    The CR term ``-0.5 * sum_k log(sum_c w_ck)`` corrects the bias from having
    estimated one mean per stratum. With strata of tens of cells that bias is the
    dominant error in a naive dispersion estimate, which is precisely why a raw
    moment estimator is not acceptable for the generating parameter.
    """
    from scipy.special import gammaln

    inv_a = 1.0 / alpha
    ll = np.zeros(y.shape[1])
    cr = np.zeros(y.shape[1])
    for k, (a, b) in enumerate(zip(starts, ends)):
        ys, ss = y[a:b], s[a:b, None]
        mu = ss * lam[k][None, :]
        mu = np.maximum(mu, 1e-12)
        ll += (
            gammaln(ys + inv_a)
            - gammaln(inv_a)
            - gammaln(ys + 1.0)
            + ys * np.log(alpha * mu)
            - (ys + inv_a) * np.log1p(alpha * mu)
        ).sum(axis=0)
        w = (mu / (1.0 + alpha * mu)).sum(axis=0)
        cr += 0.5 * np.log(np.maximum(w, 1e-12))
    return ll - cr


@dataclass
class DispersionFit:
    """Result of :func:`conditional_dispersion`."""

    gene: np.ndarray
    alpha_mle: np.ndarray
    alpha_shrunk: np.ndarray
    mean_count: np.ndarray
    trend_slope: float
    trend_intercept: float
    trend_anchor: float
    prior_var: float
    n_genes_used: int
    n_genes_total: int
    n_strata: int

    def summary(self) -> dict:
        ok = np.isfinite(self.alpha_shrunk) & (self.alpha_shrunk > 0)
        a = self.alpha_shrunk[ok]
        return {
            "dispersion_median": float(np.median(a)),
            "dispersion_mean_slope": float(self.trend_slope),
            # TOTAL spread of log alpha, reported for description only. The
            # GENERATING quantity is the residual sd about the trend; exporting the
            # total and consuming it as the residual double-counts trend variance.
            "dispersion_log_sd_total": float(np.std(np.log(a))),
            "dispersion_residual_sd": float(np.sqrt(self.prior_var)),
            # The log-rate the trend is anchored at. Must travel WITH the median,
            # which is computed over the estimable genes only; anchoring elsewhere
            # shifts every dispersion by exp(slope * offset).
            "dispersion_anchor": float(self.trend_anchor),
            "dispersion_trend_intercept": float(self.trend_intercept),
            "dispersion_mle_median": float(
                np.median(self.alpha_mle[np.isfinite(self.alpha_mle)])
            ),
            "dispersion_prior_var": float(self.prior_var),
            "dispersion_n_genes": int(self.n_genes_used),
            "dispersion_n_genes_total": int(self.n_genes_total),
            "dispersion_n_strata": int(self.n_strata),
            "dispersion_estimator": "cox-reid APL gamma-Poisson MLE + EB shrinkage",
        }


def conditional_dispersion(
    adata,
    participant_col: str = "participant",
    visit_col: str = "visit",
    celltype_col: str | None = "cell_type",
    layer: str | None = None,
    alpha_grid: np.ndarray | None = None,
    gene_chunk: int = 200,
    verbose: bool = True,
) -> DispersionFit:
    """Gamma-Poisson dispersion MLE within homogeneous strata, with EB shrinkage.

    The model is ``Y_cg ~ NB2(mu_cg = s_c * lambda_kg, alpha_g)`` where ``k`` is
    the stratum (participant x visit x cell type) and ``s_c`` the library-size
    offset. ``alpha_g`` is maximised over a log-spaced grid on the Cox-Reid
    adjusted profile likelihood, refined by parabolic interpolation, then shrunk
    toward a mean-dependent trend with an empirical-Bayes weight.

    Conditioning on cell type is not optional for real mixed samples: without it
    the estimate absorbs between-cell-type mean differences (TNBC: 0.774 pooled
    versus 0.275 within cell type).
    """
    obs = adata.obs
    keys = [participant_col, visit_col]
    if celltype_col is not None and celltype_col in obs.columns:
        keys.append(celltype_col)
    groups = obs.groupby(keys, observed=True).indices
    order, starts, ends = [], [], []
    pos = 0
    for idx in groups.values():
        if len(idx) < 2:
            continue  # no residual degrees of freedom
        order.append(np.sort(np.asarray(idx)))
        starts.append(pos)
        pos += len(idx)
        ends.append(pos)
    if not order:
        raise ValueError("no stratum has >= 2 cells; check the grouping columns")
    order = np.concatenate(order)
    n_strata = len(starts)

    X = adata.layers[layer] if layer is not None else adata.X
    # Column slicing dominates this function (one slice per gene chunk) and is
    # O(nnz) on a CSR matrix. Row slicing is needed once, for the per-stratum
    # detection counts. Keep the right layout for each rather than paying the
    # conversion inside the loop.
    import scipy.sparse as _sp

    X_csc = X.tocsc() if _sp.issparse(X) else X
    lib = np.asarray(X.sum(axis=1)).ravel()[order].astype(np.float64)
    # Offsets normalised WITHIN stratum, matching the SummaryAccumulator model.
    s = np.empty_like(lib)
    for a, b in zip(starts, ends):
        m = lib[a:b].mean()
        s[a:b] = lib[a:b] / (m if m > 0 else 1.0)

    n_genes_total = adata.n_vars
    mean_count = np.asarray(X.sum(axis=0)).ravel() / adata.n_obs
    detected_strata = np.zeros(n_genes_total)
    for a, b in zip(starts, ends):
        sub = X[order[a:b]]
        blk = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
        detected_strata += (blk.sum(axis=0) > 0).astype(float)
    del blk

    estimable = (mean_count >= _MIN_MEAN_COUNT) & (detected_strata >= _MIN_STRATA_DETECTED)
    gene_idx = np.flatnonzero(estimable)
    if gene_idx.size == 0:
        raise ValueError("no gene passes the estimability filter")

    if alpha_grid is None:
        alpha_grid = np.exp(np.linspace(np.log(1e-3), np.log(50.0), 40))
    log_grid = np.log(alpha_grid)

    alpha_mle = np.full(n_genes_total, np.nan)
    for c0 in range(0, gene_idx.size, gene_chunk):
        cols = gene_idx[c0 : c0 + gene_chunk]
        sub = X_csc[:, cols]
        y = (sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub))[order].astype(
            np.float64
        )
        apl = np.empty((len(alpha_grid), y.shape[1]))
        for ai, a_val in enumerate(alpha_grid):
            lam = _fit_group_means(y, s, starts, ends, a_val)
            apl[ai] = _apl(y, s, starts, ends, lam, a_val)
        best = np.argmax(apl, axis=0)
        # Parabolic refinement in log-alpha; falls back to the grid point at edges.
        est = log_grid[best].copy()
        interior = (best > 0) & (best < len(alpha_grid) - 1)
        if interior.any():
            j = np.flatnonzero(interior)
            b0 = best[j]
            f0 = apl[b0 - 1, j]
            f1 = apl[b0, j]
            f2 = apl[b0 + 1, j]
            denom = f0 - 2 * f1 + f2
            step = np.where(np.abs(denom) > 1e-12, 0.5 * (f0 - f2) / denom, 0.0)
            step = np.clip(step, -1.0, 1.0)
            h = log_grid[1] - log_grid[0]
            est[j] = log_grid[b0] + step * h
        alpha_mle[cols] = np.exp(est)
        if verbose:
            print(
                f"  dispersion MLE: {min(c0 + gene_chunk, gene_idx.size)}/{gene_idx.size} genes",
                flush=True,
            )

    # --- empirical-Bayes shrinkage toward a mean-dependent trend ---
    ok = np.isfinite(alpha_mle) & (alpha_mle > 0) & (mean_count > 0)
    x = np.log(mean_count[ok])
    la = np.log(alpha_mle[ok])
    slope, intercept = np.polyfit(x, la, 1)
    resid = la - (intercept + slope * x)
    # Sampling variance of log-alpha under the APL, approximated by the standard
    # NB result 2/df with df = (n_cells - n_strata). Anything left over is prior
    # spread, exactly as in the DESeq2/glmGamPoi shrinkage construction.
    # PER-GENE sampling variance. A single pooled `2/df` with df = n_cells -
    # n_strata gives 2/141311 = 1.4e-05 against a prior variance of 1.75, i.e.
    # w = 0.9999919 -- the shrinkage was a numerical no-op, and the "shrunk"
    # estimates equalled the raw MLEs to six decimal places. The information a
    # gene carries about its dispersion is governed by its COUNTS, not by the
    # number of cells: a gene at 0.02 counts/cell is nearly uninformative however
    # many cells there are.
    #
    # Var(log alpha_hat) ~= 2/df * (1 + 1/(alpha*mu))^2 for a Gamma-Poisson, which
    # collapses to the classical 2/df only when alpha*mu >> 1.
    df = max(len(order) - n_strata, 1)
    a_hat = alpha_mle[ok]
    mu_g = mean_count[ok]
    samp_var_g = (2.0 / df) * (1.0 + 1.0 / np.maximum(a_hat * mu_g * df, 1e-12)) ** 2
    samp_var_g = np.clip(samp_var_g, 1e-8, 1e4)
    prior_var = max(float(np.var(resid) - np.median(samp_var_g)), 1e-6)
    w = prior_var / (prior_var + samp_var_g)
    alpha_shrunk = np.full(n_genes_total, np.nan)
    alpha_shrunk[ok] = np.exp((intercept + slope * x) + w * resid)

    names = np.asarray(adata.var_names)
    return DispersionFit(
        gene=names,
        alpha_mle=alpha_mle,
        alpha_shrunk=alpha_shrunk,
        mean_count=mean_count,
        trend_slope=float(slope),
        trend_intercept=float(intercept),
        trend_anchor=float(np.median(x)),
        prior_var=float(prior_var),
        n_genes_used=int(ok.sum()),
        n_genes_total=int(n_genes_total),
        n_strata=n_strata,
    )


# ---------------------------------------------------------------------------
# Target measurement (run once, on the real data)
# ---------------------------------------------------------------------------


def measure_targets(
    adata,
    participant_col: str = "participant",
    visit_col: str = "visit",
    arm_col: str | None = "arm",
    celltype_col: str | None = "cell_type",
    layer: str | None = None,
    out_json: str | Path | None = None,
    out_npz: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Measure every simulator target from real data and write the canonical files.

    ALL TARGETS ARE MEASURED WITHIN CELL TYPE. The primary simulator represents one
    homogeneous cell population, so cell-level dispersion, participant and
    participant-by-visit variance, cells per participant-visit, gene rates and the
    longitudinal covariance must all be conditioned at that level. Mixing
    conditional dispersion with pooled cell counts and pooled longitudinal
    covariance is the incoherence the calibration gates kept detecting.

    Scalar targets are the MEDIAN across cell types (a typical cell type), with the
    inter-cell-type range recorded. Cell-type-pooled values are also written, with
    a ``pooled_`` prefix, for description only -- they are what a whole-sample
    analysis would see and are NOT calibration targets.
    """
    if verbose:
        print(f"summarising {adata.n_obs:,} cells x {adata.n_vars:,} genes", flush=True)

    accs = summarize_adata_per_celltype(
        adata,
        participant_col=participant_col,
        visit_col=visit_col,
        arm_col=arm_col,
        celltype_col=celltype_col or "cell_type",
        layer=layer,
    )
    if not accs:
        raise ValueError(
            f"no cell type has enough cells; check {celltype_col!r}. The primary "
            "simulator is calibrated within cell type by design."
        )
    if verbose:
        print(f"cell types available: {sorted(accs)}", flush=True)
    stats = typical_celltype_targets(accs)
    # Convert the gene-wise correlation vector to the SAME fixed quantile grid the
    # simulated arm reports, before dropping it. Without this the observed arm has
    # no grid, the distributional gate has nothing to compare against, and it
    # silently returns INSUFFICIENT -- a gate that never runs looks the same as a
    # gate that passes in a summary count.
    _r = stats.pop("_prepost_corr_genewise", None)
    if _r is not None and len(_r):
        stats["corr_quantiles"] = np.percentile(
            np.asarray(_r), np.linspace(1, 99, 99)
        ).tolist()
    anchor = stats["anchor_celltype"]
    if verbose:
        print(f"ANCHOR cell type (multivariate medoid): {anchor}", flush=True)

    # Cell-type-pooled statistics, for DESCRIPTION only.
    pooled_acc = summarize_adata(
        adata,
        participant_col=participant_col,
        visit_col=visit_col,
        arm_col=arm_col,
        celltype_col=celltype_col,
        layer=layer,
    )
    pooled = pooled_acc.statistics()
    for k, v in pooled.items():
        if not k.startswith("_") and isinstance(v, (int, float)):
            stats[f"pooled_{k}"] = float(v)

    # Dispersion from the ANCHOR cell type alone. Fitting across all cell types
    # would pool 242 strata against the simulator's 24, and the moment estimator
    # consumes one degree of freedom per stratum, so the two arms would carry
    # different biases -- the estimator asymmetry that made Gate C fail on a ~7%
    # artifact. Restricting to the anchor makes it 24 against 24.
    if verbose:
        print(f"fitting conditional Gamma-Poisson dispersion on {anchor!r} "
              "(Cox-Reid APL)", flush=True)
    anchor_mask = (adata.obs[celltype_col or "cell_type"].astype(str) == anchor).to_numpy()
    fit = conditional_dispersion(
        adata[anchor_mask],
        participant_col=participant_col,
        visit_col=visit_col,
        celltype_col=None,  # already one cell type: strata are participant x visit
        layer=layer,
        verbose=verbose,
    )
    stats.update(fit.summary())

    stats["celltype_conditioning_ratio"] = (
        float(stats["cond_alpha_median"] / stats["pooled_cond_alpha_median"])
        if stats.get("pooled_cond_alpha_median")
        else np.nan
    )

    # --- empirical pools: ALL from the anchor cell type ---
    # Every pool must come from the SAME population as the scalars. Concatenating
    # across cell types would rebuild the mixture: a library-size distribution
    # from eleven populations paired with one population's dispersion is the
    # componentwise-median problem in another guise.
    anchor_acc = accs[anchor]
    umi = np.concatenate(anchor_acc.cell_umi).astype(float)
    log_umi = np.log(umi[umi > 0])
    stats["lib_log_mean"] = float(log_umi.mean())
    stats["lib_log_sd"] = float(log_umi.std())

    cells_pv = anchor_acc.pv_frame()["n_cells"].to_numpy(dtype=float)
    stats["cells_per_pv_mean"] = float(cells_pv.mean())
    stats["cells_per_pv_cv"] = float(cells_pv.std() / cells_pv.mean())
    stats["cells_per_pv_min"] = int(cells_pv.min())
    stats["cells_per_pv_max"] = int(cells_pv.max())

    # Gene-rate profile of the ANCHOR cell type. The pooled profile is a
    # composition-weighted mixture of eleven populations and would reintroduce
    # the heterogeneity the one-population design excludes.
    prof = anchor_acc.total_counts / anchor_acc.total_counts.sum()
    gene_mean_count = prof * float(np.mean(umi))

    # PAIRED STRUCTURE of the anchor population, recorded because it is NOT the
    # nominal design. In TNBC one participant (P016) contributes no Treg cells at
    # Post, so the anchor has 23 participant-visits and 11 fully paired
    # participants against a nominal 24 and 12. Every longitudinal target -- the
    # variance components, the pre/post correlations -- is therefore estimated on
    # 11 pairs. Describing the simulator's 24 balanced strata as "like-for-like"
    # with this would be wrong, and the arm split of the retained participants
    # decides whether an arm-stratified bootstrap is even balanced.
    _pv = anchor_acc.pv_frame()
    _by_p = _pv.groupby("participant")["visit"].nunique()
    stats["n_participant_visits"] = int(len(_pv))
    stats["n_participants_any_visit"] = int(len(_by_p))
    stats["n_participants_paired"] = int((_by_p == 2).sum())
    _paired_ids = set(_by_p[_by_p == 2].index)
    _arms = (
        _pv[_pv["participant"].isin(_paired_ids)]
        .drop_duplicates("participant")["arm"]
        .value_counts()
        .to_dict()
    )
    stats["paired_participants_by_arm"] = {str(k): int(v) for k, v in _arms.items()}
    stats["n_genes_transcriptome"] = int(adata.n_vars)
    stats["n_participants"] = int(
        adata.obs[participant_col].nunique()
    )
    stats["calibration_level"] = "within_cell_type_medoid_anchor"
    stats["anchor_celltype"] = anchor
    stats["source"] = "sctrial.benchmark.calibration.measure_targets"

    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump({k: _jsonable(v) for k, v in stats.items()}, fh, indent=2)
        if verbose:
            print(f"wrote {out_json}", flush=True)
    if out_npz is not None:
        out_npz = Path(out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            library_sizes=umi.astype(np.float32),
            cells_per_pv=cells_pv.astype(np.float32),
            cond_alpha=fit.alpha_shrunk.astype(np.float32),
            cond_alpha_mle=fit.alpha_mle.astype(np.float32),
            gene_mean_count=gene_mean_count.astype(np.float32),
            calibration_level=np.array(["within_cell_type_medoid_anchor"]),
            anchor_celltype=np.array([anchor]),
        )
        if verbose:
            print(f"wrote {out_npz}", flush=True)
    return stats


def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
