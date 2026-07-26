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
        y = counts.astype(np.float64, copy=False)

        lib = y.sum(axis=1)
        self.cell_umi.append(lib.astype(np.float32))
        self.cell_genes.append((y > 0).sum(axis=1).astype(np.int32))
        self.n_cells_total += n_cells

        gene_sum = y.sum(axis=0)
        self.total_counts += gene_sum
        self.n_detected_cells += (y > 0).sum(axis=0)

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
        sum_a = y[half].sum(axis=0) if half.any() else np.zeros(self.n_genes)
        sum_b = y[~half].sum(axis=0) if (~half).any() else np.zeros(self.n_genes)

        # Participant-visit pseudobulk (summed counts) for the longitudinal gate.
        self.pv_rows.append(
            {
                "participant": participant,
                "visit": visit,
                "arm": arm,
                "stratum": stratum if stratum is not None else f"{participant}|{visit}",
                "n_cells": n_cells,
                "counts": gene_sum.astype(np.float64),
                "counts_a": sum_a.astype(np.float64),
                "counts_b": sum_b.astype(np.float64),
                "n_cells_a": int(half.sum()),
                "n_cells_b": int((~half).sum()),
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
        ys_sum = y.T @ s  # sum(y * s) per gene
        y2_sum = (y * y).sum(axis=0)

        resid = y2_sum - 2.0 * lam * ys_sum + (lam**2) * s2_sum
        self.resid_ss += resid
        self.mu_sum += lam * s_sum
        self.mu2_sum += (lam**2) * s2_sum
        self.n_strata_used += (gene_sum > 0).astype(np.float64)

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

    def variance_components(self, min_cpm: float = 10.0) -> dict:
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
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (pre_c * post_c).mean(axis=0) / (sd_pre * sd_post)
        r = r[good & np.isfinite(r)]
        if r.size:
            out["prepost_corr_genewise_median"] = float(np.median(r))
            out["prepost_corr_genewise_mean"] = float(np.mean(r))
            out["prepost_corr_genewise_sd"] = float(np.std(r))
            for q in (10, 25, 75, 90):
                out[f"prepost_corr_genewise_q{q}"] = float(np.percentile(r, q))
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
            "dispersion_log_sd": float(np.std(np.log(a))),
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
        sub = X[:, cols][order]
        y = (sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)).astype(np.float64)
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
    df = max(len(order) - n_strata, 1)
    samp_var = 2.0 / df
    prior_var = max(float(np.var(resid) - samp_var), 1e-6)
    w = prior_var / (prior_var + samp_var)
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

    Writes ``tnbc_sim_targets.json`` (the numbers the simulator is configured
    from and validated against) and ``tnbc_empirical.npz`` (the nuisance pools --
    per-cell library sizes and cells per participant-visit -- which are resampled
    rather than fitted, because a parametric fit reproduced the library-size
    median +34% and Q75 +53% off).
    """
    if verbose:
        print(f"summarising {adata.n_obs:,} cells x {adata.n_vars:,} genes", flush=True)
    acc = summarize_adata(
        adata,
        participant_col=participant_col,
        visit_col=visit_col,
        arm_col=arm_col,
        celltype_col=celltype_col,
        layer=layer,
    )
    stats = acc.statistics()
    stats.pop("_prepost_corr_genewise", None)

    if verbose:
        print("fitting conditional Gamma-Poisson dispersion (Cox-Reid APL)", flush=True)
    fit = conditional_dispersion(
        adata,
        participant_col=participant_col,
        visit_col=visit_col,
        celltype_col=celltype_col,
        layer=layer,
        verbose=verbose,
    )
    stats.update(fit.summary())

    # Marginal (cell-type-pooled) dispersion, REPORTED FOR CONTEXT ONLY. It is not
    # a calibration target: the simulator has no cell types to pool, so requiring
    # the simulated marginal to match this one would force a latent parameter to
    # absorb heterogeneity the model does not contain.
    marg = summarize_adata(
        adata,
        participant_col=participant_col,
        visit_col=visit_col,
        arm_col=arm_col,
        celltype_col=None,
        layer=layer,
    )
    m_alpha, m_mean = marg.conditional_alpha()
    m_ok = np.isfinite(m_alpha) & (m_alpha > 0)
    stats["marginal_alpha_median_CONTEXT_ONLY"] = (
        float(np.median(m_alpha[m_ok])) if m_ok.any() else np.nan
    )
    stats["celltype_conditioning_ratio"] = (
        float(stats["cond_alpha_median"] / stats["marginal_alpha_median_CONTEXT_ONLY"])
        if m_ok.any()
        else np.nan
    )

    # --- library-size and cells-per-pv fits (fallback parameters) ---
    umi = np.concatenate(acc.cell_umi).astype(np.float64)
    log_umi = np.log(umi[umi > 0])
    stats["lib_log_mean"] = float(log_umi.mean())
    stats["lib_log_sd"] = float(log_umi.std())
    pv = acc.pv_frame()
    cells_pv = pv["n_cells"].to_numpy(dtype=float)
    stats["cells_per_pv_mean"] = float(cells_pv.mean())
    stats["cells_per_pv_cv"] = float(cells_pv.std() / cells_pv.mean())
    stats["cells_per_pv_min"] = int(cells_pv.min())
    stats["cells_per_pv_max"] = int(cells_pv.max())
    stats["n_genes_transcriptome"] = int(adata.n_vars)
    stats["n_participants"] = int(pv["participant"].nunique())
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
            gene_mean_count=fit.mean_count.astype(np.float32),
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
