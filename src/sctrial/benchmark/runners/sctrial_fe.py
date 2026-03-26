"""sctrial DiD (Fixed Effects) runner for benchmarking."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _run_from_pseudobulk(pb: pd.DataFrame, gene_cols: list[str]) -> dict:
    """Run OLS DiD with participant FE on pre-aggregated pseudobulk.

    This operates on log-transformed pseudobulk means so that betas are on
    a log-expression scale comparable to edgeR/limma/dreamlet log-fold-changes.

    Model per gene:  Y_it = α_i + β₁·Post_t + β₂·(Treat_i × Post_t) + ε_it
    where Y_it is log1p(mean expression) for participant i at visit t.
    β₂ is the DiD interaction coefficient.
    """
    import statsmodels.api as sm

    out = {}
    # Build design columns
    df = pb.copy()
    df["post"] = (df["visit"] == "Post").astype(int)
    df["treated"] = (df["arm"] == "Treated").astype(int)
    df["interaction"] = df["post"] * df["treated"]

    # Participant dummies for fixed effects
    participants = df["participant"].unique()
    for p in participants[1:]:  # drop first for identification
        df[f"fe_{p}"] = (df["participant"] == p).astype(int)
    fe_cols = [c for c in df.columns if c.startswith("fe_")]

    for gene in gene_cols:
        try:
            y = df[gene].values
            if np.all(np.isnan(y)) or np.std(y) == 0:
                raise ValueError("No variance")

            X = df[["post", "interaction"] + fe_cols].copy()
            X = sm.add_constant(X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n_participants = df["participant"].nunique()
                # With few participants, cluster-robust SEs are too
                # conservative (fewer clusters than parameters). Use HC1
                # (heteroscedasticity-robust) SEs instead when clusters
                # are small.  With many participants, cluster-robust is
                # preferred.
                if n_participants <= 20:
                    model = sm.OLS(y, X, missing="drop").fit(
                        cov_type="HC1",
                    )
                else:
                    model = sm.OLS(y, X, missing="drop").fit(
                        cov_type="cluster",
                        cov_kwds={"groups": df["participant"].values},
                    )

            beta = model.params.get("interaction", np.nan)
            pval = model.pvalues.get("interaction", np.nan)
            ci = model.conf_int().loc["interaction"]
            out[gene] = {
                "beta": beta,
                "pvalue": pval,
                "ci_lo": ci.iloc[0],
                "ci_hi": ci.iloc[1],
                "converged": True,
                "failure_mode": None,
            }
        except Exception as exc:
            logger.debug("sctrial_fe gene %s failed: %s", gene, exc)
            out[gene] = {
                "beta": np.nan,
                "pvalue": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "converged": False,
                "failure_mode": "numerical",
            }

    return out


def run(
    data,
    gene_cols: list[str],
    design_kwargs: dict | None = None,
    visits: tuple[str, str] = ("Pre", "Post"),
    from_pseudobulk: bool = False,
) -> dict[str, dict]:
    """Run sctrial DiD with participant FE + cluster-robust SE.

    Parameters
    ----------
    data : AnnData or DataFrame
        Cell-level AnnData (from_pseudobulk=False) or log-pseudobulk
        DataFrame (from_pseudobulk=True).
    gene_cols : list[str]
        Gene names to test.
    design_kwargs : dict, optional
        Override TrialDesign parameters (only used with AnnData path).
    visits : tuple
        (pre, post) visit labels.
    from_pseudobulk : bool
        If True, data is a pseudobulk DataFrame with columns:
        participant, arm, visit, gene_0, gene_1, ...

    Returns
    -------
    dict : gene → {"beta", "pvalue", "ci_lo", "ci_hi", "converged", "failure_mode"}
    """
    if from_pseudobulk:
        return _run_from_pseudobulk(data, gene_cols)

    # Original cell-level path via sctrial.stats.did
    from sctrial.design import TrialDesign
    from sctrial.stats.did import did_table

    dk = {
        "participant_col": "participant",
        "visit_col": "visit",
        "arm_col": "arm",
        "arm_treated": "Treated",
        "arm_control": "Control",
    }
    if design_kwargs:
        dk.update(design_kwargs)

    design = TrialDesign(**dk)

    out = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = did_table(
                data,
                gene_cols,
                design,
                visits=visits,
                aggregate="cell",
                standardize=False,
            )

        for _, row in res.iterrows():
            out[row["feature"]] = {
                "beta": row["beta_DiD"],
                "pvalue": row["p_DiD"],
                "ci_lo": row.get("ci_lo_DiD", np.nan),
                "ci_hi": row.get("ci_hi_DiD", np.nan),
                "converged": True,
                "failure_mode": None,
            }
    except Exception as exc:
        logger.warning("sctrial_fe failed: %s", exc)
        for g in gene_cols:
            out[g] = {
                "beta": np.nan,
                "pvalue": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "converged": False,
                "failure_mode": "numerical",
            }

    # Fill missing genes
    for g in gene_cols:
        if g not in out:
            out[g] = {
                "beta": np.nan,
                "pvalue": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "converged": False,
                "failure_mode": "numerical",
            }

    return out
