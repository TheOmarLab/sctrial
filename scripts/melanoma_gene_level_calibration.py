"""Rejection rates of the gene-level DiD tests in the melanoma cohort under relabeling.

The eleven paired participants are reassigned to groups of three and eight in all 165 ways.
For every gene of the panel shown in Supplementary Figure 7 (the 2,000 highly variable genes
with a finite test), ``did_fit`` gives the analytical p-value with the cluster-robust and with
the homoskedastic covariance. Under reassignment the null hypothesis holds, so a calibrated
test rejects at its nominal level. With ``--bootstrap`` the wild cluster bootstrap p-value
(999 draws) is added.

Relabeling makes the two groups exchangeable, so it cannot show whether the changes vary
equally in the two observed groups, which the homoskedastic test assumes. The script
therefore also compares, gene by gene, the variance of the participant changes in the
responder and the non-responder group with its distribution under equal variances, and the
homoskedastic test with the Welch test on the same changes.

The results are the evidence behind ``MELANOMA_GENE_LEVEL_COV_TYPE`` in
``manuscript_figures/_shared.py``.

Run from the repository root on a compute node, with ``SCTRIAL_MANUSCRIPT_DIR`` set:

    python scripts/melanoma_gene_level_calibration.py <out_dir> [--bootstrap] [--n-jobs N]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.05, 0.01, 0.001, 1e-4, 1e-5)


def variance_comparison(frame: pd.DataFrame, unit: str, time: str, arm_bin: str,
                        genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variance of the participant changes by group, with the pooled and the Welch test."""
    ordered = frame.sort_values([unit, time])
    pre = ordered[ordered[time] == 0].set_index(unit)
    post = ordered[ordered[time] == 1].set_index(unit)
    if list(pre.index) != list(post.index):
        raise ValueError("participants of the two visits do not match")
    change = post[genes].to_numpy(dtype=float) - pre[genes].to_numpy(dtype=float)
    focal = pre[arm_bin].to_numpy() == 1
    first, second = change[focal], change[~focal]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = first.var(axis=0, ddof=1) / second.var(axis=0, ddof=1)
    per_gene = pd.DataFrame({
        "gene": genes,
        "variance_ratio": ratio,
        "p_pooled": stats.ttest_ind(first, second, equal_var=True).pvalue,
        "p_welch": stats.ttest_ind(first, second, equal_var=False).pvalue,
    })
    usable = per_gene[np.isfinite(per_gene["variance_ratio"]) & (per_gene["variance_ratio"] > 0)]
    reference = stats.f(len(first) - 1, len(second) - 1)
    flagged = usable[usable["p_pooled"] < 0.05]
    summary = pd.DataFrame([{
        "n_genes": len(per_gene),
        "n_genes_both_variances_positive": len(usable),
        "median_variance_ratio": float(usable["variance_ratio"].median()),
        "median_under_equal_variances": float(reference.median()),
        "pct_ratio_above_4": 100 * float((usable["variance_ratio"] > 4).mean()),
        "pct_above_4_under_equal_variances": 100 * float(reference.sf(4)),
        "pct_ratio_below_quarter": 100 * float((usable["variance_ratio"] < 0.25).mean()),
        "pct_below_quarter_under_equal_variances": 100 * float(reference.cdf(0.25)),
        "n_p05_pooled": int((per_gene["p_pooled"] < 0.05).sum()),
        "n_p05_welch": int((per_gene["p_welch"] < 0.05).sum()),
        "n_p05_both": int(((per_gene["p_pooled"] < 0.05) & (per_gene["p_welch"] < 0.05)).sum()),
        "median_variance_ratio_of_pooled_p05": float(flagged["variance_ratio"].median()),
        "pct_ratio_above_4_of_pooled_p05": 100 * float((flagged["variance_ratio"] > 4).mean()),
    }])
    return per_gene, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rejection rates of the melanoma gene-level DiD tests under relabeling."
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    import scanpy as sc
    from joblib import Parallel, delayed

    from manuscript_figures import _shared as shared
    from sctrial import TrialDesign, did_fit, did_table, get_did_aggregated_df, harmonize_response

    adata = shared.get_sade_feldman()
    if "log1p_tpm" not in adata.layers:
        adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
    adata = harmonize_response(adata)
    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="response_harmonized",
        arm_treated="Responder",
        arm_control="Non-responder",
    )
    visits = ("Pre", "Post")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, layer="log1p_tpm", flavor="seurat")
    genes = adata.var_names[adata.var["highly_variable"]].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        observed = did_table(
            adata, features=genes, design=design, visits=visits, layer="log1p_tpm",
            standardize=True, aggregate="participant_visit",
            cov_type=shared.MELANOMA_GENE_LEVEL_COV_TYPE,
        )
        frame, unit, time, arm_bin = get_did_aggregated_df(
            adata, genes, design, visits, layer="log1p_tpm", aggregate="participant_visit"
        )
    tested = observed.loc[observed["p_DiD"].notna(), "feature"].tolist()
    frame[unit] = frame[unit].astype(str)
    units = sorted(frame[unit].unique())
    observed_first = tuple(sorted(frame.loc[frame[arm_bin] == 1, unit].unique()))
    assignments = list(combinations(units, len(observed_first)))
    print(
        f"genes {len(genes)}, tested {len(tested)}, participants {len(units)}, "
        f"assignments {len(assignments)} (groups of {len(observed_first)} and "
        f"{len(units) - len(observed_first)})"
    )
    per_gene, variance_summary = variance_comparison(frame, unit, time, arm_bin, tested)
    per_gene.to_csv(args.out_dir / "melanoma_gene_variance_ratio.csv", index=False)
    variance_summary.to_csv(args.out_dir / "melanoma_gene_variance_ratio_summary.csv", index=False)
    print(variance_summary.T.to_string())
    base = frame[[unit, time, *tested]]
    tests = ("cluster", "nonrobust", "bootstrap") if args.bootstrap else ("cluster", "nonrobust")

    def one_gene(gene: str) -> list[dict]:
        rows = []
        data = base[[unit, time, gene]].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index, first in enumerate(assignments):
                data["arm_bin"] = data[unit].isin(first).astype(int)
                row = {"gene": gene, "assignment": index, "observed": first == observed_first}
                for cov in ("cluster", "nonrobust"):
                    fit = did_fit(data, y=gene, unit=unit, time=time, arm_bin="arm_bin",
                                  cov_type=cov, standardize=True)
                    row[f"p_{cov}"] = fit["p_DiD"]
                if args.bootstrap:
                    fit = did_fit(data, y=gene, unit=unit, time=time, arm_bin="arm_bin",
                                  cov_type="cluster", standardize=True,
                                  use_bootstrap=True, n_boot=999)
                    row["p_bootstrap"] = fit.get("p_DiD_boot", np.nan)
                rows.append(row)
        return rows

    results = Parallel(n_jobs=args.n_jobs, batch_size=8)(delayed(one_gene)(g) for g in tested)
    table = pd.DataFrame([row for rows in results for row in rows])
    tag = "bootstrap" if args.bootstrap else "analytical"
    table.to_csv(args.out_dir / f"melanoma_gene_relabeling_pvalues_{tag}.csv.gz", index=False)

    summary = []
    for test in tests:
        p = table[f"p_{test}"]
        row: dict[str, object] = {"test": test, "n_tests": int(p.notna().sum())}
        for alpha in ALPHAS:
            row[f"reject_{alpha:g}_pct"] = 100 * float((p < alpha).mean())
        observed_p = table.loc[table["observed"], f"p_{test}"]
        per_assignment = table.groupby("assignment")[f"p_{test}"].apply(lambda s: int((s < 0.05).sum()))
        n_observed = int((observed_p < 0.05).sum())
        row["observed_n_p05"] = n_observed
        row["observed_min_p"] = float(observed_p.min())
        row["relabeled_median_n_p05"] = float(per_assignment.median())
        row["relabeled_95pct_n_p05"] = float(per_assignment.quantile(0.95))
        row["share_assignments_at_least_observed"] = float((per_assignment >= n_observed).mean())
        summary.append(row)
    out = pd.DataFrame(summary)
    out.to_csv(args.out_dir / f"melanoma_gene_calibration_summary_{tag}.csv", index=False)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
