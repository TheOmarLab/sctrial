"""Melanoma pathway enrichment under three variances of the ranking t-statistic.

Genes are ranked for GSEA by the DiD coefficient divided by its standard error. In the
melanoma cohort the responder group has three paired participants, and the manuscript ranks
with the pooled (homoskedastic) standard error (``MELANOMA_GENE_LEVEL_COV_TYPE``). This script
repeats the pre-ranked GSEA of the figure code with three rankings of the same coefficients:
the pooled standard error, the cluster-robust standard error, and the Welch standard error,
which allows unequal variances of the participant changes in the two groups. It reports how
far the enrichment agrees across the three and lists the pathways named in the manuscript.

Run from the repository root on a compute node, with ``SCTRIAL_MANUSCRIPT_DIR`` set:

    python scripts/melanoma_gsea_ranking_robustness.py <out_dir> [--n-jobs N]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRERANK = {"min_size": 10, "max_size": 500, "permutation_num": 1000, "outdir": None, "no_plot": True}
NAMED = [
    ("Hallmark", "Interferon Alpha Response"),
    ("Hallmark", "E2F Targets"),
    ("Hallmark", "Myc Targets V1"),
    ("Hallmark", "Interferon Gamma Response"),
    ("Reactome", "Translation R-HSA-72766"),
    ("Reactome", "Mitochondrial Translation R-HSA-5368287"),
    ("Reactome", "Synthesis Of DNA R-HSA-69239"),
    ("GO_BP", "B Cell Receptor Signaling Pathway (GO:0050853)"),
    ("GO_BP", "B Cell Proliferation (GO:0042100)"),
    ("GO_BP", "B Cell Activation (GO:0042113)"),
]


def ranking_from_table(table: pd.DataFrame) -> pd.DataFrame:
    """The ``tstat`` ranking of ``run_gsea_did``: coefficient over standard error.

    Tied genes are ordered by name here, so an enrichment score can differ in the second
    decimal from the value of the figure pipeline; the check in ``main`` bounds the difference.
    """
    valid = table[table["n_units"] >= 4]
    rank = valid["beta_DiD"].fillna(0) / (valid["se_DiD"].fillna(1) + 1e-12)
    return finalize(pd.DataFrame({"feature": valid["feature"].to_numpy(), "rank": rank.to_numpy()}))


def finalize(ranking: pd.DataFrame) -> pd.DataFrame:
    out = ranking.replace([np.inf, -np.inf], np.nan).dropna(subset=["rank"])
    out = out.drop_duplicates(subset="feature", keep="first")
    return out.sort_values(["rank", "feature"], ascending=[False, True]).reset_index(drop=True)


def welch_statistic(frame: pd.DataFrame, unit: str, time: str, arm_bin: str,
                    genes: list[str]) -> pd.Series:
    ordered = frame.sort_values([unit, time])
    pre = ordered[ordered[time] == 0].set_index(unit)
    post = ordered[ordered[time] == 1].set_index(unit)
    if list(pre.index) != list(post.index):
        raise ValueError("participants of the two visits do not match")
    change = post[genes].to_numpy(dtype=float) - pre[genes].to_numpy(dtype=float)
    focal = pre[arm_bin].to_numpy() == 1
    first, second = change[focal], change[~focal]
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(first.var(axis=0, ddof=1) / len(first) + second.var(axis=0, ddof=1) / len(second))
        stat = (first.mean(axis=0) - second.mean(axis=0)) / se
    return pd.Series(stat, index=genes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Melanoma pathway enrichment under three variances of the ranking statistic."
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    import gseapy as gp
    from joblib import Parallel, delayed

    from manuscript_figures import _shared as shared
    from sctrial import (
        TrialDesign,
        did_table,
        get_did_aggregated_df,
        harmonize_response,
        run_gsea_did,
    )

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
    genes = adata.var_names.tolist()
    common = {"design": design, "visits": visits, "layer": "log1p_tpm"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = {
            name: did_table(adata, features=genes, standardize=True, aggregate="participant_visit",
                            cov_type=cov, **common)
            for name, cov in (("pooled", "nonrobust"), ("cluster", "cluster"))
        }
        frame, unit, time, arm_bin = get_did_aggregated_df(
            adata, genes, design, visits, layer="log1p_tpm", aggregate="participant_visit"
        )
    rankings = {name: ranking_from_table(table) for name, table in tables.items()}
    welch = welch_statistic(frame, unit, time, arm_bin, genes)
    pooled_table = tables["pooled"].set_index("feature")
    welch_rank = welch.where(pooled_table["beta_DiD"].reindex(genes).notna().to_numpy(), 0.0)
    rankings["welch"] = finalize(pd.DataFrame({"feature": genes, "rank": welch_rank.to_numpy()}))

    reference = rankings["pooled"].set_index("feature")["rank"]
    agreement = []
    for name in ("cluster", "welch"):
        other = rankings[name].set_index("feature")["rank"].reindex(reference.index)
        keep = (reference != 0) & other.notna()
        agreement.append({"ranking": name, "n_genes": int(keep.sum()),
                          "spearman_with_pooled": float(reference[keep].corr(other[keep], method="spearman"))})
    print(pd.DataFrame(agreement).to_string(index=False))

    def enrich(name: str, library: str, short: str) -> pd.DataFrame:
        res = gp.prerank(rnk=rankings[name], gene_sets=library, threads=1, **PRERANK).res2d
        return res.assign(Library=short, ranking=name)

    tasks = [(name, lib, short) for name in rankings for lib, short in shared.GSEA_LIBRARIES]
    results = pd.concat(Parallel(n_jobs=args.n_jobs)(delayed(enrich)(*t) for t in tasks), ignore_index=True)
    for column in ("NES", "FDR q-val"):
        results[column] = pd.to_numeric(results[column], errors="coerce")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        package = run_gsea_did(adata, gene_sets="MSigDB_Hallmark_2020", rank_by="tstat",
                               cov_type=shared.MELANOMA_GENE_LEVEL_COV_TYPE, **common, **PRERANK)
    mine = results[(results["ranking"] == "pooled") & (results["Library"] == "Hallmark")]
    package = package.assign(NES=pd.to_numeric(package["NES"], errors="coerce"),
                             FDR=pd.to_numeric(package["FDR q-val"], errors="coerce"))
    joined = mine.merge(package[["Term", "NES", "FDR"]], on="Term", suffixes=("", "_package"))
    enriched = joined[joined["FDR"] < 0.25]
    gap = (enriched["NES"] - enriched["NES_package"]).abs().max()
    print(f"check against run_gsea_did (Hallmark, pooled): {len(joined)} gene sets, {len(enriched)} with "
          f"FDR < 0.25, max |NES difference| among them {gap:.3g}")
    if len(joined) != len(mine) or not gap < 0.1:
        raise SystemExit("the pooled ranking of this script does not reproduce run_gsea_did")

    wide = results.pivot_table(index=["Library", "Term"], columns="ranking", values=["NES", "FDR q-val"])
    wide.columns = [f"{'FDR' if a.startswith('FDR') else 'NES'}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(args.out_dir / "melanoma_gsea_three_rankings.csv", index=False)

    summary = []
    for name in ("welch", "cluster"):
        same_sign = np.sign(wide["NES_pooled"]) == np.sign(wide[f"NES_{name}"])
        sig_pooled, sig_other = wide["FDR_pooled"] < 0.25, wide[f"FDR_{name}"] < 0.25
        row = {
            "compared_with_pooled": name,
            "n_gene_sets": len(wide),
            "nes_spearman": float(wide["NES_pooled"].corr(wide[f"NES_{name}"], method="spearman")),
            "pct_same_sign": 100 * float(same_sign.mean()),
            "n_fdr25_pooled": int(sig_pooled.sum()),
            "n_fdr25_other": int(sig_other.sum()),
            "n_fdr25_both_same_sign": int((sig_pooled & sig_other & same_sign).sum()),
        }
        for sign, label in ((-1, "negative"), (1, "positive")):
            row[f"n_fdr25_pooled_{label}"] = int((sig_pooled & (wide["NES_pooled"] * sign > 0)).sum())
            row[f"n_fdr25_other_{label}"] = int((sig_other & (wide[f"NES_{name}"] * sign > 0)).sum())
            row[f"n_fdr25_both_{label}"] = int(
                (sig_pooled & sig_other & (wide["NES_pooled"] * sign > 0) & (wide[f"NES_{name}"] * sign > 0)).sum()
            )
        summary.append(row)
    summary_table = pd.DataFrame(summary)
    summary_table.to_csv(args.out_dir / "melanoma_gsea_three_rankings_summary.csv", index=False)
    named = wide.merge(pd.DataFrame(NAMED, columns=["Library", "Term"]), on=["Library", "Term"])
    named.to_csv(args.out_dir / "melanoma_gsea_three_rankings_named.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(summary_table.T.to_string())
    print(named.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
