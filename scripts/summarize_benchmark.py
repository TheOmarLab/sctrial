"""Complete per-method summary of the frozen benchmark.

Reuses the SAME aggregation helpers and loaders that produce Figure 3 and Supp
Fig 8 (manuscript_figures._benchmark), so every number here matches a figure. The
per-(scenario, method) endpoint is computed per replicate, then averaged with a
Monte Carlo SE across replicates; scenario means are then averaged equal-weight.

Endpoints: pure-null Type I error, mixed-signal null-gene FPR (balanced +
one-directional), % null p outside 95% CI, realized FDR after BH, marginal power,
FDR-controlled discovery sensitivity (end-to-end BH TPR), bias/RMSE, robustness
families (null FPR / signal TPR / evaluability / convergence), end-to-end vs
tested-only detection, and runtime.

Run on HPC via sbatch (loads the multi-GB combined CSVs).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ORDER = ["sctrial_did", "wilcoxon_paired", "limma_voom", "dreamlet", "nebula"]
LABEL = {"sctrial_did": "sctrial", "wilcoxon_paired": "Wilcoxon", "limma_voom": "limma-voom",
         "dreamlet": "dreamlet", "nebula": "NEBULA"}
OUT = REPO / "manuscript" / "benchmark" / "validation" / "benchmark_summary"


def _boot():
    for name, path, subs in [
        ("manuscript_figures", "manuscript_figures/__init__.py", ["manuscript_figures"]),
        ("manuscript_figures._shared", "manuscript_figures/_shared.py", None),
        ("manuscript_figures._benchmark", "manuscript_figures/_benchmark.py", None),
    ]:
        kw = {"submodule_search_locations": subs} if subs else {}
        s = importlib.util.spec_from_file_location(name, path, **kw)
        m = importlib.util.module_from_spec(s)
        sys.modules[name] = m
        s.loader.exec_module(m)
    return sys.modules["manuscript_figures._benchmark"]


_LINES: list[str] = []


def emit(s=""):
    print(s, flush=True)
    _LINES.append(s)


def section(title):
    emit("\n" + "=" * 90)
    emit(title)
    emit("=" * 90)


def _tbl(title, per_method: dict, fmt="{:.4f}", note=""):
    """per_method: {method: value or (value, se)}."""
    emit(f"\n### {title}{('  — ' + note) if note else ''}")
    for m in ORDER:
        if m not in per_method:
            continue
        v = per_method[m]
        if isinstance(v, tuple):
            val, se = v
            vs = "n/a" if val is None or (isinstance(val, float) and np.isnan(val)) else fmt.format(val)
            ss = "" if se is None or (isinstance(se, float) and np.isnan(se)) else f" ± {fmt.format(se)}"
            emit(f"    {LABEL[m]:<12} {vs}{ss}")
        else:
            vs = "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
            emit(f"    {LABEL[m]:<12} {vs}")


def _scen_mean(rate, valuecol="mean"):
    """Equal-weight mean + across-scenario SE of the per-scenario values, per method."""
    out = {}
    for m, g in rate.groupby("method"):
        v = g[valuecol].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[m] = (np.nan, np.nan)
        else:
            out[m] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)
    return out


def _grid(rate, rowkey, valuecol="mean", fmt="{:.3f}"):
    """Method x rowkey grid (rate already merged with rowkey col)."""
    piv = rate.pivot_table(index=rowkey, columns="method", values=valuecol, aggfunc="mean")
    cols = [c for c in ORDER if c in piv.columns]
    header = "    " + f"{rowkey:<14}" + "".join(f"{LABEL[c]:>12}" for c in cols)
    emit(header)
    for idx, row in piv.iterrows():
        line = "    " + f"{str(idx):<14}" + "".join(
            f"{(fmt.format(row[c]) if pd.notna(row[c]) else 'n/a'):>12}" for c in cols)
        emit(line)


def main():
    b = _boot()
    OUT.mkdir(parents=True, exist_ok=True)
    emit("=" * 90)
    emit("FROZEN BENCHMARK — COMPLETE PER-METHOD SUMMARY")
    emit("Method order: sctrial (DiD), Wilcoxon (Δ scores), limma-voom, dreamlet, NEBULA")
    emit("Nominal alpha = 0.05; BH q = 0.05. Values are scenario means (± across-scenario SE).")
    emit("=" * 90)

    bench = b._load_benchmark_data(architectures=("balanced", "heterogeneous", "one_directional"))
    core = b._load_core_benchmark_data()
    emit(f"\nrows: sensitivity={len(bench):,}  core={len(core):,}")
    emit(f"core families: {sorted(core['family'].unique())}")
    emit(f"sensitivity architectures: {sorted(bench['architecture'].unique())}")

    # ============ A. CALIBRATION ============
    section("A. CALIBRATION (want ~0.05)")

    null = core[(core["family"] == "null") & (~core["is_signal"])]
    r = b._per_scenario_rate(null, on_signal=False)
    meta = null.groupby("scenario").agg(design=("design", "first"), total_n=("total_n", "first")).reset_index()
    r = r.merge(meta, on="scenario")
    _tbl("A1. Pure-null Type I error (ALL null scenarios)", _scen_mean(r))
    for dsn in ("two_arm", "single_arm"):
        _tbl(f"A1b. Pure-null Type I error — {dsn}", _scen_mean(r[r["design"] == dsn]))
    emit("\n    A1c. Pure-null Type I error by participants (total_n):")
    _grid(r, "total_n")

    for arch in ("balanced", "one_directional"):
        bb = bench[(bench["architecture"] == arch) & (bench["signal_fraction_realised"] > 0) & (~bench["is_signal"])]
        if bb.empty:
            continue
        rr = b._per_scenario_rate(bb, on_signal=False)
        mm = bb.groupby("scenario").agg(n_genes=("n_genes", "first"), signal_pct=("signal_pct", "first")).reset_index()
        rr = rr.merge(mm, on="scenario")
        _tbl(f"A2. Mixed-signal null-gene FPR — {arch}", _scen_mean(rr))
        emit(f"\n    A2b. Mixed-signal null-gene FPR by tested-panel size — {arch}:")
        _grid(rr, "n_genes")
        emit(f"\n    A2c. Mixed-signal null-gene FPR by signal fraction (%) — {arch}:")
        _grid(rr, "signal_pct")

    # % outside 95% CI (balanced, null genes) via the QQ envelope statistic
    balnull = bench[(bench["architecture"] == "balanced") & (bench["true_beta"] == 0.0)]
    rows = []
    itcol = "iteration"
    for (mth, ng, sp), g in balnull.groupby(["method", "n_genes", "signal_pct"]):
        fr = []
        for _, gg in g.groupby(itcol):
            pv = gg["pvalue"].dropna().values
            if len(pv) >= 5:
                fr.append(b._compute_frac_outside_ci(pv))
        if fr:
            rows.append({"method": mth, "scenario": f"{ng}_{sp}", "mean": float(np.mean(fr))})
    envelope = pd.DataFrame(rows)
    _tbl("A3. % of null p-values outside the 95% CI (want ~0.05)", _scen_mean(envelope))

    bal = bench[(bench["architecture"] == "balanced") & (bench["signal_fraction_realised"] > 0)]
    fdr = b._per_scenario_fdr(bal)
    _tbl("A4. Realized FDR after Benjamini-Hochberg (want <= 0.05)", _scen_mean(fdr))

    # ============ B. POWER / DISCOVERY ============
    section("B. POWER & DISCOVERY (higher = better, IF calibrated)")

    de = core[(core["family"] == "de_balanced") & (core["is_signal"])]
    rp = b._per_scenario_rate(de, on_signal=True)
    dem = de.groupby("scenario").agg(design=("design", "first"), per_arm=("n_treated", "first"),
                                     beta=("beta", "first")).reset_index()
    rp = rp.merge(dem, on="scenario")
    _tbl("B1. Marginal detection probability (ALL de_balanced signal scenarios)", _scen_mean(rp))
    emit("\n    B1b. Marginal detection probability by effect size (beta):")
    _grid(rp, "beta")

    tpr = b._per_scenario_tpr(bal, mode="end_to_end")
    tm = bal.groupby("scenario").agg(n_genes=("n_genes", "first"), signal_pct=("signal_pct", "first")).reset_index()
    tpr = tpr.merge(tm, on="scenario")
    _tbl("B2. FDR-controlled discovery sensitivity (end-to-end BH TPR)", _scen_mean(tpr))
    emit("\n    B2b. End-to-end BH TPR by signal fraction (%):")
    _grid(tpr, "signal_pct")

    # ============ C. ESTIMATION ============
    section("C. ESTIMATION ACCURACY (per method's OWN oracle; NOT cross-method)")
    br = b._compute_signal_bias_rmse_table(bench)
    bias = {m: (float(br[br.method == m]["bias"].mean()) if (br.method == m).any() else np.nan) for m in ORDER}
    rmse = {m: (float(br[br.method == m]["rmse"].mean()) if (br.method == m).any() else np.nan) for m in ORDER}
    _tbl("C1. Mean bias (beta_hat - beta), balanced signal genes", bias, fmt="{:+.4f}")
    _tbl("C2. RMSE of beta_hat, balanced signal genes", rmse)

    # ============ D. ROBUSTNESS FAMILIES ============
    section("D. ROBUSTNESS FAMILIES (core grid)")
    fams_null = core[~core["is_signal"]]
    rfn = b._per_scenario_rate(fams_null, on_signal=False)
    fm = core.groupby("scenario").agg(family=("family", "first")).reset_index()
    rfn = rfn.merge(fm, on="scenario")
    emit("\n    D1. Null-gene FPR by family x method (want ~0.05):")
    _grid(rfn[rfn["family"] != "null"], "family")

    sig = core[core["is_signal"].groupby(core["scenario"]).transform("any")]
    rft = b._per_scenario_tpr(sig, mode="end_to_end").merge(fm, on="scenario")
    emit("\n    D2. End-to-end BH TPR by signal-bearing family x method:")
    _grid(rft, "family")

    for kind in ("evaluability", "convergence"):
        q = b._per_scenario_quality(core, kind=kind).merge(fm, on="scenario")
        emit(f"\n    D3. {kind.capitalize()} by family x method (want ~1.0):")
        _grid(q, "family", fmt="{:.3f}")

    # ============ E. RUNTIME ============
    section("E. RUNTIME (wall-clock seconds per simulated dataset)")
    per_iter = (bench.groupby(["method", "scenario", "n_genes", "iteration"])["runtime_seconds"].first().reset_index())
    scen_med = per_iter.groupby(["method", "scenario", "n_genes"])["runtime_seconds"].median().reset_index()
    rt = scen_med.groupby(["method", "n_genes"])["runtime_seconds"].median().reset_index()
    emit("\n    E1. Median runtime (s) by tested-panel size:")
    _grid(rt, "n_genes", valuecol="runtime_seconds", fmt="{:.2f}")
    big = rt[rt["n_genes"] == rt["n_genes"].max()].set_index("method")["runtime_seconds"]
    if "sctrial_did" in big.index:
        base = big["sctrial_did"]
        emit(f"\n    E2. Speed vs sctrial at {int(rt['n_genes'].max()):,} genes:")
        for m in ORDER:
            if m in big.index:
                emit(f"    {LABEL[m]:<12} {big[m]/base:6.1f}x")

    (OUT / "benchmark_summary.txt").write_text("\n".join(_LINES) + "\n")
    emit(f"\nwrote {OUT}/benchmark_summary.txt")


if __name__ == "__main__":
    main()
