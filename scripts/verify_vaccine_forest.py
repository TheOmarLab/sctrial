"""Verify the Figure 3 forest Vaccine Cytotoxic vs Exhaustion effect sizes.

Reviewer flagged that Vaccine 'Cytotoxic T Cell Activity' and 'Immune Exhaustion'
both print d=+1.34 [-0.11,+2.78] (n=6). The two gene sets are disjoint, so
identical scores would be a bug. This recomputes at full precision and reports
the per-participant deltas, the paired Cohen's d + CI, the delta-array
correlation, and an element-wise identity check.

Run standalone (importlib bootstrap; needs the vaccine h5ad).
"""
import importlib.util
import sys

import numpy as np


def _bootstrap():
    for name, path, subs in [
        ("manuscript_figures", "manuscript_figures/__init__.py", ["manuscript_figures"]),
        ("manuscript_figures._shared", "manuscript_figures/_shared.py", None),
    ]:
        kw = {"submodule_search_locations": subs} if subs else {}
        s = importlib.util.spec_from_file_location(name, path, **kw)
        m = importlib.util.module_from_spec(s)
        sys.modules[name] = m
        s.loader.exec_module(m)
    return sys.modules["manuscript_figures._shared"]


def _paired_d(deltas):
    d = np.asarray(deltas, float)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    dz = mean / sd if sd > 0 else np.nan
    se = np.sqrt(1.0 / n + dz**2 / (2 * n))
    return dz, dz - 1.96 * se, dz + 1.96 * se, mean, sd


def main():
    sh = _bootstrap()
    from sctrial import TrialDesign

    vacc = sh.get_vaccine()
    print(f"vaccine: {vacc.n_obs:,} cells x {vacc.n_vars:,} genes")
    GS = sh.GENE_SIGNATURES
    for sig in ("Cytotoxic T Cell Activity", "Immune Exhaustion"):
        avail = [g for g in GS[sig] if g in vacc.var_names]
        print(f"  [{sig}] {len(avail)}/{len(GS[sig])} genes available: {avail}")

    vacc, sigs = sh.score_signatures(vacc, layer="log1p_norm")
    pid, visit = "participant_id", "visit"
    visits = ("Pre", "Post")

    deltas = {}
    for sig in ("sig_Cytotoxic T Cell Activity", "sig_Immune Exhaustion"):
        pb = vacc.obs.groupby([pid, visit], observed=True)[sig].mean().reset_index()
        arr, pids = [], []
        for p, pdf in pb.groupby(pid):
            if set(visits).issubset(set(pdf[visit])):
                pre = pdf.loc[pdf[visit] == "Pre", sig].values[0]
                post = pdf.loc[pdf[visit] == "Post", sig].values[0]
                arr.append(post - pre)
                pids.append(p)
        deltas[sig] = (np.array(arr), pids)

    cyt, cyt_pids = deltas["sig_Cytotoxic T Cell Activity"]
    exh, exh_pids = deltas["sig_Immune Exhaustion"]
    print("\n=== per-participant deltas (Post - Pre), FULL PRECISION ===")
    print(f"  participants (cyt): {cyt_pids}")
    print(f"  participants (exh): {exh_pids}")
    for p, a, b in zip(cyt_pids, cyt, exh):
        print(f"    {p}: cytotoxic={a:+.6f}   exhaustion={b:+.6f}   diff={a-b:+.2e}")

    dc, lc, hc, mc, sc = _paired_d(cyt)
    de, le, he, me, se = _paired_d(exh)
    print("\n=== paired Cohen's d, FULL PRECISION ===")
    print(f"  Cytotoxic:  d={dc:.6f}  CI=[{lc:.6f}, {hc:.6f}]  mean={mc:.6f} sd={sc:.6f}")
    print(f"  Exhaustion: d={de:.6f}  CI=[{le:.6f}, {he:.6f}]  mean={me:.6f} sd={se:.6f}")
    print(f"  |d_cyt - d_exh| = {abs(dc-de):.6e}")
    if len(cyt) == len(exh):
        print(f"  delta-array Pearson r = {np.corrcoef(cyt, exh)[0,1]:.6f}")
        print(f"  arrays element-wise identical? {np.allclose(cyt, exh)}")
    print("\nVERDICT:", "IDENTICAL (BUG)" if abs(dc - de) < 1e-6
          else f"DISTINCT (coincident to 2 decimals; true |Δd|={abs(dc-de):.4f})")


if __name__ == "__main__":
    main()
