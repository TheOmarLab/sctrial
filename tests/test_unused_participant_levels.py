"""Model results must not depend on participant categories that are not in the model.

anndata stores ``obs`` string columns as Categoricals and keeps every category through
subsetting and ``groupby``. A participant with one visit only is dropped from a paired
analysis, but its category stays, and ``C(participant)`` then adds an empty indicator
column. Those columns count as parameters in the small-sample correction of the
cluster-robust covariance, which became NaN (or raised) once parameters reached the
number of observations, so the fit fell back to the homoskedastic standard error.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st
from sctrial.stats._utils import drop_unused_levels

FEATS = ["sig_A", "sig_B", "sig_C"]
NUMERIC = ["beta_DiD", "se_DiD", "p_DiD", "p_DiD_boot", "se_DiD_boot", "ci_lo_boot", "ci_hi_boot"]


def _cells(n_treated: int, n_control: int, n_unpaired: int, *, seed: int = 0) -> AnnData:
    """Four cells per participant-visit; ``n_unpaired`` extra participants have Pre only."""
    rng = np.random.default_rng(seed)
    rows = []
    paired = [(f"T{i}", "Treated") for i in range(n_treated)] + [
        (f"C{i}", "Control") for i in range(n_control)
    ]
    for pid, arm in paired:
        shift = rng.normal()
        for visit in ("Pre", "Post"):
            for _ in range(4):
                rows.append({"participant_id": pid, "arm": arm, "visit": visit,
                             "cell_type": rng.choice(["T", "B", "Mono"]),
                             **{f: shift + rng.normal() for f in FEATS}})
    for i in range(n_unpaired):
        for _ in range(4):
            rows.append({"participant_id": f"U{i}", "arm": "Control", "visit": "Pre",
                         "cell_type": rng.choice(["T", "B", "Mono"]),
                         **{f: rng.normal() for f in FEATS}})
    obs = pd.DataFrame(rows)
    for col in ("participant_id", "arm", "visit", "cell_type"):
        obs[col] = pd.Categorical(obs[col])
    obs.index = obs.index.astype(str)
    return AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _paired_only(adata: AnnData) -> AnnData:
    """The same cells restricted to participants with both visits, categories pruned."""
    n_visits = adata.obs.groupby("participant_id", observed=True)["visit"].nunique()
    keep = n_visits[n_visits == 2].index
    sub = adata[adata.obs["participant_id"].isin(keep)].copy()
    sub.obs["participant_id"] = sub.obs["participant_id"].cat.remove_unused_categories()
    return sub


TWO_ARM = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                         arm_treated="Treated", arm_control="Control", celltype_col="cell_type")
ONE_ARM = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None,
                         celltype_col=None)


def test_drop_unused_levels_prunes_categoricals_only():
    df = pd.DataFrame({"pid": pd.Categorical(["a", "b"], categories=["a", "b", "c"]),
                       "x": [1.0, 2.0], "s": ["u", "v"]})
    out = drop_unused_levels(df)
    assert out is df
    assert list(df["pid"].cat.categories) == ["a", "b"]
    assert df["x"].tolist() == [1.0, 2.0] and df["s"].tolist() == ["u", "v"]


@pytest.mark.parametrize("n_treated,n_control,n_unpaired", [(3, 8, 21), (6, 6, 5), (4, 5, 1)])
def test_did_table_does_not_depend_on_unused_participant_levels(n_treated, n_control, n_unpaired):
    full = _cells(n_treated, n_control, n_unpaired)
    assert len(full.obs["participant_id"].cat.categories) == n_treated + n_control + n_unpaired
    kw = dict(features=FEATS, design=TWO_ARM, visits=("Pre", "Post"), standardize=True,
              aggregate="participant_visit", use_bootstrap=True, n_boot=199, seed=1)
    res_full = st.did_table(full, **kw).set_index("feature").loc[FEATS]
    res_pruned = st.did_table(_paired_only(full), **kw).set_index("feature").loc[FEATS]
    assert (res_full["cov_type_used"] == "cluster").all()
    assert np.isfinite(res_full["se_DiD"]).all()
    assert (res_full["n_units"] == n_treated + n_control).all()
    np.testing.assert_allclose(res_full[NUMERIC].to_numpy(float), res_pruned[NUMERIC].to_numpy(float),
                               rtol=0, atol=1e-10)


def test_within_arm_comparison_does_not_depend_on_unused_participant_levels():
    # 11 paired participants and 10 with one visit: 22 observations, and with the empty
    # indicators 22 parameters, the case that raised ZeroDivisionError.
    full = _cells(11, 0, 10)
    kw = dict(arm="All", features=FEATS, design=ONE_ARM, visits=("Pre", "Post"), standardize=True,
              use_bootstrap=True, n_boot=199, seed=1)
    res_full = st.within_arm_comparison(full, **kw).set_index("feature").loc[FEATS]
    res_pruned = st.within_arm_comparison(_paired_only(full), **kw).set_index("feature").loc[FEATS]
    cols = ["beta_time", "se_time", "p_time", "p_time_boot", "se_time_boot", "ci_lo_boot", "ci_hi_boot"]
    assert (res_full["cov_type_used"] == "cluster").all()
    assert (res_full["n_units"] == 11).all()
    np.testing.assert_allclose(res_full[cols].to_numpy(float), res_pruned[cols].to_numpy(float),
                               rtol=0, atol=1e-10)


def test_abundance_did_does_not_depend_on_unused_participant_levels():
    full = _cells(5, 6, 9)
    kw = dict(design=TWO_ARM, visits=("Pre", "Post"), use_bootstrap=True, n_boot=199, seed=1)
    res_full = st.abundance_did(full, **kw).set_index("celltype").sort_index()
    res_pruned = st.abundance_did(_paired_only(full), **kw).set_index("celltype").sort_index()
    cols = [c for c in ("beta_DiD", "se_DiD", "p_DiD", "p_DiD_boot", "ci_lo_boot", "ci_hi_boot")
            if c in res_full.columns]
    assert len(cols) >= 3
    np.testing.assert_allclose(res_full[cols].to_numpy(float), res_pruned[cols].to_numpy(float),
                               rtol=0, atol=1e-10)
