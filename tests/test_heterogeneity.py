import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st


def test_treatment_heterogeneity_basic():
    rng = np.random.default_rng(0)
    obs = []
    for pid in range(12):
        arm = "Treated" if pid < 6 else "Control"
        biomarker = 1 if pid % 2 == 0 else 0
        for visit in ["V1", "V2"]:
            obs.append(
                {
                    "participant_id": f"P{pid}",
                    "visit": visit,
                    "arm": arm,
                    "biomarker": biomarker,
                }
            )
    obs = pd.DataFrame(obs)
    X = rng.normal(size=(len(obs), 1))
    adata = AnnData(X=X, obs=obs)
    adata.obs["sig1"] = rng.normal(size=len(obs))

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )

    res = st.test_treatment_heterogeneity(
        adata,
        features=["sig1"],
        design=design,
        visits=("V1", "V2"),
        biomarker_col="biomarker",
    )
    assert not res.empty
    assert "beta_heterogeneity" in res.columns

    # Deeper assertions: p-value range, finite betas, n_units positive
    for _, row in res.iterrows():
        if pd.notna(row.get("p_heterogeneity")):
            assert 0 <= row["p_heterogeneity"] <= 1, "p-value out of [0,1]"
        assert np.isfinite(row["beta_heterogeneity"]), "beta should be finite"
        assert row["n_units"] > 0, "n_units should be positive"


def test_treatment_heterogeneity_numeric_biomarker():
    rng = np.random.default_rng(1)
    obs = []
    for pid in range(10):
        arm = "Treated" if pid < 5 else "Control"
        biomarker = rng.normal()
        for visit in ["V1", "V2"]:
            obs.append(
                {
                    "participant_id": f"P{pid}",
                    "visit": visit,
                    "arm": arm,
                    "biomarker_num": biomarker,
                }
            )
    obs = pd.DataFrame(obs)
    X = rng.normal(size=(len(obs), 1))
    adata = AnnData(X=X, obs=obs)
    adata.obs["sig1"] = rng.normal(size=len(obs))

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )

    res = st.test_treatment_heterogeneity(
        adata,
        features=["sig1"],
        design=design,
        visits=("V1", "V2"),
        biomarker_col="biomarker_num",
        threshold=0.0,
    )
    assert "beta_heterogeneity" in res.columns


def _heterogeneity_fixture(n=16, cells=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        arm = "Treated" if pid < n // 2 else "Control"
        biomarker = float(rng.normal())
        n_cells = 3 if cells is None else cells[pid]
        for visit in ("V1", "V2"):
            for _ in range(n_cells):
                rows.append(
                    {
                        "participant_id": f"P{pid:02d}",
                        "visit": visit,
                        "arm": arm,
                        "bmk": biomarker,
                        "sig1": float(rng.normal()),
                    }
                )
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    adata = AnnData(X=np.zeros((len(obs), 1)), obs=obs)
    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )
    return adata, design


def _heterogeneity(adata, design, **kwargs):
    return st.test_treatment_heterogeneity(
        adata, features=["sig1"], design=design, visits=("V1", "V2"), biomarker_col="bmk", **kwargs
    )


def test_heterogeneity_excludes_participants_without_biomarker():
    adata, design = _heterogeneity_fixture()
    missing = ["P03", "P11"]
    adata.obs.loc[adata.obs["participant_id"].isin(missing), "bmk"] = np.nan

    with pytest.warns(UserWarning, match="2 of 16 paired participants have no value"):
        res = _heterogeneity(adata, design)
    kept = adata[~adata.obs["participant_id"].isin(missing)].copy()
    ref = _heterogeneity(kept, design)

    assert res["n_units"].iloc[0] == 14
    for col in ("beta_heterogeneity", "p_heterogeneity", "threshold"):
        assert res[col].iloc[0] == pytest.approx(ref[col].iloc[0], rel=1e-12)


def test_heterogeneity_uses_paired_participants_only():
    adata, design = _heterogeneity_fixture()
    drop = adata.obs["participant_id"].isin(["P02", "P12"]) & (adata.obs["visit"] == "V2")
    one_visit = adata[~drop.to_numpy()].copy()
    paired = adata[~adata.obs["participant_id"].isin(["P02", "P12"])].copy()

    res = _heterogeneity(one_visit, design)
    ref = _heterogeneity(paired, design)

    assert res["n_units"].iloc[0] == 14
    for col in ("beta_heterogeneity", "p_heterogeneity", "threshold"):
        assert res[col].iloc[0] == pytest.approx(ref[col].iloc[0], rel=1e-12)


def test_heterogeneity_median_is_over_participants():
    cells = [1, 9, 2, 8, 1, 7, 3, 9, 2, 9, 1, 8, 2, 7, 1, 9]
    adata, design = _heterogeneity_fixture(cells=cells)
    per_participant = adata.obs.groupby("participant_id", observed=True)["bmk"].first()
    participant_median = float(per_participant.median())
    assert participant_median != pytest.approx(float(adata.obs["bmk"].median()))

    res = _heterogeneity(adata, design)
    explicit = _heterogeneity(adata, design, threshold=participant_median)

    assert res["threshold"].iloc[0] == pytest.approx(participant_median)
    assert res["beta_heterogeneity"].iloc[0] == pytest.approx(explicit["beta_heterogeneity"].iloc[0])
    assert res["p_heterogeneity"].iloc[0] == pytest.approx(explicit["p_heterogeneity"].iloc[0])


def test_heterogeneity_reports_non_estimable_interaction():
    adata, design = _heterogeneity_fixture()
    treated = adata.obs["arm"] == "Treated"
    adata.obs.loc[treated, "bmk"] = 5.0
    adata.obs.loc[adata.obs["participant_id"] == "P00", "bmk"] = -5.0

    with pytest.warns(UserWarning, match="not estimable.*treated/low=1"):
        res = _heterogeneity(adata, design, threshold=0.0)

    assert len(res) == 1
    assert np.isnan(res["beta_heterogeneity"].iloc[0])
    assert np.isnan(res["p_heterogeneity"].iloc[0])
    assert np.isnan(res["FDR_heterogeneity"].iloc[0])
    assert res["n_units"].iloc[0] == 16


def test_heterogeneity_counts_groups_among_the_participants_with_the_feature():
    # Every arm by biomarker group holds four participants, but three of the treated/high
    # ones have no value of the second feature: its interaction is not estimable, the first
    # feature's is.
    adata, design = _heterogeneity_fixture()
    pid = adata.obs["participant_id"]
    adata.obs["bmk"] = pid.map({f"P{i:02d}": float(i % 2) for i in range(16)})
    adata.obs["sig2"] = adata.obs["sig1"] + 1.0
    treated_high = ["P01", "P03", "P05"]
    adata.obs.loc[pid.isin(treated_high), "sig2"] = np.nan

    with pytest.warns(UserWarning, match="not estimable for 1 of 2 features.*treated/high=1 for 'sig2'"):
        res = st.test_treatment_heterogeneity(
            adata, features=["sig1", "sig2"], design=design, visits=("V1", "V2"), biomarker_col="bmk"
        ).set_index("feature")

    assert np.isfinite(res.loc["sig1", "beta_heterogeneity"]) and res.loc["sig1", "n_units"] == 16
    assert np.isnan(res.loc["sig2", "beta_heterogeneity"]) and np.isnan(res.loc["sig2", "p_heterogeneity"])
    assert res.loc["sig2", "n_units"] == 13


def test_heterogeneity_drops_a_participant_missing_the_feature_at_one_visit():
    adata, design = _heterogeneity_fixture()
    pid = adata.obs["participant_id"]
    adata.obs["bmk"] = pid.map({f"P{i:02d}": float(i % 2) for i in range(16)})
    adata.obs["sig2"] = adata.obs["sig1"]
    adata.obs.loc[(pid == "P02") & (adata.obs["visit"] == "V2"), "sig2"] = np.nan

    res = st.test_treatment_heterogeneity(
        adata, features=["sig2"], design=design, visits=("V1", "V2"), biomarker_col="bmk"
    )
    ref = _heterogeneity(adata[pid != "P02"].copy(), design)

    assert res["n_units"].iloc[0] == 15
    for col in ("beta_heterogeneity", "p_heterogeneity"):
        assert res[col].iloc[0] == pytest.approx(ref[col].iloc[0], rel=1e-10)
