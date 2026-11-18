"""Supplementary Table 2 reports the interval of the procedure that produced the p-value."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scanpy")


def test_supp_table2_reports_the_interval_of_the_procedure_behind_the_p_value(load_figure_module):
    tables = load_figure_module("supp/supp_tables")
    res = pd.DataFrame({
        "feature": ["sig_a", "sig_b"],
        "beta_DiD": [1.0, -0.5], "se_DiD": [0.5, 0.25], "se_DiD_boot": [0.3, np.nan],
        "p_DiD": [0.03, 0.2], "FDR_DiD": [0.06, 0.2],
        "ci_lo_boot": [0.1, np.nan], "ci_hi_boot": [2.4, np.nan],
        "n_units": [12, 12], "cov_type_used": ["cluster", "cluster"],
    })
    out = tables._harmonise(res, "TNBC", "DiD").set_index("feature")
    assert out.loc["sig_a", ["ci_lower", "ci_upper", "inference"]].tolist() == [0.1, 2.4, "wild cluster bootstrap"]
    assert out.loc["sig_b", "inference"] == "analytical"
    assert out.loc["sig_b", ["ci_lower", "ci_upper"]].tolist() == pytest.approx([-0.5 - 1.96 * 0.25, -0.5 + 1.96 * 0.25])
    assert out.loc["sig_a", ["se", "se_bootstrap"]].tolist() == [0.5, 0.3]


def _two_arm_adata(n_focal, n_reference, features, seed=0):
    from anndata import AnnData

    rng = np.random.default_rng(seed)
    rows = []
    people = [(f"T{i}", "anti-PDL1+Chemo") for i in range(n_focal)] + [
        (f"C{i}", "Chemo") for i in range(n_reference)] + [("U0", "Chemo")]
    for pid, arm in people:
        for visit in (("Pre",) if pid == "U0" else ("Pre", "Post")):
            for _ in range(3):
                rows.append({"participant_id": pid, "arm": arm, "visit": visit,
                             **{f: rng.normal() for f in features}})
    obs = pd.DataFrame(rows)
    for col in ("participant_id", "arm", "visit"):
        obs[col] = pd.Categorical(obs[col])
    obs.index = obs.index.astype(str)
    x = np.zeros((len(obs), 1))
    return AnnData(X=x, obs=obs, layers={"log1p_norm": x.copy()})


def test_supp_table2_corrects_immune_and_clinical_signatures_separately(
        load_figure_module, monkeypatch, tmp_path):
    from statsmodels.stats.multitest import multipletests

    tables = load_figure_module("supp/supp_tables")
    immune = [f"sig_{i}" for i in range(6)]
    clinical = ["clin_a", "clin_b"]
    adata = _two_arm_adata(4, 5, immune + clinical)
    monkeypatch.setattr(tables, "SUPP_OUTPUT", tmp_path)
    monkeypatch.setattr(tables, "get_tnbc_zhang", lambda: adata)
    monkeypatch.setattr(tables, "score_signatures", lambda ad, layer=None: (ad, immune))
    monkeypatch.setattr(tables, "score_clinical_signatures", lambda ad, layer=None: (ad, clinical))

    out = tables.table2_all_results(datasets=("TNBC",))
    assert out["signature_set"].value_counts().to_dict() == {"immune": 6, "clinical": 2}
    for _, grp in out.groupby("signature_set"):
        expected = multipletests(grp["pvalue"], method="fdr_bh")[1]
        np.testing.assert_allclose(grp["FDR"].to_numpy(), expected, atol=1e-12)
    # one family over all eight rows would give different immune values
    pooled = multipletests(out["pvalue"], method="fdr_bh")[1]
    assert not np.allclose(out["FDR"].to_numpy(), pooled)
    # the unpaired participant is in neither group
    assert (out["n_focal"] == 4).all() and (out["n_reference"] == 5).all()
    assert (out["n_units"] == 9).all()
    assert (out["cov_type_used"] == "cluster").all()


def test_supp_table5_uses_the_group_sizes_of_each_cohort(load_figure_module, monkeypatch, tmp_path):
    from sctrial.stats.power import (
        power_paired,
        power_two_group,
        sample_size_paired,
        sample_size_two_group,
        sensitivity_two_group,
    )

    tables = load_figure_module("supp/supp_tables")
    monkeypatch.setattr(tables, "SUPP_OUTPUT", tmp_path)
    pd.DataFrame([
        {"dataset": "Melanoma", "estimand": "DiD", "label": "A", "feature": "sig_a",
         "effect": 1.2, "se": 0.9, "n_units": 11, "n_focal": 3, "n_reference": 8},
        {"dataset": "COVID-19", "estimand": "Hedges' g (DFO_8-14)", "label": "A", "feature": "sig_a",
         "effect": -0.5, "se": 0.55, "n_units": 16, "n_focal": 5, "n_reference": 11},
        {"dataset": "AML", "estimand": "within-arm Δ", "label": "A", "feature": "sig_a",
         "effect": 1.0, "se": 0.4, "n_units": 11, "n_focal": np.nan, "n_reference": np.nan},
    ]).to_csv(tmp_path / "Supp_Table2_all_results.csv", index=False)
    tables._table2_fingerprint_path().write_text(tables.STATS_FINGERPRINT)

    out = tables.table5_power_analysis().set_index("dataset")

    for name, n1, n2, effect, se in (("Melanoma", 3, 8, 1.2, 0.9), ("COVID-19", 5, 11, 0.5, 0.55)):
        sigma = se / np.sqrt(1 / n1 + 1 / n2)
        row = out.loc[name]
        assert row["sigma_estimated"] == pytest.approx(sigma)
        assert row["power_at_observed_N"] == pytest.approx(power_two_group(n1, n2, effect, sigma=sigma))
        # at the observed sizes the standard error is the observed one
        z = effect / se
        from scipy.stats import norm
        assert row["power_at_observed_N"] == pytest.approx(norm.sf(1.959964 - z) + norm.cdf(-1.959964 - z), abs=1e-6)
        assert row["min_N_for_80pct_power"] == sum(
            sample_size_two_group(effect, sigma=sigma, allocation_ratio=n1 / n2))
        assert row["min_detectable_effect_80pct"] == pytest.approx(sensitivity_two_group(n1, n2, sigma=sigma))
        assert (row["n_focal"], row["n_reference"]) == (n1, n2)

    row = out.loc["AML"]
    sigma = 0.4 * np.sqrt(11 / 2)
    assert row["power_at_observed_N"] == pytest.approx(power_paired(11, 1.0, sigma=sigma))
    assert row["min_N_for_80pct_power"] == sample_size_paired(1.0, sigma=sigma)
    assert pd.isna(row["n_focal"])

    # group sizes that do not add up to the analyzed participants are an error, not a guess
    bad = pd.read_csv(tmp_path / "Supp_Table2_all_results.csv")
    bad.loc[0, "n_focal"] = 4
    bad.to_csv(tmp_path / "Supp_Table2_all_results.csv", index=False)
    with pytest.raises(ValueError, match="group sizes"):
        tables.table5_power_analysis()


def test_group_sizes_refuse_a_paired_participant_outside_both_groups(load_figure_module):
    import sctrial as st

    tables = load_figure_module("supp/supp_tables")
    design = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                            arm_treated="anti-PDL1+Chemo", arm_control="Chemo", celltype_col=None)
    adata = _two_arm_adata(3, 4, ["sig_0"])
    assert tables._paired_group_sizes(adata, design, ("Pre", "Post")) == (3, 4)

    # did_table excludes this participant with a warning; the table refuses to count around it
    adata.obs["arm"] = adata.obs["arm"].astype(str)
    adata.obs.loc[adata.obs["participant_id"] == "C0", "arm"] = "Other"
    with pytest.raises(ValueError, match="neither"):
        tables._paired_group_sizes(adata, design, ("Pre", "Post"))
