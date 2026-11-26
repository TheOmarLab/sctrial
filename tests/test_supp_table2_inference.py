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
    # the residual SD goes to the power table, not into the Table 2 file
    assert "resid_sd" not in out.columns
    assert "resid_sd" not in pd.read_csv(tmp_path / "Supp_Table2_all_results.csv").columns


def _table2_rows():
    return pd.DataFrame([
        {"dataset": "Melanoma", "estimand": "DiD", "label": "A", "feature": "sig_a",
         "effect": 1.2, "se": 0.9, "n_units": 11, "n_focal": 3, "n_reference": 8, "resid_sd": 0.7},
        {"dataset": "COVID-19", "estimand": "Hedges' g (DFO_8-14)", "label": "A", "feature": "sig_a",
         "effect": -0.5, "se": 0.55, "n_units": 16, "n_focal": 5, "n_reference": 11, "resid_sd": np.nan},
        {"dataset": "AML", "estimand": "within-arm Δ", "label": "A", "feature": "sig_a",
         "effect": 1.0, "se": 0.4, "n_units": 11, "n_focal": np.nan, "n_reference": np.nan, "resid_sd": 0.6},
    ])


def test_supp_table5_uses_the_sd_of_the_changes_and_the_t_distribution(load_figure_module):
    from scipy import stats

    from sctrial.stats.power import (
        power_paired,
        power_two_group,
        sample_size_paired,
        sample_size_two_group,
        sensitivity_paired,
        sensitivity_two_group,
    )

    tables = load_figure_module("supp/supp_tables")
    out = tables._power_table(_table2_rows()).set_index("dataset")

    # two-group longitudinal: pooled SD of the participant changes, not a value derived from the SE
    row, sigma = out.loc["Melanoma"], np.sqrt(2) * 0.7
    assert row["sigma_estimated"] == pytest.approx(sigma)
    assert row["sigma_estimated"] != pytest.approx(0.9 / np.sqrt(1 / 3 + 1 / 8))
    ncp, crit = 1.2 / (sigma * np.sqrt(1 / 3 + 1 / 8)), stats.t.ppf(0.975, 9)
    assert row["power_at_observed_N"] == pytest.approx(
        stats.nct.sf(crit, 9, ncp) + stats.nct.cdf(-crit, 9, ncp))
    assert row["power_at_observed_N"] == pytest.approx(power_two_group(3, 8, 1.2, sigma=sigma, reference="t"))
    assert row["min_N_for_80pct_power"] == sum(
        sample_size_two_group(1.2, sigma=sigma, allocation_ratio=3 / 8, reference="t"))
    assert row["min_detectable_effect_80pct"] == pytest.approx(
        sensitivity_two_group(3, 8, sigma=sigma, reference="t"))
    assert (row["n_focal"], row["n_reference"]) == (3, 8)

    # cross-sectional Hedges' g is in units of the pooled SD, so the SD is 1 whatever its SE
    row = out.loc["COVID-19"]
    assert row["sigma_estimated"] == 1.0
    assert row["power_at_observed_N"] == pytest.approx(power_two_group(5, 11, 0.5, sigma=1.0, reference="t"))
    assert row["min_detectable_effect_80pct"] == pytest.approx(sensitivity_two_group(5, 11, sigma=1.0, reference="t"))
    assert row["min_N_for_80pct_power"] == sum(
        sample_size_two_group(0.5, sigma=1.0, allocation_ratio=5 / 11, reference="t"))

    # single arm: the table reports the SD of the participant changes; the paired
    # formula is parameterized by the SD of one visit
    row = out.loc["AML"]
    assert row["sigma_estimated"] == pytest.approx(np.sqrt(2) * 0.6)
    crit = stats.t.ppf(0.975, 10)
    ncp = 1.0 / (np.sqrt(2) * 0.6 / np.sqrt(11))
    assert row["power_at_observed_N"] == pytest.approx(
        stats.nct.sf(crit, 10, ncp) + stats.nct.cdf(-crit, 10, ncp))
    assert row["power_at_observed_N"] == pytest.approx(power_paired(11, 1.0, sigma=0.6, reference="t"))
    assert row["min_N_for_80pct_power"] == sample_size_paired(1.0, sigma=0.6, reference="t")
    assert row["min_detectable_effect_80pct"] == pytest.approx(sensitivity_paired(11, sigma=0.6, reference="t"))
    assert pd.isna(row["n_focal"])


def test_supp_table5_refuses_inconsistent_rows(load_figure_module):
    tables = load_figure_module("supp/supp_tables")

    # group sizes that do not add up to the analyzed participants are an error, not a guess
    bad = _table2_rows()
    bad.loc[0, "n_focal"] = 4
    with pytest.raises(ValueError, match="group sizes"):
        tables._power_table(bad)

    # a fitted row without its residual SD is an error, not a fall back to the standard error
    for i in (0, 2):
        bad = _table2_rows()
        bad.loc[i, "resid_sd"] = np.nan
        with pytest.raises(ValueError, match="residual standard deviation"):
            tables._power_table(bad)


def test_supp_table5_takes_the_fits_of_table2(load_figure_module, monkeypatch, tmp_path):
    tables = load_figure_module("supp/supp_tables")
    monkeypatch.setattr(tables, "SUPP_OUTPUT", tmp_path)
    monkeypatch.setattr(tables, "_TABLE2_MODEL_FRAME", _table2_rows())

    out = tables.table5_power_analysis()
    saved = pd.read_csv(tmp_path / "Supp_Table5_power_analysis.csv")
    assert list(saved["dataset"]) == ["Melanoma", "COVID-19", "AML"]
    np.testing.assert_allclose(saved["power_at_observed_N"], out["power_at_observed_N"])


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


def _gene_rows(cov_type, n_units, betas, ses):
    from scipy import stats

    betas, ses = np.asarray(betas, dtype=float), np.asarray(ses, dtype=float)
    stat = np.abs(betas / ses)
    p = 2 * stats.t.sf(stat, n_units - 2) if cov_type == "nonrobust" else 2 * stats.norm.sf(stat)
    return pd.DataFrame({"feature": [f"g{i}" for i in range(len(betas))], "beta_DiD": betas,
                         "se_DiD": ses, "p_DiD": p, "n_units": n_units, "cov_type_used": cov_type})


def test_gene_level_interval_uses_the_reference_distribution_of_the_p_value(load_figure_module):
    from scipy import stats

    tables = load_figure_module("supp/supp_tables")
    betas, ses = [2.1, -1.0, 0.4, 2.2], [1.0, 0.45, 0.5, 1.0]

    pooled = tables._with_analytical_interval(_gene_rows("nonrobust", 11, betas, ses))
    half = (pooled["ci_upper"] - pooled["ci_lower"]) / 2
    assert np.allclose(half, stats.t.ppf(0.975, 9) * np.asarray(ses))
    excludes_zero = (pooled["ci_lower"] > 0) | (pooled["ci_upper"] < 0)
    assert (excludes_zero == (pooled["p_DiD"] < 0.05)).all()
    # 2.1 and 2.2 standard errors: outside +/- 1.96 SE, not significant on 9 degrees of freedom
    assert pooled["p_DiD"].iloc[0] > 0.05 and pooled["p_DiD"].iloc[3] > 0.05

    robust = tables._with_analytical_interval(_gene_rows("cluster", 11, betas, ses))
    assert np.array_equal(robust["ci_lower"], np.asarray(betas) - 1.96 * np.asarray(ses))
    assert np.array_equal(robust["ci_upper"], np.asarray(betas) + 1.96 * np.asarray(ses))


def test_gene_level_interval_refuses_p_values_from_another_distribution(load_figure_module):
    tables = load_figure_module("supp/supp_tables")
    rows = _gene_rows("cluster", 11, [2.1, -1.0], [1.0, 0.45]).assign(cov_type_used="nonrobust")
    with pytest.raises(ValueError, match="reference distribution"):
        tables._with_analytical_interval(rows)


def test_gene_level_interval_keeps_genes_without_a_fit(load_figure_module):
    tables = load_figure_module("supp/supp_tables")
    rows = _gene_rows("nonrobust", 11, [2.1, -1.0], [1.0, 0.45])
    rows.loc[1, ["beta_DiD", "se_DiD", "p_DiD", "cov_type_used"]] = [np.nan, np.nan, np.nan, np.nan]
    out = tables._with_analytical_interval(rows)
    assert np.isfinite(out.loc[0, "ci_lower"]) and np.isnan(out.loc[1, "ci_lower"])
