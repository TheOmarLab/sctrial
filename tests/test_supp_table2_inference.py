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
