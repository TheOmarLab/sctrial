"""Effect-size panels plot the statistic their axis names: Hedges' g in Figure 2H and Supp Fig 3K."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from sctrial.stats.effect_size import cohens_d_from_did, hedges_g  # noqa: E402


def _delta(arm_col, arms, n=(6, 6)):
    rng = np.random.default_rng(1)
    rows = []
    for a, k in zip(arms, n):
        for i in range(k):
            rows.append({arm_col: a, "sig_NK Cell Activity": rng.normal(0.5 if a == arms[0] else 0, 1),
                         "sig_Cell Proliferation": rng.normal(0, 1)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("rel, func, arms, n", [
    ("main/figure2_tnbc_analysis", "_panel_h", ("anti-PDL1+Chemo", "Chemo"), (6, 6)),
    ("supp/supp_fig3_melanoma_tnbc_analysis", "_panel_k_cohens_d", ("Responder", "Non-responder"), (3, 8)),
])
def test_panel_plots_hedges_g_and_says_so(load_figure_module, rel, func, arms, n):
    mod = load_figure_module(rel)
    delta = _delta(mod.DESIGN.arm_col, arms, n)
    sig = ["sig_NK Cell Activity", "sig_Cell Proliferation"]
    fig, ax = plt.subplots()
    getattr(mod, func)(ax, {"delta": delta, "sig_cols": sig})
    plotted = sorted(np.round(ax.collections[-1].get_offsets()[:, 0], 6))
    expected, uncorrected = [], []
    for col in sig:
        x = delta.loc[delta[mod.DESIGN.arm_col] == arms[0], col].to_numpy()
        y = delta.loc[delta[mod.DESIGN.arm_col] == arms[1], col].to_numpy()
        expected.append(round(hedges_g(x, y), 6))
        uncorrected.append(round(cohens_d_from_did(x, y), 6))
    assert plotted == sorted(expected) and plotted != sorted(uncorrected)
    assert ax.get_xlabel().startswith("Hedges' g")
    plt.close(fig)
