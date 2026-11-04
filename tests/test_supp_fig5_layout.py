"""Supp Fig 5 layout: gene labels are reproducible and clear, keys sit off the data."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("adjustText")


def _genes(n=20, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(scale=0.4, size=n)
    feats = [f"G{i}" for i in range(n)]
    return feats, pd.Series(x, index=feats), pd.Series(y, index=feats)


def _labelled_scatter(sf5, *, composite):
    feats, xs, ys = _genes()
    fig, ax = plt.subplots(figsize=(1.9, 1.8) if composite else (7, 6))
    sc = ax.scatter(xs.values, ys.values, s=12 if composite else 50)
    sf5._pad_limits(ax, xs.values, ys.values)
    sf5._finish_scatter(ax, feats, xs, ys, sc, fontsize=3.8 if composite else 7,
                        composite=composite)
    fig.canvas.draw()
    return fig, ax, sc, feats, xs, ys


@pytest.mark.parametrize("composite", [False, True])
def test_gene_labels_are_reproducible(load_figure_module, composite):
    sf5 = load_figure_module("supp/supp_fig5_sensitivity_robustness")
    runs = []
    for _ in range(2):
        fig, ax, *_ = _labelled_scatter(sf5, composite=composite)
        runs.append([(t.get_text(), t.get_position()) for t in ax.texts])
        plt.close(fig)
    assert runs[0] == runs[1]


def test_standalone_labels_every_gene_without_defects(load_figure_module):
    sf5 = load_figure_module("supp/supp_fig5_sensitivity_robustness")
    fig, ax, sc, feats, xs, ys = _labelled_scatter(sf5, composite=False)
    assert sorted(t.get_text() for t in ax.texts) == sorted(feats)
    xy = ax.transData.transform(np.column_stack([xs[feats], ys[feats]]))
    labels = sorted(ax.texts, key=lambda t: feats.index(t.get_text()))
    defects, _ = sf5._layout_score(ax, labels, xy, sf5._clearance_px(ax, sc))
    assert defects == 0
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    for t in ax.texts:
        bb = t.get_window_extent(renderer)
        assert frame.x0 <= bb.x0 and bb.x1 <= frame.x1
        assert frame.y0 <= bb.y0 and bb.y1 <= frame.y1
    plt.close(fig)


def test_composite_drops_the_least_discordant_genes_first(load_figure_module):
    sf5 = load_figure_module("supp/supp_fig5_sensitivity_robustness")
    fig, ax, _, feats, xs, ys = _labelled_scatter(sf5, composite=True)
    shown = {t.get_text() for t in ax.texts}
    slope, intercept = np.polyfit(xs[feats], ys[feats], 1)
    resid = (ys[feats] - (slope * xs[feats] + intercept)).abs()
    hidden = set(feats) - shown
    if hidden:
        assert resid[sorted(hidden)].max() <= resid[sorted(shown)].min()
    plt.close(fig)


def test_correlation_key_sits_below_the_axes(load_figure_module):
    sf5 = load_figure_module("supp/supp_fig5_sensitivity_robustness")
    for composite in (False, True):
        fig, ax = plt.subplots(figsize=(1.9, 1.8) if composite else (7, 6))
        ax.scatter([0, 1, 2], [0, 1, 3])
        ax.set_xlabel("x")
        sf5._corr_key(ax, 0.84, composite=composite)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        key = ax.get_legend().get_window_extent(renderer)
        assert key.y1 < ax.get_window_extent(renderer).y0
        assert ax.get_legend().get_texts()[0].get_text() == "r = 0.84"
        plt.close(fig)


def test_forest_key_is_below_every_forest(load_figure_module):
    sf5 = load_figure_module("supp/supp_fig5_sensitivity_robustness")
    feats = [f"G{i}" for i in range(6)]
    boot = {
        ds: {
            "part": pd.DataFrame({"feature": feats, "beta_DiD": np.linspace(-1, 1, 6),
                                  "se_DiD": np.full(6, 0.3)}),
            "boot": pd.DataFrame({"feature": feats, "se_DiD_boot": np.full(6, 0.35)}),
        }
        for ds in ("A", "B")
    }
    fig = plt.figure(figsize=(9, 6.5))
    sf5._panel_bootstrap_multi(fig, boot)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert all(ax.get_legend() is None for ax in fig.axes)
    (key,) = fig.legends
    top = key.get_window_extent(renderer).y1
    assert all(top < ax.get_window_extent(renderer).y0 for ax in fig.axes)
    plt.close(fig)
