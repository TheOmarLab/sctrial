"""Scatter labelling (``_shared.label_points``) and the Supp Fig 5 layout that uses it."""

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("adjustText")


def _shared():
    return sys.modules["manuscript_figures._shared"]


def _genes(n=20, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(scale=0.4, size=n)
    feats = [f"G{i}" for i in range(n)]
    return feats, pd.Series(x, index=feats), pd.Series(y, index=feats)


def _defects(ax, texts, xy, s, obstacles=None):
    """Defect count of a finished layout, measured the way label_points scores it."""
    fig = ax.figure
    fig.canvas.draw()
    r = (np.sqrt(s) / 2 + 1.0) * fig.dpi / 72
    target_px = ax.transData.transform(xy)
    target_r = np.full(len(xy), r)
    obs_px, obs_r = target_px, target_r
    if obstacles is not None:
        obs_px = np.vstack([target_px, ax.transData.transform(obstacles)])
        obs_r = np.full(len(obs_px), r)
    defects, _ = _shared().label_layout_score(texts, target_px, target_r, obs_px, obs_r,
                                              fig.canvas.get_renderer())
    return defects


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
    labels = sorted(ax.texts, key=lambda t: feats.index(t.get_text()))
    assert [t.get_text() for t in labels] == feats
    xy = np.column_stack([xs[feats], ys[feats]])
    assert _defects(ax, labels, xy, sc.get_sizes()[0]) == 0
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


def test_label_points_keeps_labels_off_obstacle_markers(load_figure_module):
    load_figure_module("supp/supp_fig5_sensitivity_robustness")
    rng = np.random.default_rng(7)
    xy = rng.uniform(0, 10, size=(8, 2))
    others = rng.uniform(0, 10, size=(40, 2))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(*others.T, s=40)
    ax.scatter(*xy.T, s=40)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    names = [f"L{i}" for i in range(len(xy))]
    shown = _shared().label_points(ax, names, xy[:, 0], xy[:, 1], 40, fontsize=7,
                                   obstacles=(others, 40))
    assert shown == names
    labels = sorted(ax.texts, key=lambda t: names.index(t.get_text()))
    assert _defects(ax, labels, xy, 40, obstacles=others) == 0
    plt.close(fig)


def test_label_points_draws_a_leader_only_for_detached_labels(load_figure_module):
    load_figure_module("supp/supp_fig5_sensitivity_robustness")
    fig, ax = plt.subplots(figsize=(4, 4))
    xy = np.array([[0.0, 0.0], [0.02, 0.0], [0.04, 0.0]])
    ax.scatter(*xy.T, s=30)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    _shared().label_points(ax, ["A", "B", "C"], xy[:, 0], xy[:, 1], 30, fontsize=8)
    from matplotlib.patches import FancyArrowPatch

    leaders = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    assert 1 <= len(leaders) <= 3
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
