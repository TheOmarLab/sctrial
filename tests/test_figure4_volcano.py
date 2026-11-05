"""Figure 4A volcano: which genes are labelled, and that the labels stay clear."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("adjustText")


def _genes():
    """A volcano with three strongly significant genes of moderate effect.

    MODA/MODB/MODC have the smallest p-values but only moderate |beta|, so a
    score that ranks |beta| and p together leaves them out of its top candidates
    (the defect that hid CD163 and CCL8 in the TNBC volcano).
    """
    rng = np.random.default_rng(0)
    n = 400
    beta = rng.normal(scale=0.8, size=n)
    p = np.clip(np.exp(-np.abs(beta) * 1.2) * rng.uniform(0.6, 1.0, size=n), 0.06, 1)
    feats = [f"GENE{i}" for i in range(n)]
    df = pd.DataFrame({"feature": feats, "beta_DiD": beta, "p_DiD": p})
    extra = pd.DataFrame({
        "feature": ["MODA", "MODB", "MODC", "NEARA", "NEARB"],
        "beta_DiD": [0.30, 0.35, -0.32, -1.80, -1.81],
        "p_DiD": [0.002, 0.004, 0.006, 0.051, 0.052],
    })
    return pd.concat([df, extra], ignore_index=True)


def test_most_significant_genes_are_labelled_and_near_twins_are_not(load_figure_module):
    f4 = load_figure_module("main/figure4_biological_discovery_multi_dataset")
    fig, ax = plt.subplots(figsize=(8, 6))
    f4._tnbc_volcano(ax, {"gene_results": _genes()}, composite=False)
    labels = {t.get_text() for t in ax.texts}
    assert {"MODA", "MODB", "MODC"} <= labels
    assert len({"NEARA", "NEARB"} & labels) == 1
    plt.close(fig)


def test_composite_volcano_keeps_significant_labels_inside_the_frame(load_figure_module):
    f4 = load_figure_module("main/figure4_biological_discovery_multi_dataset")
    fig, ax = plt.subplots(figsize=(2.2, 1.8))
    f4._tnbc_volcano(ax, {"gene_results": _genes()}, composite=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = {t.get_text() for t in ax.texts}
    assert {"MODA", "MODB", "MODC"} <= labels
    frame = ax.get_window_extent(renderer)
    for t in ax.texts:
        bb = t.get_window_extent(renderer)
        assert frame.x0 <= bb.x0 and bb.x1 <= frame.x1
        assert frame.y0 <= bb.y0 and bb.y1 <= frame.y1
    key = ax.get_legend().get_window_extent(renderer)
    assert key.y1 < frame.y0
    plt.close(fig)
