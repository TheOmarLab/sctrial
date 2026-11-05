"""Keys sit below their axes and grouped tick labels do not collide (Figure 4, Supp Fig 7)."""

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def shared():
    for name, path, subs in [
        ("manuscript_figures", "manuscript_figures/__init__.py", ["manuscript_figures"]),
        ("manuscript_figures._shared", "manuscript_figures/_shared.py", None),
    ]:
        kw = {"submodule_search_locations": [str(ROOT / s) for s in subs]} if subs else {}
        spec = importlib.util.spec_from_file_location(name, ROOT / path, **kw)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    yield sys.modules["manuscript_figures._shared"]
    for name in ("manuscript_figures._shared", "manuscript_figures"):
        sys.modules.pop(name, None)


def _extent(artist, renderer):
    return artist.get_window_extent(renderer)


@pytest.mark.parametrize("rotation", [0, 35, 90])
def test_key_below_axes_clears_tick_labels_and_x_label(shared, rotation):
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.bar(range(6), [3, 1, 2, 4, 2, 1], label="up")
    ax.bar(range(6), [-1, -2, -1, -3, -1, -2], label="down")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"Long category name {i}" for i in range(6)], rotation=rotation,
                       ha="right" if rotation else "center")
    ax.set_xlabel("Effect")
    ax.legend(loc="upper right")
    leg = shared.key_below_axes(ax, ncol=2, fontsize=6)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    key = _extent(leg, r)
    axis_text_bottom = ax.xaxis.get_tightbbox(r).y0
    assert key.y1 <= axis_text_bottom + 0.5  # wholly below every tick label and the x label
    assert key.y1 < _extent(ax, r).y0  # never over the data area
    plt.close(fig)


def test_key_below_axes_without_legend_is_a_no_op(shared):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    assert shared.key_below_axes(ax, ncol=2) is None
    plt.close(fig)


def test_group_axis_labels_sit_below_the_tick_labels(shared):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(["Pre", "Post", "Pre", "Post"])
    ax.set_xlim(-0.5, 4)
    shared.group_axis_labels(ax, [("Melanoma (DiD)\nResponder", 0, 1), ("AML (Δ)\nTreatment", 2.5, 3.5)],
                             fontsize=6)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tick_bottom = min(_extent(t, r).y0 for t in ax.get_xticklabels())
    groups = [t for t in ax.texts if "\n" in t.get_text()]
    assert len(groups) == 2
    for g in groups:
        assert _extent(g, r).y1 <= tick_bottom + 0.5
    a, b = (_extent(g, r) for g in groups)
    assert a.x1 < b.x0  # the two group labels do not overlap
    plt.close(fig)


def _trajectory_data(sf7):
    genes = list(sf7._GENE_SETS)
    rng = np.random.default_rng(0)
    data = {}
    for name, arms in [("Melanoma", ["Responder", "Non-responder"]), ("AML", ["Treatment"])]:
        rows = []
        for arm in arms:
            for pid in range(4):
                for visit in ("Pre", "Post"):
                    rows.append({"participant_id": f"{name}-{arm}-{pid}", "arm": arm,
                                 "visit": visit, **{g: float(rng.normal()) for g in genes}})
        data[name] = {"gs_pv": pd.DataFrame(rows), "cfg": {"visits": ("Pre", "Post")}}
    return data


def test_supp_fig7_trajectories_label_visits_per_tick_and_groups_once(load_figure_module):
    sf7 = load_figure_module("supp/supp_fig7_cross_dataset_biology")
    fig, ax = plt.subplots(figsize=(7, 2))
    sf7._panel_paired_trajectories(ax, _trajectory_data(sf7))
    assert {t.get_text() for t in ax.get_xticklabels()} == {"Pre", "Post"}
    groups = sorted(t.get_text() for t in ax.texts)
    assert groups == ["AML (Δ)\nTreatment", "Melanoma (DiD)\nNon-responder",
                      "Melanoma (DiD)\nResponder"]
    plt.close(fig)


def test_supp_fig7_enrichment_heatmap_keeps_groups_contiguous(load_figure_module):
    sf7 = load_figure_module("supp/supp_fig7_cross_dataset_biology")
    fig, ax = plt.subplots(figsize=(7, 2))
    sf7._panel_enrichment_heatmap(ax, _trajectory_data(sf7))
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert ticks == ["Post", "Pre"] * 3  # visits sorted within each dataset and arm
    groups = [t.get_text() for t in ax.texts]
    assert groups == ["Melanoma (DiD)\nNon-responder", "Melanoma (DiD)\nResponder",
                      "AML (Δ)\nTreatment"]
    plt.close(fig)


def test_letter_at_panel_corner_clears_long_row_labels(shared):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(np.zeros((3, 3)))
    ax.set_yticks(range(3))
    ax.set_yticklabels(["A very long pathway name", "Short", "Another long pathway name"])
    ax.set_title("Leading-Edge Gene Overlap")
    shared.letter_at_panel_corner(ax, "D", fontsize=9)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    letter = _extent(fig.texts[-1], r)
    labels = [_extent(t, r) for t in ax.get_yticklabels()]
    assert letter.x0 <= min(b.x0 for b in labels) + 0.5  # at the panel's left edge
    assert letter.y1 == pytest.approx(_extent(ax.title, r).y1, abs=1.0)  # level with the title
    assert all(letter.y0 >= b.y1 - 0.5 or letter.x1 <= b.x0 for b in labels)  # not on a label
    plt.close(fig)
