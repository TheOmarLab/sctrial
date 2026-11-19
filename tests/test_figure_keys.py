"""Keys sit below their axes and grouped tick labels do not collide (Fig 4, Supp Figs 1 and 7)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_group_axis_labels_on_the_y_axis_sit_left_of_the_tick_labels(shared):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_yticks([0, 1, 2.6, 3.6])
    ax.set_yticklabels(["Pre", "Post", "Day 0", "Day 7"])
    ax.set_ylim(4.3, -0.7)
    shared.group_axis_labels(ax, [("Melanoma", 0, 1), ("Vaccine", 2.6, 3.6)], fontsize=6,
                             axis="y")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tick_left = min(_extent(t, r).x0 for t in ax.get_yticklabels())
    for g, mid in zip(ax.texts, [0.5, 3.1]):
        assert _extent(g, r).x1 <= tick_left + 0.5
        assert g.get_verticalalignment() == "center_baseline"  # as the tick labels are
        anchor = g.get_transform().transform(g.get_position())[1]
        assert anchor == pytest.approx(ax.transData.transform((0, mid))[1], abs=0.01)
    with pytest.raises(ValueError, match="axis"):
        shared.group_axis_labels(ax, [], fontsize=6, axis="z")
    plt.close(fig)


def _completeness_data():
    visits = {"TNBC": ["Pre", "Post"], "Melanoma": ["Pre", "Post"],
              "COVID-19": ["DFO_0-7", "DFO_8-14", "DFO_15+"], "Vaccine": ["Pre", "Post"],
              "AML": ["Pre", "Post"], "CAR-T": ["Pre", "Post"]}
    loaded = {}
    for name, vs in visits.items():
        obs = pd.DataFrame([{"pid": f"{name}-{i}", "visit": v} for i in range(4) for v in vs
                            if not (i == 0 and v == vs[-1])])
        loaded[name] = {"adata": SimpleNamespace(obs=obs), "pid_col": "pid", "visit_col": "visit"}
    return loaded


def test_supp_fig1_completeness_rows_do_not_overlap_at_composite_size(load_figure_module):
    sf1 = load_figure_module("supp/supp_fig1_data_quality_cohort")
    composite_rc = {"font.size": 5, "xtick.labelsize": 4.5, "ytick.labelsize": 4.5}
    with plt.rc_context(composite_rc):
        fig, ax = plt.subplots(figsize=(1.55, 1.55))  # panel H axes in the 180 mm composite
        sf1._panel_completeness_detailed(ax, _completeness_data())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ticks = ax.get_yticklabels()
    assert [t.get_text() for t in ticks] == ["Pre", "Post"] * 2 + ["DFO_0-7", "DFO_8-14",
                                             "DFO_15+"] + ["Pre", "Post"] * 3
    boxes = sorted((_extent(t, r) for t in ticks), key=lambda b: b.y0)
    assert all(a.y1 <= b.y0 + 0.25 for a, b in zip(boxes, boxes[1:]))  # rows never overlap
    groups = [t for t in ax.texts if "/" not in t.get_text()]
    assert [g.get_text() for g in groups] == list(_completeness_data())
    tick_left = min(b.x0 for b in boxes)
    group_boxes = sorted((_extent(g, r) for g in groups), key=lambda b: b.y0)
    assert all(b.x1 <= tick_left - 0.5 * 5 * fig.dpi / 72 for b in group_boxes)  # half an em clear
    assert all(a.y1 <= b.y0 for a, b in zip(group_boxes, group_boxes[1:]))
    counts = [t.get_text() for t in ax.texts if "/" in t.get_text()]
    assert counts[:2] == ["4/4", "3/4"]
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


def _problems(shared, fig):
    return shared.layout_problems(fig)


def test_layout_problems_finds_colliding_text_and_ignores_separated_text(shared):
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.text(0.2, 0.5, "Alpha", fontsize=8, transform=ax.transAxes)
    ax.text(0.22, 0.51, "Beta", fontsize=8, transform=ax.transAxes)
    ax.text(0.7, 0.2, "Gamma", fontsize=8, transform=ax.transAxes)
    found = _problems(shared, fig)["text overlaps"]
    assert len(found) == 1 and "'Alpha'" in found[0] and "'Beta'" in found[0]
    plt.close(fig)


@pytest.mark.parametrize("pitch, expect", [(1.0, False), (0.03, True)])
def test_layout_problems_tests_rotated_labels_as_rotated_boxes(shared, pitch, expect):
    fig, ax = plt.subplots(figsize=(4, 2))
    ticks = np.arange(6) * pitch
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"Category label {i}" for i in range(6)], rotation=45, ha="right",
                       fontsize=6)
    ax.set_xlim(-0.5, ticks[-1] + 0.5)
    fig.subplots_adjust(bottom=0.5)
    overlaps = _problems(shared, fig)["text overlaps"]
    assert bool(overlaps) is expect  # axis-aligned boxes of 45 degree labels always overlap
    plt.close(fig)


def test_layout_problems_flags_text_past_the_canvas_and_ignores_out_of_view_ticks(shared):
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xticks([0, 1, 50])
    ax.set_xticklabels(["in", "view", "hidden far away label"])
    ax.set_xlim(-0.5, 1.5)
    fig.subplots_adjust(left=0.2, bottom=0.3)
    ax.text(1.4, 0.5, "runs off the right edge", transform=ax.transAxes)
    problems = _problems(shared, fig)
    assert len(problems["outside canvas"]) == 1
    assert problems["outside canvas"][0].startswith("[untitled] text 'runs off the right edge' (")
    assert not problems["text overlaps"]
    plt.close(fig)


def test_layout_problems_flags_text_on_another_panel_but_not_on_a_twin(shared):
    fig, (left, right) = plt.subplots(1, 2, figsize=(6, 2))
    fig.subplots_adjust(wspace=1.0, bottom=0.2)
    left.set_title("Left")
    right.set_title("Right")
    left.text(2.3, 0.5, "intrudes", transform=left.transAxes)
    twin = left.twinx()
    twin.set_ylabel("twin label")
    on_other = _problems(shared, fig)["text on another axes"]
    assert len(on_other) == 1 and "'intrudes'" in on_other[0] and "[Right]" in on_other[0]
    plt.close(fig)


def test_layout_problems_treats_a_legend_as_one_box(shared):
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1], label="series one")
    ax.plot([0, 1], [1, 0], label="series two")
    ax.legend(loc="upper left")
    ax.text(0.03, 0.9, "under the key", transform=ax.transAxes)
    found = _problems(shared, fig)["text overlaps"]
    assert len(found) == 1 and "legend" in found[0]
    plt.close(fig)


def test_figure4_forests_share_one_row_order(load_figure_module):
    fig4 = load_figure_module("main/figure4_biological_discovery_multi_dataset")
    order = ["Apoptosis", "Hypoxia", "NK Cell Activity", "Type I IFN"]
    df = pd.DataFrame({"label": ["Type I IFN", "Apoptosis", "NK Cell Activity"],
                       "g": [0.5, -1.0, 2.0], "lo": [0.1, -1.5, 1.0], "hi": [0.9, -0.5, 3.0]})
    fig, ax = plt.subplots()
    fig4._forest_plot(ax, df, effect_col="g", ci_lo_col="lo", ci_hi_col="hi", label_col="label",
                      order=order)
    assert [t.get_text() for t in ax.get_yticklabels()] == order[::-1]  # first name on top
    rows = {round(float(np.mean(line.get_ydata())), 6): float(np.mean(line.get_xdata()))
            for line in ax.lines if len(line.get_xdata()) == 2 and line.get_ydata()[0] ==
            line.get_ydata()[1] and line.get_xdata()[0] != line.get_xdata()[1]}
    assert rows == {3.0: -1.0, 1.0: 2.0, 0.0: 0.5}  # Hypoxia (y = 2) is absent, so its row is empty
    assert ax.get_ylim() == pytest.approx((-0.6, 3.6))
    plt.close(fig)


def test_figure4_forests_keep_effect_order_by_default(load_figure_module):
    fig4 = load_figure_module("main/figure4_biological_discovery_multi_dataset")
    df = pd.DataFrame({"label": ["b", "a", "c"], "g": [0.5, -1.0, 2.0],
                       "lo": [0.1, -1.5, 1.0], "hi": [0.9, -0.5, 3.0]})
    fig, ax = plt.subplots()
    fig4._forest_plot(ax, df, effect_col="g", ci_lo_col="lo", ci_hi_col="hi", label_col="label")
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]  # largest effect on top
    plt.close(fig)


def test_supp_fig7_exhaustion_rows_name_cell_types_and_datasets_once(load_figure_module):
    anndata = pytest.importorskip("anndata")
    sf7 = load_figure_module("supp/supp_fig7_cross_dataset_biology")
    rng = np.random.default_rng(0)
    genes = ["PDCD1", "HAVCR2", "LAG3", "CTLA4", "TOX"]
    data = {}
    for name in ("AML", "CAR-T"):
        cfg = {"design": "single_arm", "participant_col": "participant_id", "visit_col": "visit",
               "arm_col": None, "visits": ("Pre", "Post"), "layer": "x"}
        obs = pd.DataFrame([{"participant_id": f"p{p}", "visit": v, "cell_type": ct}
                            for p in range(4) for v in ("Pre", "Post")
                            for ct in ("T cells", "NK cells") for _ in range(30)])
        obs.index = obs.index.astype(str)
        ad = anndata.AnnData(X=rng.normal(size=(len(obs), len(genes))).astype("float32"), obs=obs,
                             var=pd.DataFrame(index=genes))
        ad.layers["x"] = ad.X.copy()
        data[name] = {"adata": ad, "cfg": cfg}
    with plt.rc_context({"font.size": 4.5, "ytick.labelsize": 3}):
        fig, ax = plt.subplots(figsize=(2.5, 1.6))
        sf7._panel_exhaustion_by_celltype(ax, data, label_fontsize=3.0)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    assert sorted(t.get_text() for t in ax.get_yticklabels()) == ["NK cells", "NK cells",
                                                                  "T cells", "T cells"]
    groups = [t for t in ax.texts]
    assert sorted(g.get_text() for g in groups) == ["AML (Δ)", "CAR-T (Δ)"]
    tick_left = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels())
    assert all(g.get_window_extent(r).x1 < tick_left for g in groups)
    ticks = sorted(ax.get_yticks())
    assert ticks[2] - ticks[1] > ticks[1] - ticks[0]  # a gap between the two dataset blocks
    plt.close(fig)


def test_figure4_effect_heatmap_puts_marks_above_values(load_figure_module):
    fig4 = load_figure_module("main/figure4_biological_discovery_multi_dataset")
    mat = pd.DataFrame([[-1.2, 0.3], [0.8, np.nan]], index=["A", "B"], columns=["x", "y"])
    stars = pd.DataFrame([["†", ""], ["*", ""]], index=mat.index, columns=mat.columns)
    fig, ax = plt.subplots()
    fig4.panel_f_heatmap(ax, {"heatmap_matrix": mat, "heatmap_stars": stars})
    cells = {t.get_text(): t.get_position() for t in ax.texts
             if t.get_text() not in ("F",) and "FDR" not in t.get_text()}
    assert set(cells) == {"-1.2", "0.3", "0.8", "†", "*"}  # marks are their own texts
    assert cells["†"][0] == cells["-1.2"][0] and cells["†"][1] < cells["-1.2"][1]  # above
    assert cells["*"][0] == cells["0.8"][0] and cells["*"][1] < cells["0.8"][1]
    key = next(t.get_text() for t in ax.texts if "FDR" in t.get_text())
    assert key == "* FDR < 0.05   † FDR 0.05–0.1"
    plt.close(fig)


def test_keys_in_a_row_share_one_baseline(shared):
    fig, axes = plt.subplots(1, 2, figsize=(4, 2.5))
    fig.subplots_adjust(bottom=0.45)
    for ax, xlabel in zip(axes, ["one line", "three\nline\nlabel"]):
        ax.plot([0, 1], [0, 1], label="up")
        ax.plot([0, 1], [1, 0], label="down")
        ax.legend()
        ax.set_xlabel(xlabel)
    keys = [shared.key_below_axes(ax, ncol=1, fontsize=6, clear_of=list(axes)) for ax in axes]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tops = [k.get_window_extent(r).y1 for k in keys]
    assert tops[0] == pytest.approx(tops[1], abs=0.5)
    assert tops[1] <= axes[1].xaxis.get_tightbbox(r).y0 + 0.5  # below the deepest label
    plt.close(fig)


def test_letter_at_panel_corner_takes_a_shared_top(shared):
    fig, (a, b) = plt.subplots(1, 2, figsize=(4, 2))
    a.set_title("One line")
    b.set_title("Two\nlines")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top = max(ax.title.get_window_extent(r).y1 for ax in (a, b))
    shared.letter_at_panel_corner(a, "M", fontsize=8, top=top)
    shared.letter_at_panel_corner(b, "N", fontsize=8, top=top)
    fig.canvas.draw()
    m, n = (t.get_window_extent(r) for t in fig.texts[-2:])
    assert m.y1 == pytest.approx(n.y1, abs=0.5) == pytest.approx(top, abs=1.0)
    plt.close(fig)


def test_figure4_shared_forest_names_survive_a_missing_dataset(load_figure_module):
    fig4 = load_figure_module("main/figure4_biological_discovery_multi_dataset")
    order = ["Apoptosis", "Hypoxia"]
    df = pd.DataFrame({"label": order, "g": [0.5, -1.0], "lo": [0.1, -1.5], "hi": [0.9, -0.5]})
    fig, axes = plt.subplots(1, 3)
    axes[0].axis("off")  # what a panel function does when its dataset has no results
    for ax in axes[1:]:
        fig4._forest_plot(ax, df, effect_col="g", ci_lo_col="lo", ci_hi_col="hi",
                          label_col="label", order=order)
    names_ax = fig4._show_names_once(list(axes))
    fig.canvas.draw()
    assert names_ax is axes[1]
    shown = {i: [t.get_text() for t in ax.get_yticklabels() if t.get_visible()]
             for i, ax in enumerate(axes)}
    assert shown[1] == order[::-1] and shown[2] == []
    plt.close(fig)


def test_layout_problems_skips_ticks_of_hidden_axes(shared):
    fig = plt.figure(figsize=(3, 2))
    shown = fig.add_axes([0.2, 0.25, 0.7, 0.6])
    shown.set_title("Shown")
    spacer = fig.add_axes([0.2, 0.25, 0.7, 0.6])  # a spanning spacer over the same area
    spacer.axis("off")
    spacer.set_title("Spacer title")
    problems = shared.layout_problems(fig)
    assert not any("untitled" in row or "Spacer title] x tick" in row
                   for rows in problems.values() for row in rows)
    assert any("Spacer title" in row for row in problems["text overlaps"])  # titles still drawn
    plt.close(fig)


def test_figure4_effect_heatmap_key_clears_the_title(load_figure_module, shared):
    fig4 = load_figure_module("main/figure4_biological_discovery_multi_dataset")
    rng = np.random.default_rng(3)
    mat = pd.DataFrame(rng.normal(size=(6, 19)), index=[f"D{i}" for i in range(6)],
                       columns=[f"Signature {j}" for j in range(19)])
    stars = mat.map(lambda v: "*" if v > 1.2 else ("†" if v < -1.2 else ""))
    fig, ax = plt.subplots(figsize=(12, 6))  # the standalone panel's size
    fig4.panel_f_heatmap(ax, {"heatmap_matrix": mat, "heatmap_stars": stars})
    fig.tight_layout()
    overlaps = sys.modules["manuscript_figures._shared"].layout_problems(fig)["text overlaps"]
    assert not [o for o in overlaps if "FDR" in o or "title" in o]
    plt.close(fig)


def _bar_axes(heights):
    fig, ax = plt.subplots(figsize=(3.0, 2.2), dpi=150)
    ax.bar(range(len(heights)), heights, label="value")
    ax.set_ylim(0, 10)
    ax.legend()
    return fig, ax


def test_key_inside_axes_takes_the_first_clear_location(shared):
    # tall bars on the right: the preferred upper-right corner is covered, upper left is clear
    fig, ax = _bar_axes([1, 1, 1, 1, 9.5, 9.5, 9.5])
    legend, loc, covered = shared.key_inside_axes(ax, ncol=1, fontsize=6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    key = _extent(legend, renderer)
    assert covered == 0.0
    assert loc == "upper left"  # upper left
    for bar in ax.patches:
        box = _extent(bar, renderer)
        assert not (key.x0 < box.x1 and box.x0 < key.x1 and key.y0 < box.y1 and box.y0 < key.y1)
    frame = _extent(ax, renderer)
    assert frame.x0 <= key.x0 and key.x1 <= frame.x1 and frame.y0 <= key.y0 and key.y1 <= frame.y1
    plt.close(fig)


def test_key_inside_axes_keeps_the_preferred_location_when_it_is_clear(shared):
    fig, ax = _bar_axes([9.5, 9.5, 1, 1, 1, 1, 1])
    legend, loc, covered = shared.key_inside_axes(ax, ncol=1, fontsize=6)
    assert covered == 0.0 and loc == "upper right"  # upper right, the first of the default order
    # the short bars occupy the lower right, so the second choice is taken
    legend, loc, covered = shared.key_inside_axes(ax, ncol=1, fontsize=6,
                                             prefer=("lower right", "upper right"))
    assert covered == 0.0 and loc == "upper right"
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.0, 2.2), dpi=150)
    ax.bar([0, 1], [9.5, 9.5], label="value")
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 10)
    ax.legend()
    legend, loc, covered = shared.key_inside_axes(ax, ncol=1, fontsize=6,
                                             prefer=("lower right", "upper right"))
    assert covered == 0.0 and loc == "lower right"  # lower right is clear and comes first
    plt.close(fig)


def test_key_inside_axes_reports_what_it_covers_when_nothing_is_clear(shared):
    fig, ax = _bar_axes([9.9] * 7)
    ax.axhline(5, color="k")
    legend, loc, covered = shared.key_inside_axes(ax, ncol=1, fontsize=6)
    assert covered > 0.0
    plt.close(fig)


def test_key_inside_axes_counts_a_line_where_it_runs_not_its_bounding_box(shared):
    fig, ax = plt.subplots(figsize=(3.0, 2.2), dpi=150)
    ax.plot([0, 1], [0, 1], label="diagonal")  # crosses lower left and upper right only
    ax.legend()
    legend, loc, covered = shared.key_inside_axes(ax, ncol=1, fontsize=6,
                                             prefer=("upper right", "lower right"))
    assert covered == 0.0 and loc == "lower right"  # lower right, clear of the diagonal
    plt.close(fig)


def test_key_inside_axes_without_a_legend_does_nothing(shared):
    fig, ax = plt.subplots()
    assert shared.key_inside_axes(ax, ncol=1) is None
    plt.close(fig)


def test_key_inside_axes_tries_every_location_after_the_preferred_ones(shared):
    # the two preferred corners are covered; the lower left is clear
    fig, ax = plt.subplots(figsize=(3.0, 2.2), dpi=150)
    ax.bar([4, 5, 6], [9.8, 9.8, 9.8], label="value")
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 10)
    ax.legend()
    legend, loc, covered = shared.key_inside_axes(ax, ncol=1, fontsize=6,
                                                  prefer=("upper right", "lower right"))
    assert covered == 0.0 and loc == "upper left"
    plt.close(fig)


@pytest.mark.parametrize("band,side", [("top", "upper"), ("bottom", "lower")])
def test_key_inside_axes_opens_a_band_when_nothing_is_clear(shared, band, side):
    fig, ax = _bar_axes([9.9] * 7)
    ax.axhline(5, color="k")
    before = ax.get_ylim()
    legend, loc, covered = shared.key_inside_axes(ax, ncol=1, fontsize=6, band=band)
    after = ax.get_ylim()
    assert covered == 0.0 and loc.startswith(side)
    if band == "top":
        assert after[1] > before[1] and after[0] == before[0]
    else:
        assert after[0] < before[0] and after[1] == before[1]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    key, frame = _extent(legend, renderer), _extent(ax, renderer)
    assert frame.y0 <= key.y0 and key.y1 <= frame.y1
    plt.close(fig)


def test_key_inside_axes_accepts_a_reference_line_but_prefers_a_clear_place(shared):
    # bars cover the left half, a threshold line crosses the free right half
    fig, ax = plt.subplots(figsize=(3.0, 2.2), dpi=150)
    ax.bar([0, 1, 2], [9.9, 9.9, 9.9], label="value")
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0, 10)
    ax.axvline(4.9, color="k", ls="--")
    ax.legend()
    before = ax.get_ylim()
    legend, loc, covered = shared.key_inside_axes(ax, ncol=1, fontsize=6, band="top")
    assert covered == 0.0 and loc.endswith("right")
    assert ax.get_ylim() == before  # no band was opened for a reference line
    plt.close(fig)


def _label_boxes(labels, renderer):
    return [t.get_window_extent(renderer) for t in labels]


def _crowded_scatter(figsize=(3.1, 1.1), n=19, seed=3):
    """A cluster of markers around the identity line with three markers apart from it."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(0.0, 0.08, n - 3), [-0.8, 0.7, 0.75]])
    y = 1.3 * x + rng.normal(0.0, 0.25, n)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=15, zorder=3)
    ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
    ax.set_ylim(y.min() - 0.9, y.max() + 0.9)
    names = [f"Signature name {i}" for i in range(n)]
    lines = [((-2.0, -2.0), (2.0, 2.0))]
    return fig, ax, x, y, names, lines


@pytest.fixture(scope="module")
def crowded(shared):
    """One placement of the crowded scatter, shared by the tests that only inspect it."""
    fig, ax, x, y, names, lines = _crowded_scatter()
    labels, allowances = shared.place_point_labels(
        ax, x, y, names, fontsize=4.0, marker_area=15, lines=lines, extra_orders=2
    )
    fig.canvas.draw()
    yield SimpleNamespace(fig=fig, ax=ax, x=x, y=y, names=names, lines=lines, labels=labels,
                          allowances=allowances, renderer=fig.canvas.get_renderer())
    plt.close(fig)


def _on_a_line(ax, boxes, lines) -> int:
    """Number of label boxes that one of the lines passes through."""
    count = 0
    for start, end in lines:
        pts = np.linspace(*ax.transData.transform([start, end]), 4000)
        count += sum(bool(((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1)
                           & (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1)).any()) for b in boxes)
    return count


def test_place_point_labels_keeps_labels_off_markers_and_labels(crowded):
    """The rules that hold in every layout, on a scatter with little room."""
    fig, ax, r = crowded.fig, crowded.ax, crowded.renderer
    boxes = _label_boxes(crowded.labels, r)
    frame = ax.get_window_extent(r)
    markers = ax.transData.transform(np.column_stack([crowded.x, crowded.y]))
    radius = np.sqrt(15) / 2 * fig.dpi / 72

    assert [t.get_text() for t in crowded.labels] == crowded.names
    for i, a in enumerate(boxes):
        assert frame.x0 <= a.x0 and a.x1 <= frame.x1 and frame.y0 <= a.y0 and a.y1 <= frame.y1
        for b in boxes[i + 1:]:
            assert not (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1)
        for mx, my in markers:
            assert not (a.x0 - radius < mx < a.x1 + radius and a.y0 - radius < my < a.y1 + radius)
    assert crowded.allowances <= len(crowded.names)


def test_place_point_labels_keeps_labels_off_the_lines_when_there_is_room(shared):
    fig, ax, x, y, names, lines = _crowded_scatter(figsize=(4.2, 2.6), n=10)
    lines.append(((ax.get_xlim()[0], 0.0), (ax.get_xlim()[1], 0.0)))
    labels, _ = shared.place_point_labels(
        ax, x, y, names, fontsize=4.0, marker_area=15, lines=lines, extra_orders=0
    )
    fig.canvas.draw()
    assert _on_a_line(ax, _label_boxes(labels, fig.canvas.get_renderer()), lines) == 0
    plt.close(fig)


def test_place_point_labels_leaders_end_on_their_label(crowded):
    r = crowded.renderer
    boxes = {t.get_text(): t.get_window_extent(r) for t in crowded.labels}
    leaders = [t for t in crowded.ax.texts if not t.get_text()]
    assert leaders
    for leader in leaders:
        box = boxes[leader.anncoords.get_text()]
        path = leader.arrow_patch.get_path().vertices
        ends = leader.arrow_patch.get_transform().transform(path[[0, -1]])
        on_edge = [
            box.x0 - 1 <= ex <= box.x1 + 1 and box.y0 - 1 <= ey <= box.y1 + 1
            and (min(abs(ex - box.x0), abs(ex - box.x1)) <= 1 or min(abs(ey - box.y0), abs(ey - box.y1)) <= 1)
            for ex, ey in ends
        ]
        assert any(on_edge)


def test_place_point_labels_is_reproducible(shared, crowded):
    fig, ax, x, y, names, lines = _crowded_scatter()
    labels, _ = shared.place_point_labels(
        ax, x, y, names, fontsize=4.0, marker_area=15, lines=lines, extra_orders=2
    )
    assert [t.get_position() for t in labels] == [t.get_position() for t in crowded.labels]
    plt.close(fig)


def test_place_point_labels_raises_when_the_axes_cannot_hold_the_labels(shared):
    fig, ax, x, y, names, lines = _crowded_scatter(figsize=(0.9, 0.5))
    with pytest.raises(RuntimeError, match="cannot all be placed"):
        shared.place_point_labels(ax, x, y, names, fontsize=6.0, marker_area=15, lines=lines,
                                  extra_orders=0)
    plt.close(fig)


def test_segments_through_boxes(shared):
    box = np.array([0.0, 0.0, 2.0, 1.0])
    start = np.array([[-1.0, 0.5], [-1.0, 2.0], [1.0, -1.0], [-1.0, 0.0], [0.5, 0.5]])
    end = np.array([[3.0, 0.5], [3.0, 2.0], [1.0, 3.0], [3.0, 0.0], [0.6, 0.6]])
    hit = shared._segments_through_boxes(start, end, box[None, :])
    assert hit.tolist() == [True, False, True, False, True]


def test_place_point_labels_uses_a_line_only_when_no_layout_clears_it(shared):
    fig, ax = plt.subplots(figsize=(2.0, 0.5))
    x = np.linspace(-0.6, 0.6, 5)
    y = np.zeros(5)
    ax.scatter(x, y, s=15, zorder=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    names = [f"Signature {i}" for i in range(5)]
    lines = [((-1.0, level), (1.0, level)) for level in np.linspace(-0.95, 0.95, 9)]
    labels, allowances = shared.place_point_labels(
        ax, x, y, names, fontsize=5.0, marker_area=15, lines=lines, extra_orders=0
    )
    fig.canvas.draw()
    assert len(labels) == 5
    assert _on_a_line(ax, _label_boxes(labels, fig.canvas.get_renderer()), lines) > 0
    assert allowances > 0
    plt.close(fig)


def test_key_inside_axes_band_on_a_log_axis_extends_in_log_units(shared):
    fig, ax = plt.subplots(figsize=(2.2, 1.2))
    heights = [20000, 25000, 30000, 28000, 26000, 24000, 29000, 27000]
    ax.bar(range(8), heights, width=1.0, label="cells")
    ax.set_ylim(500, 32000)
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    ax.legend(loc="upper right")
    leg, loc, covered = shared.key_inside_axes(ax, ncol=1, fontsize=6, band="top")
    new_low, new_high = ax.get_ylim()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    key, frame = leg.get_window_extent(r), ax.get_window_extent(r)
    tallest = ax.transData.transform((0, max(heights)))[1]

    assert new_low == pytest.approx(low)
    assert new_high > high
    assert loc.startswith("upper") and covered == 0.0
    assert key.y0 >= tallest - 0.5 and key.y1 <= frame.y1
    # the added band is a share of the axis height, not a linear amount of cells
    share = (np.log10(new_high) - np.log10(high)) / (np.log10(new_high) - np.log10(new_low))
    assert 0.1 < share < 0.5
    plt.close(fig)
