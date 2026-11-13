"""Layout helpers behind Supp Figs 2, 6 and 8: alluvial geometry, keys, tick labels."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("plotly")
pytest.importorskip("scanpy")

_MM = 1 / 25.4


@pytest.fixture
def sf2(load_figure_module):
    return load_figure_module("supp/supp_fig2_annotation_baseline")


def _flows():
    """Twelve cell types, two of them in very few participants, across two arms."""
    counts = {"B cell": (9, 8), "CD4+ T": (10, 9), "CD8+ T": (10, 9), "DC": (8, 7),
              "Erythroid": (1, 0), "HSC/Prog": (1, 1), "Monocyte": (10, 9), "NK": (9, 9),
              "Other": (3, 2), "Plasma": (6, 5), "T other": (10, 9), "Treg": (9, 8)}
    rows = [{"Cell type": ct, "Arm": arm, "count": n}
            for ct, pair in counts.items() for arm, n in zip(("Chemo", "anti-PDL1+Chemo"), pair)
            if n > 0]
    return pd.DataFrame(rows)


def _bars(ax, x0):
    return [p for p in ax.patches
            if isinstance(p, mpatches.Rectangle) and np.isclose(p.get_x(), x0)]


def test_alluvial_sides_share_one_height_scale(sf2):
    flow = _flows()
    fig = plt.figure(figsize=(30 * _MM, 40 * _MM))
    ax = fig.add_axes([0.35, 0.05, 0.3, 0.85])
    sf2._draw_alluvial(ax, flow, "Arm", fontsize=4)
    left = sum(p.get_height() for p in _bars(ax, 0.0))
    # The right side draws coloured segments and an outline per arm; compare outlines.
    right = sum(p.get_height() for p in _bars(ax, 0.95) if p.get_facecolor()[3] == 0)
    assert left == pytest.approx(right)
    per_participant = left / flow["count"].sum()
    for p in _bars(ax, 0.0):
        assert p.get_height() / per_participant == pytest.approx(round(p.get_height() / per_participant))
    plt.close(fig)


def test_alluvial_labels_never_overlap(sf2):
    import manuscript_figures._shared as sh

    fig = plt.figure(figsize=(30 * _MM, 40 * _MM))
    ax = fig.add_axes([0.35, 0.05, 0.3, 0.8])
    sf2._draw_alluvial(ax, _flows(), "Arm", fontsize=4)
    problems = sh.layout_problems(fig)
    assert problems["text overlaps"] == []
    plt.close(fig)


def test_standalone_alluvial_title_clears_headers(sf2):
    import manuscript_figures._shared as sh

    fig = sf2._alluvial_standalone("TNBC", {"flow": _flows(), "right_label": "Arm"})
    problems = sh.layout_problems(fig)
    assert problems["text overlaps"] == [] and problems["outside canvas"] == []
    ax = fig.axes[0]
    r = fig.canvas.get_renderer()
    headers = max(t.get_window_extent(r).y1 for t in ax.texts if t.get_text() in ("Cell type", "Arm"))
    assert (ax.title.get_window_extent(r).y0 - headers) * 72 / fig.dpi >= 4  # a visible gap, pt
    plt.close(fig)


def test_spread_labels_keeps_gap_and_bounds(sf2):
    y = np.array([0.10, 0.11, 0.12, 0.50, 0.98, 0.99])
    out = sf2._spread_labels(y, 0.05, 0.02, 0.98)
    s = np.sort(out)
    assert np.all(np.diff(s) >= 0.05 - 1e-12)
    assert s[0] >= 0.02 - 1e-12 and s[-1] <= 0.98 + 1e-12
    assert np.argsort(out).tolist() == np.argsort(y).tolist()  # order is preserved


def test_key_below_axes_on_axes_without_x_text(load_figure_module):
    load_figure_module("supp/supp_fig2_annotation_baseline")
    import manuscript_figures._shared as sh

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.scatter([0, 1], [0, 1], label="a")
    ax.scatter([1, 0], [0, 1], label="b")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    leg = sh.key_below_axes(ax, ncol=2)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    assert leg.get_window_extent(r).y1 <= ax.get_window_extent(r).y0
    plt.close(fig)


def test_family_tick_labels_grouped_in_a_composite_cell(load_figure_module):
    load_figure_module("supp/supp_fig6_benchmark_calibration")
    import manuscript_figures._benchmark as bm
    import manuscript_figures._shared as sh

    labels = ["50 cells/PV", "250 cells/PV", "1,000 cells/PV", "emp. het. yield", "10% miss",
              "20% miss", "3:7 arm imbalance", "5:10 arm imbalance", "10:20 arm imbalance",
              "het. effect", "comp. stress"]
    fig = plt.figure(figsize=(180 * _MM, 40 * _MM))
    ax = fig.add_axes([0.55, 0.45, 0.4, 0.4])  # half-width cell, as panels K to N
    ax.set_xticks(range(len(labels)))
    bm._family_ticklabels(ax, labels, composite=True, fontsize=3.8)
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "50", "250", "1,000", "emp.\nhet.\nyield", "10%", "20%", "3:7", "5:10", "10:20",
        "het.\neffect", "comp.\nstress"]
    assert [t.get_text() for t in ax.texts] == ["cells/PV", "missing", "arm imbalance"]
    problems = sh.layout_problems(fig)
    assert problems["text overlaps"] == [] and problems["outside canvas"] == []
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_xticks(range(len(labels)))
    bm._family_ticklabels(ax, labels, composite=False, fontsize=8)
    assert [t.get_text() for t in ax.get_xticklabels()] == labels  # standalone unchanged
    assert all(t.get_rotation() == 20 for t in ax.get_xticklabels())
    plt.close(fig)


def test_figure3_bootstrap_labels_are_reproducible(load_figure_module):
    fig3 = load_figure_module("main/figure3_robustness_benchmarking")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"feature": [f"Signature {i}" for i in range(12)],
                       "se_DiD": rng.uniform(0.6, 1.4, 12)})
    df["se_DiD_boot"] = df["se_DiD"] * rng.uniform(0.8, 1.1, 12)

    def layout(global_seed):
        np.random.seed(global_seed)  # whatever state the caller left
        fig, ax = plt.subplots(figsize=(4, 3))
        pts = fig3._panel_a_single(ax, {"df_boot": df}, title="t", dataset="melanoma")
        fig3._label_panel_a(ax, pts)
        out = [(t.get_text(), t.get_position()) for t in ax.texts]
        plt.close(fig)
        return out

    first = layout(1)
    assert first == layout(2)
    assert first  # signatures were named


def test_figure3_forest_rows_grouped_and_clear(load_figure_module):
    fig3 = load_figure_module("main/figure3_robustness_benchmarking")
    import manuscript_figures._shared as sh

    sigs = ["Cytotoxic T Cell Activity", "T Cell Exhaustion", "Inflammatory Response",
            "Type I Interferon", "T Cell Activation"]
    rng = np.random.default_rng(3)
    rows = []
    for ds, n in (("TNBC", 12), ("Sade-Feldman", 11), ("Vaccine", 6)):
        for s in sigs:
            d = float(rng.normal(0, 0.8))
            rows.append({"dataset": ds, "signature": s, "d": d, "d_lower": d - 1,
                         "d_upper": d + 1, "n_participants": n})
    data = {"scale_data": {"effect_df": pd.DataFrame(rows)}}
    fig, ax = plt.subplots(figsize=(4, 4))
    fig3._panel_e_cross_dataset(ax, data)
    fig.tight_layout()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 15 and "TNBC" not in labels  # no header rows
    texts = [t.get_text() for t in ax.texts]
    assert "TNBC\nn = 12" in texts and "Melanoma\nn = 11" in texts  # one group label each
    assert sum(t.startswith(("+", "-")) for t in texts) == 15  # one value per row
    problems = sh.layout_problems(fig)
    assert problems["text overlaps"] == []
    plt.close(fig)


_NEBULA_LOG = """\
MODEL-COMPATIBLE NULL: DGP matches NEBULA's assumed model exactly
  mean beta      : -0.0003  (MCSE 0.0003)
  FPR at 0.05    : 0.0473

SIGMA_U ABLATION: is the participant-by-visit level responsible?
   sigma_u   mean beta     MCSE  FPR@0.05  FPR MCSE  converged
     0.000     +0.0002   0.0003    0.0489    0.0025     1.0000
     0.250     -0.0044   0.0016    0.6837    0.0058     1.0000
     0.500     -0.0081   0.0034    0.8158    0.0052     1.0000
     0.768     -0.0154   0.0073    0.8651    0.0037     1.0000

WORKAROUND DIAGNOSTIC: id = participant x visit, participant fixed effects
  standard (id=participant)              FPR 0.8530 +/- 0.0080   mean beta -0.0261
  workaround (id=participant x visit)    FPR 0.2245 +/- 0.0109   mean beta -0.0127
"""


@pytest.mark.parametrize("composite", [True, False])
def test_nebula_hierarchy_annotations_do_not_overlap(load_figure_module, tmp_path, composite):
    sf6 = load_figure_module("supp/supp_fig6_benchmark_calibration")
    import manuscript_figures._shared as sh

    (tmp_path / "nebula_hierarchy_diagnostic.log").write_text(_NEBULA_LOG)
    sf6._DIAG_DIR = tmp_path
    if composite:  # the panel's own cell in the composite, where the axes are 15 mm tall
        fig, cell = sf6._composite_cells()
        ax = fig.add_subplot(cell["DGH"].subgridspec(1, 3, width_ratios=sf6._DGH_WIDTHS,
                                                     wspace=0.32)[1])
        sf6._panel_nebula_hierarchy(fig, ax=ax, composite=True)
    else:
        fig = plt.figure(figsize=(7, 5))
        sf6._panel_nebula_hierarchy(fig)
        ax = fig.axes[0]
    assert len(ax.texts) == 2  # the calibrated-sigma label and the control box
    problems = sh.layout_problems(fig)
    assert problems["text overlaps"] == [] and problems["outside canvas"] == []
    plt.close(fig)


@pytest.mark.parametrize("scale", [1.0, 0.35])
def test_size_key_matches_dot_areas(sf2, scale):
    fig, ax = plt.subplots(figsize=(3, 3))
    sf2._size_key(ax, fontsize=5, title_fontsize=6, scale=scale)
    sizes = [h.get_markersize() for h in ax.get_legend().legend_handles]
    # a dot for a fraction f has area f * 200 * scale pt^2, i.e. diameter sqrt of that
    assert sizes == pytest.approx([np.sqrt(f * 200 * scale) for f in (0.25, 0.5, 0.75)])
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    marks = [h.get_window_extent(r) for h in ax.get_legend().legend_handles]
    for upper, lower in zip(marks, marks[1:]):  # rows far enough apart that circles do not touch
        assert upper.y0 - lower.y1 >= 0
    plt.close(fig)


def test_key_below_axes_keeps_title_only_when_asked(load_figure_module):
    load_figure_module("supp/supp_fig2_annotation_baseline")
    import manuscript_figures._shared as sh

    for keep in (False, True):
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [0, 1], label="a")
        ax.legend(title="Runtime relative to sctrial")
        leg = sh.key_below_axes(ax, ncol=1, keep_title=keep)
        assert leg.get_title().get_text() == ("Runtime relative to sctrial" if keep else "")
        plt.close(fig)
