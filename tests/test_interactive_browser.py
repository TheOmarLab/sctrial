"""Browser regressions for the packaged offline workspace."""

from __future__ import annotations

import json
import os
import re
from statistics import median

import pytest

from sctrial.interactive._document import build_workspace


@pytest.fixture(scope="module")
def browser():
    driver = pytest.importorskip(
        "playwright.sync_api", reason="Install Playwright to run browser tests"
    )
    with driver.sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("SCTRIAL_TEST_CHROMIUM")
        )
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def page(browser, tmp_path):
    document = tmp_path / "workspace.html"
    document.write_bytes(build_workspace())
    context = browser.new_context(offline=True)
    try:
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(document.as_uri())
        yield page
        assert not errors, errors
    finally:
        context.close()


def app_menu(page, name):
    page.get_by_role("menubar").get_by_role("menuitem", name=name, exact=True).click()
    return page.get_by_role("menu", name=name, exact=True)


def menu_action(page, menu, item):
    popup = app_menu(page, menu)
    command = popup.get_by_role("menuitem", name=item, exact=True)
    selection = popup.get_by_role("menuitemradio", name=item, exact=True)
    command.or_(selection).click()


def comparison_settings(page):
    page.get_by_role("button", name="Configure", exact=True).click()
    return page.get_by_role("dialog", name="Configure comparison")


def open_csv(page, path):
    from playwright.sync_api import expect

    menu_action(page, "File", "Open study")
    dialog = page.get_by_role("dialog", name="Open study")
    dialog.locator('input[type="file"]').set_input_files(path)
    expect(dialog).not_to_be_visible()


def show_comparison(page):
    page.locator("#chartMode").select_option("comparison")


def test_cell_population_scene_and_analysis_handoff(page, tmp_path):
    from playwright.sync_api import expect

    scene = page.locator("#cellStage")
    expect(page.locator("#chartMode")).to_have_value("cells")
    expect(scene).to_be_visible()
    expect(
        page.locator("#participantMatrix, #effectPlot, #pairMap, #projectSidebar")
    ).to_have_count(0)
    expect(scene).to_contain_text("Illustrative cell models")
    for mode in ("genes", "pathways", "modules", "trajectory"):
        button = page.locator(f'#scienceModes [data-mode="{mode}"]')
        expect(button).to_be_visible()
    groups = scene.get_by_role("combobox", name="Cell population")
    groups.select_option(label="CD8 T cells")
    expect(scene).to_contain_text("1,402 cells")
    expect(scene).to_contain_text("12 participants")
    scene.get_by_role("button", name="Analyze population").click()
    expect(page.locator("#chartMode")).to_have_value("comparison")
    expect(page.locator("#effectPlot")).to_be_visible()
    settings = comparison_settings(page)
    expect(settings.get_by_role("combobox", name="Cell population")).to_have_value("CD8 T cells")
    settings.get_by_role("button", name="Cancel").click()

    foreign = tmp_path / "Distinct types.csv"
    foreign.write_text(
        "participant,arm,visit,population,cells,Score\n"
        "P1,A,Baseline,Cluster X17,4,1\nP1,A,Post,Cluster X17,9,2\n"
        "P2,B,Baseline,Cluster X17,7,3\nP2,B,Post,Cluster X17,1,\n"
        "P1,A,Baseline,Other cluster,200,1\n",
        encoding="utf-8",
    )
    open_csv(page, foreign)
    expect(page.locator("#chartMode")).to_have_value("cells")
    groups.select_option(label="Cluster X17")
    expect(scene).to_contain_text("21 cells")
    expect(scene).to_contain_text("2 participants")
    page.set_viewport_size({"width": 390, "height": 844})
    expect(scene.get_by_role("button", name="Analyze population")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    scene.get_by_role("button", name="Analyze population").click()
    expect(page.locator("#effectPlot .sce-link")).to_have_count(1)


def test_plot_controls_and_participant_inspection(page):
    from playwright.sync_api import expect

    show_comparison(page)

    plot = page.locator("#effectPlot")
    expect(plot).to_be_visible()
    expect(page.locator("#chartMode")).to_have_value("comparison")
    expect(page.locator("#participantMatrix, #pairMap, #projectSidebar")).to_have_count(0)
    expect(plot).to_contain_text("+0.314")
    expect(plot).to_contain_text("95% CI")
    expect(plot.locator(".sce-link")).to_have_count(10)
    mark = plot.locator(".sce-observation").first
    mark.click()
    dialog = page.get_by_role("dialog", name=re.compile("^Participant "))
    expect(dialog).to_contain_text("Baseline")
    dialog.get_by_role("button", name="Close record").click()
    mark.press("Enter")
    expect(dialog).to_be_visible()
    dialog.get_by_role("button", name="Close record").click()

    file_button = page.get_by_role("menubar").get_by_role("menuitem", name="File", exact=True)
    file_button.press("Enter")
    menu = page.get_by_role("menu", name="File", exact=True)
    expect(menu).to_be_visible()
    menu.get_by_role("menuitem").first.press("Escape")
    expect(menu).not_to_be_visible()
    expect(file_button).to_be_focused()


def test_reversed_contrast_and_frozen_pin(page):
    from playwright.sync_api import expect

    show_comparison(page)

    plot = page.locator("#effectPlot")
    settings = comparison_settings(page)
    settings.get_by_role("combobox", name="Follow-up visit").select_option("Week 12")
    settings.get_by_role("button", name="Apply changes").press("Enter")
    page.get_by_role("button", name="Undo contrast change").click()
    expect(plot).to_contain_text("+0.314")
    arm_a = plot.get_by_role("group", name="Arm A participant observations")
    original_color = arm_a.locator(".sce-arm-label").evaluate(
        "(label) => getComputedStyle(label).fill"
    )
    menu_action(page, "Analysis", "Pin contrast")
    settings = comparison_settings(page)
    settings.get_by_role("combobox", name="Group order").select_option("Arm B")
    settings.get_by_role("button", name="Apply changes").click()
    expect(plot).to_contain_text("−0.314")
    assert (
        arm_a.locator(".sce-arm-label").evaluate("(label) => getComputedStyle(label).fill")
        == original_color
    )
    menu_action(page, "Analysis", "Pinned contrasts")
    page.get_by_role("button", name=re.compile(r"^Reopen ")).click()
    expect(plot).to_contain_text("+0.314")
    expect(page.locator("#chartMode")).to_have_value("comparison")


def test_private_csv_feature_specific_participants(page, tmp_path):
    from playwright.sync_api import expect

    study = tmp_path / "Independent.csv"
    study.write_text(
        "participant,arm,visit,population,cells,Response,Marker\n"
        "P1,A,Baseline,T,2,0,2\nP1,A,Post,T,80,2,4\n"
        "P2,A,Baseline,T,60,1,2\nP2,A,Post,T,3,5,7\n"
        "P3,B,Baseline,T,12,1,0\nP3,B,Post,T,5,2,1\nP4,B,Baseline,T,90,0,1\n"
        "P5,B,Baseline,T,22,2,0\nP5,B,Post,T,19,,2\n",
        encoding="utf-8",
    )
    open_csv(page, study)
    show_comparison(page)

    plot = page.locator("#effectPlot")
    expect(plot).to_contain_text("Descriptive")
    expect(plot.locator(".sce-link")).to_have_count(3)
    expect(plot.locator(".sce-ci")).to_have_count(0)
    view = page.locator("#chartMode")
    view.select_option("measurements")
    matrix = page.get_by_role("region", name="Participant measurements")
    expect(matrix.get_by_role("button", name=re.compile(r"^Included"))).to_contain_text("3")
    matrix.get_by_role("button", name=re.compile(r"^Excluded")).click()
    row = matrix.get_by_role("row").filter(
        has=page.get_by_role("button", name="Inspect participant P5")
    )
    expect(row.get_by_role("cell").nth(1)).to_have_attribute("aria-label", "Missing measurement")
    expect(matrix.get_by_role("button", name="Inspect participant P1")).to_have_count(0)
    view.select_option("trajectory")
    visit_counts = page.get_by_role("table", name="Participants measured at each visit")
    group_b = visit_counts.get_by_role("row").filter(
        has=page.get_by_role("rowheader", name="B", exact=True)
    )
    expect(group_b.get_by_role("cell")).to_have_text(["3", "1"])
    expect(page.get_by_text("Mean at each visit", exact=True)).to_be_visible()
    view.select_option("comparison")
    page.get_by_role("combobox", name="Outcome", exact=True).select_option(label="Marker")
    expect(plot.locator(".sce-link")).to_have_count(4)

    menu_action(page, "View", "Results table")
    table = page.get_by_role("table", name="Descriptive participant-level contrasts")
    row = table.get_by_role("row").filter(has_text="Response")
    expect(row.get_by_role("cell").first).to_have_text("+2.00")
    expect(row.get_by_role("cell").last).to_have_text("3")
    marker = table.get_by_role("row").filter(has_text="Marker")
    expect(marker.get_by_role("cell").last).to_have_text("4")
    expect(table.get_by_role("columnheader", name="95% CI")).to_have_count(0)

    assert page.evaluate("localStorage.length + sessionStorage.length") == 0


def test_molecular_views_use_measured_data_and_time_cursor(page):
    from playwright.sync_api import expect

    page.locator('#scienceModes [data-mode="genes"]').click()
    gene = page.get_by_role("combobox", name="Gene", exact=True)
    gene.select_option(label="IFIT1")
    stage = page.locator("#molecularStage")
    study = json.loads(page.locator("#dataset").text_content())
    gene_id = next(f["id"] for f in study["features"] if f["label"] == "IFIT1")
    observed = next(
        r["scores"][gene_id]
        for r in study["observations"]
        if (r["id"], r["population"], r["visit"]) == ("A01", "All annotated cells", "Baseline")
    )
    tile = stage.locator(f'[data-gene-cell][data-id="A01"][data-feature="{gene_id}"]')
    assert float(tile.get_attribute("data-value")) == pytest.approx(observed)
    page.locator('#scienceModes [data-mode="pathways"]').click()
    expect(stage.locator("[data-membership]")).to_have_count(3)
    expect(stage.locator("figcaption")).to_contain_text("membership")
    pathway = stage.get_by_role("combobox", name="Pathway", exact=True).input_value()
    baseline_rows = [
        r
        for r in study["observations"]
        if (r["population"], r["visit"]) == ("All annotated cells", "Baseline")
    ]
    pathway_values = [
        r["scores"][pathway] for r in baseline_rows if r["scores"].get(pathway) is not None
    ]
    hub = stage.locator("[data-hub-mean]")
    assert int(hub.get_attribute("data-n")) == len(pathway_values)
    assert float(hub.get_attribute("data-hub-mean")) == pytest.approx(
        sum(pathway_values) / len(pathway_values)
    )
    page.locator('#scienceModes [data-mode="modules"]').click()
    score_id = stage.get_by_role("combobox", name="Score", exact=True).input_value()
    reading = stage.locator(
        f'[data-reading][data-id="A01"][data-feature="{score_id}"][data-visit="Baseline"]'
    )
    expect(reading).to_have_count(1)
    arm_values = [
        r["scores"][score_id]
        for r in baseline_rows
        if r["arm"] == "Arm A" and r["scores"].get(score_id) is not None
    ]
    mark = stage.locator(f'[data-median="{score_id}"][data-arm="Arm A"]')
    assert float(mark.get_attribute("data-value")) == pytest.approx(median(arm_values))

    page.locator('#scienceModes [data-mode="trajectory"]').click()
    visits = page.get_by_role("slider", name="Browse observed visits")
    visits.press("End")
    expect(page.locator("#trajectoryFrame")).to_contain_text("Week 12")
    expect(page.get_by_role("combobox", name="Follow-up visit")).to_have_value("Week 4")
    visits.press("Home")
    page.get_by_role("button", name="Play visits").click()
    expect(page.locator("#trajectoryFrame")).to_contain_text("Week 12", timeout=7000)
    expect(page.get_by_role("button", name="Play visits")).to_be_visible()
    show_comparison(page)
    selected = page.get_by_role("combobox", name="Outcome", exact=True).input_value()
    result = next(
        r
        for r in study["results"]["did|Arm A|Week 4|All annotated cells"]
        if r["feature"] == selected
    )
    estimate = page.locator("#effectPlot .sce-estimate").get_attribute("aria-label")
    assert float(estimate.split()[-1].replace("−", "-")) == pytest.approx(result["estimate"])


def test_typed_molecular_data_and_unmapped_views(page, tmp_path):
    from playwright.sync_api import expect

    study = tmp_path / "Measured features.csv"
    study.write_text(
        "participant,arm,visit,population,cells,gene:ACTB,pathway:My pathway,module:My score\n"
        "P1,A,Baseline,T,2,0,3,2\nP1,A,Post,T,80,4,4,5\n"
        "P2,A,Baseline,T,60,1,4,2\nP2,A,Post,T,3,,6,8\n"
        "P3,B,Baseline,T,12,3,7,5\nP3,B,Post,T,5,2,6,6\n",
        encoding="utf-8",
    )
    open_csv(page, study)

    def science(mode):
        page.locator(f'#scienceModes [data-mode="{mode}"]').click()
        return page.locator("#molecularStage")

    surface = science("genes")
    expect(surface.get_by_role("combobox", name="Gene", exact=True)).to_contain_text("ACTB")
    surface.get_by_role("combobox", name="Visit", exact=True).select_option(label="Baseline")
    exact_zero = surface.locator('[data-gene-cell][data-id="P1"][data-value="0"]')
    expect(exact_zero).to_have_count(1)
    surface.get_by_role("combobox", name="Visit", exact=True).select_option(label="Post")
    expect(surface.locator('[data-gene-cell][data-id="P1"][data-value="4"]')).to_have_count(1)
    expect(surface.locator('[data-gene-cell][data-id="P2"][data-value="0"]')).to_have_count(0)
    surface = science("pathways")
    expect(surface).to_contain_text("My pathway")
    expect(surface.locator("[data-membership]")).to_have_count(0)
    gene_sets = tmp_path / "membership.gmt"
    gene_sets.write_text("My pathway\tStudy definition\tACTB\tUNMEASURED\n", encoding="utf-8")
    surface.get_by_role("button", name="Link gene set…").click()
    mapping = page.get_by_role("dialog", name="Link gene set")
    mapping.locator('input[type="file"]').set_input_files(gene_sets)
    mapping.get_by_role("button", name="Link membership").click()
    expect(surface.locator("[data-membership]")).to_have_count(2)
    expect(surface).to_contain_text("UNMEASURED")
    surface = science("modules")
    expect(surface).to_contain_text("My score")
    expect(surface.locator('[data-reading][data-id="P1"][data-value="5"]')).to_have_count(1)

    untyped = tmp_path / "No gene metadata.csv"
    untyped.write_text(
        "participant,arm,visit,population,cells,Unknown score\nP1,A,Post,T,5,0\nP2,A,Post,T,7,10\n",
        encoding="utf-8",
    )
    open_csv(page, untyped)
    for mode in ("genes", "pathways"):
        empty = science(mode)
        expect(empty.get_by_role("button", name="Open study", exact=True)).to_be_visible()
        expect(empty.locator("[data-gene-cell], [data-membership]")).to_have_count(0)


def test_single_arm_and_cross_sectional_inputs(page, tmp_path):
    from playwright.sync_api import expect

    paired = tmp_path / "One arm.csv"
    paired.write_text(
        "participant,arm,visit,population,cells,Score\n"
        "P1,A,Baseline,T,10,1\nP1,A,Post,T,5,4\n"
        "P2,A,Baseline,T,6,2\nP2,A,Post,T,16,6\n",
        encoding="utf-8",
    )
    open_csv(page, paired)
    show_comparison(page)
    plot = page.locator("#effectPlot")
    expect(plot.locator(".sce-link")).to_have_count(2)
    expect(plot).to_contain_text("+3.5")
    settings = comparison_settings(page)
    expect(settings.get_by_role("combobox", name="Study design")).to_have_value("paired")
    settings.get_by_role("button", name="Cancel").click()

    cross = tmp_path / "Independent groups.csv"
    cross.write_text(
        "participant,arm,visit,population,cells,Score\n"
        "P1,A,Day 1,T,5,5\nP2,A,Day 1,T,9,1\n"
        "P3,B,Day 1,T,6,1\nP4,B,Day 1,T,16,2\n",
        encoding="utf-8",
    )
    open_csv(page, cross)
    show_comparison(page)
    expect(plot.locator(".sce-link")).to_have_count(0)
    expect(plot).to_contain_text("+1.5")
    settings = comparison_settings(page)
    expect(settings.get_by_role("combobox", name="Study design")).to_have_value("cross")
    expect(settings.get_by_role("combobox", name="Selected visit")).to_have_value("Day 1")


def test_sheet_reopens_its_analysis_and_mobile_navigation(page):
    from playwright.sync_api import expect

    show_comparison(page)

    outcome = page.get_by_role("combobox", name="Outcome", exact=True)
    outcome.select_option(label="Cytotoxicity")
    page.get_by_role("button", name="New analysis").click()
    show_comparison(page)
    expect(page.get_by_role("tab", name="Analysis 2", exact=True)).to_have_attribute(
        "aria-selected", "true"
    )
    expect(outcome).to_have_value("interferon")

    page.get_by_role("button", name="Actions for Analysis 2").click()
    page.get_by_role("menuitem", name="Rename").click()
    title = page.get_by_role("textbox", name="Analysis name")
    title.fill("Sensitivity analysis")
    title.press("Enter")
    expect(page.get_by_role("tab", name="Sensitivity analysis", exact=True)).to_have_attribute(
        "aria-selected", "true"
    )
    page.get_by_role("tab", name="Primary analysis", exact=True).click()
    expect(outcome).to_have_value("cytotoxicity")

    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("tab", name="Sensitivity analysis", exact=True).click()
    expect(outcome).to_have_value("interferon")
    menu = app_menu(page, "Menu")
    expect(menu).to_be_visible()
    menu.get_by_role("menuitem").first.press("Escape")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def test_workbook_download_and_reopen(page, tmp_path):
    from playwright.sync_api import expect

    show_comparison(page)

    page.get_by_role("button", name="New analysis").click()
    show_comparison(page)
    page.get_by_role("combobox", name="Outcome", exact=True).select_option(label="Proliferation")
    with page.expect_download() as downloaded:
        menu_action(page, "File", "Save workbook")
    archive = tmp_path / downloaded.value.suggested_filename
    downloaded.value.save_as(archive)
    content = json.loads(archive.read_text(encoding="utf-8"))
    assert [sheet["title"] for sheet in content["sheets"]] == ["Primary analysis", "Analysis 2"]
    assert "participants" not in content["study"]
    assert "observations" not in content["study"]

    page.get_by_role("tab", name="Primary analysis", exact=True).click()
    menu_action(page, "File", "Open study")
    with page.expect_file_chooser() as picker:
        page.get_by_role("button", name=re.compile(r"^Open analysis workbook")).click()
    picker.value.set_files(archive)
    expect(page.get_by_role("tab", name="Analysis 2 (2)", exact=True)).to_have_attribute(
        "aria-selected", "true"
    )
    expect(page.get_by_role("combobox", name="Outcome", exact=True)).to_have_value("proliferation")
    page.get_by_role("tab", name="Primary analysis", exact=True).click()
    expect(page.get_by_role("combobox", name="Outcome", exact=True)).to_have_value("interferon")
