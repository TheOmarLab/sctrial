"""Browser regressions for the packaged offline workspace."""

from __future__ import annotations

import json
import os
import re

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


def atlas(page):
    return page.locator("[data-atlas]")


def layer(page, name):
    atlas(page).get_by_role("button", name=name, exact=True).click()


def select(page, control, label):
    atlas(page).get_by_role("combobox", name=control, exact=True).select_option(label=label)


def open_plot(page):
    atlas(page).get_by_role("button", name="Plot results", exact=True).click()
    return page.locator("#atlasResultMount #effectPlot")


def comparison_settings(page):
    menu_action(page, "Analysis", "Configure comparison…")
    return page.get_by_role("dialog", name="Configure comparison")


def open_csv(page, path):
    from playwright.sync_api import expect

    menu_action(page, "File", "Open study")
    dialog = page.get_by_role("dialog", name="Open study")
    dialog.locator('input[type="file"]').first.set_input_files(path)
    expect(dialog).not_to_be_visible()


def select_population(page, name="CD8 T cells"):
    select(page, "Cell population", name)


def test_cluster_workspace_persists_across_layers_and_plot(page):
    from playwright.sync_api import expect

    stage = atlas(page).locator(".at-scene")
    expect(stage).to_be_visible()
    scene = stage.locator("canvas").first.element_handle()
    select_population(page)
    for name, control in (
        ("Expression", "Gene"),
        ("Pathways", "Pathway"),
        ("Scores", "Module score"),
    ):
        layer(page, name)
        expect(atlas(page).get_by_role("button", name=name, exact=True)).to_have_attribute(
            "aria-pressed", "true"
        )
        expect(atlas(page).get_by_role("combobox", name=control, exact=True)).to_be_visible()
        assert stage.locator("canvas").first.evaluate("(canvas, first) => canvas === first", scene)
    plot = open_plot(page)
    expect(plot).to_be_visible()
    expect(stage).to_be_visible()
    expect(plot).to_contain_text("95% CI")
    atlas(page).get_by_role("button", name="Close results").click()
    expect(plot).not_to_be_visible()
    assert stage.locator("canvas").first.evaluate("(canvas, first) => canvas === first", scene)


def test_private_measurements_use_participants_at_selected_visit(page, tmp_path):
    from playwright.sync_api import expect

    study = tmp_path / "Independent.csv"
    study.write_text(
        "participant,arm,visit,population,cells,gene:ACTB,pathway:My pathway,module:Score\n"
        "P1,A,Baseline,T,1,0,1,0\nP1,A,Post,T,80,2,3,2\n"
        "P2,A,Baseline,T,101,10,1,1\nP2,A,Post,T,3,4,5,5\n"
        "P3,B,Baseline,T,12,2,1,1\nP3,B,Post,T,5,,2,2\n"
        "P4,B,Baseline,B,1,,1,1\nP4,B,Post,B,1,0,3,2\n",
        encoding="utf-8",
    )
    open_csv(page, study)
    layer(page, "Expression")
    select(page, "Gene", "ACTB")
    select_population(page, "T")
    select(page, "Visit", "Baseline")
    group = atlas(page).locator('[data-population="T"][data-value]')
    expect(group).to_have_attribute("data-n", "3")
    assert float(group.get_attribute("data-value")) == pytest.approx(4)
    other = atlas(page).locator('[data-population="B"][data-n]')
    expect(other).to_have_attribute("data-n", "0")
    select(page, "Visit", "Post")
    expect(group).to_have_attribute("data-n", "2")
    assert float(group.get_attribute("data-value")) == pytest.approx(3)
    expect(other).to_have_attribute("data-value", "0")
    layer(page, "Pathways")
    select(page, "Pathway", "My pathway")
    expect(atlas(page).get_by_role("combobox", name="Visit", exact=True)).to_have_value("Post")
    plot = open_plot(page)
    expect(plot).to_contain_text("Descriptive")
    expect(plot.locator(".sce-link")).to_have_count(3)
    expect(plot.locator(".sce-ci")).to_have_count(0)
    # (mean of +2 and +4 in A) minus (+1 in B).
    expect(plot.locator(".sce-estimate")).to_have_attribute("aria-label", "Estimate 2")
    view = page.locator("#atlasPlotType")
    view.select_option("measurements")
    matrix = page.get_by_role("region", name="Participant measurements")
    expect(matrix.get_by_role("button", name=re.compile(r"^Included"))).to_contain_text("3")
    view.select_option("trajectory")
    counts = page.get_by_role("table", name="Participants measured at each visit")
    expect(
        counts.get_by_role("row")
        .filter(has=page.get_by_role("rowheader", name="A", exact=True))
        .get_by_role("cell")
    ).to_have_text(["2", "2"])
    expect(
        counts.get_by_role("row")
        .filter(has=page.get_by_role("rowheader", name="B", exact=True))
        .get_by_role("cell")
    ).to_have_text(["1", "1"])
    assert page.evaluate("localStorage.length + sessionStorage.length") == 0


def test_gene_contrast_matches_selected_population_and_visit(page):
    from playwright.sync_api import expect

    select_population(page)
    layer(page, "Expression")
    select(page, "Gene", "IFIT1")
    select(page, "Visit", "Week 12")
    study = json.loads(page.locator("#dataset").text_content())
    feature = next(f for f in study["features"] if f["label"] == "IFIT1")
    reference = next(
        r
        for r in study["results"]["did|Arm A|Week 12|CD8 T cells"]
        if r["feature"] == feature["id"]
    )
    plot = open_plot(page)
    expect(plot).to_contain_text("Week 12")
    estimate = plot.locator(".sce-estimate").get_attribute("aria-label")
    assert float(estimate.split()[-1].replace("−", "-")) == pytest.approx(reference["estimate"])
    expect(plot.locator(".sce-link")).to_have_count(reference["n"])
    mark = plot.locator(".sce-observation").first
    mark.press("Enter")
    dialog = page.get_by_role("dialog", name=re.compile("^Participant "))
    expect(dialog).to_contain_text("Week 12")
    dialog.get_by_role("button", name="Close dialog").click()
    expect(atlas(page).locator(".at-scene")).to_be_visible()


def test_reverse_arm_direction_and_reopen_frozen_pin(page):
    from playwright.sync_api import expect

    select_population(page)
    layer(page, "Scores")
    select(page, "Module score", "Cytotoxicity")
    select(page, "Visit", "Week 4")
    plot = open_plot(page)
    estimate = plot.locator(".sce-estimate")
    initial = estimate.get_attribute("aria-label")
    menu_action(page, "Analysis", "Pin contrast")
    settings = comparison_settings(page)
    settings.get_by_role("combobox", name="Group order").select_option("Arm B")
    settings.get_by_role("button", name="Apply changes").click()
    expect(estimate).not_to_have_attribute("aria-label", initial)
    reversed_value = float(estimate.get_attribute("aria-label").split()[-1].replace("−", "-"))
    assert reversed_value == pytest.approx(-float(initial.split()[-1].replace("−", "-")))
    menu_action(page, "Analysis", "Pinned contrasts")
    page.get_by_role("button", name=re.compile(r"^Reopen ")).click()
    expect(estimate).to_have_attribute("aria-label", initial)


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
    plot = open_plot(page)
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
    open_plot(page)
    expect(plot.locator(".sce-link")).to_have_count(0)
    expect(plot).to_contain_text("+1.5")
    settings = comparison_settings(page)
    expect(settings.get_by_role("combobox", name="Study design")).to_have_value("cross")


def test_analysis_workbook_restores_layer_population_and_feature(page, tmp_path):
    from playwright.sync_api import expect

    select_population(page)
    layer(page, "Expression")
    select(page, "Gene", "IFIT1")
    select(page, "Visit", "Week 12")
    with page.expect_download() as downloaded:
        menu_action(page, "File", "Save workbook")
    archive = tmp_path / downloaded.value.suggested_filename
    downloaded.value.save_as(archive)
    content = json.loads(archive.read_text(encoding="utf-8"))
    assert "participants" not in content["study"]
    assert "observations" not in content["study"]
    layer(page, "Scores")
    menu_action(page, "File", "Open study")
    with page.expect_file_chooser() as picker:
        page.get_by_role("button", name=re.compile(r"^Open analysis workbook")).click()
    picker.value.set_files(archive)
    expect(atlas(page).get_by_role("button", name="Expression", exact=True)).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(atlas(page).get_by_role("combobox", name="Cell population", exact=True)).to_have_value(
        "CD8 T cells"
    )
    expect(atlas(page).get_by_role("combobox", name="Visit", exact=True)).to_have_value("Week 12")
    expect(
        atlas(page).get_by_role("combobox", name="Gene", exact=True).locator("option:checked")
    ).to_have_text("IFIT1")


@pytest.mark.parametrize("width,height", [(390, 844), (1280, 800)])
def test_atlas_playback_and_result_drawer_fit_viewport(page, width, height):
    from playwright.sync_api import expect

    page.set_viewport_size({"width": width, "height": height})
    layer(page, "Expression")
    select(page, "Visit", "Baseline")
    atlas(page).get_by_role("button", name="Play visits").click()
    expect(atlas(page).get_by_role("combobox", name="Visit", exact=True)).not_to_have_value(
        "Baseline", timeout=5000
    )
    plot = open_plot(page)
    expect(plot).to_be_visible()
    expect(atlas(page).locator(".at-scene")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    atlas(page).get_by_role("button", name="Close results").click()
    expect(plot).not_to_be_visible()


def test_linked_pathway_definition_preserves_recorded_score(page, tmp_path):
    from playwright.sync_api import expect

    csv = tmp_path / "Pathway.csv"
    csv.write_text(
        "participant,arm,visit,population,cells,pathway:Measured set,gene:MYC\n"
        "P1,A,Pre,T,200,3,0\nP2,B,Pre,T,5,7,5\n",
        encoding="utf-8",
    )
    definition = tmp_path / "study.gmt"
    definition.write_text("Measured set\tExample definition\tMYC\tBCL2\n", encoding="utf-8")
    open_csv(page, csv)
    layer(page, "Pathways")
    group = atlas(page).locator('[data-population="T"][data-value]')
    expect(group).to_have_attribute("data-value", "5")
    atlas(page).get_by_text("Gene set", exact=True).click()
    atlas(page).get_by_role("button", name="Link gene set…", exact=True).click()
    dialog = page.get_by_role("dialog", name="Link gene set")
    dialog.locator('input[type="file"]').set_input_files(definition)
    dialog.get_by_role("button", name="Link membership").click()
    atlas(page).get_by_text("Gene set", exact=True).click()
    expect(atlas(page).locator(".at-members")).to_contain_text("MYC")
    expect(atlas(page).locator(".at-members")).to_contain_text("BCL2")
    expect(group).to_have_attribute("data-value", "5")


def test_playback_closes_open_contrast_before_browsing_other_visits(page):
    from playwright.sync_api import expect

    select_population(page)
    layer(page, "Expression")
    select(page, "Visit", "Week 4")
    plot = open_plot(page)
    expect(plot).to_contain_text("Week 4")
    atlas(page).get_by_role("button", name="Play visits").click()
    expect(atlas(page).get_by_role("combobox", name="Visit", exact=True)).to_have_value("Week 12")
    expect(plot).not_to_be_visible()
    # Playback leaves the last applied contrast intact until Plot results is chosen.
    settings = comparison_settings(page)
    expect(settings.get_by_role("combobox", name="Follow-up visit", exact=True)).to_have_value(
        "Week 4"
    )
    settings.get_by_role("button", name="Cancel").click()
    open_plot(page)
    expect(plot).to_contain_text("Week 12")
