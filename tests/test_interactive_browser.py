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
        page.goto(document.as_uri())
        yield page
    finally:
        context.close()


def comparison_settings(page):
    page.get_by_role("button", name="Configure", exact=True).click()
    return page.get_by_role("dialog", name="Configure comparison")


def open_csv(page, path):
    from playwright.sync_api import expect

    page.get_by_role("button", name="Open study", exact=True).click()
    dialog = page.get_by_role("dialog", name="Open study")
    dialog.locator('input[type="file"]').set_input_files(path)
    expect(dialog).not_to_be_visible()


def test_keyboard_navigation_and_frozen_pin(page):
    from playwright.sync_api import expect

    settings = comparison_settings(page)
    settings.get_by_role("combobox", name="Follow-up visit").select_option("Week 12")
    settings.get_by_role("button", name="Apply changes").press("Enter")
    expect(page.get_by_role("button", name="Configure Visits")).to_contain_text("Week 12")
    page.get_by_role("button", name="Undo contrast change").click()
    expect(page.get_by_role("button", name="Configure Visits")).to_contain_text("Week 4")

    page.get_by_role("button", name="Inspect participant A01").press("Enter")
    record = page.get_by_role("dialog", name="Participant A01")
    expect(record).to_contain_text("Paired change")
    record.get_by_role("button", name="Close dialog").click()

    def arm_a_color():
        dots = page.get_by_label("Arm A participant observations").locator(".sce-arm-mark")
        return dots.first.evaluate("(dot) => getComputedStyle(dot).fill")

    original_color = arm_a_color()
    changes = page.get_by_role("button", name="Changes", exact=True)
    changes.click()
    page.get_by_role("button", name="Pin contrast", exact=True).click()
    settings = comparison_settings(page)
    settings.get_by_role("combobox", name="Group order").select_option("Arm B")
    settings.get_by_role("button", name="Apply changes").click()

    page.get_by_role("button", name="Comparison", exact=True).click()
    assert arm_a_color() == original_color
    page.get_by_role("button", name=re.compile(r"^Pinned contrasts")).click()
    page.get_by_role("button", name=re.compile(r"^Reopen ")).click()
    expect(page.get_by_role("button", name="Configure Group order")).to_contain_text(
        "Arm A − Arm B"
    )
    expect(page.get_by_role("button", name="Configure Visits")).to_contain_text("Week 4")
    expect(changes).to_have_attribute("aria-pressed", "true")


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

    matrix = page.get_by_role("region", name="Participant measurements")
    expect(matrix.get_by_role("button", name=re.compile(r"^Included"))).to_contain_text("3")
    paired = page.get_by_role("button", name=re.compile(r"^Open sampling record for P1\."))
    missing_post = page.get_by_role("button", name=re.compile(r"^Open sampling record for P5\."))
    expect(paired).to_have_accessible_name(re.compile("Baseline measured, Post measured"))
    expect(missing_post).to_have_accessible_name(re.compile("Baseline measured, Post missing"))
    expect(matrix.get_by_role("row", name="P1, A")).to_contain_text("Included")
    missing_post.press("Enter")
    record = page.get_by_role("dialog", name="Participant P5")
    expect(record).to_contain_text("Post")
    record.get_by_role("button", name="Close dialog").click()
    matrix.get_by_role("button", name=re.compile(r"^Excluded")).click()
    row = matrix.get_by_role("row").filter(
        has=page.get_by_role("button", name="Inspect participant P5")
    )
    expect(row.get_by_role("cell").nth(1)).to_have_attribute("aria-label", "Missing measurement")
    expect(matrix.get_by_role("button", name="Inspect participant P1")).to_have_count(0)
    page.get_by_role("button", name="Trajectories", exact=True).click()
    visit_counts = page.get_by_role("table", name="Participants measured at each visit")
    group_b = visit_counts.get_by_role("row").filter(
        has=page.get_by_role("rowheader", name="B", exact=True)
    )
    expect(group_b.get_by_role("cell")).to_have_text(["3", "1"])
    expect(page.get_by_text("Mean at each visit", exact=True)).to_be_visible()
    page.get_by_role("button", name="Comparison", exact=True).click()
    page.get_by_role("combobox", name="Outcome", exact=True).select_option(label="Marker")
    expect(matrix.get_by_role("button", name=re.compile(r"^Included"))).to_contain_text("4")

    page.get_by_role("button", name="Results table", exact=True).click()
    table = page.get_by_role("table", name="Descriptive participant-level contrasts")
    row = table.get_by_role("row").filter(has_text="Response")
    expect(row.get_by_role("cell").first).to_have_text("+2.00")
    expect(row.get_by_role("cell").last).to_have_text("3")
    marker = table.get_by_role("row").filter(has_text="Marker")
    expect(marker.get_by_role("cell").last).to_have_text("4")
    expect(table.get_by_role("columnheader", name="95% CI")).to_have_count(0)

    page.get_by_role("button", name="Pin contrast", exact=True).click()
    badge = page.locator("#pinCount")
    expect(badge).to_have_text("1")
    page.get_by_role("button", name="New analysis").click()
    assert page.evaluate("localStorage.length + sessionStorage.length") == 0
    open_csv(page, study)
    expect(badge).to_be_hidden()
    expect(page.get_by_role("button", name="Analysis 2", exact=True)).to_have_count(0)


def test_sheet_reopens_its_analysis_and_mobile_navigation(page):
    from playwright.sync_api import expect

    outcome = page.get_by_role("combobox", name="Outcome", exact=True)
    outcome.select_option(label="Cytotoxicity")
    page.get_by_role("button", name="New analysis").click()
    expect(page.get_by_role("heading", level=1)).to_have_text("Analysis 2")
    expect(outcome).to_have_value("interferon")

    page.get_by_role("button", name="Actions for Analysis 2").click()
    page.get_by_role("menuitem", name="Rename").click()
    title = page.get_by_role("textbox", name="Analysis name")
    title.fill("Sensitivity analysis")
    title.press("Enter")
    expect(page.get_by_role("heading", level=1)).to_have_text("Sensitivity analysis")
    page.get_by_role("button", name="Primary analysis", exact=True).click()
    expect(outcome).to_have_value("cytotoxicity")

    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("button", name="Open project navigator").click()
    page.get_by_role("button", name="Sensitivity analysis", exact=True).click()
    expect(outcome).to_have_value("interferon")
    expect(page.get_by_role("button", name="Open project navigator")).to_have_attribute(
        "aria-expanded", "false"
    )
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def test_workbook_download_and_reopen(page, tmp_path):
    from playwright.sync_api import expect

    page.get_by_role("button", name="New analysis").click()
    page.get_by_role("combobox", name="Outcome", exact=True).select_option(label="Proliferation")
    page.get_by_role("button", name="Export", exact=True).click()
    with page.expect_download() as downloaded:
        page.get_by_role("button", name=re.compile(r"^Save analysis workbook")).click()
    archive = tmp_path / downloaded.value.suggested_filename
    downloaded.value.save_as(archive)
    content = json.loads(archive.read_text(encoding="utf-8"))
    assert [sheet["title"] for sheet in content["sheets"]] == ["Primary analysis", "Analysis 2"]
    assert "participants" not in content["study"]
    assert "observations" not in content["study"]

    page.get_by_role("button", name="Primary analysis", exact=True).click()
    page.get_by_role("button", name="Open study", exact=True).click()
    with page.expect_file_chooser() as picker:
        page.get_by_role("button", name=re.compile(r"^Open analysis workbook")).click()
    picker.value.set_files(archive)
    expect(page.get_by_role("heading", level=1)).to_have_text("Analysis 2 (2)")
    expect(page.get_by_role("combobox", name="Outcome", exact=True)).to_have_value("proliferation")
    page.get_by_role("button", name="Primary analysis", exact=True).click()
    expect(page.get_by_role("combobox", name="Outcome", exact=True)).to_have_value("interferon")
