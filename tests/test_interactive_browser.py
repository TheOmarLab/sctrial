"""Browser regressions for the packaged offline workspace."""

from __future__ import annotations

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


def test_keyboard_navigation_and_frozen_pin(page):
    from playwright.sync_api import expect

    visit = page.get_by_role("combobox", name="Follow-up visit")
    page.get_by_role("slider", name="Selected visit").press("ArrowRight")
    expect(visit).to_have_value("Week 12")
    page.get_by_role("button", name="Undo contrast change").click()
    expect(visit).to_have_value("Week 4")

    trace = page.get_by_role("button", name=re.compile(r"^A01 · Arm A ·"))
    trace.press("Enter")
    expect(trace).to_have_attribute("aria-pressed", "true")
    expect(page.get_by_role("complementary", name="Inspector")).to_contain_text(
        "Participant inspector"
    )
    page.get_by_role("button", name="Clear participant selection").click()

    changes = page.get_by_role("button", name="Changes", exact=True)
    changes.click()
    page.get_by_role("button", name="Pin contrast", exact=True).click()
    groups = page.get_by_role("combobox", name="Group order")
    groups.select_option("Arm B")
    page.get_by_role("button", name=re.compile(r"^Pins")).click()
    page.get_by_role("button", name=re.compile(r"^Reopen ")).click()
    expect(groups).to_have_value("Arm A")
    expect(visit).to_have_value("Week 4")
    expect(changes).to_have_attribute("aria-pressed", "true")


def test_private_csv_counts_and_pin_lifetime(page, tmp_path):
    from playwright.sync_api import expect

    study = tmp_path / "Private.csv"
    study.write_text(
        "participant,arm,visit,population,cells,Response\n"
        "P1,A,Baseline,T,2,0\nP1,A,Post,T,80,2\n"
        "P2,A,Baseline,T,60,1\nP2,A,Post,T,3,5\n"
        "P3,B,Baseline,T,12,1\nP3,B,Post,T,5,2\nP4,B,Baseline,T,90,0\n",
        encoding="utf-8",
    )

    def open_study():
        page.get_by_role("button", name="Open study", exact=True).click()
        dialog = page.get_by_role("dialog", name="Open study")
        dialog.locator('input[type="file"]').set_input_files(study)
        expect(dialog).not_to_be_visible()

    open_study()
    page.get_by_role("button", name="Results", exact=True).click()
    table = page.get_by_role("table", name="Descriptive participant-level contrasts")
    row = table.get_by_role("row").filter(has_text="Response")
    expect(row.get_by_role("cell").first).to_have_text("+2.00")
    expect(row.get_by_role("cell").last).to_have_text("3")
    expect(table.get_by_role("columnheader", name="95% CI")).to_have_count(0)

    page.get_by_role("button", name="Pin contrast", exact=True).click()
    badge = page.locator("#pinCount")
    expect(badge).to_have_text("1")
    assert page.evaluate("localStorage.length + sessionStorage.length") == 0
    open_study()
    expect(badge).to_be_hidden()
