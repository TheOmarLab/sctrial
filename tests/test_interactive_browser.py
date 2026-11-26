"""Exercise the packaged workspace with a local Chromium installation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from sctrial.interactive._document import build_workspace


class WorkspaceBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested = os.environ.get("SCTRIAL_TEST_CHROMIUM")
        names = (
            (requested,)
            if requested
            else ("chromium", "chromium-browser", "google-chrome", "chrome", "msedge")
        )
        for name in names:
            executable = shutil.which(name)
            if (
                executable
                and subprocess.run(
                    [executable, "--version"], capture_output=True, timeout=5
                ).returncode
                == 0
            ):
                cls.browser = executable
                return
        if requested:
            raise ValueError("SCTRIAL_TEST_CHROMIUM must name a working Chromium executable")
        raise unittest.SkipTest("No Chromium on PATH. Set SCTRIAL_TEST_CHROMIUM to its executable.")

    def run_workspace(self, steps):
        # A separate page drives the actual offline document through its rendered controls.
        harness = """
<title>Testing</title><iframe src="workspace.html"></iframe>
<script>
document.querySelector('iframe').onload = async function () {
  const app = this.contentWindow, doc = app.document;
  const find = selector => {
    const element = doc.querySelector(selector);
    if (!element) throw new Error('Missing control: ' + selector);
    return element;
  };
  const check = (value, message) => { if (!value) throw new Error(message); };
  const click = selector => find(selector).click();
  const change = (selector, value) => {
    const element = find(selector);
    element.value = value;
    element.dispatchEvent(new app.Event('change', {bubbles: true}));
  };
  const key = (selector, value) => {
    const element = find(selector);
    element.focus();
    element.dispatchEvent(new app.KeyboardEvent('keydown', {key: value, bubbles: true}));
  };

  try {
    STEPS
    document.title = 'Result: Passed';
  } catch (error) { document.title = 'Result: ' + (error.stack || String(error)); }
};
</script>"""
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "workspace.html").write_bytes(build_workspace())
            driver = folder / "test.html"
            driver.write_text(harness.replace("STEPS", steps), encoding="utf-8")
            command = [
                self.browser,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
                "--allow-file-access-from-files",
                f"--user-data-dir={folder / 'profile'}",
                "--remote-debugging-address=127.0.0.1",
                "--remote-debugging-port=0",
                driver.as_uri(),
            ]
            with subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ) as proc:
                try:
                    result = self.browser_result(proc, folder / "profile" / "DevToolsActivePort")
                finally:
                    proc.terminate()
            self.assertEqual(result, "Result: Passed")

    def browser_result(self, proc, port_file):
        # Chromium's local target list exposes the test page title without a browser driver.
        direct = build_opener(ProxyHandler({}))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self.fail(f"Chromium exited with status {proc.returncode} before running the test")
            if port_file.exists():
                lines = port_file.read_text().splitlines()
                if lines:
                    try:
                        with direct.open(f"http://127.0.0.1:{lines[0]}/json/list", timeout=2) as r:
                            pages = json.load(r)
                        for page in pages:
                            if page["title"].startswith("Result: "):
                                return page["title"]
                    except URLError:
                        pass  # The port can be published before it accepts connections.
            time.sleep(0.05)
        self.fail("The browser did not finish the workspace interaction")

    def test_keyboard_navigation_and_frozen_pin(self):
        self.run_workspace("""
const visit = '[data-param="visit"]';
const initial = find(visit).value;
key('#plot [role="slider"]', 'ArrowRight');
check(find(visit).value !== initial, 'Right arrow must change the follow-up visit');
click('[data-action="undo"]');
check(find(visit).value === initial, 'Undo must restore the previous visit');
key('#plot .cc-person', 'Enter');
check(doc.querySelector('#plot .cc-person[aria-pressed="true"]'), 'Trace must be selected');
check(find('#inspector').textContent.includes('Participant inspector'), 'Inspector must follow trace');
click('#clearSelection');
click('[data-mode="changes"]');
const arm = find('[data-param="arm"]');
const original = arm.value, reversed = [...arm.options].find(o => o.value !== original).value;
click('#pinButton');
change('[data-param="arm"]', reversed);
click('[data-view="bench"]');
click('button[aria-label^="Reopen "]');
check(find('[data-param="arm"]').value === original, 'Reopen must restore pinned group order');
check(find(visit).value === initial, 'Reopen must restore the pinned visit');
check(find('[data-mode="changes"]').getAttribute('aria-pressed') === 'true', 'Reopen plot view');
""")

    def test_private_csv_counts_and_pin_lifetime(self):
        self.run_workspace(r"""
const input = 'participant,arm,visit,population,cells,Response\n' +
  ['P1,A,Baseline,T,2,0','P1,A,Post,T,80,2','P2,A,Baseline,T,60,1',
   'P2,A,Post,T,3,5','P3,B,Baseline,T,12,1','P3,B,Post,T,5,2',
   'P4,B,Baseline,T,90,0'].join('\n');
async function openFile() {
  click('.study-button');
  const field = find('input[type=file]');
  const files = new app.DataTransfer();
  files.items.add(new app.File([input], 'Private.csv', {type: 'text/csv'}));
  field.files = files.files;
  field.dispatchEvent(new app.Event('change', {bubbles: true}));
  while (doc.querySelector('#dialog[open]')) await new Promise(resolve => setTimeout(resolve, 5));
}
await openFile();
click('[data-view="results"]');
const row = find('table.results-table tbody tr');
check(row.querySelector('td').textContent === '+2.00', 'Participants must contribute equally');
check(row.lastElementChild.textContent === '3', 'Only complete participant pairs contribute');
check(!find('table.results-table').textContent.includes('95% CI'), 'Summaries are descriptive');
click('#pinButton');
check(find('#pinCount').textContent === '1', 'Private pin must be available in the current study');
check(app.localStorage.length === 0 && app.sessionStorage.length === 0, 'Private pins stay in memory');
await openFile();
check(find('#pinCount').hidden, 'Reopening a private file must start a fresh pin set');
""")
