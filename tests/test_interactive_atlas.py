"""Browser regressions using every observation in the packaged public GSE169246 atlas."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager
from importlib.resources import files
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sctrial.interactive._document import build_workspace
from sctrial.interactive._server import WorkspaceHandler, create_server
from sctrial.interactive._study import LocalStudy


@pytest.fixture(scope="module")
def public_h5ad(tmp_path_factory):
    """Re-encode released per-cell values at their display precision; no participant IDs were released."""
    assets = files("sctrial.interactive").joinpath("atlas").joinpath("data")
    study = json.loads(assets.joinpath("study.json").read_text("utf-8"))
    layout = study["map"]
    payload = assets.joinpath("map.bin").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == layout["sha256"]
    rows = np.frombuffer(payload, dtype=np.uint8).reshape(layout["rows"], layout["stride"])
    assert len(rows) == 141_553
    genes = [f for f in study["features"] if f["kind"] == "gene"]
    assert len(genes) == 6
    values = np.column_stack(
        [rows[:, 7 + layout["featureIds"].index(g["id"])] * (g["domain"][1] / 255.0) for g in genes]
    )
    xy = np.empty((len(rows), 2))
    for axis, domain in enumerate((layout["xDomain"], layout["yDomain"])):
        codes = rows[:, 2 * axis].astype(float) + 256.0 * rows[:, 2 * axis + 1]
        xy[:, axis] = domain[0] + (domain[1] - domain[0]) * codes / 65_535
    obs = pd.DataFrame(index=pd.Index(np.arange(len(rows)).astype(str), name="public_map_row"))
    for column, key, offset in (
        ("cell_type", "populations", 4),
        ("arm", "arms", 5),
        ("visit", "visits", 6),
    ):
        obs[column] = pd.Categorical.from_codes(
            rows[:, offset], [item["label"] for item in study[key]]
        )
    names = [g["id"] for g in genes]
    data = ad.AnnData(X=values, obs=obs, var=pd.DataFrame(index=names), obsm={"X_umap": xy})
    data.uns["source"] = "GSE169246 public sctrial atlas, 8-bit display precision, log1p(CP10K)"
    path = tmp_path_factory.mktemp("public-atlas") / "GSE169246_display.h5ad"
    data.write_h5ad(path)

    expected = {}
    for visit in study["visits"]:
        matched = (obs["visit"] == visit["label"]) & (obs["arm"] == "Atezolizumab + paclitaxel")
        pop = matched & (obs["cell_type"] == "CD8 T cell")
        marker = values[pop, names.index("GZMB")]
        interferon = values[pop][:, [names.index("IFIT1"), names.index("ISG15")]]
        expected[visit["id"]] = {
            "cells": int(pop.sum()),
            "total": int(matched.sum()),
            "mean": marker.mean(),
            "nonzero": int(np.count_nonzero(marker)),
            "bin2": int(
                np.histogram(marker, bins=32, range=(0, values[:, names.index("GZMB")].max()))[0][1]
            ),
            "pathway": interferon.mean(axis=1).mean(),
        }
    yield path, expected


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


@contextmanager
def serve(path=None):
    requests = []

    class RecordingHandler(WorkspaceHandler):
        def log_request(self, code="-", size="-"):
            requests.append((self.command, self.path, int(code)))

    with create_server(LocalStudy(path) if path else None, build_workspace()) as server:
        server.RequestHandlerClass = RecordingHandler
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server, requests
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("launcher", [False, True], ids=["browser-file", "launcher-file"])
def test_open_real_study(browser, public_h5ad, launcher):
    from playwright.sync_api import expect

    path, expected = public_h5ad
    with serve(path if launcher else None) as (server, requests):
        context = browser.new_context(reduced_motion="reduce")
        origins = []
        context.on("request", lambda req: origins.append(urlsplit(req.url).netloc))
        try:
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(server.origin)
            if not launcher:
                page.get_by_role("button", name="Open H5AD").click()
                page.get_by_label("Choose an H5AD file").set_input_files(str(path))
            dlg = page.get_by_role("dialog", name="Map your study")
            expect(dlg).to_be_visible(timeout=30_000)
            expect(dlg.locator('[data-im="participant"]')).to_have_value("")
            dlg.locator('[data-im="normalization"]').select_option("log1p")
            dlg.locator('input[value="ifn_ab"]').check()
            dlg.get_by_role("button", name="Explore study").click()
            expect(page.locator("#atlasTitle")).to_have_text(path.stem, timeout=45_000)
            expect(page.locator("#cohortCells")).to_have_text("141,553")
            expect(page.locator("#participantStat")).to_be_hidden()
            page.locator("#arm").select_option(label="Atezolizumab + paclitaxel")
            page.locator("#choosePop").click()
            page.locator("#popList button").filter(
                has=page.get_by_text("CD8 T cell", exact=True)
            ).click()
            for visit in ("pre", "post"):
                page.locator(f'button[data-visit="{visit}"]').click()
                g = expected[visit]
                expect(page.locator("#popMeta")).to_contain_text(f"{g['cells']:,} cells")
                expect(page.locator("#captureCount")).to_contain_text(f"{g['total']:,} cells")
                expect(page.locator(f".ev-end.ev-{visit}")).to_contain_text(
                    f"{g['cells']:,} / {g['total']:,}"
                )
            expect(page.locator(".ev-pair")).to_have_count(0)
            page.get_by_role("button", name="Visit lens", exact=True).click()
            expect(page.locator("#map")).to_have_attribute(
                "aria-label", re.compile("Follow-up.*Visit lens active")
            )
            page.locator("#map").press("ArrowRight")
            page.get_by_role("button", name="Gene expression", exact=True).click()
            page.get_by_label("Gene", exact=True).select_option(label="GZMB")
            expect(page.locator(".ev-metric strong").first).to_contain_text(
                f"{expected['post']['mean']:.2f}"
            )
            for visit, g in expected.items():
                expect(page.locator(f".ev-end.ev-{visit}")).to_contain_text(
                    f"{g['nonzero']:,} / {g['cells']:,}"
                )
            hist = page.locator("[data-ev-hist]")
            hist.press("Home")
            hist.press("ArrowRight")
            for key, visit in (("pre", "Pre-treatment"), ("post", "Follow-up")):
                expect(page.locator("[data-ev-bin]")).to_contain_text(
                    f"{visit} {expected[key]['bin2']:,}"
                )
            page.get_by_role("button", name="Pathways", exact=True).click()
            assert sorted(page.locator(".pw-menu option").all_text_contents()) == ["IFIT1", "ISG15"]
            for visit, g in expected.items():
                expect(page.locator(f".ev-end.ev-{visit} strong")).to_have_text(
                    f"{g['pathway']:.3f}"
                )
            assert not errors
            assert set(origins) <= {server.host_header, ""}
            assert {method for method, _, _ in requests} <= {"GET", "HEAD"}
            assert (
                any(route == "/local/h5ad" and code == 206 for _, route, code in requests)
                == launcher
            )
        finally:
            context.close()


def test_launcher_byte_ranges_read_the_original_file(public_h5ad):
    path, _ = public_h5ad
    digest = hashlib.sha256(path.read_bytes()).digest()
    opener = build_opener(ProxyHandler({}))
    with serve(path) as (server, _):
        url = server.origin + "/local/h5ad"
        token = {"X-sctrial-session": server.session}
        with opener.open(Request(url, headers=token, method="HEAD")) as response:
            assert response.status == 200
            assert int(response.headers["Content-Length"]) == path.stat().st_size
            assert response.headers["Accept-Ranges"] == "bytes"
        first, last = 123, 8193
        with opener.open(
            Request(url, headers={**token, "Range": f"bytes={first}-{last}"})
        ) as response:
            assert response.status == 206
            assert (
                response.headers["Content-Range"] == f"bytes {first}-{last}/{path.stat().st_size}"
            )
            with path.open("rb") as source:
                source.seek(first)
                assert response.read() == source.read(last - first + 1)
        with pytest.raises(HTTPError) as invalid_range:
            opener.open(
                Request(
                    url,
                    headers={
                        **token,
                        "Range": f"bytes={path.stat().st_size}-{path.stat().st_size}",
                    },
                )
            )
        assert invalid_range.value.code == 416
        with pytest.raises(HTTPError) as missing_token:
            opener.open(Request(url, headers={"Range": "bytes=0-7"}))
        assert missing_token.value.code == 403
    assert hashlib.sha256(path.read_bytes()).digest() == digest
