"""Integration tests for a local H5AD study and its participant-level fits."""

from __future__ import annotations

import hashlib
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import sparse

import sctrial as st
from sctrial.interactive._document import build_workspace
from sctrial.interactive._server import create_server
from sctrial.interactive._study import LocalStudy
from sctrial.stats.did import did_fit


class WorkspacePreviewTest(unittest.TestCase):
    def test_standalone_preview_matches_packaged_workspace(self):
        root = Path(__file__).resolve().parents[1]
        preview = root / "design" / "interactive-mode" / "sctrial-analysis-worksheet.html"
        self.assertTrue(
            preview.read_bytes() == build_workspace(),
            "Rebuild the standalone analysis worksheet after changing the workspace assets.",
        )


def make_study():
    rng = np.random.default_rng(10)
    rows = []
    for arm in ("A", "B"):
        for number in range(7):
            person = f"{arm}{number}"
            subject = rng.normal()
            for day, visit in enumerate(("Before", "After", "Later")):
                if (person, visit) == ("A6", "After"):
                    continue
                for cell_type in ("T cell", "B cell"):
                    if (person, visit, cell_type) == ("B5", "After", "T cell"):
                        continue
                    shift = day * (0.5 if arm == "A" else -0.2)
                    for _ in range(2 + number % 3):
                        score = rng.normal(subject + shift)
                        marker = rng.normal(subject - shift)
                        if (person, visit, cell_type) == ("B2", "After", "T cell"):
                            marker = np.nan
                        rows.append((person, arm, visit, cell_type, score, marker))
    frame = pd.DataFrame(rows, columns=["donor", "group", "visit", "cell_type", "score", "marker"])
    frame.index = pd.Index([f"raw-cell-{i}" for i in range(len(frame))])
    return ad.AnnData(obs=frame, X=np.zeros((len(frame), 3), dtype=np.float32))


class InteractiveWorkspaceTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "study.h5ad"
        self.data = make_study()
        self.data.write_h5ad(self.path)
        self.fingerprint = hashlib.sha256(self.path.read_bytes()).digest()
        self.workspace = LocalStudy(self.path)
        self.mapping = {
            "participant": "donor",
            "arm": "group",
            "visit": "visit",
            "population": "cell_type",
            "scores": ["score", "marker"],
        }

    def tearDown(self):
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).digest(), self.fingerprint)

    def test_read_only_summary_excludes_cells_and_expression_matrix(self):
        schema = self.workspace.schema()
        self.assertEqual(schema["filename"], "study.h5ad")
        self.assertEqual(schema["nCells"], self.data.n_obs)
        self.assertTrue(next(col["numeric"] for col in schema["columns"] if col["name"] == "score"))
        # Include opening the file as well as mapping its observation table.
        original = h5py.Group.__getitem__

        def read_metadata(group, key):
            if group.name == "/" and isinstance(key, str):
                self.assertNotIn(key.strip("/").split("/")[0], {"X", "layers", "raw"})
            return original(group, key)

        with patch.object(h5py.Group, "__getitem__", new=read_metadata):
            study = LocalStudy(self.path).prepare(self.mapping)
        self.assertEqual(len(study["participants"]), 14)
        self.assertNotIn("raw-cell-", json.dumps(study))
        self.assertNotIn("raw-cell-", json.dumps(schema))
        matches = [
            row
            for row in study["observations"]
            if (row["id"], row["visit"], row["population"]) == ("A1", "Before", "T cell")
        ]
        self.assertEqual(len(matches), 1)
        row = matches[0]
        self.assertEqual(row["cells"], 3)
        key = next(f["id"] for f in study["features"] if f["label"] == "score")
        expected = self.data.obs.query(
            "donor == 'A1' and visit == 'Before' and cell_type == 'T cell'"
        )
        self.assertAlmostEqual(row["scores"][key], expected.score.mean())
        json.dumps(study, allow_nan=False)

    def test_visit_axis_honors_ordered_categories_with_shuffled_cells(self):
        other_path = self.path.with_name("ordered.h5ad")
        study = self.data[::-1].copy()
        study.obs["visit"] = pd.Categorical(
            study.obs["visit"], categories=["Before", "Unused", "After", "Later"], ordered=True
        )
        study.write_h5ad(other_path)
        result = LocalStudy(other_path).prepare(self.mapping)
        self.assertTrue(result["orderedVisits"])
        self.assertEqual(result["visits"], ["Before", "After", "Later"])

    def test_selected_dense_and_sparse_expression_preserves_x_scale(self):
        n = self.data.n_obs
        index = np.arange(n)
        matrix = np.column_stack([np.full(n, np.inf), index % 5, (index % 11) * -0.125])
        matrix[index % 19 == 0, 2] = np.nan
        reference = self.data.obs.copy()
        reference["chosen"] = matrix[:, 2]
        reference["second"] = matrix[:, 1]
        reference = reference.groupby(["donor", "visit", "cell_type"], observed=True)[
            ["chosen", "second"]
        ].mean()
        original_group_get = h5py.Group.__getitem__
        for storage in (np.asarray, sparse.csr_matrix, sparse.csc_matrix):
            with self.subTest(storage=storage.__name__):
                path = self.path.with_name(storage.__name__ + ".h5ad")
                data = self.data.copy()
                data.var_names = ["UNSELECTED", "G1", "G2"]
                data.X = storage(matrix)
                data.obs["module:Proliferation"] = data.obs["score"]
                data.write_h5ad(path)
                local = LocalStudy(path)
                self.assertTrue(local.schema()["geneInput"])
                with patch.object(
                    h5py.Group,
                    "__getitem__",
                    new=lambda group, key: (
                        self.fail("Gene search read expression data")
                        if group.name == "/" and key == "X"
                        else original_group_get(group, key)
                    ),
                ):
                    self.assertEqual(local.find_genes({"query": "g1"}), {"genes": ["G1"]})
                    self.assertEqual(local.find_genes({"query": "not present"}), {"genes": []})
                mapping = {
                    **self.mapping,
                    "scores": ["module:Proliferation", "marker"],
                    "scoreKinds": {"marker": "pathway"},
                    "genes": ["G2", "G1"],
                }
                original = h5py.Dataset.__getitem__

                def guard(dataset, selected):
                    if dataset.name == "/X":
                        self.assertIsInstance(selected, tuple)
                        self.assertIn(selected[1], [1, 2])
                    return original(dataset, selected)

                with patch.object(h5py.Dataset, "__getitem__", new=guard):
                    study = local.prepare(mapping)
                features = study["features"]
                self.assertEqual(
                    [f["kind"] for f in features], ["module", "pathway", "gene", "gene"]
                )
                self.assertEqual(
                    [f["label"] for f in features], ["Proliferation", "marker", "G2", "G1"]
                )
                self.assertEqual(features[2]["unit"], "Expression (X scale)")
                self.assertEqual(study["mapping"], mapping)
                distributions = study["cellDistributions"]
                g2 = features[2]["id"]
                low, high = float(np.nanmin(matrix[:, 2])), float(np.nanmax(matrix[:, 2]))
                self.assertEqual(distributions["features"][g2], [low, high])
                seen = set()
                for group in distributions["groups"]:
                    key = group["arm"], group["visit"], group["population"]
                    self.assertNotIn(key, seen)
                    seen.add(key)
                    mask = (
                        (data.obs.group == key[0])
                        & (data.obs.visit == key[1])
                        & (data.obs.cell_type == key[2])
                    )
                    raw = matrix[mask, 2]
                    finite = raw[np.isfinite(raw)]
                    counts = group["bins"][g2]
                    expected_counts, _ = np.histogram(finite, bins=len(counts), range=(low, high))
                    self.assertEqual(counts, expected_counts.tolist())
                    self.assertEqual(group["zeros"][g2], np.count_nonzero(finite == 0))
                self.assertEqual(
                    seen, set(data.obs[["group", "visit", "cell_type"]].itertuples(index=False, name=None))
                )
                for r in study["observations"]:
                    if r["population"] == study["populations"][0]:
                        continue
                    expected = reference.loc[(r["id"], r["visit"], r["population"])]
                    self.assertAlmostEqual(r["scores"]["gene_1"], expected.chosen)
                    self.assertAlmostEqual(r["scores"]["gene_2"], expected.second)
                self.assertEqual(local.prepare(mapping)["studyId"], study["studyId"])
                genes_only = local.prepare({**self.mapping, "scores": [], "genes": ["G1"]})
                self.assertEqual([f["label"] for f in genes_only["features"]], ["G1"])
                fit = local.fit(
                    {
                        "studyId": genes_only["studyId"],
                        "design": "cross",
                        "arm": "A",
                        "reference": "B",
                        "visit": "Before",
                        "population": "T cell",
                    }
                )["rows"][0]
                values = reference.xs(("Before", "T cell"), level=("visit", "cell_type")).second
                in_a = values.index.astype(str).str.startswith("A")
                self.assertAlmostEqual(fit["estimate"], values[in_a].mean() - values[~in_a].mean())
                self.assertEqual(fit["n"], len(values))
                with self.assertRaisesRegex(ValueError, "Unknown X variable"):
                    local.prepare({**self.mapping, "genes": ["MISSING"]})

    def test_fits_match_package_with_missing_pairs_and_feature_values(self):
        study = self.workspace.prepare(self.mapping)
        selected = self.data[self.data.obs.cell_type == "T cell"].copy()
        for arm, reference in (("A", "B"), ("B", "A")):
            design = st.TrialDesign(
                participant_col="donor",
                visit_col="visit",
                arm_col="group",
                arm_treated=arm,
                arm_control=reference,
                celltype_col="cell_type",
            )
            common = {"features": ["score", "marker"], "design": design, "standardize": False}
            for mode, suffix in (("did", "DiD"), ("paired", "time"), ("cross", "arm")):
                with self.subTest(mode=mode, arm=arm):
                    spec = {
                        "studyId": study["studyId"],
                        "design": mode,
                        "arm": arm,
                        "reference": reference,
                        "baseline": "Before",
                        "visit": "After",
                        "population": "T cell",
                    }
                    actual = self.workspace.fit(spec)
                    if mode == "did":
                        table = st.did_table(selected, visits=("Before", "After"), **common)
                    elif mode == "paired":
                        table = st.within_arm_comparison(
                            selected, arm=arm, visits=("Before", "After"), **common
                        )
                    else:
                        table = st.between_arm_comparison(selected, visit="After", **common)
                    expected = table.set_index("feature")
                    self.assertEqual(len(actual["rows"]), 2)
                    for row in actual["rows"]:
                        direct = expected.loc[row["label"]]
                        self.assertEqual(row["n"], direct["n_units"])
                        self.assertAlmostEqual(row["estimate"], direct["beta_" + suffix], places=8)
                        self.assertAlmostEqual(row["p"], direct["p_" + suffix], places=8)
                        self.assertAlmostEqual(row["q"], direct["FDR_" + suffix], places=8)
                        self.assertAlmostEqual(row["lo"], direct["ci_lo_" + suffix], places=8)
                        self.assertAlmostEqual(row["hi"], direct["ci_hi_" + suffix], places=8)
                    json.dumps(actual, allow_nan=False)

    def test_did_intervals_follow_actual_covariance_and_identifiability(self):
        rng = np.random.default_rng(48)
        frame = pd.DataFrame(
            {
                "participant": np.repeat(np.arange(12), 2),
                "post": np.tile([0, 1], 12),
                "treated": np.repeat([0, 1], 12),
            }
        )
        frame["score"] = frame.participant * 0.2 + frame.post * (
            1.4 * frame.treated + np.repeat(rng.normal(size=12), 2)
        )
        args = dict(
            y="score", unit="participant", time="post", arm_bin="treated", standardize=False
        )
        for covariance in ("cluster", "nonrobust"):
            with self.subTest(covariance=covariance):
                model = smf.ols("score ~ post + post:treated + C(participant)", data=frame)
                fit_kwargs = (
                    {"cov_kwds": {"groups": frame.participant}} if covariance == "cluster" else {}
                )
                direct = model.fit(cov_type=covariance, **fit_kwargs)
                result = did_fit(frame, cov_type=covariance, **args)
                bounds = direct.conf_int().loc["post:treated"]
                np.testing.assert_allclose(
                    [result["ci_lo_DiD"], result["ci_hi_DiD"]], bounds, rtol=1e-9
                )
        exact = frame.assign(score=2.0 * frame.participant + frame.post * frame.treated)
        result = did_fit(exact, **args)
        self.assertAlmostEqual(result["beta_DiD"], 1.0)
        self.assertTrue(np.isnan(result["ci_lo_DiD"]) and np.isnan(result["ci_hi_DiD"]))
        one_group = did_fit(frame[frame.treated == 1], **args)
        for key in ("beta_DiD", "ci_lo_DiD", "ci_hi_DiD"):
            self.assertTrue(np.isnan(one_group[key]))

    def test_mappings_are_reused_and_stale_fits_rejected(self):
        first = self.workspace.prepare(self.mapping)
        self.assertEqual(self.workspace.prepare(self.mapping)["studyId"], first["studyId"])
        second = self.workspace.prepare({**self.mapping, "scores": ["marker"]})
        self.assertNotEqual(first["studyId"], second["studyId"])
        with self.assertRaises(ValueError):
            self.workspace.fit({"studyId": first["studyId"]})

    def test_selected_participants_determine_the_fitted_contrast(self):
        study = self.workspace.prepare(self.mapping)
        ids = ["A1", "A2", "A3", "B0", "B2", "B3"]
        opts = {
            "studyId": study["studyId"],
            "design": "did",
            "arm": "A",
            "reference": "B",
            "baseline": "Before",
            "visit": "After",
            "population": "T cell",
            "participants": ids,
        }
        fitted = {r["label"]: r for r in self.workspace.fit(opts)["rows"]}
        self.assertEqual(fitted["score"]["n"], 6)
        self.assertEqual(fitted["marker"]["n"], 5)
        obs = self.data.obs
        selected = obs[obs.donor.isin(ids) & obs.cell_type.eq("T cell")]
        means = selected.groupby(["donor", "group", "visit"], observed=True).score.mean().unstack()
        change = (means.After - means.Before).groupby(level="group", observed=True).mean()
        self.assertAlmostEqual(fitted["score"]["estimate"], change["A"] - change["B"])
        missing = self.workspace.fit({**opts, "participants": []})
        for row in missing["rows"]:
            self.assertEqual(row["n"], 0)
            self.assertIsNone(row["estimate"])

    def test_single_group_and_single_occasion(self):
        single = self.data[self.data.obs["group"] == "A"].copy()
        single.obs.drop(columns=["group", "cell_type"], inplace=True)
        path = self.path.with_name("single-group.h5ad")
        single.write_h5ad(path)
        local = LocalStudy(path)
        mapping = {**self.mapping, "arm": "", "population": ""}
        study = local.prepare(mapping)
        self.assertEqual(len(study["participants"]), 7)
        self.assertEqual(len(study["populations"]), 1)
        group = study["participants"][0]["arm"]
        out = local.fit(
            {
                "studyId": study["studyId"],
                "design": "paired",
                "arm": group,
                "baseline": "Before",
                "visit": "After",
                "population": study["populations"][0],
            }
        )
        self.assertTrue(all(row["n"] == 6 for row in out["rows"]))

        cross = self.data[self.data.obs.visit == "After"].copy()
        cross.obs.drop(columns=["visit"], inplace=True)
        path = self.path.with_name("single-occasion.h5ad")
        cross.write_h5ad(path)
        local = LocalStudy(path)
        study = local.prepare({**self.mapping, "visit": ""})
        self.assertEqual(len(study["visits"]), 1)
        result = local.fit(
            {
                "studyId": study["studyId"],
                "design": "cross",
                "arm": "A",
                "reference": "B",
                "visit": study["visits"][0],
                "population": "T cell",
            }
        )
        counts = {r["label"]: r["n"] for r in result["rows"]}
        self.assertEqual(counts, {"score": 12, "marker": 11})

    def test_local_session(self):
        server = create_server(self.workspace, b"<html><head></head><body>Study</body></html>")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(worker.join, 2)
        self.addCleanup(server.shutdown)
        opener = build_opener(ProxyHandler({}))

        def get(path, headers=None, payload=None):
            body = None if payload is None else json.dumps(payload).encode()
            req = Request(server.origin + path, data=body, headers=headers or {})
            with opener.open(req, timeout=3) as response:
                return response.read()

        document = get("/")
        self.assertIn(server.session.encode(), document)
        self.assertEqual(
            get("/sctrial-workbook.html"), b"<html><head></head><body>Study</body></html>"
        )
        self.assertNotIn(b"raw-cell-", document)
        with self.assertRaises(HTTPError) as denied:
            get("/local/schema")
        self.assertIn(denied.exception.code, (401, 403))
        with self.assertRaises(HTTPError) as denied:
            get("/", {"Host": "example.org"})
        self.assertIn(denied.exception.code, (400, 401, 403, 421))
        headers = {
            "X-sctrial-session": server.session,
            "Content-Type": "application/json",
            "Origin": server.origin,
        }
        self.assertEqual(json.loads(get("/local/schema", headers))["filename"], "study.h5ad")
        with self.assertRaises(HTTPError) as denied:
            get("/local/study", dict(headers, Origin="https://example.org"), self.mapping)
        self.assertIn(denied.exception.code, (401, 403))
        with self.assertRaises(HTTPError) as denied:
            get("/local/fit", {**headers, "X-sctrial-session": "unknown"}, {})
        self.assertIn(denied.exception.code, (401, 403))
        prepared = json.loads(get("/local/study", headers, self.mapping))
        self.assertEqual(len(prepared["participants"]), 14)
        self.assertEqual(prepared["version"], st.__version__)
        self.assertEqual(prepared["mapping"], self.mapping)
        fit = {
            "studyId": prepared["studyId"],
            "design": "cross",
            "arm": "A",
            "reference": "B",
            "baseline": "Before",
            "visit": "After",
            "population": "T cell",
        }
        self.assertEqual(len(json.loads(get("/local/fit", headers, fit))["rows"]), 2)
