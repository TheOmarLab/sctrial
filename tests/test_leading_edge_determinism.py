"""Leading-edge heatmaps (Figure 4D, Supp Fig 7D) must not depend on the string hash seed."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = textwrap.dedent("""
    import importlib.util, json, os, sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    root, rel, func = sys.argv[1], sys.argv[2], sys.argv[3]
    os.chdir(root)
    def load(name, path, search=None):
        kw = {"submodule_search_locations": [search]} if search else {}
        spec = importlib.util.spec_from_file_location(name, path, **kw)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    load("manuscript_figures", "manuscript_figures/__init__.py", "manuscript_figures")
    pkg = rel.split("/")[0]
    load(f"manuscript_figures.{pkg}", f"manuscript_figures/{pkg}/__init__.py", f"manuscript_figures/{pkg}")
    mod = load("manuscript_figures." + rel.replace("/", "."), f"manuscript_figures/{rel}.py")
    mod.is_display_gene = lambda gene: True  # synthetic gene names; this test is about ordering
    # Many genes shared by exactly the same number of pathways: the top-10 cut falls in a tie.
    genes_up = [f"UP{i:02d}" for i in range(30)]
    genes_dn = [f"DN{i:02d}" for i in range(30)]
    rows = []
    for k in range(6):
        rows.append({"Term": f"Up pathway {k}", "NES": 2.0 - 0.1 * k, "FDR q-val": 0.01,
                     "Lead_genes": ";".join(genes_up[k:k + 20])})
        rows.append({"Term": f"Down pathway {k}", "NES": -2.0 + 0.1 * k, "FDR q-val": 0.01,
                     "Lead_genes": ";".join(genes_dn[k:k + 20])})
    fig, ax = plt.subplots(figsize=(10, 7))
    getattr(mod, func)(ax, {"gsea_results": pd.DataFrame(rows)}, composite=True)
    labels = [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
    print(json.dumps(labels))
""")


@pytest.mark.parametrize("rel,func", [
    ("main/figure4_biological_discovery_multi_dataset", "_tnbc_leading_edge"),
    ("supp/supp_fig7_cross_dataset_biology", "panel_D"),
])
def test_leading_edge_genes_do_not_depend_on_hash_seed(rel, func, tmp_path):
    outs = []
    for seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed, SCTRIAL_MANUSCRIPT_DIR=str(tmp_path))
        res = subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT), rel, func],
                             capture_output=True, text=True, env=env, timeout=300)
        assert res.returncode == 0, res.stderr[-2000:]
        outs.append(json.loads(res.stdout.strip().splitlines()[-1]))
    assert any(lab.startswith(("UP", "DN")) for lab in outs[0]), "no gene labels were drawn"
    assert outs[0] == outs[1] == outs[2]
