# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
# Dev install (editable)
pip install -e ".[dev,all]"

# Python environment with sctrial installed
/opt/anaconda3/bin/python
```

## Running Manuscript Figure Scripts

```bash
# MUST run from sctrial/sc_trial_inference/ as module (relative imports)
cd /Users/omarm/Documents/Research/projects/sc-trialdiff/sctrial/sc_trial_inference
python -m manuscript_figures.main.figure3_robustness_benchmarking
```

**Do NOT** run as `python manuscript_figures/main/figure3_robustness_benchmarking.py` — relative imports will fail.

## Running Main Figure Scripts (importlib bootstrap)

Main figure modules also use relative imports. To run a single main figure standalone:
```bash
cd /Users/omarm/Documents/Research/projects/sc-trialdiff/sctrial/sc_trial_inference
/opt/anaconda3/bin/python -c "
import importlib, sys
for name, path, subs in [
    ('manuscript_figures', 'manuscript_figures/__init__.py', ['manuscript_figures']),
    ('manuscript_figures._shared', 'manuscript_figures/_shared.py', None),
    ('manuscript_figures.main', 'manuscript_figures/main/__init__.py', ['manuscript_figures/main']),
]:
    kw = {'submodule_search_locations': subs} if subs else {}
    s = importlib.util.spec_from_file_location(name, path, **kw)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
sys.modules['manuscript_figures._shared'].apply_style()
s = importlib.util.spec_from_file_location('manuscript_figures.main.figure3_robustness_benchmarking', 'manuscript_figures/main/figure3_robustness_benchmarking.py')
m = importlib.util.module_from_spec(s); sys.modules[s.name] = m; s.loader.exec_module(m)
m.generate()
"
```

## Running Supplementary Figure Scripts

Supp modules use `from .._shared import ...` (relative imports). Run individual modules with importlib bootstrap.
**Note:** Some supp modules import from others (e.g. SF3→Figure3). When bootstrapping, pre-load dependency modules before the target.
```bash
cd /Users/omarm/Documents/Research/projects/sc-trialdiff/sctrial/sc_trial_inference
/opt/anaconda3/bin/python -c "
import importlib, sys
for name, path, subs in [
    ('manuscript_figures', 'manuscript_figures/__init__.py', ['manuscript_figures']),
    ('manuscript_figures._shared', 'manuscript_figures/_shared.py', None),
    ('manuscript_figures.supp', 'manuscript_figures/supp/__init__.py', ['manuscript_figures/supp']),
]:
    kw = {'submodule_search_locations': subs} if subs else {}
    s = importlib.util.spec_from_file_location(name, path, **kw)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
sys.modules['manuscript_figures._shared'].apply_style()
# Now import and run the target module
s = importlib.util.spec_from_file_location('manuscript_figures.supp.supp_fig1_data_quality_cohort', 'manuscript_figures/supp/supp_fig1_data_quality_cohort.py')
m = importlib.util.module_from_spec(s); sys.modules[s.name] = m; s.loader.exec_module(m)
m.generate()
"
```

## Running All Figures (run_all.py)

```bash
cd /Users/omarm/Documents/Research/projects/sc-trialdiff/sctrial/sc_trial_inference
python -m manuscript_figures.run_all              # everything
python -m manuscript_figures.run_all --main       # main figures only
python -m manuscript_figures.run_all --supp       # supplementary only
python -m manuscript_figures.run_all --figure 2   # single main figure
python -m manuscript_figures.run_all --supp-fig 3 # single supp figure
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_did.py

# Run a specific test
python -m pytest tests/test_did.py::test_did_table_basic -v

# Annotation tests (require scanpy, ~60s)
python -m pytest tests/test_annotation.py -v

# With coverage
python -m pytest tests/ --cov=src/sctrial --cov-report=term-missing
```

## Linting

```bash
# Check
python -m ruff check src/ tests/ manuscript_figures/

# Auto-fix
python -m ruff check --fix src/ tests/ manuscript_figures/
```

Ruff config: line-length 100, target Python 3.9, rules E/F/W/I/UP.

## Documentation

```bash
# Build Sphinx docs
cd docs && make html

# Tutorials live in two synced locations:
#   tutorials/           (repo root, runnable)
#   docs/source/tutorials/  (Sphinx copies, must stay in sync)
```

## Architecture

### Package layout (`src/sctrial/`)

- **`design.py`** — `TrialDesign` dataclass: defines arm/visit/participant columns and labels. Nearly every function takes a `TrialDesign` as its first argument.
- **`stats/did.py`** — Core DiD engine: `did_fit()` (single OLS + wild cluster bootstrap), `did_table()` (loop over features), `DiDConfig` (settings). Central to the package.
- **`stats/comparisons.py`** — `within_arm_comparison()`, `between_arm_comparison()`: paired/unpaired Wilcoxon/t-tests with FDR.
- **`stats/effect_size.py`** — Cohen's d, Hedges' g, bootstrap CIs for effect sizes.
- **`stats/gsea.py`** — Gene-set enrichment wrappers around gseapy, operating on DiD results.
- **`stats/abundance.py`** — Cell-type abundance DiD with cluster-robust SEs.
- **`stats/power.py`** — Power analysis: `power_did()`, `power_curve()`, `sample_size_did()`.
- **`scoring.py`** — Gene-set scoring: z-mean, Seurat-style, AUCell methods.
- **`plotting.py`** — All visualization: forest plots, interaction plots, volcanoes, UMAPs, GSEA heatmaps. Functions accept `ax=` for subplot embedding.
- **`datasets.py`** — Loaders for built-in datasets (Sade-Feldman, Stephenson, vaccine). Sade-Feldman includes marker-based cell-type annotation via `_annotate_immune_celltypes()` (requires scanpy).
- **`workflow.py`** — `TrialWorkflow` / `workflow()`: end-to-end convenience pipeline.
- **`validation.py`** — `TrialDataValidator`, AnnData structure checks.
- **`analysis.py`** — High-level analysis helpers (e.g. `run_did_analysis()`).
- **`preprocessing.py`** — AnnData preprocessing (filtering, normalization, HVG selection).
- **`adata_tools.py`** — AnnData manipulation utilities (subsetting, merging, pseudobulk).
- **`convenience.py`** — One-liner wrappers and shortcut functions.
- **`utils.py`** — Shared internal utilities (logging, formatting).
- **`_env.py`** — Environment detection and optional-dependency guards.

### Dataset directory structure

- **Layout**: `datasets/[name]/raw/` for downloads, `datasets/[name]/processed/` for cached h5ad. Single canonical cache path per loader — no multi-directory fallback search. All 5 loaders default `data_dir="datasets/[name]"`.
- **No subsampling defaults**: `max_cells_per_sample`, `max_participants`, `max_cells_per_group` all default to `None`. Never re-add subsampling defaults.
- **Cache validation**: Loaders warn (`UserWarning`) when cached h5ad lacks `processing_params` in `.uns`. Params mismatch triggers automatic reprocessing.
- **Vaccine processed filename**: `vaccine_gse171964.h5ad` (not `vaccine_gse171964_day0_day7.h5ad` or `_full` variant). Day 0/7 filter is in the loader, not the filename.
- **Full dataset sizes**: Sade-Feldman ~8.2GB, Stephenson ~11GB, Vaccine ~3GB, AML ~1.1GB (41K cells), CAR-T ~5GB (405K cells). Total ~28GB. Check `df -h` before reprocessing in worktrees.

### Key design patterns

- **Log-scale axes: powers-of-10 ticks ONLY** — Use `LogLocator(base=10)` + `NullLocator()` for minor. NEVER use 1-2-5 sub-decade ticks (500, 1000, 2000, 5000). This is a hard rule.
- **Memory benchmarks use tracemalloc, not RSS** — RSS delta is too noisy (OS page reclamation, GC). tracemalloc precisely tracks Python heap allocations.
- **Heatmaps with categorical columns: use `imshow`** — `pcolormesh` with numeric x-coords creates non-uniform cell widths. Use `imshow` with integer indices for uniform grids.
- **`adjustText` is incompatible with log-scale axes** — arrows misconnect. Use `ax.annotate` with `textcoords="offset points"` instead.
- **UMAP scatter styling**: Always use `cmap="magma"` for continuous coloring, point size `s=1.5`–`3.0`, `alpha=0.7`, `edgecolors="none"`, `rasterized=True`.
- **Never force-assign biologically incorrect cell-type labels** — if marker scoring fails, use `"Unassigned"`, never a specific type like "CD4 T cell". Mislabeling is worse than leaving unassigned.
- **scanpy is optional but required for Sade-Feldman** — `load_sade_feldman()` raises `ImportError` if scanpy is missing. Guard with try/except at the call site, never inside the annotation function.
- **Editing notebook text cells**: Use `json.load`/`json.dump` to modify `.ipynb` source cells programmatically. Never hand-edit notebook JSON.
- **`TrialDesign` threading**: All statistical functions receive a `TrialDesign` that encodes the experimental structure. Do not hardcode column names.
- **Pseudobulk aggregation**: Cell-level data is aggregated to participant-level before statistical testing to avoid pseudoreplication. The unit of analysis is always the participant (or participant×visit).
- **`ax=` parameter convention**: Plotting functions accept an optional `matplotlib.axes.Axes` for embedding in multi-panel figures.
- **Panel functions that create their own figure**: Return `plt.Figure | None` instead of taking an `ax` argument. Used for small-multiples (power curves) and heatmaps. Save with `save_panel(fig, ...)` directly.
- **When inlining helper modules**: Rename `_prepare_data()` to distinguish (e.g. `_prepare_sf_data()`), store helper data under a sub-key (e.g. `data["scale_data"]`), and remove all `from . import helper as alias` references.
- **Feature columns use spaces in names**: e.g. `sig_Cytotoxic T Cell Activity`, not underscores.
- **Dotplot color scale: raw mean log-expression** — Use `cmap="Reds"`, `vmin=0`, color = `mean(log1p_expr)` per cell type. Never z-score for dotplots.
- **`canonical_markers` keys must match `HARMONIZED_CELLTYPE_ORDER` exactly** — Use "CD4+ T", "B cell", "NK", "T other" (not "CD4 T", "B cells", "NK cells"). Mismatches silently collapse to "Other".
- **pandas Categorical phantom bars** — A `pd.Categorical` column retains defined-but-absent levels as x-axis ticks in seaborn/groupby. Fix: `.astype(str)` before groupby, or `.cat.remove_unused_categories()`.
- **Dataset config pattern for supp figures**: Use explicit `"design": "two_arm"/"single_arm"` field + `_ds_label()` helper for DiD/Δ estimand labels. Never use `arm_treated == arm_control` to infer design type.
- **Single-arm cfg values can be None**: `arm_treated`, `arm_control`, `arm_col` may be `None` for single-arm datasets (e.g., Vaccine). Always use `.get()` with guard: `if t and t in values`.
- **Scientific heatmaps: never fillna(0)** — Missing data must use `mask=mat.isna()` in seaborn, not `fillna(0)` which fabricates zero effects.
- **Effect-size thresholds for sign counting** — When counting positive/negative directions, threshold at |Δ| > 0.05 to filter near-zero noise. Without this, agreement metrics are inflated.
- **Variance decomposition** — η² = SS_between / SS_total (one-way ANOVA). Use group-size-weighted SS_between = Σ(n_k × (mean_k − grand)²). Never compute SS_within separately and add.
- **Two-sample SE for two-arm effects** — SE = sqrt(var_t/n_t + var_c/n_c) (Welch). One-sample SE is only valid for single-arm deltas.
- **Participant delta indexing** — Index by `(participant_id, arm)` tuple to prevent arm misassignment when participant IDs repeat across treatment strata.
- **LOO memory optimization** — Before leave-one-out loops over AnnData, slice to only needed genes: `adata = adata[:, feats].copy()`. Full AnnData copies per iteration cause OOM.
- **Assumption tests on model residuals: per-feature, not pooled** — Shapiro-Wilk, Breusch-Pagan, etc. must run on each feature's residuals separately when residuals come from different OLS models. Pooling violates i.i.d. assumptions. Summarize as median statistic + fraction passing per dataset.
- **Never hardcode column names in diagnostic panels** — Always use `cfg["participant_col"]` (or equivalent config key) for participant column lookups. Column names vary across datasets. Checking for literal `"participant"` will silently fail.
- **Welch t-test: use `scipy.stats.ttest_ind(equal_var=False)`** — Don't mix Welch SE with pooled df. The scipy function handles both SE and Welch-Satterthwaite df correctly.

### Manuscript figures (`manuscript_figures/`)

- `main/figure{1-6}_*.py` — 6 main figures, each self-contained with `_prepare_data()` + `_panel_*()` + `generate()`:
  - `figure1_problem_framework.py` — Pseudoreplication problem & sctrial framework
  - `figure2_melanoma_analysis.py` — Melanoma primary analysis & clinical outcome
  - `figure3_robustness_benchmarking.py` — Statistical robustness, scalability, power, method comparison (9 panels A-H + D2)
  - `figure4_biological_discovery.py` — Pathway & gene-level analysis
  - `figure5_multi_dataset.py` — Cross-cohort generalizability & cell-type analysis
  - `figure6_validation_dynamics.py` — Permutation validation, heterogeneity & temporal dynamics (8 panels A-H)
- Output goes to `/Users/omarm/Documents/Research/projects/sc-trialdiff/manuscript/main/` and `/Users/omarm/Documents/Research/projects/sc-trialdiff/manuscript/supp/`. **Note: `/Users/omarm/Documents/Research/projects/sc-trialdiff/manuscript/` is OUTSIDE the git repo** (repo root is `sctrial/sc_trial_inference/`). It contains datasets, GSEA results, and processed h5ad files that are not version-controlled.
- h5ad files are gitignored. Processed datasets live in `datasets/[name]/processed/` (e.g. `datasets/sade_feldman/processed/sade_feldman_processed_v6.h5ad`). GSEA CSVs at `/Users/omarm/Documents/Research/projects/sc-trialdiff/manuscript/gsea_*/` — these exist only locally.
- Scripts use a `_DATA_CACHE` dict and `_CODE_VERSION` tag for JSON cache invalidation. Bump `_CODE_VERSION` when changing analysis logic.
- Cache stored in `manuscript_figures/_cache/` (NOT inside figure output dirs).
- **Dataset loading in figures**: Use `get_sade_feldman()`, `get_stephenson()`, `get_vaccine()`, `get_aml()`, `get_cart()` from `_shared.py`. Never use `load_clinical_trial_dataset()` (deprecated).

## CI/CD

GitHub Actions (`test.yml`): runs pytest on Python 3.9–3.11 on ubuntu-latest.

## Git workflow

- Default: commit and push directly to `dev` branch. PRs go `dev` → `main` only. Never create feature branches.
- When separating concerns from an existing dev→main PR, cherry-pick to a new branch targeting dev, and force-push dev to drop the commit.
- If dev falls behind main (from merge commits), rebase: `git checkout dev && git rebase origin/main && git push --force-with-lease`
- **NEVER reference Claude/AI in commits, PRs, or any public content** — no Co-Authored-By, no "Generated with Claude Code", no claude branch names, nothing. This is a hard rule.
- After making code changes, always push to the remote and check CI status before considering the task complete. Do not wait to be reminded.
- **Real git repo root**: `/Users/omarm/Documents/Research/projects/sc-trialdiff/sctrial/sc_trial_inference` — always use `git -C` with this path. The top-level `sc-trialdiff/` may have a separate `.git` that is NOT the real repo.

## Workflow preferences

- Always work locally (edit files, run code) rather than delegating to sub-agents or planning remotely, unless explicitly told otherwise.
- Never report notebook or script results as successful without actually verifying the output. If execution fails, say so clearly.

## Bug fixing

- When fixing bugs in notebooks or scripts, check for ALL instances of the same class of error (e.g., wrong layer names, wrong column references) across all files before reporting the fix as complete. Do not fix one instance and miss others.

## Code quality

- Primary language is Python. When editing Python code, ensure imports are correct and test data fixtures are properly set up before claiming tests pass.

## Future features to work on

- **Single-arm power analysis** — `power_paired()` / `sample_size_paired()` for single-arm pre/post designs. Currently the vaccine tutorial defines a custom `power_paired()` because the package only has `power_did()` for two-arm DiD. This is a gap in the API.
