---
name: notebook-sync
description: Sync tutorial notebooks between tutorials/ and docs/source/tutorials/. Use before committing or creating PRs that touch notebooks.
disable-model-invocation: true
---

# Notebook Sync Workflow

Ensures tutorial notebooks stay in sync between the two locations.

## Steps

1. **Identify changed notebooks** in `tutorials/`:
   ```bash
   cd /Users/omarm/Documents/Research/projects/sc-trialdiff/sctrial/sc_trial_inference
   git diff --name-only HEAD tutorials/
   ```

2. **Copy changed notebooks** to docs:
   ```bash
   cp tutorials/<notebook>.ipynb docs/source/tutorials/<notebook>.ipynb
   ```

3. **Verify sync**:
   ```bash
   diff tutorials/<notebook>.ipynb docs/source/tutorials/<notebook>.ipynb
   ```

4. **Stage both copies** for commit.

## Locations

- Source: `tutorials/`
- Mirror: `docs/source/tutorials/`
- Both directories must contain identical notebook files
