# Contributing to sctrial

Thank you for your interest in contributing to `sctrial`!

## Code Style
We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Please ensure your code adheres to the project's standards by running `pre-commit`.
Type checking is enforced with **mypy**. Use `python -m mypy src` before opening a PR.

## Development Setup
1. Clone the repository.
2. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Install the pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests
Tests are located in the `tests/` directory and can be run using `pytest`:
```bash
pytest
```

### Coverage Expectations
- New features must include tests.
- If you touch statistical logic, add at least one test that verifies a known effect.
- If you touch tutorials or notebooks, ensure outputs are updated.

## Documentation
- Update docstrings when modifying function behavior or return values.
- Run a docs build locally when changing docs or APIs:
  ```bash
  sphinx-build -b html docs/source docs/build/html
  ```
- If you change example notebooks, execute them and sync to `docs/source/examples/`.

## Branch Naming
Use descriptive branch names:
- `fix/<short-description>`
- `feature/<short-description>`
- `docs/<short-description>`

## Commit Message Format
Prefer short, imperative messages:
- `Add DiDConfig and workflow API`
- `Fix AUCell typing`

## Pull Request Process
1. Create a new branch for your changes.
2. Ensure tests pass locally.
3. Run mypy and ruff checks (CI will enforce both).
3. Submit a pull request with a clear description of the changes.
