# Contributing to sctrial

Thank you for your interest in contributing to `sctrial`!

## Code Style
We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Please ensure your code adheres to the project's standards by running `pre-commit`.

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

## Pull Request Process
1. Create a new branch for your changes.
2. Ensure tests pass locally.
3. Submit a pull request with a clear description of the changes.
