# Contributing to OinkGameRL

First off, thanks for taking the time to contribute!

## Development Setup

We use modern Python tooling. Please follow these steps:

1.  **Install `uv`**: We use uv for dependency management.
    ```bash
    pip install uv
    ```
2.  **Install dependencies**:
    ```bash
    uv sync --dev
    ```
3.  **Install Pre-commit hooks**:
    ```bash
    uv run pre-commit install
    ```

## Pull Request Process

1.  Ensure all tests pass: `uv run pytest`
2.  Ensure type checking passes: `uv run mypy .`
3.  Update documentation if you change any public APIs.
4.  Submit your PR to the `main` branch.

## Code Style

We use `ruff` for formatting and linting. The pre-commit hook will handle this automatically, but you can run it manually:
```bash
uv run ruff check . --fix
uv run ruff format .
