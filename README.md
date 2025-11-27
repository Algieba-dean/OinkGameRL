# OinkGameRL

This project implements a game environment and AI agents using `gymnasium`. It follows modern Python engineering practices, utilizing **uv** for dependency management, **Ruff** for linting, and **Pre-commit** for workflow safety.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** (3.12 Recommended)
- **uv** (An extremely fast Python package installer and resolver)

### Installing uv

MacOS / Linux:

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

Windows:

```powershell
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

---

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone [https://github.com/your-username/OinkAllStar.git](https://github.com/your-username/OinkAllStar.git)
   cd OinkAllStar
   ```

2. **Sync the environment**
   This command creates a virtual environment (`.venv`) and installs all dependencies (including dev tools like `pytest` and `ruff`) defined in `pyproject.toml`.

   ```bash
   uv sync --all-extras --dev
   ```

3. **Install Git Hooks (Crucial)**
   We use `pre-commit` to ensure code quality and security before every commit.

   ```bash
   uv run pre-commit install
   ```

   _(Optional) If you encounter issues with `detect-secrets`, initialize the baseline:_

   ```bash
   uv tool run detect-secrets scan > .secrets.baseline
   ```

---

## Development Workflow

We use `uv run` to execute commands within the project's virtual environment. You generally **do not** need to manually activate the venv.

### 1. Running Tests

Run all unit tests using `pytest`:

```bash
uv run pytest
```

Generate a coverage report (HTML report will be in `htmlcov/`):

```bash
uv run pytest --cov --cov-report=html
```

### 2. Linting & Formatting (Ruff)

We use **Ruff** for both linting and formatting.

Check for code issues:

```bash
uv run ruff check .
```

Auto-fix issues and format code:

```bash
uv run ruff check --fix .
uv run ruff format .
```

### 3. Pre-commit Checks

These checks run automatically when you `git commit`. You can also trigger them manually:

```bash
uv run pre-commit run --all-files
```

---

## Project Rules & Best Practices

### Git Workflow

- **Main Branch Protection:** Direct pushes to `main` (or `master`) are **blocked**.
- **Feature Branches:** Always create a new branch for your changes:
  ```bash
  git checkout -b feature/my-new-feature
  ```
- **Pull Requests:** Submit a PR to merge your changes. CI checks must pass before merging.

### Secrets Detection

We use `detect-secrets` to prevent committing API keys or passwords.

- If the hook blocks your commit due to a "false positive" (a random string that looks like a secret), you can update the baseline:
  ```bash
  uv run detect-secrets scan --update .secrets.baseline
  git add .secrets.baseline
  ```

### Code Style

- **Type Hints:** Use type hints for function arguments and return values.
- **Imports:** Ruff handles import sorting automatically.
- **Testing:** New features must include unit tests.

---

## Project Structure

```text
OinkAllStar/
├── games/               # Source code for environments and agents
├── tests/               # Pytest test suite
├── .github/             # GitHub Actions CI configuration
├── .venv/               # Virtual environment (managed by uv)
├── pyproject.toml       # Project configuration & dependencies
├── uv.lock              # Dependency lock file (DO NOT EDIT MANUALLY)
├── .pre-commit-config.yaml # Git hooks configuration
└── README.md            # This file
```

---

## Troubleshooting

**Q: `pre-commit` failed with formatting errors.**
A: Ruff likely auto-fixed your files. Just `git add` the modified files and try `git commit` again.

**Q: CI failed on GitHub but passes locally.**
A: Ensure you have run `uv sync` locally to match the lock file. Check if you forgot to add new files to git.

**Q: How do I add a new library?**
A: Use `uv add <package_name>`. For dev tools (like testing libraries), use `uv add --dev <package_name>`.
