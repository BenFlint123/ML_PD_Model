# ML_PD_Model — AI Coding Guide

Workspace-wide guidance. Path-scoped instructions live under
[`.github/instructions/`](./instructions/) and are loaded automatically when
you edit matching files.

## Project overview

A sandbox for applying ML approaches to credit risk, specifically
probability of default (PD) modelling. Interpretable champion models (e.g.
logistic regression) are benchmarked against ML challengers (e.g. LightGBM).
Proofs of concept are built on the clean Taiwan credit card dataset, then
tested and iterated on the larger, messier Lending Club dataset.

Layout: numbered notebooks for exploration and modelling, a `lib/` package
for reusable modelling code (see `lib/model_development.py`), shared
constants in `config.py`, and a gitignored `data/` folder. Someone should be
able to clone this, run `uv sync`, drop the raw data in `data/raw/`, and run
the notebooks in order.

## Stable conventions

- **Flat `lib/` package, editable install via `uv`.** `uv sync` installs
  `lib` in editable mode, so `import lib` (or `from lib.x import y`) works
  identically in VS Code notebooks, plain CLI Jupyter, scripts, and pytest.
  No `sys.path` manipulation anywhere — if imports fail, fix the install
  (`uv sync`), not the path.
- **`data/` is never committed**, except the `.gitkeep` markers that keep
  `raw/`, `interim/`, `processed/` present after a fresh clone. Don't add
  data files to git, however small — this repo is meant to be handed a
  dataset, not to carry one.
- **Notebooks live under `notebooks/`**, numbered/prefixed by topic (e.g.
  `01_explore.ipynb`). `nbstripout` strips outputs automatically on commit
  — don't disable it or hand-edit notebook JSON to work around it.
- **`pyproject.toml` + `uv.lock` are the source of truth for dependencies.**
  No `requirements.txt`, no `setup.py`.
- **Python 3.13 minimum** (pinned via `.python-version`).

## Tooling

- **uv** — Python install, venv, dependency management, and (via the
  minimal `hatchling` build backend) the editable install that makes
  imports work.
- **Ruff** — sole linter and formatter.
- **pytest + pytest-cov** — tests for `lib/`. Coverage is *reported*, never
  *gated*.
- **pre-commit** — commit stage: ruff + hygiene + nbstripout; push stage:
  pytest.

Deliberately absent: mypy, bandit, docstring enforcement, and a
version/release process — none of it earns its keep for exploratory
analysis work. Add any of it back per-project if a specific need arises.

## Pull request conventions

Prefix PR titles with the type of change. Supported types:
`feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `deps`, `revert`.

Example: `feat: add random forest challenger`.

Link related issues in the PR body using `Closes #<n>` or `Relates to #<n>`.

## Common pitfalls (workspace-wide)

- ❌ `sys.path.insert(...)` anywhere → ✅ rely on the editable install
- ❌ `python.analysis.extraPaths` in `.vscode/settings.json` → ✅ leave it minimal
- ❌ `requirements.txt` → ✅ `pyproject.toml` + `uv.lock`
- ❌ Committing files under `data/` (besides `.gitkeep`) → ✅ keep data local, out of git
- ❌ Committing a notebook with cell outputs still attached → ✅ let `nbstripout` do its job
- ❌ `--cov-fail-under` in pytest addopts → ✅ coverage is reported, never gated

## Where to look for path-scoped guidance

| When editing… | See |
|---|---|
| `tests/**` | [tests.instructions.md](./instructions/tests.instructions.md) |
| `notebooks/**` | [notebooks.instructions.md](./instructions/notebooks.instructions.md) |
| `.github/workflows/**` | [workflows.instructions.md](./instructions/workflows.instructions.md) |
| `pyproject.toml` | [pyproject.instructions.md](./instructions/pyproject.instructions.md) |
