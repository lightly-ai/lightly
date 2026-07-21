# CLAUDE.md

Guidance for AI assistants working in this repo. For full contributor etiquette (issues, PR process, forking), see `CONTRIBUTING.md`; for maintainer-only tasks, see `MAINTAINING.md`.

## Project overview

`lightly` (lightly-ssl) is a self-supervised learning (SSL) library for
computer vision, built on PyTorch / PyTorch Lightning. It implements 19+ SSL
methods (SimCLR, MoCo, BYOL, DINO, DINOv2, SwaV, VICReg, Barlow Twins, FroSSL,
DetConS, LeJEPA, etc.).

Main entry points:
- `lightly.models` — backbones + method-specific heads
- `lightly.loss` — SSL loss implementations
- `lightly.transforms` — method-specific augmentation pipelines
- `lightly.data` — dataset wrapper + collate functions
- `lightly.cli` — `lightly-*` command-line tools (train, embed, crop, download, serve, magic)
- `lightly.core` — one-liner convenience APIs
- `lightly.api` — communication with the Lightly web app

## Repo layout

- `lightly/` — the package (flat layout, no `src/`)
- `tests/` — mirrors `lightly/`'s structure
- `examples/` — training examples in three variants: `pytorch/`, `pytorch_lightning/`, `pytorch_lightning_distributed/`, plus generated `notebooks/`
- `docs/source/` — Sphinx docs, including one `.rst` page per SSL method under `docs/source/examples/`
- `benchmarks/` — benchmark scripts
- `CONTRIBUTING.md`, `MAINTAINING.md`, `Makefile`, `pyproject.toml`

## Setup & common commands

Package manager is `uv`.

```bash
uv venv && source .venv/bin/activate
make install-dev   # installs all extras + pre-commit hooks
```

| Command | Purpose |
|---|---|
| `make format` | Auto-fix imports/formatting with ruff |
| `make format-check` | Check formatting without fixing |
| `make lint` | Ruff lint (`lint-lightly` + `lint-tests`) |
| `make type-check` | mypy on `lightly` and `tests` |
| `make static-checks` | `format-check` + `type-check` |
| `make test` | `pytest tests --runslow` (full suite) |
| `make test-fast` | `pytest tests` (skips `@pytest.mark.slow`) |
| `make all-checks` | `static-checks` + `test` |
| `make generate-example-notebooks` | Regenerate `examples/notebooks/*` from `examples/{pytorch,pytorch_lightning,pytorch_lightning_distributed}` |

If `make format` reports changes, re-run it before `make all-checks`.

## Code style (see `CONTRIBUTING.md` for full detail)

- Google + PyTorch styleguide. Docstrings use triple double quotes and the
  Google convention (checked by ruff's pydocstyle rules); required on public
  functions unless very short and obvious.
- Full type hints everywhere (mypy-clean). Use Python 3.10-style unions (`str |
  Path`, not `Union[str, Path]`); this requires `from __future__ import
  annotations` at the top of the module (package still declares
  `requires-python = ">=3.6"` and CI tests old Python versions).
- Prefer keyword arguments when calling functions with more than one argument.
- Import functions via their module (`from module import submodule;
  submodule.fn(...)`); import classes directly (`from module.submodule import
  MyClass`).

## Adding or modifying an SSL method (or a standalone transform/loss/model)

Touch points, mirroring an existing method (e.g. BYOL) as the template:

1. `lightly/models/<name>.py`, `lightly/loss/<name>_loss.py`, `lightly/transforms/<name>_transform.py` as needed
2. Export the new symbol from the relevant subpackage `__init__.py`
3. Add `docs/source/examples/<name>.rst` and wire it into its parent toctree
4. Add example scripts under all three `examples/` variants, then run `make
   generate-example-notebooks` and commit the regenerated notebooks (they're
   tracked in git)
5. Add tests mirrored under `tests/{models,loss,transforms}/`, following
   existing naming (e.g. `test_ModelsBYOL.py`, `test_barlow_twins_loss.py`,
   `test_byol_transform.py`)

## Testing

- `pytest`, config in `tests/conftest.py`. Test tree mirrors `lightly/`.
- Slow tests are marked `@pytest.mark.slow` and skipped unless `--runslow` is passed (`make test` passes it, `make test-fast` doesn't).
- `LIGHTLY_SERVER_LOCATION` is set to a dummy URL in `conftest.py` so tests don't hit the real API.

## CI (`.github/workflows/`)

- `test_code_format.yml` — `make static-checks`
- `test.yml` — main unit test suite
- `tests_unmocked.yml` — tests against real dependencies/API where relevant
- `test_minimal_deps.yml` — install with lowest-pinned direct dependencies, then test
- `test_api_deps_only.yml` — install with only API-client dependencies
- `test_setup.yml` — package build/install sanity check
- `check_example_nbs.yml` — verifies generated notebooks are up to date with `examples/`
- `weekly_dependency_test.yml` — scheduled test against latest dependency versions
- `release_pypi.yml` — publish to PyPI

## PR workflow

- Branch off `upstream/master`; never commit directly to `master`.
- Before committing: `make format`, then `make all-checks` (or at least `static-checks` + targeted tests for faster iterations).
- If you touched `examples/`, regenerate notebooks (`make generate-example-notebooks`) and commit them.
- If you touched `docs/source`, verify the docs still build (see `docs/README.md`).
