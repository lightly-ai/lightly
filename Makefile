# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

# All commands run through uv and do not require manually activating a virtual
# environment. Every install target is a single uv command so that the resulting
# environment is fully described by that command plus pyproject.toml.

.DEFAULT_GOAL := help


### Configuration

# Python directories to format and lint.
PYTHON_DIRS = benchmarks docs examples lightly tests

# Min and max Python versions we test against. 3.8 is the lowest version uv can
# provision and the lowest version pyproject.toml's requires-python allows.
MINIMAL_PYTHON_VERSION := 3.8
MAXIMAL_PYTHON_VERSION := 3.12

# Date until which dependencies installed with --exclude-newer must have been released.
# Dependencies released after this date are ignored. This keeps CI stable when new
# versions of dependencies are released. Must be a full timestamp: a bare date is
# interpreted in the local timezone, which would make local and CI runs disagree.
EXCLUDE_NEWER_DATE := "2025-08-07T22:00:00Z"

# Install the package in non-editable mode in CI.
ifdef CI
EDITABLE :=
NO_EDITABLE := --no-editable
else
EDITABLE := -e
NO_EDITABLE :=
endif

# Pytest options shared by all the CI test targets.
PYTEST_CI_OPTS := -v --durations=20 --runslow

# How to invoke tools inside the project environment. Defaults to syncing from the
# lockfile, which is what you want locally. Targets that run against an environment
# built by one of the install-* targets override this with `uv run --no-sync`, because
# syncing would replace the pinned dependency versions with the lockfile's versions.
UV_RUN ?= uv run --frozen


.PHONY: help
help:
	@echo "Common targets:"
	@echo "  install-dev              install the package for local development"
	@echo "  format                   auto-fix imports and formatting with ruff"
	@echo "  static-checks            lock-check + format-check + type-check"
	@echo "  test                     run the full test suite"
	@echo "  test-fast                run the test suite without slow tests"
	@echo "  all-checks               static-checks + test"
	@echo "  generate-example-notebooks  regenerate examples/notebooks from examples/"
	@echo ""
	@echo "See the Makefile for the install-*/test-* targets used by CI."


### Cleaning

.PHONY: clean
clean: clean-build clean-pyc clean-out

# Remove build artifacts.
.PHONY: clean-build
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

# Remove Python file artifacts.
.PHONY: clean-pyc
clean-pyc:
	find . -name '__pycache__' -exec rm -fr {} +

# Remove hydra outputs.
.PHONY: clean-out
clean-out:
	rm -fr outputs/
	rm -fr lightly_outputs/
	rm -fr lightning_logs/
	rm -fr lightly_epoch_*.ckpt
	rm -fr last.ckpt


### Static checks

# Format code with ruff and fix auto-fixable lint errors.
.PHONY: format
format:
	uv run --frozen ruff check --fix $(PYTHON_DIRS)
	uv run --frozen ruff format $(PYTHON_DIRS)

# Check that the code is formatted and free of lint errors.
.PHONY: format-check
format-check:
	@echo "Checking code format..."
	uv run --frozen ruff check $(PYTHON_DIRS)
	uv run --frozen ruff format --check $(PYTHON_DIRS)

# Check typing.
.PHONY: type-check
type-check:
	uv run --frozen mypy lightly tests

# Check that uv.lock is up-to-date with pyproject.toml. Without this, the
# `uv run --frozen` targets above would silently run against a stale environment.
#
# --exclude-newer has to match `lock` below: it is part of the resolution inputs, so
# omitting it here makes uv re-resolve and report the lockfile as outdated.
.PHONY: lock-check
lock-check:
	uv lock --check --exclude-newer $(EXCLUDE_NEWER_DATE)

.PHONY: static-checks
static-checks: lock-check format-check type-check

.PHONY: all-checks
all-checks: static-checks test


### Dependencies

# Update the lockfile using dependencies released before the cutoff date.
.PHONY: lock
lock:
	uv lock --exclude-newer $(EXCLUDE_NEWER_DATE)

# Install the package for local development, including the dependencies needed to
# build the docs.
.PHONY: install-dev
install-dev:
	uv sync --frozen --all-extras --group docs
	uv run --frozen pre-commit install

# Install system dependencies required to build PyAV from source.
#
# Both minimal targets need this, including install-minimal which selects no extras:
# uv always resolves every optional dependency, so --resolution=lowest-direct pulls in
# av==8.0.3, which ships no wheel for Python 3.8 and therefore builds from source. The
# maximal targets get an av wheel instead and need no system libraries.
#
# This installs packages system-wide with apt-get, so it refuses to run outside CI.
.PHONY: _av-system-deps
_av-system-deps:
ifndef CI
	$(error _av-system-deps installs system packages with apt-get and must only run in CI (CI is unset))
endif
	sudo apt-get update
	sudo apt-get install -y libavformat-dev libavdevice-dev

# Install the package with the lowest supported dependency versions.
#
# Explanation of the flags:
# --resolution=lowest-direct: Only install minimal versions for direct dependencies.
#   Transitive dependencies use the latest compatible version. Using --resolution=lowest
#   would also downgrade transitive dependencies, which is not a realistic scenario and
#   results in some extremely old dependencies being installed.
# --exclude-newer: Ignore dependencies released after that date to keep CI stable.
# --group minimal: Provides the version floors to resolve against. Note that this must
#   not be `--extra minimal`: that extra is empty now, so it would silently be a no-op.
#
# These targets deliberately do not pass --frozen: it would take precedence over
# --resolution and install the lockfile's versions instead of the lowest ones.
.PHONY: install-minimal
install-minimal: _av-system-deps
	uv sync --python=$(MINIMAL_PYTHON_VERSION) --resolution=lowest-direct \
		--exclude-newer $(EXCLUDE_NEWER_DATE) $(NO_EDITABLE) --group minimal

# Install the package with the lowest supported dependency versions, including extras.
# See install-minimal for an explanation of the flags.
.PHONY: install-minimal-extras
install-minimal-extras: _av-system-deps
	uv sync --python=$(MINIMAL_PYTHON_VERSION) --resolution=lowest-direct \
		--exclude-newer $(EXCLUDE_NEWER_DATE) $(NO_EDITABLE) --group minimal \
		--extra matplotlib --extra timm --extra video

# Install the package with dependencies pinned to the latest compatible version
# available at EXCLUDE_NEWER_DATE.
.PHONY: install-maximal
install-maximal:
	uv sync --python=$(MAXIMAL_PYTHON_VERSION) --exclude-newer $(EXCLUDE_NEWER_DATE) \
		$(NO_EDITABLE)

# Install the package with all extras and dependencies pinned to the latest compatible
# version available at EXCLUDE_NEWER_DATE.
.PHONY: install-maximal-extras
install-maximal-extras:
	uv sync --python=$(MAXIMAL_PYTHON_VERSION) --exclude-newer $(EXCLUDE_NEWER_DATE) \
		$(NO_EDITABLE) --all-extras

# Install the package with the extras needed to regenerate the example notebooks.
# Excludes the video extra as PyAV is not needed to convert notebooks.
.PHONY: install-notebook
install-notebook:
	uv sync --python=$(MAXIMAL_PYTHON_VERSION) --exclude-newer $(EXCLUDE_NEWER_DATE) \
		$(NO_EDITABLE) --extra matplotlib --extra timm

# Install the package with the latest version of all dependencies. The --upgrade flag
# ensures that the lockfile is ignored.
.PHONY: install-latest
install-latest:
	uv sync --python=$(MAXIMAL_PYTHON_VERSION) --upgrade --reinstall $(NO_EDITABLE) \
		--all-extras

# Install only the tooling needed to build and publish the package.
.PHONY: install-dist
install-dist:
	uv sync --frozen --only-group dist

# Install the package with everything needed to build the documentation.
.PHONY: install-docs
install-docs:
	uv sync --python=$(MAXIMAL_PYTHON_VERSION) --exclude-newer $(EXCLUDE_NEWER_DATE) \
		$(NO_EDITABLE) --all-extras --group docs


### Deprecated dependency targets
#
# These are only kept because lightly-core's build_docs.yml workflow calls them to
# build the docs for docs.lightly.ai. They install into whatever virtual environment
# is already active, which that workflow creates itself.
#
# TODO: Remove once lightly-core's build_docs.yml uses astral-sh/setup-uv and
# `make install-docs` instead.

.PHONY: install-uv
install-uv:
	curl -LsSf https://astral.sh/uv/0.12.3/install.sh | sh

.PHONY: install-pinned-extras
install-pinned-extras:
	uv pip install --exclude-newer $(EXCLUDE_NEWER_DATE) --reinstall $(EDITABLE) . \
		--all-extras --requirement pyproject.toml --group dev --group docs


### Testing

# The CI test targets use `uv run --no-sync` instead of `uv run --frozen` because the
# environment they run in was installed by one of the install-* targets above. With
# --frozen, uv would re-sync the environment to match the lockfile and undo that.

# Run all tests.
.PHONY: test
test:
	uv run --frozen pytest tests --runslow

# Run all tests except the slow ones.
.PHONY: test-fast
test-fast:
	uv run --frozen pytest tests

# One target per CI scenario, so that every workflow job reads
# `make install-<scenario>` followed by `make test-<scenario>`. They currently share
# the same pytest invocation and only differ in the environment the matching
# install-<scenario> target built.
.PHONY: _test-ci
_test-ci:
	uv run --no-sync pytest tests $(PYTEST_CI_OPTS)

.PHONY: test-minimal test-minimal-extras test-maximal test-maximal-extras test-latest
test-minimal test-minimal-extras test-maximal test-maximal-extras test-latest: _test-ci

# Run the @pytest.mark.DDP tests on the shared gloo pool (real multi-rank collective,
# not mocked). USE_PYTEST_POOL enables the pool; it is off by default. See #1982.
#
# python -m pytest is required (instead of plain pytest) because it adds the repo
# root to sys.path, which the spawned pool workers need to import the tests package.
.PHONY: test-distributed
test-distributed:
	USE_PYTEST_POOL=1 uv run --no-sync python -m pytest tests -m DDP $(PYTEST_CI_OPTS)

# Check that the committed example notebooks are up-to-date with examples/.
.PHONY: test-notebooks
test-notebooks:
	$(MAKE) generate-example-notebooks UV_RUN="uv run --no-sync"
	git add examples/notebooks/
	@if ! git diff --cached --exit-code; then \
		echo "Notebooks have changed! Please run 'make generate-example-notebooks' and commit the changes."; \
		exit 1; \
	fi

# Smoke test that the CLI entry points are installed and runnable.
.PHONY: test-cli
test-cli:
	uv run --no-sync lightly-crop --help
	uv run --no-sync lightly-ssl-train --help
	uv run --no-sync lightly-embed --help
	uv run --no-sync lightly-magic --help
	uv run --no-sync lightly-version

# Smoke test the CLI on an actual dataset.
.PHONY: test-cli-dataset
test-cli-dataset:
	rm -rf clothing_dataset_small
	git clone --depth 1 https://github.com/alexeygrigorev/clothing-dataset-small clothing_dataset_small
	uv run --no-sync lightly-ssl-train input_dir=clothing_dataset_small/test/dress \
		trainer.max_epochs=1 loader.num_workers=2
	uv run --no-sync lightly-embed input_dir=clothing_dataset_small/test/dress


### Packaging

# Build source and wheel package.
.PHONY: dist
dist: clean
	uv build
	ls -l dist


### Examples

# Generate notebooks from the example scripts.
.PHONY: generate-example-notebooks
generate-example-notebooks:
	$(UV_RUN) python examples/create_example_nbs.py examples/pytorch examples/notebooks/pytorch
	$(UV_RUN) python examples/create_example_nbs.py examples/pytorch_lightning examples/notebooks/pytorch_lightning
	$(UV_RUN) python examples/create_example_nbs.py examples/pytorch_lightning_distributed examples/notebooks/pytorch_lightning_distributed
