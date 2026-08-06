# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

.PHONY: clean clean-build clean-pyc clean-out docs help
.DEFAULT_GOAL := help

# TODO
help:

## make clean
clean: clean-build clean-pyc clean-out

## remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

## remove python file artifacts
clean-pyc:
	find . -name '__pycache__' -exec rm -fr {} +

## remove hydra outputs
clean-out:
	rm -fr outputs/
	rm -fr lightly_outputs/
	rm -fr lightning_logs/
	rm -fr lightly_epoch_*.ckpt
	rm -fr last.ckpt


# Python directories to format and lint
PYTHON_DIRS = benchmarks docs examples lightly tests

# format code with ruff
format:
	uv run --frozen ruff check --fix --select I $(PYTHON_DIRS)
	uv run --frozen ruff format $(PYTHON_DIRS)

# check if code is formatted with ruff
format-check:
	@echo "Checking code format..."
	uv run --frozen ruff check --select I $(PYTHON_DIRS)
	uv run --frozen ruff format --check $(PYTHON_DIRS)

# lint code with ruff
lint: lint-lightly lint-tests

## lint lightly code with ruff
lint-lightly:
	uv run --frozen ruff check lightly

## lint tests with ruff
lint-tests:
	uv run --frozen ruff check tests

## run tests
test:
	uv run --frozen pytest tests --runslow

test-fast:
	uv run --frozen pytest tests

## check typing
type-check:
	uv run --frozen mypy lightly tests

## run format checks
static-checks: format-check type-check

## run format checks and tests
all-checks: static-checks test

## build source and wheel package
dist: clean
	uv build
	ls -l dist



# uninstall package from active site
uninstall: clean
	pip uninstall lightly


## helper for renaming
find: 
	@read -p "Enter Term: " term; \
	grep -rnw ./ -e "$$term"


### Virtual Environment

.PHONY: install-uv
install-uv:
	curl -LsSf https://astral.sh/uv/0.2.25/install.sh | sh


.PHONY: reset-venv
reset-venv:
	rm -rf .venv
	uv venv


### Dependencies

# Project commands use uv and do not require manually activating the virtual environment.

# Install the package in non-editable mode in CI.
ifdef CI
EDITABLE=
NO_EDITABLE=--no-editable
else
EDITABLE=-e
NO_EDITABLE=
endif

# Date until which dependencies installed with --exclude-newer must have been released.
# Dependencies released after this date are ignored.
EXCLUDE_NEWER_DATE="2025-08-07"

# Min and max Python versions for dependency testing.
MINIMAL_PYTHON_VERSION=3.8
MAXIMAL_PYTHON_VERSION=3.12

# Update the lockfile using dependencies released before the cutoff date.
.PHONY: lock
lock:
	uv lock --exclude-newer ${EXCLUDE_NEWER_DATE}

# Install package for local development.
.PHONY: install-dev
install-dev:
	uv sync --frozen --all-extras
	uv run --frozen pre-commit install


# Install package with API dependencies only.
# Should be same command as in the docs with some extra flags for CI:
# https://docs.lightly.ai/docs/install-lightly#install-the-lightly-python-client
.PHONY: install-api-only
install-api-only:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --requirement requirements/base.txt
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} . --no-deps

# Install package with minimal dependencies.
# 
# The dev dependencies are installed together with the package's minimal dependencies.
# setuptools<50 is then installed for compatibility with old PyTorch Lightning versions
# that do not include the correct setuptools dependencies.
#
# Explanation of flags:
# --exclude-newer: We don't want to install dependencies released after that date to
#   keep CI stable.
# --resolution=lowest-direct: Only install minimal versions for direct dependencies.
#   Transitive dependencies will use the latest compatible version.
# 	Using --resolution=lowest would also download the latest versions for transitive
#   dependencies which is not a realistic scenario and results in some extremely old
#   dependencies being installed.
.PHONY: install-minimal
install-minimal:
	uv sync --python=${MINIMAL_PYTHON_VERSION} --resolution=lowest-direct --exclude-newer ${EXCLUDE_NEWER_DATE} ${NO_EDITABLE} --group dev --extra minimal --upgrade-group dev
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall "setuptools<50"

# Install package with minimal dependencies including extras.
# See install-minimal for explanation of flags.
# Install selected extras separately so their minimal versions can be tested.
# Development dependencies are installed from the dev dependency group.
.PHONY: install-minimal-extras
install-minimal-extras:
	uv sync --python=${MINIMAL_PYTHON_VERSION} --resolution=lowest-direct --exclude-newer ${EXCLUDE_NEWER_DATE} ${NO_EDITABLE} --group dev --extra matplotlib --extra minimal --extra timm --extra video --upgrade-group dev
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall "setuptools<50"

# Install package with dependencies pinned to the latest compatible version available at
# EXCLUDE_NEWER_DATE. This keeps CI stable if new versions of dependencies are released.
.PHONY: install-pinned
install-pinned:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall ${EDITABLE} . --requirement pyproject.toml

# Install package with all extras and dependencies pinned to the latest compatible
# version available at EXCLUDE_NEWER_DATE. This keeps CI stable if new versions of
# dependencies are released.
.PHONY: install-pinned-extras
install-pinned-extras:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall ${EDITABLE} . --all-extras --requirement pyproject.toml

# Install package with selected extras pinned to the latest compatible version
# available at EXCLUDE_NEWER_DATE. This excludes video dependencies.
.PHONY: install-pinned-extras-no-video
install-pinned-extras-no-video:
	uv sync --exclude-newer ${EXCLUDE_NEWER_DATE} ${NO_EDITABLE} --group dev --extra matplotlib --extra minimal --extra timm --upgrade-group dev

# Install package with pinned extras for notebook CI checks.
.PHONY: install-pinned-notebook
install-pinned-notebook: install-pinned-extras-no-video

# Install system dependencies for PyAV in CI.
.PHONY: install-av-system-deps
install-av-system-deps:
ifdef CI
	sudo apt-get update
	sudo apt-get install -y libssh-gcrypt-4=0.9.6-2build1 libavformat-dev libavdevice-dev
else
	@echo "Skipping PyAV system dependencies (CI only)."
endif

# Install package with pinned extras for specific Python versions used in CI.
.PHONY: install-pinned-extras-3.7 install-pinned-extras-3.12
install-pinned-extras-3.7: install-pinned-extras-no-video
install-pinned-extras-3.12: install-av-system-deps install-pinned-extras

# Install package with the latest dependencies.
.PHONY: install-latest
install-latest:
	uv sync --python=${MAXIMAL_PYTHON_VERSION} --upgrade --reinstall ${NO_EDITABLE} --group dev --all-extras


# Generate Notebooks from examples
.PHONY: generate-example-notebooks
generate-example-notebooks:
	uv run --frozen python examples/create_example_nbs.py examples/pytorch examples/notebooks/pytorch
	uv run --frozen python examples/create_example_nbs.py examples/pytorch_lightning examples/notebooks/pytorch_lightning
	uv run --frozen python examples/create_example_nbs.py examples/pytorch_lightning_distributed examples/notebooks/pytorch_lightning_distributed
