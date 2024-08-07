# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

.PHONY: clean clean-build clean-pyc clean-out docs help
.DEFAULT_GOAL := help

# TODO
help:

## make clean
clean: clean-tox clean-build clean-pyc clean-out

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

## remove tox cache
clean-tox:
	rm -fr .tox

# format code with isort and black
format:
	isort .
	black .

# check if code is formatted with isort and black
format-check:
	@echo "⚫ Checking code format..."
	isort --check-only --diff .
	black --check .

# check style with flake8
lint: lint-lightly lint-tests

## check lightly style with flake8
lint-lightly:
	pylint --rcfile=pylintrc lightly

## check tests style with flake8
lint-tests:
	pylint --rcfile=pylintrc tests

## run tests
test:
	pytest tests --runslow

test-fast:
	pytest tests

## check typing
type-check:
	mypy lightly tests

## run format checks
static-checks: format-check type-check

## run format checks and tests
all-checks: static-checks test

## build source and wheel package
dist: clean
	python -m build
	ls -l dist



# uninstall package from active site
uninstall: clean
	pip uninstall lightly

## run tests in tox envs
tox:
	tox

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
	deactivate || true
	rm -rf .venv
	uv venv .venv


### Dependencies

# When running these commands locally, it is recommended to first reset the environment
# with: `make reset-venv && source .venv/bin/activate`
# Otherwise old dependencies might linger around.

# Set EDITABLE to -e to install the package in editable mode outside of CI. This is
# useful for local development.
ifdef CI
EDITABLE=
else
EDITABLE=-e
endif

# All extras without dev requirements.
EXTRAS = [matplotlib,openapi,timm,video]

# Date until which dependencies installed with --exclude-newer must have been released.
# Dependencies released after this date are ignored.
EXCLUDE_NEWER_DATE="2024-08-07"

# Install package for local development.
.PHONY: install-dev
install-dev:
	uv pip install ${EDITABLE} . --all-extras --requirement pyproject.toml


# Install package with API dependencies only.
# Should be same command as in the docs with some extra flags for CI:
# https://docs.lightly.ai/docs/install-lightly#install-the-lightly-python-client
.PHONY: install-api-only
install-api-only:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --requirement requirements/base.txt
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} . --no-deps

# Install package with minimal dependencies.
# 
# This command is split into two steps:
# 1. Install the dev dependencies to be able to run tests. We don't want to use
#    the minimal versions for these dependencies.
# 2. Then we reinstall the package with minimal dependencies.
#
# Explanation of flags:
# --exclude-newer: We don't want to install dependencies released after that date to
#   keep CI stable.
# --resolution=lowest-direct: Only install minimal versions for direct dependencies.
#   Transitive dependencies will use the latest compatible version.
# 	Using --resolution=lowest would also download the latest versions for transitive
#   dependencies which is not a realistic scenario and results in some extremely old
#   dependencies being installed.
# --reinstall: Reinstall dependencies to make sure they satisfy the constraints.
.PHONY: install-minimal
install-minimal:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} ${EDITABLE} ".[dev]"
	uv pip install --resolution=lowest-direct --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall ${EDITABLE} .

# Install package with minimal dependencies including extras.
# See install-minimal for explanation of flags.
# We use ".${EXTRAS}" instead of --all-extras because --all-extras includes the dev
# dependencies for which we don't want to install the minimal versions.
.PHONY: install-minimal-extras
install-minimal-extras:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} ${EDITABLE} ".[dev]"
	uv pip install --resolution=lowest-direct --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall ${EDITABLE} ".${EXTRAS}" --requirement pyproject.toml

# Install package with fixed dependencies.
.PHONY: install-fixed
install-fixed:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall ${EDITABLE} . --requirement pyproject.toml

# Install package with fixed dependencies and extras.
.PHONY: install-fixed-extras
install-fixed-extras:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall ${EDITABLE} . --all-extras --requirement pyproject.toml

# Install package with the latest dependencies.
.PHONY: install-latest
install-latest:
	uv pip install --upgrade --reinstall ${EDITABLE} . --all-extras --requirement pyproject.toml
