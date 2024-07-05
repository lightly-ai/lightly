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

## install the package to active site
install: clean 
	pip install .

# Should be same command as in the docs:
# https://docs.lightly.ai/docs/install-lightly#install-the-lightly-python-client
install-api-only: clean
	pip install -r requirements/base.txt
	pip install . --no-deps

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
