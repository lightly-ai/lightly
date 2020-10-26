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
	rm -fr lightning_logs/
	rm -fr lightly_epoch_*.ckpt

## remove tox cache
clean-tox:
	rm -fr .tox

# check style with flake8
lint: lint-lightly lint-tests

## check lightly style with flake8
lint-lightly:
	flake8 lightly

## check tests style with flake8
lint-tests:
	flake8 tests

## run tests
test:
	pytest tests -n 4

## build source and wheel package
dist: clean 
	python setup.py sdist bdist_wheel
	ls -l dist

## install the package to active site
install: clean 
	pip install .

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
