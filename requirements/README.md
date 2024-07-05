All dependencies are tracked in `pyproject.toml` and no new files should be added to
the `requirements` directory.

We maintain `base.txt` to allow installing the package only with the dependencies
necessary to use the API part of the package. The package can be installed with API
only dependencies by running:
```
pip install -r requirements/base.txt
pip install lightly --no-deps
```
This is also documented in our Lightly Worker docs:
https://docs.lightly.ai/docs/install-lightly#install-the-lightly-python-client

It is currently not possible to move these dependencies to an optional dependency
group in `pyproject.toml` because pip does not support installing only optional
dependencies. See https://github.com/pypa/pip/issues/11440

`openapi.txt` is automatically created by the API generator and should not be modified
manually.

There are tests in [`tests/test_requirements.py`](../tests/test_requirements.py) that
check that the dependencies in `base.txt` are in sync with the dependencies in
`pyproject.toml` and `openapi.txt`.

There is also a [GitHub Action](../.github/workflows/test_api_deps_only.yml) that
verifies that installing only the API part of the package works correctly. 
