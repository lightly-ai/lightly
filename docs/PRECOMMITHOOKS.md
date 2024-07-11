# Pre-Commit Hooks
We use pre-commit hooks to identify simple issues before submission to code review. In particular, our hooks currently check for:
* Private keys in the commit
* Large files in the commit (>500kB)
* Units which don't pass their unit tests (on push only)

## Install Pre-Commit

`pre-commit` comes as a pip package and is specified in `requirements/dev.txt`.

To install it either run:
```
$ pip install .[dev]
```
Or, to install it separately:
```
$ pip install pre-commit
```

Test your installation:
```
$ pre-commit --version
```
If the installation failed, try
```
$ curl https://pre-commit.com/install-local.py | python -
```
or see the [documentation of pre-commit](https://pre-commit.com/) for more information.

## Install Pre-Commit Hooks
To install the pre-commit hooks specified in `.pre-commit-hooks.yaml`, simply run
```
$ pre-commit install
```
Install the pre-push hooks like this
```
$ pre-commit install --hook-type pre-push
```

You can verify that the hooks were installed correctly with
```
$ pre-commit run --all-files
```
The output should look like this:
```
$ pre-commit run --all-files
Detect Private Key................................Passed
Check for added large files.......................Passed 
```

## Usage
With the new setup, checks for private keys and large files are made before every commit and all tests must pass for a push.
