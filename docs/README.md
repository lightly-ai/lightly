# Documentation Guide
All the commands in here are assumed to be run from within the `docs` directory.
## Prerequisites
In a virtual environment, make sure that you install the development dependencies:
```bash
pip install -e "..[dev]"
```
Or if your package manager of choice is `uv`, you can run:
```bash
(cd .. && make install-dev)
```

Maintainers will additionally require an installation including `detectron2` for the release, however it is not necessary for contributors. This isn't handled in requirements because the version you'll need depends on your GPU/hardware. For installing `detectron2`, follow the [instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Build the Docs
The `sphinx` documentation generator provides a Makefile. To build the `html` documentation without running python files (tutorials) use:
``` 
make html-noplot
```
This is the recommended way to locally build the docs before creating a PR and will put the build `.html` files inside `docs/build`. Since above command uses caching to speed up the build, some warnings may not appear after the initial build. It is therefore advisable to do a clean build from time to time by running:
```
make clean-html-noplot
```
The built docs can be viewed by calling:
```bash
make serve-local
```
For building the full docs with python files (including tutorials), run (usually not necessary during development):
```bash
make html
```


## Deploy the Docs

Only Lightly core team members will have access to deploy new docs. 

1. Open a terminal and go to the `docs/` folder. 
1. If not done yet, authenticate your account using `gcloud auth login`
1. Deploy to app engine using `gcloud app deploy app.yaml`
