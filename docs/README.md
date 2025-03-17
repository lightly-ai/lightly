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

To create a shortcut for building the documentation with environment variables for the active-learning tutorial, use:
```
LIGHTLY_SERVER_LOCATION='https://api.lightly.ai' LIGHTLY_TOKEN='YOUR_TOKEN' AL_TUTORIAL_DATASET_ID='YOUR_DATASET_ID' make html && python -m http.server 1234 -d build/html
```

## Deploy the Docs

Only Lightly core team members will have access to deploy new docs. 

1. Open a terminal and go to the `docs/` folder. 
1. If not done yet, authenticate your account using `gcloud auth login`
1. Deploy to app engine using `gcloud app deploy app.yaml`

## Docstrings and Style Guide
We build our code based on the [Google Python Styleguide](https://google.github.io/styleguide/pyguide.html).

Important notes:
- Always use triple double quotes (`"""`).
- A function must include a docstring unless it meets all the following criteria: it is not externally visible, is very short, and is obvious.
- Make your functions checkable through static typecheckers (`mypy`). This means that it must have proper [type hints](https://docs.python.org/3/library/typing.html) everywhere.
- Don't overlook the `Raises`.
- Use punctuation.
- **Please look carefully at the examples provided below (from the styleguide)**.

### Packages and Modules
Packages (i.e. the `__init__.py` files) and modules should start with a docstring describing the contents and usage of the package / module. 

Example:
```python
"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
```

### Functions

Example of a function:
```python
from smalltable import Table
from typing import Sequence, Union, Mapping, Tuple


def fetch_smalltable_rows(
  table_handle: Table,
  keys: Sequence[Union[bytes, str]],
  require_all_keys: bool = False,
) -> Mapping[bytes, Tuple[str]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
      table_handle:
        An open smalltable.Table instance.
      keys:
        A sequence of strings representing the key of each table row to
        fetch.  String keys will be UTF-8 encoded.
      require_all_keys:
        Optional; If require_all_keys is True only rows with values set
        for all keys will be returned.

    Returns:
      A dict mapping keys to the corresponding table row data
      fetched. Each row is represented as a tuple of strings. For
      example:

      {b'Serak': ('Rigel VII', 'Preparer'),
       b'Zim': ('Irk', 'Invader'),
       b'Lrrr': ('Omicron Persei 8', 'Emperor')}

      Returned keys are always bytes.  If a key from the keys argument is
      missing from the dictionary, then that row was not found in the
      table (and require_all_keys must have been False).

    Raises:
      IOError: An error occurred accessing the smalltable.
    """
```

### Classes

Attributes of a class should be documented at the class level if they are meant to be public.

Example:
```python
class SampleClass:
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam:
            A boolean indicating if we like SPAM or not.
        eggs:
            An integer count of the eggs we have laid.
    """

    def __init__(self, likes_spam=False):
        """Inits SampleClass with blah.
        
        Args:
            likes_spam:
                Boolean value indicating if we like SPAM or not.
        """
        self.likes_spam = likes_spam
        self.eggs = 0

    def public_method(self):
        """Performs operation blah."""

    def public_method_2(self, x: str):
        """Performs operation blah 2. 
        
        Args:
            x:
                Some explanation for x.
        """
```
