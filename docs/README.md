# Documentation Guide

## Prerequisites
Make sure you installed dev dependencies:
```
pip install -r ../requirements/dev.txt
```

You may have to set up a clean environment (e.g. with Conda) and use setuptools from the parent directory:
```
conda create -n lightly python=3.7
conda activate lightly
pip install -e .["all"]
```

For building docs with python files (including tutorials) install detectron2.
This isn't handled in requirements because the version you'll need depends on your GPU/ hardware.
[Follow instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

## Build the Docs
`sphinx` provides a Makefile, so to build the `html` documentation, simply type:
```
make html
```

To build docs without running python files (tutorials) use
``` 
make html-noplot
```

Shortcut to build the docs (with env variables for active-learning tutorial) use:
```
LIGHTLY_SERVER_LOCATION='https://api.lightly.ai' TOKEN='YOUR_TOKEN' AL_TUTORIAL_DATASET_ID='YOUR_DATASET_ID' make html && python -m http.server 1234 -d build/html
```

You can host the docs after building using the following python command 
`python -m http.server 1234 -d build/html` from the docs folder.
Open a browser and go to `http://localhost:1234` to see the documentation.

Once the docs are built they are cached in `docs/build`. A new build will only recompile changed files.
The cache can be cleared with `make clean`.

## Deploy the Docs

Only Lightly core team members will have access to deploy new docs. 

1. Open a terminal and go to the `docs/` folder. 
1. If not done yet, authenticate your account using `gcloud auth login`
1. Deploy to app engine using `gcloud app deploy app.yaml`

## Docstrings and Style Guide
We build our code based on the [Google Python Styleguide]().

Important notes:
- Always use three double-quotes (`"""`).
- A function must have a docstring, unless it meets all of the following criteria: not externally visible, very short, obvious.
- Always use type hints when possible.
- Don't overlook the `Raises`.
- Use punctuation.
- Provide examples only for cli commands and core.py atm.
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
````

### Functions

Example:
```python
def fetch_smalltable_rows(table_handle: smalltable.Table,
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

Attributes of a class should follow the same rules as the arguments for a function.

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
        """Inits SampleClass with blah."""
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

### Moving Pages to readme.com

If a page is to be moved, update the "redirects" dictionary in `source/conf.py` mapping the old page to the new one.

More info is available at [reredirects docs](https://documatt.gitlab.io/sphinx-reredirects/usage.html).
