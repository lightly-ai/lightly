# How to contribute to lightly?

Everyone is welcome to contribute, and we value everybody's contribution. Code is thus not the only way to help the community. Answering questions, helping others, reaching out and improving the documentations are immensely valuable to the community.

It also helps us if you spread the word: reference the library from blog posts on the awesome projects it made possible, shout out on Twitter every time it has helped you, or simply star the repo to say "thank you".

## You can contribute in so many ways!

There are 4 ways you can contribute to lightly:
* Fixing outstanding issues with the existing code;
* Implementing new models;
* Contributing to the examples or to the documentation;
* Submitting issues related to bugs or desired new features.

*All are equally valuable to the community.*

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

First, **please make sure the bug was not already reported** (use the search bar on Github under Issues).

* Include your **OS type and version**, the versions of **Python**, **PyTorch**, and **PyTorch Lightning**.
* A code snippet that allows us to reproduce the bug in less than 30s.
* Provide the *full* traceback if an exception is raised.

### Do you want to implement a new self-supervised model?

Awesome! Please provide the following information:

* Short description of the model and link to the paper;
* Link to the implementation if it's open source;

If you are willing to contribute the model yourself, let us know so we can best
guide you.

### Do you want a new feature (that is not a model)?

A world-class feature request addresses the following points:

1. Motivation first:
  * Is it related to a problem/frustration with the library? If so, please explain
    why. Providing a code snippet that demonstrates the problem is best.
  * Is it related to something you would need for a project? We'd love to hear
    about it!
  * Is it something you worked on and think could benefit the community?
    Awesome! Tell us what problem it solved for you.
2. Provide a **code snippet** that demonstrates its future use;
3. Attach any additional information (drawings, screenshots, etc.) you think may help.


## Pull Requests

Before writing code, we strongly advise you to search through the exising PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

Follow these steps to start contributing:

1. Fork the [repository](https://github.com/lightly-ai/lightly/) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:lightly-ai/lightly.git
   cd lightly
   git remote add upstream https://github.com/lightly-ai/lightly.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a_descriptive_name_for_my_changes upstream/master
   ```

   **do not** work on the `master` branch.

4. Set up a development environment by running the following command in a virtual environment:

   ```bash
   pip install -e ".[dev]"
   ```

   If you are using [uv](https://github.com/astral-sh/uv) instead of pip, you can use
   the following command:

   ```bash
   make install-dev
   ```

5. **(Optional)** Install pre-commit hooks:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

   We use pre-commit hooks to identify simple issues before submission to code review. In particular, our hooks currently check for:
   * Private keys in the commit
   * Large files in the commit (>500kB)
   * Run formatting checks using `black`, `isort` and `mypy`.
   * Units which don't pass their unit tests (on push only)

   You can verify that the hooks were installed correctly with
   ```
   pre-commit run --all-files
   ```
   The output should look like this:
   ```
   pre-commit run --all-files
   Detect Private Key................................Passed
   Check for added large files.......................Passed
   black.............................................Passed
   isort.............................................Passed
   mypy..............................................Passed
   ```

6. Develop the features on your branch.

   As you work on the features, you should make sure that the code is formatted and the
   test suite passes:

   ```bash
   make format
   make all-checks
   ```

   If you get an error from isort or black, please run `make format` again before
   running `make all-checks`.

   If you're modifying examples under `examples/`, make sure to update the corresponding notebooks by
   running the following command:

   ```bash
   make generate-example-notebooks
   ```
   
   If you're modifying documents under `docs/source`, make sure to validate that
   they can still be built. This check also runs in CI and the build instructions can be found in `docs/README.md`.

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   git add modified_file.py
   git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

   Push the changes to your account using:

   ```bash
   git push -u upstream a_descriptive_name_for_my_changes
   ```

1. Once you are satisfied, go to the webpage of your fork on GitHub.
   Click on 'Pull request' to send your changes to the project maintainers for review. If there is a change in the docs, please make sure to print the changes made to the webpage as PDF and include them in the PR.

2. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

3. We have a extensive Continuous Integration system that runs tests on all Pull Requests. This
   is to make sure that the changes introduced by the commits don’t introduce errors. When
   all CI tests in a workflow pass, it implies that the changes introduced by a commit do not introduce any errors.
   We have workflows that check unit tests, dependencies, and formatting.

### Style guide

`lightly` follows the [Google styleguide](https://google.github.io/styleguide/pyguide.html) and the [PyTorch styleguide](https://github.com/IgorSusmelj/pytorch-styleguide) by Igor Susmelj.

Important notes:
- Always use triple double quotes (`"""`).
- A function must include a docstring unless it meets all the following criteria: it is not externally visible, is very short, and is obvious.
- Make your functions checkable through static typecheckers (`mypy`). This means that it must have proper [type hints](https://docs.python.org/3/library/typing.html) everywhere. We use Python 3.10-style type-hints for Union-types, i.e. `str | Path` instead of `Union[str, Path]`. For backwards-compatibility, this requires that every module using such type-hints imports `from __future__ import annotations` at the very top of the module.
- Don't overlook the `Raises`.
- Use punctuation.
- **Please look carefully at the examples provided below (from the styleguide)**.

#### Packages and Modules
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

#### Functions

Example of a function:
```python
from __future__ import annotations

from smalltable import Table
from typing import Sequence, Union, Mapping, Tuple


def fetch_smalltable_rows(
  table_handle: Table,
  keys: Sequence[bytes | str],
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

#### Classes

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

#### This guide was inspired by Transformers [transformers guide to contributing](https://github.com/huggingface/transformers/blob/master/CONTRIBUTING.md) which was influenced by Scikit-learn [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md).