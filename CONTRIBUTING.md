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
   git checkout -b a_descriptive_name_for_my_changes
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

   If you're modifying documents under `docs/source`, make sure to validate that
   they can still be built. This check also runs in CI. 

   ```bash
   cd docs
   make html
   ```
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
   git push -u origin a_descriptive_name_for_my_changes
   ```

7. Once you are satisfied, go to the webpage of your fork on GitHub.
   Click on 'Pull request' to send your changes to the project maintainers for review.

8. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

9. We have a extensive Continuous Integration system that runs tests on all Pull Requests. This
   is to make sure that the changes introduced by the commits don’t introduce errors. When
   all CI tests in a workflow pass, it implies that the changes introduced by a commit do not introduce any errors.
   We have workflows that check unit tests, dependencies, and formatting.

### Style guide

`lightly` follows the [Google styleguide](https://google.github.io/styleguide/pyguide.html) and the [PyTorch styleguide](https://github.com/IgorSusmelj/pytorch-styleguide) by Igor Susmelj.
Check our [documentation writing guide](https://github.com/lightly-ai/lightly/docs/README.md) for more information.

#### This guide was inspired by Transformers [transformers guide to contributing](https://github.com/huggingface/transformers/blob/master/CONTRIBUTING.md) which was influenced by Scikit-learn [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md).