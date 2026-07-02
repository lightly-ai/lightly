from pathlib import Path
from typing import Any, Dict, List, Optional, Set, cast

import toml
import yaml


def test_requirements_base__pyproject() -> None:
    """Check that all dependencies in requirements/base.txt are also in
    pyproject.toml.
    """
    missing_deps = _requirements("base") - _pyproject("dependencies")
    assert not missing_deps


def test_requirements_base__openapi() -> None:
    """Check that all dependencies in requirements/openapi.txt are also in
    requirements/base.txt.
    """
    openapi = {
        dep for dep in _requirements("openapi") if not dep.startswith("setuptools")
    }
    missing_deps = openapi - _requirements("base")
    assert not missing_deps


def test_pre_commit_config__mypy_hook_config() -> None:
    """Check mypy hook uses the project's installed environment, runs on the
    full project rather than individual staged files, and is scoped to the
    pre-push stage (a full mypy run is too slow for every commit).
    """
    hook = _pre_commit_mypy_hook()
    assert hook is not None, "mypy hook not found in .pre-commit-config.yaml"
    assert hook.get("language") == "system", (
        "mypy hook must use language: system so it runs with the project's "
        "installed torch and type stubs, not an isolated environment"
    )
    assert hook.get("pass_filenames") is False, (
        "mypy hook must set pass_filenames: false — mypy needs to analyse the "
        "whole project at once, not individual staged files"
    )
    args = hook.get("args", [])
    assert "lightly" in args and "tests" in args, (
        "mypy hook must pass 'lightly' and 'tests' as args — without them mypy "
        "runs with no targets and checks nothing"
    )
    assert "pre-push" in hook.get("stages", []), (
        "mypy hook must be scoped to the pre-push stage so a full mypy run "
        "does not slow down every commit"
    )


def _pre_commit_mypy_hook() -> Optional[Dict[str, Any]]:
    """Returns the mypy hook dict from .pre-commit-config.yaml."""
    with open(".pre-commit-config.yaml") as f:
        config = yaml.safe_load(f)
    for repo in config["repos"]:
        for hook in repo.get("hooks", []):
            if hook["id"] == "mypy":
                return cast(Dict[str, Any], hook)
    return None


def _pyproject(name: str) -> Set[str]:
    """Returns dependencies from pyproject.toml."""
    return _normalize_dependencies(toml.load("pyproject.toml")["project"][name])


def _requirements(name: str) -> Set[str]:
    """Returns dependencies from requirements files."""
    path = Path(__file__).parent.parent / "requirements" / f"{name}.txt"
    return _normalize_dependencies(path.read_text().splitlines())


def _normalize_dependencies(deps: List[str]) -> Set[str]:
    """Remove spaces, empty lines, and comments."""
    return {dep.replace(" ", "") for dep in deps if dep and not dep.startswith("#")}
