from typing import List, Set
import toml
from pathlib import Path


def test_requirements_base__pyproject() -> None:
    # Check that all dependencies in requirements/base.txt are also in
    # pyproject.toml.
    missing_deps = _requirements("base") - _pyproject("dependencies")
    assert not missing_deps


def test_requirements_base__openapi() -> None:
    # Check that all dependencies in requirements/openapi.txt are also in
    # requirements/base.txt.
    openapi = {
        dep for dep in _requirements("openapi") if not dep.startswith("setuptools")
    }
    missing_deps = openapi - _requirements("base")
    assert not missing_deps


def _pyproject(name: str) -> Set[str]:
    """Returns dependencies from pyproject.toml."""
    return _normalize_dependencies(toml.load("pyproject.toml")["project"][name])


def _requirements(name: str) -> Set[str]:
    """Returns dependencies from requirements files."""
    path = Path(__file__).parent.parent / "requirements" / f"{name}.txt"
    return _normalize_dependencies(path.read_text().splitlines())


def _normalize_dependencies(deps: List[str]) -> Set[str]:
    # Remove spaces, empty lines, and comments.
    return {dep.replace(" ", "") for dep in deps if dep and not dep.startswith("#")}
