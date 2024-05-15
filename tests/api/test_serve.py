from pathlib import Path

import pytest

from lightly.api import serve


def test_validate_input_mount(tmp_path: Path) -> None:
    (tmp_path / "image.png").touch()
    serve.validate_input_mount(input_mount=tmp_path)


def test_validate_input_mount__not_exist(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match=f"Path for 'input_mount' argument '{tmp_path}/not-existant' does not exist.",
    ):
        serve.validate_input_mount(input_mount=tmp_path / "not-existant")


def test_validate_input_mount__not_directory(tmp_path: Path) -> None:
    (tmp_path / "file.txt").touch()
    with pytest.raises(
        ValueError,
        match=f"Path for 'input_mount' argument '{tmp_path}/file.txt' is not a directory.",
    ):
        serve.validate_input_mount(input_mount=tmp_path / "file.txt")


def test_validate_input_mount__no_files(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match=(
            f"Path for 'input_mount' argument '{tmp_path}' does not contain any images "
            "or videos"
        ),
    ):
        serve.validate_input_mount(input_mount=tmp_path)


def test_validate_lightly_mount(tmp_path: Path) -> None:
    serve.validate_lightly_mount(lightly_mount=tmp_path)


def test_validate_lightly_mount__not_exist(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match=(
            f"Path for 'lightly_mount' argument '{tmp_path}/not-existant' does not "
            "exist."
        ),
    ):
        serve.validate_lightly_mount(lightly_mount=tmp_path / "not-existant")


def test_validate_lightly_mount__not_directory(tmp_path: Path) -> None:
    (tmp_path / "file.txt").touch()
    with pytest.raises(
        ValueError,
        match=(
            f"Path for 'lightly_mount' argument '{tmp_path}/file.txt' is not a "
            "directory."
        ),
    ):
        serve.validate_lightly_mount(lightly_mount=tmp_path / "file.txt")


def test__translate_path(tmp_path: Path) -> None:
    tmp_file = tmp_path / "hello/world.txt"
    assert serve._translate_path(path="/hello/world.txt", directories=[]) == ""
    assert serve._translate_path(path="/hello/world.txt", directories=[tmp_path]) == ""
    tmp_file.mkdir(parents=True, exist_ok=True)
    tmp_file.touch()
    assert serve._translate_path(
        path="/hello/world.txt", directories=[tmp_path]
    ) == str(tmp_file)
    assert serve._translate_path(
        path="/world.txt",
        directories=[tmp_path / "hi", tmp_path / "hello"],
    ) == str(tmp_file)


def test__translate_path__special_chars(tmp_path: Path) -> None:
    (tmp_path / "white space.txt").touch()
    assert serve._translate_path(
        path="/white%20space.txt", directories=[tmp_path]
    ) == str(tmp_path / "white space.txt")

    (tmp_path / "parens(1).txt").touch()
    assert serve._translate_path(
        path="/parens%281%29.txt", directories=[tmp_path]
    ) == str(tmp_path / "parens(1).txt")
