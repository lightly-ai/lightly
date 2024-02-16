from pathlib import Path

from lightly.api import serve


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
