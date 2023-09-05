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
