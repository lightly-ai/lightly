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


def test__strip_leading_slashes() -> None:
    assert serve._strip_leading_slashes("/") == ""
    assert serve._strip_leading_slashes("//") == ""
    assert serve._strip_leading_slashes("/hello") == "hello"
    assert serve._strip_leading_slashes("/hello/world") == "hello/world"
    assert serve._strip_leading_slashes("//hello/world/") == "hello/world/"
