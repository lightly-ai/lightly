import argparse
from pathlib import Path

import jupytext
import nbformat
from nbformat import NotebookNode


def add_installation_cell(nb: NotebookNode, script_path: Path) -> NotebookNode:
    # Find installation snippet
    with open(script_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# pip install"):
                pip_command = "!" + line.lstrip("# ").strip()
                # Create a new code cell
                code_cell = nbformat.v4.new_code_cell(pip_command)
                # Add the cell to the notebook
                nb.cells.insert(1, code_cell)
                break

    return nb


def covert_to_nbs(scripts_dir: Path, notebooks_dir: Path) -> None:
    # Loop through all Python files in the directory
    for py_file_path in scripts_dir.rglob("*.py"):
        # Construct the full paths
        notebook_path = notebooks_dir / py_file_path.relative_to(
            scripts_dir
        ).with_suffix(".ipynb")
        print(f"Converting {py_file_path} to notebook...")
        notebook = jupytext.read(py_file_path)
        notebook = add_installation_cell(notebook, py_file_path)
        # Make cell ids deterministic to avoid changing ids everytime a notebook is (re)generated.
        for i, cell in enumerate(notebook.cells):
            cell.id = str(i)
        jupytext.write(notebook, notebook_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scripts_dir",
        help="path to the directory containing the python scripts",
    )
    parser.add_argument(
        "notebooks_dir",
        help="path to directory where the generated notebooks are stored",
    )
    args = parser.parse_args()
    covert_to_nbs(Path(args.scripts_dir), Path(args.notebooks_dir))
