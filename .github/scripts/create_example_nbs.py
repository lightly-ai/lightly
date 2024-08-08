import argparse
import os

import nbformat
from jupytext import read, write
from nbformat import NotebookNode

# python snippet to install lightly
code_content: str = """!pip install lightly"""


def add_installation_cell(
    nb: NotebookNode, content: str = code_content
) -> NotebookNode:
    # Create a new code cell
    code_cell = nbformat.v4.new_code_cell(content)

    # Add the code cell to the beginning of the notebook
    nb.cells.insert(0, code_cell)

    return nb


def covert_to_nbs(scripts_dir: str, notebooks_dir: str) -> None:
    # Loop through all Python files in the directory
    for script in os.listdir(scripts_dir):
        if script.endswith(".py"):
            # Construct the full paths
            py_file_path = os.path.join(scripts_dir, script)
            notebook_name = os.path.splitext(script)[0] + ".ipynb"
            notebook_path = os.path.join(notebooks_dir, notebook_name)

            if not os.path.exists(notebook_path):
                print(f"Converting {script} to notebook...")
                notebook = read(py_file_path)
                notebook = add_installation_cell(notebook)
                write(notebook, notebook_path)
            else:
                print(f"Notebook {notebook_name} already exists. Skipping conversion.")


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
    covert_to_nbs(args.scripts_dir, args.notebooks_dir)
