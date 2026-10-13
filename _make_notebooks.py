"""
Generate Energy_Model.ipynb and Household_EF.ipynb from their .py scripts
(no nbformat dependency — stdlib json only).

The .py files are the source of truth. Each `# %% <title>` marker starts a
new cell: the title becomes a markdown heading and the code below it a code
cell. The module docstring becomes the notebook's intro text.

Run after editing either script:

    python _make_notebooks.py
"""
import ast
import json
import os
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {
    "Energy_Model.py": ("Energy_Model.ipynb", "Energy Model"),
    "Household_EF.py": ("Household_EF.ipynb", "Household Energy Footprint"),
}


def _src(s):
    # Notebook expects source as list of lines, each ending with \n
    # (last line may omit trailing \n; either is fine).
    lines = s.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def md(text):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": _src(text),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    }


def py_to_cells(path, title):
    text = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    module = ast.parse(text)
    doc = ast.get_docstring(module) or ""
    # Drop the docstring from the code; it becomes the intro markdown.
    if doc:
        end_line = module.body[0].end_lineno
        text = "\n".join(text.split("\n")[end_line:])

    cells = [md(f"# {title}\n\n{doc}")]
    chunks = []
    header, body = None, []
    for line in text.split("\n"):
        if line.startswith("# %%"):
            chunks.append((header, body))
            header, body = line[4:].strip(), []
        else:
            body.append(line)
    chunks.append((header, body))

    for header, body in chunks:
        body_text = "\n".join(body).strip("\n")
        if header:
            cells.append(md(f"## {header}"))
        if body_text.strip():
            cells.append(code(body_text))
    return cells


def main():
    for script, (notebook, title) in SCRIPTS.items():
        nb = {
            "cells": py_to_cells(os.path.join(_HERE, script), title),
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        out = os.path.join(_HERE, notebook)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
