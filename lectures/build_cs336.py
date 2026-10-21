#!/usr/bin/env python3
"""Convert a Stanford CS336 EdTrace lecture to a Jupyter notebook."""

import argparse
import ast
import copy
import json
import os
import pathlib
import re
import textwrap

HELPERS = {
    "merge",
    "get_compression_ratio",
    "get_gpt5_tokenizer",
    "output_tokenizer",
    "train_bpe",
    "count_adjacent_pairs",
}

LINK_FUNCS = {"link", "post_link", "article_link", "video_link"}


def convert(source: pathlib.Path, output: pathlib.Path):
    source = source.resolve()
    output = output.resolve()

    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    tree = ast.parse("".join(lines), filename=str(source))

    functions = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }

    if "main" not in functions:
        raise ValueError("Il file non contiene main(): formato non riconosciuto")

    cells = []
    visited = set()

    def original(node):
        return "".join(lines[node.lineno - 1:node.end_lineno]).rstrip()

    def name(call):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
            return call.func.id
        return None

    def literal(node):
        try:
            return ast.literal_eval(node)
        except (TypeError, ValueError, SyntaxError, MemoryError):
            return None

    def add_cell(kind, content):
        content = content.strip("\n")
        if not content.strip():
            return

        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": content.splitlines(keepends=True),
        }

        if kind == "code":
            cell.update(execution_count=None, outputs=[])

        cells.append(cell)

    def link_markdown(call):
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        arg = call.args[0] if call.args else None

        url = (
            literal(kwargs.get("url"))
            if "url" in kwargs
            else literal(arg)
        )
        title = literal(kwargs.get("title")) if "title" in kwargs else None

        if isinstance(url, str):
            return f" [{title or url}]({url})"

        if isinstance(arg, ast.Name):
            return f" *[{arg.id}]*"

        if arg is not None:
            return f" `{ast.unparse(arg)}`"

        return ""

    def markdown(expr):
        if isinstance(expr, ast.Tuple):
            parts = [markdown(item) for item in expr.elts]
            return (
                "".join(parts)
                if all(part is not None for part in parts)
                else None
            )

        if not isinstance(expr, ast.Call):
            return None

        func = name(expr)

        if func == "text" and expr.args:
            value = literal(expr.args[0])
            return value if isinstance(value, str) else None

        if func == "image" and expr.args:
            url = literal(expr.args[0])

            if not isinstance(url, str):
                return None

            if not url.startswith(("https://", "http://")):
                local_image = source.parent / url

                if local_image.is_file():
                    url = pathlib.Path(
                        os.path.relpath(local_image, output.parent)
                    ).as_posix()
                else:
                    url = (
                        "https://raw.githubusercontent.com/"
                        "stanford-cs336/lectures/main/" + url
                    )

            return f"![Lecture image]({url})"

        if func in LINK_FUNCS:
            return link_markdown(expr)

        return None

    # --------------------------------------------------
    # Notebook initialization
    # --------------------------------------------------

    add_cell(
        "markdown",
        f"# CS336 — {source.stem}\n\n"
        "Converted from the original Stanford executable lecture.",
    )

    add_cell(
        "code",
        "\n".join([
            "import os",
            "import regex",
            "import tiktoken",
            "from abc import ABC",
            "from dataclasses import dataclass",
            "from collections import defaultdict",
        ]),
    )

    # --------------------------------------------------
    # Preserve classes and helper functions
    # --------------------------------------------------

    definitions = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            beginning = min(
                (d.lineno for d in node.decorator_list),
                default=node.lineno,
            )

            definitions.append(
                "".join(lines[beginning - 1:node.end_lineno]).rstrip()
            )

        elif isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            if node.name == "train_bpe":
                clean = copy.deepcopy(node)

                clean.body = [
                    statement
                    for statement in clean.body
                    if not (
                        isinstance(statement, ast.Expr)
                        and markdown(statement.value) is not None
                    )
                ]

                definitions.append(ast.unparse(clean))

            else:
                definitions.append(original(node))

    add_cell(
        "markdown",
        "## Definitions\n\n"
        "Classes and helper functions used by the following examples.",
    )

    add_cell("code", "\n\n\n".join(definitions))

    # --------------------------------------------------
    # Convert lecture content in execution order
    # --------------------------------------------------

    def process(function_name):
        if function_name in visited:
            raise ValueError(
                f"La funzione {function_name} è richiamata più volte: "
                "conversione ambigua"
            )

        visited.add(function_name)

        pending_markdown = []
        pending_code = []

        def flush_markdown():
            if pending_markdown:
                add_cell("markdown", "\n\n".join(pending_markdown))
                pending_markdown.clear()

        def flush_code():
            if pending_code:
                add_cell("code", "\n\n".join(pending_code))
                pending_code.clear()

        for statement in functions[function_name].body:

            # Convert text(), image(), link() to Markdown
            if isinstance(statement, ast.Expr):
                converted = markdown(statement.value)

                if converted is not None:
                    flush_code()
                    pending_markdown.append(converted)
                    continue

                # Follow calls to lecture sections
                if isinstance(statement.value, ast.Call):
                    child_name = name(statement.value)

                    if child_name in functions and child_name not in HELPERS:
                        flush_code()
                        flush_markdown()
                        process(child_name)
                        continue

            # Move the explanatory text from train_bpe
            # immediately before the executable example.
            if (
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Call)
                and name(statement.value) == "train_bpe"
            ):
                flush_code()
                pending_markdown.append(
                    "Start with the list of bytes of `string`."
                )

            flush_markdown()
            pending_code.append(textwrap.dedent(original(statement)))

        flush_code()
        flush_markdown()

    process("main")

    # --------------------------------------------------
    # Build notebook
    # --------------------------------------------------

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    # Validate Python syntax and residual EdTrace calls
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])

            compile(code, f"cell_{index}", "exec")

            if re.search(r"\b(?:text|image|link)\s*\(", code):
                raise ValueError(
                    f"EdTrace call remaining in Python cell {index}"
                )

    # Save notebook
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    n_markdown = sum(c["cell_type"] == "markdown" for c in cells)
    n_code = sum(c["cell_type"] == "code" for c in cells)

    print(f"Created: {output}")
    print(f"Total cells: {len(cells)}")
    print(f"Markdown cells: {n_markdown}")
    print(f"Python cells: {n_code}")


# --------------------------------------------------
# Command-line interface
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a CS336 EdTrace lecture into a Jupyter notebook."
    )

    parser.add_argument(
        "source",
        type=pathlib.Path,
        help="Path to the original lecture .py file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output notebook path (default: alongside source .py)",
    )

    args = parser.parse_args()

    output = args.output or args.source.with_suffix(".ipynb")

    convert(args.source, output)


if __name__ == "__main__":
    main()