#!/usr/bin/env python3
"""
Recursively find all Python files in a given directory and consolidate them
into a single Markdown file with headings and code blocks.
"""

import os
import argparse
from pathlib import Path


def consolidate_py_files(
    input_dir: str, output_file: str, include_relative: bool = True
) -> None:
    """
    Walk through input_dir, collect all .py files, and write them into a single
    Markdown file.

    Args:
        input_dir: Root directory to search for Python files.
        output_file: Path to the output Markdown file.
        include_relative: If True, use paths relative to input_dir in headings.
    """
    input_path = Path(input_dir).resolve()
    py_files   = []

    # Recursively collect all .py files
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".py"):
                full_path = Path(root) / file
                py_files.append(full_path)

    if not py_files:
        print(f"No .py files found under '{input_path}'.")
        return

    # Sort for consistent order
    py_files.sort()

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"# Python Files Consolidated\n\n")
        out.write(f"**Source directory:** `{input_path}`\n\n")
        out.write(f"**Total files:** {len(py_files)}\n\n")
        out.write("---\n\n")

        for idx, file_path in enumerate(py_files, 1):
            # Compute heading path
            if include_relative:
                heading_path = file_path.relative_to(input_path)
            else:
                heading_path = file_path

            out.write(f"## {idx}. `{heading_path}`\n\n")

            try:
                # Read file content
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Fallback to latin-1 if UTF-8 fails (rare for .py)
                try:
                    content = file_path.read_text(encoding="latin-1")
                except Exception as e:
                    content = f"**Error reading file:** {e}\n"
            except Exception as e:
                content = f"**Error reading file:** {e}\n"

            out.write("```python\n")
            out.write(content)
            # Ensure trailing newline
            if not content.endswith("\n"):
                out.write("\n")
            out.write("```\n\n")

    print(f"Successfully consolidated {len(py_files)} files into '{output_file}'")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate all Python files from a folder into one Markdown file."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Root directory to search for .py files (default: current directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="consolidated_python_files.md",
        help="Output Markdown file path (default: consolidated_python_files.md).",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Use absolute paths in headings instead of relative.",
    )
    args = parser.parse_args()

    consolidate_py_files(
        input_dir       =args.input_dir,
        output_file     =args.output,
        include_relative=not args.absolute,
    )


if __name__ == "__main__":
    main()
