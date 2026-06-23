#!/usr/bin/env python3
"""
nb_to_md.py — Convert Qwen3 GRPO ToolCalling notebook to readable markdown.

Usage:
    python scripts/nb_to_md.py \
        --notebook Qwen3_(4B)_GRPO_ToolCalling.ipynb \
        --output docs/implementation.md

Extracts code cells with their preceding markdown context into a
structured markdown document suitable for review, documentation,
or static rendering.
"""

import json
import re
import sys
from pathlib import Path
from typing import TextIO


def extract_cell_number(cell: dict, nb: dict) -> int:
    """Return the execution count for a code cell, or 0 for markdown."""
    ec = cell.get("execution_count")
    if isinstance(ec, int):
        return ec
    return 0


def clean_source(source_lines: list[str]) -> str:
    """Join notebook source lines, adding trailing newline if missing."""
    result = []
    for line in source_lines:
        if line and not line.endswith("\n"):
            line += "\n"
        result.append(line)
    return "".join(result)


def tag_for_section(source: str) -> str | None:
    """Detect section header from `# =====` comment lines in code."""
    m = re.search(r"# ={5,}\s*(.+?)\s*={5,}", source)
    if m:
        tag = m.group(1).strip()
        # Remove trailing `py` if present (from `.py` file header)
        tag = re.sub(r"\s*py\s*$", "", tag)
        return tag
    return None


def should_skip_cell(source: str) -> bool:
    """Skip boilerplate cells (pip installs, dataset downloads, etc.)."""
    lines = source.strip().splitlines()
    if not lines:
        return True
    first = lines[0].strip()
    # Skip pure pip installs
    if first.startswith("!pip install") or first.startswith("%pip install"):
        return True
    # Skip purely empty or whitespace
    if all(not l.strip() for l in lines):
        return True
    # Skip git clone
    if "git clone" in first:
        return True
    return False


def format_output(output: dict) -> str | None:
    """Format a single cell output to markdown."""
    otype = output.get("output_type")
    text = ""

    if otype == "stream":
        parts = output.get("text", [])
        text = "".join(parts)
    elif otype in ("execute_result", "display_data"):
        data = output.get("data", {})
        if "text/plain" in data:
            text = "".join(data["text/plain"])
        elif "text/html" in data:
            return None  # skip HTML outputs
    elif otype == "error":
        ename = output.get("ename", "Error")
        evalue = output.get("evalue", "")
        trace = "".join(output.get("traceback", []))
        text = f"{ename}: {evalue}\n"
        # Clean ANSI codes
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        text = re.sub(r"\x1b\[.*?m", "", text)

    if not text or not text.strip():
        return None

    # Clean ANSI escapes from text too
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = text.strip()

    if len(text) > 5000:
        text = text[:5000] + "\n... (truncated)"

    return text


def convert_nb_to_md(notebook_path: str, output_path: str, include_outputs: bool = False) -> dict:
    """Convert a Jupyter notebook to markdown.

    Returns stats dict.
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")

    lines: list[str] = []
    stats = {"markdown": 0, "code": 0, "skipped": 0, "outputs": 0}

    # --- Title ---
    nb_meta = nb.get("metadata", {})
    title = ""
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            src = clean_source(cell["source"]).strip()
            if src.startswith("# "):
                title = src.split("\n")[0].lstrip("# ").strip()
                break

    if title:
        lines.append(f"# {title}\n")
    else:
        lines.append(f"# Notebook: {Path(notebook_path).name}\n")

    lines.append(f"\n> Auto-generated from `{Path(notebook_path).name}`  \n")
    lines.append(f"> Total cells: {len(cells)}  \n")
    lines.append(f"> Kernel: {lang}\n")
    lines.append("---\n")

    for ci, cell in enumerate(cells):
        ctype = cell.get("cell_type")

        if ctype == "markdown":
            src = clean_source(cell["source"]).strip()
            if not src:
                continue
            lines.append(f"\n{src}\n")
            stats["markdown"] += 1

        elif ctype == "code":
            src_code = clean_source(cell["source"]).strip()
            if should_skip_cell(src_code):
                stats["skipped"] += 1
                continue

            # Section detection
            section_tag = tag_for_section(src_code)
            if section_tag:
                lines.append(f"\n## `{section_tag}`\n")

            # Execution count
            exec_count = extract_cell_number(cell, nb)
            label = f"Cell [{exec_count}]" if exec_count else f"Cell #{ci}"

            # Add annotation for .py file imports
            first_line = src_code.split("\n")[0] if src_code else ""
            annotation = ""
            if first_line.startswith("# =") and "py" in first_line:
                annotation = f"> Source file: `{first_line.replace('#', '').replace('=', '').strip()}`\n\n"

            lines.append(f"\n{annotation}```{lang} {label}\n{src_code}\n```\n")

            # Optional outputs
            if include_outputs:
                outputs = cell.get("outputs", [])
                for o in outputs:
                    formatted = format_output(o)
                    if formatted and formatted.strip():
                        lines.append(f"\n**Output:**\n```\n{formatted}\n```\n")
                        stats["outputs"] += 1

            stats["code"] += 1

    # Write
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.writelines(lines)

    stats["total"] = len(cells)
    return stats


def print_stats(stats: dict) -> None:
    print("\n=== Conversion Stats ===")
    print(f"  Total cells:  {stats['total']}")
    print(f"  Markdown:     {stats['markdown']}")
    print(f"  Code:         {stats['code']}")
    print(f"  Skipped:      {stats['skipped']}")
    print(f"  Outputs:      {stats['outputs']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Qwen3 GRPO ToolCalling notebook to markdown"
    )
    parser.add_argument(
        "--notebook",
        default="Qwen3_(4B)_GRPO_ToolCalling.ipynb",
        help="Path to the Jupyter notebook",
    )
    parser.add_argument(
        "--output",
        default="docs/implementation.md",
        help="Output markdown file path",
    )
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="Include cell outputs in the markdown",
    )
    args = parser.parse_args()

    stats = convert_nb_to_md(
        notebook_path=args.notebook,
        output_path=args.output,
        include_outputs=args.include_outputs,
    )
    print_stats(stats)
    print(f"\nWritten to: {args.output}")
