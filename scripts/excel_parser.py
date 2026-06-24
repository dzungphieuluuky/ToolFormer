"""
excel_parser.py
────────────────
Parse telecom function definitions from Excel (.xlsx) or JSON schema files.

Functions
─────────
  parse_telecom_functions(excel_path, output_path=None) → dict
    Parse an Excel workbook with function definitions into a library dict.

  load_function_library(library_path) → dict
    Load a pre-saved JSON function library.

  load_function_schema(schema_path) → dict
    Load a function_schema.json file (alias for load_function_library).
"""

from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _safe_json(value: Any, fallback: Any = None) -> Any:
    """Parse a cell that may contain JSON or a Python literal."""
    if not isinstance(value, str) or not value.strip():
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except Exception:
            return fallback


def _safe_list(value: Any) -> list[str]:
    """Parse a cell that may contain a list (JSON or comma-separated)."""
    result = _safe_json(value, fallback=None)
    if isinstance(result, list):
        return result
    if isinstance(value, str):
        return [v.strip() for v in value.replace("\n", ",").split(",") if v.strip()]
    return []


def parse_telecom_functions(
    excel_path: str, output_path: Optional[str] = None
) -> dict[str, dict[str, Any]]:
    """
    Parse an Excel workbook containing telecom function definitions.

    Expected columns:
      function_name, description, parameters,
      example_queries (optional), domain_info (optional),
      constraints (optional), tags (optional)

    Returns a dict of {function_name: schema_dict}.
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(path)
    required_cols = {"function_name", "description", "parameters"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing required columns: {missing}")

    library: dict[str, dict[str, Any]] = {}
    for _idx, row in df.iterrows():
        name = str(row["function_name"]).strip()
        if not name:
            continue

        params = _safe_json(row.get("parameters", "{}"), fallback={})
        examples = _safe_list(row.get("example_queries", "[]"))
        domain = _safe_json(row.get("domain_info", "{}"), fallback={})
        constraints = _safe_json(row.get("constraints", "{}"), fallback={})
        tags = _safe_list(row.get("tags", "[]"))

        func_schema: dict[str, Any] = {
            "name": name,
            "description": str(row["description"]).strip(),
            "parameters": params,
            "examples": examples,
            "domain": domain,
            "constraints": constraints,
            "tags": tags,
        }
        library[name] = func_schema

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(library, fh, indent=2, ensure_ascii=False)
        print(f"[excel_parser] Saved {len(library)} functions → {out}")

    return library


def load_function_library(library_path: str) -> dict[str, Any]:
    """Load a pre-saved JSON function library."""
    with open(library_path, "r", encoding="utf-8") as fh:
        return dict(json.load(fh))


def load_function_schema(schema_path: str) -> dict[str, Any]:
    """Load a function_schema.json file (alias for load_function_library)."""
    return load_function_library(schema_path)
