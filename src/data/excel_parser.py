"""
excel_parser.py
---------------
Reads telecom_functions.xlsx and converts each row into a structured
JSON schema entry.  Also supports loading directly from function_schema.json
if the Excel file is not yet available.
"""

import json
import ast
from pathlib import Path
from typing import Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_json(value: str, fallback=None):
    """Parse a JSON string; return fallback on any error."""
    if not isinstance(value, str) or not value.strip():
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except Exception:
            return fallback


def _safe_list(value: str) -> list:
    """Parse a Python list string or JSON array."""
    result = _safe_json(value, fallback=None)
    if isinstance(result, list):
        return result
    if isinstance(value, str):
        # treat as newline/comma-separated plain text
        return [v.strip() for v in value.replace("\n", ",").split(",") if v.strip()]
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_telecom_functions(
    excel_path: str,
    output_path: Optional[str] = "data/processed/function_library.json",
) -> dict:
    """
    Parse *telecom_functions.xlsx* → function_library.json.

    Expected columns
    ────────────────
    function_name   : str
    description     : str
    parameters      : JSON string  →  {"param_name": {"type", "required", ...}}
    example_queries : JSON/list    →  ["query1", "query2", ...]
    domain_info     : JSON string  →  {"domain": ..., "category": ...}
    constraints     : JSON string  →  {"param_name": {"min": ..., "max": ...}}   [optional]
    tags            : list / str   →  ["tag1", "tag2"]                           [optional]

    Returns
    ───────
    dict  –  {function_name: schema_dict}
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(path)
    required_cols = {"function_name", "description", "parameters"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Excel is missing required columns: {missing}")

    library: dict = {}

    for idx, row in df.iterrows():
        name = str(row["function_name"]).strip()
        if not name:
            continue

        params = _safe_json(row.get("parameters", "{}"), fallback={})
        examples = _safe_list(row.get("example_queries", "[]"))
        domain = _safe_json(row.get("domain_info", "{}"), fallback={})
        constraints = _safe_json(row.get("constraints", "{}"), fallback={})
        tags = _safe_list(row.get("tags", "[]"))

        func_schema = {
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


def load_function_library(library_path: str) -> dict:
    """Load a previously parsed function_library.json."""
    with open(library_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_function_schema(schema_path: str) -> dict:
    """
    Load the raw function_schema.json provided by the mentor.
    This is the PRIMARY input for dataset generation when the Excel is unavailable.

    Expected format
    ───────────────
    {
      "function_name_1": {
          "name": "...",
          "description": "...",
          "parameters": { ... },
          "examples": [...],
          ...
      },
      ...
    }
    """
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    print(f"[excel_parser] Loaded function_schema.json  →  {len(schema)} functions")
    return schema