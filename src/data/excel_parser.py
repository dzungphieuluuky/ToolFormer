import ast
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def _safe_json(value: str, fallback=None):
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
    result = _safe_json(value, fallback=None)
    if isinstance(result, list):
        return result
    if isinstance(value, str):
        return [v.strip() for v in value.replace("\n", ",").split(",") if v.strip()]
    return []


def parse_telecom_functions(excel_path: str, output_path: Optional[str] = "data/processed/function_library.json") -> dict:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    df = pd.read_excel(path)
    required_cols = {"function_name", "description", "parameters"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing required columns: {missing}")
    library = {}
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
        print(f"[excel_parser] Saved {len(library)} functions -> {out}")
    return library


def load_function_library(library_path: str) -> dict:
    with open(library_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_function_schema(schema_path: str) -> dict:
    with open(schema_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
