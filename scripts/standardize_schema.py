import pandas as pd
import json
import re
from typing import Dict, Any, Optional


def parse_parameter(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single parameter line like:
    "- location_code [string][require]: Mô tả, Default: VNM"
    "- type_station [string][require]: Loại trạm ..., thuộc danh sách [4g_vtt, ...], Default: all"

    Returns a dict with:
        name, type, required (bool), description, default (or None), enum (or None)
    """
    # Remove leading dash and spaces
    line = line.lstrip("- ").strip()
    if not line:
        return None

    # Extract parameter name (first word)
    name_match = re.match(r"^(\w+)", line)
    if not name_match:
        return None
    name = name_match.group(1)

    # Extract type and required/optional inside brackets
    # Pattern: [string][require] or [string][optional]
    type_required_match = re.search(r"\[(\w+)\]\[(require|optional)\]", line)
    if not type_required_match:
        # Try alternative: [string][require] maybe without brackets
        # We'll assume default type string and required
        param_type = "string"
        required = True
    else:
        param_type = type_required_match.group(1)
        required = type_required_match.group(2) == "require"

    # Remove bracketed parts to get description
    desc = re.sub(r"\[(\w+)\]\[(require|optional)\]", "", line).strip()

    # Extract enum if present (thuộc danh sách [...])
    enum_match = re.search(r"thuộc danh sách \[(.*?)\]", desc)
    enum_values = None
    if enum_match:
        enum_str = enum_match.group(1)
        enum_values = [v.strip() for v in enum_str.split(",")]
        desc = desc.replace(enum_match.group(0), "").strip()

    # Extract default if present (Default: value)
    default_match = re.search(r"Default:\s*([^,]+?)(?:,|$)", desc)
    default_val = None
    if default_match:
        default_val = default_match.group(1).strip()
        # Remove the Default part from description
        desc = desc.replace(default_match.group(0), "").strip()

    # Clean up leftover commas and spaces
    desc = desc.rstrip(",").strip()

    return {
        "name": name,
        "type": param_type,
        "required": required,
        "description": desc,
        "default": default_val,
        "enum": enum_values,
    }


def parse_function(row: pd.Series) -> Dict[str, Any]:
    func_name = row["Tên hàm"].strip()
    description = row["Mô tả"].strip()

    params_text = row["Tham số"]
    param_lines = [p.strip() for p in params_text.split("\n") if p.strip()]
    params = {}
    for line in param_lines:
        if not line.startswith("-"):
            continue
        param_info = parse_parameter(line)
        if param_info is None:
            print(f"Warning: Could not parse parameter line: '{line}'")
            continue
        pname = param_info.pop("name")
        # Build the parameter entry
        entry = {
            "type": param_info["type"],
            "required": param_info["required"],
            "description": param_info["description"],
        }
        if param_info.get("default") is not None:
            entry["default"] = param_info["default"]
        if param_info.get("enum"):
            entry["enum"] = param_info["enum"]
        params[pname] = entry

    # Examples: split by newline
    examples_text = row.get("Ví dụ câu truy vấn", "")
    examples = [e.strip() for e in examples_text.split("\n") if e.strip()]

    # Domain and tags (customizable)
    domain = {"category": "telecom_operations"}
    tags = []
    keywords = [
        "KPI",
        "alarm",
        "traffic",
        "coverage",
        "subscriber",
        "performance",
        "configuration",
        "topology",
    ]
    for kw in keywords:
        if kw.lower() in description.lower() or kw.lower() in func_name.lower():
            tags.append(kw.lower())
    if not tags:
        tags.append("general")

    return {
        "name": func_name,
        "description": description,
        "parameters": params,
        "examples": examples,
        "domain": domain,
        "constraints": {},
        "tags": tags,
    }


def convert_csv_to_schema(csv_path: str, output_json: str):
    df = pd.read_csv(csv_path, encoding="utf-8")
    schema = {}
    for _, row in df.iterrows():
        entry = parse_function(row)
        schema[entry["name"]] = entry
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"✅ Schema written to {output_json} with {len(schema)} functions.")


if __name__ == "__main__":
    convert_csv_to_schema(
        "data/raw/function.csv", "data/processed/function_schema.json"
    )
