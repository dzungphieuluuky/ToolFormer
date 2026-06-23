#!/usr/bin/env python3
"""
clean_dataset.py — Repair & normalize toolformer dataset JSONL files.

Usage:
    python clean_dataset.py \
        --data-dir data/processed \
        --output-dir data/processed/clean

Operations:
    1. Drop malformed JSON lines
    2. Drop rows missing required fields
    3. Normalize ground_truth (fix null calls, missing keys)
    4. Drop samples with undefined function calls
    5. Deduplicate by id
    6. Fix date formats (coerce to YYYY-MM-DD)
    7. Clean retrieved_argument_values (sort by score, drop empty entries)
    8. Split into train / test output files
    9. Save cleaning report
"""

import calendar
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

DATE_SLASH_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
DATE_DOT_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})$")
DATE_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}$")

VALID_WORKFLOWS = {"single_call", "parallel", "sequential", "abstention"}
VALID_SPLITS = {"train", "test"}

REQUIRED_FIELDS = [
    "id",
    "query",
    "workflow_type",
    "function_name",
    "ground_truth",
    "retrieved_functions",
    "split",
    "retrieved_argument_values",
]


# ── Helpers ─────────────────────────────────────────────────────────────


def _normalize_ground_truth(gt: Any) -> dict:
    """Mirror the notebook's normalize_ground_truth logic."""
    if not isinstance(gt, dict):
        return {"calls": [], "workflow": "single_call", "reasoning": ""}
    calls = gt.get("calls")
    if not isinstance(calls, list):
        calls = []
    cleaned = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        func = c.get("function")
        if not isinstance(func, str) or not func.strip():
            continue
        args = c.get("arguments")
        if not isinstance(args, dict):
            args = {}
        cleaned.append({"function": func, "arguments": args})
    return {
        "calls": cleaned,
        "workflow": gt.get("workflow", "single_call") or "single_call",
        "reasoning": gt.get("reasoning", "") or "",
    }


def _fix_date(val: str) -> str:
    """Coerce common date formats to YYYY-MM-DD."""
    m = DATE_TIMESTAMP_RE.match(val)
    if m:
        return m.group(1)
    m = DATE_SLASH_RE.match(val)
    if m:
        d, mth, y = m.groups()
        return f"{y}-{int(mth):02d}-{int(d):02d}"
    m = DATE_DOT_RE.match(val)
    if m:
        y, mth, d = m.groups()
        return f"{y}-{mth}-{d}"
    return val


def _fix_call_dates(call: dict) -> int:
    fixes = 0
    args = call.get("arguments")
    if not isinstance(args, dict):
        return 0
    for key in ("from_date", "to_date", "start_date", "end_date"):
        if key in args and isinstance(args[key], str):
            fixed = _fix_date(args[key])
            if fixed != args[key]:
                args[key] = fixed
                fixes += 1
    return fixes


# ── Date extraction helpers (from notebook's smart_retriever.py) ────


def _last_day(year: int, month: int) -> str:
    """Last day of a given month."""
    last = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-{last:02d}"


def _quarter_range(q: int, year: int) -> tuple[str, str]:
    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return (f"{year}-{starts[q]}", f"{year}-{ends[q]}")


_DATE_PATTERNS = [
    # "tháng 6/2026", "thang 6 nam 2026"
    (
        r"(?:thang|tháng)\s*(\d{1,2})\s*[/\-\.]\s*(\d{4})",
        lambda m: (f"{m.group(2)}-{int(m.group(1)):02d}-01",
                   _last_day(int(m.group(2)), int(m.group(1)))),
    ),
    # "tháng 6 năm 2026"
    (
        r"(?:thang|tháng)\s*(\d{1,2})\s*(?:nam|năm)\s*(\d{4})",
        lambda m: (f"{m.group(2)}-{int(m.group(1)):02d}-01",
                   _last_day(int(m.group(2)), int(m.group(1)))),
    ),
    # "năm 2022", "nam 2022"
    (
        r"(?:toan\s*)?(?:nam|năm)\s*(\d{4})",
        lambda m: (f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"),
    ),
    # "Q1/2025", "quý 2 năm 2025"
    (
        r"(?:q|quy|quý)\s*(\d)\s*[/\-]?\s*(\d{4})",
        lambda m: _quarter_range(int(m.group(1)), int(m.group(2))),
    ),
    # "06/2026" (MM/YYYY)
    (
        r"(\d{1,2})\s*/\s*(\d{4})",
        lambda m: (f"{m.group(2)}-{int(m.group(1)):02d}-01",
                   _last_day(int(m.group(2)), int(m.group(1)))),
    ),
    # "2022-01-01" to "2022-12-31" (explicit ISO range)
    (
        r"(\d{4}-\d{2}-\d{2})\s*(?:den|đến|to|\-)\s*(\d{4}-\d{2}-\d{2})",
        lambda m: (m.group(1), m.group(2)),
    ),
    # "từ ngày dd/mm/YYYY đến ngày dd/mm/YYYY"
    (
        r"(?:tu|từ)\s*(?:ngay|ngày)?\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*(?:den|đến)\s*(?:ngay|ngày)?\s*(\d{1,2})/(\d{1,2})/(\d{4})",
        lambda m: (f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}",
                   f"{m.group(6)}-{int(m.group(5)):02d}-{int(m.group(4)):02d}"),
    ),
    # "ngày dd/mm/YYYY" (single date)
    (
        r"(?:ngay|ngày)\s*(\d{1,2})/(\d{1,2})/(\d{4})",
        lambda m: (f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}",
                   f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
    ),
    # "dd/mm/YYYY" (bare date)
    (
        r"(\d{1,2})/(\d{1,2})/(\d{4})",
        lambda m: (f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}",
                   f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
    ),
]


def _extract_dates(query: str) -> dict[str, str]:
    """Extract from_date and to_date from Vietnamese query.

    Skips matches preceded by week/ordinal indicators (e.g. "tuần 22/2026").
    """
    query_lower = query.lower()
    for pattern, extractor in _DATE_PATTERNS:
        for match in re.finditer(pattern, query_lower):
            start = match.start()
            prefix = query_lower[max(0, start - 8):start].strip()
            # Skip patterns like "tuần 22/2026", "thứ 24/2026", "week 22/2026"
            if re.search(r"(?:tuần|tuan|thứ|thu|week)\s*$", prefix):
                continue
            try:
                from_date, to_date = extractor(match)
                return {"from_date": from_date, "to_date": to_date}
            except (ValueError, IndexError, calendar.IllegalMonthError):
                continue
    return {}


def _extract_data_level(query: str) -> str | None:
    """Extract aggregation level from Vietnamese query."""
    _DATA_LEVEL_PATTERNS = [
        (r"(?:theo|tong\s*hp)\s*(?:ngay|ngày|daily)", "day"),
        (r"(?:theo|tong\s*hp)\s*(?:tuan|tuần|weekly)", "week"),
        (r"(?:theo|tong\s*hp)\s*(?:thang|tháng|monthly)", "month"),
        (r"(?:theo|tong\s*hp)\s*(?:nam|năm|yearly)", "year"),
        (r"(?:hang|hàng)\s*(?:ngay|ngày)", "day"),
        (r"(?:hang|hàng)\s*(?:tuan|tuần)", "week"),
        (r"(?:hang|hàng)\s*(?:thang|tháng)", "month"),
        (r"(?:thang|tháng)\s*\d{1,2}", "month"),
    ]
    for pattern, level in _DATA_LEVEL_PATTERNS:
        if re.search(pattern, query.lower()):
            return level
    return None


def _extract_tech_type(query: str) -> str | None:
    """Extract 2G/3G/4G/5G tech from query."""
    for pat in [r"\b5G\b", r"\b4G\b", r"\b3G\b", r"\b2G\b"]:
        m = re.search(pat, query)
        if m:
            return m.group(0)
    return None


def _extract_best_value(retrieved_vals: dict, param_name: str) -> Any:
    """Return the code of the highest-scored entry for param_name."""
    entries = retrieved_vals.get(param_name)
    if isinstance(entries, list) and entries:
        valid = [e for e in entries if isinstance(e, dict) and e.get("code")]
        if valid:
            best = max(
                valid,
                key=lambda e: e.get("score", 0)
                if isinstance(e.get("score"), (int, float))
                else 0,
            )
            return best["code"]
    return None


# ── Single-call call filler ─────────────────────────────────────────────

# Parameter name aliases: map function-param-name → retrieved_argument_values key
_PARAM_ALIASES = {
    "object_code": "location_code",
    "station_code": "location_code",
}

# Tech/cell type parameter names that map to _extract_tech_type
_TECH_PARAMS = {"tech_type", "cell_type"}

# Date parameter names
_DATE_PARAMS = {"from_date", "to_date", "date", "start_date", "end_date"}

# Functions whose data_level enum excludes quarter/year
_DATA_LEVEL_RESTRICTED = {"RADIO_TRAFFIC", "RADIO_KPI"}
_ALLOWED_RESTRICTED_LEVELS = {"day", "week", "month"}


def _build_args(
    func_name: str,
    params: dict,
    retrieved_vals: dict,
    query: str,
) -> dict[str, Any]:
    """Build tool-call arguments dict for a single_call sample."""
    args: dict[str, Any] = {}
    dates = _extract_dates(query)
    data_level_from_query = _extract_data_level(query)
    tech_type = _extract_tech_type(query)

    for pname, pschema in params.items():
        required = pschema.get("required", False)
        raw_default = pschema.get("default")
        ptype = pschema.get("type", "string")
        enum_vals = pschema.get("enum")

        value = None

        # 1. Try direct lookup in retrieved_argument_values
        best = _extract_best_value(retrieved_vals, pname)
        if best is not None:
            value = best

        # 2. Try aliased lookup
        if value is None and pname in _PARAM_ALIASES:
            best = _extract_best_value(retrieved_vals, _PARAM_ALIASES[pname])
            if best is not None:
                value = best

        # 3. Tech/cell type from query
        if value is None and pname in _TECH_PARAMS and tech_type:
            value = tech_type

        # 4. Data level: try retrieved → query extract → default
        if value is None and pname == "data_level":
            if data_level_from_query:
                value = data_level_from_query
            # If function restricts data_level, clamp to allowed set
            if func_name in _DATA_LEVEL_RESTRICTED and value in _DATA_LEVEL_RESTRICTED:
                # Not in allowed set → try next best
                pass  # handled below

        # 5. Date params from query
        if value is None and pname in _DATE_PARAMS:
            if pname in dates:
                value = dates[pname]
            elif pname == "from_date" and "from_date" in dates:
                value = dates["from_date"]
            elif pname == "to_date" and "to_date" in dates:
                value = dates["to_date"]
            elif pname == "start_date" and "from_date" in dates:
                value = dates["from_date"]
            elif pname == "end_date" and "to_date" in dates:
                value = dates["to_date"]
            # If one date and we have both from/to, use 'from' for 'date'
            elif pname == "date" and "from_date" in dates:
                value = dates["from_date"]

        # 6. Fall back to schema default
        if value is None:
            if raw_default is not None:
                # Handle special sentinel defaults
                if isinstance(raw_default, str) and raw_default in (
                    "<previous day>", "<current date>", "<previous day>",
                ):
                    value = "2026-06-15"
                else:
                    value = raw_default
            # Parameters with no explicit default but required
            elif pname == "station_code":
                value = "all"
            elif pname == "kpi_code":
                value = "call_setup_success_rate"
            elif pname == "date":
                value = "2026-06-15"
            elif pname == "start_date":
                value = "2026-06-01"
            elif pname == "end_date":
                value = "2026-06-30"

        # 7. Validate against enum if defined
        if value is not None and enum_vals:
            # Clamp data_level for restricted radio functions
            if pname == "data_level" and func_name in _DATA_LEVEL_RESTRICTED:
                if value not in _ALLOWED_RESTRICTED_LEVELS:
                    value = "month"
            # General enum validation
            if value not in enum_vals:
                # Try schema default first
                fallback = raw_default if raw_default is not None else None
                if fallback is not None and fallback in enum_vals:
                    value = fallback
                elif enum_vals:
                    # Pick first non-empty enum value
                    non_empty = [e for e in enum_vals if e != '""']
                    value = non_empty[0] if non_empty else enum_vals[0]

        # 8. Convert numeric types
        if value is not None and ptype in ("int", "integer"):
            if isinstance(value, str):
                if value.isdigit():
                    value = int(value)
                else:
                    value = int(raw_default) if raw_default and str(raw_default).isdigit() else 5

        # Skip empty/sentinel string defaults for OPTIONAL params only
        if value is not None:
            if isinstance(value, str) and (value.strip() == "" or value == '""'):
                if not required:
                    continue
            args[pname] = value

    return args


def step_fill_single_call_calls(
    samples: list[dict],
    function_library: dict,
) -> tuple[list[dict], dict[str, int]]:
    """Fill empty `calls` for single_call samples by constructing valid calls.

    Returns (samples, fix_counts_by_function) — does NOT drop any samples.
    """
    fix_counts: dict[str, int] = {}
    total_filled = 0

    for s in samples:
        if s.get("workflow_type") != "single_call":
            continue
        gt = s.get("ground_truth")
        if not isinstance(gt, dict):
            continue
        calls = gt.get("calls")
        if not isinstance(calls, list) or len(calls) > 0:
            continue

        func_name = s.get("function_name")
        if not func_name or func_name not in function_library:
            continue

        schema = function_library[func_name]
        params = schema.get("parameters", {})
        retrieved_vals = s.get("retrieved_argument_values", {})
        query = s.get("query", "")

        args = _build_args(func_name, params, retrieved_vals, query)
        gt["calls"] = [{"function": func_name, "arguments": args}]
        total_filled += 1
        fix_counts[func_name] = fix_counts.get(func_name, 0) + 1

    return samples, fix_counts


# ── Cleaning pipeline steps ─────────────────────────────────────────────


def step_drop_invalid_json(lines: list[str]) -> tuple[list[dict], list[int]]:
    valid = []
    dropped = []
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            valid.append(json.loads(line))
        except json.JSONDecodeError:
            dropped.append(lineno)
    return valid, dropped


def step_drop_missing_required(samples: list[dict]) -> tuple[list[dict], list[int]]:
    kept = []
    dropped = []
    for idx, s in enumerate(samples):
        ok = True
        for field in REQUIRED_FIELDS:
            if field not in s or s[field] is None:
                ok = False
                break
        if ok:
            kept.append(s)
        else:
            dropped.append(idx)
    return kept, dropped


def step_normalize_ground_truth(samples: list[dict]) -> tuple[list[dict], int]:
    fixed_count = 0
    for s in samples:
        gt = s.get("ground_truth")
        normalized = _normalize_ground_truth(gt)
        if normalized != gt:
            fixed_count += 1
            s["ground_truth"] = normalized
    return samples, fixed_count


def step_drop_undefined_functions(
    samples: list[dict], function_library: dict
) -> tuple[list[dict], list[str]]:
    kept = []
    dropped_ids = []
    for s in samples:
        gt = s.get("ground_truth")
        if not isinstance(gt, dict):
            dropped_ids.append(s.get("id", "<?>"))
            continue
        calls = gt.get("calls", [])
        if not isinstance(calls, list):
            dropped_ids.append(s.get("id", "<?>"))
            continue
        all_defined = True
        for c in calls:
            if not isinstance(c, dict):
                all_defined = False
                break
            fname = c.get("function")
            if not isinstance(fname, str) or fname not in function_library:
                all_defined = False
                break
        if all_defined:
            kept.append(s)
        else:
            dropped_ids.append(s.get("id", "<?>"))
    return kept, dropped_ids


def step_deduplicate(samples: list[dict]) -> tuple[list[dict], list[str]]:
    seen: set[str] = set()
    kept = []
    removed_ids = []
    for s in samples:
        sid = s.get("id")
        if sid and sid in seen:
            removed_ids.append(sid)
        else:
            if sid:
                seen.add(sid)
            kept.append(s)
    return kept, removed_ids


def step_fix_dates(samples: list[dict]) -> tuple[list[dict], int]:
    total_fixes = 0
    for s in samples:
        gt = s.get("ground_truth")
        if not isinstance(gt, dict):
            continue
        calls = gt.get("calls", [])
        if not isinstance(calls, list):
            continue
        for c in calls:
            if isinstance(c, dict):
                total_fixes += _fix_call_dates(c)
    return samples, total_fixes


def step_clean_retrieved_argument_values(samples: list[dict]) -> tuple[list[dict], int]:
    total_fixed = 0
    for s in samples:
        rav = s.get("retrieved_argument_values")
        if not isinstance(rav, dict):
            continue
        cleaned: dict[str, list[dict]] = {}
        for pname, entries in rav.items():
            if not isinstance(entries, list):
                continue
            valid = [e for e in entries if isinstance(e, dict) and e.get("code") and e.get("label")]
            valid.sort(key=lambda e: e.get("score", 0) if isinstance(e.get("score"), (int, float)) else 0, reverse=True)
            if len(valid) != len(entries):
                total_fixed += 1
            cleaned[pname] = valid
        s["retrieved_argument_values"] = cleaned
    return samples, total_fixed


def step_validate_workflow_type(samples: list[dict]) -> tuple[list[dict], list[str]]:
    kept = []
    dropped_ids = []
    for s in samples:
        wt = s.get("workflow_type")
        if wt in VALID_WORKFLOWS:
            kept.append(s)
        else:
            dropped_ids.append(s.get("id", "<?>"))
    return kept, dropped_ids


def step_validate_split(samples: list[dict]) -> tuple[list[dict], list[str]]:
    kept = []
    dropped_ids = []
    for s in samples:
        sp = s.get("split")
        if sp in VALID_SPLITS:
            kept.append(s)
        else:
            dropped_ids.append(s.get("id", "<?>"))
    return kept, dropped_ids


def step_strip_empty_enum_values(
    samples: list[dict],
    function_library: dict,
) -> tuple[list[dict], int]:
    """Remove empty string values from optional enum fields in calls.

    Required params with empty-string defaults are kept (the schema says
    they must be present, even if the value is a sentinel empty string).
    """
    fixed = 0
    for s in samples:
        gt = s.get("ground_truth")
        if not isinstance(gt, dict):
            continue
        calls = gt.get("calls", [])
        if not isinstance(calls, list):
            continue
        for c in calls:
            if not isinstance(c, dict):
                continue
            fn = c.get("function", "")
            schema = function_library.get(fn, {})
            params = schema.get("parameters", {})
            args = c.get("arguments")
            if not isinstance(args, dict):
                continue
            to_pop = []
            for k, v in args.items():
                if isinstance(v, str) and (v.strip() == "" or v == '""'):
                    pinfo = params.get(k, {})
                    if not pinfo.get("required"):
                        to_pop.append(k)
            for k in to_pop:
                args.pop(k)
                fixed += 1
    return samples, fixed


def step_fill_defaults(samples: list[dict]) -> tuple[list[dict], int]:
    filled = 0
    for s in samples:
        if "ground_truth" not in s:
            s["ground_truth"] = {"calls": [], "workflow": "single_call", "reasoning": ""}
            filled += 1
        if "retrieved_functions" not in s or not isinstance(s.get("retrieved_functions"), list):
            s["retrieved_functions"] = []
            filled += 1
        if "retrieved_argument_values" not in s or not isinstance(s.get("retrieved_argument_values"), dict):
            s["retrieved_argument_values"] = {}
            filled += 1
        if "workflow_type" not in s:
            s["workflow_type"] = "single_call"
            filled += 1
        if "split" not in s:
            s["split"] = "train"
            filled += 1
    return samples, filled


# ── Main pipeline ───────────────────────────────────────────────────────


def run_cleaning_pipeline(
    data_dir: str,
    output_dir: str,
) -> dict:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load files
    train_path = data_dir / "train_dataset.jsonl"
    test_path = data_dir / "test_dataset.jsonl"
    library_path = data_dir / "function_library.json"

    for p in [train_path, test_path, library_path]:
        assert p.exists(), f"Required file not found: {p}"

    with open(library_path, "r", encoding="utf-8") as f:
        function_library = json.load(f)

    report: dict[str, Any] = {
        "input": {},
        "steps": {},
        "output": {},
    }

    for split_name, input_path in [("train", train_path), ("test", test_path)]:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        n_input = len([l for l in raw_lines if l.strip()])
        report["input"][split_name] = n_input

        step_log: dict[str, Any] = {}

        # 1. Drop invalid JSON
        samples, dropped_json = step_drop_invalid_json(raw_lines)
        step_log["dropped_invalid_json"] = len(dropped_json)
        step_log["dropped_invalid_json_lines"] = dropped_json[:20]

        # 2. Fill defaults for missing optional fields
        samples, filled = step_fill_defaults(samples)
        step_log["filled_defaults"] = filled

        # 3. Drop missing required fields
        samples, dropped_req = step_drop_missing_required(samples)
        step_log["dropped_missing_required"] = len(dropped_req)

        # 4. Normalize ground_truth
        samples, norm_fixes = step_normalize_ground_truth(samples)
        step_log["normalized_ground_truth"] = norm_fixes

        # 5. Validate workflow_type
        samples, dropped_wt = step_validate_workflow_type(samples)
        step_log["dropped_invalid_workflow"] = len(dropped_wt)

        # 6. Validate split
        samples, dropped_sp = step_validate_split(samples)
        step_log["dropped_invalid_split"] = len(dropped_sp)

        # 7. Drop undefined functions
        samples, dropped_undef = step_drop_undefined_functions(samples, function_library)
        step_log["dropped_undefined_functions"] = len(dropped_undef)
        step_log["dropped_undefined_function_ids"] = dropped_undef[:20]

        # 8. Deduplicate
        samples, removed_dups = step_deduplicate(samples)
        step_log["removed_duplicates"] = len(removed_dups)
        step_log["removed_duplicate_ids"] = removed_dups[:20]

        # 9. Fix dates
        samples, date_fixes = step_fix_dates(samples)
        step_log["fixed_dates"] = date_fixes

        # 10. Clean retrieved_argument_values
        samples, rav_fixes = step_clean_retrieved_argument_values(samples)
        step_log["cleaned_retrieved_argument_values"] = rav_fixes

        # 11. Clamp data_level for restricted functions (RADIO_TRAFFIC, RADIO_KPI)
        _RESTRICTED_DL_FNS = {"RADIO_TRAFFIC", "RADIO_KPI"}
        _RESTRICTED_DL_ALLOWED = {"day", "week", "month"}
        dl_fixes = 0
        for s in samples:
            gt = s.get("ground_truth")
            if not isinstance(gt, dict):
                continue
            for c in gt.get("calls", []):
                if not isinstance(c, dict):
                    continue
                if c.get("function") not in _RESTRICTED_DL_FNS:
                    continue
                args = c.get("arguments")
                if not isinstance(args, dict):
                    continue
                dl = args.get("data_level")
                if dl is not None and dl not in _RESTRICTED_DL_ALLOWED:
                    args["data_level"] = "month"
                    dl_fixes += 1
        step_log["clamped_data_level"] = dl_fixes

        # 12. Strip empty string enum values from arguments
        samples, empty_fixes = step_strip_empty_enum_values(samples, function_library)
        step_log["stripped_empty_enum_values"] = empty_fixes

        # 12. Fill empty calls for single_call samples (train only)
        if split_name == "train":
            samples, filled_calls = step_fill_single_call_calls(
                samples, function_library
            )
            step_log["filled_single_call_calls"] = sum(filled_calls.values())
            step_log["filled_single_call_calls_by_function"] = dict(
                sorted(filled_calls.items(), key=lambda x: -x[1])
            )
        else:
            step_log["filled_single_call_calls"] = 0
            step_log["filled_single_call_calls_by_function"] = {}

        report["steps"][split_name] = step_log
        report["output"][split_name] = len(samples)

        # Write output
        output_path = output_dir / f"{split_name}_dataset_clean.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(samples)} samples → {output_path}")

    # Compute totals
    total_in = report["input"].get("train", 0) + report["input"].get("test", 0)
    total_out = report["output"].get("train", 0) + report["output"].get("test", 0)
    total_dropped = total_in - total_out

    report["summary"] = {
        "total_input": total_in,
        "total_output": total_out,
        "total_dropped": total_dropped,
        "dropped_rate": f"{total_dropped / max(total_in, 1) * 100:.2f}%",
    }

    # Save report
    report_path = output_dir / "clean_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved report → {report_path}")

    return report


def print_summary(report: dict) -> None:
    print("\n" + "=" * 60)
    print("  DATASET CLEANING REPORT")
    print("=" * 60)
    for split in ("train", "test"):
        steps = report["steps"].get(split, {})
        print(f"\n  [{split.upper()}]")
        print(f"    Input:  {report['input'].get(split, 0)} samples")
        print(f"    Output: {report['output'].get(split, 0)} samples")
        drops = []
        for key in (
            "dropped_invalid_json",
            "dropped_missing_required",
            "dropped_invalid_workflow",
            "dropped_invalid_split",
            "dropped_undefined_functions",
            "removed_duplicates",
        ):
            v = steps.get(key, 0)
            if v:
                drops.append(f"{key}={v}")
        if drops:
            print(f"    Dropped: {', '.join(drops)}")
        fixes = []
        for key in ("normalized_ground_truth", "filled_defaults", "fixed_dates", "cleaned_retrieved_argument_values"):
            v = steps.get(key, 0)
            if v:
                fixes.append(f"{key}={v}")
        filled_calls = steps.get("filled_single_call_calls", 0)
        if filled_calls:
            fixes.append(f"filled_single_calls={filled_calls}")
        if fixes:
            print(f"    Fixed:   {', '.join(fixes)}")
            filled_by_fn = steps.get("filled_single_call_calls_by_function", {})
            if filled_by_fn:
                print(f"    By function: {dict(list(filled_by_fn.items())[:10])}")
    s = report.get("summary", {})
    print(f"\n  TOTAL: {s.get('total_input', 0)} → {s.get('total_output', 0)} "
          f"(dropped {s.get('total_dropped', 0)}, {s.get('dropped_rate', 'N/A')})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean toolformer dataset")
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Path to the data/processed directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/clean",
        help="Output directory for cleaned files",
    )
    args = parser.parse_args()

    report = run_cleaning_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print_summary(report)
