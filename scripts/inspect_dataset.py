#!/usr/bin/env python
"""
inspect_dataset.py
───────────────────
Comprehensive Exploratory Data Analysis (EDA) for the telecom dataset.

Generates both CLI output and a markdown report file covering:
  - Dataset overview
  - Workflow distribution
  - Function name distribution (primary + call functions)
  - Schema coverage
  - Argument parameter profiling
  - Query characteristics
  - Data quality metrics
  - Clean vs original comparison
  - Retrieved functions & argument values
  - Cross-split analysis
  - Per-function detailed analysis (with library metadata)

Usage:
    python scripts/inspect_dataset.py \
        --train data/processed/train_dataset_uncleaned.jsonl \
        --test data/processed/test_dataset_uncleaned.jsonl \
        --train-schema data/processed/function_schema_train.json \
        --test-schema data/processed/function_schema_test.json \
        --library data/processed/function_library.json \
        [--train-clean data/processed/train_dataset_clean.jsonl] \
        [--test-clean data/processed/test_dataset_clean.jsonl] \
        [--report reports/dataset_report.md]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Helpers ────────────────────────────────────────────────────────────────

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_BLUE = "\033[94m"
ANSI_CYAN = "\033[96m"


def no_color(text: str) -> str:
    return text


def colorize(use_color: bool):
    if use_color:
        return lambda text: text
    return no_color


def load_jsonl(path: str) -> list[dict]:
    samples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * n / total:.1f}%"


def fmt_ratio(n: int, total: int) -> str:
    return f"{n}/{total} ({fmt_pct(n, total)})"


def stat_str(values: list[int | float]) -> dict:
    if not values:
        return {"min": "-", "max": "-", "mean": "-", "median": "-", "sum": "-"}
    s = sorted(values)
    n = len(s)
    return {
        "min": s[0],
        "max": s[-1],
        "mean": round(sum(s) / n, 2),
        "median": s[n // 2],
        "sum": sum(s),
    }


def heading(text: str, level: int = 2) -> str:
    return f"{'#' * level} {text}\n"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        for row in rows
    )
    return f"{hdr}\n{sep}\n{body}\n"


def ascii_bar(pct: float, width: int = 20) -> str:
    """Render a simple ASCII horizontal bar."""
    filled = int(round(pct / 100.0 * width))
    empty = width - filled
    return "[" + "#" * filled + "." * empty + "]"


def dotted_line(char: str = "─", width: int = 70) -> str:
    return char * width


# ── Analyzer ───────────────────────────────────────────────────────────────


class DatasetAnalyzer:
    def __init__(self, samples: list[dict], label: str):
        self.samples = samples
        self.label   = label
        self.n       = len(samples)

    # ── Top-level analysis ────────────────────────────────────────────────

    def top_level_keys(self) -> Counter:
        c = Counter()
        for s in self.samples:
            for k in s:
                c[k] += 1
        return c

    def workflow_distribution(self) -> Counter:
        return Counter(s.get("workflow_type", "unknown") for s in self.samples)

    def function_name_distribution(self) -> Counter:
        return Counter(s.get("function_name", "unknown") for s in self.samples)

    def ground_truth_keys(self) -> Counter:
        c = Counter()
        for s in self.samples:
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                for k in gt:
                    c[k] += 1
        return c

    def query_lengths(self) -> list[int]:
        return [len(s.get("query", "")) for s in self.samples]

    def argument_counts(self) -> list[int]:
        counts = []
        for s in self.samples:
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                args = gt.get("arguments", {})
                if isinstance(args, dict):
                    counts.append(len(args))
        return counts

    def call_counts(self) -> list[int]:
        counts = []
        for s in self.samples:
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                calls = gt.get("calls", [])
                if isinstance(calls, list):
                    counts.append(len(calls))
        return counts

    def retrieved_functions_counts(self) -> list[int]:
        return [len(s.get("retrieved_functions", [])) for s in self.samples]

    def argument_value_presence(self) -> tuple[int, int]:
        present = sum(
            1 for s in self.samples if s.get("retrieved_argument_values")
        )
        return present, self.n

    def primary_vs_gt_function_mismatch(self) -> list[str]:
        mismatches = []
        for s in self.samples:
            top_fn = s.get("function_name")
            gt     = s.get("ground_truth", {})
            if isinstance(gt, dict):
                gt_fn = gt.get("function")
                if top_fn and gt_fn and top_fn != gt_fn and top_fn != "none":
                    mismatches.append(s.get("id", "?"))
        return mismatches

    def workflow_mismatch(self) -> list[str]:
        mismatches = []
        for s in self.samples:
            wf = s.get("workflow_type")
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                gwf = gt.get("workflow")
                if wf and gwf and wf != gwf:
                    mismatches.append(s.get("id", "?"))
        return mismatches

    def id_analysis(self) -> dict:
        ids = [s.get("id", "") for s in self.samples]
        valid = sum(1 for i in ids if UUID_RE.match(i))
        dupes = len(ids) - len(set(ids))
        empty = sum(1 for i in ids if not i)
        return {
            "valid_uuids": valid,
            "total": len(ids),
            "duplicates": dupes,
            "empty": empty,
        }

    # ── Call function analysis ────────────────────────────────────────────

    def call_function_distribution(self) -> Counter:
        c = Counter()
        for s in self.samples:
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                calls = gt.get("calls", [])
                if isinstance(calls, list):
                    for call in calls:
                        if isinstance(call, dict) and call.get("function"):
                            c[call["function"]] += 1
        return c

    # ── Argument parameter profiling ──────────────────────────────────────

    def argument_params(self) -> tuple[Counter, Counter, defaultdict]:
        keys                       = Counter()
        types                      = Counter()
        value_samples: defaultdict = defaultdict(list)
        for s in self.samples:
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                args = gt.get("arguments", {})
                if isinstance(args, dict):
                    for k, v in args.items():
                        keys[k] += 1
                        types[type(v).__name__] += 1
                        if len(value_samples[k]) < 20:
                            value_samples[k].append(v)
        return keys, types, value_samples

    def sample_argument_params(self) -> Counter:
        keys = Counter()
        for s in self.samples:
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                calls = gt.get("calls", [])
                if isinstance(calls, list):
                    for call in calls:
                        if isinstance(call, dict):
                            args = call.get("arguments", {})
                            if isinstance(args, dict):
                                for k in args:
                                    keys[k] += 1
        return keys

    # ── Missing values ────────────────────────────────────────────────────

    def missing_fields(self) -> dict:
        required      = ["id", "query", "workflow_type", "function_name", "ground_truth"]
        missing: dict = {}
        for field in required:
            count = sum(1 for s in self.samples if field not in s)
            if count:
                missing[field] = count
        return missing

    # ── Split ─────────────────────────────────────────────────────────────

    def split_distribution(self) -> Counter:
        return Counter(s.get("split", "unknown") for s in self.samples)

    # ── Retrieved argument values stats ───────────────────────────────────

    def retrieved_arg_value_stats(self) -> dict:
        present_count         = 0
        param_counts: Counter = Counter()
        for s in self.samples:
            rav = s.get("retrieved_argument_values")
            if rav and isinstance(rav, dict):
                present_count += 1
                for p in rav:
                    param_counts[p] += 1
        return {
            "present": present_count,
            "params": dict(param_counts.most_common(20)),
        }

    # ── Per-function analysis ──────────────────────────────────────────────

    def collect_all_referenced_functions(self) -> Counter:
        """Count SAMPLES (unique per sample) where each function is referenced.

        For each sample, collects a set of function names from:
          - function_name (primary field)
          - ground_truth.function (single_call target)
          - ground_truth.calls[].function (sequential/parallel targets)
        Then increments the counter ONCE per unique function per sample.
        """
        c = Counter()
        for s in self.samples:
            seen: set[str] = set()
            fn = s.get("function_name", "none")
            if fn and fn != "none":
                seen.add(fn)
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                gf = gt.get("function")
                if gf and gf != "none":
                    seen.add(gf)
                calls = gt.get("calls", [])
                if isinstance(calls, list):
                    for call in calls:
                        if isinstance(call, dict) and call.get("function"):
                            seen.add(call["function"])
            for f in seen:
                c[f] += 1
        return c


# ── Report Builder ─────────────────────────────────────────────────────────


class ReportBuilder:
    def __init__(self, title: str = "Dataset Inspection Report"):
        self.lines: list[str] = []
        self.title            = title

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def h1(self, text: str) -> None:
        self.add(heading(text, 1))

    def h2(self, text: str) -> None:
        self.add(heading(text, 2))

    def h3(self, text: str) -> None:
        self.add(heading(text, 3))

    def add_table(self, headers: list[str], rows: list[list[Any]]) -> None:
        self.add(table(headers, rows))

    def add_key_value(
        self, key: str, value: Any, indent: int = 0, unit: str = ""
    ) -> None:
        prefix = "  " * indent
        if unit:
            self.add(f"{prefix}- **{key}**: {value} {unit}")
        else:
            self.add(f"{prefix}- **{key}**: {value}")

    def add_counter_table(
        self,
        counter: Counter,
        title: str,
        top_n: int = 15,
        total: int | None = None,
    ) -> None:
        if not counter:
            self.add(f"_{title}: (empty)_\n")
            return
        self.h3(title)
        rows = []
        for name, count in counter.most_common(top_n):
            pct = fmt_pct(count, total or sum(counter.values()))
            rows.append([name, count, pct])
        if not rows:
            rows.append(["(no data)", "-", "-"])
        self.add_table(["Name", "Count", "Percent"], rows)

    def build(self) -> str:
        header = f"""# {self.title}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---
"""
        return header + "\n".join(self.lines)


# ── Analysis functions ─────────────────────────────────────────────────────


def analyze_dataset(
    analyzer: DatasetAnalyzer,
    schema: dict | None,
    full_library: dict | None,
    report: ReportBuilder,
) -> dict:
    n = analyzer.n
    report.h2(f"Dataset: {analyzer.label} ({n} samples)")

    # ── Overview ──────────────────────────────────────────────────────────
    report.h3("Overview")
    keys = analyzer.top_level_keys()
    report.add_key_value("Total samples", n)
    report.add_key_value("Top-level fields", len(keys))
    report.add(", ".join(f"`{k}` ({v})" for k, v in keys.most_common()))

    missing = analyzer.missing_fields()
    if missing:
        report.add(f"\n**Missing required fields:** {missing}")
    else:
        report.add("\n**Missing required fields:** None")

    # IDs
    id_info = analyzer.id_analysis()
    report.add_key_value("Valid UUIDs", fmt_ratio(id_info["valid_uuids"], id_info["total"]))
    if id_info["duplicates"]:
        report.add_key_value("Duplicate IDs", id_info["duplicates"])
    if id_info["empty"]:
        report.add_key_value("Empty IDs", id_info["empty"])

    # Split
    split_dist = analyzer.split_distribution()
    if split_dist:
        report.add_key_value("Split distribution", dict(split_dist))

    report.add()

    # ── Workflow ──────────────────────────────────────────────────────────
    wf = analyzer.workflow_distribution()
    report.add_counter_table(wf, "Workflow Distribution", top_n=10, total=n)

    # Workflow mismatches
    wf_mm = analyzer.workflow_mismatch()
    if wf_mm:
        report.add(f"- ⚠️ **Workflow mismatches** (top-level != ground_truth): {len(wf_mm)} samples")
    else:
        report.add("- ✅ No workflow mismatches")

    # Primary function vs ground_truth function mismatches
    fn_mm = analyzer.primary_vs_gt_function_mismatch()
    if fn_mm:
        report.add(
            f"- ⚠️ **Function name mismatches** (function_name != ground_truth['function']): "
            f"{len(fn_mm)} samples"
        )
    else:
        report.add("- ✅ No function name mismatches")

    report.add()

    # ── Function distribution ─────────────────────────────────────────────
    fn = analyzer.function_name_distribution()
    report.add_counter_table(fn, "Primary Function Distribution", top_n=15, total=n)

    # Ground truth keys
    gt_keys = analyzer.ground_truth_keys()
    report.add_counter_table(gt_keys, "Ground Truth Fields", top_n=10, total=n)

    # Call functions
    call_fn = analyzer.call_function_distribution()
    if call_fn:
        report.add_counter_table(
            call_fn, "Functions Referenced in `calls[]`", top_n=15
        )

    # ── Query characteristics ─────────────────────────────────────────────
    report.h3("Query Characteristics")
    qlens = analyzer.query_lengths()
    qs    = stat_str(qlens)
    report.add_key_value("Length (characters)", f"mean={qs['mean']}, median={qs['median']}, min={qs['min']}, max={qs['max']}")
    empty_queries = sum(1 for q in qlens if q == 0)
    if empty_queries:
        report.add(f"- ⚠️ **Empty queries**: {empty_queries}")

    report.add()

    # ── Argument profiling ────────────────────────────────────────────────
    arg_keys, arg_types, arg_samples = analyzer.argument_params()
    if arg_keys:
        report.h3("Ground Truth Arguments")
        rows = []
        for name, count in arg_keys.most_common(20):
            pct         = fmt_pct(count, n)
            sample_vals = arg_samples.get(name, [])
            sample_repr = (
                ", ".join(repr(v) for v in sample_vals[:5])
                if sample_vals
                else "-"
            )
            rows.append([name, count, pct, sample_repr])
        report.add_table(
            ["Parameter", "Samples", "Coverage", "Sample Values"],
            rows,
        )

    arg_count_stats = stat_str(analyzer.argument_counts())
    report.add_key_value("Args per ground_truth", f"mean={arg_count_stats['mean']}, median={arg_count_stats['median']}, range=[{arg_count_stats['min']}, {arg_count_stats['max']}]")

    report.add()

    # ── Calls analysis ────────────────────────────────────────────────────
    call_counts = analyzer.call_counts()
    call_stats  = stat_str(call_counts)
    with_calls = sum(1 for c in call_counts if c > 0)
    report.h3("Multi-Call Workflow Analysis")
    report.add_key_value("Samples with calls[]", fmt_ratio(with_calls, n))
    report.add_key_value("Calls per sample", f"mean={call_stats['mean']}, median={call_stats['median']}, range=[{call_stats['min']}, {call_stats['max']}]")

    # Stacked argument params across calls
    call_arg_keys = analyzer.sample_argument_params()
    if call_arg_keys:
        report.add_counter_table(
            call_arg_keys, "Argument Parameters in `calls[]`", top_n=15
        )

    report.add()

    # ── Retrieved functions ──────────────────────────────────────────────
    rf_counts = analyzer.retrieved_functions_counts()
    rf_stats  = stat_str(rf_counts)
    report.h3("Retrieved Functions")
    report.add_key_value("Count", f"mean={rf_stats['mean']}, median={rf_stats['median']}, range=[{rf_stats['min']}, {rf_stats['max']}]")

    # Check if primary function is in retrieved_functions
    primary_missing_retrieval = 0
    for s in analyzer.samples:
        primary   = s.get("function_name")
        retrieved = s.get("retrieved_functions", [])
        if primary and primary != "none" and isinstance(retrieved, list):
            if primary not in retrieved:
                primary_missing_retrieval += 1
    if primary_missing_retrieval:
        report.add(
            f"- ⚠️ **Primary function missing from retrieved_functions**: "
            f"{primary_missing_retrieval} samples"
        )

    report.add()

    # ── Retrieved argument values ─────────────────────────────────────────
    rav_stats = analyzer.retrieved_arg_value_stats()
    report.h3("Retrieved Argument Values")
    report.add_key_value("Samples enriched", fmt_ratio(rav_stats["present"], n))
    if rav_stats["params"]:
        rows = [[p, c, fmt_pct(c, n)] for p, c in rav_stats["params"].items()]
        report.add_table(
            ["Parameter", "Samples", "Coverage"],
            rows,
        )

    report.add()

    # ── Schema coverage ───────────────────────────────────────────────────
    if schema:
        report.h3("Schema Coverage")
        schema_funcs  = set(schema.keys())
        dataset_funcs = set()
        for s in analyzer.samples:
            sfn = s.get("function_name")
            if sfn and sfn != "none":
                dataset_funcs.add(sfn)
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                calls = gt.get("calls", [])
                if isinstance(calls, list):
                    for c in calls:
                        if isinstance(c, dict) and c.get("function"):
                            dataset_funcs.add(c["function"])

        in_schema     = dataset_funcs & schema_funcs
        not_in_schema = dataset_funcs - schema_funcs
        report.add_key_value("Functions in schema", fmt_ratio(len(in_schema), len(dataset_funcs)))
        if not_in_schema and full_library:
            in_lib = not_in_schema & set(full_library.keys())
            report.add_key_value("In full library (not in schema)", len(in_lib) if in_lib else 0)
            truly_missing = not_in_schema - set(full_library.keys())
            if truly_missing:
                report.add(f"- ⚠️ **Hallucinated functions**: {len(truly_missing)} — {', '.join(sorted(truly_missing))}")

    report.add()
    report.add("---")
    report.add()

    return {
        "n": n,
        "workflow": dict(wf.most_common()),
        "functions": dict(fn.most_common(20)),
    }


def cross_split_analysis(
    train_analyzer: DatasetAnalyzer,
    test_analyzer: DatasetAnalyzer,
    train_schema: dict,
    test_schema: dict,
    full_library: dict | None,
    report: ReportBuilder,
) -> None:
    report.h2("Cross-Split Analysis")

    # Primary function overlap
    train_primaries = set()
    for s in train_analyzer.samples:
        fn = s.get("function_name")
        if fn and fn != "none":
            train_primaries.add(fn)

    test_primaries = set()
    for s in test_analyzer.samples:
        fn = s.get("function_name")
        if fn and fn != "none":
            test_primaries.add(fn)

    overlap    = train_primaries & test_primaries
    train_only = train_primaries - test_primaries
    test_only  = test_primaries - train_primaries

    report.h3("Primary Function Overlap")
    report.add_key_value("Train-only primary functions", len(train_only))
    report.add_key_value("Test-only primary functions", len(test_only))
    if overlap:
        report.add_key_value(
            "Overlapping primary functions", f"{len(overlap)} — {', '.join(sorted(overlap))}"
        )
    else:
        report.add("- ✅ No overlap in primary function names")

    # All functions referenced (including calls)
    all_train_funcs = set(train_primaries)
    for s in train_analyzer.samples:
        gt = s.get("ground_truth", {})
        if isinstance(gt, dict):
            calls = gt.get("calls", [])
            if isinstance(calls, list):
                for c in calls:
                    if isinstance(c, dict) and c.get("function"):
                        all_train_funcs.add(c["function"])

    all_test_funcs = set(test_primaries)
    for s in test_analyzer.samples:
        gt = s.get("ground_truth", {})
        if isinstance(gt, dict):
            calls = gt.get("calls", [])
            if isinstance(calls, list):
                for c in calls:
                    if isinstance(c, dict) and c.get("function"):
                        all_test_funcs.add(c["function"])

    report.h3("Schema Consistency")
    if full_library:
        missing = (all_train_funcs | all_test_funcs) - set(full_library.keys())
        if missing:
            report.add(f"- ❌ **Functions not in full library**: {len(missing)} — {', '.join(sorted(missing))}")
        else:
            report.add("- ✅ All referenced functions exist in full library")

    # Train schema coverage
    train_missing_from_schema = all_train_funcs - set(train_schema.keys())
    if train_missing_from_schema:
        in_lib = train_missing_from_schema & set(full_library.keys()) if full_library else set()
        report.add(
            f"- ⚠️ Train functions not in train schema: {len(train_missing_from_schema)} "
            f"({len(in_lib)} in full library, {len(train_missing_from_schema - in_lib)} hallucinated)"
        )

    test_missing_from_schema = all_test_funcs - set(test_schema.keys())
    if test_missing_from_schema:
        in_lib = test_missing_from_schema & set(full_library.keys()) if full_library else set()
        report.add(
            f"- ⚠️ Test functions not in test schema: {len(test_missing_from_schema)} "
            f"({len(in_lib)} in full library, {len(test_missing_from_schema - in_lib)} hallucinated)"
        )

    report.add()

    # Sample size comparison
    report.h3("Dataset Size Comparison")

    rows = [
        ["Metric", "Train", "Test"],
        ["Samples", str(train_analyzer.n), str(test_analyzer.n)],
    ]

    for wf in sorted(
        set(list(train_analyzer.workflow_distribution().keys()) + list(test_analyzer.workflow_distribution().keys()))
    ):
        tc  = train_analyzer.workflow_distribution().get(wf, 0)
        tsc = test_analyzer.workflow_distribution().get(wf, 0)
        tp  = fmt_pct(tc, train_analyzer.n)
        tsp = fmt_pct(tsc, test_analyzer.n)
        rows.append([f"  {wf}", f"{tc} ({tp})", f"{tsc} ({tsp})"])

    rows.append(["Unique primary functions", str(len(train_primaries)), str(len(test_primaries))])
    rows.append(["Schema functions", str(len(train_schema)), str(len(test_schema))])

    report.add_table(rows[0], rows[1:])
    report.add()


def clean_comparison(
    analyzer_orig: DatasetAnalyzer,
    analyzer_clean: DatasetAnalyzer | None,
    label: str,
    report: ReportBuilder,
) -> None:
    if analyzer_clean is None:
        return
    report.h2(f"Cleaning Impact: {label}")
    removed = analyzer_orig.n - analyzer_clean.n
    report.add_key_value("Original samples", analyzer_orig.n)
    report.add_key_value("Clean samples", analyzer_clean.n)
    report.add_key_value("Removed", f"{removed} ({fmt_pct(removed, analyzer_orig.n)})")

    for wf in sorted(
        set(list(analyzer_orig.workflow_distribution().keys()) + list(analyzer_clean.workflow_distribution().keys()))
    ):
        oc = analyzer_orig.workflow_distribution().get(wf, 0)
        cc = analyzer_clean.workflow_distribution().get(wf, 0)
        if oc != cc:
            report.add(f"- {wf}: {oc} → {cc} (removed {oc - cc})")

    report.add()


# ── CLI Summary ─────────────────────────────────────────────────────────────


def _section_header(text: str, width: int = 70) -> None:
    print(f"\n  {ANSI_BOLD}{text}{ANSI_RESET}")
    print(f"  {dotted_line('─', width)}")


def _warn(text: str) -> None:
    print(f"  {ANSI_YELLOW}⚠ {text}{ANSI_RESET}")


def _ok(text: str) -> None:
    print(f"  {ANSI_GREEN}✓ {text}{ANSI_RESET}")


def _err(text: str) -> None:
    print(f"  {ANSI_RED}✗ {text}{ANSI_RESET}")


def _info(text: str) -> None:
    print(f"  {ANSI_CYAN}{text}{ANSI_RESET}")


def _bold(text: str) -> str:
    return f"{ANSI_BOLD}{text}{ANSI_RESET}"


def _dim(text: str) -> str:
    return f"{ANSI_DIM}{text}{ANSI_RESET}"


def print_cli_table(headers: list[str], rows: list[list[Any]]) -> None:
    """Print a formatted ASCII table to CLI."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    # Cap column widths
    col_widths = [min(w, 50) for w in col_widths]
    sep = "  " + "─" * (sum(col_widths) + 3 * (len(col_widths) - 1))
    hdr = "  " + " │ ".join(h.ljust(w)[:w] for h, w in zip(headers, col_widths))
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        line = "  " + " │ ".join(
            str(c).ljust(w)[:w] for c, w in zip(row, col_widths)
        )
        print(line)


def print_cli_summary(
    label: str,
    analyzer: DatasetAnalyzer,
    schema: dict | None,
    full_library: dict | None,
) -> None:
    n           = analyzer.n
    wf          = analyzer.workflow_distribution()
    fn          = analyzer.function_name_distribution()
    qlens       = analyzer.query_lengths()
    qs          = stat_str(qlens)
    arg_counts  = analyzer.argument_counts()
    acs         = stat_str(arg_counts)
    call_counts = analyzer.call_counts()
    ccs         = stat_str(call_counts)

    print(f"\n{ANSI_BOLD}{'=' * 70}{ANSI_RESET}")
    print(f"  {ANSI_BOLD}{label} ({n} samples){ANSI_RESET}")
    print(f"{ANSI_BOLD}{'=' * 70}{ANSI_RESET}")

    _section_header("WORKFLOW DISTRIBUTION")
    if wf:
        max_cnt = max(wf.values())
        for wt, cnt in wf.most_common():
            bar_w = int(cnt / max_cnt * 30)
            bar = "█" * bar_w
            print(f"    {wt:25s} {cnt:5d} ({fmt_pct(cnt, n):>5s})  {ANSI_CYAN}{bar}{ANSI_RESET}")

    _section_header("QUERY CHARACTERISTICS")
    print(f"    Length:       mean={qs['mean']}, median={qs['median']}, range=[{qs['min']}, {qs['max']}]")
    empty_queries = sum(1 for q in qlens if q == 0)
    if empty_queries:
        _warn(f"Empty queries: {empty_queries}")

    _section_header("ARGUMENTS")
    print(f"    Params/sample:  mean={acs['mean']}, median={acs['median']}, range=[{acs['min']}, {acs['max']}]")
    arg_keys, arg_types, _ = analyzer.argument_params()
    print(f"    Unique param names: {len(arg_keys)}")
    if arg_keys:
        print(f"    Top params: {dict(arg_keys.most_common(8))}")
    with_calls = sum(1 for c in call_counts if c > 0)
    print(f"    Samples with calls[]: {fmt_ratio(with_calls, n)}")
    print(f"    Calls/sample:         mean={ccs['mean']}, median={ccs['median']}, range=[{ccs['min']}, {ccs['max']}]")

    _section_header("RETRIEVAL COVERAGE")
    rf_counts = analyzer.retrieved_functions_counts()
    rfs       = stat_str(rf_counts)
    print(f"    Retrieved functions:  mean={rfs['mean']}, range=[{rfs['min']}, {rfs['max']}]")
    rav_present, rav_total = analyzer.argument_value_presence()
    print(f"    Arg values enriched:  {fmt_ratio(rav_present, rav_total)}")

    _section_header("DATA QUALITY")
    id_info = analyzer.id_analysis()
    if id_info["valid_uuids"] == id_info["total"]:
        _ok(f"All {id_info['total']} IDs are valid UUIDs")
    else:
        _warn(f"UUIDs: {fmt_ratio(id_info['valid_uuids'], id_info['total'])}")
    if id_info["duplicates"]:
        _err(f"Duplicate IDs: {id_info['duplicates']}")
    if id_info["empty"]:
        _err(f"Empty IDs: {id_info['empty']}")

    wf_mm = analyzer.workflow_mismatch()
    if wf_mm:
        _warn(f"Workflow mismatches: {len(wf_mm)}")
    else:
        _ok("No workflow mismatches")

    fn_mm = analyzer.primary_vs_gt_function_mismatch()
    if fn_mm:
        _warn(f"Function name mismatches: {len(fn_mm)}")
    else:
        _ok("No function name mismatches")

    missing = analyzer.missing_fields()
    if missing:
        _err(f"Missing fields: {missing}")
    else:
        _ok("No missing required fields")

    call_fn = analyzer.call_function_distribution()
    if call_fn:
        print(f"    Unique call functions: {_bold(str(len(call_fn)))}")

    _section_header("TOP 10 PRIMARY FUNCTIONS")
    print_cli_table(
        ["Function", "Count", "Percent", "Distribution"],
        [
            [fname, str(cnt), fmt_pct(cnt, n), ascii_bar(100.0 * cnt / n, 25)]
            for fname, cnt in fn.most_common(10)
        ],
    )

    # Schema coverage
    if schema:
        schema_funcs  = set(schema.keys())
        dataset_funcs = set()
        for s in analyzer.samples:
            fn = s.get("function_name")
            if fn and fn != "none":
                dataset_funcs.add(fn)
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                calls = gt.get("calls", [])
                if isinstance(calls, list):
                    for c in calls:
                        if isinstance(c, dict) and c.get("function"):
                            dataset_funcs.add(c["function"])
        not_in_schema = dataset_funcs - schema_funcs
        if not_in_schema and full_library:
            truly_missing = not_in_schema - set(full_library.keys())
            if truly_missing:
                print(f"\n  {ANSI_RED}Hallucinated functions: {len(truly_missing)} — {sorted(truly_missing)}{ANSI_RESET}")

    print()


def print_function_analysis(
    train_analyzer: DatasetAnalyzer,
    test_analyzer: DatasetAnalyzer,
    full_library: dict | None,
) -> None:
    """Print a detailed table of ALL functions referenced in train and/or test."""
    train_ref = train_analyzer.collect_all_referenced_functions()
    test_ref = test_analyzer.collect_all_referenced_functions()
    all_funcs = sorted(set(list(train_ref.keys()) + list(test_ref.keys())))

    if not all_funcs:
        _warn("No functions found in datasets")
        return

    print(f"\n{ANSI_BOLD}{'=' * 70}{ANSI_RESET}")
    print(f"  {ANSI_BOLD}COMPREHENSIVE FUNCTION ANALYSIS ({len(all_funcs)} total){ANSI_RESET}")
    print(f"{ANSI_BOLD}{'=' * 70}{ANSI_RESET}")

    headers = ["Function", "Params (lib)", "Req (lib)", "Train (samples)", "Test (samples)", "Total", "Library Info"]
    rows = []

    for func_name in all_funcs:
        tc = train_ref.get(func_name, 0)
        tsc = test_ref.get(func_name, 0)
        total_ref = tc + tsc

        # Library info
        lib_info = ""
        if full_library and func_name in full_library:
            schema = full_library[func_name]
            desc = schema.get("description", "")
            lib_info = desc[:60] + ("..." if len(desc) > 60 else "")
        else:
            lib_info = f"{ANSI_RED}NOT IN LIBRARY{ANSI_RESET}"

        param_count = 0
        required_count = 0
        if full_library and func_name in full_library:
            params = full_library[func_name].get("parameters", {})
            param_count = len(params)
            required_count = sum(
                1 for p in params.values()
                if isinstance(p, dict) and p.get("required")
            )

        rows.append([
            func_name,
            str(param_count),
            str(required_count),
            str(tc),
            str(tsc),
            str(total_ref),
            lib_info,
        ])

    print_cli_table(headers, rows)
    print()

    # Summary stats about function coverage
    _section_header("FUNCTION COVERAGE SUMMARY")
    train_only = sorted(set(train_ref.keys()) - set(test_ref.keys()))
    test_only = sorted(set(test_ref.keys()) - set(train_ref.keys()))
    overlap = sorted(set(train_ref.keys()) & set(test_ref.keys()))

    if train_only:
        _info(f"Train-only: {len(train_only)} — {', '.join(train_only)}")
    if test_only:
        _info(f"Test-only:  {len(test_only)} — {', '.join(test_only)}")
    if overlap:
        _ok(f"Overlap:    {len(overlap)} functions appear in both splits")

    # Library coverage
    if full_library:
        in_lib = [f for f in all_funcs if f in full_library]
        not_in_lib = [f for f in all_funcs if f not in full_library]
        if not_in_lib:
            _err(f"HALLUCINATED: {len(not_in_lib)} — {', '.join(not_in_lib)}")
        else:
            _ok(f"All {len(all_funcs)} referenced functions exist in library ({len(full_library)} total)")

        # Functions in library but never referenced
        never_ref = sorted(set(full_library.keys()) - set(all_funcs))
        if never_ref:
            _warn(f"Library functions never used: {len(never_ref)} — {', '.join(never_ref)}")
        else:
            _ok("All library functions are referenced in at least one split")

    print()


def compare_datasets_cli(
    train_analyzer: DatasetAnalyzer,
    test_analyzer: DatasetAnalyzer,
    full_library: dict | None = None,
) -> None:
    print(f"\n{ANSI_BOLD}{'=' * 70}{ANSI_RESET}")
    print(f"  {ANSI_BOLD}TRAIN vs TEST COMPARISON{ANSI_RESET}")
    print(f"{ANSI_BOLD}{'=' * 70}{ANSI_RESET}")
    print_cli_table(
        ["Metric", "Train", "Test"],
        [
            ["Samples", str(train_analyzer.n), str(test_analyzer.n)],
        ],
    )

    all_wf = sorted(
        set(list(train_analyzer.workflow_distribution().keys()) + list(test_analyzer.workflow_distribution().keys()))
    )
    if all_wf:
        _section_header("Workflow Split")
        for wf in all_wf:
            tc  = train_analyzer.workflow_distribution().get(wf, 0)
            tsc = test_analyzer.workflow_distribution().get(wf, 0)
            tp  = fmt_pct(tc, train_analyzer.n)
            tsp = fmt_pct(tsc, test_analyzer.n)
            print(f"    {wf:25s}  {tc:5d} ({tp:>5s})  {tsc:5d} ({tsp:>5s})")

    train_fn = train_analyzer.function_name_distribution()
    test_fn  = test_analyzer.function_name_distribution()
    train_primaries = {k for k in train_fn if k != "none"}
    test_primaries = {k for k in test_fn if k != "none"}

    train_call_fn = train_analyzer.call_function_distribution()
    test_call_fn  = test_analyzer.call_function_distribution()
    train_all     = train_primaries | set(train_call_fn.keys())
    test_all      = test_primaries | set(test_call_fn.keys())

    _section_header("Function Coverage")
    print(f"    {'Primary functions (unique)':40s} {len(train_primaries):>5d} {len(test_primaries):>5d}")
    print(f"    {'All referenced functions':40s} {len(train_all):>5d} {len(test_all):>5d}")
    print(f"    {'Overlap':40s} {len(train_primaries & test_primaries):>5d}")
    print(f"    {'Train-only':40s} {len(train_primaries - test_primaries):>5d}")
    print(f"    {'Test-only':40s} {len(test_primaries - train_primaries):>5d}")

    # Show top-5 most imbalanced functions
    _section_header("Top-5 Most Imbalanced Functions")
    all_joint = sorted(set(train_primaries | test_primaries))
    imbalances = []
    for fn in all_joint:
        tc = train_fn.get(fn, 0)
        tsc = test_fn.get(fn, 0)
        if tc + tsc > 0:
            ratio = tc / (tc + tsc) if (tc + tsc) > 0 else 0
            imbalances.append((abs(0.5 - ratio), fn, tc, tsc))
    imbalances.sort(reverse=True)
    for _, fn, tc, tsc in imbalances[:5]:
        bar = ascii_bar(100.0 * tc / (tc + tsc) if (tc + tsc) > 0 else 50, 20)
        print(f"    {fn:40s}  train={tc:>4d}  test={tsc:>4d}  {bar}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Comprehensive dataset EDA")
    parser.add_argument("--train", required=True, help="Train JSONL dataset path")
    parser.add_argument("--test", required=True, help="Test JSONL dataset path")
    parser.add_argument("--train-schema", required=True, help="Train schema path")
    parser.add_argument("--test-schema", required=True, help="Test schema path")
    parser.add_argument(
        "--library", default="data/processed/function_library.json", help="Full library path"
    )
    parser.add_argument("--train-clean", help="Optional cleaned train path")
    parser.add_argument("--test-clean", help="Optional cleaned test path")
    parser.add_argument("--report", default="", help="Path for markdown report output")
    args = parser.parse_args()

    # Load data
    train_samples = load_jsonl(args.train)
    test_samples  = load_jsonl(args.test)
    train_schema  = load_json(args.train_schema)
    test_schema   = load_json(args.test_schema)

    full_library = None
    if args.library and Path(args.library).exists():
        full_library = load_json(args.library)

    train_clean_analyzer = None
    test_clean_analyzer  = None
    if args.train_clean and Path(args.train_clean).exists():
        train_clean_analyzer = DatasetAnalyzer(
            load_jsonl(args.train_clean), "Train (Cleaned)"
        )
    if args.test_clean and Path(args.test_clean).exists():
        test_clean_analyzer = DatasetAnalyzer(
            load_jsonl(args.test_clean), "Test (Cleaned)"
        )

    train_analyzer = DatasetAnalyzer(train_samples, "Train")
    test_analyzer  = DatasetAnalyzer(test_samples, "Test")

    # ── CLI output ────────────────────────────────────────────────────────
    print_cli_summary("TRAIN DATASET", train_analyzer, train_schema, full_library)
    print_cli_summary("TEST DATASET", test_analyzer, test_schema, full_library)
    print_function_analysis(train_analyzer, test_analyzer, full_library)
    compare_datasets_cli(train_analyzer, test_analyzer, full_library)

    if train_clean_analyzer:
        removed = train_analyzer.n - train_clean_analyzer.n
        print(f"\n  CLEAN TRAIN: {train_analyzer.n} → {train_clean_analyzer.n} ({removed} removed, {fmt_pct(removed, train_analyzer.n)})")
    if test_clean_analyzer:
        removed = test_analyzer.n - test_clean_analyzer.n
        print(f"  CLEAN TEST:  {test_analyzer.n} → {test_clean_analyzer.n} ({removed} removed, {fmt_pct(removed, test_analyzer.n)})")

    # ── Markdown report ───────────────────────────────────────────────────
    if args.report:
        report = ReportBuilder("Telecom Dataset Inspection Report")

        report.add(f"**Train dataset**: `{args.train}` ({train_analyzer.n} samples)")
        report.add(f"**Test dataset**:  `{args.test}` ({test_analyzer.n} samples)")
        report.add(f"**Train schema**: `{args.train_schema}` ({len(train_schema)} functions)")
        report.add(f"**Test schema**:  `{args.test_schema}` ({len(test_schema)} functions)")
        if full_library:
            report.add(f"**Full library**: `{args.library}` ({len(full_library)} functions)")
        report.add()

        analyze_dataset(train_analyzer, train_schema, full_library, report)
        analyze_dataset(test_analyzer, test_schema, full_library, report)

        cross_split_analysis(
            train_analyzer, test_analyzer, train_schema, test_schema, full_library, report
        )

        if train_clean_analyzer:
            clean_comparison(train_analyzer, train_clean_analyzer, "Train", report)
        if test_clean_analyzer:
            clean_comparison(test_analyzer, test_clean_analyzer, "Test", report)

        function_analysis_report(train_analyzer, test_analyzer, full_library, report)

        report.add(_build_recommendations(train_analyzer, test_analyzer, full_library))

        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report.build())
        print(f"\n📄 Report written to {args.report}")


def function_analysis_report(
    train_analyzer: DatasetAnalyzer,
    test_analyzer: DatasetAnalyzer,
    full_library: dict | None,
    report: ReportBuilder,
) -> None:
    """Add a detailed per-function analysis section to the markdown report."""
    report.h2("Per-Function Analysis")

    train_ref = train_analyzer.collect_all_referenced_functions()
    test_ref = test_analyzer.collect_all_referenced_functions()
    all_funcs = sorted(set(list(train_ref.keys()) + list(test_ref.keys())))

    if not all_funcs:
        report.add("_No functions found in datasets._")
        return

    # Build parameter summary for each function
    func_param_info: dict[str, dict] = {}
    if full_library:
        for fname, schema in full_library.items():
            params = schema.get("parameters", {})
            required = [
                p for p, info in params.items()
                if isinstance(info, dict) and info.get("required")
            ]
            optional = [
                p for p, info in params.items()
                if isinstance(info, dict) and not info.get("required")
            ]
            func_param_info[fname] = {
                "total_params": len(params),
                "required": required,
                "optional": optional,
                "description": schema.get("description", ""),
            }

    rows = []
    for func_name in all_funcs:
        tc = train_ref.get(func_name, 0)
        tsc = test_ref.get(func_name, 0)
        total_ref = tc + tsc

        in_lib = full_library and func_name in full_library
        if in_lib and func_name in func_param_info:
            info = func_param_info[func_name]
            desc_short = info["description"][:80]
            params_str = f"{info['total_params']} ({len(info['required'])} req, {len(info['optional'])} opt)"
            req_params = ", ".join(info["required"]) if info["required"] else "—"
        else:
            desc_short = "⚠️ NOT IN LIBRARY"
            params_str = "—"
            req_params = "—"

        train_only_str = "✓" if func_name in train_ref and func_name not in test_ref else ""
        test_only_str = "✓" if func_name in test_ref and func_name not in train_ref else ""

        split_info = "both" if tc > 0 and tsc > 0 else "train" if tc > 0 else "test"
        rows.append([
            func_name,
            desc_short,
            params_str,
            req_params,
            str(tc),
            str(tsc),
            str(total_ref),
            split_info,
        ])

    report.h3(f"All Referenced Functions ({len(all_funcs)} total)")
    report.add_table(
        ["Function", "Description", "Params (lib)", "Required Params (lib)", "Train (samples)", "Test (samples)", "Total", "Split"],
        rows,
    )

    # Summary stats
    report.h3("Function Coverage Summary")
    in_lib_count = sum(1 for f in all_funcs if full_library and f in full_library)
    not_in_lib = [f for f in all_funcs if not (full_library and f in full_library)]
    train_only = sorted(set(train_ref.keys()) - set(test_ref.keys()))
    test_only = sorted(set(test_ref.keys()) - set(train_ref.keys()))
    overlap = sorted(set(train_ref.keys()) & set(test_ref.keys()))

    if overlap:
        report.add(f"- ✅ **Overlapping functions** (both splits): {len(overlap)}")
    if train_only:
        report.add(f"- 📌 **Train-only functions**: {len(train_only)} — {', '.join(train_only)}")
    if test_only:
        report.add(f"- 📌 **Test-only functions**: {len(test_only)} — {', '.join(test_only)}")

    report.add_key_value("Referenced functions in library", fmt_ratio(in_lib_count, len(all_funcs)))
    if not_in_lib:
        report.add(f"- ❌ **Not in library**: {len(not_in_lib)} — {', '.join(not_in_lib)}")

    if full_library:
        never_ref = sorted(set(full_library.keys()) - set(all_funcs))
        if never_ref:
            report.add(f"- ⚠️ **Library functions never referenced**: {len(never_ref)} — {', '.join(never_ref)}")
        else:
            report.add("- ✅ All library functions are referenced in at least one split")

    # Distribution balance table
    report.h3("Most Imbalanced Functions")
    balance_rows = []
    for fn in all_funcs:
        tc = train_ref.get(fn, 0)
        tsc = test_ref.get(fn, 0)
        if tc + tsc > 0:
            train_pct = 100.0 * tc / (tc + tsc)
            test_pct = 100.0 * tsc / (tc + tsc)
            imbalance = abs(50.0 - train_pct)
            if imbalance > 15:
                balance_rows.append([fn, tc, tsc, f"{train_pct:.0f}% / {test_pct:.0f}%", f"{imbalance:.0f}%"])

    if balance_rows:
        balance_rows.sort(key=lambda r: -abs(50.0 - float(r[3].split("%")[0])))
        report.add_table(
            ["Function", "Train", "Test", "Split %", "Imbalance"],
            balance_rows,
        )

    report.add()
    report.add("---")
    report.add()


def _build_recommendations(
    train_analyzer: DatasetAnalyzer,
    test_analyzer: DatasetAnalyzer,
    full_library: dict | None,
) -> str:
    lines = ["## Recommendations\n"]

    # Check imbalance
    train_wf = train_analyzer.workflow_distribution()
    test_wf  = test_analyzer.workflow_distribution()

    if full_library:
        # Check for hallucinated functions
        all_train_funcs = set()
        for s in train_analyzer.samples:
            fn = s.get("function_name")
            if fn and fn != "none":
                all_train_funcs.add(fn)
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                calls = gt.get("calls", [])
                if isinstance(calls, list):
                    for c in calls:
                        if isinstance(c, dict) and c.get("function"):
                            all_train_funcs.add(c["function"])
        hallucinated = all_train_funcs - set(full_library.keys())
        if hallucinated:
            lines.append(
                f"- ❌ **Train dataset** contains {len(hallucinated)} hallucinated function(s): "
                f"{', '.join(sorted(hallucinated))}. "
                f"Run `scripts/clean_dataset.py` to remove these samples."
            )

        all_test_funcs = set()
        for s in test_analyzer.samples:
            fn = s.get("function_name")
            if fn and fn != "none":
                all_test_funcs.add(fn)
            gt = s.get("ground_truth", {})
            if isinstance(gt, dict):
                calls = gt.get("calls", [])
                if isinstance(calls, list):
                    for c in calls:
                        if isinstance(c, dict) and c.get("function"):
                            all_test_funcs.add(c["function"])
        hallucinated_test = all_test_funcs - set(full_library.keys())
        if hallucinated_test:
            lines.append(
                f"- ❌ **Test dataset** contains {len(hallucinated_test)} hallucinated function(s): "
                f"{', '.join(sorted(hallucinated_test))}. "
                f"Run `scripts/clean_dataset.py` to remove these samples."
            )

    # Workflow imbalance
    def _check_imbalance(wf: Counter, label: str):
        total = sum(wf.values())
        if total == 0:
            return
        for wt, cnt in wf.most_common():
            pct = cnt / total
            if wt in ("single_call",) and pct < 0.30:
                lines.append(
                    f"- ⚠️ **{label}**: `{wt}` is only {fmt_pct(cnt, total)} "
                    f"— consider increasing for better coverage."
                )
            if wt in ("parallel", "sequential") and pct < 0.05:
                lines.append(
                    f"- ⚠️ **{label}**: `{wt}` is only {fmt_pct(cnt, total)} "
                    f"— few-shot examples may be limited."
                )
            if wt in ("abstention",) and pct < 0.02:
                lines.append(
                    f"- ⚠️ **{label}**: `{wt}` is only {fmt_pct(cnt, total)} "
                    f"— very few rejection samples."
                )

    _check_imbalance(train_wf, "Train")
    _check_imbalance(test_wf, "Test")

    lines.append(
        "- 🔍 Run `scripts/validate_dataset.py` to verify data integrity "
        "after any cleaning or regeneration."
    )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
