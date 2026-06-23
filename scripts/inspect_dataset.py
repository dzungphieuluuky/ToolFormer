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

Usage:
    python scripts/inspect_dataset.py \\
        --train data/processed/train_dataset_uncleaned.jsonl \\
        --test data/processed/test_dataset_uncleaned.jsonl \\
        --train-schema data/processed/function_schema_train.json \\
        --test-schema data/processed/function_schema_test.json \\
        --library data/processed/function_library.json \\
        [--train-clean data/processed/train_dataset_clean.jsonl] \\
        [--test-clean data/processed/test_dataset_clean.jsonl] \\
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

    print(f"\n{'=' * 70}")
    print(f"  {label} ({n} samples)")
    print(f"{'=' * 70}")

    print(f"\n  WORKFLOW")
    for wt, cnt in wf.most_common():
        print(f"    {wt:20s}  {cnt:5d}  ({fmt_pct(cnt, n)})")

    print(f"\n  TOP FUNCTIONS")
    for fname, cnt in fn.most_common(10):
        print(f"    {fname:35s}  {cnt:5d}  ({fmt_pct(cnt, n)})")

    print(f"\n  QUERY")
    print(f"    length:  mean={qs['mean']}, median={qs['median']}, range=[{qs['min']}, {qs['max']}]")

    print(f"\n  ARGUMENTS")
    print(f"    params per sample:  mean={acs['mean']}, median={acs['median']}, range=[{acs['min']}, {acs['max']}]")
    arg_keys, arg_types, _ = analyzer.argument_params()
    print(f"    unique param names: {len(arg_keys)}")
    if arg_keys:
        print(f"    top: {dict(arg_keys.most_common(8))}")

    print(f"\n  CALLS")
    with_calls = sum(1 for c in call_counts if c > 0)
    print(f"    samples with calls: {fmt_ratio(with_calls, n)}")
    print(f"    calls per sample:   mean={ccs['mean']}, median={ccs['median']}, range=[{ccs['min']}, {ccs['max']}]")

    print(f"\n  RETRIEVED FUNCTIONS")
    rf_counts = analyzer.retrieved_functions_counts()
    rfs       = stat_str(rf_counts)
    print(f"    count: mean={rfs['mean']}, range=[{rfs['min']}, {rfs['max']}]")

    print(f"\n  RETRIEVED ARGUMENT VALUES")
    rav_present, rav_total = analyzer.argument_value_presence()
    print(f"    enriched: {fmt_ratio(rav_present, rav_total)}")

    print(f"\n  DATA QUALITY")
    id_info = analyzer.id_analysis()
    print(f"    valid UUIDs: {fmt_ratio(id_info['valid_uuids'], id_info['total'])}")
    if id_info["duplicates"]:
        print(f"    duplicate IDs: {id_info['duplicates']}")

    wf_mm = analyzer.workflow_mismatch()
    if wf_mm:
        print(f"    workflow mismatches: {len(wf_mm)}")

    fn_mm = analyzer.primary_vs_gt_function_mismatch()
    if fn_mm:
        print(f"    function name mismatches: {len(fn_mm)}")

    missing = analyzer.missing_fields()
    if missing:
        print(f"    missing fields: {missing}")

    call_fn = analyzer.call_function_distribution()
    if call_fn:
        print(f"    unique call functions: {len(call_fn)}")

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
                print(f"    hallucinated functions: {len(truly_missing)} — {sorted(truly_missing)}")

    print()


def compare_datasets_cli(
    train_analyzer: DatasetAnalyzer, test_analyzer: DatasetAnalyzer
) -> None:
    print(f"{'=' * 70}")
    print("  TRAIN vs TEST COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'':30s} {'TRAIN':>10s} {'TEST':>10s}")
    print(f"  {'Samples':30s} {train_analyzer.n:>10d} {test_analyzer.n:>10d}")

    for wf in sorted(
        set(list(train_analyzer.workflow_distribution().keys()) + list(test_analyzer.workflow_distribution().keys()))
    ):
        tc  = train_analyzer.workflow_distribution().get(wf, 0)
        tsc = test_analyzer.workflow_distribution().get(wf, 0)
        tp  = fmt_pct(tc, train_analyzer.n)
        tsp = fmt_pct(tsc, test_analyzer.n)
        print(f"  {wf:30s} {tc:>5d} ({tp:>5s}) {tsc:>5d} ({tsp:>5s})")

    train_fn = train_analyzer.function_name_distribution()
    test_fn  = test_analyzer.function_name_distribution()
    train_primaries = {k for k in train_fn if k != "none"}
    test_primaries = {k for k in test_fn if k != "none"}
    print(f"  {'Unique primary functions':30s} {len(train_primaries):>10d} {len(test_primaries):>10d}")
    print(f"  {'Overlap':30s} {len(train_primaries & test_primaries):>10d}")
    print(f"  {'Train-only':30s} {len(train_primaries - test_primaries):>10d}")
    print(f"  {'Test-only':30s} {len(test_primaries - train_primaries):>10d}")

    train_call_fn = train_analyzer.call_function_distribution()
    test_call_fn  = test_analyzer.call_function_distribution()
    train_all     = train_primaries | set(train_call_fn.keys())
    test_all      = test_primaries | set(test_call_fn.keys())
    print(f"  {'All referenced functions':30s} {len(train_all):>10d} {len(test_all):>10d}")
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
    compare_datasets_cli(train_analyzer, test_analyzer)

    if train_clean_analyzer:
        removed = train_analyzer.n - train_clean_analyzer.n
        print(f"  CLEAN TRAIN: {train_analyzer.n} → {train_clean_analyzer.n} ({removed} removed)")
    if test_clean_analyzer:
        removed = test_analyzer.n - test_clean_analyzer.n
        print(f"  CLEAN TEST:  {test_analyzer.n} → {test_clean_analyzer.n} ({removed} removed)")

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

        report.add(_build_recommendations(train_analyzer, test_analyzer, full_library))

        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report.build())
        print(f"\n📄 Report written to {args.report}")


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
