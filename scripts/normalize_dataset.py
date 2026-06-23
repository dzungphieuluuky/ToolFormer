"""One-time migration: normalise ground_truth to always have a 'calls' array.

Before: ground_truth varies by workflow type:
  - single_call: {function, arguments, workflow, reasoning}
  - sequential:  {function, arguments, workflow, calls: [{step, function, arguments, depends_on}], reasoning}
  - parallel:    {function, arguments, workflow, calls: [{function, arguments}], reasoning}
  - abstention:  {function: null, arguments: {}, workflow, refusal_message, reasoning}

After: every sample has:
  - calls: [{function, arguments}, ...]   (always present; empty for abstention)
  - workflow, reasoning, refusal_message  (as before)
  - top-level function/arguments removed (redundant — subsumed by calls)
  - step/depends_on stripped from sequential calls (unused by pipeline)
"""

import json
from pathlib import Path


def normalise_gt(gt: dict) -> dict:
    wf = gt.get("workflow", "single_call")

    if wf in ("sequential", "parallel"):
        raw_calls = gt.get("calls", [])
        calls = [
            {"function": c["function"], "arguments": c.get("arguments", {})}
            for c in raw_calls
        ]
    elif wf == "single_call":
        calls = [
            {"function": gt.get("function"), "arguments": gt.get("arguments", {})}
        ]
    else:  # abstention
        calls = []

    result = {
        "calls": calls,
        "workflow": wf,
        "reasoning": gt.get("reasoning", ""),
    }
    if "refusal_message" in gt:
        result["refusal_message"] = gt["refusal_message"]

    return result


def normalise_jsonl(path: Path) -> tuple[int, int]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            s["ground_truth"] = normalise_gt(s.get("ground_truth", {}))
            samples.append(s)

    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return len(samples)


def main():
    data_dir = Path("data/processed")
    for name in ("train_dataset.jsonl", "test_dataset.jsonl"):
        path = data_dir / name
        if path.exists():
            count = normalise_jsonl(path)
            print(f"✅ {name}: {count} samples normalised")
        else:
            print(f"⚠️  {name} not found, skipping")


if __name__ == "__main__":
    main()
