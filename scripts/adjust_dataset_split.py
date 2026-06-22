import json
import argparse
from pathlib import Path
from collections import defaultdict

def load_jsonl(path: Path):
    data = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(path: Path, records):
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Adjust train/test split to satisfy custom function count constraints.")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train_dataset.jsonl"), help="Path to original train jsonl")
    parser.add_argument("--test", type=Path, default=Path("data/processed/test_dataset.jsonl"), help="Path to original test jsonl")
    parser.add_argument("--out-train", type=Path, default=Path("data/processed/train_dataset_adjusted.jsonl"))
    parser.add_argument("--out-test", type=Path, default=Path("data/processed/test_dataset_adjusted.jsonl"))
    args = parser.parse_args()

    train_records = load_jsonl(args.train)
    test_records = load_jsonl(args.test)

    # Determine distinct functions in original train
    train_funcs_order = []
    seen = set()
    for rec in train_records:
        fn = rec.get("function_name")
        if fn not in seen:
            seen.add(fn)
            train_funcs_order.append(fn)
    # Keep only first 20 functions for train
    desired_train_funcs = set(train_funcs_order[:20])

    # Reassign records
    new_train = []
    new_test = []
    # Start with original test records
    new_test.extend(test_records)

    # Process original train records
    for rec in train_records:
        fn = rec.get("function_name")
        if fn in desired_train_funcs:
            new_train.append(rec)
        else:
            new_test.append(rec)

    # Ensure test contains at least one sample for each of the 20 train functions
    existing_test_funcs = {rec.get("function_name") for rec in new_test}
    for fn in desired_train_funcs:
        if fn not in existing_test_funcs:
            # Find first occurrence in original train and copy to test
            for rec in train_records:
                if rec.get("function_name") == fn:
                    new_test.append(rec)
                    break

    # Add 6 additional functions not in train (if available)
    all_funcs = {rec.get("function_name") for rec in train_records + test_records}
    remaining_funcs = list(all_funcs - desired_train_funcs)
    additional_funcs = remaining_funcs[:6]
    # Ensure they are present in test (they already are, but add if missing)
    for fn in additional_funcs:
        if fn not in existing_test_funcs:
            # find any record with this function from original test or train
            for rec in test_records + train_records:
                if rec.get("function_name") == fn:
                    new_test.append(rec)
                    break

    # Add 5 miscellaneous functions (next distinct ones after the above)
    misc_funcs = remaining_funcs[6:11]
    for fn in misc_funcs:
        if fn not in existing_test_funcs:
            for rec in test_records + train_records:
                if rec.get("function_name") == fn:
                    new_test.append(rec)
                    break

    # Write adjusted files
    write_jsonl(args.out_train, new_train)
    write_jsonl(args.out_test, new_test)
    print(f"Adjusted dataset written. Train records: {len(new_train)} (functions: {len({r.get('function_name') for r in new_train})}). Test records: {len(new_test)} (functions: {len({r.get('function_name') for r in new_test})}).")

if __name__ == "__main__":
    main()
