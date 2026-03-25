#!/usr/bin/env python3
"""
Build data/phase_1/test.jsonl by holding out ~100 prompts that do not appear in SFT data.

Reads data/phase_1/raw_prompts.jsonl and optionally data/phase_1/sft_train.jsonl (for holdout logic).
If data/phase_1/complete_dataset.jsonl exists, loads input -> gold docstring and sets "reference"
on each test row for Phase 7 evaluation (rows without a match omit "reference" and a warning is printed).

Writes data/phase_1/test.jsonl (overwrites each run).
"""

import json
import random
import sys
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "phase_1"
TEST_SIZE = 100


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_docstring_lookup(complete_path: Path) -> dict[str, str]:
    """Map input source string -> gold docstring body."""
    if not complete_path.exists():
        return {}
    lookup: dict[str, str] = {}
    for row in load_jsonl(complete_path):
        if not isinstance(row, dict):
            continue
        if row.get("error"):
            continue
        inp = row.get("input")
        doc = row.get("docstring")
        if isinstance(inp, str) and isinstance(doc, str) and doc.strip():
            lookup[inp] = doc.strip()
    return lookup


def main() -> None:
    data_dir = DEFAULT_DATA_DIR
    raw_path = data_dir / "raw_prompts.jsonl"
    sft_path = data_dir / "sft_train.jsonl"
    complete_path = data_dir / "complete_dataset.jsonl"
    test_path = data_dir / "test.jsonl"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw prompts not found: {raw_path}. Run generate_datasets.py first.")

    raw = load_jsonl(raw_path)
    sft_inputs = set()
    if sft_path.exists():
        for row in load_jsonl(sft_path):
            inp = row.get("input")
            if inp is not None:
                sft_inputs.add(inp)

    doc_lookup = load_docstring_lookup(complete_path)

    # Only use raw prompts whose input is not in SFT set
    eligible = [r for r in raw if r.get("input") not in sft_inputs]
    if len(eligible) < TEST_SIZE:
        n = len(eligible)
        print(f"Warning: only {n} prompts not in SFT; writing {n} to test.jsonl.")
        test_rows = eligible
    else:
        random.shuffle(eligible)
        test_rows = eligible[:TEST_SIZE]

    data_dir.mkdir(parents=True, exist_ok=True)
    missing_ref = 0
    with open(test_path, "w", encoding="utf-8") as f:
        for row in tqdm(test_rows, desc="write test.jsonl", unit="row"):
            inp = row.get("input")
            out: dict = {"input": inp}
            if "reference" in row:
                out["reference"] = row["reference"]
            elif isinstance(inp, str) and inp in doc_lookup:
                out["reference"] = doc_lookup[inp]
            else:
                missing_ref += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {len(test_rows)} rows to {test_path}")
    if doc_lookup and missing_ref:
        print(
            f"Warning: {missing_ref} test rows have no gold docstring in {complete_path}; "
            "Phase 7 similarity metrics need reference or complete data.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
