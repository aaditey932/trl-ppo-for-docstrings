#!/usr/bin/env python3
"""
Build data/phase_1/sft_train.jsonl from data/phase_1/complete_dataset.jsonl.

Each output row: input, response, reference; optional category (basic / typed / edge_cases) when present
in complete_dataset.jsonl.
Rows are skipped when docstring is missing/empty or when the complete row has an "error" field (LLM failure).

Run after stream_complete_dataset.py and before train_sft.py / make_dataset.py.

Usage: python src/phase_1/build_sft_train.py
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "phase_1"


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


def main() -> None:
    p = argparse.ArgumentParser(description="Build sft_train.jsonl from complete_dataset.jsonl")
    p.add_argument(
        "--complete",
        type=Path,
        default=DEFAULT_DATA_DIR / "complete_dataset.jsonl",
        help="Path to complete_dataset.jsonl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATA_DIR / "sft_train.jsonl",
        help="Output path for SFT JSONL",
    )
    args = p.parse_args()

    if not args.complete.exists():
        raise FileNotFoundError(
            f"Complete dataset not found: {args.complete}. Run stream_complete_dataset.py first."
        )

    rows = load_jsonl(args.complete)
    written = 0
    skipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="build sft_train", unit="row"):
            if not isinstance(row, dict):
                skipped += 1
                continue
            if row.get("error"):
                skipped += 1
                continue
            inp = row.get("input")
            doc = row.get("docstring")
            if not isinstance(inp, str) or not inp.strip():
                skipped += 1
                continue
            if not isinstance(doc, str) or not doc.strip():
                skipped += 1
                continue
            out_row = {
                "input": inp,
                "response": doc.strip(),
                "reference": doc.strip(),
            }
            cat = row.get("category")
            if isinstance(cat, str) and cat:
                out_row["category"] = cat
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows to {args.out} (skipped {skipped})", file=sys.stderr)


if __name__ == "__main__":
    main()
