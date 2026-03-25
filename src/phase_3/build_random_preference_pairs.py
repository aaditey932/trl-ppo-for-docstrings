#!/usr/bin/env python3
"""
Build data/phase_3/preference_pairs.jsonl from candidates.jsonl without an LLM judge:
for each row with ≥2 unique candidates (after the same repair as build_preference_pairs),
pick two distinct candidates at random and assign chosen/rejected at random.

Run from repo root: python src/phase_3/build_random_preference_pairs.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "phase_3"))

from clean_candidates import dedupe_candidate_strings, load_jsonl_with_line_errors, repair_row_for_judge

DEFAULT_INPUT = REPO_ROOT / "data" / "phase_3" / "candidates.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "phase_3" / "preference_pairs.jsonl"


def random_pair(uniq: list[str], rng: random.Random) -> tuple[str, str]:
    a, b = rng.sample(uniq, 2)
    if rng.random() < 0.5:
        return a, b
    return b, a


def main() -> None:
    p = argparse.ArgumentParser(description="Random chosen/rejected pairs from candidates.jsonl.")
    p.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    args = p.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Candidates not found: {args.input}")

    rows, json_line_errors = load_jsonl_with_line_errors(args.input)
    if json_line_errors:
        print("Invalid JSON (fix or remove these lines before continuing):", file=sys.stderr)
        for line_no, msg in json_line_errors[:100]:
            print(f"  line {line_no}: {msg}", file=sys.stderr)
        raise SystemExit(1)
    if not rows:
        raise ValueError(f"No rows in {args.input}.")

    rng = random.Random(args.seed)
    n_written = 0
    skip_reasons: dict[str, int] = {
        "empty_prompt": 0,
        "empty_input": 0,
        "lt2_candidates_after_repair": 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for row in rows:
            row, _ = repair_row_for_judge(row)
            prompt = row.get("prompt", "")
            user_input = row.get("input", "")
            if not isinstance(user_input, str):
                user_input = ""

            if not user_input.strip():
                skip_reasons["empty_input"] += 1
                continue
            if not prompt or not str(prompt).strip():
                skip_reasons["empty_prompt"] += 1
                continue

            uniq = dedupe_candidate_strings(row.get("candidates"))
            if len(uniq) < 2:
                skip_reasons["lt2_candidates_after_repair"] += 1
                continue

            chosen, rejected = random_pair(uniq, rng)
            out_f.write(
                json.dumps(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(f"Wrote {n_written} preference pairs to {args.output}.")
    for reason, count in skip_reasons.items():
        if count:
            print(f"  skipped {reason}: {count}")


if __name__ == "__main__":
    main()
