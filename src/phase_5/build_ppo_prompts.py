#!/usr/bin/env python3
"""
Phase 5: Build prompt-only PPO dataset from preference pairs.

Reads data/phase_3/preference_pairs.jsonl, collects unique prompts, and writes
data/phase_5/ppo_prompts.jsonl with one row per prompt: {"prompt": "..."}.

Requires Phase 3 data. Run from repo root: python src/phase_5/build_ppo_prompts.py
"""

import json
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PAIRS_DIR = REPO_ROOT / "data" / "phase_3"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase_5"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
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
    pairs_dir = DEFAULT_PAIRS_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    pairs_path = pairs_dir / "preference_pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Preference pairs not found: {pairs_path}. Run Phase 3 first (generate_candidates.py, build_preference_pairs.py)."
        )

    rows = load_jsonl(pairs_path)
    if not rows:
        raise ValueError(f"No valid rows in {pairs_path}.")

    # Collect unique prompts (preserve order of first occurrence)
    seen: set[str] = set()
    unique_prompts: list[str] = []
    for r in rows:
        prompt = r.get("prompt")
        if prompt is not None and prompt not in seen:
            seen.add(prompt)
            unique_prompts.append(prompt)

    if not unique_prompts:
        raise ValueError(f"No prompts found in {pairs_path}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ppo_prompts.jsonl"
    with open(out_path, "w") as f:
        for prompt in tqdm(unique_prompts, desc="ppo prompts", unit="prompt"):
            f.write(json.dumps({"prompt": prompt}) + "\n")

    print(f"Wrote {len(unique_prompts)} unique prompts to {out_path}")


if __name__ == "__main__":
    main()
