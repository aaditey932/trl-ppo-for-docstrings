#!/usr/bin/env python3
"""
Generate multiple candidate docstring completions per prompt using the SFT model.
Reads data/phase_1/raw_prompts.jsonl, writes data/phase_3/candidates.jsonl.

Gold docstrings for preference scoring are taken from row["reference"] when present,
otherwise from data/phase_1/complete_dataset.jsonl keyed by input (same as build_sft_train).

Each output row: {"prompt": "...", "candidates": [...], "input": "...", "reference": "...",
"source": "human"|"gpt"} — first half of rows human, second half gpt (500/500 when N=1000).
Requires outputs/sft_policy from Phase 2.

Use --max-prompts N to process only N prompts with an even split across categories
(basic / typed / edge_cases). Rows without "category" infer category from file order
using the same block size as Phase 1 (see CATEGORY_BLOCK_SIZE).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from config import policy_from_pretrained_kwargs, tokenizer_pretrained_kwargs
from utils import make_prompt

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "phase_1"
DEFAULT_SFT_DIR = REPO_ROOT / "outputs" / "sft_policy"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase_3"
DEFAULT_TEMPERATURES = "0.4,0.7,1.0"
DEFAULT_MAX_NEW_TOKENS = 192

# Must match src/phase_1/generate_datasets.py RAW_PER_CATEGORY when "category" is missing on a row.
CATEGORY_BLOCK_SIZE = 1000
CATEGORIES = ("basic", "typed", "edge_cases")


def infer_category(row: dict, line_index: int) -> str:
    c = row.get("category")
    if isinstance(c, str) and c in CATEGORIES:
        return c
    return CATEGORIES[min(line_index // CATEGORY_BLOCK_SIZE, len(CATEGORIES) - 1)]


def even_split_counts(n: int, n_parts: int) -> list[int]:
    """Split n into n_parts buckets; sizes differ by at most 1."""
    if n_parts <= 0:
        return []
    base = n // n_parts
    rem = n % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def interleave_lists(lists: list[list[dict]]) -> list[dict]:
    """Round-robin merge (b0, t0, e0, b1, t1, e1, ...)."""
    out: list[dict] = []
    max_len = max((len(x) for x in lists), default=0)
    for i in range(max_len):
        for lst in lists:
            if i < len(lst):
                out.append(lst[i])
    return out


def select_rows_even_by_category(rows: list[dict], max_prompts: int | None) -> tuple[list[dict], str]:
    """
    If max_prompts is None, return all rows unchanged.
    Otherwise take up to max_prompts rows with (almost) equal counts per category, interleaved.
    """
    if max_prompts is None:
        return rows, "all rows (no cap)"

    if max_prompts <= 0:
        return [], "0 requested"

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_cat[infer_category(row, idx)].append(row)

    quotas = even_split_counts(max_prompts, len(CATEGORIES))
    per_cat_selected: list[list[dict]] = []
    shortfall_msgs: list[str] = []

    for cat, q in zip(CATEGORIES, quotas):
        pool = by_cat[cat]
        if len(pool) < q:
            shortfall_msgs.append(f"{cat}: want {q}, have {len(pool)}")
        per_cat_selected.append(pool[:q])

    selected = interleave_lists(per_cat_selected)
    note = f"even split target {max_prompts} -> quotas {dict(zip(CATEGORIES, quotas))}"
    if shortfall_msgs:
        note += "; " + "; ".join(shortfall_msgs)
    return selected, note


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


def load_docstring_lookup(path: Path) -> dict[str, str]:
    """Map function source string -> gold docstring (same filters as build_sft_train)."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("error"):
                continue
            inp = row.get("input")
            doc = row.get("docstring")
            if not isinstance(inp, str) or not inp.strip():
                continue
            if not isinstance(doc, str) or not doc.strip():
                continue
            out[inp] = doc.strip()
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate candidate docstrings per prompt (SFT model).",
    )
    p.add_argument(
        "--max-prompts",
        "-n",
        type=int,
        default=1000,
        metavar="N",
        help="Process at most N prompts, split evenly across basic / typed / edge_cases "
        "(remainder goes to the first categories). Default: 1000. Use 0 or a negative value "
        "for all rows in raw_prompts.jsonl.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing raw_prompts.jsonl and complete_dataset.jsonl",
    )
    p.add_argument(
        "--sft-dir",
        type=Path,
        default=DEFAULT_SFT_DIR,
        help="SFT policy directory (outputs/sft_policy)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for candidates.jsonl",
    )
    p.add_argument(
        "--temperatures",
        type=str,
        default=DEFAULT_TEMPERATURES,
        help="Comma-separated sampling temperatures",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Generation max_new_tokens",
    )
    args = p.parse_args()

    data_dir = args.data_dir
    sft_dir = args.sft_dir
    output_dir = args.out_dir
    max_prompts = args.max_prompts if args.max_prompts > 0 else None
    temperatures_str = args.temperatures
    max_new_tokens = args.max_new_tokens

    raw_path = data_dir / "raw_prompts.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw prompts not found: {raw_path}. Run Phase 1 first.")

    if not sft_dir.exists():
        raise FileNotFoundError(f"SFT model not found: {sft_dir}. Run Phase 2 first.")

    tok_kw = tokenizer_pretrained_kwargs()
    model_kw = policy_from_pretrained_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(str(sft_dir), **tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(sft_dir), **model_kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_rows = load_jsonl(raw_path)
    raw_rows, selection_note = select_rows_even_by_category(all_rows, max_prompts)
    print(f"Selection: {selection_note} (loaded {len(all_rows)} rows from raw)", flush=True)

    complete_path = data_dir / "complete_dataset.jsonl"
    ref_lookup = load_docstring_lookup(complete_path)

    temperatures = [float(t.strip()) for t in temperatures_str.split(",") if t.strip()]
    if not temperatures:
        temperatures = [0.4, 0.7, 1.0]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "candidates.jsonl"
    n_total = len(raw_rows)
    print(f"Generating candidates for {n_total} prompts -> {out_path}", flush=True)

    def source_for_row_index(i: int, n: int) -> str:
        """First half of rows human, second half gpt (500/500 when n==1000)."""
        return "human" if i < (n // 2) else "gpt"

    with open(out_path, "w") as out_f:
        for i, row in enumerate(tqdm(raw_rows, desc="prompts", unit="prompt", total=n_total)):
            user_input = row.get("input", "")
            reference = row.get("reference", "") or ""
            if not reference and isinstance(user_input, str):
                reference = ref_lookup.get(user_input, "")
            full_prompt = make_prompt(user_input)

            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            input_length = inputs["input_ids"].shape[1]
            inputs = {k: v.to(device) for k, v in inputs.items()}

            candidates = []
            seen = set()
            with torch.no_grad():
                for temp in temperatures:
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temp,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        top_p=0.95,
                    )
                    # Decode only the generated part
                    new_tokens = out[0][input_length:]
                    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    # Dedupe by normalized content
                    key = completion[:200]
                    if key not in seen:
                        seen.add(key)
                        candidates.append(completion)

            record = {
                "prompt": full_prompt,
                "candidates": candidates,
                "input": user_input,
                "reference": reference,
                "source": source_for_row_index(i, n_total),
            }
            cat = row.get("category")
            if isinstance(cat, str) and cat:
                record["category"] = cat
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {n_total} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
