#!/usr/bin/env python3
"""
Generate data/phase_1/raw_prompts.jsonl via the LLM: Python function sources only.
Each line is JSON: {"input": "<source>", "category": "basic"|"typed"|"edge_cases"}.

One top-level function per API call. Few-shot examples live in few_shot_examples.py.

Each sample targets a length chosen to fill **length bins** evenly (see LENGTH_BINS and stratified logic below).
Validation uses actual line counts: outputs must fall in [MIN_CODE_LINES, MAX_CODE_LINES] and into a bin that
is not yet at quota for that category.

Each validated row is appended immediately (with flush) so progress survives interrupts.
Re-running resumes from existing rows (order is basic, then typed, then edge_cases; 1000 each).

Next steps (docstring pipeline): stream_complete_dataset.py -> build_sft_train.py -> make_dataset.py

Usage: python src/phase_1/generate_datasets.py
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "phase_1"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from few_shot_examples import build_few_shot_messages
from helper import LLMClient

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase_1"

RAW_PER_CATEGORY = 1000
RAW_BATCH_SIZE = 1
# Generated function source length (physical lines in the string, including blanks inside it).
MIN_CODE_LINES = 10
MAX_CODE_LINES = 50
# Inclusive ranges; must partition [MIN_CODE_LINES, MAX_CODE_LINES] with no gaps or overlaps.
LENGTH_BINS: tuple[tuple[int, int], ...] = (
    (10, 19),
    (20, 29),
    (30, 39),
    (40, 50),
)
RETRIES = 2
RETRY_DELAY = 2.0
CALL_DELAY = 1.0
# If stratified acceptance fails this many times in a row (JSON/parse/bin-full rejects), fill the rest without bin quotas.
STRATIFY_STALL_BEFORE_RELAX = 300

CATEGORIES = ("basic", "typed", "edge_cases")


def code_system_text() -> str:
    return f"""You are a data generator for Python training data.
You output only valid JSON: a single JSON array of objects. No markdown, no explanation.
Each object must have exactly: "input" (string, valid Python source for exactly one top-level function: one `def` or `async def`, no class).
Nested functions inside that definition are allowed. Do not emit multiple sibling top-level definitions.
Each "input" must be between {MIN_CODE_LINES} and {MAX_CODE_LINES} physical lines (count physical lines, including blank lines inside the string).
Each user message gives a target line count for that sample — aim for that length while staying within the range above.
Use nested logic, inner defs, validation, branching, and clear structure — not padding or repeated comments.
Do not include docstrings inside the generated code. Generate diverse, realistic examples."""


def build_category_user(category: str, target_lines: int) -> str:
    """User message for one generation: random target_lines drives variety."""
    head = (
        f"Target length for this sample: **{target_lines}** physical lines "
        f"(count newlines inside the string, including blank lines). "
        f"Stay between {MIN_CODE_LINES} and {MAX_CODE_LINES} lines inclusive. No docstrings.\n\n"
    )
    if category == "basic":
        return (
            head
            + "Generate exactly one top-level Python function (no class, no module-level code besides imports if needed).\n"
            + 'Return only valid JSON: one array with one object: [{"input": "..."}].'
        )
    if category == "typed":
        return (
            head
            + "Generate exactly one top-level Python function with type hints (parameters, return type, `typing` imports allowed at the top of the string).\n"
            + 'Return only valid JSON: [{"input": "..."}].'
        )
    if category == "edge_cases":
        return (
            head
            + "Generate exactly one top-level Python function using at least one of: async def, generators (yield), decorators, or contextlib.\n"
            + 'Return only valid JSON: [{"input": "..."}].'
        )
    raise ValueError(f"Unknown category: {category}")


def get_client() -> LLMClient:
    return LLMClient()


def parse_json_array(text: str) -> list:
    """Extract a JSON array from response, handling markdown code blocks."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def call_api_with_retry(
    client: LLMClient,
    system: str,
    user: str,
    few_shot: list[dict] | None = None,
) -> str:
    messages: list[dict] = [{"role": "system", "content": system}]
    if few_shot:
        messages.extend(few_shot)
    messages.append({"role": "user", "content": user})

    model_name = "GPT"
    model_url = os.environ["GPT_OSS_20B_URL"]
    verify_cert = os.environ["MODEL_ACCESS_CERT"]
    temperature = 0.3
    max_tokens = int(os.environ.get("GENERATE_MAX_TOKENS", "4096"))

    for attempt in range(RETRIES + 1):
        try:
            result = client._generate_llm_response(
                model_name=model_name,
                model_url=model_url,
                messages=messages,
                verify_cert=verify_cert,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result or ""
        except Exception:
            if attempt == RETRIES:
                raise
            time.sleep(RETRY_DELAY)

    return ""


def _bin_index_for_line_count(n: int) -> int:
    for i, (lo, hi) in enumerate(LENGTH_BINS):
        if lo <= n <= hi:
            return i
    return -1


def extract_inputs_from_code_batch(items: list) -> list[str]:
    out: list[str] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        inp = obj.get("input")
        if not inp:
            continue
        s = str(inp).strip()
        nlines = len(s.splitlines())
        if MIN_CODE_LINES <= nlines <= MAX_CODE_LINES:
            out.append(s)
    return out


def _validate_length_bins() -> None:
    lo0 = LENGTH_BINS[0][0]
    hi_last = LENGTH_BINS[-1][1]
    if lo0 != MIN_CODE_LINES or hi_last != MAX_CODE_LINES:
        raise ValueError("LENGTH_BINS must span MIN_CODE_LINES..MAX_CODE_LINES")
    for i in range(len(LENGTH_BINS) - 1):
        if LENGTH_BINS[i][1] + 1 != LENGTH_BINS[i + 1][0]:
            raise ValueError("LENGTH_BINS must be contiguous with no gaps")


def quota_per_bin(total_rows: int, n_bins: int) -> list[int]:
    """Split total_rows across bins as evenly as possible (larger bins get +1 first)."""
    base = total_rows // n_bins
    rem = total_rows % n_bins
    return [base + (1 if j < rem else 0) for j in range(n_bins)]


def load_existing_bin_counts(
    path: Path,
    raw_per_category: int,
    categories: tuple[str, ...],
) -> dict[str, list[int]]:
    """Per category, count rows whose actual line count falls in each LENGTH_BIN (file order: cat blocks)."""
    _validate_length_bins()
    n_bins = len(LENGTH_BINS)
    counts: dict[str, list[int]] = {c: [0] * n_bins for c in categories}
    if not path.is_file():
        return counts
    rows: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            inp = row.get("input")
            if isinstance(inp, str):
                rows.append(inp)
    for idx, inp in enumerate(tqdm(rows, desc="resume bin counts", unit="row", leave=False)):
        cat_i = min(idx // raw_per_category, len(categories) - 1)
        cat = categories[cat_i]
        nlines = len(inp.splitlines())
        bi = _bin_index_for_line_count(nlines)
        if bi >= 0:
            counts[cat][bi] += 1
    return counts


def pick_stratified_target(
    bin_counts: list[int],
    bin_quotas: list[int],
) -> tuple[int, int] | None:
    """
    Choose a bin that is under quota (largest deficit first), then a random target line count inside that bin.
    Returns (target_lines, bin_index) or None if all bins full.
    """
    deficits: list[tuple[int, int]] = []  # (deficit, bin_index)
    for j, (q, c) in enumerate(zip(bin_quotas, bin_counts, strict=True)):
        d = q - c
        if d > 0:
            deficits.append((d, j))
    if not deficits:
        return None
    deficits.sort(reverse=True)
    top_def = deficits[0][0]
    candidates = [j for d, j in deficits if d == top_def]
    j = random.choice(candidates)
    lo, hi = LENGTH_BINS[j]
    return random.randint(lo, hi), j


def count_nonempty_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    _validate_length_bins()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_prompts.jsonl"

    total_target = RAW_PER_CATEGORY * len(CATEGORIES)
    existing = count_nonempty_lines(raw_path)
    if existing >= total_target:
        print(f"Already complete ({existing} rows, target {total_target}) at {raw_path}")
        return

    if existing > 0:
        print(f"Resuming: {existing} rows on disk, {total_target - existing} remaining.")

    n_bins = len(LENGTH_BINS)
    bin_quotas = quota_per_bin(RAW_PER_CATEGORY, n_bins)
    bin_counts_by_cat = load_existing_bin_counts(raw_path, RAW_PER_CATEGORY, CATEGORIES)

    client = get_client()
    raw_rows_written = 0
    few_shot = build_few_shot_messages(MIN_CODE_LINES, MAX_CODE_LINES)
    relaxed_warned = False

    file_mode = "a" if existing > 0 else "w"
    with open(raw_path, file_mode, encoding="utf-8") as raw_f:
        for i, category in enumerate(CATEGORIES):
            collected = min(
                RAW_PER_CATEGORY,
                max(0, existing - i * RAW_PER_CATEGORY),
            )
            if collected >= RAW_PER_CATEGORY:
                continue
            bin_counts = bin_counts_by_cat[category]
            stall = 0
            relax = False

            def maybe_relax() -> None:
                nonlocal relax, relaxed_warned
                if stall >= STRATIFY_STALL_BEFORE_RELAX and not relax:
                    relax = True
                    if not relaxed_warned:
                        print(
                            f"Stratified line bins: {stall} stalled attempts for category {category!r}; "
                            "relaxing to any length in range until category is full.",
                            file=sys.stderr,
                        )
                        relaxed_warned = True

            with tqdm(
                total=RAW_PER_CATEGORY,
                initial=collected,
                desc=f"generate [{category}]",
                unit="row",
                leave=True,
            ) as pbar:
                while collected < RAW_PER_CATEGORY:
                    batch_n = min(RAW_BATCH_SIZE, RAW_PER_CATEGORY - collected)
                    picked = pick_stratified_target(bin_counts, bin_quotas)
                    if picked is None:
                        if relax:
                            target_lines = random.randint(MIN_CODE_LINES, MAX_CODE_LINES)
                        else:
                            break
                    else:
                        target_lines, _ = picked
                    code_user = build_category_user(category, target_lines)
                    code_content = call_api_with_retry(
                        client,
                        code_system_text(),
                        code_user,
                        few_shot=few_shot,
                    )
                    time.sleep(CALL_DELAY)
                    try:
                        code_items = parse_json_array(code_content)
                    except (json.JSONDecodeError, TypeError):
                        stall += 1
                        maybe_relax()
                        continue
                    if not isinstance(code_items, list):
                        stall += 1
                        maybe_relax()
                        continue

                    inputs = extract_inputs_from_code_batch(code_items)
                    if len(inputs) > batch_n:
                        inputs = inputs[:batch_n]
                    if not inputs:
                        stall += 1
                        maybe_relax()
                        continue

                    inp = inputs[0]
                    nlines = len(inp.splitlines())
                    bi = _bin_index_for_line_count(nlines)
                    if bi < 0:
                        stall += 1
                        maybe_relax()
                        continue
                    if not relax and bin_counts[bi] >= bin_quotas[bi]:
                        stall += 1
                        maybe_relax()
                        continue

                    stall = 0
                    if collected >= RAW_PER_CATEGORY:
                        break
                    row = {"input": inp, "category": category}
                    raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw_f.flush()
                    raw_rows_written += 1
                    collected += 1
                    bin_counts[bi] += 1
                    pbar.update(1)

    final = existing + raw_rows_written
    print(f"Wrote {raw_rows_written} new rows to {raw_path} (total {final})")


if __name__ == "__main__":
    main()
