#!/usr/bin/env python3
"""
Build data/phase_3/preference_pairs.jsonl from candidates.jsonl.

For each row with at least two unique candidate docstrings, calls GPT OSS (via LLMClient)
as a blind judge: function source + labeled candidates only (no gold reference).
The model returns JSON with chosen_label and rejected_label; we map those to strings
and write TRL rows {"prompt", "chosen", "rejected"}.

Requires data/phase_3/candidates.jsonl from generate_candidates.py.
Same API env as Phase 1 (GPT_OSS_20B_URL, MODEL_ACCESS_CERT, MSAL vars for LLMClient).

Run from repo root: python src/phase_3/build_preference_pairs.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "phase_1"))
sys.path.insert(0, str(REPO_ROOT / "src" / "phase_3"))
from candidates_preflight import (
    dedupe_candidate_strings,
    load_jsonl_with_line_errors,
    repair_row_for_judge,
)
from helper import LLMClient

DEFAULT_INPUT = REPO_ROOT / "data" / "phase_3" / "candidates.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "phase_3" / "preference_pairs.jsonl"

JUDGE_TEMPERATURE = 0.2
JUDGE_MAX_TOKENS = 512
JUDGE_RETRIES = 3
CALL_DELAY = 0.5
MAX_CODE_CHARS = 16000
MAX_CANDIDATE_CHARS = 12000

LABEL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalize_key(s: str) -> str:
    return " ".join(s.split()).strip().lower()


def truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def extract_json_object(text: str) -> dict | None:
    """Parse a JSON object from model output, allowing markdown fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def build_label_map(uniq: list[str]) -> tuple[list[str], dict[str, str]]:
    """Return list of labels and label -> candidate text."""
    if len(uniq) > len(LABEL_CHARS):
        raise ValueError(f"At most {len(LABEL_CHARS)} candidates supported, got {len(uniq)}")
    labels = [LABEL_CHARS[i] for i in range(len(uniq))]
    return labels, dict(zip(labels, uniq))


def judge_messages(code: str, labels: list[str], label_to_text: dict[str, str]) -> tuple[list[dict], dict[str, str]]:
    """Build chat messages and return (messages, label_to_text) for the API call."""
    code_block = truncate(code.strip(), MAX_CODE_CHARS)
    parts = [
        "You compare Python docstring candidates for the single top-level function below.",
        "Which candidate better describes the function's behavior, parameters, and return value?",
        "Which candidate is worse (less accurate, vague, or misleading)?",
        "",
        "Respond with ONLY valid JSON (no markdown), exactly this shape:",
        '{"chosen_label":"<LETTER>","rejected_label":"<LETTER>"}',
        "where LETTERS are from the labels below. chosen_label must differ from rejected_label.",
        "",
        "Function source:",
        "```python",
        code_block,
        "```",
        "",
        "Candidates:",
    ]
    for lab in labels:
        body = truncate(label_to_text[lab], MAX_CANDIDATE_CHARS)
        parts.append(f"{lab}:\n{body}\n")

    system = (
        "You are an expert Python reviewer. Judge docstring quality for correctness and clarity "
        "against the function source only. Output only the requested JSON object."
    )
    user = "\n".join(parts)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages, label_to_text


def call_judge(
    client: LLMClient,
    code: str,
    uniq: list[str],
) -> tuple[str, str] | None:
    """Return (chosen, rejected) docstrings or None if judge fails after retries."""
    if len(uniq) > len(LABEL_CHARS):
        uniq = uniq[: len(LABEL_CHARS)]
    labels, label_to_text = build_label_map(uniq)
    messages, _ = judge_messages(code, labels, label_to_text)

    model_url = os.environ["GPT_OSS_20B_URL"]
    verify_cert = os.environ["MODEL_ACCESS_CERT"]

    for attempt in range(JUDGE_RETRIES):
        try:
            raw = client._generate_llm_response(
                model_name="GPT",
                model_url=model_url,
                messages=messages,
                verify_cert=verify_cert,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception:
            if attempt + 1 >= JUDGE_RETRIES:
                return None
            time.sleep(2.0 * (attempt + 1))
            continue

        obj = extract_json_object(raw or "")
        if not obj:
            time.sleep(0.5 * (attempt + 1))
            continue

        c_lab = obj.get("chosen_label")
        r_lab = obj.get("rejected_label")
        if not isinstance(c_lab, str) or not isinstance(r_lab, str):
            time.sleep(0.5 * (attempt + 1))
            continue

        c_lab = c_lab.strip().upper()[:1]
        r_lab = r_lab.strip().upper()[:1]
        if c_lab not in label_to_text or r_lab not in label_to_text:
            time.sleep(0.5 * (attempt + 1))
            continue

        chosen = label_to_text[c_lab]
        rejected = label_to_text[r_lab]
        if normalize_key(chosen) == normalize_key(rejected):
            time.sleep(0.5 * (attempt + 1))
            continue

        return chosen, rejected

    return None


def main() -> None:
    input_path = DEFAULT_INPUT
    output_path = DEFAULT_OUTPUT

    if not input_path.exists():
        raise FileNotFoundError(
            f"Candidates not found: {input_path}. Run generate_candidates.py first."
        )

    rows, json_line_errors = load_jsonl_with_line_errors(input_path)
    if json_line_errors:
        print("Invalid JSON (fix or remove these lines before continuing):", file=sys.stderr)
        for line_no, msg in json_line_errors[:100]:
            print(f"  line {line_no}: {msg}", file=sys.stderr)
        if len(json_line_errors) > 100:
            print(f"  ... and {len(json_line_errors) - 100} more", file=sys.stderr)
        raise SystemExit(1)
    if not rows:
        raise ValueError(f"No rows in {input_path}.")

    client = LLMClient()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    skip_reasons: dict[str, int] = {
        "empty_prompt": 0,
        "empty_input": 0,
        "lt2_candidates_after_repair": 0,
        "judge_failed": 0,
    }

    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in tqdm(rows, desc="preference pairs", unit="row"):
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

            pair = call_judge(client, user_input, uniq)
            if pair is None:
                skip_reasons["judge_failed"] += 1
                continue

            chosen, rejected = pair
            out_f.write(
                json.dumps(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1
            time.sleep(CALL_DELAY)

    n_skipped = sum(skip_reasons.values())
    print(f"Wrote {n_written} preference pairs to {output_path} (skipped {n_skipped}).")
    for reason, count in skip_reasons.items():
        if count:
            print(f"  skipped {reason}: {count}")


if __name__ == "__main__":
    main()
