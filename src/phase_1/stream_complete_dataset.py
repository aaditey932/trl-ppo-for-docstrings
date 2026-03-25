#!/usr/bin/env python3
"""
Follow data/phase_1/raw_prompts.jsonl as it grows, generate a docstring for each
row's Python `input`, and append completed rows to complete_dataset.jsonl.

Designed to run in parallel while another process appends to raw_prompts.jsonl:
reads only complete lines, tracks file offset in a sidecar state file, and
deduplicates by hash so restarts are safe.

Each output line is JSON with at least:
  - input: original source (unchanged)
  - docstring: generated docstring text (no surrounding quotes)
  - input_with_docstring: same code with the docstring inserted under the first
    top-level function (if parsing succeeds; else null)

Environment (same as generate_datasets.py): GPT_OSS_20B_URL, MODEL_ACCESS_CERT,
KUBEFLOW_OIDC_SCOPE, CCI_AIL_DOCUMENT_HIERARCHY_CLIENTID, PRGX_AZURE_TENANT_ID,
optional CCI_AIL_DOCUMENT_HIERARCHY_SECRET.

Usage:
  python src/phase_1/stream_complete_dataset.py
  python src/phase_1/stream_complete_dataset.py --raw data/phase_1/raw_prompts.jsonl \\
      --out data/phase_1/complete_dataset.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "phase_1"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from helper import LLMClient

DEFAULT_RAW = REPO_ROOT / "data" / "phase_1" / "raw_prompts.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "phase_1" / "complete_dataset.jsonl"
DEFAULT_STATE = REPO_ROOT / "data" / "phase_1" / ".complete_dataset_stream.state"

RETRIES = 2
RETRY_DELAY = 2.0
CALL_DELAY = 0.5
EOF_SLEEP = 0.35
INCOMPLETE_SLEEP = 0.12

DOCSTRING_SYSTEM = """You write concise, accurate Python docstrings for a single top-level function.
The code may contain nested inner functions; document only the outermost function's behavior.
Use Google style: short summary line, then Args/Returns/Raises if useful. Do not repeat the entire implementation.
Output ONLY the docstring body text: no surrounding triple quotes, no markdown fences, no commentary."""

DOCSTRING_USER_TEMPLATE = """Add a docstring for the top-level function in this Python source:

```python
{code}
```
"""


def input_fingerprint(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def load_state(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("raw_offset", 0))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0


def save_state(path: Path, raw_offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"raw_offset": raw_offset}, indent=0) + "\n", encoding="utf-8")


def load_processed_fingerprints(complete_path: Path) -> set[str]:
    seen: set[str] = set()
    if not complete_path.is_file():
        return seen
    with open(complete_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            inp = obj.get("input")
            if isinstance(inp, str):
                seen.add(input_fingerprint(inp))
    return seen


def strip_wrapping_quotes(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if (t.startswith('"""') and t.endswith('"""') and len(t) >= 6) or (
        t.startswith("'''") and t.endswith("'''") and len(t) >= 6
    ):
        t = t[3:-3].strip()
    return t


def merge_docstring(source: str, docstring: str) -> str | None:
    """Insert docstring as first statement of the first module-level function."""
    docstring = docstring.strip()
    if not docstring:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body.insert(0, ast.Expr(ast.Constant(value=docstring)))
            break
    else:
        return None
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def get_client() -> LLMClient:
    return LLMClient()


def call_docstring_llm(client: LLMClient, code: str) -> str:
    user = DOCSTRING_USER_TEMPLATE.format(code=code)
    messages: list[dict] = [
        {"role": "system", "content": DOCSTRING_SYSTEM},
        {"role": "user", "content": user},
    ]
    model_name = "GPT"
    model_url = os.environ["GPT_OSS_20B_URL"]
    verify_cert = os.environ["MODEL_ACCESS_CERT"]
    temperature = 0.2
    max_tokens = int(os.environ.get("DOCSTRING_MAX_TOKENS", "2048"))

    for attempt in range(RETRIES + 1):
        try:
            raw = client._generate_llm_response(
                model_name=model_name,
                model_url=model_url,
                messages=messages,
                verify_cert=verify_cert,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return strip_wrapping_quotes(raw or "")
        except Exception as e:
            if attempt >= RETRIES:
                raise
            print(f"Docstring LLM error (retry {attempt + 1}): {e}", file=sys.stderr)
            time.sleep(RETRY_DELAY)
    return ""


def process_row(
    client: LLMClient,
    row: dict,
    out_f,
    processed: set[str],
    pbar: tqdm | None = None,
) -> None:
    inp = row.get("input")
    if not isinstance(inp, str) or not inp.strip():
        return
    fp = input_fingerprint(inp)
    if fp in processed:
        return
    out_obj: dict = {"input": inp, "docstring": "", "input_with_docstring": None}
    if "label" in row:
        out_obj["label"] = row["label"]
    if "category" in row:
        out_obj["category"] = row["category"]
    try:
        doc = call_docstring_llm(client, inp)
        out_obj["docstring"] = doc
        out_obj["input_with_docstring"] = merge_docstring(inp, doc) if doc else None
    except Exception as e:
        out_obj["error"] = str(e)
    out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
    out_f.flush()
    processed.add(fp)
    if pbar is not None:
        pbar.update(1)
    time.sleep(CALL_DELAY)


def tail_and_process(
    raw_path: Path,
    complete_path: Path,
    state_path: Path,
) -> None:
    client = get_client()
    processed = load_processed_fingerprints(complete_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path.parent.mkdir(parents=True, exist_ok=True)

    start_offset = load_state(state_path)
    print(
        f"Streaming from {raw_path} (start offset {start_offset}), "
        f"{len(processed)} inputs already in {complete_path}",
        file=sys.stderr,
    )

    with open(complete_path, "a", encoding="utf-8") as out_f, tqdm(
        desc="docstrings",
        unit="row",
    ) as pbar:
        while True:
            if not raw_path.is_file():
                time.sleep(EOF_SLEEP)
                continue
            try:
                size = raw_path.stat().st_size
            except OSError:
                time.sleep(EOF_SLEEP)
                continue
            raw_offset = load_state(state_path)
            if raw_offset > size:
                raw_offset = 0
                save_state(state_path, raw_offset)

            with open(raw_path, "r", encoding="utf-8") as raw_f:
                raw_f.seek(raw_offset)
                while True:
                    pos_before = raw_f.tell()
                    line = raw_f.readline()
                    if line == "":
                        save_state(state_path, pos_before)
                        break
                    if not line.endswith("\n"):
                        save_state(state_path, pos_before)
                        time.sleep(INCOMPLETE_SLEEP)
                        break
                    raw_offset = raw_f.tell()
                    save_state(state_path, raw_offset)
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Skip bad JSON at offset {pos_before}: {e}", file=sys.stderr)
                        continue
                    if not isinstance(row, dict):
                        continue
                    process_row(client, row, out_f, processed, pbar=pbar)

            time.sleep(EOF_SLEEP)


def main() -> None:
    p = argparse.ArgumentParser(description="Tail raw_prompts.jsonl and write complete_dataset.jsonl")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Path to raw_prompts.jsonl")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output complete_dataset.jsonl")
    p.add_argument("--state", type=Path, default=DEFAULT_STATE, help="Sidecar JSON with raw file offset")
    args = p.parse_args()
    tail_and_process(args.raw, args.out, args.state)


if __name__ == "__main__":
    main()
