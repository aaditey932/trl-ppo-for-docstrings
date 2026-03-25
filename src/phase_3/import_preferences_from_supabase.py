#!/usr/bin/env python3
"""
Import preference rows from a Supabase table into data/phase_3/preference_pairs.jsonl.

Expects columns for function source (passed through make_prompt), chosen docstring, and
rejected docstring. Writes TRL rows: {"prompt", "chosen", "rejected"}.

Environment (same pattern as Phase 1):
  SUPABASE_URL   — project URL
  SUPABASE_KEY   — service role or anon key with SELECT on the table

Run from repo root:
  python src/phase_3/import_preferences_from_supabase.py --table preference_pairs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

load_dotenv(REPO_ROOT / ".env")

from supabase import Client, create_client

from utils import make_prompt

DEFAULT_OUTPUT = REPO_ROOT / "data" / "phase_3" / "preference_pairs.jsonl"
PAGE_SIZE = 1000


def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        print(
            "Set SUPABASE_URL and SUPABASE_KEY in the environment or .env",
            file=sys.stderr,
        )
        sys.exit(1)
    return create_client(url, key)


def select_columns(input_col: str, chosen_col: str, rejected_col: str) -> str:
    # Quote-like identifiers not needed for simple names; comma-separated for PostgREST
    return f"{input_col},{chosen_col},{rejected_col}"


def iter_supabase_rows(
    client: Client,
    table: str,
    select_str: str,
    *,
    page_size: int,
    start_offset: int,
    max_rows: int | None,
):
    """Yield rows using inclusive range pagination (PostgREST)."""
    fetched = 0
    db_start = start_offset
    while True:
        remaining = None if max_rows is None else max_rows - fetched
        if remaining is not None and remaining <= 0:
            break
        take = page_size if remaining is None else min(page_size, remaining)
        db_end = db_start + take - 1
        resp = client.table(table).select(select_str).range(db_start, db_end).execute()
        batch = resp.data or []
        if not batch:
            break
        for row in batch:
            yield row
            fetched += 1
            if max_rows is not None and fetched >= max_rows:
                return
        if len(batch) < take:
            break
        db_start += len(batch)


def row_key(row: dict) -> tuple[str, str, str]:
    return (row["prompt"], row["chosen"], row["rejected"])


def load_existing_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not path.is_file():
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            p, c, r = obj.get("prompt"), obj.get("chosen"), obj.get("rejected")
            if isinstance(p, str) and isinstance(c, str) and isinstance(r, str):
                keys.add((p, c, r))
    return keys


def transform_row(
    raw: dict,
    input_col: str,
    chosen_col: str,
    rejected_col: str,
) -> dict | None:
    src = raw.get(input_col)
    chosen = raw.get(chosen_col)
    rejected = raw.get(rejected_col)
    if src is None or chosen is None or rejected is None:
        return None
    if not isinstance(src, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    src = src.strip()
    chosen = chosen.strip()
    rejected = rejected.strip()
    if not src or not chosen or not rejected:
        return None
    if chosen == rejected:
        return None
    return {
        "prompt": make_prompt(src),
        "chosen": chosen,
        "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Supabase → preference_pairs.jsonl")
    parser.add_argument("--table", required=True, help="Supabase table name")
    parser.add_argument(
        "--input-column",
        default="input",
        help="Column with Python function source for make_prompt (default: input)",
    )
    parser.add_argument("--chosen-column", default="chosen", help="Preferred docstring body")
    parser.add_argument("--rejected-column", default="rejected", help="Dispreferred docstring body")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--mode",
        choices=("append", "write"),
        default="append",
        help="append: merge with existing file (dedupe); write: only Supabase rows",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many rows at the start of the table (pagination offset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Import at most this many rows from Supabase (after offset)",
    )
    args = parser.parse_args()

    client = get_client()
    select_str = select_columns(args.input_column, args.chosen_column, args.rejected_column)

    existing_keys: set[tuple[str, str, str]] = set()
    existing_rows: list[dict] = []
    if args.mode == "append" and args.output.is_file():
        existing_keys = load_existing_keys(args.output)
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    new_rows: list[dict] = []
    n_fetched = 0
    n_skipped = 0

    for raw in iter_supabase_rows(
        client,
        args.table,
        select_str,
        page_size=PAGE_SIZE,
        start_offset=args.offset,
        max_rows=args.limit,
    ):
        n_fetched += 1
        out = transform_row(raw, args.input_column, args.chosen_column, args.rejected_column)
        if out is None:
            n_skipped += 1
            continue
        key = row_key(out)
        if key in existing_keys:
            n_skipped += 1
            continue
        existing_keys.add(key)
        new_rows.append(out)

    if args.mode == "write":
        to_write = new_rows
    else:
        to_write = existing_rows + new_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in to_write:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Supabase rows read: {n_fetched}, "
        f"new TRL rows added: {len(new_rows)}, "
        f"skipped (invalid or duplicate): {n_skipped}, "
        f"total lines written: {len(to_write)} → {args.output}"
    )


if __name__ == "__main__":
    main()
