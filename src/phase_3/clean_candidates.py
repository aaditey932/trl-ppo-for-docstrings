"""
Validate and repair candidates.jsonl rows so build_preference_pairs.py can judge each row.

Repairs (when possible):
- Missing/empty ``prompt`` but valid ``input`` -> ``make_prompt(input)``
- ``candidates`` not a list -> normalized to list
- Fewer than 2 unique string candidates -> append ``reference`` if distinct
- Still < 2 -> append FALLBACK_SECOND_CANDIDATE (clearly distinct placeholder)

Also caps at 26 unique candidates (judge label limit).

CLI: python src/phase_3/candidates_preflight.py [--fix] [-i path] [-o path]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils import make_prompt

# Last resort when model produced a single duplicate; must differ from normal docstrings.
FALLBACK_SECOND_CANDIDATE = (
    "[Placeholder] No second model completion available; compare against this stub."
)
FALLBACK_ALTERNATE = (
    "[Alternate stub] Minimal placeholder docstring for pairwise comparison only."
)

MAX_JUDGE_CANDIDATES = 26


def dedupe_candidate_strings(candidates: object) -> list[str]:
    """Same dedupe as build_preference_pairs: unique strings by first 500 chars of strip."""
    uniq: list[str] = []
    seen: set[str] = set()
    for c in candidates or []:
        if not isinstance(c, str):
            continue
        key = c.strip()[:500]
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def repair_row_for_judge(row: dict) -> tuple[dict, list[str]]:
    """
    Return a copy of row with prompt/candidates adjusted so dedupe_candidate_strings
    yields at least 2 strings when possible, and at most MAX_JUDGE_CANDIDATES.
    """
    fixes: list[str] = []
    out = dict(row)

    inp = out.get("input")
    if not isinstance(inp, str):
        inp = ""
        out["input"] = inp
        if row.get("input") is not None:
            fixes.append("coerced_input_to_str")

    prompt = out.get("prompt") or ""
    if not str(prompt).strip() and inp.strip():
        out["prompt"] = make_prompt(inp)
        fixes.append("filled_prompt_from_input")

    cands = out.get("candidates")
    if not isinstance(cands, list):
        out["candidates"] = [] if cands is None else list(cands)
        fixes.append("normalized_candidates_to_list")
        cands = out["candidates"]

    def append_if_distinct(text: str, label: str) -> None:
        nonlocal cands
        if not isinstance(text, str) or not text.strip():
            return
        keys = {c.strip()[:500] for c in dedupe_candidate_strings(cands)}
        if text.strip()[:500] not in keys:
            cands = list(cands) + [text.strip()]
            out["candidates"] = cands
            fixes.append(label)

    uniq = dedupe_candidate_strings(out["candidates"])
    if len(uniq) < 2:
        ref = out.get("reference")
        if isinstance(ref, str) and ref.strip():
            append_if_distinct(ref, "appended_reference_as_candidate")
    uniq = dedupe_candidate_strings(out["candidates"])
    if len(uniq) < 2:
        append_if_distinct(FALLBACK_SECOND_CANDIDATE, "appended_fallback_second_candidate")
    uniq = dedupe_candidate_strings(out["candidates"])
    if len(uniq) < 2:
        append_if_distinct(FALLBACK_ALTERNATE, "appended_fallback_alternate_candidate")

    uniq = dedupe_candidate_strings(out["candidates"])
    if len(uniq) > MAX_JUDGE_CANDIDATES:
        # Keep first 26 unique strings in order of first appearance
        kept: list[str] = []
        seen: set[str] = set()
        for c in out["candidates"]:
            if not isinstance(c, str):
                continue
            key = c.strip()[:500]
            if key in seen:
                continue
            seen.add(key)
            kept.append(c)
            if len(kept) >= MAX_JUDGE_CANDIDATES:
                break
        out["candidates"] = kept
        fixes.append(f"truncated_to_{MAX_JUDGE_CANDIDATES}_candidates")

    return out, fixes


def load_jsonl_with_line_errors(path: Path) -> tuple[list[dict], list[tuple[int, str]]]:
    """Parse JSONL; return (rows, list of (line_number, error) for bad lines)."""
    rows: list[dict] = []
    errors: list[tuple[int, str]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append((line_num, str(e)))
    return rows, errors


def scan_rows(rows: list[dict]) -> dict[str, list[int]]:
    """Classify rows by issue (0-based indices). Does not apply repairs."""
    from collections import defaultdict

    bad: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        inp = row.get("input")
        if not isinstance(inp, str):
            inp = ""
        if not inp.strip():
            bad["empty_input"].append(i)
        prompt = row.get("prompt") or ""
        if not str(prompt).strip() and inp.strip():
            bad["empty_prompt_fixable"].append(i)
        elif not str(prompt).strip():
            bad["empty_prompt_unfixable"].append(i)
        u = dedupe_candidate_strings(row.get("candidates"))
        if len(u) < 2:
            bad["lt2_unique_candidates_before_repair"].append(i)
    return dict(bad)


def main() -> None:
    p = argparse.ArgumentParser(description="Scan or fix candidates.jsonl for preference judging.")
    p.add_argument("-i", "--input", type=Path, default=REPO_ROOT / "data" / "phase_3" / "candidates.jsonl")
    p.add_argument("-o", "--output", type=Path, default=None, help="Write fixed JSONL (default with --fix: overwrite -i)")
    p.add_argument("--fix", action="store_true", help="Apply repairs and write output")
    args = p.parse_args()

    path = args.input
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows, json_errs = load_jsonl_with_line_errors(path)
    if json_errs:
        print(f"Invalid JSON on {len(json_errs)} line(s):", file=sys.stderr)
        for ln, msg in json_errs[:50]:
            print(f"  line {ln}: {msg}", file=sys.stderr)
        if len(json_errs) > 50:
            print(f"  ... and {len(json_errs) - 50} more", file=sys.stderr)
        if not args.fix:
            sys.exit(1)

    if not rows:
        print("No rows parsed.", file=sys.stderr)
        sys.exit(1)

    report = scan_rows(rows)
    print("Scan (0-based row indices, before repair):")
    if not report:
        print("  (no issues)")
    else:
        for k, v in sorted(report.items()):
            print(f"  {k}: {len(v)} {v[:15]}{'...' if len(v) > 15 else ''}")

    if not args.fix:
        return

    out_path = args.output if args.output is not None else path
    fixed_rows: list[dict] = []
    all_fixes = 0
    for row in rows:
        new_row, fixes = repair_row_for_judge(row)
        fixed_rows.append(new_row)
        all_fixes += len(fixes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in fixed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(fixed_rows)} rows to {out_path} ({all_fixes} repair tags applied).")


if __name__ == "__main__":
    main()
