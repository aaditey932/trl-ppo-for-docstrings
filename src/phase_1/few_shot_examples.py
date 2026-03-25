"""
Few-shot prompt/response pairs for generate_datasets.

Each example is exactly one top-level function (one `def` or `async def`); nested defs inside
are allowed. Assistant turns are JSON arrays of one object: [{"input": "<source>"}].
Examples are kept under MAX_LINE_LENGTH used by generate_datasets so the model sees feasible sizes.
"""

from __future__ import annotations

import json

# --- Example 1: plain Python, shorter function ------------------------------------------------

EXAMPLE_FUNCTION_PLAIN = '''def partition_records(rows: list, key: str, min_val: float) -> dict:
    def _ok(r):
        if not isinstance(r, dict):
            return False
        if key not in r:
            return False
        try:
            return float(r[key]) >= min_val
        except (TypeError, ValueError):
            return False

    def _split(items):
        hi, lo = [], []
        for r in items:
            if _ok(r):
                hi.append(dict(r))
            else:
                lo.append(dict(r))
        return hi, lo

    if not isinstance(rows, list):
        return {"error": "rows must be a list", "above": [], "below": []}
    above, below = _split(rows)
    return {
        "above": above,
        "below": below,
        "counts": {"above": len(above), "below": len(below)},
    }'''

# --- Example 2: typed -------------------------------------------------------------------------

EXAMPLE_FUNCTION_TYPED = '''from typing import Any, Dict, Iterable, List, Optional, Tuple

def normalize_tags(
    items: List[Dict[str, Any]],
    tag_key: str = "tag",
    default: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    seen: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []

    def _norm(s: Any) -> str:
        if s is None:
            return (default or "").strip().lower()
        return str(s).strip().lower()

    for it in items:
        if not isinstance(it, dict):
            continue
        raw = it.get(tag_key)
        t = _norm(raw)
        if not t:
            continue
        seen[t] = seen.get(t, 0) + 1
        row = dict(it)
        row[tag_key] = t
        out.append(row)

    return out, dict(sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])))'''

# --- Example 3: async -------------------------------------------------------------------------

EXAMPLE_FUNCTION_ASYNC = '''import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

async def take_first_n(
    stream: AsyncIterator[int],
    n: int,
    timeout: float = 1.0,
) -> Dict[str, Any]:
    buf: List[int] = []
    err: Optional[str] = None

    async def _drain() -> None:
        nonlocal err
        try:
            async for x in stream:
                buf.append(int(x))
                if len(buf) >= n:
                    break
        except asyncio.TimeoutError:
            err = "timeout"
        except Exception as e:
            err = str(e)

    try:
        await asyncio.wait_for(_drain(), timeout=timeout)
    except asyncio.TimeoutError:
        err = err or "timeout"
    return {
        "values": buf[:n],
        "taken": min(len(buf), n),
        "error": err,
    }'''


def pack_one_function_json(code: str) -> str:
    return json.dumps([{"input": code}], ensure_ascii=False)


def build_few_shot_messages(min_lines: int, max_lines: int) -> list[dict]:
    """
    Three user/assistant turns: each assistant message is a JSON array with one object,
    whose "input" is a single function's source.
    """
    span = f"between {min_lines} and {max_lines} physical lines (including blank lines)"
    u1 = (
        "Generate exactly one top-level Python function (no class). "
        "No docstrings. Use nested logic and inner defs if helpful. "
        f'Return only valid JSON: one array with one object: [{{"input": "..."}}]. '
        f"The function source must be {span}."
    )
    u2 = (
        "Generate exactly one top-level Python function with type hints (from typing if needed). "
        f'Same JSON shape. The function source must be {span}.'
    )
    u3 = (
        "Generate exactly one top-level async function (async def). "
        f'Same JSON shape. The function source must be {span}.'
    )
    return [
        {"role": "user", "content": u1},
        {"role": "assistant", "content": pack_one_function_json(EXAMPLE_FUNCTION_PLAIN)},
        {"role": "user", "content": u2},
        {"role": "assistant", "content": pack_one_function_json(EXAMPLE_FUNCTION_TYPED)},
        {"role": "user", "content": u3},
        {"role": "assistant", "content": pack_one_function_json(EXAMPLE_FUNCTION_ASYNC)},
    ]
