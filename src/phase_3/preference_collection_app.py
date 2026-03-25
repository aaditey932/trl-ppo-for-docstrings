#!/usr/bin/env python3
"""
Streamlit UI: show random function code + two docstring candidates; save chosen/rejected to Supabase.

Uses the same Supabase row shape as export_preferences_to_supabase.py:
  input    — Python function source
  chosen   — preferred docstring body
  rejected — dispreferred docstring body

Environment (see .env):
  SUPABASE_URL
  SUPABASE_KEY

Run from repo root:
  streamlit run src/phase_3/preference_collection_app.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from supabase import Client, create_client

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "phase_3"))

load_dotenv(REPO_ROOT / ".env")

from clean_candidates import dedupe_candidate_strings, load_jsonl_with_line_errors, repair_row_for_judge

DEFAULT_CANDIDATES = REPO_ROOT / "data" / "phase_3" / "candidates.jsonl"
DEFAULT_TABLE = "preferences"


def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_KEY in the environment or .env"
        )
    return create_client(url, key)


@st.cache_data(show_spinner="Loading candidates…")
def load_eligible_rows(candidates_path: str) -> list[dict]:
    """Rows with non-empty input and at least two unique candidates after repair."""
    path = Path(candidates_path)
    rows, json_line_errors = load_jsonl_with_line_errors(path)
    if json_line_errors:
        raise ValueError(
            "Invalid JSON in candidates file: "
            + ", ".join(f"line {n}: {m}" for n, m in json_line_errors[:5])
        )
    eligible: list[dict] = []
    for row in rows:
        row, _ = repair_row_for_judge(row)
        user_input = row.get("input", "")
        if not isinstance(user_input, str) or not user_input.strip():
            continue
        prompt = row.get("prompt", "")
        if not prompt or not str(prompt).strip():
            continue
        uniq = dedupe_candidate_strings(row.get("candidates"))
        if len(uniq) < 2:
            continue
        eligible.append({"input": user_input.strip(), "uniq": uniq})
    return eligible


def draw_sample(eligible: list[dict], rng: random.Random) -> tuple[str, str, str]:
    """Return (function_source, docstring_a, docstring_b) with A/B in random order."""
    row = rng.choice(eligible)
    a, b = rng.sample(row["uniq"], 2)
    return row["input"], a, b


def main() -> None:
    st.set_page_config(
        page_title="Docstring preferences",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Which docstring is better?")
    st.caption(
        "You see one Python function and two candidate docstring bodies. "
        "Pick the one that better describes the function."
    )

    with st.sidebar:
        candidates_path = st.text_input(
            "Candidates JSONL",
            value=str(DEFAULT_CANDIDATES),
            help="Usually data/phase_3/candidates.jsonl",
        )
        table_name = st.text_input("Supabase table", value=DEFAULT_TABLE)
        seed_str = st.text_input(
            "Optional RNG seed",
            value="",
            help="Integer for reproducible sampling; leave empty for nondeterministic.",
        )

    try:
        eligible = load_eligible_rows(candidates_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if not eligible:
        st.error("No eligible rows: need non-empty `input` and ≥2 unique candidates.")
        st.stop()

    st.sidebar.success(f"{len(eligible)} rows available for sampling.")

    if seed_str.strip():
        try:
            seed_int = int(seed_str.strip())
        except ValueError:
            st.sidebar.error("Seed must be an integer.")
            st.stop()
    else:
        seed_int = None

    if (
        "rng" not in st.session_state
        or st.session_state.get("_seed_applied") != (seed_int, seed_str)
    ):
        st.session_state.rng = random.Random(seed_int)
        st.session_state._seed_applied = (seed_int, seed_str)
        st.session_state.current = draw_sample(eligible, st.session_state.rng)

    if "current" not in st.session_state:
        st.session_state.current = draw_sample(eligible, st.session_state.rng)

    src, doc_a, doc_b = st.session_state.current

    st.subheader("Function code")
    st.code(src, language="python")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Option A**")
        st.text_area("A", value=doc_a, height=320, disabled=True, label_visibility="collapsed")
    with col2:
        st.markdown("**Option B**")
        st.text_area("B", value=doc_b, height=320, disabled=True, label_visibility="collapsed")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        prefer_a = st.button("A is better", type="primary", use_container_width=True)
    with c2:
        prefer_b = st.button("B is better", type="primary", use_container_width=True)
    with c3:
        skip = st.button("Skip (new pair)", use_container_width=True)

    if skip:
        st.session_state.current = draw_sample(eligible, st.session_state.rng)
        st.rerun()

    if prefer_a or prefer_b:
        chosen, rejected = (doc_a, doc_b) if prefer_a else (doc_b, doc_a)
        row = {"input": src, "chosen": chosen.strip(), "rejected": rejected.strip()}
        if row["chosen"] == row["rejected"]:
            st.warning("Chosen and rejected are identical; skipping.")
        else:
            try:
                get_client().table(table_name).insert(row).execute()
                st.success("Saved to Supabase.")
            except Exception as e:
                st.error(f"Supabase insert failed: {e}")
                st.stop()
        st.session_state.current = draw_sample(eligible, st.session_state.rng)
        st.rerun()


if __name__ == "__main__":
    main()
