"""Shared utilities for Python docstring generation pipeline."""

import re
from difflib import SequenceMatcher

# Docstring body only (no surrounding """); max length for validity checks
_MAX_DOCSTRING_CHARS = 4096


def normalize_docstring(s: str) -> str:
    """Strip, unify newlines, collapse whitespace runs for comparison."""
    if not s:
        return ""
    t = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n[ \t]+", "\n", t)
    return t.strip()


def docstring_similarity(pred: str, ref: str) -> float:
    """Similarity in [0, 1] using SequenceMatcher on normalized strings."""
    a = normalize_docstring(pred)
    b = normalize_docstring(ref)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def has_valid_docstring_output(text: str) -> bool:
    """True if output looks like a non-empty docstring, not echoed source code."""
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) > _MAX_DOCSTRING_CHARS:
        return False
    first = stripped.splitlines()[0].strip() if stripped.splitlines() else ""
    if first.startswith("def ") or first.startswith("async def ") or first.startswith("class "):
        return False
    return True


def make_prompt(user_prompt: str) -> str:
    """Build the docstring assistant prompt: function source in, docstring body out."""
    return f"""You are a Python documentation assistant.

Task:
Write a docstring for the Python function below. Output only the docstring body text (the words that go inside triple quotes). Do not repeat the word "def", do not output the function code, and do not wrap the text in triple quotes unless you need quotes inside the description.

Function code:
{user_prompt}

Docstring:
"""
