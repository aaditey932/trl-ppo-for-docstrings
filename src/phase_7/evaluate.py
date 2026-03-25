#!/usr/bin/env python3
"""
Phase 7: Evaluation. Compare SFT vs PPO on data/phase_1/test.jsonl.

What the metrics measure (for both checkpoints):

**Overlap with gold** — Mean similarity and exact match with reference docstrings.
Evaluates how well the model recovers the intended documentation (character-level overlap).

**Semantic correctness** — Mean embedding cosine similarity (sentence-transformers MiniLM):
prediction vs gold reference when `reference` exists; otherwise function source vs prediction
as a reference-free proxy. Captures paraphrases and meaning overlap that string similarity misses.

**Constraint / format adherence** — Checks for valid docstring-shaped output (not echoed `def` / source).
Ensures outputs follow the expected format instead of copying code.

**Learned preference (reward score)** — Mean reward assigned to generated outputs.
Tracks alignment with the reward model’s notion of quality.

Writes metrics to report/metrics.json, error cases to report/error_cases.json,
and SFT vs PPO samples for human preference to report/sft_vs_ppo_samples.jsonl.

Requires: Phase 1 (test.jsonl via make_dataset.py), Phase 2 (SFT), Phase 4 (reward model).
PPO is optional; if outputs/ppo_policy/ is missing, only SFT is evaluated.

Run from repo root: python src/phase_7/evaluate.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "phase_5"))

from config import policy_from_pretrained_kwargs, tokenizer_pretrained_kwargs
from utils import (
    docstring_similarity,
    has_valid_docstring_output,
    make_prompt,
    normalize_docstring,
)
from run_ppo import get_reward, load_reward_model_and_tokenizer

# Paths
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "phase_1" / "test.jsonl"
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "phase_1" / "raw_prompts.jsonl"
DEFAULT_SFT_PATH = REPO_ROOT / "data" / "phase_1" / "sft_train.jsonl"
DEFAULT_SFT_POLICY_DIR = REPO_ROOT / "outputs" / "sft_policy"
DEFAULT_PPO_POLICY_DIR = REPO_ROOT / "outputs" / "ppo_policy"
DEFAULT_REWARD_MODEL_DIR = REPO_ROOT / "outputs" / "reward_model"
REPORT_DIR = REPO_ROOT / "report"
MAX_TEST_PROMPTS = 100
HUMAN_PREFERENCE_SUBSET = 50
GENERATION_MAX_NEW_TOKENS = 192
GENERATION_TEMPERATURE = 0.3
GENERATION_DO_SAMPLE = True
# MiniLM: fast sentence embeddings for semantic correctness (pred vs ref, or code vs pred).
SEMANTIC_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MAX_LENGTH = 512
CODE_SNIPPET_CHARS = 4000


def _mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


class SemanticEmbedder:
    """Sentence embeddings for semantic correctness (cosine similarity in [0, 1] after mapping)."""

    def __init__(self, device: str, model_id: str = SEMANTIC_EMBED_MODEL_ID) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, device=self.device)
        out = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=SEMANTIC_MAX_LENGTH,
            return_tensors="pt",
        )
        out = {k: v.to(self.device) for k, v in out.items()}
        hidden = self.model(**out).last_hidden_state
        pooled = _mean_pool(hidden, out["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def pairwise_cosine_01(self, texts_a: list[str], texts_b: list[str]) -> list[float]:
        """Cosine similarities in [0, 1] (mapped from [-1, 1])."""
        if len(texts_a) != len(texts_b):
            raise ValueError("texts_a and texts_b must have the same length")
        if not texts_a:
            return []
        ea = self.encode(texts_a)
        eb = self.encode(texts_b)
        cos = (ea * eb).sum(dim=1).clamp(-1.0, 1.0)
        # Map to [0, 1] for a monotonic "correctness" style score
        return ((cos + 1.0) / 2.0).cpu().tolist()


def semantic_correctness_scores(
    examples: list[dict], responses: list[str], embedder: SemanticEmbedder
) -> list[float]:
    """Per-example semantic correctness: pred vs reference, or code vs pred if no reference."""
    texts_a: list[str] = []
    texts_b: list[str] = []
    for ex, resp in zip(examples, responses):
        ref = (ex.get("reference") or "").strip()
        code = ex.get("input") or ""
        if ref:
            texts_a.append(resp)
            texts_b.append(ref)
        else:
            texts_a.append(f"Python function:\n{code[:CODE_SNIPPET_CHARS]}")
            texts_b.append(resp)
    return embedder.pairwise_cosine_01(texts_a, texts_b)

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


def get_test_prompts(
    test_path: Path | None = None,
    raw_path: Path | None = None,
    sft_path: Path | None = None,
    max_prompts: int = MAX_TEST_PROMPTS,
) -> list[dict]:
    """Load test set: test.jsonl if present, else hold out from raw_prompts (exclude SFT inputs)."""
    test_path = test_path or DEFAULT_TEST_PATH
    raw_path = raw_path or DEFAULT_RAW_PATH
    sft_path = sft_path or DEFAULT_SFT_PATH

    if test_path.exists():
        rows = load_jsonl(test_path)
        return rows[:max_prompts]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}. Run Phase 1 make_dataset.py to create test.jsonl, or ensure raw_prompts.jsonl exists."
        )
    raw = load_jsonl(raw_path)
    sft_inputs = set()
    if sft_path.exists():
        for row in load_jsonl(sft_path):
            inp = row.get("input")
            if inp is not None:
                sft_inputs.add(inp)
    eligible = [r for r in raw if r.get("input") not in sft_inputs]
    return eligible[:max_prompts]


def load_policy(path: Path, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok_kw = tokenizer_pretrained_kwargs()
    model_kw = policy_from_pretrained_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(str(path), **tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(path), **model_kw)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def generate_responses(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
    temperature: float = GENERATION_TEMPERATURE,
    batch_size: int = 4,
) -> list[str]:
    """Generate one completion per prompt. Prompts are full strings (make_prompt(input))."""
    responses = []
    n = len(prompts)
    for i in tqdm(
        range(0, n, batch_size),
        desc="generate",
        unit="batch",
        leave=False,
    ):
        batch = prompts[i : i + batch_size]
        out = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        out = {k: v.to(device) for k, v in out.items()}
        with torch.no_grad():
            gen = model.generate(
                **out,
                max_new_tokens=max_new_tokens,
                do_sample=GENERATION_DO_SAMPLE,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode only the generated part
        for j, (input_ids, gen_ids) in enumerate(zip(out["input_ids"], gen)):
            new_len = gen_ids.shape[0] - input_ids.shape[0]
            if new_len <= 0:
                responses.append("")
                continue
            new_tokens = gen_ids[input_ids.shape[0] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(text.strip())
    return responses


def write_figures_tables(
    report_dir: Path,
    metrics: dict,
    error_cases: list[dict],
) -> None:
    """Write report/figures_tables.md with metrics table, error cases, and analysis placeholders."""
    lines = [
        "# Evaluation Report: SFT vs PPO",
        "",
        "## What the metrics measure",
        "",
        "**Overlap with gold** — Mean similarity and exact match with reference docstrings. "
        "Evaluates how well the model recovers the intended documentation (character-level overlap).",
        "",
        "**Semantic correctness** — Mean embedding cosine similarity (sentence-transformers MiniLM): "
        "prediction vs gold reference when `reference` exists; otherwise function source vs prediction "
        "as a reference-free proxy. Captures paraphrases and meaning overlap that string similarity misses.",
        "",
        "**Constraint / format adherence** — Checks for valid docstring-shaped output (not echoed `def` / source). "
        "Ensures outputs follow the expected format instead of copying code.",
        "",
        "**Learned preference (reward score)** — Mean reward assigned to generated outputs. "
        "Tracks alignment with the reward model’s notion of quality.",
        "",
        "## 1. Metrics Summary",
        "",
        "| Model | Mean similarity | Semantic correctness | Exact match | Format adherence | Mean reward |",
        "|-------|-----------------|------------------------|-------------|------------------|-------------|",
    ]
    for name, m in metrics.items():
        if m is None:
            continue
        ms = m.get("mean_similarity", 0)
        sem = m.get("mean_semantic_correctness", 0)
        em = m.get("exact_match_rate", 0)
        fmt = m.get("format_adherence", 0)
        rew = m.get("mean_reward_score", 0)
        lines.append(
            f"| {name.upper()} | {ms:.4f} | {sem:.4f} | {em:.2%} | {fmt:.2%} | {rew:.4f} |"
        )
    lines.extend(["", "## 2. Error Cases (sample)", ""])
    for block in error_cases:
        model = block.get("model", "?")
        cases = block.get("cases", [])
        lines.append(f"### {model.upper()}")
        lines.append("")
        for i, c in enumerate(cases, 1):
            inp = (c.get("input") or "")[:120]
            if len(c.get("input") or "") > 120:
                inp += "..."
            resp = (c.get("response_snippet") or "")[:200]
            if len(c.get("response_snippet") or "") > 200:
                resp += "..."
            sem = c.get("semantic_correctness")
            sem_s = f"{sem:.4f}" if isinstance(sem, (int, float)) else "n/a"
            lines.append(
                f"- **Case {i}** (similarity={c.get('similarity')}, semantic_correctness={sem_s}, reward={c.get('reward')})"
            )
            lines.append(f"  - Input: {inp}")
            lines.append(f"  - Response: {resp}")
            lines.append("")
    lines.extend([
        "## 3. Analysis",
        "",
        "### 3.1 Coverage vs reference and semantic correctness",
        "Compare generated docstrings to gold references: missing Args/Returns, wrong behavior, or vague text. **mean_semantic_correctness** uses sentence embeddings (paraphrase-tolerant vs string similarity). (Fill in from error cases.)",
        "",
        "### 3.2 Style drift",
        "The model may converge on a fixed template (e.g. one-line summaries) instead of matching reference style. (Fill in with examples.)",
        "",
        "### 3.3 Reward hacking",
        "If the reward model favors length or certain phrases, PPO may optimize for those instead of accuracy. (Fill in with examples.)",
        "",
        "### 3.4 Echoing source code",
        "Invalid outputs may repeat `def` or restate the function instead of writing a docstring. (Fill in with examples.)",
        "",
    ])
    (report_dir / "figures_tables.md").write_text("\n".join(lines))


def compute_metrics(
    examples: list[dict],
    responses: list[str],
    format_ok: list[bool],
    rewards: list[float],
    semantic_scores: list[float],
) -> dict:
    """Mean similarity, semantic correctness, exact match, format adherence, mean reward."""
    n = len(examples)
    if n == 0:
        return {}

    similarities: list[float] = []
    exact_matches = 0
    for ex, resp in zip(examples, responses):
        ref = ex.get("reference") or ""
        similarities.append(docstring_similarity(resp, ref))
        if normalize_docstring(resp) == normalize_docstring(ref):
            exact_matches += 1

    mean_sem = sum(semantic_scores) / n if semantic_scores and len(semantic_scores) == n else 0.0

    return {
        "num_prompts": n,
        "mean_similarity": sum(similarities) / n if similarities else 0.0,
        "mean_semantic_correctness": mean_sem,
        "exact_match_rate": exact_matches / n,
        "format_adherence": sum(format_ok) / n,
        "mean_reward_score": sum(rewards) / n if rewards else 0.0,
    }


def run_evaluation(
    test_path: Path | None = None,
    sft_policy_dir: Path | None = None,
    ppo_policy_dir: Path | None = None,
    reward_model_dir: Path | None = None,
    report_dir: Path | None = None,
    max_prompts: int = MAX_TEST_PROMPTS,
    human_preference_n: int = HUMAN_PREFERENCE_SUBSET,
) -> dict:
    test_path = test_path or DEFAULT_TEST_PATH
    sft_policy_dir = sft_policy_dir or DEFAULT_SFT_POLICY_DIR
    ppo_policy_dir = ppo_policy_dir or DEFAULT_PPO_POLICY_DIR
    reward_model_dir = reward_model_dir or DEFAULT_REWARD_MODEL_DIR
    report_dir = report_dir or REPORT_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = get_test_prompts(
        test_path=test_path, max_prompts=max_prompts
    )
    if not examples:
        raise ValueError("No test examples found.")

    full_prompts = [make_prompt(ex["input"]) for ex in examples]

    # Reward model for scoring
    rm_tokenizer, reward_model = load_reward_model_and_tokenizer(reward_model_dir)

    sem_embedder: SemanticEmbedder | None = None

    def get_sem_embedder() -> SemanticEmbedder:
        nonlocal sem_embedder
        if sem_embedder is None:
            print("Loading semantic embedding model for semantic correctness...")
            sem_embedder = SemanticEmbedder(device)
        return sem_embedder

    results = {"sft": None, "ppo": None, "error_cases": [], "metadata": {}}

    # ---- SFT ----
    if not sft_policy_dir.exists():
        print(f"SFT policy not found at {sft_policy_dir}; skipping SFT evaluation.")
    else:
        print("Evaluating SFT policy...")
        sft_tokenizer, sft_model = load_policy(sft_policy_dir, device)
        sft_responses = generate_responses(
            full_prompts, sft_tokenizer, sft_model, device
        )
        sft_format = [has_valid_docstring_output(r) for r in sft_responses]
        sft_rewards = [
            get_reward(p, r, rm_tokenizer, reward_model)
            for p, r in tqdm(
                zip(full_prompts, sft_responses),
                total=len(full_prompts),
                desc="reward SFT",
                unit="prompt",
                leave=False,
            )
        ]
        sft_sem = semantic_correctness_scores(examples, sft_responses, get_sem_embedder())
        results["sft"] = {
            "metrics": compute_metrics(
                examples, sft_responses, sft_format, sft_rewards, sft_sem
            ),
            "responses": sft_responses,
            "rewards": sft_rewards,
            "semantic_scores": sft_sem,
        }

    # ---- PPO ----
    if not ppo_policy_dir.exists():
        print(f"PPO policy not found at {ppo_policy_dir}; skipping PPO evaluation.")
    else:
        print("Evaluating PPO policy...")
        ppo_tokenizer, ppo_model = load_policy(ppo_policy_dir, device)
        ppo_responses = generate_responses(
            full_prompts, ppo_tokenizer, ppo_model, device
        )
        ppo_format = [has_valid_docstring_output(r) for r in ppo_responses]
        ppo_rewards = [
            get_reward(p, r, rm_tokenizer, reward_model)
            for p, r in tqdm(
                zip(full_prompts, ppo_responses),
                total=len(full_prompts),
                desc="reward PPO",
                unit="prompt",
                leave=False,
            )
        ]
        ppo_sem = semantic_correctness_scores(examples, ppo_responses, get_sem_embedder())
        results["ppo"] = {
            "metrics": compute_metrics(
                examples, ppo_responses, ppo_format, ppo_rewards, ppo_sem
            ),
            "responses": ppo_responses,
            "rewards": ppo_rewards,
            "semantic_scores": ppo_sem,
        }

    # ---- Error cases (up to 10): low similarity to reference or low reward ----
    def collect_errors(
        resps: list,
        rewards: list,
        sem_scores: list[float] | None = None,
    ) -> list[dict]:
        errs = []
        for i, ex in enumerate(examples):
            ref = ex.get("reference")
            resp = resps[i] if i < len(resps) else ""
            sim = docstring_similarity(resp, ref or "")
            rew = rewards[i] if i < len(rewards) else None
            sem = sem_scores[i] if sem_scores and i < len(sem_scores) else None
            if sim < 0.5 or (rew is not None and rew < 0):
                errs.append({
                    "input": ex.get("input", ""),
                    "reference": ref,
                    "similarity": sim,
                    "semantic_correctness": sem,
                    "response_snippet": (resps[i][:300] + "...") if len(resps[i]) > 300 else resps[i],
                    "reward": rew,
                })
        return errs[:10]

    if results["sft"]:
        results["error_cases"].append({
            "model": "sft",
            "cases": collect_errors(
                results["sft"]["responses"],
                results["sft"]["rewards"],
                results["sft"].get("semantic_scores"),
            ),
        })
    if results["ppo"]:
        results["error_cases"].append({
            "model": "ppo",
            "cases": collect_errors(
                results["ppo"]["responses"],
                results["ppo"]["rewards"],
                results["ppo"].get("semantic_scores"),
            ),
        })

    results["metadata"] = {
        "num_test_prompts": len(examples),
        "human_preference_subset_n": min(human_preference_n, len(examples)),
    }

    # ---- Write outputs ----
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "sft": results["sft"]["metrics"] if results["sft"] else None,
        "ppo": results["ppo"]["metrics"] if results["ppo"] else None,
    }
    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(report_dir / "error_cases.json", "w") as f:
        json.dump(results["error_cases"], f, indent=2)

    # Write report/figures_tables.md (metrics table + error cases + analysis placeholders)
    write_figures_tables(report_dir, metrics_out, results["error_cases"])

    # Human preference: SFT vs PPO side-by-side for first N prompts
    if results["sft"] and results["ppo"]:
        n_human = min(human_preference_n, len(examples))
        samples = []
        for i in range(n_human):
            ref = examples[i].get("reference", "")
            sft_r = results["sft"]["responses"][i]
            ppo_r = results["ppo"]["responses"][i]
            samples.append({
                "input": examples[i]["input"],
                "reference": ref,
                "sft_response": sft_r,
                "ppo_response": ppo_r,
                "sft_similarity": docstring_similarity(sft_r, ref),
                "ppo_similarity": docstring_similarity(ppo_r, ref),
                "sft_semantic_correctness": results["sft"]["semantic_scores"][i],
                "ppo_semantic_correctness": results["ppo"]["semantic_scores"][i],
            })
        with open(report_dir / "sft_vs_ppo_samples.jsonl", "w") as f:
            for s in tqdm(samples, desc="write human preference", unit="sample", leave=False):
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {len(samples)} SFT vs PPO samples to report/sft_vs_ppo_samples.jsonl for human preference.")

    print(f"Wrote report/metrics.json and report/error_cases.json.")

    print("\n--- Code and model responses ---")
    for i, ex in enumerate(examples):
        print(f"\n{'=' * 60}\nExample {i + 1}/{len(examples)}\n{'=' * 60}")
        print("--- Code (input) ---")
        print(ex.get("input", ""))
        if results["sft"]:
            print("--- SFT response ---")
            print(results["sft"]["responses"][i])
        if results["ppo"]:
            print("--- PPO response ---")
            print(results["ppo"]["responses"][i])

    return results


def main() -> None:
    if not DEFAULT_SFT_POLICY_DIR.exists():
        raise FileNotFoundError(
            f"SFT policy not found at {DEFAULT_SFT_POLICY_DIR}. Run Phase 2 first (train_sft.py)."
        )
    if not DEFAULT_REWARD_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Reward model not found at {DEFAULT_REWARD_MODEL_DIR}. Run Phase 4 first (train_reward_model.py)."
        )

    results = run_evaluation()
    print("\n--- Metrics summary ---")
    if results["sft"]:
        print("SFT:", json.dumps(results["sft"]["metrics"], indent=2))
    if results["ppo"]:
        print("PPO:", json.dumps(results["ppo"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
