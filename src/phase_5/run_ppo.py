#!/usr/bin/env python3
"""
Phase 5: Reward function for PPO. Phase 6 will add the PPO training loop.

Provides get_reward(prompt, response, rm_tokenizer, reward_model) that scores
prompt+response using the trained reward model from outputs/reward_model/.

Requires Phase 4 (train_reward_model.py) to be run first.
"""

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_REWARD_MODEL_DIR = REPO_ROOT / "outputs" / "reward_model"


def get_reward(
    prompt: str,
    response: str,
    rm_tokenizer: AutoTokenizer,
    reward_model: torch.nn.Module,
) -> float:
    """
    Score prompt+response with the reward model. Returns a scalar.

    Uses the same concatenation and tokenization as Phase 4 reward evaluation.
    """
    text = prompt + response
    out = rm_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=rm_tokenizer.model_max_length,
        padding=True,
    )
    device = next(reward_model.parameters()).device
    out = {k: v.to(device) for k, v in out.items()}
    reward_model.eval()
    with torch.no_grad():
        logits = reward_model(**out).logits
    return float(logits.squeeze(-1).item())


def load_reward_model_and_tokenizer(
    reward_model_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[AutoTokenizer, torch.nn.Module]:
    """
    Load tokenizer and reward model from outputs/reward_model/ (or given path).
    Model is set to eval mode and moved to CPU or CUDA.
    If device is None, uses cuda:0 when CUDA is available (see train_ppo CUDA_DEVICE).
    """
    path = Path(reward_model_dir) if reward_model_dir else DEFAULT_REWARD_MODEL_DIR
    if not path.exists():
        raise FileNotFoundError(
            f"Reward model not found at {path}. Run Phase 4 first (python src/phase_4/train_reward_model.py)."
        )
    tok_kw = {"trust_remote_code": True}
    rm_kw = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(str(path), **tok_kw)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path), num_labels=1, **rm_kw
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


if __name__ == "__main__":
    # Sanity check: load reward model and score one chosen/rejected pair
    tokenizer, model = load_reward_model_and_tokenizer()
    pairs_path = REPO_ROOT / "data" / "phase_3" / "preference_pairs.jsonl"
    if pairs_path.exists():
        rows = load_jsonl(pairs_path)
        if rows:
            ex = rows[0]
            prompt = ex.get("prompt", "")
            chosen = ex.get("chosen", "")
            rejected = ex.get("rejected", "")
            r_chosen = get_reward(prompt, chosen, tokenizer, model)
            r_rejected = get_reward(prompt, rejected, tokenizer, model)
            print("Sanity check (first preference pair):")
            print(f"  reward(chosen)  = {r_chosen:.4f}")
            print(f"  reward(rejected)= {r_rejected:.4f}")
            print(f"  chosen > rejected: {r_chosen > r_rejected}")
        else:
            print("No rows in preference_pairs.jsonl; skipping sanity check.")
    else:
        print(f"{pairs_path} not found; skipping sanity check.")
