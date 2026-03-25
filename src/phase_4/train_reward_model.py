#!/usr/bin/env python3
"""
Phase 4: Train a reward model on preference pairs (chosen > rejected) using TRL RewardTrainer.
Uses the same base model family/tokenizer space as the SFT policy.
Saves to outputs/reward_model/. After training, evaluates pairwise accuracy, margin, and logs error cases.

Requires data/phase_3/preference_pairs.jsonl from Phase 3.
Requires outputs/sft_policy from Phase 2.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from config import reward_model_from_pretrained_kwargs, tokenizer_pretrained_kwargs
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "phase_3"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "reward_model"
DEFAULT_SFT_POLICY_DIR = REPO_ROOT / "outputs" / "sft_policy"

EVAL_SPLIT_RATIO = 0.2
MAX_ERROR_CASES = 10
TRUNCATE_LEN = 200


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


def main() -> None:
    data_dir = DEFAULT_DATA_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    sft_policy_dir = DEFAULT_SFT_POLICY_DIR

    pairs_path = data_dir / "preference_pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Preference pairs not found: {pairs_path}. Run Phase 3 first."
        )

    if not sft_policy_dir.exists():
        raise FileNotFoundError(
            f"SFT policy not found at {sft_policy_dir}. Run Phase 2 first."
        )

    rows = load_jsonl(pairs_path)
    if not rows:
        raise ValueError(f"No valid rows in {pairs_path}.")

    data = [
        {
            "prompt": r.get("prompt", ""),
            "chosen": r.get("chosen", ""),
            "rejected": r.get("rejected", ""),
        }
        for r in rows
        if r.get("prompt") is not None
        and r.get("chosen") is not None
        and r.get("rejected") is not None
    ]
    if not data:
        raise ValueError(f"No rows with prompt/chosen/rejected in {pairs_path}.")

    full_dataset = Dataset.from_list(data)
    split = full_dataset.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    tok_kw = tokenizer_pretrained_kwargs()
    rm_kw = reward_model_from_pretrained_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(str(sft_policy_dir), **tok_kw)

    # Decoder-only models often need pad_token set manually
    if tokenizer.eos_token is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        str(sft_policy_dir),
        num_labels=1,
        **rm_kw,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    config = RewardConfig(
        output_dir=str(output_dir),
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="tensorboard",
        center_rewards_coefficient=1e-2,
        logging_steps=10,
    )

    trainer = RewardTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved reward model and tokenizer to {output_dir}")

    # Evaluation
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    margins = []
    error_cases = []

    def _reward(prompt: str, completion: str) -> float:
        text = prompt + "\n" + completion
        out = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=min(tokenizer.model_max_length, 1024),
            padding=True,
        )
        out = {k: v.to(device) for k, v in out.items()}
        with torch.no_grad():
            logits = model(**out).logits
        return float(logits.squeeze(-1).item())

    for ex in tqdm(eval_dataset, desc="reward eval", unit="pair"):
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        r_chosen = _reward(prompt, chosen)
        r_rejected = _reward(prompt, rejected)

        margin = r_chosen - r_rejected
        margins.append(margin)

        if r_chosen > r_rejected:
            correct += 1
        else:
            if len(error_cases) < MAX_ERROR_CASES:
                error_cases.append(
                    {
                        "prompt": (prompt[:TRUNCATE_LEN] + "...") if len(prompt) > TRUNCATE_LEN else prompt,
                        "chosen": (chosen[:TRUNCATE_LEN] + "...") if len(chosen) > TRUNCATE_LEN else chosen,
                        "rejected": (rejected[:TRUNCATE_LEN] + "...") if len(rejected) > TRUNCATE_LEN else rejected,
                        "reward_chosen": r_chosen,
                        "reward_rejected": r_rejected,
                    }
                )

    n_eval = len(eval_dataset)
    pairwise_accuracy = correct / n_eval if n_eval else 0.0
    avg_margin = sum(margins) / len(margins) if margins else 0.0

    print("\n--- Reward model evaluation (eval set) ---")
    print(f"Pairwise accuracy (chosen > rejected): {pairwise_accuracy:.4f} ({correct}/{n_eval})")
    print(f"Average reward margin (chosen - rejected): {avg_margin:.4f}")
    print(f"Error cases (rejected >= chosen): {len(error_cases)} shown (max {MAX_ERROR_CASES})")

    for j, err in enumerate(error_cases, 1):
        print(f"\n  Error case {j}: reward_chosen={err['reward_chosen']:.4f}, reward_rejected={err['reward_rejected']:.4f}")
        print(f"  prompt: {err['prompt'][:120]}...")
        print(f"  chosen: {err['chosen'][:120]}...")
        print(f"  rejected: {err['rejected'][:120]}...")


if __name__ == "__main__":
    main()