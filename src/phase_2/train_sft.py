#!/usr/bin/env python3
"""
SFT warm start: train the policy LM (see src/config.py BASE_MODEL) on data/phase_1/sft_train.jsonl
to generate docstrings for Python function code. Saves to outputs/sft_policy/.

Requires data/phase_1/sft_train.jsonl from Phase 1: build_sft_train.py after
complete_dataset.jsonl exists (from stream_complete_dataset.py over raw_prompts.jsonl).
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from config import BASE_MODEL, policy_from_pretrained_kwargs, tokenizer_pretrained_kwargs
from utils import make_prompt

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "phase_1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "sft_policy"


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
    model_name = BASE_MODEL
    tok_kw = tokenizer_pretrained_kwargs()
    model_kw = policy_from_pretrained_kwargs()

    sft_path = data_dir / "sft_train.jsonl"
    if not sft_path.exists():
        raise FileNotFoundError(
            f"SFT training file not found: {sft_path}. Run: "
            "python src/phase_1/build_sft_train.py (after complete_dataset.jsonl from stream_complete_dataset.py)."
        )

    examples = load_jsonl(sft_path)
    if not examples:
        raise ValueError(f"No valid examples in {sft_path}.")

    text_examples = [
        {
            "text": make_prompt(ex.get("input", ""))
            + (ex.get("response") or ex.get("reference") or "")
        }
        for ex in tqdm(examples, desc="prepare SFT texts", unit="ex")
    ]
    train_data = Dataset.from_list(text_examples)

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kw)

    config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved SFT policy and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
