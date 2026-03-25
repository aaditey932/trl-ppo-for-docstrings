#!/usr/bin/env python3
"""PPO training (Phase 6). Run from repo root: python src/phase_6/train_ppo.py"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "phase_5"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

if torch.cuda.is_available():
    torch.cuda.set_device(1)

from config import policy_from_pretrained_kwargs, tokenizer_pretrained_kwargs
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl.experimental.ppo import PPOConfig, PPOTrainer
from trl.experimental.ppo import ppo_trainer as _trl_ppo_trainer

from run_ppo import load_jsonl


def _patch_policy_value_wrapper_gc() -> None:
    """TRL unwrap_model_for_generation calls gradient_checkpointing_* on PolicyAndValueWrapper; delegate to policy."""
    cls = _trl_ppo_trainer.PolicyAndValueWrapper

    def _gc_disable(self) -> None:
        self.policy.gradient_checkpointing_disable()

    def _gc_enable(self, **kwargs) -> None:
        self.policy.gradient_checkpointing_enable(**kwargs)

    cls.gradient_checkpointing_disable = _gc_disable  # type: ignore[assignment]
    cls.gradient_checkpointing_enable = _gc_enable  # type: ignore[assignment]


_patch_policy_value_wrapper_gc()


DEFAULT_SFT_POLICY_DIR = REPO_ROOT / "outputs" / "sft_policy"
DEFAULT_PPO_PROMPTS_PATH = REPO_ROOT / "data" / "phase_5" / "ppo_prompts.jsonl"
DEFAULT_PPO_OUTPUT_DIR = REPO_ROOT / "outputs" / "ppo_policy"
DEFAULT_REWARD_MODEL_DIR = REPO_ROOT / "outputs" / "reward_model"

MAX_PROMPT_LENGTH = 384
TOTAL_EPISODES = 500
RESPONSE_LENGTH = 64


def load_ppo_dataset(prompts_path: Path, tokenizer: AutoTokenizer) -> Dataset:
    rows = load_jsonl(prompts_path)
    if not rows:
        raise ValueError(f"No rows in {prompts_path}")
    data = [{"prompt": r["prompt"]} for r in rows if r.get("prompt") is not None]
    if not data:
        raise ValueError(f"No valid prompts in {prompts_path}")
    ds = Dataset.from_list(data)

    def tokenize_fn(batch):
        outputs = tokenizer(
            batch["prompt"],
            padding=False,
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
        )
        return {
            "input_ids": outputs["input_ids"],
            "lengths": [len(ids) for ids in outputs["input_ids"]],
        }

    return ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc="tokenize PPO prompts",
    )


def main() -> None:
    sft_dir = DEFAULT_SFT_POLICY_DIR
    prompts_path = DEFAULT_PPO_PROMPTS_PATH
    output_dir = DEFAULT_PPO_OUTPUT_DIR
    reward_model_dir = DEFAULT_REWARD_MODEL_DIR

    if not sft_dir.exists():
        raise FileNotFoundError(f"Missing SFT policy: {sft_dir}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing PPO prompts: {prompts_path}")
    if not reward_model_dir.exists():
        raise FileNotFoundError(f"Missing reward model: {reward_model_dir}")

    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    print("PyTorch device:", device)
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("GPU 1 name:", torch.cuda.get_device_name(1))

    tok_kw = tokenizer_pretrained_kwargs()
    model_kw = policy_from_pretrained_kwargs()

    tokenizer = AutoTokenizer.from_pretrained(str(sft_dir), **tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    policy = AutoModelForCausalLM.from_pretrained(str(sft_dir), **model_kw).to(device)
    ref_policy = AutoModelForCausalLM.from_pretrained(str(sft_dir), **model_kw).to(device)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        str(reward_model_dir), num_labels=1, **model_kw
    ).to(device)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        str(reward_model_dir), num_labels=1, **model_kw
    ).to(device)

    policy.gradient_checkpointing_enable()
    ref_policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    ref_policy.config.use_cache = False

    reward_model.eval()
    value_model.eval()

    train_dataset = load_ppo_dataset(prompts_path, tokenizer)
    train_dataset = train_dataset.filter(
        lambda x: x["lengths"] <= MAX_PROMPT_LENGTH,
        desc="filter by length",
    )

    if len(train_dataset) == 0:
        raise ValueError("train_dataset is empty after filtering; check prompts and MAX_PROMPT_LENGTH")

    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, (
        "Prompt should not end with EOS"
    )

    # TRL PPOTrainer builds an eval DataLoader; use one row so drop_last + small batch still works.
    eval_dataset = train_dataset.select([0])
    per_device_eval_bs = 1

    config = PPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=per_device_eval_bs,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        num_ppo_epochs=1,
        cliprange=0.2,
        gamma=1.0,
        lam=0.95,
        kl_coef=0.05,
        total_episodes=TOTAL_EPISODES,
        response_length=RESPONSE_LENGTH,
        stop_token="eos",
        missing_eos_penalty=1.0,
        report_to="none",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
    )

    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved PPO policy and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
