"""Policy LM hub id and Hugging Face loading defaults (Qwen2.5, etc.)."""

import os

import torch

# Cold-start policy (SFT before `outputs/sft_policy` exists). Override: export BASE_MODEL=org/name
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B")
# 1.5B fits in less VRAM than 7B; if OOM during SFT/PPO, lower per_device_train_batch_size or enable gradient checkpointing.


def policy_base_model_id() -> str:
    """Hub repo id for the base (unfine-tuned) policy."""
    return BASE_MODEL


def tokenizer_pretrained_kwargs() -> dict:
    return {"trust_remote_code": True}


def policy_from_pretrained_kwargs() -> dict:
    kw: dict = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kw["dtype"] = torch.bfloat16
    else:
        kw["dtype"] = torch.float32
    return kw


def reward_model_from_pretrained_kwargs() -> dict:
    """Loading seq-classification heads from the same checkpoint family as the policy."""
    return policy_from_pretrained_kwargs().copy()
