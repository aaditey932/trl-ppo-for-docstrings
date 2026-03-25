# RLHF for Python docstrings

End-to-end pipeline to train a language model to write **Python docstring bodies** from function source code using **supervised fine-tuning (SFT)**, **preference modeling**, and **proximal policy optimization (PPO)**—the same high-level pattern as RLHF: warm-start with demonstrations, learn a reward from preferences, then optimize the policy with that reward.

Default policy backbone is configurable via `BASE_MODEL` (see [`src/config.py`](src/config.py)); the codebase targets Hugging Face models such as Qwen2.5.

## What this repo does

1. **Data** — Generate or stream Python function snippets, attach gold docstrings, and build SFT and test splits ([`src/phase_1/`](src/phase_1/README.md)).
2. **SFT** — Train the policy to imitate gold docstrings ([`src/phase_2/`](src/phase_2/README.md)).
3. **Preferences** — Sample multiple docstrings per function from the SFT model and collect pairwise preferences (e.g. via a blind LLM judge) ([`src/phase_3/`](src/phase_3/README.md)).
4. **Reward model** — Train a sequence-classification-style reward model on chosen vs rejected pairs ([`src/phase_4/`](src/phase_4/README.md)).
5. **PPO setup** — Build prompt-only data and a reward hook for RL ([`src/phase_5/`](src/phase_5/README.md)).
6. **PPO** — Fine-tune the policy with TRL’s PPO trainer against the reward model ([`src/phase_6/`](src/phase_6/README.md)).
7. **Evaluation** — Compare SFT vs PPO on overlap, embeddings, format checks, and reward scores ([`src/phase_7/`](src/phase_7/README.md)).

Shared prompting and metrics helpers live in [`src/utils.py`](src/utils.py) (`make_prompt`, similarity, validity checks).

## Quick start

### Environment

- Python 3.10+ recommended.
- GPU strongly recommended for SFT, reward training, and PPO (`torch` + CUDA as appropriate).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env` from your template and set variables required by the scripts you run. Phase 1 data generation and docstring completion use an LLM client that expects endpoint and auth-related variables (see [`src/phase_1/helper.py`](src/phase_1/helper.py) and the Phase 1–3 READMEs). At minimum, many flows need `GPT_OSS_20B_URL` and `MODEL_ACCESS_CERT` where those scripts apply.

Optional: `import_preferences_from_supabase.py` uses `SUPABASE_URL` and `SUPABASE_KEY` if you import preferences from Supabase instead of generating them locally.

### Run order (full pipeline)

From the repository root:

| Step | Command |
|------|---------|
| 1 — Data | `python src/phase_1/generate_datasets.py` → `stream_complete_dataset.py` → `build_sft_train.py` → `make_dataset.py` |
| 2 — SFT | `python src/phase_2/train_sft.py` |
| 3 — Preferences | `python src/phase_3/generate_candidates.py` → `build_preference_pairs.py` |
| 4 — Reward | `python src/phase_4/train_reward_model.py` |
| 5 — PPO data | `python src/phase_5/build_ppo_prompts.py` (optional: `run_ppo.py` for reward sanity checks) |
| 6 — PPO | `python src/phase_6/train_ppo.py` |
| 7 — Eval | `python src/phase_7/evaluate.py` |

Each phase’s `README.md` lists inputs, outputs, and prerequisites.

### Outputs

| Artifact | Typical path |
|----------|----------------|
| SFT policy | `outputs/sft_policy/` |
| Reward model | `outputs/reward_model/` |
| PPO policy | `outputs/ppo_policy/` |
| Metrics and samples | `report/` (`metrics.json`, `sft_vs_ppo_samples.jsonl`, etc.) |

### Notebook

[`base_lm_demo.ipynb`](base_lm_demo.ipynb) demonstrates working with the base or fine-tuned models in a notebook environment.

## Design notes

- **Prompt format** — The model is asked for the docstring *body* only (no surrounding `"""`), via `make_prompt` in [`src/utils.py`](src/utils.py).
- **Reward model** — Initialized from the SFT checkpoint so tokenizer and backbone stay aligned with the policy.
- **Evaluation** — Phase 7 combines string similarity, embedding similarity, heuristic validity, and mean reward for SFT and (optionally) PPO checkpoints.

## Documentation map

- [Phase 1 — Data](src/phase_1/README.md)
- [Phase 2 — SFT](src/phase_2/README.md)
- [Phase 3 — Candidates & preferences](src/phase_3/README.md)
- [Phase 4 — Reward model](src/phase_4/README.md)
- [Phase 5 — PPO prompts & reward](src/phase_5/README.md)
- [Phase 6 — PPO training](src/phase_6/README.md)
- [Phase 7 — Evaluation](src/phase_7/README.md)

For the conceptual write-up that motivated this workflow (RLHF-style training for docstrings), see your local copy of **RLHF for Docstring** (PDF); this README stays in sync with the code in *this* repository.