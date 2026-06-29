# Muslim-mode-finetuning

QLoRA fine-tuning of **Karnak-6B** (`Applied-Innovation-Center/Karnak-6B-v1.0`) to become the brain of the **مُسلِم** (Muslim) Arabic Islamic voice agent.

## What this is

A light LoRA that teaches **behavior, not facts**:

- **Tool routing** — call the right tool (`get_tafsir_verse`, `play_ayah`, `search_hadith`, …) instead of answering from memory
- **Scripture guardrail** — never recite Qur'an or hadith text from weights; always route to audio/lookup tools
- **Persona & identity** — self-identifies as «مُسلِم», creator «يحيى النوساني»
- **Scope discipline** — one-line redirect for off-topic (football, coding, …)
- **Measured rulings** — calm, sourced responses; no unprompted hardline verdicts
- **TTS-clean Arabic** — zero digits, markdown, or Latin in spoken output

Facts (Qur'an text, hadith, tafsir) are supplied at inference by the agent's MCP tools — never baked into the weights.

## Model output

The fine-tuned model (v1, LoRA merged into base) is publicly available on Hugging Face:

**[NightPrince/Muslim-6B](https://huggingface.co/NightPrince/Muslim-6B)**

## Base model

[Applied-Innovation-Center/Karnak-6B-v1.0](https://huggingface.co/Applied-Innovation-Center/Karnak-6B-v1.0) — a Qwen3-based Arabic LLM depth-extended to ~5.94B parameters (54 layers, vocab 192,728), Apache-2.0 license.

## Repo structure

```
dataset/
  muslim_lora_train_v2.jsonl   # 362 training examples
  muslim_lora_val_v2.jsonl     # 31 validation examples
  build_lora_dataset.py        # deterministic dataset builder
  validate_dataset.py          # schema / TTS-clean / dedup checks
  DATACARD.md                  # full dataset provenance and behavior budget
  muslim_system_prompt.txt     # the real agent system prompt (used as system turn)
  quran.json                   # bundled surah metadata (standalone ground truth)

train/
  sft_lora.py                  # main training script (TRL SFTTrainer, QLoRA)
  karnak_training_chat_template.jinja  # patched template with {% generation %} markers
  merge_and_push.py            # merge LoRA into base and push to HF Hub
  MODEL_CARD.md                # model card template

eval/
  run_eval_gate.py             # eval gate runner (base vs base+LoRA probe comparison)
  probe_prompts.py             # 18 probes across B1–B6 behavior categories
  probe_prompts_v2.py          # expanded probe set for v2 eval

docs/
  Muslim-Karnak-finetune-plan.md  # full design brief
```

## Training setup

- **Hardware:** RTX 2080 Ti (Turing, SM75, 11 GB) — fp16 only, no bf16
- **Method:** QLoRA 4-bit nf4, LoRA r=16 α=32, 3 epochs, lr=2e-4 cosine
- **Target modules:** `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`
- **Dataset:** 393 examples total (362 train / 31 val), 71% tool-calling traces
- **Runtime:** ~42 min on a single 2080 Ti

## Eval gate results (v1)

18 probes across all behavior categories. Clear wins:

| Category | Base | LoRA |
|---|---|---|
| B1 tool routing | 2/8 correct tool calls | 8/8 correct |
| B2 scripture guardrail | 1/3 | 3/3 |
| B3 persona (Arabic + English) | ✗ hallucinated "OpenAI" as creator | ✓ |
| B4 scope redirect | ✗ wrote real Python code | ✓ one-line redirect |
| TTS violations | 4 (markdown fence + digits) | 0 |

Two caveats documented transparently in the [model card](https://huggingface.co/NightPrince/Muslim-6B) — v2 training targets both.

## Creator

**يحيى النوساني** (Yahya Alnwsany)
