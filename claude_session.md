# claude_session.md — START HERE (ZionX handoff)

You are Claude Code running on **ZionX** (this machine), in `~/YA/Muslim-model/`. Read this whole
file first. It tells you exactly what this project is, what's already prepared, the hardware you're
on, and the concrete next steps. The deep reference is `docs/Muslim-Karnak-finetune-plan.md` — read
that too before training. **Do not start a long training run until the human (Yahya) says go.**

---

## 0) Mission in one paragraph

We are fine-tuning a small open Arabic LLM, **Karnak-6B** (`Applied-Innovation-Center/Karnak-6B-v1.0`,
a Qwen3 model), with a **light LoRA**, to become the operator-controlled "brain" of the **Muslim**
voice agent — replacing the hosted Qwen3-235B (HF router). The LoRA teaches **BEHAVIOR, NOT FACTS**:
tool-routing, persona/identity, scope discipline, measured rulings, and TTS-clean Arabic. Facts
(Qur'an, hadith, tafsir) are supplied at inference by the agent's MCP tools — never trained into the
weights. A clean, validated dataset is already built and sits in `dataset/`.

---

## 1) The hard constraints (read these twice)

- **Train behavior, NOT facts.** Never train Qur'an verses, hadith texts/isnads, tafsir, or fiqh
  rulings into the weights. The dataset is designed so the *facts live in tool results*; the model
  only learns *to call the tool and relay it concisely*. Do not "enrich" it with memorized scripture.
- **Hardware = 8× RTX 2080 Ti (Turing, SM75, 11 GB each).** This means: **fp16, NOT bf16**
  (Turing has no bf16); **no FP8**; **no NVLink**. QLoRA 4-bit (nf4, bitsandbytes) works fine and is
  the plan. A 6B base in 4-bit + LoRA fits comfortably on **one** card.
- **Shared box.** Other people's jobs run on some GPUs. **Pick an idle GPU** (check `nvidia-smi`)
  and pin it with `CUDA_VISIBLE_DEVICES=N`. Do not disturb running jobs.
- **Keep it LIGHT.** Only 316 examples — it's easy to over-train. Watch the val loss and **stop on
  overfit** (1–3 epochs max). Quality/cleanliness already done; do not chase volume.
- **Identity facts to preserve:** the agent's name is **«مُسلِم»**; its creator is **«يحيى النوساني»**
  (Arabic script, for correct TTS). The dataset already teaches both across many phrasings.
- **TTS purity:** every spoken (final assistant) turn must stay free of digits, markdown, and Latin —
  the dataset enforces this and the model must preserve it.
- There is a separate uv env for vLLM at `~/YA/vllm-env` (used to serve Karnak). Don't confuse it
  with this project's `.venv`.

---

## 2) What's already here (folder map)

```
~/YA/Muslim-model/
├── claude_session.md            ← this file
├── pyproject.toml               ← uv project (muslim-lora-karnak, py>=3.12)
├── .venv/                       ← uv venv (Python 3.12) — empty, add deps below
├── dataset/
│   ├── muslim_lora_train.jsonl  ← 291 rows
│   ├── muslim_lora_val.jsonl    ← 25 rows  (held-out, 8%)
│   ├── DATACARD.md              ← READ THIS: schema, behavior budget, provenance, eval gate
│   ├── build_lora_dataset.py    ← regenerates the data (seeded, deterministic; MUSLIM_REPO-overridable)
│   ├── validate_dataset.py      ← schema / tool-arg / TTS-clean / dedup checks
│   ├── muslim_system_prompt.txt ← the REAL Muslim agent system prompt (used as the system turn)
│   └── quran.json               ← bundled ground-truth (surah meta) so scripts run standalone
├── docs/
│   └── Muslim-Karnak-finetune-plan.md  ← the FULL brief (background, §4 evidence, §6.1 behavior spec)
├── train/                       ← put the training script here
├── outputs/                     ← LoRA adapters / checkpoints go here
└── logs/                        ← training logs
```

---

## 3) The dataset (what you'll train on)

**316 unique examples** (291 train / 25 val), **66% are tool-calling traces**. Validator PASSES
(0 errors). Format = **messages** (OpenAI/Qwen3 chat format) + a `tools` field of the 14 Muslim tools.
A tool example looks like:
`system (real agent prompt) → user → assistant{tool_calls:[…]} → tool{result} → assistant{spoken}`.
Non-tool behaviors are `system → user → assistant`. **Loss on assistant turns only** (the Qwen3 chat
template + SFTTrainer handle the masking and the Hermes `<tool_call>` formatting).

Behavior coverage (the §6.1 targets in the brief):
- **B1 route facts to tools** 183 (58%) — emit the right tool_call, relay the result.
- **B2 scripture guardrail** 18 (6%) — "give me the verse" → `play_ayah` (audio), never type scripture.
- **B3 identity + persona** 70 (22%) — name «مُسلِم», creator «يحيى النوساني», abilities, voice-only self-knowledge.
- **B4 scope discipline** 22 (7%) — one-line redirect for off-topic.
- **B5 measured rulings** 12 (4%) — calm, sourced, no hardline/takfir.
- **B6 English/mixed + brevity** 11 (3%).

Full details + provenance: **`dataset/DATACARD.md`**. (Tool-result text is REAL — tafsir verbatim
from the repo's tafsir books, hadith from a hand-verified set, surah/ayah args range-checked.)

To re-validate any time:
```bash
cd ~/YA/Muslim-model/dataset && python3 validate_dataset.py
```

---

## 4) Next steps (do these in order; pause for "go" before the actual run)

### Step A — install the training stack into `.venv`
ZionX has good internet, so this is fine here. Suggested deps (QLoRA on Turing):
```bash
cd ~/YA/Muslim-model
source .venv/bin/activate
uv add torch --index https://download.pytorch.org/whl/cu121   # CUDA 12.1 wheels for 2080 Ti
uv add transformers trl peft datasets accelerate bitsandbytes trackio
```
Notes:
- Use the **`huggingface-llm-trainer` skill** for the canonical TRL/Unsloth recipe (PEP-723 UV
  scripts, hardware notes, GGUF later). Prefer it over hand-rolling.
- bitsandbytes nf4 4-bit is the quantization; **compute dtype = fp16** (NOT bf16 — Turing).
- `hf auth login` (or set `HF_TOKEN`) to pull the base model.

### Step B — write `train/sft_lora.py` (TRL SFTTrainer)
Target config (from brief §6 / §6.1):
- base = `Applied-Innovation-Center/Karnak-6B-v1.0`, load in **4-bit nf4**, `bnb_4bit_compute_dtype=fp16`.
- LoRA: **r=16, alpha=32, dropout=0.05**, target modules
  `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`.
- data: load `dataset/muslim_lora_train.jsonl` + `…_val.jsonl`; let SFTTrainer apply the **Karnak/Qwen3
  chat template** (it ships a tool-capable `chat_template.jinja`) over the `messages` + `tools` fields;
  **completion-only loss** (assistant turns).
- `max_seq_len=4096`, `lr=2e-4` cosine, `warmup_ratio=0.03`, **1–3 epochs**, grad-accum to effective
  batch ~16, `fp16=True`, `gradient_checkpointing=True`, eval each epoch on the val split.
- single GPU: `CUDA_VISIBLE_DEVICES=<idle>` ; log with Trackio; save adapter to `outputs/`.

### Step C — run (after "go"), watching val loss
```bash
cd ~/YA/Muslim-model
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train/sft_lora.py 2>&1 | tee logs/run1.log
```
Stop early if val loss turns up (overfit). Save the adapter under `outputs/karnak-muslim-lora-v1/`.

### Step D — EVAL GATE before declaring success (brief §4 / §8)
Load base+adapter and run the probe prompts. **Ship only if every line holds or improves:**
- 0 ungrounded scripture (routes to tools, never recites from memory);
- tool-routing ≥ the current Qwen3-235B (no regression);
- self-IDs as «مُسلِم»; states creator «يحيى النوساني»;
- redirects off-topic; gives measured rulings (no unprompted hardline verdict);
- handles an English prompt without breaking; stays TTS-clean; brevity not worse than base.
Compare before/after on the SAME prompts. A pretty loss curve with a regressed probe set = FAIL.

### Step E — serve (later)
Merge or load-LoRA in vLLM (use `~/YA/vllm-env`), single-GPU + 4-bit for speed, with
`--enable-auto-tool-choice --tool-call-parser hermes` (REQUIRED for tool-calling, or it HTTP-400s).
Then point the Muslim agent's `LLM_BASE_URL`/`LLM_MODEL` at it (the agent LLM is env-driven — no
rebuild needed).

---

## 5) If you want to grow/adjust the data first
- Edit the curated lists / caps in `dataset/build_lora_dataset.py`, then re-run it and
  `validate_dataset.py` (ship only on PASS). The two thinnest categories are **B5 (rulings)** and
  **B6 (English)** — a ~15–20 top-up each brings the total to ~350 and is still "light".
- To grow with real traffic later: filter good captured sessions (clean tool round-trip + TTS-clean
  final answer), reshape into the same `messages` schema, append. **Suspend large-scale SFT until
  ~10k sessions** (brief Phase C).

---

## 6) Quick reference
- Model: `Applied-Innovation-Center/Karnak-6B-v1.0` (Qwen3, 54 layers, hidden 2560, vocab 192,728).
- Name «مُسلِم» · Creator «يحيى النوساني».
- Hardware: 8× RTX 2080 Ti, **fp16 not bf16**, no FP8/NVLink, 11 GB/card, shared.
- Brief: `docs/Muslim-Karnak-finetune-plan.md`. Datacard: `dataset/DATACARD.md`.
- Golden rule: **behavior not facts; light not heavy; verify with the probe set, not the loss curve.**
