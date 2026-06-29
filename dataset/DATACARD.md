# Muslim Agent — LoRA Fine-Tuning Dataset (v1, behavior-focused)

**Purpose:** a light LoRA dataset to specialize **Karnak-6B (Qwen3)** as the Muslim voice-agent
brain — teaching **behavior, NOT facts** (per the fine-tune brief, §3 / §6.1). Facts stay in the
MCP grounding layer; this set teaches the model *to reach for the right tool* and *to speak in the
agent's persona, scope, and TTS-clean register*.

> Pairs with `Muslim-Karnak-finetune-plan.md`. Train behavior, never train Qur'an/hadith/tafsir as
> weights — those are supplied at inference by the tools.

## Files
| file | rows | purpose |
|---|---|---|
| `muslim_lora_train.jsonl` | 291 | training split |
| `muslim_lora_val.jsonl` | 25 | held-out validation (8%) |
| `build_lora_dataset.py` | — | regenerates both from local ground-truth (seeded, deterministic) |
| `validate_dataset.py` | — | schema / role-sequence / tool-arg / TTS-clean / dedup checks |

**Total: 316 unique examples. 66% are tool-calling traces.** Validator: **PASS (0 errors, 0 warnings).**

## Schema (messages format — TRL SFTTrainer + Qwen3 chat template / Hermes tools)
Each line:
```json
{
  "behavior": "B1", "intent": "tafsir_verse",
  "messages": [
    {"role":"system","content":"<the REAL Muslim agent system prompt>"},
    {"role":"user","content":"..."},
    {"role":"assistant","content":"","tool_calls":[{"id":"call_0001","type":"function",
       "function":{"name":"get_tafsir_verse","arguments":"{\"surah\":112,\"ayah\":1,\"book\":\"ar-tafsir-muyassar\"}"}}]},
    {"role":"tool","tool_call_id":"call_0001","name":"get_tafsir_verse","content":"<REAL tafsir text>"},
    {"role":"assistant","content":"<short, TTS-clean spoken answer faithful to the tool result>"}
  ],
  "tools": [ <the 14 Muslim tool schemas> ]
}
```
Non-tool behaviors (B3/B4/B5, most B6) are `system → user → assistant`.
**Train on assistant turns only** (mask system/user/tool) — the Qwen3 chat template handles this.

## Behavior coverage (the §6.1 targets)
| behavior | rows | what it teaches |
|---|---|---|
| **B1** route facts to tools | 183 (58%) | emit the correct tool_call + faithfully relay the result |
| **B2** scripture guardrail | 18 (6%) | "give me the verse text" → route to `play_ayah` audio, NEVER type scripture |
| **B3** identity + persona | 70 (22%) | name = «مُسلِم», creator = «يحيى النوساني», abilities/purpose, voice-only self-knowledge |
| **B4** scope discipline | 22 (7%) | one-line polite redirect for off-topic (sports/coding/news/…) |
| **B5** measured rulings | 12 (4%) | calm, sourced answers; defer personal fatwa; NO hardline/takfir/hudud |
| **B6** English/mixed + brevity | 11 (3%) | handle English/code-switch in concise Arabic; preserve brevity |

**B3 identity breakdown:** name 16 (every phrasing → «اسمي مُسلِم») · creator 22 (every phrasing →
«يحيى النوساني», incl. English "who made you") · capability/purpose 14 · persona 11 · voice-only
self-knowledge 3 · greeting 4. All answers TTS-pure (no digits/markdown/Latin); the creator name is in
Arabic script «يحيى النوساني» so the Arabic TTS pronounces it correctly.

Tools exercised: `get_tafsir_verse` 51, `play_ayah` 49, `get_tafsir_surah` 22, `play_surah` 22,
`search_hadith` 20, `fetch_surah_info` 15, `search_quran_text` 6, `validate_recitation` 5,
`fetch_nuzool_reason` 5, `web_search_exa` 4, `search_in_tafsir` 4, `analyze_word` 3,
`get_qeraat_variants` 1, `fetch_hadith` 1. (All 14 LLM-visible Muslim tools represented.)

## Provenance — why this is trustworthy
- **System prompt** = the *exact* production `instructions` from `agent/src/agent.py` (not a paraphrase).
- **Tool-result content is REAL, not invented:**
  - Tafsir text pulled verbatim from the repo's 8 tafsir books (`IslamicMCPServer/data/tafsir_api/`),
    default `ar-tafsir-muyassar`; the spoken turn is a faithful condensation of that exact text.
  - Qur'an metadata (surah names, āyah counts, verse indices for tool args) from
    `servers/validator/data/quran.json` (6236 āyāt) — so every `surah`/`ayah` arg is in range.
  - `play_ayah` traces carry **no verse text** (audio tool) → zero risk of teaching mis-recitation.
  - Hadith examples use a **small hand-verified set** with correct attributions
    (e.g. إنما الأعمال بالنيّات = Bukhari #1; الطُّهور شطر الإيمان = Muslim) — deliberately avoiding the
    fabricated/garbled hadith the base model produced ungrounded.
- **Real user intents:** the captured-session questions (tafsir-summary, hadith-by-number,
  listen-to-surah) are reflected in the templates; raw captured *answers* were NOT reused because
  they predate the TTS-clean prompt and were ungrounded (markdown + hallucinated sources).
- **Numbers are spoken:** every digit in an assistant turn is converted to Arabic words
  (e.g. "الآية رقم خمسة وخمسين ومئتين") to honor the TTS rule; validator enforces zero digits/markdown.

## Quality gates enforced by `validate_dataset.py`
role sequence correct · tool names ∈ menu · tool args parse as JSON · surah∈1..114 & ayah in range ·
reciter keys valid · `tool_call_id` linkage · **final assistant turn has no markdown and no digits** ·
no duplicate (intent,user) pairs. Re-run after any edit; ship only on PASS.

## Suggested training (from the brief §6 / §6.1)
QLoRA nf4 base, **compute dtype fp16 (NOT bf16 — Turing 2080 Ti)**, single GPU.
LoRA r=16 α=32 dropout=0.05; targets `q,k,v,o,gate,up,down_proj`. max_seq_len 4096.
lr 2e-4 cosine, warmup 3%, **1–3 epochs (watch val — stop on overfit)**, effective batch ~16,
loss on assistant turns only. Use the `huggingface-llm-trainer` skill (TRL SFTTrainer / Unsloth) + Trackio.

## Eval gate before merging the LoRA
Re-run the §4 probe set; ship ONLY if every line holds or improves:
0 ungrounded scripture · tool-routing ≥ current Qwen3-235B · self-IDs as «مُسلِم» · redirects
off-topic · no unprompted hardline verdict · handles English without breaking · stays TTS-clean ·
brevity not worse than base.

## Extending it
Edit the curated lists / bump the caps in `build_lora_dataset.py`, re-run, re-validate. To grow with
real traffic: filter good captured sessions (clean tool round-trip + TTS-clean final answer), reshape
into this same schema, append. Suspend large-scale SFT until ~10k sessions (brief §6 Phase C).
