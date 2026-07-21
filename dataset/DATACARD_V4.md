# Muslim Agent — LoRA Fine-Tuning Dataset (v4 / Muslim-6B-PRO)

**Purpose:** the dataset behind `NightPrince/Muslim-6B-PRO`, a full retrain from
`Applied-Innovation-Center/Karnak-6B-v1.0` (not an incremental patch on v3). Behavior + tool-routing
is taught; Qur'an/hadith/tafsir/fatwa **facts requiring exact, source-cited text** are never
memorized into weights — those stay in the MCP grounding layer. The exception, new in v4: general,
well-established Islamic knowledge with no dedicated retrieval tool (Seerah, stories of the
prophets, aqeedah, broad fiqh concepts, akhlaq, foundational history, comparative/interfaith) is
trained for calibrated tone and confidence, the same style/calibration pattern already used for
measured fiqh rulings — not new risky fact-injection.

## Why a full retrain (not another LoRA patch on v3)

A systematic audit of v3's real production tool-calling (real MCP servers, real vLLM serving, real
generated conversations) surfaced three root-caused problems, all traced to the source and fixed
here rather than patched over:

1. **Double-JSON-encoding of `tool_call.arguments`.** One line, shared by both this repo's
   training-time chat template AND the base model's own shipped `chat_template.jinja`
   (`tool_call.arguments | tojson` — re-encoding an already-JSON-string value). Fixed in both
   places; `train/merge_and_push.py` asserts the fix is actually present in what gets shipped.
2. **Tool-name hallucination concentrated on tools added after v3's cutoff**: 0% on the originally
   trained tool set (879 real calls) vs. 23.5% on newly-added MCP tools. Fixed with live-probed,
   schema-verified coverage of every tool actually served in production — including catching that
   `analyze_word` had been trained with the WRONG schema in every prior version
   (`{word: string}` vs. the real `{surah, ayah, word_no}` position lookup).
3. **Surah name/number confusion**, concentrated on surahs never seen in training. Fixed with
   `fetch_surah_info` coverage of all 114 surahs, plus a **hand-verified** (not just generated) set
   of alternate/colloquial surah names and named-ayah nicknames — each cross-checked against
   `mcp.tafsir.net`'s real scholarly names_info text for uniqueness before inclusion. Several
   plausible names were explicitly REJECTED after this check found them ambiguous across two
   different surahs (see `dataset/verify_surah_facts.py`'s module docstring for the specific
   collisions caught).

## Files

| file | rows | purpose |
|---|---|---|
| `muslim_lora_train_v4.jsonl` | 2,513 | training split |
| `muslim_lora_val_v4.jsonl` | 218 | held-out validation (8%) |
| `build_lora_dataset_v4.py` | — | regenerates both; extends v1/v2's hand-curated behaviors with B7-B13, systematic surah coverage, alt-surah-names, new-tool coverage, and merges the two ground-truth-checked real-data sources below |
| `merge_dspark_conversations.py` | — | ground-truth-checks the 1,594 real tool-augmented conversations generated against v3 this project; kept 1,416 (89%), corrected the mechanical double-encoding bug, DROPPED (not guessed-and-patched) anything with a hallucinated tool name, out-of-range surah/ayah, or scripture recited verbatim instead of routed to `play_ayah` |
| `merge_voice_sessions.py` | — | full re-pass of all 1,326 turns from the real production `NightPrince/muslim-voice-sessions` dataset (not just the ~70 hand-picked earlier); kept 455 (34% — most rejections were genuine empty/session-artifact turns, verified by spot-check, not a bug in the filter) |
| `verify_surah_facts.py` | — | independent fact-check gate for every hardcoded surah/ayah claim (named verses + alt-surah-names) against quran.json + live-fetched scholarly ground truth; **must PASS before training** |
| `validate_dataset.py` | — | schema / role-sequence / tool-arg-range / TTS-clean / dedup checks |
| `tafsir_net_surah_ground_truth.jsonl` | 114 | real `fetch_surah_info` + `get_surah_statistics` output for every surah, fetched live from `mcp.tafsir.net` this session — the source of truth `verify_surah_facts.py` checks against |
| `voice_sessions_turns.parquet` | 1,326 | real production turns (text/tool-call columns only; audio bytes stripped) from the private `NightPrince/muslim-voice-sessions` HF dataset |

**Total: 2,731 unique examples (2,513 train / 218 val), 59% tool-calling traces.**
Validator: **PASS (0 errors)**. Surah-fact validator: **PASS (0 errors)**.

## Composition by source

| source | count |
|---|---|
| Hand-curated (v1/v2 behaviors B1-B6 + v4's new identity/B7-B13/new-tool/surah-coverage examples) | 869 |
| Ground-truth-checked real Dspark-generated conversations | 1,407 |
| Ground-truth-checked real production voice-session turns | 455 |

## Behavior coverage

B1 tool-routing (1,943) · B2 scripture-audio guardrail (42) · B3 persona/identity incl. adversarial
override resistance (271) · B4 scope discipline (63) · B5 measured rulings (167) · B6 English/mixed
(22) · B7 Seerah (66) · B8 stories of the prophets (78) · B9 aqeedah (21) · B10 broad fiqh concepts
(26) · B11 akhlaq (14) · B12 Islamic history (11) · B13 comparative/interfaith (7).

31 distinct real tool names exercised (see `validate_dataset.py` output for the exact per-tool
counts), matching the tool menu actually served in production this session (local
play_ayah/play_surah + IslamicMCPServer + HadithMCPServer + mcp.tafsir.net's 17 tools +
islamqa-mcp.org's 5 tools).

## Schema (unchanged from v1/v2)

```json
{
  "behavior": "B1", "intent": "tafsir_verse",
  "messages": [
    {"role": "system", "content": "<the REAL Muslim agent system prompt>"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0001", "type": "function",
       "function": {"name": "get_tafsir_verse", "arguments": "{\"surah\":112,\"ayah\":1,\"book\":\"ar-tafsir-muyassar\"}"}}]},
    {"role": "tool", "tool_call_id": "call_0001", "name": "get_tafsir_verse", "content": "<REAL tafsir text>"},
    {"role": "assistant", "content": "<short, TTS-clean spoken answer faithful to the tool result>"}
  ],
  "tools": [ <the current, live-verified tool schemas> ]
}
```

Train on assistant turns only (mask system/user/tool) — `assistant_only_loss=True` in
`train/sft_lora.py`, using the fixed training-time chat template's `{% generation %}` markers.
