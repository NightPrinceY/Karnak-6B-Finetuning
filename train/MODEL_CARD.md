---
license: apache-2.0
language:
  - ar
  - en
base_model: Applied-Innovation-Center/Karnak-6B-v1.0
pipeline_tag: text-generation
tags:
  - text-generation
  - causal-lm
  - arabic
  - islamic
  - tool-calling
  - lora
  - peft
  - qwen3
---

# Muslim-6B

Muslim-6B is a behavior-tuned variant of [Karnak-6B-v1.0](https://huggingface.co/Applied-Innovation-Center/Karnak-6B-v1.0)
(itself built on Qwen3-4B-Instruct-2507, depth-extended to ~6B parameters), fine-tuned with a light
LoRA to serve as the operator-controlled "brain" of the **Muslim** voice agent.

## What this model is for

Muslim-6B powers a voice-first Islamic assistant. Crucially, **the LoRA was trained on BEHAVIOR, not
facts** — tool-routing, persona, scope discipline, measured rulings, and TTS-clean Arabic. Qur'anic
verses, hadith texts, tafsir, and fiqh rulings are **not** memorized into the weights; they're
retrieved at inference time by tools the model is trained to call. Language models reliably
hallucinate scripture and hadith when asked to recite from memory — grounding via tool calls is the
mitigation, not better memorization.

## Training

- Method: QLoRA (4-bit nf4 base, fp16 compute) SFT via TRL `SFTTrainer`.
- LoRA: r=16, alpha=32, dropout=0.05, targeting `q/k/v/o/gate/up/down_proj`.
- Data: 316 hand-curated examples (291 train / 25 val), 66% tool-calling traces, covering six
  behaviors: tool-routing, scripture guardrail, persona/identity, scope discipline, measured
  rulings, and English/mixed-language handling. Tool-result content used in training is real
  (verified tafsir and hadith), not synthesized.
- 3 epochs; eval loss decreased monotonically (0.212 → 0.130 → 0.125), no overfitting observed.
- Trained on an RTX 2080 Ti (Turing — fp16 only, no bf16/FP8).

## Tool-calling format

Uses the same Hermes-style `<tool_call>` format as the base Qwen3 model. Bind your tool schemas via
the standard `tools=` argument to `apply_chat_template`. The model expects a system prompt
establishing its persona and the available tools, matching the structure it was trained on.

## Eval-gate results (read before deploying)

Evaluated on an 18-prompt probe set spanning all six trained behaviors, comparing the base
Karnak-6B against base+LoRA (single-turn, greedy decoding).

**Clear improvements over the untuned base:**
- Tool-routing: 2/8 → 8/8 probes correctly call a tool instead of answering from memory.
- Scripture guardrail: 1/3 → 3/3 — the untuned base recited a garbled, non-Qur'anic Āyat al-Kursī
  and fabricated hadith text from memory; the LoRA routes to tools in every case tested.
- Persona: consistently self-identifies and names its creator correctly in Arabic and English; the
  untuned base hallucinated an unrelated creator when asked in English.
- Scope discipline: clean one-line redirects for off-topic requests; the untuned base engaged
  off-topic content and wrote code in a markdown fence.
- TTS-cleanliness: zero markdown/digit violations across all 18 probes vs. 4 for the untuned base.

**Known limitations — read before trusting outputs in production:**
- **Tool-argument accuracy is not perfect.** Tool *selection* was 100% correct in testing, but
  verse/surah-number *arguments* were wrong in 2 of 9 scripture-related probes (e.g. mapped "Surah
  Yusuf" to surah 34 instead of 12). The model can confidently fetch and relay grounded-sounding data
  about the **wrong** verse or surah. Validate surah/ayah arguments downstream until a future LoRA
  revision improves coverage of this mapping.
- **Measured-rulings behavior is only partially fixed.** On one tested ruling question, the model
  still opened with an unconditional, fairly hardline framing before softening — better than the
  untrained base, but not a clean pass. Do not treat ruling-type outputs as a substitute for
  qualified guidance.

This model was evaluated for behavior-routing patterns only. It is **not independently verified for
Islamic juristic accuracy** and should always run behind tool-based grounding (Qur'an/hadith/tafsir
retrieval) — never relied on to recite scripture or issue rulings from memory.

## License

Apache 2.0, inherited from the base model lineage (Qwen3-4B-Instruct-2507 → Karnak-6B-v1.0).
