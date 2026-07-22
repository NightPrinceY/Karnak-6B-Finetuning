---
license: apache-2.0
language:
  - ar
  - en
base_model: Applied-Innovation-Center/Karnak-6B-v1.0
pipeline_tag: text-generation
library_name: transformers
tags:
  - text-generation
  - causal-lm
  - arabic
  - islamic
  - tool-calling
  - lora
  - peft
  - qwen3
  - voice-assistant
---

<p align="center">
  <img src="https://huggingface.co/NightPrince/Muslim-6B-PRO/resolve/main/muslim-6b-pro-banner-dark.png" alt="Muslim-6B-PRO" width="100%" />
</p>

# Muslim-6B-PRO

**Muslim-6B-PRO** is a behavior-tuned Islamic voice-assistant model, fine-tuned from
[Karnak-6B-v1.0](https://huggingface.co/Applied-Innovation-Center/Karnak-6B-v1.0) (a
depth-extended Qwen3-4B-Instruct-2507) to serve as the reasoning core of **Muslim**, a
voice-first Islamic assistant. It is trained for tool-call routing, persona/scope discipline,
and calibrated general Islamic knowledge — not for reciting scripture from memory.

## Model Details

| | |
|---|---|
| **Base model** | [Karnak-6B-v1.0](https://huggingface.co/Applied-Innovation-Center/Karnak-6B-v1.0) (Qwen3 architecture) |
| **Parameters** | 5.94B |
| **Layers** | 54 |
| **Hidden size** | 2,560 |
| **Attention heads** | 32 (8 KV heads, GQA) |
| **Vocabulary** | 192,728 |
| **Context length** | 262,144 tokens |
| **Fine-tuning method** | QLoRA (4-bit NF4 base, fp16 compute) |
| **License** | Apache 2.0 |
| **Languages** | Arabic, English |

## Key Capabilities

- **Reliable tool-call routing** across the full Qur'an/hadith/tafsir/fatwa retrieval toolset
  (31 tools, including mcp.tafsir.net's 17 tools, IslamQA's 5 tools, and local Qur'an audio
  playback), with schemas verified against the live tool servers rather than assumed.
- **Clean, standards-correct tool-call JSON** — `tool_call.arguments` decodes with a single
  `json.loads()`, matching the Hermes-style format used by the base Qwen3 model.
- **Full 114-surah coverage**, including alternate/colloquial surah names and named-ayah
  nicknames (e.g. آية الكرسي, سورة براءة, سورة تبارك), each verified against real scholarly
  source text to avoid ambiguous name→number mappings.
- **Calibrated general Islamic knowledge** — Seerah, stories of the prophets, aqeedah basics,
  broad fiqh concepts, akhlaq, foundational history, and comparative/interfaith framing, with
  appropriate hedging on genuinely contested specifics rather than flat assertions.
- **Persona and scope discipline**, including resistance to adversarial attempts to override
  its identity or push it outside its intended scope.

## Intended Use

Muslim-6B-PRO is trained on **behavior**, not memorized facts, for anything requiring
exact, source-cited text — Qur'an wording, hadith matn/isnad, tafsir attribution. Those are
retrieved at inference time via tool calls, never generated from memory, because language
models reliably hallucinate scripture when asked to recite it directly. The one exception is
well-established, broadly-agreed general Islamic knowledge with no dedicated retrieval tool
(Seerah, stories of the prophets, aqeedah basics, broad fiqh concepts, akhlaq, foundational
history, comparative/interfaith framing) — there, the model is trained for calibrated tone and
appropriate hedging on contested specifics, not fact injection.

**This model is designed to be served with a tool-calling layer** (Qur'an/hadith/tafsir
retrieval, audio playback) and a system prompt defining its persona and scope. It is not
intended as a general-purpose scripture-reciting or fatwa-issuing model on its own.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "NightPrince/Muslim-6B-PRO"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

messages = [
    {"role": "system", "content": "<your Muslim agent system prompt>"},
    {"role": "user", "content": "ما هي آية الكرسي؟"},
]
inputs = tokenizer.apply_chat_template(
    messages, tools=your_tool_schemas, add_generation_prompt=True,
    return_tensors="pt", return_dict=True,
).to(model.device)

out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Tool-calling format

Uses the same Hermes-style `<tool_call>` format as the base Qwen3 model. Bind your tool
schemas via the standard `tools=` argument to `apply_chat_template`. `tool_call.arguments`
decodes cleanly with a single `json.loads()` call.

### GGUF quantizations

Quantized GGUF builds (Q2_K through Q8_0, plus F16) for `llama.cpp`-based local inference are
published separately at **NightPrince/Muslim-6B-PRO-GGUF**.

## Training Data

2,731 examples (59% tool-calling traces), from three ground-truth-checked sources: hand-curated
examples, real production voice-session turns, and real tool-augmented conversations — each
example checked against source-of-truth references, with anything that couldn't be verified
mechanically excluded rather than guessed at.

| Behavior | Count | Description |
|---|---|---|
| B1 | 1,943 | Tool routing |
| B2 | 42 | Scripture-audio guardrail |
| B3 | 271 | Persona/identity, incl. adversarial-override resistance |
| B4 | 63 | Scope discipline |
| B5 | 167 | Measured fiqh rulings |
| B6 | 22 | English / mixed-language |
| B7 | 66 | Seerah |
| B8 | 78 | Stories of the prophets |
| B9 | 21 | Aqeedah |
| B10 | 26 | Broad fiqh concepts |
| B11 | 14 | Akhlaq |
| B12 | 11 | Islamic history |
| B13 | 7 | Comparative / interfaith |

## Training Procedure

- **Method**: QLoRA (4-bit NF4 base, fp16 compute — trained on hardware with no native bf16
  support) via TRL `SFTTrainer`.
- **LoRA config**: r=16, alpha=32, dropout=0.05, targeting `q/k/v/o/gate/up/down_proj`.
- **Schedule**: 3 epochs, cosine LR decay from 2e-4, 3% warmup, effective batch size 16.
- **Best checkpoint selection**: `load_best_model_at_end` on held-out eval loss across the full
  3-epoch run — the published weights are the best-performing checkpoint, not simply the last.

## Limitations

- Not intended for direct scripture recitation or fatwa-issuing without the retrieval tool
  layer it was trained to route through.
- Behavioral eval-gate results (57 adversarial/generalization probes) are pending publication —
  loss curves alone do not fully capture tool-routing correctness or persona robustness; treat
  this card as provisional on that front until updated.
- Trained and evaluated primarily on Arabic Islamic-assistant use cases; general-purpose
  capability outside that domain is inherited from the base model and not separately verified.

## Citation

If you use this model, please cite it as:

```bibtex
@misc{muslim6bpro2026,
  title        = {Muslim-6B-PRO: A Behavior-Tuned Islamic Voice-Assistant Language Model},
  author       = {Alnwsany, Yahya},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/NightPrince/Muslim-6B-PRO}},
  note         = {Fine-tuned from Karnak-6B-v1.0}
}
```

**Related resources:**
- Base model: [Applied-Innovation-Center/Karnak-6B-v1.0](https://huggingface.co/Applied-Innovation-Center/Karnak-6B-v1.0)
- Training dataset: [NightPrince/muslim-6b-v1-dataset](https://huggingface.co/datasets/NightPrince/muslim-6b-v1-dataset)
- GGUF quantizations: [NightPrince/Muslim-6B-PRO-GGUF](https://huggingface.co/NightPrince/Muslim-6B-PRO-GGUF)
- Fine-tuning code: [github.com/NightPrinceY/Karnak-6B-Finetuning](https://github.com/NightPrinceY/Karnak-6B-Finetuning)

## Copyright & License

Copyright © 2026 Yahya Alnwsany (NightPrince). This model's fine-tuning work — the LoRA
adapter, training data curation, and this model card — is released under the **Apache License
2.0**; see [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) for the full text. Use of the
base model [Karnak-6B-v1.0](https://huggingface.co/Applied-Innovation-Center/Karnak-6B-v1.0)
remains subject to its own license terms from Applied Innovation Center.
