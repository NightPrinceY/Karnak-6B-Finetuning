"""Generate the Muslim-6B-PRO model card from REAL numbers -- training metrics
pulled from trainer_state.json (written by the HF Trainer), dataset composition
pulled from the actual train/val files, not hand-typed guesses.

Run AFTER training finishes, BEFORE merge_and_push.py:
    .venv/bin/python train/generate_model_card.py
Writes train/MODEL_CARD_PRO.md, which merge_and_push.py reads.
"""
import json
import pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "karnak-muslim-lora-v4"
TRAIN_FILE = REPO_ROOT / "dataset" / "muslim_lora_train_v4.jsonl"
VAL_FILE = REPO_ROOT / "dataset" / "muslim_lora_val_v4.jsonl"


def load_trainer_state():
    state_path = OUTPUT_DIR / "trainer_state.json"
    if not state_path.exists():
        # load_best_model_at_end may leave the state under a checkpoint-* dir
        candidates = sorted(OUTPUT_DIR.glob("checkpoint-*/trainer_state.json"))
        if not candidates:
            raise FileNotFoundError(f"no trainer_state.json found under {OUTPUT_DIR}")
        state_path = candidates[-1]
    return json.loads(state_path.read_text(encoding="utf-8"))


def summarize_dataset():
    behav, intents, tool_share_n, n = Counter(), Counter(), 0, 0
    for path in (TRAIN_FILE, VAL_FILE):
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                n += 1
                behav[d["behavior"]] += 1
                intents[d["intent"]] += 1
                if any(m["role"] == "tool" for m in d["messages"]):
                    tool_share_n += 1
    return n, behav, tool_share_n / n


def main():
    state = load_trainer_state()
    log_history = state["log_history"]
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    n_total, behav, tool_share = summarize_dataset()

    eval_loss_str = " → ".join(f"{loss:.3f}" for _, loss in eval_losses) if eval_losses else "n/a"
    final_train_loss = train_losses[-1] if train_losses else None

    behavior_lines = "\n".join(
        f"  - {b}: {n}" for b, n in sorted(behav.items())
    )

    card = f"""---
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

# Muslim-6B-PRO

Muslim-6B-PRO is a behavior-tuned variant of [Karnak-6B-v1.0](https://huggingface.co/Applied-Innovation-Center/Karnak-6B-v1.0)
(Qwen3-4B-Instruct-2507, depth-extended to ~6B parameters), fine-tuned with QLoRA to serve as the
operator-controlled "brain" of the **Muslim** voice agent. This is a full retrain from the base
model (not an incremental patch on the prior Muslim-6B-v3), replacing all earlier Muslim-6B
versions after a systematic audit of their real production tool-calling behavior surfaced three
concrete, root-caused problems -- all fixed at the source here:

1. **Double-JSON-encoding of tool-call arguments.** Traced to a one-line bug shared by BOTH the
   fine-tuning repo's training-time chat template AND the base model's own shipped
   `chat_template.jinja` (`tool_call.arguments | tojson`, re-encoding an already-JSON-string
   value). Fixed at both the training-template and merge-time levels.
2. **Tool-name hallucination on tools added after v3's fine-tuning cutoff** (measured: 0% on the
   originally-trained tool set vs. 23.5% on newly-added MCP tools). Fixed via live-probed, real
   schema coverage of every tool actually served in production (mcp.tafsir.net's 17 tools,
   IslamQA's 5 tools, hadith cross-references) -- schemas verified against the live servers this
   session, not hand-typed guesses (this also caught `analyze_word`'s schema being wrong in every
   prior version: it's a `{{surah, ayah, word_no}}` position lookup, not a `{{word: string}}` search).
3. **Surah name/number confusion**, worse on less-common surahs never seen in training. Fixed with
   systematic `fetch_surah_info` coverage of all 114 surahs, plus a hand-verified (not just
   pattern-generated) set of alternate/colloquial surah names and named-ayah nicknames (e.g. آية
   الكرسي, سورة براءة, سورة تبارك) -- each cross-checked against mcp.tafsir.net's real scholarly
   names_info text for uniqueness before inclusion, specifically to avoid training a colliding or
   ambiguous name→number mapping.

## What this model is for

Muslim-6B-PRO powers a voice-first Islamic assistant. **The LoRA is trained on BEHAVIOR, not
facts** for anything requiring exact, source-cited text (Qur'an wording, hadith matn+isnad, tafsir
attribution) -- those are retrieved at inference time via tool calls, never memorized, because
language models reliably hallucinate scripture when asked to recite from memory. The exception is
well-established, broadly-agreed general Islamic knowledge with no dedicated retrieval tool
(Seerah, stories of the prophets, aqeedah basics, broad fiqh concepts, akhlaq, foundational
history, comparative/interfaith framing) -- there, the LoRA reinforces calibrated, correctly-toned
answers on mainstream points and appropriate hedging on genuinely contested specifics, the same
style/calibration pattern already proven for measured fiqh rulings.

## Training

- Method: QLoRA (4-bit nf4 base, fp16 compute -- Turing/2080Ti has no bf16 tensor cores) SFT via
  TRL `SFTTrainer`.
- LoRA: r=16, alpha=32, dropout=0.05, targeting `q/k/v/o/gate/up/down_proj` (see train/sft_lora.py
  LORA_CONFIG for the exact, current values -- kept in sync by hand since this script doesn't
  import that module, matching this repo's existing pattern of hand-synced constants across
  scripts).
- Data: {n_total} examples ({len(train_losses) and 'see dataset card' or ''}), {tool_share:.0%} tool-calling traces, spanning:
{behavior_lines}
  (B1 tool-routing, B2 scripture-audio guardrail, B3 persona/identity incl. adversarial-override
  resistance, B4 scope discipline, B5 measured rulings, B6 English/mixed-language, B7 Seerah,
  B8 stories of the prophets, B9 aqeedah, B10 broad fiqh concepts, B11 akhlaq, B12 Islamic
  history, B13 comparative/interfaith.)
- Sources: hand-curated examples, ground-truth-checked real production voice sessions, and
  ground-truth-checked real tool-augmented conversations generated against v3 (corrected for the
  double-encoding bug, or excluded where the correct answer couldn't be established mechanically
  -- see dataset/merge_dspark_conversations.py and dataset/merge_voice_sessions.py for exact
  inclusion/exclusion criteria; nothing was speculatively "fixed" by guessing at content).
- {log_history[-1]['epoch']:.0f} epochs; eval loss by checkpoint: {eval_loss_str}
- Trained on an RTX 2080 Ti (Turing -- fp16 only, no bf16/FP8).

## Tool-calling format

Uses the same Hermes-style `<tool_call>` format as the base Qwen3 model. Bind your tool schemas via
the standard `tools=` argument to `apply_chat_template`. **Verify** `tool_call.arguments` decodes
with a single `json.loads()` -- this was the exact bug this version fixes.

## Eval-gate results

See the eval gate transcript (`logs/eval_gate.log`) run against `eval/probe_prompts.py` +
`probe_prompts_v2.py` + `probe_prompts_v4.py` (the latter added this version: new-tool coverage,
adversarial identity pressure, alt-surah-name resolution, and held-out generalization probes for
every new knowledge category). Human judgment on tone/calibration is the actual gate; mechanical
checks (tool-call presence, TTS-cleanliness) are necessary but not sufficient.
"""
    out_path = HERE / "MODEL_CARD_PRO.md"
    out_path.write_text(card, encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"final train loss: {final_train_loss}")
    print(f"eval losses by epoch: {eval_losses}")


if __name__ == "__main__":
    main()
