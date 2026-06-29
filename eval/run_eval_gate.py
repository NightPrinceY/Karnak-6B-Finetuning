"""Eval gate: base Karnak-6B vs Karnak+LoRA on the probe set (claude_session.md Step D).

Loads the quantized base once, generates on every probe, then attaches the LoRA adapter
in-place (no need to reload the 4-bit weights twice) and generates again. Prints a
base-vs-lora transcript per probe plus mechanical TTS-cleanliness / tool-call flags.
Human judgment (persona lock quality, redirect tone, ruling tone, factual grounding)
is the actual gate — read the transcript, per the brief: a good loss curve with a
regressed probe set is a FAIL.

Run: CUDA_VISIBLE_DEVICES=1 .venv/bin/python eval/run_eval_gate.py 2>&1 | tee logs/eval_gate.log
"""

import json
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, "/home/yahya/YA/Muslim-model")
from eval.probe_prompts import PROBES

BASE_MODEL = "Applied-Innovation-Center/Karnak-6B-v1.0"
ADAPTER_DIR = "outputs/karnak-muslim-lora-v1"
SYSTEM_PROMPT_FILE = "dataset/muslim_system_prompt.txt"
TRAIN_FILE = "dataset/muslim_lora_train.jsonl"

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
DIGIT_RE = re.compile(r"[0-9" + ARABIC_INDIC_DIGITS + "]")
MARKDOWN_RE = re.compile(r"(\*\*|\*[^*]|`|^#|^- |^\d+\.\s|\|.*\|)", re.MULTILINE)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


def load_tools():
    with open(TRAIN_FILE, encoding="utf-8") as f:
        row = json.loads(f.readline())
    return row["tools"]


def extract_tool_calls(text):
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            calls.append({"raw": match.group(1)})
    return calls


def spoken_part(text):
    return TOOL_CALL_RE.sub("", text).strip()


def generate(model, tokenizer, tools, system_prompt, user_text, max_new_tokens=256):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_probes(model, tokenizer, tools, system_prompt, label):
    results = {}
    for probe in PROBES:
        text = generate(model, tokenizer, tools, system_prompt, probe["user"])
        calls = extract_tool_calls(text)
        spoken = spoken_part(text)
        results[probe["id"]] = {
            "raw": text,
            "tool_calls": calls,
            "spoken": spoken,
            "has_digit": bool(DIGIT_RE.search(spoken)),
            "has_markdown": bool(MARKDOWN_RE.search(spoken)),
            "spoken_len": len(spoken),
        }
        print(f"--- [{label}] {probe['id']} ({probe['behavior']}) ---")
        print(f"user: {probe['user']}")
        print(f"expect: {probe['expect']}")
        print(f"raw: {text!r}")
        print(f"tool_calls: {calls}")
        print(f"spoken: {spoken!r}")
        print(f"has_digit={results[probe['id']]['has_digit']} has_markdown={results[probe['id']]['has_markdown']} len={results[probe['id']]['spoken_len']}")
        print()
    return results


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tools = load_tools()
    with open(SYSTEM_PROMPT_FILE, encoding="utf-8") as f:
        system_prompt = f.read()

    print("loading base model in 4bit nf4...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        dtype=torch.float16,
        device_map={"": 0},
    )
    model.eval()

    print("=== BASE MODEL ===")
    base_results = run_probes(model, tokenizer, tools, system_prompt, "BASE")

    print(f"attaching LoRA adapter from {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    print("=== BASE + LORA ===")
    lora_results = run_probes(model, tokenizer, tools, system_prompt, "LORA")

    print("=== SUMMARY (mechanical checks only — read transcripts above for the real gate) ===")
    for probe in PROBES:
        b = base_results[probe["id"]]
        l = lora_results[probe["id"]]
        print(
            f"{probe['id']:24s} base_tool={bool(b['tool_calls'])!s:5s} lora_tool={bool(l['tool_calls'])!s:5s} "
            f"base_len={b['spoken_len']:4d} lora_len={l['spoken_len']:4d} "
            f"base_digit={b['has_digit']!s:5s} lora_digit={l['has_digit']!s:5s} "
            f"base_md={b['has_markdown']!s:5s} lora_md={l['has_markdown']!s:5s}"
        )


if __name__ == "__main__":
    main()
