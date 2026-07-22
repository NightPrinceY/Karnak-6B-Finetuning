"""Merge the LoRA adapter into the base Karnak-6B weights and push as a standalone
model to the Hugging Face Hub under NightPrince/Muslim-6B.

Merging needs the full (non-quantized) base in memory at once (~12GB fp16) — that
doesn't fit on a single 11GB 2080 Ti, so this runs on CPU (plenty of system RAM).
No GPU needed; this is a one-time weight merge, not training.

Run (merge + save locally only, no Hub push):
    .venv/bin/python train/merge_and_push.py
Run (merge, then also push to the Hub):
    .venv/bin/python train/merge_and_push.py --push
The merge/push steps are split so GGUF conversion, old-repo cleanup, etc. can happen
between the local merge and the actual public Hub push.
"""

import argparse

import torch
from huggingface_hub import create_repo
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Applied-Innovation-Center/Karnak-6B-v1.0"
ADAPTER_DIR = "outputs/karnak-muslim-lora-v4"
MERGED_DIR = "outputs/Muslim-6B-PRO"
HUB_REPO = "NightPrince/Muslim-6B-PRO"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="also push the merged model to the Hub")
    args = parser.parse_args()

    print("loading base model on CPU in fp16 for merge...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map="cpu")

    print("attaching LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    print("merging adapter into base weights...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    # CRITICAL FIX (v4): sft_lora.py deliberately restores the tokenizer's
    # ORIGINAL chat_template (as shipped by Karnak-6B-v1.0 upstream) before
    # saving the adapter, so the training-time patched copy
    # (train/karnak_training_chat_template.jinja, fixed earlier this
    # project) never reaches this point. Verified this session: Karnak's
    # own shipped chat_template has the IDENTICAL bug --
    # `tool_call.arguments | tojson` double-encodes an already-JSON-string
    # argument. Without this patch, the merged v4/V1 model would ship with
    # the exact same double-encoding bug that broke every real tool call in
    # v3's production serving. Apply the same one-line fix here, to the
    # template that actually gets saved/served.
    if "tool_call.arguments | tojson" in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "tool_call.arguments | tojson", "tool_call.arguments"
        )
        print("patched chat_template: removed double-JSON-encoding of tool_call.arguments")
    else:
        raise RuntimeError(
            "expected 'tool_call.arguments | tojson' in the tokenizer's chat_template "
            "but did not find it -- the upstream template may have changed; verify the "
            "double-encoding bug is still present/absent before proceeding, don't silently skip this."
        )

    print(f"saving merged model to {MERGED_DIR} ...")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)

    with open("train/MODEL_CARD_PRO.md", encoding="utf-8") as f:
        readme = f.read()
    with open(f"{MERGED_DIR}/README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    if not args.push:
        print(f"merge complete, saved locally to {MERGED_DIR} (not pushed -- rerun with --push when ready)")
        return

    print(f"creating Hub repo {HUB_REPO} (public)...")
    create_repo(HUB_REPO, private=False, exist_ok=True)

    print(f"pushing to {HUB_REPO} ...")
    model.push_to_hub(HUB_REPO)
    tokenizer.push_to_hub(HUB_REPO)
    from huggingface_hub import upload_file

    upload_file(
        path_or_fileobj=f"{MERGED_DIR}/README.md",
        path_in_repo="README.md",
        repo_id=HUB_REPO,
    )

    print(f"done: https://huggingface.co/{HUB_REPO}")


if __name__ == "__main__":
    main()
