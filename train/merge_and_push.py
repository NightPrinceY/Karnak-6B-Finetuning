"""Merge the LoRA adapter into the base Karnak-6B weights and push as a standalone
model to the Hugging Face Hub under NightPrince/Muslim-6B.

Merging needs the full (non-quantized) base in memory at once (~12GB fp16) — that
doesn't fit on a single 11GB 2080 Ti, so this runs on CPU (plenty of system RAM).
No GPU needed; this is a one-time weight merge, not training.

Run: .venv/bin/python train/merge_and_push.py
"""

import torch
from huggingface_hub import create_repo
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Applied-Innovation-Center/Karnak-6B-v1.0"
ADAPTER_DIR = "outputs/karnak-muslim-lora-v1"
MERGED_DIR = "outputs/Muslim-6B"
HUB_REPO = "NightPrince/Muslim-6B"


def main():
    print("loading base model on CPU in fp16 for merge...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map="cpu")

    print("attaching LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    print("merging adapter into base weights...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    print(f"saving merged model to {MERGED_DIR} ...")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"creating Hub repo {HUB_REPO} (public)...")
    create_repo(HUB_REPO, private=False, exist_ok=True)

    with open("train/MODEL_CARD.md", encoding="utf-8") as f:
        readme = f.read()
    with open(f"{MERGED_DIR}/README.md", "w", encoding="utf-8") as f:
        f.write(readme)

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
