"""v5 CORRECTIVE merge: attach the v5 corrective LoRA adapter to the already-
merged Muslim-6B-PRO (not the Karnak base -- this is a second-generation
merge on top of an already-shipped model, not a from-scratch merge). Forked
from train/merge_and_push.py, which stays untouched as the record of how
Muslim-6B-PRO itself was produced.

Merging needs the full (non-quantized) base in memory at once (~12GB fp16) — that
doesn't fit on a single 11GB 2080 Ti, so this runs on CPU (plenty of system RAM).
No GPU needed; this is a one-time weight merge, not training.

Deliberately outputs to a NEW local dir and a NEW Hub repo name rather than
overwriting Muslim-6B-PRO in place, so the currently-shipped model stays an
instant rollback until this corrective build passes the eval gate.

Run (merge + save locally only, no Hub push):
    .venv/bin/python train/merge_and_push_v5.py
Run (merge, then also push to the Hub):
    .venv/bin/python train/merge_and_push_v5.py --push
The merge/push steps are split so re-running the eval gate against the local
merge can happen before the actual public Hub push.
"""

import argparse

import torch
from huggingface_hub import create_repo
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "outputs/Muslim-6B-PRO"  # already-merged, already-shipped model --
                                       # this corrective pass builds on top of it
ADAPTER_DIR = "outputs/karnak-muslim-lora-v5-corrective"
MERGED_DIR = "outputs/Muslim-6B-PRO-v5"
HUB_REPO = "NightPrince/Muslim-6B-PRO-v5"


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

    # v5: the double-JSON-encoding fix (see train/merge_and_push.py for the
    # full history) was already applied when Muslim-6B-PRO was first merged,
    # and confirmed present in outputs/Muslim-6B-PRO/chat_template.jinja
    # (zero occurrences of the buggy pattern). sft_lora_v5_corrective.py's
    # own template-restore step carries that already-clean template forward
    # unchanged, so this should be a no-op here -- but keep the check as a
    # defensive guard against an unexpectedly-regressed base, rather than a
    # hard assertion that would now unconditionally fail.
    if "tool_call.arguments | tojson" in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "tool_call.arguments | tojson", "tool_call.arguments"
        )
        print("WARNING: found and patched a regressed double-JSON-encoding bug -- "
              "investigate why outputs/Muslim-6B-PRO's chat_template changed")
    else:
        print("chat_template already clean (no double-JSON-encoding pattern found), as expected")

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
