"""QLoRA SFT of Karnak-6B for the Muslim agent -- HF Jobs variant (L4, 24GB,
native bf16). Adapted from train/sft_lora.py, which targets a local Turing
(RTX 2080 Ti, fp16-only, 11GB) GPU. Differences, all deliberate:

  - bf16 instead of fp16: L4 (Ada Lovelace) has native bf16 tensor cores,
    so the fp32-master-weight workaround sft_lora.py needs for Turing isn't
    needed here -- TRL's normal QLoRA bf16 adapter casting works as intended.
  - per_device_train_batch_size=4 (up from 2): 24GB VRAM comfortably fits a
    bigger batch than the 2080Ti's 11GB, which was running at 99% memory
    utilization locally. grad_accum drops to 4 to keep the same effective
    batch size (16) the recipe was tuned around.
  - Data loaded from a mounted HF dataset volume (/data/...), not local
    repo-relative paths -- this runs in an ephemeral HF Jobs container.
  - Adapter is pushed directly to the Hub at the end (the container's
    filesystem disappears when the job ends), not saved locally.

Run via:
    hf jobs uv run --flavor l4x1 --with transformers --with peft --with trl \\
        --with bitsandbytes --with accelerate --with huggingface_hub --with liger-kernel --with datasets \\
        -v hf://datasets/NightPrince/muslim-6b-v1-dataset:/dataset \\
        --secrets HF_TOKEN \\
        train/sft_lora_hfjobs.py
"""

import os

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

HF_TOKEN = os.environ.get("HF_TOKEN")  # explicit, not relying on implicit env pickup
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))  # >0 = short diagnostic run, skips eval/save/push

BASE_MODEL = "Applied-Innovation-Center/Karnak-6B-v1.0"
DATA_DIR = "/dataset"
OUTPUT_DIR = "/tmp/karnak-muslim-lora-v4-hfjobs"
ADAPTER_HUB_REPO = "NightPrince/muslim-6b-v1-lora-adapter"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_chat_template = tokenizer.chat_template
    with open(f"{DATA_DIR}/karnak_training_chat_template.jinja", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{DATA_DIR}/muslim_lora_train_v4.jsonl",
            "validation": f"{DATA_DIR}/muslim_lora_val_v4.jsonl",
        },
    )
    train_dataset = dataset["train"].remove_columns(["behavior", "intent"])
    eval_dataset = dataset["validation"].remove_columns(["behavior", "intent"])

    is_diagnostic = MAX_STEPS > 0
    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        max_steps=MAX_STEPS if is_diagnostic else -1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # matches local exactly (2*8=16) -- batch=4 was tested and
                                        # measured to add ~6% padding waste on this length-variable
                                        # dataset without a clear compute win, so removed as a variable
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=5120,
        packing=False,
        assistant_only_loss=True,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        use_cache=False,
        use_liger_kernel=True,
        eval_strategy="no" if is_diagnostic else "epoch",
        save_strategy="no" if is_diagnostic else "epoch",
        save_total_limit=1,
        load_best_model_at_end=not is_diagnostic,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1 if is_diagnostic else 5,
        report_to="none",
        run_name="karnak-muslim-lora-v4-hfjobs",
        seed=42,
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
        peft_config=LORA_CONFIG,
        processing_class=tokenizer,
    )

    trainer.train()

    if is_diagnostic:
        print(f"DIAGNOSTIC RUN (MAX_STEPS={MAX_STEPS}) -- skipping save/push, timing only.")
        return

    trainer.save_model(OUTPUT_DIR)
    tokenizer.chat_template = original_chat_template
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"pushing adapter to {ADAPTER_HUB_REPO} ...")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(ADAPTER_HUB_REPO, private=False, exist_ok=True, token=HF_TOKEN)
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=ADAPTER_HUB_REPO, token=HF_TOKEN)
    print(f"done: https://huggingface.co/{ADAPTER_HUB_REPO}")


if __name__ == "__main__":
    main()
