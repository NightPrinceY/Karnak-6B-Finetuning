"""QLoRA SFT of Karnak-6B for the Muslim agent. Behavior only — see dataset/DATACARD.md.

Run pinned to one idle GPU, e.g.:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python train/sft_lora.py 2>&1 | tee logs/run1.log
"""

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import trackio

BASE_MODEL = "Applied-Innovation-Center/Karnak-6B-v1.0"
OUTPUT_DIR = "outputs/karnak-muslim-lora-v2"
TRAIN_FILE = "dataset/muslim_lora_train_v2.jsonl"
VAL_FILE = "dataset/muslim_lora_val_v2.jsonl"

# Turing (RTX 2080 Ti, sm75) has no bf16 tensor cores: compute dtype must stay fp16 everywhere.
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
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


TRAINING_CHAT_TEMPLATE_PATH = "train/karnak_training_chat_template.jinja"


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Karnak's shipped chat_template.jinja has no {% generation %} markers, so TRL can't
    # auto-derive assistant-only-loss masks from it. Swap in a hand-patched copy (same
    # rendering, just with generation markers) for training, then restore the original
    # before saving so the artifact served by vLLM keeps the untouched production template.
    original_chat_template = tokenizer.chat_template
    with open(TRAINING_CHAT_TEMPLATE_PATH, encoding="utf-8") as f:
        tokenizer.chat_template = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        dtype=torch.float16,
        device_map={"": 0},
    )

    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    )
    train_dataset = dataset["train"].remove_columns(["behavior", "intent"])
    eval_dataset = dataset["validation"].remove_columns(["behavior", "intent"])

    trackio.init(
        project="muslim-karnak-lora",
        name="karnak-muslim-lora-v2",
        config={
            "base_model": BASE_MODEL,
            "lora_r": LORA_CONFIG.r,
            "lora_alpha": LORA_CONFIG.lora_alpha,
            "epochs": 3,
            "learning_rate": 2e-4,
        },
    )

    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=4096,
        packing=False,
        assistant_only_loss=True,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        use_cache=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="trackio",
        run_name="karnak-muslim-lora-v2",
        project="muslim-karnak-lora",
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

    # TRL force-casts QLoRA adapter params to bf16 (QLoRA paper convention, assumes Ampere+).
    # Turing has no native bf16. Restore fp32 (PEFT's own default master-weight dtype for
    # quantized models) so fp16 mixed precision + GradScaler work correctly: fp32 master
    # params, fp16-autocast compute. Casting to literal fp16 leaf params instead breaks
    # GradScaler ("Attempting to unscale FP16 gradients").
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.chat_template = original_chat_template
    tokenizer.save_pretrained(OUTPUT_DIR)
    # TRL's own Trackio integration already closes the run when the training loop ends;
    # calling finish() again here would raise "Call trackio.init() before trackio.finish()".


if __name__ == "__main__":
    main()
