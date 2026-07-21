"""Isolated diagnostic: is Liger Kernel actually helping or hurting step time
on this dataset's dynamic per-batch padding? Runs a handful of real training
steps with Liger on vs off, same GPU, same data, same everything else, and
prints per-step wall time. Not a full training run -- max_steps caps it short.

Run: CUDA_VISIBLE_DEVICES=1 .venv/bin/python train/diag_liger_test.py --liger
     CUDA_VISIBLE_DEVICES=1 .venv/bin/python train/diag_liger_test.py --no-liger
"""
import argparse
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Applied-Innovation-Center/Karnak-6B-v1.0"
TRAIN_FILE = "dataset/muslim_lora_train_v4.jsonl"
VAL_FILE = "dataset/muslim_lora_val_v4.jsonl"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
LORA_CONFIG = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


class StepTimer(TrainerCallback):
    def __init__(self):
        self.times = []
        self.last = None

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if self.last is not None:
            self.times.append(now - self.last)
        self.last = now


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--liger", action="store_true")
    p.add_argument("--no-liger", dest="liger", action="store_false")
    p.set_defaults(liger=True)
    p.add_argument("--steps", type=int, default=6)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    with open("train/karnak_training_chat_template.jinja", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=BNB_CONFIG, dtype=torch.float16, device_map={"": 0},
    )

    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    train_dataset = dataset["train"].remove_columns(["behavior", "intent"])
    eval_dataset = dataset["validation"].remove_columns(["behavior", "intent"])

    config = SFTConfig(
        output_dir="/tmp/diag_liger_test",
        max_steps=args.steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_length=5120,
        packing=False,
        assistant_only_loss=True,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        use_cache=False,
        use_liger_kernel=args.liger,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model, train_dataset=train_dataset, eval_dataset=eval_dataset,
        args=config, peft_config=LORA_CONFIG, processing_class=tokenizer,
    )
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    timer = StepTimer()
    trainer.add_callback(timer)

    print(f"=== LIGER={args.liger} STEPS={args.steps} ===")
    t0 = time.time()
    trainer.train()
    total = time.time() - t0
    print(f"TOTAL: {total:.1f}s for {args.steps} steps ({total/args.steps:.1f}s/step avg)")
    if timer.times:
        print(f"per-step times (excl. first): {[f'{t:.1f}' for t in timer.times]}")


if __name__ == "__main__":
    main()
