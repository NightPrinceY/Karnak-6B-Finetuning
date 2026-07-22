"""QLoRA SFT of Karnak-6B for the Muslim agent. Behavior only — see dataset/DATACARD.md.

Single GPU:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python train/sft_lora.py 2>&1 | tee logs/run1.log

Multi-GPU (DDP, one quantized model replica per GPU -- only the LoRA adapter's
gradients need to sync across cards since the 4-bit base is frozen, so this is
cheap even over plain PCIe with no NVLink):
    .venv/bin/accelerate launch --multi_gpu --num_processes=8 train/sft_lora.py 2>&1 | tee logs/run4.log

gradient_accumulation_steps is computed from the actual process count at
runtime so the EFFECTIVE batch size (and therefore the LR/epoch schedule
this recipe was tuned around) stays the same regardless of how many GPUs
this is launched with.
"""

import glob
import os

import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import trackio

TARGET_EFFECTIVE_BATCH = 16  # per_device_train_batch_size(2) * grad_accum(8) * 1 GPU, the proven single-GPU recipe
# NOTE: multi-GPU DDP was attempted this session (only the LoRA adapter's gradients
# need to sync, which should have been cheap) but reliably OOM'd on this hardware even
# after fixing the underlying memory issues below (Liger kernel, tool-result
# truncation) -- DDP's own overhead (NCCL buffers, replica bookkeeping) apparently
# doesn't leave enough headroom on an 11GB 2080Ti stacked on top of a 6B QLoRA model
# at max_length=5120. Reverted to the proven single-GPU batch size; the code above
# still auto-adapts grad_accum if a future GPU with more headroom makes multi-GPU
# viable again.
PER_DEVICE_TRAIN_BATCH_SIZE = 2

BASE_MODEL = "Applied-Innovation-Center/Karnak-6B-v1.0"
OUTPUT_DIR = "outputs/karnak-muslim-lora-v4"
TRAIN_FILE = "dataset/muslim_lora_train_v4.jsonl"
VAL_FILE = "dataset/muslim_lora_val_v4.jsonl"

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

    # One quantized model replica per process/GPU under `accelerate launch`;
    # PartialState().process_index is 0 with a single, un-launched process
    # (plain `python train/sft_lora.py`), so this is safe either way.
    device_string = PartialState().process_index
    num_processes = PartialState().num_processes
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        dtype=torch.float16,
        device_map={"": device_string},
    )

    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    )
    train_dataset = dataset["train"].remove_columns(["behavior", "intent"])
    eval_dataset = dataset["validation"].remove_columns(["behavior", "intent"])

    is_main_process = PartialState().is_main_process
    if is_main_process:
        trackio.init(
            project="muslim-karnak-lora",
            name="karnak-muslim-lora-v4",
            config={
                "base_model": BASE_MODEL,
                "lora_r": LORA_CONFIG.r,
                "lora_alpha": LORA_CONFIG.lora_alpha,
                "epochs": 3,
                "learning_rate": 2e-4,
            },
        )

    # Keep the effective batch size (and therefore the proven LR/epoch
    # schedule) identical regardless of GPU count: grad_accum shrinks as
    # num_processes grows. Must divide evenly -- fail loudly rather than
    # silently train with a different effective batch size than intended.
    denom = PER_DEVICE_TRAIN_BATCH_SIZE * num_processes
    if TARGET_EFFECTIVE_BATCH % denom != 0:
        raise ValueError(
            f"TARGET_EFFECTIVE_BATCH={TARGET_EFFECTIVE_BATCH} not evenly divisible by "
            f"per_device_train_batch_size({PER_DEVICE_TRAIN_BATCH_SIZE}) * num_processes({num_processes}); "
            "pick a process count that divides evenly instead of silently changing the effective batch size."
        )
    grad_accum_steps = TARGET_EFFECTIVE_BATCH // denom
    print(f"num_processes={num_processes}, per_device_train_batch_size={PER_DEVICE_TRAIN_BATCH_SIZE}, "
          f"grad_accum_steps={grad_accum_steps} -> effective batch={TARGET_EFFECTIVE_BATCH}")

    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=grad_accum_steps,
        ddp_find_unused_parameters=False if num_processes > 1 else None,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # Raised from 4096: measured this session that the real 31-tool schema +
        # system prompt alone costs ~2,527 tokens on EVERY example, pushing
        # median example length to ~3,200 and ~9% of the v4 dataset past 4096
        # even after capping pathological tool-result outliers (see
        # MAX_TOOL_RESULT_CHARS in build_lora_dataset_v4.py / the merge
        # scripts). 5120 covers p99 (4,689 measured) with margin; max observed
        # after the outlier fix is 5,621, so a small number of the longest
        # examples still truncate from the end -- acceptable long-tail loss,
        # not the 9% this would otherwise be.
        max_length=5120,
        packing=False,
        assistant_only_loss=True,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        use_cache=False,
        use_liger_kernel=True,  # fused, memory-efficient CE loss -- avoids the float32-upcast
                                # memory spike in TRL's default chunked CE path that OOM'd on
                                # long sequences (median example is ~3200 tokens once the real,
                                # full 31-tool schema is included; some exceed max_length=4096)
        # Step-based, not epoch-based: a prior run died silently (WSL2/host
        # environment restarted underneath it -- outside anything tmux can
        # protect against) at step 68/474, well before the ~158-step first
        # epoch boundary, losing 2.5 hours with nothing to resume from.
        # Checkpointing every 40 steps (~1.3-1.5h at observed pace) bounds
        # the worst-case loss from any future interruption to that window.
        eval_strategy="steps",
        eval_steps=100,  # raised back to 100 after resuming from checkpoint-350 (near the end
                         # of the run, 474 total steps): next eval lands at 400 (redoing the one
                         # lost to a connection-layer tmux kill), then nothing until the forced
                         # final eval at 474 -- deliberately no eval at 450 per user request.
                         # NOTE: on resume, the actually-effective value comes from
                         # TrainerState.eval_steps/save_steps (loaded from the checkpoint's
                         # trainer_state.json), NOT from this TrainingArguments value --
                         # DefaultFlowCallback checks state.eval_steps, and resuming does not
                         # overwrite it from args (only prints a mismatch warning). Any future
                         # change here MUST also be hand-patched into the latest checkpoint's
                         # trainer_state.json before resuming, or it will be silently ignored.
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,  # keep the 2 most recent -- enough margin if the latest write
                            # is ever mid-flush during an interruption, without wasting disk
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="trackio",
        run_name="karnak-muslim-lora-v4",
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

    checkpoints = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    resume_from = checkpoints[-1] if checkpoints else None
    if resume_from:
        print(f"resuming from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(OUTPUT_DIR)  # HF Trainer.save_model already gates on is_world_process_zero internally
    if is_main_process:
        tokenizer.chat_template = original_chat_template
        tokenizer.save_pretrained(OUTPUT_DIR)
    # TRL's own Trackio integration already closes the run when the training loop ends;
    # calling finish() again here would raise "Call trackio.init() before trackio.finish()".


if __name__ == "__main__":
    main()
