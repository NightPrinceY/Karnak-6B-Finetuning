"""v5 CORRECTIVE QLoRA pass on top of the already-merged Muslim-6B-PRO --
NOT a retrain from the Karnak base. Forked from train/sft_lora.py (which
stays untouched as the record of how Muslim-6B-PRO itself was produced).

This trains a small, gentle new adapter on dataset/muslim_lora_train_v5_corrective.jsonl
(built by dataset/build_corrective_dataset_v5.py: every genuinely-new v5 row
plus a stratified 20%-per-behavior resample of the original v4 set, for
forgetting mitigation) targeting the specific findings from the 57-probe eval
gate re-verification (see the corrective plan): alt-name/play_surah routing,
B7-B13 tool-chaining, sihr ruling phrasing, Ayat-al-Kursi text-recitation
guard, identity without the system prompt, and related fixes.

Single GPU:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python train/sft_lora_v5_corrective.py 2>&1 | tee logs/run_v5_corrective.log

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

BASE_MODEL = "outputs/Muslim-6B-PRO"  # the already-merged, already-shipped model --
                                       # this pass corrects it further, doesn't retrain it
OUTPUT_DIR = "outputs/karnak-muslim-lora-v5-corrective"
TRAIN_FILE = "dataset/muslim_lora_train_v5_corrective.jsonl"
VAL_FILE = "dataset/muslim_lora_val_v5_corrective.jsonl"

# Turing (RTX 2080 Ti, sm75) has no bf16 tensor cores: compute dtype must stay fp16 everywhere.
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    # Gentler than the original full run (r=16/alpha=32): a narrow-capacity
    # adapter matches the narrow correction goal and reduces the chance of
    # overwriting broad behaviors that already work well (identity-holding,
    # B1 tool-routing discipline -- both scored cleanly on the eval gate).
    r=8,
    lora_alpha=16,
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
            name="karnak-muslim-lora-v5-corrective",
            config={
                "base_model": BASE_MODEL,
                "lora_r": LORA_CONFIG.r,
                "lora_alpha": LORA_CONFIG.lora_alpha,
                "epochs": 1,
                "learning_rate": 5e-5,
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
        # Gentler than the original full run: this is a narrow correction on
        # top of an already-converged model, not a rewrite. 1 epoch + a ~4x
        # lower LR reduces the risk of the small corrective set (with its
        # stratified v4 resample) overfitting or drowning out the broad
        # behaviors that already work.
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=grad_accum_steps,
        ddp_find_unused_parameters=False if num_processes > 1 else None,
        learning_rate=5e-5,
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
        # Step-based, not epoch-based: same rationale as the original v4 run
        # (WSL2/host can restart underneath a long training process outside
        # anything tmux can protect against). Started at eval_steps=15 (~6
        # checkpoints across the ~95-step run); each eval pass measured
        # ~36 minutes under heavy shared-GPU contention on this machine, so
        # raised to 40 (~2-3 checkpoints total, still enough for the
        # protected-best-checkpoint mechanism to have real signal without
        # burning hours on eval overhead alone). Eval/save frequency doesn't
        # touch gradients or the training trajectory -- resuming from
        # checkpoint-15 with a coarser schedule reproduces the identical run,
        # just measured less often.
        # NOTE: on resume, the actually-effective value comes from
        # TrainerState.eval_steps/save_steps (loaded from the checkpoint's
        # trainer_state.json), NOT from this TrainingArguments value --
        # DefaultFlowCallback checks state.eval_steps, and resuming does not
        # overwrite it from args (only prints a mismatch warning). Any future
        # change here MUST also be hand-patched into the latest checkpoint's
        # trainer_state.json before resuming, or it will be silently ignored.
        eval_strategy="steps",
        eval_steps=40,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=2,  # keep the 2 most recent -- enough margin if the latest write
                            # is ever mid-flush during an interruption, without wasting disk
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="trackio",
        run_name="karnak-muslim-lora-v5-corrective",
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
