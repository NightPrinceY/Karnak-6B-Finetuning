"""Push the Muslim-6B-V1 training dataset to the Hugging Face Hub.

Pushes only the FINAL, transformed training/val JSONL files + the dataset
card -- NOT the raw voice_sessions_turns.parquet intermediate (that's a more
direct copy of the private NightPrince/muslim-voice-sessions source with
session_ids/timestamps; the final JSONL already incorporates its content in
de-identified, transformed training-example form, which is what "the
dataset" means here).

Run: .venv/bin/python dataset/push_dataset.py
"""
import os
import pathlib

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
load_dotenv(REPO_ROOT / ".env")

HUB_REPO = "NightPrince/muslim-6b-v1-dataset"

FILES = {
    "muslim_lora_train_v4.jsonl": "muslim_lora_train_v4.jsonl",
    "muslim_lora_val_v4.jsonl": "muslim_lora_val_v4.jsonl",
    "DATACARD_V4.md": "README.md",
    "build_lora_dataset_v4.py": "build_lora_dataset_v4.py",
    "merge_dspark_conversations.py": "merge_dspark_conversations.py",
    "merge_voice_sessions.py": "merge_voice_sessions.py",
    "verify_surah_facts.py": "verify_surah_facts.py",
    "validate_dataset.py": "validate_dataset.py",
    "tafsir_net_surah_ground_truth.jsonl": "tafsir_net_surah_ground_truth.jsonl",
}


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (check .env)")
    api = HfApi(token=token)

    print(f"creating dataset repo {HUB_REPO} (public)...")
    create_repo(HUB_REPO, repo_type="dataset", private=False, exist_ok=True, token=token)

    for local_name, repo_path in FILES.items():
        local_path = HERE / local_name
        if not local_path.exists():
            print(f"SKIP (not found): {local_name}")
            continue
        print(f"uploading {local_name} -> {repo_path} ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=HUB_REPO,
            repo_type="dataset",
        )

    print(f"done: https://huggingface.co/datasets/{HUB_REPO}")


if __name__ == "__main__":
    main()
