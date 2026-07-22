"""Push the already-merged model directory to the Hub directly, without reloading
the model into memory (the merge already happened via merge_and_push.py; this just
uploads the resulting files -- avoids competing for CPU/RAM with GGUF quantization
running in parallel).

Run: .venv/bin/python train/push_merged_only.py
"""
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MERGED_DIR = "outputs/Muslim-6B-PRO"
HUB_REPO = "NightPrince/Muslim-6B-PRO"

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not set (check .env)")

api = HfApi(token=token)
print(f"creating Hub repo {HUB_REPO} (public)...")
create_repo(HUB_REPO, private=False, exist_ok=True, token=token)

print(f"uploading {MERGED_DIR} -> {HUB_REPO} ...")
api.upload_folder(
    folder_path=MERGED_DIR,
    repo_id=HUB_REPO,
    repo_type="model",
)
print(f"done: https://huggingface.co/{HUB_REPO}")
