"""Delete the superseded Muslim model repos from HuggingFace, once
NightPrince/Muslim-6B-PRO is published and has passed its eval gate.

DO NOT RUN until that gate has actually passed -- confirmed with the user
(2026-07-20): delete all old versions once V1 ships and is verified good.
This is a real, irreversible deletion of public HF repos (anyone who
already cloned a copy keeps it, but the repos/links disappear).

Run: .venv/bin/python train/delete_old_versions.py --confirm
"""
import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

OLD_REPOS = [
    "NightPrince/Muslim-6B-V1.0",  # the original first fine-tune (predates this project's v2/v3)
    "NightPrince/Muslim-6B-v2",
    "NightPrince/Muslim-6B-v3",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--confirm", action="store_true", help="actually delete (otherwise dry-run)")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (check .env)")
    api = HfApi(token=token)

    for repo in OLD_REPOS:
        if not args.confirm:
            print(f"[DRY RUN] would delete model repo: {repo}")
            continue
        try:
            api.delete_repo(repo_id=repo, repo_type="model")
            print(f"deleted: {repo}")
        except Exception as e:
            print(f"FAILED to delete {repo}: {e}")

    if not args.confirm:
        print("\nRe-run with --confirm to actually delete. Only do this after "
              "NightPrince/Muslim-6B-PRO has passed its eval gate.")


if __name__ == "__main__":
    main()
