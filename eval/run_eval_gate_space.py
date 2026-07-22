"""Eval gate v2: run all 57 probes against the REAL deployed Space
(NightPrince/muslim-6b-pro-demo), not local HF generate(). This gets fast
ZeroGPU inference AND exercises the actual tool-execution round-trip (real
MCP calls, real audio URLs) rather than just checking whether a <tool_call>
was syntactically emitted -- a stricter test than the original run_eval_gate.py.

Each probe gets a FRESH conversation (empty history) so probes don't
cross-contaminate each other's context.

SAFETY (ZeroGPU quota is a hard daily budget, not free to burn on retries):
- Every probe result is appended to a JSONL file and flushed IMMEDIATELY --
  a Ctrl-C or crash mid-run never loses completed probes.
- Re-running this script SKIPS probes already present in the results file --
  it never re-spends quota on a probe that already has a real answer.
- --budget-seconds caps total wall-clock spent calling the Space; the script
  stops cleanly (not mid-probe) once the budget is exhausted, leaving the
  rest for a later run once quota resets.

Run: .venv/bin/python eval/run_eval_gate_space.py --budget-seconds 1500
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

from dotenv import load_dotenv
from gradio_client import Client

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))
from eval.probe_prompts import PROBES
from eval.probe_prompts_v2 import PROBES_V2
from eval.probe_prompts_v4 import PROBES_V4

ALL_PROBES = PROBES + PROBES_V2 + PROBES_V4
SPACE_ID = "NightPrince/muslim-6b-pro-demo"
RESULTS_PATH = REPO_ROOT / "logs" / "eval_gate_space_results.jsonl"

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
DIGIT_RE = re.compile(r"[0-9" + ARABIC_INDIC_DIGITS + "]")
MARKDOWN_RE = re.compile(r"(\*\*|\*[^*]|`|^#|^- |^\d+\.\s|\|.*\|)", re.MULTILINE)
TOOL_MARKER_RE = re.compile(r"🔧 \*\*(\w+)\*\*")


def load_existing_results() -> dict[str, dict]:
    if not RESULTS_PATH.exists():
        return {}
    out = {}
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["id"]] = r
    return out


def append_result(r: dict) -> None:
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_probe(client: Client, probe: dict) -> dict:
    result = client.predict(probe["user"], [], api_name="/respond")
    chat, audio_url, _ = result

    tool_calls = []
    final_text = ""
    for msg in chat:
        content = msg["content"]
        text = content[0]["text"] if isinstance(content, list) else content
        if msg["role"] == "assistant":
            m = TOOL_MARKER_RE.search(text)
            if m:
                tool_calls.append(m.group(1))
            elif not text.startswith("↩︎") and not text.startswith("↩"):
                final_text = text  # last non-tool-marker, non-tool-result assistant turn

    return {
        "id": probe["id"],
        "behavior": probe["behavior"],
        "user": probe["user"],
        "expect": probe["expect"],
        "tool_calls": tool_calls,
        "final_text": final_text,
        "audio_url": audio_url,
        "has_digit": bool(DIGIT_RE.search(final_text)),
        "has_markdown": bool(MARKDOWN_RE.search(final_text)),
        "final_len": len(final_text),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-seconds", type=float, default=1200,
                         help="stop cleanly once this much wall-clock time calling the Space has elapsed")
    args = parser.parse_args()

    existing = load_existing_results()
    todo = [p for p in ALL_PROBES if p["id"] not in existing]
    print(f"{len(existing)} probes already have saved results, skipping them.")
    print(f"{len(todo)} probes remaining. Budget: {args.budget_seconds:.0f}s.")
    if not todo:
        print("Nothing to do.")
        return

    print(f"connecting to {SPACE_ID} ...")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (check .env) -- required so calls are recognized as "
                          "the Space owner, not an anonymous visitor subject to daily ZeroGPU quota")
    client = Client(SPACE_ID, token=token, httpx_kwargs={"timeout": 300})

    spent = 0.0
    n_done = 0
    for i, probe in enumerate(todo, 1):
        if spent >= args.budget_seconds:
            print(f"Budget exhausted ({spent:.0f}s >= {args.budget_seconds:.0f}s) -- stopping cleanly "
                  f"before probe '{probe['id']}'. {len(todo) - n_done} probes left for next run.")
            break

        print(f"--- [{i}/{len(todo)} this run | {len(existing)+n_done}/{len(ALL_PROBES)} total] "
              f"{probe['id']} ({probe['behavior']}) ---")
        print(f"user: {probe['user']}")
        print(f"expect: {probe['expect']}")
        t0 = time.time()
        try:
            r = run_probe(client, probe)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}")
            r = {"id": probe["id"], "behavior": probe["behavior"], "user": probe["user"],
                 "expect": probe["expect"], "tool_calls": [], "final_text": f"[ERROR: {e}]",
                 "audio_url": None, "has_digit": False, "has_markdown": False, "final_len": 0}
        dt = time.time() - t0
        spent += dt
        append_result(r)  # SAVED IMMEDIATELY -- safe to Ctrl-C after this point
        n_done += 1
        print(f"tool_calls: {r['tool_calls']}")
        print(f"final: {r['final_text']!r}")
        print(f"digit={r['has_digit']} markdown={r['has_markdown']} len={r['final_len']} "
              f"({dt:.1f}s, {spent:.0f}s/{args.budget_seconds:.0f}s spent)")
        print()

    total_done = len(existing) + n_done
    print("=" * 70)
    print(f"{total_done}/{len(ALL_PROBES)} probes have saved results ({len(ALL_PROBES) - total_done} remaining).")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
