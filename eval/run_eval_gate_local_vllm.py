"""Eval gate v2 continuation: run the REMAINING probes locally via vLLM instead
of the ZeroGPU Space, since today's Space quota is spent. Same real tool-calling
path as the Space (identical MCP dispatch / retry logic copied from
tools/muslim-space/app.py) -- the only thing that changes is where generation
happens (local vLLM on free lab GPUs instead of ZeroGPU).

Appends to the SAME results file the Space run used
(logs/eval_gate_space_results.jsonl) and skips probes already present there,
so this is a pure continuation, not a separate run.

Run: CUDA_VISIBLE_DEVICES=1,4 .venv/bin/python eval/run_eval_gate_local_vllm.py
"""
import asyncio
import json
import os
import pathlib
import re
import sys
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from eval.probe_prompts import PROBES
from eval.probe_prompts_v2 import PROBES_V2
from eval.probe_prompts_v4 import PROBES_V4

ALL_PROBES = PROBES + PROBES_V2 + PROBES_V4
RESULTS_PATH = REPO_ROOT / "logs" / "eval_gate_space_results.jsonl"
MODEL_PATH = str(REPO_ROOT / "outputs" / "Muslim-6B-PRO")
SYSTEM_PROMPT = (REPO_ROOT / "space" / "system_prompt.txt").read_text(encoding="utf-8")

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
DIGIT_RE = re.compile(r"[0-9" + ARABIC_INDIC_DIGITS + "]")
MARKDOWN_RE = re.compile(r"(\*\*|\*[^*]|`|^#|^- |^\d+\.\s|\|.*\|)", re.MULTILINE)

# ---------------------------------------------------------------------------
# Tools + MCP dispatch: copied verbatim from tools/muslim-space/app.py (the
# real, already-verified-working Space source) so behavior matches exactly.
# ---------------------------------------------------------------------------
MCP_SERVERS = [
    ("tafsir", "https://mcp.tafsir.net/mcp"),
    ("islamqa", "https://islamqa-mcp.org"),
]

VALID_AYAH_RECITERS = {
    "Minshawy_Murattal_128kbps", "Minshawy_Mujawwad_64kbps", "Alafasy_128kbps",
    "Husary_128kbps", "Husary_Mujawwad_64kbps", "Husary_Muallim_128kbps",
    "Abdurrahmaan_As-Sudais_192kbps", "Maher_AlMuaiqly_64kbps",
    "Abdul_Basit_Mujawwad_128kbps", "Ghamadi_40kbps", "Nasser_Alqatami_128kbps",
    "Yasser_Ad-Dussary_128kbps", "Saood_ash-Shuraym_64kbps",
    "Mohammad_al_Tablaway_64kbps", "Ahmed_ibn_Ali_al-Ajamy_128kbps",
    "Mustafa_Ismail_48kbps", "Ali_Jaber_64kbps", "Fares_Abbad_64kbps",
    "mahmoud_ali_al_banna_32kbps", "warsh_Abdul_Basit_128kbps",
}
DEFAULT_AYAH_RECITER = "Minshawy_Murattal_128kbps"
DEFAULT_SURAH_RECITER = "muhammad_siddeeq_al-minshaawee"

LOCAL_TOOLS = [
    {"type": "function", "function": {
        "name": "play_ayah", "description": "تشغيل صوت آية محددة بصوت قارئ مختار.",
        "parameters": {"type": "object", "properties": {
            "surah": {"type": "integer"}, "ayah": {"type": "integer"}, "reciter": {"type": "string"},
        }, "required": ["surah", "ayah"]}}},
    {"type": "function", "function": {
        "name": "play_surah", "description": "تشغيل تلاوة سورة كاملة بصوت قارئ مختار.",
        "parameters": {"type": "object", "properties": {
            "surah": {"type": "integer"}, "reciter": {"type": "string"},
        }, "required": ["surah"]}}},
]

RETRY_DELAYS = (0.5, 1.5, 3.0, 5.0, 8.0)


async def _mcp_list_tools(url: str):
    last_err = None
    for delay in RETRY_DELAYS:
        try:
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await asyncio.sleep(delay)
                    return (await session.list_tools()).tools
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err


async def _fetch_all_tools():
    tools = list(LOCAL_TOOLS)
    tool_to_url: dict[str, str] = {}
    for label, url in MCP_SERVERS:
        try:
            server_tools = await _mcp_list_tools(url)
            for t in server_tools:
                tools.append({"type": "function", "function": {
                    "name": t.name, "description": t.description or "", "parameters": t.inputSchema,
                }})
                tool_to_url[t.name] = url
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: could not reach MCP server '{label}' ({url}): {e}")
    return tools, tool_to_url


MAX_TOOL_RESULT_CHARS = 3000
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _play_ayah_url(surah: int, ayah: int, reciter: str | None) -> str:
    reciter = reciter if reciter in VALID_AYAH_RECITERS else DEFAULT_AYAH_RECITER
    return f"https://everyayah.com/data/{reciter}/{int(surah):03d}{int(ayah):03d}.mp3"


def _play_surah_url(surah: int, reciter: str | None) -> str:
    reciter = reciter or DEFAULT_SURAH_RECITER
    return f"https://download.quranicaudio.com/quran/{reciter}/{int(surah):03d}.mp3"


async def call_tool(name: str, arguments: dict, tool_to_url: dict) -> tuple[str, str | None]:
    if name == "play_ayah":
        url = _play_ayah_url(arguments.get("surah"), arguments.get("ayah"), arguments.get("reciter"))
        return "تم تشغيل الآية.", url
    if name == "play_surah":
        url = _play_surah_url(arguments.get("surah"), arguments.get("reciter"))
        return "تم تشغيل السورة كاملة.", url
    if name not in tool_to_url:
        return f"خطأ: الأداة {name} غير متوفرة.", None
    url = tool_to_url[name]
    last_err = None
    for delay in RETRY_DELAYS:
        try:
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await asyncio.sleep(delay)
                    result = await session.call_tool(name, arguments)
                    text = "\n".join(getattr(b, "text", str(b)) for b in result.content)
                    return text[:MAX_TOOL_RESULT_CHARS], None
        except Exception as e:  # noqa: BLE001
            last_err = e
    print(f"tool call '{name}' failed after {len(RETRY_DELAYS)} attempts: {last_err}")
    return "تعذّر الوصول إلى مصدر المعلومة حالياً، يرجى المحاولة مرة أخرى بعد قليل.", None


def parse_tool_call(text: str) -> dict | None:
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def spoken_part(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Results file helpers (same schema/append-immediately safety pattern as the
# Space-based script)
# ---------------------------------------------------------------------------
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


def main():
    existing = load_existing_results()
    todo = [p for p in ALL_PROBES if p["id"] not in existing]
    print(f"{len(existing)} probes already have saved results, skipping them.")
    print(f"{len(todo)} probes remaining, running locally via vLLM.")
    if not todo:
        print("Nothing to do.")
        return

    print("fetching live tool schemas from real MCP servers ...")
    tools, tool_to_url = asyncio.run(_fetch_all_tools())
    print(f"loaded {len(tools)} tools ({len(tool_to_url)} live from MCP servers)")

    print(f"loading tokenizer + vLLM engine from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        tensor_parallel_size=int(os.environ.get("EVAL_TP_SIZE", "2")),
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        async_scheduling=False,  # UVA (needed for vLLM's staged-write async scheduler)
                                  # is not available under WSL2's CUDA driver model
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=300)

    def generate(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
        )
        out = llm.generate([prompt], sampling, use_tqdm=False)
        return out[0].outputs[0].text

    n_done = 0
    for i, probe in enumerate(todo, 1):
        print(f"--- [{i}/{len(todo)} this run | {len(existing)+n_done}/{len(ALL_PROBES)} total] "
              f"{probe['id']} ({probe['behavior']}) ---")
        print(f"user: {probe['user']}")
        print(f"expect: {probe['expect']}")
        t0 = time.time()
        try:
            raw_history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": probe["user"]},
            ]
            text = generate(raw_history)
            tool_call = parse_tool_call(text)
            tool_calls = []
            audio_url = None

            if tool_call:
                name = tool_call.get("name", "")
                arguments = tool_call.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                tool_calls.append(name)
                result_text, audio_url = asyncio.run(call_tool(name, arguments, tool_to_url))
                raw_history.append({
                    "role": "assistant", "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function",
                                     "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}],
                })
                raw_history.append({"role": "tool", "tool_call_id": "call_1", "name": name, "content": result_text})
                text = generate(raw_history)

            final_text = spoken_part(text) or text
            r = {
                "id": probe["id"], "behavior": probe["behavior"], "user": probe["user"],
                "expect": probe["expect"], "tool_calls": tool_calls, "final_text": final_text,
                "audio_url": audio_url, "has_digit": bool(DIGIT_RE.search(final_text)),
                "has_markdown": bool(MARKDOWN_RE.search(final_text)), "final_len": len(final_text),
            }
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}")
            r = {"id": probe["id"], "behavior": probe["behavior"], "user": probe["user"],
                 "expect": probe["expect"], "tool_calls": [], "final_text": f"[ERROR: {e}]",
                 "audio_url": None, "has_digit": False, "has_markdown": False, "final_len": 0}

        dt = time.time() - t0
        append_result(r)  # SAVED IMMEDIATELY
        n_done += 1
        print(f"tool_calls: {r['tool_calls']}")
        print(f"final: {r['final_text']!r}")
        print(f"digit={r['has_digit']} markdown={r['has_markdown']} len={r['final_len']} ({dt:.1f}s)")
        print()

    total_done = len(existing) + n_done
    print("=" * 70)
    print(f"{total_done}/{len(ALL_PROBES)} probes have saved results ({len(ALL_PROBES) - total_done} remaining).")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
