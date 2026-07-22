"""Muslim-6B-PRO — live chat demo with real tool-calling.

Loads the merged model and, at startup, fetches the REAL tool schemas from the
same public MCP servers (mcp.tafsir.net, islamqa-mcp.org) the model was
trained against — not a hardcoded snapshot. When the model emits a
<tool_call>, this actually calls the tool (a real MCP request, or a real
audio-CDN URL for play_ayah/play_surah) and feeds the result back for a
second generation pass, matching the exact training-time tool-call/tool-
response format.
"""
import spaces  # noqa: F401  -- MUST be imported before torch

import asyncio
import json
import re

import gradio as gr
import torch
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "NightPrince/Muslim-6B-PRO"

with open("system_prompt.txt", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# ---------------------------------------------------------------------------
# Model: load at module scope, real GPU attaches only inside @spaces.GPU calls
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# ---------------------------------------------------------------------------
# Tools: local audio tools + live-fetched schemas from real public MCP servers
# ---------------------------------------------------------------------------
MCP_SERVERS = [
    ("tafsir", "https://mcp.tafsir.net/mcp"),
    ("islamqa", "https://islamqa-mcp.org"),
]

# Real everyayah.com per-ayah reciter keys (production audio_player.py's own
# valid set) -- an invalid key here would silently 404 in real playback.
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
    {
        "type": "function",
        "function": {
            "name": "play_ayah",
            "description": "تشغيل صوت آية محددة بصوت قارئ مختار.",
            "parameters": {
                "type": "object",
                "properties": {
                    "surah": {"type": "integer"},
                    "ayah": {"type": "integer"},
                    "reciter": {"type": "string"},
                },
                "required": ["surah", "ayah"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_surah",
            "description": "تشغيل تلاوة سورة كاملة بصوت قارئ مختار.",
            "parameters": {
                "type": "object",
                "properties": {
                    "surah": {"type": "integer"},
                    "reciter": {"type": "string"},
                },
                "required": ["surah"],
            },
        },
    },
]


RETRY_DELAYS = (0.5, 1.5, 3.0, 5.0, 8.0)  # some MCP servers need a beat after the initialize
                                 # handshake before they'll serve a real request --
                                 # without this, list_tools/call_tool can race and raise
                                 # "Session terminated" on an otherwise-healthy server.
                                 # Backoff instead of a single fixed delay since the exact
                                 # warmup time varies with network path.


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
                tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    },
                })
                tool_to_url[t.name] = url
        except Exception as e:  # noqa: BLE001 -- best-effort at startup
            print(f"WARNING: could not reach MCP server '{label}' ({url}): {e}")
    return tools, tool_to_url


TOOLS, TOOL_TO_URL = asyncio.run(_fetch_all_tools())
print(f"loaded {len(TOOLS)} tools ({len(TOOL_TO_URL)} live from MCP servers)")

MAX_TOOL_RESULT_CHARS = 3000
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _play_ayah_url(surah: int, ayah: int, reciter: str | None) -> str:
    reciter = reciter if reciter in VALID_AYAH_RECITERS else DEFAULT_AYAH_RECITER
    return f"https://everyayah.com/data/{reciter}/{int(surah):03d}{int(ayah):03d}.mp3"


def _play_surah_url(surah: int, reciter: str | None) -> str:
    reciter = reciter or DEFAULT_SURAH_RECITER
    return f"https://download.quranicaudio.com/quran/{reciter}/{int(surah):03d}.mp3"


async def call_tool(name: str, arguments: dict) -> tuple[str, str | None]:
    """Dispatch a real tool call. Returns (text_result_for_model, audio_url_or_none)."""
    if name == "play_ayah":
        url = _play_ayah_url(arguments.get("surah"), arguments.get("ayah"), arguments.get("reciter"))
        return "تم تشغيل الآية.", url
    if name == "play_surah":
        url = _play_surah_url(arguments.get("surah"), arguments.get("reciter"))
        return "تم تشغيل السورة كاملة.", url
    if name not in TOOL_TO_URL:
        return f"خطأ: الأداة {name} غير متوفرة.", None
    url = TOOL_TO_URL[name]
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


@spaces.GPU(duration=60)
def _generate(messages: list[dict]) -> str:
    inputs = tokenizer.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=300, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


async def respond(message: str, chat_history: list[dict], raw_history: list[dict]):
    """Send a message to Muslim-6B-PRO, routing through real Islamic-knowledge
    tools (Qur'an tafsir, hadith search, fatwa lookup, ayah/surah audio) when
    the model calls one, instead of answering from memory.
    """
    raw_history = raw_history or []
    if not raw_history:
        raw_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    raw_history.append({"role": "user", "content": message})
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})

    text = _generate(raw_history)
    tool_call = parse_tool_call(text)
    audio_url = None

    if tool_call:
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        chat_history.append({
            "role": "assistant",
            "content": f"🔧 **{name}**(`{json.dumps(arguments, ensure_ascii=False)}`)",
        })
        result_text, audio_url = await call_tool(name, arguments)
        chat_history.append({"role": "assistant", "content": f"↩︎ {result_text[:300]}"})

        raw_history.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}],
        })
        raw_history.append({"role": "tool", "tool_call_id": "call_1", "name": name, "content": result_text})

        text = _generate(raw_history)

    final = spoken_part(text) or text
    chat_history.append({"role": "assistant", "content": final})
    raw_history.append({"role": "assistant", "content": final})

    return chat_history, raw_history, audio_url, ""


def clear_chat():
    return [], [], None, ""


with gr.Blocks(title="Muslim-6B-PRO") as demo:
    gr.Markdown(
        "# 🕌 Muslim-6B-PRO\n"
        "مساعد صوتي إسلامي — الإجابات مبنية على أدوات حقيقية (تفسير، حديث، فتوى، تلاوة) "
        "وليست محفوظة من الذاكرة. جرّب: «ما هي آية الكرسي؟» أو «شغّل سورة الفاتحة»."
    )
    chatbot = gr.Chatbot(height=480, rtl=True, label="المحادثة")
    audio_out = gr.Audio(label="تشغيل", autoplay=True)
    raw_state = gr.State([])

    with gr.Row():
        msg = gr.Textbox(placeholder="اكتب سؤالك هنا…", scale=5, show_label=False, rtl=True)
        send = gr.Button("إرسال", scale=1, variant="primary")
    clear = gr.Button("محادثة جديدة")

    gr.Examples(
        examples=[
            "ما هي آية الكرسي؟",
            "شغّل سورة الفاتحة",
            "فسّر لي الآية الأولى من سورة البقرة",
            "ابحث لي عن حديث عن الصبر",
        ],
        inputs=msg,
    )

    send.click(respond, [msg, chatbot, raw_state], [chatbot, raw_state, audio_out, msg])
    msg.submit(respond, [msg, chatbot, raw_state], [chatbot, raw_state, audio_out, msg])
    clear.click(clear_chat, None, [chatbot, raw_state, audio_out, msg])

demo.launch(mcp_server=True)
