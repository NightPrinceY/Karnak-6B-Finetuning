# Muslim-6B-PRO — Live Demo (Hugging Face Space)

**Live**: [huggingface.co/spaces/NightPrince/muslim-6b-pro-demo](https://huggingface.co/spaces/NightPrince/muslim-6b-pro-demo)

A Gradio chat demo for [NightPrince/Muslim-6B-PRO](https://huggingface.co/NightPrince/Muslim-6B-PRO)
that actually exercises the model's core capability: **real tool-calling against real Islamic
knowledge sources**, not a mocked or scripted response. When the model needs a tafsir, a hadith,
a fatwa, or ayah/surah audio, this app calls the actual tool — live — and feeds the real result
back to the model for its final answer, exactly matching the training-time tool-call format.

This is not the deployment source of truth (that's the Space's own git repo on the Hub) — it's
mirrored here so the demo's code, and the reasoning behind it, live alongside the model that
powers it.

## What it does

1. Loads `NightPrince/Muslim-6B-PRO` on **ZeroGPU** (dynamic, per-request GPU allocation — free
   for the Space creator, no dedicated hardware billing).
2. At startup, **live-fetches the real tool schemas** from the same public MCP servers the model
   was trained against:
   - [mcp.tafsir.net](https://mcp.tafsir.net) — 17 tools (tafsir, nuzool reasons, qeraat variants,
     word analysis, hadith search, cross-references, …)
   - [islamqa-mcp.org](https://islamqa-mcp.org) — 5 tools (fatwa search/fetch)
   - Plus 2 local tools (`play_ayah`, `play_surah`) that construct real audio-CDN URLs
     (everyayah.com / quranicaudio.com), matching the production agent's own audio player.
3. On each chat turn: generates a response, and if the model emits a `<tool_call>`, actually
   dispatches it — a real MCP request, or a real audio URL — appends the real result as a
   `tool`-role message, and generates the final spoken answer from that.
4. Also exposes an **MCP server** (`demo.launch(mcp_server=True)`), so other agents can call this
   Space as a tool.

## Why this matters for this model specifically

Muslim-6B-PRO's entire design premise is **"retrieved, not recited"** — it is deliberately *not*
trained to answer Qur'an/hadith/tafsir questions from memory, because language models reliably
hallucinate scripture when asked to recite it directly. A demo that faked or mocked tool responses
would misrepresent the model entirely. This demo exists to prove the real behavior: watch it
actually call `fetch_ayah(2, 255)` against a live server and get back the real text of Ayat
al-Kursi, not a memorized string.

## Files

| File | Purpose |
|---|---|
| `app.py` | The full Gradio app: model loading, live MCP tool discovery, chat loop, tool dispatch, audio playback |
| `requirements.txt` | `transformers`, `accelerate`, `mcp` (the official MCP Python client) — `gradio`/`spaces` are platform-managed and deliberately not pinned here |
| `system_prompt.txt` | The real production system prompt (`dataset/muslim_system_prompt.txt`), adapted to drop the one tool (`web_search_exa`) not included in this demo |

## Architecture notes

### ZeroGPU pattern

The model is loaded once at **module scope** (`AutoModelForCausalLM.from_pretrained(..., device_map="cuda")`)
— `import spaces` monkey-patches `torch` so this succeeds even though no real GPU is attached at
import time. The actual generation call (`_generate`, decorated `@spaces.GPU(duration=60)`) is
the only place real GPU compute happens; ZeroGPU streams the packed weights into VRAM on first
entry and reuses the warm worker across requests.

Each chat turn can call `_generate` up to twice (initial generation, then again after a tool
result comes back) — both calls, plus the tool dispatch in between, happen within the same async
turn rather than each `@spaces.GPU` entry racing independently.

### A real bug found and fixed: MCP session race condition

Early versions of this app intermittently failed to reach `mcp.tafsir.net` with
`McpError: Session terminated`, thrown from inside `list_tools()`/`call_tool()` right after a
successful `initialize()` handshake. Debugging (see the commit history) traced this to a genuine
race condition: some MCP servers need a brief pause after the initialize handshake completes
before they're ready to serve the next request. The fix is a retry-with-backoff wrapper
(`RETRY_DELAYS = (0.5, 1.5, 3.0, 5.0, 8.0)` seconds) around every MCP call, applied both at
startup tool-discovery and at live tool-call time. This was **not** a code bug in the request
itself — a plain retry with an inter-request delay resolves it, and the same exact call succeeds
immediately when run from a lower-latency network path.

### Gradio 6 note

`gr.Chatbot(type="messages")` will crash on Gradio 6.x — the `type` parameter was removed because
the legacy "tuples" format no longer exists; messages format is now the only option. This app
already reflects that (no `type=` kwarg).

## Running it yourself

```bash
pip install -r requirements.txt
python app.py
```

No GPU is required to *develop* against this file locally — `@spaces.GPU` is a no-op off of
ZeroGPU, and `import spaces` succeeds everywhere. A real GPU (or CPU, slowly) is needed to
actually run `model.generate()`, since ZeroGPU's dynamic allocation only exists on the Hub.

To redeploy to a Space:

```bash
hf repos create <namespace>/<name> --type space --space-sdk gradio --flavor zero-a10g --public
hf upload <namespace>/<name> . --repo-type space
```

## Related resources

- Full-precision model: [NightPrince/Muslim-6B-PRO](https://huggingface.co/NightPrince/Muslim-6B-PRO)
- GGUF quantizations (for local `llama.cpp` inference): [NightPrince/Muslim-6B-PRO-GGUF](https://huggingface.co/NightPrince/Muslim-6B-PRO-GGUF)
- Training dataset: [NightPrince/muslim-6b-v1-dataset](https://huggingface.co/datasets/NightPrince/muslim-6b-v1-dataset)

See the main [project README](../README.md) and the [model card](https://huggingface.co/NightPrince/Muslim-6B-PRO)
for the full training pipeline, dataset composition, and evaluation status.
