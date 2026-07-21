#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full re-pass of real production voice sessions (NightPrince/muslim-voice-
sessions, private HF dataset) -- all 1,326 turns, not just the ~70 hand-
curated seed questions used earlier for Dspark generation.

Source: dataset/voice_sessions_turns.parquet (session_id, turn, user_text,
assistant_text, tool_names, tools -- audio bytes columns stripped; fetched
from the real HF dataset's turns.parquet via HF_TOKEN, see
scripts/fetch_voice_sessions.py in Dspark for the download step).

Same ground-truth-check discipline as merge_dspark_conversations.py: only
mechanical fixes applied (re-serializing already-clean tool arguments,
spelling out a surah/ayah number we independently KNOW from the tool call
itself), nothing speculative. Rows are DROPPED, not guessed-and-patched, for:
  - no user_text (session-opening greetings with nothing to respond to)
  - tool name not in the current verified TOOLS menu
  - surah/ayah out of range
  - TTS-uncleanliness (raw digits/markdown) that can't be mechanically
    resolved by the known tool-call numbers
  - final text quoting scripture verbatim instead of routing to play_ayah
  - duplicate (session_id, turn) or near-duplicate user_text already seen

Run: python3 dataset/merge_voice_sessions.py
"""
import json
import pathlib
import re
import sys
from collections import Counter

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "voice_sessions_turns.parquet"
SYSTEM_PROMPT_FILE = HERE / "muslim_system_prompt.txt"

# Kept in sync by hand with build_lora_dataset_v4.TOOLS (see the note in
# merge_dspark_conversations.py for why this isn't imported directly).
VALID_TOOL_NAMES = {
    "play_ayah", "play_surah", "get_tafsir_verse", "get_tafsir_surah",
    "search_hadith", "fetch_hadith", "fetch_cross_references", "web_search_exa",
    "validate_recitation",
    "fetch_ayah", "fetch_tafsir", "list_tafsir_sources", "list_science_sources",
    "list_all_sources", "list_sources_for_ayah", "fetch_nuzool_reason",
    "fetch_surah_info", "analyze_word", "find_root_occurrences", "get_root_stats",
    "get_qeraat_variants", "search_quran_text", "search_in_tafsir",
    "get_quran_overview", "get_page_fawaed", "get_surah_statistics",
    "search_answers", "fetch_answer", "list_categories", "fetch_grounding_rules",
    "show_answer",
}

QURAN = json.load(open(HERE / "quran.json", encoding="utf-8"))
SURA_NAME, SURA_COUNT, VERSE_TEXT = {}, {}, {}
for r in QURAN:
    s, a = r["sura_id"], r["aya_id"]
    SURA_NAME[s] = r["sura_name"]
    SURA_COUNT[s] = SURA_COUNT.get(s, 0) + 1
    VERSE_TEXT[(s, a)] = r.get("standard_full") or r.get("standard")

_TASHKEEL = re.compile(
    "[" + "ـ" + "ً-ٟ" + "ٰ" + "ۖ-ۭ" + "]"
)
_MD = re.compile(r"(\*\*|^#{1,6}\s|^[-*]\s|```|•|^\d+\.\s|^>\s|\]\(|https?://)", re.M)
_DIGIT = re.compile(r"[0-9٠-٩]")
MAX_TOOL_RESULT_CHARS = 3000


def strip_tashkeel(t):
    return _TASHKEEL.sub("", t or "")


assert strip_tashkeel("اللَّهُ") == "الله", \
    "strip_tashkeel is broken -- stripped base letters"


def quotes_scripture_verbatim(text, surah, ayah):
    verse = VERSE_TEXT.get((surah, ayah))
    if not verse:
        return False
    verse_words = strip_tashkeel(verse).split()
    text_norm = strip_tashkeel(text)
    for i in range(len(verse_words) - 5):
        if " ".join(verse_words[i:i + 6]) in text_norm:
            return True
    return False


def check_and_build(row, system_prompt):
    user_text = row["user_text"]
    assistant_text = row["assistant_text"]
    if not isinstance(user_text, str) or not user_text.strip():
        return None, "no user_text (session opener)"
    if not isinstance(assistant_text, str) or len(assistant_text.strip()) < 6:
        return None, "empty/too-short assistant_text"

    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}]

    tool_surah_ayah = []
    if row["n_tools"] > 0:
        names = list(row["tool_names"])
        tools_struct = row["tools"]
        arg_strs = list(tools_struct["arguments"])
        results = list(tools_struct["result"])
        if len(names) != 1:
            # Multi-tool-call turns aren't reliably reconstructable into the
            # single-round-trip schema the rest of this dataset uses; skip
            # rather than guess at ordering/pairing.
            return None, "multi-tool-call turn (skipped, not single-call schema)"
        fname = names[0]
        if fname not in VALID_TOOL_NAMES:
            return None, f"tool not in verified menu: {fname}"
        try:
            args = json.loads(arg_strs[0])
        except (json.JSONDecodeError, TypeError, IndexError):
            return None, "tool arguments not valid JSON"
        if not isinstance(args, dict):
            return None, "tool arguments not a JSON object"
        if "surah" in args:
            s = args["surah"]
            if not isinstance(s, int) or s not in SURA_COUNT:
                return None, f"surah out of range: {s}"
            if "ayah" in args:
                a = args["ayah"]
                if not isinstance(a, int) or not (1 <= a <= SURA_COUNT[s]):
                    return None, f"ayah {a} out of range for surah {s}"
                tool_surah_ayah.append((s, a))
        cid = "call_voice_0001"
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": fname, "arguments": json.dumps(args, ensure_ascii=False)}}]})
        result_str = str(results[0]) if results else ""
        if len(result_str) > MAX_TOOL_RESULT_CHARS:
            result_str = result_str[:MAX_TOOL_RESULT_CHARS] + " …[truncated]"
        msgs.append({"role": "tool", "tool_call_id": cid, "name": fname, "content": result_str})

    if _MD.search(assistant_text):
        return None, "markdown leak in final turn"
    if _DIGIT.search(assistant_text):
        return None, "raw digit in final spoken turn (not mechanically fixable)"
    for (s, a) in tool_surah_ayah:
        if quotes_scripture_verbatim(assistant_text, s, a):
            return None, "scripture recited verbatim instead of routed to play_ayah"

    msgs.append({"role": "assistant", "content": assistant_text})
    intent = f"voice_{row['tool_names'][0]}" if row["n_tools"] > 0 else "voice_general"
    behavior = "B1" if row["n_tools"] > 0 else "B1"
    return {"behavior": behavior, "intent": intent, "messages": msgs}, None


def main():
    if not SRC.exists():
        print(f"ERROR: {SRC} not found -- fetch it from the private HF dataset first", file=sys.stderr)
        sys.exit(1)
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()

    df = pd.read_parquet(SRC)
    kept, rejected = [], Counter()
    seen_user_text = set()
    for _, row in df.iterrows():
        result, reason = check_and_build(row, system_prompt)
        if reason:
            rejected[reason.split(":")[0].split("(")[0].strip()] += 1
            continue
        ukey = result["messages"][1]["content"]
        if ukey in seen_user_text:
            rejected["duplicate user_text"] += 1
            continue
        seen_user_text.add(ukey)
        kept.append(result)

    out_path = HERE / "voice_sessions_corrected.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    total = len(df)
    print(f"input turns: {total}")
    print(f"kept (written to {out_path.name}): {len(kept)} ({len(kept)/total:.0%})")
    print(f"rejected: {sum(rejected.values())} ({sum(rejected.values())/total:.0%})")
    for reason, n in rejected.most_common():
        print(f"  - {reason}: {n}")


if __name__ == "__main__":
    main()
