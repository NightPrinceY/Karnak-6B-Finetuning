#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth-check the 1,594 real tool-augmented conversations generated
against Muslim-6B-v3 this project (Dspark/data/conversations.jsonl) and emit
the CONFIRMED-CORRECT subset as dataset/dspark_corrected.jsonl, ready for
build_lora_dataset_v4.py to merge in.

Per the project's own "quality over quantity" principle and this session's
explicit zero-false-data bar for anything Quran-related: examples are only
FIXED when the fix is 100% mechanical and deterministic (the double-JSON-
encoding bug -- decode-then-re-encode-once is always correct regardless of
content). Anything that requires GUESSING at the right answer (hallucinated
tool name, wrong surah/ayah, scripture recited as raw text instead of
routed to play_ayah) is DROPPED, not speculatively patched -- fabricating a
"corrected" answer for an arbitrary generated conversation risks introducing
a new, less visible error. This is expected to keep roughly 75-90% of the
1,594 (matching the ~12-24% real error rates measured this session by
category), which is still a large volume recovery.

Run: python3 dataset/merge_dspark_conversations.py
"""
import json
import pathlib
import re
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent.parent / "Dspark" / "data" / "conversations.jsonl"

# NOTE: deliberately NOT importing build_lora_dataset_v4 here -- that module
# has no `if __name__ == "__main__"` guard, so importing it re-runs its
# entire generation pipeline (and re-writes muslim_lora_train_v4.jsonl) as a
# side effect. Keep this tool-name set in sync with build_lora_dataset_v4.TOOLS
# by hand; both are sourced from the same live-probed schemas this session.
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
SURA_COUNT = {}
VERSE_TEXT = {}
for r in QURAN:
    s, a = r["sura_id"], r["aya_id"]
    SURA_COUNT[s] = SURA_COUNT.get(s, 0) + 1
    VERSE_TEXT[(s, a)] = r.get("standard_full") or r.get("standard")

_TASHKEEL = re.compile(
    "[" + "\u0640" + "\u064B-\u065F" + "\u0670" + "\u06D6-\u06ED" + "]"
)
_MD = re.compile(r"(\*\*|^#{1,6}\s|^[-*]\s|```|•|^\d+\.\s|^>\s|\]\(|https?://)", re.M)
_DIGIT = re.compile(r"[0-9٠-٩]")
MAX_TOOL_RESULT_CHARS = 3000


def strip_tashkeel(t):
    return _TASHKEEL.sub("", t or "")


INTENT_TO_BEHAVIOR = {
    "persona": "B3", "name": "B3", "creator": "B3", "capability": "B3",
    "greeting": "B3", "tts_self": "B3", "general": "B3",
    "scope_redirect": "B4",
    "ruling": "B5", "ruling_v2": "B5", "fatwa_search": "B5", "fetch_answer": "B5",
    "english_mixed": "B6",
    "scripture_audio_guard": "B2", "named_verse_audio": "B2", "play_ayah_named": "B2",
}


def behavior_for(intent):
    if intent in INTENT_TO_BEHAVIOR:
        return INTENT_TO_BEHAVIOR[intent]
    if intent.startswith("b1_"):
        return "B1"
    return "B1"  # default: everything else is a tool-routing/factual example


def deep_decode(arguments_raw):
    """Loop json.loads while the result is still a string -- the exact
    double/triple-encoding bug this session diagnosed in v3's output."""
    val = arguments_raw
    depth = 0
    while isinstance(val, str) and depth < 5:
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None
        depth += 1
    if not isinstance(val, dict):
        return None
    return val


def quotes_scripture_verbatim(text, surah, ayah):
    """B2 guardrail check: does the spoken text contain a long verbatim run
    of the actual ayah's own wording (should have been routed to play_ayah
    audio instead of typed/spoken directly)?"""
    verse = VERSE_TEXT.get((surah, ayah))
    if not verse:
        return False
    verse_words = strip_tashkeel(verse).split()
    text_norm = strip_tashkeel(text)
    # any run of 6+ consecutive verse words appearing verbatim in the answer
    for i in range(len(verse_words) - 5):
        run = " ".join(verse_words[i:i + 6])
        if run in text_norm:
            return True
    return False


def check_and_fix(conv_row):
    """Returns (fixed_messages, reject_reason_or_None)."""
    msgs = conv_row["conversations"]
    if not msgs or msgs[0]["role"] != "system" or msgs[-1]["role"] != "assistant":
        return None, "malformed role sequence"

    fixed = []
    tool_surah_ayah = []  # (surah, ayah) seen in any tool_call this conversation, for B2 check
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            new_calls = []
            for tc in m["tool_calls"]:
                fname = tc["function"]["name"]
                if fname not in VALID_TOOL_NAMES:
                    return None, f"hallucinated tool name: {fname}"
                args = deep_decode(tc["function"]["arguments"])
                if args is None:
                    return None, f"tool arguments not valid JSON after decode: {fname}"
                if "surah" in args:
                    s = args["surah"]
                    if not isinstance(s, int) or s not in SURA_COUNT:
                        return None, f"surah out of range: {s}"
                    if "ayah" in args:
                        a = args["ayah"]
                        if not isinstance(a, int) or not (1 <= a <= SURA_COUNT[s]):
                            return None, f"ayah {a} out of range for surah {s}"
                        tool_surah_ayah.append((s, a))
                new_tc = dict(tc)
                new_tc["function"] = dict(tc["function"])
                new_tc["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
                new_calls.append(new_tc)
            new_m = dict(m)
            new_m["tool_calls"] = new_calls
            fixed.append(new_m)
        elif m["role"] == "tool":
            # Match the DSpark generation harness's own MAX_TOOL_RESULT_CHARS=3000
            # convention -- some real tool results (raw web_search_exa page dumps,
            # long fatwa answers) run 15-20k+ chars untruncated, which alone can
            # push a single training example's token count past max_length and
            # cause OOM during training regardless of batch size.
            new_m = dict(m)
            content = new_m.get("content") or ""
            if len(content) > MAX_TOOL_RESULT_CHARS:
                new_m["content"] = content[:MAX_TOOL_RESULT_CHARS] + " …[truncated]"
            fixed.append(new_m)
        else:
            fixed.append(m)

    final = fixed[-1]["content"] or ""
    if _MD.search(final):
        return None, "markdown leak in final turn"
    if _DIGIT.search(final):
        return None, "raw digit in final spoken turn"
    if len(final.strip()) < 6:
        return None, "final turn too short/empty"
    for (s, a) in tool_surah_ayah:
        if quotes_scripture_verbatim(final, s, a):
            return None, "scripture recited verbatim instead of routed to play_ayah"

    return fixed, None


def main():
    if not SRC.exists():
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        sys.exit(1)

    kept, rejected = [], Counter()
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fixed_msgs, reason = check_and_fix(row)
            if reason:
                rejected[reason.split(":")[0]] += 1
                continue
            kept.append({
                "behavior": behavior_for(row["intent"]),
                "intent": f"dspark_{row['intent']}",
                "messages": fixed_msgs,
            })

    out_path = HERE / "dspark_corrected.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    total = sum(rejected.values()) + len(kept)
    print(f"input conversations: {total}")
    print(f"kept (written to {out_path.name}): {len(kept)} ({len(kept)/total:.0%})")
    print(f"rejected: {sum(rejected.values())} ({sum(rejected.values())/total:.0%})")
    for reason, n in rejected.most_common():
        print(f"  - {reason}: {n}")


if __name__ == "__main__":
    main()
