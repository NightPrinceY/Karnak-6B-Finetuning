#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the Muslim LoRA dataset: schema, role-sequence, tool sanity, TTS-cleanliness, dedup."""
import json, re, sys, glob, pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
FILES = [HERE / "muslim_lora_train.jsonl", HERE / "muslim_lora_val.jsonl"]

VALID_SURAH = range(1, 115)
import os
_qp = HERE / "quran.json"
if not _qp.exists():
    _qp = pathlib.Path(os.getenv("MUSLIM_REPO", "/home/yahya/src/Muslim")) / "servers/validator/data/quran.json"
SURA_COUNT = {}
for r in json.load(open(_qp, encoding="utf-8")):
    SURA_COUNT[r["sura_id"]] = SURA_COUNT.get(r["sura_id"], 0) + 1
AYAH_KEYS = {"Minshawy_Murattal_128kbps", "Alafasy_128kbps", "Husary_128kbps",
             "Abdurrahmaan_As-Sudais_192kbps", "Maher_AlMuaiqly_64kbps", "Abdul_Basit_Mujawwad_128kbps"}
SURAH_KEYS = {"muhammad_siddeeq_al-minshaawee", "mishaari_raashid_al_3afaasee",
              "abdurrahmaan_as-sudays", "mahmood_khaleel_al-husaree"}

_MD = re.compile(r"(\*\*|^#{1,6}\s|^[-*]\s|```|•|^\d+\.\s|^>\s|\]\(|https?://)", re.M)
_DIGIT = re.compile(r"\d")

errors, warns = [], []
seen_user = {}
n = 0
behav, intents = Counter(), Counter()
tool_use = 0
toolname_counts = Counter()

def err(i, msg):
    errors.append(f"[line {i}] {msg}")

for path in FILES:
    if not path.exists():
        err(0, f"missing file {path}")
        continue
    for li, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        n += 1
        tag = f"{path.name}:{li}"
        try:
            d = json.loads(line)
        except Exception as e:
            err(tag, f"JSON parse error: {e}"); continue

        # top-level
        for k in ("behavior", "intent", "messages", "tools"):
            if k not in d:
                err(tag, f"missing key {k}")
        if "messages" not in d:
            continue
        behav[d.get("behavior")] += 1
        intents[d.get("intent")] += 1
        msgs = d["messages"]
        tool_names = {t["function"]["name"] for t in d.get("tools", [])}

        # roles & sequence
        if msgs[0]["role"] != "system" or not msgs[0]["content"].strip():
            err(tag, "first msg must be non-empty system")
        if msgs[1]["role"] != "user" or not msgs[1]["content"].strip():
            err(tag, "second msg must be non-empty user")
        if msgs[-1]["role"] != "assistant":
            err(tag, "last msg must be assistant")

        # dedup on (intent,user)
        ukey = (d.get("intent"), msgs[1]["content"])
        if ukey in seen_user:
            warns.append(f"[{tag}] duplicate of {seen_user[ukey]}")
        else:
            seen_user[ukey] = tag

        # tool round-trip validation
        has_tool = any(m["role"] == "tool" for m in msgs)
        if has_tool:
            tool_use += 1
            # find assistant-with-tool_calls -> tool -> assistant
            tc_msg = next((m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")), None)
            tl_msg = next((m for m in msgs if m["role"] == "tool"), None)
            if not tc_msg:
                err(tag, "tool message present but no assistant tool_calls")
            else:
                for tc in tc_msg["tool_calls"]:
                    fnname = tc["function"]["name"]
                    toolname_counts[fnname] += 1
                    if fnname not in tool_names:
                        err(tag, f"tool_call '{fnname}' not in tools menu")
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except Exception as e:
                        err(tag, f"tool args not JSON: {e}"); continue
                    # arg sanity
                    if "surah" in args and args["surah"] not in VALID_SURAH:
                        err(tag, f"surah out of range: {args['surah']}")
                    if "ayah" in args and "surah" in args:
                        if args["surah"] in SURA_COUNT and not (1 <= args["ayah"] <= SURA_COUNT[args["surah"]]):
                            err(tag, f"ayah {args['ayah']} out of range for surah {args['surah']}")
                    if fnname == "play_ayah" and "reciter" in args and args["reciter"] not in AYAH_KEYS:
                        warns.append(f"[{tag}] unknown ayah reciter {args['reciter']}")
                    if fnname == "play_surah" and "reciter" in args and args["reciter"] not in SURAH_KEYS:
                        warns.append(f"[{tag}] unknown surah reciter {args['reciter']}")
                    # tool_call_id linkage
                    if tl_msg and tl_msg.get("tool_call_id") != tc["id"]:
                        warns.append(f"[{tag}] tool_call_id mismatch")

        # TTS-cleanliness of the FINAL assistant turn (spoken)
        final = msgs[-1]["content"]
        if _MD.search(final):
            err(tag, f"markdown leak in final turn: {final[:60]}")
        if _DIGIT.search(final):
            err(tag, f"raw digit in final spoken turn: {final[:60]}")
        if len(final) < 8:
            err(tag, "final turn too short")

print("=" * 60)
print(f"examples checked: {n}")
print(f"tool-calling examples: {tool_use} ({tool_use/max(n,1):.0%})")
print("behavior:", dict(sorted(behav.items())))
print("tools exercised:", dict(toolname_counts.most_common()))
print(f"warnings: {len(warns)}")
for w in warns[:10]:
    print("  ⚠", w)
print(f"ERRORS: {len(errors)}")
for e in errors[:30]:
    print("  ✗", e)
print("=" * 60)
print("RESULT:", "PASS ✓" if not errors else "FAIL ✗")
sys.exit(1 if errors else 0)
