#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent fact-check for every hardcoded Quran surah/ayah claim used in
build_lora_dataset_v4.py: NAMED_VERSES (ayah nicknames) and ALT_SURAH_NAMES
(surah nicknames). Run this any time either list is edited.

Checks, for every entry:
  1. surah/ayah numbers are in valid range (quran.json ground truth).
  2. For NAMED_VERSES: the claimed ayah's real text contains an expected
     distinctive phrase (independent authorship, so a wrong ayah number
     can't silently pass just because the number is in-range).
  3. For ALT_SURAH_NAMES: the alt name appears in the REAL scholarly
     names_info text (dataset/tafsir_net_surah_ground_truth.jsonl, fetched
     live from mcp.tafsir.net) for the claimed surah's touqifi/ijtihadi
     name blocks specifically, AND does not appear in any OTHER surah's
     name blocks (catches ambiguous/colliding nicknames before they can
     train a wrong mapping).

Exits non-zero on any failure -- treat this as a hard gate before training.
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

QURAN = json.load(open(HERE / "quran.json", encoding="utf-8"))
SURA_NAME, SURA_COUNT, VERSE_TEXT = {}, {}, {}
for r in QURAN:
    s, a = r["sura_id"], r["aya_id"]
    SURA_NAME[s] = r["sura_name"]
    SURA_COUNT[s] = SURA_COUNT.get(s, 0) + 1
    VERSE_TEXT[(s, a)] = r.get("standard_full") or r.get("standard")

# Import the actual lists from the builder so this check can never drift
# from what training data really uses (re-derive by exec'ing just the
# module up to where these are defined would be fragile; instead the
# builder script re-declares them here verbatim -- keep in sync by hand,
# OR run this AFTER build_lora_dataset_v4.py and diff manually).
NAMED_VERSES = [
    ("آية الكرسي", 2, 255, "الحي القيوم"),
    ("آية النور", 24, 35, "نور السماوات"),
    ("آية الدّين", 2, 282, None),   # checked by length instead (longest ayah)
    ("آخر آية في سورة البقرة", 2, 286, None),
    ("آية الوضوء", 5, 6, "فاغسلوا وجوهكم"),
]

ALT_SURAH_NAMES = [
    ("أم الكتاب", 1), ("أم القرآن", 1), ("السبع المثاني", 1),
    ("براءة", 9),
    ("بني إسرائيل", 17),
    ("الملائكة", 35),
    ("المؤمن", 40),
    ("حم السجدة", 41),
    ("القتال", 47),
    ("تبارك", 67),
    ("هل أتى", 76),
    ("قل هو الله أحد", 112),
]

errors = []

# Arabic diacritics (tashkeel) + tatweel: strip before any text comparison,
# since quran.json's "standard_full" is fully diacritized but hand-typed
# expected phrases (and the tafsir.net scholarly quotes) normally are not.
_TASHKEEL = re.compile(
    "[" + "\u0640" + "\u064B-\u065F" + "\u0670" + "\u06D6-\u06ED" + "]"
)


def strip_tashkeel(t):
    return _TASHKEEL.sub("", t or "")


# Self-check: a prior version of this pattern (typed as literal Arabic
# glyphs in a regex range) silently stripped base letters too, which would
# make every containment check below spuriously pass against an empty
# string. Guard against that class of bug regenerating.
_self_check = strip_tashkeel("اللَّهُ")  # "اللَّهُ"
assert _self_check == "الله", (  # "الله"
    f"strip_tashkeel is broken -- expected 'الله', got {_self_check!r}"
)


def check_named_verses():
    for name, s, a, phrase in NAMED_VERSES:
        if s not in SURA_COUNT:
            errors.append(f"NAMED_VERSES: {name!r} surah {s} does not exist")
            continue
        if not (1 <= a <= SURA_COUNT[s]):
            errors.append(f"NAMED_VERSES: {name!r} ayah {a} out of range for surah {s} (max {SURA_COUNT[s]})")
            continue
        text = VERSE_TEXT.get((s, a))
        if not text:
            errors.append(f"NAMED_VERSES: {name!r} -> ({s},{a}) has no verse text in quran.json")
            continue
        if phrase and strip_tashkeel(phrase) not in strip_tashkeel(text):
            errors.append(f"NAMED_VERSES: {name!r} -> ({s},{a}) text does not contain expected phrase {phrase!r}: {text!r}")
        if name == "آية الدّين":
            # Ayat ad-Dayn (2:282) is the LONGEST ayah in the Quran -- verify
            # this structurally instead of by fixed phrase.
            longest = max(VERSE_TEXT.items(), key=lambda kv: len(kv[1] or ""))
            if longest[0] != (s, a):
                errors.append(f"NAMED_VERSES: 'آية الدّين' claimed as ({s},{a}) but the actual longest ayah is {longest[0]}")


def extract_name_blocks(names_info):
    """Return (touqifi_names, ijtihadi_names) parsed from the real scholarly text."""
    def block(headers):
        for h in headers:
            m = re.search(re.escape(h) + r"\s*\n(.+?)(?:\n\n|\n\*|\nأسماؤها|\nاسمها)", names_info, re.S)
            if m:
                return m.group(1).strip()
        return ""

    return block(["أسماؤها التوقيفية:", "اسمها التوقيفي:"]), block(["أسماؤها الاجتهادية:", "اسمها الاجتهادي:"])


def split_names(block_text, canonical):
    """Split a raw name-block string into individual name segments."""
    if not block_text:
        return []
    names = re.split(r"،\s*و?|,\s*", block_text)
    out = []
    for n in names:
        n = n.strip().rstrip(".").strip()
        n = re.sub(r"^سورة\s+", "", n)
        n = n.strip("{}").strip()
        if n and strip_tashkeel(n) != canonical:
            out.append(n)
    return out


def check_alt_surah_names():
    gt_path = HERE / "tafsir_net_surah_ground_truth.jsonl"
    if not gt_path.exists():
        errors.append(f"ALT_SURAH_NAMES: ground truth file missing: {gt_path} -- cannot verify, treat as FAIL")
        return
    stats_by_surah = {}
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            stats_by_surah[d["surah"]] = d["stats"]

    # Per surah, the set of EXACT name segments (split on the Arabic comma
    # separator, tashkeel-stripped) attributed to it -- not a raw substring
    # search, which would false-positive on compound names that merely
    # CONTAIN another name as a fragment (e.g. "سجدة المؤمن" for surah 41
    # contains the substring "المؤمن", which is surah 40's actual own
    # exact name -- these are different names, not a collision).
    segments_by_surah = {}
    for s in range(1, 115):
        ni = stats_by_surah[s].get("names_info") or ""
        canonical = strip_tashkeel(stats_by_surah[s].get("name") or "")
        touq, ijti = extract_name_blocks(ni)
        segs = set()
        for block_text in (touq, ijti):
            for n in split_names(block_text, canonical):
                segs.add(strip_tashkeel(n))
        segments_by_surah[s] = segs

    for name, claimed_s in ALT_SURAH_NAMES:
        if claimed_s not in SURA_COUNT:
            errors.append(f"ALT_SURAH_NAMES: {name!r} claimed surah {claimed_s} does not exist")
            continue
        norm_name = strip_tashkeel(name)
        if norm_name not in segments_by_surah.get(claimed_s, set()):
            errors.append(f"ALT_SURAH_NAMES: {name!r} NOT found as an exact name segment in surah {claimed_s}'s "
                          f"({SURA_NAME[claimed_s]}) real touqifi/ijtihadi name blocks")
            continue
        collisions = [s for s in range(1, 115) if s != claimed_s and norm_name in segments_by_surah.get(s, set())]
        if collisions:
            errors.append(f"ALT_SURAH_NAMES: {name!r} claimed unique to surah {claimed_s} "
                          f"but ALSO appears as an exact name segment for surah(s) {collisions} -- ambiguous, must exclude")


def check_full_quran_totals():
    if len(SURA_COUNT) != 114:
        errors.append(f"quran.json has {len(SURA_COUNT)} surahs, expected 114")
    total = sum(SURA_COUNT.values())
    if total != 6236:
        errors.append(f"quran.json has {total} total ayat, expected 6236")


if __name__ == "__main__":
    check_full_quran_totals()
    check_named_verses()
    check_alt_surah_names()

    print(f"NAMED_VERSES checked: {len(NAMED_VERSES)}")
    print(f"ALT_SURAH_NAMES checked: {len(ALT_SURAH_NAMES)}")
    print(f"ERRORS: {len(errors)}")
    for e in errors:
        print("  ✗", e)
    print("RESULT:", "PASS ✓" if not errors else "FAIL ✗")
    sys.exit(1 if errors else 0)
