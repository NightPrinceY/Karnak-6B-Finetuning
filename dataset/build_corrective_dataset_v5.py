#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the actual training set for the v5 corrective QLoRA pass.

This is NOT "train on all of muslim_lora_train_v5.jsonl" -- that would mean
2,717 gradient steps' worth of data dominated by content the model already
handles correctly, diluting the narrow correction signal this pass exists
to deliver. Instead: every row that's NEW in v5 (not present in v4 at all --
the actual corrective content: alt-names, sihr, B7-B13 tool-chaining, etc.)
plus a stratified-by-behavior random resample of the ORIGINAL v4 rows, so
the corrective LoRA sees enough of the model's already-correct behavior
interleaved to guard against catastrophic forgetting, without retraining on
the full 2,731-row v4 set from scratch.

Run: MUSLIM_REPO=/home/elijah/src/Muslim python3 dataset/build_corrective_dataset_v5.py
"""
import hashlib
import json
import pathlib
import random
from collections import Counter, defaultdict

random.seed(1407)  # same seed as build_lora_dataset_v5.py, for reproducibility

HERE = pathlib.Path(__file__).resolve().parent
RESAMPLE_FRACTION = 0.20  # ~20% of v4, stratified by behavior


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(d):
    # Same discriminator as build_lora_dataset_v5.py's dedup: intent + system
    # + user content. Two rows with this triple identical are the same
    # training example.
    return hashlib.md5(
        (d["intent"] + "||" + d["messages"][0]["content"] + "||" + d["messages"][1]["content"]).encode()
    ).hexdigest()


def main():
    v4_train = load_jsonl(HERE / "muslim_lora_train_v4.jsonl")
    v4_val = load_jsonl(HERE / "muslim_lora_val_v4.jsonl")
    v5_train = load_jsonl(HERE / "muslim_lora_train_v5.jsonl")
    v5_val = load_jsonl(HERE / "muslim_lora_val_v5.jsonl")

    v4_all = v4_train + v4_val
    v5_all = v5_train + v5_val
    v4_keys = {row_key(d) for d in v4_all}

    new_rows = [d for d in v5_all if row_key(d) not in v4_keys]
    print(f"v4 total: {len(v4_all)}, v5 total: {len(v5_all)}, new-in-v5: {len(new_rows)}")

    # Stratified resample of v4 by behavior, so B1's ~1943-row dominance
    # doesn't crowd out the small-but-critical B2/B3/B9 slices in the sample.
    by_behavior = defaultdict(list)
    for d in v4_all:
        by_behavior[d["behavior"]].append(d)

    resampled = []
    for behavior, rows in sorted(by_behavior.items()):
        k = max(1, round(len(rows) * RESAMPLE_FRACTION))
        resampled.extend(random.sample(rows, k))
    print(f"stratified v4 resample ({RESAMPLE_FRACTION:.0%} per behavior): {len(resampled)}")

    combined = new_rows + resampled
    # de-dup in case a resampled v4 row and a "new" row somehow collide
    seen, uniq = set(), []
    for d in combined:
        k = row_key(d)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(d)
    random.shuffle(uniq)

    n_val = max(12, int(len(uniq) * 0.08))
    val, train = uniq[:n_val], uniq[n_val:]

    with open(HERE / "muslim_lora_train_v5_corrective.jsonl", "w", encoding="utf-8") as f:
        for d in train:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(HERE / "muslim_lora_val_v5_corrective.jsonl", "w", encoding="utf-8") as f:
        for d in val:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    bc = Counter(d["behavior"] for d in uniq)
    new_bc = Counter(d["behavior"] for d in new_rows)
    print(f"TOTAL corrective set: {len(uniq)} (train {len(train)} / val {len(val)})")
    print("by behavior (full corrective set):", dict(sorted(bc.items())))
    print("by behavior (new-only subset):", dict(sorted(new_bc.items())))


if __name__ == "__main__":
    main()
