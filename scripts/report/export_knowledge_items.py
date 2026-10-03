#!/usr/bin/env python3
"""Export the follow-up knowledge probe items for public replay.

Sources (working tree only - the exports themselves are what ships):
  runs_archprobe/followup_20261003/plan_frozen.json   (own synthetic items)
  runs_archprobe/followup_20261003/sources.json      (URL + SHA per source page)

Writes, under data_report/followup_20261003/:
  knowledge-questions.json   model-facing: id/question/options, NO gold
  knowledge-answer-key.json  grader-facing: id/date/answer/source URL+SHA
  knowledge-arena-replay.txt plaintext copy-paste replay format (no gold)

The set is a factual-recall probe of recent-news recall - NOT a known
pretraining-cutoff test - with 24 unique items each run under two option
rotations (repeated measures on the same items, not 48 independent facts).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "runs_archprobe/followup_20261003/plan_frozen.json"
SRCS = ROOT / "runs_archprobe/followup_20261003/v2/sources_v2.json"
OUT = ROOT / "data_report/followup_20261003"

NOTE = ("Factual-recall probe items about dated public events (recent-news "
        "recall). This is NOT a known-pretraining-cutoff test: no cutoff is "
        "asserted. Each item was run under two option rotations as repeated "
        "measures on the same item (48 rows total, 24 unique facts).")


def items() -> list[dict]:
    st = json.loads(PLAN.read_text())["stimuli"]["fu_horizon"]
    rows = st["items"] if isinstance(st, dict) and "items" in st else st
    return [r for r in rows if r.get("item_id")]


def main() -> None:
    rows = items()
    srcs = json.loads(SRCS.read_text()).get("items", {})
    rotations = json.loads(PLAN.read_text())["stimuli"]["fu_horizon"]["rotations"]
    question_items, answer_items = [], []
    for number, row in enumerate(rows, 1):
        public_id = f"q{number:02d}"
        base = list(row["options"])
        variants = [
            {"rotation": offset, "options": base[offset:] + base[:offset]}
            for offset in rotations
        ]
        question_items.append({"id": public_id, "question": row["question"],
                               "variants": variants})
        source = srcs.get(row["item_id"], {})
        support = ([{"url": source["source_url"],
                     "sha256": source["source_sha256"]}]
                   if source.get("verified") and source.get("source_url") else [])
        if row.get("sources") and not support:
            raise ValueError(f"missing verified source for {row['item_id']}")
        answer_items.append({
            "id": public_id, "source_id": row["item_id"],
            "date": row["event_date"], "answer": row["gold"], "source": support,
            "variants": [{"rotation": v["rotation"],
                          "choice": f"o{v['options'].index(row['gold'])}"}
                         for v in variants],
        })
    questions = {
        "schema": "jev-knowledge-questions.v2",
        "version": "followup-20261003.v1",
        "use": "factual recall probe (model-facing; no gold answers included)",
        "note": NOTE,
        "n_unique_items": len(rows),
        "rotations_per_item": len(rotations),
        "n_variants": len(rows) * len(rotations),
        "items": question_items,
    }
    key = {
        "schema": "jev-knowledge-answer-key.v2",
        "version": "followup-20261003.v1",
        "use": "grader-facing answer key with source provenance (URL + SHA-256 of the fetched page)",
        "note": NOTE,
        "items": answer_items,
    }
    lines = ["Jev knowledge probe - plaintext Arena-style replay",
             "24 unique items x 2 option rotations (repeated measures).",
             "Factual recall probe; NOT a known pretraining-cutoff test.",
             "No answers are included here - see knowledge-answer-key.json.",
             ""]
    lines.append("Copy each variant separately; reply with only the option key.")
    lines.append("")
    for row in question_items:
        for variant in row["variants"]:
            lines.append(f"{row['id']} / rotation {variant['rotation']}")
            lines.append("Context: This question is about public events.")
            lines.append(row["question"])
            for j, opt in enumerate(variant["options"]):
                lines.append(f"   o{j}. {opt}")
            lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "knowledge-questions.json").write_text(
        json.dumps(questions, indent=2) + "\n", encoding="utf-8")
    (OUT / "knowledge-answer-key.json").write_text(
        json.dumps(key, indent=2) + "\n", encoding="utf-8")
    (OUT / "knowledge-arena-replay.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"[ok] exported {len(rows)} items -> {OUT}")


if __name__ == "__main__":
    main()
