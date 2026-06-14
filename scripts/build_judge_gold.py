"""
scripts/build_judge_gold.py -- Bootstrap data/judge_gold.jsonl from OEQ human scores.

Uses OSF crowdsourced human labels (validation_human, indirectness_human,
framing_human) on human OEQ responses as the gold standard for Phase 15.

Usage:
  python -m scripts.build_judge_gold
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.load_elephant import ELEPHANT_SEED, load_elephant

GOLD_PATH = Path("data/judge_gold.jsonl")
N_GOLD = 30


def main() -> int:
    data_dir = Path("data/elephant")
    items = load_elephant("oeq", n=150, seed=ELEPHANT_SEED, data_dir=data_dir, allow_sample=True)
    gold = []
    for item in items:
        if not item.human_response:
            continue
        hs = item.human_scores or {}
        entry = {
            "item_id": item.id,
            "prompt": item.prompt[:500],
            "response": item.human_response[:2000],
            "source": "human_oeq",
        }
        for metric in ("validation", "indirectness", "framing"):
            key = f"{metric}_human"
            if key in hs:
                entry[f"human_{metric}"] = int(hs[key])
            elif metric in hs:
                entry[f"human_{metric}"] = int(hs[metric])
        if any(f"human_{m}" in entry for m in ("validation", "indirectness", "framing")):
            gold.append(entry)
        if len(gold) >= N_GOLD:
            break

    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_PATH.open("w", encoding="utf-8") as f:
        for row in gold:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(gold)} gold rows to {GOLD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
