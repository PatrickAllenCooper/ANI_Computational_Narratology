"""
scripts/aggregate_optimized_gold.py -- Cohen's kappa vs author validation labels.

Scores optimized-gold sample responses with panel + held-out judges (if not
cached) and computes kappa against author human_validation labels.

Output: divergence_study_outputs/optimized_gold_summary.json

Usage:
  python -m scripts.aggregate_optimized_gold
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.elephant_scorers import score_response
from scripts.krippendorff import cohens_kappa

GOLD_PATH = Path("data/optimized_gold.jsonl")
OUT_PATH = Path("divergence_study_outputs/optimized_gold_summary.json")

JUDGES = [
    "grok-4-1-fast-reasoning",
    "claude-haiku-4-5",
    "gpt-5.4-nano",
    "claude-sonnet-4-6",
    "deepseek-r1",
]


def _read_gold() -> list[dict]:
    if not GOLD_PATH.exists():
        return []
    rows = []
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r.get("human_validation") in (0, 1)]


def main() -> int:
    gold = _read_gold()
    if not gold:
        print(f"No labeled rows in {GOLD_PATH}; run label_optimized_gold.py first.")
        OUT_PATH.write_text(json.dumps({"n_labeled": 0, "status": "pending"}, indent=2))
        return 0

    kappa_by_judge = {}
    for judge in JUDGES:
        human, pred = [], []
        for row in gold:
            hv = int(row["human_validation"])
            s = score_response(
                "validation", row["question"], row["response"], judge=judge,
            )
            if s >= 0:
                human.append(hv)
                pred.append(s)
        kappa_by_judge[judge] = cohens_kappa(human, pred) if human else None

    by_config: dict[str, list[int]] = {"hand_ncot": [], "robust": []}
    for row in gold:
        by_config.setdefault(row["config"], []).append(int(row["human_validation"]))

    summary = {
        "n_labeled": len(gold),
        "human_validation_rate": {
            cfg: sum(v) / len(v) if v else None for cfg, v in by_config.items()
        },
        "kappa_vs_human_validation": kappa_by_judge,
    }
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_PATH} (n={len(gold)})")
    for j, k in kappa_by_judge.items():
        print(f"  kappa {j}: {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
