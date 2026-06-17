"""
scripts/aggregate_brokenmath_from_cache.py -- Aggregate BrokenMath from gen/score caches.

Builds brokenmath_raw.csv and brokenmath_summary.json from bm_gen_* / bm_score_*
caches without re-running generation. Useful while long-running nano cells complete.

Usage:
  python -m scripts.aggregate_brokenmath_from_cache
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.aggregate_brokenmath import aggregate
from scripts.brokenmath_scorer import score_brokenmath_response
from scripts.load_brokenmath import load_brokenmath
from scripts.run_brokenmath import _gen_cache, _system_prompt
from scripts.run_phase1_quartet import OUT_DIR

OUT_CSV = OUT_DIR / "brokenmath_raw.csv"
OUT_JSON = OUT_DIR / "brokenmath_summary.json"
DEFAULT_JUDGE = "claude-haiku-4-5"
GENERATORS = [
    "gpt-5.4-nano", "claude-haiku-4-5", "claude-sonnet-4-6", "grok-4-1-fast-reasoning",
]
ARMS = ["raw", "standard_cot", "narrative_cot"]


def main() -> int:
    items = load_brokenmath()
    rows = []
    for item in items:
        for gen in GENERATORS:
            for arm in ARMS:
                cache = _gen_cache(item.problem_id, gen, arm)
                if not cache.exists():
                    continue
                obj = json.loads(cache.read_text())
                response = (obj.get("response") or "").strip()
                if not response:
                    continue
                scored = score_brokenmath_response(
                    item.problem, item.original_problem, response,
                    problem_id=item.problem_id, judge=DEFAULT_JUDGE,
                )
                rows.append({
                    "problem_id": item.problem_id,
                    "question_type": item.question_type,
                    "generator": gen,
                    "arm": arm,
                    "response_len": len(response),
                    "empty_response": 0,
                    **scored,
                })

    if not rows:
        print("No cached BrokenMath cells found.")
        return 1

    fieldnames = sorted({k for r in rows for k in r})
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = aggregate(rows)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    for gen in GENERATORS:
        key = f"{gen}|standard_cot"
        if key in summary.get("cell_stats", {}):
            cs = summary["cell_stats"][key]
            print(f"  {gen} CoT: n={cs['n']} rate={cs['rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
