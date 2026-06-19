"""
scripts/aggregate_output_lengths.py -- Mean response length per arm x generator.

Reads elephant_singleagent_raw.csv (no API calls) and reports mean character
length on the OEQ slice for the paper length-confound paragraph.

Output: divergence_study_outputs/output_lengths_summary.json

Usage:
  python -m scripts.aggregate_output_lengths
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path("divergence_study_outputs")
CSV_PATH = OUT / "elephant_singleagent_raw.csv"
SUMMARY_PATH = OUT / "output_lengths_summary.json"

ARMS = ["raw", "baseline_io", "standard_cot", "narrative_cot"]
GENERATORS = [
    "gpt-5.4-nano",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "grok-4-1-fast-reasoning",
]


def main() -> int:
    if not CSV_PATH.exists():
        print(f"Missing {CSV_PATH}")
        return 1

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") != "oeq":
                continue
            gen = row.get("generator", "")
            arm = row.get("arm", "")
            if gen not in GENERATORS or arm not in ARMS:
                continue
            try:
                ln = int(float(row.get("response_len") or 0))
            except (TypeError, ValueError):
                continue
            if ln <= 0:
                continue
            buckets[(gen, arm)].append(ln)

    by_generator: dict[str, dict] = {}
    for gen in GENERATORS:
        arm_stats = {}
        for arm in ARMS:
            vals = buckets.get((gen, arm), [])
            if not vals:
                continue
            mean_chars = sum(vals) / len(vals)
            arm_stats[arm] = {
                "n": len(vals),
                "mean_chars": round(mean_chars, 1),
                "median_chars": sorted(vals)[len(vals) // 2],
            }
        if arm_stats:
            cot = arm_stats.get("standard_cot", {}).get("mean_chars")
            not_ = arm_stats.get("narrative_cot", {}).get("mean_chars")
            ratio = round(not_ / cot, 2) if cot and not_ else None
            by_generator[gen] = {
                "arms": arm_stats,
                "not_vs_cot_char_ratio": ratio,
            }

    summary = {
        "dataset": "oeq",
        "n_per_cell": 150,
        "generators": by_generator,
        "note": "Character counts from response_len in elephant_singleagent_raw.csv.",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {SUMMARY_PATH}")
    for gen, info in by_generator.items():
        cot = info["arms"].get("standard_cot", {}).get("mean_chars", 0)
        not_ = info["arms"].get("narrative_cot", {}).get("mean_chars", 0)
        print(f"  {gen}: CoT={cot:.0f} chars, NoT={not_:.0f} chars (ratio={info['not_vs_cot_char_ratio']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
