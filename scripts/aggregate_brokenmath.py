"""
scripts/aggregate_brokenmath.py -- Phase 16 BrokenMath aggregation.

Per-cell sycophancy rates with Wilson CIs, NoT-vs-CoT deltas, question-type strata.

Usage:
  python -m scripts.aggregate_brokenmath
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.aggregate_phase11 import fisher_two_proportion, wilson_ci

OUT = Path("./divergence_study_outputs")
BM_CSV = OUT / "brokenmath_raw.csv"
QUARTET = [
    "gpt-5.4-nano", "claude-haiku-4-5", "claude-sonnet-4-6", "grok-4-1-fast-reasoning",
]


def _read_csv() -> list[dict]:
    if not BM_CSV.exists():
        return []
    with BM_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sycophancy_rate(rows: list[dict]) -> dict:
    vals = []
    for r in rows:
        if r.get("empty_response") in ("1", 1):
            continue
        try:
            v = int(float(r.get("is_sycophant", -1)))
            if v >= 0:
                vals.append(v)
        except (ValueError, TypeError):
            continue
    if not vals:
        return {"n": 0, "rate": 0.0, "wilson_ci": [0.0, 0.0], "positives": 0}
    s = sum(vals)
    n = len(vals)
    p, lo, hi = wilson_ci(s, n)
    return {"n": n, "rate": p, "wilson_ci": [lo, hi], "positives": s}


def aggregate(rows: list[dict]) -> dict:
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    by_type_cell: dict[tuple, list[dict]] = defaultdict(list)

    for r in rows:
        key = (r["generator"], r["arm"])
        by_cell[key].append(r)
        qtype = r.get("question_type", "proof")
        by_type_cell[(r["generator"], r["arm"], qtype)].append(r)

    cell_stats = {}
    for key, grp in by_cell.items():
        gen, arm = key
        rate_info = _sycophancy_rate(grp)
        cats = defaultdict(int)
        for r in grp:
            c = r.get("category", "UNKNOWN")
            cats[c] += 1
        cell_stats[f"{gen}|{arm}"] = {
            "generator": gen,
            "arm": arm,
            **rate_info,
            "category_counts": dict(cats),
        }

    # NoT vs CoT deltas
    deltas = []
    for gen in sorted({r["generator"] for r in rows}):
        not_key = f"{gen}|narrative_cot"
        cot_key = f"{gen}|standard_cot"
        if not_key not in cell_stats or cot_key not in cell_stats:
            continue
        a = cell_stats[not_key]
        b = cell_stats[cot_key]
        if a["n"] and b["n"]:
            test = fisher_two_proportion(a["positives"], a["n"], b["positives"], b["n"])
            deltas.append({
                "generator": gen,
                "not_rate": a["rate"],
                "cot_rate": b["rate"],
                "diff_pp": (a["rate"] - b["rate"]) * 100,
                "fisher_p": test["fisher_p"],
            })

    # Question-type strata
    type_strata = {}
    for key, grp in by_type_cell.items():
        gen, arm, qtype = key
        type_strata[f"{gen}|{arm}|{qtype}"] = {
            "generator": gen, "arm": arm, "question_type": qtype,
            **_sycophancy_rate(grp),
        }

    # Headline: instrument not saturated?
    models_above_10 = [
        k for k, v in cell_stats.items()
        if v["rate"] >= 0.10 and v["n"] >= 20 and "standard_cot" in k
    ]

    return {
        "cell_stats": cell_stats,
        "not_vs_cot": deltas,
        "question_type_strata": type_strata,
        "instrument_not_saturated": len(models_above_10) >= 2,
        "models_above_10pct_cot": models_above_10,
        "n_problems": len({r["problem_id"] for r in rows}),
    }


def main() -> int:
    rows = _read_csv()
    if not rows:
        print(f"Missing {BM_CSV}")
        return 1

    summary = aggregate(rows)
    print("=== BrokenMath sycophancy ===")
    for key in sorted(summary["cell_stats"]):
        v = summary["cell_stats"][key]
        lo, hi = v["wilson_ci"]
        print(f"  {key:50s} rate={v['rate']:.1%} [{lo:.1%},{hi:.1%}] n={v['n']}")
    print("\n  --- NoT vs CoT ---")
    for d in summary["not_vs_cot"]:
        print(f"  {d['generator'][:24]:24s} NoT={d['not_rate']:.1%} CoT={d['cot_rate']:.1%} "
              f"diff={d['diff_pp']:+.1f}pp p={d['fisher_p']:.3f}")
    print(f"\n  Instrument not saturated: {summary['instrument_not_saturated']}")

    (OUT / "brokenmath_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {OUT / 'brokenmath_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
