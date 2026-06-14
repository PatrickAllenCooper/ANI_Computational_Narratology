"""
scripts/aggregate_judge_reliability.py -- Phase 15 judge reliability aggregation.

Computes Krippendorff's alpha and Cohen's kappa from judge_panel_raw.csv
and data/judge_gold.jsonl. Emits judge_reliability_summary.json.

Usage:
  python -m scripts.aggregate_judge_reliability
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.krippendorff import cohens_kappa, krippendorff_alpha_nominal

OUT = Path("./divergence_study_outputs")
PANEL_CSV = OUT / "judge_panel_raw.csv"
GOLD_PATH = Path("data/judge_gold.jsonl")
ALPHA_THRESHOLD = 0.67


def _read_panel() -> list[dict]:
    if not PANEL_CSV.exists():
        return []
    with PANEL_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_gold() -> list[dict]:
    if not GOLD_PATH.exists():
        return []
    rows = []
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _majority_vote(scores: list[int]) -> int:
    valid = [s for s in scores if s >= 0]
    if not valid:
        return -1
    return 1 if sum(valid) > len(valid) / 2 else 0


def aggregate_panel(rows: list[dict]) -> dict:
    """Per-metric Krippendorff alpha and pairwise kappa across judges."""
    by_item_metric: dict[tuple, dict[str, int]] = defaultdict(dict)
    for r in rows:
        try:
            score = int(float(r["score"]))
        except (ValueError, TypeError):
            score = -1
        key = (r["item_id"], r["arm"], r["generator"], r["metric"])
        by_item_metric[key][r["judge"]] = score

    judges = sorted({r["judge"] for r in rows})
    metrics = sorted({r["metric"] for r in rows})

    alpha_by_metric = {}
    kappa_pairs = []
    majority_flags = {}

    for metric in metrics:
        items = [k for k in by_item_metric if k[3] == metric]
        if not items:
            continue
        rater_vecs = []
        for judge in judges:
            rater_vecs.append([by_item_metric[k].get(judge) for k in items])
        alpha = krippendorff_alpha_nominal(rater_vecs)
        alpha_by_metric[metric] = {
            "alpha": alpha,
            "n_items": len(items),
            "low_reliability": alpha < ALPHA_THRESHOLD if alpha == alpha else True,
            "reporting_rule": (
                "majority_vote" if alpha < ALPHA_THRESHOLD else "any_judge_ok"
            ),
        }

        for i, j1 in enumerate(judges):
            for j2 in judges[i + 1:]:
                a = [by_item_metric[k].get(j1, -1) for k in items]
                b = [by_item_metric[k].get(j2, -1) for k in items]
                kappa_pairs.append({
                    "metric": metric,
                    "judge_a": j1,
                    "judge_b": j2,
                    "kappa": cohens_kappa(a, b),
                    "n": sum(1 for x, y in zip(a, b) if x >= 0 and y >= 0),
                })

        # Majority vs original primary judge scores
        disagreements = 0
        for k in items:
            panel_scores = [by_item_metric[k].get(j) for j in judges]
            maj = _majority_vote([s for s in panel_scores if s is not None and s >= 0])
            if maj >= 0:
                disagreements += 0  # placeholder for future original comparison
        majority_flags[metric] = alpha_by_metric[metric]["low_reliability"]

    return {
        "alpha_by_metric": alpha_by_metric,
        "kappa_pairs": kappa_pairs,
        "judges": judges,
        "majority_vote_metrics": [m for m, v in majority_flags.items() if v],
    }


def aggregate_gold(gold: list[dict], panel_rows: list[dict]) -> dict:
    """Judge vs human gold Cohen's kappa per metric."""
    if not gold:
        return {"note": "No judge_gold.jsonl; human kappa skipped."}

    gold_by_id = {g["item_id"]: g for g in gold}
    judges = sorted({r["judge"] for r in panel_rows}) if panel_rows else []

    # Also score gold items directly from gold file judge columns if present
    kappa_vs_human = []
    for metric in ("validation", "indirectness", "framing"):
        human_key = f"human_{metric}"
        for judge in judges:
            human_vals = []
            judge_vals = []
            for g in gold:
                if human_key not in g:
                    continue
                try:
                    hv = int(g[human_key])
                except (ValueError, TypeError):
                    continue
                # Panel score for same item if available
                panel_match = [
                    r for r in panel_rows
                    if r["item_id"] == g["item_id"] and r["metric"] == metric and r["judge"] == judge
                ]
                if panel_match:
                    try:
                        jv = int(float(panel_match[0]["score"]))
                    except (ValueError, TypeError):
                        continue
                elif f"judge_{metric}" in g:
                    jv = int(g[f"judge_{metric}"])
                else:
                    continue
                if hv >= 0 and jv >= 0:
                    human_vals.append(hv)
                    judge_vals.append(jv)
            if human_vals:
                kappa_vs_human.append({
                    "metric": metric,
                    "judge": judge,
                    "kappa_vs_human": cohens_kappa(judge_vals, human_vals),
                    "n": len(human_vals),
                })

    return {
        "gold_n": len(gold),
        "kappa_vs_human": kappa_vs_human,
        "gold_by_id_count": len(gold_by_id),
    }


def main() -> int:
    panel = _read_panel()
    gold = _read_gold()

    summary: dict = {"alpha_threshold": ALPHA_THRESHOLD}
    if panel:
        summary["panel"] = aggregate_panel(panel)
        print("=== Judge panel reliability ===")
        for m, info in summary["panel"]["alpha_by_metric"].items():
            flag = "LOW" if info["low_reliability"] else "OK"
            print(f"  {m:14s} alpha={info['alpha']:.3f} n={info['n_items']} [{flag}]")
        if summary["panel"]["majority_vote_metrics"]:
            print(f"  Majority vote required: {summary['panel']['majority_vote_metrics']}")
    else:
        print(f"Missing {PANEL_CSV} (run judge_panel first)")
        summary["panel"] = {}

    summary["gold"] = aggregate_gold(gold, panel)
    if gold:
        print("\n=== Judge vs human gold ===")
        for row in summary["gold"].get("kappa_vs_human", []):
            print(f"  {row['metric']:14s} {row['judge'][:24]:24s} "
                  f"kappa={row['kappa_vs_human']:.3f} n={row['n']}")

    (OUT / "judge_reliability_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
    )
    print(f"\nWrote {OUT / 'judge_reliability_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
