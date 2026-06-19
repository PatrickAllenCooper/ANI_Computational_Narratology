"""
scripts/bootstrap_reliability_cis.py -- Bootstrap CIs for judge reliability anchors.

Computes 95% percentile bootstrap CIs for Krippendorff alpha (panel) and
Cohen's kappa (judge vs human gold) from existing CSV/JSONL files. No API calls.

Output: divergence_study_outputs/reliability_bootstrap_cis.json

Usage:
  python -m scripts.bootstrap_reliability_cis [--n-bootstrap 2000]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from scripts.krippendorff import cohens_kappa, krippendorff_alpha_nominal

OUT = Path("divergence_study_outputs")
PANEL_CSV = OUT / "judge_panel_raw.csv"
GOLD_PATH = Path("data/judge_gold.jsonl")
GOLD_SCORES = OUT / "judge_gold_scores.jsonl"
OUT_PATH = OUT / "reliability_bootstrap_cis.json"

PANEL_JUDGES = [
    "gpt-5.4-nano",
    "claude-haiku-4-5",
    "grok-4-1-fast-reasoning",
]
METRICS = ("validation", "indirectness", "framing")


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = int(round(p / 100 * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, idx))]


def _bootstrap_ci_items(
    items: list,
    stat_fn,
    n_bootstrap: int = 2000,
    seed: int = 44,
) -> dict:
    rng = random.Random(seed)
    n = len(items)
    if n < 2:
        return {"point": float("nan"), "ci95": [float("nan"), float("nan")], "n_bootstrap": 0}
    point = stat_fn(items)
    boots = []
    for _ in range(n_bootstrap):
        draw = [items[rng.randrange(n)] for _ in range(n)]
        boots.append(stat_fn(draw))
    return {
        "point": point,
        "ci95": [_percentile(boots, 2.5), _percentile(boots, 97.5)],
        "n_bootstrap": n_bootstrap,
    }


def _bootstrap_ci_pairs(
    pairs: list[tuple[int, int]],
    n_bootstrap: int = 2000,
    seed: int = 44,
) -> dict:
    rng = random.Random(seed)
    n = len(pairs)
    if n < 2:
        return {"point": float("nan"), "ci95": [float("nan"), float("nan")], "n_bootstrap": 0}
    point = cohens_kappa([p[0] for p in pairs], [p[1] for p in pairs])
    boots = []
    for _ in range(n_bootstrap):
        draw = [pairs[rng.randrange(n)] for _ in range(n)]
        boots.append(cohens_kappa([p[0] for p in draw], [p[1] for p in draw]))
    return {
        "point": point,
        "ci95": [_percentile(boots, 2.5), _percentile(boots, 97.5)],
        "n_bootstrap": n_bootstrap,
    }


def _panel_items() -> dict[str, list[tuple[str, dict[str, int]]]]:
    """metric -> list of (item_key, {judge: score}).

    The unit of analysis is a (item_id, arm, generator) response, matching
    aggregate_judge_reliability.aggregate_panel; keying on item_id alone would
    collapse responses that share a source prompt across arms/generators.
    """
    if not PANEL_CSV.exists():
        return {}
    by_metric: dict[str, list[tuple[str, dict[str, int]]]] = defaultdict(list)
    bucket: dict[tuple, dict[str, int]] = defaultdict(dict)
    with PANEL_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["item_id"], row["arm"], row["generator"], row["metric"])
            try:
                score = int(float(row["score"]))
            except (ValueError, TypeError):
                score = -1
            bucket[key][row["judge"]] = score
    for (item_id, arm, generator, metric), scores in bucket.items():
        # Krippendorff's alpha handles missing data natively (>=2 raters/item);
        # keep every item rather than complete-case filtering, so the bootstrap
        # point estimate matches the canonical full-panel alpha in
        # judge_reliability_summary.json.
        by_metric[metric].append(((item_id, arm, generator), scores))
    return by_metric


def _alpha_for_items(items: list[tuple[str, dict[str, int]]]) -> float:
    if len(items) < 2:
        return float("nan")
    rater_vecs = []
    for judge in PANEL_JUDGES:
        rater_vecs.append([
            (it[1][judge] if it[1].get(judge, -1) >= 0 else None)
            for it in items
        ])
    return krippendorff_alpha_nominal(rater_vecs)


def _read_gold_pairs() -> dict[tuple[str, str], list[tuple[int, int]]]:
    """(metric, judge) -> list of (human, judge) pairs."""
    gold = {}
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                g = json.loads(line)
                gold[g["item_id"]] = g
    scores_by: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    if not GOLD_SCORES.exists():
        return scores_by
    with GOLD_SCORES.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            gs = json.loads(line)
            metric = gs.get("metric")
            judge = gs.get("judge")
            item_id = gs.get("item_id")
            g = gold.get(item_id)
            if not g:
                continue
            hk = f"human_{metric}"
            if hk not in g:
                continue
            try:
                hv = int(g[hk])
                jv = int(gs["score"])
            except (KeyError, ValueError, TypeError):
                continue
            if hv >= 0 and jv >= 0:
                scores_by[(metric, judge)].append((hv, jv))
    return scores_by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()

    summary: dict = {"n_bootstrap": args.n_bootstrap}

    panel_items = _panel_items()
    summary["panel_alpha"] = {}
    for metric in METRICS:
        items = panel_items.get(metric, [])
        if len(items) < 3:
            continue

        summary["panel_alpha"][metric] = _bootstrap_ci_items(
            items, _alpha_for_items, n_bootstrap=args.n_bootstrap,
        )
        summary["panel_alpha"][metric]["n_items"] = len(items)

    gold_pairs = _read_gold_pairs()
    summary["kappa_vs_human"] = []
    for metric in METRICS:
        for judge in PANEL_JUDGES:
            pairs = gold_pairs.get((metric, judge), [])
            if len(pairs) < 5:
                continue

            ci = _bootstrap_ci_pairs(pairs, n_bootstrap=args.n_bootstrap)
            ci["metric"] = metric
            ci["judge"] = judge
            ci["n"] = len(pairs)
            summary["kappa_vs_human"].append(ci)

    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {OUT_PATH}")
    for m, info in summary.get("panel_alpha", {}).items():
        lo, hi = info["ci95"]
        print(f"  alpha {m}: {info['point']:.3f} [{lo:.3f}, {hi:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
