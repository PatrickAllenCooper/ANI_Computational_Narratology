#!/usr/bin/env python3
"""Addendum 17.11 amendment 2 (registered 2026-09-23): every single-agent prompt placed against the
rate of the human answers to the same ELEPHANT OEQ items, scored by the same judge. This is ELEPHANT's
own definition of social sycophancy (a model's rate minus the human rate; below the human rate is
over-correction, arXiv 2505.13995v2 Section 4.3).

Rubrics and judges: validation under the five 17.8 judges; accepting framing under the production
judge and gpt-4o. Human-answer scores come from the rescore caches written by
judge_reference_agreement.py (same path, full text). Arms: the 17.7 cells (standard_cot_rep,
narrative_cot_rep, not_checklist, not_narrative_only, advisor_cot) and the 17.12 baselines, on the
four 17.7 generators.

Per judge and rubric, pooled over the four generators and per generator: mean over (item, generator)
of arm score minus the human answer's score on the same item, item-clustered percentile bootstrap,
8,000 draws, seed 20260923. Readings (pre-declared): ABOVE-HUMAN (interval above 0), BELOW-HUMAN
(interval below 0), AT-HUMAN (includes 0). Descriptive.

Run:  python -m scripts.analyze_human_anchor
      python -m scripts.analyze_human_anchor --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import scripts.rescore_elephant_full_judge as rfj
from scripts.rescore_elephant_untruncated import _key, load_questions

GENS = ("claude-haiku-4-5", "grok-4-1-fast-reasoning", "gpt-5.4-nano", "Llama-3.3-70B-Instruct")
ARMS = ("standard_cot_rep", "narrative_cot_rep", "not_checklist", "not_narrative_only", "advisor_cot",
        "cot_prepend_naive", "cot_prepend_context", "cot_perspective_shift", "cot_sceptical")
JUDGES = {"validation": ("claude-haiku-4-5", "gpt-4o", "gpt-5.4-nano", "Llama-3.3-70B-Instruct", "grok-4-1-fast-reasoning"),
          "framing": ("claude-haiku-4-5", "gpt-4o")}
DRAWS, SEED = 8000, 20260923
OUT = Path("divergence_study_outputs/human_anchor_readout.json")


def _cached(metric, q, a, judge):
    k = _key(metric, q, a, judge, None)
    if not k.exists():
        return None
    try:
        return int(json.loads(k.read_text()).get("score", -1))
    except Exception:
        return None


def human_scores(metric: str, judge: str) -> dict:
    from scripts.load_elephant import load_elephant
    out = {}
    for it in load_elephant("oeq", n=150):
        if it.human_response:
            s = _cached(metric, it.prompt, it.human_response, judge)
            if s in (0, 1):
                out[it.id] = s
    return out


def diff_boot(pairs_by_item: dict, draws=DRAWS, seed=SEED) -> dict:
    items = sorted(pairs_by_item)
    if len(items) < 20:
        return {"n_items": len(items), "diff": None}
    sums = np.array([sum(pairs_by_item[i]) for i in items], dtype=float)
    cnts = np.array([len(pairs_by_item[i]) for i in items], dtype=float)
    pt = sums.sum() / cnts.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(draws, len(items)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    return {"n_items": len(items), "n_pairs": int(cnts.sum()), "diff": float(pt),
            "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def reading(c: dict) -> str:
    if c.get("diff") is None:
        return "UNDER-GATED"
    if c["lo"] > 0:
        return "ABOVE-HUMAN"
    if c["hi"] < 0:
        return "BELOW-HUMAN"
    return "AT-HUMAN"


def run() -> dict:
    q = load_questions("oeq", n=150)
    rows = rfj.load_rows(GENS, ARMS)
    res = {"draws": DRAWS, "seed": SEED, "rubrics": {}}
    for metric, judges in JUDGES.items():
        rfj.METRIC = metric
        mr = {}
        for j in judges:
            hs = human_scores(metric, j)
            arm_scores = defaultdict(dict)
            for r in rows:
                s = rfj.cached_score(rfj.qfor(r, q), r["response"], j)
                if s in (0, 1):
                    arm_scores[(r["generator"], r["arm"])][r["item_id"]] = s
            jr = {"human_rate": float(np.mean(list(hs.values()))) if hs else None, "n_human": len(hs),
                  "pooled": {}, "per_generator": {}, "readings": {}}
            for a in ARMS:
                by = defaultdict(list)
                for g in GENS:
                    for i, s in arm_scores.get((g, a), {}).items():
                        if i in hs:
                            by[i].append(s - hs[i])
                jr["pooled"][a] = diff_boot(by)
                jr["readings"][a] = reading(jr["pooled"][a])
                jr["per_generator"][a] = {}
                for g in GENS:
                    byg = {i: [s - hs[i]] for i, s in arm_scores.get((g, a), {}).items() if i in hs}
                    jr["per_generator"][a][g] = diff_boot(byg)
            mr[j] = jr
        res["rubrics"][metric] = mr
    rfj.METRIC = "validation"
    return res


def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("[ok] " if cond else "[FAIL] ") + msg)
        ok = ok and cond
    up = {str(i): [1] * 3 for i in range(40)}
    down = {str(i): [-1] * 3 for i in range(40)}
    mix = {str(i): [1 if i % 2 else -1] for i in range(40)}
    check("all positive differences read ABOVE-HUMAN", reading(diff_boot(up, 500, 1)) == "ABOVE-HUMAN")
    check("all negative differences read BELOW-HUMAN", reading(diff_boot(down, 500, 1)) == "BELOW-HUMAN")
    check("balanced differences read AT-HUMAN", reading(diff_boot(mix, 500, 1)) == "AT-HUMAN")
    check("fewer than 20 items reads UNDER-GATED", reading(diff_boot({"a": [1]}, 500, 1)) == "UNDER-GATED")
    print("\nselftest: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=OUT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    res = run()
    for metric, mr in res["rubrics"].items():
        for j, jr in mr.items():
            print(f"\n== {metric} | judge {j} | human rate {jr['human_rate']:.3f} (n {jr['n_human']})  arm minus human, pooled over four generators")
            for arm in ARMS:
                c = jr["pooled"][arm]
                if c.get("diff") is None:
                    print(f"   {arm:24s} under-gated"); continue
                print(f"   {arm:24s} {100 * c['diff']:+.1f} [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]  {jr['readings'][arm]}")
    a.json.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
