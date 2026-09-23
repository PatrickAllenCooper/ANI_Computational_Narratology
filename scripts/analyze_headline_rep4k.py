"""
scripts/analyze_headline_rep4k.py -- Addendum 17.10 readout: the headline NoT minus CoT
validation drop, replicated same-day on all seven generators with BOTH arms at a 4,096-token
cap (`standard_cot_rep4k`, `narrative_cot_rep4k`), under every judge of the 17.8 panel.

Per generator and judge: rates, drop with an item-clustered paired bootstrap, relative
reduction. Per arm: share of responses that still end at the cap (finish reason recorded in
the generation cache since 2026-09-22). Readings as registered:
  HEADLINE-HOLDS             production judge negative with CI excluding 0 AND at least three
                             of the four other judges likewise
  HEADLINE-JUDGE-DEPENDENT   production judge negative with CI excluding 0, fewer than three others
  HEADLINE-FAILS             production judge CI includes 0 or is positive

Run: python -m scripts.analyze_headline_rep4k [--judges ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.rescore_elephant_full_judge import (DEFAULT_GENERATORS, DEFAULT_JUDGES, PRODUCTION_JUDGE,
                                                  load_rows, paired_drop, table)
from scripts.rescore_elephant_untruncated import load_questions
from scripts.run_elephant import GENEROUS_CAP, _gen_cache_path

COT, NOT = "standard_cot_rep4k", "narrative_cot_rep4k"
OUT = Path("divergence_study_outputs/headline_rep4k_readout.json")


def cap_share(gen: str, arm: str, items) -> dict:
    n = hit = known = 0
    for i in items:
        p = _gen_cache_path("oeq", i, gen, arm)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        n += 1
        fr = d.get("finish_reason")
        if fr:
            known += 1
            hit += int(fr in ("length", "max_tokens", "MAX_TOKENS"))
    return {"n": n, "finish_reason_known": known, "at_cap": hit,
            "share_at_cap": (hit / known) if known else None}


def reading(per_judge: dict) -> str:
    p = per_judge.get(PRODUCTION_JUDGE)
    if not p or p.get("drop") is None or not (p["hi"] < 0):
        return "HEADLINE-FAILS"
    others = sum(1 for j, v in per_judge.items()
                 if j != PRODUCTION_JUDGE and v.get("drop") is not None and v["hi"] < 0)
    return "HEADLINE-HOLDS" if others >= 3 else "HEADLINE-JUDGE-DEPENDENT"


def run(judges) -> dict:
    rows = load_rows(DEFAULT_GENERATORS, (COT, NOT))
    q = load_questions("oeq", n=150)
    tabs = {j: table(rows, j, q)["cells"] for j in judges}
    res = {"judges": list(judges), "per_generator": {}}
    for g in DEFAULT_GENERATORS:
        items = sorted({r["item_id"] for r in rows if r["generator"] == g})
        pj = {}
        for j in judges:
            c = paired_drop(tabs[j].get((g, COT), {}), tabs[j].get((g, NOT), {}))
            if c.get("drop") is not None and c.get("rate_a"):
                c["relative"] = -c["drop"] / c["rate_a"]
            pj[j] = c
        res["per_generator"][g] = {"per_judge": pj, "reading": reading(pj),
                                   "cap": {a: cap_share(g, a, items) for a in (COT, NOT)},
                                   "n_rows": {a: sum(1 for r in rows if r["generator"] == g and r["arm"] == a)
                                              for a in (COT, NOT)}}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    ap.add_argument("--arm-suffix", default="",
                    help="17.10 amendment run 2: '_b' reads standard_cot_rep4k_b / narrative_cot_rep4k_b "
                         "and writes headline_rep4k_b_readout.json")
    a = ap.parse_args(argv)
    global COT, NOT, OUT
    if a.arm_suffix:
        COT, NOT = COT + a.arm_suffix, NOT + a.arm_suffix
        OUT = Path(f"divergence_study_outputs/headline_rep4k{a.arm_suffix}_readout.json")
    res = run([j for j in a.judges.split(",") if j])
    for g, v in res["per_generator"].items():
        cap = v["cap"]
        print(f"\n== {g}  rows CoT {v['n_rows'][COT]} NoT {v['n_rows'][NOT]}  at cap: CoT "
              f"{cap[COT]['at_cap']}/{cap[COT]['finish_reason_known']}  NoT {cap[NOT]['at_cap']}/"
              f"{cap[NOT]['finish_reason_known']}  -> {v['reading']}")
        for j, c in v["per_judge"].items():
            if c.get("drop") is None:
                print(f"   {j:26s} under-gated ({c.get('n_items')} items)")
                continue
            print(f"   {j:26s} {c['rate_a']:.3f} -> {c['rate_b']:.3f}  {100 * c['drop']:+.1f} "
                  f"[{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]  rel {100 * c.get('relative', 0):.0f}%  n {c['n_items']}")
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
