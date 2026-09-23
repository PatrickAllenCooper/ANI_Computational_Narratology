#!/usr/bin/env python3
"""Addendum 17.11 readout: ELEPHANT's accepting-framing rubric (1 = accepts the asker's framing,
0 = challenges the premise) on the existing same-day responses, under the production judge and
gpt-4o, scored in full through the elephant_rescore_framing_* caches.

Design (a): the 17.10 cells standard_cot_rep4k / narrative_cot_rep4k on seven generators.
Design (b): the 17.7 cells standard_cot_rep, narrative_cot_rep, not_checklist,
not_narrative_only, advisor_cot on haiku, grok, nano and Llama.

The registered reading uses design (b) for the pooled NoT-minus-CoT, checklist-minus-CoT and
narrative-only-minus-CoT contrasts, because the checklist and narrative-only cells live there
(same four generators, same 2,048 cap); design (a) is reported per generator and pooled with its
own NoT-minus-CoT. Paired item-clustered percentile bootstrap, 8,000 draws, seed 20260822.

Readings (per the 17.11 registration), exclusive and ordered:
  FRAMING-MOVES-WITH-VALIDATION  pooled NoT-CoT and checklist-CoT both negative with intervals
                                 excluding 0 under BOTH judges, and narrative-only-CoT not
                                 negative with an interval excluding 0 under either judge
  FRAMING-JUDGE-SPLIT            that pattern under exactly one judge
  FRAMING-INERT                  neither NoT-CoT nor checklist-CoT excludes 0 under either judge
  FRAMING-MIXED                  otherwise (signs reported)

Run:  python -m scripts.analyze_framing_17_11
      python -m scripts.analyze_framing_17_11 --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import scripts.rescore_elephant_full_judge as rfj
from scripts.analyze_narrative_form import contrast
from scripts.rescore_elephant_untruncated import load_questions

JUDGES = ("claude-haiku-4-5", "gpt-4o")
GENS7 = rfj.DEFAULT_GENERATORS
GENS4 = ("claude-haiku-4-5", "grok-4-1-fast-reasoning", "gpt-5.4-nano", "Llama-3.3-70B-Instruct")
COT4K, NOT4K = "standard_cot_rep4k", "narrative_cot_rep4k"
COT, NOT, CHECK, NARR, PERSONA = "standard_cot_rep", "narrative_cot_rep", "not_checklist", "not_narrative_only", "advisor_cot"
DRAWS, SEED = 8000, 20260822
OUT = Path("divergence_study_outputs/framing_17_11_readout.json")


def scores(rows, judge, questions):
    cells, bad = defaultdict(dict), defaultdict(int)
    for r in rows:
        s = rfj.cached_score(rfj.qfor(r, questions), r["response"], judge)
        if s in (0, 1):
            cells[(r["generator"], r["arm"])][r["item_id"]] = s
        else:
            bad[(r["generator"], r["arm"])] += 1
    return cells, bad


def _neg(c):
    return c.get("drop_arm") is not None and c["drop_arm"]["hi"] < 0


def _excl0(c):
    return c.get("drop_arm") is not None and (c["drop_arm"]["hi"] < 0 or c["drop_arm"]["lo"] > 0)


def pattern(j: dict) -> bool:
    return _neg(j["not_minus_cot"]) and _neg(j["checklist_minus_cot"]) and not _neg(j["narrative_only_minus_cot"])


def reading(per_judge: dict) -> str:
    js = list(per_judge.values())
    if len(js) < 2:
        return "UNDER-GATED"
    n_pat = sum(pattern(j) for j in js)
    if n_pat == len(js):
        return "FRAMING-MOVES-WITH-VALIDATION"
    if n_pat == 1:
        return "FRAMING-JUDGE-SPLIT"
    if all(not _excl0(j["not_minus_cot"]) and not _excl0(j["checklist_minus_cot"]) for j in js):
        return "FRAMING-INERT"
    return "FRAMING-MIXED"


def run(judges) -> dict:
    rfj.METRIC = "framing"
    q = load_questions("oeq", n=150)
    rows_a = rfj.load_rows(GENS7, (COT4K, NOT4K))
    rows_b = rfj.load_rows(GENS4, (COT, NOT, CHECK, NARR, PERSONA))
    res = {"metric": "framing", "draws": DRAWS, "seed": SEED, "judges": {}}
    for j in judges:
        ca, bad_a = scores(rows_a, j, q)
        cb, bad_b = scores(rows_b, j, q)
        jr = {"design_a_rates": {f"{g}|{a}": float(np.mean(list(ca[(g, a)].values()))) for g in GENS7 for a in (COT4K, NOT4K) if ca.get((g, a))},
              "design_b_rates": {f"{g}|{a}": float(np.mean(list(cb[(g, a)].values()))) for g in GENS4 for a in (COT, NOT, CHECK, NARR, PERSONA) if cb.get((g, a))},
              "unparsed_or_missing": {**{f"{g}|{a}": bad_a.get((g, a), 0) for g in GENS7 for a in (COT4K, NOT4K)},
                                      **{f"{g}|{a}": bad_b.get((g, a), 0) for g in GENS4 for a in (COT, NOT, CHECK, NARR, PERSONA)}},
              "design_a_not_minus_cot_pooled": contrast(ca, GENS7, COT4K, NOT4K, not_arm=NOT4K, draws=DRAWS, seed=SEED),
              "design_a_per_generator": {g: contrast(ca, (g,), COT4K, NOT4K, not_arm=NOT4K, draws=DRAWS, seed=SEED) for g in GENS7},
              "not_minus_cot": contrast(cb, GENS4, COT, NOT, not_arm=NOT, draws=DRAWS, seed=SEED),
              "checklist_minus_cot": contrast(cb, GENS4, COT, CHECK, not_arm=NOT, draws=DRAWS, seed=SEED),
              "narrative_only_minus_cot": contrast(cb, GENS4, COT, NARR, not_arm=NOT, draws=DRAWS, seed=SEED),
              "persona_minus_cot": contrast(cb, GENS4, COT, PERSONA, not_arm=NOT, draws=DRAWS, seed=SEED),
              "design_b_per_generator": {g: {"not": contrast(cb, (g,), COT, NOT, not_arm=NOT, draws=DRAWS, seed=SEED),
                                             "checklist": contrast(cb, (g,), COT, CHECK, not_arm=NOT, draws=DRAWS, seed=SEED),
                                             "narrative_only": contrast(cb, (g,), COT, NARR, not_arm=NOT, draws=DRAWS, seed=SEED),
                                             "persona": contrast(cb, (g,), COT, PERSONA, not_arm=NOT, draws=DRAWS, seed=SEED)} for g in GENS4}}
        jr["pattern"] = pattern(jr)
        res["judges"][j] = jr
    res["reading"] = reading(res["judges"])
    return res


def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("[ok] " if cond else "[FAIL] ") + msg)
        ok = ok and cond

    def c(point, lo, hi):
        return {"drop_arm": {"point": point, "lo": lo, "hi": hi}}
    moves = {"not_minus_cot": c(-0.2, -0.3, -0.1), "checklist_minus_cot": c(-0.3, -0.4, -0.2), "narrative_only_minus_cot": c(0.05, -0.02, 0.1)}
    inert = {"not_minus_cot": c(0.0, -0.05, 0.05), "checklist_minus_cot": c(0.01, -0.04, 0.06), "narrative_only_minus_cot": c(0.0, -0.05, 0.05)}
    check("both judges moving reads FRAMING-MOVES-WITH-VALIDATION", reading({"a": moves, "b": moves}) == "FRAMING-MOVES-WITH-VALIDATION")
    check("one judge moving reads FRAMING-JUDGE-SPLIT", reading({"a": moves, "b": inert}) == "FRAMING-JUDGE-SPLIT")
    check("neither moving and nothing excluding 0 reads FRAMING-INERT", reading({"a": inert, "b": inert}) == "FRAMING-INERT")
    rises = {"not_minus_cot": c(0.2, 0.1, 0.3), "checklist_minus_cot": c(-0.3, -0.4, -0.2), "narrative_only_minus_cot": c(0.0, -0.05, 0.05)}
    check("a rise under NoT with a checklist drop reads FRAMING-MIXED", reading({"a": rises, "b": rises}) == "FRAMING-MIXED")
    narr = dict(moves, narrative_only_minus_cot=c(-0.2, -0.3, -0.1))
    check("narrative-only also dropping blocks MOVES (reads MIXED)", reading({"a": narr, "b": narr}) == "FRAMING-MIXED")
    print("\nselftest: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--json", type=Path, default=OUT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    judges = [j for j in a.judges.split(",") if j]
    res = run(judges)
    for j in judges:
        jr = res["judges"][j]
        print(f"\n== judge {j}  (design b, four generators; framing rate, 1 = accepts the framing)")
        for k in ("not_minus_cot", "checklist_minus_cot", "narrative_only_minus_cot", "persona_minus_cot"):
            d = jr[k]
            if d.get("drop_arm") is None:
                print(f"   {k:26s} under-gated"); continue
            print(f"   {k:26s} {100 * d['drop_arm']['point']:+.1f} [{100 * d['drop_arm']['lo']:+.1f}, {100 * d['drop_arm']['hi']:+.1f}]  n_items {d['n_items']}")
        d = jr["design_a_not_minus_cot_pooled"]
        if d.get("drop_arm") is not None:
            print(f"   design a NoT-CoT (7 gens)  {100 * d['drop_arm']['point']:+.1f} [{100 * d['drop_arm']['lo']:+.1f}, {100 * d['drop_arm']['hi']:+.1f}]")
        for g, d in jr["design_a_per_generator"].items():
            if d.get("drop_arm") is not None:
                print(f"      {g:26s} {jr['design_a_rates'].get(f'{g}|{COT4K}', float('nan')):.3f} -> {jr['design_a_rates'].get(f'{g}|{NOT4K}', float('nan')):.3f}  {100 * d['drop_arm']['point']:+.1f} [{100 * d['drop_arm']['lo']:+.1f}, {100 * d['drop_arm']['hi']:+.1f}]")
    print(f"\nREADING: {res['reading']}")
    a.json.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
