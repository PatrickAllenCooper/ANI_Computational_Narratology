"""
analyze_counterparty_causal.py -- Addendum 17.5: does the counterparty-representation
manipulation, isolated from the rest of the NoT scaffold, reproduce Z2's stratified
pattern, and does a style-matched control that names no counterparty reproduce it too?

Two new arms (installed in scripts.run_elephant.CAUSAL_ARMS):
  not_counterparty_only  a single sentence asking the model to consider everyone
                          affected, no other NoT structure (the manipulation)
  not_style_control       a single sentence about checking premises, same length and
                          register, unrelated content (the control)

Reuses the Z2 counterparty coding cache (divergence_study_outputs/elephant_cp_*.json,
one YES/NO judge call per item, already paid for) rather than re-coding.

  python -m scripts.analyze_counterparty_causal
  python -m scripts.analyze_counterparty_causal --selftest
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

from scripts.analyze_length_matched_elephant import RAW, _cluster_bootstrap, _rows, _scored, apply_corrected_scores

OUT = Path(__file__).resolve().parents[1] / "divergence_study_outputs"
COL = "sycophantic_validation"
GENS = ("claude-haiku-4-5", "grok-4-1-fast-reasoning", "gpt-5.4-nano")
ARMS = ("not_counterparty_only", "not_style_control")


def load_coding() -> dict[str, str]:
    coding = {}
    for p in OUT.glob("elephant_cp_*.json"):
        rec = json.loads(p.read_text())
        coding[rec["item_id"]] = rec["answer"]
    return coding


def cell(rows, gen, arm):
    return {r["item_id"]: int(r[COL]) for r in rows if r["generator"] == gen and r["arm"] == arm and _scored(r, COL)}


def rate(d: dict[str, int]) -> float | None:
    return 100.0 * st.mean(d.values()) if d else None


def strat_diff_stat(present_ids, absent_ids, arm_by_item, cot_by_item):
    """Per-resample: (arm - CoT) mean on present items minus (arm - CoT) mean on absent."""
    def stat(recs):
        ids = {r["item_id"] for r in recs}
        p = [arm_by_item[i] - cot_by_item[i] for i in ids & present_ids if i in arm_by_item and i in cot_by_item]
        a = [arm_by_item[i] - cot_by_item[i] for i in ids & absent_ids if i in arm_by_item and i in cot_by_item]
        if len(p) < 5 or len(a) < 5:
            return None
        return 100.0 * (st.mean(p) - st.mean(a))
    return stat


def build(rows, coding, *, b=8000, seed=20260922):
    present = {i for i, a in coding.items() if a == "YES"}
    absent = {i for i, a in coding.items() if a == "NO"}
    res = {"per_generator": {}, "pooled": {}}
    pooled_by_arm = {arm: [] for arm in ARMS}
    for gen in GENS:
        cot = cell(rows, gen, "standard_cot")
        intact = cell(rows, gen, "narrative_cot")
        g = {"cot_rate": rate(cot), "intact_rate": rate(intact), "n_cot": len(cot), "n_intact": len(intact)}
        for arm in ARMS:
            a = cell(rows, gen, arm)
            drop_recs = [{"item_id": i, "d": a[i] - cot[i]} for i in set(a) & set(cot)]
            drop_ci = _cluster_bootstrap(drop_recs, lambda rs: 100.0 * st.mean([r["d"] for r in rs]) if rs else None,
                                         b=b, seed=seed)
            recs_all = [{"item_id": i} for i in set(a) & set(cot)]
            sdiff = _cluster_bootstrap(recs_all, strat_diff_stat(present, absent, a, cot), b=b, seed=seed)
            n_present = len(set(a) & set(cot) & present)
            n_absent = len(set(a) & set(cot) & absent)
            g[arm] = {"rate": rate(a), "n": len(a), "drop": drop_ci,
                      "stratified_diff": sdiff, "n_present": n_present, "n_absent": n_absent}
            pooled_by_arm[arm].append({"gen": gen, "a": a, "cot": cot})
        res["per_generator"][gen] = g

    for arm in ARMS:
        items = [{"item_id": f"{c['gen']}:{i}"} for c in pooled_by_arm[arm] for i in set(c["a"]) & set(c["cot"])]

        def drop_stat(recs, arm=arm):
            per_gen = []
            groups: dict[str, list[str]] = {}
            for r in recs:
                gen, item = r["item_id"].split(":", 1)
                groups.setdefault(gen, []).append(item)
            for c in pooled_by_arm[arm]:
                its = groups.get(c["gen"], [])
                d = [c["a"][i] - c["cot"][i] for i in its if i in c["a"] and i in c["cot"]]
                if len(d) >= 5:
                    per_gen.append(100.0 * st.mean(d))
            return st.mean(per_gen) if per_gen else None

        def sdiff_stat(recs, arm=arm):
            per_gen = []
            groups: dict[str, list[str]] = {}
            for r in recs:
                gen, item = r["item_id"].split(":", 1)
                groups.setdefault(gen, []).append(item)
            for c in pooled_by_arm[arm]:
                its = set(groups.get(c["gen"], []))
                p = [c["a"][i] - c["cot"][i] for i in its & present if i in c["a"] and i in c["cot"]]
                ab = [c["a"][i] - c["cot"][i] for i in its & absent if i in c["a"] and i in c["cot"]]
                if len(p) >= 5 and len(ab) >= 5:
                    per_gen.append(100.0 * (st.mean(p) - st.mean(ab)))
            return st.mean(per_gen) if per_gen else None

        res["pooled"][arm] = {
            "drop": _cluster_bootstrap(items, drop_stat, b=b, seed=seed),
            "stratified_diff": _cluster_bootstrap(items, sdiff_stat, b=b, seed=seed),
        }
    return res


def reading(pooled: dict) -> str:
    """
    NOTE (found 2026-09-22, after the run): the three branches pre-declared in
    Addendum 17.5 (COUNTERPARTY-CAUSAL, GENERIC-INSTRUCTION-EFFECT, STRUCTURE-NEEDED)
    all presumed the manipulation arm's drop would be zero or negative. It came back
    positive with a CI excluding zero, a REVERSAL the registration did not anticipate
    and has no branch for. Adding REVERSED here rather than forcing the result into
    STRUCTURE-NEEDED (which the un-patched logic would have silently done, since it only
    tested "is the CI's upper bound below zero" and a positive-excluding-zero CI also
    fails that test) is itself part of the honest readout: the pre-declared branch set
    was incomplete, and the gap is reported, not papered over.
    """
    manip = pooled["not_counterparty_only"]
    ctrl = pooled["not_style_control"]
    manip_neg_sig = manip["drop"]["hi"] is not None and manip["drop"]["hi"] < 0
    manip_pos_sig = manip["drop"]["lo"] is not None and manip["drop"]["lo"] > 0
    manip_strat_sig = manip["stratified_diff"]["hi"] is not None and manip["stratified_diff"]["hi"] < 0
    ctrl_strat_sig = ctrl["stratified_diff"]["hi"] is not None and ctrl["stratified_diff"]["hi"] < 0
    if manip_pos_sig:
        return "REVERSED (not pre-declared; the isolated manipulation RAISES validation, CI excludes 0 above)"
    if manip_neg_sig and manip_strat_sig and not ctrl_strat_sig:
        return "COUNTERPARTY-CAUSAL"
    if manip_strat_sig and ctrl_strat_sig:
        return "GENERIC-INSTRUCTION-EFFECT"
    if not manip_neg_sig:
        return "STRUCTURE-NEEDED"
    return "MIXED"


def _selftest() -> int:
    rows = [
        {"generator": "g", "arm": "standard_cot", "item_id": "p1", COL: "1"},
        {"generator": "g", "arm": "standard_cot", "item_id": "p2", COL: "1"},
        {"generator": "g", "arm": "standard_cot", "item_id": "p3", COL: "1"},
        {"generator": "g", "arm": "standard_cot", "item_id": "p4", COL: "1"},
        {"generator": "g", "arm": "standard_cot", "item_id": "p5", COL: "1"},
        {"generator": "g", "arm": "standard_cot", "item_id": "a1", COL: "1"},
        {"generator": "g", "arm": "standard_cot", "item_id": "a2", COL: "1"},
        {"generator": "g", "arm": "standard_cot", "item_id": "a3", COL: "1"},
        {"generator": "g", "arm": "standard_cot", "item_id": "a4", COL: "1"},
        {"generator": "g", "arm": "standard_cot", "item_id": "a5", COL: "1"},
    ]
    for i in ("p1", "p2", "p3", "p4", "p5"):
        rows.append({"generator": "g", "arm": "not_counterparty_only", "item_id": i, COL: "0"})
        rows.append({"generator": "g", "arm": "not_style_control", "item_id": i, COL: "1"})
    for i in ("a1", "a2", "a3", "a4", "a5"):
        rows.append({"generator": "g", "arm": "not_counterparty_only", "item_id": i, COL: "1"})
        rows.append({"generator": "g", "arm": "not_style_control", "item_id": i, COL: "1"})
    coding = {f"p{k}": "YES" for k in range(1, 6)} | {f"a{k}": "NO" for k in range(1, 6)}
    global GENS
    old = GENS
    GENS = ("g",)
    try:
        res = build(rows, coding, b=200)
    finally:
        GENS = old
    g = res["per_generator"]["g"]["not_counterparty_only"]
    ok = abs(g["drop"]["point"] + 50.0) < 1e-9 and abs(g["stratified_diff"]["point"] + 100.0) < 1e-9
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--bootstrap", type=int, default=8000)
    ap.add_argument("--json", type=Path, default=OUT / "counterparty_causal_analysis.json")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    rows = _rows(a.raw, "oeq")
    n_rep, n_miss = apply_corrected_scores(rows, "oeq", "validation")
    print(f"[corrected] replaced {n_rep} truncated scores; {n_miss} not cached (left as published)")
    coding = load_coding()
    print(f"counterparty coding: {sum(1 for v in coding.values() if v == 'YES')} present, "
          f"{sum(1 for v in coding.values() if v == 'NO')} absent, of {len(coding)}")
    res = build(rows, coding, b=a.bootstrap)
    for gen, g in res["per_generator"].items():
        print(f"{gen}: CoT {g['cot_rate']:.1f} intact NoT {g['intact_rate']:.1f}")
        for arm in ARMS:
            v = g[arm]
            d, sd = v["drop"], v["stratified_diff"]
            print(f"    {arm:22s} rate {v['rate']:5.1f} (n={v['n']:3d})  drop {d['point']:+.1f} [{d['lo']:+.1f},{d['hi']:+.1f}]"
                  f"  strat.diff {sd['point']:+.1f} [{sd['lo']:+.1f},{sd['hi']:+.1f}] (present n={v['n_present']}, absent n={v['n_absent']})")
    print("POOLED:")
    for arm in ARMS:
        v = res["pooled"][arm]
        d, sd = v["drop"], v["stratified_diff"]
        print(f"    {arm:22s} drop {d['point']:+.1f} [{d['lo']:+.1f},{d['hi']:+.1f}]  "
              f"strat.diff {sd['point']:+.1f} [{sd['lo']:+.1f},{sd['hi']:+.1f}]")
    r = reading(res["pooled"])
    res["reading"] = r
    print("READING:", r)
    a.json.write_text(json.dumps(res, indent=1))
    print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
