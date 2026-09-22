"""
verify_pillar3_headline.py -- an independent recomputation of the pillar-3
headline numbers straight from the CSVs, sharing no code with the registered
analyzers (analyze_router_decomposition, analyze_topology_2x2, analyze_flooding).

Recomputes, for a community (rows + votes CSV) and a judge comparator CSV:

  collective accuracy          S2 `verdict` against `gold_verdict` (published ESH coding)
  counter fire rate            n_objectors >= 2, cross-checked against the votes' objected_r3
  one-loser lift               P(wrong | fired) vs P(wrong | not) within YTA/NTA verdicts, ratio
  routed accuracy              S2 unless fired, then the judge's standard majority-of-3 (ties -> S2)
  routed minus collective      paired, item-clustered percentile bootstrap (own seed)
  per-sample, per-arm, per-gold breakdowns of routed minus collective (descriptive)

Run:
  python -m scripts.verify_pillar3_headline                # grok panel vs its registered numbers
  python -m scripts.verify_pillar3_headline --community haiku_249
  python -m scripts.verify_pillar3_headline --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "divergence_study_outputs"

COMMUNITIES = {
    "grok": {"rows": "cg_deliberation_rows.csv", "votes": "cg_deliberation_votes.csv",
             "judge": "cg_sonnet_actuator_rows.csv", "judge_model": "claude-sonnet-4-6",
             "registered": {"source": "router_decomposition.json (16.15.1 / 16.22, grok panel)",
                            "n": 1677, "collective": 0.8426, "fire": 0.1956, "n_fired": 328,
                            "ol_n": 1287, "ol_fired": 93, "ol_wrong_fired": 0.387, "ol_wrong_unfired": 0.039,
                            "ol_ratio": 9.83, "routed": 0.9129, "delta": 0.0704, "lo": 0.0477, "hi": 0.0941}},
    "haiku_249": {"rows": "cg_deliberation_haiku_249_rows.csv", "votes": "cg_deliberation_haiku_249_votes.csv",
                  "judge": "cg_sonnet_actuator_rows.csv", "judge_model": "claude-sonnet-4-6",
                  "registered": {"source": "router_decomposition_haiku.json (16.23)",
                                 "n": 487, "collective": 0.8542, "fire": 0.193, "n_fired": 94,
                                 "ol_n": 424, "ol_fired": 65, "ol_wrong_fired": 0.108, "ol_wrong_unfired": 0.072,
                                 "ol_ratio": 1.49, "routed": 0.8912, "delta": 0.0370, "lo": 0.0145, "hi": 0.0593}},
    "noedge": {"rows": "cg_deliberation_noedge_rows.csv", "votes": "cg_deliberation_noedge_votes.csv",
               "judge": "cg_sonnet_actuator_rows.csv", "judge_model": "claude-sonnet-4-6",
               "registered": {"source": "router_decomposition_noedge_k4.json (16.25, edges-off)",
                              "n": 1675, "collective": 0.8322, "fire": 0.1467, "n_fired": 246,
                              "ol_n": 1263, "ol_fired": 78, "ol_wrong_fired": 0.282, "ol_wrong_unfired": 0.056,
                              "ol_ratio": 5.06, "routed": 0.8800, "delta": 0.0480, "lo": 0.0320, "hi": 0.0650}},
    "llama": {"rows": "cg_deliberation_llama3370binstruct_rows.csv", "votes": "cg_deliberation_llama3370binstruct_votes.csv",
              "judge": "cg_sonnet_actuator_rows.csv", "judge_model": "claude-sonnet-4-6",
              "registered": {"source": "router_decomposition_llama.json (16.24, Llama-3.3-70B-Instruct)",
                             "n": 897, "collective": 0.6198, "fire": 0.0792, "n_fired": 71,
                             "ol_n": 804, "ol_fired": 64, "ol_wrong_fired": 0.469, "ol_wrong_unfired": 0.307,
                             "ol_ratio": 1.53, "routed": 0.6611, "delta": 0.0412, "lo": 0.0279, "hi": 0.0559}},
}
SEATS = ("writer_advocate", "counterparty", "neutral_adjudicator")
# 2026-09-21: the edges-off cell installs its seats under a "_noedge" suffix
# (run_crowdgold_topology.install), so the diagnostic n_objectors-vs-votes
# cross-check below needs the matching seat names or it silently reports a
# false mismatch (the fired/lift/delta figures never depended on this; they
# read n_objectors straight from the rows CSV throughout).
SEATS_BY_COMMUNITY = {"noedge": tuple(s + "_noedge" for s in SEATS)}


def code(v: str):
    """Published ESH coding: YTA and ESH are at fault (1), NTA and NAH are not (0), else None."""
    return {"YTA": 1, "ESH": 1, "NTA": 0, "NAH": 0}.get(v)


def load(name: str):
    spec = COMMUNITIES[name]
    seats = SEATS_BY_COMMUNITY.get(name, SEATS)
    rows = list(csv.DictReader(open(OUT / spec["rows"])))
    votes = list(csv.DictReader(open(OUT / spec["votes"])))
    judge = list(csv.DictReader(open(OUT / spec["judge"])))
    obj = defaultdict(int)
    for v in votes:
        if v["role_id"] in seats and v.get("objected_r3") == "1":
            obj[(v["arm"], v["item_id"], v["sample_idx"])] += 1
    jm = defaultdict(list)
    for j in judge:
        if j["model"] == spec["judge_model"] and j["scaffold"] == "standard":
            jm[(j["arm"], j["item_id"])].append(code(j["verdict"]))
    deb = []
    for r in rows:
        s1, s2 = code(r.get("synthesis_verdict", "")), code(r["verdict"])
        if s1 is None or s2 is None:
            continue
        gold = 1 if r["gold_verdict"] == "YTA" else 0
        key = (r["arm"], r["item_id"], r["sample_idx"])
        n_obj = int(r.get("n_objectors") or 0)
        codes = [c for c in jm.get((r["arm"], r["item_id"]), []) if c is not None]
        n1 = sum(codes); n0 = len(codes) - n1
        jv = None if (not codes or n1 == n0) else int(n1 > n0)
        deb.append({"arm": r["arm"], "item": r["item_id"], "sample": int(r["sample_idx"]), "gold": gold,
                    "s2": s2, "ok": int(s2 == gold), "n_obj": n_obj, "n_obj_votes": obj.get(key, 0),
                    "fired": int(n_obj >= 2), "one_loser": r["verdict"] in ("YTA", "NTA"),
                    "judge_ok": None if jv is None else int(jv == gold)})
    return deb


def routed_ok(d):
    if d["fired"] and d["judge_ok"] is not None:
        return d["judge_ok"]
    return d["ok"]


def boot_delta(deb, draws=4000, seed=101):
    items = sorted({d["item"] for d in deb})
    per = {i: [] for i in items}
    for d in deb:
        per[d["item"]].append(routed_ok(d) - d["ok"])
    arr = [np.array(per[i], dtype=float) for i in items]
    sums = np.array([a.sum() for a in arr]); cnts = np.array([len(a) for a in arr])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(draws, len(items)))
    stats = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(sums.sum() / cnts.sum()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def compute(deb):
    n = len(deb)
    coll = sum(d["ok"] for d in deb) / n
    n_fired = sum(d["fired"] for d in deb)
    mismatch = sum(1 for d in deb if d["n_obj"] != d["n_obj_votes"])
    ol = [d for d in deb if d["one_loser"]]
    olf = [d for d in ol if d["fired"]]; olu = [d for d in ol if not d["fired"]]
    wf = 1 - sum(d["ok"] for d in olf) / len(olf); wu = 1 - sum(d["ok"] for d in olu) / len(olu)
    routed = sum(routed_ok(d) for d in deb) / n
    point, lo, hi = boot_delta(deb)
    br = {}
    for key, fn in (("sample", lambda d: d["sample"]), ("arm", lambda d: d["arm"]), ("gold", lambda d: d["gold"])):
        groups = defaultdict(list)
        for d in deb:
            groups[fn(d)].append(d)
        br[key] = {str(k): {"n": len(g), "collective": round(sum(x["ok"] for x in g) / len(g), 4),
                            "routed": round(sum(routed_ok(x) for x in g) / len(g), 4),
                            "delta": round((sum(routed_ok(x) for x in g) - sum(x["ok"] for x in g)) / len(g), 4),
                            "n_fired": sum(x["fired"] for x in g)}
                   for k, g in sorted(groups.items())}
    return {"n": n, "collective": coll, "fire": n_fired / n, "n_fired": n_fired,
            "n_objectors_vote_mismatch": mismatch, "ol_n": len(ol), "ol_fired": len(olf),
            "ol_wrong_fired": wf, "ol_wrong_unfired": wu, "ol_ratio": wf / wu if wu else None,
            "routed": routed, "delta": point, "lo": lo, "hi": hi, "breakdown": br}


TOL = {"n": 0, "n_fired": 0, "ol_n": 0, "ol_fired": 0, "collective": 0.0015, "fire": 0.0015, "routed": 0.0015,
       "ol_wrong_fired": 0.0015, "ol_wrong_unfired": 0.0015, "ol_ratio": 0.05, "delta": 0.0015, "lo": 0.006, "hi": 0.006}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--community", default="grok", choices=sorted(COMMUNITIES))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        deb = [{"arm": "x", "item": "i1", "sample": 0, "gold": 1, "s2": 0, "ok": 0, "n_obj": 2, "n_obj_votes": 2,
                "fired": 1, "one_loser": True, "judge_ok": 1},
               {"arm": "x", "item": "i2", "sample": 0, "gold": 0, "s2": 0, "ok": 1, "n_obj": 0, "n_obj_votes": 0,
                "fired": 0, "one_loser": True, "judge_ok": 0}]
        r = compute(deb)
        ok = r["collective"] == 0.5 and r["routed"] == 1.0 and r["delta"] == 0.5 and r["ol_ratio"] is None
        print("selftest", "OK" if ok else "FAILED"); return 0 if ok else 1
    spec = COMMUNITIES[a.community]
    deb = load(a.community)
    r = compute(deb)
    reg = spec["registered"]
    print(f"PILLAR 3 HEADLINE, independent recomputation -- {a.community} -- judge {spec['judge_model']} standard majority-of-3")
    print(f"  registered values from {reg['source']}")
    print(f"  {'quantity':18s} {'recomputed':>12s} {'registered':>12s}  agree")
    agree_all = True
    for k in ("n", "collective", "fire", "n_fired", "ol_n", "ol_fired", "ol_wrong_fired", "ol_wrong_unfired",
              "ol_ratio", "routed", "delta", "lo", "hi"):
        got, want = r[k], reg[k]
        ag = abs(got - want) <= TOL[k]
        agree_all &= ag
        print(f"  {k:18s} {got:12.4f} {want:12.4f}  {'yes' if ag else 'NO'}")
    print(f"  n_objectors column vs votes' objected_r3 sum: {r['n_objectors_vote_mismatch']} mismatches of {r['n']}")
    print("  breakdown of routed minus collective (descriptive, no CI):")
    for key, tab in r["breakdown"].items():
        for k, v in tab.items():
            print(f"    {key:7s} {k:14s} n {v['n']:5d} fired {v['n_fired']:4d} collective {v['collective']:.4f} routed {v['routed']:.4f} delta {v['delta']:+.4f}")
    print("  VERDICT:", "ALL REGISTERED VALUES REPRODUCED" if agree_all else "DISAGREEMENT (see rows marked NO)")
    out = a.out or OUT / f"verify_pillar3_headline_{a.community}.json"
    out.write_text(json.dumps({"community": a.community, "judge": spec["judge_model"], "registered": reg,
                               "recomputed": r, "tolerances": TOL, "all_agree": agree_all,
                               "bootstrap": {"draws": 4000, "seed": 101, "kind": "item-clustered percentile, numpy"}},
                              indent=1))
    print("  wrote", out)
    return 0 if agree_all else 2


if __name__ == "__main__":
    sys.exit(main())
