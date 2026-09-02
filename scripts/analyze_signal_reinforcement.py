"""
scripts/analyze_signal_reinforcement.py -- Addendum 5 analysis (S1, S2, S2b).

S1  Sensor x actuator selection on item half 0 (sha256 parity), confirmation
    on half 1.  Objective is COMPOSED-SYSTEM accuracy over all codable
    debates: sensor fires -> actuator verdict (fallback S2), else S2.
    Positive iff the half-0 winner beats the registered configuration
    (composite flag + haiku standard majority-3) on half 1 with a 95%
    item-clustered CI excluding zero.  Otherwise exploratory.
S2  Crux escalation: haiku reading post + R3 objections (k=3 majority, tie
    -> S2) vs haiku reading the post cold (A3a), paired on the flagged
    debates.  Two-sided 95% CI.
S2b Actuator stabilisation: haiku majority-5 vs majority-3 on flagged.

All inputs are cached artefacts; this script makes no API calls.

Usage
-----
  python -m scripts.analyze_signal_reinforcement --selftest
  python -m scripts.analyze_signal_reinforcement            # S1 only
  python -m scripts.analyze_signal_reinforcement --crux divergence_study_outputs/crux_escalation_rows.csv
  python -m scripts.analyze_signal_reinforcement --k5-rows divergence_study_outputs/<k5 rows>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_actuator_ladder import (
    DEFAULT_DRAWS, actuated_ok, attach_r2_verdicts, correct, item_half,
    load_comparator, load_debates, make_rule_vendor, evaluate,
)
from scripts.analyze_loop_step import flagged, flagged_severe
from scripts.analyze_crowdgold_sdt import code_response
from scripts.run_phase1_quartet import OUT_DIR

ANALYSIS_PATH = OUT_DIR / "signal_reinforcement_analysis.json"
REGISTERED = ("composite", "haiku_maj3")

# ---------------------------------------------------------------------------
# Pre-declared candidates (Addendum 5, S1)
# ---------------------------------------------------------------------------

def _revised(d: dict) -> bool:
    a, b = code_response(d["s1"], "published"), code_response(d["s2"], "published")
    return a is not None and b is not None and a != b


SENSORS: dict[str, Callable[[dict], bool]] = {
    "composite": flagged,
    "severe": flagged_severe,
    "any_objection": lambda d: d["n_objectors"] >= 1,
    "any_reject_vote": lambda d: d["n_reject"] >= 1,
    "not_unanimous_accept": lambda d: not d["unanimous_accept"],
    "verdict_revised": _revised,
    "composite_or_reject": lambda d: flagged(d) or d["n_reject"] >= 1,
}


def _majority(codes: list[int]) -> Optional[str]:
    if not codes:
        return None
    n1 = sum(codes)
    n0 = len(codes) - n1
    if n1 == n0:
        return None
    return "YTA" if n1 > n0 else "NTA"


def _codes(comp, model, scaffold, d) -> list[int]:
    vs = comp.get((model, scaffold, d["arm"], d["item"]), [])
    return [c for c in (code_response(v, "published") for v in vs) if c is not None]


def make_actuators(comp) -> dict[str, Callable[[dict], Optional[str]]]:
    haiku = make_rule_vendor(comp, "claude-haiku-4-5", "standard")

    def six_vote(d):
        return _majority(_codes(comp, "claude-haiku-4-5", "standard", d)
                         + _codes(comp, "gpt-5.4-nano", "standard", d))

    def unanimous(d):
        cs = _codes(comp, "claude-haiku-4-5", "standard", d)
        if len(cs) < 3 or len(set(cs)) != 1:
            return None
        return "YTA" if cs[0] == 1 else "NTA"

    return {"haiku_maj3": haiku, "haiku_nano_6vote": six_vote,
            "haiku_unanimous_else_s2": unanimous}


# ---------------------------------------------------------------------------
# Composed-system accuracy and paired comparison
# ---------------------------------------------------------------------------

def composed_ok(d: dict, sensor, actuator) -> bool:
    return actuated_ok(d, actuator) if sensor(d) else bool(d["s2_ok"])


def clustered_ci(recs: Sequence[dict], key: str, *, draws: int, seed: int,
                 alpha: float = 0.05) -> tuple[Optional[float], Optional[float], Optional[float]]:
    by_item: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        by_item[r["item"]].append(r[key])
    ids = sorted(by_item)
    if not ids:
        return None, None, None
    rng = random.Random(seed)

    def mean_of(sel):
        vals = [v for i in sel for v in by_item[i]]
        return sum(vals) / len(vals)

    point = mean_of(ids)
    boots = sorted(mean_of([ids[rng.randrange(len(ids))] for _ in ids])
                   for _ in range(draws))
    lo = boots[int(alpha / 2 * len(boots))]
    hi = boots[min(int((1 - alpha / 2) * len(boots)), len(boots) - 1)]
    return point, lo, hi


def run_s1(debates: Sequence[dict], comp, *, draws: int, seed: int = 31) -> dict:
    acts = make_actuators(comp)
    h0 = [d for d in debates if item_half(d["item"]) == 0]
    h1 = [d for d in debates if item_half(d["item"]) == 1]

    def acc(ds, s, a):
        return sum(composed_ok(d, SENSORS[s], acts[a]) for d in ds) / len(ds)

    ranking = sorted(
        ({"sensor": s, "actuator": a, "acc_half0": acc(h0, s, a),
          "fire_rate_half0": sum(SENSORS[s](d) for d in h0) / len(h0)}
         for s in SENSORS for a in acts),
        key=lambda r: -r["acc_half0"])
    best = ranking[0]
    sel = (best["sensor"], best["actuator"])
    recs = [{"item": d["item"],
             "d": int(composed_ok(d, SENSORS[sel[0]], acts[sel[1]]))
                  - int(composed_ok(d, SENSORS[REGISTERED[0]], acts[REGISTERED[1]]))}
            for d in h1]
    point, lo, hi = clustered_ci(recs, "d", draws=draws, seed=seed)
    s2_acc_h1 = sum(d["s2_ok"] for d in h1) / len(h1)
    return {
        "n_half0": len(h0), "n_half1": len(h1),
        "ranking_half0": ranking,
        "selected": {"sensor": sel[0], "actuator": sel[1]},
        "registered": {"sensor": REGISTERED[0], "actuator": REGISTERED[1]},
        "selected_is_registered": sel == REGISTERED,
        "half1": {
            "acc_s2": s2_acc_h1,
            "acc_registered": acc(h1, *REGISTERED),
            "acc_selected": acc(h1, *sel),
            "delta_selected_minus_registered": point, "lo": lo, "hi": hi,
            "n_items": len({r["item"] for r in recs}),
        },
        "positive": bool(not (sel == REGISTERED) and lo is not None and lo > 0),
        # exploratory: every candidate on half 1, for the record
        "half1_all": sorted(
            ({"sensor": s, "actuator": a, "acc_half1": acc(h1, s, a)}
             for s in SENSORS for a in acts),
            key=lambda r: -r["acc_half1"]),
    }


# ---------------------------------------------------------------------------
# S2: crux escalation vs cold
# ---------------------------------------------------------------------------

def load_crux(path: Path) -> dict[tuple, list[str]]:
    out: dict[tuple, list[str]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["item_id"], r["arm"], int(r["sample_idx"]))] = (
                r["crux_verdicts"].split("|") if r["crux_verdicts"] else [])
    return out


def run_s2(flagged_debates: Sequence[dict], comp, crux: dict[tuple, list[str]],
           *, draws: int, seed: int = 37) -> dict:
    cold = make_rule_vendor(comp, "claude-haiku-4-5", "standard")

    def crux_rule(d):
        vs = crux.get((d["item"], d["arm"], d["sample_idx"]), [])
        return _majority([c for c in (code_response(v, "published") for v in vs)
                          if c is not None])

    ds = [d for d in flagged_debates
          if (d["item"], d["arm"], d["sample_idx"]) in crux]
    recs = [{"item": d["item"], "arm": d["arm"],
             "crux": int(actuated_ok(d, crux_rule)),
             "cold": int(actuated_ok(d, cold)),
             "s2": int(d["s2_ok"])} for d in ds]
    for r in recs:
        r["d_crux_cold"] = r["crux"] - r["cold"]
        r["d_crux_s2"] = r["crux"] - r["s2"]
    p, lo, hi = clustered_ci(recs, "d_crux_cold", draws=draws, seed=seed)
    p2, lo2, hi2 = clustered_ci(recs, "d_crux_s2", draws=draws, seed=seed + 1)
    by_arm = {a: (sum(r["d_crux_cold"] for r in recs if r["arm"] == a)
                  / max(1, sum(1 for r in recs if r["arm"] == a)))
              for a in sorted({r["arm"] for r in recs})}
    n_fallback = sum(1 for d in ds if crux_rule(d) is None)
    n_parse_fail = sum(1 for d in ds
                       if any(code_response(v, "published") is None
                              for v in crux[(d["item"], d["arm"], d["sample_idx"])]))
    return {
        "n": len(recs), "n_items": len({r["item"] for r in recs}),
        "n_missing_crux_rows": len(flagged_debates) - len(ds),
        "acc_s2": sum(r["s2"] for r in recs) / len(recs) if recs else None,
        "acc_cold_haiku": sum(r["cold"] for r in recs) / len(recs) if recs else None,
        "acc_crux_haiku": sum(r["crux"] for r in recs) / len(recs) if recs else None,
        "delta_crux_minus_cold": p, "lo": lo, "hi": hi,
        "delta_crux_minus_cold_by_arm": by_arm,
        "delta_crux_minus_s2": p2, "lo_s2": lo2, "hi_s2": hi2,
        "n_crux_tie_fallback": n_fallback,
        "n_debates_with_any_uncodable_crux_sample": n_parse_fail,
        "agreement_crux_cold": (sum(1 for r in recs if r["crux"] == r["cold"])
                                / len(recs) if recs else None),
        "positive_two_sided": bool(lo is not None and (lo > 0 or hi < 0)),
        "direction": (None if lo is None else
                      "crux_better" if lo > 0 else
                      "crux_worse_recorrelation" if hi < 0 else "null"),
    }


# ---------------------------------------------------------------------------
# S2b: majority-5 vs majority-3
# ---------------------------------------------------------------------------

def run_s2b(flagged_debates: Sequence[dict], comp3, comp5, *, draws: int,
            seed: int = 41) -> dict:
    r3 = make_rule_vendor(comp3, "claude-haiku-4-5", "standard")
    r5 = make_rule_vendor(comp5, "claude-haiku-4-5", "standard")
    n5 = [len(_codes(comp5, "claude-haiku-4-5", "standard", d)) for d in flagged_debates]
    recs = [{"item": d["item"], "d": int(actuated_ok(d, r5)) - int(actuated_ok(d, r3)),
             "m5": int(actuated_ok(d, r5)), "m3": int(actuated_ok(d, r3))}
            for d in flagged_debates]
    p, lo, hi = clustered_ci(recs, "d", draws=draws, seed=seed)
    e5 = evaluate(flagged_debates, r5, alpha=0.05, draws=draws, seed=seed + 1,
                  label="A3a haiku standard majority-5")
    return {
        "n": len(recs), "samples_per_cell_k5": {
            "min": min(n5) if n5 else None, "max": max(n5) if n5 else None},
        "acc_maj3": sum(r["m3"] for r in recs) / len(recs) if recs else None,
        "acc_maj5": sum(r["m5"] for r in recs) / len(recs) if recs else None,
        "delta_maj5_minus_maj3": p, "lo": lo, "hi": hi,
        "A3a_with_maj5": e5,
    }


# ---------------------------------------------------------------------------

def print_report(res: dict) -> None:
    if "S1" in res:
        s = res["S1"]
        print(f"\nS1  selection on half 0 (n={s['n_half0']}), confirmation on half 1 "
              f"(n={s['n_half1']})")
        print("  half-0 ranking (top 8):")
        for r in s["ranking_half0"][:8]:
            print(f"    {r['acc_half0']:.3f}  fire={r['fire_rate_half0']:.2f}  "
                  f"{r['sensor']} + {r['actuator']}")
        h = s["half1"]
        print(f"  selected  : {s['selected']['sensor']} + {s['selected']['actuator']}"
              f"{'  (== registered)' if s['selected_is_registered'] else ''}")
        print(f"  half 1    : S2 {h['acc_s2']:.3f}  registered {h['acc_registered']:.3f}"
              f"  selected {h['acc_selected']:.3f}")
        if h["delta_selected_minus_registered"] is not None:
            print(f"  delta(sel - reg) = {h['delta_selected_minus_registered']:+.3f} "
                  f"95% CI [{h['lo']:+.3f}, {h['hi']:+.3f}]")
        print(f"  POSITIVE: {s['positive']}")
    if "S2" in res:
        s = res["S2"]
        print(f"\nS2  crux escalation, n={s['n']} flagged debates "
              f"({s['n_missing_crux_rows']} missing)")
        print(f"  acc S2 {s['acc_s2']:.3f}  cold haiku {s['acc_cold_haiku']:.3f}  "
              f"crux haiku {s['acc_crux_haiku']:.3f}")
        print(f"  delta(crux - cold) = {s['delta_crux_minus_cold']:+.3f} "
              f"95% CI [{s['lo']:+.3f}, {s['hi']:+.3f}]  by arm "
              f"{ {k: round(v, 3) for k, v in s['delta_crux_minus_cold_by_arm'].items()} }")
        print(f"  delta(crux - S2)   = {s['delta_crux_minus_s2']:+.3f} "
              f"95% CI [{s['lo_s2']:+.3f}, {s['hi_s2']:+.3f}]")
        print(f"  crux/cold agreement {s['agreement_crux_cold']:.3f}; tie fallbacks "
              f"{s['n_crux_tie_fallback']}; uncodable-sample debates "
              f"{s['n_debates_with_any_uncodable_crux_sample']}")
        print(f"  direction: {s['direction']}  POSITIVE(two-sided): "
              f"{s['positive_two_sided']}")
    if "S2b" in res:
        s = res["S2b"]
        print(f"\nS2b majority-5 vs majority-3 on flagged, n={s['n']} "
              f"(k5 samples per cell {s['samples_per_cell_k5']})")
        print(f"  acc maj3 {s['acc_maj3']:.3f}  maj5 {s['acc_maj5']:.3f}  "
              f"delta {s['delta_maj5_minus_maj3']:+.3f} 95% CI [{s['lo']:+.3f}, {s['hi']:+.3f}]")
        e = s["A3a_with_maj5"]
        print(f"  A3a(maj5) vs S2: delta {e['delta']:+.3f} CI [{e['lo']:+.3f}, {e['hi']:+.3f}]"
              f"  halves {e['delta_by_item_half']}  arms {e['delta_by_arm']}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--crux", type=Path, default=None)
    ap.add_argument("--k5-rows", type=Path, default=None)
    ap.add_argument("--skip-s1", action="store_true")
    ap.add_argument("--json", type=Path, default=ANALYSIS_PATH)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    debates = load_debates()
    fl = [d for d in debates if flagged(d)]
    comp = load_comparator()
    res: dict = {"n_codable": len(debates), "n_flagged": len(fl)}
    if not a.skip_s1:
        res["S1"] = run_s1(debates, comp, draws=a.draws)
    if a.crux is not None:
        res["S2"] = run_s2(fl, comp, load_crux(a.crux), draws=a.draws)
    if a.k5_rows is not None:
        res["S2b"] = run_s2b(fl, comp, load_comparator(a.k5_rows), draws=a.draws)
    print_report(res)
    existing = json.loads(a.json.read_text()) if a.json.exists() else {}
    existing.update(res)
    a.json.write_text(json.dumps(existing, indent=1))
    print(f"\nwrote {a.json}")
    return 0


# ---------------------------------------------------------------------------

def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    rng = random.Random(3)
    debates, comp = [], defaultdict(list)
    # Synthetic world: S2 is right 60%; haiku is right 85% on flagged items;
    # nano is a coin flip; composite flag fires on half the debates.
    for i in range(160):
        item = f"it{i:03d}"
        gold = "YTA" if i % 2 else "NTA"
        for arm in ("third_person", "as_asker"):
            flag = rng.random() < 0.5
            s2 = gold if rng.random() < 0.6 else ("NTA" if gold == "YTA" else "YTA")
            d = {"model": "m", "item": item, "arm": arm, "sample_idx": 0,
                 "gold": gold, "s1": s2, "s2": s2,
                 "syn_ok": s2 == gold, "s2_ok": s2 == gold, "syn_code": 1,
                 "n_objectors": 1 if flag else 0, "n_reject": 1 if flag and rng.random() < .5 else 0,
                 "unanimous_accept": not flag, "verdict_revised": False,
                 "obj_writer_advocate": "x" if flag else "",
                 "undermined_writer_advocate": False,
                 "undermined_counterparty": False,
                 "obj_counterparty": "", "obj_neutral_adjudicator": "",
                 "rej_writer_advocate": "", "rej_counterparty": "",
                 "rej_neutral_adjudicator": ""}
            debates.append(d)
            for _ in range(3):
                comp[("claude-haiku-4-5", "standard", arm, item)].append(
                    gold if rng.random() < 0.85 else ("NTA" if gold == "YTA" else "YTA"))
                comp[("gpt-5.4-nano", "standard", arm, item)].append(
                    gold if rng.random() < 0.5 else ("NTA" if gold == "YTA" else "YTA"))
    # sensor fields analyze_loop_step reads: make sure flagged() sees them
    n_fl = sum(flagged(d) for d in debates)
    check("synthetic composite flag fires on a nontrivial share", 0.2 < n_fl / len(debates) < 0.8)

    s1 = run_s1(debates, comp, draws=300)
    check("S1 ranking covers all 21 candidates", len(s1["ranking_half0"]) == 21)
    check("S1 half-0 winner is a haiku-only actuator (nano is noise)",
          s1["selected"]["actuator"] != "haiku_nano_6vote")
    check("S1 half-1 fields present",
          s1["half1"]["acc_registered"] is not None and s1["half1"]["acc_selected"] is not None)
    # a candidate that fires on everything with an 85% actuator should beat the
    # registered composite (which only fires on ~half): expect positive delta
    any_all = next(r for r in s1["half1_all"]
                   if r["sensor"] == "not_unanimous_accept" and r["actuator"] == "haiku_maj3")
    check("held-out accuracy computed for every candidate", any_all["acc_half1"] > 0.5)

    fl = [d for d in debates if flagged(d)]
    crux_good = {(d["item"], d["arm"], d["sample_idx"]): [d["gold"]] * 3 for d in fl}
    s2 = run_s2(fl, comp, crux_good, draws=300)
    check("S2 oracle crux beats cold haiku with CI > 0",
          s2["lo"] is not None and s2["lo"] > 0 and s2["direction"] == "crux_better")
    crux_bad = {k: [("NTA" if v[0] == "YTA" else "YTA")] * 3 for k, v in crux_good.items()}
    s2b_ = run_s2(fl, comp, crux_bad, draws=300)
    check("S2 anti-oracle crux is flagged as re-correlation",
          s2b_["direction"] == "crux_worse_recorrelation")
    crux_tie = {k: ["YTA", "NTA", "UNRESOLVED"] for k in crux_good}
    s2t = run_s2(fl, comp, crux_tie, draws=100)
    check("S2 ties fall back to S2 and are counted",
          s2t["n_crux_tie_fallback"] == len(fl))

    comp5 = defaultdict(list, {k: list(v) for k, v in comp.items()})
    for k, v in comp.items():
        if k[0] == "claude-haiku-4-5":
            gold = "YTA" if int(k[3][2:]) % 2 else "NTA"
            comp5[k] = v + [gold, gold]
    s2b = run_s2b(fl, comp, comp5, draws=300)
    check("S2b sees 5 samples per cell", s2b["samples_per_cell_k5"] == {"min": 5, "max": 5})
    check("S2b majority-5 with two extra correct votes is not worse",
          s2b["delta_maj5_minus_maj3"] >= 0)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
