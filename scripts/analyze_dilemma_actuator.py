"""
scripts/analyze_dilemma_actuator.py -- Addendum 12 Stage 2: repeat
Addendum 4's A3a rung (cross-vendor escalation from a cold single-agent
read) on the Scruples Dilemmas instrument, using grok's composite stake
flag (confirmed to have grip on this instrument in Stage 1, G3 +0.755) as
the sensor.

WHAT THIS REPLICATES, AND WHY A SEPARATE MODULE AGAIN

Same reasoning as scripts/analyze_dilemma_grip.py: analyze_actuator_ladder's
`evaluate()`/`correct()`/`make_rule_vendor()` are usable in principle
(evaluate's own internals are generic -- item id, arm, a boolean s2_ok, and
a rule callback), but `correct()` and `make_rule_vendor()` both route
through `analyze_crowdgold_sdt.code_response`, which is AITA's five-token
collapsing step and does not apply to a clean two-way ACTION_A/ACTION_B
instrument. This module reimplements the small amount of AITA-specific
glue natively and imports the genuinely generic pieces (`item_half`)
unchanged.

THE TEST

Population: Stage 1's (and any registered extension's) composite-flagged
debates -- neutral objection OR mis-localised advocate objection, exactly
Stage 1's G1 sensor. Actuator: `scripts.run_dilemma_singleagent`'s
cross-vendor (haiku, by default) cold majority-of-3 read on the SAME item.
Outcome: does substituting the actuator's verdict for the deliberation's
own S2 verdict, on flagged debates only, raise accuracy against gold?
Same positive-outcome bar as every actuator test in this programme
(analyze_actuator_ladder.py, analyze_induced_sensor_actuator.py): paired,
item-clustered bootstrap; single actuator tested on 2 populations
(flagged, unflagged-as-context) so alpha=0.025 (Bonferroni family size 2,
matching Addendum 9/10's convention); CI excludes zero; delta >= 0.05;
positive in both item halves (sha256 parity, single arm here so no
by-arm split is meaningful).

Usage:
  python -m scripts.analyze_dilemma_actuator --selftest
  python -m scripts.analyze_dilemma_actuator --tag cg_dilemma_stage1 --baseline-tag cg_dilemma_baseline
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.analyze_actuator_ladder import item_half
from scripts.analyze_dilemma_grip import flagged, load_debates
from scripts.run_phase1_quartet import OUT_DIR

MIN_DELTA = 0.05
FAMILY_ALPHA = 0.05 / 2   # single actuator, 2 populations (flagged + context)
DEFAULT_DRAWS = 4000
VERDICT_SET = ("ACTION_A", "ACTION_B")


def correct(verdict: Optional[str], gold: str) -> Optional[bool]:
    """None (not codable) for anything outside the two committal tokens."""
    if verdict not in VERDICT_SET:
        return None
    return verdict == gold


def load_baseline(path: Path) -> dict[str, list[str]]:
    """item_id -> list of the actuator's per-sample verdicts."""
    out: dict[str, list[str]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["item_id"]].append(r["verdict"])
    return out


def make_rule(baseline: dict[str, list[str]]) -> Callable[[dict], Optional[str]]:
    def rule(d: dict) -> Optional[str]:
        vs = baseline.get(d["item"], [])
        a = sum(1 for v in vs if v == "ACTION_A")
        b = sum(1 for v in vs if v == "ACTION_B")
        if a == b:
            return None          # tie or no data -> fall back to S2
        return "ACTION_A" if a > b else "ACTION_B"
    return rule


def actuated_ok(d: dict, rule: Callable[[dict], Optional[str]]) -> bool:
    v = rule(d)
    ok = correct(v, d["gold"]) if v is not None else None
    return d["s2_ok"] if ok is None else ok


def evaluate(debates: Sequence[dict], rule: Callable[[dict], Optional[str]],
            *, alpha: float, draws: int, seed: int, label: str) -> dict:
    recs = [{"item": d["item"], "arm": d["arm"],
            "delta": int(actuated_ok(d, rule)) - int(d["s2_ok"]),
            "act": int(actuated_ok(d, rule)), "s2": int(d["s2_ok"]),
            "changed": int(rule(d) is not None
                          and correct(rule(d), d["gold"]) is not None
                          and (correct(rule(d), d["gold"]) != d["s2_ok"]))}
           for d in debates]
    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_item[r["item"]].append(r)
    ids = sorted(by_item)
    rng = random.Random(seed)

    def mean_delta(sel_ids):
        rs = [r for i in sel_ids for r in by_item[i]]
        return sum(r["delta"] for r in rs) / len(rs) if rs else None

    point = mean_delta(ids)
    vals = sorted(v for v in (
        mean_delta([ids[rng.randrange(len(ids))] for _ in ids])
        for _ in range(draws)) if v is not None)
    lo = vals[int((alpha / 2) * len(vals))] if vals else None
    hi = vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)] if vals else None

    def sub(pred):
        rs = [r for r in recs if pred(r)]
        return (sum(r["delta"] for r in rs) / len(rs)) if rs else None

    halves = {h: sub(lambda r, h=h: item_half(r["item"]) == h) for h in (0, 1)}
    n_changed = sum(r["changed"] for r in recs)
    n_fallback = sum(1 for d in debates if rule(d) is None
                     or correct(rule(d), d["gold"]) is None)

    crit_ci = lo is not None and lo > 0
    crit_size = point is not None and point >= MIN_DELTA
    crit_rep = all(v is not None and v > 0 for v in halves.values())
    return {
        "label": label, "n": len(recs), "n_items": len(ids),
        "ci_level": 1 - alpha,
        "acc_s2": sum(r["s2"] for r in recs) / len(recs) if recs else None,
        "acc_actuated": sum(r["act"] for r in recs) / len(recs) if recs else None,
        "delta": point, "lo": lo, "hi": hi,
        "delta_by_item_half": halves,
        "n_verdict_changed": n_changed, "n_fallback_to_s2": n_fallback,
        "criteria": {"ci_excludes_zero": crit_ci, "delta_ge_0.05": crit_size,
                     "replicates_both_halves": crit_rep},
        "positive": bool(crit_ci and crit_size and crit_rep),
    }


def _fmt(e: dict) -> str:
    if e.get("delta") is None:
        return "n/a"
    return (f"{e['delta']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}] "
           f"(n={e['n']}, S2 {e['acc_s2']:.3f} -> {e['acc_actuated']:.3f}, "
           f"changed {e['n_verdict_changed']}, fallback {e['n_fallback_to_s2']})")


def run_stage2(rows_path: Path, votes_path: Path, baseline_path: Path, *,
              draws: int = DEFAULT_DRAWS, seed: int = 31) -> dict:
    debates = load_debates(rows_path, votes_path)
    baseline = load_baseline(baseline_path)
    fl = [d for d in debates if flagged(d)]
    unfl = [d for d in debates if not flagged(d)]
    rule = make_rule(baseline)
    fl_res = evaluate(fl, rule, alpha=FAMILY_ALPHA, draws=draws, seed=seed,
                      label="A3a-dilemma flagged")
    unfl_res = evaluate(unfl, rule, alpha=FAMILY_ALPHA, draws=draws,
                        seed=seed + 1,
                        label="A3a-dilemma unflagged (context)")
    sens_gain = (None if fl_res["delta"] is None or unfl_res["delta"] is None
                else fl_res["delta"] - unfl_res["delta"])
    return {
        "n_debates": len(debates), "n_flagged": len(fl),
        "flag_coverage": len(fl) / len(debates) if debates else None,
        "flagged": fl_res, "unflagged_context": unfl_res,
        "sensor_specific_gain": sens_gain,
        "positive": fl_res["positive"],
    }


def print_stage2(res: dict) -> None:
    print("\n" + "=" * 72)
    print("ADDENDUM 12 STAGE 2: cross-vendor actuator on the composite flag "
         "(Scruples Dilemmas)")
    print("=" * 72)
    print(f"  n_debates={res['n_debates']}  n_flagged={res['n_flagged']}  "
         f"flag_coverage={res['flag_coverage']}")
    for key, label in (("flagged", "FLAGGED"), ("unflagged_context",
                                                "UNFLAGGED (context)")):
        e = res[key]
        tag = "POSITIVE" if e["positive"] else "null"
        print(f"  {label:<22} {tag:<8} {_fmt(e)}")
        print(f"      halves {({h: round(v, 3) if v is not None else None for h, v in e['delta_by_item_half'].items()})}"
             f"  criteria {e['criteria']}")
    g = res["sensor_specific_gain"]
    print(f"\n  sensor-specific gain (flagged - unflagged): "
         f"{'n/a' if g is None else f'{g:+.3f}'}")
    print(f"\n  STAGE 2 RESULT: {'POSITIVE' if res['positive'] else 'null'}")


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("correct() is None for a non-committal token",
          correct("UNRESOLVED", "ACTION_A") is None)
    check("correct() is True on a match", correct("ACTION_A", "ACTION_A") is True)
    check("correct() is False on a mismatch",
          correct("ACTION_B", "ACTION_A") is False)

    baseline = {"i1": ["ACTION_A", "ACTION_A", "ACTION_B"],
               "i2": ["ACTION_A", "ACTION_B"],       # tie
               "i3": []}                              # no data
    rule = make_rule(baseline)
    check("make_rule resolves a real majority", rule({"item": "i1"}) == "ACTION_A")
    check("make_rule returns None on a tie", rule({"item": "i2"}) is None)
    check("make_rule returns None on no data", rule({"item": "i3"}) is None)

    d_match = {"item": "i1", "gold": "ACTION_A", "s2_ok": False, "arm": "neutral"}
    check("actuated_ok uses the actuator's verdict when it disagrees with a "
         "wrong S2 and is correct",
         actuated_ok(d_match, rule) is True)
    d_tie = {"item": "i2", "gold": "ACTION_A", "s2_ok": True, "arm": "neutral"}
    check("actuated_ok falls back to S2 on a tied actuator read",
         actuated_ok(d_tie, rule) is True)

    # Engineered population: 30 flagged debates where the actuator FIXES a
    # wrong S2 on 20 of them and is neutral (agrees, or ties, both mapping to
    # the same outcome) on the rest, and 30 unflagged debates where the
    # actuator does nothing (context, no positive claim). This checks the
    # end-to-end wiring rather than just the primitives above.
    debates = []
    baseline2: dict[str, list[str]] = {}
    for i in range(30):
        item = f"fl{i}"
        gold = "ACTION_A" if i % 2 == 0 else "ACTION_B"
        wrong = "ACTION_B" if gold == "ACTION_A" else "ACTION_A"
        fixed = i < 20
        debates.append({"item": item, "arm": "neutral", "gold": gold,
                        "s2_ok": not fixed, "syn_ok": True})
        # actuator votes correctly (gold) 3-0 on the fixed subset, and
        # agrees with the (already-correct) S2 on the rest.
        baseline2[item] = [gold] * 3 if fixed else [gold if not fixed else wrong] * 3
    for i in range(30):
        item = f"un{i}"
        debates.append({"item": item, "arm": "neutral", "gold": "ACTION_A",
                        "s2_ok": True, "syn_ok": True})
        baseline2[item] = ["ACTION_A"] * 3   # agrees with the already-correct S2

    rule2 = make_rule(baseline2)
    fl = [d for d in debates if d["item"].startswith("fl")]
    unfl = [d for d in debates if d["item"].startswith("un")]
    fl_res = evaluate(fl, rule2, alpha=FAMILY_ALPHA, draws=1000, seed=1,
                      label="test flagged")
    unfl_res = evaluate(unfl, rule2, alpha=FAMILY_ALPHA, draws=1000, seed=2,
                        label="test unflagged")
    check("engineered flagged delta is exactly 20/30 = 0.667",
         fl_res["delta"] is not None and abs(fl_res["delta"] - 20 / 30) < 1e-9)
    check("engineered flagged result clears the CI-excludes-zero criterion",
         fl_res["criteria"]["ci_excludes_zero"])
    check("engineered flagged result clears the delta>=0.05 criterion",
         fl_res["criteria"]["delta_ge_0.05"])
    check("engineered flagged result is POSITIVE end to end",
         fl_res["positive"])
    check("engineered unflagged-context delta is 0 (actuator never used, "
         "correct S2 unchanged)", unfl_res["delta"] == 0)
    check("engineered unflagged-context result is null (delta below floor)",
         not unfl_res["positive"])

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag", default="cg_dilemma_stage1")
    ap.add_argument("--baseline-tag", default="cg_dilemma_baseline")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    rows_path = OUT_DIR / f"{args.tag}_rows.csv"
    votes_path = OUT_DIR / f"{args.tag}_votes.csv"
    baseline_path = OUT_DIR / f"{args.baseline_tag}_rows.csv"
    for p in (rows_path, votes_path, baseline_path):
        if not p.exists():
            print(f"\nERROR: {p} not found.\n")
            return 2
    res = run_stage2(rows_path, votes_path, baseline_path,
                     draws=args.n_boot, seed=args.seed)
    print_stage2(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
