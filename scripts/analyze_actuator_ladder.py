"""
scripts/analyze_actuator_ladder.py -- zero-spend rungs of the actuator ladder
(Guidance_Documents/prereg_embodiment_community.md, Addendum 4).

Population: flagged debates from the k=4 deliberation rows (composite flag,
frozen in analyze_loop_step). Comparator in every rung: S2, the integrated
proposal's verdict, on the SAME debates (paired). Non-codable actuated
verdicts fall back to S2 so abstention cannot manufacture a gain.

Rungs computed here (all free; iteration 1 replayed under the no-spend guard):
  A0   selective emission baseline (not a stopping rung)
  A1   statement against interest: mis-localised objector's own R2 verdict
  A2   stakeless deference: neutral adjudicator's R2 verdict when it objected
  A3a  cross-vendor escalation from cache: haiku standard majority-of-3

Positive-outcome criteria are applied mechanically, exactly as registered:
  (i)   CI excludes zero (98.75% for the zero-spend family of four)
  (ii)  delta >= +0.05
  (iii) point delta > 0 in both item halves (sha256 parity) and both arms

Usage
-----
  python -m scripts.analyze_actuator_ladder --selftest
  python -m scripts.analyze_actuator_ladder
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_loop_step import (
    flagged, flagged_severe, mislocalised, mislocalised_reject,
    neutral_objected, neutral_rejected,
)
from scripts.run_loop_iteration import no_spend, replay_debate
from scripts.run_phase1_quartet import OUT_DIR
from scripts.verdict_format import extract_verdict

ROWS_PATH = OUT_DIR / "cg_deliberation_rows.csv"
VOTES_PATH = OUT_DIR / "cg_deliberation_votes.csv"
COMPARATOR_PATH = OUT_DIR / "cg_scaffold_combined_rows_k3.csv"
ANALYSIS_PATH = OUT_DIR / "actuator_ladder_analysis.json"

STAKE_SEATS = ("writer_advocate", "counterparty")
ZERO_SPEND_ALPHA = 0.05 / 4      # Bonferroni over the four zero-spend rungs
MIN_DELTA = 0.05
DEFAULT_DRAWS = 4000


def correct(verdict: str, gold: str) -> Optional[bool]:
    c = code_response(verdict, "published")
    return None if c is None else c == (1 if gold == "YTA" else 0)


def item_half(item_id: str) -> int:
    return hashlib.sha256(item_id.encode()).digest()[-1] % 2


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_debates(rows_path: Path = ROWS_PATH, votes_path: Path = VOTES_PATH,
                 ) -> list[dict]:
    """All codable debates with the sensor fields analyze_loop_step expects."""
    debates: dict[tuple, dict] = {}
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            key = (r["item_id"], r["arm"], r["sample_idx"])
            syn_code = code_response(r["synthesis_verdict"], "published")
            debates[key] = {
                "model": r["model"], "item": r["item_id"], "arm": r["arm"],
                "sample_idx": int(r["sample_idx"]), "gold": r["gold_verdict"],
                "s1": r["synthesis_verdict"], "s2": r["verdict"],
                "syn_ok": correct(r["synthesis_verdict"], r["gold_verdict"]),
                "s2_ok": correct(r["verdict"], r["gold_verdict"]),
                "syn_code": syn_code,
            }
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            d = debates.get((v["item_id"], v["arm"], v["sample_idx"]))
            if d is None:
                continue
            seat = v["role_id"]
            d[f"obj_{seat}"] = v["objected_r3"] == "1"
            d[f"rej_{seat}"] = v["r3_label"] == "REJECT"
            want = v["role_stake"]
            if d["syn_code"] is not None and want in ("at_fault", "not_at_fault"):
                d[f"undermined_{seat}"] = (d["syn_code"] == 1) != (want == "at_fault")
    return [d for d in debates.values()
            if d["syn_ok"] is not None and d["s2_ok"] is not None]


def attach_r2_verdicts(debates: Sequence[dict], items_by_id: dict) -> None:
    """Replay iteration 1 from cache (no spend) and attach each seat's R2 verdict."""
    with no_spend():
        for d in debates:
            st = replay_debate(d["model"], d["arm"], items_by_id[d["item"]],
                               d["sample_idx"])
            for rid, text in st["r2"].items():
                d[f"r2_{rid}"] = extract_verdict(text, rcd.INSTRUMENT)


def load_comparator(path: Path = COMPARATOR_PATH) -> dict[tuple, list[str]]:
    """(model, scaffold, arm, item_id) -> list of cached single-agent verdicts."""
    out: dict[tuple, list[str]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["model"], r["scaffold"], r["arm"], r["item_id"])].append(
                r["verdict"])
    return out


# ---------------------------------------------------------------------------
# Rung rules: each maps a debate -> actuated verdict string or None (fallback)
# ---------------------------------------------------------------------------

def mislocal_seat(d: dict) -> Optional[str]:
    seats = [s for s in STAKE_SEATS
             if d.get(f"obj_{s}") and not d.get(f"undermined_{s}", True)]
    return seats[0] if len(seats) == 1 else None


def rule_a1_r2(d: dict) -> Optional[str]:
    s = mislocal_seat(d)
    return d.get(f"r2_{s}") if s else None


def rule_a1_flip(d: dict) -> Optional[str]:
    """Verdict class opposite S1; needs no R2 parse."""
    if d["syn_code"] is None:
        return None
    return "NTA" if d["syn_code"] == 1 else "YTA"


def rule_a2_neutral(d: dict) -> Optional[str]:
    return d.get("r2_neutral_adjudicator")


def make_rule_vendor(comp: dict[tuple, list[str]], model: str, scaffold: str,
                     ) -> Callable[[dict], Optional[str]]:
    def rule(d: dict) -> Optional[str]:
        vs = comp.get((model, scaffold, d["arm"], d["item"]), [])
        codes = [code_response(v, "published") for v in vs]
        codes = [c for c in codes if c is not None]
        if not codes:
            return None
        n1 = sum(1 for c in codes if c == 1)
        n0 = len(codes) - n1
        if n1 == n0:
            return None          # tie -> fall back to S2
        return "YTA" if n1 > n0 else "NTA"
    return rule


# ---------------------------------------------------------------------------
# Paired, item-clustered evaluation with the registered positive criteria
# ---------------------------------------------------------------------------

def actuated_ok(d: dict, rule) -> bool:
    v = rule(d)
    ok = correct(v, d["gold"]) if v is not None else None
    return d["s2_ok"] if ok is None else ok


def evaluate(debates: Sequence[dict], rule, *, alpha: float, draws: int,
             seed: int, label: str) -> dict:
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
    arms = {a: sub(lambda r, a=a: r["arm"] == a)
            for a in sorted({r["arm"] for r in recs})}
    n_changed = sum(r["changed"] for r in recs)
    n_fallback = sum(1 for d in debates if rule(d) is None
                     or correct(rule(d), d["gold"]) is None)

    crit_ci = lo is not None and lo > 0
    crit_size = point is not None and point >= MIN_DELTA
    crit_rep = (all(v is not None and v > 0 for v in halves.values())
                and all(v is not None and v > 0 for v in arms.values()))
    return {
        "label": label, "n": len(recs), "n_items": len(ids),
        "ci_level": 1 - alpha,
        "acc_s2": sum(r["s2"] for r in recs) / len(recs) if recs else None,
        "acc_actuated": sum(r["act"] for r in recs) / len(recs) if recs else None,
        "delta": point, "lo": lo, "hi": hi,
        "delta_by_item_half": halves, "delta_by_arm": arms,
        "n_verdict_changed": n_changed, "n_fallback_to_s2": n_fallback,
        "criteria": {"ci_excludes_zero": crit_ci, "delta_ge_0.05": crit_size,
                     "replicates_halves_and_arms": crit_rep},
        "positive": bool(crit_ci and crit_size and crit_rep),
    }


def selective_emission(debates: Sequence[dict]) -> dict:
    out = {}
    n = len(debates)
    acc_all = sum(d["s2_ok"] for d in debates) / n
    for name, sig in (("composite_flag", flagged), ("severe_flag", flagged_severe)):
        keep = [d for d in debates if not sig(d)]
        out[name] = {
            "coverage": len(keep) / n,
            "accuracy_emitted": (sum(d["s2_ok"] for d in keep) / len(keep)
                                 if keep else None),
            "accuracy_held": (sum(d["s2_ok"] for d in debates if sig(d))
                              / max(1, n - len(keep))),
        }
    out["accuracy_all"] = acc_all
    return out


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

def run_ladder(debates: Sequence[dict], comp: dict[tuple, list[str]], *,
               draws: int = DEFAULT_DRAWS, seed: int = 23) -> dict:
    fl = [d for d in debates if flagged(d)]
    unfl = [d for d in debates if not flagged(d)]
    a = ZERO_SPEND_ALPHA
    res: dict = {"n_debates": len(debates), "n_flagged": len(fl),
                 "A0_selective_emission": selective_emission(debates)}

    # A1 -- statement against interest
    pop = [d for d in fl if mislocal_seat(d)]
    res["A1"] = evaluate(pop, rule_a1_r2, alpha=a, draws=draws, seed=seed,
                         label="A1 against-interest objector's R2 verdict")
    res["A1_sens_reject_only"] = evaluate(
        [d for d in pop if mislocalised_reject(d)], rule_a1_r2, alpha=a,
        draws=draws, seed=seed + 1, label="A1 REJECT-only")
    res["A1_sens_flip"] = evaluate(pop, rule_a1_flip, alpha=a, draws=draws,
                                   seed=seed + 2, label="A1 flip rule")

    # A2 -- stakeless deference
    pop2 = [d for d in fl if neutral_objected(d)]
    res["A2"] = evaluate(pop2, rule_a2_neutral, alpha=a, draws=draws,
                         seed=seed + 3, label="A2 neutral's R2 verdict")
    res["A2_sens_reject_only"] = evaluate(
        [d for d in pop2 if neutral_rejected(d)], rule_a2_neutral, alpha=a,
        draws=draws, seed=seed + 4, label="A2 neutral REJECT only")

    # A3a -- cross-vendor escalation from cache
    haiku_std = make_rule_vendor(comp, "claude-haiku-4-5", "standard")
    res["A3a"] = evaluate(fl, haiku_std, alpha=a, draws=draws, seed=seed + 5,
                          label="A3a haiku standard majority-3")
    res["A3a_unflagged"] = evaluate(unfl, haiku_std, alpha=a, draws=draws,
                                    seed=seed + 6,
                                    label="A3a rule on UNFLAGGED (context)")
    d_fl, d_un = res["A3a"]["delta"], res["A3a_unflagged"]["delta"]
    res["A3a_sensor_specific_gain"] = (
        None if d_fl is None or d_un is None else d_fl - d_un)
    for tag, m, sc in (("nano_std", "gpt-5.4-nano", "standard"),
                       ("haiku_narr", "claude-haiku-4-5", "narrative_cot"),
                       ("nano_narr", "gpt-5.4-nano", "narrative_cot")):
        res[f"A3a_sens_{tag}"] = evaluate(
            fl, make_rule_vendor(comp, m, sc), alpha=a, draws=draws,
            seed=seed + 7, label=f"A3a {m} {sc}")

    order = ("A1", "A2", "A3a")
    first = next((k for k in order if res[k]["positive"]), None)
    res["ladder_result"] = first or "no zero-spend rung positive"
    res["positives_in_order"] = [k for k in order if res[k]["positive"]]
    return res


def _fmt(e: dict) -> str:
    if e.get("delta") is None:
        return "n/a"
    return (f"{e['delta']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}] "
            f"(n={e['n']}, S2 {e['acc_s2']:.3f} -> {e['acc_actuated']:.3f}, "
            f"changed {e['n_verdict_changed']}, fallback {e['n_fallback_to_s2']})")


def print_report(res: dict) -> None:
    print(f"\n=== actuator ladder, zero-spend rungs "
          f"({res['n_debates']} debates, {res['n_flagged']} flagged) ===")
    a0 = res["A0_selective_emission"]
    print(f"A0 selective emission: overall {a0['accuracy_all']:.3f}")
    for k in ("composite_flag", "severe_flag"):
        s = a0[k]
        print(f"   hold {k:<15} coverage {s['coverage']:.3f}  emitted "
              f"{s['accuracy_emitted']:.3f}  held {s['accuracy_held']:.3f}")
    for k in ("A1", "A1_sens_reject_only", "A1_sens_flip", "A2",
              "A2_sens_reject_only", "A3a", "A3a_unflagged",
              "A3a_sens_nano_std", "A3a_sens_haiku_narr", "A3a_sens_nano_narr"):
        e = res[k]
        tag = "POSITIVE" if e["positive"] else "null"
        primary = k in ("A1", "A2", "A3a")
        print(f"{'>>' if primary else '  '} {k:<22} {tag:<8} {_fmt(e)}")
        if primary:
            print(f"      halves {({h: round(v, 3) if v is not None else None for h, v in e['delta_by_item_half'].items()})}"
                  f"  arms {({h: round(v, 3) if v is not None else None for h, v in e['delta_by_arm'].items()})}"
                  f"  criteria {e['criteria']}")
    g = res["A3a_sensor_specific_gain"]
    print(f"\nA3a sensor-specific gain (flagged - unflagged): "
          f"{'n/a' if g is None else f'{g:+.3f}'}")
    print(f"\nLADDER RESULT: {res['ladder_result']}")


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    rng = random.Random(9)
    debates = []
    for i in range(600):
        gold = "YTA" if rng.random() < 0.4 else "NTA"
        arm = "third_person" if i % 2 else "as_asker"
        # S1 favours NTA (code 0) so writer_advocate's stake is NOT undermined;
        # make S2 correct 59% of the time.
        s2_ok = rng.random() < 0.59
        s2 = gold if s2_ok else ("NTA" if gold == "YTA" else "YTA")
        d = {"model": "m", "item": f"i{i % 120}", "arm": arm, "sample_idx": 0,
             "gold": gold, "s1": "NTA", "s2": s2, "syn_code": 0,
             "syn_ok": gold == "NTA", "s2_ok": s2_ok,
             "obj_writer_advocate": True, "undermined_writer_advocate": False,
             "obj_counterparty": False, "undermined_counterparty": True,
             "obj_neutral_adjudicator": False,
             "rej_writer_advocate": True, "rej_counterparty": False,
             "rej_neutral_adjudicator": False}
        # planted: the against-interest seat's R2 verdict is right 85% of the time
        d["r2_writer_advocate"] = gold if rng.random() < 0.85 else (
            "NTA" if gold == "YTA" else "YTA")
        d["r2_neutral_adjudicator"] = "UNRESOLVED"     # non-codable -> fallback
        debates.append(d)
    comp: dict[tuple, list[str]] = defaultdict(list)
    for d in debates:                                  # haiku = coin flip
        comp[("claude-haiku-4-5", "standard", d["arm"], d["item"])] = [
            "YTA" if rng.random() < 0.5 else "NTA" for _ in range(3)]

    res = run_ladder(debates, comp, draws=400)
    check("every synthetic debate is flagged (mis-localised by construction)",
          res["n_flagged"] == 600)
    check("A1 recovers the planted against-interest gain and is POSITIVE",
          res["A1"]["positive"] and res["A1"]["delta"] > 0.15)
    # S1 is NTA and gold is NTA 60% of the time, so flipping S1 is worse than
    # S2 (59%) here: the flip rule must come out NEGATIVE, not positive.
    check("A1 flip rule negative when S1 is usually right",
          res["A1_sens_flip"]["delta"] < 0
          and not res["A1_sens_flip"]["positive"])
    check("A2 population empty (neutral never objected) -> n=0, not positive",
          res["A2"]["n"] == 0 and not res["A2"]["positive"])
    # A coin-flip vendor against a 59% S2 loses about 9 points: not positive,
    # and negative in sign.
    check("A3a coin-flip vendor is not positive and loses to S2",
          not res["A3a"]["positive"] and res["A3a"]["delta"] < 0)
    check("ladder result is A1", res["ladder_result"] == "A1")

    # Fallback: a rule returning None must contribute exactly zero delta.
    e = evaluate(debates, lambda d: None, alpha=0.05, draws=50, seed=1,
                 label="none")
    check("None rule -> delta 0 and full fallback",
          e["delta"] == 0 and e["n_fallback_to_s2"] == len(debates))
    # Replication check must fail when one arm is negative.
    def arm_biased(d):
        return d["gold"] if d["arm"] == "as_asker" else (
            "NTA" if d["gold"] == "YTA" else "YTA")
    e2 = evaluate(debates, arm_biased, alpha=0.05, draws=50, seed=2, label="ab")
    check("arm-carried gain fails the replication criterion",
          not e2["criteria"]["replicates_halves_and_arms"])
    check("item_half is deterministic and binary",
          item_half("abc") in (0, 1) and item_half("abc") == item_half("abc"))

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Actuator ladder, zero-spend rungs")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--json", type=Path, default=ANALYSIS_PATH)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    debates = load_debates()
    items = rcd.load_items(source="scruples", n_yta=99, n_nta=150,
                           min_votes=50, min_consensus=0.90, seed=44)
    by_id = {i.item_id: i for i in items}
    fl = [d for d in debates if flagged(d)]
    print(f"{len(debates)} codable debates, {len(fl)} flagged; replaying "
          f"flagged from cache for R2 verdicts ...")
    attach_r2_verdicts(fl, by_id)
    comp = load_comparator()
    res = run_ladder(debates, comp, draws=a.draws)
    print_report(res)
    a.json.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
