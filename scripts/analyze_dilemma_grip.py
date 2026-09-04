"""
scripts/analyze_dilemma_grip.py -- Addendum 12 Stage 1: the stake-grip
screen, ported to the Scruples Dilemmas instrument.

One-to-one correspondence with scripts/analyze_stake_grip.py

Grip is the same three conditions, same thresholds, same item-clustered
bootstrap, on the two person_*_advocate seats instead of AITA's
writer_advocate / counterparty:

  G1  composite-flag fire rate <= 0.50
  G2  REJECT share of R4 votes >= 0.05
  G3  stake concentration on the stake seats (R4 reject rate | undermined
      minus | not undermined) >= +0.20, 95% item-clustered CI excludes 0

WHY A SEPARATE MODULE INSTEAD OF PARAMETERISING THE EXISTING ONE

analyze_stake_grip.py (via analyze_actuator_ladder.load_debates and
analyze_loop_step.flagged / mislocalised) hard-codes STAKE_SEATS =
("writer_advocate", "counterparty") and routes every verdict through
analyze_crowdgold_sdt.code_response, which collapses AITA's five-token
vocabulary (YTA/NTA/ESH/NAH/UNRESOLVED) into a published/private at-fault
bit. Dilemmas' instrument is already a clean two-way token (ACTION_A /
ACTION_B) with no collapsing step to do, and patching a module five addenda
of standing AITA results depend on, to carry a second task's role names, is
a correctness risk for a roughly 150-line saving. This module reimplements
the small amount of logic natively; every threshold and formula below is
copy-checked against analyze_stake_grip.py.

Usage:
  python -m scripts.analyze_dilemma_grip --selftest
  python -m scripts.analyze_dilemma_grip --tag cg_dilemma
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.run_phase1_quartet import OUT_DIR

STAKE_SEATS = ("person_a_advocate", "person_b_advocate")
G1_MAX_FIRE, G2_MIN_REJECT, G3_MIN_CONC = 0.50, 0.05, 0.20


def load_debates(rows_path: Path, votes_path: Path) -> list[dict]:
    """One record per debate, with the sensor fields flagged/mislocalised
    below expect: obj_<seat>, undermined_<seat>.
    """
    debates: dict[tuple, dict] = {}
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            key = (r["item_id"], r["arm"], r["sample_idx"])
            debates[key] = {
                "item": r["item_id"], "arm": r["arm"],
                "sample_idx": int(r["sample_idx"]),
                "gold": r["gold_verdict"], "verdict": r["verdict"],
                "synthesis_verdict": r["synthesis_verdict"],
                "syn_ok": (r["synthesis_verdict"] == r["gold_verdict"]
                          if r["synthesis_verdict"] in ("ACTION_A", "ACTION_B")
                          else None),
                "s2_ok": (r["verdict"] == r["gold_verdict"]
                         if r["verdict"] in ("ACTION_A", "ACTION_B") else None),
                "n_objectors": int(r.get("n_objectors") or 0),
                "n_reject": int(r.get("n_reject") or 0),
                "unanimous_accept": r.get("unanimous_accept") == "1",
                "verdict_revised": r.get("verdict_revised") == "1",
            }
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            d = debates.get((v["item_id"], v["arm"], v["sample_idx"]))
            if d is None:
                continue
            seat = v["role_id"]
            d[f"obj_{seat}"] = v["objected_r3"] == "1"
            d[f"rej_{seat}"] = v["r3_label"] == "REJECT"
            und = v.get("stake_undermined")
            if seat in STAKE_SEATS and und not in (None, ""):
                d[f"undermined_{seat}"] = bool(int(und))
    return [d for d in debates.values()
            if d["syn_ok"] is not None and d["s2_ok"] is not None]


def mislocalised(d: dict) -> bool:
    return any(d.get(f"obj_{s}") and not d.get(f"undermined_{s}", True)
              for s in STAKE_SEATS)


def neutral_objected(d: dict) -> bool:
    return bool(d.get("obj_neutral_adjudicator"))


def flagged(d: dict) -> bool:
    """The composite sensor: neutral objection OR mis-localised objection.

    Identical definition to analyze_loop_step.flagged.
    """
    return neutral_objected(d) or mislocalised(d)


def load_votes(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append({
                "item": r["item_id"], "arm": r["arm"], "role": r["role_id"],
                "sample_idx": int(r["sample_idx"]),
                "reject": int(r.get("reject") or 0),
                "undermined": int(r.get("stake_undermined") or 0),
                "objected": int(r.get("objected_r3") or 0),
                "r3_label": r.get("r3_label", ""),
            })
    return out


def _boot_ci(by_item: dict[str, list], stat: Callable[[list], Optional[float]],
            *, draws: int, seed: int, alpha: float = 0.05):
    ids = sorted(by_item)
    rng = random.Random(seed)
    point = stat([x for i in ids for x in by_item[i]])
    vals = sorted(v for v in (
        stat([x for i in (ids[rng.randrange(len(ids))] for _ in ids)
             for x in by_item[i]]) for _ in range(draws)) if v is not None)
    if not vals:
        return point, None, None
    return (point, vals[int(alpha / 2 * len(vals))],
           vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)])


def stake_concentration(votes: Sequence[dict], *, draws: int, seed: int) -> dict:
    sv = [v for v in votes if v["role"] in STAKE_SEATS]
    by_item: dict[str, list] = defaultdict(list)
    for v in sv:
        by_item[v["item"]].append(v)

    def stat(vs):
        u = [v["reject"] for v in vs if v["undermined"]]
        n = [v["reject"] for v in vs if not v["undermined"]]
        if not u or not n:
            return None
        return sum(u) / len(u) - sum(n) / len(n)

    p, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed)
    return {
        "n_votes": len(sv), "n_undermined": sum(v["undermined"] for v in sv),
        "reject_rate_undermined": (sum(v["reject"] for v in sv if v["undermined"])
                                   / max(1, sum(v["undermined"] for v in sv))),
        "reject_rate_not_undermined": (
            sum(v["reject"] for v in sv if not v["undermined"])
            / max(1, sum(1 for v in sv if not v["undermined"]))),
        "delta": p, "lo": lo, "hi": hi,
    }


def grip(debates: Sequence[dict], votes: Sequence[dict], *, draws: int = 2000,
         seed: int = 7) -> dict:
    n = len(debates)
    fire = sum(flagged(d) for d in debates) / n if n else None
    rej = sum(v["reject"] for v in votes) / len(votes) if votes else None
    conc = stake_concentration(votes, draws=draws, seed=seed)
    seat = {}
    for s in STAKE_SEATS + ("neutral_adjudicator",):
        sv = [v for v in votes if v["role"] == s]
        seat[s] = {
            "objected_r3": sum(v["objected"] for v in sv) / len(sv) if sv else None,
            "reject_r4": sum(v["reject"] for v in sv) / len(sv) if sv else None,
            "undermined": (sum(v["undermined"] for v in sv) / len(sv)
                          if sv and s in STAKE_SEATS else None),
        }
    g1 = fire is not None and fire <= G1_MAX_FIRE
    g2 = rej is not None and rej >= G2_MIN_REJECT
    g3 = (conc["delta"] is not None and conc["delta"] >= G3_MIN_CONC
         and conc["lo"] is not None and conc["lo"] > 0)
    return {
        "n_debates": n, "n_items": len({d["item"] for d in debates}),
        "n_votes": len(votes),
        "G1_fire_rate": fire, "G2_reject_share": rej,
        "G3_stake_concentration": conc, "per_seat": seat,
        "neutral_objected_rate": sum(neutral_objected(d) for d in debates) / n if n else None,
        "mislocalised_rate": sum(mislocalised(d) for d in debates) / n if n else None,
        "s2_accuracy": sum(d["s2_ok"] for d in debates) / n if n else None,
        "criteria": {"G1": g1, "G2": g2, "G3": g3},
        "grip": bool(g1 and g2 and g3),
    }


def print_grip(g: dict) -> None:
    print("\n" + "=" * 72)
    print("STAKE-GRIP SCREEN (Addendum 12 Stage 1, Scruples Dilemmas)")
    print("=" * 72)
    print(f"  n_debates={g['n_debates']}  n_items={g['n_items']}  "
          f"n_votes={g['n_votes']}  S2 accuracy={g['s2_accuracy']}")
    print(f"  G1 fire rate {g['G1_fire_rate']:.3f} (<= {G1_MAX_FIRE}) "
          f"-> {'PASS' if g['criteria']['G1'] else 'fail'}")
    print(f"  G2 reject share {g['G2_reject_share']:.3f} (>= {G2_MIN_REJECT}) "
          f"-> {'PASS' if g['criteria']['G2'] else 'fail'}")
    c = g["G3_stake_concentration"]
    lo = "n/a" if c["lo"] is None else f"{c['lo']:+.3f}"
    hi = "n/a" if c["hi"] is None else f"{c['hi']:+.3f}"
    delta = "n/a" if c["delta"] is None else f"{c['delta']:+.3f}"
    print(f"  G3 stake concentration {delta} [{lo}, {hi}] (>= "
          f"+{G3_MIN_CONC}, CI excl. 0) -> "
          f"{'PASS' if g['criteria']['G3'] else 'fail'}")
    print(f"  reject|undermined {c['reject_rate_undermined']:.3f}  "
          f"reject|not_undermined {c['reject_rate_not_undermined']:.3f}")
    print(f"\n  GRIP: {'yes' if g['grip'] else 'no'}")


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("STAKE_SEATS names the two dilemma advocates",
          STAKE_SEATS == ("person_a_advocate", "person_b_advocate"))

    d_flagged_neutral = {"obj_neutral_adjudicator": True}
    d_flagged_misloc = {"obj_person_a_advocate": True,
                        "undermined_person_a_advocate": False}
    d_clean = {"obj_person_a_advocate": True,
              "undermined_person_a_advocate": True,
              "obj_neutral_adjudicator": False}
    check("neutral objection alone flags", flagged(d_flagged_neutral))
    check("mis-localised advocate objection alone flags",
          flagged(d_flagged_misloc))
    check("a correctly-localised objection with no neutral objection "
          "does not flag", not flagged(d_clean))

    # A tiny synthetic run: 20 items, 3 seats each, engineered so G3 is
    # unambiguously positive and G1/G2 pass, to check the arithmetic end to
    # end rather than just the sensor predicates above. Half the items'
    # A-advocate seat rejects (10/20), which is a large enough sample and
    # effect that the bootstrap CI reliably excludes zero.
    n_items = 20
    votes = []
    for i in range(n_items):
        item = f"i{i}"
        # A-advocate: undermined every time (verdict is always ACTION_A),
        # rejects on half the items when undermined.
        rej = int(i % 2 == 0)
        votes.append({"item": item, "arm": "neutral", "role": "person_a_advocate",
                     "sample_idx": 0, "reject": rej, "undermined": 1,
                     "objected": rej,
                     "r3_label": "REJECT" if rej else "ACCEPT"})
        # B-advocate: never undermined (verdict is always ACTION_A, which
        # serves B's stake), never rejects.
        votes.append({"item": item, "arm": "neutral", "role": "person_b_advocate",
                     "sample_idx": 0, "reject": 0, "undermined": 0,
                     "objected": 0, "r3_label": "ACCEPT"})
        votes.append({"item": item, "arm": "neutral", "role": "neutral_adjudicator",
                     "sample_idx": 0, "reject": 0, "undermined": 0,
                     "objected": 0, "r3_label": "ACCEPT"})
    conc = stake_concentration(votes, draws=1000, seed=1)
    check("stake concentration on the synthetic run is exactly the "
          "engineered 0.5 - 0 = 0.5",
          conc["delta"] is not None and abs(conc["delta"] - 0.5) < 1e-9)
    check("CI excludes zero on a clean, engineered-positive signal",
          conc["lo"] is not None and conc["lo"] > 0)

    debates = [{"item": f"i{i}", "syn_ok": True, "s2_ok": True,
               "obj_neutral_adjudicator": False,
               "obj_person_a_advocate": i % 2 == 0,
               "undermined_person_a_advocate": True,
               "obj_person_b_advocate": False} for i in range(n_items)]
    g = grip(debates, votes, draws=1000, seed=2)
    check("G1 fire rate is 0 (no mislocalised or neutral objections here)",
          g["G1_fire_rate"] == 0.0 and g["criteria"]["G1"])
    check("G2 reject share is 10/60, clears the 0.05 floor",
          abs(g["G2_reject_share"] - 10 / 60) < 1e-9 and g["criteria"]["G2"])
    check("G3 clears +0.20 with CI excluding zero on this engineered signal",
          g["criteria"]["G3"])
    check("grip is True when all three criteria pass",
          g["grip"] is True)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag", default="cg_dilemma")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    rows_path = OUT_DIR / f"{args.tag}_rows.csv"
    votes_path = OUT_DIR / f"{args.tag}_votes.csv"
    if not rows_path.exists() or not votes_path.exists():
        print(f"\nERROR: {rows_path} / {votes_path} not found. Run "
              "scripts.run_crowdgold_dilemma first.\n")
        return 2
    debates = load_debates(rows_path, votes_path)
    votes = load_votes(votes_path)
    g = grip(debates, votes, draws=args.n_boot, seed=args.seed)
    print_grip(g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
