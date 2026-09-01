"""
scripts/analyze_loop_step.py -- retrospective control-loop analysis of the
deliberation protocol (Phase 0 of the loop programme; see
Guidance_Documents/not_origins_review.md Sec 8 and the E4/E5 registrations in
prereg_embodiment_community.md).

THE OBSERVATION THIS EXPLOITS
-----------------------------
Every deliberation debate already contains ONE closed loop iteration:

    synthesis (recorded verdict)  ->  R3 seat objections  ->  integration
    ->  final group verdict (recorded)

with crowd gold on every item.  So the loop's one-step dynamics are computable
from the E4/E5 caches at zero spend:

  CORRECTION   P(final correct | synthesis wrong)   -- the fix rate
  CORRUPTION   P(final wrong | synthesis right)     -- the break rate
  NET-COUNTS   accuracy(final) - accuracy(synthesis): at high synthesis
               accuracy an UNGATED loop turn can be net-negative in counts
               even when fix-rate >> break-rate, because the right pool is
               much larger than the wrong pool.  This is the case for a
               sensor: gate the loop on an error signal.
  SENSOR       which R3 objection patterns predict a wrong synthesis.  Two
               candidates: (a) the neutral adjudicator objecting (it has no
               stake, so its objection is never interest-explained), and
               (b) MIS-LOCALISED objections -- a stake-bearing seat objecting
               although the synthesis did NOT undermine its stake.
  GATED VALUE  the counterfactual: apply the integration step only on flagged
               debates, keep the raw synthesis otherwise.

All interval estimates are item-clustered bootstrap percentile CIs.

Usage
-----
  python -m scripts.analyze_loop_step --selftest
  python -m scripts.analyze_loop_step
  python -m scripts.analyze_loop_step --json divergence_study_outputs/loop_step_analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from scripts.analyze_crowdgold_sdt import code_response
from scripts.run_phase1_quartet import OUT_DIR

ROWS_PATH = OUT_DIR / "cg_deliberation_rows.csv"
VOTES_PATH = OUT_DIR / "cg_deliberation_votes.csv"
ANALYSIS_PATH = OUT_DIR / "loop_step_analysis.json"
SEATS = ("writer_advocate", "counterparty", "neutral_adjudicator")
STAKE_SEATS = ("writer_advocate", "counterparty")
DEFAULT_DRAWS = 4000


def _correct(verdict: str, gold: str) -> Optional[bool]:
    c = code_response(verdict, "published")
    if c is None:
        return None
    return c == (1 if gold == "YTA" else 0)


def load_debates(rows_path: Path = ROWS_PATH, votes_path: Path = VOTES_PATH,
                 ) -> list[dict]:
    """One record per debate with codable synthesis AND final verdicts."""
    debates: dict[tuple, dict] = {}
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            key = (r["item_id"], r["arm"], r["sample_idx"])
            debates[key] = {
                "item": r["item_id"], "arm": r["arm"],
                "syn_ok": _correct(r["synthesis_verdict"], r["gold_verdict"]),
                "fin_ok": _correct(r["verdict"], r["gold_verdict"]),
                "syn_code": code_response(r["synthesis_verdict"], "published"),
            }
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            d = debates.get((v["item_id"], v["arm"], v["sample_idx"]))
            if d is None:
                continue
            seat = v["role_id"]
            d[f"obj_{seat}"] = v["objected_r3"] == "1"
            want = v["role_stake"]
            if d["syn_code"] is not None and want in ("at_fault", "not_at_fault"):
                d[f"undermined_{seat}"] = (
                    (d["syn_code"] == 1) != (want == "at_fault"))
    return [d for d in debates.values()
            if d["syn_ok"] is not None and d["fin_ok"] is not None]


def mislocalised(d: dict) -> bool:
    """A stake-bearing seat objected although its stake was not undermined."""
    return any(
        d.get(f"obj_{s}") and not d.get(f"undermined_{s}", True)
        for s in STAKE_SEATS
    )


def neutral_objected(d: dict) -> bool:
    return bool(d.get("obj_neutral_adjudicator"))


def flagged(d: dict) -> bool:
    """The composite sensor: neutral objection OR mis-localised objection."""
    return neutral_objected(d) or mislocalised(d)


# ---------------------------------------------------------------------------
# Item-clustered bootstrap over debate-level statistics
# ---------------------------------------------------------------------------

def boot(stat_fn, by_item: dict[str, list[dict]], *, draws: int, seed: int,
         ) -> dict:
    ids = sorted(by_item)
    rng = random.Random(seed)

    def ev(sample_ids):
        return stat_fn([d for i in sample_ids for d in by_item[i]])

    point = ev(ids)
    vals = []
    for _ in range(draws):
        v = ev([ids[rng.randrange(len(ids))] for _ in ids])
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return {"point": point, "lo95": None, "hi95": None}
    return {"point": point,
            "lo95": vals[int(0.025 * len(vals))],
            "hi95": vals[min(int(0.975 * len(vals)), len(vals) - 1)]}


def _rate(ds: Sequence[dict], cond, out) -> Optional[float]:
    sel = [d for d in ds if cond(d)]
    return sum(out(d) for d in sel) / len(sel) if sel else None


def analyze(debates: Sequence[dict], *, draws: int = DEFAULT_DRAWS,
            seed: int = 7) -> dict:
    by_item: dict[str, list[dict]] = defaultdict(list)
    for d in debates:
        by_item[d["item"]].append(d)

    res: dict = {"n_debates": len(debates), "n_items": len(by_item)}

    def acc_syn(ds):
        return _rate(ds, lambda d: True, lambda d: d["syn_ok"])

    def acc_fin(ds):
        return _rate(ds, lambda d: True, lambda d: d["fin_ok"])

    def fix(ds):
        return _rate(ds, lambda d: not d["syn_ok"], lambda d: d["fin_ok"])

    def brk(ds):
        return _rate(ds, lambda d: d["syn_ok"], lambda d: not d["fin_ok"])

    def net_counts(ds):
        a, b = acc_fin(ds), acc_syn(ds)
        return None if a is None or b is None else a - b

    def gated_net_counts(ds):
        """Counterfactual: integrate only flagged debates, else keep synthesis."""
        vals = [(d["fin_ok"] if flagged(d) else d["syn_ok"]) - d["syn_ok"]
                for d in ds]
        return sum(vals) / len(vals) if vals else None

    res["synthesis_accuracy"] = boot(acc_syn, by_item, draws=draws, seed=seed)
    res["final_accuracy"] = boot(acc_fin, by_item, draws=draws, seed=seed + 1)
    res["fix_rate"] = boot(fix, by_item, draws=draws, seed=seed + 2)
    res["break_rate"] = boot(brk, by_item, draws=draws, seed=seed + 3)
    res["net_gain_rates"] = boot(
        lambda ds: (lambda f, b: None if f is None or b is None else f - b)(
            fix(ds), brk(ds)),
        by_item, draws=draws, seed=seed + 4)
    res["net_gain_counts_ungated"] = boot(net_counts, by_item, draws=draws,
                                          seed=seed + 5)
    res["net_gain_counts_gated"] = boot(gated_net_counts, by_item, draws=draws,
                                        seed=seed + 6)

    # Sensor validity: P(synthesis wrong | signal) - P(wrong | no signal).
    for name, sig in (("neutral_objection", neutral_objected),
                      ("mislocalised_objection", mislocalised),
                      ("composite_flag", flagged)):
        def lift(ds, s=sig):
            a = _rate(ds, s, lambda d: not d["syn_ok"])
            b = _rate(ds, lambda d: not s(d), lambda d: not d["syn_ok"])
            return None if a is None or b is None else a - b

        res[f"sensor_{name}_lift"] = boot(lift, by_item, draws=draws,
                                          seed=seed + 7)
        sel = [d for d in debates if sig(d)]
        res[f"sensor_{name}_n"] = len(sel)
        res[f"sensor_{name}_precision"] = (
            _rate(sel, lambda d: True, lambda d: not d["syn_ok"]))
        wrong = [d for d in debates if not d["syn_ok"]]
        res[f"sensor_{name}_recall"] = (
            _rate(wrong, lambda d: True, lambda d: 1 if sig(d) else 0))

    # Does the loop already use the sensor: fix rate flagged vs unflagged.
    def fix_gap(ds):
        a = _rate(ds, lambda d: not d["syn_ok"] and flagged(d),
                  lambda d: d["fin_ok"])
        b = _rate(ds, lambda d: not d["syn_ok"] and not flagged(d),
                  lambda d: d["fin_ok"])
        return None if a is None or b is None else a - b

    res["fix_rate_flagged_minus_unflagged"] = boot(fix_gap, by_item,
                                                   draws=draws, seed=seed + 8)
    return res


def _fmt(d: dict) -> str:
    if d.get("point") is None:
        return "n/a"
    lo = f"{d['lo95']:+.3f}" if d.get("lo95") is not None else "?"
    hi = f"{d['hi95']:+.3f}" if d.get("hi95") is not None else "?"
    return f"{d['point']:+.3f} [{lo}, {hi}]"


def print_report(res: dict) -> None:
    print(f"\n=== one-step loop dynamics ({res['n_debates']} debates, "
          f"{res['n_items']} items) ===")
    print(f"synthesis accuracy:      {_fmt(res['synthesis_accuracy'])}")
    print(f"final accuracy:          {_fmt(res['final_accuracy'])}")
    print(f"fix rate  P(ok|wrong):   {_fmt(res['fix_rate'])}")
    print(f"break rate P(bad|right): {_fmt(res['break_rate'])}")
    print(f"net gain (rates):        {_fmt(res['net_gain_rates'])}")
    print(f"net gain (counts, UNGATED loop): {_fmt(res['net_gain_counts_ungated'])}")
    print(f"net gain (counts, GATED loop):   {_fmt(res['net_gain_counts_gated'])}")
    print("\nsensors: P(synthesis wrong | signal) - P(wrong | no signal)")
    for name in ("neutral_objection", "mislocalised_objection", "composite_flag"):
        p = res[f"sensor_{name}_precision"]
        r = res[f"sensor_{name}_recall"]
        print(f"  {name:<24} lift {_fmt(res[f'sensor_{name}_lift'])}   "
              f"precision {p:.3f}  recall {r:.3f}  n={res[f'sensor_{name}_n']}")
    print(f"\nfix-rate gap (flagged - unflagged wrong syntheses): "
          f"{_fmt(res['fix_rate_flagged_minus_unflagged'])}")


# ---------------------------------------------------------------------------
# Selftest on synthetic debates
# ---------------------------------------------------------------------------

def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    rng = random.Random(1)
    debates = []
    # 400 debates, 80 items; wrong syntheses carry the neutral objection 70%
    # of the time vs 10% otherwise; loop fixes flagged-wrong at 60%, breaks 2%.
    for i in range(400):
        syn_ok = rng.random() < 0.8
        obj_n = rng.random() < (0.10 if syn_ok else 0.70)
        fin_ok = syn_ok
        if not syn_ok and obj_n and rng.random() < 0.6:
            fin_ok = True
        if syn_ok and rng.random() < 0.02:
            fin_ok = False
        debates.append({
            "item": f"i{i % 80}", "arm": "third_person",
            "syn_ok": syn_ok, "fin_ok": fin_ok, "syn_code": 1,
            "obj_neutral_adjudicator": obj_n,
            "obj_writer_advocate": False, "obj_counterparty": False,
            "undermined_writer_advocate": True,
            "undermined_counterparty": True,
        })
    res = analyze(debates, draws=400)
    check("fix rate recovered (~0.42 = 0.7*0.6)",
          res["fix_rate"]["point"] is not None
          and 0.25 < res["fix_rate"]["point"] < 0.60)
    check("break rate small", res["break_rate"]["point"] < 0.06)
    check("neutral sensor lift positive and CI excludes zero",
          res["sensor_neutral_objection_lift"]["lo95"] > 0)
    check("gated loop beats ungated in counts",
          res["net_gain_counts_gated"]["point"]
          >= res["net_gain_counts_ungated"]["point"])
    check("no mislocalised objections in this synthetic set",
          res["sensor_mislocalised_objection_n"] == 0)

    # mislocalised() logic: objection without undermining flags; with does not.
    d1 = {"obj_writer_advocate": True, "undermined_writer_advocate": False,
          "obj_counterparty": False}
    d2 = {"obj_writer_advocate": True, "undermined_writer_advocate": True,
          "obj_counterparty": False}
    check("mislocalised flags stakeless objection", mislocalised(d1))
    check("mislocalised ignores stake-explained objection", not mislocalised(d2))

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One-step loop dynamics")
    ap.add_argument("--rows", type=Path, default=ROWS_PATH)
    ap.add_argument("--votes", type=Path, default=VOTES_PATH)
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--json", type=Path, default=ANALYSIS_PATH)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    debates = load_debates(a.rows, a.votes)
    res = analyze(debates, draws=a.draws)
    print_report(res)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
