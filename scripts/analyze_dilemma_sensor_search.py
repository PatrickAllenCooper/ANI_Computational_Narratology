"""
scripts/analyze_dilemma_sensor_search.py -- addresses point 2 of the
opposing-interests review (see Guidance_Documents/prereg_embodiment_community.md,
Addendum 12 Stage 2 close): the composite flag (`analyze_dilemma_grip.flagged`)
fires on only 1.0% of codable Dilemmas debates (6/624), two orders of
magnitude rarer than the same predicate on AITA (20.09%, 337/1677). Before
concluding "no task-general trigger exists on this instrument," this module
asks the question directly: does ANY other native, non-induced signal
computable from grok's OWN Dilemmas deliberation transcripts do better as a
routing sensor?

WHY THIS IS A DIFFERENT LINE FROM ADDENDUM 11's CLOSED SENSOR SEARCH

Addendum 11 Phase 0 (`scripts/analyze_sensor_search.py`) searched for a
routing sensor on AITA's INDUCED (few-shot-taught) haiku/nano population,
built for a signal that was itself trained into the model. That line is
closed; nothing here reopens it. This module searches grok's NATIVE
Dilemmas population -- the same 624-debate, never-induced panel Addendum
12 Stage 1/2 already used -- for a different reason: not because the
composite flag might be broken, but because Stage 2 already showed
Dilemmas' stake-tracking is close to deterministic (reject rate ~77-81%
when a seat's stake is undermined vs ~6-10% when it is not), which
predicts in advance that any candidate wide enough to raise coverage above
the composite's 1.0% will mostly re-admit the ordinary, correctly-localised
reject the composite flag was built to exclude. This module tests that
prediction rather than assuming it.

WHY A SEPARATE MODULE, NOT A PATCH TO analyze_sensor_search.py

Same reasoning as `analyze_dilemma_grip.py` and `analyze_dilemma_actuator.py`:
`analyze_sensor_search.py` routes through `analyze_actuator_ladder.evaluate`,
which in turn routes through `analyze_crowdgold_sdt.code_response` --
AITA's five-token collapsing step, not applicable to Dilemmas' clean
ACTION_A/ACTION_B instrument. This module reuses only the genuinely
generic pieces already built for Dilemmas: `load_debates`/`load_votes`
from `analyze_dilemma_grip.py`, and `evaluate`/`correct`/`make_rule`/
`load_baseline` from `analyze_dilemma_actuator.py` (both already generic
over item id / arm / a boolean s2_ok / a rule callback). No AITA-specific
code is touched or imported.

CANDIDATES (fixed list, registered before any is scored)

  1. composite                 -- flagged() (control; Addendum 12's own sensor)
  2. either_advocate_reject    -- either advocate seat rejects at R4,
                                   regardless of stake direction (the
                                   coverage-maximizing relaxation of the
                                   composite's mis-localisation filter)
  3. both_advocates_reject     -- both advocate seats reject at R4
  4. r4_nonunanimous           -- final vote round is not unanimous accept
  5. r4_reject_ge2             -- >=2 of 3 R4 votes are REJECT
  6. neutral_alone             -- neutral objects AND no mis-localised
                                   advocate objection (splits the composite
                                   into its two components)
  7. r3_any_reject             -- >=1 R3 label is REJECT
  8. r3_reject_ge2             -- >=2 R3 labels are REJECT
  9. verdict_revised           -- synthesis verdict changed between S1 and S2
 10. undermined_reject         -- a stake seat rejects AND is itself
                                   undermined (the "expected"/majority-class
                                   dissent -- a diagnostic complement, not a
                                   sensor candidate we expect to route on)
 11. objection_kind_unresolvable -- any seat's R3 objection is coded
                                   "unresolvable" (a content-based split of
                                   the near-universal dissent, rather than a
                                   raw count)

 12. controversial_item (SEPARATE CATEGORY, not part of the 11-candidate
     Bonferroni family's interpretation as a "sensor," though scored under
     the same statistical bar for comparability) -- the corpus's own
     pre-existing crowd-annotation `controversial` flag. This is a
     corpus-difficulty prior, not a deliberation behaviour: it costs zero
     API calls to compute, so a positive result here says nothing about
     whether the deliberation diagnoses its own error, and must never be
     reported as a "grip sensor" alongside 1-11.

Design mirrors `analyze_sensor_search.py`'s own architecture (fixed
candidate dict -> score against a shared actuator rule -> rank), adapted
to Dilemmas' evaluate()/rule shape.

Positivity bar, derived (not copied uncritically from Addendum 5/11):
  - coverage floor: n_flagged >= 15 on the 624-debate panel before any CI
    is even attempted (Addendum 11 Phase 0's own documented failure mode:
    a 5/94 candidate looked best by point estimate and was pure noise).
  - statistical floor: `analyze_dilemma_actuator.evaluate`'s own criteria
    (item-clustered bootstrap CI on actuator gain excludes zero, delta on
    the flagged set >= +0.05) at Bonferroni alpha = 0.05/12 (family size =
    the full candidate list, 11 native + the covariate scored for
    comparability).
  - economic floor: coverage * delta >= 0.01 (roughly a quarter of AITA's
    own validated actuator's aggregate lift, 0.20 * 0.193 ~= 0.039) -- a
    technically-significant but economically negligible candidate should
    not be promoted to a growth stage.

Usage
-----
  python -m scripts.analyze_dilemma_sensor_search --selftest
  python -m scripts.analyze_dilemma_sensor_search --tag cg_dilemma_stage1 --baseline-tag cg_dilemma_baseline_stage2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.analyze_dilemma_actuator import evaluate, load_baseline, make_rule
from scripts.analyze_dilemma_grip import STAKE_SEATS, flagged, load_debates
from scripts.run_phase1_quartet import OUT_DIR

ANALYSIS_PATH = OUT_DIR / "dilemma_sensor_search_analysis.json"
N_CANDIDATES = 12                 # 11 native + 1 covariate, scored together
FAMILY_ALPHA = 0.05 / N_CANDIDATES
MIN_COVERAGE_N = 15
MIN_DELTA = 0.05
MIN_ECONOMIC = 0.01
DEFAULT_DRAWS = 4000


def attach_extra_fields(debates: Sequence[dict], rows_path: Path,
                        votes_path: Path) -> None:
    """`n_r3_reject` and `controversial` live on the rows CSV but aren't
    carried by analyze_dilemma_grip.load_debates; `r3_objection_kind` is a
    per-vote field on the votes CSV. Merge all three in directly."""
    extra: dict[tuple, dict] = {}
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            key = (r["item_id"], r["arm"], r["sample_idx"])
            extra[key] = {
                "n_r3_reject": int(r.get("n_r3_reject") or 0),
                "controversial": r.get("controversial") == "1",
            }
    unresolvable: dict[tuple, bool] = {}
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            key = (v["item_id"], v["arm"], v["sample_idx"])
            if v.get("r3_objection_kind") == "unresolvable":
                unresolvable[key] = True
    for d in debates:
        key = (d["item"], d["arm"], str(d["sample_idx"]))
        e = extra.get(key, {})
        d["n_r3_reject"] = e.get("n_r3_reject", 0)
        d["controversial"] = e.get("controversial", False)
        d["any_unresolvable"] = unresolvable.get(key, False)


def either_advocate_reject(d: dict) -> bool:
    return any(d.get(f"rej_{s}") for s in STAKE_SEATS)


def both_advocates_reject(d: dict) -> bool:
    return all(d.get(f"rej_{s}") for s in STAKE_SEATS)


def r4_nonunanimous(d: dict) -> bool:
    return not d.get("unanimous_accept", True)


def r4_reject_ge2(d: dict) -> bool:
    return d.get("n_reject", 0) >= 2


def neutral_alone(d: dict) -> bool:
    return bool(d.get("obj_neutral_adjudicator")) and not mislocalised_local(d)


def mislocalised_local(d: dict) -> bool:
    return any(d.get(f"obj_{s}") and not d.get(f"undermined_{s}", True)
              for s in STAKE_SEATS)


def r3_any_reject(d: dict) -> bool:
    return d.get("n_r3_reject", 0) >= 1


def r3_reject_ge2(d: dict) -> bool:
    return d.get("n_r3_reject", 0) >= 2


def verdict_revised(d: dict) -> bool:
    return bool(d.get("verdict_revised"))


def undermined_reject(d: dict) -> bool:
    return any(d.get(f"rej_{s}") and d.get(f"undermined_{s}") for s in STAKE_SEATS)


def objection_kind_unresolvable(d: dict) -> bool:
    return bool(d.get("any_unresolvable"))


def controversial_item(d: dict) -> bool:
    return bool(d.get("controversial"))


CANDIDATES: dict[str, Callable[[dict], bool]] = {
    "composite": flagged,
    "either_advocate_reject": either_advocate_reject,
    "both_advocates_reject": both_advocates_reject,
    "r4_nonunanimous": r4_nonunanimous,
    "r4_reject_ge2": r4_reject_ge2,
    "neutral_alone": neutral_alone,
    "r3_any_reject": r3_any_reject,
    "r3_reject_ge2": r3_reject_ge2,
    "verdict_revised": verdict_revised,
    "undermined_reject": undermined_reject,
    "objection_kind_unresolvable": objection_kind_unresolvable,
}
COVARIATE = "controversial_item"


def score_candidate(debates: Sequence[dict], sig: Callable[[dict], bool],
                    rule: Callable[[dict], Optional[str]], *, alpha: float,
                    draws: int, seed: int, label: str) -> dict:
    fl = [d for d in debates if sig(d)]
    unfl = [d for d in debates if not sig(d)]
    n = len(debates)
    acc_all = sum(d["s2_ok"] for d in debates) / n if n else None
    acc_fl = sum(d["s2_ok"] for d in fl) / len(fl) if fl else None
    acc_unfl = sum(d["s2_ok"] for d in unfl) / len(unfl) if unfl else None
    raw_lift = None if acc_fl is None or acc_unfl is None else acc_unfl - acc_fl
    fl_eval = evaluate(fl, rule, alpha=alpha, draws=draws, seed=seed,
                       label=f"{label} flagged")
    unfl_eval = evaluate(unfl, rule, alpha=alpha, draws=draws, seed=seed + 1,
                         label=f"{label} unflagged (context)")
    d_fl, d_un = fl_eval["delta"], unfl_eval["delta"]
    sens_gain = None if d_fl is None or d_un is None else d_fl - d_un
    coverage = len(fl) / n if n else None
    economic = (None if coverage is None or d_fl is None
               else coverage * d_fl)
    meets_coverage_floor = len(fl) >= MIN_COVERAGE_N
    meets_statistical_floor = bool(fl_eval["positive"])
    meets_economic_floor = economic is not None and economic >= MIN_ECONOMIC
    return {
        "candidate": label, "n_flagged": len(fl), "n_total": n,
        "coverage": coverage,
        "acc_all": acc_all, "acc_flagged": acc_fl, "acc_unflagged": acc_unfl,
        "raw_lift": raw_lift,       # positive => flagged set is MORE error-prone
        "actuator_flagged": fl_eval, "actuator_unflagged": unfl_eval,
        "sensor_specific_gain": sens_gain,
        "economic_lift": economic,
        "criteria": {
            "coverage_floor_n_ge_15": meets_coverage_floor,
            "statistical_floor": meets_statistical_floor,
            "economic_floor_ge_0.01": meets_economic_floor,
        },
        "promoted": bool(meets_coverage_floor and meets_statistical_floor
                        and meets_economic_floor),
    }


def run_search(rows_path: Path, votes_path: Path, baseline_path: Path, *,
              draws: int = DEFAULT_DRAWS, seed: int = 41) -> dict:
    debates = load_debates(rows_path, votes_path)
    attach_extra_fields(debates, rows_path, votes_path)
    baseline = load_baseline(baseline_path)
    rule = make_rule(baseline)
    scores = {
        name: score_candidate(debates, sig, rule, alpha=FAMILY_ALPHA,
                              draws=draws, seed=seed + i, label=name)
        for i, (name, sig) in enumerate(CANDIDATES.items())
    }
    covariate = score_candidate(debates, controversial_item, rule,
                                alpha=FAMILY_ALPHA, draws=draws,
                                seed=seed + len(CANDIDATES),
                                label=COVARIATE)
    promoted = [n for n, s in scores.items() if s["promoted"]]
    return {
        "n_debates": len(debates),
        "candidates": scores,
        "covariate_controversial_item": covariate,
        "promoted_candidates": promoted,
        "verdict": ("PROMOTE: " + ", ".join(promoted)) if promoted
                  else "NULL -- no native candidate clears the coverage, "
                       "statistical, and economic floors jointly",
    }


def print_search(res: dict) -> None:
    print("\n" + "=" * 78)
    print("DILEMMA NATIVE SENSOR SEARCH (point 2, zero-spend, cached data)")
    print("=" * 78)
    print(f"  n_debates={res['n_debates']}")
    order = sorted(res["candidates"],
                   key=lambda n: (res["candidates"][n]["promoted"],
                                  res["candidates"][n]["n_flagged"]),
                   reverse=True)
    for name in order:
        s = res["candidates"][name]
        tag = "PROMOTE" if s["promoted"] else "null"
        fe = s["actuator_flagged"]
        delta = "n/a" if fe["delta"] is None else f"{fe['delta']:+.3f}"
        lo = "n/a" if fe.get("lo") is None else f"{fe['lo']:+.3f}"
        hi = "n/a" if fe.get("hi") is None else f"{fe['hi']:+.3f}"
        print(f"  {name:<28} {tag:<8} n_flagged={s['n_flagged']:<4} "
             f"coverage={s['coverage']:.3f}  actuator_delta={delta} "
             f"[{lo}, {hi}]  econ={s['economic_lift']}")
    c = res["covariate_controversial_item"]
    print(f"\n  [covariate, not a sensor] controversial_item  "
         f"n_flagged={c['n_flagged']}  coverage={c['coverage']:.3f}  "
         f"actuator_delta={c['actuator_flagged']['delta']}")
    print(f"\n  VERDICT: {res['verdict']}")


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("candidate list has exactly the 11 registered native sensors",
          len(CANDIDATES) == 11)
    check("composite candidate is the same function analyze_dilemma_grip exports",
          CANDIDATES["composite"] is flagged)
    check("family alpha is 0.05 / 12 (11 native + covariate)",
          abs(FAMILY_ALPHA - 0.05 / 12) < 1e-12)

    d_either = {"rej_person_a_advocate": True, "rej_person_b_advocate": False}
    d_both = {"rej_person_a_advocate": True, "rej_person_b_advocate": True}
    d_neither = {"rej_person_a_advocate": False, "rej_person_b_advocate": False}
    check("either_advocate_reject fires on one rejecting seat",
          either_advocate_reject(d_either))
    check("both_advocates_reject requires both",
          both_advocates_reject(d_both) and not both_advocates_reject(d_either))
    check("neither seat rejecting flags neither predicate",
          not either_advocate_reject(d_neither)
          and not both_advocates_reject(d_neither))

    d_misloc = {"obj_person_a_advocate": True, "undermined_person_a_advocate": False,
               "obj_neutral_adjudicator": True}
    d_neutral_only = {"obj_person_a_advocate": False, "obj_neutral_adjudicator": True}
    check("neutral_alone excludes a debate that is ALSO mis-localised",
          not neutral_alone(d_misloc))
    check("neutral_alone fires when neutral objects with no mis-localisation",
          neutral_alone(d_neutral_only))

    check("r3_any_reject / r3_reject_ge2 threshold correctly",
          r3_any_reject({"n_r3_reject": 1}) and not r3_reject_ge2({"n_r3_reject": 1})
          and r3_reject_ge2({"n_r3_reject": 2}))

    # Engineered end-to-end population: 20 flagged-by-either_advocate_reject
    # debates where the actuator fixes 10 of them, 60 unflagged debates
    # where it does nothing -- checks the coverage/statistical/economic
    # floors compute and gate correctly, not just the predicates above.
    debates = []
    baseline: dict[str, list[str]] = {}
    for i in range(20):
        item = f"fl{i}"
        gold = "ACTION_A" if i % 2 == 0 else "ACTION_B"
        fixed = i < 10
        debates.append({"item": item, "arm": "neutral", "gold": gold,
                        "s2_ok": not fixed, "syn_ok": True,
                        "sample_idx": 0, "rej_person_a_advocate": True,
                        "rej_person_b_advocate": False,
                        "n_r3_reject": 1, "controversial": False,
                        "any_unresolvable": False, "n_reject": 1,
                        "unanimous_accept": False, "verdict_revised": False})
        # actuator votes correctly (gold) 3-0 on the fixed subset (where S2
        # was wrong), and agrees with the already-correct S2 on the rest.
        baseline[item] = [gold] * 3
    for i in range(60):
        item = f"un{i}"
        debates.append({"item": item, "arm": "neutral", "gold": "ACTION_A",
                        "s2_ok": True, "syn_ok": True, "sample_idx": 0,
                        "rej_person_a_advocate": False,
                        "rej_person_b_advocate": False, "n_r3_reject": 0,
                        "controversial": False, "any_unresolvable": False,
                        "n_reject": 0, "unanimous_accept": True,
                        "verdict_revised": False})
        baseline[item] = ["ACTION_A"] * 3
    rule = make_rule(baseline)
    s = score_candidate(debates, either_advocate_reject, rule,
                        alpha=FAMILY_ALPHA, draws=1000, seed=1,
                        label="test_either")
    check("engineered coverage is exactly 20/80 = 0.25",
          abs(s["coverage"] - 0.25) < 1e-9)
    check("engineered flagged delta is exactly 10/20 = 0.5",
          abs(s["actuator_flagged"]["delta"] - 0.5) < 1e-9)
    check("engineered candidate clears the coverage floor (n=20 >= 15)",
          s["criteria"]["coverage_floor_n_ge_15"])
    check("engineered candidate clears the statistical floor",
          s["criteria"]["statistical_floor"])
    check("engineered candidate clears the economic floor "
         "(0.25 * 0.5 = 0.125 >= 0.01)",
         s["criteria"]["economic_floor_ge_0.01"])
    check("engineered candidate is promoted end to end", s["promoted"])

    # A low-coverage engineered candidate (n=3) must fail the coverage
    # floor even with a perfect point estimate -- the Addendum-11 lesson.
    tiny_sig = lambda d: d["item"] in ("fl0", "fl1", "fl2")
    s_tiny = score_candidate(debates, tiny_sig, rule, alpha=FAMILY_ALPHA,
                             draws=1000, seed=2, label="test_tiny")
    check("a 3-debate candidate fails the coverage floor regardless of "
         "point estimate", not s_tiny["criteria"]["coverage_floor_n_ge_15"]
         and not s_tiny["promoted"])

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag", default="cg_dilemma_stage1")
    ap.add_argument("--baseline-tag", default="cg_dilemma_baseline_stage2")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
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
    res = run_search(rows_path, votes_path, baseline_path,
                     draws=args.n_boot, seed=args.seed)
    print_search(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
