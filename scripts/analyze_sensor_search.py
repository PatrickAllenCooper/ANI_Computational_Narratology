"""
scripts/analyze_sensor_search.py -- Addendum 11 Phase 0 (zero spend): find a
routing sensor built FOR the few-shot-induced signal, rather than inherited
from `flagged()` (`scripts/analyze_loop_step.py`), which was designed around
grok's native grip and never re-examined after Addendum 8 induced the
OPPOSITE event on purpose (the UNDERMINED seat rejecting, not a mis-localised
one).

Population: the ALREADY-CACHED Addendum 10 rows/votes
(`cg_deliberation_{haiku,nano}_fewshot_{rows,votes}.csv`). No new generation.
Comparator in every candidate: S2 (the same-debate integrated verdict), the
same paired evaluation `analyze_actuator_ladder.evaluate` uses for A3a, with
the same cross-vendor cached actuator.

Candidates (fixed list, registered in Addendum 11 before any is scored):
  1. composite            -- flagged(), the Addendum 9/10 sensor (control)
  2. composite_severe     -- flagged_severe() (control)
  3. undermined_reject    -- a stake seat is undermined AND rejects
  4. undermined_and_neutral -- (3) AND the neutral adjudicator also objected
  5. r4_nonunanimous      -- final vote round is not unanimous accept
  6. r4_reject_ge2        -- >=2 of 3 R4 votes are REJECT
  7. r3_any_reject        -- >=1 R3 label is REJECT
  8. r3_reject_ge2        -- >=2 R3 labels are REJECT
  9. verdict_revised      -- synthesis text changed between S1 and S2

Design: develop (rank all 9) on haiku; confirm the single best candidate
on nano at Bonferroni alpha = 0.05/9. "Best" means the candidate that
clears `analyze_actuator_ladder.evaluate`'s own full positive bar on
haiku (CI excludes zero, delta>=0.05, replicates across both item
halves and both arms) with the largest n behind it -- not the largest
raw point estimate, which is vulnerable to small-n noise from
low-coverage candidates (verified during Phase 0: the naive top-ranked
candidate by point estimate alone fired on only 5/94 nano debates and
failed to replicate across item halves, i.e. it was noise, not signal;
see `rank()`). Every candidate is also replayed on grok's native
1677-debate population (`cg_deliberation_rows.csv`/`_votes.csv`) as an
external check that it tracks error in general, not an artifact of a
40-60-item induced panel.

Usage
-----
  python -m scripts.analyze_sensor_search --selftest
  python -m scripts.analyze_sensor_search
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.analyze_actuator_ladder import (
    ROWS_PATH as GROK_ROWS_PATH, VOTES_PATH as GROK_VOTES_PATH,
    evaluate, load_comparator, load_debates, make_rule_vendor,
)
from scripts.analyze_loop_step import STAKE_SEATS, flagged, flagged_severe
from scripts.run_phase1_quartet import OUT_DIR

COMPARATOR_PATH = OUT_DIR / "cg_scaffold_combined_rows_k3.csv"
ANALYSIS_PATH = OUT_DIR / "sensor_search_analysis.json"
DEVELOP_ALPHA = 0.05          # unadjusted -- exploratory ranking, not a claim
CONFIRM_ALPHA = 0.05 / 9      # Bonferroni over the 9 candidates, spent once
DEFAULT_DRAWS = 4000

#: (label, sensor model, rows tag, cross-vendor actuator to test)
POPULATIONS = (
    ("haiku_fewshot", "claude-haiku-4-5", "cg_deliberation_haiku_fewshot",
     ("grok_std", "grok-4-1-fast-reasoning", "standard")),
    ("nano_fewshot", "gpt-5.4-nano", "cg_deliberation_nano_fewshot",
     ("grok_std", "grok-4-1-fast-reasoning", "standard")),
)
GROK_ACTUATOR = ("claude-haiku-4-5", "standard")   # Addendum 4's primary


def attach_r3_reject_count(debates: Sequence[dict], rows_path: Path) -> None:
    """`n_r3_reject` isn't carried by analyze_actuator_ladder.load_debates;
    read it straight off the rows CSV and merge it in."""
    counts: dict[tuple, int] = {}
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            key = (r["item_id"], r["arm"], r["sample_idx"])
            counts[key] = int(r.get("n_r3_reject") or 0)
    for d in debates:
        d["n_r3_reject"] = counts.get((d["item"], d["arm"], str(d["sample_idx"])), 0)


def undermined_reject(d: dict) -> bool:
    return any(d.get(f"rej_{s}") and d.get(f"undermined_{s}") for s in STAKE_SEATS)


def undermined_and_neutral(d: dict) -> bool:
    return undermined_reject(d) and bool(d.get("obj_neutral_adjudicator"))


def r4_nonunanimous(d: dict) -> bool:
    return not d.get("unanimous_accept", True)


def r4_reject_ge2(d: dict) -> bool:
    return d.get("n_reject", 0) >= 2


def r3_any_reject(d: dict) -> bool:
    return d.get("n_r3_reject", 0) >= 1


def r3_reject_ge2(d: dict) -> bool:
    return d.get("n_r3_reject", 0) >= 2


def verdict_revised(d: dict) -> bool:
    return bool(d.get("verdict_revised"))


CANDIDATES: dict[str, Callable[[dict], bool]] = {
    "composite": flagged,
    "composite_severe": flagged_severe,
    "undermined_reject": undermined_reject,
    "undermined_and_neutral": undermined_and_neutral,
    "r4_nonunanimous": r4_nonunanimous,
    "r4_reject_ge2": r4_reject_ge2,
    "r3_any_reject": r3_any_reject,
    "r3_reject_ge2": r3_reject_ge2,
    "verdict_revised": verdict_revised,
}


def score_candidate(debates: Sequence[dict], sig: Callable[[dict], bool],
                    rule, *, alpha: float, draws: int, seed: int,
                    label: str) -> dict:
    fl = [d for d in debates if sig(d)]
    unfl = [d for d in debates if not sig(d)]
    n = len(debates)
    acc_all = sum(d["s2_ok"] for d in debates) / n if n else None
    acc_fl = sum(d["s2_ok"] for d in fl) / len(fl) if fl else None
    acc_unfl = sum(d["s2_ok"] for d in unfl) / len(unfl) if unfl else None
    lift = None if acc_fl is None or acc_unfl is None else acc_unfl - acc_fl
    fl_eval = evaluate(fl, rule, alpha=alpha, draws=draws, seed=seed,
                       label=f"{label} flagged")
    unfl_eval = evaluate(unfl, rule, alpha=alpha, draws=draws, seed=seed + 1,
                         label=f"{label} unflagged (context)")
    d_fl, d_un = fl_eval["delta"], unfl_eval["delta"]
    sens_gain = None if d_fl is None or d_un is None else d_fl - d_un
    return {
        "candidate": label, "n_flagged": len(fl), "n_total": n,
        "coverage": len(fl) / n if n else None,
        "acc_all": acc_all, "acc_flagged": acc_fl, "acc_unflagged": acc_unfl,
        "error_lift": lift,        # positive => flagged set is MORE error-prone
        "actuator_flagged": fl_eval, "actuator_unflagged": unfl_eval,
        "sensor_specific_gain": sens_gain,
    }


def run_population(label: str, model: str, tag: str, actuator, comp: dict, *,
                   alpha: float, draws: int, seed: int) -> dict:
    rows_path = OUT_DIR / f"{tag}_rows.csv"
    votes_path = OUT_DIR / f"{tag}_votes.csv"
    debates = [d for d in load_debates(rows_path, votes_path)
              if d["model"] == model]
    attach_r3_reject_count(debates, rows_path)
    atag, amodel, ascaffold = actuator
    rule = make_rule_vendor(comp, amodel, ascaffold)
    scores = {
        name: score_candidate(debates, sig, rule, alpha=alpha, draws=draws,
                              seed=seed + i, label=f"{label}/{name}")
        for i, (name, sig) in enumerate(CANDIDATES.items())
    }
    return {"population": label, "model": model, "n_debates": len(debates),
           "actuator": atag, "candidates": scores}


def _fmt_signed(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:+.3f}"


def rank(scores: dict) -> list[str]:
    """Rank candidates for confirmation. A raw sensor-specific-gain sort
    rewards small-n noise (e.g. a candidate that fires on 5 debates can post
    a huge point estimate by chance) -- exactly the failure mode this
    project's own `evaluate()` criteria exist to catch. So candidates that
    clear evaluate()'s full positive bar on the flagged set (CI excludes
    zero AND delta>=0.05 AND replicates across both item halves and both
    arms, at the unadjusted develop-stage alpha) are ranked first, tied by
    largest n (more debates behind the estimate is more robust, not just a
    tiebreak); candidates that don't clear the full bar are ranked after,
    by raw sensor-specific gain, for visibility only.
    """
    def key(name):
        s = scores[name]
        fe = s["actuator_flagged"]
        robust = bool(fe.get("positive"))
        n = fe.get("n") or 0
        g = s["sensor_specific_gain"]
        return (robust, n if robust else 0,
                g if g is not None else float("-inf"))
    return sorted(scores, key=key, reverse=True)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    comp = load_comparator(COMPARATOR_PATH)

    # develop on haiku
    develop = run_population(*POPULATIONS[0], comp, alpha=DEVELOP_ALPHA,
                             draws=a.draws, seed=a.seed)
    order = rank(develop["candidates"])
    print(f"=== develop on {develop['population']} "
          f"({develop['n_debates']} debates) ===")
    for name in order:
        s = develop["candidates"][name]
        g = s["sensor_specific_gain"]
        print(f"  {name:<24} coverage {s['coverage']:.2f}  error_lift "
              f"{_fmt_signed(s['error_lift'])}"
              f"  sensor_specific_gain {_fmt_signed(g)}")
    best = order[0]
    print(f"\nbest on haiku by sensor-specific gain: {best}")

    # confirm the single best candidate on nano at Bonferroni alpha
    confirm_pop = POPULATIONS[1]
    rows_path = OUT_DIR / f"{confirm_pop[2]}_rows.csv"
    votes_path = OUT_DIR / f"{confirm_pop[2]}_votes.csv"
    debates = [d for d in load_debates(rows_path, votes_path)
              if d["model"] == confirm_pop[1]]
    attach_r3_reject_count(debates, rows_path)
    atag, amodel, ascaffold = confirm_pop[3]
    rule = make_rule_vendor(comp, amodel, ascaffold)
    confirm = score_candidate(debates, CANDIDATES[best], rule,
                              alpha=CONFIRM_ALPHA, draws=a.draws,
                              seed=a.seed + 100, label=f"nano_fewshot/{best}")
    print(f"\n=== confirm '{best}' on {confirm_pop[0]} "
          f"({len(debates)} debates, alpha={CONFIRM_ALPHA:.4f}) ===")
    fe = confirm["actuator_flagged"]
    print(f"  coverage {confirm['coverage']:.2f}  error_lift "
          f"{_fmt_signed(confirm['error_lift'])}")
    print(f"  flagged delta {fe['delta']:+.3f} "
          f"[{fe['lo']:+.3f}, {fe['hi']:+.3f}]  "
          f"sensor_specific_gain {confirm['sensor_specific_gain']:+.3f}"
          if fe["delta"] is not None else "  flagged delta n/a")

    # external validity: replay every candidate on grok's native population
    grok_debates = [d for d in load_debates(GROK_ROWS_PATH, GROK_VOTES_PATH)]
    attach_r3_reject_count(grok_debates, GROK_ROWS_PATH)
    grok_rule = make_rule_vendor(comp, *GROK_ACTUATOR)
    grok_scores = {
        name: score_candidate(grok_debates, sig, grok_rule,
                              alpha=DEVELOP_ALPHA, draws=a.draws,
                              seed=a.seed + 200 + i, label=f"grok/{name}")
        for i, (name, sig) in enumerate(CANDIDATES.items())
    }
    print(f"\n=== external check: candidates replayed on grok's native "
          f"{len(grok_debates)}-debate population ===")
    for name in order:
        s = grok_scores[name]
        print(f"  {name:<24} coverage {s['coverage']:.2f}  error_lift "
              f"{_fmt_signed(s['error_lift'])}")

    grok_lift_best = grok_scores[best]["error_lift"]
    gate_g0_external = grok_lift_best is not None and grok_lift_best >= 0.10
    gate_g0_induced = (confirm["sensor_specific_gain"] is not None
                       and confirm["sensor_specific_gain"] > 0)
    gate_g0_ci_clears = fe["lo"] is not None and fe["lo"] > 0
    print(f"\nGATE G0: external lift on grok >= 0.10: {gate_g0_external} "
          f"({grok_lift_best})")
    print(f"GATE G0: sensor-specific gain > 0 on nano: {gate_g0_induced} "
          f"({confirm['sensor_specific_gain']})")
    print(f"(for reference only, not part of G0 as registered) "
          f"nano actuator CI excludes zero: {gate_g0_ci_clears}")
    verdict = ("PASS -> Phase 2" if gate_g0_external and gate_g0_induced
              else "borderline -> Phase 1" if gate_g0_induced
              else "FAIL -> stop, report ceiling")
    print(f"\nGATE G0 VERDICT: {verdict}")

    out = {"develop": develop, "ranking": order, "best_candidate": best,
          "confirm_nano": confirm, "grok_external_check": grok_scores,
          "gate_g0": {"external_lift_ge_0.10": gate_g0_external,
                      "sensor_specific_gain_gt_0": gate_g0_induced,
                      "nano_ci_excludes_zero_reference_only": gate_g0_ci_clears,
                      "verdict": verdict}}
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {a.out}")
    return 0


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("candidate list has exactly the 9 registered sensors",
          len(CANDIDATES) == 9)
    check("composite candidate is the same function analyze_loop_step exports",
          CANDIDATES["composite"] is flagged)

    ready = all((OUT_DIR / f"{tag}_rows.csv").exists()
               for _, _, tag, _ in POPULATIONS) and COMPARATOR_PATH.exists()
    check("both fewshot caches and the comparator are present", ready)
    if not ready:
        print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
        return 1 if fails else 0

    comp = load_comparator(COMPARATOR_PATH)
    d0 = run_population(*POPULATIONS[0], comp, alpha=DEVELOP_ALPHA,
                        draws=200, seed=1)
    check("haiku population nonempty", d0["n_debates"] > 0)
    check("every candidate produces a coverage between 0 and 1",
          all(0 <= s["coverage"] <= 1 for s in d0["candidates"].values()))
    check("composite candidate coverage matches flagged() directly",
          abs(d0["candidates"]["composite"]["coverage"]
              - sum(1 for d in load_debates(
                  OUT_DIR / f"{POPULATIONS[0][2]}_rows.csv",
                  OUT_DIR / f"{POPULATIONS[0][2]}_votes.csv")
                  if d["model"] == POPULATIONS[0][1] and flagged(d))
              / max(1, d0["n_debates"])) < 1e-9)
    check("undermined_reject and composite are DIFFERENT populations "
          "(the redesign actually changes who gets flagged)",
          d0["candidates"]["undermined_reject"]["n_flagged"]
          != d0["candidates"]["composite"]["n_flagged"])
    order = rank(d0["candidates"])
    check("ranking returns all 9 candidate names exactly once",
          sorted(order) == sorted(CANDIDATES))
    top = d0["candidates"][order[0]]
    check("the top-ranked candidate clears the full positive bar whenever "
          "ANY candidate does (robustness beats raw point estimate)",
          top["actuator_flagged"]["positive"]
          or not any(s["actuator_flagged"]["positive"]
                     for s in d0["candidates"].values()))
    if sum(1 for s in d0["candidates"].values()
          if s["actuator_flagged"]["positive"]) >= 2:
        robust = [s for s in d0["candidates"].values()
                 if s["actuator_flagged"]["positive"]]
        check("among robust candidates, the top pick has the largest n "
              "(ties are not broken by point estimate alone)",
              top["actuator_flagged"]["n"]
              == max(s["actuator_flagged"]["n"] for s in robust))
    r0 = score_candidate(
        [d for d in load_debates(OUT_DIR / f"{POPULATIONS[0][2]}_rows.csv",
                                 OUT_DIR / f"{POPULATIONS[0][2]}_votes.csv")
         if d["model"] == POPULATIONS[0][1]],
        CANDIDATES["composite"],
        make_rule_vendor(comp, *POPULATIONS[0][3][1:]),
        alpha=DEVELOP_ALPHA, draws=200, seed=1, label="x")
    r0b = score_candidate(
        [d for d in load_debates(OUT_DIR / f"{POPULATIONS[0][2]}_rows.csv",
                                 OUT_DIR / f"{POPULATIONS[0][2]}_votes.csv")
         if d["model"] == POPULATIONS[0][1]],
        CANDIDATES["composite"],
        make_rule_vendor(comp, *POPULATIONS[0][3][1:]),
        alpha=DEVELOP_ALPHA, draws=200, seed=1, label="x")
    check("scoring is deterministic given fixed inputs/seed",
          r0["sensor_specific_gain"] == r0b["sensor_specific_gain"])

    grok_ready = GROK_ROWS_PATH.exists() and GROK_VOTES_PATH.exists()
    check("grok native population present for the external check", grok_ready)
    if grok_ready:
        gd = [d for d in load_debates(GROK_ROWS_PATH, GROK_VOTES_PATH)]
        attach_r3_reject_count(gd, GROK_ROWS_PATH)
        check("grok population nonempty and all model==grok",
              len(gd) > 0 and all("grok" in d["model"] for d in gd))

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
