"""
scripts/analyze_claim_audit.py -- three zero-spend audits of the central claim,
run BEFORE any paper is drafted or any further money is spent.

Each check can independently sink or save the headline. All three run on data
already on disk. No generation, no new arms.

CHECK 1 -- STAKE-BLIND SENSOR BAKE-OFF (does embodiment's stake structure do the
selecting, or does raw dissent volume?)
  The composite flag needs to know each seat's ASSIGNED STAKE and the synthesis
  direction to compute `undermined_<seat>`. A plain objection count needs
  neither. If a stake-blind counter matches the composite's composed-system
  accuracy at matched coverage, then the stake-localisation apparatus -- the
  entire embodiment story at the READOUT layer -- is not what selects the
  errors, and the claim has to be relocated to the generator (where opposed
  interests make dissent rare and diagnostic) rather than the sensor.
  Comparator family is deliberately all stake-blind: objection counts,
  non-unanimity, reject counts, and S1->S2 verdict instability.

CHECK 2 -- CONDITIONAL DECOMPOSITION (is the flag a sign-selector or an
error-density concentrator?)
  The claim "dissent marks where an outside check will help" implies the judge
  should REPAIR more and DAMAGE less on flagged debates. Decompose the A3a
  partition into P(S2 wrong), P(judge repairs | S2 wrong), P(judge damages |
  S2 right), per stratum. If the judge's own skill profile is flat across
  strata, the entire gain is base-rate concentration and the paper must say so.

CHECK 3 -- DILEMMAS RAW vs CODABLE FLAG LIFT (is the second corpus's
diagnosticity real, or an abstention artifact?)
  analyze_dilemma_grip.load_debates drops non-codable debates, and every
  registered Dilemmas result uses that codable population. If the flag's
  apparent error-lift on the RAW population comes from flagged debates being
  disproportionately non-codable, then the flag on Dilemmas is detecting
  abstention, not stake-content, and AITA's lift and Dilemmas' lift are not two
  cells of one quantity.

Usage
-----
  python -m scripts.analyze_claim_audit --selftest
  python -m scripts.analyze_claim_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.analyze_actuator_ladder import (
    COMPARATOR_PATH, ROWS_PATH, VOTES_PATH, actuated_ok, correct,
    load_comparator, load_debates, make_rule_vendor,
)
from scripts.analyze_loop_step import flagged
from scripts.run_phase1_quartet import OUT_DIR

ANALYSIS_PATH = OUT_DIR / "claim_audit_analysis.json"
A3A_ACTUATOR = ("claude-haiku-4-5", "standard")   # Addendum 4's certified rung
DEFAULT_DRAWS = 4000
DILEMMA_TAG = "cg_dilemma_stage1"


# ---------------------------------------------------------------------------
# Stake-blind sensor family. NONE of these needs role_stake or the synthesis
# direction; each is a plain count off the rows CSV.
# ---------------------------------------------------------------------------

def n_objectors_ge2(d: dict) -> bool:
    return d.get("n_objectors", 0) >= 2


def n_objectors_ge1(d: dict) -> bool:
    return d.get("n_objectors", 0) >= 1


def nonunanimous(d: dict) -> bool:
    return not d.get("unanimous_accept", True)


def n_reject_ge1(d: dict) -> bool:
    return d.get("n_reject", 0) >= 1


def verdict_revised(d: dict) -> bool:
    return bool(d.get("verdict_revised"))


STAKE_BLIND: dict[str, Callable[[dict], bool]] = {
    "n_objectors_ge2": n_objectors_ge2,
    "n_objectors_ge1": n_objectors_ge1,
    "nonunanimous": nonunanimous,
    "n_reject_ge1": n_reject_ge1,
    "verdict_revised": verdict_revised,
}


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------

def composed_ok(d: dict, sensor: Callable[[dict], bool], rule) -> bool:
    """The deployed system: deliberation verdict unless flagged, actuator if
    flagged (actuated_ok already falls back to S2 on a non-codable actuator
    read, so abstention cannot manufacture a gain)."""
    return actuated_ok(d, rule) if sensor(d) else d["s2_ok"]


def _boot_ci_paired(vals_by_item: dict[str, list[float]], *, draws: int,
                    seed: int, alpha: float = 0.05):
    ids = sorted(vals_by_item)
    rng = random.Random(seed)

    def mean(sel):
        xs = [x for i in sel for x in vals_by_item[i]]
        return sum(xs) / len(xs) if xs else None

    point = mean(ids)
    vals = sorted(v for v in (
        mean([ids[rng.randrange(len(ids))] for _ in ids])
        for _ in range(draws)) if v is not None)
    if not vals:
        return point, None, None
    return (point, vals[int(alpha / 2 * len(vals))],
            vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)])


def paired_sensor_diff(debates: Sequence[dict], sensor_a, sensor_b, rule, *,
                       draws: int, seed: int) -> dict:
    """Composed accuracy of sensor_a minus sensor_b, paired per debate,
    item-clustered bootstrap."""
    by_item: dict[str, list[float]] = defaultdict(list)
    acc_a = acc_b = 0
    for d in debates:
        a = int(composed_ok(d, sensor_a, rule))
        b = int(composed_ok(d, sensor_b, rule))
        acc_a += a
        acc_b += b
        by_item[d["item"]].append(a - b)
    n = len(debates)
    point, lo, hi = _boot_ci_paired(by_item, draws=draws, seed=seed)
    return {"acc_a": acc_a / n if n else None, "acc_b": acc_b / n if n else None,
            "delta": point, "lo": lo, "hi": hi, "n": n}


def trim_sensor(sensor: Callable[[dict], bool], debates: Sequence[dict],
                target_n: int, seed: int) -> Callable[[dict], bool]:
    """A coverage-matched version of `sensor`: fires on a random `target_n`
    subset of the debates it would normally fire on. Keyed by debate identity
    so the returned predicate is deterministic."""
    fires = [(d["item"], d["arm"], d["sample_idx"]) for d in debates if sensor(d)]
    rng = random.Random(seed)
    keep = set(fires if len(fires) <= target_n else rng.sample(fires, target_n))

    def trimmed(d: dict) -> bool:
        return (d["item"], d["arm"], d["sample_idx"]) in keep
    return trimmed


# ---------------------------------------------------------------------------
# CHECK 1
# ---------------------------------------------------------------------------

def check1_bakeoff(debates: Sequence[dict], rule, *, draws: int, seed: int,
                   trim_seeds: int = 200) -> dict:
    n = len(debates)
    n_comp = sum(1 for d in debates if flagged(d))
    base_s2 = sum(d["s2_ok"] for d in debates) / n
    comp_acc = sum(composed_ok(d, flagged, rule) for d in debates) / n

    rivals = {}
    for name, sig in STAKE_BLIND.items():
        n_fire = sum(1 for d in debates if sig(d))
        nat = paired_sensor_diff(debates, flagged, sig, rule, draws=draws,
                                 seed=seed)
        # coverage-matched: trim the rival to the composite's exact n, averaged
        # over many tie-break draws so the comparison does not hinge on one.
        trimmed_accs = []
        for s in range(trim_seeds):
            t = trim_sensor(sig, debates, n_comp, seed + 1000 + s)
            trimmed_accs.append(
                sum(composed_ok(d, t, rule) for d in debates) / n)
        trimmed_accs.sort()
        rivals[name] = {
            "n_fire": n_fire, "coverage": n_fire / n,
            "composed_acc_natural": nat["acc_b"],
            "composite_minus_rival_natural": {
                "delta": nat["delta"], "lo": nat["lo"], "hi": nat["hi"]},
            "composed_acc_trimmed_to_composite_n": {
                "mean": sum(trimmed_accs) / len(trimmed_accs),
                "p05": trimmed_accs[int(0.05 * len(trimmed_accs))],
                "p95": trimmed_accs[min(int(0.95 * len(trimmed_accs)),
                                        len(trimmed_accs) - 1)],
                "n_seeds": trim_seeds},
        }

    # random routing at matched coverage: the floor of the family
    rand_accs = []
    for s in range(trim_seeds):
        t = trim_sensor(lambda d: True, debates, n_comp, seed + 5000 + s)
        rand_accs.append(sum(composed_ok(d, t, rule) for d in debates) / n)
    rand_accs.sort()

    return {
        "n_debates": n, "n_composite_flagged": n_comp,
        "composite_coverage": n_comp / n,
        "acc_deliberation_only": base_s2,
        "acc_composite_routed": comp_acc,
        "rivals": rivals,
        "random_routing_matched_coverage": {
            "mean": sum(rand_accs) / len(rand_accs),
            "p05": rand_accs[int(0.05 * len(rand_accs))],
            "max": rand_accs[-1], "n_seeds": trim_seeds},
    }


# ---------------------------------------------------------------------------
# CHECK 2
# ---------------------------------------------------------------------------

def check2_decomposition(debates: Sequence[dict], rule, *, draws: int,
                         seed: int) -> dict:
    out = {}
    for label, sel in (("flagged", lambda d: flagged(d)),
                       ("unflagged", lambda d: not flagged(d))):
        sub = [d for d in debates if sel(d)]
        n = len(sub)
        wrong = [d for d in sub if not d["s2_ok"]]
        right = [d for d in sub if d["s2_ok"]]
        # the judge only gets a chance to repair/damage when it emits a codable
        # verdict; otherwise actuated_ok falls back to S2 by construction.
        def spoke(d):
            v = rule(d)
            return v is not None and correct(v, d["gold"]) is not None
        wrong_spoke = [d for d in wrong if spoke(d)]
        right_spoke = [d for d in right if spoke(d)]
        out[label] = {
            "n": n,
            "p_s2_wrong": len(wrong) / n if n else None,
            "judge_emitted_rate": (sum(1 for d in sub if spoke(d)) / n
                                   if n else None),
            "p_repair_given_s2_wrong": (
                sum(1 for d in wrong_spoke if actuated_ok(d, rule))
                / len(wrong_spoke) if wrong_spoke else None),
            "n_wrong_judge_spoke": len(wrong_spoke),
            "p_damage_given_s2_right": (
                sum(1 for d in right_spoke if not actuated_ok(d, rule))
                / len(right_spoke) if right_spoke else None),
            "n_right_judge_spoke": len(right_spoke),
        }
    f, u = out["flagged"], out["unflagged"]
    out["gaps"] = {
        "error_density_ratio": (f["p_s2_wrong"] / u["p_s2_wrong"]
                                if u["p_s2_wrong"] else None),
        "repair_gap": (f["p_repair_given_s2_wrong"] - u["p_repair_given_s2_wrong"]
                       if None not in (f["p_repair_given_s2_wrong"],
                                       u["p_repair_given_s2_wrong"]) else None),
        "damage_gap": (f["p_damage_given_s2_right"] - u["p_damage_given_s2_right"]
                       if None not in (f["p_damage_given_s2_right"],
                                       u["p_damage_given_s2_right"]) else None),
    }
    return out


# ---------------------------------------------------------------------------
# CHECK 3
# ---------------------------------------------------------------------------

def load_dilemma_debates_raw(rows_path: Path, votes_path: Path) -> list[dict]:
    """Every Dilemmas debate INCLUDING non-codable ones. Mirrors
    analyze_dilemma_grip.load_debates exactly except for the final filter."""
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
            }
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            d = debates.get((v["item_id"], v["arm"], v["sample_idx"]))
            if d is None:
                continue
            seat = v["role_id"]
            d[f"obj_{seat}"] = v["objected_r3"] == "1"
            und = v.get("stake_undermined")
            if seat in ("person_a_advocate", "person_b_advocate") and und not in (None, ""):
                d[f"undermined_{seat}"] = bool(int(und))
    return list(debates.values())


def _dilemma_flagged(d: dict) -> bool:
    misloc = any(d.get(f"obj_{s}") and not d.get(f"undermined_{s}", True)
                 for s in ("person_a_advocate", "person_b_advocate"))
    return bool(d.get("obj_neutral_adjudicator")) or misloc


def check3_dilemma_lift(rows_path: Path, votes_path: Path, *, draws: int,
                        seed: int) -> dict:
    raw = load_dilemma_debates_raw(rows_path, votes_path)
    codable = [d for d in raw if d["syn_ok"] is not None and d["s2_ok"] is not None]

    def lift(pop, score):
        fl = [d for d in pop if _dilemma_flagged(d)]
        un = [d for d in pop if not _dilemma_flagged(d)]
        if not fl or not un:
            return None
        af = sum(score(d) for d in fl) / len(fl)
        au = sum(score(d) for d in un) / len(un)
        by_item: dict[str, list[float]] = defaultdict(list)
        for d in pop:
            by_item[d["item"]].append(0.0)   # placeholder, CI computed below
        return {"n_flagged": len(fl), "n_unflagged": len(un),
                "err_flagged": 1 - af, "err_unflagged": 1 - au,
                "error_lift": au - af}

    # raw: a non-codable debate is scored as an ERROR (the inflating convention)
    raw_lift = lift(raw, lambda d: bool(d["s2_ok"]))
    cod_lift = lift(codable, lambda d: bool(d["s2_ok"]))

    n_raw_fl = sum(1 for d in raw if _dilemma_flagged(d))
    noncod_fl = sum(1 for d in raw
                    if _dilemma_flagged(d)
                    and (d["syn_ok"] is None or d["s2_ok"] is None))
    n_raw_un = len(raw) - n_raw_fl
    noncod_un = sum(1 for d in raw
                    if not _dilemma_flagged(d)
                    and (d["syn_ok"] is None or d["s2_ok"] is None))
    return {
        "n_raw": len(raw), "n_codable": len(codable),
        "raw_population": raw_lift,
        "codable_population": cod_lift,
        "p_noncodable_given_flagged": noncod_fl / n_raw_fl if n_raw_fl else None,
        "p_noncodable_given_unflagged": noncod_un / n_raw_un if n_raw_un else None,
        "n_raw_flagged": n_raw_fl, "n_noncodable_flagged": noncod_fl,
    }


# ---------------------------------------------------------------------------

def print_audit(res: dict) -> None:
    c1, c2, c3 = res["check1_bakeoff"], res["check2_decomposition"], res["check3_dilemma"]
    print("\n" + "=" * 78)
    print("CHECK 1 -- STAKE-BLIND SENSOR BAKE-OFF (AITA, composed-system accuracy)")
    print("=" * 78)
    print(f"  n={c1['n_debates']}  composite fires {c1['n_composite_flagged']} "
          f"({c1['composite_coverage']:.3f})")
    print(f"  deliberation only        {c1['acc_deliberation_only']:.4f}")
    print(f"  composite routed         {c1['acc_composite_routed']:.4f}   <-- stake-AWARE")
    r = c1["random_routing_matched_coverage"]
    print(f"  random @ matched cov     {r['mean']:.4f} (max over seeds {r['max']:.4f})")
    print("\n  stake-BLIND rivals:")
    for name, v in c1["rivals"].items():
        d = v["composite_minus_rival_natural"]
        t = v["composed_acc_trimmed_to_composite_n"]
        ci = (f"[{d['lo']:+.4f}, {d['hi']:+.4f}]" if d["lo"] is not None else "n/a")
        print(f"    {name:<18} cov {v['coverage']:.3f}  composed {v['composed_acc_natural']:.4f}"
              f"  trimmed {t['mean']:.4f}  composite-minus-rival {d['delta']:+.4f} {ci}")

    print("\n" + "=" * 78)
    print("CHECK 2 -- CONDITIONAL DECOMPOSITION (is it sign-selection or base rate?)")
    print("=" * 78)
    for k in ("flagged", "unflagged"):
        s = c2[k]
        print(f"  {k:<10} n={s['n']:<5} P(S2 wrong)={s['p_s2_wrong']:.3f}  "
              f"judge emitted {s['judge_emitted_rate']:.3f}")
        print(f"             P(repair | S2 wrong)={s['p_repair_given_s2_wrong']:.3f} "
              f"(n={s['n_wrong_judge_spoke']})   "
              f"P(damage | S2 right)={s['p_damage_given_s2_right']:.3f} "
              f"(n={s['n_right_judge_spoke']})")
    g = c2["gaps"]
    print(f"  error-density ratio {g['error_density_ratio']:.2f}x   "
          f"repair gap {g['repair_gap']:+.3f}   damage gap {g['damage_gap']:+.3f}")

    print("\n" + "=" * 78)
    print("CHECK 3 -- DILEMMAS RAW vs CODABLE FLAG LIFT (real, or abstention?)")
    print("=" * 78)
    print(f"  n_raw={c3['n_raw']}  n_codable={c3['n_codable']}")
    for k in ("raw_population", "codable_population"):
        v = c3[k]
        if v:
            print(f"  {k:<20} n_flagged={v['n_flagged']:<4} "
                  f"err_flagged={v['err_flagged']:.3f} "
                  f"err_unflagged={v['err_unflagged']:.3f} "
                  f"error_lift={v['error_lift']:+.3f}")
    print(f"  P(non-codable | flagged)   {c3['p_noncodable_given_flagged']:.3f} "
          f"({c3['n_noncodable_flagged']}/{c3['n_raw_flagged']})")
    print(f"  P(non-codable | unflagged) {c3['p_noncodable_given_unflagged']:.3f}")


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("every stake-blind sensor is computable without role_stake or "
          "synthesis direction",
          all(sig({"n_objectors": 2, "n_reject": 1, "unanimous_accept": False,
                   "verdict_revised": True}) in (True, False)
              for sig in STAKE_BLIND.values()))

    d_hi = {"n_objectors": 2, "n_reject": 0, "unanimous_accept": True,
            "verdict_revised": False}
    check("n_objectors_ge2 fires at 2 and not at 1",
          n_objectors_ge2(d_hi) and not n_objectors_ge2({"n_objectors": 1}))
    check("nonunanimous reads the unanimous_accept flag",
          nonunanimous({"unanimous_accept": False})
          and not nonunanimous({"unanimous_accept": True}))

    # composed_ok: flagged -> actuator, unflagged -> S2, and fallback to S2
    rule_ok = lambda d: "YTA"
    rule_none = lambda d: None
    d = {"item": "i", "arm": "a", "sample_idx": 0, "gold": "YTA", "s2_ok": False}
    check("composed_ok routes a flagged debate to the actuator",
          composed_ok(d, lambda x: True, rule_ok) is True)
    check("composed_ok leaves an unflagged debate on S2",
          composed_ok(d, lambda x: False, rule_ok) is False)
    check("composed_ok falls back to S2 when the actuator is silent",
          composed_ok(d, lambda x: True, rule_none) is False)

    # trim_sensor reduces coverage to exactly the target
    pop = [{"item": f"i{i}", "arm": "a", "sample_idx": 0} for i in range(50)]
    t = trim_sensor(lambda x: True, pop, 10, seed=1)
    check("trim_sensor fires on exactly the target count",
          sum(1 for d in pop if t(d)) == 10)
    t_small = trim_sensor(lambda x: x["item"] in ("i0", "i1"), pop, 10, seed=1)
    check("trim_sensor is a no-op when the sensor already fires below target",
          sum(1 for d in pop if t_small(d)) == 2)

    # engineered decomposition: flagged stratum is error-dense, judge is
    # perfectly repairing there and neutral elsewhere.
    debates = []
    for i in range(20):
        debates.append({"item": f"f{i}", "arm": "a", "sample_idx": 0,
                        "gold": "YTA", "s2_ok": i >= 10, "n_objectors": 2,
                        "n_reject": 0, "unanimous_accept": True,
                        "verdict_revised": False,
                        "obj_neutral_adjudicator": True})
    for i in range(20):
        debates.append({"item": f"u{i}", "arm": "a", "sample_idx": 0,
                        "gold": "YTA", "s2_ok": True, "n_objectors": 0,
                        "n_reject": 0, "unanimous_accept": True,
                        "verdict_revised": False,
                        "obj_neutral_adjudicator": False})
    dec = check2_decomposition(debates, lambda d: "YTA", draws=100, seed=1)
    check("engineered P(S2 wrong | flagged) is 0.5",
          abs(dec["flagged"]["p_s2_wrong"] - 0.5) < 1e-9)
    check("engineered P(S2 wrong | unflagged) is 0.0",
          dec["unflagged"]["p_s2_wrong"] == 0.0)
    check("engineered P(repair | S2 wrong, flagged) is 1.0 (judge always right)",
          dec["flagged"]["p_repair_given_s2_wrong"] == 1.0)
    check("engineered P(damage | S2 right) is 0.0",
          dec["flagged"]["p_damage_given_s2_right"] == 0.0)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--trim-seeds", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    debates = load_debates(ROWS_PATH, VOTES_PATH)
    comp = load_comparator(COMPARATOR_PATH)
    rule = make_rule_vendor(comp, *A3A_ACTUATOR)

    res = {
        "check1_bakeoff": check1_bakeoff(debates, rule, draws=args.n_boot,
                                         seed=args.seed,
                                         trim_seeds=args.trim_seeds),
        "check2_decomposition": check2_decomposition(debates, rule,
                                                     draws=args.n_boot,
                                                     seed=args.seed + 1),
        "check3_dilemma": check3_dilemma_lift(
            OUT_DIR / f"{DILEMMA_TAG}_rows.csv",
            OUT_DIR / f"{DILEMMA_TAG}_votes.csv",
            draws=args.n_boot, seed=args.seed + 2),
    }
    print_audit(res)
    args.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
