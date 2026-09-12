"""
scripts/analyze_unembodied_ablation.py -- Addendum 16.8: the readout for the
stake-free deliberation control (scripts/run_crowdgold_unembodied.py).

WHY THE OBVIOUS READOUT IS NOT ENOUGH

The registered comparison was "does the stake-blind counter `n_objectors >= 2`
stay rare (0.196) and diagnostic (+0.320) without assigned interests?". A raw
fire rate cannot answer it, because the fire rate is a compound of two things:

  (i)  how often an individual seat objects at all, and
  (ii) how those objections DISTRIBUTE across the three seats.

Measured on the embodied grok panel (1,680 debates, 5,040 seat-votes): the
per-seat objection rate is 0.398, so under independence P(>=2 objectors) would
be 0.349 -- nearly double the observed 0.196 -- and P(exactly 1) would be 0.433
against an observed 0.792. The embodied protocol's rarity is therefore ENTIRELY
a negative-dependence effect: when there is an objection it is concentrated in
one seat, far more often than chance. That concentration is what "grip" looks
like at the debate level, and it is the thing the embodiment claim is about.

So a fire rate that moved could mean "seats stopped localising" OR "seats
objected more/less overall", which are different findings. This module reports
both channels separately and makes the localisation excess

    LOC = P(exactly 1 objector) - 3p(1-p)^2      [p = per-seat objection rate]

the primary statistic, since it is automatically controlled for the overall
objection rate. Embodied baseline: LOC = 0.792 - 0.433 = +0.359.

POPULATION DISCIPLINE

Addendum 16.3 is this programme's own proof that the raw-vs-codable choice can
flip the sign of an error lift (Dilemmas: +0.314 raw, -0.023 codable, because
P(non-codable | flagged) was 0.727). Stake-free readers have less reason to
commit to a verdict, so an elevated non-codable rate is a live risk here. Both
populations are therefore always reported side by side, together with
P(non-codable | fired), and the codable population is primary.

GATES, pre-declared (see the Addendum 16.8 registration):
  - R3 parse gate: mean n_r3_unparsed per debate must not exceed the embodied
    baseline by more than 0.05 seats, else the fire rate moved for a parsing
    reason and no comparison is read.
  - Minimum-flagged gate: the error lift is only read when the arm has >= 60
    fired debates; below that only the fire rate and LOC are reported.

Usage
-----
  python -m scripts.analyze_unembodied_ablation --selftest
  python -m scripts.analyze_unembodied_ablation --tag cg_deliberation_unembodied \
      --baseline-tag cg_deliberation
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.analyze_crowdgold_sdt import code_response
from scripts.run_phase1_quartet import OUT_DIR

ANALYSIS_PATH = OUT_DIR / "unembodied_ablation_analysis.json"
DEFAULT_DRAWS = 4000
MIN_FLAGGED_FOR_LIFT = 60
PARSE_GATE_SLACK = 0.05
COMMITTAL = ("YTA", "NTA", "ESH", "NAH")


def verdict_correct(verdict: str, gold: str):
    """AITA is a FIVE-token instrument (NTA/YTA/ESH/NAH/UNRESOLVED) that every
    registered result in this programme scores by collapsing to a published
    at-fault bit via analyze_crowdgold_sdt.code_response. Scoring it by raw
    string equality instead treats ESH and NAH as wrong whenever gold is
    YTA/NTA and gives 0.7173 where the registered convention gives 0.8427 --
    an earlier draft of this module did exactly that and every lift it printed
    was wrong. Returns None when the verdict is not codable."""
    c = code_response(verdict, "published")
    if c is None:
        return None
    return c == (1 if gold == "YTA" else 0)


def load_rows(rows_path: Path, model: Optional[str] = None,
              keep_items: Optional[set] = None) -> list[dict]:
    """One record per debate. Deliberately does NOT use
    analyze_loop_step.flagged / analyze_actuator_ladder.load_debates: both key
    on STAKE_SEATS = (writer_advocate, counterparty) and on undermined_<seat>,
    none of which exist on control rows -- flagged() would silently return
    False for every debate and report a 0.000 fire rate. The statistics here
    are role-agnostic by construction so both arms go through identical code.
    """
    out = []
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            if model and r.get("model") != model:
                continue
            if keep_items is not None and r["item_id"] not in keep_items:
                continue
            syn, ver = r.get("synthesis_verdict", ""), r.get("verdict", "")
            out.append({
                "item": r["item_id"], "arm": r["arm"],
                "sample_idx": int(r["sample_idx"]),
                "gold": r["gold_verdict"],
                "n_objectors": int(r.get("n_objectors") or 0),
                "n_r3_unparsed": int(r.get("n_r3_unparsed") or 0),
                "codable": (code_response(syn, "published") is not None
                            and code_response(ver, "published") is not None),
                # correctness of the deliberation's own final verdict, scored
                # by the registered five-token collapse, NOT string equality
                "s2_ok": verdict_correct(ver, r["gold_verdict"]),
            })
    return out


def per_seat_objection_rates(votes_path: Path, model: Optional[str] = None,
                             keep_items: Optional[set] = None) -> dict:
    """Per-seat objection MARGINALS plus the pooled rate. The marginals are the
    point: Addendum 16.8 records that using a single pooled p to build the
    independence baseline scored seat heterogeneity as dependence and inflated
    the embodied localisation excess from +0.291 to +0.359. The seats' true
    rates are 0.430 / 0.690 / 0.073 -- nowhere near exchangeable."""
    per: dict = defaultdict(lambda: [0, 0])
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            if model and v.get("model") != model:
                continue
            if keep_items is not None and v["item_id"] not in keep_items:
                continue
            s_ = v["role_id"]
            per[s_][0] += 1 if v.get("objected_r3") == "1" else 0
            per[s_][1] += 1
    marg = {k: a / b for k, (a, b) in per.items() if b}
    tot_o = sum(a for a, _ in per.values())
    tot_n = sum(b for _, b in per.values())
    return {"per_seat": marg, "pooled": tot_o / tot_n if tot_n else 0.0,
            "n_votes": tot_n}


def poisson_binomial(ps: Sequence[float]) -> list[float]:
    """P(exactly k successes) for independent Bernoulli trials with DIFFERENT
    probabilities. The correct independence null for heterogeneous seats."""
    dist = [1.0]
    for q in ps:
        nd = [0.0] * (len(dist) + 1)
        for i, w in enumerate(dist):
            nd[i] += w * (1 - q)
            nd[i + 1] += w * q
        dist = nd
    return dist


def _boot(by_item: dict, stat: Callable[[list], Optional[float]], *,
          draws: int, seed: int, alpha: float = 0.05):
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


def localisation(rows: Sequence[dict], rates: dict) -> dict:
    """Observed objector-count distribution vs the POISSON-BINOMIAL null built
    from the per-seat marginals (not a pooled p). LOC = observed P(exactly 1)
    minus that null."""
    n = len(rows)
    obs = {k: sum(1 for r in rows if r["n_objectors"] == k) / n if n else None
           for k in range(4)}
    ps = [rates["per_seat"][k] for k in sorted(rates["per_seat"])]
    ind = poisson_binomial(ps)
    p = rates["pooled"]
    hom = [comb(3, k) * p ** k * (1 - p) ** (3 - k) for k in range(4)]
    return {
        "per_seat_objection_rate": p,
        "per_seat_marginals": rates["per_seat"],
        "observed": obs,
        "independence_poisson_binomial": {k: ind[k] for k in range(4)},
        "independence_pooled_INCORRECT": {k: hom[k] for k in range(4)},
        "fire_observed": (obs[2] + obs[3]) if n else None,
        "fire_independence": ind[2] + ind[3],
        "localisation_excess": (obs[1] - ind[1]) if n else None,
        "localisation_excess_pooled_INCORRECT": (obs[1] - hom[1]) if n else None,
    }


def analyse(rows_path: Path, votes_path: Path, *, model: Optional[str],
            draws: int, seed: int, label: str,
            keep_items: Optional[set] = None) -> dict:
    rows = load_rows(rows_path, model, keep_items)
    rates = per_seat_objection_rates(votes_path, model, keep_items)
    p, n_votes = rates["pooled"], rates["n_votes"]
    codable = [r for r in rows if r["codable"]]
    n = len(rows)

    def fired(r):
        return r["n_objectors"] >= 2

    def lift_on(pop):
        fl = [r for r in pop if fired(r)]
        un = [r for r in pop if not fired(r)]
        if not fl or not un:
            return None
        ef = 1 - sum(bool(r["s2_ok"]) for r in fl) / len(fl)
        eu = 1 - sum(bool(r["s2_ok"]) for r in un) / len(un)
        return {"n_fired": len(fl), "err_fired": ef, "err_unfired": eu,
                "error_lift": ef - eu}

    by_item_fire: dict = defaultdict(list)
    for r in rows:
        by_item_fire[r["item"]].append(1.0 if fired(r) else 0.0)
    fire_pt, fire_lo, fire_hi = _boot(
        by_item_fire, lambda xs: sum(xs) / len(xs) if xs else None,
        draws=draws, seed=seed)

    by_item_loc: dict = defaultdict(list)
    for r in rows:
        by_item_loc[r["item"]].append(r)
    loc_pt, loc_lo, loc_hi = _boot(
        by_item_loc,
        lambda rs, _pb=poisson_binomial(
            [rates["per_seat"][k] for k in sorted(rates["per_seat"])])[1]:
            (sum(1 for r in rs if r["n_objectors"] == 1) / len(rs) - _pb)
            if rs else None,
        draws=draws, seed=seed + 1)

    by_item_lift: dict = defaultdict(list)
    for r in codable:
        by_item_lift[r["item"]].append(r)

    def lift_stat(rs):
        fl = [r for r in rs if fired(r)]
        un = [r for r in rs if not fired(r)]
        if not fl or not un:
            return None
        return ((1 - sum(bool(r["s2_ok"]) for r in fl) / len(fl))
                - (1 - sum(bool(r["s2_ok"]) for r in un) / len(un)))

    lift_pt, lift_lo, lift_hi = _boot(by_item_lift, lift_stat, draws=draws,
                                      seed=seed + 2)

    n_fired_raw = sum(1 for r in rows if fired(r))
    noncod_fired = sum(1 for r in rows if fired(r) and not r["codable"])
    n_unfired_raw = n - n_fired_raw
    noncod_unfired = sum(1 for r in rows if not fired(r) and not r["codable"])

    return {
        "label": label, "n_debates": n, "n_items": len({r["item"] for r in rows}),
        "n_codable": len(codable), "n_seat_votes": n_votes,
        "localisation": localisation(rows, rates),
        "fire_rate": {"point": fire_pt, "lo": fire_lo, "hi": fire_hi},
        "localisation_excess_ci": {"point": loc_pt, "lo": loc_lo, "hi": loc_hi},
        "lift_codable": lift_on(codable),
        "lift_raw_noncodable_as_error": lift_on(rows),
        "lift_codable_ci": {"point": lift_pt, "lo": lift_lo, "hi": lift_hi},
        "p_noncodable_given_fired": (noncod_fired / n_fired_raw
                                     if n_fired_raw else None),
        "p_noncodable_given_unfired": (noncod_unfired / n_unfired_raw
                                       if n_unfired_raw else None),
        "mean_n_r3_unparsed": sum(r["n_r3_unparsed"] for r in rows) / n if n else None,
        "n_fired": n_fired_raw,
        "lift_is_powered": n_fired_raw >= MIN_FLAGGED_FOR_LIFT,
    }


def compare(control: dict, baseline: dict) -> dict:
    gate_parse = (control["mean_n_r3_unparsed"]
                  <= baseline["mean_n_r3_unparsed"] + PARSE_GATE_SLACK)
    return {
        "parse_gate_passed": gate_parse,
        "d_fire": control["fire_rate"]["point"] - baseline["fire_rate"]["point"],
        "d_localisation_excess": (control["localisation"]["localisation_excess"]
                                  - baseline["localisation"]["localisation_excess"]),
        "d_per_seat_objection_rate": (
            control["localisation"]["per_seat_objection_rate"]
            - baseline["localisation"]["per_seat_objection_rate"]),
        "d_lift": ((control["lift_codable"]["error_lift"]
                    - baseline["lift_codable"]["error_lift"])
                   if control["lift_codable"] and baseline["lift_codable"] else None),
        "lift_comparison_powered": control["lift_is_powered"],
    }


def print_report(res: dict) -> None:
    c, b, d = res["control"], res["baseline"], res["comparison"]
    print("\n" + "=" * 78)
    print("ADDENDUM 16.8 -- STAKE-FREE DELIBERATION ABLATION")
    print("=" * 78)
    for tag, a in (("EMBODIED (baseline)", b), ("STAKE-FREE (control)", c)):
        L = a["localisation"]
        print(f"\n  {tag}: n={a['n_debates']} debates, {a['n_items']} items, "
              f"{a['n_codable']} codable")
        print(f"    per-seat objection rate   {L['per_seat_objection_rate']:.3f}")
        print(f"    P(exactly 1 objector)     {L['observed'][1]:.3f}   "
              f"(independence {L['independence_poisson_binomial'][1]:.3f})")
        print(f"    LOCALISATION EXCESS       {L['localisation_excess']:+.3f}  "
              f"[{a['localisation_excess_ci']['lo']:+.3f}, "
              f"{a['localisation_excess_ci']['hi']:+.3f}]")
        print(f"    fire rate P(>=2)          {L['fire_observed']:.3f}   "
              f"(independence {L['fire_independence']:.3f})  "
              f"CI [{a['fire_rate']['lo']:.3f}, {a['fire_rate']['hi']:.3f}]")
        lc = a["lift_codable"]
        if lc:
            print(f"    error lift (codable)      {lc['error_lift']:+.3f}  "
                  f"[{a['lift_codable_ci']['lo']:+.3f}, "
                  f"{a['lift_codable_ci']['hi']:+.3f}]  n_fired={lc['n_fired']}"
                  f"{'' if a['lift_is_powered'] else '  << UNDERPOWERED'}")
        lr = a["lift_raw_noncodable_as_error"]
        if lr:
            print(f"    error lift (raw)          {lr['error_lift']:+.3f}")
        print(f"    P(non-codable | fired)    {a['p_noncodable_given_fired']}")
        print(f"    mean n_r3_unparsed        {a['mean_n_r3_unparsed']:.4f}")

    print("\n  " + "-" * 74)
    print(f"  R3 parse gate: {'PASS' if d['parse_gate_passed'] else 'FAIL -- do not read'}")
    print(f"  d(per-seat objection rate) {d['d_per_seat_objection_rate']:+.3f}")
    print(f"  d(LOCALISATION EXCESS)     {d['d_localisation_excess']:+.3f}   <-- primary")
    print(f"  d(fire rate)               {d['d_fire']:+.3f}")
    d_lift = "n/a" if d["d_lift"] is None else format(d["d_lift"], "+.3f")
    powered = "" if d["lift_comparison_powered"] else "  (UNDERPOWERED)"
    print(f"  d(error lift)              {d_lift}{powered}")


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # localisation: perfectly localised panel (always exactly one objector)
    rows = [{"n_objectors": 1} for _ in range(100)]
    R3 = {"per_seat": {"a": 1/3, "b": 1/3, "c": 1/3}, "pooled": 1/3, "n_votes": 300}
    L = localisation(rows, R3)
    check("observed P(exactly 1) is 1.0 on a perfectly localised panel",
          L["observed"][1] == 1.0)
    exp1 = 3 * (1 / 3) * (2 / 3) ** 2
    check("independence prediction matches the binomial by hand",
          abs(L["independence_poisson_binomial"][1] - exp1) < 1e-12)
    check("localisation excess is observed minus independence",
          abs(L["localisation_excess"] - (1.0 - exp1)) < 1e-12)
    check("fire rate is 0 when every debate has exactly one objector",
          L["fire_observed"] == 0.0)

    # independence sanity: a panel drawn AT independence should show ~0 excess
    rng = random.Random(5)
    p = 0.4
    rows2 = []
    for _ in range(20000):
        k = sum(1 for _ in range(3) if rng.random() < p)
        rows2.append({"n_objectors": k})
    L2 = localisation(rows2, {"per_seat": {"a": p, "b": p, "c": p},
                              "pooled": p, "n_votes": 60000})
    check("a genuinely independent panel shows a localisation excess near zero",
          abs(L2["localisation_excess"]) < 0.02)
    check("and its observed fire rate matches the independence prediction",
          abs(L2["fire_observed"] - L2["fire_independence"]) < 0.02)

    # the embodied numbers this whole design rests on, recomputed here
    check("embodied independence P(>=2) at p=0.398 is ~0.349",
          abs((3 * 0.398 ** 2 * 0.602 + 0.398 ** 3) - 0.349) < 0.002)
    check("embodied independence P(exactly 1) at p=0.398 is ~0.433",
          abs(3 * 0.398 * 0.602 ** 2 - 0.433) < 0.002)

    # load_rows must not depend on stake seats, and must gate codability
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "t_rows.csv"
        with open(rp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "model", "item_id", "arm", "sample_idx", "gold_verdict",
                "verdict", "synthesis_verdict", "n_objectors", "n_r3_unparsed"])
            w.writeheader()
            w.writerow({"model": "m", "item_id": "i1", "arm": "a",
                        "sample_idx": "0", "gold_verdict": "YTA",
                        "verdict": "YTA", "synthesis_verdict": "NTA",
                        "n_objectors": "2", "n_r3_unparsed": "0"})
            w.writerow({"model": "m", "item_id": "i2", "arm": "a",
                        "sample_idx": "0", "gold_verdict": "YTA",
                        "verdict": "UNRESOLVED", "synthesis_verdict": "YTA",
                        "n_objectors": "1", "n_r3_unparsed": "1"})
        got = load_rows(rp)
        check("load_rows reads both debates without any stake-seat column",
              len(got) == 2)
        check("a committal verdict is codable and scored", got[0]["codable"]
              and got[0]["s2_ok"] is True)
        check("a non-committal verdict is not codable and has s2_ok None",
              (not got[1]["codable"]) and got[1]["s2_ok"] is None)

    # REGRESSION GUARDS for the two bugs found after the first real run
    check("ESH is codable and counts as at-fault, so gold=YTA scores CORRECT "
          "(string equality would have called this wrong)",
          verdict_correct("ESH", "YTA") is True)
    check("NAH is codable and counts as not-at-fault against gold=NTA",
          verdict_correct("NAH", "NTA") is True)
    check("UNRESOLVED is not codable", verdict_correct("UNRESOLVED", "YTA") is None)
    check("a five-token instrument is NOT scored by raw string equality",
          verdict_correct("ESH", "YTA") != ("ESH" == "YTA"))

    pb = poisson_binomial([0.430, 0.690, 0.073])
    check("Poisson-binomial reproduces the registered embodied null "
          "P(exactly 1) = 0.501", abs(pb[1] - 0.501) < 0.002)
    check("and P(>=2) = 0.335, not the pooled null's 0.349",
          abs(pb[2] + pb[3] - 0.335) < 0.002)
    check("Poisson-binomial equals the binomial when all rates are equal",
          all(abs(poisson_binomial([0.4] * 3)[k]
                  - comb(3, k) * 0.4 ** k * 0.6 ** (3 - k)) < 1e-12
              for k in range(4)))

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag", default="cg_deliberation_unembodied")
    ap.add_argument("--baseline-tag", default="cg_deliberation")
    ap.add_argument("--model", default="grok-4-1-fast-reasoning")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    paths = {}
    for name, tag in (("control", a.tag), ("baseline", a.baseline_tag)):
        rp, vp = OUT_DIR / f"{tag}_rows.csv", OUT_DIR / f"{tag}_votes.csv"
        if not rp.exists() or not vp.exists():
            print(f"\nERROR: {rp} / {vp} not found.\n")
            return 2
        paths[name] = (rp, vp)

    # The control lost 43 of 420 debates to connection failures, so it covers
    # 194 of the embodied panel's 210 items (a strict subset, verified). The
    # headline comparison restricts the baseline to those same items; the
    # unrestricted baseline is reported alongside so the restriction's effect
    # is visible rather than assumed harmless.
    ctrl_items = {r["item"] for r in load_rows(paths["control"][0], a.model)}
    res = {
        "control": analyse(*paths["control"], model=a.model, draws=a.n_boot,
                           seed=a.seed, label=a.tag),
        "baseline": analyse(*paths["baseline"], model=a.model, draws=a.n_boot,
                            seed=a.seed + 50, label=a.baseline_tag + " [items matched to control]",
                            keep_items=ctrl_items),
        "baseline_full_panel": analyse(
            *paths["baseline"], model=a.model, draws=a.n_boot,
            seed=a.seed + 70, label=a.baseline_tag + " [full 210-item panel]"),
        "n_items_matched": len(ctrl_items),
    }
    res["comparison"] = compare(res["control"], res["baseline"])
    print_report(res)
    a.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
