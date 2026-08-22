"""
scripts/analyze_deliberation_dynamics.py -- what happens DURING deliberation,
as distinct from what the vote says at the end.

WHY THIS EXISTS
---------------
The programme's multi-agent work has been read out through its terminal vote,
and that read-out is uninformative: the Round-4 vote profile across 82 debates
is statistically indistinguishable from three independent Bernoulli draws
(observed rejections-per-debate {0:78, 1:4, 2:0, 3:0} against {0:78.07, 1:3.87,
2:0.06, 3:0.00} expected under independence; unanimity 0.9512 vs p^3 = 0.9520).
Voting aggregates fixed positions. Deliberation is supposed to *transform*
them. So the quantity of interest is not the tally but the movement: who moves,
when, and toward what.

WHAT IT MEASURES
----------------
1. DIVERSITY COLLAPSE. Distinct positions per debate at each round, and the
   share of debates that are unanimous. Deliberation that synthesises should
   not look the same as deliberation that merely converges.

2. CONFORMITY vs SYNTHESIS, against a chance baseline. When an agent changes
   its position, does it land on a position another agent ALREADY HELD
   (conformity) or on one nobody held (synthesis)? The raw share is
   uninterpretable on its own -- with a small action space most moves land on a
   held position by luck -- so the chance rate is computed per debate from the
   realised action space and the observed opening positions, and the excess is
   tested against it.

3. CONFIDENCE CALIBRATION ACROSS ROUNDS. If stated confidence does not fall in
   the rounds where positions are about to move, the confidence channel is not
   tracking the instability of the position.

4. ROLE ASYMMETRY. Whether the agent with the most at stake is the one that
   capitulates -- the deliberative analogue of the sycophancy question.

WHY THE CHANCE BASELINE MATTERS
-------------------------------
Scenario action spaces here range from 2 to 5 labels. A binary scenario gives a
conforming move ~100% of the time by construction. Pooling scenarios without
correcting for this manufactures a conformity finding out of the taxonomy. The
per-debate correction is the difference between a real result and an artefact,
and this repo has already published three artefacts of exactly that shape.

Usage:
  python -m scripts.analyze_deliberation_dynamics --selftest
  python -m scripts.analyze_deliberation_dynamics
  python -m scripts.analyze_deliberation_dynamics --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

from scripts.run_phase1_quartet import OUT_DIR

ROUNDS = ("round0", "round1", "round2")
DEFAULT_ROUNDS_CSV = OUT_DIR / "debate_all_round_decisions.csv"
DEFAULT_ROLES_CSV = OUT_DIR / "debate_mind_change_rates.csv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_debates(path: Path, *, round_alias: Optional[dict] = None,
                 resolve_novel: bool = True) -> dict:
    """(scenario, sample, model) -> {perspective_id: {round: row}}.

    resolve_novel guards a real artefact. In the open-action-space arm the
    ``decision`` column is the literal string "NOVEL" for every agent that
    proposes its own action -- 219 of 300 rows. Three agents proposing three
    COMPLETELY DIFFERENT actions therefore all read as the same position, and
    the arm scores 74% unanimous. Resolving each proposal to its own
    ``novel_action_label`` puts unanimity at 13%. The uncorrected reading
    inverts the finding, so the correction is on by default and the flag exists
    only to demonstrate the artefact.
    """
    rows = list(csv.DictReader(path.open()))
    cell: dict = defaultdict(dict)
    for r in rows:
        rd = (round_alias or {}).get(r["round"], r["round"])
        row = dict(r)
        if resolve_novel and row.get("novel_action_label"):
            row["decision"] = row["novel_action_label"]
        key = (r["scenario_id"], r["sample_idx"], r["gen_model"])
        cell[key].setdefault(r["perspective_id"], {})[rd] = row
    return dict(cell)


def structural_comparison(named: dict) -> dict:
    """Contagion measures side by side across structural variants."""
    out: dict = {}
    print("\n" + "=" * 74)
    print("STRUCTURAL COMPARISON -- which structures propagate, which resist")
    print("=" * 74)
    print(f"{'structure':<26}{'unanim r0':>10}{'unanim r2':>10}{'distinct r2':>13}"
          f"{'conformity excess':>19}{'p':>10}")
    for lab, d in named.items():
        div = diversity_by_round(d)
        e = conformity_all_baselines(d)["empirical"]
        if not div or not e:
            continue
        out[lab] = {"diversity": div, "conformity_empirical": e,
                    "movement": movement_by_transition(d)}
        print(f"{lab:<26}{div['round0']['unanimous_rate']:>9.0%}"
              f"{div['round2']['unanimous_rate']:>10.0%}"
              f"{div['round2']['mean_distinct']:>13.2f}"
              f"{e['excess']:>+19.1%}{e['p_value']:>10.2g}")
    return out


def action_space(agents: dict) -> set[str]:
    """Every label realised anywhere in this debate, across agents and rounds."""
    return {v[rd]["decision"] for v in agents.values() for rd in v}


# --- Chance baselines -------------------------------------------------------
#
# The conforming-move share is meaningless without a null, and the null is a
# real modelling choice that changes the answer, so all four are computed and
# reported rather than one being chosen quietly.
#
#   declared  -- uniform over the scenario's DECLARED decision taxonomy. This is
#                the option set the agents were actually given. It is the most
#                generous denominator: hospital_allocation declares four labels
#                and the agents only ever emit two, so a uniform draw over the
#                declared set overstates how many places a move could land.
#   scenario  -- uniform over labels REALISED anywhere in that scenario. Drops
#                the dead options but still pools across debates.
#   empirical -- drawn from the scenario's empirical position distribution.
#                Preferred: it respects that DEFER is technically available and
#                effectively never chosen, without conditioning on the outcome
#                of the individual debate.
#   debate    -- uniform over labels realised in THAT debate. Reported for
#                completeness and NOT preferred: it conditions on the very
#                outcome under test, so a debate in which everyone converged
#                looks binary by construction and the null is inflated toward 1.
BASELINES = ("declared", "scenario", "empirical", "debate")


def declared_taxonomies() -> dict[str, set[str]]:
    """scenario_id -> declared decision-taxonomy labels, from the live source."""
    try:
        from scripts.run_debate_std_cot_ablation import SCENARIOS
    except Exception:
        return {}
    return {s.id: set(s.decision_taxonomy) for s in SCENARIOS}


def scenario_spaces(debates: dict) -> tuple[dict, dict]:
    """(labels realised per scenario, Counter of positions per scenario)."""
    space: dict = defaultdict(set)
    marg: dict = defaultdict(Counter)
    for (scen, _, _), agents in debates.items():
        for v in agents.values():
            for rd in v:
                space[scen].add(v[rd]["decision"])
                marg[scen][v[rd]["decision"]] += 1
    return dict(space), dict(marg)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def diversity_by_round(debates: dict, n_agents: int = 3) -> dict:
    out: dict = {}
    for i, rd in enumerate(ROUNDS):
        counts, total = Counter(), 0
        for agents in debates.values():
            if len(agents) != n_agents:
                continue
            pos = [v[rd]["decision"] for v in agents.values() if rd in v]
            if len(pos) != n_agents:
                continue
            counts[len(set(pos))] += 1
            total += 1
        if not total:
            continue
        out[rd] = {
            "distinct_counts": dict(sorted(counts.items())),
            "mean_distinct": sum(k * v for k, v in counts.items()) / total,
            "unanimous": counts[1],
            "n": total,
            "unanimous_rate": counts[1] / total,
        }
    return out


def movement_by_transition(debates: dict) -> dict:
    out: dict = {}
    for a, b in zip(ROUNDS, ROUNDS[1:]):
        ch = tot = 0
        for agents in debates.values():
            for v in agents.values():
                if a in v and b in v:
                    tot += 1
                    ch += v[a]["decision"] != v[b]["decision"]
        if tot:
            out[f"{a}->{b}"] = {"changed": ch, "n": tot, "rate": ch / tot}
    return out


def conformity_all_baselines(debates: dict, **kw) -> dict:
    """Conformity excess under every chance baseline, so the sensitivity shows."""
    return {b: conformity(debates, baseline=b, **kw) for b in BASELINES}


def conformity(debates: dict, *, n_agents: int = 3, baseline: str = "empirical",
               first: str = "round0", last: str = "round2") -> dict:
    """Conforming-move share against a per-debate chance baseline.

    A move is CONFORMING if it lands on a label another agent already held at
    ``first``. The chance rate for one move is |alternatives that another agent
    held| / |alternatives|, where alternatives excludes the mover's own opening
    label -- i.e. the probability of conforming if the mover picked uniformly
    among the labels it could have moved to.
    """
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}, got {baseline!r}")
    declared = declared_taxonomies()
    scen_space, scen_marg = scenario_spaces(debates)

    obs = n = 0
    exp = 0.0
    per_role: dict = defaultdict(lambda: {"obs": 0, "n": 0, "exp": 0.0})
    for key, agents in debates.items():
        if len(agents) != n_agents:
            continue
        scen = key[0]
        if baseline == "debate":
            space = action_space(agents)
        elif baseline == "scenario" or baseline == "empirical":
            space = scen_space.get(scen, action_space(agents))
        else:  # declared
            space = declared.get(scen) or scen_space.get(scen, action_space(agents))
        for pid, v in agents.items():
            if first not in v or last not in v:
                continue
            start, end = v[first]["decision"], v[last]["decision"]
            if start == end:
                continue
            others = {o[first]["decision"] for q, o in agents.items()
                      if q != pid and first in o}
            alt = [c for c in space if c != start]
            if not alt:
                continue
            if baseline == "empirical":
                # Draw the destination from the scenario's empirical position
                # distribution, renormalised over the labels the mover could
                # move TO. Respects that some declared options are never used.
                w = {c: scen_marg.get(scen, Counter()).get(c, 0) for c in alt}
                tot_w = sum(w.values())
                chance = (sum(v for c, v in w.items() if c in others) / tot_w
                          if tot_w else len([c for c in alt if c in others]) / len(alt))
            else:
                chance = len([c for c in alt if c in others]) / len(alt)
            hit = int(end in others)
            obs += hit
            exp += chance
            n += 1
            per_role[pid]["obs"] += hit
            per_role[pid]["exp"] += chance
            per_role[pid]["n"] += 1
    if not n:
        return {}
    p = exp / n
    pval = binom_sf(obs, n, p)
    return {
        "observed": obs, "n": n,
        "observed_rate": obs / n,
        "chance_rate": p,
        "excess": obs / n - p,
        "p_value": pval,
        "per_role": {k: dict(v, observed_rate=v["obs"] / v["n"],
                             chance_rate=v["exp"] / v["n"],
                             excess=v["obs"] / v["n"] - v["exp"] / v["n"])
                     for k, v in per_role.items() if v["n"]},
    }


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact, stdlib only."""
    if p <= 0:
        return 1.0 if k <= 0 else 0.0
    if p >= 1:
        return 1.0 if k <= n else 0.0
    return min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                        for i in range(k, n + 1)))


def confidence_by_round(debates: dict) -> dict:
    out: dict = {}
    for rd in ROUNDS:
        vals = []
        for agents in debates.values():
            for v in agents.values():
                if rd in v and v[rd].get("decision_confidence"):
                    try:
                        vals.append(float(v[rd]["decision_confidence"]))
                    except ValueError:
                        pass
        if vals:
            out[rd] = {"mean": sum(vals) / len(vals), "n": len(vals)}
    return out


def confidence_of_movers(debates: dict) -> dict:
    """Is stated confidence lower on the moves that are about to change?

    A confidence channel that tracks the instability of a position should read
    LOWER at round0 on the agents that go on to move.
    """
    mv, st = [], []
    for agents in debates.values():
        for v in agents.values():
            if "round0" not in v or "round2" not in v:
                continue
            try:
                c = float(v["round0"]["decision_confidence"])
            except (ValueError, KeyError, TypeError):
                continue
            (mv if v["round0"]["decision"] != v["round2"]["decision"] else st).append(c)
    if not mv or not st:
        return {}
    return {
        "movers_mean": sum(mv) / len(mv), "movers_n": len(mv),
        "stayers_mean": sum(st) / len(st), "stayers_n": len(st),
        "difference": sum(mv) / len(mv) - sum(st) / len(st),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(debates: dict) -> dict:
    div = diversity_by_round(debates)
    mov = movement_by_transition(debates)
    con_all = conformity_all_baselines(debates)
    con = con_all["empirical"]
    cbr = confidence_by_round(debates)
    cmv = confidence_of_movers(debates)

    print("=" * 74)
    print("DELIBERATION DYNAMICS -- what moves, not what the vote said")
    print("=" * 74)

    print("\n1. DIVERSITY COLLAPSE")
    print(f"   {'round':<10}{'mean distinct':>15}{'unanimous':>12}{'n':>6}   distribution")
    for rd, v in div.items():
        print(f"   {rd:<10}{v['mean_distinct']:>15.2f}{v['unanimous_rate']:>11.0%}"
              f"{v['n']:>6}   {v['distinct_counts']}")
    if len(div) >= 2:
        a, b = list(div.values())[0], list(div.values())[-1]
        print(f"   -> unanimity {a['unanimous_rate']:.0%} -> {b['unanimous_rate']:.0%}; "
              f"{a['distinct_counts'].get(1,0)} of {a['n']} debates open with "
              f"NOTHING TO DELIBERATE.")

    print("\n2. MOVEMENT")
    for k, v in mov.items():
        print(f"   {k:<22}{v['changed']:>5}/{v['n']:<6} = {v['rate']:.1%}")

    print("\n3. CONFORMITY vs SYNTHESIS")
    if con:
        print(f"   conforming moves observed : {con['observed']}/{con['n']} = "
              f"{con['observed_rate']:.1%}")
        print(f"   synthesis (novel position): {con['n'] - con['observed']}/{con['n']} = "
              f"{1 - con['observed_rate']:.1%}")
        print("\n   THE ANSWER DEPENDS ON THE CHANCE BASELINE, so all four are shown.")
        print(f"   {'baseline':<12}{'chance':>9}{'excess':>10}{'p':>12}   note")
        NOTE = {"declared": "uniform over the DECLARED taxonomy (most generous)",
                "scenario": "uniform over labels realised in the scenario",
                "empirical": "scenario's empirical distribution (PREFERRED)",
                "debate": "labels realised in that debate (endogenous, NOT preferred)"}
        for b in BASELINES:
            c = con_all.get(b) or {}
            if not c:
                continue
            print(f"   {b:<12}{c['chance_rate']:>8.1%}{c['excess']:>+10.1%}"
                  f"{c['p_value']:>12.1e}   {NOTE[b]}")
        # The per-debate null conditions on the outcome under test, so it is
        # reported but excluded from the verdict. Judging on it would let an
        # endogenous null veto a real effect.
        DEFENSIBLE = ("declared", "scenario", "empirical")
        signs = {b: (con_all[b]["excess"] > 0) for b in DEFENSIBLE if con_all.get(b)}
        sig = sum(1 for b in signs if con_all[b]["p_value"] < 0.05)
        rng = [con_all[b]["excess"] for b in DEFENSIBLE if con_all.get(b)]
        if len(set(signs.values())) > 1:
            print("\n   -> THE SIGN FLIPS ACROSS DEFENSIBLE BASELINES. Not established.")
        elif sig == len(signs):
            print(f"\n   -> Same sign and significant under all {len(signs)} defensible")
            print(f"      baselines. Conformity above chance HOLDS, with an effect size")
            print(f"      of {min(rng):+.1%} to {max(rng):+.1%} depending on how generous")
            print(f"      the null is. Quote the smallest ({min(rng):+.1%}).")
        else:
            print(f"\n   -> Same sign under all defensible baselines but significant in")
            print(f"      only {sig}/{len(signs)}: suggestive, not established.")
        if con_all.get("debate") and con_all["debate"]["excess"] <= 0:
            print("      (The per-debate null shows nothing, as expected: it conditions")
            print("       on the convergence it is meant to test.)")

    print("\n4. CONFIDENCE")
    for rd, v in cbr.items():
        print(f"   {rd:<10}mean {v['mean']:.3f}   n={v['n']}")
    if cmv:
        print(f"   round0 confidence of agents who LATER MOVE : {cmv['movers_mean']:.3f} "
              f"(n={cmv['movers_n']})")
        print(f"   round0 confidence of agents who DO NOT     : {cmv['stayers_mean']:.3f} "
              f"(n={cmv['stayers_n']})")
        print(f"   difference                                 : {cmv['difference']:+.3f}")
        if abs(cmv["difference"]) < 0.02:
            print("   -> confidence does NOT anticipate movement: the channel is not")
            print("      tracking the instability of the position.")

    return {"diversity": div, "conformity_all_baselines": con_all,
            "movement": mov, "conformity": con,
            "confidence_by_round": cbr, "confidence_of_movers": cmv}


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _mk(pos_by_round: dict, conf: float = 0.9) -> dict:
    return {rd: {"decision": d, "decision_confidence": str(conf)}
            for rd, d in pos_by_round.items()}


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("binom_sf(0,n,p) == 1", abs(binom_sf(0, 10, 0.3) - 1.0) < 1e-12)
    check("binom_sf(n+1,...) == 0", binom_sf(11, 10, 0.3) == 0.0)
    check("binom_sf matches a hand value",
          abs(binom_sf(2, 3, 0.5) - 0.5) < 1e-12)

    # A debate where everyone already agrees: no movement, no conformity moves.
    same = {"a": _mk({r: "X" for r in ROUNDS}),
            "b": _mk({r: "X" for r in ROUNDS}),
            "c": _mk({r: "X" for r in ROUNDS})}
    d1 = {("s", "0", "m"): same}
    check("unanimous debate reads as 1 distinct position",
          diversity_by_round(d1)["round0"]["mean_distinct"] == 1.0)
    check("unanimous debate yields no conforming moves",
      conformity(d1, baseline="debate") == {})

    # A pure conformity move in a BINARY space: chance is 1.0, so the excess
    # must be zero. This is the artefact the chance baseline exists to prevent.
    binary = {"a": _mk({"round0": "X", "round1": "Y", "round2": "Y"}),
              "b": _mk({r: "Y" for r in ROUNDS}),
              "c": _mk({r: "Y" for r in ROUNDS})}
    cb = conformity({("s", "0", "m"): binary}, baseline="debate")
    check("binary space: conforming move observed", cb["observed_rate"] == 1.0)
    check("binary space: chance baseline is also 1.0", cb["chance_rate"] == 1.0)
    check("binary space: EXCESS IS ZERO (the artefact guard)",
          abs(cb["excess"]) < 1e-12)

    # A 4-way space where the mover conforms: chance is 1/3, excess is +2/3.
    wide = {"a": _mk({"round0": "W", "round1": "X", "round2": "X"}),
            "b": _mk({r: "X" for r in ROUNDS}),
            "c": _mk({r: "X" for r in ROUNDS})}
    # give the space four labels by having someone visit Y and Z
    wide["c"] = _mk({"round0": "X", "round1": "Y", "round2": "Z"})
    cw = conformity({("s", "0", "m"): wide}, baseline="debate")
    check("wide space: chance baseline below 1", cw["chance_rate"] < 1.0)
    check("wide space: excess is positive when the mover conforms",
          cw["excess"] > 0)

    # Synthesis: mover lands on a label nobody held.
    syn = {"a": _mk({"round0": "W", "round1": "Q", "round2": "Q"}),
           "b": _mk({r: "X" for r in ROUNDS}),
           "c": _mk({r: "X" for r in ROUNDS})}
    cs = conformity({("s", "0", "m"): syn}, baseline="debate")
    check("synthesis move is not counted as conforming", cs["observed"] == 0)

    # Movement detection
    mv = movement_by_transition({("s", "0", "m"): binary})
    check("movement detected on r0->r1", mv["round0->round1"]["changed"] == 1)
    check("no movement on r1->r2", mv["round1->round2"]["changed"] == 0)

    # Confidence-of-movers separates the two groups
    cm = confidence_of_movers({("s", "0", "m"): {
        "a": _mk({"round0": "X", "round1": "Y", "round2": "Y"}, conf=0.5),
        "b": _mk({r: "Y" for r in ROUNDS}, conf=0.9),
        "c": _mk({r: "Y" for r in ROUNDS}, conf=0.9)}})
    check("confidence_of_movers separates movers from stayers",
          abs(cm["difference"] + 0.4) < 1e-9)

    allb = conformity_all_baselines({("s", "0", "m"): wide})
    check("all four baselines compute without error",
          all(allb.get(b) for b in BASELINES))
    check("the endogenous per-debate null is never more generous than declared",
          allb["debate"]["chance_rate"] >= allb["declared"]["chance_rate"] - 1e-9)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Deliberation dynamics, not vote tallies")
    ap.add_argument("--rounds", type=Path, default=DEFAULT_ROUNDS_CSV)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if not a.rounds.exists():
        print(f"ERROR: {a.rounds} not found")
        return 2
    debates = load_debates(a.rounds)
    print(f"loaded {len(debates)} debates from {a.rounds}")
    out = report(debates)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
