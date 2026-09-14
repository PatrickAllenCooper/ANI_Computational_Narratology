"""
scripts/analyze_coalition.py -- Addendum 16.19 readout: coalition advocates.

WHAT IS READ

The coalition cell (scripts/run_crowdgold_coalition.py, tag
cg_deliberation_coalition) seats FOUR advocates on the registered AITA panel
with edges on: writer_advocate_co and writer_ally_co on the writer's side
(stake not_at_fault, LIKE-biased), counterparty_co (stake at_fault) and
neutral_adjudicator_co (no stake). Krishna and Morgan (2001) predict that the
second like-biased expert adds no information. This module reads that
prediction off the cell four ways, every statistic through the conventions
of the 16.10 table (scripts/analyze_topology_2x2.py): item-clustered
bootstrap CIs, five-token verdicts scored by the registered code_response
collapse and never by string equality, raw and codable populations side by
side, lifts read only at >= 60 fired, partial panels reported and never
retried.

  1. ROLE-LOCK for each advocate. Per-seat P(at_fault) at r0 from the cached
     r0 records; lock = |P(at_fault) advocate - P(at_fault) counterparty|,
     reported for the writer's advocate AND for the ally (embodied/on:
     0.971), plus the like-biased pair's r0 gap and per-debate concordance.
  2. The LIKE-BIASED PAIR'S OBJECTION PHI. Phi coefficient between
     objected_r3 of writer_advocate_co and writer_ally_co over debates in
     which both R3 labels parsed, with the 2x2 table and the conditionals
     P(ally objects | writer objects) and P(ally objects | writer does not);
     the two opposed pairs' phi for contrast. K-M: near +1.
     READ IT AS A POST-COMMUNICATION STATISTIC. The cell runs with edges ON
     (run_crowdgold_coalition.r1_user_co / r2_user_co quote all three
     neighbours), so the advocate and the ally have each read the other's r0
     and r1 before either labels the synthesis at R3: a phi near +1 mixes
     shared stake with two rounds of mutual persuasion, and the opposed-pair
     contrast under the same regime controls that only partly. The
     PRE-communication, independence reading is item 1's like-pair r0
     verdict concordance (r0 is written before any seat reads any other);
     the report prints the two side by side and flags a divergence. If they
     diverge, the clean K-M test is an EDGES-OFF coalition cell
     (run_crowdgold_topology.r1_user_noedge / r2_user_noedge under new
     `_co_noedge` ids), which is a separate registration, not built here.
  3. The POOLED FLAG'S LIFT. The 16.10 flag is ">= 2 of 3 seats objected at
     R3". It is read three ways on the same debates: THREE-SEAT (the ally
     ignored: the embodied flag reconstructed), POOLED over all four seats at
     >= 2, and pooled at >= 3. Lift = P(S2 wrong | fired) - P(S2 wrong | not)
     on the codable population; the paired delta (pooled - three-seat)
     carries an item-clustered CI. K-M: pooling the ally does not raise the
     lift. The information test conditions on the writer's objection and
     asks whether the ally's still separates error; at phi = 1 there are no
     discordant debates to condition on, which is the point.
  4. The AGAINST-INTEREST OBJECTION COUNT, the generalised sensor. A seat
     objects AGAINST INTEREST when it objects at R3 to a synthesis verdict
     that SERVES its own stake: the writer's advocate or the ally objecting
     to a not-at-fault synthesis, the counterparty objecting to an at-fault
     one. The per-debate count is 0-2 (the two writer-side seats on a
     not-at-fault synthesis, the counterparty alone on an at-fault one) and
     blank when the synthesis is not codable. The runner writes it as
     n_against_interest_r3; it is RECOMPUTED here from the votes CSV (role
     stake x synthesis verdict x objected_r3) and any mismatch with the
     column is reported loudly. Fire = count >= 1: fire rate, lift with CI,
     per-seat against-interest rates, transfer to the cached grok solo
     (standard majority-of-3, which no deliberation cell touched), and the
     SAME sensor run on the embodied cell's own votes as the three-seat
     baseline.
  5. S2 ACCURACY, PAIRED against the embodied cell. The coalition group
     verdict's accuracy on the same (arm, item) debates as cg_deliberation:
     paired against the embodied sample 0 (the matching seed draw) and
     against the embodied per-debate mean over its samples (the 0.843 of
     the 16.10 table), with item-clustered CIs on the paired difference.

Plus the standard 16.10 columns: per-seat objection marginals, localisation
excess against the n-seat Poisson-binomial null (analyze_unembodied_ablation
.localisation hard-codes three seats -- comb(3, k), range(4) -- and is NOT
used), fire rate P(>= 2 of 4), G3 stake concentration over the three stake
seats, and the parse guards.

GUARDS (the literal GUARD FAILED is printed, exit 4 is returned and NO
analysis JSON is written when any fails):
  * the runner's own records: <tag>_summary.json must exist with
    round_guard.pass and outcome_guard.pass True, and <tag>_completeness.json
    (every launched debate is a row; run_crowdgold_coalition writes it) must
    exist with pass True. A missing record is a failed guard: the runner
    quarantines a failed run's CSVs as <tag>_GUARD_FAILED_*, but a CSV pair
    without its guard records is not a readable cell either.
  * run_guards on the rows/votes: S2-codable share >= 0.95 (the base guard's
    call_guard_row passes a legal "VERDICT: UNRESOLVED" as parsed, so an
    UNRESOLVED / NOVERDICT flood at synthesis, R3, R4 or integration is
    visible only here), mean n_r3_unparsed <= 0.05, R4 vote_parsed share
    >= 0.95, all four seats present on >= 0.99 of debates, and the runner's
    n_against_interest_r3 column agreeing with the recomputation on every
    codable debate.

Model lists are flags (--model, --scaffold, --tag, --embodied-tag,
--comparator) so the fourth vendor's deployment reads through the same code
with no edit.

Usage
-----
  python -m scripts.analyze_coalition --selftest    # synthetic votes in a tempdir
  python -m scripts.analyze_coalition               # the cell on disk
  python -m scripts.analyze_coalition --model <deployment>
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tempfile
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_actuator_ladder import COMPARATOR_PATH, correct, load_comparator, make_rule_vendor
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_unembodied_ablation import (
    _boot, per_seat_objection_rates, poisson_binomial, verdict_correct,
)
from scripts.load_scruples import AT_FAULT_VERDICTS, NOT_AT_FAULT_VERDICTS
from scripts.run_phase1_quartet import OUT_DIR
from scripts.verdict_format import extract_verdict, is_noncommittal

TAG = "cg_deliberation_coalition"
EMBODIED_TAG = "cg_deliberation"
MODEL = "grok-4-1-fast-reasoning"
SCAFFOLD = "narrative_cot"
ANALYSIS_NAME = "coalition_analysis.json"
R0_CAP = 2560
MIN_FIRED = 60
MIN_CODABLE_SHARE = 0.95
MAX_MEAN_R3_UNPARSED = 0.05
MIN_VOTE_PARSED_SHARE = 0.95
MIN_SEATS_PRESENT_SHARE = 0.99
#: 16.10 table, embodied/on, codable population over all four samples.
EMBODIED_S2_REGISTERED = 0.843

WRITER = "writer_advocate_co"
ALLY = "writer_ally_co"
COUNTER = "counterparty_co"
NEUTRAL = "neutral_adjudicator_co"
SEATS = (WRITER, ALLY, COUNTER, NEUTRAL)
THREE_SEATS = (WRITER, COUNTER, NEUTRAL)          # the embodied flag, ally ignored
STAKE_SEATS = (WRITER, ALLY, COUNTER)
LIKE_PAIR = (WRITER, ALLY)
OPPOSED_PAIRS = ((WRITER, COUNTER), (ALLY, COUNTER))
EMBODIED_SEATS = ("writer_advocate", "counterparty", "neutral_adjudicator")


# ---------------------------------------------------------------------------
# Stake logic, keyed on the STAKE string so `_co` and embodied seats read alike
# ---------------------------------------------------------------------------

def undermined_by(stake: str, verdict: str) -> Optional[bool]:
    """run_crowdgold_deliberation.stake_undermined keyed on the stake string
    rather than the role id, so it needs no install(). None when the verdict
    is not codable."""
    if is_noncommittal(verdict) or verdict == rcd.NOVERDICT:
        return None
    at_fault = verdict in AT_FAULT_VERDICTS
    not_at_fault = verdict in NOT_AT_FAULT_VERDICTS
    if not (at_fault or not_at_fault):
        return None
    if stake == "not_at_fault":
        return at_fault
    if stake == "at_fault":
        return not_at_fault
    return False


def against_interest(seat_votes: dict[str, dict], synthesis_verdict: str,
                     seats: Sequence[str]) -> tuple[Optional[int], tuple[str, ...]]:
    """(count, role ids) of stake-bearing seats whose R3 objection was lodged
    against a synthesis verdict that SERVES their stake. Mirrors
    run_crowdgold_coalition.against_interest_count. None when the synthesis
    is not codable (the debate is excluded, not coded 0)."""
    roles = []
    for rid in seats:
        v = seat_votes.get(rid)
        if v is None or v["role_stake"] not in ("at_fault", "not_at_fault"):
            continue
        und = undermined_by(v["role_stake"], synthesis_verdict)
        if und is None:
            return None, ()
        if v["objected_r3"] and not und:
            roles.append(rid)
    return len(roles), tuple(roles)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_votes(votes_path: Path, model: Optional[str], keep_items: Optional[set] = None) -> dict:
    """(arm, item, sample) -> {role_id: typed vote row}."""
    out: dict = defaultdict(dict)
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            if model and v.get("model") != model:
                continue
            if keep_items is not None and v["item_id"] not in keep_items:
                continue
            su = v.get("stake_undermined")
            out[(v["arm"], v["item_id"], int(v["sample_idx"]))][v["role_id"]] = {
                "role_id": v["role_id"],
                "role_stake": v.get("role_stake", ""),
                "objected_r3": int(v.get("objected_r3") or 0),
                "r3_label_parsed": int(v.get("r3_label_parsed") or 0),
                "reject": int(v.get("reject") or 0),
                "vote_parsed": int(v.get("vote_parsed") or 0),
                "stake_undermined": None if su in ("", None) else int(su),
                "synthesis_verdict": v.get("synthesis_verdict", ""),
            }
    return dict(out)


def load_debates(rows_path: Path, votes_path: Path, model: Optional[str],
                 keep_items: Optional[set] = None, seats: Sequence[str] = SEATS) -> list[dict]:
    """One record per debate, role-agnostic except for the per-seat objection
    map and the against-interest count recomputed from the votes."""
    votes = load_votes(votes_path, model, keep_items)
    out = []
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            if model and r.get("model") != model:
                continue
            if keep_items is not None and r["item_id"] not in keep_items:
                continue
            syn, ver = r.get("synthesis_verdict", ""), r.get("verdict", "")
            key = (r["arm"], r["item_id"], int(r["sample_idx"]))
            sv = votes.get(key, {})
            n_ai, ai_roles = against_interest(sv, syn, seats)
            col = r.get("n_against_interest_r3")
            out.append({
                "item": r["item_id"], "arm": r["arm"], "sample_idx": int(r["sample_idx"]),
                "gold": r["gold_verdict"], "verdict": ver, "synthesis_verdict": syn,
                "n_objectors": int(r.get("n_objectors") or 0),
                "n_r3_unparsed": int(r.get("n_r3_unparsed") or 0),
                "codable": (code_response(syn, "published") is not None
                            and code_response(ver, "published") is not None),
                "s2_ok": verdict_correct(ver, r["gold_verdict"]),
                "objected": {s: (sv[s]["objected_r3"] if s in sv else None) for s in seats},
                "r3_parsed": {s: (sv[s]["r3_label_parsed"] if s in sv else 0) for s in seats},
                "n_against_interest": n_ai,
                "against_interest_roles": ai_roles,
                "n_against_interest_column": None if col in ("", None) else int(col),
            })
    return out


# ---------------------------------------------------------------------------
# Flags on a debate record. None = the debate is excluded from that flag.
# ---------------------------------------------------------------------------

def n_objected(d: dict, seats: Sequence[str]) -> Optional[int]:
    vals = [d["objected"].get(s) for s in seats]
    return None if any(v is None for v in vals) else sum(vals)


def flag_three_seat(d: dict) -> Optional[bool]:
    n = n_objected(d, THREE_SEATS)
    return None if n is None else n >= 2


def flag_pooled(d: dict) -> Optional[bool]:
    n = n_objected(d, SEATS)
    return None if n is None else n >= 2


def flag_pooled_3(d: dict) -> Optional[bool]:
    n = n_objected(d, SEATS)
    return None if n is None else n >= 3


def flag_against_interest(d: dict) -> Optional[bool]:
    n = d["n_against_interest"]
    return None if n is None else n >= 1


def flag_embodied_ge2(d: dict) -> Optional[bool]:
    return d["n_objectors"] >= 2


FLAGS: dict[str, Callable[[dict], Optional[bool]]] = {
    "three_seat_ge2": flag_three_seat,
    "pooled_ge2": flag_pooled,
    "pooled_ge3": flag_pooled_3,
    "against_interest_ge1": flag_against_interest,
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def phi_table(pairs: Sequence[tuple[int, int]]) -> dict:
    """Phi coefficient and 2x2 table of two binary series. phi is None when
    either marginal is constant (the coefficient is undefined, not zero)."""
    a = sum(1 for x, y in pairs if x and y)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    d = sum(1 for x, y in pairs if not x and not y)
    n = a + b + c + d
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return {
        "n": n, "both": a, "first_only": b, "second_only": c, "neither": d,
        "phi": None if denom == 0 else (a * d - b * c) / denom,
        "agreement": (a + d) / n if n else None,
        "p_first": (a + b) / n if n else None,
        "p_second": (a + c) / n if n else None,
        "p_second_given_first": a / (a + b) if a + b else None,
        "p_second_given_not_first": c / (c + d) if c + d else None,
    }


def pair_phi(debates: Sequence[dict], seat_a: str, seat_b: str, *, draws: int, seed: int) -> dict:
    """Objection phi between two seats over debates in which both R3 labels
    parsed, item-clustered bootstrap CI."""
    by: dict = defaultdict(list)
    for d in debates:
        x, y = d["objected"].get(seat_a), d["objected"].get(seat_b)
        if x is None or y is None or not (d["r3_parsed"].get(seat_a) and d["r3_parsed"].get(seat_b)):
            continue
        by[d["item"]].append((x, y))
    if not by:
        return {"seats": [seat_a, seat_b], "n": 0, "phi": None, "lo": None, "hi": None}
    p, lo, hi = _boot(by, lambda ps: phi_table(ps)["phi"], draws=draws, seed=seed)
    t = phi_table([x for i in by for x in by[i]])
    t.update({"seats": [seat_a, seat_b], "lo": lo, "hi": hi})
    return t


def localisation_n(rows: Sequence[dict], rates: dict, seats: Sequence[str]) -> dict:
    """Observed objector-count distribution over 0..n_seats vs the
    Poisson-binomial null from the per-seat marginals. Fire = P(>= 2)."""
    n = len(rows)
    k_max = len(seats)
    obs = {k: (sum(1 for r in rows if r["n_objectors"] == k) / n if n else None)
           for k in range(k_max + 1)}
    ps = [rates["per_seat"].get(s, 0.0) for s in seats]
    ind = poisson_binomial(ps)
    return {
        "n_seats": k_max,
        "per_seat_marginals": {s: rates["per_seat"].get(s) for s in seats},
        "pooled": rates["pooled"],
        "observed": obs,
        "independence_poisson_binomial": {k: ind[k] for k in range(k_max + 1)},
        "fire_observed": (sum(obs[k] for k in range(2, k_max + 1)) if n else None),
        "fire_independence": sum(ind[2:]),
        "localisation_excess": (obs[1] - ind[1]) if n else None,
    }


def _err_split(ds: Sequence[dict], flag: Callable[[dict], Optional[bool]]) -> Optional[float]:
    fl = [d for d in ds if flag(d)]
    un = [d for d in ds if flag(d) is False]
    if not fl or not un:
        return None
    return ((1 - sum(bool(d["s2_ok"]) for d in fl) / len(fl))
            - (1 - sum(bool(d["s2_ok"]) for d in un) / len(un)))


def lift(debates: Sequence[dict], flag: Callable[[dict], Optional[bool]], *,
         draws: int, seed: int, min_fired: int = MIN_FIRED) -> dict:
    """P(S2 wrong | fired) - P(S2 wrong | not fired) on the codable population.
    Fire rate and n_fired are on the RAW population (every debate the flag is
    defined on); a debate the flag is not defined on is excluded."""
    defined = [d for d in debates if flag(d) is not None]
    n_fired = sum(1 for d in defined if flag(d))
    pop = [d for d in defined if d["codable"]]
    fl = [d for d in pop if flag(d)]
    un = [d for d in pop if not flag(d)]
    by: dict = defaultdict(list)
    for d in pop:
        by[d["item"]].append(d)
    p, lo, hi = (_boot(by, lambda ds: _err_split(ds, flag), draws=draws, seed=seed)
                 if by else (None, None, None))
    return {
        "n_defined": len(defined), "n_excluded": len(debates) - len(defined),
        "n_fired": n_fired,
        "fire_rate": (n_fired / len(defined)) if defined else None,
        "n_codable": len(pop), "n_fired_codable": len(fl),
        "err_fired": (1 - sum(bool(d["s2_ok"]) for d in fl) / len(fl)) if fl else None,
        "err_unfired": (1 - sum(bool(d["s2_ok"]) for d in un) / len(un)) if un else None,
        "delta": p, "lo": lo, "hi": hi,
        "powered": n_fired >= min_fired,
        "p_noncodable_given_fired": ((n_fired - len(fl)) / n_fired) if n_fired else None,
    }


def paired_lift_delta(debates: Sequence[dict], flag_a: Callable, flag_b: Callable, *,
                      draws: int, seed: int) -> dict:
    """lift(flag_b) - lift(flag_a) on the SAME codable debates, item-clustered."""
    pop = [d for d in debates if d["codable"] and flag_a(d) is not None and flag_b(d) is not None]
    by: dict = defaultdict(list)
    for d in pop:
        by[d["item"]].append(d)

    def stat(ds):
        la, lb = _err_split(ds, flag_a), _err_split(ds, flag_b)
        return None if la is None or lb is None else lb - la
    p, lo, hi = _boot(by, stat, draws=draws, seed=seed) if by else (None, None, None)
    return {"n": len(pop), "delta": p, "lo": lo, "hi": hi}


def ally_information(debates: Sequence[dict]) -> dict:
    """Error rate by (writer objected, ally objected) on codable debates, and
    whether the ally's objection separates error once the writer's is fixed."""
    cells = {}
    for w in (0, 1):
        for a in (0, 1):
            ds = [d for d in debates if d["codable"]
                  and d["objected"].get(WRITER) == w and d["objected"].get(ALLY) == a]
            cells[f"writer={w},ally={a}"] = {
                "n": len(ds),
                "err": (1 - sum(bool(d["s2_ok"]) for d in ds) / len(ds)) if ds else None}
    c11, c10 = cells["writer=1,ally=1"], cells["writer=1,ally=0"]
    c01, c00 = cells["writer=0,ally=1"], cells["writer=0,ally=0"]
    within = (c11["err"] - c10["err"]) if c11["err"] is not None and c10["err"] is not None else None
    within0 = (c01["err"] - c00["err"]) if c01["err"] is not None and c00["err"] is not None else None
    return {
        "cells": cells,
        "n_discordant": c10["n"] + c01["n"],
        "ally_lift_given_writer_objected": within,
        "ally_lift_given_writer_silent": within0,
    }


def r0_code(out_dir: Path, model: str, scaffold: str, arm: str, item: str, idx: int,
            role_id: str) -> Optional[int]:
    """Published at-fault code of a seat's cached r0 verdict; the file name
    comes from the runner's own call_cache_path so the key stays in sync."""
    p = Path(out_dir) / rcd.call_cache_path(model, scaffold, arm, item, idx, "r0", role_id, R0_CAP).name
    if not p.exists():
        return None
    txt = json.loads(p.read_text()).get("output") or ""
    return code_response(extract_verdict(txt, rcd.INSTRUMENT), "published")


def role_lock(debates: Sequence[dict], *, out_dir: Path, model: str, scaffold: str,
              seats: Sequence[str] = SEATS, advocates: Sequence[str] = LIKE_PAIR,
              opponent: str = COUNTER, like_pair: tuple[str, str] = LIKE_PAIR) -> dict:
    at = {s: [0, 0] for s in seats}
    acc = {s: [0, 0] for s in seats}
    conc = [0, 0]
    for d in debates:
        codes = {}
        g = 1 if d["gold"] == "YTA" else 0
        for s in seats:
            c = r0_code(out_dir, model, scaffold, d["arm"], d["item"], d["sample_idx"], s)
            if c is None:
                continue
            codes[s] = c
            at[s][0] += c; at[s][1] += 1
            acc[s][0] += int(c == g); acc[s][1] += 1
        if like_pair[0] in codes and like_pair[1] in codes:
            conc[0] += int(codes[like_pair[0]] == codes[like_pair[1]]); conc[1] += 1
    p_at = {s: (at[s][0] / at[s][1] if at[s][1] else None) for s in seats}
    seat_acc = {s: (acc[s][0] / acc[s][1] if acc[s][1] else None) for s in seats}
    locks = {a: (abs(p_at[a] - p_at[opponent]) if None not in (p_at[a], p_at[opponent]) else None)
             for a in advocates}
    return {
        "p_at_fault": p_at, "seat_r0_accuracy": seat_acc,
        "n_r0_records": {s: at[s][1] for s in seats},
        "role_lock": locks,
        "like_pair_r0_gap": (abs(p_at[like_pair[0]] - p_at[like_pair[1]])
                             if None not in (p_at[like_pair[0]], p_at[like_pair[1]]) else None),
        "like_pair_r0_concordance": (conc[0] / conc[1]) if conc[1] else None,
        "n_like_pair_debates": conc[1],
    }


def g3_stake_concentration(votes: dict, *, draws: int, seed: int) -> dict:
    """R4 reject rate | own stake undermined minus | not, over stake seats;
    the 16.10 statistic on the votes as written (stake_undermined is against
    the GROUP verdict there, as in the runner)."""
    sv = []
    for (arm, item, idx), seat_votes in votes.items():
        for rid, v in seat_votes.items():
            if v["role_stake"] in ("at_fault", "not_at_fault") and v["stake_undermined"] is not None:
                sv.append({"item": item, "role": rid, "reject": v["reject"], "und": v["stake_undermined"]})
    if not sv:
        return {"applicable": False}
    by: dict = defaultdict(list)
    for v in sv:
        by[v["item"]].append(v)

    def stat(vs):
        u = [x["reject"] for x in vs if x["und"]]
        n = [x["reject"] for x in vs if not x["und"]]
        return None if not u or not n else sum(u) / len(u) - sum(n) / len(n)
    p, lo, hi = _boot(by, stat, draws=draws, seed=seed)
    per_seat = {}
    for rid in sorted({v["role"] for v in sv}):
        vs = [v for v in sv if v["role"] == rid]
        per_seat[rid] = {"n": len(vs), "delta": stat(vs)}
    return {"applicable": True, "n_stake_votes": len(sv), "delta": p, "lo": lo, "hi": hi,
            "per_seat": per_seat}


def against_interest_rates(votes: dict, seats: Sequence[str]) -> dict:
    """Per stake seat: P(objected | synthesis SERVES the stake) -- the
    against-interest rate -- beside P(objected | undermined)."""
    out = {}
    for rid in seats:
        serves = [0, 0]
        und = [0, 0]
        for seat_votes in votes.values():
            v = seat_votes.get(rid)
            if v is None or v["role_stake"] not in ("at_fault", "not_at_fault"):
                continue
            u = undermined_by(v["role_stake"], v["synthesis_verdict"])
            if u is None:
                continue
            tgt = und if u else serves
            tgt[0] += v["objected_r3"]; tgt[1] += 1
        if serves[1] or und[1]:
            out[rid] = {"n_serves": serves[1],
                        "p_object_against_interest": (serves[0] / serves[1]) if serves[1] else None,
                        "n_undermined": und[1],
                        "p_object_undermined": (und[0] / und[1]) if und[1] else None}
    return out


def transfer(debates: Sequence[dict], flag: Callable[[dict], Optional[bool]], *,
             comparator: dict, model: str, draws: int, seed: int) -> dict:
    """P(grok solo wrong | cell flagged) vs not; solo = cached standard
    majority-of-3 the deliberation never touched. A cell is flagged when the
    flag fired on a majority of its samples."""
    solo = make_rule_vendor(comparator, model, "standard")
    fl_c: dict = defaultdict(int)
    k_c: dict = defaultdict(int)
    gold = {}
    for d in debates:
        f = flag(d)
        if f is None:
            continue
        c = (d["arm"], d["item"])
        fl_c[c] += int(f); k_c[c] += 1; gold[c] = d["gold"]
    trec = []
    for c, g in gold.items():
        v = solo({"arm": c[0], "item": c[1]})
        ok = correct(v, g) if v is not None else None
        if ok is None:
            continue
        trec.append({"item": c[1], "w": int(not ok), "f": int(fl_c[c] * 2 > k_c[c])})
    by: dict = defaultdict(list)
    for x in trec:
        by[x["item"]].append(x)

    def stat(rs):
        f = [x for x in rs if x["f"]]; u = [x for x in rs if not x["f"]]
        return None if not f or not u else sum(x["w"] for x in f) / len(f) - sum(x["w"] for x in u) / len(u)
    p, lo, hi = _boot(by, stat, draws=draws, seed=seed) if by else (None, None, None)
    tf = [x for x in trec if x["f"]]; tu = [x for x in trec if not x["f"]]
    return {"n_cells": len(trec), "n_flagged_cells": len(tf),
            "p_wrong_flagged": (sum(x["w"] for x in tf) / len(tf)) if tf else None,
            "p_wrong_unflagged": (sum(x["w"] for x in tu) / len(tu)) if tu else None,
            "lift": p, "lo": lo, "hi": hi}


def s2_paired(debates: Sequence[dict], embodied_rows_path: Path, *, model: str,
              draws: int, seed: int) -> dict:
    """Coalition S2 accuracy paired against the embodied cell on the same
    (arm, item): against embodied sample 0 and against the embodied
    per-debate mean over its samples. Codable population on both sides."""
    emb: dict = defaultdict(dict)
    with open(embodied_rows_path) as f:
        for r in csv.DictReader(f):
            if model and r.get("model") != model:
                continue
            ok = verdict_correct(r.get("verdict", ""), r["gold_verdict"])
            if ok is None:
                continue
            emb[(r["arm"], r["item_id"])][int(r["sample_idx"])] = int(ok)
    co: dict = defaultdict(list)
    for d in debates:
        if d["s2_ok"] is not None:
            co[(d["arm"], d["item"])].append(int(d["s2_ok"]))
    items = {c[1] for c in co}
    emb_on_items = {c: v for c, v in emb.items() if c[1] in items}
    by0: dict = defaultdict(list)
    bym: dict = defaultdict(list)
    for c, vals in co.items():
        e = emb.get(c)
        if not e:
            continue
        cm = sum(vals) / len(vals)
        if 0 in e:
            by0[c[1]].append(cm - e[0])
        bym[c[1]].append(cm - sum(e.values()) / len(e))
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    p0, lo0, hi0 = _boot(by0, mean, draws=draws, seed=seed) if by0 else (None, None, None)
    pm, lom, him = _boot(bym, mean, draws=draws, seed=seed + 1) if bym else (None, None, None)
    co_all = [x for vals in co.values() for x in vals]
    emb_all = [x for e in emb_on_items.values() for x in e.values()]
    emb_s0 = [e[0] for e in emb_on_items.values() if 0 in e]
    return {
        "coalition": {"n": len(co_all), "accuracy": mean(co_all)},
        "embodied_all_samples": {"n": len(emb_all), "accuracy": mean(emb_all),
                                 "registered": EMBODIED_S2_REGISTERED},
        "embodied_sample0": {"n": len(emb_s0), "accuracy": mean(emb_s0)},
        "paired_vs_sample0": {"n_pairs": sum(len(v) for v in by0.values()),
                              "delta": p0, "lo": lo0, "hi": hi0},
        "paired_vs_mean": {"n_pairs": sum(len(v) for v in bym.values()),
                           "delta": pm, "lo": lom, "hi": him},
    }


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def run_guards(debates: Sequence[dict], votes: dict) -> dict:
    """The analyzer's own guards on the rows / votes (see GUARDS in the
    module docstring). `pass` is False on any breach; the report prints the
    literal GUARD FAILED and main() exits 4 without writing the JSON."""
    n = len(debates)
    codable = [d for d in debates if d["codable"]]
    seats_full = sum(1 for d in debates if all(d["objected"].get(s) is not None for s in SEATS))
    vote_rows = [v for key in ((d["arm"], d["item"], d["sample_idx"]) for d in debates)
                 for v in votes.get(key, {}).values()]
    n_votes = len(vote_rows)
    vote_parsed = sum(1 for v in vote_rows if v.get("vote_parsed"))
    mism = sum(1 for d in codable if d["n_against_interest_column"] is not None
               and d["n_against_interest_column"] != d["n_against_interest"])
    col_missing = sum(1 for d in codable if d["n_against_interest_column"] is None
                      and d["n_against_interest"] is not None)
    g = {
        "n_debates": n, "n_codable": len(codable),
        "codable_share": (len(codable) / n) if n else 0.0,
        "mean_n_r3_unparsed": (sum(d["n_r3_unparsed"] for d in debates) / n) if n else 0.0,
        "vote_parsed_share": (vote_parsed / n_votes) if n_votes else 0.0,
        "all_seats_present_share": (seats_full / n) if n else 0.0,
        "n_against_interest_mismatch": mism, "n_against_interest_column_missing": col_missing,
        "limits": {"codable_share": MIN_CODABLE_SHARE, "mean_n_r3_unparsed": MAX_MEAN_R3_UNPARSED,
                   "vote_parsed_share": MIN_VOTE_PARSED_SHARE,
                   "all_seats_present_share": MIN_SEATS_PRESENT_SHARE},
    }
    g["pass"] = bool(n and g["codable_share"] >= MIN_CODABLE_SHARE
                     and g["mean_n_r3_unparsed"] <= MAX_MEAN_R3_UNPARSED
                     and g["vote_parsed_share"] >= MIN_VOTE_PARSED_SHARE
                     and g["all_seats_present_share"] >= MIN_SEATS_PRESENT_SHARE
                     and mism == 0 and col_missing == 0)
    return g


def runner_record_guard(tag: str, out_dir: Path) -> dict:
    """The runner's own verdicts: <tag>_summary.json (round / outcome guards)
    and <tag>_completeness.json (every launched debate is a row). Missing =
    failed."""
    reasons: list[str] = []
    sp = out_dir / f"{tag}_summary.json"
    if sp.exists():
        try:
            s = json.loads(sp.read_text())
        except json.JSONDecodeError:
            s = {}
        rg, og = s.get("round_guard", {}), s.get("outcome_guard", {})
        if not (rg.get("pass", False) and og.get("pass", False)):
            reasons.append(f"summary: round guard {rg.get('pass')}, outcome guard {og.get('pass')}")
    else:
        reasons.append(f"{sp.name} missing: the run's guard verdict is unknown")
    cp = out_dir / f"{tag}_completeness.json"
    if cp.exists():
        try:
            c = json.loads(cp.read_text())
        except json.JSONDecodeError:
            c = {}
        if not c.get("pass", False):
            reasons.append(f"completeness: {c.get('n_missing')} of {c.get('n_expected')} launched "
                           "debates missing from the rows")
    else:
        reasons.append(f"{cp.name} missing: it is not known whether every launched debate is a row")
    return {"pass": not reasons, "reasons": reasons}


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------

def analyse(*, rows_path: Path, votes_path: Path, out_dir: Path, model: str,
            scaffold: str = SCAFFOLD, embodied_rows_path: Optional[Path] = None,
            embodied_votes_path: Optional[Path] = None,
            comparator_path: Optional[Path] = None, keep_items: Optional[set] = None,
            draws: int = 3000, seed: int = 29) -> Optional[dict]:
    if not rows_path.exists() or not votes_path.exists():
        return None
    debates = load_debates(rows_path, votes_path, model, keep_items, seats=SEATS)
    if not debates:
        return None
    votes = load_votes(votes_path, model, keep_items)
    n = len(debates)
    codable = [d for d in debates if d["codable"]]
    rates = per_seat_objection_rates(votes_path, model, keep_items)
    loc = localisation_n(debates, rates, SEATS)

    mism = [d for d in debates if d["n_against_interest_column"] is not None
            and d["n_against_interest_column"] != d["n_against_interest"]]
    missing_col = sum(1 for d in debates if d["n_against_interest_column"] is None
                      and d["n_against_interest"] is not None)

    lifts = {name: lift(debates, fn, draws=draws, seed=seed + i)
             for i, (name, fn) in enumerate(FLAGS.items())}
    res: dict = {
        "tag": rows_path.name.rsplit("_rows.csv", 1)[0], "model": model, "seats": list(SEATS),
        "n_debates": n, "n_items": len({d["item"] for d in debates}), "n_codable": len(codable),
        "n_samples": len({d["sample_idx"] for d in debates}),
        "role_lock": role_lock(debates, out_dir=out_dir, model=model, scaffold=scaffold),
        "objection_phi": {
            "like_pair": pair_phi(debates, *LIKE_PAIR, draws=draws, seed=seed + 11),
            "opposed_pairs": [pair_phi(debates, a, b, draws=draws, seed=seed + 12 + i)
                              for i, (a, b) in enumerate(OPPOSED_PAIRS)],
            "neutral_pairs": [pair_phi(debates, a, NEUTRAL, draws=draws, seed=seed + 15 + i)
                              for i, a in enumerate(LIKE_PAIR)],
        },
        "objection": {"per_seat": rates["per_seat"], "pooled": rates["pooled"],
                      "localisation": loc,
                      "localisation_excess": loc["localisation_excess"],
                      "fire_rate_ge2_of_4": loc["fire_observed"],
                      "fire_independence": loc["fire_independence"]},
        "lift": lifts,
        "pooled_minus_three_seat": {
            "ge2": paired_lift_delta(debates, flag_three_seat, flag_pooled, draws=draws, seed=seed + 21),
            "ge3": paired_lift_delta(debates, flag_three_seat, flag_pooled_3, draws=draws, seed=seed + 22),
        },
        "ally_information": ally_information(debates),
        "against_interest": {
            "column_agreement": {"n_checked": sum(1 for d in debates if d["n_against_interest_column"] is not None),
                                 "n_mismatch": len(mism), "n_column_missing": missing_col,
                                 "mismatches": [(d["arm"], d["item"], d["sample_idx"],
                                                 d["n_against_interest_column"], d["n_against_interest"])
                                                for d in mism[:20]]},
            "count_distribution": {k: sum(1 for d in debates if d["n_against_interest"] == k) for k in (0, 1, 2)},
            "n_excluded_noncodable_synthesis": sum(1 for d in debates if d["n_against_interest"] is None),
            "per_seat": against_interest_rates(votes, STAKE_SEATS),
            "role_composition": {rid: sum(1 for d in debates if rid in d["against_interest_roles"])
                                 for rid in STAKE_SEATS},
            "lift": lifts["against_interest_ge1"],
        },
        "g3": g3_stake_concentration(votes, draws=draws, seed=seed + 31),
        "guards": {**run_guards(debates, votes),
                   "p_noncodable": 1 - len(codable) / n,
                   "p_noncodable_given_fired": {name: l["p_noncodable_given_fired"] for name, l in lifts.items()}},
    }
    if comparator_path is not None and Path(comparator_path).exists():
        comp = load_comparator(Path(comparator_path))
        res["transfer_grok_solo"] = {
            name: transfer(debates, fn, comparator=comp, model=model, draws=draws, seed=seed + 41 + i)
            for i, (name, fn) in enumerate(FLAGS.items())}
    if embodied_rows_path is not None and Path(embodied_rows_path).exists():
        res["s2_paired"] = s2_paired(debates, Path(embodied_rows_path), model=model, draws=draws, seed=seed + 51)
        if embodied_votes_path is not None and Path(embodied_votes_path).exists():
            emb = load_debates(Path(embodied_rows_path), Path(embodied_votes_path), model,
                               {d["item"] for d in debates}, seats=EMBODIED_SEATS)
            ev = load_votes(Path(embodied_votes_path), model, {d["item"] for d in debates})
            res["embodied_baseline"] = {
                "n_debates": len(emb),
                "against_interest": {
                    "lift": lift(emb, flag_against_interest, draws=draws, seed=seed + 61),
                    "per_seat": against_interest_rates(ev, ("writer_advocate", "counterparty")),
                    "count_distribution": {k: sum(1 for d in emb if d["n_against_interest"] == k) for k in (0, 1, 2)},
                },
                "ge2_of_3": lift(emb, flag_embodied_ge2, draws=draws, seed=seed + 62),
                "objection_phi_opposed": pair_phi(emb, "writer_advocate", "counterparty", draws=draws, seed=seed + 63),
            }
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def _ci(d: dict, lo="lo", hi="hi"):
    return f"[{_f(d.get(lo))}, {_f(d.get(hi))}]"


def print_report(res: dict) -> None:
    rl, ph, ai = res["role_lock"], res["objection_phi"], res["against_interest"]
    print("\n" + "=" * 90)
    print(f"ADDENDUM 16.19 -- COALITION ADVOCATES   {res['tag']}   model {res['model']}")
    print("=" * 90)
    print(f"  debates {res['n_debates']}  items {res['n_items']}  samples {res['n_samples']}  "
          f"codable {res['n_codable']}  seats {res['seats']}")

    print("\n  1. ROLE-LOCK (r0)                              embodied/on: 0.971")
    for s in SEATS:
        print(f"     {s:<24} P(at_fault)={_f(rl['p_at_fault'][s], '.3f')}  "
              f"r0 acc={_f(rl['seat_r0_accuracy'][s], '.3f')}  n={rl['n_r0_records'][s]}")
    for a, v in rl["role_lock"].items():
        print(f"     lock {a:<20} {_f(v, '.3f')}")
    print(f"     like pair r0 gap {_f(rl['like_pair_r0_gap'], '.3f')}   "
          f"concordance {_f(rl['like_pair_r0_concordance'], '.3f')}  (n={rl['n_like_pair_debates']})"
          "   <- PRE-communication (independence) reading")

    lp = ph["like_pair"]
    print("\n  2. OBJECTION PHI                                K-M: like pair near +1")
    print("     (POST-communication: edges on, each has read the other's r0 and r1 before R3; "
          "compare with the r0 concordance above)")
    print(f"     like pair   writer x ally      phi {_f(lp['phi'])} {_ci(lp)}  n={lp['n']}  "
          f"table both/w-only/a-only/neither = {lp.get('both')}/{lp.get('first_only')}/"
          f"{lp.get('second_only')}/{lp.get('neither')}")
    print(f"        P(ally objects | writer objects) {_f(lp.get('p_second_given_first'), '.3f')}   "
          f"P(ally objects | writer silent) {_f(lp.get('p_second_given_not_first'), '.3f')}   "
          f"agreement {_f(lp.get('agreement'), '.3f')}")
    for t in ph["opposed_pairs"]:
        print(f"     opposed     {t['seats'][0]:<18} x {t['seats'][1]:<16} phi {_f(t['phi'])} {_ci(t)}  n={t['n']}")
    for t in ph["neutral_pairs"]:
        print(f"     vs neutral  {t['seats'][0]:<18} x {t['seats'][1]:<16} phi {_f(t['phi'])} {_ci(t)}  n={t['n']}")

    ob = res["objection"]
    print("\n  3. FLAGS AND LIFT (codable S2 error, fired - not)   K-M: pooling adds no lift")
    print(f"     per-seat objection " + "  ".join(f"{s.split('_')[0][:7]}={_f(v, '.3f')}"
                                                  for s, v in ob["per_seat"].items()))
    print(f"     localisation excess (P-binom, 4 seats) {_f(ob['localisation_excess'])}   "
          f"fire P(>=2 of 4) {_f(ob['fire_rate_ge2_of_4'], '.3f')} (indep {_f(ob['fire_independence'], '.3f')})")
    print(f"     {'flag':<22}{'fire':>7}{'n fired':>9}{'err|fired':>11}{'err|not':>9}{'lift':>8}  CI               powered")
    for name, l in res["lift"].items():
        print(f"     {name:<22}{_f(l['fire_rate'], '.3f'):>7}{l['n_fired']:>9}{_f(l['err_fired'], '.3f'):>11}"
              f"{_f(l['err_unfired'], '.3f'):>9}{_f(l['delta']):>8}  {_ci(l):<18}"
              f"{'yes' if l['powered'] else 'NO (<' + str(MIN_FIRED) + ')'}")
    pm = res["pooled_minus_three_seat"]
    print(f"     pooled(>=2) - three-seat(>=2)  {_f(pm['ge2']['delta'])} {_ci(pm['ge2'])}   "
          f"pooled(>=3) - three-seat  {_f(pm['ge3']['delta'])} {_ci(pm['ge3'])}")
    info = res["ally_information"]
    print("     information test: err by (writer, ally) objection  "
          + "  ".join(f"{k}: {_f(v['err'], '.3f')} (n={v['n']})" for k, v in info["cells"].items()))
    print(f"        discordant debates {info['n_discordant']}   ally lift | writer objected "
          f"{_f(info['ally_lift_given_writer_objected'])}   | writer silent "
          f"{_f(info['ally_lift_given_writer_silent'])}")

    l = ai["lift"]
    print("\n  4. AGAINST-INTEREST OBJECTIONS (generalised sensor)")
    ca = ai["column_agreement"]
    flag = "" if ca["n_mismatch"] == 0 else "   *** MISMATCH WITH n_against_interest_r3 COLUMN ***"
    print(f"     column agreement: {ca['n_checked']} checked, {ca['n_mismatch']} mismatch, "
          f"{ca['n_column_missing']} column missing{flag}")
    print(f"     count distribution {ai['count_distribution']}   excluded (non-codable synthesis) "
          f"{ai['n_excluded_noncodable_synthesis']}   composition {ai['role_composition']}")
    for s, v in ai["per_seat"].items():
        print(f"     {s:<24} P(object | serves stake) {_f(v['p_object_against_interest'], '.3f')} "
              f"(n={v['n_serves']})   P(object | undermined) {_f(v['p_object_undermined'], '.3f')} (n={v['n_undermined']})")
    print(f"     fire {_f(l['fire_rate'], '.3f')}  n fired {l['n_fired']}  lift {_f(l['delta'])} {_ci(l)}  "
          f"{'powered' if l['powered'] else 'UNDERPOWERED'}")
    if "transfer_grok_solo" in res:
        for name, t in res["transfer_grok_solo"].items():
            print(f"     transfer {name:<22} P(solo wrong|flagged) {_f(t['p_wrong_flagged'], '.3f')}  "
                  f"| not {_f(t['p_wrong_unflagged'], '.3f')}  lift {_f(t['lift'])} {_ci(t)}  "
                  f"cells {t['n_flagged_cells']}/{t['n_cells']}")
    if "embodied_baseline" in res:
        eb = res["embodied_baseline"]
        el = eb["against_interest"]["lift"]
        print(f"     embodied baseline (same items, {eb['n_debates']} debates): against-interest fire "
              f"{_f(el['fire_rate'], '.3f')}  lift {_f(el['delta'])} {_ci(el)}   >=2-of-3 lift "
              f"{_f(eb['ge2_of_3']['delta'])}   opposed phi {_f(eb['objection_phi_opposed']['phi'])}")
        for s, v in eb["against_interest"]["per_seat"].items():
            print(f"        {s:<21} P(object | serves) {_f(v['p_object_against_interest'], '.3f')} (n={v['n_serves']})")

    g = res["g3"]
    print(f"\n  G3 stake concentration (3 stake seats) {_f(g.get('delta'))} {_ci(g)}  n={g.get('n_stake_votes')}"
          if g["applicable"] else "\n  G3 n/a")
    if g["applicable"]:
        print("     per seat " + "  ".join(f"{s.split('_')[0][:7]}={_f(v['delta'])}" for s, v in g["per_seat"].items()))

    if "s2_paired" in res:
        s = res["s2_paired"]
        print("\n  5. S2 ACCURACY, PAIRED vs embodied/on")
        print(f"     coalition {_f(s['coalition']['accuracy'], '.3f')} (n={s['coalition']['n']})   "
              f"embodied all samples {_f(s['embodied_all_samples']['accuracy'], '.3f')} "
              f"(n={s['embodied_all_samples']['n']}; registered {EMBODIED_S2_REGISTERED})   "
              f"embodied sample 0 {_f(s['embodied_sample0']['accuracy'], '.3f')}")
        print(f"     paired delta vs sample 0 {_f(s['paired_vs_sample0']['delta'])} {_ci(s['paired_vs_sample0'])} "
              f"(pairs {s['paired_vs_sample0']['n_pairs']})   vs mean {_f(s['paired_vs_mean']['delta'])} "
              f"{_ci(s['paired_vs_mean'])}")
    lpc = rl.get("like_pair_r0_concordance")
    lpp = lp.get("phi")
    if lpc is not None and lpp is not None:
        print(f"\n  like pair: r0 verdict concordance {lpc:.3f} (PRE-communication, the independence "
              f"reading) vs R3 objection phi {lpp:+.3f} (POST-communication, edges on); if these "
              "diverge, the clean K-M test is an edges-off coalition cell (separate registration)")
    gd = res["guards"]
    print(f"\n  guards: codable {gd['codable_share']:.3f} (>= {MIN_CODABLE_SHARE})   mean n_r3_unparsed "
          f"{gd['mean_n_r3_unparsed']:.4f} (<= {MAX_MEAN_R3_UNPARSED})   vote parsed {gd['vote_parsed_share']:.3f} "
          f"(>= {MIN_VOTE_PARSED_SHARE})   all four seats {gd['all_seats_present_share']:.3f} "
          f"(>= {MIN_SEATS_PRESENT_SHARE})   count mismatches {gd['n_against_interest_mismatch']}   "
          f"-> {'PASS' if gd['pass'] else 'GUARD FAILED'}")
    if "runner_record_guard" in res:
        rr = res["runner_record_guard"]
        print(f"  runner records: {'PASS' if rr['pass'] else 'GUARD FAILED -- ' + '; '.join(rr['reasons'])}")
    if not gd["pass"] or not res.get("runner_record_guard", {"pass": True})["pass"]:
        print("\n  GUARD FAILED -- the numbers above are NOT readable as behaviour.")


# ---------------------------------------------------------------------------
# Selftest on synthetic votes
# ---------------------------------------------------------------------------

def _synth_cell(td: Path, *, n_items: int, ally_mode: str = "copy", model: str = MODEL) -> dict:
    """A planted four-seat cell plus an embodied cell and a solo comparator.

    items i0..i{n-1}; gold YTA on even i; two arms; sample 0.
      synthesis  = gold, except WRONG (flipped) on i % 5 == 0; S2 = synthesis.
      writer     objects when undermined by the synthesis, and AGAINST
                 INTEREST when the synthesis is wrong and not-at-fault.
      ally       'copy' -> = writer (phi +1); 'flip' -> = 1 - writer (phi -1).
      counterparty objects when undermined, and against interest when the
                 synthesis is wrong and at-fault.
      neutral    objects only on wrong, even-i debates (keeps every flag exact).
      R4 reject  iff undermined by the group verdict (G3 exactly +1).
    So: against-interest fires iff synthesis wrong (fire 0.2, lift +1.0);
    three-seat >=2 fires iff wrong (lift +1.0); pooled >=2 also fires on every
    correct at-fault synthesis (lift < 1); pooled >=3 iff wrong.
      r0 caches: writer/ally NTA, counterparty YTA, neutral = gold.
      embodied rows/votes (three embodied seats): samples 0,1; sample s
                 wrong on i % 4 == s; stake seats object when undermined and
                 against interest iff wrong (baseline sensor: fire 0.25, lift +1).
      comparator: solo standard k3 wrong iff i % 5 == 0.
    """
    rows, votes, emb_rows, emb_votes, comp = [], [], [], [], []
    for i in range(n_items):
        item = f"i{i:03d}"
        gold = "YTA" if i % 2 == 0 else "NTA"
        wrong = i % 5 == 0
        syn = ("NTA" if gold == "YTA" else "YTA") if wrong else gold
        for arm in ("third_person", "as_asker"):
            und_w = undermined_by("not_at_fault", syn)
            und_c = undermined_by("at_fault", syn)
            obj = {WRITER: int(und_w or (wrong and syn in NOT_AT_FAULT_VERDICTS)),
                   COUNTER: int(und_c or (wrong and syn in AT_FAULT_VERDICTS)),
                   NEUTRAL: int(wrong and i % 2 == 0)}
            obj[ALLY] = obj[WRITER] if ally_mode == "copy" else 1 - obj[WRITER]
            stake = {WRITER: "not_at_fault", ALLY: "not_at_fault", COUNTER: "at_fault", NEUTRAL: "none"}
            sv = {}
            for s in SEATS:
                u = undermined_by(stake[s], syn)
                sv[s] = {"role_id": s, "role_stake": stake[s], "objected_r3": obj[s],
                         "r3_label_parsed": 1, "reject": int(bool(u)), "vote_parsed": 1,
                         "stake_undermined": None if u is None else int(u), "synthesis_verdict": syn}
                votes.append({"model": model, "scaffold": SCAFFOLD, "arm": arm, "item_id": item,
                              "sample_idx": 0, "gold_verdict": gold, "role_id": s, "role_stake": stake[s],
                              "group_verdict": syn, "synthesis_verdict": syn,
                              "r3_label": "ACCEPT_WITH_MODIFICATION" if obj[s] else "ACCEPT",
                              "r3_label_parsed": 1, "objected_r3": obj[s],
                              "vote": "REJECT" if u else "ACCEPT", "reject": int(bool(u)),
                              "vote_parsed": 1, "stake_undermined": "" if u is None else int(u)})
            n_ai, ai_roles = against_interest(sv, syn, SEATS)
            rows.append({"model": model, "scaffold": "deliberation_ncot", "arm": arm, "item_id": item,
                         "sample_idx": 0, "gold_verdict": gold, "verdict": syn, "synthesis_verdict": syn,
                         "n_objectors": sum(obj.values()), "n_r3_unparsed": 0, "n_agents": 4,
                         "n_against_interest_r3": "" if n_ai is None else n_ai,
                         "against_interest_roles": "|".join(ai_roles)})
            r0v = {WRITER: "NTA", ALLY: "NTA", COUNTER: "YTA", NEUTRAL: gold}
            for s in SEATS:
                p = td / rcd.call_cache_path(model, SCAFFOLD, arm, item, 0, "r0", s, R0_CAP).name
                p.write_text(json.dumps({"output": f"Section 5 - Decision.\nVERDICT: {r0v[s]}"}))
            for s_idx in (0, 1):
                ewrong = i % 4 == s_idx
                ev = ("NTA" if gold == "YTA" else "YTA") if ewrong else gold
                estake = {"writer_advocate": "not_at_fault", "counterparty": "at_fault",
                          "neutral_adjudicator": "none"}
                eobj = {}
                for es, st in estake.items():
                    u = undermined_by(st, ev)
                    # objects when undermined, and against interest iff the synthesis is wrong
                    eobj[es] = int(bool(u) or (ewrong and st != "none"))
                    emb_votes.append({"model": model, "scaffold": SCAFFOLD, "arm": arm, "item_id": item,
                                      "sample_idx": s_idx, "gold_verdict": gold, "role_id": es,
                                      "role_stake": st, "group_verdict": ev, "synthesis_verdict": ev,
                                      "r3_label": "ACCEPT_WITH_MODIFICATION" if eobj[es] else "ACCEPT",
                                      "r3_label_parsed": 1, "objected_r3": eobj[es],
                                      "vote": "REJECT" if u else "ACCEPT", "reject": int(bool(u)),
                                      "vote_parsed": 1, "stake_undermined": "" if u is None else int(u)})
                emb_rows.append({"model": model, "scaffold": "deliberation_ncot", "arm": arm,
                                 "item_id": item, "sample_idx": s_idx, "gold_verdict": gold,
                                 "verdict": ev, "synthesis_verdict": ev, "n_objectors": sum(eobj.values()),
                                 "n_r3_unparsed": 0})
            solo = ("NTA" if gold == "YTA" else "YTA") if wrong else gold
            for k in range(3):
                comp.append({"model": model, "scaffold": "standard", "arm": arm, "item_id": item,
                             "sample_idx": k, "verdict": solo})

    def write(path, recs):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]))
            w.writeheader(); w.writerows(recs)
    paths = {"rows": td / f"{TAG}_rows.csv", "votes": td / f"{TAG}_votes.csv",
             "emb_rows": td / f"{EMBODIED_TAG}_rows.csv", "emb_votes": td / f"{EMBODIED_TAG}_votes.csv",
             "comp": td / "comparator.csv"}
    write(paths["rows"], rows); write(paths["votes"], votes)
    write(paths["emb_rows"], emb_rows); write(paths["emb_votes"], emb_votes); write(paths["comp"], comp)
    return paths


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # ---- stake logic -------------------------------------------------------
    check("undermined_by: not_at_fault stake is undermined by YTA/ESH, served by NTA/NAH, None otherwise",
          undermined_by("not_at_fault", "YTA") is True and undermined_by("not_at_fault", "ESH") is True
          and undermined_by("not_at_fault", "NTA") is False and undermined_by("not_at_fault", "NAH") is False
          and undermined_by("not_at_fault", rcd.UNRESOLVED) is None
          and undermined_by("not_at_fault", rcd.NOVERDICT) is None and undermined_by("not_at_fault", "") is None)
    check("undermined_by: at_fault stake mirrors; a stake of none is never undermined",
          undermined_by("at_fault", "NTA") is True and undermined_by("at_fault", "YTA") is False
          and undermined_by("none", "YTA") is False and undermined_by("none", "NTA") is False)
    check("undermined_by agrees with rcd.stake_undermined on the embodied seats for every verdict",
          all(undermined_by(rcd.ROLE_BY_ID[rid].stake, v) == rcd.stake_undermined(rid, v)
              for rid in EMBODIED_SEATS for v in ("YTA", "NTA", "ESH", "NAH", rcd.UNRESOLVED, rcd.NOVERDICT)))
    sv = {WRITER: {"role_stake": "not_at_fault", "objected_r3": 1},
          ALLY: {"role_stake": "not_at_fault", "objected_r3": 1},
          COUNTER: {"role_stake": "at_fault", "objected_r3": 1},
          NEUTRAL: {"role_stake": "none", "objected_r3": 1}}
    check("against_interest: NTA synthesis -> writer + ally (2), not the counterparty, never the neutral",
          against_interest(sv, "NTA", SEATS) == (2, (WRITER, ALLY)))
    check("against_interest: YTA synthesis -> counterparty alone (1)",
          against_interest(sv, "YTA", SEATS) == (1, (COUNTER,)))
    check("against_interest: non-codable synthesis -> None (excluded, not 0)",
          against_interest(sv, rcd.UNRESOLVED, SEATS) == (None, ()))
    check("against_interest: the count can never exceed 2 (the two sides cannot both be served)",
          max(against_interest(sv, v, SEATS)[0] or 0 for v in ("YTA", "NTA", "ESH", "NAH")) == 2)
    check("against_interest reads the embodied seats through the same code",
          against_interest({"writer_advocate": sv[WRITER], "counterparty": sv[COUNTER]}, "NTA", EMBODIED_SEATS)
          == (1, ("writer_advocate",)))

    # ---- phi ---------------------------------------------------------------
    t = phi_table([(1, 1)] * 5 + [(0, 0)] * 7)
    check("phi_table: identical series -> phi +1, agreement 1, conditionals 1/0",
          abs(t["phi"] - 1) < 1e-12 and t["agreement"] == 1
          and t["p_second_given_first"] == 1 and t["p_second_given_not_first"] == 0)
    check("phi_table: opposite series -> phi -1", abs(phi_table([(1, 0)] * 4 + [(0, 1)] * 6)["phi"] + 1) < 1e-12)
    check("phi_table: balanced 2x2 -> phi exactly 0",
          phi_table([(1, 1), (1, 0), (0, 1), (0, 0)] * 3)["phi"] == 0)
    check("phi_table: a constant marginal -> phi None (undefined), not 0",
          phi_table([(1, 1), (1, 0)])["phi"] is None and phi_table([])["phi"] is None)

    # ---- localisation over four seats ------------------------------------------
    rates = {"per_seat": {WRITER: 0.4, ALLY: 0.4, COUNTER: 0.7, NEUTRAL: 0.1}, "pooled": 0.4}
    fake = [{"n_objectors": k} for k in (0, 1, 1, 2, 3, 4, 4, 4)]
    loc = localisation_n(fake, rates, SEATS)
    check("localisation_n: observed and null both run 0..4 and each sums to 1",
          sorted(loc["observed"]) == [0, 1, 2, 3, 4]
          and abs(sum(loc["observed"].values()) - 1) < 1e-12
          and abs(sum(loc["independence_poisson_binomial"].values()) - 1) < 1e-12)
    check("localisation_n: fire_observed counts k=2,3,4 (a three-seat range(4) would drop k=4)",
          abs(loc["fire_observed"] - 5 / 8) < 1e-12 and abs(loc["observed"][4] - 3 / 8) < 1e-12
          and abs(loc["fire_independence"] - sum(poisson_binomial([0.4, 0.4, 0.7, 0.1])[2:])) < 1e-12)

    # ---- lift on hand-built debates ---------------------------------------------
    def mk(i, ok, fired, codable=True):
        return {"item": f"i{i}", "codable": codable, "s2_ok": ok, "_f": fired}
    hand = [mk(i, False, True) for i in range(70)] + [mk(i + 100, True, False) for i in range(70)] \
        + [mk(999, None, True, codable=False)]
    l = lift(hand, lambda d: d["_f"], draws=50, seed=1)
    check("lift: planted err|fired 1, err|not 0 -> delta +1.0, powered at 71 fired, CI [1,1], "
          "non-codable fired debate counted in fire rate and in P(non-codable|fired)",
          l["delta"] == 1.0 and l["lo"] == 1.0 and l["hi"] == 1.0 and l["powered"]
          and l["n_fired"] == 71 and l["n_fired_codable"] == 70
          and abs(l["p_noncodable_given_fired"] - 1 / 71) < 1e-12)
    l2 = lift(hand[:20] + hand[70:90], lambda d: d["_f"], draws=10, seed=1)
    check("lift: 20 fired is reported UNPOWERED (< 60) with the point estimate still shown",
          not l2["powered"] and l2["delta"] == 1.0)
    check("lift: a flag returning None excludes the debate",
          lift(hand, lambda d: None if d["item"] == "i0" else d["_f"], draws=10, seed=1)["n_excluded"] == 1)

    # ---- the synthetic cell ------------------------------------------------------
    real_out = rcd.OUT_DIR
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        paths = _synth_cell(td, n_items=200, ally_mode="copy")
        res = analyse(rows_path=paths["rows"], votes_path=paths["votes"], out_dir=td, model=MODEL,
                      embodied_rows_path=paths["emb_rows"], embodied_votes_path=None,
                      comparator_path=paths["comp"], draws=200, seed=3)
        check("synthetic cell analysed: 400 debates, 200 items, 1 sample, all codable",
              res is not None and res["n_debates"] == 400 and res["n_items"] == 200
              and res["n_samples"] == 1 and res["n_codable"] == 400)
        rl = res["role_lock"]
        check("role-lock: writer 1.0 and ally 1.0 against the counterparty from the planted r0 caches",
              rl["role_lock"][WRITER] == 1.0 and rl["role_lock"][ALLY] == 1.0
              and rl["p_at_fault"][COUNTER] == 1.0 and rl["p_at_fault"][NEUTRAL] == 0.5)
        check("role-lock: like pair r0 gap 0, concordance 1.0, neutral r0 accuracy 1.0, 400 records per seat",
              rl["like_pair_r0_gap"] == 0 and rl["like_pair_r0_concordance"] == 1.0
              and rl["seat_r0_accuracy"][NEUTRAL] == 1.0 and rl["n_r0_records"][WRITER] == 400)
        lp = res["objection_phi"]["like_pair"]
        check("objection phi: planted copy -> like pair phi +1 with CI [1, 1] on 400 debates",
              abs(lp["phi"] - 1) < 1e-12 and lp["lo"] == 1.0 and lp["hi"] == 1.0 and lp["n"] == 400
              and lp["first_only"] == 0 and lp["second_only"] == 0)
        op = res["objection_phi"]["opposed_pairs"]
        check("objection phi: the two opposed pairs are identical to each other (ally = writer) and negative",
              op[0]["phi"] == op[1]["phi"] and op[0]["phi"] < 0)
        L = res["lift"]
        check("lift: three-seat >=2 fires iff the synthesis is wrong -> fire 0.2, lift +1.0, powered (80 fired)",
              abs(L["three_seat_ge2"]["fire_rate"] - 0.2) < 1e-12 and L["three_seat_ge2"]["delta"] == 1.0
              and L["three_seat_ge2"]["n_fired"] == 80 and L["three_seat_ge2"]["powered"])
        check("lift: pooled >=2 also fires on every correct at-fault synthesis -> higher fire rate, LOWER lift",
              L["pooled_ge2"]["fire_rate"] > 0.2 and L["pooled_ge2"]["delta"] < 1.0
              and L["pooled_ge2"]["err_unfired"] == 0.0)
        check("lift: pooled >=3 fires iff wrong -> lift +1.0 again",
              L["pooled_ge3"]["delta"] == 1.0 and L["pooled_ge3"]["n_fired"] == 80)
        pm = res["pooled_minus_three_seat"]
        check("paired lift delta (pooled - three-seat): negative for >=2 with CI below 0, exactly 0 for >=3",
              pm["ge2"]["delta"] < 0 and pm["ge2"]["hi"] < 0 and pm["ge3"]["delta"] == 0.0)
        info = res["ally_information"]
        check("information test: zero discordant debates when phi = +1, ally lifts undefined",
              info["n_discordant"] == 0 and info["ally_lift_given_writer_objected"] is None
              and info["ally_lift_given_writer_silent"] is None)
        ai = res["against_interest"]
        check("against-interest: fires iff the synthesis is wrong -> fire 0.2, lift +1.0, CI [1,1], powered",
              abs(ai["lift"]["fire_rate"] - 0.2) < 1e-12 and ai["lift"]["delta"] == 1.0
              and ai["lift"]["lo"] == 1.0 and ai["lift"]["powered"])
        check("against-interest: count 2 on wrong not-at-fault syntheses (writer + ally), 1 on wrong at-fault",
              ai["count_distribution"] == {0: 320, 1: 40, 2: 40}
              and ai["role_composition"] == {WRITER: 40, ALLY: 40, COUNTER: 40})
        check("against-interest: the runner's n_against_interest_r3 column agrees on every row",
              ai["column_agreement"]["n_checked"] == 400 and ai["column_agreement"]["n_mismatch"] == 0
              and ai["column_agreement"]["n_column_missing"] == 0)
        ps = ai["per_seat"]
        check("against-interest per seat: writer/ally object on 40 of the 200 syntheses that serve them, "
              "the counterparty on 40 of its 200; every seat objects on 100% of undermined ones",
              all(ps[s]["n_serves"] == 200 and ps[s]["n_undermined"] == 200
                  and abs(ps[s]["p_object_against_interest"] - 0.2) < 1e-12
                  and ps[s]["p_object_undermined"] == 1.0 for s in STAKE_SEATS))
        tr = res["transfer_grok_solo"]
        check("transfer: solo wrong iff flagged by the against-interest flag -> lift +1.0 on 400 cells",
              tr["against_interest_ge1"]["lift"] == 1.0 and tr["against_interest_ge1"]["n_cells"] == 400
              and tr["against_interest_ge1"]["n_flagged_cells"] == 80)
        g = res["g3"]
        check("G3: reject iff undermined -> +1.0 pooled and per seat over the three stake seats",
              g["applicable"] and g["delta"] == 1.0 and sorted(g["per_seat"]) == sorted(STAKE_SEATS)
              and all(v["delta"] == 1.0 for v in g["per_seat"].values()))
        s2 = res["s2_paired"]
        check("S2 paired: coalition 0.80, embodied sample 0 0.75, all samples 0.75 -> paired deltas +0.05 with CIs",
              abs(s2["coalition"]["accuracy"] - 0.8) < 1e-12 and abs(s2["embodied_sample0"]["accuracy"] - 0.75) < 1e-12
              and abs(s2["embodied_all_samples"]["accuracy"] - 0.75) < 1e-12
              and abs(s2["paired_vs_sample0"]["delta"] - 0.05) < 1e-12 and abs(s2["paired_vs_mean"]["delta"] - 0.05) < 1e-12
              and s2["paired_vs_sample0"]["n_pairs"] == 400 and s2["paired_vs_sample0"]["lo"] is not None)
        check("guards: no unparsed R3 label, nothing non-codable; run_guards PASS with every share at 1.0",
              res["guards"]["mean_n_r3_unparsed"] == 0 and res["guards"]["p_noncodable"] == 0
              and res["guards"]["pass"] and res["guards"]["codable_share"] == 1.0
              and res["guards"]["vote_parsed_share"] == 1.0 and res["guards"]["all_seats_present_share"] == 1.0)
        # an UNRESOLVED flood: every R3 label / R4 vote / group verdict non-codable
        flood_rows = [dict(r, verdict=rcd.UNRESOLVED, synthesis_verdict=rcd.UNRESOLVED)
                      for r in csv.DictReader(open(paths["rows"]))]
        flood_votes = [dict(v, vote=rcd.UNRESOLVED, vote_parsed=0, r3_label=rcd.UNRESOLVED, r3_label_parsed=0,
                            objected_r3=0, synthesis_verdict=rcd.UNRESOLVED, group_verdict=rcd.UNRESOLVED)
                       for v in csv.DictReader(open(paths["votes"]))]
        (td / "fl").mkdir()
        for name, recs in (("rows", flood_rows), ("votes", flood_votes)):
            with open(td / "fl" / f"{TAG}_{name}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
        res_fl = analyse(rows_path=td / "fl" / f"{TAG}_rows.csv", votes_path=td / "fl" / f"{TAG}_votes.csv",
                         out_dir=td, model=MODEL, draws=20, seed=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(res_fl)
        check("an UNRESOLVED flood (100% non-codable, no vote parsed) FAILS run_guards and the report "
              "prints the literal GUARD FAILED",
              res_fl is not None and not res_fl["guards"]["pass"] and res_fl["guards"]["codable_share"] == 0.0
              and res_fl["guards"]["vote_parsed_share"] == 0.0 and "GUARD FAILED" in buf.getvalue())
        rr_missing = runner_record_guard(TAG, td)
        (td / f"{TAG}_summary.json").write_text(json.dumps(
            {"round_guard": {"pass": True}, "outcome_guard": {"pass": False}}))
        (td / f"{TAG}_completeness.json").write_text(json.dumps({"pass": True}))
        rr_failed = runner_record_guard(TAG, td)
        (td / f"{TAG}_summary.json").write_text(json.dumps(
            {"round_guard": {"pass": True}, "outcome_guard": {"pass": True}}))
        rr_ok = runner_record_guard(TAG, td)
        (td / f"{TAG}_completeness.json").write_text(json.dumps({"pass": False, "n_missing": 2, "n_expected": 400}))
        rr_short = runner_record_guard(TAG, td)
        check("runner_record_guard: missing summary + sidecar -> failed (two reasons); a failed outcome guard "
              "-> failed; both intact -> pass; a completeness shortfall -> failed and named",
              not rr_missing["pass"] and len(rr_missing["reasons"]) == 2
              and not rr_failed["pass"] and "outcome guard False" in rr_failed["reasons"][0]
              and rr_ok["pass"] and not rr_short["pass"] and "2 of 400" in rr_short["reasons"][0])
        # main() end to end in the tempdir: refuses on the runner's records, then reads
        real_outdir = globals()["OUT_DIR"]
        globals()["OUT_DIR"] = td
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_m1 = main(["--n-boot", "20", "--out", str(td / "co_analysis.json")])
            check("main() on a cell whose completeness sidecar reports a shortfall prints GUARD FAILED, "
                  "exits 4 and writes NO analysis JSON",
                  rc_m1 == 4 and "GUARD FAILED" in buf.getvalue() and not (td / "co_analysis.json").exists())
            (td / f"{TAG}_completeness.json").write_text(json.dumps({"pass": True}))
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_m2 = main(["--n-boot", "20", "--out", str(td / "co_analysis.json")])
            check("main() with intact runner records and passing run_guards exits 0 and writes the JSON "
                  "with both guard blocks",
                  rc_m2 == 0 and (td / "co_analysis.json").exists()
                  and json.loads((td / "co_analysis.json").read_text())["runner_record_guard"]["pass"]
                  and json.loads((td / "co_analysis.json").read_text())["guards"]["pass"])
            (td / f"{TAG}_rows.csv").write_text((td / "fl" / f"{TAG}_rows.csv").read_text())
            (td / f"{TAG}_votes.csv").write_text((td / "fl" / f"{TAG}_votes.csv").read_text())
            (td / "co_analysis.json").unlink()
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_m3 = main(["--n-boot", "20", "--out", str(td / "co_analysis.json")])
            check("main() on the UNRESOLVED flood (runner records intact, since the base guard passes a "
                  "legal UNRESOLVED as parsed) prints GUARD FAILED, exits 4, writes no JSON",
                  rc_m3 == 4 and "GUARD FAILED" in buf.getvalue() and not (td / "co_analysis.json").exists())
        finally:
            globals()["OUT_DIR"] = real_outdir
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(res)
        out = buf.getvalue()
        check("print_report renders every section without error",
              all(k in out for k in ("ROLE-LOCK", "OBJECTION PHI", "FLAGS AND LIFT", "AGAINST-INTEREST",
                                     "S2 ACCURACY, PAIRED", "G3 stake concentration")))
        # embodied baseline through the same code, on the embodied seats
        paths2 = _synth_cell(td / "b", n_items=40, ally_mode="flip") if (td / "b").mkdir() is None else None
        res2 = analyse(rows_path=paths2["rows"], votes_path=paths2["votes"], out_dir=td / "b", model=MODEL,
                       draws=50, seed=5)
        lp2 = res2["objection_phi"]["like_pair"]
        check("objection phi: planted flip -> like pair phi -1 with 40 items x 2 arms",
              abs(lp2["phi"] + 1) < 1e-12 and lp2["both"] == 0 and lp2["neither"] == 0)
        check("analyse() returns None when the cell is not on disk",
              analyse(rows_path=td / "nope_rows.csv", votes_path=td / "nope_votes.csv", out_dir=td, model=MODEL) is None)
        check("analyse() returns None when the model has no rows in the cell",
              analyse(rows_path=paths["rows"], votes_path=paths["votes"], out_dir=td,
                      model="no-such-deployment", draws=5, seed=1) is None)
        res3 = analyse(rows_path=paths["rows"], votes_path=paths["votes"], out_dir=td, model=MODEL,
                       embodied_rows_path=paths["emb_rows"], embodied_votes_path=paths["emb_votes"],
                       comparator_path=paths["comp"], draws=50, seed=7)
        eb = res3.get("embodied_baseline")
        check("embodied baseline: the same sensor on the synthetic three-seat cell -> 800 debates, "
              "against-interest fire 0.25, lift +1.0, two stake seats read",
              eb is not None and eb["n_debates"] == 800
              and abs(eb["against_interest"]["lift"]["fire_rate"] - 0.25) < 1e-12
              and eb["against_interest"]["lift"]["delta"] == 1.0
              and sorted(eb["against_interest"]["per_seat"]) == ["counterparty", "writer_advocate"]
              and eb["against_interest"]["count_distribution"] == {0: 600, 1: 200, 2: 0})
        check("embodied baseline: >=2-of-3 flag and the opposed phi are read through the same code",
              eb["ge2_of_3"]["delta"] == 1.0 and eb["objection_phi_opposed"]["phi"] is not None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(res3)
        check("print_report renders the transfer and embodied-baseline lines",
              "embodied baseline" in buf.getvalue() and "transfer" in buf.getvalue())
        check("nothing was written outside the tempdir (rcd.OUT_DIR untouched)", rcd.OUT_DIR == real_out)

    # ---- the real embodied cell anchors the paired comparison --------------------------
    emb_rows = OUT_DIR / f"{EMBODIED_TAG}_rows.csv"
    if emb_rows.exists():
        emb_ok = []
        with open(emb_rows) as f:
            for r in csv.DictReader(f):
                if r.get("model") == MODEL:
                    ok = verdict_correct(r.get("verdict", ""), r["gold_verdict"])
                    if ok is not None:
                        emb_ok.append(int(ok))
        acc = sum(emb_ok) / len(emb_ok)
        check(f"embodied/on S2 accuracy on the codable population reproduces the registered "
              f"{EMBODIED_S2_REGISTERED} ({acc:.4f}, n={len(emb_ok)})", abs(acc - EMBODIED_S2_REGISTERED) < 0.002)
    else:
        check("embodied/on rows present on disk (needed for the paired S2 readout)", False)

    # ---- the real coalition cell, if it has been run --------------------------------
    rows_path = OUT_DIR / f"{TAG}_rows.csv"
    if rows_path.exists():
        res = analyse(rows_path=rows_path, votes_path=OUT_DIR / f"{TAG}_votes.csv", out_dir=OUT_DIR,
                      model=MODEL, draws=50, seed=1)
        check("real cell: the runner's n_against_interest_r3 column agrees with the recomputation on every row",
              res is not None and res["against_interest"]["column_agreement"]["n_mismatch"] == 0)
    else:
        print(f"[ok] coalition cell not on disk yet ({rows_path.name}); real-cell checks skipped")

    # ---- cross-check against the runner's own against_interest_count (install() mutates rcd; LAST) ----
    import scripts.run_crowdgold_coalition as rcc
    rcc.install()
    vr = [{"role_id": s, "role_stake": v["role_stake"], "objected_r3": v["objected_r3"]} for s, v in sv.items()]
    check("against_interest agrees with run_crowdgold_coalition.against_interest_count on every codable verdict",
          all(against_interest(sv, v, SEATS) == rcc.against_interest_count(vr, v)
              for v in ("YTA", "NTA", "ESH", "NAH", rcd.UNRESOLVED, rcd.NOVERDICT)))

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Addendum 16.19 coalition readout (reads only; no API)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--scaffold", default=SCAFFOLD, help="r0 cache scaffold key")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--embodied-tag", default=EMBODIED_TAG)
    ap.add_argument("--comparator", type=Path, default=COMPARATOR_PATH)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", type=Path, default=OUT_DIR / ANALYSIS_NAME)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    rows_path, votes_path = OUT_DIR / f"{a.tag}_rows.csv", OUT_DIR / f"{a.tag}_votes.csv"
    if not rows_path.exists() or not votes_path.exists():
        print(f"coalition cell not on disk: {rows_path} / {votes_path}")
        return 2
    rr = runner_record_guard(a.tag, OUT_DIR)
    if not rr["pass"]:
        print(f"\n  GUARD FAILED -- {a.tag} is not readable: {'; '.join(rr['reasons'])}. "
              "No analysis JSON written.\n")
        return 4
    res = analyse(rows_path=rows_path, votes_path=votes_path, out_dir=OUT_DIR, model=a.model,
                  scaffold=a.scaffold,
                  embodied_rows_path=OUT_DIR / f"{a.embodied_tag}_rows.csv",
                  embodied_votes_path=OUT_DIR / f"{a.embodied_tag}_votes.csv",
                  comparator_path=a.comparator, draws=a.n_boot, seed=a.seed)
    if res is None:
        print(f"no rows for model {a.model} in {rows_path}")
        return 2
    res["runner_record_guard"] = rr
    print_report(res)
    if res["against_interest"]["column_agreement"]["n_mismatch"]:
        print("\n  *** the runner's n_against_interest_r3 column disagrees with the recomputation; "
              "the rows CSV and the votes CSV are not from the same run ***")
    if not res["guards"]["pass"]:
        print(f"\n  GUARD FAILED -- {a.tag} is not readable (analyzer guards). No analysis JSON written.\n")
        return 4
    a.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
