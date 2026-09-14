"""
scripts/analyze_multivendor.py -- Task 16.18 readout: the cross-vendor
decider layer (scripts/run_crowdgold_multivendor.py, tag
cg_deliberation_multivendor), read through the same statistics as the
16.10 table and composed, at zero extra spend, with the embodied sensor.

THE THREE QUESTIONS, AND THE NUMBERS THAT ANSWER THEM

  1. LADHA / CONDORCET. Three seats, three vendors, no seat reads any other
     (the (identical, off) topology), so each seat's R0 verdict is an
     independent draw from a DIFFERENT model. Does the majority of the three
     beat the best single seat? In 16.10 the same cell with three grok seats
     gave majority - best = -0.002 [-0.020, +0.015]: three draws of one model
     are not three voters. The Condorcet jury theorem needs independent
     errors; Ladha (1992) shows the majority still beats its best member when
     the average pairwise error correlation is low enough. Reported: per-seat
     R0 accuracy (each seat named by its model), the majority's accuracy, the
     paired item-clustered difference majority - best seat (the best seat is
     picked on the same data, which flatters the seat, so a positive interval
     is conservative), majority - each seat, the independence prediction
     p1p2 + p1p3 + p2p3 - 2p1p2p3 next to the observed majority, and the
     pairwise error correlations (phi) that say how far from independent the
     three actually are. R2 (each seat's final position, still unread by the
     others) is the sensitivity.
  2. THE COLLECTIVE. The cell's S2 accuracy (the moderator's integrated
     verdict, scored by the registered code_response collapse) against the
     single-vendor (identical, off) cell (registered 0.875 on 210 items) and
     against grok answering alone (cached standard majority-of-3, 0.911 on
     the 420 cells). Both are PAIRED on the matched (arm, item) cells with
     item-clustered CIs; the references are also restated on the matched
     panel, because this cell's content-filter screen is the union over
     three vendors (197 items, a subset of the 210).
  3. THE COMPOSED BIPARTITE GRAPH. The embodied cell (cg_deliberation, 1,677
     codable debates) supplies the SENSOR -- the stake-blind counter
     n_objectors >= 2 of 16.15 -- and this cell's cross-vendor majority
     supplies the ACTUATOR: a flagged debate takes the majority's verdict,
     an unflagged one keeps its own S2, and a debate the majority cannot
     decide (tie, non-codable, or an item outside this cell's 197) falls back
     to S2 (analyze_claim_audit.composed_ok, exactly as for the haiku and
     sonnet judges). Nothing is generated: every input is a cached record.
     Reported next to the haiku-judge and sonnet-judge routers of
     router_decomposition.json (16.15.1), which this module RECOMPUTES
     through the same code on the same population (agreement is checked to
     1e-9 and reported) rather than quotes, and the paired difference
     cross-vendor - haiku judge with an item-clustered CI. All four routers
     of 16.15.1 (collective alone, counter, verdict type, union) are run
     under every actuator, on the full population and on the covered
     sub-population.

Also reported, so the cell sits in the 16.10 table: per-seat objection
rates, localisation excess vs the Poisson-binomial null, fire rate, error
lift of the counter at >= 60 fired, transfer to grok solo, G3 (not
applicable: no seat carries a stake), and the parse guards.

GUARDS (the literal GUARD FAILED is printed, the exit code is 3, NO analysis
JSON is written, and the numbers are not to be read):
  * the runner's own records, read by main() BEFORE any number: <tag>_panel.json
    must exist with exit_code 0 and a completeness record with pass True
    (run_crowdgold_multivendor compares the rows CSV with every launched
    debate; rcd.run drops an errored debate silently), and <tag>_summary.json
    must exist with round_guard.pass and outcome_guard.pass True (the base
    runner's truncation / empty-output / parse guard over all 17 calls, e.g.
    a haiku seat truncating R4 votes at the 3072 floor). A missing record is
    a failed guard: the runner quarantines a failed run's CSVs as
    <tag>_GUARD_FAILED_*, but a CSV pair without its records is not readable.
  * run_guards on the rows/votes: codable share of the cell's debates >= 0.95,
    mean n_r3_unparsed <= 0.05 seats per debate, every seat's R0 codable share
    >= 0.95 (a seat that abstains turns the majority into a different object),
    every seat's R4 vote_parsed share >= 0.95 and truncated share <= 0.05
    (from the votes CSV's vote_parsed / truncated columns), and at least one
    debate.

Conventions inherited from 16.8 / 16.10 / 16.15: verdicts scored by
code_response(., "published") (ESH -> at fault, NAH -> not at fault), never
string equality; item-clustered percentile bootstrap (4,000 draws, seed 31);
partial panels reported, never retried; a cell-level flag fires when more
than half of the cell's debate samples fire.

Usage
-----
  python -m scripts.analyze_multivendor --selftest
  python -m scripts.analyze_multivendor                 # writes multivendor_analysis.json
  python -m scripts.analyze_multivendor --tag cg_deliberation_multivendor_b   # a --namespace panel
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_actuator_ladder import (
    COMPARATOR_PATH, ROWS_PATH as EMBODIED_ROWS, VOTES_PATH as EMBODIED_VOTES, correct,
    load_comparator, load_debates, make_rule_vendor,
)
from scripts.analyze_claim_audit import composed_ok
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_router_decomposition import (
    ANALYSIS_PATH as ROUTER_JSON_PATH, GROK, GROK_SOLO, HAIKU_JUDGE, ROUTERS,
    SONNET_JUDGE, SONNET_ROWS_PATH, verdict_type,
)
from scripts.analyze_unembodied_ablation import (
    _boot, localisation, per_seat_objection_rates,
)
from scripts.run_crowdgold_multivendor import (
    DEFAULT_SEAT_MODELS, DEFAULT_TAG, PANEL_LABEL, ROW_FIELDS_MV, VOTE_FIELDS_MV, seat_ids,
)
from scripts.run_crowdgold_topology import CELLS as TOPOLOGY_CELLS
from scripts.run_phase1_quartet import OUT_DIR

ANALYSIS_PATH = OUT_DIR / "multivendor_analysis.json"
IDENT_OFF_TAG = TOPOLOGY_CELLS[("identical", "off")]["tag"]
DEFAULT_DRAWS = 4000
DEFAULT_SEED = 31
MIN_FIRED = 60

#: Registered reference values this readout is compared against (restated
#: live from the files when present; these are the quoted numbers).
REGISTERED = {
    # 16.10 table row "group verdict (S2) acc", read on the table's
    # AGGREGATION population: debates whose three R0 seat verdicts and S2 are
    # all codable (n=409 of 420). Plain S2 over the cell's 419 codable
    # debates is 0.871; both are restated live below, and the paired
    # comparison here is on codable S2 of matched cells.
    "identical_off_s2": 0.875,
    "identical_off_s2_codable": 0.871,
    "grok_solo_standard_maj3": 0.911,  # 16.10 / 16.15.2, 420 cells
    "identical_off_majority_minus_best": {"delta": -0.002, "lo": -0.020, "hi": 0.015},
    "haiku_judge_counter": 0.880,     # 16.15.1 composed accuracy
    "sonnet_judge_counter": 0.913,
    "collective_alone": 0.843,
}

# Guards (checked on the literal string GUARD FAILED before any number is read)
MIN_CODABLE_SHARE = 0.95
MAX_MEAN_R3_UNPARSED = 0.05
MIN_SEAT_R0_CODABLE = 0.95
MIN_SEAT_VOTE_PARSED = 0.95
MAX_SEAT_TRUNCATED = 0.05


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cell(rows_path: Path, votes_path: Path, model: Optional[str] = PANEL_LABEL,
              ) -> list[dict]:
    """One record per debate of a multivendor-schema cell, its three seats
    attached from the votes CSV (seat model, R0 / R2 verdicts, R3 objection,
    R4 vote). Every verdict is carried both as the string and as its
    code_response code; None codes mark non-codable verdicts."""
    debates: dict[tuple, dict] = {}
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            if model and r.get("model") != model:
                continue
            gold = r["gold_verdict"]
            key = (r["item_id"], r["arm"], int(r["sample_idx"]))
            syn_ok = correct(r.get("synthesis_verdict", ""), gold)
            s2_ok = correct(r.get("verdict", ""), gold)
            debates[key] = {
                "item": r["item_id"], "arm": r["arm"], "sample_idx": int(r["sample_idx"]),
                "gold": gold, "s1": r.get("synthesis_verdict", ""), "s2": r.get("verdict", ""),
                "syn_ok": syn_ok, "s2_ok": s2_ok,
                "codable": syn_ok is not None and s2_ok is not None,
                "n_objectors": int(r.get("n_objectors") or 0),
                "n_r3_unparsed": int(r.get("n_r3_unparsed") or 0),
                "n_reject": int(r.get("n_reject") or 0),
                "unanimous_accept": r.get("unanimous_accept") == "1",
                "verdict_revised": r.get("verdict_revised") == "1",
                "seat_models": r.get("seat_models", ""),
                "moderator_model": r.get("moderator_model", ""),
                "seats": {},
            }
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            if model and v.get("model") != model:
                continue
            d = debates.get((v["item_id"], v["arm"], int(v["sample_idx"])))
            if d is None:
                continue
            r0, r2 = v.get("r0_verdict", ""), v.get("r2_verdict", "")
            d["seats"][v["role_id"]] = {
                "role_id": v["role_id"], "index": int(v.get("seat_index") or 0),
                "model": v.get("seat_model", ""),
                "r0": r0, "r0_code": code_response(r0, "published"), "r0_ok": correct(r0, d["gold"]),
                "r2": r2, "r2_code": code_response(r2, "published"), "r2_ok": correct(r2, d["gold"]),
                "objected": v.get("objected_r3") == "1",
                "r3_label": v.get("r3_label", ""),
                "vote": v.get("vote", ""), "reject": v.get("reject") == "1",
                # an absent or blank column (a synthetic cell) reads as parsed / not
                # truncated; the base runner always writes 0/1
                "vote_parsed": str(v.get("vote_parsed") or "1") == "1",
                "truncated": str(v.get("truncated") or "0") == "1",
            }
    return sorted(debates.values(), key=lambda d: (d["item"], d["arm"], d["sample_idx"]))


def seat_order(debates: Sequence[dict]) -> list[str]:
    """Seat role ids in seat order (seat_index, then id), from the votes."""
    seen: dict[str, int] = {}
    for d in debates:
        for rid, s in d["seats"].items():
            seen.setdefault(rid, s["index"])
    return sorted(seen, key=lambda rid: (seen[rid], rid))


def seat_models(debates: Sequence[dict], seats: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for d in debates:
        for rid in seats:
            if rid in d["seats"] and d["seats"][rid]["model"]:
                out.setdefault(rid, d["seats"][rid]["model"])
    return {rid: out.get(rid, "") for rid in seats}


def majority_code(codes: Sequence[Optional[int]], *, require_all: bool = True) -> Optional[int]:
    """Majority of the codable codes; None on a tie, on no codable code, and
    (when require_all) whenever any code is missing."""
    cs = [c for c in codes if c is not None]
    if not cs or (require_all and len(cs) < len(codes)):
        return None
    n1 = sum(cs)
    n0 = len(cs) - n1
    if n1 == n0:
        return None
    return 1 if n1 > n0 else 0


def _code_str(c: Optional[int]) -> Optional[str]:
    return None if c is None else ("YTA" if c == 1 else "NTA")


# ---------------------------------------------------------------------------
# Estimator helpers (item-clustered percentile bootstrap via _boot)
# ---------------------------------------------------------------------------

def _by_item(recs: Sequence[dict]) -> dict:
    by: dict = defaultdict(list)
    for r in recs:
        by[r["item"]].append(r)
    return by


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def mean_ci(recs: Sequence[dict], key: str, *, draws: int, seed: int) -> dict:
    pt, lo, hi = _boot(_by_item(recs), lambda rs: _mean([r[key] for r in rs]),
                       draws=draws, seed=seed)
    return {"value": pt, "lo": lo, "hi": hi, "n": len(recs)}


def paired_delta(recs: Sequence[dict], a: str, b: str, *, draws: int, seed: int) -> dict:
    """mean(a) - mean(b), paired per record, item-clustered CI."""
    pt, lo, hi = _boot(_by_item(recs), lambda rs: _mean([r[a] - r[b] for r in rs]),
                       draws=draws, seed=seed)
    return {"acc_a": _mean([r[a] for r in recs]), "acc_b": _mean([r[b] for r in recs]),
            "delta": pt, "lo": lo, "hi": hi, "n": len(recs),
            "ci_excludes_zero": (lo is not None and hi is not None and (lo > 0 or hi < 0))}


# ---------------------------------------------------------------------------
# 1. Ladha / Condorcet: majority of three vendors vs the best single seat
# ---------------------------------------------------------------------------

def ladha_block(debates: Sequence[dict], seats: Sequence[str], *, round_key: str,
                draws: int, seed: int) -> dict:
    """On the debates where all three seats' verdicts for `round_key` (r0 or
    r2) are codable: per-seat accuracy, majority accuracy, majority minus
    best seat (paired), majority minus each seat, the independence
    prediction and the pairwise error correlations."""
    models = seat_models(debates, seats)
    recs = []
    n_dropped = 0
    for d in debates:
        codes = {s: d["seats"][s][f"{round_key}_code"] for s in seats if s in d["seats"]}
        if len(codes) < len(seats) or any(c is None for c in codes.values()):
            n_dropped += 1
            continue
        g = 1 if d["gold"] == "YTA" else 0
        maj = majority_code([codes[s] for s in seats])
        rec = {"item": d["item"], "maj": int(maj == g),
               "s2": None if d["s2_ok"] is None else int(d["s2_ok"])}
        for s in seats:
            rec[f"ok_{s}"] = int(codes[s] == g)
        recs.append(rec)
    n = len(recs)
    if n == 0:
        return {"round": round_key, "n": 0, "n_dropped": n_dropped, "applicable": False}
    seat_acc = {s: mean_ci(recs, f"ok_{s}", draws=draws, seed=seed + i)
                for i, s in enumerate(seats)}
    best = max(seats, key=lambda s: seat_acc[s]["value"])
    maj_acc = mean_ci(recs, "maj", draws=draws, seed=seed + 10)
    vs_best = paired_delta(recs, "maj", f"ok_{best}", draws=draws, seed=seed + 11)
    vs_each = {s: paired_delta(recs, "maj", f"ok_{s}", draws=draws, seed=seed + 20 + i)
               for i, s in enumerate(seats)}
    p = [seat_acc[s]["value"] for s in seats]
    pred = p[0] * p[1] + p[0] * p[2] + p[1] * p[2] - 2 * p[0] * p[1] * p[2]
    pairs = {}
    phis = []
    for i in range(len(seats)):
        for j in range(i + 1, len(seats)):
            a, b = seats[i], seats[j]
            wa = [1 - r[f"ok_{a}"] for r in recs]
            wb = [1 - r[f"ok_{b}"] for r in recs]
            pa, pb = _mean(wa), _mean(wb)
            both = _mean([x * y for x, y in zip(wa, wb)])
            var = pa * (1 - pa) * pb * (1 - pb)
            phi = ((both - pa * pb) / var ** 0.5) if var > 0 else None
            if phi is not None:
                phis.append(phi)
            pairs[f"{a}|{b}"] = {"p_both_wrong": both, "p_both_wrong_if_independent": pa * pb,
                                 "phi": phi}
    # the 16.10 table's "group verdict (S2) acc" is read on THIS population
    s2_recs = [{"item": r["item"], "ok": r["s2"]} for r in recs if r["s2"] is not None]
    return {
        "round": round_key, "applicable": True, "n": n, "n_dropped_non_codable": n_dropped,
        "group_verdict_s2_on_this_population": mean_ci(s2_recs, "ok", draws=draws, seed=seed + 12),
        "seat_models": models, "seat_accuracy": seat_acc, "best_seat": best,
        "best_seat_model": models.get(best, ""),
        "majority_accuracy": maj_acc,
        "majority_minus_best_seat": vs_best,
        "majority_minus_each_seat": vs_each,
        "majority_equals_best_seat_rate": _mean([int(r["maj"] == r[f"ok_{best}"]) for r in recs]),
        "independence_prediction": {"majority_if_independent": pred,
                                    "observed_majority": maj_acc["value"],
                                    "observed_minus_predicted": maj_acc["value"] - pred},
        "pairwise_error": pairs,
        "mean_pairwise_phi": _mean(phis),
    }


# ---------------------------------------------------------------------------
# 2. The collective vs the single-vendor collective and vs grok solo
# ---------------------------------------------------------------------------

def cell_key(d: dict) -> tuple:
    return (d["arm"], d["item"])


def reference_s2_by_cell(rows_path: Path, model: str) -> dict[tuple, float]:
    """(arm, item) -> mean S2 correctness over that cell's codable samples in
    another deliberation cell's rows CSV (the (identical, off) cell here)."""
    acc: dict[tuple, list[int]] = defaultdict(list)
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            if model and r.get("model") != model:
                continue
            ok = correct(r.get("verdict", ""), r["gold_verdict"])
            if ok is not None:
                acc[(r["arm"], r["item_id"])].append(int(ok))
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def comparison_block(debates: Sequence[dict], seats: Sequence[str], *,
                     ident_s2: Optional[dict[tuple, float]], solo_rule, draws: int,
                     seed: int) -> dict:
    """Paired, item-clustered contrasts on the matched (arm, item) cells:
    the cell's S2 and its R0 majority against the (identical, off) S2 and
    against grok solo (standard majority-of-3)."""
    def own_s2(d):
        return None if d["s2_ok"] is None else int(d["s2_ok"])

    def own_maj(d):
        codes = [d["seats"][s]["r0_code"] for s in seats if s in d["seats"]]
        m = majority_code(codes) if len(codes) == len(seats) else None
        return None if m is None else int(m == (1 if d["gold"] == "YTA" else 0))

    def ref_ident(d):
        return None if ident_s2 is None else ident_s2.get(cell_key(d))

    def ref_solo(d):
        v = solo_rule({"arm": d["arm"], "item": d["item"]}) if solo_rule else None
        ok = correct(v, d["gold"]) if v is not None else None
        return None if ok is None else int(ok)

    out: dict = {}
    for name, fa, fb in (("s2_vs_identical_off", own_s2, ref_ident),
                         ("s2_vs_grok_solo", own_s2, ref_solo),
                         ("majority_vs_identical_off", own_maj, ref_ident),
                         ("majority_vs_grok_solo", own_maj, ref_solo),
                         ("majority_vs_own_s2", own_maj, own_s2)):
        recs = []
        for d in debates:
            a, b = fa(d), fb(d)
            if a is None or b is None:
                continue
            recs.append({"item": d["item"], "a": a, "b": b})
        out[name] = (paired_delta(recs, "a", "b", draws=draws, seed=seed + len(out))
                     if recs else {"n": 0, "delta": None, "lo": None, "hi": None})
        out[name]["n_items"] = len({r["item"] for r in recs})
    return out


# ---------------------------------------------------------------------------
# 16.10-style structure: objection rates, localisation, fire, lift, transfer
# ---------------------------------------------------------------------------

def structure_block(debates: Sequence[dict], votes_path: Path, *, draws: int, seed: int,
                    solo_rule) -> dict:
    rates = per_seat_objection_rates(votes_path, PANEL_LABEL, None)
    loc = localisation(list(debates), rates) if rates["per_seat"] else None
    fired = lambda d: d["n_objectors"] >= 2
    codable = [d for d in debates if d["codable"]]
    n_fired = sum(1 for d in debates if fired(d))
    byl = defaultdict(list)
    for d in codable:
        byl[d["item"]].append(d)

    def lift_stat(rs):
        fl = [r for r in rs if fired(r)]
        un = [r for r in rs if not fired(r)]
        if not fl or not un:
            return None
        return ((1 - sum(bool(r["s2_ok"]) for r in fl) / len(fl))
                - (1 - sum(bool(r["s2_ok"]) for r in un) / len(un)))
    l_p, l_lo, l_hi = _boot(byl, lift_stat, draws=draws, seed=seed + 1)
    fl = [d for d in codable if fired(d)]
    un = [d for d in codable if not fired(d)]

    # transfer to grok solo (cell level, flag = majority of the cell's samples)
    fl_c: dict = defaultdict(int)
    k_c: dict = defaultdict(int)
    gold = {}
    for d in debates:
        c = cell_key(d)
        fl_c[c] += int(fired(d))
        k_c[c] += 1
        gold[c] = d["gold"]
    trec = []
    for c, g in gold.items():
        v = solo_rule({"arm": c[0], "item": c[1]}) if solo_rule else None
        ok = correct(v, g) if v is not None else None
        if ok is None:
            continue
        trec.append({"item": c[1], "w": int(not ok), "f": int(fl_c[c] * 2 > k_c[c])})
    byt = defaultdict(list)
    for x in trec:
        byt[x["item"]].append(x)

    def tstat(rs):
        f = [x for x in rs if x["f"]]
        u = [x for x in rs if not x["f"]]
        return None if not f or not u else _mean([x["w"] for x in f]) - _mean([x["w"] for x in u])
    t_p, t_lo, t_hi = _boot(byt, tstat, draws=draws, seed=seed + 2)
    tf = [x for x in trec if x["f"]]
    tu = [x for x in trec if not x["f"]]
    return {
        "objection": {"per_seat": rates["per_seat"], "pooled": rates["pooled"],
                      "localisation_excess": loc["localisation_excess"] if loc else None,
                      "fire_rate": loc["fire_observed"] if loc else None,
                      "fire_independence": loc["fire_independence"] if loc else None},
        "g3": {"applicable": False, "reason": "no seat carries a stake"},
        "lift": {"n_fired": n_fired, "powered": n_fired >= MIN_FIRED,
                 "codable": ({"n_fired": len(fl),
                              "err_fired": 1 - _mean([bool(r["s2_ok"]) for r in fl]),
                              "err_unfired": 1 - _mean([bool(r["s2_ok"]) for r in un])}
                             if fl and un else None),
                 "delta": l_p, "lo": l_lo, "hi": l_hi},
        "transfer_grok_solo": {"n_cells": len(trec), "n_flagged_cells": len(tf),
                               "p_wrong_flagged": _mean([x["w"] for x in tf]),
                               "p_wrong_unflagged": _mean([x["w"] for x in tu]),
                               "lift": t_p, "lo": t_lo, "hi": t_hi},
    }


# ---------------------------------------------------------------------------
# 3. The composed bipartite graph: embodied sensor -> cross-vendor actuator
# ---------------------------------------------------------------------------

def actuator_from_cell(debates: Sequence[dict], seats: Sequence[str], kind: str,
                       ) -> Callable[[dict], Optional[str]]:
    """A rule in the analyze_actuator_ladder sense: debate -> 'YTA' / 'NTA'
    or None (fall back to S2), keyed by the debate's (arm, item), from this
    cell's own records. kind: 'r0_majority' / 'r2_majority' (majority over
    the codable seat verdicts pooled across the cell's samples; a tie or no
    codable verdict is None) or 's2' (the cell's own integrated verdict,
    majority across samples)."""
    codes: dict[tuple, list[int]] = defaultdict(list)
    for d in debates:
        c = cell_key(d)
        if kind == "s2":
            k = code_response(d["s2"], "published")
            if k is not None:
                codes[c].append(k)
        else:
            rk = "r0_code" if kind == "r0_majority" else "r2_code"
            for s in seats:
                if s in d["seats"] and d["seats"][s][rk] is not None:
                    codes[c].append(d["seats"][s][rk])
    table = {c: _code_str(majority_code(v, require_all=False)) for c, v in codes.items()}

    def rule(d: dict) -> Optional[str]:
        return table.get((d["arm"], d["item"]))
    return rule


def paired_rule_diff(pop: Sequence[dict], sensor, rule_a, rule_b, *, draws: int,
                     seed: int) -> dict:
    """Composed accuracy under rule_a minus under rule_b, same sensor, paired
    per debate, item-clustered CI."""
    recs = [{"item": d["item"], "a": int(composed_ok(d, sensor, rule_a)),
             "b": int(composed_ok(d, sensor, rule_b))} for d in pop]
    return paired_delta(recs, "a", "b", draws=draws, seed=seed)


def composed_block(emb: Sequence[dict], rules: dict[str, Callable], covered: set, *,
                   draws: int, seed: int, primary: str = "crossvendor_r0_majority",
                   judges: Sequence[str] = ("haiku_judge", "sonnet_judge")) -> dict:
    """Every router of 16.15.1 under every actuator, on the full embodied
    population (uncovered debates fall back to S2) and on the covered
    sub-population, plus the paired differences that answer the question."""
    pops = {"full": list(emb), "covered": [d for d in emb if cell_key(d) in covered]}
    out: dict = {"n_debates_full": len(pops["full"]), "n_debates_covered": len(pops["covered"]),
                 "n_items_full": len({d["item"] for d in emb}),
                 "n_items_covered": len({d["item"] for d in pops["covered"]}),
                 "populations": {}}
    k = 0
    for pname, pop in pops.items():
        blk: dict = {"n": len(pop), "actuators": {}}
        for rname, rule in rules.items():
            rb: dict = {}
            for sname, sensor in ROUTERS.items():
                recs = [{"item": d["item"], "ok": int(composed_ok(d, sensor, rule))} for d in pop]
                fired = [d for d in pop if sensor(d)]
                emitted = sum(1 for d in fired if rule(d) is not None
                              and correct(rule(d), d["gold"]) is not None)
                k += 1
                rb[sname] = {
                    "n_fire": len(fired), "coverage": len(fired) / len(pop) if pop else None,
                    "composed": mean_ci(recs, "ok", draws=draws, seed=seed + k),
                    "actuator_emitted_on_fired": (emitted / len(fired)) if fired else None,
                    "n_fallback_to_s2": len(fired) - emitted,
                    "by_stratum": {
                        st: {"n": len(sub), "n_fired": sum(1 for d in sub if sensor(d)),
                             "acc_s2": _mean([d["s2_ok"] for d in sub]),
                             "acc_composed": _mean([composed_ok(d, sensor, rule) for d in sub])}
                        for st, sub in ((s_, [d for d in pop if verdict_type(d) == s_])
                                        for s_ in ("one_loser", "both_party"))},
                }
            blk["actuators"][rname] = rb
        pd: dict = {}
        if primary in rules:
            for sname in ("counter", "union"):
                sensor = ROUTERS[sname]
                k += 1
                pd[f"{primary}_minus_collective_alone|{sname}"] = paired_rule_diff(
                    pop, sensor, rules[primary], lambda d: None, draws=draws, seed=seed + k)
                for j in judges:
                    if j in rules:
                        k += 1
                        pd[f"{primary}_minus_{j}|{sname}"] = paired_rule_diff(
                            pop, sensor, rules[primary], rules[j], draws=draws, seed=seed + k)
        blk["paired_deltas"] = pd
        out["populations"][pname] = blk
    return out


def router_json_agreement(comp_full: dict, router_json: Optional[dict]) -> dict:
    """The haiku / sonnet judges recomputed here must equal router_
    decomposition.json's composed accuracies exactly (same population, same
    rule, same code); the check is on the point estimates to 1e-9."""
    if not router_json:
        return {"available": False}
    out: dict = {"available": True, "fields": {}, "agrees": True}
    acts = comp_full["actuators"]
    for j, key in (("haiku_judge", "composed_haiku"), ("sonnet_judge", "composed_sonnet")):
        if j not in acts or key not in router_json:
            continue
        blk = router_json[key]["block"] if key == "composed_sonnet" else router_json[key]
        for sname in ROUTERS:
            got = acts[j][sname]["composed"]["value"]
            ref = blk["routers"][sname]["composed"]["acc"]
            ok = got is not None and ref is not None and abs(got - ref) < 1e-9
            out["fields"][f"{j}|{sname}"] = {"got": got, "json": ref, "agrees": ok}
            out["agrees"] &= ok
    return out


# ---------------------------------------------------------------------------
# Guards, readings, report
# ---------------------------------------------------------------------------

def run_guards(debates: Sequence[dict], seats: Sequence[str]) -> dict:
    n = len(debates)
    checks: dict = {}
    codable = sum(1 for d in debates if d["codable"])
    checks["n_debates"] = {"value": n, "limit": 1, "ok": n >= 1}
    checks["codable_share"] = {"value": codable / n if n else None, "limit": MIN_CODABLE_SHARE,
                               "ok": bool(n) and codable / n >= MIN_CODABLE_SHARE}
    m = _mean([d["n_r3_unparsed"] for d in debates])
    checks["mean_n_r3_unparsed"] = {"value": m, "limit": MAX_MEAN_R3_UNPARSED,
                                    "ok": m is not None and m <= MAX_MEAN_R3_UNPARSED}
    for s in seats:
        have = [d for d in debates if s in d["seats"]]
        cod = sum(1 for d in have if d["seats"][s]["r0_code"] is not None)
        v = cod / len(have) if have else None
        checks[f"seat_r0_codable_share|{s}"] = {"value": v, "limit": MIN_SEAT_R0_CODABLE,
                                                "ok": v is not None and v >= MIN_SEAT_R0_CODABLE}
        vp = (sum(1 for d in have if d["seats"][s].get("vote_parsed", True)) / len(have)) if have else None
        checks[f"seat_vote_parsed_share|{s}"] = {"value": vp, "limit": MIN_SEAT_VOTE_PARSED,
                                                 "ok": vp is not None and vp >= MIN_SEAT_VOTE_PARSED}
        tr = (sum(1 for d in have if d["seats"][s].get("truncated", False)) / len(have)) if have else None
        checks[f"seat_r4_truncated_share|{s}"] = {"value": tr, "limit": MAX_SEAT_TRUNCATED,
                                                  "ok": tr is not None and tr <= MAX_SEAT_TRUNCATED}
    checks["three_seats_present"] = {"value": len(seats), "limit": 3, "ok": len(seats) == 3}
    return {"n_raw": n, "n_codable": codable, "checks": checks,
            "passed": all(c["ok"] for c in checks.values())}


def runner_record_guard(tag: str, out_dir: Path) -> dict:
    """The runner's own verdicts: <tag>_panel.json (exit_code, completeness)
    and <tag>_summary.json (round / outcome guards). Missing = failed."""
    reasons: list[str] = []
    pp = out_dir / f"{tag}_panel.json"
    if pp.exists():
        try:
            pj = json.loads(pp.read_text())
        except json.JSONDecodeError:
            pj = {}
        if pj.get("exit_code") != 0:
            reasons.append(f"panel.json exit_code {pj.get('exit_code')}")
        comp = pj.get("completeness")
        if not comp or not comp.get("pass", False):
            reasons.append("completeness: "
                           + (f"{comp.get('n_missing')} of {comp.get('n_expected')} launched debates "
                              "missing from the rows" if comp else "no record in panel.json"))
    else:
        reasons.append(f"{pp.name} missing: the run's exit code and completeness are unknown")
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
    return {"pass": not reasons, "reasons": reasons}


def _sign(r: dict) -> str:
    if r.get("delta") is None:
        return "not computable"
    if r.get("ci_excludes_zero"):
        return "above zero" if r["delta"] > 0 else "below zero"
    return "interval includes zero"


def readings(res: dict) -> dict:
    lad = res["ladha"]["r0"]
    cmp_ = res["comparison"]
    comp = res["composed"]["populations"]["full"]["paired_deltas"]
    out = {
        "16.18_ladha": {
            "claim": "the majority of three different-vendor seats beats its best single seat",
            "delta": lad.get("majority_minus_best_seat", {}).get("delta"),
            "lo": lad.get("majority_minus_best_seat", {}).get("lo"),
            "hi": lad.get("majority_minus_best_seat", {}).get("hi"),
            "reading": _sign(lad.get("majority_minus_best_seat", {})),
            "holds": bool(lad.get("majority_minus_best_seat", {}).get("ci_excludes_zero"))
                     and (lad.get("majority_minus_best_seat", {}).get("delta") or 0) > 0,
            "single_vendor_reference": REGISTERED["identical_off_majority_minus_best"],
        },
        "16.18_collective_vs_single_vendor": {
            "claim": "the cross-vendor collective's S2 beats the single-vendor (identical, off) S2",
            **{k: cmp_["s2_vs_identical_off"].get(k) for k in ("acc_a", "acc_b", "delta", "lo", "hi", "n")},
            "reading": _sign(cmp_["s2_vs_identical_off"]),
            "registered_reference": REGISTERED["identical_off_s2"],
        },
        "16.18_collective_vs_grok_solo": {
            "claim": "the cross-vendor collective's S2 beats grok answering alone",
            **{k: cmp_["s2_vs_grok_solo"].get(k) for k in ("acc_a", "acc_b", "delta", "lo", "hi", "n")},
            "reading": _sign(cmp_["s2_vs_grok_solo"]),
            "registered_reference": REGISTERED["grok_solo_standard_maj3"],
        },
        "16.18_majority_vs_grok_solo": {
            "claim": "the cross-vendor R0 majority alone beats grok answering alone",
            **{k: cmp_["majority_vs_grok_solo"].get(k) for k in ("acc_a", "acc_b", "delta", "lo", "hi", "n")},
            "reading": _sign(cmp_["majority_vs_grok_solo"]),
        },
    }
    key = "crossvendor_r0_majority_minus_haiku_judge|counter"
    if key in comp:
        out["16.18_composed_vs_haiku_judge"] = {
            "claim": "routing the embodied cell's flagged debates to the cross-vendor majority "
                     "beats routing them to the haiku judge (counter router, full population)",
            **{k: comp[key].get(k) for k in ("acc_a", "acc_b", "delta", "lo", "hi", "n")},
            "reading": _sign(comp[key]),
            "registered_reference": REGISTERED["haiku_judge_counter"],
        }
    key = "crossvendor_r0_majority_minus_sonnet_judge|counter"
    if key in comp:
        out["16.18_composed_vs_sonnet_judge"] = {
            "claim": "... beats routing them to the sonnet judge",
            **{k: comp[key].get(k) for k in ("acc_a", "acc_b", "delta", "lo", "hi", "n")},
            "reading": _sign(comp[key]),
            "registered_reference": REGISTERED["sonnet_judge_counter"],
        }
    key = "crossvendor_r0_majority_minus_collective_alone|counter"
    if key in comp:
        out["16.18_composed_vs_collective_alone"] = {
            "claim": "the composed graph beats the embodied collective unrouted",
            **{k: comp[key].get(k) for k in ("acc_a", "acc_b", "delta", "lo", "hi", "n")},
            "reading": _sign(comp[key]),
            "registered_reference": REGISTERED["collective_alone"],
        }
    return out


def analyse(debates: Sequence[dict], votes_path: Path, *, ident_s2: Optional[dict],
            comp: Optional[dict], sonnet_comp: Optional[dict], emb: Optional[Sequence[dict]],
            router_json: Optional[dict], draws: int, seed: int) -> dict:
    seats = seat_order(debates)
    models = seat_models(debates, seats)
    solo_rule = make_rule_vendor(comp, *GROK_SOLO) if comp else None
    res: dict = {
        "registration": "Task 16.18, cross-vendor decider layer",
        "draws": draws, "seed": seed,
        "n_debates": len(debates), "n_items": len({d["item"] for d in debates}),
        "n_codable": sum(1 for d in debates if d["codable"]),
        "seats": list(seats), "seat_models": models,
        "moderator_model": next((d["moderator_model"] for d in debates if d["moderator_model"]), ""),
        "s2_accuracy": mean_ci([{"item": d["item"], "ok": int(d["s2_ok"])}
                                for d in debates if d["s2_ok"] is not None], "ok",
                               draws=draws, seed=seed),
        "ladha": {rk: ladha_block(debates, seats, round_key=rk, draws=draws, seed=seed + 100 * (i + 1))
                  for i, rk in enumerate(("r0", "r2"))},
        "comparison": comparison_block(debates, seats, ident_s2=ident_s2, solo_rule=solo_rule,
                                       draws=draws, seed=seed + 300),
        "structure": structure_block(debates, votes_path, draws=draws, seed=seed + 400,
                                     solo_rule=solo_rule),
        "references": dict(REGISTERED),
    }
    if ident_s2:
        res["references"]["identical_off_s2_full_panel_live"] = _mean(list(ident_s2.values()))
        res["references"]["identical_off_n_cells"] = len(ident_s2)
    if emb is not None:
        rules: dict[str, Callable] = {
            "crossvendor_r0_majority": actuator_from_cell(debates, seats, "r0_majority"),
            "crossvendor_r2_majority": actuator_from_cell(debates, seats, "r2_majority"),
            "crossvendor_s2": actuator_from_cell(debates, seats, "s2"),
        }
        if comp:
            rules["haiku_judge"] = make_rule_vendor(comp, *HAIKU_JUDGE)
        if sonnet_comp:
            rules["sonnet_judge"] = make_rule_vendor(sonnet_comp, *SONNET_JUDGE)
        covered = {cell_key(d) for d in debates}
        res["composed"] = composed_block(emb, rules, covered, draws=draws, seed=seed + 500)
        res["composed"]["router_json_agreement"] = router_json_agreement(
            res["composed"]["populations"]["full"], router_json)
        res["composed"]["note"] = (
            "sensor = embodied cell's stake-blind counter n_objectors >= 2 (and the other "
            "16.15.1 routers); actuator = this cell's cross-vendor majority keyed by (arm, "
            "item); a debate the actuator cannot decide falls back to S2; zero spend")
    else:
        res["composed"] = {"populations": {"full": {"paired_deltas": {}}},
                           "note": "embodied cell rows/votes absent: composed graph NOT COMPUTABLE"}
    res["guards"] = run_guards(debates, seats)
    res["pre_declared_readings"] = readings(res)
    return res


def _f(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def _ci(r: dict, fmt="+.3f") -> str:
    return f"[{_f(r.get('lo'), fmt)}, {_f(r.get('hi'), fmt)}]"


def print_report(res: dict, *, label: str) -> None:
    g = res["guards"]
    print("\n" + "=" * 96)
    print(f"TASK 16.18 -- CROSS-VENDOR DECIDER LAYER   ({label}: {res['n_debates']} debates, "
          f"{res['n_items']} items, {res['n_codable']} codable)")
    print("=" * 96)
    print("  seats: " + "  ".join(f"{s}={m}" for s, m in res["seat_models"].items())
          + f"   moderator {res['moderator_model'] or 'n/a'}")
    s2 = res["s2_accuracy"]
    print(f"  S2 accuracy (group verdict)     {_f(s2['value'], '.3f')} {_ci(s2, '.3f')}  n={s2['n']}")

    for rk in ("r0", "r2"):
        lad = res["ladha"][rk]
        print(f"\n  LADHA / CONDORCET on {rk.upper()} verdicts (n={lad.get('n')} debates with three "
              f"codable seats; dropped {lad.get('n_dropped_non_codable', 0)})")
        if not lad.get("applicable"):
            print("    not computable")
            continue
        for s, a in lad["seat_accuracy"].items():
            print(f"    seat {s:<18} {lad['seat_models'].get(s, ''):<26} acc {_f(a['value'], '.3f')} "
                  f"{_ci(a, '.3f')}")
        ma = lad["majority_accuracy"]
        print(f"    MAJORITY of three                                     acc {_f(ma['value'], '.3f')} "
              f"{_ci(ma, '.3f')}")
        gs = lad["group_verdict_s2_on_this_population"]
        print(f"    group verdict S2 on this population                   acc {_f(gs['value'], '.3f')} "
              f"{_ci(gs, '.3f')}   [16.10 identical/off: {REGISTERED['identical_off_s2']:.3f}]")
        vb = lad["majority_minus_best_seat"]
        print(f"    majority - best seat ({lad['best_seat_model']}) {_f(vb['delta'])} {_ci(vb)}  "
              f"{'*' if vb['ci_excludes_zero'] else ''}   [single-vendor 16.10: "
              f"{_f(REGISTERED['identical_off_majority_minus_best']['delta'])} "
              f"[{_f(REGISTERED['identical_off_majority_minus_best']['lo'])}, "
              f"{_f(REGISTERED['identical_off_majority_minus_best']['hi'])}]]")
        for s, r in lad["majority_minus_each_seat"].items():
            print(f"      majority - {lad['seat_models'].get(s, s):<26} {_f(r['delta'])} {_ci(r)}")
        ip = lad["independence_prediction"]
        print(f"    if errors were independent: majority {_f(ip['majority_if_independent'], '.3f')}; "
              f"observed {_f(ip['observed_majority'], '.3f')}; mean pairwise error phi "
              f"{_f(lad['mean_pairwise_phi'])}")
        for k, v in lad["pairwise_error"].items():
            print(f"      {k:<40} P(both wrong) {_f(v['p_both_wrong'], '.3f')} vs independent "
                  f"{_f(v['p_both_wrong_if_independent'], '.3f')}  phi {_f(v['phi'])}")

    c = res["comparison"]
    print("\n  THE COLLECTIVE vs THE REFERENCES (paired on matched (arm, item) cells)")
    for name, ref in (("s2_vs_identical_off", REGISTERED["identical_off_s2"]),
                      ("s2_vs_grok_solo", REGISTERED["grok_solo_standard_maj3"]),
                      ("majority_vs_identical_off", REGISTERED["identical_off_s2"]),
                      ("majority_vs_grok_solo", REGISTERED["grok_solo_standard_maj3"]),
                      ("majority_vs_own_s2", None)):
        r = c[name]
        print(f"    {name:<28} {_f(r.get('acc_a'), '.3f')} vs {_f(r.get('acc_b'), '.3f')}  "
              f"delta {_f(r.get('delta'))} {_ci(r)}  n={r.get('n')}"
              f"{'  *' if r.get('ci_excludes_zero') else ''}"
              + (f"   [registered reference on the full panel: {ref:.3f}]" if ref else ""))

    st = res["structure"]
    print("\n  16.10-STYLE STRUCTURE")
    print(f"    objection per seat  " + "  ".join(f"{s}={_f(v, '.3f')}" for s, v in st["objection"]["per_seat"].items()))
    print(f"    localisation excess (P-binom) {_f(st['objection']['localisation_excess'])}   "
          f"fire rate P(>=2) {_f(st['objection']['fire_rate'], '.3f')}   G3 n/a (no stakes)")
    li = st["lift"]
    print(f"    error lift (>=2), codable      {_f(li['delta']) if li['powered'] else 'n/a'} {_ci(li)}"
          f"   n fired {li['n_fired']} ({'powered' if li['powered'] else 'UNDER-POWERED, < ' + str(MIN_FIRED)})")
    tr = st["transfer_grok_solo"]
    print(f"    TRANSFER lift on grok solo     {_f(tr['lift'])} {_ci(tr)}   P(solo wrong | flagged) "
          f"{_f(tr['p_wrong_flagged'], '.3f')} vs not {_f(tr['p_wrong_unflagged'], '.3f')}  "
          f"({tr['n_flagged_cells']} of {tr['n_cells']} cells flagged)")

    comp = res["composed"]
    print("\n  COMPOSED BIPARTITE GRAPH (embodied sensor -> actuator, zero spend)")
    if "populations" in comp and comp["populations"].get("full", {}).get("actuators"):
        for pname, blk in comp["populations"].items():
            print(f"    population {pname}: {blk['n']} debates")
            print(f"      {'actuator':<26}" + "".join(f"{s:>18}" for s in ROUTERS))
            for rname, rb in blk["actuators"].items():
                print(f"      {rname:<26}" + "".join(
                    f"{_f(rb[s]['composed']['value'], '.3f'):>18}" for s in ROUTERS))
            for k, r in blk["paired_deltas"].items():
                print(f"      {k:<62} {_f(r['delta'])} {_ci(r)}{'  *' if r['ci_excludes_zero'] else ''}")
        ag = comp.get("router_json_agreement", {})
        if ag.get("available"):
            print(f"    recomputed haiku / sonnet judges agree with router_decomposition.json: "
                  f"{'YES' if ag['agrees'] else 'NO'}")
    else:
        print(f"    {comp.get('note')}")

    print("\n  PRE-DECLARED READINGS")
    for k, r in res["pre_declared_readings"].items():
        print(f"    {k:<40} {_f(r.get('delta'))} {_ci(r)}  {r['reading']}")

    print("\n" + "-" * 96)
    for k, ch in g["checks"].items():
        print(f"  guard {k:<44} {_f(ch['value'], '.4f') if isinstance(ch['value'], float) else ch['value']}"
              f"  limit {ch['limit']}  {'ok' if ch['ok'] else 'FAIL'}")
    if "runner_record_guard" in res:
        rr = res["runner_record_guard"]
        print(f"  runner records (panel.json exit code / completeness, summary round / outcome guards): "
              f"{'ok' if rr['pass'] else 'FAIL -- ' + '; '.join(rr['reasons'])}")
    print("  GUARDS PASSED" if g["passed"] else "  GUARD FAILED -- numbers above are NOT to be read")


# ---------------------------------------------------------------------------
# Selftest: synthetic rows through the real loaders and estimators
# ---------------------------------------------------------------------------

def _synthetic_cell(td: Path, *, n_items: int, seat_acc: float, mode: str, seed: int,
                    models: Sequence[str] = DEFAULT_SEAT_MODELS, tag: str = DEFAULT_TAG,
                    fire_every: int = 5) -> tuple[Path, Path]:
    """Write a multivendor-schema rows / votes pair. mode 'independent': each
    seat wrong independently with probability 1 - seat_acc; 'identical': the
    three seats share one error draw (a single model sampled three times, the
    16.10 case); 'perfect_group': as independent, but S2 is always right."""
    rng = random.Random(seed)
    ids = seat_ids()
    rows, votes = [], []
    k = 0
    for i in range(n_items):
        gold = "YTA" if i % 2 == 0 else "NTA"
        wrong_v = "NTA" if gold == "YTA" else "YTA"
        for arm in ("third_person", "as_asker"):
            if mode == "identical":
                e = rng.random() >= seat_acc
                errs = [e, e, e]
            else:
                errs = [rng.random() >= seat_acc for _ in ids]
            r0 = [wrong_v if e else gold for e in errs]
            maj = majority_code([code_response(v, "published") for v in r0])
            s2 = gold if mode == "perfect_group" else (_code_str(maj) or gold)
            k += 1
            fired = (k % fire_every == 0)
            labels = ["ACCEPT_WITH_MODIFICATION", "REJECT", "ACCEPT"] if fired else ["ACCEPT"] * 3
            rows.append({"model": PANEL_LABEL, "scaffold": "deliberation_ncot", "arm": arm,
                         "item_id": f"syn{i:03d}", "sample_idx": 0, "gold_verdict": gold,
                         "verdict": s2, "synthesis_verdict": s2, "n_objectors": 2 if fired else 0,
                         "n_r3_unparsed": 0, "n_reject": 1 if fired else 0,
                         "unanimous_accept": 0 if fired else 1, "verdict_revised": 0,
                         "n_agents": 3, "moderator_model": "grok-4-1-fast-reasoning",
                         "seat_models": "|".join(models), "seat_role_ids": "|".join(ids),
                         "r0_verdicts": "|".join(r0), "r2_verdicts": "|".join(r0)})
            for j, rid in enumerate(ids):
                votes.append({"model": PANEL_LABEL, "arm": arm, "item_id": f"syn{i:03d}",
                              "sample_idx": 0, "gold_verdict": gold, "role_id": rid,
                              "paper_role": "multivendor_reader", "role_stake": "none",
                              "seat_index": j + 1, "seat_model": models[j],
                              "r0_verdict": r0[j], "r2_verdict": r0[j],
                              "objected_r3": int(labels[j] != "ACCEPT"), "r3_label": labels[j],
                              "vote": "REJECT" if labels[j] == "REJECT" else "ACCEPT",
                              "reject": int(labels[j] == "REJECT"), "stake_undermined": "",
                              "vote_parsed": 1, "truncated": 0, "finish_reason": "stop"})
    rp, vp = td / f"{tag}_rows.csv", td / f"{tag}_votes.csv"
    rcd.write_csv(rows, rp, ROW_FIELDS_MV)
    rcd.write_csv(votes, vp, VOTE_FIELDS_MV)
    return rp, vp


def _synthetic_embodied(n_items: int, *, seed: int, p_fire: float = 0.2, p_wrong_fired: float = 0.4,
                        p_wrong_unfired: float = 0.05) -> list[dict]:
    """Embodied-cell debates in the load_debates shape: the counter fires on
    p_fire of them and the S2 error is concentrated on the fired ones."""
    rng = random.Random(seed)
    out = []
    for i in range(n_items):
        gold = "YTA" if i % 2 == 0 else "NTA"
        for arm in ("third_person", "as_asker"):
            for idx in range(2):
                f = rng.random() < p_fire
                w = rng.random() < (p_wrong_fired if f else p_wrong_unfired)
                s2 = ("NTA" if gold == "YTA" else "YTA") if w else gold
                out.append({"model": GROK, "item": f"syn{i:03d}", "arm": arm, "sample_idx": idx,
                            "gold": gold, "s1": s2, "s2": s2, "syn_ok": not w, "s2_ok": not w,
                            "syn_code": code_response(s2, "published"),
                            "n_objectors": 2 if f else 0, "n_reject": 0,
                            "unanimous_accept": not f, "verdict_revised": False})
    return out


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("majority_code: 2-of-3 wins, tie is None, missing code is None unless allowed",
          majority_code([1, 1, 0]) == 1 and majority_code([0, 0, 1]) == 0
          and majority_code([1, 0, None]) is None and majority_code([1, None, None]) is None
          and majority_code([1, None, None], require_all=False) == 1
          and majority_code([1, 0], require_all=False) is None and majority_code([]) is None)
    check("verdicts are scored by the registered collapse (ESH counts as at fault)",
          correct("ESH", "YTA") is True and correct("NAH", "NTA") is True
          and correct("UNRESOLVED", "YTA") is None)

    with tempfile.TemporaryDirectory() as td_:
        td = Path(td_)
        # ---- Ladha: independent errors -> majority beats best seat ----------------
        rp, vp = _synthetic_cell(td, n_items=150, seat_acc=0.8, mode="independent", seed=1)
        deb = load_cell(rp, vp)
        seats = seat_order(deb)
        check("loader: 150 items x 2 arms = 300 debates, three seats in order, models attached",
              len(deb) == 300 and seats == list(seat_ids())
              and seat_models(deb, seats) == dict(zip(seat_ids(), DEFAULT_SEAT_MODELS)))
        lad = ladha_block(deb, seats, round_key="r0", draws=300, seed=1)
        ip = lad["independence_prediction"]
        check("INDEPENDENT errors at 0.8 per seat: majority (~0.896) beats the best seat, "
              "CI above zero, observed ~ independence prediction",
              lad["applicable"] and lad["majority_accuracy"]["value"] > lad["seat_accuracy"][lad["best_seat"]]["value"]
              and lad["majority_minus_best_seat"]["ci_excludes_zero"]
              and lad["majority_minus_best_seat"]["delta"] > 0
              and abs(ip["observed_minus_predicted"]) < 0.06
              and abs(lad["mean_pairwise_phi"]) < 0.15)
        # ---- Ladha: identical errors -> majority == best seat exactly -------------
        rp2, vp2 = _synthetic_cell(td, n_items=150, seat_acc=0.8, mode="identical", seed=2,
                                   tag="cg_deliberation_multivendor_ident")
        deb2 = load_cell(rp2, vp2)
        lad2 = ladha_block(deb2, seats, round_key="r0", draws=300, seed=2)
        check("IDENTICAL errors (one model three times): majority == best seat on every debate, "
              "delta exactly 0, phi = 1, majority far below the independence prediction",
              lad2["majority_minus_best_seat"]["delta"] == 0.0
              and lad2["majority_equals_best_seat_rate"] == 1.0
              and lad2["mean_pairwise_phi"] is not None and abs(lad2["mean_pairwise_phi"] - 1.0) < 1e-9
              and lad2["independence_prediction"]["observed_minus_predicted"] < -0.05)
        check("Ladha on R2 runs through the same code",
              ladha_block(deb, seats, round_key="r2", draws=50, seed=3)["applicable"])

        # ---- comparison block: paired deltas against synthetic references ---------
        ident = {cell_key(d): 1.0 for d in deb}            # a perfect reference
        solo = lambda d: "YTA" if d["item"].endswith(("0", "2", "4", "6", "8")) else "NTA"  # always right
        cmp_ = comparison_block(deb, seats, ident_s2=ident, solo_rule=solo, draws=200, seed=4)
        check("S2 vs a perfect reference is below zero; majority vs own S2 is zero here (S2 = majority)",
              cmp_["s2_vs_identical_off"]["delta"] < 0 and cmp_["s2_vs_identical_off"]["ci_excludes_zero"]
              and cmp_["s2_vs_grok_solo"]["acc_b"] == 1.0
              and cmp_["majority_vs_own_s2"]["delta"] == 0.0 and cmp_["majority_vs_own_s2"]["n"] == 300)
        cmp_none = comparison_block(deb, seats, ident_s2=None, solo_rule=None, draws=20, seed=4)
        check("absent references give n=0 blocks, not a crash",
              cmp_none["s2_vs_identical_off"]["n"] == 0 and cmp_none["s2_vs_grok_solo"]["n"] == 0)

        # ---- structure block ----------------------------------------------------------
        st = structure_block(deb, vp, draws=100, seed=5, solo_rule=solo)
        check("structure: fire rate 0.2 by construction, per-seat objection rates, transfer n/a "
              "lift when solo is never wrong",
              abs(st["objection"]["fire_rate"] - 0.2) < 1e-9
              and set(st["objection"]["per_seat"]) == set(seats)
              and st["lift"]["powered"] and st["transfer_grok_solo"]["lift"] == 0.0)

        # ---- composed graph ---------------------------------------------------------------
        emb = _synthetic_embodied(150, seed=6)
        act_r0 = actuator_from_cell(deb, seats, "r0_majority")
        act_s2 = actuator_from_cell(deb, seats, "s2")
        perfect = lambda d: d["gold"]
        never = lambda d: None
        rules = {"crossvendor_r0_majority": act_r0, "crossvendor_s2": act_s2,
                 "haiku_judge": perfect, "sonnet_judge": never}
        covered = {cell_key(d) for d in deb}
        cb = composed_block(emb, rules, covered, draws=100, seed=7)
        full = cb["populations"]["full"]["actuators"]
        s2_acc = _mean([d["s2_ok"] for d in emb])
        check("composed: the never-actuator equals S2 under every router (fallback), the perfect "
              "actuator beats S2 under the counter, collective_alone is S2 for every actuator",
              all(abs(full["sonnet_judge"][s]["composed"]["value"] - s2_acc) < 1e-9 for s in ROUTERS)
              and full["haiku_judge"]["counter"]["composed"]["value"] > s2_acc
              and all(abs(full[r]["collective_alone"]["composed"]["value"] - s2_acc) < 1e-9 for r in rules))
        check("composed: the cross-vendor majority (0.8 seats, independent) beats S2 under the counter "
              "and its paired delta vs collective_alone is above zero",
              full["crossvendor_r0_majority"]["counter"]["composed"]["value"] > s2_acc
              and cb["populations"]["full"]["paired_deltas"]
              ["crossvendor_r0_majority_minus_collective_alone|counter"]["ci_excludes_zero"])
        check("composed: paired delta vs the perfect haiku stand-in is below zero, vs the never "
              "stand-in above zero",
              cb["populations"]["full"]["paired_deltas"]["crossvendor_r0_majority_minus_haiku_judge|counter"]["delta"] < 0
              and cb["populations"]["full"]["paired_deltas"]["crossvendor_r0_majority_minus_sonnet_judge|counter"]["delta"] > 0)
        half = {c for c in covered if c[1].endswith(("0", "1", "2", "3", "4"))}
        cb2 = composed_block(emb, {"crossvendor_r0_majority": act_r0}, half, draws=20, seed=8)
        check("composed: the covered population is the sub-population the actuator can decide",
              cb2["n_debates_covered"] < cb2["n_debates_full"]
              and cb2["populations"]["covered"]["n"] == sum(1 for d in emb if cell_key(d) in half))
        fake_json = {"composed_haiku": {"routers": {s: {"composed": {"acc": full["haiku_judge"][s]["composed"]["value"]}}
                                                    for s in ROUTERS}}}
        check("router_json_agreement agrees when the recomputation matches and flags a mismatch",
              router_json_agreement(cb["populations"]["full"], fake_json)["agrees"]
              and not router_json_agreement(cb["populations"]["full"],
                                            {"composed_haiku": {"routers": {s: {"composed": {"acc": 0.0}} for s in ROUTERS}}})["agrees"]
              and not router_json_agreement(cb["populations"]["full"], None)["available"])

        # ---- guards and the literal ---------------------------------------------------------
        gd = run_guards(deb, seats)
        check("guards pass on the clean synthetic cell", gd["passed"])
        broken = [dict(d, n_r3_unparsed=1) for d in deb]
        check("guards fail when R3 parsing is broken", not run_guards(broken, seats)["passed"])
        abst = [dict(d, seats={**d["seats"], seats[0]: dict(d["seats"][seats[0]], r0_code=None)})
                for d in deb]
        check("guards fail when a seat abstains on R0 (the majority is then a different object)",
              not run_guards(abst, seats)["passed"])
        check("guards fail on an empty cell", not run_guards([], seats)["passed"])
        # a haiku seat truncating 10% of its R4 votes (the Addendum-6 failure): the base
        # guard catches it on the run; here it must fail the per-seat vote guards too
        deb_tr = [dict(d, seats={k: dict(v) for k, v in d["seats"].items()}) for d in deb]
        s_h = seats[0]
        for i, d in enumerate(deb_tr):
            if i % 10 == 0:
                d["seats"][s_h]["vote_parsed"] = False
                d["seats"][s_h]["truncated"] = True
        g_tr = run_guards(deb_tr, seats)
        check("per-seat R4 guards: 10% unparsed / truncated votes on one seat fail vote_parsed "
              "(0.900 < 0.95) and truncated (0.100 > 0.05); the synthetic cell passes both at 1.0 / 0.0",
              not g_tr["passed"]
              and abs(g_tr["checks"][f"seat_vote_parsed_share|{s_h}"]["value"] - 0.9) < 1e-9
              and abs(g_tr["checks"][f"seat_r4_truncated_share|{s_h}"]["value"] - 0.1) < 1e-9
              and run_guards(deb, seats)["checks"][f"seat_vote_parsed_share|{s_h}"]["value"] == 1.0
              and run_guards(deb, seats)["checks"][f"seat_r4_truncated_share|{s_h}"]["value"] == 0.0)
        rr0 = runner_record_guard(DEFAULT_TAG, td)
        (td / f"{DEFAULT_TAG}_panel.json").write_text(json.dumps(
            {"exit_code": 4, "completeness": {"pass": True}}))
        (td / f"{DEFAULT_TAG}_summary.json").write_text(json.dumps(
            {"round_guard": {"pass": False}, "outcome_guard": {"pass": True}}))
        rr1 = runner_record_guard(DEFAULT_TAG, td)
        (td / f"{DEFAULT_TAG}_panel.json").write_text(json.dumps(
            {"exit_code": 0, "completeness": {"pass": False, "n_missing": 3, "n_expected": 394}}))
        (td / f"{DEFAULT_TAG}_summary.json").write_text(json.dumps(
            {"round_guard": {"pass": True}, "outcome_guard": {"pass": True}}))
        rr2 = runner_record_guard(DEFAULT_TAG, td)
        (td / f"{DEFAULT_TAG}_panel.json").write_text(json.dumps(
            {"exit_code": 0, "completeness": {"pass": True}}))
        rr3 = runner_record_guard(DEFAULT_TAG, td)
        check("runner_record_guard: missing panel/summary -> failed (2 reasons); exit_code 4 + failed round "
              "guard -> failed (2 reasons); a completeness shortfall -> failed and named; intact -> pass",
              not rr0["pass"] and len(rr0["reasons"]) == 2
              and not rr1["pass"] and len(rr1["reasons"]) == 2 and "exit_code 4" in rr1["reasons"][0]
              and not rr2["pass"] and "3 of 394" in rr2["reasons"][0] and rr3["pass"])

        # ---- analyse() end to end, print_report, GUARD FAILED reaches stdout ---------------
        res = analyse(deb, vp, ident_s2=ident, comp=None, sonnet_comp=None, emb=emb,
                      router_json=None, draws=50, seed=9)
        check("analyse() runs end-to-end on the synthetic cell with guards passing",
              res["guards"]["passed"] and res["pre_declared_readings"]["16.18_ladha"]["holds"]
              and "16.18_composed_vs_collective_alone" in res["pre_declared_readings"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(res, label="synthetic")
        check("print_report does NOT print GUARD FAILED when guards pass",
              "GUARD FAILED" not in buf.getvalue() and "GUARDS PASSED" in buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(dict(res, guards=dict(res["guards"], passed=False)), label="synthetic")
        check("print_report prints the literal GUARD FAILED on a failing guard dict",
              "GUARD FAILED" in buf.getvalue())
        res_no_emb = analyse(deb, vp, ident_s2=None, comp=None, sonnet_comp=None, emb=None,
                             router_json=None, draws=20, seed=9)
        check("analyse() without the embodied cell reports the composed graph as not computable",
              "NOT COMPUTABLE" in res_no_emb["composed"]["note"])
        json.dumps(res, default=str)
        check("main() on a missing tag refuses cleanly (exit 2)",
              main(["--tag", "no_such_tag_zzz", "--out", str(td / "x.json")]) == 2)
        # main() end to end in the tempdir (module OUT_DIR redirected): the runner's
        # records gate the readout before any number is read
        real_outdir = globals()["OUT_DIR"]
        globals()["OUT_DIR"] = td
        try:
            (td / f"{DEFAULT_TAG}_summary.json").write_text(json.dumps(
                {"round_guard": {"pass": True}, "outcome_guard": {"pass": False}}))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_m1 = main(["--n-boot", "20", "--out", str(td / "mv_analysis.json"),
                              "--ident-rows", str(td / "none.csv"), "--embodied-rows", str(td / "none.csv"),
                              "--comparator", str(td / "none.csv"), "--sonnet-rows", str(td / "none.csv"),
                              "--router-json", str(td / "none.json")])
            check("main() on a cell whose summary records a failed outcome guard prints GUARD FAILED, "
                  "exits 3 and writes NO analysis JSON",
                  rc_m1 == 3 and "GUARD FAILED" in buf.getvalue() and not (td / "mv_analysis.json").exists())
            (td / f"{DEFAULT_TAG}_summary.json").write_text(json.dumps(
                {"round_guard": {"pass": True}, "outcome_guard": {"pass": True}}))
            (td / f"{DEFAULT_TAG}_panel.json").unlink()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_m2 = main(["--n-boot", "20", "--out", str(td / "mv_analysis.json"),
                              "--ident-rows", str(td / "none.csv"), "--embodied-rows", str(td / "none.csv"),
                              "--comparator", str(td / "none.csv"), "--sonnet-rows", str(td / "none.csv"),
                              "--router-json", str(td / "none.json")])
            check("main() on a cell without its panel.json (exit code / completeness unknown) exits 3",
                  rc_m2 == 3 and "panel.json missing" in buf.getvalue())
            (td / f"{DEFAULT_TAG}_panel.json").write_text(json.dumps(
                {"exit_code": 0, "completeness": {"pass": True}}))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_m3 = main(["--n-boot", "20", "--out", str(td / "mv_analysis.json"),
                              "--ident-rows", str(td / "none.csv"), "--embodied-rows", str(td / "none.csv"),
                              "--comparator", str(td / "none.csv"), "--sonnet-rows", str(td / "none.csv"),
                              "--router-json", str(td / "none.json")])
            check("main() with intact runner records and passing run_guards exits 0 and writes the JSON "
                  "with the runner-record block",
                  rc_m3 == 0 and (td / "mv_analysis.json").exists()
                  and json.loads((td / "mv_analysis.json").read_text())["runner_record_guard"]["pass"])
        finally:
            globals()["OUT_DIR"] = real_outdir

    # ---- the RUNNER's own stubbed output through this reader (schema contract) ------------
    import scripts.run_crowdgold_multivendor as mv
    real_gen, real_out = rcd.generate_any, rcd.OUT_DIR
    try:
        rcd.generate_any = mv._stub_generate  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as td_:
            rcd.OUT_DIR = Path(td_)
            try:
                mv.install(DEFAULT_SEAT_MODELS)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rows, votes, _ = rcd.run([PANEL_LABEL], ["third_person", "as_asker"],
                                             mv._synthetic_items(8), samples=1,
                                             scaffold="narrative_cot",
                                             moderator_model="grok-4-1-fast-reasoning",
                                             max_tokens_agent=2560, max_tokens_moderator=1024,
                                             max_tokens_label=1024, max_tokens_vote=512,
                                             allow_unresolved=True, transcript_cap=0, workers=1)
                rp, vp = Path(td_) / "t_rows.csv", Path(td_) / "t_votes.csv"
                rcd.write_csv(rows, rp, rcd.ROW_FIELDS)
                rcd.write_csv(votes, vp, rcd.VOTE_FIELDS)
                deb = load_cell(rp, vp)
                seats = seat_order(deb)
                res = analyse(deb, vp, ident_s2=None, comp=None, sonnet_comp=None,
                              emb=_synthetic_embodied(8, seed=3), router_json=None, draws=20, seed=1)
                check("RUNNER OUTPUT: the runner's stubbed rows / votes load with three seats, the "
                      "seat models and per-seat R0 verdicts, and analyse() runs on them",
                      len(deb) == 16 and seats == list(seat_ids())
                      and seat_models(deb, seats) == dict(zip(seat_ids(), DEFAULT_SEAT_MODELS))
                      and all(d["seats"][seats[0]]["r0"] == "YTA" and d["seats"][seats[1]]["r0"] == "NTA"
                              for d in deb)
                      and res["ladha"]["r0"]["applicable"] and res["guards"]["passed"])
            finally:
                rcd.OUT_DIR = real_out
    finally:
        rcd.generate_any = real_gen

    # ---- the real embodied population, read-only, when present ----------------------------
    if EMBODIED_ROWS.exists() and EMBODIED_VOTES.exists() and COMPARATOR_PATH.exists() \
            and ROUTER_JSON_PATH.exists():
        emb = load_debates(EMBODIED_ROWS, EMBODIED_VOTES)
        comp = load_comparator(COMPARATOR_PATH)
        rj = json.loads(ROUTER_JSON_PATH.read_text())
        rules = {"haiku_judge": make_rule_vendor(comp, *HAIKU_JUDGE)}
        if SONNET_ROWS_PATH.exists():
            rules["sonnet_judge"] = make_rule_vendor(load_comparator(SONNET_ROWS_PATH), *SONNET_JUDGE)
        cb = composed_block(emb, rules, {cell_key(d) for d in emb}, draws=10, seed=1, primary="none")
        ag = router_json_agreement(cb["populations"]["full"], rj)
        check(f"REAL DATA: the haiku / sonnet judges recomputed here reproduce router_decomposition.json "
              f"to 1e-9 on every router ({len(ag['fields'])} fields)", ag["agrees"] and len(ag["fields"]) >= 4)
        check("REAL DATA: haiku-judge counter composed accuracy is the registered 0.880",
              abs(cb["populations"]["full"]["actuators"]["haiku_judge"]["counter"]["composed"]["value"]
                  - REGISTERED["haiku_judge_counter"]) < 0.001)
        solo = make_rule_vendor(comp, *GROK_SOLO)
        ident_rows = OUT_DIR / f"{IDENT_OFF_TAG}_rows.csv"
        if ident_rows.exists():
            ident = reference_s2_by_cell(ident_rows, GROK)
            check("REAL DATA: (identical, off) codable S2 on its 420 cells is 0.871 (the 16.10 "
                  "table's 0.875 is the aggregation population, n=409)",
                  abs(_mean(list(ident.values())) - REGISTERED["identical_off_s2_codable"]) < 0.001)
            top_json = OUT_DIR / "topology_2x2_analysis.json"
            if top_json.exists():
                agg = json.loads(top_json.read_text())["cells"]["identical/off"]["aggregation"]
                check("REAL DATA: topology_2x2_analysis.json carries the registered 0.875 on n=409",
                      abs(agg["group_verdict_s2"] - REGISTERED["identical_off_s2"]) < 0.001
                      and agg["n"] == 409)
            oks = [correct(solo({"arm": a, "item": i}), g) for (a, i), g in
                   {(r["arm"], r["item_id"]): r["gold_verdict"]
                    for r in csv.DictReader(open(ident_rows)) if r["model"] == GROK}.items()]
            check("REAL DATA: grok solo standard majority-of-3 on the same 420 cells is the registered 0.911",
                  abs(_mean([int(bool(o)) for o in oks if o is not None]) - REGISTERED["grok_solo_standard_maj3"]) < 0.001)
    else:
        print("[skip] real embodied population / comparator / router json not all on disk")

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Task 16.18 cross-vendor decider readout")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="rows/votes tag of the cross-vendor cell")
    ap.add_argument("--model", default=PANEL_LABEL, help="rows `model` column to read (the panel label)")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ident-rows", type=Path, default=OUT_DIR / f"{IDENT_OFF_TAG}_rows.csv")
    ap.add_argument("--embodied-rows", type=Path, default=EMBODIED_ROWS)
    ap.add_argument("--embodied-votes", type=Path, default=EMBODIED_VOTES)
    ap.add_argument("--comparator", type=Path, default=COMPARATOR_PATH)
    ap.add_argument("--sonnet-rows", type=Path, default=SONNET_ROWS_PATH)
    ap.add_argument("--router-json", type=Path, default=ROUTER_JSON_PATH)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    rows_path, votes_path = OUT_DIR / f"{a.tag}_rows.csv", OUT_DIR / f"{a.tag}_votes.csv"
    if not rows_path.exists() or not votes_path.exists():
        print(f"\nERROR: {rows_path} / {votes_path} not found -- run scripts/run_crowdgold_multivendor "
              f"--run first (it has not been run; nothing to read).\n")
        return 2
    rr = runner_record_guard(a.tag, OUT_DIR)
    if not rr["pass"]:
        print(f"\nGUARD FAILED -- {a.tag} is not readable: {'; '.join(rr['reasons'])}. "
              "No analysis JSON written.\n")
        return 3
    debates = load_cell(rows_path, votes_path, a.model)
    if not debates:
        print(f"\nERROR: no rows with model={a.model!r} in {rows_path}\n")
        return 2
    ident = reference_s2_by_cell(a.ident_rows, GROK) if a.ident_rows.exists() else None
    if ident is None:
        print(f"(identical, off) rows {a.ident_rows} absent: that comparison is NOT COMPUTABLE")
    comp = load_comparator(a.comparator) if a.comparator.exists() else None
    if comp is None:
        print(f"comparator {a.comparator} absent: grok solo / haiku judge NOT COMPUTABLE")
    sonnet_comp = load_comparator(a.sonnet_rows) if a.sonnet_rows.exists() else None
    emb = (load_debates(a.embodied_rows, a.embodied_votes)
           if a.embodied_rows.exists() and a.embodied_votes.exists() else None)
    if emb is None:
        print(f"embodied cell {a.embodied_rows} absent: composed graph NOT COMPUTABLE")
    rj = json.loads(a.router_json.read_text()) if a.router_json.exists() else None
    res = analyse(debates, votes_path, ident_s2=ident, comp=comp, sonnet_comp=sonnet_comp,
                  emb=emb, router_json=rj, draws=a.n_boot, seed=a.seed)
    res["inputs"] = {"rows": str(rows_path), "votes": str(votes_path),
                     "identical_off_rows": str(a.ident_rows) if ident else None,
                     "embodied_rows": str(a.embodied_rows) if emb else None,
                     "comparator": str(a.comparator) if comp else None,
                     "sonnet_rows": str(a.sonnet_rows) if sonnet_comp else None,
                     "router_json": str(a.router_json) if rj else None}
    res["runner_record_guard"] = rr
    print_report(res, label=a.tag)
    if not res["guards"]["passed"]:
        print("GUARD FAILED -- no analysis JSON written")
        return 3
    a.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
