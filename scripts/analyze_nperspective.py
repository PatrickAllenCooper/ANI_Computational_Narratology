"""
scripts/analyze_nperspective.py -- Addendum 16.21 readout: the k-loser
generalisation on the four-seat (n-perspective) AITA cell.

Reads the cell scripts/run_crowdgold_nperspective.py writes
(divergence_study_outputs/cg_deliberation_nperspective_{rows,votes}.csv) and
the registered embodied cell (cg_deliberation_*) for the paired comparison,
and produces, through the same definitions as scripts/analyze_topology_2x2.py
(load_rows-style records, per-seat objection marginals, Poisson-binomial
localisation, role-lock, fire rate, G3, lift read only at >= MIN_FIRED fired,
transfer to grok solo, S2 accuracy with item-clustered CIs):

  k                    per debate, the number of ADVOCATES the SYNTHESIS
                       verdict goes against: writer / counterparty by the
                       registered stake_undermined (published collapse), the
                       third party by its per-item stake from the screen
                       (none -> never a loser, never favoured). Recomputed here
                       from synthesis_verdict + third_party_stake and checked
                       against the row's k_losers_synthesis (guard).
  against-interest     R3 objections from advocates the synthesis FAVOURS
  count                (n_against_interest_r3), recomputed from the votes CSV
                       and checked against the row (guard).
  fire rate, lift      P(count >= 1) and P(wrong | fired) - P(wrong | not),
                       codable population, by k (1, 2), pooled, and by
                       synthesis verdict type (one_loser / both_party); read
                       only at >= 60 fired; item-clustered bootstrap CIs. The
                       registered stake-blind counter n_objectors >= 2 is
                       reported alongside for comparability with 16.10.
  role-lock            |P(at_fault) writer - counterparty| at r0 (the 16.10
                       definition) and, for the third party, |P(at_fault | stake
                       YTA) - P(at_fault | stake NTA)| at r0 plus P(r0 serves
                       own stake); per-seat r0 accuracy.
  G3                   R4 reject | own stake undermined minus | not, over the
                       three advocates (within-role), item-clustered CI.
  S2 accuracy          group verdict correctness by the registered five-token
                       collapse, overall and by k, with a PAIRED delta against
                       the embodied cell on the same (arm, item, sample)
                       debates and an item-clustered CI.
  transfer             P(grok-solo wrong | cell fired) - P(| not), grok solo =
                       cached standard majority-of-3 the cell never touched.
  prediction table     the pre-declared k-loser table (k=0 silence, k=1 sensor,
                       k=2 sensor with less coverage, k=n flooding) with each
                       row marked reachable / unreachable on this instrument
                       and the measured cell filled in where it exists.

GUARDS (the literal GUARD FAILED is printed, exit 4 returned and NO analysis
JSON written when any fails):
  * the runner's own records, read by main() BEFORE anything else:
    <tag>_summary.json must exist with round_guard.pass and outcome_guard.pass
    True (the base runner's truncation / empty-output / parse guard over all
    22 calls), and <tag>_completeness.json (every launched debate is a row;
    run_crowdgold_nperspective writes it) must exist with pass True. A
    missing record is a failed guard. The runner quarantines a failed run's
    CSVs as <tag>_GUARD_FAILED_*, but a CSV pair without its guard records
    is not a readable cell either.
  * run_guards on the rows/votes: S2-codable share >= 0.95; mean
    n_r3_unparsed <= 0.05; R4 vote_parsed share >= 0.95; the row's k and
    against-interest count agree with the recomputation on >= 0.99 of
    codable debates; every codable debate carries all four seats' votes on
    >= 0.99.

Usage
-----
  python -m scripts.analyze_nperspective --selftest
  python -m scripts.analyze_nperspective            # the cell on disk
  python -m scripts.analyze_nperspective --model <deployment>   # a later model in the same cell
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_actuator_ladder import COMPARATOR_PATH, correct, load_comparator, make_rule_vendor
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_unembodied_ablation import poisson_binomial, verdict_correct
from scripts.run_crowdgold_nperspective import (
    ADVOCATES_NP, DEFAULT_TAG, ROLE_IDS_NP, THIRD_PARTY_ID, favoured, losers,
    stake_undermined_np, verdict_type,
)
from scripts.run_phase1_quartet import OUT_DIR
from scripts.verdict_format import extract_verdict

ANALYSIS_PATH = OUT_DIR / "nperspective_analysis.json"
DEFAULT_MODEL = "grok-4-1-fast-reasoning"
EMBODIED_TAG = "cg_deliberation"
EMBODIED_SEATS = ("writer_advocate", "counterparty", "neutral_adjudicator")
R0_CAP = 2560
MIN_FIRED = 60
MIN_CODABLE_SHARE = 0.95
MAX_MEAN_UNPARSED = 0.05
MIN_AGREE = 0.99
MIN_VOTE_PARSED_SHARE = 0.95

#: The pre-declared k-loser table. "reachable" is a property of THIS
#: instrument (two opposed party advocates under the published collapse plus a
#: third party with stake YTA / NTA / none): k = 1 + [third party undermined].
PREDICTION_TABLE = (
    {"k": 0, "favoured": "n (all)", "mode": "SILENCE",
     "prediction": "nothing deviates; plain objections ~0; against-interest count ~0",
     "reachable": False,
     "why": "every codable verdict undermines exactly one of writer / counterparty"},
    {"k": 1, "favoured": "n-1", "mode": "SENSOR",
     "prediction": "against-interest count rare (fire << 0.5) with positive error lift",
     "reachable": True, "why": "one party loses; the third party is served or has no stake"},
    {"k": 2, "favoured": "n-2 = 1", "mode": "SENSOR (less coverage)",
     "prediction": "count comes from one favoured seat; same-sign lift, lower fire rate; the "
                   "stake-blind n_objectors >= 2 counter fires by construction (two losers) and "
                   "carries no information here",
     "reachable": True, "why": "one party AND the third party lose"},
    {"k": 3, "favoured": "0", "mode": "FLOODING",
     "prediction": "plain objections flood (P(>=2) > 0.5); against-interest count "
                   "structurally 0 (no favoured seat)",
     "reachable": False,
     "why": "writer and counterparty cannot both be undermined under the published collapse; "
            "the both-party verdict type (ESH/NAH) is the flooding stratum, reported separately"},
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _boot(by_item: dict, stat: Callable, *, draws: int, seed: int, alpha: float = 0.05):
    ids = sorted(by_item)
    if not ids:
        return None, None, None
    rng = random.Random(seed)
    point = stat([x for i in ids for x in by_item[i]])
    vals = sorted(v for v in (
        stat([x for i in (ids[rng.randrange(len(ids))] for _ in ids) for x in by_item[i]])
        for _ in range(draws)) if v is not None)
    if not vals or point is None:
        return point, None, None
    return point, vals[int(alpha / 2 * len(vals))], vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)]


def _int(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def load_votes(votes_path: Path, model: Optional[str], keep_items: Optional[set]) -> dict:
    """(arm, item, sample) -> {role_id: vote record}."""
    out: dict = defaultdict(dict)
    with open(votes_path) as f:
        for v in csv.DictReader(f):
            if model and v.get("model") != model:
                continue
            if keep_items is not None and v["item_id"] not in keep_items:
                continue
            key = (v["arm"], v["item_id"], int(v["sample_idx"]))
            out[key][v["role_id"]] = {
                "objected_r3": 1 if v.get("objected_r3") == "1" else 0,
                "reject": _int(v.get("reject"), 0),
                "vote_parsed": _int(v.get("vote_parsed"), 0),
                "stake_undermined": v.get("stake_undermined", ""),
                "role_stake": v.get("role_stake", ""),
            }
    return out


def load_np_rows(rows_path: Path, votes: dict, model: Optional[str],
                 keep_items: Optional[set]) -> list[dict]:
    """One record per debate with k and the against-interest count RECOMPUTED
    from synthesis_verdict + third_party_stake + the votes, next to the row's
    own columns (the guard compares them)."""
    out = []
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            if model and r.get("model") != model:
                continue
            if keep_items is not None and r["item_id"] not in keep_items:
                continue
            syn, ver = r.get("synthesis_verdict", ""), r.get("verdict", "")
            stake = (r.get("third_party_stake") or "none").upper()
            if stake not in ("YTA", "NTA"):
                stake = "none"
            key = (r["arm"], r["item_id"], int(r["sample_idx"]))
            seats = votes.get(key, {})
            ls = losers(syn, stake)
            fav = favoured(syn, stake)
            n_ai = (None if fav is None else
                    sum(seats[rid]["objected_r3"] for rid in fav if rid in seats))
            out.append({
                "item": r["item_id"], "arm": r["arm"], "sample_idx": int(r["sample_idx"]),
                "gold": r["gold_verdict"], "syn": syn, "verdict": ver,
                "tp_stake": stake, "tp_description": r.get("third_party_description", ""),
                "tp_stake_source": r.get("third_party_stake_source", ""),
                "verdict_type": verdict_type(syn),
                "k": None if ls is None else len(ls),
                "k_row": _int(r.get("k_losers_synthesis")),
                "n_favoured": None if fav is None else len(fav),
                "n_ai": n_ai,
                "n_ai_row": _int(r.get("n_against_interest_r3")),
                "n_objectors": _int(r.get("n_objectors"), 0),
                "n_r3_unparsed": _int(r.get("n_r3_unparsed"), 0),
                "n_seats_voted": len(seats),
                "codable": (code_response(syn, "published") is not None
                            and code_response(ver, "published") is not None),
                "s2_ok": verdict_correct(ver, r["gold_verdict"]),
                "seats": seats,
            })
    return out


def load_baseline(rows_path: Path, model: Optional[str], keep_items: Optional[set]) -> dict:
    """(arm, item, sample) -> s2_ok of the embodied cell."""
    out = {}
    if not rows_path.exists():
        return out
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            if model and r.get("model") != model:
                continue
            if keep_items is not None and r["item_id"] not in keep_items:
                continue
            out[(r["arm"], r["item_id"], int(r["sample_idx"]))] = verdict_correct(
                r.get("verdict", ""), r["gold_verdict"])
    return out


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------

def lift_block(recs: Sequence[dict], fired: Callable[[dict], bool], *, draws: int, seed: int) -> dict:
    """Fire rate on the whole population; lift on the CODABLE population, read
    only at >= MIN_FIRED fired (raw)."""
    n = len(recs)
    n_fired = sum(1 for r in recs if fired(r))
    cod = [r for r in recs if r["codable"]]
    fl = [r for r in cod if fired(r)]
    un = [r for r in cod if not fired(r)]
    by = defaultdict(list)
    for r in cod:
        by[r["item"]].append(r)

    def stat(rs):
        a = [r for r in rs if fired(r)]
        b = [r for r in rs if not fired(r)]
        if not a or not b:
            return None
        return ((1 - sum(bool(r["s2_ok"]) for r in a) / len(a))
                - (1 - sum(bool(r["s2_ok"]) for r in b) / len(b)))
    p, lo, hi = _boot(by, stat, draws=draws, seed=seed)
    return {
        "n": n, "n_codable": len(cod), "n_fired": n_fired,
        "fire_rate": (n_fired / n) if n else None,
        "powered": n_fired >= MIN_FIRED,
        "err_fired": (1 - sum(bool(r["s2_ok"]) for r in fl) / len(fl)) if fl else None,
        "err_unfired": (1 - sum(bool(r["s2_ok"]) for r in un) / len(un)) if un else None,
        "delta": p, "lo": lo, "hi": hi,
        "ratio": ((1 - sum(bool(r["s2_ok"]) for r in fl) / len(fl))
                  / max(1e-9, 1 - sum(bool(r["s2_ok"]) for r in un) / len(un))
                  if fl and un and (1 - sum(bool(r["s2_ok"]) for r in un) / len(un)) > 0 else None),
    }


def ai_fired(r: dict) -> bool:
    return (r["n_ai"] or 0) >= 1


def plain_fired(r: dict) -> bool:
    return r["n_objectors"] >= 2


def by_stratum(recs: Sequence[dict], *, draws: int, seed: int) -> dict:
    """The sensor by k, pooled, and by synthesis verdict type."""
    out = {"pooled": {"against_interest": lift_block(recs, ai_fired, draws=draws, seed=seed),
                      "plain_ge2": lift_block(recs, plain_fired, draws=draws, seed=seed + 1)}}
    for k in (0, 1, 2, 3):
        sub = [r for r in recs if r["k"] == k]
        out[f"k={k}"] = {
            "n": len(sub),
            "against_interest": lift_block(sub, ai_fired, draws=draws, seed=seed + 10 + k) if sub else None,
            "plain_ge2": lift_block(sub, plain_fired, draws=draws, seed=seed + 20 + k) if sub else None,
        }
    for vt in ("one_loser", "both_party"):
        sub = [r for r in recs if r["verdict_type"] == vt]
        out[vt] = {
            "n": len(sub),
            "against_interest": lift_block(sub, ai_fired, draws=draws, seed=seed + 30) if sub else None,
            "plain_ge2": lift_block(sub, plain_fired, draws=draws, seed=seed + 40) if sub else None,
            "p_objectors_ge2": (sum(1 for r in sub if r["n_objectors"] >= 2) / len(sub)) if sub else None,
        }
    for st in ("YTA", "NTA", "none"):
        sub = [r for r in recs if r["tp_stake"] == st]
        out[f"stake={st}"] = {"n": len(sub),
                              "n_k2": sum(1 for r in sub if r["k"] == 2),
                              "against_interest": lift_block(sub, ai_fired, draws=draws, seed=seed + 50)
                              if sub else None}
    return out


def per_seat_objection_rates(recs: Sequence[dict]) -> dict:
    per: dict = defaultdict(lambda: [0, 0])
    for r in recs:
        for rid, v in r["seats"].items():
            per[rid][0] += v["objected_r3"]
            per[rid][1] += 1
    marg = {k: a / b for k, (a, b) in per.items() if b}
    tot_o = sum(a for a, _ in per.values())
    tot_n = sum(b for _, b in per.values())
    return {"per_seat": marg, "pooled": tot_o / tot_n if tot_n else 0.0, "n_votes": tot_n}


def localisation4(recs: Sequence[dict], rates: dict) -> dict:
    """Observed objector-count distribution over FOUR seats vs the Poisson-
    binomial null from the per-seat marginals (analyze_unembodied_ablation's
    localisation hard-codes three seats)."""
    n = len(recs)
    n_seats = len(ROLE_IDS_NP)
    obs = {k: (sum(1 for r in recs if r["n_objectors"] == k) / n if n else None)
           for k in range(n_seats + 1)}
    ps = [rates["per_seat"].get(rid, 0.0) for rid in ROLE_IDS_NP]
    ind = poisson_binomial(ps)
    return {
        "per_seat_marginals": rates["per_seat"], "observed": obs,
        "independence_poisson_binomial": {k: ind[k] for k in range(n_seats + 1)},
        "fire_observed": (sum(obs[k] for k in range(2, n_seats + 1)) if n else None),
        "fire_independence": sum(ind[2:]),
        "localisation_excess": (obs[1] - ind[1]) if n else None,
    }


def r0_verdict_code(model: str, arm: str, item: str, idx: int, role_id: str) -> Optional[int]:
    p = rcd.call_cache_path(model, "narrative_cot", arm, item, idx, "r0", role_id, R0_CAP)
    if not p.exists():
        return None
    txt = json.loads(p.read_text()).get("output") or ""
    return code_response(extract_verdict(txt, rcd.INSTRUMENT), "published")


def role_lock(recs: Sequence[dict], model: str) -> dict:
    """Per-seat P(at_fault) at r0; the 16.10 lock on the party pair; the third
    party's lock by stake and P(r0 serves own stake)."""
    at = {s: [0, 0] for s in ROLE_IDS_NP}
    acc = {s: [0, 0] for s in ROLE_IDS_NP}
    tp_by_stake = {"YTA": [0, 0], "NTA": [0, 0], "none": [0, 0]}
    tp_serves = [0, 0]
    for r in recs:
        g = 1 if r["gold"] == "YTA" else 0
        for s in ROLE_IDS_NP:
            c = r0_verdict_code(model, r["arm"], r["item"], r["sample_idx"], s)
            if c is None:
                continue
            at[s][0] += c; at[s][1] += 1
            acc[s][0] += int(c == g); acc[s][1] += 1
            if s == THIRD_PARTY_ID:
                tp_by_stake[r["tp_stake"]][0] += c
                tp_by_stake[r["tp_stake"]][1] += 1
                if r["tp_stake"] in ("YTA", "NTA"):
                    tp_serves[0] += int(c == (1 if r["tp_stake"] == "YTA" else 0))
                    tp_serves[1] += 1
    p_at = {s: (at[s][0] / at[s][1] if at[s][1] else None) for s in ROLE_IDS_NP}
    seat_acc = {s: (acc[s][0] / acc[s][1] if acc[s][1] else None) for s in ROLE_IDS_NP}
    w, c = "writer_advocate_np", "counterparty_np"
    lock_pair = abs(p_at[w] - p_at[c]) if None not in (p_at[w], p_at[c]) else None
    tp_p = {k: (v[0] / v[1] if v[1] else None) for k, v in tp_by_stake.items()}
    lock_tp = (abs(tp_p["YTA"] - tp_p["NTA"]) if None not in (tp_p["YTA"], tp_p["NTA"]) else None)
    return {"p_at_fault": p_at, "seat_r0_accuracy": seat_acc, "n_r0_read": {s: at[s][1] for s in ROLE_IDS_NP},
            "role_lock_pair": lock_pair,
            "third_party": {"p_at_fault_by_stake": tp_p, "role_lock_by_stake": lock_tp,
                            "p_r0_serves_own_stake": (tp_serves[0] / tp_serves[1]) if tp_serves[1] else None,
                            "n_with_stake": tp_serves[1]}}


def g3_stake_concentration(recs: Sequence[dict], *, draws: int, seed: int) -> dict:
    sv = []
    for r in recs:
        for rid in ADVOCATES_NP:
            v = r["seats"].get(rid)
            if not v or v["stake_undermined"] in ("", None) or not v["vote_parsed"]:
                continue
            sv.append({"item": r["item"], "reject": v["reject"], "und": int(v["stake_undermined"]),
                       "role": rid})
    if not sv:
        return {"applicable": False}
    by = defaultdict(list)
    for v in sv:
        by[v["item"]].append(v)

    def stat(vs):
        u = [x["reject"] for x in vs if x["und"]]
        n = [x["reject"] for x in vs if not x["und"]]
        return None if not u or not n else sum(u) / len(u) - sum(n) / len(n)
    p, lo, hi = _boot(by, stat, draws=draws, seed=seed)
    per_role = {}
    for rid in ADVOCATES_NP:
        rs = [x for x in sv if x["role"] == rid]
        per_role[rid] = {"n": len(rs), "delta": stat(rs) if rs else None}
    return {"applicable": True, "n_stake_votes": len(sv), "delta": p, "lo": lo, "hi": hi,
            "per_role": per_role}


def s2_block(recs: Sequence[dict], baseline: dict, *, draws: int, seed: int) -> dict:
    cod = [r for r in recs if r["s2_ok"] is not None]
    by = defaultdict(list)
    for r in cod:
        by[r["item"]].append(r)
    acc_stat = lambda rs: (sum(bool(r["s2_ok"]) for r in rs) / len(rs)) if rs else None
    p, lo, hi = _boot(by, acc_stat, draws=draws, seed=seed)
    out = {"n_codable": len(cod), "accuracy": p, "lo": lo, "hi": hi, "by_k": {}}
    for k in (1, 2):
        sub = [r for r in cod if r["k"] == k]
        out["by_k"][k] = {"n": len(sub), "accuracy": acc_stat(sub)}
    paired = [{"item": r["item"], "d": int(bool(r["s2_ok"])) - int(bool(baseline[(r["arm"], r["item"], r["sample_idx"])]))}
              for r in cod if (r["arm"], r["item"], r["sample_idx"]) in baseline
              and baseline[(r["arm"], r["item"], r["sample_idx"])] is not None]
    byp = defaultdict(list)
    for x in paired:
        byp[x["item"]].append(x)
    dp, dlo, dhi = _boot(byp, lambda xs: (sum(x["d"] for x in xs) / len(xs)) if xs else None,
                         draws=draws, seed=seed + 1)
    out["paired_vs_embodied"] = {"n_pairs": len(paired), "delta": dp, "lo": dlo, "hi": dhi,
                                 "embodied_accuracy_on_pairs": (
                                     sum(int(bool(baseline[(r["arm"], r["item"], r["sample_idx"])]))
                                         for r in cod if (r["arm"], r["item"], r["sample_idx"]) in baseline
                                         and baseline[(r["arm"], r["item"], r["sample_idx"])] is not None)
                                     / len(paired)) if paired else None}
    return out


def transfer_grok_solo(recs: Sequence[dict], fired: Callable[[dict], bool], *, draws: int, seed: int,
                       comparator_path: Path = COMPARATOR_PATH) -> dict:
    if not comparator_path.exists():
        return {"available": False}
    comp = load_comparator(comparator_path)
    solo = make_rule_vendor(comp, DEFAULT_MODEL, "standard")
    fl_c = defaultdict(int); k_c = defaultdict(int); gold = {}
    for r in recs:
        c = (r["arm"], r["item"]); fl_c[c] += int(fired(r)); k_c[c] += 1; gold[c] = r["gold"]
    trec = []
    for c, g in gold.items():
        v = solo({"arm": c[0], "item": c[1]}); ok = correct(v, g) if v is not None else None
        if ok is None:
            continue
        trec.append({"item": c[1], "w": int(not ok), "f": int(fl_c[c] * 2 > k_c[c])})
    byt = defaultdict(list)
    for x in trec:
        byt[x["item"]].append(x)

    def tstat(rs):
        f = [x for x in rs if x["f"]]; u = [x for x in rs if not x["f"]]
        return None if not f or not u else sum(x["w"] for x in f) / len(f) - sum(x["w"] for x in u) / len(u)
    p, lo, hi = _boot(byt, tstat, draws=draws, seed=seed)
    tf = [x for x in trec if x["f"]]; tu = [x for x in trec if not x["f"]]
    return {"available": True, "n_cells": len(trec), "n_flagged_cells": len(tf),
            "p_wrong_flagged": (sum(x["w"] for x in tf) / len(tf)) if tf else None,
            "p_wrong_unflagged": (sum(x["w"] for x in tu) / len(tu)) if tu else None,
            "lift": p, "lo": lo, "hi": hi}


def run_guards(recs: Sequence[dict]) -> dict:
    n = len(recs)
    cod = [r for r in recs if r["codable"]]
    agree_k = sum(1 for r in cod if r["k_row"] == r["k"])
    agree_ai = sum(1 for r in cod if r["n_ai_row"] == r["n_ai"])
    full = sum(1 for r in cod if r["n_seats_voted"] == len(ROLE_IDS_NP))
    votes = [v for r in recs for v in r["seats"].values()]
    g = {
        "n_debates": n, "n_codable": len(cod),
        "s2_codable_share": (len(cod) / n) if n else 0.0,
        "mean_n_r3_unparsed": (sum(r["n_r3_unparsed"] for r in recs) / n) if n else 0.0,
        "vote_parsed_share": (sum(1 for v in votes if v["vote_parsed"]) / len(votes)) if votes else 0.0,
        "k_agrees_share": (agree_k / len(cod)) if cod else 0.0,
        "against_interest_agrees_share": (agree_ai / len(cod)) if cod else 0.0,
        "all_seats_voted_share": (full / len(cod)) if cod else 0.0,
    }
    g["pass"] = bool(n and g["s2_codable_share"] >= MIN_CODABLE_SHARE
                     and g["mean_n_r3_unparsed"] <= MAX_MEAN_UNPARSED
                     and g["vote_parsed_share"] >= MIN_VOTE_PARSED_SHARE
                     and g["k_agrees_share"] >= MIN_AGREE
                     and g["against_interest_agrees_share"] >= MIN_AGREE
                     and g["all_seats_voted_share"] >= MIN_AGREE)
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


def analyse(rows_path: Path, votes_path: Path, *, model: str, baseline_rows: Path,
            keep_items: Optional[set], draws: int, seed: int,
            comparator_path: Path = COMPARATOR_PATH) -> dict:
    votes = load_votes(votes_path, model, keep_items)
    recs = load_np_rows(rows_path, votes, model, keep_items)
    guards = run_guards(recs)
    rates = per_seat_objection_rates(recs)
    baseline = load_baseline(baseline_rows, model, keep_items)
    k_counts = {str(k): sum(1 for r in recs if r["k"] == k) for k in (0, 1, 2, 3)}
    k_counts["non_codable"] = sum(1 for r in recs if r["k"] is None)
    return {
        "model": model, "tag": DEFAULT_TAG, "seats": list(ROLE_IDS_NP),
        "n_debates": len(recs), "n_items": len({r["item"] for r in recs}),
        "n_codable": sum(1 for r in recs if r["codable"]),
        "k_distribution": k_counts,
        "stake_distribution": {s: sum(1 for r in recs if r["tp_stake"] == s) for s in ("YTA", "NTA", "none")},
        "stake_source": {s: sum(1 for r in recs if r["tp_stake_source"] == s)
                         for s in sorted({r["tp_stake_source"] for r in recs})},
        "guards": guards,
        "objection": {"per_seat": rates["per_seat"], "pooled": rates["pooled"],
                      **localisation4(recs, rates)},
        "role_lock": role_lock(recs, model),
        "sensor": by_stratum(recs, draws=draws, seed=seed),
        "g3": g3_stake_concentration(recs, draws=draws, seed=seed + 4),
        "s2": s2_block(recs, baseline, draws=draws, seed=seed + 5),
        "transfer_grok_solo": {
            "against_interest": transfer_grok_solo(recs, ai_fired, draws=draws, seed=seed + 6,
                                                   comparator_path=comparator_path),
            "plain_ge2": transfer_grok_solo(recs, plain_fired, draws=draws, seed=seed + 7,
                                            comparator_path=comparator_path)},
        "prediction_table": [dict(row, measured=_measured_row(row["k"], recs)) for row in PREDICTION_TABLE],
    }


def _measured_row(k: int, recs: Sequence[dict]) -> dict:
    sub = [r for r in recs if r["k"] == k]
    if not sub:
        return {"n": 0}
    cod = [r for r in sub if r["codable"]]
    fl = [r for r in cod if ai_fired(r)]; un = [r for r in cod if not ai_fired(r)]
    return {"n": len(sub),
            "fire_rate_against_interest": sum(1 for r in sub if ai_fired(r)) / len(sub),
            "p_plain_ge2": sum(1 for r in sub if plain_fired(r)) / len(sub),
            "err_fired": (1 - sum(bool(r["s2_ok"]) for r in fl) / len(fl)) if fl else None,
            "err_unfired": (1 - sum(bool(r["s2_ok"]) for r in un) / len(un)) if un else None,
            "n_fired": len(fl)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def _ci(b: Optional[dict], key="delta") -> str:
    if not b or b.get(key) is None:
        return "n/a"
    return f"{_f(b[key])} [{_f(b.get('lo'))}, {_f(b.get('hi'))}]"


def print_report(res: dict) -> None:
    print("\n" + "=" * 96)
    print(f"ADDENDUM 16.21 -- N PERSPECTIVES (k-loser generalisation)   model {res['model']}   "
          f"debates {res['n_debates']}  items {res['n_items']}  codable {res['n_codable']}")
    print("=" * 96)
    g = res["guards"]
    print(f"  guards: S2-codable {g['s2_codable_share']:.3f}  mean n_r3_unparsed {g['mean_n_r3_unparsed']:.4f}  "
          f"vote parsed {g['vote_parsed_share']:.3f}  "
          f"k agrees {g['k_agrees_share']:.3f}  count agrees {g['against_interest_agrees_share']:.3f}  "
          f"all seats voted {g['all_seats_voted_share']:.3f}  -> {'PASS' if g['pass'] else 'GUARD FAILED'}")
    if "runner_record_guard" in res:
        rr = res["runner_record_guard"]
        print(f"  runner records: {'PASS' if rr['pass'] else 'GUARD FAILED -- ' + '; '.join(rr['reasons'])}")
    print(f"  k distribution {res['k_distribution']}   third-party stake {res['stake_distribution']} "
          f"(source {res['stake_source']})")
    rl = res["role_lock"]
    print(f"\n  ROLE-LOCK  pair |writer - counterparty| {_f(rl['role_lock_pair'], '.3f')}   "
          f"third party |P(at_fault|YTA) - P(at_fault|NTA)| {_f(rl['third_party']['role_lock_by_stake'], '.3f')}   "
          f"P(r0 serves own stake) {_f(rl['third_party']['p_r0_serves_own_stake'], '.3f')} "
          f"(n {rl['third_party']['n_with_stake']})")
    print("     P(at_fault) " + "  ".join(f"{s.replace('_np', '')[:12]}={_f(v, '.3f')}" for s, v in rl["p_at_fault"].items()))
    print("     r0 accuracy " + "  ".join(f"{s.replace('_np', '')[:12]}={_f(v, '.3f')}" for s, v in rl["seat_r0_accuracy"].items()))
    ob = res["objection"]
    print(f"\n  OBJECTIONS  per seat " + "  ".join(f"{s.replace('_np', '')[:12]}={v:.3f}" for s, v in ob["per_seat"].items()))
    print(f"     localisation excess (P-binom, 4 seats) {_f(ob['localisation_excess'])}   "
          f"fire P(>=2) {_f(ob['fire_observed'], '.3f')} vs independence {ob['fire_independence']:.3f}")
    g3 = res["g3"]
    if g3.get("applicable"):
        print(f"  G3 (three advocates, within-role) {_ci(g3)}   per role " +
              "  ".join(f"{k.replace('_np', '')[:12]}={_f(v['delta'])} (n {v['n']})" for k, v in g3["per_role"].items()))
    s2 = res["s2"]
    print(f"\n  S2 ACCURACY {_f(s2['accuracy'], '.3f')} [{_f(s2['lo'], '.3f')}, {_f(s2['hi'], '.3f')}] (n {s2['n_codable']})   "
          f"by k: " + "  ".join(f"k={k} {_f(v['accuracy'], '.3f')} (n {v['n']})" for k, v in s2["by_k"].items()))
    pe = s2["paired_vs_embodied"]
    print(f"     paired vs embodied on the same debates: {_ci(pe)} (n pairs {pe['n_pairs']}, embodied "
          f"{_f(pe['embodied_accuracy_on_pairs'], '.3f')})")
    print("\n  THE SENSOR: against-interest objection count (fire = count >= 1), lift on codable, read at >= "
          f"{MIN_FIRED} fired")
    print(f"  {'stratum':<14}{'n':>6}{'fired':>7}{'fire':>8}{'err|fired':>11}{'err|not':>10}{'lift [CI]':>32}{'plain>=2 fire':>15}{'plain lift':>28}")
    sens = res["sensor"]
    for name in ("pooled", "k=1", "k=2", "one_loser", "both_party", "stake=YTA", "stake=NTA", "stake=none"):
        blk = sens.get(name) or {}
        ai = blk.get("against_interest")
        pl = blk.get("plain_ge2")
        if not ai:
            print(f"  {name:<14}{blk.get('n', 0):>6}{'':>7}{'(empty)':>8}")
            continue
        lift = _ci(ai) if ai["powered"] else f"underpowered (n fired {ai['n_fired']})"
        print(f"  {name:<14}{ai['n']:>6}{ai['n_fired']:>7}{_f(ai['fire_rate'], '.3f'):>8}"
              f"{_f(ai['err_fired'], '.3f'):>11}{_f(ai['err_unfired'], '.3f'):>10}{lift:>32}"
              f"{_f(pl['fire_rate'], '.3f') if pl else 'n/a':>15}"
              f"{(_ci(pl) if pl and pl['powered'] else 'underpowered') if pl else 'n/a':>28}")
    tr = res["transfer_grok_solo"]["against_interest"]
    if tr.get("available"):
        print(f"\n  TRANSFER to grok solo (standard maj-3): against-interest {_ci(tr, 'lift')}  "
              f"P(wrong|flagged) {_f(tr['p_wrong_flagged'], '.3f')} vs {_f(tr['p_wrong_unflagged'], '.3f')} "
              f"({tr['n_flagged_cells']}/{tr['n_cells']} cells)")
        tp = res["transfer_grok_solo"]["plain_ge2"]
        print(f"                                          plain >=2       {_ci(tp, 'lift')}  "
              f"P(wrong|flagged) {_f(tp['p_wrong_flagged'], '.3f')} vs {_f(tp['p_wrong_unflagged'], '.3f')}")
    print("\n  PREDICTION TABLE (pre-declared) vs measured")
    print(f"  {'k':<3}{'favoured':<10}{'mode':<24}{'reachable':<11}{'n':>5}{'AI fire':>9}{'err|f':>8}{'err|not':>9}{'plain>=2':>10}  prediction")
    for row in res["prediction_table"]:
        m = row["measured"]
        print(f"  {row['k']:<3}{row['favoured']:<10}{row['mode']:<24}{('yes' if row['reachable'] else 'NO'):<11}"
              f"{m.get('n', 0):>5}{_f(m.get('fire_rate_against_interest'), '.3f'):>9}"
              f"{_f(m.get('err_fired'), '.3f'):>8}{_f(m.get('err_unfired'), '.3f'):>9}"
              f"{_f(m.get('p_plain_ge2'), '.3f'):>10}  {row['prediction']}")
        if not row["reachable"]:
            print(f"     unreachable: {row['why']}")
    if not g["pass"]:
        print("\n  GUARD FAILED -- the numbers above are NOT readable as behaviour.")


# ---------------------------------------------------------------------------
# Selftest: synthetic rows/votes with planted structure, in a tempdir
# ---------------------------------------------------------------------------

def _synthetic(n_items: int = 360, seed: int = 3) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Rows + votes for the np cell and rows for an embodied baseline. Plants:
    writer NTA / counterparty YTA at r0 (lock 1.0); third party tracks its
    stake; every undermined advocate objects; a favoured advocate objects
    against interest on 30% of debates and those debates are wrong 60% of the
    time vs 6% otherwise (lift > 0 at k=1 and k=2, each clearing MIN_FIRED);
    one ESH debate in ten with every advocate objecting (flooding in the
    both-party stratum on both counters); the embodied baseline is right 85%
    of the time."""
    rng = random.Random(seed)
    rows, votes, base, r0 = [], [], [], {}
    stakes = ("YTA", "NTA", "none")
    for i in range(n_items):
        item = f"i{i:03d}"
        gold = "YTA" if i % 3 == 0 else "NTA"
        stake = stakes[i % 3]
        for arm in ("third_person", "as_asker"):
            u = rng.random()
            syn = "ESH" if u < 0.10 else ("YTA" if rng.random() < 0.45 else "NTA")
            fav = favoured(syn, stake) or ()
            ls = losers(syn, stake) or ()
            against = rng.random() < 0.30 and bool(fav)
            wrong = rng.random() < (0.60 if against else 0.06)
            gold_bit = 1 if gold == "YTA" else 0
            verdict = syn if syn == "ESH" else (
                ("NTA" if gold_bit else "YTA") if wrong else ("YTA" if gold_bit else "NTA"))
            if syn == "ESH":
                wrong = gold_bit == 0
            ai_seat = rng.choice(fav) if against else None
            seats = {}
            for rid in ROLE_IDS_NP:
                obj = (rid in ls) or (rid == ai_seat) or (syn == "ESH" and rid in ADVOCATES_NP)
                und = stake_undermined_np(rid, verdict, stake)
                seats[rid] = {"objected_r3": int(obj), "reject": int(obj and rng.random() < 0.5),
                              "vote_parsed": 1, "stake_undermined": "" if und is None else int(und),
                              "role_stake": ("at_fault" if stake == "YTA" else "not_at_fault" if stake == "NTA" else "none")
                              if rid == THIRD_PARTY_ID else ("not_at_fault" if rid.startswith("writer") else
                                                             "at_fault" if rid.startswith("counter") else "none")}
                votes.append({"model": DEFAULT_MODEL, "arm": arm, "item_id": item, "sample_idx": 0,
                              "role_id": rid, **seats[rid]})
                r0[(arm, item, rid)] = ("NTA" if rid.startswith("writer") else "YTA" if rid.startswith("counter")
                                        else (stake if stake != "none" else rng.choice(("YTA", "NTA")))
                                        if rid == THIRD_PARTY_ID else rng.choice(("YTA", "NTA")))
            n_obj = sum(v["objected_r3"] for v in seats.values())
            n_ai = sum(seats[rid]["objected_r3"] for rid in fav)
            rows.append({"model": DEFAULT_MODEL, "arm": arm, "item_id": item, "sample_idx": 0,
                         "gold_verdict": gold, "synthesis_verdict": syn, "verdict": verdict,
                         "third_party_stake": stake, "third_party_description": "the writer's sibling",
                         "third_party_stake_source": "heuristic",
                         "k_losers_synthesis": len(ls), "n_against_interest_r3": n_ai,
                         "n_objectors": n_obj, "n_r3_unparsed": 0})
            base_ok = rng.random() < 0.85
            base.append({"model": DEFAULT_MODEL, "arm": arm, "item_id": item, "sample_idx": 0,
                         "gold_verdict": gold,
                         "verdict": ("YTA" if gold_bit else "NTA") if base_ok else ("NTA" if gold_bit else "YTA")})
    return rows, votes, base, r0


def _write(path: Path, rows: list[dict]) -> None:
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _selftest() -> int:
    import tempfile
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("prediction table covers k = 0..3 with exactly k=1 and k=2 reachable",
          [r["k"] for r in PREDICTION_TABLE] == [0, 1, 2, 3]
          and [r["reachable"] for r in PREDICTION_TABLE] == [False, True, True, False])
    check("k and favoured come from the runner module (one definition, no drift)",
          losers("YTA", "NTA") == ("writer_advocate_np", "third_party_advocate_np")
          and favoured("YTA", "NTA") == ("counterparty_np",))
    rows, votes, base, r0 = _synthetic()
    real_out = rcd.OUT_DIR
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rp, vp, bp = td / "np_rows.csv", td / "np_votes.csv", td / "emb_rows.csv"
        _write(rp, rows); _write(vp, votes); _write(bp, base)
        # r0 cache files for the role-lock, in the tempdir
        rcd.OUT_DIR = td
        try:
            for (arm, item, rid), v in r0.items():
                p = rcd.call_cache_path(DEFAULT_MODEL, "narrative_cot", arm, item, 0, "r0", rid, R0_CAP)
                p.write_text(json.dumps({"output": f"Section 5 - Decision.\nVERDICT: {v}"}))
            res = analyse(rp, vp, model=DEFAULT_MODEL, baseline_rows=bp, keep_items=None,
                          draws=300, seed=1, comparator_path=td / "no_comparator.csv")
        finally:
            rcd.OUT_DIR = real_out
        print_report(res)
        g = res["guards"]
        check("guards PASS on the synthetic cell (k and count recomputed == row on every codable debate)",
              g["pass"] and g["k_agrees_share"] == 1.0 and g["against_interest_agrees_share"] == 1.0)
        kd = res["k_distribution"]
        check(f"k distribution has k=1 and k=2 only, k=0 and k=3 empty {kd}",
              kd["1"] > 0 and kd["2"] > 0 and kd["0"] == 0 and kd["3"] == 0)
        rl = res["role_lock"]
        check("planted role-lock: pair 1.000, third party by stake 1.000, P(serves own stake) 1.000",
              abs(rl["role_lock_pair"] - 1.0) < 1e-9
              and abs(rl["third_party"]["role_lock_by_stake"] - 1.0) < 1e-9
              and abs(rl["third_party"]["p_r0_serves_own_stake"] - 1.0) < 1e-9)
        s = res["sensor"]
        check("planted lift: against-interest count lifts error at k=1 and at k=2 (CIs exclude 0)",
              all(s[k]["against_interest"]["powered"] and s[k]["against_interest"]["lo"] > 0
                  for k in ("k=1", "k=2")))
        check("against-interest count never fires on the third party with no stake at k=1 "
              "beyond the favoured seats (count <= n_favoured everywhere)",
              all((r["n_ai"] or 0) <= (r["n_favoured"] or 0)
                  for r in load_np_rows(rp, load_votes(vp, DEFAULT_MODEL, None), DEFAULT_MODEL, None)
                  if r["codable"]))
        check("planted flooding: on every both-party (ESH) debate the plain counter fires AND the "
              "against-interest count fires (ESH under the published collapse still leaves a "
              "favoured seat; the count is structurally empty only at k=n)",
              s["both_party"]["n"] > 0 and abs(s["both_party"]["p_objectors_ge2"] - 1.0) < 1e-9
              and abs(s["both_party"]["against_interest"]["fire_rate"] - 1.0) < 1e-9)
        check("at k=2 the stake-blind >=2 counter fires by construction (two losers object) while "
              "the against-interest count does not",
              abs(s["k=2"]["plain_ge2"]["fire_rate"] - 1.0) < 1e-9
              and s["k=2"]["against_interest"]["fire_rate"] < 0.5)
        check("one-loser stratum: fire rate below 0.5 (a sensor, not a flood)",
              s["one_loser"]["against_interest"]["fire_rate"] < 0.5)
        s2 = res["s2"]
        check("S2 accuracy with CI and a paired delta against the baseline on every debate",
              s2["accuracy"] is not None and s2["lo"] is not None
              and s2["paired_vs_embodied"]["n_pairs"] == len(rows))
        check("G3 over the three advocates is applicable and positive on the planted votes",
              res["g3"]["applicable"] and res["g3"]["delta"] > 0
              and set(res["g3"]["per_role"]) == set(ADVOCATES_NP))
        ob = res["objection"]
        check("four-seat localisation: observed distribution sums to 1 over 0..4 objectors",
              abs(sum(ob["observed"].values()) - 1.0) < 1e-9 and set(ob["observed"]) == {0, 1, 2, 3, 4})
        check("transfer reports unavailable without a comparator file (never fabricated)",
              res["transfer_grok_solo"]["against_interest"] == {"available": False})
        pt = res["prediction_table"]
        check("prediction table rows carry measured n for k=1/k=2 and n=0 for the unreachable rows",
              pt[1]["measured"]["n"] > 0 and pt[2]["measured"]["n"] > 0
              and pt[0]["measured"]["n"] == 0 and pt[3]["measured"]["n"] == 0)
        # planted guard failure: corrupt the row's k on many debates
        bad = [dict(r, k_losers_synthesis=9) for r in rows]
        _write(rp, bad)
        res_bad = analyse(rp, vp, model=DEFAULT_MODEL, baseline_rows=bp, keep_items=None,
                          draws=50, seed=1, comparator_path=td / "none.csv")
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(res_bad)
        check("a row whose k disagrees with the recomputation fails the guard and the literal "
              "GUARD FAILED is printed", not res_bad["guards"]["pass"] and "GUARD FAILED" in buf.getvalue())
        _write(rp, rows)
        bad_votes = [dict(v, vote_parsed=0) if i % 10 == 0 else v for i, v in enumerate(votes)]
        _write(vp, bad_votes)
        res_bv = analyse(rp, vp, model=DEFAULT_MODEL, baseline_rows=bp, keep_items=None,
                         draws=50, seed=1, comparator_path=td / "none.csv")
        _write(vp, votes)
        check("10% unparsed R4 votes (the Addendum-6 failure, which the base guard sees only as "
              "truncation) fails the vote_parsed guard",
              not res_bv["guards"]["pass"] and abs(res_bv["guards"]["vote_parsed_share"] - 0.9) < 1e-9
              and res["guards"]["vote_parsed_share"] == 1.0)
        rr0 = runner_record_guard("np_cell", td)
        (td / "np_cell_summary.json").write_text(json.dumps(
            {"round_guard": {"pass": False}, "outcome_guard": {"pass": True}}))
        (td / "np_cell_completeness.json").write_text(json.dumps({"pass": True}))
        rr1 = runner_record_guard("np_cell", td)
        (td / "np_cell_summary.json").write_text(json.dumps(
            {"round_guard": {"pass": True}, "outcome_guard": {"pass": True}}))
        rr2 = runner_record_guard("np_cell", td)
        (td / "np_cell_completeness.json").write_text(json.dumps({"pass": False, "n_missing": 1, "n_expected": 720}))
        rr3 = runner_record_guard("np_cell", td)
        check("runner_record_guard: missing records -> failed (2 reasons); failed round guard -> failed; "
              "intact -> pass; completeness shortfall -> failed and named",
              not rr0["pass"] and len(rr0["reasons"]) == 2 and not rr1["pass"] and "round guard False" in rr1["reasons"][0]
              and rr2["pass"] and not rr3["pass"] and "1 of 720" in rr3["reasons"][0])
        # main() end to end in the tempdir (module OUT_DIR redirected): refuses on the
        # runner's records before reading a number; reads once they are intact
        _write(td / f"{DEFAULT_TAG}_rows.csv", rows); _write(td / f"{DEFAULT_TAG}_votes.csv", votes)
        _write(td / f"{EMBODIED_TAG}_rows.csv", base)
        real_outdir = globals()["OUT_DIR"]
        globals()["OUT_DIR"] = td
        rcd.OUT_DIR = td
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_m1 = main(["--n-boot", "20", "--out", str(td / "np_analysis.json")])
            check("main() on a cell without its summary / completeness records prints GUARD FAILED, "
                  "exits 4 and writes NO analysis JSON",
                  rc_m1 == 4 and "GUARD FAILED" in buf.getvalue() and not (td / "np_analysis.json").exists())
            (td / f"{DEFAULT_TAG}_summary.json").write_text(json.dumps(
                {"round_guard": {"pass": True}, "outcome_guard": {"pass": True}}))
            (td / f"{DEFAULT_TAG}_completeness.json").write_text(json.dumps({"pass": True}))
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_m2 = main(["--n-boot", "20", "--out", str(td / "np_analysis.json")])
            check("main() with intact runner records and passing run_guards exits 0 and writes the JSON",
                  rc_m2 == 0 and (td / "np_analysis.json").exists()
                  and json.loads((td / "np_analysis.json").read_text())["runner_record_guard"]["pass"])
            (td / f"{DEFAULT_TAG}_summary.json").write_text(json.dumps(
                {"round_guard": {"pass": True}, "outcome_guard": {"pass": False}}))
            (td / "np_analysis.json").unlink()
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_m3 = main(["--n-boot", "20", "--out", str(td / "np_analysis.json")])
            check("main() on a cell whose summary records a failed outcome guard exits 4 without a JSON",
                  rc_m3 == 4 and "outcome guard False" in buf.getvalue() and not (td / "np_analysis.json").exists())
            (td / f"{DEFAULT_TAG}_summary.json").write_text(json.dumps(
                {"round_guard": {"pass": True}, "outcome_guard": {"pass": True}}))
            _write(td / f"{DEFAULT_TAG}_votes.csv", bad_votes)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_m4 = main(["--n-boot", "20", "--out", str(td / "np_analysis.json")])
            check("main() with intact runner records but 10% unparsed votes exits 4 (analyzer guard) "
                  "without a JSON",
                  rc_m4 == 4 and "GUARD FAILED" in buf.getvalue() and not (td / "np_analysis.json").exists())
        finally:
            globals()["OUT_DIR"] = real_outdir
            rcd.OUT_DIR = real_out
    rp_real = OUT_DIR / f"{DEFAULT_TAG}_rows.csv"
    if rp_real.exists():
        print(f"  (cell present on disk: {rp_real.name}; run without --selftest for the readout)")
    else:
        print("  (no n-perspective cell on disk yet; the readout runs once the cell has been run)")
    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--baseline-tag", default=EMBODIED_TAG)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    rows_path, votes_path = OUT_DIR / f"{a.tag}_rows.csv", OUT_DIR / f"{a.tag}_votes.csv"
    if not rows_path.exists() or not votes_path.exists():
        print(f"no cell on disk at {rows_path} / {votes_path}; run scripts/run_crowdgold_nperspective.py first")
        return 2
    rr = runner_record_guard(a.tag, OUT_DIR)
    if not rr["pass"]:
        print(f"\n  GUARD FAILED -- {a.tag} is not readable: {'; '.join(rr['reasons'])}. "
              "No analysis JSON written.\n")
        return 4
    res = analyse(rows_path, votes_path, model=a.model, baseline_rows=OUT_DIR / f"{a.baseline_tag}_rows.csv",
                  keep_items=None, draws=a.n_boot, seed=a.seed)
    res["runner_record_guard"] = rr
    print_report(res)
    if not res["guards"]["pass"]:
        print(f"\n  GUARD FAILED -- {a.tag} is not readable (analyzer guards). No analysis JSON written.\n")
        return 4
    a.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
