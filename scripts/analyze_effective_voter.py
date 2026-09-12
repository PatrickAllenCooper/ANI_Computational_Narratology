"""
scripts/analyze_effective_voter.py -- Addendum 16.15.4: "one effective voter",
interval and scope. Zero spend: every number is replayed from the cached r0
opening statements (`cgd_<model>_narrative_cot_t2560_<arm>_<item>_<idx>_r0_<seat>.json`)
and the `<tag>_rows.csv` files the deliberation runs already wrote.

THE QUESTION (hostile-reviewer finding C2)
------------------------------------------
The record says the grok collective has one effective voter: the three-seat
majority equals the neutral seat on every codable debate, and a cross-validated
oracle over the seat emissions scores BELOW the neutral seat alone (gap -0.021,
no interval in the record). Two things were missing: (a) an item-clustered CI
on that oracle gap, and (b) the scope of the claim -- does it hold for every
model with a cached three-seat community, or only where role-lock does?

WHAT IS COMPUTED (registration, prereg "Addendum 16.15" / 16.15.4)
------------------------------------------------------------------
(a) grok: the cross-validated oracle gap. The oracle is the best FIXED function
    of the three r0 verdicts (writer's advocate, counterparty, neutral), each
    collapsed to an at-fault bit by the registered five-token collapse
    `code_response(extract_verdict(...), "published")`, i.e. a lookup table
    from the 8 possible profiles to a predicted at-fault bit. It is fitted on
    one half of the ITEMS (majority gold per profile; ties and unseen profiles
    fall back to the training half's majority gold class) and scored on the
    other half, in both directions, and the gap is oracle accuracy minus the
    neutral seat's accuracy on the same debates. Item-clustered bootstrap CI,
    4,000 draws, seed 13, with the fold assignment fixed per item (the split
    itself is drawn from seed 13; split sensitivity is reported separately as
    the mean / min / max over 200 further splits). The in-sample (non-CV)
    oracle is reported alongside as the headroom upper bound. A secondary,
    non-registered oracle over the RAW five-token profile (216 cells) is
    reported for comparison only.
(b) every model with a cached three-seat community (grok, sonnet, nano, haiku;
    the tags of analyze_rolelock_by_model.PANELS, which is the script that
    wrote papers/embodied_sensor/figures/fig1_rolelock.json): the share of
    debates on which the majority of the three r0 verdicts equals the neutral
    seat's verdict, the accuracy of each seat and of the majority on the same
    debates, the best seat, the writer's-advocate-minus-neutral and
    best-minus-neutral paired differences, and role-lock
    |P(at_fault | writer_advocate) - P(at_fault | counterparty)| at r0 computed
    exactly as fig1_rolelock.json was built (per-seat codable denominators,
    `analyze_rolelock_by_model.analyse_panel`), each with an item-clustered CI.

POPULATION
----------
(a) and the majority statistics use the TRIPLE population: debates on which
all three r0 verdicts are codable under the collapse (grok 1,642 of 1,680).
Role-lock uses each seat's own codable rows, as fig1_rolelock.json did. The
record's "1,663 of 1,663" was computed in-session on an undocumented
population; no r0-cache population reproduces that count (see the JSON field
`population_alternatives`), but majority == neutral holds on 100% of every
candidate population for grok.

SEEDS
-----
The registered seed is 13. Each panel is analysed with base seed
13 + 100 * panel_index (grok 13, sonnet 113, nano 213, haiku 313) and every
interval inside a panel is drawn with base + k, k fixed per statistic
(majority == neutral 1; seat accuracy 2-4; majority accuracy 5; seat minus
neutral 6-8; majority minus neutral 9; role-lock 10), so that no two
intervals share a resample stream. The primary 16.15.4(a) readout, the grok
CV oracle gap, uses seed 13 exactly (fold split and bootstrap). The seed
actually used is recorded next to every CI in the JSON (`seed` field) and
the rule is stated in the `estimator` block.

WHAT THIS ARTEFACT DOES NOT ANSWER
----------------------------------
The record's -0.021 (papers/embodied_sensor/headline_theory.html R3) is a CV
oracle over the FULL panel profile (nine seat verdicts, both moderator
verdicts, all labels and votes), scored 0.831 vs 0.852. That is a different
quantity from the registered "best fixed function of the three r0 verdicts",
and it is NOT COMPUTABLE here: it needs a replay over the r1/r2/moderator
caches, i.e. a separate zero-spend registration. The paper must either cite
the registered readout from this artefact or keep -0.021 without an
interval and with its full-profile definition.

GUARD
-----
Prints the literal string GUARD FAILED if, for any panel, a cached r0 call is
missing from disk, the non-codable share of triples exceeds 0.10, or fewer
than 20 triples survive. A failed panel is still written with `guard_ok`
false so the failure is auditable.

Usage
-----
  python -m scripts.analyze_effective_voter --selftest
  python -m scripts.analyze_effective_voter            # writes effective_voter.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_rolelock_by_model import PANELS, R0_CAP, SEATS, analyse_panel, r0_verdict_code
from scripts.analyze_unembodied_ablation import _boot, load_rows
from scripts.run_phase1_quartet import OUT_DIR
from scripts.verdict_format import extract_verdict

ANALYSIS_PATH = OUT_DIR / "effective_voter.json"
DEFAULT_DRAWS = 4000
DEFAULT_SEED = 13
N_SPLITS = 200
MAX_NONCODABLE_TRIPLE = 0.10
MIN_TRIPLES = 20
W, C, N = 0, 1, 2          # seat indices into `codes`, order of SEATS
GUARD_FAILED = "GUARD FAILED"
PANEL_SEED_STRIDE = 100    # base seed of panel i = seed + PANEL_SEED_STRIDE * i
#: per-statistic seed offsets inside a panel (base + k)
SEED_OFFSETS = {"majority_equals_neutral": 1, "seat_acc_writer_advocate": 2,
                "seat_acc_counterparty": 3, "seat_acc_neutral_adjudicator": 4,
                "majority_acc_on_triples": 5, "writer_advocate_minus_neutral": 6,
                "counterparty_minus_neutral": 7, "neutral_adjudicator_minus_neutral": 8,
                "majority_minus_neutral": 9, "role_lock": 10}


# ---------------------------------------------------------------------------
# Loading: one record per debate, three r0 verdicts each
# ---------------------------------------------------------------------------

def r0_verdict_raw(model: str, arm: str, item: str, idx: int, role_id: str) -> Optional[str]:
    """The raw five-token r0 verdict (None when the cached call is absent).
    The registered collapse of this string is asserted by --selftest to equal
    analyze_rolelock_by_model.r0_verdict_code on every grok row."""
    p = rcd.call_cache_path(model, "narrative_cot", arm, item, idx, "r0", role_id, R0_CAP)
    if not p.exists():
        return None
    txt = json.loads(p.read_text()).get("output") or ""
    return extract_verdict(txt, rcd.INSTRUMENT)


def load_panel(model: str, tag: str) -> Optional[list[dict]]:
    rows_path = OUT_DIR / f"{tag}_rows.csv"
    if not rows_path.exists():
        return None
    recs = []
    for r in load_rows(rows_path, model):
        raws, codes, missing = [], [], 0
        for s in SEATS:
            raw = r0_verdict_raw(model, r["arm"], r["item"], r["sample_idx"], s)
            if raw is None:
                missing += 1
            raws.append(raw)
            codes.append(None if raw is None else code_response(raw, "published"))
        recs.append({"item": r["item"], "arm": r["arm"], "sample_idx": r["sample_idx"],
                     "gold": 1 if r["gold"] == "YTA" else 0,
                     "codes": tuple(codes), "raws": tuple(raws), "missing": missing,
                     "debate_codable": r["codable"]})
    return recs


def triples(recs: Sequence[dict]) -> list[dict]:
    return [r for r in recs if None not in r["codes"]]


def majority(codes: Sequence[int]) -> int:
    return 1 if sum(codes) >= 2 else 0


# ---------------------------------------------------------------------------
# The oracle: best fixed function of the three r0 verdicts
# ---------------------------------------------------------------------------

def profile_bits(r: dict) -> tuple:
    return r["codes"]


def profile_raw(r: dict) -> tuple:
    return r["raws"]


def fit_table(train: Sequence[dict], profile: Callable[[dict], tuple]) -> tuple[dict, int]:
    """Lookup table profile -> at-fault bit that maximises training accuracy
    (majority gold per profile). Returns (table, default) where `default` is
    the training majority class, used for ties and unseen profiles."""
    tally: dict = defaultdict(lambda: [0, 0])
    for r in train:
        tally[profile(r)][r["gold"]] += 1
    n1 = sum(r["gold"] for r in train)
    default = 1 if 2 * n1 > len(train) else 0
    table = {}
    for p, (n0, n1_) in tally.items():
        table[p] = 1 if n1_ > n0 else (0 if n0 > n1_ else default)
    return table, default


def oracle_correct(train: Sequence[dict], test: Sequence[dict],
                   profile: Callable[[dict], tuple]) -> int:
    table, default = fit_table(train, profile)
    return sum(int(table.get(profile(r), default) == r["gold"]) for r in test)


def split_items(items: Sequence[str], seed: int) -> dict[str, int]:
    ids = sorted(set(items))
    random.Random(seed).shuffle(ids)
    half = len(ids) // 2
    return {i: (0 if k < half else 1) for k, i in enumerate(ids)}


def cv_oracle_stat(recs: Sequence[dict], fold: dict[str, int],
                   profile: Callable[[dict], tuple]) -> Optional[dict]:
    """Two-fold CV oracle accuracy, both directions, on the debates given (a
    bootstrap multiset keeps all copies of an item in that item's fold)."""
    a = [r for r in recs if fold[r["item"]] == 0]
    b = [r for r in recs if fold[r["item"]] == 1]
    if not a or not b:
        return None
    correct = oracle_correct(a, b, profile) + oracle_correct(b, a, profile)
    n = len(recs)
    neutral = sum(int(r["codes"][N] == r["gold"]) for r in recs)
    return {"oracle": correct / n, "neutral": neutral / n, "gap": (correct - neutral) / n}


def in_sample_oracle(recs: Sequence[dict], profile: Callable[[dict], tuple]) -> dict:
    table, default = fit_table(recs, profile)
    acc = oracle_correct(recs, recs, profile) / len(recs)
    neutral = sum(int(r["codes"][N] == r["gold"]) for r in recs) / len(recs)
    return {"oracle": acc, "neutral": neutral, "gap": acc - neutral,
            "table": {"|".join(map(str, k)): v for k, v in sorted(table.items(), key=str)},
            "default": default}


def cv_oracle(recs: Sequence[dict], *, draws: int, seed: int,
              profile: Callable[[dict], tuple] = profile_bits,
              n_splits: int = N_SPLITS) -> dict:
    items = [r["item"] for r in recs]
    fold = split_items(items, seed)
    point = cv_oracle_stat(recs, fold, profile)
    by_item: dict = defaultdict(list)
    for r in recs:
        by_item[r["item"]].append(r)
    _, lo, hi = _boot(by_item, lambda rs: (lambda s: None if s is None else s["gap"])(
        cv_oracle_stat(rs, fold, profile)), draws=draws, seed=seed)
    gaps = []
    for k in range(n_splits):
        s = cv_oracle_stat(recs, split_items(items, seed + 1 + k), profile)
        if s is not None:
            gaps.append(s["gap"])
    ins = in_sample_oracle(recs, profile)
    return {
        "n": len(recs), "n_items": len(set(items)), "split_seed": seed,
        "fold_sizes": [sum(1 for i in set(items) if fold[i] == 0),
                       sum(1 for i in set(items) if fold[i] == 1)],
        "cv_oracle_acc": point["oracle"], "neutral_acc": point["neutral"],
        "gap": point["gap"], "lo": lo, "hi": hi,
        "split_sensitivity": {"n_splits": len(gaps), "mean_gap": sum(gaps) / len(gaps) if gaps else None,
                              "min_gap": min(gaps) if gaps else None, "max_gap": max(gaps) if gaps else None},
        "in_sample": ins,
    }


# ---------------------------------------------------------------------------
# Majority, seats, lock
# ---------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def lock_stat(recs: Sequence[dict]) -> Optional[float]:
    """|P(at_fault | writer) - P(at_fault | counterparty)| with per-seat
    codable denominators, the fig1_rolelock.json construction."""
    w = [r["codes"][W] for r in recs if r["codes"][W] is not None]
    c = [r["codes"][C] for r in recs if r["codes"][C] is not None]
    if not w or not c:
        return None
    return abs(_mean(w) - _mean(c))


def guard(recs: Sequence[dict]) -> dict:
    n = len(recs)
    n_missing = sum(r["missing"] for r in recs)
    n_trip = len(triples(recs))
    noncod = (n - n_trip) / n if n else 1.0
    reasons = []
    if n_missing:
        reasons.append(f"{n_missing} cached r0 calls missing from disk")
    if noncod > MAX_NONCODABLE_TRIPLE:
        reasons.append(f"non-codable triple share {noncod:.3f} > {MAX_NONCODABLE_TRIPLE}")
    if n_trip < MIN_TRIPLES:
        reasons.append(f"only {n_trip} triples < {MIN_TRIPLES}")
    return {"ok": not reasons, "reasons": reasons, "n_debates": n, "n_triples": n_trip,
            "n_missing_r0_calls": n_missing, "noncodable_triple_share": noncod}


def analyse_panel_voter(short: str, model: str, tag: str, *, draws: int, seed: int,
                        with_oracle: bool = True) -> Optional[dict]:
    recs = load_panel(model, tag)
    if recs is None:
        return None
    return _panel_from_recs(recs, draws=draws, seed=seed, with_oracle=with_oracle,
                            model=model, tag=tag)


def _panel_from_recs(recs: Sequence[dict], *, draws: int, seed: int,
                     with_oracle: bool = False, model: Optional[str] = None,
                     tag: Optional[str] = None) -> dict:
    """The per-panel block on a list of debate records (real or synthetic).
    `fig1_reference` is filled only when a real (model, tag) is given."""
    g = guard(recs)
    trip = triples(recs)
    by_item: dict = defaultdict(list)
    for r in trip:
        by_item[r["item"]].append(r)
    by_item_all: dict = defaultdict(list)
    for r in recs:
        by_item_all[r["item"]].append(r)

    def ci(stat, by, k):
        """Item-clustered CI drawn with seed `seed + k`; the seed used is
        recorded next to the interval (16.15 audit: per-statistic seeds)."""
        p, lo, hi = _boot(by, stat, draws=draws, seed=seed + k)
        return {"point": p, "lo": lo, "hi": hi, "seed": seed + k, "draws": draws}

    out: dict = {"model": model, "tag": tag, "guard": g, "base_seed": seed,
                 "n_debates": len(recs), "n_items": len(by_item_all), "n_triples": len(trip),
                 "n_items_in_triples": len(by_item)}
    if not trip:
        return out
    K = SEED_OFFSETS
    eq = [int(majority(r["codes"]) == r["codes"][N]) for r in trip]
    out["majority_equals_neutral"] = {"count": sum(eq), "n": len(trip),
                                      **ci(lambda rs: _mean([int(majority(r["codes"]) == r["codes"][N]) for r in rs]), by_item, K["majority_equals_neutral"])}
    seat_acc = {}
    for i, s in enumerate(SEATS):
        seat_acc[s] = ci(lambda rs, i=i: _mean([int(r["codes"][i] == r["gold"]) for r in rs]), by_item, K[f"seat_acc_{s}"])
    out["seat_acc_on_triples"] = seat_acc
    out["majority_acc_on_triples"] = ci(lambda rs: _mean([int(majority(r["codes"]) == r["gold"]) for r in rs]), by_item, K["majority_acc_on_triples"])
    best = max(SEATS, key=lambda s: seat_acc[s]["point"])
    out["best_seat"] = best
    # paired seat-minus-neutral differences; one bootstrap per seat so that
    # "best minus neutral" and "writer minus neutral" are the SAME numbers
    # whenever the best seat is the writer's advocate
    out["seat_minus_neutral"] = {
        s: ci(lambda rs, i=i: _mean([int(r["codes"][i] == r["gold"]) - int(r["codes"][N] == r["gold"]) for r in rs]), by_item, K[f"{s}_minus_neutral"])
        for i, s in enumerate(SEATS)}
    out["best_minus_neutral"] = out["seat_minus_neutral"][best]
    out["writer_minus_neutral"] = out["seat_minus_neutral"][SEATS[W]]
    out["majority_minus_neutral"] = ci(lambda rs: _mean([int(majority(r["codes"]) == r["gold"]) - int(r["codes"][N] == r["gold"]) for r in rs]), by_item, K["majority_minus_neutral"])
    out["role_lock"] = ci(lock_stat, by_item_all, K["role_lock"])
    out["p_at_fault_r0"] = {s: _mean([r["codes"][i] for r in recs if r["codes"][i] is not None])
                            for i, s in enumerate(SEATS)}
    out["profile_counts"] = {"|".join(map(str, p)): {"n": n_, "gold_at_fault": sum(r["gold"] for r in trip if r["codes"] == p)}
                             for p, n_ in sorted(Counter(r["codes"] for r in trip).items())}
    # the fig1_rolelock.json construction, reused verbatim, for the record
    ref = analyse_panel(model, tag) if model and tag else None
    out["fig1_reference"] = {"n_r0": ref["n_r0"], "role_lock": ref["role_lock"],
                             "seat_acc_per_seat_codable": ref["seat_acc"],
                             "p_at_fault": ref["p_at_fault"]} if ref else None
    if with_oracle:
        out["cv_oracle_bits"] = cv_oracle(trip, draws=draws, seed=seed, profile=profile_bits)
        out["cv_oracle_raw_tokens_secondary"] = cv_oracle(trip, draws=draws, seed=seed, profile=profile_raw)
    return out


def population_alternatives(recs: Sequence[dict]) -> dict:
    """Candidate populations for the record's '1,663 of 1,663' -- none of
    them reproduces the count, all of them give majority == neutral on 100%."""
    def maj_codable(codes):
        c = [v for v in codes if v is not None]
        if len(c) < 2 or (len(c) == 2 and sum(c) == 1):
            return None
        return 1 if 2 * sum(c) > len(c) else 0
    pops = {
        "all_three_r0_codable": [r for r in recs if None not in r["codes"]],
        "majority_defined_and_neutral_codable": [r for r in recs if maj_codable(r["codes"]) is not None and r["codes"][N] is not None],
        "all_three_r0_codable_and_debate_codable": [r for r in recs if None not in r["codes"] and r["debate_codable"]],
        "neutral_codable_and_debate_codable": [r for r in recs if r["codes"][N] is not None and r["debate_codable"]],
    }
    out = {}
    for k, p in pops.items():
        eqn = sum(1 for r in p if maj_codable(r["codes"]) == r["codes"][N])
        out[k] = {"n": len(p), "majority_equals_neutral": eqn,
                  "neutral_acc": _mean([int(r["codes"][N] == r["gold"]) for r in p])}
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def _ci(d: dict, fmt=".3f") -> str:
    return f"{_f(d['point'], fmt)} [{_f(d['lo'], fmt)}, {_f(d['hi'], fmt)}]"


def print_report(res: dict) -> None:
    print("\n" + "=" * 100)
    print("ADDENDUM 16.15.4 -- ONE EFFECTIVE VOTER: interval and scope")
    print("=" * 100)
    g = res["panels"].get("grok")
    if g and "cv_oracle_bits" in g:
        o = g["cv_oracle_bits"]
        print(f"\n(a) grok CV oracle gap, best fixed function of the three r0 verdicts (bits), n={o['n']} triples, "
              f"{o['n_items']} items, folds {o['fold_sizes']}")
        print(f"    CV oracle acc {o['cv_oracle_acc']:.3f}   neutral acc {o['neutral_acc']:.3f}   "
              f"gap {o['gap']:+.3f} [{_f(o['lo'])}, {_f(o['hi'])}]   (record: -0.021, no CI)")
        s = o["split_sensitivity"]
        print(f"    split sensitivity over {s['n_splits']} splits: mean {_f(s['mean_gap'])} "
              f"min {_f(s['min_gap'])} max {_f(s['max_gap'])}")
        i = o["in_sample"]
        print(f"    in-sample oracle (headroom upper bound) acc {i['oracle']:.3f}, gap {i['gap']:+.3f}; table {i['table']}")
        o2 = g["cv_oracle_raw_tokens_secondary"]
        print(f"    secondary (NOT registered) raw five-token profile: CV oracle {o2['cv_oracle_acc']:.3f}, "
              f"gap {o2['gap']:+.3f} [{_f(o2['lo'])}, {_f(o2['hi'])}], in-sample gap {o2['in_sample']['gap']:+.3f}")
    print(f"\n(b) every cached three-seat community")
    print(f"  {'model':7s} {'n':>5s} {'trip':>5s}  {'maj==neutral':>22s}  {'lock':>22s}  "
          f"{'acc writer':>22s}  {'acc counterp':>22s}  {'acc neutral':>22s}  {'acc majority':>22s}  best")
    for k, p in res["panels"].items():
        if p is None:
            print(f"  {k:7s}  (rows file absent)")
            continue
        if "majority_equals_neutral" not in p:
            print(f"  {k:7s} {p['n_debates']:5d} {p['n_triples']:5d}  (no triples)")
            continue
        sa = p["seat_acc_on_triples"]
        print(f"  {k:7s} {p['n_debates']:5d} {p['n_triples']:5d}  {_ci(p['majority_equals_neutral']):>22s}  "
              f"{_ci(p['role_lock']):>22s}  {_ci(sa[SEATS[0]]):>22s}  {_ci(sa[SEATS[1]]):>22s}  "
              f"{_ci(sa[SEATS[2]]):>22s}  {_ci(p['majority_acc_on_triples']):>22s}  {p['best_seat']}")
    print()
    for k, p in res["panels"].items():
        if not p or "majority_equals_neutral" not in p:
            continue
        print(f"  {k:7s} writer - neutral {_ci(p['writer_minus_neutral'], '+.3f')}   "
              f"best({p['best_seat'].split('_')[0]}) - neutral {_ci(p['best_minus_neutral'], '+.3f')}   "
              f"majority - neutral {_ci(p['majority_minus_neutral'], '+.3f')}   "
              f"guard {'ok' if p['guard']['ok'] else GUARD_FAILED}")
        if "cv_oracle_bits" in p and k != "grok":
            o = p["cv_oracle_bits"]
            print(f"          CV oracle gap (bits, extension) {o['gap']:+.3f} [{_f(o['lo'])}, {_f(o['hi'])}]  "
                  f"in-sample gap {o['in_sample']['gap']:+.3f}")
    print_agreement(res)
    if not res["guard_ok"]:
        print(f"\n{GUARD_FAILED}: " + "; ".join(res["guard_failures"]))
    print()


#: the auditor's motivating estimates, bracketed in the 16.15.4 registration
#: and the 16.8 preamble; they are not results until reproduced here.
REPLAY = {
    "grok_oracle_gap": -0.021,          # 16.15.4 / 16.8 preamble, "no CI in the record"
    "grok_majority_equals_neutral": 1.0,  # "1,663 of 1,663"
    "haiku_majority_equals_neutral_share": 0.963,
    "haiku_n_triples": 187,
    "haiku_writer_acc": 0.818,
    "haiku_neutral_acc": 0.701,
}


def replay_agreement(res: dict) -> dict:
    """Does the registered recomputation agree with each bracketed value?"""
    g, h = res["panels"].get("grok"), res["panels"].get("haiku")
    out = {}
    if g and "cv_oracle_bits" in g:
        o = g["cv_oracle_bits"]
        out["grok_oracle_gap"] = {
            "replay": REPLAY["grok_oracle_gap"], "computed": o["gap"], "lo": o["lo"], "hi": o["hi"],
            "agrees": o["lo"] is not None and o["lo"] <= REPLAY["grok_oracle_gap"] <= o["hi"],
            "same_quantity": False,
            "note": "the record's -0.021 was a 2-fold CV oracle over a much richer profile (nine seat "
                    "verdicts, both moderator verdicts, all labels and votes; headline_theory.html R3); "
                    "the registered readout is over the three r0 verdicts only, where the in-sample "
                    "oracle is exactly the neutral seat, so the gap is 0 by construction and can only "
                    "go negative through unseen profiles under CV",
            "paper_must_pick_one": (
                f"cite the registered readout ({o['gap']:+.3f} [{_f(o['lo'])}, {_f(o['hi'])}], n {o['n']}, "
                f"{o['n_items']} items: the best fixed function of the three opening verdicts IS the "
                f"neutral seat), OR keep -0.021 explicitly without an interval and with its "
                f"full-profile definition; this artefact does not attach an interval to -0.021")}
        out["grok_majority_equals_neutral"] = {
            "replay": REPLAY["grok_majority_equals_neutral"], "replay_n": 1663,
            "computed": g["majority_equals_neutral"]["point"], "computed_n": g["majority_equals_neutral"]["n"],
            "agrees": g["majority_equals_neutral"]["point"] == 1.0,
            "replay_n_reproduced": False,
            "note": "share agrees (100%); the record's n of 1,663 is not reproduced by any r0-cache "
                    "population (see population_alternatives_grok)",
            "paper_wording": (
                f"majority equals the neutral seat on {g['majority_equals_neutral']['count']:,} of "
                f"{g['majority_equals_neutral']['n']:,} debates with three codable opening verdicts "
                f"({g['n_items_in_triples']} items); drop 1,663")}
    if h and "majority_equals_neutral" in h:
        m = h["majority_equals_neutral"]
        out["haiku_majority_equals_neutral"] = {
            "replay": REPLAY["haiku_majority_equals_neutral_share"], "replay_n": REPLAY["haiku_n_triples"],
            "computed": m["point"], "computed_n": m["n"],
            "agrees": round(m["point"], 3) == REPLAY["haiku_majority_equals_neutral_share"] and m["n"] == REPLAY["haiku_n_triples"]}
        wa = h["seat_acc_on_triples"][SEATS[W]]["point"]
        na = h["seat_acc_on_triples"][SEATS[N]]["point"]
        ma = h["majority_acc_on_triples"]["point"]
        out["haiku_writer_acc"] = {"replay": REPLAY["haiku_writer_acc"], "computed": wa,
                                   "agrees": round(wa, 3) == REPLAY["haiku_writer_acc"]}
        wn = h["writer_minus_neutral"]
        out["haiku_neutral_acc"] = {
            "replay": REPLAY["haiku_neutral_acc"], "computed_neutral_on_triples": na,
            "computed_neutral_per_seat_codable": h["fig1_reference"]["seat_acc_per_seat_codable"][SEATS[N]] if h["fig1_reference"] else None,
            "computed_majority_on_triples": ma,
            "agrees": round(na, 3) == REPLAY["haiku_neutral_acc"],
            "replay_is_majority_acc": round(ma, 3) == REPLAY["haiku_neutral_acc"],
            "note": "the replay's 0.701 is the MAJORITY's accuracy on the 187 triples, not the neutral "
                    "seat's; the neutral seat scores 0.717 on the triples (0.708 on its 192 codable rows)",
            "paper_wording": (
                f"writer's advocate {wa:.3f} vs neutral {na:.3f} (paired {wn['point']:+.3f} "
                f"[{_f(wn['lo'])}, {_f(wn['hi'])}], n {h['n_triples']}); the majority ({ma:.3f}) tracks "
                f"the neutral and is below the best seat; cite the paired seat difference, not the "
                f"extension CV oracle gap")}
    return out


def print_agreement(res: dict) -> None:
    print("agreement with the registration's bracketed replay values:")
    for k, v in res.get("replay_agreement", {}).items():
        comp = v.get("computed", v.get("computed_neutral_on_triples"))
        print(f"  {k:36s} replay {v['replay']:+.3f}  computed {comp:+.3f}  "
              f"{'AGREES' if v['agrees'] else 'DISAGREES'}" + (f"  -- {v['note']}" if v.get("note") else ""))


def run(*, draws: int, seed: int) -> dict:
    res: dict = {"registration": "Addendum 16.15.4", "estimator": {
        "bootstrap": "item-clustered percentile", "draws": draws, "seed": seed,
        "seed_rule": (f"registered seed {seed}; panel i uses base seed {seed} + {PANEL_SEED_STRIDE} * i "
                      f"({', '.join(f'{k} {seed + PANEL_SEED_STRIDE * i}' for i, k in enumerate(PANELS))}); "
                      f"each interval inside a panel is drawn with base + k, k per statistic as in "
                      f"`seed_offsets`, and the seed used is recorded next to every CI; the grok CV "
                      f"oracle gap (the primary 16.15.4(a) readout) uses seed {seed} exactly"),
        "seed_offsets": dict(SEED_OFFSETS),
        "collapse": "code_response(extract_verdict(r0), 'published')",
        "oracle": "2-fold CV over items, both directions, lookup table on the three r0 at-fault bits, "
                  "ties/unseen -> training majority class; fold fixed per item inside the bootstrap"},
        "not_computable_here": {
            "record_full_profile_oracle_gap_minus_0_021": (
                "the record's -0.021 (headline_theory.html R3) is a CV oracle over the full panel "
                "profile (nine seat verdicts, both moderator verdicts, all labels and votes), 0.831 vs "
                "0.852; it is a different quantity from the registered three-r0-verdict oracle and "
                "needs a replay over the r1/r2/moderator caches -- a separate zero-spend registration")},
        "panels": {}, "guard_ok": True, "guard_failures": []}
    for i, (short, (model, tag)) in enumerate(PANELS.items()):
        p = analyse_panel_voter(short, model, tag, draws=draws, seed=seed + PANEL_SEED_STRIDE * i)
        res["panels"][short] = p
        if p and not p["guard"]["ok"]:
            res["guard_ok"] = False
            res["guard_failures"].append(f"{short}: " + "; ".join(p["guard"]["reasons"]))
    grok = load_panel(*PANELS["grok"])
    if grok:
        res["population_alternatives_grok"] = population_alternatives(grok)
    res["replay_agreement"] = replay_agreement(res)
    return res


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _synthetic(kind: str, n_items: int = 200) -> list[dict]:
    """Deterministic synthetic panels. 'lock1': writer always not-at-fault,
    counterparty always at-fault, neutral = gold with 20% flips -> lock 1.
    'lock0_good_advocate': writer = gold, counterparty = gold with 30% flips
    and neutral = gold with 40% flips, both flip sets balanced across gold
    classes so P(at_fault) is 0.5 for every seat -> lock 0."""
    recs = []
    for i in range(n_items):
        gold = i % 2
        if kind == "lock1":
            w, c = 0, 1
            nn = gold if i % 10 < 8 else 1 - gold
        elif kind == "lock0_good_advocate":
            w = gold
            c = gold if i % 20 < 14 else 1 - gold
            nn = gold if i % 20 < 12 else 1 - gold
        else:
            raise ValueError(kind)
        recs.append({"item": f"i{i}", "arm": "a", "sample_idx": 0, "gold": gold,
                     "codes": (w, c, nn), "raws": ("YTA" if w else "NTA", "YTA" if c else "NTA", "YTA" if nn else "NTA"),
                     "missing": 0, "debate_codable": True})
    return recs


def _selftest(draws: int = 300) -> int:
    fails = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("five-token collapse: ESH->1, YTA->1, NTA->0, NAH->0, NOVERDICT->None",
          (code_response("ESH", "published"), code_response("YTA", "published"), code_response("NTA", "published"),
           code_response("NAH", "published"), code_response("NOVERDICT", "published")) == (1, 1, 0, 0, None))
    check("majority of bits", (majority((0, 1, 1)), majority((0, 1, 0)), majority((1, 1, 1))) == (1, 0, 1))

    # lock = 1 forces majority == neutral on 100%
    s1 = _synthetic("lock1")
    check("synthetic lock=1: lock is exactly 1.0", lock_stat(s1) == 1.0)
    eq = [majority(r["codes"]) == r["codes"][N] for r in s1]
    check("synthetic lock=1: majority == neutral on 100% of triples", all(eq))
    o1 = cv_oracle(s1, draws=draws, seed=DEFAULT_SEED, n_splits=20)
    check("synthetic lock=1: in-sample oracle equals the neutral seat (gap 0)", abs(o1["in_sample"]["gap"]) < 1e-12)
    check("synthetic lock=1: CV oracle gap is 0 with CI containing 0",
          abs(o1["gap"]) < 1e-12 and o1["lo"] <= 0 <= o1["hi"])

    # lock = 0 with an accurate advocate makes the advocate the best seat
    s0 = _synthetic("lock0_good_advocate")
    check("synthetic lock=0: lock is exactly 0.0", lock_stat(s0) == 0.0)
    acc = {s: _mean([int(r["codes"][i] == r["gold"]) for r in s0]) for i, s in enumerate(SEATS)}
    check("synthetic lock=0: writer's advocate is the best seat (1.000 vs 0.700 / 0.600)",
          max(SEATS, key=lambda s: acc[s]) == "writer_advocate"
          and abs(acc["counterparty"] - 0.7) < 1e-9 and abs(acc["neutral_adjudicator"] - 0.6) < 1e-9)
    check("synthetic lock=0: majority != neutral on some triples",
          any(majority(r["codes"]) != r["codes"][N] for r in s0))
    o0 = cv_oracle(s0, draws=draws, seed=DEFAULT_SEED, n_splits=20)
    check("synthetic lock=0: CV oracle learns the advocate, gap +0.400 with CI > 0",
          abs(o0["gap"] - 0.4) < 1e-9 and o0["lo"] > 0)

    # guard
    bad = [dict(r, codes=(None, r["codes"][1], r["codes"][2])) if k % 2 else r for k, r in enumerate(s1)]
    check("guard fails on a 50% non-codable panel", not guard(bad)["ok"])
    check("guard fails when a cached r0 call is missing", not guard([dict(s1[0], missing=1)] + s1[1:])["ok"])
    check("guard passes on a clean panel", guard(s1)["ok"])
    # the literal GUARD FAILED must reach stdout through print_report
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report({"panels": {}, "guard_ok": False, "guard_failures": ["synthetic"], "replay_agreement": {}})
    check("print_report prints the literal GUARD FAILED on a failing guard dict",
          GUARD_FAILED in buf.getvalue() and "synthetic" in buf.getvalue())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report({"panels": {}, "guard_ok": True, "guard_failures": [], "replay_agreement": {}})
    check("print_report does NOT print GUARD FAILED when guards pass", GUARD_FAILED not in buf.getvalue())

    # per-statistic seeds are recorded next to every CI, on the stated offsets
    check("SEED_OFFSETS covers the ten per-panel statistics with distinct offsets 1..10",
          sorted(SEED_OFFSETS.values()) == list(range(1, 11)))
    p_syn = _panel_from_recs(s0, draws=draws, seed=DEFAULT_SEED)
    check("every CI in a panel block records the seed it was drawn with (base + offset)",
          p_syn["majority_equals_neutral"]["seed"] == DEFAULT_SEED + SEED_OFFSETS["majority_equals_neutral"]
          and p_syn["role_lock"]["seed"] == DEFAULT_SEED + SEED_OFFSETS["role_lock"]
          and p_syn["seat_acc_on_triples"]["neutral_adjudicator"]["seed"] == DEFAULT_SEED + SEED_OFFSETS["seat_acc_neutral_adjudicator"]
          and p_syn["writer_minus_neutral"]["seed"] == DEFAULT_SEED + SEED_OFFSETS["writer_advocate_minus_neutral"]
          and p_syn["base_seed"] == DEFAULT_SEED)
    p_re = _panel_from_recs(s0, draws=draws, seed=DEFAULT_SEED)
    check("a panel block is bit-identical when re-drawn with the recorded seeds",
          json.dumps(p_re, sort_keys=True, default=str) == json.dumps(p_syn, sort_keys=True, default=str))

    # fold discipline: every copy of an item lands in that item's fold
    fold = split_items([r["item"] for r in s1], 13)
    check("split is a half/half partition of items", sum(fold.values()) == len(fold) // 2)

    # real data: the grok panel must reproduce the fig1_rolelock.json construction
    grok = load_panel(*PANELS["grok"])
    if grok is None:
        check("grok rows file present", False)
    else:
        same = all(code_response(r["raws"][i], "published") == r0_verdict_code(PANELS["grok"][0], r["arm"], r["item"], r["sample_idx"], s)
                   for r in grok for i, s in enumerate(SEATS) if r["raws"][i] is not None)
        check("grok: raw-read collapse equals analyze_rolelock_by_model.r0_verdict_code on every row", same)
        ref = analyse_panel(*PANELS["grok"])
        check("grok: lock equals the fig1 construction exactly (0.971)",
              abs(lock_stat(grok) - ref["role_lock"]) < 1e-12 and round(ref["role_lock"], 3) == 0.971)
        t = triples(grok)
        check("grok: 1,642 triples, no missing r0 calls, guard ok",
              len(t) == 1642 and guard(grok)["ok"])
        check("grok: majority == neutral on 100% of triples",
              all(majority(r["codes"]) == r["codes"][N] for r in t))
        check("grok: in-sample oracle over the three r0 bits IS the neutral seat (gap exactly 0)",
              abs(in_sample_oracle(t, profile_bits)["gap"]) < 1e-12)
    haiku = load_panel(*PANELS["haiku"])
    if haiku is None:
        check("haiku rows file present", False)
    else:
        t = triples(haiku)
        eqn = sum(1 for r in t if majority(r["codes"]) == r["codes"][N])
        check("haiku: replay reproduces 180/187 majority == neutral (0.963)",
              (eqn, len(t)) == (180, 187))
        wa = _mean([int(r["codes"][W] == r["gold"]) for r in t])
        check("haiku: writer's advocate 0.818 on the 187 triples", round(wa, 3) == 0.818)
    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--json", type=Path, default=ANALYSIS_PATH)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    res = run(draws=a.draws, seed=a.seed)
    print_report(res)
    a.json.write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {a.json}")
    return 0 if res["guard_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
