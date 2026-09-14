"""
scripts/analyze_aggregation_rules.py -- the aggregation theory the paper leans
on (Nitzan-Paroush / Shapley-Grofman weighted majority, Condorcet with
correlated votes, Golub-Jackson's dictator condition) tested on our OWN seat
verdicts. Zero spend: every number is replayed from the cached r0 opening
statements (`cgd_<model>_narrative_cot_t2560_<arm>_<item>_<idx>_r0_<seat>.json`),
the `<tag>_rows.csv` files the deliberation runs already wrote, and the
`aggregation_passthrough.json` artefact already on disk. No model or API call
is made and no registration document is edited.

WHY
---
related_work.tex says "the optimal rule over voters of unequal competence is
a log-odds weighted majority (Shapley-Grofman 1984, Nitzan-Paroush 1982).
With our measured seat accuracies that rule is a dictatorship of the
competent seat, which is what we observe." That sentence was written from
the seat accuracies, not computed. This module computes it, with the weights
fitted OUT OF SAMPLE (two-fold CV over items, both directions) so the rule is
a genuine decision procedure and not an in-sample fit, and compares it with
every other rule the record discusses on the same debates.

WHAT IS COMPUTED, per model with a cached three-seat community (grok, sonnet,
nano, haiku, the PANELS of analyze_rolelock_by_model) and per cell of the
16.10 roles x edges 2x2 (topology CSVs, seats resolved by
analyze_topology_2x2.CELL_SPECS)
--------------------------------------------------------------------------
Population: the TRIPLE population, debates on which all three r0 verdicts
are codable under the registered five-token collapse
`code_response(extract_verdict(r0), "published")`, exactly as
effective_voter.json used. The moderator's synthesis verdict and the
integrated verdict (S2, the `verdict` column) are read from the rows CSV and
scored on the triples where they are codable (n stated); paired differences
against S2 use the intersection.

Decision rules over the three opening verdicts (writer's advocate,
counterparty, neutral adjudicator, in that seat order; identical cells use
reader alpha / beta / gamma):
  * each single seat;
  * simple majority of the three at-fault bits;
  * the neutral seat alone, the dictator (seat index 2; for identical cells
    this is reader gamma and is reported only for symmetry);
  * the Shapley-Grofman / Nitzan-Paroush log-odds weighted majority, weights
    w_i = log(p_i / (1 - p_i)) with p_i the seat's Laplace-smoothed accuracy
    (k_i + 1) / (n + 2) on the OTHER fold; verdict 1 when
    sum_i w_i (2 x_i - 1) > 0, 0 when < 0, the simple majority on an exact
    tie; a seat with p_i < 0.5 therefore votes inverted, as the theorem
    requires; the in-sample rule (weights from all triples) is reported as
    the headroom upper bound;
  * the moderator's synthesis verdict and the integrated verdict S2;
  * the CV oracle over the eight r0 profiles, reused verbatim from
    analyze_effective_voter.cv_oracle (same seed, same fold split).

Fold split: analyze_effective_voter.split_items with the community's base
seed, fixed per item inside the bootstrap; a bootstrap multiset keeps every
copy of an item in that item's fold, and in each resample the weights are
re-fitted on the resampled training fold and scored on the resampled test
fold, exactly as the oracle is.

Intervals: item-clustered percentile bootstrap, 4,000 draws. All rules of a
community are resampled in ONE stream (seed base + 1) so every paired
difference is computed on the same resamples; the CV oracle keeps its own
stream (seed base, as effective_voter.json). Base seeds: panel i of PANELS
uses 13 + 100 i (grok 13, sonnet 113, nano 213, haiku 313, as
effective_voter.json), topology cell j uses 13 + 100 (4 + j). The seed used
is recorded next to every interval.

THEORY TESTS
------------
(T1) Shapley-Grofman dictatorship. The log-odds rule is a dictatorship of
     seat i exactly when w_i > sum_{j != i} |w_j|. Prediction: where lock is
     near 1 and the neutral seat is the best seat (grok, nano) the weighted
     rule EQUALS the neutral dictator (paired difference 0, agreement 1.0);
     where an advocate is the best seat (haiku) it BEATS the dictator.
     Readout: per model, the fitted weights per fold, which seat (if any)
     dominates, the share of debates on which the weighted rule returns the
     neutral verdict, and the paired difference weighted minus dictator with
     its CI. Also weighted minus best seat, weighted minus majority, and
     weighted minus CV oracle (the oracle is the best fixed function of the
     profile and bounds every deterministic rule from above in sample).
(T2) Condorcet with correlated votes, the Ladha (1992) framework. From
     aggregation_passthrough.json, the same-model x3 and cross-vendor x3
     majority gains and mean pairwise error correlations. For three voters
     of common competence p and mean pairwise correlation rho of their
     correctness indicators, the majority's accuracy is confined by the
     eight-profile distribution's moment constraints to
        [3 p2 - 2 min(p2, 1 - 3p + 3 p2),  3 p2 - 2 max(0, 2 p2 - p)],
     p2 = p^2 + rho p (1 - p), so the gain over an individual is at most
     p (1 - p) (1 - rho) whenever 2 p2 >= p (this is the max-min-over-the-
     profile-simplex construction; the selftest checks the closed form
     against a linear programme over the eight profiles). Independence
     (Condorcet) would give 3 p^2 - 2 p^3 - p. Readout: per panel the
     observed gain, the correlation ceiling, the share of the ceiling
     realised, the independence prediction, whether the observed gain sits
     under the ceiling, and whether it sits under the independence value.
     HONESTY NOTE: Ladha 1992 states its sufficient conditions as an upper
     bound on the average pairwise correlation; the primary text could not
     be fetched in this session (JSTOR refused the fetch, Springer requires
     a cookie), so no formula is attributed to Ladha here. The ceiling
     above is derived and verified in this module. The same quantities are
     computed on our own communities (mean seat accuracy, the three pairwise
     error correlations, the observed majority gain over the mean seat and
     over the best seat); the common-competence ceiling applies to the
     identical cells, where the seats are exchangeable, and is reported for
     the embodied cells only with the warning that the advocates are
     anti-correlated by construction (lock), so the common-p band is not
     the right object there.
     FOUND, NOT PREDICTED: under lock the advocates' log-odds weights are
     gold base-rate logits (writer always 0, counterparty always 1), so on
     an item resample that tilts the gold balance the two advocates can
     jointly outweigh the neutral seat, the rule stops being a dictatorship
     and degenerates to a near-constant verdict (accuracy falls by 0.17 to
     0.25 on grok). This happens on 2 to 7 percent of draws depending on
     the fold split, which is why the CV rule's lower bound differs between
     the 'grok' stream (base 13) and the 'embodied/on' stream (base 413) on
     the SAME 1,642 debates. The share of such draws is written next to
     every T1 readout (`dominance_broken_share_of_draws`) and the in-sample
     rule (weights from all triples, no CV) is reported alongside. It is a
     property of the theorem, not of the data: Nitzan-Paroush optimality
     assumes conditional independence, and two locked seats are perfectly
     anti-correlated (error correlation -0.94 on grok).
(T3) The moderator against the best rule over seats. S2 minus the CV
     weighted majority, S2 minus the best seat, S2 minus the majority, S2
     minus the neutral dictator, S2 minus the CV oracle, and the synthesis
     verdict minus the weighted majority, each a paired item-clustered CI
     on the intersection population. Reading rule: the moderator is a
     better aggregator than the optimal weighted majority when the S2 minus
     weighted interval lies above 0, worse when it lies below 0, and
     indistinguishable otherwise.
(16.10) The four topology cells: majority vs best seat vs S2 (the record's
     table row), plus the weighted rule and the oracle; for the identical
     cells the share of debates on which the weighted rule returns the
     simple majority (exchangeable seats should give near-equal weights).

GUARD
-----
Prints the literal string GUARD FAILED if, for any community, a cached r0
call is missing from disk, the non-codable share of triples exceeds 0.10, or
fewer than 20 triples survive. A failed community is still written with
`guard_ok` false so the failure is auditable. Sonnet (31 debates, 16 items)
passes the guard and is flagged uninformative, as the record does.

Usage
-----
  python -m scripts.analyze_aggregation_rules --selftest
  python -m scripts.analyze_aggregation_rules            # writes aggregation_rules.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_effective_voter import (
    cv_oracle, fit_table, load_panel, profile_bits, split_items,
)
from scripts.analyze_rolelock_by_model import PANELS, R0_CAP, SEATS
from scripts.analyze_topology_2x2 import CELL_SPECS, MODEL as TOPOLOGY_MODEL
from scripts.run_phase1_quartet import OUT_DIR
from scripts.verdict_format import extract_verdict

ANALYSIS_PATH = OUT_DIR / "aggregation_rules.json"
PASSTHROUGH_PATH = OUT_DIR / "aggregation_passthrough.json"
DEFAULT_DRAWS = 4000
DEFAULT_SEED = 13
PANEL_SEED_STRIDE = 100
MAX_NONCODABLE_TRIPLE = 0.10
MIN_TRIPLES = 20
UNINFORMATIVE_N = 50
DICTATOR = 2               # seat index of the neutral adjudicator (reader gamma on identical cells)
GUARD_FAILED = "GUARD FAILED"
RULES = ("seat0", "seat1", "seat2", "majority", "dictator", "weighted_cv",
         "weighted_insample", "synthesis", "s2")


# ---------------------------------------------------------------------------
# Loading: one record per debate; three r0 bits, synthesis bit, S2 bit
# ---------------------------------------------------------------------------

def r0_raw(model: str, arm: str, item: str, idx: int, role_id: str) -> Optional[str]:
    p = rcd.call_cache_path(model, "narrative_cot", arm, item, idx, "r0", role_id, R0_CAP)
    if not p.exists():
        return None
    txt = json.loads(p.read_text()).get("output") or ""
    return extract_verdict(txt, rcd.INSTRUMENT)


def load_community(model: str, tag: str, seats: Sequence[str]) -> Optional[list[dict]]:
    """Mirrors analyze_effective_voter.load_panel (asserted equal on the r0
    bits by --selftest) but takes the seat ids as a parameter, so the
    identical-reader cells load through the same code, and adds the
    moderator's synthesis verdict and the integrated verdict S2 as at-fault
    bits under the registered collapse."""
    rows_path = OUT_DIR / f"{tag}_rows.csv"
    if not rows_path.exists():
        return None
    recs = []
    mods: set = set()
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            if r.get("model") != model:
                continue
            arm, item, idx = r["arm"], r["item_id"], int(r["sample_idx"])
            mods.add(r.get("moderator_model", ""))
            codes, missing = [], 0
            for s in seats:
                raw = r0_raw(model, arm, item, idx, s)
                if raw is None:
                    missing += 1
                codes.append(None if raw is None else code_response(raw, "published"))
            recs.append({"item": item, "arm": arm, "sample_idx": idx,
                         "gold": 1 if r["gold_verdict"] == "YTA" else 0,
                         "codes": tuple(codes), "missing": missing,
                         "synthesis": code_response(r.get("synthesis_verdict", ""), "published"),
                         "s2": code_response(r.get("verdict", ""), "published")})
    for r in recs:
        r["moderator_model"] = "|".join(sorted(mods))
    return recs


def triples(recs: Sequence[dict]) -> list[dict]:
    return [r for r in recs if None not in r["codes"]]


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


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def majority(codes: Sequence[int]) -> int:
    return 1 if sum(codes) >= 2 else 0


def logodds_weights(train: Sequence[dict]) -> list[float]:
    """w_i = log(p_i / (1 - p_i)), p_i the seat's Laplace-smoothed accuracy on
    the training records. Smoothing keeps a perfect seat (sonnet's neutral,
    31 of 31) finite; it moves a weight by O(1/n)."""
    n = len(train)
    out = []
    for i in range(3):
        k = sum(int(r["codes"][i] == r["gold"]) for r in train)
        p = (k + 1) / (n + 2)
        out.append(math.log(p / (1 - p)))
    return out


def weighted_vote(codes: Sequence[int], w: Sequence[float]) -> int:
    s = sum(wi * (2 * c - 1) for wi, c in zip(w, codes))
    if s > 0:
        return 1
    if s < 0:
        return 0
    return majority(codes)


def dominant_seat(w: Sequence[float]) -> Optional[int]:
    """The seat whose weight exceeds the sum of the others' absolute weights;
    the weighted rule is then that seat's verdict on every profile (a
    negative dominant weight would be that seat inverted; not observed)."""
    for i, wi in enumerate(w):
        if abs(wi) > sum(abs(wj) for j, wj in enumerate(w) if j != i):
            return i
    return None


def map_rule_independent(p: Sequence[float]) -> dict:
    """The Bayes-optimal (MAP, equal priors) verdict for every profile of three
    CONDITIONALLY INDEPENDENT voters with competences p; used by the selftest
    to check that the log-odds rule reproduces it on every profile."""
    out = {}
    for x in itertools.product((0, 1), repeat=3):
        lik = {}
        for g in (0, 1):
            v = 1.0
            for pi, xi in zip(p, x):
                v *= pi if xi == g else 1 - pi
            lik[g] = v
        out[x] = 1 if lik[1] > lik[0] else (0 if lik[0] > lik[1] else None)
    return out


# ---------------------------------------------------------------------------
# Correctness vectors, in-sample and cross-validated, aligned with the records
# ---------------------------------------------------------------------------

def cv_weights_by_fold(rs: Sequence[dict], fold: dict) -> Optional[dict]:
    a = [r for r in rs if fold[r["item"]] == 0]
    b = [r for r in rs if fold[r["item"]] == 1]
    if not a or not b:
        return None
    return {0: logodds_weights(b), 1: logodds_weights(a)}   # fold f is scored with weights fitted on the other fold


def cv_oracle_tables(rs: Sequence[dict], fold: dict) -> Optional[dict]:
    a = [r for r in rs if fold[r["item"]] == 0]
    b = [r for r in rs if fold[r["item"]] == 1]
    if not a or not b:
        return None
    return {0: fit_table(b, profile_bits), 1: fit_table(a, profile_bits)}


def predictions(rs: Sequence[dict], fold: dict) -> Optional[dict]:
    """rule -> list of predicted at-fault bits aligned with rs (None where the
    rule is undefined on that debate: synthesis / S2 not codable)."""
    wf = cv_weights_by_fold(rs, fold)
    ot = cv_oracle_tables(rs, fold)
    if wf is None or ot is None:
        return None
    w_all = logodds_weights(rs)
    pred = {k: [] for k in RULES + ("oracle_cv",)}
    for r in rs:
        c = r["codes"]
        f = fold[r["item"]]
        pred["seat0"].append(c[0]); pred["seat1"].append(c[1]); pred["seat2"].append(c[2])
        pred["majority"].append(majority(c))
        pred["dictator"].append(c[DICTATOR])
        pred["weighted_cv"].append(weighted_vote(c, wf[f]))
        pred["weighted_insample"].append(weighted_vote(c, w_all))
        table, default = ot[f]
        pred["oracle_cv"].append(table.get(c, default))
        pred["synthesis"].append(r["synthesis"])
        pred["s2"].append(r["s2"])
    return pred


def correctness(rs: Sequence[dict], pred: dict) -> dict:
    return {k: [None if p is None else int(p == r["gold"]) for p, r in zip(v, rs)]
            for k, v in pred.items()}


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _paired(a: Sequence[Optional[int]], b: Sequence[Optional[int]]) -> Optional[float]:
    d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    return sum(d) / len(d) if d else None


PAIRS = (("weighted_cv", "dictator"), ("weighted_cv", "best_seat"), ("weighted_cv", "majority"),
         ("weighted_cv", "oracle_cv"), ("weighted_insample", "dictator"), ("weighted_insample", "best_seat"),
         ("majority", "best_seat"), ("majority", "dictator"),
         ("s2", "weighted_cv"), ("s2", "best_seat"), ("s2", "majority"), ("s2", "dictator"),
         ("s2", "oracle_cv"), ("synthesis", "weighted_cv"), ("s2", "synthesis"))


def community_stats(rs: Sequence[dict], fold: dict, best_seat: int,
                    dominant_point: Optional[int] = None) -> Optional[dict]:
    """Every statistic of one community on one (re)sample, as a flat dict.
    `best_seat` is fixed from the point estimate so that 'best seat' means the
    same seat in every resample (the paired 'weighted minus best seat' is then
    a difference against a named seat, not against a moving target).
    `dominant_point` is the in-sample dominant seat of the point estimate;
    `dominance_broken` is 1 on a resample whose CV weights on either fold no
    longer make that seat dominant (under lock the advocates' weights are
    base-rate logits, and on an item resample that tilts the gold balance
    they can jointly outweigh the neutral seat and turn the rule into a
    near-constant verdict; the bootstrap mean of this flag is the share of
    such draws)."""
    pred = predictions(rs, fold)
    if pred is None:
        return None
    ok = correctness(rs, pred)
    wf = cv_weights_by_fold(rs, fold)
    ok["best_seat"] = ok[f"seat{best_seat}"]
    out = {}
    for k, v in ok.items():
        out[f"acc_{k}"] = _mean(v)
    for a, b in PAIRS:
        out[f"diff_{a}_minus_{b}"] = _paired(ok[a], ok[b])
    out["share_weighted_equals_dictator"] = _mean([int(x == y) for x, y in zip(pred["weighted_cv"], pred["dictator"])])
    out["share_weighted_equals_majority"] = _mean([int(x == y) for x, y in zip(pred["weighted_cv"], pred["majority"])])
    out["share_majority_equals_dictator"] = _mean([int(x == y) for x, y in zip(pred["majority"], pred["dictator"])])
    out["share_s2_equals_weighted"] = _mean([None if x is None else int(x == y) for x, y in zip(pred["s2"], pred["weighted_cv"])])
    out["dominance_broken"] = (None if dominant_point is None else
                               int(dominant_seat(wf[0]) != dominant_point or dominant_seat(wf[1]) != dominant_point))
    return out


def boot_multi(by_item: dict, stat: Callable[[list], Optional[dict]], *,
               draws: int, seed: int, alpha: float = 0.05) -> tuple[dict, dict]:
    """One item-clustered resampling stream for a dict-valued statistic; the
    percentile convention is analyze_unembodied_ablation._boot's exactly
    (sorted values, index int(alpha/2 n) and min(int((1-alpha/2) n), n-1))."""
    ids = sorted(by_item)
    rng = random.Random(seed)
    point = stat([x for i in ids for x in by_item[i]])
    samples: dict = defaultdict(list)
    for _ in range(draws):
        s = stat([x for i in (ids[rng.randrange(len(ids))] for _ in ids) for x in by_item[i]])
        if s is None:
            continue
        for k, v in s.items():
            if v is not None:
                samples[k].append(v)
    ci = {}
    for k in point:
        vals = sorted(samples.get(k, []))
        if not vals:
            ci[k] = {"point": point[k], "lo": None, "hi": None, "boot_mean": None, "n_draws_defined": 0, "seed": seed}
            continue
        ci[k] = {"point": point[k], "lo": vals[int(alpha / 2 * len(vals))],
                 "hi": vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)],
                 "boot_mean": sum(vals) / len(vals), "n_draws_defined": len(vals), "seed": seed}
    return point, ci


# ---------------------------------------------------------------------------
# T2: Condorcet with correlated votes
# ---------------------------------------------------------------------------

def majority_band(p: float, rho: float) -> dict:
    """Exact range of the three-voter majority accuracy for common competence p
    and common pairwise correlation rho of the correctness indicators, from
    the moment constraints on the eight-profile distribution. The ceiling on
    the gain equals p(1-p)(1-rho) whenever 2 p2 >= p."""
    p2 = p * p + rho * p * (1 - p)
    p3_min = max(0.0, 2 * p2 - p)
    p3_max = min(p2, 1 - 3 * p + 3 * p2)
    hi = 3 * p2 - 2 * p3_min
    lo = 3 * p2 - 2 * p3_max
    return {"p": p, "rho": rho, "p2": p2, "majority_min": lo, "majority_max": hi,
            "gain_ceiling": hi - p, "gain_floor": lo - p,
            "gain_ceiling_closed_form_p1mp1mrho": p * (1 - p) * (1 - rho),
            "closed_form_applies": 2 * p2 >= p,
            "independence_majority": 3 * p * p - 2 * p ** 3,
            "independence_gain": 3 * p * p - 2 * p ** 3 - p}


def majority_band_lp(p: float, rho: float) -> Optional[tuple[float, float]]:
    """Linear programme over the eight profiles with the three marginals and
    three pairwise moments as constraints (selftest cross-check of the
    closed form; needs scipy)."""
    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError:
        return None
    profiles = list(itertools.product((0, 1), repeat=3))
    p2 = p * p + rho * p * (1 - p)
    A, b = [], []
    for i in range(3):
        A.append([pr[i] for pr in profiles]); b.append(p)
    for i, j in itertools.combinations(range(3), 2):
        A.append([pr[i] * pr[j] for pr in profiles]); b.append(p2)
    A.append([1] * 8); b.append(1)
    c = np.array([1.0 if sum(pr) >= 2 else 0.0 for pr in profiles])
    hi = linprog(-c, A_eq=A, b_eq=b, bounds=[(0, 1)] * 8)
    lo = linprog(c, A_eq=A, b_eq=b, bounds=[(0, 1)] * 8)
    if not (hi.success and lo.success):
        return None
    return lo.fun, -hi.fun


def pearson(x: Sequence[int], y: Sequence[int]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if not vx or not vy:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (vx * vy) ** 0.5


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = k + 1.0
        return r
    return pearson(ranks(x), ranks(y)) if len(x) >= 3 else None


def t2_from_passthrough(path: Path = PASSTHROUGH_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    d = json.loads(path.read_text()).get("diversity", {})
    panels = {}
    for k, v in d.items():
        band = majority_band(v["individual_acc"], v["error_rho"])
        gain = v["gain"]
        panels[k] = {
            "kind": "same_model" if k.startswith("same-model") else "cross_vendor",
            "n_items": v["n_items"], "individual_acc": v["individual_acc"], "majority_acc": v["majority_acc"],
            "gain": gain, "error_rho": v["error_rho"],
            "gain_ceiling": band["gain_ceiling"], "gain_floor": band["gain_floor"],
            "closed_form_applies": band["closed_form_applies"],
            "share_of_ceiling_realised": gain / band["gain_ceiling"] if band["gain_ceiling"] > 0 else None,
            "independence_gain": band["independence_gain"],
            "under_ceiling": gain <= band["gain_ceiling"] + 1e-12,
            "under_independence": gain <= band["independence_gain"] + 1e-12,
        }
    same = [v for v in panels.values() if v["kind"] == "same_model"]
    cross = [v for v in panels.values() if v["kind"] == "cross_vendor"]
    rhos = [v["error_rho"] for v in panels.values()]
    gains = [v["gain"] for v in panels.values()]
    return {
        "source": str(path.name),
        "panels": panels,
        "same_model": {"n": len(same), "mean_gain": _mean([v["gain"] for v in same]),
                       "mean_rho": _mean([v["error_rho"] for v in same]),
                       "mean_ceiling": _mean([v["gain_ceiling"] for v in same]),
                       "mean_independence_gain": _mean([v["independence_gain"] for v in same]),
                       "all_under_ceiling": all(v["under_ceiling"] for v in same),
                       "all_under_independence": all(v["under_independence"] for v in same)},
        "cross_vendor": {"n": len(cross), "mean_gain": _mean([v["gain"] for v in cross]),
                         "mean_rho": _mean([v["error_rho"] for v in cross]),
                         "mean_ceiling": _mean([v["gain_ceiling"] for v in cross]),
                         "mean_independence_gain": _mean([v["independence_gain"] for v in cross]),
                         "all_under_ceiling": all(v["under_ceiling"] for v in cross),
                         "all_under_independence": all(v["under_independence"] for v in cross)},
        "spearman_rho_vs_gain_across_panels": spearman(rhos, gains),
        "reading": ("majority gain decreases with mean pairwise error correlation across the six "
                    "panels; every observed gain sits under the correlation ceiling p(1-p)(1-rho); "
                    "the independence (Condorcet) prediction over-states every same-model gain"),
    }


def own_seats_t2(trip: Sequence[dict]) -> dict:
    """The T2 quantities on one of our communities: seat accuracies, the
    three pairwise error correlations, the majority's gain over the mean seat
    and over the best seat, and the common-competence ceiling at the mean
    seat accuracy and mean pairwise correlation."""
    ok = [[int(r["codes"][i] == r["gold"]) for r in trip] for i in range(3)]
    acc = [_mean(v) for v in ok]
    cors = {f"{i}-{j}": pearson(ok[i], ok[j]) for i, j in itertools.combinations(range(3), 2)}
    rho_vals = [v for v in cors.values() if v is not None]
    rho = _mean(rho_vals)
    maj = _mean([int(majority(r["codes"]) == r["gold"]) for r in trip])
    pbar = _mean(acc)
    band = majority_band(pbar, rho) if rho is not None else None
    return {"seat_acc": acc, "mean_seat_acc": pbar, "best_seat_acc": max(acc),
            "pairwise_error_correlation": cors, "mean_pairwise_error_correlation": rho,
            "majority_acc": maj, "gain_over_mean_seat": maj - pbar, "gain_over_best_seat": maj - max(acc),
            "ceiling_at_mean_p_and_mean_rho": band["gain_ceiling"] if band else None,
            "independence_gain_at_mean_p": band["independence_gain"] if band else None,
            "under_ceiling": (maj - pbar) <= band["gain_ceiling"] + 1e-12 if band else None,
            "seats_exchangeable": max(acc) - min(acc) < 0.05,
            "note": ("common-competence band applies (seats within 0.05 of each other)" if max(acc) - min(acc) < 0.05
                     else "seats are heterogeneous (lock): the common-p band is not the right object; the "
                          "Shapley-Grofman rule is")}


# ---------------------------------------------------------------------------
# One community
# ---------------------------------------------------------------------------

def analyse_community(recs: Sequence[dict], *, seats: Sequence[str], draws: int, seed: int,
                      label: str, model: Optional[str] = None, tag: Optional[str] = None) -> dict:
    g = guard(recs)
    trip = triples(recs)
    out: dict = {"label": label, "model": model, "tag": tag, "seats": list(seats), "guard": g,
                 "moderator_model": recs[0].get("moderator_model") if recs else None,
                 "base_seed": seed, "n_debates": len(recs),
                 "n_items": len({r["item"] for r in recs}), "n_triples": len(trip),
                 "n_items_in_triples": len({r["item"] for r in trip}),
                 "n_triples_synthesis_codable": sum(1 for r in trip if r["synthesis"] is not None),
                 "n_triples_s2_codable": sum(1 for r in trip if r["s2"] is not None),
                 "uninformative": len(trip) < UNINFORMATIVE_N}
    if not trip:
        return out
    items = [r["item"] for r in trip]
    fold = split_items(items, seed)
    seat_acc = [_mean([int(r["codes"][i] == r["gold"]) for r in trip]) for i in range(3)]
    best = max(range(3), key=lambda i: seat_acc[i])
    out["best_seat"] = {"index": best, "id": seats[best], "acc": seat_acc[best]}
    out["fold"] = {"split_seed": seed, "sizes": [sum(1 for i in set(items) if fold[i] == 0),
                                                  sum(1 for i in set(items) if fold[i] == 1)]}
    by_item: dict = defaultdict(list)
    for r in trip:
        by_item[r["item"]].append(r)
    dom_point = dominant_seat(logodds_weights(trip))
    point, ci = boot_multi(by_item, lambda rs: community_stats(rs, fold, best, dom_point), draws=draws, seed=seed + 1)
    out["rules"] = {k: ci[f"acc_{k}"] for k in RULES + ("oracle_cv", "best_seat")}
    out["dominance_broken_share_of_draws"] = (None if dom_point is None else ci["dominance_broken"]["boot_mean"])
    out["paired"] = {f"{a}_minus_{b}": ci[f"diff_{a}_minus_{b}"] for a, b in PAIRS}
    out["shares"] = {k: ci[k] for k in ("share_weighted_equals_dictator", "share_weighted_equals_majority",
                                        "share_majority_equals_dictator", "share_s2_equals_weighted")}
    # the fitted weights, point estimate (fold f scored with weights from the other fold)
    wf = cv_weights_by_fold(trip, fold)
    w_all = logodds_weights(trip)
    out["weights"] = {
        "seat_order": list(seats),
        "applied_to_fold_0_fitted_on_fold_1": wf[0], "applied_to_fold_1_fitted_on_fold_0": wf[1],
        "in_sample": w_all,
        "dominant_seat_fold_0": dominant_seat(wf[0]), "dominant_seat_fold_1": dominant_seat(wf[1]),
        "dominant_seat_in_sample": dominant_seat(w_all),
        "smoothing": "Laplace (k+1)/(n+2) on the training fold",
        "tie_rule": "exact zero score falls back to the simple majority"}
    # the registered oracle, reused verbatim (its own stream, seed = base)
    out["oracle_cv_effective_voter"] = cv_oracle(trip, draws=draws, seed=seed, profile=profile_bits)
    out["t2_own_seats"] = own_seats_t2(trip)
    # T1 verdict
    wd = out["paired"]["weighted_cv_minus_dictator"]
    dom = out["weights"]["dominant_seat_in_sample"]
    if out["shares"]["share_weighted_equals_dictator"]["point"] == 1.0 and wd["point"] == 0.0:
        t1 = "weighted rule EQUALS the neutral dictator on every debate"
    elif wd["lo"] is not None and wd["lo"] > 0:
        t1 = "weighted rule BEATS the neutral dictator (CI above 0)"
    elif wd["hi"] is not None and wd["hi"] < 0:
        t1 = "weighted rule is WORSE than the neutral dictator (CI below 0)"
    else:
        t1 = "weighted rule and neutral dictator indistinguishable (CI includes 0)"
    out["t1"] = {"verdict": t1, "dominant_seat_in_sample": None if dom is None else seats[dom],
                 "weighted_minus_dictator": wd,
                 "weighted_insample_minus_dictator": out["paired"]["weighted_insample_minus_dictator"],
                 "dominance_broken_share_of_draws": out["dominance_broken_share_of_draws"],
                 "note": ("the CV rule's lower tail comes from resamples in which the dominant seat's weight no "
                          "longer exceeds the others' (share above); under lock the advocates' weights are gold "
                          "base-rate logits, so an item resample that tilts the gold balance can make the two "
                          "advocates jointly outweigh the neutral seat and the rule degenerates to a near-constant "
                          "verdict; Shapley-Grofman optimality assumes conditional independence, which lock violates")}
    # T3 verdict
    sw = out["paired"]["s2_minus_weighted_cv"]
    if sw["lo"] is not None and sw["lo"] > 0:
        t3 = "moderator (S2) is a BETTER aggregator than the CV weighted majority (CI above 0)"
    elif sw["hi"] is not None and sw["hi"] < 0:
        t3 = "moderator (S2) is a WORSE aggregator than the CV weighted majority (CI below 0)"
    else:
        t3 = "moderator (S2) and the CV weighted majority indistinguishable (CI includes 0)"
    sd = out["paired"]["s2_minus_dictator"]
    if sd["lo"] is not None and sd["lo"] > 0:
        t3d = "S2 BETTER than the neutral dictator (CI above 0)"
    elif sd["hi"] is not None and sd["hi"] < 0:
        t3d = "S2 WORSE than the neutral dictator (CI below 0)"
    else:
        t3d = "S2 and the neutral dictator indistinguishable (CI includes 0)"
    out["t3"] = {"verdict": t3, "s2_minus_weighted_cv": sw, "verdict_vs_dictator": t3d, "s2_minus_dictator": sd}
    out["profile_counts"] = {"|".join(map(str, p)): {"n": n_, "gold_at_fault": sum(r["gold"] for r in trip if r["codes"] == p)}
                             for p, n_ in sorted(((p, sum(1 for r in trip if r["codes"] == p))
                                                  for p in set(r["codes"] for r in trip)))}
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def _ci(d: Optional[dict], fmt=".3f") -> str:
    if d is None:
        return "n/a"
    return f"{_f(d['point'], fmt)} [{_f(d['lo'], fmt)}, {_f(d['hi'], fmt)}]"


def print_report(res: dict) -> None:
    print("\n" + "=" * 110)
    print("AGGREGATION RULES ON OUR OWN SEAT VERDICTS (zero spend, replayed from cached r0 calls and rows CSVs)")
    print("=" * 110)
    comms = {**res.get("panels", {}), **res.get("topology_cells", {})}
    print(f"\nRule accuracies on the triple population, item-clustered 95% CIs, {res['estimator']['draws']} draws")
    hdr = f"  {'community':16s} {'n':>5s} {'items':>5s}  {'seat0':>20s}  {'seat1':>20s}  {'seat2 (dictator)':>20s}  {'majority':>20s}  {'weighted CV':>20s}  {'oracle CV':>20s}  {'synthesis':>20s}  {'S2':>20s}"
    print(hdr)
    for k, p in comms.items():
        if p is None:
            print(f"  {k:16s}  (rows file absent)")
            continue
        if "rules" not in p:
            print(f"  {k:16s} {p['n_debates']:5d}  (no triples)")
            continue
        r = p["rules"]
        print(f"  {k:16s} {p['n_triples']:5d} {p['n_items_in_triples']:5d}  {_ci(r['seat0']):>20s}  {_ci(r['seat1']):>20s}  "
              f"{_ci(r['seat2']):>20s}  {_ci(r['majority']):>20s}  {_ci(r['weighted_cv']):>20s}  "
              f"{_ci(r['oracle_cv']):>20s}  {_ci(r['synthesis']):>20s}  {_ci(r['s2']):>20s}"
              + ("   [uninformative n]" if p.get("uninformative") else ""))
    print("\n(T1) Shapley-Grofman: weights (seat0, seat1, seat2), dominant seat, and weighted minus dictator")
    for k, p in comms.items():
        if not p or "rules" not in p:
            continue
        w = p["weights"]
        dom = w["dominant_seat_in_sample"]
        print(f"  {k:16s} best seat {p['best_seat']['id']:28s} w(fold1->0) {['%+.2f' % x for x in w['applied_to_fold_0_fitted_on_fold_1']]} "
              f"w(fold0->1) {['%+.2f' % x for x in w['applied_to_fold_1_fitted_on_fold_0']]}  dominant {'none' if dom is None else p['seats'][dom]}")
        print(f"  {'':16s} weighted==dictator {_ci(p['shares']['share_weighted_equals_dictator'])}   "
              f"weighted-dictator {_ci(p['paired']['weighted_cv_minus_dictator'], '+.3f')}   "
              f"weighted-best {_ci(p['paired']['weighted_cv_minus_best_seat'], '+.3f')}   "
              f"weighted-majority {_ci(p['paired']['weighted_cv_minus_majority'], '+.3f')}   "
              f"weighted-oracle {_ci(p['paired']['weighted_cv_minus_oracle_cv'], '+.3f')}")
        print(f"  {'':16s} in-sample weighted-dictator {_ci(p['paired']['weighted_insample_minus_dictator'], '+.3f')}   "
              f"draws with broken dominance {_f(p['dominance_broken_share_of_draws'], '.3f')}")
        print(f"  {'':16s} -> {p['t1']['verdict']}")
    t2 = res.get("t2_passthrough")
    if t2:
        print("\n(T2) Condorcet with correlated votes: aggregation_passthrough.json panels")
        print(f"  {'panel':40s} {'n':>4s} {'p':>6s} {'rho':>6s} {'gain':>7s} {'ceiling':>8s} {'share':>6s} {'indep':>7s}  under ceiling / under indep")
        for k, v in t2["panels"].items():
            print(f"  {k:40s} {v['n_items']:4d} {v['individual_acc']:6.3f} {v['error_rho']:6.3f} {v['gain']:+7.3f} "
                  f"{v['gain_ceiling']:+8.3f} {_f(v['share_of_ceiling_realised'], '5.2f'):>6s} {v['independence_gain']:+7.3f}  "
                  f"{'yes' if v['under_ceiling'] else 'NO'} / {'yes' if v['under_independence'] else 'NO'}")
        for kind in ("same_model", "cross_vendor"):
            s = t2[kind]
            print(f"  {kind:14s} mean gain {s['mean_gain']:+.3f}  mean rho {s['mean_rho']:.3f}  mean ceiling {s['mean_ceiling']:+.3f}  "
                  f"mean independence gain {s['mean_independence_gain']:+.3f}  all under ceiling {s['all_under_ceiling']}  "
                  f"all under independence {s['all_under_independence']}")
        print(f"  Spearman(rho, gain) across the six panels {_f(t2['spearman_rho_vs_gain_across_panels'])}")
    print("\n(T2, own seats) mean seat acc, pairwise error correlations, majority gain, ceiling at (mean p, mean rho)")
    for k, p in comms.items():
        if not p or "rules" not in p:
            continue
        t = p["t2_own_seats"]
        c = t["pairwise_error_correlation"]
        print(f"  {k:16s} seats {['%.3f' % a for a in t['seat_acc']]} mean {t['mean_seat_acc']:.3f}  rho "
              f"{{{', '.join(f'{kk}: {_f(v)}' for kk, v in c.items())}}}  majority {t['majority_acc']:.3f}  "
              f"gain/mean {t['gain_over_mean_seat']:+.3f}  gain/best {t['gain_over_best_seat']:+.3f}  "
              f"ceiling {_f(t['ceiling_at_mean_p_and_mean_rho'])}  indep {_f(t['independence_gain_at_mean_p'])}  "
              f"{'exchangeable' if t['seats_exchangeable'] else 'heterogeneous'}")
    print("\n(T3) the moderator against the best rule over seats (paired, intersection population)")
    for k, p in comms.items():
        if not p or "rules" not in p:
            continue
        q = p["paired"]
        print(f"  {k:16s} S2-weighted {_ci(q['s2_minus_weighted_cv'], '+.3f')}  S2-best {_ci(q['s2_minus_best_seat'], '+.3f')}  "
              f"S2-majority {_ci(q['s2_minus_majority'], '+.3f')}  S2-dictator {_ci(q['s2_minus_dictator'], '+.3f')}  "
              f"S2-oracle {_ci(q['s2_minus_oracle_cv'], '+.3f')}  synthesis-weighted {_ci(q['synthesis_minus_weighted_cv'], '+.3f')}  "
              f"S2-synthesis {_ci(q['s2_minus_synthesis'], '+.3f')}")
        print(f"  {'':16s} -> {p['t3']['verdict']}; {p['t3']['verdict_vs_dictator']}")
    print("\n(16.10) topology cells: majority vs best seat vs S2, and weighted vs majority on the identical cells")
    for k, p in res.get("topology_cells", {}).items():
        if not p or "rules" not in p:
            continue
        r = p["rules"]
        print(f"  {k:16s} majority {_ci(r['majority'])}  best seat ({p['best_seat']['id']}) {_ci(r['best_seat'])}  S2 {_ci(r['s2'])}  "
              f"majority-best {_ci(p['paired']['majority_minus_best_seat'], '+.3f')}  S2-majority {_ci(p['paired']['s2_minus_majority'], '+.3f')}  "
              f"weighted==majority {_ci(p['shares']['share_weighted_equals_majority'])}")
    ra = res.get("record_agreement")
    if ra:
        print("\nagreement with the record (16.10 table, 16.15.4, aggregation_passthrough.json)")
        for k, v in ra.items():
            flags = {kk: vv for kk, vv in v.items() if kk.endswith("agrees")}
            print(f"  {k:32s} record {v.get('record', v.get('record_point', v.get('record_about')))}  computed "
                  f"{v.get('computed', v.get('computed_in_sample_weighted_minus_dictator'))}  "
                  f"{' '.join(kk + '=' + ('YES' if vv else 'NO') for kk, vv in flags.items())}"
                  + (f"  -- {v['note']}" if v.get("note") else ""))
    if not res["guard_ok"]:
        print(f"\n{GUARD_FAILED}: " + "; ".join(res["guard_failures"]))
    print()


def run(*, draws: int, seed: int) -> dict:
    res: dict = {
        "registration": "not registered; zero-spend replay proposed for the aggregation-theory addendum",
        "estimator": {
            "bootstrap": "item-clustered percentile", "draws": draws, "seed": seed,
            "seed_rule": (f"panel i of PANELS uses base {seed} + {PANEL_SEED_STRIDE} i (as effective_voter.json); "
                          f"topology cell j uses base {seed} + {PANEL_SEED_STRIDE} (4 + j); the fold split uses the base "
                          f"seed; all rules of a community share one resampling stream at base + 1; the reused "
                          f"effective_voter CV oracle keeps its own stream at the base seed"),
            "population": "triples: all three r0 verdicts codable under code_response(extract_verdict(r0), 'published')",
            "weighted_rule": ("log-odds weights log(p/(1-p)), p Laplace-smoothed seat accuracy on the other fold; "
                              "sign of sum w_i (2 x_i - 1); exact tie -> simple majority"),
            "oracle": "analyze_effective_voter.cv_oracle over the three r0 bits, fold fixed per item",
            "t2_ceiling": "gain <= p(1-p)(1-rho) for three voters of common competence p and mean pairwise "
                          "error correlation rho (moment band over the eight profiles, LP-checked in --selftest)",
        },
        "panels": {}, "topology_cells": {}, "guard_ok": True, "guard_failures": []}
    for i, (short, (model, tag)) in enumerate(PANELS.items()):
        recs = load_community(model, tag, SEATS)
        if recs is None:
            res["panels"][short] = None
            continue
        p = analyse_community(recs, seats=SEATS, draws=draws, seed=seed + PANEL_SEED_STRIDE * i,
                              label=short, model=model, tag=tag)
        res["panels"][short] = p
        if not p["guard"]["ok"]:
            res["guard_ok"] = False
            res["guard_failures"].append(f"{short}: " + "; ".join(p["guard"]["reasons"]))
    for j, (cell, (tag, seats)) in enumerate(CELL_SPECS.items()):
        label = f"{cell[0]}/{cell[1]}"
        recs = load_community(TOPOLOGY_MODEL, tag, seats)
        if recs is None:
            res["topology_cells"][label] = None
            continue
        p = analyse_community(recs, seats=seats, draws=draws, seed=seed + PANEL_SEED_STRIDE * (len(PANELS) + j),
                              label=label, model=TOPOLOGY_MODEL, tag=tag)
        res["topology_cells"][label] = p
        if not p["guard"]["ok"]:
            res["guard_ok"] = False
            res["guard_failures"].append(f"{label}: " + "; ".join(p["guard"]["reasons"]))
    res["t2_passthrough"] = t2_from_passthrough()
    res["record_agreement"] = record_agreement(res)
    res["headline"] = headline(res)
    return res


#: the record's values this artefact must reproduce (16.10 RESULTS table,
#: 16.15.4 RESULTS, aggregation_passthrough.json as summarised in the task)
RECORD = {
    "16.10": {"embodied/on": (0.860, 0.860, 0.843), "embodied/off": (0.854, 0.854, 0.822),
              "identical/on": (0.877, 0.880, 0.870), "identical/off": (0.880, 0.883, 0.875)},
    "16.15_grok_oracle_gap": (0.000, -0.004, 0.000),
    "16.15_grok_neutral_acc": 0.860,
    "16.15_grok_majority_equals_neutral": (1642, 1642),
    "16.15_haiku_writer_minus_neutral_point": 0.102,
    "passthrough_same_model_gain_about": 0.01,
    "passthrough_cross_vendor_gain_about": 0.04,
}


def record_agreement(res: dict) -> dict:
    out = {}
    g = res["panels"].get("grok")
    if g and "rules" in g:
        o = g["oracle_cv_effective_voter"]
        want = RECORD["16.15_grok_oracle_gap"]
        out["grok_cv_oracle_gap"] = {"record": want, "computed": (o["gap"], o["lo"], o["hi"]),
                                     "agrees": tuple(round(x, 3) for x in (o["gap"], o["lo"], o["hi"])) == want}
        out["grok_neutral_acc"] = {"record": RECORD["16.15_grok_neutral_acc"], "computed": g["rules"]["dictator"]["point"],
                                   "agrees": round(g["rules"]["dictator"]["point"], 3) == RECORD["16.15_grok_neutral_acc"]}
        out["grok_majority_equals_neutral"] = {
            "record": RECORD["16.15_grok_majority_equals_neutral"],
            "computed": (round(g["shares"]["share_majority_equals_dictator"]["point"] * g["n_triples"]), g["n_triples"]),
            "agrees": g["shares"]["share_majority_equals_dictator"]["point"] == 1.0 and g["n_triples"] == 1642}
    h = res["panels"].get("haiku")
    if h and "rules" in h:
        d = h["paired"]["weighted_insample_minus_dictator"]["point"]
        out["haiku_writer_minus_neutral"] = {
            "record_point": RECORD["16.15_haiku_writer_minus_neutral_point"], "computed_in_sample_weighted_minus_dictator": d,
            "agrees": round(d, 3) == RECORD["16.15_haiku_writer_minus_neutral_point"],
            "note": "the in-sample log-odds rule on haiku IS the writer's advocate, so this paired difference is the "
                    "record's writer minus neutral; its interval here comes from a different resampling stream"}
    for cell, (maj, best, s2) in RECORD["16.10"].items():
        c = res["topology_cells"].get(cell)
        if not c or "rules" not in c:
            continue
        cm, cb, cs = c["rules"]["majority"]["point"], c["rules"]["best_seat"]["point"], c["rules"]["s2"]["point"]
        out[f"16.10_{cell}"] = {
            "record": {"majority": maj, "best_seat": best, "s2": s2},
            "computed": {"majority": cm, "best_seat": cb, "s2": cs},
            "majority_agrees": round(cm, 3) == maj, "s2_agrees": round(cs, 3) == s2,
            "best_seat_agrees": round(cb, 3) == best,
            "note": ("best seat is chosen on the triples here; 16.10 chose it on each seat's own codable rows and "
                     "scored it on the triples, which can pick a different reader on the identical cells "
                     "(within 0.003)" if round(cb, 3) != best else "")}
    t2 = res.get("t2_passthrough")
    if t2:
        out["passthrough_gains"] = {
            "record_about": (RECORD["passthrough_same_model_gain_about"], RECORD["passthrough_cross_vendor_gain_about"]),
            "computed": (t2["same_model"]["mean_gain"], t2["cross_vendor"]["mean_gain"]),
            "agrees": abs(t2["same_model"]["mean_gain"] - 0.01) < 0.005 and abs(t2["cross_vendor"]["mean_gain"] - 0.04) < 0.005}
    return out


def headline(res: dict) -> dict:
    """One line per community per test, for the design document."""
    out = {}
    for k, p in {**res["panels"], **res["topology_cells"]}.items():
        if not p or "rules" not in p:
            continue
        out[k] = {"t1": p["t1"]["verdict"], "t1_dominant_seat": p["t1"]["dominant_seat_in_sample"],
                  "t1_weighted_cv_minus_dictator": _ci(p["paired"]["weighted_cv_minus_dictator"], "+.3f"),
                  "t1_weighted_insample_minus_dictator": _ci(p["paired"]["weighted_insample_minus_dictator"], "+.3f"),
                  "t1_broken_dominance_share_of_draws": p["dominance_broken_share_of_draws"],
                  "t3": p["t3"]["verdict"], "t3_vs_dictator": p["t3"]["verdict_vs_dictator"],
                  "t3_s2_minus_weighted_cv": _ci(p["paired"]["s2_minus_weighted_cv"], "+.3f"),
                  "t3_s2_minus_dictator": _ci(p["paired"]["s2_minus_dictator"], "+.3f"),
                  "t2_own_seats_gain_over_mean_seat": p["t2_own_seats"]["gain_over_mean_seat"],
                  "t2_own_seats_ceiling": p["t2_own_seats"]["ceiling_at_mean_p_and_mean_rho"],
                  "t2_own_seats_applicable": p["t2_own_seats"]["seats_exchangeable"],
                  "uninformative": p["uninformative"]}
    return out


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _synthetic(kind: str, n_items: int = 200) -> list[dict]:
    """'lock1': writer always 0, counterparty always 1, neutral = gold with 20%
    flips. 'lock0_good_advocate': writer = gold, counterparty = gold with 30%
    flips, neutral = gold with 40% flips, flips balanced across gold classes.
    'independent': three seats with competences 0.9 / 0.7 / 0.6 drawn from a
    fixed-seed generator, conditionally independent given gold."""
    rng = random.Random(7)
    recs = []
    for i in range(n_items):
        gold = i % 2
        if kind == "lock1":
            w, c = 0, 1
            nn = gold if i % 10 < 8 else 1 - gold
        elif kind == "lock1_imbalanced":
            # gold 0 on three items in four, neutral = gold with 30% flips:
            # the advocates' base-rate weights (+-logit 0.75 = +-1.10, sum
            # 2.20) outweigh the neutral's (logit 0.70 = 0.85), so the
            # log-odds rule is no dictatorship and returns the majority
            # class on every profile
            gold = 0 if i % 4 else 1
            w, c = 0, 1
            nn = gold if i % 10 < 7 else 1 - gold
        elif kind == "lock0_good_advocate":
            w = gold
            c = gold if i % 20 < 14 else 1 - gold
            nn = gold if i % 20 < 12 else 1 - gold
        elif kind == "independent":
            w, c, nn = (gold if rng.random() < p else 1 - gold for p in (0.9, 0.7, 0.6))
        else:
            raise ValueError(kind)
        recs.append({"item": f"i{i}", "arm": "a", "sample_idx": 0, "gold": gold,
                     "codes": (w, c, nn), "missing": 0, "synthesis": nn, "s2": nn})
    return recs


def _selftest(draws: int = 300) -> int:
    fails = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # rules
    check("five-token collapse: ESH->1, YTA->1, NTA->0, NAH->0, NOVERDICT->None",
          (code_response("ESH", "published"), code_response("YTA", "published"), code_response("NTA", "published"),
           code_response("NAH", "published"), code_response("NOVERDICT", "published")) == (1, 1, 0, 0, None))
    check("weighted vote: dominant positive weight decides, negative weight inverts, zero score -> majority",
          weighted_vote((0, 1, 1), (3.0, 0.5, 0.5)) == 0 and weighted_vote((1, 1, 0), (0.5, 0.5, -3.0)) == 1
          and weighted_vote((0, 1, 1), (1.0, -0.5, -0.5)) == 0 and weighted_vote((1, 1, 0), (0.0, 0.0, 0.0)) == 1)
    check("dominant_seat: (0.4, -0.3, 1.8) -> seat 2; (1.0, 1.0, 1.0) -> none; (1.5, -0.2, 0.9) -> seat 0",
          dominant_seat((0.4, -0.3, 1.8)) == 2 and dominant_seat((1.0, 1.0, 1.0)) is None and dominant_seat((1.5, -0.2, 0.9)) == 0)
    # Nitzan-Paroush / Shapley-Grofman optimality under independence: the log-odds
    # rule reproduces the MAP verdict on every profile, including inverted seats
    ok_all = True
    for p in ((0.9, 0.7, 0.6), (0.85, 0.55, 0.51), (0.6, 0.6, 0.6), (0.9, 0.3, 0.6), (0.7, 0.2, 0.2), (0.95, 0.9, 0.9)):
        w = [math.log(pi / (1 - pi)) for pi in p]
        mp = map_rule_independent(p)
        for x, g in mp.items():
            if g is not None and weighted_vote(x, w) != g:
                ok_all = False
    check("log-odds weighted majority equals the MAP rule on all 8 profiles for six competence vectors (independent voters)", ok_all)
    check("dictatorship condition: MAP rule for (0.9, 0.7, 0.6) is seat 0's verdict on every profile iff w0 > |w1| + |w2|",
          (dominant_seat([math.log(p / (1 - p)) for p in (0.9, 0.7, 0.6)]) == 0)
          == all(v == x[0] for x, v in map_rule_independent((0.9, 0.7, 0.6)).items()))

    # T2 band: closed form vs LP, and the band contains the independence point
    lp_ok, contains = True, True
    lp_available = majority_band_lp(0.8, 0.3) is not None
    for p, rho in ((0.85, 0.65), (0.84, 0.70), (0.905, 0.589), (0.85, 0.35), (0.86, 0.336), (0.6, 0.2), (0.55, 0.9), (0.95, 0.05), (0.7, 0.0)):
        b = majority_band(p, rho)
        lp = majority_band_lp(p, rho)
        if lp is None or abs(lp[0] - b["majority_min"]) > 1e-6 or abs(lp[1] - b["majority_max"]) > 1e-6:
            lp_ok = False
        if rho == 0.0 and not (b["majority_min"] - 1e-9 <= b["independence_majority"] <= b["majority_max"] + 1e-9):
            contains = False
        if b["closed_form_applies"] and abs(b["gain_ceiling"] - b["gain_ceiling_closed_form_p1mp1mrho"]) > 1e-12:
            lp_ok = False
    check("T2 band: closed form equals the LP over the eight profiles on nine (p, rho) points (scipy present)", lp_available and lp_ok)
    check("T2 band: at rho = 0 the independence value lies inside the band, and the ceiling is p(1-p)(1-rho) when 2 p2 >= p", contains)
    check("T2 band: ceiling falls with rho (p = 0.85: rho 0.65 -> 0.045, rho 0.35 -> 0.083)",
          abs(majority_band(0.85, 0.65)["gain_ceiling"] - 0.044625) < 1e-6 and abs(majority_band(0.85, 0.35)["gain_ceiling"] - 0.082875) < 1e-6)
    ex = own_seats_t2(_synthetic("independent", 2000))
    check("own-seat T2 on an independent synthetic community: pairwise error correlations within 0.1 of 0 and gain under ceiling",
          all(abs(v) < 0.1 for v in ex["pairwise_error_correlation"].values()) and ex["under_ceiling"])

    # synthetic communities
    s1 = _synthetic("lock1")
    fold = split_items([r["item"] for r in s1], DEFAULT_SEED)
    st = community_stats(s1, fold, 2)
    check("synthetic lock=1: weighted rule == neutral dictator on 100%, weighted - dictator = 0, majority == dictator",
          st["share_weighted_equals_dictator"] == 1.0 and st["diff_weighted_cv_minus_dictator"] == 0.0 and st["share_majority_equals_dictator"] == 1.0)
    s0 = _synthetic("lock0_good_advocate")
    fold0 = split_items([r["item"] for r in s0], DEFAULT_SEED)
    st0 = community_stats(s0, fold0, 0)
    check("synthetic lock=0 with a perfect advocate: weighted rule is the advocate (acc 1.0), beats dictator by +0.400",
          abs(st0["acc_weighted_cv"] - 1.0) < 1e-12 and abs(st0["diff_weighted_cv_minus_dictator"] - 0.4) < 1e-12)
    a0 = analyse_community(s0, seats=SEATS, draws=draws, seed=DEFAULT_SEED, label="syn")
    check("synthetic lock=0: T1 verdict is BEATS (CI above 0) and dominant seat is the writer's advocate",
          a0["t1"]["verdict"].startswith("weighted rule BEATS") and a0["t1"]["dominant_seat_in_sample"] == "writer_advocate")
    a1 = analyse_community(s1, seats=SEATS, draws=draws, seed=DEFAULT_SEED, label="syn")
    check("synthetic lock=1: T1 verdict is EQUALS and the dominant seat is the neutral adjudicator",
          a1["t1"]["verdict"].startswith("weighted rule EQUALS") and a1["t1"]["dominant_seat_in_sample"] == "neutral_adjudicator")
    check("synthetic: S2 set equal to the neutral seat gives S2 - dictator exactly 0 with a degenerate CI",
          a1["paired"]["s2_minus_dictator"]["point"] == 0.0 and a1["paired"]["s2_minus_dictator"]["lo"] == 0.0)
    si = _synthetic("independent", 600)
    ai = analyse_community(si, seats=SEATS, draws=draws, seed=DEFAULT_SEED, label="syn")
    check("synthetic independent (0.9/0.7/0.6): weighted CV rule >= simple majority, weighted - majority CI not below 0",
          ai["paired"]["weighted_cv_minus_majority"]["point"] >= 0 and ai["paired"]["weighted_cv_minus_majority"]["hi"] >= 0)
    a1b = analyse_community(s1, seats=SEATS, draws=draws, seed=DEFAULT_SEED, label="syn")
    check("a community block is bit-identical when re-drawn with the recorded seeds",
          json.dumps(a1b, sort_keys=True, default=str) == json.dumps(a1, sort_keys=True, default=str))
    check("synthetic lock=1 (balanced gold): the neutral's dominance breaks on fewer than 2% of resamples",
          a1["dominance_broken_share_of_draws"] < 0.02)
    s1i = _synthetic("lock1_imbalanced", 400)
    wi = logodds_weights(s1i)
    fold_i = split_items([r["item"] for r in s1i], DEFAULT_SEED)
    sti = community_stats(s1i, fold_i, 2, 2)
    check("synthetic lock=1 with gold 3:1 and a 0.70 neutral: advocates' base-rate weights outweigh the neutral (no dominant "
          "seat), the log-odds rule returns the base-rate class (acc 0.75 > neutral 0.70) and the flag reports broken dominance",
          dominant_seat(wi) is None and abs(sti["acc_weighted_insample"] - 0.75) < 1e-12 and abs(sti["acc_dictator"] - 0.70) < 1e-12
          and sti["dominance_broken"] == 1)
    check("every CI records its seed: rules at base + 1, oracle at base",
          a1["rules"]["majority"]["seed"] == DEFAULT_SEED + 1 and a1["oracle_cv_effective_voter"]["split_seed"] == DEFAULT_SEED)

    # guard and the literal
    bad = [dict(r, codes=(None, r["codes"][1], r["codes"][2])) if k % 2 else r for k, r in enumerate(s1)]
    check("guard fails on a 50% non-codable community", not guard(bad)["ok"])
    check("guard fails when a cached r0 call is missing", not guard([dict(s1[0], missing=1)] + s1[1:])["ok"])
    check("guard passes on a clean community", guard(s1)["ok"])
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report({"panels": {}, "topology_cells": {}, "guard_ok": False, "guard_failures": ["synthetic"],
                      "estimator": {"draws": 0}, "t2_passthrough": None})
    check("print_report prints the literal GUARD FAILED on a failing guard dict", GUARD_FAILED in buf.getvalue() and "synthetic" in buf.getvalue())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report({"panels": {}, "topology_cells": {}, "guard_ok": True, "guard_failures": [],
                      "estimator": {"draws": 0}, "t2_passthrough": None})
    check("print_report does NOT print GUARD FAILED when guards pass", GUARD_FAILED not in buf.getvalue())

    # real data
    grok = load_community(*PANELS["grok"], SEATS)
    ref = load_panel(*PANELS["grok"])
    if grok is None or ref is None:
        check("grok rows file present", False)
    else:
        check("grok: load_community r0 bits equal analyze_effective_voter.load_panel on every debate",
              len(grok) == len(ref) and all(a["codes"] == b["codes"] and a["gold"] == b["gold"] and a["item"] == b["item"]
                                            for a, b in zip(grok, ref)))
        t = triples(grok)
        check("grok: 1,642 triples, no missing r0 calls, guard ok", len(t) == 1642 and guard(grok)["ok"])
        fold_g = split_items([r["item"] for r in t], DEFAULT_SEED)
        st_g = community_stats(t, fold_g, 2)
        check("grok: neutral seat 0.860 and majority == neutral on 100% of triples",
              round(st_g["acc_dictator"], 3) == 0.860 and st_g["share_majority_equals_dictator"] == 1.0)
        wf = cv_weights_by_fold(t, fold_g)
        check("grok: the neutral seat dominates the fitted weights on both folds (T1 prediction at lock 0.971)",
              dominant_seat(wf[0]) == 2 and dominant_seat(wf[1]) == 2)
        check("grok: weighted CV rule == neutral dictator on 100% of triples, paired difference exactly 0",
              st_g["share_weighted_equals_dictator"] == 1.0 and st_g["diff_weighted_cv_minus_dictator"] == 0.0)
        check("grok: S2 codable on 1,640 of the 1,642 triples, synthesis on 1,641, both on 1,639 (the record's 'triples and debate codable')",
              sum(1 for r in t if r["s2"] is not None) == 1640 and sum(1 for r in t if r["synthesis"] is not None) == 1641
              and sum(1 for r in t if r["s2"] is not None and r["synthesis"] is not None) == 1639)
    haiku = load_community(*PANELS["haiku"], SEATS)
    if haiku is None:
        check("haiku rows file present", False)
    else:
        t = triples(haiku)
        fold_h = split_items([r["item"] for r in t], DEFAULT_SEED + PANEL_SEED_STRIDE * 3)
        st_h = community_stats(t, fold_h, 0)
        check("haiku: 187 triples, writer's advocate 0.818, neutral 0.717, majority 0.701",
              len(t) == 187 and round(st_h["acc_seat0"], 3) == 0.818 and round(st_h["acc_dictator"], 3) == 0.717
              and round(st_h["acc_majority"], 3) == 0.701)
        wf = cv_weights_by_fold(t, fold_h)
        check("haiku: the writer's advocate dominates the in-sample weights and the weights fitted on fold 1; on fold 0's "
              "weights no seat dominates (a genuine weighted majority), and the CV rule beats the dictator by > +0.05",
              dominant_seat(logodds_weights(t)) == 0 and dominant_seat(wf[0]) == 0 and dominant_seat(wf[1]) is None
              and st_h["diff_weighted_cv_minus_dictator"] > 0.05)
    ev = OUT_DIR / "effective_voter.json"
    if ev.exists() and grok is not None:
        e = json.loads(ev.read_text())["panels"]["grok"]["cv_oracle_bits"]
        o = cv_oracle(triples(grok), draws=draws, seed=DEFAULT_SEED, profile=profile_bits, n_splits=1)
        check("grok: the reused CV oracle reproduces effective_voter.json's point gap (+0.000) and oracle accuracy",
              abs(o["gap"] - e["gap"]) < 1e-12 and abs(o["cv_oracle_acc"] - e["cv_oracle_acc"]) < 1e-12)
    for cell, (tag, seats) in CELL_SPECS.items():
        recs = load_community(TOPOLOGY_MODEL, tag, seats)
        if recs is None:
            check(f"topology cell {cell}: rows file present", False)
            continue
        g = guard(recs)
        check(f"topology cell {cell[0]}/{cell[1]}: {g['n_triples']} triples, no missing r0 calls, guard ok", g["ok"])
    t2 = t2_from_passthrough()
    check("T2: aggregation_passthrough.json read, six panels, three same-model and three cross-vendor",
          t2 is not None and len(t2["panels"]) == 6 and t2["same_model"]["n"] == 3 and t2["cross_vendor"]["n"] == 3)
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
