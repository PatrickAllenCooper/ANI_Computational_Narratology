"""
scripts/analyze_flooding.py -- Addendum 16.12: the third sensor failure mode,
FLOODING. Zero spend; every number is a recomputation on the cached grok AITA
deliberation CSVs (and, as a companion only, the Dilemmas Stage 1 CSVs).

THE QUESTION

The dissent sensor is a deviation detector on an anti-correlated advocate
edge that exists only when a verdict has exactly ONE loser. Addendum 16.12
predicted, and tabulated at $0 on the cached 1,680 embodied debates, that a
both-party verdict (ESH "everyone at fault", NAH "no one at fault") collapses
the edge and floods the flag:

  | group verdict          |     n | P(exactly 1) | P(2+ object) | phi(writer, counterparty) | wrong |
  | YTA / NTA (one loser)  | 1,288 |        0.922 |        0.072 |                    -0.901 | 0.064 |
  | ESH (both at fault)    |   272 |        0.382 |        0.614 |                    -0.191 | 0.441 |
  | NAH (no one at fault)  |   118 |        0.331 |        0.576 |                    -0.275 | 0.517 |

That table sits under a non-RESULTS heading with no JSON artefact behind it,
so results.tex stopped citing it (its 1,288 / 0.922 / 0.614 / 0.576 and the
three phis). This module is the zero-spend artefact that restores it: every
cell is recomputed from `cg_deliberation_rows.csv` / `cg_deliberation_votes.csv`
with the registered `code_response` collapse and the registered estimator, and
agreement with the 16.12 quotes is recorded cell by cell.

WHAT IS COMPUTED

Per integrated-verdict type (YTA, NTA, ESH, NAH, INFO -- INFO never occurs in
the cache and is non-codable under the collapse, so it is reported with n = 0),
and for the pooled one-loser (YTA or NTA), both-party (ESH or NAH) and
all-codable rows:

  n, items with a debate in the group, the objector-count distribution
  (P(0), P(exactly 1), P(2+) -- the registered stake-blind counter
  `n_objectors >= 2` is P(2+)), P(wrong) on the deliberation's own final
  verdict, the phi coefficient between the writer_advocate's and the
  counterparty's R3 objections (`objected_r3`) with its 2x2 table, the
  per-seat objection rates and P(both advocates object).

Two populations are reported, because the record uses both:
  s2_codable_16_12   S2 (the `verdict` column, the deployed integrated
                     verdict) codable under the collapse: 1,678 of 1,680.
                     This is the 16.12 population (1,288 + 272 + 118).
  s1_s2_codable_16_15  S1 synthesis AND S2 both codable: 1,677 of 1,680, the
                     16.15 population (1,287 one-loser / 390 both-party,
                     fired 93 / 235), which results.tex cites.
Point estimates, counts and every n are identical wherever the one extra
debate (S2 NTA, S1 NOVERDICT) does not enter.

Verdict type is read from the FINAL integrated verdict (`verdict`, S2), the
column that alone reproduces 16.12 (the S1 synthesis column gives 279 ESH and
73 NAH and reproduces neither, per 16.15.1 RESULTS). Correctness is the
registered five-token collapse (ESH -> at fault, NAH -> not at fault), never
string equality. Objections are the votes CSV's `objected_r3` bit per seat;
the rows CSV's `n_objectors` column is checked against the per-seat sum
(guard) and used as the counter, exactly as analyze_claim_audit.n_objectors_ge2.

ESTIMATOR. Item-clustered percentile bootstrap, 4,000 draws, seed 13,
imported from analyze_unembodied_ablation._boot (the 16.15 estimator).
RESAMPLING FRAME as fixed after the 16.15 audit: every interval resamples the
items of the FULL population (the 210 items behind the codable debates) and
applies the verdict-type restriction INSIDE each resample, treating verdict
type as an outcome of the debate. Each block records `n_items_frame` and
`n_items_kept`. A draw on which a statistic is undefined (an empty group, a
phi with a zero margin) is dropped from that statistic's percentile interval,
which is _boot's own convention.

DILEMMAS (companion, NOT a reproduction). 16.12 did not tabulate the Scruples
Dilemmas corpus; its prose names it as the SILENCE mode. The same readout is
computed on `cg_dilemma_stage1_rows.csv` / `_votes.csv` (verdict tokens
ACTION_A / ACTION_B, always exactly one loser, no both-party type exists in
that instrument; seats person_a_advocate / person_b_advocate; correctness is
token equality, the convention of analyze_dilemma_grip). Those values carry
their own intervals and no 16.12 quote to agree with; the block says so.

GUARDS, checked on the literal string GUARD FAILED before any number is read,
per instrument: S2-codable share >= 0.95; mean `n_r3_unparsed` per debate
<= 0.05; every codable debate carries all three seats' votes (>= 0.99); the
rows CSV's `n_objectors` equals the per-seat `objected_r3` sum on >= 0.99 of
codable debates; the one-loser and both-party strata partition the codable
population. The AITA guard governs the registered readout and the exit code
(3 on failure, the block recorded UNREAD). The Dilemmas companion has its own
guard; on failure only the companion is recorded UNREAD (the 16.14 per-arm
convention), a distinct line COMPANION GUARD NOT MET [dilemmas] is printed and
the exit code stays 0. The literal GUARD FAILED is reserved for the registered
readout: it never appears in the output of a run whose registered guard
passed, so a log grep for it (the 16.15 RESULTS convention) does not classify
a successful run as failed because the companion is UNREAD. On the cache as of
2026-09-12 the companion's S2-codable share is 0.907 (59 genuine UNRESOLVED
and 6 NOVERDICT of 700), so the companion is UNREAD unless a registered
exemption for the instrument's explicit UNRESOLVED token is added; none is
assumed here.

Usage
-----
  python -m scripts.analyze_flooding --selftest
  python -m scripts.analyze_flooding
  python -m scripts.analyze_flooding --draws 400 --seed 13 --no-dilemmas
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.analyze_actuator_ladder import ROWS_PATH, VOTES_PATH, correct as aita_correct
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_unembodied_ablation import _boot as boot_ci
from scripts.run_phase1_quartet import OUT_DIR

ANALYSIS_PATH = OUT_DIR / "flooding_analysis.json"
DILEMMA_ROWS_PATH = OUT_DIR / "cg_dilemma_stage1_rows.csv"
DILEMMA_VOTES_PATH = OUT_DIR / "cg_dilemma_stage1_votes.csv"

DEFAULT_DRAWS = 4000
DEFAULT_SEED = 13
GUARD_FAILED = "GUARD FAILED"          # reserved for the registered (AITA, 16.12) readout
COMPANION_GUARD_FAILED = "COMPANION GUARD NOT MET"   # the Dilemmas companion, exit stays 0
assert GUARD_FAILED not in COMPANION_GUARD_FAILED, "companion token must not trip a GUARD FAILED grep"

# Guards
MIN_CODABLE_SHARE = 0.95
MAX_MEAN_R3_UNPARSED = 0.05
MIN_SEAT_COVERAGE = 0.99
MIN_OBJECTOR_CONSISTENCY = 0.99

ONE_LOSER = ("YTA", "NTA")
BOTH_PARTY = ("ESH", "NAH")
AITA_TYPES = ("YTA", "NTA", "ESH", "NAH", "INFO")

#: The 16.12 table, verbatim (three rows, five numbers each). Rates are quoted
#: to three decimals, so agreement is |got - quoted| <= 0.0005; n is exact.
REPLAY_16_12 = {
    "one_loser": {"n": 1288, "p_exactly_one": 0.922, "p_two_plus": 0.072,
                  "phi": -0.901, "p_wrong": 0.064},
    "ESH": {"n": 272, "p_exactly_one": 0.382, "p_two_plus": 0.614,
            "phi": -0.191, "p_wrong": 0.441},
    "NAH": {"n": 118, "p_exactly_one": 0.331, "p_two_plus": 0.576,
            "phi": -0.275, "p_wrong": 0.517},
}
#: The 16.15.1 RESULTS strata (1,677 population), which results.tex's flooding
#: paragraph cites: n, counter firings, fire rate, P(wrong).
REPLAY_16_15_1 = {
    "one_loser": {"n": 1288 - 1, "n_two_plus": 93, "p_two_plus": 0.072, "p_wrong": 0.064},
    "both_party": {"n": 390, "n_two_plus": 235, "p_two_plus": 0.603, "p_wrong": 0.464},
    "pooled": {"n": 1677, "n_two_plus": 328},
}
RATE_TOL = 0.0005 + 1e-9


# ---------------------------------------------------------------------------
# Instruments: which seats form the advocate edge, which tokens are codable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instrument:
    name: str
    advocate_seats: tuple[str, str]
    neutral_seat: str
    types: tuple[str, ...]                      # single verdict types reported
    groups: dict                                # group name -> verdict tokens
    codable: Callable[[str], bool]
    correct: Callable[[str, str], Optional[bool]]


AITA = Instrument(
    name="aita_grok_deliberation",
    advocate_seats=("writer_advocate", "counterparty"),
    neutral_seat="neutral_adjudicator",
    types=AITA_TYPES,
    groups={**{t: (t,) for t in AITA_TYPES},
            "one_loser": ONE_LOSER, "both_party": BOTH_PARTY,
            "pooled": ONE_LOSER + BOTH_PARTY},
    codable=lambda v: code_response(v, "published") is not None,
    correct=aita_correct,
)

# 16.10 / 16.25: the edges-off cell installs its seats under a "_noedge" suffix
# (run_crowdgold_topology.install), so the votes CSV's role_id column reads
# writer_advocate_noedge / counterparty_noedge / neutral_adjudicator_noedge.
# Same instrument otherwise -- same gold, same verdict coding, same groups.
AITA_NOEDGE = Instrument(
    name="aita_grok_deliberation_noedge",
    advocate_seats=("writer_advocate_noedge", "counterparty_noedge"),
    neutral_seat="neutral_adjudicator_noedge",
    types=AITA_TYPES,
    groups={**{t: (t,) for t in AITA_TYPES},
            "one_loser": ONE_LOSER, "both_party": BOTH_PARTY,
            "pooled": ONE_LOSER + BOTH_PARTY},
    codable=lambda v: code_response(v, "published") is not None,
    correct=aita_correct,
)

DILEMMA_TYPES = ("ACTION_A", "ACTION_B")
DILEMMAS = Instrument(
    name="dilemmas_stage1_grok_deliberation",
    advocate_seats=("person_a_advocate", "person_b_advocate"),
    neutral_seat="neutral_adjudicator",
    types=DILEMMA_TYPES,
    groups={**{t: (t,) for t in DILEMMA_TYPES},
            "one_loser": DILEMMA_TYPES, "pooled": DILEMMA_TYPES},
    codable=lambda v: v in DILEMMA_TYPES,
    correct=lambda v, g: (v == g) if v in DILEMMA_TYPES else None,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def build_debates(rows: Sequence[dict], votes: Sequence[dict], inst: Instrument,
                  ) -> list[dict]:
    """One record per debate from row dicts and vote dicts (CSV schema).
    Keeps EVERY debate; codability is a field, not a filter, so both the
    16.12 (S2 codable) and 16.15 (S1 and S2 codable) populations can be cut
    from the same list. Seat objections come from the votes' `objected_r3`."""
    debates: dict[tuple, dict] = {}
    for r in rows:
        key = (r["item_id"], r["arm"], str(r["sample_idx"]))
        s1, s2, gold = r.get("synthesis_verdict", ""), r.get("verdict", ""), r["gold_verdict"]
        debates[key] = {
            "item": r["item_id"], "arm": r["arm"], "sample_idx": int(r["sample_idx"]),
            "gold": gold, "s1": s1, "s2": s2,
            "s1_codable": inst.codable(s1), "s2_codable": inst.codable(s2),
            "s2_ok": inst.correct(s2, gold),
            "n_objectors": int(r.get("n_objectors") or 0),
            "n_r3_unparsed": int(r.get("n_r3_unparsed") or 0),
            "seats": set(),
        }
    for v in votes:
        d = debates.get((v["item_id"], v["arm"], str(v["sample_idx"])))
        if d is None:
            continue
        seat = v["role_id"]
        d["seats"].add(seat)
        d[f"obj_{seat}"] = v.get("objected_r3") == "1"
    all_seats = (*inst.advocate_seats, inst.neutral_seat)
    for d in debates.values():
        d["seats_complete"] = all(s in d["seats"] for s in all_seats)
        d["n_objectors_votes"] = sum(1 for s in all_seats if d.get(f"obj_{s}"))
        d["seats"] = sorted(d["seats"])
    return sorted(debates.values(), key=lambda d: (d["item"], d["arm"], d["sample_idx"]))


def population(debates: Sequence[dict], which: str) -> list[dict]:
    if which == "s2":
        return [d for d in debates if d["s2_codable"]]
    if which == "s1s2":
        return [d for d in debates if d["s2_codable"] and d["s1_codable"]]
    raise ValueError(which)


# ---------------------------------------------------------------------------
# Statistics on a group of debates (restriction applied inside the resample)
# ---------------------------------------------------------------------------

def phi_from_table(a: int, b: int, c: int, d: int) -> Optional[float]:
    """phi for the 2x2 [[a, b], [c, d]] = [[both, x only], [y only, neither]].
    None when a margin is zero (undefined)."""
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den <= 0:
        return None
    return (a * d - b * c) / math.sqrt(den)


def _pack(d: dict, inst: Instrument) -> tuple:
    """Compact per-debate tuple for the bootstrap: (verdict, n_objectors,
    wrong, advocate-1 objected, advocate-2 objected, neutral objected)."""
    w, c = inst.advocate_seats
    return (d["s2"], d["n_objectors"], int(not d["s2_ok"]),
            int(bool(d.get(f"obj_{w}"))), int(bool(d.get(f"obj_{c}"))),
            int(bool(d.get(f"obj_{inst.neutral_seat}"))))


def _sel(recs: Sequence[tuple], tokens: tuple) -> list[tuple]:
    return [r for r in recs if r[0] in tokens]


def stat_p_objectors(tokens: tuple, pred: Callable[[int], bool]):
    def stat(recs):
        g = _sel(recs, tokens)
        return (sum(1 for r in g if pred(r[1])) / len(g)) if g else None
    return stat


def stat_p_wrong(tokens: tuple):
    def stat(recs):
        g = _sel(recs, tokens)
        return (sum(r[2] for r in g) / len(g)) if g else None
    return stat


def stat_phi(tokens: tuple):
    def stat(recs):
        g = _sel(recs, tokens)
        a = sum(1 for r in g if r[3] and r[4])
        b = sum(1 for r in g if r[3] and not r[4])
        c = sum(1 for r in g if not r[3] and r[4])
        return phi_from_table(a, b, c, len(g) - a - b - c) if g else None
    return stat


def _by_item(debates: Sequence[dict], inst: Instrument) -> dict:
    by: dict = defaultdict(list)
    for d in debates:
        by[d["item"]].append(_pack(d, inst))
    return by


def _ci_block(by_item: dict, stat, *, draws: int, seed: int) -> dict:
    pt, lo, hi = boot_ci(by_item, stat, draws=draws, seed=seed)
    return {"point": pt, "lo": lo, "hi": hi}


def group_block(pop: Sequence[dict], by_item: dict, tokens: tuple, inst: Instrument,
                *, draws: int, seed: int) -> dict:
    """All readouts for one verdict group. `pop` is the full population (the
    resampling frame); the group restriction is applied inside each draw."""
    g = [d for d in pop if d["s2"] in tokens]
    n = len(g)
    w, c = inst.advocate_seats
    counts_by_k = Counter(d["n_objectors"] for d in g)
    a = sum(1 for d in g if d.get(f"obj_{w}") and d.get(f"obj_{c}"))
    b = sum(1 for d in g if d.get(f"obj_{w}") and not d.get(f"obj_{c}"))
    cc = sum(1 for d in g if not d.get(f"obj_{w}") and d.get(f"obj_{c}"))
    dd = n - a - b - cc
    out: dict = {
        "verdict_tokens": list(tokens), "n": n,
        "n_items_kept": len({d["item"] for d in g}),
        "n_items_frame": len(by_item),
        "objector_counts": {str(k): counts_by_k.get(k, 0) for k in range(4)},
        "n_zero": counts_by_k.get(0, 0),
        "n_exactly_one": counts_by_k.get(1, 0),
        "n_two_plus": sum(v for k, v in counts_by_k.items() if k >= 2),
        "n_wrong": sum(1 for d in g if not d["s2_ok"]),
        "p_zero": {"point": (counts_by_k.get(0, 0) / n) if n else None,
                   "note": "point only, no interval"},
        "p_exactly_one": _ci_block(by_item, stat_p_objectors(tokens, lambda k: k == 1),
                                   draws=draws, seed=seed),
        "p_two_plus": _ci_block(by_item, stat_p_objectors(tokens, lambda k: k >= 2),
                                draws=draws, seed=seed),
        "p_wrong": _ci_block(by_item, stat_p_wrong(tokens), draws=draws, seed=seed),
        "phi_advocates": {
            **_ci_block(by_item, stat_phi(tokens), draws=draws, seed=seed),
            "seats": [w, c],
            "table": {"both": a, f"{w}_only": b, f"{c}_only": cc, "neither": dd},
        },
        "seat_objection_rate": {
            s: ((sum(1 for d in g if d.get(f"obj_{s}")) / n) if n else None)
            for s in (w, c, inst.neutral_seat)},
        "p_both_advocates_object": (a / n) if n else None,
        "seat_rates_note": "seat_objection_rate and p_both_advocates_object are point values, no interval",
    }
    out["p_two_plus"]["counter"] = "stake-blind n_objectors >= 2 (analyze_claim_audit.n_objectors_ge2)"
    return out


def analyse_population(debates: Sequence[dict], inst: Instrument, which: str, *,
                       draws: int, seed: int) -> dict:
    pop = population(debates, which)
    by_item = _by_item(pop, inst)
    out: dict = {
        "population": which,
        "description": ("S2 (final integrated `verdict`) codable under the collapse"
                        if which == "s2" else
                        "S1 synthesis AND S2 final verdict both codable under the collapse"),
        "n": len(pop), "n_items": len(by_item),
        "groups": {name: group_block(pop, by_item, tokens, inst, draws=draws, seed=seed)
                   for name, tokens in inst.groups.items()},
    }
    out["raw_verdict_counts"] = dict(sorted(Counter(d["s2"] for d in pop).items()))
    return out


# ---------------------------------------------------------------------------
# Agreement with the record
# ---------------------------------------------------------------------------

def _agree(got, quoted, tol):
    ok = got is not None and abs(got - quoted) <= tol
    return {"got": got, "quoted": quoted, "agrees": bool(ok),
            "abs_diff": (abs(got - quoted) if got is not None else None)}


def agreement_16_12(groups: dict, replay: dict = REPLAY_16_12) -> dict:
    """Cell-by-cell agreement with the 16.12 table on the 16.12 population."""
    out: dict = {"population": "s2_codable_16_12",
                 "tolerance": {"n": 0, "rates_and_phi": RATE_TOL}, "rows": {}}
    for row, q in replay.items():
        g = groups[row]
        fields = {
            "n": _agree(g["n"], q["n"], 0),
            "p_exactly_one": _agree(g["p_exactly_one"]["point"], q["p_exactly_one"], RATE_TOL),
            "p_two_plus": _agree(g["p_two_plus"]["point"], q["p_two_plus"], RATE_TOL),
            "phi": _agree(g["phi_advocates"]["point"], q["phi"], RATE_TOL),
            "p_wrong": _agree(g["p_wrong"]["point"], q["p_wrong"], RATE_TOL),
        }
        out["rows"][row] = {"fields": fields,
                            "agrees": all(v["agrees"] for v in fields.values())}
    out["n_cells"] = sum(len(r["fields"]) for r in out["rows"].values())
    out["n_cells_agree"] = sum(sum(1 for v in r["fields"].values() if v["agrees"])
                               for r in out["rows"].values())
    out["all_agree"] = out["n_cells_agree"] == out["n_cells"]
    return out


def agreement_16_15_1(groups: dict, replay: dict = REPLAY_16_15_1) -> dict:
    """Agreement with the 16.15.1 RESULTS strata on the 16.15 population (the
    values results.tex's flooding paragraph cites)."""
    out: dict = {"population": "s1_s2_codable_16_15",
                 "tolerance": {"counts": 0, "rates": RATE_TOL}, "rows": {}}
    for row, q in replay.items():
        g = groups[row]
        fields = {"n": _agree(g["n"], q["n"], 0),
                  "n_two_plus": _agree(g["n_two_plus"], q["n_two_plus"], 0)}
        if "p_two_plus" in q:
            fields["p_two_plus"] = _agree(g["p_two_plus"]["point"], q["p_two_plus"], RATE_TOL)
        if "p_wrong" in q:
            fields["p_wrong"] = _agree(g["p_wrong"]["point"], q["p_wrong"], RATE_TOL)
        out["rows"][row] = {"fields": fields,
                            "agrees": all(v["agrees"] for v in fields.values())}
    out["all_agree"] = all(r["agrees"] for r in out["rows"].values())
    return out


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def run_guards(debates: Sequence[dict], inst: Instrument) -> dict:
    n_raw = len(debates)
    pop = population(debates, "s2")
    n_cod = len(pop)
    share = n_cod / n_raw if n_raw else 0.0
    mean_unparsed = (sum(d["n_r3_unparsed"] for d in debates) / n_raw) if n_raw else None
    seat_cov = (sum(1 for d in pop if d["seats_complete"]) / n_cod) if n_cod else 0.0
    consistent = (sum(1 for d in pop if d["n_objectors"] == d["n_objectors_votes"]) / n_cod
                  if n_cod else 0.0)
    strata = [t for name, t in inst.groups.items() if name in ("one_loser", "both_party")]
    in_strata = sum(1 for d in pop if any(d["s2"] in t for t in strata))
    checks = {
        "s2_codable_share": {"value": share, "limit": MIN_CODABLE_SHARE,
                             "ok": share >= MIN_CODABLE_SHARE},
        "mean_n_r3_unparsed": {"value": mean_unparsed, "limit": MAX_MEAN_R3_UNPARSED,
                               "ok": mean_unparsed is not None
                               and mean_unparsed <= MAX_MEAN_R3_UNPARSED},
        "seat_vote_coverage_on_codable": {"value": seat_cov, "limit": MIN_SEAT_COVERAGE,
                                          "ok": seat_cov >= MIN_SEAT_COVERAGE},
        "n_objectors_matches_per_seat_sum": {"value": consistent,
                                             "limit": MIN_OBJECTOR_CONSISTENCY,
                                             "ok": consistent >= MIN_OBJECTOR_CONSISTENCY},
        "strata_partition_codable": {"value": in_strata, "limit": n_cod,
                                     "ok": in_strata == n_cod},
    }
    return {"instrument": inst.name, "n_raw": n_raw, "n_s2_codable": n_cod,
            "checks": checks, "passed": all(c["ok"] for c in checks.values())}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _aita_block(aita_debates: Sequence[dict], *, draws: int, seed: int, inst: Instrument = AITA) -> dict:
    raw_counts = dict(sorted(Counter(d["s2"] for d in aita_debates).items()))
    aita: dict = {
        "status": "READ",
        "instrument": inst.name, "n_raw": len(aita_debates),
        "n_items": len({d["item"] for d in aita_debates}),
        "raw_verdict_counts": raw_counts,
        "verdict_types": {t: {"n_raw": raw_counts.get(t, 0), "codable_under_collapse": inst.codable(t),
                              "collapse": code_response(t, "published")}
                          for t in AITA_TYPES},
        "populations": {
            "s2_codable_16_12": analyse_population(aita_debates, inst, "s2", draws=draws, seed=seed),
            "s1_s2_codable_16_15": analyse_population(aita_debates, inst, "s1s2", draws=draws, seed=seed),
        },
    }
    g12 = aita["populations"]["s2_codable_16_12"]["groups"]
    aita["agreement_16_12"] = agreement_16_12(g12)
    aita["agreement_16_15_1"] = agreement_16_15_1(aita["populations"]["s1_s2_codable_16_15"]["groups"])
    # figure-facing view of the 16.12 table (same values as the group blocks)
    aita["table_16_12"] = {
        "population": "s2_codable_16_12",
        "columns": ["n", "p_exactly_one", "p_two_plus", "phi", "p_wrong"],
        "rows": {row: {"n": g12[row]["n"],
                       "p_exactly_one": g12[row]["p_exactly_one"],
                       "p_two_plus": {k: g12[row]["p_two_plus"][k] for k in ("point", "lo", "hi")},
                       "phi": {k: g12[row]["phi_advocates"][k] for k in ("point", "lo", "hi")},
                       "p_wrong": g12[row]["p_wrong"],
                       "quoted_16_12": REPLAY_16_12.get(row)}
                 for row in ("YTA", "NTA", "one_loser", "ESH", "NAH", "both_party", "pooled")},
    }
    aita["notes"] = {
        "info_verdict": ("INFO never occurs in the cache (n_raw 0) and code_response maps it to "
                         "None, so it would be non-codable under the registered collapse"),
        "pooled_vs_within_type_phi": {
            "one_loser_pooled": g12["one_loser"]["phi_advocates"]["point"],
            "YTA": g12["YTA"]["phi_advocates"]["point"],
            "NTA": g12["NTA"]["phi_advocates"]["point"],
            "reading": ("the pooled one-loser phi is carried by which seat loses: on YTA the "
                        "writer_advocate objects and on NTA the counterparty does; within a "
                        "single one-loser type the residual phi is weaker (see per-type rows "
                        "and seat_objection_rate)"),
        },
    }
    return aita


DILEMMA_NOTE = ("COMPANION ONLY. Addendum 16.12 did not tabulate the Dilemmas corpus, so "
                "there is no 16.12 quote to reproduce here; the instrument's verdict space "
                "(ACTION_A / ACTION_B) has exactly one loser by construction and no "
                "both-party type. Values carry their own intervals and are not a "
                "reproduction of any record line.")


def _dilemma_block(dilemma_debates: Sequence[dict], *, draws: int, seed: int) -> dict:
    return {
        "status": "READ",
        "instrument": DILEMMAS.name, "note": DILEMMA_NOTE,
        "n_raw": len(dilemma_debates),
        "n_items": len({d["item"] for d in dilemma_debates}),
        "raw_verdict_counts": dict(sorted(Counter(d["s2"] for d in dilemma_debates).items())),
        "n_s1_s2_codable": len(population(dilemma_debates, "s1s2")),
        "populations": {
            "s2_codable": analyse_population(dilemma_debates, DILEMMAS, "s2",
                                             draws=draws, seed=seed)},
    }


def _unread(debates: Sequence[dict], inst: Instrument, note: Optional[str]) -> dict:
    """A block whose guard failed: identity and raw counts only, no readout."""
    out = {"status": "UNREAD", "instrument": inst.name, "n_raw": len(debates),
           "n_items": len({d["item"] for d in debates}),
           "raw_verdict_counts": dict(sorted(Counter(d["s2"] for d in debates).items())),
           "reason": "guard failed; numbers recorded UNREAD"}
    if note:
        out["note"] = note
    return out


def analyse(aita_debates: Sequence[dict], dilemma_debates: Optional[Sequence[dict]],
            *, draws: int, seed: int, aita_inst: Instrument = AITA) -> dict:
    """Guards run per instrument BEFORE any number is computed. The AITA guard
    governs the registered 16.12 readout and the exit code; the Dilemmas
    companion has its own guard and is recorded UNREAD on its own failure
    without touching the registered block (the 16.14 per-arm convention)."""
    res: dict = {
        "registration": "Addendum 16.12 (zero-spend FLOODING table), reproduced with the registered estimator",
        "draws": draws, "seed": seed,
        "estimator": ("item-clustered percentile bootstrap, analyze_unembodied_ablation._boot; "
                      "verdicts scored by the registered code_response collapse "
                      "(ESH -> at fault, NAH -> not at fault), never string equality"),
        "resampling_frame": ("every interval resamples the items of the FULL population and "
                             "applies the verdict-type restriction inside each draw; see "
                             "n_items_frame / n_items_kept per block; draws on which a "
                             "statistic is undefined are dropped from its interval"),
        "verdict_type_source": "S2 final integrated verdict (`verdict` column)",
        "objection_source": "votes CSV `objected_r3` per seat; counter = rows CSV `n_objectors` >= 2",
    }
    guards: dict = {"aita": run_guards(aita_debates, aita_inst)}
    res["aita"] = (_aita_block(aita_debates, draws=draws, seed=seed, inst=aita_inst)
                   if guards["aita"]["passed"] else _unread(aita_debates, aita_inst, None))
    if dilemma_debates is not None:
        guards["dilemmas"] = run_guards(dilemma_debates, DILEMMAS)
        res["dilemmas"] = (_dilemma_block(dilemma_debates, draws=draws, seed=seed)
                           if guards["dilemmas"]["passed"]
                           else _unread(dilemma_debates, DILEMMAS, DILEMMA_NOTE))
    else:
        res["dilemmas"] = None
    res["guards"] = {
        **guards,
        "passed": guards["aita"]["passed"],
        "companion_passed": (guards["dilemmas"]["passed"] if "dilemmas" in guards else None),
        "rule": ("`passed` is the registered (AITA, 16.12) guard and drives the exit code; "
                 "a failing companion guard leaves only the companion UNREAD"),
    }
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, fmt=".3f"):
    return "n/a" if x is None else format(x, fmt)


def _ci(e: dict, fmt=".3f") -> str:
    return f"{_f(e['point'], fmt)} [{_f(e['lo'], fmt)}, {_f(e['hi'], fmt)}]"


def _table_lines(block: dict, inst: Instrument) -> None:
    print(f"  population {block['population']}: {block['description']}; "
          f"n={block['n']} debates, {block['n_items']} items; verdicts {block['raw_verdict_counts']}")
    print(f"    {'group':<11} {'n':>5}  {'P(exactly 1)':<24} {'P(2+ object)':<24} "
          f"{'phi(advocates)':<26} {'P(wrong)':<24} 2x2 both/x/y/neither")
    for name, g in block["groups"].items():
        t = g["phi_advocates"]["table"]
        tb = "/".join(str(v) for v in t.values())
        print(f"    {name:<11} {g['n']:>5}  {_ci(g['p_exactly_one']):<24} {_ci(g['p_two_plus']):<24} "
              f"{_ci(g['phi_advocates'], '+.3f'):<26} {_ci(g['p_wrong']):<24} {tb}")
    print("    seat objection rates (point):")
    for name, g in block["groups"].items():
        if g["n"] == 0:
            continue
        rates = "  ".join(f"{s} {_f(v)}" for s, v in g["seat_objection_rate"].items())
        print(f"      {name:<11} {rates}  both-advocates {_f(g['p_both_advocates_object'])}")


def _agreement_lines(a: dict, title: str) -> None:
    print(f"\n  {title} (population {a['population']}):")
    for row, r in a["rows"].items():
        print(f"    {row:<11} {'OK     ' if r['agrees'] else 'DIFFERS'} " + "  ".join(
            f"{k} {_f(v['got'], '.3f' if k not in ('n', 'n_two_plus') else 'd')}/{v['quoted']}"
            f"{'' if v['agrees'] else ' (!)'}"
            for k, v in r["fields"].items()))


def print_report(res: dict) -> None:
    print("\n" + "=" * 78)
    print("ADDENDUM 16.12 -- FLOODING: the advocate edge by integrated-verdict type")
    print("=" * 78)
    print(f"  estimator item-clustered bootstrap, {res['draws']} draws, seed {res['seed']}; "
          f"verdict type from S2; collapse code_response")
    a = res["aita"]
    print(f"  AITA grok deliberation: {a['n_raw']} raw debates, {a['n_items']} items, "
          f"raw verdicts {a['raw_verdict_counts']}")
    if a["status"] == "UNREAD":
        print(f"  AITA block UNREAD ({a['reason']})")
    else:
        for key in ("s2_codable_16_12", "s1_s2_codable_16_15"):
            print()
            _table_lines(a["populations"][key], AITA)
        _agreement_lines(a["agreement_16_12"], "AGREEMENT WITH THE 16.12 TABLE")
        ag = a["agreement_16_12"]
        print(f"    {ag['n_cells_agree']} of {ag['n_cells']} cells agree -> "
              f"{'ALL 16.12 CELLS REPRODUCED' if ag['all_agree'] else 'NOT ALL CELLS REPRODUCED'}")
        _agreement_lines(a["agreement_16_15_1"], "AGREEMENT WITH 16.15.1 RESULTS strata (results.tex)")
        pv = a["notes"]["pooled_vs_within_type_phi"]
        print(f"\n  note: one-loser pooled phi {_f(pv['one_loser_pooled'], '+.3f')} vs within-YTA "
              f"{_f(pv['YTA'], '+.3f')} and within-NTA {_f(pv['NTA'], '+.3f')}: {pv['reading']}")
    if res.get("dilemmas"):
        d = res["dilemmas"]
        print("\n" + "-" * 78)
        print(f"  DILEMMAS companion (NOT in 16.12): {d['n_raw']} raw debates, {d['n_items']} items, "
              f"raw verdicts {d['raw_verdict_counts']}")
        if d["status"] == "UNREAD":
            print(f"  companion block UNREAD ({d['reason']})")
        else:
            print(f"  S1&S2 codable {d['n_s1_s2_codable']}")
            _table_lines(d["populations"]["s2_codable"], DILEMMAS)
    print("\n" + "-" * 78)
    g = res["guards"]
    for inst_name in ("aita", "dilemmas"):
        if inst_name not in g:
            continue
        for k, c in g[inst_name]["checks"].items():
            print(f"  guard [{inst_name}] {k:<36} {_f(c['value'], '.4f')}  limit {c['limit']}  "
                  f"{'ok' if c['ok'] else 'FAIL'}")
    if "dilemmas" in g and not g["dilemmas"]["passed"]:
        print(f"  {COMPANION_GUARD_FAILED} [dilemmas] -- companion numbers recorded UNREAD")
    print("  GUARDS PASSED (registered 16.12 readout)" if g["passed"]
          else f"  {GUARD_FAILED} -- numbers above are NOT to be read")


# ---------------------------------------------------------------------------
# Selftest on synthetic rows: ESH fires on every debate, YTA on none
# ---------------------------------------------------------------------------

def _synthetic_aita(seats: tuple = ("writer_advocate", "counterparty", "neutral_adjudicator")
                    ) -> tuple[list[dict], list[dict]]:
    """Hand case, 22 debates on 22 items. YTA x8 (gold YTA, right): exactly
    one advocate objects, the writer on half and the counterparty on the
    other half -> P(2+) = 0, phi = -1. NTA x4 (gold YTA, wrong): the
    counterparty objects alone. ESH x4 (gold NTA, wrong under the collapse):
    BOTH advocates object on every debate, the neutral on half -> P(2+) = 1,
    phi undefined (no variance). NAH x4 (gold NTA, right): the four advocate
    patterns once each -> phi = 0, P(2+) = 1/4. One UNRESOLVED S2 (out of
    both populations) and one NTA (gold NTA, right) with a NOVERDICT S1 (in
    the 16.12 population, out of the 16.15 one)."""
    rows, votes = [], []

    def add(item, idx, s2, gold, obj, s1=None):
        n_obj = sum(obj)
        rows.append({"item_id": item, "arm": "as_asker", "sample_idx": str(idx),
                     "gold_verdict": gold, "verdict": s2,
                     "synthesis_verdict": s1 if s1 is not None else s2,
                     "n_objectors": str(n_obj), "n_r3_unparsed": "0"})
        for s, o in zip(seats, obj):
            votes.append({"item_id": item, "arm": "as_asker", "sample_idx": str(idx),
                          "role_id": s, "objected_r3": "1" if o else "0"})

    for i in range(8):
        add(f"yta{i}", 0, "YTA", "YTA", (1, 0, 0) if i < 4 else (0, 1, 0))
    for i in range(4):
        add(f"nta{i}", 0, "NTA", "YTA", (0, 1, 0))
    for i in range(4):
        add(f"esh{i}", 0, "ESH", "NTA", (1, 1, 1) if i < 2 else (1, 1, 0))
    for i, pat in enumerate(((1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0))):
        add(f"nah{i}", 0, "NAH", "NTA", pat)
    add("unres0", 0, "UNRESOLVED", "YTA", (1, 0, 0))
    add("nta_s1nov", 0, "NTA", "NTA", (0, 1, 0), s1="NOVERDICT")
    return rows, votes


def _synthetic_dilemmas() -> tuple[list[dict], list[dict]]:
    rows, votes = [], []
    seats = ("person_a_advocate", "person_b_advocate", "neutral_adjudicator")

    def add(item, s2, gold, obj):
        rows.append({"item_id": item, "arm": "neutral", "sample_idx": "0",
                     "gold_verdict": gold, "verdict": s2, "synthesis_verdict": s2,
                     "n_objectors": str(sum(obj)), "n_r3_unparsed": "0"})
        for s, o in zip(seats, obj):
            votes.append({"item_id": item, "arm": "neutral", "sample_idx": "0",
                          "role_id": s, "objected_r3": "1" if o else "0"})

    for i in range(10):
        add(f"a{i}", "ACTION_A", "ACTION_A", (0, 1, 0))     # B-advocate loses, objects
    for i in range(10):
        add(f"b{i}", "ACTION_B", "ACTION_A", (1, 0, 0))     # wrong; A-advocate objects
    add("u0", "UNRESOLVED", "ACTION_A", (0, 0, 0))
    return rows, votes


def _selftest(draws: int = 300) -> int:
    import contextlib
    import io
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("phi of a perfectly anti-correlated 2x2 is -1 and of an independent one is 0",
          phi_from_table(0, 3, 3, 0) == -1.0 and phi_from_table(1, 1, 1, 1) == 0.0)
    check("phi is undefined (None) when a margin is zero",
          phi_from_table(4, 0, 0, 0) is None and phi_from_table(0, 0, 0, 0) is None)
    check("code_response collapse: ESH -> 1, NAH -> 0, INFO/UNRESOLVED -> None",
          code_response("ESH", "published") == 1 and code_response("NAH", "published") == 0
          and code_response("INFO", "published") is None
          and code_response("UNRESOLVED", "published") is None)

    rows, votes = _synthetic_aita()
    debates = build_debates(rows, votes, AITA)
    check("build_debates keeps every row (22 debates incl. the UNRESOLVED one)", len(debates) == 22)
    s2 = population(debates, "s2")
    s1s2 = population(debates, "s1s2")
    check("S2-codable population drops UNRESOLVED only (21); S1&S2 also drops the NOVERDICT-S1 debate (20)",
          len(s2) == 21 and len(s1s2) == 20)
    check("n_objectors recomputed from the votes matches the rows column on every debate",
          all(d["n_objectors"] == d["n_objectors_votes"] for d in debates))

    # Regression check, 2026-09-21: _aita_block used to hardcode AITA internally, so an
    # alternate instrument's debates (correctly built with the right seat names) had their
    # phi and seat_objection_rate silently zeroed out (wrong, unsuffixed keys looked up
    # against a dict that only has suffixed keys -- found reading out the 16.25 no-edge
    # k=4 cell, whose seats carry a "_noedge" suffix). The same synthetic case run through
    # a differently-named instrument must give the SAME phi values as the default AITA run.
    noedge_seats = ("writer_advocate_noedge", "counterparty_noedge", "neutral_adjudicator_noedge")
    aita_noedge_inst = Instrument(name="aita_noedge_selftest", advocate_seats=noedge_seats[:2],
                                  neutral_seat=noedge_seats[2], types=AITA_TYPES,
                                  groups=AITA.groups, codable=AITA.codable, correct=AITA.correct)
    rows_ne, votes_ne = _synthetic_aita(noedge_seats)
    debates_ne = build_debates(rows_ne, votes_ne, aita_noedge_inst)
    res_ne = analyse(debates_ne, None, draws=draws, seed=13, aita_inst=aita_noedge_inst)
    G_ne = res_ne["aita"]["populations"]["s2_codable_16_12"]["groups"]
    check("a non-default instrument's phi and table are NOT silently zeroed (regression for "
          "the _aita_block hardcoded-AITA bug)",
          G_ne["YTA"]["phi_advocates"]["point"] == -1.0
          and G_ne["YTA"]["phi_advocates"]["table"]["both"] == 0
          and sum(G_ne["YTA"]["phi_advocates"]["table"].values()) == G_ne["YTA"]["n"] == 8)

    res = analyse(debates, None, draws=draws, seed=13)
    G = res["aita"]["populations"]["s2_codable_16_12"]["groups"]
    check("per-type n on the 16.12 population: YTA 8, NTA 5, ESH 4, NAH 4, INFO 0",
          [G[t]["n"] for t in AITA_TYPES] == [8, 5, 4, 4, 0])
    check("pooled rows: one_loser = YTA + NTA (13), both_party = ESH + NAH (8), pooled 21",
          G["one_loser"]["n"] == 13 and G["both_party"]["n"] == 8 and G["pooled"]["n"] == 21)
    check("ESH fires (2+ objectors) on EVERY debate: P(2+)=1, P(exactly 1)=0",
          G["ESH"]["p_two_plus"]["point"] == 1.0 and G["ESH"]["p_exactly_one"]["point"] == 0.0
          and G["ESH"]["n_two_plus"] == 4)
    check("YTA fires on NONE: P(2+)=0, P(exactly 1)=1",
          G["YTA"]["p_two_plus"]["point"] == 0.0 and G["YTA"]["p_exactly_one"]["point"] == 1.0)
    check("a group that fires everywhere / nowhere has a degenerate interval [1, 1] / [0, 0]",
          G["ESH"]["p_two_plus"]["lo"] == 1.0 and G["ESH"]["p_two_plus"]["hi"] == 1.0
          and G["YTA"]["p_two_plus"]["lo"] == 0.0 and G["YTA"]["p_two_plus"]["hi"] == 0.0)
    check("YTA advocate edge is perfectly anti-correlated: phi = -1, 2x2 = 0/4/4/0",
          G["YTA"]["phi_advocates"]["point"] == -1.0
          and list(G["YTA"]["phi_advocates"]["table"].values()) == [0, 4, 4, 0])
    check("ESH edge collapses to a constant (both always object): phi undefined, 2x2 = 4/0/0/0",
          G["ESH"]["phi_advocates"]["point"] is None
          and list(G["ESH"]["phi_advocates"]["table"].values()) == [4, 0, 0, 0])
    check("NAH edge is independent: phi = 0, P(2+) = 1/4, P(exactly 1) = 1/2, P(0) = 1/4",
          abs(G["NAH"]["phi_advocates"]["point"]) < 1e-12
          and abs(G["NAH"]["p_two_plus"]["point"] - 0.25) < 1e-12
          and abs(G["NAH"]["p_exactly_one"]["point"] - 0.5) < 1e-12
          and abs(G["NAH"]["p_zero"]["point"] - 0.25) < 1e-12)
    check("one-loser pooled phi = -1 (writer objects on YTA, counterparty on NTA, never both; 2x2 0/4/9/0)",
          G["one_loser"]["phi_advocates"]["point"] == -1.0
          and list(G["one_loser"]["phi_advocates"]["table"].values()) == [0, 4, 9, 0])
    check("P(wrong) by the collapse: YTA 0, NTA 4/5, ESH 1 (gold NTA), NAH 0, one-loser 4/13",
          G["YTA"]["p_wrong"]["point"] == 0.0 and abs(G["NTA"]["p_wrong"]["point"] - 0.8) < 1e-12
          and G["ESH"]["p_wrong"]["point"] == 1.0 and G["NAH"]["p_wrong"]["point"] == 0.0
          and abs(G["one_loser"]["p_wrong"]["point"] - 4 / 13) < 1e-12)
    check("INFO row: n 0, every statistic None, flagged non-codable under the collapse",
          G["INFO"]["n"] == 0 and G["INFO"]["p_two_plus"]["point"] is None
          and G["INFO"]["phi_advocates"]["point"] is None
          and res["aita"]["verdict_types"]["INFO"]["codable_under_collapse"] is False)
    check("seat objection rates: on YTA writer 1/2, counterparty 1/2, neutral 0; on ESH neutral 1/2",
          G["YTA"]["seat_objection_rate"]["writer_advocate"] == 0.5
          and G["YTA"]["seat_objection_rate"]["counterparty"] == 0.5
          and G["YTA"]["seat_objection_rate"]["neutral_adjudicator"] == 0.0
          and G["ESH"]["seat_objection_rate"]["neutral_adjudicator"] == 0.5
          and G["ESH"]["p_both_advocates_object"] == 1.0)
    check("full resampling frame: every group resamples all 21 items and keeps only its own",
          G["ESH"]["n_items_frame"] == 21 and G["ESH"]["n_items_kept"] == 4
          and G["one_loser"]["n_items_kept"] == 13 and G["pooled"]["n_items_kept"] == 21)
    G15 = res["aita"]["populations"]["s1_s2_codable_16_15"]["groups"]
    check("16.15 population: one-loser drops the NOVERDICT-S1 debate (12 vs 13), both-party unchanged (8)",
          G15["one_loser"]["n"] == 12 and G15["both_party"]["n"] == 8
          and G15["NTA"]["n"] == 4 and G15["NTA"]["p_wrong"]["point"] == 1.0)
    # a bootstrap interval must bracket the point for a mixed group
    bp = G["both_party"]["p_two_plus"]
    check("both-party P(2+) = 5/8 with an interval bracketing the point",
          abs(bp["point"] - 5 / 8) < 1e-12 and bp["lo"] <= bp["point"] <= bp["hi"]
          and bp["lo"] < bp["hi"])
    check("_boot is item-clustered: a two-item group with opposite outcomes gets lo 0 and hi 1",
          (lambda e: e["lo"] == 0.0 and e["hi"] == 1.0)(
              _ci_block({"a": [("YTA", 2, 0, 0, 0, 0)], "b": [("YTA", 1, 0, 0, 0, 0)]},
                        stat_p_objectors(("YTA",), lambda k: k >= 2), draws=400, seed=13)))

    # agreement machinery: a quoted table that matches passes, one that does not fails
    quoted_ok = {"ESH": {"n": 4, "p_exactly_one": 0.0, "p_two_plus": 1.0, "phi": None,
                         "p_wrong": 1.0}}
    quoted_ok["ESH"]["phi"] = -0.191    # phi is None here -> must NOT agree
    ag = agreement_16_12(G, quoted_ok)
    check("agreement: matching cells agree, an undefined phi against a quoted phi does not",
          ag["rows"]["ESH"]["fields"]["n"]["agrees"] and ag["rows"]["ESH"]["fields"]["p_two_plus"]["agrees"]
          and not ag["rows"]["ESH"]["fields"]["phi"]["agrees"] and not ag["all_agree"]
          and ag["n_cells_agree"] == 4 and ag["n_cells"] == 5)
    ag2 = agreement_16_12(G, {"NAH": {"n": 4, "p_exactly_one": 0.5, "p_two_plus": 0.25,
                                      "phi": 0.0, "p_wrong": 0.0}})
    check("agreement: a fully matching row reports all_agree", ag2["all_agree"])
    ag3 = agreement_16_12(G, {"NAH": {"n": 4, "p_exactly_one": 0.5, "p_two_plus": 0.251,
                                      "phi": 0.0, "p_wrong": 0.0}})
    check("agreement: a rate off by 0.001 (beyond 3-dp rounding) DIFFERS",
          not ag3["rows"]["NAH"]["fields"]["p_two_plus"]["agrees"])
    ag4 = agreement_16_15_1(G15, {"one_loser": {"n": 12, "n_two_plus": 0, "p_two_plus": 0.0,
                                                "p_wrong": 1 / 3}})
    check("agreement_16_15_1 checks n, firing count and rates", ag4["all_agree"])

    # guards
    gd = res["guards"]
    check("guards pass on the clean synthetic panel", gd["passed"] and gd["aita"]["passed"])
    rows_bad = [dict(r, n_objectors="3") for r in rows]
    gd2 = run_guards(build_debates(rows_bad, votes, AITA), AITA)
    check("guard fails when the rows' n_objectors disagrees with the per-seat objections",
          not gd2["passed"] and not gd2["checks"]["n_objectors_matches_per_seat_sum"]["ok"])
    votes_missing = [v for v in votes if v["role_id"] != "counterparty"]
    gd3 = run_guards(build_debates(rows, votes_missing, AITA), AITA)
    check("guard fails when a seat's votes are missing",
          not gd3["passed"] and not gd3["checks"]["seat_vote_coverage_on_codable"]["ok"])
    rows_unc = rows + [dict(rows[0], item_id=f"u{i}", verdict="UNRESOLVED") for i in range(5)]
    votes_unc = votes + [dict(v, item_id=f"u{i}") for i in range(5) for v in votes[:3]]
    gd4 = run_guards(build_debates(rows_unc, votes_unc, AITA), AITA)
    check("guard fails when too many debates are non-codable",
          not gd4["passed"] and not gd4["checks"]["s2_codable_share"]["ok"])
    rows_unp = [dict(r, n_r3_unparsed="1") for r in rows]
    gd5 = run_guards(build_debates(rows_unp, votes, AITA), AITA)
    check("guard fails when R3 parsing is broken", not gd5["passed"])

    # Dilemmas companion
    drows, dvotes = _synthetic_dilemmas()
    dd = build_debates(drows, dvotes, DILEMMAS)
    res2 = analyse(debates, dd, draws=draws, seed=13)
    D = res2["dilemmas"]["populations"]["s2_codable"]["groups"]
    check("dilemmas: ACTION_A 10 right, ACTION_B 10 wrong, UNRESOLVED excluded, pooled 20 of 21",
          D["ACTION_A"]["n"] == 10 and D["ACTION_A"]["p_wrong"]["point"] == 0.0
          and D["ACTION_B"]["n"] == 10 and D["ACTION_B"]["p_wrong"]["point"] == 1.0
          and D["pooled"]["n"] == 20 and res2["dilemmas"]["n_raw"] == 21)
    check("dilemmas: the losing advocate objects alone -> P(2+) 0, pooled phi -1 on the person seats",
          D["pooled"]["p_two_plus"]["point"] == 0.0 and D["pooled"]["phi_advocates"]["point"] == -1.0
          and D["pooled"]["phi_advocates"]["seats"] == ["person_a_advocate", "person_b_advocate"])
    check("dilemmas block is labelled as a companion, not a 16.12 reproduction",
          "COMPANION ONLY" in res2["dilemmas"]["note"]
          and "did not tabulate" in res2["dilemmas"]["note"])
    check("dilemmas guards pass and the registered guard passes",
          res2["guards"]["passed"] and res2["guards"]["companion_passed"]
          and res2["dilemmas"]["status"] == "READ")
    check("table_16_12 view carries the same values as the group blocks",
          res2["aita"]["table_16_12"]["rows"]["ESH"]["p_two_plus"]["point"]
          == G["ESH"]["p_two_plus"]["point"]
          and res2["aita"]["table_16_12"]["rows"]["one_loser"]["quoted_16_12"] == REPLAY_16_12["one_loser"])
    # companion guard failure: companion UNREAD, registered block untouched, exit stays 0
    drows_bad = drows + [dict(drows[0], item_id=f"uu{i}", verdict="UNRESOLVED",
                              synthesis_verdict="UNRESOLVED") for i in range(6)]
    dvotes_bad = dvotes + [dict(v, item_id=f"uu{i}") for i in range(6) for v in dvotes[:3]]
    res3 = analyse(debates, build_debates(drows_bad, dvotes_bad, DILEMMAS), draws=draws, seed=13)
    check("a failing companion guard records the companion UNREAD with no readout numbers",
          res3["dilemmas"]["status"] == "UNREAD" and "populations" not in res3["dilemmas"]
          and res3["guards"]["companion_passed"] is False)
    check("a failing companion guard leaves the registered block READ and `passed` True",
          res3["aita"]["status"] == "READ" and res3["guards"]["passed"]
          and res3["aita"]["agreement_16_12"] is not None)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(res3)
    check("print_report prints COMPANION GUARD NOT MET [dilemmas] and GUARDS PASSED for the registered readout",
          f"{COMPANION_GUARD_FAILED} [dilemmas]" in buf.getvalue()
          and "GUARDS PASSED" in buf.getvalue())
    check("a failing companion guard never prints the literal GUARD FAILED (reserved for the registered readout)",
          GUARD_FAILED not in buf.getvalue() and GUARD_FAILED not in COMPANION_GUARD_FAILED)
    # registered guard failure: AITA block UNREAD, no numbers computed
    res4 = analyse(build_debates(rows_unc, votes_unc, AITA), dd, draws=draws, seed=13)
    check("a failing registered guard records the AITA block UNREAD and `passed` False",
          res4["aita"]["status"] == "UNREAD" and "populations" not in res4["aita"]
          and not res4["guards"]["passed"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(res4)
    check("print_report prints the literal GUARD FAILED when the registered guard fails",
          f"{GUARD_FAILED} -- numbers above are NOT to be read" in buf.getvalue())

    # the literal GUARD FAILED must reach stdout through print_report
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(res2)
    check("print_report does NOT print GUARD FAILED when guards pass",
          GUARD_FAILED not in buf.getvalue() and "GUARDS PASSED" in buf.getvalue())
    res_bad = dict(res2, guards=dict(res2["guards"], passed=False))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(res_bad)
    check("print_report prints the literal GUARD FAILED on a failing guard dict",
          GUARD_FAILED in buf.getvalue())
    check("registered guard does not fail for the companion's sake",
          res3["guards"]["passed"] and not res3["guards"]["companion_passed"])
    check("artefact is JSON-serialisable", json.dumps(res2, default=str) is not None)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--rows", type=Path, default=ROWS_PATH)
    ap.add_argument("--votes", type=Path, default=VOTES_PATH)
    ap.add_argument("--instrument", choices=("aita", "aita_noedge"), default="aita",
                    help="aita_noedge selects the edges-off cell's _noedge seat ids (16.10/16.25)")
    ap.add_argument("--dilemma-rows", type=Path, default=DILEMMA_ROWS_PATH)
    ap.add_argument("--dilemma-votes", type=Path, default=DILEMMA_VOTES_PATH)
    ap.add_argument("--no-dilemmas", action="store_true",
                    help="skip the Dilemmas companion block")
    ap.add_argument("--json", type=Path, default=ANALYSIS_PATH)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    inst = AITA_NOEDGE if a.instrument == "aita_noedge" else AITA
    aita = build_debates(read_csv(a.rows), read_csv(a.votes), inst)
    dilemmas = None
    if not a.no_dilemmas and a.dilemma_rows.exists() and a.dilemma_votes.exists():
        dilemmas = build_debates(read_csv(a.dilemma_rows), read_csv(a.dilemma_votes), DILEMMAS)
    elif not a.no_dilemmas:
        print(f"dilemma CSVs absent ({a.dilemma_rows}, {a.dilemma_votes}): companion block skipped")
    res = analyse(aita, dilemmas, draws=a.draws, seed=a.seed, aita_inst=inst)
    res["inputs"] = {"rows": str(a.rows), "votes": str(a.votes),
                     "dilemma_rows": str(a.dilemma_rows) if dilemmas is not None else None,
                     "dilemma_votes": str(a.dilemma_votes) if dilemmas is not None else None}
    print_report(res)
    a.json.write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {a.json}")
    if not res["guards"]["passed"]:
        print(GUARD_FAILED)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
