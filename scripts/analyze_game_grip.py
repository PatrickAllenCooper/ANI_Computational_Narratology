"""
scripts/analyze_game_grip.py -- the stake-grip screen, role-lock, Nash-pull
and extent readouts for the TMGBench binding-joint-plan deliberation
(scripts/run_crowdgold_game.py), one third-seat arm at a time, with the
registered planner-vs-neutral contrast on the 16 value-region games when
both arms are on disk.

One-to-one correspondence with scripts/analyze_dilemma_grip.py for the grip
screen: the same three conditions, the same thresholds (imported from it),
the same item-clustered bootstrap (_boot_ci imported from it), on the two
player_*_advocate seats instead of Dilemmas' person_*_advocate seats:

  G1  composite-flag fire rate <= 0.50
  G2  REJECT share of R4 votes >= 0.05
  G3  stake concentration on the stake seats (R4 reject rate | undermined
      minus | not undermined) >= +0.20, 95% item-clustered CI excludes 0

WHY A SEPARATE MODULE INSTEAD OF PARAMETERISING THE DILEMMAS ONE

analyze_dilemma_grip.py hard-codes STAKE_SEATS = (person_a_advocate,
person_b_advocate), the third seat name neutral_adjudicator and the two-way
ACTION_A / ACTION_B instrument. This instrument has five verdict tokens, a
third seat whose role id DIFFERS BY ARM (neutral_reader | plan_mediator), a
gold that is NO_UNIQUE_PLAN on 31 of 144 games, and four readouts the
Dilemmas run never had (role-lock, Nash-pull, the extent table, the third-
seat contrast). The sensor predicates and the G1/G2/G3 formulas below are
copy-checked against the sibling; everything else is native.

WHAT THIS MODULE READS

  <tag>_rows.csv / <tag>_votes.csv   written by run_crowdgold_game (ROW_FIELDS /
                                     VOTE_FIELDS), one arm per tag
  data/tmgbench_classic + gold       through tmg_games.load_games (sha-pinned);
                                     every cell's ItemView is RE-RENDERED from
                                     (game_id, sample_idx) and checked against
                                     the CSV's permutation, names and gold, so
                                     the payoffs behind every seat's verdict
                                     are the ones the model actually saw
  cgd_*_game_plan_ncot_* cache       optional, for the truncation column of
                                     the guard (see GUARD)

GUARD (runs first; on failure nothing else is printed)

Per-call guard rows are reconstructed from the CSVs: every one of the 17
calls per cell has its parsed token in the CSVs (r0/r1/r2 verdicts, r3
label, r4 vote per seat; synthesis and integration verdicts per cell), so
the NOVERDICT share is complete. A token outside that round's COMMITTAL set
(run_crowdgold_game.ROUND_INSTRUMENT x verdict_format.committal_tokens) is
coded NOVERDICT here whatever the CSV cell says -- a blank, a stray
UNRESOLVED (registered as legal by verdict_format, never offered by any
instruction) or anything else. Without that a run whose every vote was
UNRESOLVED would pass the guard and read as G1 = 0 / G2 = 0, grip False: a
silent false negative on two registered predictions. The selftest pins it.
The CSVs record finish_reason only for the
integration and r4 calls; the other 13 calls' truncation flags are read from
the call cache when the records can be found by name (largest token cap
wins when a cell was re-run at several caps), and the coverage is printed.
Thresholds are tmg_games.MAX_TRUNCATION_SHARE / MAX_NOVERDICT_SHARE (0.05,
mirroring run_crowdgold_aita), grouped per (model, round:role_id, arm)
exactly as the runner's own call_guard_row groups them. When the guard
fails the report prints the literal string GUARD FAILED and refuses to
print or return any grip, role-lock, Nash-pull or accuracy number.

DEFINITIONS

  plan_correct(verdict, gold)   tmg_games.score_plan: string match on the
      committal tokens, NOVERDICT / UNRESOLVED scored wrong and kept in the
      denominator (the registered verdict_format rule). NO_UNIQUE_PLAN is a
      committal token: against a null-gold game (gold NO_UNIQUE_PLAN) it is
      CORRECT. The selftest pins this as a regression guard.

  favours(plan, player)   the plan names a cell at which that player's own
      ordinal payoff is 3 or 4 -- one of the player's two HIGHER payoffs
      among the four cells a binding agreement can name (under a binding
      agreement every cell is a group-plausible plan; a plan does not have
      to be self-enforcing, which is exactly why Nash-pull is a separate
      readout). This needs no comparison of one player's ranks with the
      other's, is defined for every PLAN_xy, and is binary per player. A
      cell can favour both players ((3,3), (4,4), (3,4), (4,3)) or neither
      ((1,1), (2,2), (1,2), (2,1)). NO_UNIQUE_PLAN and NOVERDICT favour no
      one and are excluded from the favour rates (their shares are reported
      alongside). Alternative considered and rejected: restricting to the
      Pareto-efficient cells makes every ASSURANCE / HARMONY game (a single
      efficient cell) uninformative by construction and drops the dominated
      cell a locked advocate actually names in a stag hunt.

  role-lock   the manipulation check that the advocates pull in opposite,
      stake-consistent directions at r0 (before any exchange):
        pull_row = P(favours row | R-advocate r0) - P(favours row | C-advocate r0)
        pull_col = P(favours col | C-advocate r0) - P(favours col | R-advocate r0)
        role_lock = |(pull_row + pull_col) / 2|
      pull_row alone is the task's one-sided form |P(favours row) R - C|;
      the two one-sided forms are not each other's complements (a cell can
      favour both or neither), so the report carries both, signed, and the
      lock is the absolute value of their mean. 0 for two seats that behave
      identically, 1 for a pair that always names its own player's higher
      cells and never the other's. Item-clustered CI on the signed mean.

  Nash-pull   P(seat's verdict is a pure Nash cell | verdict is a PLAN cell),
      by seat, overall / by region / by gold_type, for r0 and r2. Games with
      no pure Nash cell (CYCLIC_NO_IDEAL) contribute 0 by construction.

  accuracy   plan_correct on every cell (non-committal = wrong): the group's
      final (integration) verdict and the synthesis verdict, and each seat's
      r0 and r2 verdict; overall, by region and by gold_type (the extent
      table: value = DILEMMA_BOTH_PREFER + ASSURANCE, neutral =
      INCENTIVE_COMPATIBLE_EFFICIENT + HARMONY, cannot_assist = CYCLIC +
      CONFLICT + ZERO_SUM).

  G1 / G2 / G3   as the sibling. flagged(d) = third seat objected at r3 OR
      an advocate objected while its stake was NOT undermined (mis-
      localised); an advocate whose undermined flag is undefined is never
      mis-localised (the sibling's `.get(..., True)` default). The grip
      population is the debates whose synthesis and final verdicts are both
      committal, as the sibling's load_debates filters. ONE DELIBERATE
      DIFFERENCE from the sibling's load_votes: a blank stake_undermined is
      kept as None and EXCLUDED from G3, not coded 0. In the Dilemmas run a
      blank only meant a non-committal group verdict (bounded by the guard);
      here it also means the group verdict was NO_UNIQUE_PLAN (no plan, no
      payoff to compare) or the game has no pure Nash cell (the reference is
      undefined) -- a large, structured share that would otherwise be
      silently coded "not undermined". The runner's docstring registers
      this exclusion; the count of excluded votes is reported.

  flag concentration (stake-blind)   fired(d) = n_objectors >= 2, the
      stake-blind counter analyze_topology_2x2 uses for its transfer readout
      because it needs no stake semantics. Reported: fire rate, and the
      error lift P(group wrong | fired) - P(group wrong | not fired) on the
      committal cells, overall and by region, read only at >= 60 fired
      (analyze_topology_2x2.MIN_FIRED). With --solo-rows a transfer lift
      onto an independent single-agent run's verdicts on the same games is
      computed the topology way: an item is flagged when a majority of its
      cells fired; P(solo wrong | flagged item) - P(solo wrong | not).

  contrast   when a second tag is given and its third seat differs, the
      value-region cells shared by both arms are compared PAIRED on items
      (each bootstrap draw resamples item ids once and evaluates both arms
      on the same list, as analyze_stake_grip.paired_g3_delta): planner
      minus neutral on group final accuracy, synthesis accuracy, third-seat
      r0 and r2 accuracy, third-seat r0 Nash-pull, stake-blind fire rate,
      and G3. The registered prediction is a positive group-accuracy delta.

Usage:
  python -m scripts.analyze_game_grip --selftest
  python -m scripts.analyze_game_grip --tag cg_game_neutral
  python -m scripts.analyze_game_grip --tag cg_game_neutral --contrast-tag cg_game_planner --out /tmp/x.json

Exit codes: 0 ok, 2 missing / empty inputs, 4 guard failed (nothing read).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import tempfile
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.analyze_dilemma_grip import G1_MAX_FIRE, G2_MIN_REJECT, G3_MIN_CONC, _boot_ci
from scripts.run_crowdgold_game import (
    ARM_OF,
    ROUNDS,
    ROUND_INSTRUMENT,
    ROW_FIELDS,
    SCAFFOLD_NAME,
    STAKE_SEATS,
    TAG_PREFIX,
    THIRD_NEUTRAL,
    THIRD_PLANNER,
    THIRD_ROLE,
    VOTE_FIELDS,
    GameItem,
    moderator_role_id,
    stake_undermined,
)
from scripts.run_phase1_quartet import OUT_DIR, _safe
from scripts.tmg_games import (
    GOLD_TYPES,
    NO_UNIQUE_PLAN,
    PLAN_VERDICTS,
    REGIONS,
    REGION_VALUE,
    VERDICT_SET,
    Game,
    ItemView,
    cache_files_with_prefix,
    load_games,
    parse_guard,
    print_parse_guard,
    render_item,
    score_plan,
)
from scripts.verdict_format import NOVERDICT, committal_tokens

ROW_SEAT, COL_SEAT = STAKE_SEATS
THIRD_ROLE_IDS = tuple(r.role_id for r in THIRD_ROLE.values())
#: favours(plan, player): the player's own payoff at the plan is one of its
#: two higher ordinal payoffs (3 or 4 of 1..4).
FAVOUR_MIN_PAYOFF = 3
#: analyze_topology_2x2's stake-blind counter and its power floor.
FIRE_MIN_OBJECTORS = 2
MIN_FIRED = 60
SEAT_ROUNDS = ("r0", "r2")
#: Column labels for the extent table.
SEAT_SHORT = {ROW_SEAT: "R-adv", COL_SEAT: "C-adv", "neutral_reader": "neutral",
              "plan_mediator": "planner"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    return int(v)


def plan_correct(verdict: str, gold: str) -> Optional[bool]:
    """tmg_games.score_plan: non-committal is WRONG and stays in the
    denominator; NO_UNIQUE_PLAN against a null-gold game is CORRECT. None only
    when the gold itself is not a legal token (a malformed row)."""
    if gold not in VERDICT_SET:
        return None
    return bool(score_plan(verdict or NOVERDICT, gold)["correct"])


def committal_correct(verdict: str, gold: str) -> Optional[bool]:
    """The sibling's syn_ok / s2_ok: None unless the verdict is committal."""
    if verdict not in VERDICT_SET:
        return None
    return plan_correct(verdict, gold)


def favours(view: ItemView, verdict: str, player: str) -> Optional[bool]:
    """Does the plan give ``player`` ('row' | 'col') one of its two higher
    payoffs? None for NO_UNIQUE_PLAN / NOVERDICT / anything not a PLAN cell."""
    p = view.payoffs_for(verdict)
    if p is None:
        return None
    return (p[0] if player == "row" else p[1]) >= FAVOUR_MIN_PAYOFF


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_debates(rows_path: Path, votes_path: Path, *,
                 model: Optional[str] = None) -> list[dict]:
    """One record per debate (cell), with the sensor fields flagged /
    mislocalised expect (obj_<seat>, undermined_<seat>) plus everything the
    guard and the readouts need. Unlike the sibling this returns EVERY
    debate; grip_population() applies the sibling's committal filter."""
    debates: dict[tuple, dict] = {}
    with open(rows_path, newline="") as f:
        for r in csv.DictReader(f):
            if model and r.get("model") != model:
                continue
            key = (r["item_id"], r["arm"], r["sample_idx"])
            third = r["third_role_id"]
            debates[key] = {
                "item": r["item_id"], "game_id": r.get("game_id") or r["item_id"],
                "arm": r["arm"], "third_seat": r["third_seat"],
                "third_role_id": third, "seats": (ROW_SEAT, COL_SEAT, third),
                "model": r["model"],
                "moderator_model": r.get("moderator_model") or r["model"],
                "scaffold": r.get("scaffold") or SCAFFOLD_NAME,
                "sample_idx": int(r["sample_idx"]),
                "gold_type": r["gold_type"], "region": r["region"],
                "gold": r["gold_verdict"],
                "row_perm": r["row_perm"], "col_perm": r["col_perm"],
                "row_name": r["row_name"], "col_name": r["col_name"],
                "verdict": r["verdict"], "synthesis_verdict": r["synthesis_verdict"],
                "correct": plan_correct(r["verdict"], r["gold_verdict"]),
                "syn_correct": plan_correct(r["synthesis_verdict"], r["gold_verdict"]),
                "syn_ok": committal_correct(r["synthesis_verdict"], r["gold_verdict"]),
                "s2_ok": committal_correct(r["verdict"], r["gold_verdict"]),
                "r0": {ROW_SEAT: r["r0_verdict_row_advocate"],
                       COL_SEAT: r["r0_verdict_col_advocate"],
                       third: r["r0_verdict_third"]},
                "r1": {ROW_SEAT: r["r1_verdict_row_advocate"],
                       COL_SEAT: r["r1_verdict_col_advocate"],
                       third: r["r1_verdict_third"]},
                "r2": {ROW_SEAT: r["r2_verdict_row_advocate"],
                       COL_SEAT: r["r2_verdict_col_advocate"],
                       third: r["r2_verdict_third"]},
                "n_objectors": int(r.get("n_objectors") or 0),
                "n_reject": int(r.get("n_reject") or 0),
                "n_r3_unparsed": int(r.get("n_r3_unparsed") or 0),
                "n_vote_unparsed": int(r.get("n_vote_unparsed") or 0),
                "unanimous_accept": r.get("unanimous_accept") == "1",
                "verdict_revised": r.get("verdict_revised") == "1",
                "truncated": _int(r.get("truncated")),
                "output_len": _int(r.get("output_len")),
                "synthesis_len": _int(r.get("synthesis_len")),
            }
    with open(votes_path, newline="") as f:
        for v in csv.DictReader(f):
            d = debates.get((v["item_id"], v["arm"], v["sample_idx"]))
            if d is None:
                continue
            seat = v["role_id"]
            d[f"obj_{seat}"] = v["objected_r3"] == "1"
            d[f"rej_{seat}"] = v["r3_label"] == "REJECT"
            d[f"label_{seat}"] = v.get("r3_label") or NOVERDICT
            d[f"vote_{seat}"] = v.get("vote") or NOVERDICT
            d[f"vote_truncated_{seat}"] = _int(v.get("truncated"))
            d[f"vote_len_{seat}"] = _int(v.get("output_len"))
            und = v.get("stake_undermined")
            if seat in STAKE_SEATS and und not in (None, ""):
                d[f"undermined_{seat}"] = bool(int(und))
    return list(debates.values())


def grip_population(debates: Sequence[dict]) -> list[dict]:
    """The sibling's filter: both moderator verdicts committal."""
    return [d for d in debates if d["syn_ok"] is not None and d["s2_ok"] is not None]


def load_votes(path: Path, *, model: Optional[str] = None) -> list[dict]:
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if model and r.get("model") != model:
                continue
            out.append({
                "item": r["item_id"], "arm": r["arm"], "role": r["role_id"],
                "stake": r.get("role_stake", ""),
                "sample_idx": int(r["sample_idx"]),
                "gold_type": r["gold_type"], "region": r["region"],
                "gold": r["gold_verdict"],
                "reject": int(r.get("reject") or 0),
                # None when undefined -- EXCLUDED from G3, never coded 0
                "undermined": _int(r.get("stake_undermined")),
                "objected": int(r.get("objected_r3") or 0),
                "r3_label": r.get("r3_label", ""), "vote": r.get("vote", ""),
                "r0_verdict": r.get("r0_verdict", ""),
                "r1_verdict": r.get("r1_verdict", ""),
                "r2_verdict": r.get("r2_verdict", ""),
                "group_verdict": r.get("group_verdict", ""),
                "truncated": _int(r.get("truncated")),
                "output_len": _int(r.get("output_len")),
            })
    return out


def _perm_str(perm: Sequence[int]) -> str:
    return "".join(str(p) for p in perm)


def build_views(debates: Sequence[dict],
                games: Optional[Sequence[Game]] = None) -> dict[tuple, ItemView]:
    """Re-render every (game_id, sample_idx) from the pinned data and refuse
    on any disagreement with the CSV (permutation, names, gold): the payoffs
    behind a verdict must be the ones the model saw."""
    games = games if games is not None else load_games()
    by_id = {g.game_id: g for g in games}
    views: dict[tuple, ItemView] = {}
    for d in debates:
        key = (d["item"], d["sample_idx"])   # what vote rows carry; rendered from game_id
        if key in views:
            continue
        g = by_id.get(d["game_id"])
        if g is None:
            raise ValueError(f"game {d['game_id']!r} in the CSV is not in the pinned data")
        v = render_item(g, d["sample_idx"])
        got = (_perm_str(v.row_perm), _perm_str(v.col_perm), v.row_name,
               v.col_name, v.gold_verdict)
        want = (d["row_perm"], d["col_perm"], d["row_name"], d["col_name"], d["gold"])
        if got != want:
            raise ValueError(f"rendering mismatch on {key}: data gives {got}, "
                             f"CSV says {want}; the CSV was not produced from "
                             f"this data / renderer")
        views[key] = v
    return views


# ---------------------------------------------------------------------------
# Item-clustered statistics (CI machinery imported from the sibling)
# ---------------------------------------------------------------------------


def clustered_mean(recs: Sequence[dict], key: Callable[[dict], Optional[int]], *,
                   draws: int, seed: int) -> dict:
    by_item: dict[str, list] = defaultdict(list)
    for r in recs:
        v = key(r)
        if v is None:
            continue
        by_item[str(r["item"])].append(int(v))
    n = sum(len(v) for v in by_item.values())
    if not n:
        return {"n": 0, "n_items": 0, "point": None, "lo": None, "hi": None}

    def stat(xs: list) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    p, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed)
    return {"n": n, "n_items": len(by_item), "point": p, "lo": lo, "hi": hi}


def _sliced(recs: Sequence[dict], key, *, draws: int, seed: int) -> dict:
    """all / by_region / by_gold_type, every cell a clustered mean."""
    return {
        "all": clustered_mean(recs, key, draws=draws, seed=seed),
        "by_region": {rg: clustered_mean([r for r in recs if r["region"] == rg], key,
                                         draws=draws, seed=seed) for rg in REGIONS},
        "by_gold_type": {gt: clustered_mean([r for r in recs if r["gold_type"] == gt],
                                            key, draws=draws, seed=seed)
                         for gt in GOLD_TYPES},
    }


def paired_delta(by_a: dict, by_b: dict, stat: Callable[[list], Optional[float]], *,
                 draws: int, seed: int) -> dict:
    """b - a on the items both carry, one item resample per draw evaluated
    on BOTH sides (analyze_stake_grip.paired_g3_delta's scheme)."""
    ids = sorted(set(by_a) & set(by_b))
    if not ids:
        return {"n_shared_items": 0, "a": None, "b": None, "delta": None,
                "lo": None, "hi": None, "positive": False}

    def on(by, sel):
        return stat([x for i in sel for x in by.get(i, [])])

    rng = random.Random(seed)
    pa, pb = on(by_a, ids), on(by_b, ids)
    point = None if pa is None or pb is None else pb - pa
    diffs = []
    for _ in range(draws):
        sel = [ids[rng.randrange(len(ids))] for _ in ids]
        ga, gb = on(by_a, sel), on(by_b, sel)
        if ga is not None and gb is not None:
            diffs.append(gb - ga)
    diffs.sort()
    lo = hi = None
    if diffs:
        lo = diffs[int(0.025 * len(diffs))]
        hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]
    return {"n_shared_items": len(ids), "n_boot_valid": len(diffs), "a": pa, "b": pb,
            "delta": point, "lo": lo, "hi": hi,
            "positive": bool(point is not None and lo is not None and lo > 0)}


# ---------------------------------------------------------------------------
# Role-lock
# ---------------------------------------------------------------------------


def _favour_records(debates: Sequence[dict], views: dict, round_name: str) -> list[dict]:
    recs = []
    for d in debates:
        v = views[(d["item"], d["sample_idx"])]
        for seat in STAKE_SEATS:
            verdict = d[round_name][seat]
            recs.append({"item": d["item"], "seat": seat, "verdict": verdict,
                         "fr": favours(v, verdict, "row"),
                         "fc": favours(v, verdict, "col"),
                         "region": d["region"], "gold_type": d["gold_type"]})
    return recs


def _lock_block(recs: Sequence[dict], *, draws: int, seed: int) -> dict:
    by_item: dict[str, list] = defaultdict(list)
    for x in recs:
        by_item[x["item"]].append(x)

    def rate(xs, seat, k):
        vals = [x[k] for x in xs if x["seat"] == seat and x[k] is not None]
        return sum(vals) / len(vals) if vals else None

    def pull_row(xs):
        a, b = rate(xs, ROW_SEAT, "fr"), rate(xs, COL_SEAT, "fr")
        return None if a is None or b is None else a - b

    def pull_col(xs):
        c, e = rate(xs, COL_SEAT, "fc"), rate(xs, ROW_SEAT, "fc")
        return None if c is None or e is None else c - e

    def signed(xs):
        r, c = pull_row(xs), pull_col(xs)
        return None if r is None or c is None else (r + c) / 2

    def ci(stat):
        p, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed) if by_item else (None, None, None)
        return {"point": p, "lo": lo, "hi": hi}

    seats = {}
    for seat in STAKE_SEATS:
        sv = [x for x in recs if x["seat"] == seat]
        n_plan = sum(1 for x in sv if x["fr"] is not None)
        seats[seat] = {
            "n": len(sv), "n_plan": n_plan,
            "p_favours_row": rate(recs, seat, "fr"),
            "p_favours_col": rate(recs, seat, "fc"),
            "p_no_unique_plan": (sum(1 for x in sv if x["verdict"] == NO_UNIQUE_PLAN)
                                 / len(sv) if sv else None),
            "p_noverdict": (sum(1 for x in sv if x["verdict"] not in VERDICT_SET)
                            / len(sv) if sv else None),
        }
    s = ci(signed)
    return {"n_debates": len(recs) // len(STAKE_SEATS), "n_items": len(by_item),
            "seats": seats, "pull_row": ci(pull_row), "pull_col": ci(pull_col),
            "signed": s, "role_lock": None if s["point"] is None else abs(s["point"])}


def role_lock(debates: Sequence[dict], views: dict, *, round_name: str = "r0",
              draws: int, seed: int) -> dict:
    recs = _favour_records(debates, views, round_name)
    return {
        "round": round_name, "favours_min_payoff": FAVOUR_MIN_PAYOFF,
        "all": _lock_block(recs, draws=draws, seed=seed),
        "by_region": {rg: _lock_block([x for x in recs if x["region"] == rg],
                                      draws=draws, seed=seed) for rg in REGIONS},
        "by_gold_type": {gt: _lock_block([x for x in recs if x["gold_type"] == gt],
                                         draws=draws, seed=seed) for gt in GOLD_TYPES},
    }


# ---------------------------------------------------------------------------
# Nash-pull and accuracy
# ---------------------------------------------------------------------------


def _seat_sort_key(seat: str) -> tuple:
    order = {ROW_SEAT: 0, COL_SEAT: 1}
    return (order.get(seat, 2), seat)


def _seat_records(votes: Sequence[dict], views: dict, seat: str, round_name: str) -> list[dict]:
    out = []
    for v in votes:
        if v["role"] != seat:
            continue
        view = views[(v["item"], v["sample_idx"])]
        verdict = v[f"{round_name}_verdict"]
        out.append({"item": v["item"], "region": v["region"], "gold_type": v["gold_type"],
                    "verdict": verdict, "is_nash": view.is_nash(verdict),
                    "correct": plan_correct(verdict, v["gold"])})
    return out


def nash_pull(votes: Sequence[dict], views: dict, *, draws: int, seed: int) -> dict:
    seats = sorted({v["role"] for v in votes}, key=_seat_sort_key)
    out: dict = {"seats": seats, "rounds": SEAT_ROUNDS, "per_seat": {}}
    for seat in seats:
        out["per_seat"][seat] = {}
        for rnd in SEAT_ROUNDS:
            recs = _seat_records(votes, views, seat, rnd)
            out["per_seat"][seat][rnd] = _sliced(
                recs, lambda r: None if r["is_nash"] is None else int(r["is_nash"]),
                draws=draws, seed=seed)
    return out


def accuracy_tables(debates: Sequence[dict], votes: Sequence[dict], views: dict, *,
                    draws: int, seed: int) -> dict:
    seats = sorted({v["role"] for v in votes}, key=_seat_sort_key)
    group = {
        "final": _sliced(debates, lambda d: int(bool(d["correct"])), draws=draws, seed=seed),
        "synthesis": _sliced(debates, lambda d: int(bool(d["syn_correct"])),
                             draws=draws, seed=seed),
        "verdict_counts": dict(Counter(d["verdict"] for d in debates)),
        "noncommittal_share": (sum(1 for d in debates if d["verdict"] not in VERDICT_SET)
                               / len(debates) if debates else None),
    }
    per_seat = {}
    for seat in seats:
        per_seat[seat] = {rnd: _sliced(_seat_records(votes, views, seat, rnd),
                                       lambda r: int(bool(r["correct"])),
                                       draws=draws, seed=seed) for rnd in SEAT_ROUNDS}
    cells_by_type = Counter(d["gold_type"] for d in debates)
    games_by_type = Counter()
    for gt, gid in {(d["gold_type"], d["game_id"]) for d in debates}:
        games_by_type[gt] += 1
    return {"seats": seats, "group": group, "per_seat": per_seat,
            "cells_by_gold_type": dict(cells_by_type),
            "games_by_gold_type": dict(games_by_type)}


# ---------------------------------------------------------------------------
# Grip (formulas copy-checked against analyze_dilemma_grip)
# ---------------------------------------------------------------------------


def mislocalised(d: dict) -> bool:
    return any(d.get(f"obj_{s}") and not d.get(f"undermined_{s}", True)
               for s in STAKE_SEATS)


def third_objected(d: dict) -> bool:
    return bool(d.get(f"obj_{d['third_role_id']}"))


def flagged(d: dict) -> bool:
    """The composite sensor: third-seat objection OR mis-localised objection."""
    return third_objected(d) or mislocalised(d)


def stake_concentration(votes: Sequence[dict], *, draws: int, seed: int) -> dict:
    sv_all = [v for v in votes if v["role"] in STAKE_SEATS]
    sv = [v for v in sv_all if v["undermined"] is not None]
    by_item: dict[str, list] = defaultdict(list)
    for v in sv:
        by_item[v["item"]].append(v)

    def stat(vs):
        u = [v["reject"] for v in vs if v["undermined"]]
        n = [v["reject"] for v in vs if not v["undermined"]]
        if not u or not n:
            return None
        return sum(u) / len(u) - sum(n) / len(n)

    p, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed) if by_item else (None, None, None)
    n_und = sum(v["undermined"] for v in sv)
    return {
        "n_votes": len(sv), "n_excluded_undefined": len(sv_all) - len(sv),
        "n_undermined": n_und,
        "reject_rate_undermined": (sum(v["reject"] for v in sv if v["undermined"])
                                   / max(1, n_und)),
        "reject_rate_not_undermined": (
            sum(v["reject"] for v in sv if not v["undermined"])
            / max(1, sum(1 for v in sv if not v["undermined"]))),
        "delta": p, "lo": lo, "hi": hi,
    }


def grip(debates: Sequence[dict], votes: Sequence[dict], *, draws: int = 2000,
         seed: int = 7) -> dict:
    """G1/G2/G3 on the grip population (committal moderator verdicts)."""
    pop = grip_population(debates)
    n = len(pop)
    fire = clustered_mean(pop, lambda d: int(flagged(d)), draws=draws, seed=seed)
    rej = clustered_mean(votes, lambda v: v["reject"], draws=draws, seed=seed + 1)
    conc = stake_concentration(votes, draws=draws, seed=seed + 2)
    seats = sorted({v["role"] for v in votes}, key=_seat_sort_key)
    seat = {}
    for s in seats:
        sv = [v for v in votes if v["role"] == s]
        und = [v["undermined"] for v in sv if v["undermined"] is not None]
        seat[s] = {
            "n": len(sv),
            "objected_r3": sum(v["objected"] for v in sv) / len(sv) if sv else None,
            "reject_r4": sum(v["reject"] for v in sv) / len(sv) if sv else None,
            "undermined": (sum(und) / len(und) if und and s in STAKE_SEATS else None),
            "n_undermined_defined": len(und) if s in STAKE_SEATS else None,
        }
    g1 = fire["point"] is not None and fire["point"] <= G1_MAX_FIRE
    g2 = rej["point"] is not None and rej["point"] >= G2_MIN_REJECT
    g3 = (conc["delta"] is not None and conc["delta"] >= G3_MIN_CONC
          and conc["lo"] is not None and conc["lo"] > 0)
    return {
        "n_debates": n, "n_items": len({d["item"] for d in pop}),
        "n_excluded_noncommittal": len(debates) - n, "n_votes": len(votes),
        "G1_fire_rate": fire["point"], "G1_ci": fire,
        "G2_reject_share": rej["point"], "G2_ci": rej,
        "G3_stake_concentration": conc, "per_seat": seat,
        "third_objected_rate": sum(third_objected(d) for d in pop) / n if n else None,
        "mislocalised_rate": sum(mislocalised(d) for d in pop) / n if n else None,
        "s2_accuracy": sum(d["s2_ok"] for d in pop) / n if n else None,
        "criteria": {"G1": g1, "G2": g2, "G3": g3},
        "grip": bool(g1 and g2 and g3),
    }


# ---------------------------------------------------------------------------
# Stake-blind flag concentration (analyze_topology_2x2's counter)
# ---------------------------------------------------------------------------


def fired(d: dict) -> bool:
    return d["n_objectors"] >= FIRE_MIN_OBJECTORS


def _lift(pop: Sequence[dict], *, draws: int, seed: int) -> dict:
    fl = [d for d in pop if fired(d)]
    un = [d for d in pop if not fired(d)]
    by_item: dict[str, list] = defaultdict(list)
    for d in pop:
        by_item[d["item"]].append(d)

    def stat(ds):
        f = [d for d in ds if fired(d)]
        u = [d for d in ds if not fired(d)]
        if not f or not u:
            return None
        return ((1 - sum(bool(d["s2_ok"]) for d in f) / len(f))
                - (1 - sum(bool(d["s2_ok"]) for d in u) / len(u)))

    p, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed) if by_item else (None, None, None)
    return {
        "n_codable": len(pop), "n_fired": len(fl), "powered": len(fl) >= MIN_FIRED,
        "p_wrong_fired": (1 - sum(bool(d["s2_ok"]) for d in fl) / len(fl)) if fl else None,
        "p_wrong_unfired": (1 - sum(bool(d["s2_ok"]) for d in un) / len(un)) if un else None,
        "delta": p, "lo": lo, "hi": hi,
    }


def load_solo_rows(path: Path) -> list[dict]:
    """An independent single-agent run's verdicts on the same games: any CSV
    with item_id, verdict, gold_verdict (scored by plan_correct)."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ok = plan_correct(r.get("verdict", ""), r.get("gold_verdict", ""))
            if ok is None:
                continue
            out.append({"item": r["item_id"], "wrong": int(not ok)})
    return out


def transfer_to_solo(debates: Sequence[dict], solo: Sequence[dict], *,
                     draws: int, seed: int) -> dict:
    """An item is flagged when a majority of its cells fired; the lift is
    P(solo wrong | flagged item) - P(solo wrong | not), item bootstrap."""
    fl_c: dict[str, int] = defaultdict(int)
    k_c: dict[str, int] = defaultdict(int)
    for d in debates:
        fl_c[d["item"]] += int(fired(d))
        k_c[d["item"]] += 1
    wrong: dict[str, list] = defaultdict(list)
    for s in solo:
        if s["item"] in k_c:
            wrong[s["item"]].append(s["wrong"])
    recs = [{"item": i, "w": sum(w) / len(w), "f": int(fl_c[i] * 2 > k_c[i])}
            for i, w in wrong.items()]
    by_item = {r["item"]: [r] for r in recs}

    def stat(rs):
        f = [x for x in rs if x["f"]]
        u = [x for x in rs if not x["f"]]
        if not f or not u:
            return None
        return sum(x["w"] for x in f) / len(f) - sum(x["w"] for x in u) / len(u)

    p, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed) if by_item else (None, None, None)
    tf = [x for x in recs if x["f"]]
    tu = [x for x in recs if not x["f"]]
    return {"n_items": len(recs), "n_flagged_items": len(tf),
            "p_wrong_flagged": sum(x["w"] for x in tf) / len(tf) if tf else None,
            "p_wrong_unflagged": sum(x["w"] for x in tu) / len(tu) if tu else None,
            "lift": p, "lo": lo, "hi": hi}


def flag_concentration(debates: Sequence[dict], *, draws: int, seed: int,
                       solo: Optional[Sequence[dict]] = None) -> dict:
    codable = [d for d in debates if d["s2_ok"] is not None]
    out = {
        "counter": f"n_objectors >= {FIRE_MIN_OBJECTORS}", "min_fired": MIN_FIRED,
        "fire_rate": _sliced(debates, lambda d: int(fired(d)), draws=draws, seed=seed),
        "lift": {"all": _lift(codable, draws=draws, seed=seed + 1),
                 "by_region": {rg: _lift([d for d in codable if d["region"] == rg],
                                         draws=draws, seed=seed + 1) for rg in REGIONS}},
    }
    if solo is not None:
        out["transfer_solo"] = transfer_to_solo(debates, solo, draws=draws, seed=seed + 2)
    return out


# ---------------------------------------------------------------------------
# Planner-vs-neutral contrast on the value region (paired on items)
# ---------------------------------------------------------------------------


def _by_item(recs: Sequence[dict], key) -> dict:
    by: dict[str, list] = defaultdict(list)
    for r in recs:
        v = key(r)
        if v is not None:
            by[r["item"]].append(v)
    return by


def _mean(xs: list) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def value_region_contrast(a: dict, b: dict, *, draws: int, seed: int) -> dict:
    """b minus a on the value-region items both arms carry. When exactly one
    side is the planner arm, b is the planner (so every delta reads planner
    minus neutral); otherwise the order given is kept and labelled."""
    if a["third_seat"] == THIRD_PLANNER and b["third_seat"] != THIRD_PLANNER:
        a, b = b, a
    da = [d for d in a["debates"] if d["region"] == REGION_VALUE]
    db = [d for d in b["debates"] if d["region"] == REGION_VALUE]
    va = [v for v in a["votes"] if v["region"] == REGION_VALUE]
    vb = [v for v in b["votes"] if v["region"] == REGION_VALUE]

    def third_recs(side, votes, rnd):
        return _seat_records(votes, side["views"], side["third_role_id"], rnd)

    stats: dict = {}
    stats["group_accuracy"] = paired_delta(
        _by_item(da, lambda d: int(bool(d["correct"]))),
        _by_item(db, lambda d: int(bool(d["correct"]))), _mean, draws=draws, seed=seed)
    stats["synthesis_accuracy"] = paired_delta(
        _by_item(da, lambda d: int(bool(d["syn_correct"]))),
        _by_item(db, lambda d: int(bool(d["syn_correct"]))), _mean, draws=draws, seed=seed + 1)
    for i, rnd in enumerate(SEAT_ROUNDS):
        stats[f"third_{rnd}_accuracy"] = paired_delta(
            _by_item(third_recs(a, va, rnd), lambda r: int(bool(r["correct"]))),
            _by_item(third_recs(b, vb, rnd), lambda r: int(bool(r["correct"]))),
            _mean, draws=draws, seed=seed + 2 + i)
    stats["third_r0_nash_pull"] = paired_delta(
        _by_item(third_recs(a, va, "r0"), lambda r: None if r["is_nash"] is None else int(r["is_nash"])),
        _by_item(third_recs(b, vb, "r0"), lambda r: None if r["is_nash"] is None else int(r["is_nash"])),
        _mean, draws=draws, seed=seed + 4)
    stats["fire_rate"] = paired_delta(
        _by_item(da, lambda d: int(fired(d))), _by_item(db, lambda d: int(fired(d))),
        _mean, draws=draws, seed=seed + 5)

    def g3_stat(vs):
        u = [v["reject"] for v in vs if v["undermined"]]
        n = [v["reject"] for v in vs if not v["undermined"]]
        return None if not u or not n else sum(u) / len(u) - sum(n) / len(n)

    def stake_by_item(votes):
        by: dict[str, list] = defaultdict(list)
        for v in votes:
            if v["role"] in STAKE_SEATS and v["undermined"] is not None:
                by[v["item"]].append(v)
        return by

    stats["G3_stake_concentration"] = paired_delta(
        stake_by_item(va), stake_by_item(vb), g3_stat, draws=draws, seed=seed + 6)
    return {"a": {"tag": a["tag"], "third_seat": a["third_seat"], "n_value_cells": len(da)},
            "b": {"tag": b["tag"], "third_seat": b["third_seat"], "n_value_cells": len(db)},
            "direction": f"{b['tag']} minus {a['tag']}", "region": REGION_VALUE,
            "stats": stats}


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_CACHE_RE = re.compile(
    rf"^cgd_(?P<model>.+?)_{re.escape(SCAFFOLD_NAME)}_t(?P<cap>\d+)_"
    rf"(?P<arm>{'|'.join(re.escape(a) for a in ARM_OF.values())})_"
    rf"(?P<item>[^_]+)_(?P<idx>\d{{2}})_(?P<round>{'|'.join(ROUNDS)})_"
    rf"(?P<role>.+)\.json$")


def index_cache(out_dir: Path) -> dict[tuple, Path]:
    """One pass over the cache directory: (model, arm, item, idx, round,
    role) -> the record at the LARGEST token cap."""
    best: dict[tuple, tuple[int, Path]] = {}
    if not out_dir.is_dir():
        return {}
    for p in out_dir.iterdir():
        m = _CACHE_RE.match(p.name)
        if not m:
            continue
        key = (m["model"], m["arm"], m["item"], int(m["idx"]), m["round"], m["role"])
        cap = int(m["cap"])
        if key not in best or cap > best[key][0]:
            best[key] = (cap, p)
    return {k: v[1] for k, v in best.items()}


def guard_token(rnd: str, verdict) -> str:
    """The CSV's token for round ``rnd`` if it is one of that round's
    COMMITTAL tokens, else NOVERDICT (module docstring, GUARD). PLAN rounds:
    the five PLAN / NO_UNIQUE_PLAN tokens; r3_label: ACCEPT /
    ACCEPT_WITH_MODIFICATION / REJECT; r4_vote: ACCEPT / REJECT."""
    v = (verdict or "").strip()
    return v if v in committal_tokens(ROUND_INSTRUMENT[rnd]) else NOVERDICT


def guard_rows_from_csv(debates: Sequence[dict]) -> list[dict]:
    """One guard row per call (17 per cell), shaped for tmg_games.parse_guard
    and grouped like the runner's call_guard_row: scaffold/round:role_id."""
    rows = []
    for d in debates:
        mod_rid = moderator_role_id(d["moderator_model"], d["third_seat"])

        def add(rnd, role, model, verdict, truncated, output_len):
            rows.append({
                "model": model, "scaffold": f"{d['scaffold']}/{rnd}:{role}",
                "arm": d["arm"], "verdict": guard_token(rnd, verdict),
                "truncated": truncated, "output_len": output_len,
                "truncation_known": truncated is not None,
                "item": d["item"], "sample_idx": d["sample_idx"], "round": rnd,
                "role_id": role,
            })

        for seat in d["seats"]:
            for rnd in ("r0", "r1", "r2"):
                add(rnd, seat, d["model"], d[rnd][seat], None, None)
            add("r3_label", seat, d["model"], d.get(f"label_{seat}"), None, None)
            add("r4_vote", seat, d["model"], d.get(f"vote_{seat}"),
                d.get(f"vote_truncated_{seat}"), d.get(f"vote_len_{seat}"))
        add("synthesis", mod_rid, d["moderator_model"], d["synthesis_verdict"],
            None, d["synthesis_len"])
        add("integration", mod_rid, d["moderator_model"], d["verdict"],
            d["truncated"], d["output_len"])
    return rows


def enrich_guard_from_cache(grows: list[dict], out_dir: Path) -> dict:
    """Fill truncated / output_len from the call cache where a record is
    found by name. Never overrides a truncation flag the CSV already has."""
    idx = index_cache(out_dir)
    found = missing = 0
    for g in grows:
        key = (_safe(g["model"]), g["arm"], _safe(g["item"]), g["sample_idx"],
               g["round"], _safe(g["role_id"]))
        p = idx.get(key)
        if p is None:
            missing += 1
            continue
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            missing += 1
            continue
        found += 1
        if g["truncated"] is None:
            g["truncated"] = int(rec.get("finish_reason", "") in ("max_tokens", "length", "MAX_TOKENS"))
            g["truncation_known"] = True
        if g["output_len"] is None:
            g["output_len"] = len(rec.get("output") or "")
    return {"cache_dir": str(out_dir), "n_found": found, "n_missing": missing}


def guard_report(grows: Sequence[dict], coverage: Optional[dict] = None) -> dict:
    rep = parse_guard([{**g, "truncated": g["truncated"] or 0,
                        "output_len": g["output_len"] or 0} for g in grows])
    rep["n_calls"] = len(grows)
    rep["n_truncation_known"] = sum(1 for g in grows if g["truncation_known"])
    rep["n_noverdict"] = sum(1 for g in grows if g["verdict"] == NOVERDICT)
    rep["n_truncated"] = sum(1 for g in grows if g["truncated"])
    rep["cache"] = coverage
    return rep


# ---------------------------------------------------------------------------
# Per-arm analysis
# ---------------------------------------------------------------------------


def load_arm(rows_path: Path, votes_path: Path, *, tag: str, model: Optional[str] = None,
             games: Optional[Sequence[Game]] = None) -> dict:
    debates = load_debates(rows_path, votes_path, model=model)
    votes = load_votes(votes_path, model=model)
    if not debates:
        return {"tag": tag, "debates": [], "votes": votes, "views": {},
                "third_seat": None, "third_role_id": None}
    third_seats = {d["third_seat"] for d in debates}
    third_ids = {d["third_role_id"] for d in debates}
    if len(third_seats) != 1 or len(third_ids) != 1:
        raise ValueError(f"{tag}: one arm per tag expected, found third seats "
                         f"{sorted(third_seats)} / role ids {sorted(third_ids)}")
    return {"tag": tag, "debates": debates, "votes": votes,
            "views": build_views(debates, games),
            "third_seat": next(iter(third_seats)),
            "third_role_id": next(iter(third_ids))}


def analyse_arm(arm: dict, *, cache_dir: Optional[Path], solo: Optional[Sequence[dict]] = None,
                draws: int, seed: int) -> dict:
    """The guard first; on failure the result carries ONLY the guard."""
    debates, votes, views = arm["debates"], arm["votes"], arm["views"]
    grows = guard_rows_from_csv(debates)
    cov = enrich_guard_from_cache(grows, cache_dir) if cache_dir is not None else None
    guard = guard_report(grows, cov)
    res = {
        "tag": arm["tag"], "third_seat": arm["third_seat"],
        "third_role_id": arm["third_role_id"],
        "arm": next(iter({d["arm"] for d in debates}), None),
        "models": sorted({d["model"] for d in debates}),
        "n_debates": len(debates), "n_items": len({d["item"] for d in debates}),
        "n_votes": len(votes), "guard": guard, "refused": not guard["pass"],
    }
    if not guard["pass"]:
        return res
    res["role_lock"] = role_lock(debates, views, draws=draws, seed=seed)
    res["nash_pull"] = nash_pull(votes, views, draws=draws, seed=seed + 10)
    res["accuracy"] = accuracy_tables(debates, votes, views, draws=draws, seed=seed + 20)
    res["grip"] = grip(debates, votes, draws=draws, seed=seed + 30)
    res["flag_concentration"] = flag_concentration(debates, draws=draws, seed=seed + 40,
                                                   solo=solo)
    return res


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _f(x, fmt=".3f") -> str:
    return "n/a" if x is None else format(x, fmt)


def _ci(c: dict, fmt=".3f") -> str:
    if not c or not c.get("n"):
        return f"{'n/a':>7}   {'':>16}  n=0"
    return (f"{_f(c['point'], fmt):>7}   [{_f(c['lo'], fmt):>6}, {_f(c['hi'], fmt):>6}]"
            f"  n={c['n']}")


def _ci3(c: dict, fmt="+.3f") -> str:
    return f"{_f(c['point'], fmt)} [{_f(c['lo'], fmt)}, {_f(c['hi'], fmt)}]"


def print_guard(rep: dict) -> None:
    print_parse_guard(rep)
    print(f"  {rep['n_calls']} calls reconstructed from the CSVs: NOVERDICT on "
          f"{rep['n_noverdict']}, truncated on {rep['n_truncated']} of "
          f"{rep['n_truncation_known']} with a known finish_reason "
          f"({rep['n_calls'] - rep['n_truncation_known']} unknown; the chars column "
          f"counts an unknown length as 0)")
    if rep.get("cache"):
        c = rep["cache"]
        print(f"  cache records found for {c['n_found']} calls, missing for "
              f"{c['n_missing']} ({c['cache_dir']})")


def print_role_lock(rl: dict) -> None:
    a = rl["all"]
    print(f"\nROLE-LOCK ({rl['round']}; favours = the plan gives that player payoff "
          f">= {rl['favours_min_payoff']})")
    print(f"  {'seat':<20}{'n':>5}{'n_plan':>8}{'P(fav row)':>12}{'P(fav col)':>12}"
          f"{'P(NO_UNIQUE)':>14}{'P(NOVERDICT)':>14}")
    for seat, s in a["seats"].items():
        print(f"  {seat:<20}{s['n']:>5}{s['n_plan']:>8}{_f(s['p_favours_row']):>12}"
              f"{_f(s['p_favours_col']):>12}{_f(s['p_no_unique_plan']):>14}"
              f"{_f(s['p_noverdict']):>14}")
    print(f"  pull toward row  P(fav row) R-adv minus C-adv   {_ci3(a['pull_row'])}")
    print(f"  pull toward col  P(fav col) C-adv minus R-adv   {_ci3(a['pull_col'])}")
    print(f"  ROLE-LOCK = |mean of the two|                   {_f(a['role_lock'])}"
          f"   signed CI [{_f(a['signed']['lo'], '+.3f')}, {_f(a['signed']['hi'], '+.3f')}]")
    print("  by region / gold_type (role-lock, signed CI, n_items):")
    for name, blk in list(rl["by_region"].items()) + list(rl["by_gold_type"].items()):
        s = blk["signed"]
        print(f"    {name:<32}{_f(blk['role_lock']):>7}   [{_f(s['lo'], '+.3f'):>6}, "
              f"{_f(s['hi'], '+.3f'):>6}]  n_items={blk['n_items']}")


def print_nash_pull(np_: dict) -> None:
    print("\nNASH-PULL  P(verdict is a pure Nash cell | verdict is a PLAN cell), "
          "item-clustered 95% CI")
    for seat in np_["seats"]:
        for rnd in np_["rounds"]:
            sl = np_["per_seat"][seat][rnd]
            print(f"  {seat:<20}{rnd:<4}{'all':<32}{_ci(sl['all'])}")
            if rnd != "r0":
                continue
            for rg, c in sl["by_region"].items():
                print(f"  {'':<20}{'':<4}{rg:<32}{_ci(c)}")
            for gt, c in sl["by_gold_type"].items():
                print(f"  {'':<20}{'':<4}{gt:<32}{_ci(c)}")


def print_accuracy(acc: dict) -> None:
    g = acc["group"]
    print("\nACCURACY vs gold (non-committal = wrong), item-clustered 95% CI")
    print(f"  group final      overall   {_ci(g['final']['all'])}")
    print(f"  group synthesis  overall   {_ci(g['synthesis']['all'])}")
    for seat in acc["seats"]:
        for rnd in SEAT_ROUNDS:
            print(f"  {seat:<16} {rnd:<9} {_ci(acc['per_seat'][seat][rnd]['all'])}")
    print("  by region:")
    for rg in REGIONS:
        print(f"    {rg:<16} group final {_ci(g['final']['by_region'][rg])}")
        for seat in acc["seats"]:
            print(f"    {'':<16} {seat:<20} r0 {_ci(acc['per_seat'][seat]['r0']['by_region'][rg])}")
    print("  by gold_type (the extent table): group final [CI]; then synthesis, "
          "seat r0 points")
    hdr = "".join(f"{SEAT_SHORT.get(s, s[:7]) + ' r0':>11}" for s in acc["seats"])
    print(f"    {'gold_type':<32}{'games':>6}{'cells':>6}  {'group final':<30}{'synth':>7}{hdr}")
    for gt in GOLD_TYPES:
        c = g["final"]["by_gold_type"][gt]
        seats = "".join(f"{_f(acc['per_seat'][s]['r0']['by_gold_type'][gt]['point']):>11}"
                        for s in acc["seats"])
        print(f"    {gt:<32}{acc['games_by_gold_type'].get(gt, 0):>6}"
              f"{acc['cells_by_gold_type'].get(gt, 0):>6}  "
              f"{_f(c['point']):>7} [{_f(c['lo']):>5}, {_f(c['hi']):>5}]     "
              f"{_f(g['synthesis']['by_gold_type'][gt]['point']):>7}{seats}")
    print(f"  verdict counts: {g['verdict_counts']}   noncommittal share "
          f"{_f(g['noncommittal_share'])}")


def print_grip(g: dict) -> None:
    print("\nSTAKE-GRIP SCREEN (analyze_dilemma_grip thresholds, player advocates)")
    print(f"  n_debates={g['n_debates']}  n_items={g['n_items']}  n_votes={g['n_votes']}  "
          f"S2 accuracy={_f(g['s2_accuracy'])}  excluded non-committal="
          f"{g['n_excluded_noncommittal']}")
    print(f"  G1 fire rate {_f(g['G1_fire_rate'])} [{_f(g['G1_ci']['lo'])}, "
          f"{_f(g['G1_ci']['hi'])}] (<= {G1_MAX_FIRE}) -> "
          f"{'PASS' if g['criteria']['G1'] else 'fail'}")
    print(f"  G2 reject share {_f(g['G2_reject_share'])} [{_f(g['G2_ci']['lo'])}, "
          f"{_f(g['G2_ci']['hi'])}] (>= {G2_MIN_REJECT}) -> "
          f"{'PASS' if g['criteria']['G2'] else 'fail'}")
    c = g["G3_stake_concentration"]
    print(f"  G3 stake concentration {_f(c['delta'], '+.3f')} [{_f(c['lo'], '+.3f')}, "
          f"{_f(c['hi'], '+.3f')}] (>= +{G3_MIN_CONC}, CI excl. 0) -> "
          f"{'PASS' if g['criteria']['G3'] else 'fail'}")
    print(f"  reject|undermined {_f(c['reject_rate_undermined'])}  "
          f"reject|not_undermined {_f(c['reject_rate_not_undermined'])}  "
          f"stake votes {c['n_votes']} (undermined {c['n_undermined']}; "
          f"{c['n_excluded_undefined']} excluded, undefined)")
    print(f"  third objected {_f(g['third_objected_rate'])}  mislocalised "
          f"{_f(g['mislocalised_rate'])}")
    print(f"  {'seat':<20}{'objected_r3':>12}{'reject_r4':>10}{'undermined':>12}{'n':>6}")
    for s, v in g["per_seat"].items():
        print(f"  {s:<20}{_f(v['objected_r3']):>12}{_f(v['reject_r4']):>10}"
              f"{_f(v['undermined']):>12}{v['n']:>6}")
    print(f"\n  GRIP: {'yes' if g['grip'] else 'no'}")


def print_flag_concentration(fc: dict) -> None:
    print(f"\nSTAKE-BLIND FLAG ({fc['counter']}): fire rate and error lift on the "
          f"group's own verdict (read at >= {fc['min_fired']} fired)")
    print(f"  fire rate  all              {_ci(fc['fire_rate']['all'])}")
    for rg in REGIONS:
        print(f"  fire rate  {rg:<16} {_ci(fc['fire_rate']['by_region'][rg])}")
    l = fc["lift"]["all"]
    print(f"  lift P(wrong|fired) - P(wrong|not)   {_f(l['delta'], '+.3f')} "
          f"[{_f(l['lo'], '+.3f')}, {_f(l['hi'], '+.3f')}]  fired {l['n_fired']}"
          f"{'' if l['powered'] else '  (UNDERPOWERED)'}  "
          f"P(wrong|fired)={_f(l['p_wrong_fired'])} P(wrong|not)={_f(l['p_wrong_unfired'])}")
    for rg in REGIONS:
        l = fc["lift"]["by_region"][rg]
        print(f"    {rg:<16} {_f(l['delta'], '+.3f')} [{_f(l['lo'], '+.3f')}, "
              f"{_f(l['hi'], '+.3f')}]  fired {l['n_fired']}"
              f"{'' if l['powered'] else '  (underpowered)'}")
    if "transfer_solo" in fc:
        t = fc["transfer_solo"]
        print(f"  TRANSFER to solo run: P(solo wrong|flagged item)={_f(t['p_wrong_flagged'])} "
              f"vs not={_f(t['p_wrong_unflagged'])}  lift {_f(t['lift'], '+.3f')} "
              f"[{_f(t['lo'], '+.3f')}, {_f(t['hi'], '+.3f')}]  "
              f"items {t['n_items']} flagged {t['n_flagged_items']}")


def print_contrast(c: dict) -> None:
    print("\n" + "=" * 72)
    print(f"THIRD-SEAT CONTRAST on the {c['region']} region: {c['direction']}  "
          f"({c['b']['third_seat']} minus {c['a']['third_seat']}; paired on items)")
    print("=" * 72)
    print(f"  {'statistic':<26}{'a':>8}{'b':>8}{'delta':>9}{'lo':>9}{'hi':>9}  n_items")
    for name, s in c["stats"].items():
        print(f"  {name:<26}{_f(s['a']):>8}{_f(s['b']):>8}{_f(s['delta'], '+.3f'):>9}"
              f"{_f(s['lo'], '+.3f'):>9}{_f(s['hi'], '+.3f'):>9}  {s['n_shared_items']}"
              f"{'   POSITIVE' if s['positive'] else ''}")


def print_arm_report(res: dict) -> None:
    print("\n" + "=" * 72)
    print(f"GAME GRIP -- tag {res['tag']}  arm {res['arm']}  third seat "
          f"{res['third_seat']} ({res['third_role_id']})  models {res['models']}")
    print(f"  rows={res['n_debates']}  items={res['n_items']}  votes={res['n_votes']}")
    print("=" * 72)
    print_guard(res["guard"])
    if res["refused"]:
        print("\n  *** GUARD FAILED: no role-lock, Nash-pull, accuracy or grip number "
              "is printed for this arm. Raise the failing round's token cap and "
              "re-run. ***")
        return
    print_role_lock(res["role_lock"])
    print_nash_pull(res["nash_pull"])
    print_accuracy(res["accuracy"])
    print_grip(res["grip"])
    print_flag_concentration(res["flag_concentration"])


# ---------------------------------------------------------------------------
# Selftest: synthetic records from the real games, written through the
# runner's CSV columns, values hand-computed by an independent path
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _wrong_plan(view: ItemView) -> str:
    if view.row_best_verdict != view.gold_verdict:
        return view.row_best_verdict
    if view.col_best_verdict != view.gold_verdict:
        return view.col_best_verdict
    return next(p for p in PLAN_VERDICTS if p != view.gold_verdict)


def _synthetic_arm(games: Sequence[Game], third_seat: str, *, wrong_every: Optional[int],
                   model: str = "stub-model", extra_value_sample: bool = True,
                   noverdict_r0_every: Optional[int] = None) -> tuple[list[dict], list[dict]]:
    """Planted behaviour (see the selftest for the values it implies):
      R-advocate r0 = row player's best cell, C-advocate r0 = column player's
      best cell, third r0 = gold; every seat's r2 = gold; synthesis = gold;
      group = gold except on every `wrong_every`-th game (id order), where
      it is a wrong PLAN. Objections: R iff wrong, C iff undermined, third
      iff wrong. Rejects: advocates iff undermined, third iff wrong."""
    third_id = THIRD_ROLE[third_seat].role_id
    arm = ARM_OF[third_seat]
    rows, votes = [], []
    ordered = sorted(games, key=lambda g: g.game_id)
    for gi, g in enumerate(ordered):
        samples = (0, 1) if (extra_value_sample and g.region == REGION_VALUE) else (0,)
        wrong = wrong_every is not None and gi % wrong_every == wrong_every - 1
        for idx in samples:
            view = render_item(g, idx)
            item = GameItem(view)
            gold = view.gold_verdict
            verdict = _wrong_plan(view) if wrong else gold
            r0 = {ROW_SEAT: view.row_best_verdict, COL_SEAT: view.col_best_verdict,
                  third_id: gold}
            if noverdict_r0_every is not None and gi % noverdict_r0_every == 0:
                r0[ROW_SEAT] = NOVERDICT
            r2 = {s: gold for s in r0}
            und = {s: stake_undermined(s, verdict, item) for s in r0}
            objected = {ROW_SEAT: wrong, COL_SEAT: bool(und[COL_SEAT]), third_id: wrong}
            rejects = {ROW_SEAT: bool(und[ROW_SEAT]), COL_SEAT: bool(und[COL_SEAT]),
                       third_id: wrong}
            labels = {s: ("ACCEPT_WITH_MODIFICATION" if objected[s] else "ACCEPT") for s in r0}
            vote = {s: ("REJECT" if rejects[s] else "ACCEPT") for s in r0}
            order = (ROW_SEAT, COL_SEAT, third_id)
            rows.append({
                "model": model, "scaffold": SCAFFOLD_NAME, "arm": arm,
                "third_seat": third_seat, "third_role_id": third_id,
                "item_id": g.game_id, "game_id": g.game_id, "sample_idx": idx,
                "gold_type": g.gold_type, "region": g.region, "gold_verdict": gold,
                "nash_verdicts": "|".join(sorted(view.nash_verdicts)),
                "n_nash": len(view.nash_verdicts),
                "row_perm": _perm_str(view.row_perm), "col_perm": _perm_str(view.col_perm),
                "row_name": view.row_name, "col_name": view.col_name,
                "row_best_verdict": view.row_best_verdict,
                "col_best_verdict": view.col_best_verdict,
                "verdict": verdict, "correct": int(score_plan(verdict, gold)["correct"]),
                "noncommittal": 0, "output_len": 300, "finish_reason": "stop",
                "truncated": 0, "moderator_model": model,
                "synthesis_verdict": gold, "synthesis_len": 250,
                "synthesis_correct": 1, "verdict_revised": int(verdict != gold),
                "r0_verdict_row_advocate": r0[ROW_SEAT],
                "r0_verdict_col_advocate": r0[COL_SEAT], "r0_verdict_third": r0[third_id],
                "r1_verdict_row_advocate": r0[ROW_SEAT],
                "r1_verdict_col_advocate": r0[COL_SEAT], "r1_verdict_third": r0[third_id],
                "r2_verdict_row_advocate": gold, "r2_verdict_col_advocate": gold,
                "r2_verdict_third": gold,
                "r3_labels": "|".join(labels[s] for s in order),
                "n_r3_unparsed": 0,
                "n_objectors": sum(objected.values()),
                "n_accept": sum(1 for s in order if vote[s] == "ACCEPT"),
                "n_reject": sum(1 for s in order if vote[s] == "REJECT"),
                "n_vote_unparsed": 0,
                "unanimous_accept": int(all(vote[s] == "ACCEPT" for s in order)),
                "rejecting_roles": "|".join(s for s in order if vote[s] == "REJECT"),
            })
            for s in order:
                votes.append({
                    "model": model, "scaffold": SCAFFOLD_NAME, "arm": arm,
                    "third_seat": third_seat, "item_id": g.game_id, "sample_idx": idx,
                    "gold_type": g.gold_type, "region": g.region, "gold_verdict": gold,
                    "role_id": s, "role_stake": {ROW_SEAT: "row", COL_SEAT: "col"}.get(s, "none"),
                    "group_verdict": verdict, "synthesis_verdict": gold,
                    "r0_verdict": r0[s], "r1_verdict": r0[s], "r2_verdict": r2[s],
                    "r0_is_nash": "" if view.is_nash(r0[s]) is None else int(view.is_nash(r0[s])),
                    "r2_is_nash": "" if view.is_nash(r2[s]) is None else int(view.is_nash(r2[s])),
                    "r0_correct": int(score_plan(r0[s], gold)["correct"]),
                    "r2_correct": 1,
                    "r3_label": labels[s], "r3_label_parsed": 1,
                    "objected_r3": int(objected[s]), "vote": vote[s],
                    "reject": int(vote[s] == "REJECT"), "vote_parsed": 1,
                    "stake_undermined": "" if und[s] is None else int(und[s]),
                    "finish_reason": "stop", "truncated": 0, "output_len": 120,
                })
    return rows, votes


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    real_tag_files = len(cache_files_with_prefix(TAG_PREFIX))

    check("STAKE_SEATS names the two player advocates; third role ids are the "
          "runner's neutral_reader / plan_mediator",
          STAKE_SEATS == ("player_r_advocate", "player_c_advocate")
          and set(THIRD_ROLE_IDS) == {"neutral_reader", "plan_mediator"})
    check("thresholds are the sibling's (0.50 / 0.05 / +0.20)",
          (G1_MAX_FIRE, G2_MIN_REJECT, G3_MIN_CONC) == (0.50, 0.05, 0.20))

    # ---- scoring: the NO_UNIQUE_PLAN regression guard ----------------------------
    check("REGRESSION GUARD: NO_UNIQUE_PLAN is scored CORRECT against a null gold "
          "(gold NO_UNIQUE_PLAN)",
          plan_correct(NO_UNIQUE_PLAN, NO_UNIQUE_PLAN) is True)
    check("a PLAN against a null gold is wrong; NO_UNIQUE_PLAN against a PLAN gold is wrong",
          plan_correct("PLAN_11", NO_UNIQUE_PLAN) is False
          and plan_correct(NO_UNIQUE_PLAN, "PLAN_12") is False)
    check("NOVERDICT / UNRESOLVED / empty score wrong (in the denominator), and "
          "committal_correct makes them None (the sibling's syn_ok / s2_ok)",
          plan_correct(NOVERDICT, "PLAN_12") is False
          and plan_correct("UNRESOLVED", NO_UNIQUE_PLAN) is False
          and plan_correct("", "PLAN_12") is False
          and committal_correct(NOVERDICT, "PLAN_12") is None
          and committal_correct(NO_UNIQUE_PLAN, NO_UNIQUE_PLAN) is True)

    # ---- guard token coding -------------------------------------------------------
    check("guard_token: committal tokens pass per round; UNRESOLVED, blank, a PLAN "
          "on the vote round and a vote on a PLAN round are NOVERDICT",
          guard_token("r4_vote", "ACCEPT") == "ACCEPT"
          and guard_token("r4_vote", "REJECT") == "REJECT"
          and guard_token("r3_label", "ACCEPT_WITH_MODIFICATION") == "ACCEPT_WITH_MODIFICATION"
          and guard_token("r0", "PLAN_12") == "PLAN_12"
          and guard_token("integration", NO_UNIQUE_PLAN) == NO_UNIQUE_PLAN
          and guard_token("r4_vote", "UNRESOLVED") == NOVERDICT
          and guard_token("r3_label", "UNRESOLVED") == NOVERDICT
          and guard_token("r2", "UNRESOLVED") == NOVERDICT
          and guard_token("r4_vote", "") == NOVERDICT
          and guard_token("r4_vote", None) == NOVERDICT
          and guard_token("r4_vote", "PLAN_12") == NOVERDICT
          and guard_token("r4_vote", "ACCEPT_WITH_MODIFICATION") == NOVERDICT
          and guard_token("r0", "ACCEPT") == NOVERDICT)

    # ---- data and the favours predicate -----------------------------------------
    try:
        games = load_games()
    except Exception as e:  # noqa: BLE001
        check(f"load_games() ({type(e).__name__}: {e})", False)
        print(f"\n{len(fails)} FAILED")
        return 1
    check("144 games load", len(games) == 144)
    pd_game = next(g for g in games if g.gold_type == "DILEMMA_BOTH_PREFER")
    pdv = render_item(pd_game, 0)
    nash = next(iter(pdv.nash_verdicts))
    check("favours on a DILEMMA game: row-best favours row only, col-best favours "
          "col only, gold (3,3) favours both, the Nash cell (2,2) favours neither, "
          "NO_UNIQUE_PLAN / NOVERDICT favour no one (None)",
          favours(pdv, pdv.row_best_verdict, "row") is True
          and favours(pdv, pdv.row_best_verdict, "col") is False
          and favours(pdv, pdv.col_best_verdict, "col") is True
          and favours(pdv, pdv.col_best_verdict, "row") is False
          and favours(pdv, pdv.gold_verdict, "row") is True
          and favours(pdv, pdv.gold_verdict, "col") is True
          and favours(pdv, nash, "row") is False and favours(pdv, nash, "col") is False
          and favours(pdv, NO_UNIQUE_PLAN, "row") is None
          and favours(pdv, NOVERDICT, "col") is None)

    # ---- sensor predicates (as the sibling) ---------------------------------------
    d_third = {"third_role_id": "neutral_reader", "obj_neutral_reader": True}
    d_misloc = {"third_role_id": "plan_mediator", "obj_player_r_advocate": True,
                "undermined_player_r_advocate": False}
    d_clean = {"third_role_id": "neutral_reader", "obj_player_r_advocate": True,
               "undermined_player_r_advocate": True, "obj_neutral_reader": False}
    d_undef = {"third_role_id": "neutral_reader", "obj_player_c_advocate": True}
    check("third-seat objection alone flags (either variant's role id)", flagged(d_third))
    check("mis-localised advocate objection alone flags", flagged(d_misloc))
    check("a correctly-localised objection with no third objection does not flag",
          not flagged(d_clean))
    check("an objection with an UNDEFINED undermined flag is not mis-localised "
          "(the sibling's .get(..., True) default)", not flagged(d_undef))

    # ---- role-lock on two literal hand cases ---------------------------------------
    views = {(pd_game.game_id, 0): pdv}
    base = {"item": pd_game.game_id, "game_id": pd_game.game_id, "sample_idx": 0,
            "region": pd_game.region, "gold_type": pd_game.gold_type}
    locked = [{**base, "r0": {ROW_SEAT: pdv.row_best_verdict, COL_SEAT: pdv.col_best_verdict}}]
    same = [{**base, "r0": {ROW_SEAT: pdv.gold_verdict, COL_SEAT: pdv.gold_verdict}}]
    rl_l = role_lock(locked, views, draws=50, seed=1)["all"]
    rl_s = role_lock(same, views, draws=50, seed=1)["all"]
    check("role-lock is exactly 1 when R names row-best and C names col-best "
          "(pull_row = pull_col = +1)",
          rl_l["role_lock"] == 1.0 and rl_l["pull_row"]["point"] == 1.0
          and rl_l["pull_col"]["point"] == 1.0)
    check("role-lock is exactly 0 when both advocates name the same cell",
          rl_s["role_lock"] == 0.0 and rl_s["pull_row"]["point"] == 0.0)

    # ---- synthetic neutral arm through the runner's CSV columns --------------------
    WRONG_EVERY = 3
    rows_n, votes_n = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY)
    check("synthetic rows / votes carry every ROW_FIELDS / VOTE_FIELDS column used",
          all(k in ROW_FIELDS for k in rows_n[0]) and all(k in VOTE_FIELDS for k in votes_n[0]))
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        _write_csv(tdp / "cg_game_neutral_rows.csv", rows_n, ROW_FIELDS)
        _write_csv(tdp / "cg_game_neutral_votes.csv", votes_n, VOTE_FIELDS)
        arm = load_arm(tdp / "cg_game_neutral_rows.csv", tdp / "cg_game_neutral_votes.csv",
                       tag="cg_game_neutral", games=games)
        n_cells = len(rows_n)
        check(f"load_arm: {n_cells} debates (144 games + a second sample on the 16 "
              f"value games), 144 items, views re-rendered and consistent with the CSV",
              len(arm["debates"]) == n_cells == 160
              and len({d["item"] for d in arm["debates"]}) == 144
              and len(arm["views"]) == 160 and arm["third_role_id"] == "neutral_reader")
        check("load_votes keeps an undefined stake_undermined as None (NO_UNIQUE_PLAN "
              "group verdicts / no-Nash games), never 0",
              any(v["undermined"] is None for v in arm["votes"] if v["role"] in STAKE_SEATS)
              and all(v["undermined"] is None for v in arm["votes"]
                      if v["role"] in STAKE_SEATS and v["group_verdict"] == NO_UNIQUE_PLAN))
        res = analyse_arm(arm, cache_dir=None, draws=300, seed=3)
        check("guard passes on the clean synthetic arm (17 calls per cell, 0 NOVERDICT)",
              res["guard"]["pass"] and res["guard"]["n_calls"] == 17 * n_cells
              and res["guard"]["n_noverdict"] == 0 and not res["refused"])

        # independent expectations straight from the payoff matrices
        exp = {"fr_R": [], "fr_C": [], "fc_R": [], "fc_C": []}
        nash_R: dict[str, list] = defaultdict(list)
        n_wrong = n_wrong_cannot = n_cannot = 0
        for d in arm["debates"]:
            g = next(x for x in games if x.game_id == d["game_id"])
            v = render_item(g, d["sample_idx"])
            rb = next(c for c in ((0, 0), (0, 1), (1, 0), (1, 1)) if g.pA[c[0]][c[1]] == 4)
            cb = next(c for c in ((0, 0), (0, 1), (1, 0), (1, 1)) if g.pB[c[0]][c[1]] == 4)
            exp["fr_R"].append(1)                          # row gets 4 at its best cell
            exp["fc_R"].append(int(g.pB[rb[0]][rb[1]] >= 3))
            exp["fc_C"].append(1)
            exp["fr_C"].append(int(g.pA[cb[0]][cb[1]] >= 3))
            nash_R[g.gold_type].append(int(rb in g.nash_cells))
            if d["verdict"] != d["gold"]:
                n_wrong += 1
                n_wrong_cannot += int(g.region == "cannot_assist")
            n_cannot += int(g.region == "cannot_assist")
        m = lambda xs: sum(xs) / len(xs)  # noqa: E731
        exp_pull_row = m(exp["fr_R"]) - m(exp["fr_C"])
        exp_pull_col = m(exp["fc_C"]) - m(exp["fc_R"])
        rl = res["role_lock"]["all"]
        check(f"role-lock pulls reproduce the payoff-matrix computation "
              f"(row {exp_pull_row:+.3f}, col {exp_pull_col:+.3f})",
              abs(rl["pull_row"]["point"] - exp_pull_row) < 1e-9
              and abs(rl["pull_col"]["point"] - exp_pull_col) < 1e-9
              and abs(rl["role_lock"] - abs((exp_pull_row + exp_pull_col) / 2)) < 1e-9
              and rl["seats"][ROW_SEAT]["n_plan"] == n_cells)
        check("role-lock CI excludes 0 on the locked synthetic seats",
              rl["signed"]["lo"] is not None and rl["signed"]["lo"] > 0)
        npR = res["nash_pull"]["per_seat"][ROW_SEAT]["r0"]["by_gold_type"]
        check("Nash-pull of the R-advocate by gold_type reproduces P(row-best cell is "
              "a pure Nash cell) from the matrices, all seven types",
              all(abs(npR[gt]["point"] - m(nash_R[gt])) < 1e-9 for gt in GOLD_TYPES))
        cyc = {s: res["nash_pull"]["per_seat"][s]["r0"]["by_gold_type"]["CYCLIC_NO_IDEAL"]
               for s in res["nash_pull"]["seats"]}
        check("Nash-pull is 0 by construction on CYCLIC_NO_IDEAL for the advocates "
              "(16 PLAN verdicts each); the third seat's NO_UNIQUE_PLAN verdicts there "
              "are excluded (n=0, not coded 0)",
              all(cyc[s]["point"] == 0.0 and cyc[s]["n"] == 16 for s in STAKE_SEATS)
              and cyc["neutral_reader"]["n"] == 0 and cyc["neutral_reader"]["point"] is None)
        acc = res["accuracy"]
        check("third seat r0 accuracy is exactly 1.0 (it names the gold) and every "
              "seat's r2 accuracy is 1.0",
              acc["per_seat"]["neutral_reader"]["r0"]["all"]["point"] == 1.0
              and all(acc["per_seat"][s]["r2"]["all"]["point"] == 1.0 for s in acc["seats"]))
        check(f"group final accuracy is exactly 1 - {n_wrong}/{n_cells}",
              abs(acc["group"]["final"]["all"]["point"] - (1 - n_wrong / n_cells)) < 1e-9
              and acc["group"]["synthesis"]["all"]["point"] == 1.0)
        check("REGRESSION GUARD on the extent table: the cannot_assist region "
              "(gold NO_UNIQUE_PLAN) scores its NO_UNIQUE_PLAN group verdicts CORRECT "
              f"(accuracy = 1 - {n_wrong_cannot}/{n_cannot})",
              abs(acc["group"]["final"]["by_region"]["cannot_assist"]["point"]
                  - (1 - n_wrong_cannot / n_cannot)) < 1e-9
              and acc["group"]["final"]["by_region"]["cannot_assist"]["point"] > 0.5
              and all(acc["group"]["final"]["by_gold_type"][gt]["point"] > 0.5
                      for gt in ("CYCLIC_NO_IDEAL", "CONFLICT_OF_EQUILIBRIA", "ZERO_SUM")))
        check("extent table has all seven gold types with the registered game counts",
              acc["games_by_gold_type"] == {"INCENTIVE_COMPATIBLE_EFFICIENT": 70, "HARMONY": 27,
                                            "CYCLIC_NO_IDEAL": 16, "CONFLICT_OF_EQUILIBRIA": 9,
                                            "ASSURANCE": 9, "DILEMMA_BOTH_PREFER": 7, "ZERO_SUM": 6})
        gr = res["grip"]
        n_und = sum(1 for v in arm["votes"] if v["undermined"])
        n_rej = sum(v["reject"] for v in arm["votes"])
        check("synthetic design produced undermined advocate votes (else G3 is untestable)",
              n_und > 0)
        check(f"G1 fire rate is exactly the wrong share {n_wrong}/{n_cells} (third objects "
              "iff wrong; R's stake-blind objections are mis-localised only when wrong)",
              abs(gr["G1_fire_rate"] - n_wrong / n_cells) < 1e-9 and gr["criteria"]["G1"])
        check(f"G2 reject share is exactly {n_rej}/{len(arm['votes'])} and clears 0.05",
              abs(gr["G2_reject_share"] - n_rej / len(arm["votes"])) < 1e-9 and gr["criteria"]["G2"])
        c3 = gr["G3_stake_concentration"]
        check("G3 is exactly +1.0 (advocates reject iff undermined), CI excludes 0, "
              "undefined votes excluded not coded 0",
              abs(c3["delta"] - 1.0) < 1e-9 and c3["lo"] > 0 and gr["criteria"]["G3"]
              and c3["n_excluded_undefined"] > 0
              and c3["n_votes"] + c3["n_excluded_undefined"] == 2 * n_cells)
        check("grip is True when all three pass", gr["grip"] is True)
        fc = res["flag_concentration"]
        l = fc["lift"]["all"]
        check(f"stake-blind fire rate is exactly {n_wrong}/{n_cells} and the error lift "
              "is exactly +1.0 with CI excluding 0 (fired iff wrong by construction)",
              abs(fc["fire_rate"]["all"]["point"] - n_wrong / n_cells) < 1e-9
              and abs(l["delta"] - 1.0) < 1e-9 and l["lo"] > 0 and l["n_fired"] == n_wrong)
        check("lift power flag reads >= 60 fired", l["powered"] == (n_wrong >= MIN_FIRED))
        # transfer onto a synthetic solo run: wrong exactly on the wrong games
        solo_path = tdp / "solo.csv"
        wrong_items = {d["item"] for d in arm["debates"] if d["verdict"] != d["gold"]}
        with open(solo_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "verdict", "gold_verdict"])
            w.writeheader()
            for d in arm["debates"]:
                if d["sample_idx"] != 0:
                    continue
                w.writerow({"item_id": d["item"], "gold_verdict": d["gold"],
                            "verdict": _wrong_plan(arm["views"][(d["item"], 0)])
                            if d["item"] in wrong_items else d["gold"]})
        t = flag_concentration(arm["debates"], draws=200, seed=1,
                               solo=load_solo_rows(solo_path))["transfer_solo"]
        check("transfer to a solo run: P(solo wrong | flagged item) = 1, | not = 0, "
              "lift +1.0",
              t["p_wrong_flagged"] == 1.0 and t["p_wrong_unflagged"] == 0.0
              and abs(t["lift"] - 1.0) < 1e-9 and t["n_flagged_items"] == len(wrong_items))
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_arm_report(res)
        out = buf.getvalue()
        check("report prints guard PASSED, ROLE-LOCK, NASH-PULL, extent table, grip "
              "and stake-blind flag sections",
              "guard PASSED" in out and "ROLE-LOCK" in out and "NASH-PULL" in out
              and "extent table" in out and "GRIP: yes" in out and "STAKE-BLIND" in out
              and "GUARD FAILED" not in out)

        # ---- GUARD FAILED path: 10% NOVERDICT on the R-advocate's r0 ---------------
        rows_b, votes_b = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY,
                                         noverdict_r0_every=10)
        _write_csv(tdp / "bad_rows.csv", rows_b, ROW_FIELDS)
        _write_csv(tdp / "bad_votes.csv", votes_b, VOTE_FIELDS)
        bad = load_arm(tdp / "bad_rows.csv", tdp / "bad_votes.csv", tag="bad", games=games)
        res_b = analyse_arm(bad, cache_dir=None, draws=50, seed=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_arm_report(res_b)
        out_b = buf.getvalue()
        check("10% NOVERDICT in one round:seat fails the guard, the result carries no "
              "grip / role-lock / accuracy keys, and the report prints GUARD FAILED "
              "and nothing else",
              not res_b["guard"]["pass"] and res_b["refused"]
              and not any(k in res_b for k in ("grip", "role_lock", "nash_pull", "accuracy"))
              and "GUARD FAILED" in out_b and "ROLE-LOCK" not in out_b
              and "GRIP:" not in out_b and "G3" not in out_b)
        check("the failing guard cell is the R-advocate's r0 group",
              any(k.endswith(f"/r0:{ROW_SEAT}|{ARM_OF[THIRD_NEUTRAL]}") and v["noverdict"] > 0.05
                  for k, v in res_b["guard"]["cells"].items()))

        # ---- GUARD FAILED path: every r4 vote is a stray UNRESOLVED -----------------
        rows_u, votes_u = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY)
        for v in votes_u:
            v["vote"], v["reject"], v["vote_parsed"] = "UNRESOLVED", 0, 0
        for r in rows_u:
            r["n_accept"], r["n_reject"], r["n_vote_unparsed"] = 0, 0, 3
            r["vote_row_advocate"] = r["vote_col_advocate"] = r["vote_third"] = "UNRESOLVED"
        _write_csv(tdp / "unres_rows.csv", rows_u, ROW_FIELDS)
        _write_csv(tdp / "unres_votes.csv", votes_u, VOTE_FIELDS)
        unres = load_arm(tdp / "unres_rows.csv", tdp / "unres_votes.csv", tag="unres",
                         games=games)
        res_u = analyse_arm(unres, cache_dir=None, draws=50, seed=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_arm_report(res_u)
        check("an arm whose every r4 vote is 'UNRESOLVED' (written as-is by a "
              "runner) FAILS the guard with NOVERDICT 100% on every r4 group, prints "
              "GUARD FAILED, and returns no G2 (was: guard pass, G2 = 0.0, grip False)",
              not res_u["guard"]["pass"] and res_u["refused"]
              and res_u["guard"]["worst_noverdict"] == 1.0
              and all(v["noverdict"] == 1.0 for k, v in res_u["guard"]["cells"].items()
                      if "/r4_vote:" in k)
              and "grip" not in res_u and "GUARD FAILED" in buf.getvalue()
              and "G2" not in buf.getvalue())
        rows_l, votes_l = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY)
        for v in votes_l:
            v["r3_label"], v["r3_label_parsed"] = "UNRESOLVED", 0
        _write_csv(tdp / "unres_l_rows.csv", rows_l, ROW_FIELDS)
        _write_csv(tdp / "unres_l_votes.csv", votes_l, VOTE_FIELDS)
        unres_l = load_arm(tdp / "unres_l_rows.csv", tdp / "unres_l_votes.csv",
                           tag="unres_l", games=games)
        res_ul = analyse_arm(unres_l, cache_dir=None, draws=50, seed=1)
        check("...and likewise when every r3 label is 'UNRESOLVED' (no silent G1 = 0)",
              not res_ul["guard"]["pass"] and res_ul["refused"]
              and all(v["noverdict"] == 1.0 for k, v in res_ul["guard"]["cells"].items()
                      if "/r3_label:" in k))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "unres", "--dir", str(tdp), "--no-cache", "--n-boot", "20"])
        check("main() exits 4 with GUARD FAILED on the all-UNRESOLVED arm",
              rc == 4 and "GUARD FAILED" in buf.getvalue())
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "bad", "--dir", str(tdp), "--no-cache", "--n-boot", "20"])
        check("main() exits 4 on a failed guard and prints GUARD FAILED",
              rc == 4 and "GUARD FAILED" in buf.getvalue())

        # ---- cache enrichment of the truncation column --------------------------------
        cdir = tdp / "cache"
        cdir.mkdir()
        d0 = arm["debates"][0]
        base_name = (f"cgd_{_safe(d0['model'])}_{SCAFFOLD_NAME}_t{{cap}}_{d0['arm']}_"
                     f"{d0['item']}_{d0['sample_idx']:02d}_r0_{ROW_SEAT}.json")
        (cdir / base_name.format(cap=1024)).write_text(json.dumps(
            {"finish_reason": "length", "output": "x" * 50}))
        (cdir / base_name.format(cap=2560)).write_text(json.dumps(
            {"finish_reason": "stop", "output": "y" * 900}))
        syn_name = (f"cgd_{_safe(d0['moderator_model'])}_{SCAFFOLD_NAME}_t1024_{d0['arm']}_"
                    f"{d0['item']}_{d0['sample_idx']:02d}_synthesis_"
                    f"{_safe(moderator_role_id(d0['moderator_model'], d0['third_seat']))}.json")
        (cdir / syn_name).write_text(json.dumps({"finish_reason": "length", "output": "z" * 10}))
        grows = guard_rows_from_csv(arm["debates"])
        known_before = sum(1 for g in grows if g["truncation_known"])
        cov = enrich_guard_from_cache(grows, cdir)
        r0_row = next(g for g in grows if g["item"] == d0["item"] and g["sample_idx"] == d0["sample_idx"]
                      and g["round"] == "r0" and g["role_id"] == ROW_SEAT)
        syn_row = next(g for g in grows if g["item"] == d0["item"] and g["sample_idx"] == d0["sample_idx"]
                       and g["round"] == "synthesis")
        check("CSV alone knows truncation for 4 of 17 calls per cell (integration + r4)",
              known_before == 4 * n_cells)
        check("cache enrichment: 2 records found, the LARGER cap wins for r0 "
              "(finish stop, 900 chars), the synthesis record marks truncation",
              cov["n_found"] == 2 and r0_row["truncated"] == 0 and r0_row["output_len"] == 900
              and r0_row["truncation_known"] and syn_row["truncated"] == 1
              and sum(1 for g in grows if g["truncation_known"]) == known_before + 2)
        # a whole round:seat group truncated in cache must fail the guard
        for d in arm["debates"]:
            nm = (f"cgd_{_safe(d['model'])}_{SCAFFOLD_NAME}_t2560_{d['arm']}_{d['item']}_"
                  f"{d['sample_idx']:02d}_r1_{COL_SEAT}.json")
            (cdir / nm).write_text(json.dumps({"finish_reason": "length", "output": "q"}))
        res_c = analyse_arm(arm, cache_dir=cdir, draws=50, seed=1)
        check("truncation read from the cache alone (VERDICT lines intact) fails the guard",
              not res_c["guard"]["pass"] and res_c["refused"]
              and res_c["guard"]["cache"]["n_found"] == 2 + n_cells)

        # ---- contrast with a synthetic planner arm ------------------------------------
        rows_p, votes_p = _synthetic_arm(games, THIRD_PLANNER, wrong_every=None)
        _write_csv(tdp / "cg_game_planner_rows.csv", rows_p, ROW_FIELDS)
        _write_csv(tdp / "cg_game_planner_votes.csv", votes_p, VOTE_FIELDS)
        planner = load_arm(tdp / "cg_game_planner_rows.csv", tdp / "cg_game_planner_votes.csv",
                           tag="cg_game_planner", games=games)
        check("planner arm loads with plan_mediator as the third seat and a distinct arm",
              planner["third_role_id"] == "plan_mediator"
              and {d["arm"] for d in planner["debates"]} == {ARM_OF[THIRD_PLANNER]})
        con = value_region_contrast(planner, arm, draws=300, seed=5)  # order deliberately swapped
        n_val = sum(1 for d in arm["debates"] if d["region"] == REGION_VALUE)
        n_val_wrong = sum(1 for d in arm["debates"]
                          if d["region"] == REGION_VALUE and d["verdict"] != d["gold"])
        ga = con["stats"]["group_accuracy"]
        check("contrast orients itself planner minus neutral whichever order is given",
              con["b"]["third_seat"] == THIRD_PLANNER and con["a"]["third_seat"] == THIRD_NEUTRAL)
        check(f"value-region contrast: 16 shared items, group accuracy delta exactly "
              f"{n_val_wrong}/{n_val}, CI excludes 0",
              ga["n_shared_items"] == 16 and abs(ga["delta"] - n_val_wrong / n_val) < 1e-9
              and ga["positive"] and n_val == 32)
        check("contrast: third-seat accuracy delta is 0 (both third seats name the gold), "
              "fire-rate delta is negative (planner arm never fires)",
              con["stats"]["third_r0_accuracy"]["delta"] == 0.0
              and con["stats"]["fire_rate"]["delta"] < 0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_contrast(con)
        check("contrast prints with the POSITIVE marker on group accuracy",
              "THIRD-SEAT CONTRAST" in buf.getvalue() and "POSITIVE" in buf.getvalue())
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "cg_game_neutral", "--contrast-tag", "cg_game_planner",
                       "--dir", str(tdp), "--no-cache", "--n-boot", "30",
                       "--out", str(tdp / "analysis.json")])
        check("main() end to end on both arms exits 0, prints both reports and the "
              "contrast, writes JSON",
              rc == 0 and buf.getvalue().count("GAME GRIP -- tag") == 2
              and "THIRD-SEAT CONTRAST" in buf.getvalue()
              and json.loads((tdp / "analysis.json").read_text())["contrast"]["stats"]["group_accuracy"]["positive"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "nope", "--dir", str(tdp), "--no-cache"])
        check("main() exits 2 when the tag's CSVs are absent", rc == 2)

    check("the selftest wrote nothing under the real tag prefix in the cache dir",
          len(cache_files_with_prefix(TAG_PREFIX)) == real_tag_files)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Grip / role-lock / Nash-pull / extent readouts for "
                    "run_crowdgold_game CSVs (offline; reads only).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag", default=f"{TAG_PREFIX}_{THIRD_NEUTRAL}")
    ap.add_argument("--contrast-tag", default=None,
                    help="second arm's tag; adds the value-region contrast")
    ap.add_argument("--model", default=None, help="restrict to one agent model")
    ap.add_argument("--dir", type=Path, default=OUT_DIR,
                    help="directory holding <tag>_rows.csv / _votes.csv and the call cache")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not read the call cache for the guard's truncation column")
    ap.add_argument("--solo-rows", type=Path, default=None,
                    help="an independent single-agent run's CSV (item_id, verdict, "
                         "gold_verdict) for the transfer lift")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None, help="write the full result as JSON")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    games = None
    arms = []
    for tag in [args.tag] + ([args.contrast_tag] if args.contrast_tag else []):
        rows_path = args.dir / f"{tag}_rows.csv"
        votes_path = args.dir / f"{tag}_votes.csv"
        if not rows_path.exists() or not votes_path.exists():
            print(f"\nERROR: {rows_path} / {votes_path} not found. Run "
                  "scripts.run_crowdgold_game first.\n")
            return 2
        if games is None:
            games = load_games()
        arm = load_arm(rows_path, votes_path, tag=tag, model=args.model, games=games)
        if not arm["debates"]:
            print(f"\nERROR: no rows in {rows_path}"
                  f"{' for model ' + args.model if args.model else ''}.\n")
            return 2
        arms.append(arm)

    solo = load_solo_rows(args.solo_rows) if args.solo_rows else None
    cache_dir = None if args.no_cache else args.dir
    results = []
    refused = False
    for i, arm in enumerate(arms):
        res = analyse_arm(arm, cache_dir=cache_dir, solo=solo, draws=args.n_boot,
                          seed=args.seed + 100 * i)
        print_arm_report(res)
        results.append(res)
        refused = refused or res["refused"]
    out = {"arms": results, "contrast": None}
    if refused:
        print("\n*** GUARD FAILED on at least one arm: no grip number is reported; "
              "the contrast is not computed. ***")
    elif len(arms) == 2:
        if arms[0]["third_seat"] == arms[1]["third_seat"]:
            print(f"\n  both tags carry the same third seat ({arms[0]['third_seat']}); "
                  "the contrast reads as second minus first")
        out["contrast"] = value_region_contrast(arms[0], arms[1], draws=args.n_boot,
                                                seed=args.seed + 500)
        print_contrast(out["contrast"])
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 4 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
