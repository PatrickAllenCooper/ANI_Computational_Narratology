"""
scripts/analyze_game_nb.py -- the Addendum 16.13c readouts for the NON-BINDING
regime of the TMGBench games deliberation (scripts/run_crowdgold_game_nb.py,
scaffold game_action_ncot, arms third_neutral_nb / third_planner_nb), one
third-seat arm at a time, with the registered planner-minus-neutral contrast
paired on items when both arms are on disk, and the registered
neutral-minus-solo contrast when the solo non-binding rows are given.

This module makes NO model calls and spends nothing; it reads CSVs the runner
wrote and the sha-pinned game data. There is no --run flag because nothing
here can generate.

WHAT IS REUSED FROM scripts/analyze_game_grip.py (the binding-regime sibling)
AND FROM scripts/tmg_games_nb.py (the 16.13c definition site)

Everything item-agnostic is IMPORTED from the sibling: the item-clustered
bootstrap (clustered_mean, _sliced, paired_delta, _by_item, _mean, and
_boot_ci / the G1-G3 thresholds behind them), the guard report wrapper, the
formatting helpers and the CSV writer. Everything regime-specific that
another module also reads is IMPORTED from tmg_games_nb and aliased under
this module's names: the registered names and the tag prefix, the two
instruments, the gold recommendation (nb_gold_verdict), gold_action /
private_action / dominant_action (gold_act_for / private_act_for /
dominant_act_for) and stake_undermined_nb (argument order (view, verdict,
stake), tmg_games_nb's). Nothing that could disagree with the runner is
defined here; the selftest keeps the runner-vs-analyzer pins as regression
guards. The G1/G2/G3 formulas are the sibling's, on the two *_nb advocate
seats.

THE REGISTERED DESIGN THIS MODULE READS (prereg, Addendum 16.13c)

  scaffold        game_action_ncot
  arms            third_neutral_nb | third_planner_nb
  role ids        player_r_advocate_nb, player_c_advocate_nb,
                  neutral_reader_nb | plan_mediator_nb,
                  modgamenb-<variant>-<model> (moderator)
  tag             cgnb_game_<variant>    (rows / votes CSVs; the runner's
                  TAG_PREFIX, tmg_games_nb.NB_TAG_PREFIX -- the registration
                  fixes every other name; NOT an extension of the binding
                  runner's 'cg_game', whose tag namespace is a startswith test)
  advocate rounds r0 / r1 / r2 answer for their OWN player only:
                  VERDICT: ACT_1 | ACT_2  (the player's DISPLAYED action index)
  third seat, synthesis, integration:
                  VERDICT: PLAN_xy | NO_RECOMMENDATION
  r3 label        ACCEPT | ACCEPT_WITH_MODIFICATION | REJECT
  r4 vote         ACCEPT | REJECT  (on the group's joint recommendation)
  gold            the cell best for all parties: DILEMMA_BOTH_PREFER and
                  ASSURANCE -> gold_cell (the Pareto-dominant cell both prefer
                  / the payoff-dominant equilibrium); neutral region ->
                  gold_cell; cannot-assist -> NO_RECOMMENDATION. Advocate
                  accuracy is scored against the gold cell's action for that
                  player (undefined, hence excluded, on cannot-assist games).
  stake_undermined(advocate, group plan) := the player has a STRICTLY
                  profitable unilateral deviation from the recommended cell
                  (the cell is not a best reply for that player). None when
                  the group verdict is non-committal or NO_RECOMMENDATION;
                  the third seat is False whenever the verdict is committal.

A NOTE THE SELFTEST PINS. The registration says "in DILEMMA the gold cell
undermines BOTH advocates by construction; in ASSURANCE it undermines
neither". Under the registered definition, on the 7 DILEMMA_BOTH_PREFER
games of the pinned data the gold cell undermines BOTH advocates on exactly
ONE game (111, the symmetric case) and exactly ONE advocate on the other six
(the gold cell already gives the other player its maximum, 4, so that player
has no profitable deviation); on the 9 ASSURANCE games it undermines
neither. The definition is implemented as registered; the "BOTH" claim is
not, because it is false on the data. Every readout below therefore also
reports the advocate split by undermined_at_gold (the "tempted" advocate,
whose player gains by deviating from the gold cell, versus the other), so
the registered pooled P1' statistic can be read next to the one the
mechanism actually predicts.

DEFINITIONS

  gold_recommendation(view)   PLAN_xy for the view's gold cell in DISPLAYED
      option numbers, else NO_RECOMMENDATION (tmg_games' gold_verdict with
      NO_UNIQUE_PLAN renamed; the cell is the same registered gold).

  gold_action(view, player)   ACT_x, the player's displayed option at the
      gold cell; None on cannot-assist games.

  private_action(view, player)   ACT_x, the player's displayed option at the
      game's UNIQUE pure Nash cell; None when the game has no or several
      pure Nash cells (ASSURANCE, CONFLICT, CYCLIC, and 2 of the 6 ZERO_SUM
      games). On every DILEMMA_BOTH_PREFER game it differs from gold_action
      for BOTH players (a cell that strictly Pareto-dominates a unique 2x2
      Nash cell can share neither its row nor its column; the selftest pins
      this on the data). This is the registered P1' "private action".
      Supplementary: dominant_action(view, player), the player's strictly
      dominant displayed option (None when none; only game 111 has one for
      both players).

  private-action share (P1')   P(advocate's verdict == its player's private
      action), NOVERDICT counted as NOT private (in the denominator; the
      guard bounds it at 5%), per seat and pooled, at r0 and r2, in DILEMMA
      (registered) and overall on games where it is defined; split by
      undermined_at_gold.

  accuracy   group final (integration) and synthesis verdicts vs the gold
      recommendation, recommendation_correct: a non-committal verdict is
      wrong and stays in the denominator (the registered verdict_format
      rule); NO_RECOMMENDATION against a cannot-assist gold is CORRECT.
      Advocates: action_correct vs gold_action (None -> excluded). Third
      seat: recommendation_correct. Overall, by region, by gold_type.

  G1 / G2 / G3   as the sibling: flagged(d) = third seat objected at r3 OR
      an advocate objected while its stake was NOT undermined (mis-
      localised; an undefined undermined flag is never mis-localised). Grip
      population = debates whose synthesis and final verdicts are both
      committal. Undefined stake_undermined is EXCLUDED from G3, never 0.

  P5 (flooding)   the composite flag's fire rate by region and its error lift
      P(group wrong | flagged) - P(group wrong | not) on the committal cells
      by region; flooding = fire rate on DILEMMA > 0.5 AND the value-region
      lift's CI includes 0. The stake-blind counter (n_objectors >= 2) and
      its lift are reported alongside, as the sibling does, with the >= 60
      fired power floor.

  P6 (steering)   among ADVOCATE r4 votes on cells whose group verdict IS
      the gold recommendation, the ACCEPT share; per arm (overall / by
      region), and planner minus neutral paired on items.

  P2' (drift)   neutral-arm group accuracy on DILEMMA minus the solo
      non-binding accuracy on the same items (paired; each draw resamples
      item ids once and evaluates both sides), read as holding when the CI
      sits below 0. Needs --solo-rows.

  P3' (the mediator holds; PRIMARY)   planner minus neutral group accuracy
      on the value region, paired on items; holds when the CI lower bound
      exceeds 0.

  contrast   when two tags with different third seats are given, every
      delta reads planner minus neutral whichever order was given: group
      final and synthesis accuracy (value region), third-seat r0/r2 accuracy,
      advocate private-action share at r0 (DILEMMA), P6 accept-on-gold share,
      composite and stake-blind fire rates, G3.

GUARD (runs first; on failure nothing else is printed or returned)

As the sibling: 17 guard rows per cell reconstructed from the CSVs, grouped
per (model, scaffold/round:role_id, arm) exactly as the runner's own
call_guard_row groups them. The committal set is PER ROUND AND PER SEAT:
advocates' r0/r1/r2 tokens must be ACT_1 / ACT_2; the third seat's and the
moderator's PLAN rounds must be PLAN_xy / NO_RECOMMENDATION; r3 the three
labels; r4 ACCEPT / REJECT. Anything else -- a blank, a stray UNRESOLVED,
the binding regime's NO_UNIQUE_PLAN, a PLAN from an advocate or an ACT from
the third seat -- is NOVERDICT against the 5% limit. Truncation flags are
read from the CSV where it has them (integration and r4) and otherwise from
the call cache when a record is found by name (largest token cap wins).
When the guard fails the report prints the literal string GUARD FAILED and
refuses to print or return any number.

WHAT THIS MODULE READS -- THE ROW SCHEMA THE RUNNER MUST WRITE

The runner (scripts/run_crowdgold_game_nb.py) is a sibling of
run_crowdgold_game.py with the same protocol and CSV shape; the analyzer
re-renders every (game_id, sample_idx) from the pinned data and refuses on
any disagreement with the CSV's permutation, names and gold, so derived
payoff quantities are always recomputed here. Columns marked R are required;
O are optional and cross-checked against the analyzer's own computation when
present (a disagreement is an error, not a warning).

  <tag>_rows.csv, one row per cell (item_id, arm, sample_idx unique)
    R model, scaffold (= game_action_ncot), arm (third_neutral_nb |
      third_planner_nb), third_seat (neutral | planner), third_role_id
      (neutral_reader_nb | plan_mediator_nb), item_id, game_id (both the
      3-digit game id), sample_idx (int), gold_type, region (tmg_games),
      gold_verdict (PLAN_xy | NO_RECOMMENDATION), row_perm, col_perm
      ("01" | "10"), row_name, col_name, moderator_model
    R verdict            integration verdict: PLAN_xy | NO_RECOMMENDATION |
                         NOVERDICT (written as parsed, never coerced)
    R synthesis_verdict  same set
    R r0_verdict_row_advocate, r0_verdict_col_advocate   ACT_1 | ACT_2 | NOVERDICT
    R r0_verdict_third                                   PLAN_xy | NO_RECOMMENDATION | NOVERDICT
    R r1_verdict_* / r2_verdict_*  (same three columns each)
    R n_objectors        seats whose r3 label was ACCEPT_WITH_MODIFICATION or REJECT
    R truncated, output_len   of the integration call;  synthesis_len
    O n_reject, n_r3_unparsed, n_vote_unparsed, unanimous_accept,
      verdict_revised, gold_action_row, gold_action_col (cross-checked),
      n_calls_type_named, type_named_first_round (transcript-leak covariate)
  <tag>_votes.csv, one row per (cell, seat)
    R model, arm, item_id, sample_idx, gold_type, region, gold_verdict,
      role_id (one of the four *_nb ids), role_stake (row | col | none),
      group_verdict, r0_verdict, r1_verdict, r2_verdict, r3_label,
      objected_r3 (0/1), vote (ACCEPT | REJECT | NOVERDICT), reject (0/1),
      truncated, output_len
    O stake_undermined ("" | 0 | 1; cross-checked against the registered
      definition), gold_action (cross-checked), finish_reason,
      type_named_in_prompt, type_named_rounds
  --solo-rows CSV (run_game_singleagent --regime nonbinding): item_id,
    verdict (PLAN_xy | NO_RECOMMENDATION | NOVERDICT), gold_verdict; optional
    sample_idx, region, gold_type (else taken from the pinned data), masked
    (rows with masked = 1 are dropped), scaffold (when present every row must
    carry tmg_solo_nb -- the BINDING solo run, scaffold tmg_solo, is refused,
    exit 2, as is any file with a NO_UNIQUE_PLAN gold: it is not the
    registered solo non-binding comparator).

Cache files this analyzer looks for (truncation column only, never written):
  cgd_<model>_game_action_ncot_t<cap>_<arm>_<game>_<idx>_<round>_<role>.json

COORDINATION WITH THE RUNNER

This module never imports the runner at import time (the runner may import
from here without a cycle). When scripts/run_crowdgold_game_nb.py exists,
the selftest imports it lazily and pins: the registered names and instrument
tokens are identical; its ROW_FIELDS / VOTE_FIELDS carry every required
column; its gold_action, private_action, dominant_action and
stake_undermined agree with this module's on every rendering (144 games x 3
samples) and every verdict token; and its own offline stub, driven through
its run_deliberation with the cache redirected to a temp dir, produces CSVs
that load here, re-render consistently, pass the guard and yield the
expected numbers. When the runner is absent, the selftest instead asserts
that every 16.13c namespace in divergence_study_outputs is empty (nothing
could legitimately have written it). Either way it asserts the selftest
itself added nothing to the real cache directory.

Usage:
  python -m scripts.analyze_game_nb --selftest
  python -m scripts.analyze_game_nb --tag cgnb_game_neutral
  python -m scripts.analyze_game_nb --tag cgnb_game_neutral --contrast-tag cgnb_game_planner \\
      --solo-rows divergence_study_outputs/game_solo_nb_rows.csv --out /tmp/nb.json

Exit codes: 0 ok, 2 missing / empty / foreign inputs, 4 guard failed (nothing read).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tempfile
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import importlib
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.analyze_dilemma_grip import G1_MAX_FIRE, G2_MIN_REJECT, G3_MIN_CONC, _boot_ci
from scripts.analyze_game_grip import (
    FIRE_MIN_OBJECTORS,
    MIN_FIRED,
    SEAT_ROUNDS,
    _by_item,
    _ci,
    _ci3,
    _f,
    _int,
    _mean,
    _perm_str,
    _sliced,
    _write_csv,
    clustered_mean,
    guard_report,
    paired_delta,
    print_guard,
)
from scripts.run_crowdgold_game import (
    GAME_LABEL_INSTRUMENT,
    GAME_VOTE_INSTRUMENT,
    ROUNDS,
    THIRD_NEUTRAL,
    THIRD_PLANNER,
    THIRD_SEATS,
)
from scripts.run_phase1_quartet import OUT_DIR, _safe
from scripts.tmg_games import (
    CELLS,
    DILEMMA_BOTH_PREFER,
    GOLD_TYPES,
    PLAN_VERDICTS,
    REGIONS,
    REGION_CANNOT_ASSIST,
    REGION_VALUE,
    Game,
    ItemView,
    cache_files_with_prefix,
    forbidden_hits,
    load_games,
    plan_for_cell,
    render_item,
)
from scripts.verdict_format import (
    NOVERDICT,
    UNRESOLVED,
    committal_tokens,
    score_against_gold,
)
# The single definition site of 16.13c (names, instruments, gold, the four
# item-level definitions). This module aliases, never re-defines; the
# selftest keeps every cross-module pin as a regression guard. tmg_games_nb
# imports nothing from here, so there is no cycle.
from scripts.tmg_games_nb import (
    ACT_1,
    ACT_2,
    ACT_INSTRUMENT,
    ACT_VERDICTS,
    NB_ARM_OF,
    NB_INSTRUMENT,
    NB_MODERATOR_PREFIX,
    NB_ROLE_NEUTRAL_READER,
    NB_ROLE_PLAN_MEDIATOR,
    NB_ROLE_PLAYER_C,
    NB_ROLE_PLAYER_R,
    NB_SCAFFOLD_NAME,
    NB_TAG_PREFIX,
    NB_VERDICT_SET,
    NO_RECOMMENDATION,
    dominant_act_for,
    gold_act_for,
    nb_gold_verdict,
    private_act_for,
    score_act,
    stake_undermined_nb,
)

# ---------------------------------------------------------------------------
# Registered names (Addendum 16.13c), IMPORTED from scripts.tmg_games_nb and
# re-exported under this module's names; the selftest pins the literals
# against the registration and against scripts.run_crowdgold_game_nb when it
# exists.
# ---------------------------------------------------------------------------

SCAFFOLD_NB = NB_SCAFFOLD_NAME
TAG_PREFIX_NB = NB_TAG_PREFIX
MODERATOR_PREFIX_NB = NB_MODERATOR_PREFIX
ARM_OF_NB = dict(NB_ARM_OF)
ROW_SEAT_NB = NB_ROLE_PLAYER_R
COL_SEAT_NB = NB_ROLE_PLAYER_C
STAKE_SEATS_NB = (ROW_SEAT_NB, COL_SEAT_NB)
THIRD_ROLE_ID_NB = {THIRD_NEUTRAL: NB_ROLE_NEUTRAL_READER, THIRD_PLANNER: NB_ROLE_PLAN_MEDIATOR}
THIRD_ROLE_IDS_NB = tuple(THIRD_ROLE_ID_NB.values())
ALL_ROLE_IDS_NB = STAKE_SEATS_NB + THIRD_ROLE_IDS_NB
SEAT_STAKE_NB = {ROW_SEAT_NB: "row", COL_SEAT_NB: "col",
                 THIRD_ROLE_ID_NB[THIRD_NEUTRAL]: "none",
                 THIRD_ROLE_ID_NB[THIRD_PLANNER]: "none"}
SEAT_SHORT_NB = {ROW_SEAT_NB: "R-adv", COL_SEAT_NB: "C-adv",
                 THIRD_ROLE_ID_NB[THIRD_NEUTRAL]: "neutral",
                 THIRD_ROLE_ID_NB[THIRD_PLANNER]: "planner"}
CALLS_PER_CELL_NB = 17
#: Every namespace the runner introduces; each is a substring of a cache file
#: name (cgd_ files) or a prefix (the tag).
NB_NAMESPACES = (SCAFFOLD_NB, *ARM_OF_NB.values(), *ALL_ROLE_IDS_NB,
                 MODERATOR_PREFIX_NB, TAG_PREFIX_NB)

# ---------------------------------------------------------------------------
# Instruments (registered with verdict_format by tmg_games_nb, not edited
# into it). The label and vote rounds carry the binding runner's tokens under
# its instrument names; the two nb instruments are this regime's own.
# ---------------------------------------------------------------------------

ACTION_INSTRUMENT = ACT_INSTRUMENT
ACTION_SET = ACT_VERDICTS
RECOMMENDATION_INSTRUMENT = NB_INSTRUMENT
RECOMMENDATION_SET = NB_VERDICT_SET

LABEL_INSTRUMENT_NB = GAME_LABEL_INSTRUMENT
VOTE_INSTRUMENT_NB = GAME_VOTE_INSTRUMENT
PLAN_ROUNDS = ("r0", "r1", "r2", "synthesis", "integration")


def round_instrument(round_name: str, role_id: str) -> str:
    """The committal instrument of one call: advocates answer ACT on the
    three PLAN-bearing agent rounds, every other seat answers a
    recommendation; r3 / r4 are the shared label / vote instruments."""
    if round_name == "r3_label":
        return LABEL_INSTRUMENT_NB
    if round_name == "r4_vote":
        return VOTE_INSTRUMENT_NB
    if round_name not in PLAN_ROUNDS:
        raise ValueError(f"unknown round {round_name!r}")
    return ACTION_INSTRUMENT if role_id in STAKE_SEATS_NB else RECOMMENDATION_INSTRUMENT


def moderator_role_id_nb(mod_model: str, third_seat: str) -> str:
    return f"{MODERATOR_PREFIX_NB}-{third_seat}-{_safe(mod_model)}"


#: Columns the loaders read (module docstring, ROW SCHEMA).
ROWS_REQUIRED = (
    "model", "scaffold", "arm", "third_seat", "third_role_id", "item_id",
    "game_id", "sample_idx", "gold_type", "region", "gold_verdict", "row_perm",
    "col_perm", "row_name", "col_name", "moderator_model", "verdict",
    "synthesis_verdict", "r0_verdict_row_advocate", "r0_verdict_col_advocate",
    "r0_verdict_third", "r1_verdict_row_advocate", "r1_verdict_col_advocate",
    "r1_verdict_third", "r2_verdict_row_advocate", "r2_verdict_col_advocate",
    "r2_verdict_third", "n_objectors", "truncated", "output_len", "synthesis_len",
)
ROWS_OPTIONAL = (
    "n_reject", "n_r3_unparsed", "n_vote_unparsed", "unanimous_accept",
    "verdict_revised", "gold_action_row", "gold_action_col",
    "n_calls_type_named", "type_named_first_round",
)
VOTES_REQUIRED = (
    "model", "arm", "item_id", "sample_idx", "gold_type", "region",
    "gold_verdict", "role_id", "role_stake", "group_verdict", "r0_verdict",
    "r1_verdict", "r2_verdict", "r3_label", "objected_r3", "vote", "reject",
    "truncated", "output_len",
)
VOTES_OPTIONAL = ("stake_undermined", "gold_action", "finish_reason",
                  "type_named_in_prompt", "type_named_rounds", "synthesis_verdict")


# ---------------------------------------------------------------------------
# Item-level definitions (registered; everything recomputed from the view):
# tmg_games_nb's, aliased under this module's names. Argument order is
# tmg_games_nb's -- (view, player) and (view, verdict, stake).
# ---------------------------------------------------------------------------

#: PLAN_xy for the gold cell in displayed numbers, else NO_RECOMMENDATION.
gold_recommendation = nb_gold_verdict
#: The player's displayed option at the gold cell; None on cannot-assist.
gold_action = gold_act_for
#: The player's displayed option at the game's unique pure Nash cell; None
#: when the game has no or several pure Nash cells.
private_action = private_act_for
#: The player's strictly dominant displayed option, or None.
dominant_action = dominant_act_for


def unique_nash_cell(view: ItemView) -> Optional[tuple[int, int]]:
    cells = [view.cell_for(v) for v in view.nash_verdicts]
    return cells[0] if len(cells) == 1 else None


def plan_action(verdict: str, player: str) -> Optional[str]:
    """The player's displayed option named by a PLAN_xy; None otherwise."""
    if verdict not in PLAN_VERDICTS:
        return None
    return f"ACT_{verdict[5] if player == 'row' else verdict[6]}"


def undermined_at_gold(view: ItemView, player: str) -> Optional[bool]:
    """Would the gold recommendation itself undermine this player? The
    'tempted' advocate of a one-sided dilemma. None on cannot-assist."""
    return stake_undermined_nb(view, gold_recommendation(view), player)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def recommendation_correct(verdict: str, gold: str) -> Optional[bool]:
    """verdict_format's rule on the recommendation instrument: non-committal
    is WRONG and stays in the denominator; NO_RECOMMENDATION against a
    cannot-assist gold is CORRECT. None only for a malformed gold."""
    if gold not in RECOMMENDATION_SET:
        return None
    return bool(score_against_gold(verdict or NOVERDICT, gold,
                                   RECOMMENDATION_INSTRUMENT)["correct"])


def committal_recommendation_correct(verdict: str, gold: str) -> Optional[bool]:
    """The sibling's syn_ok / s2_ok: None unless the verdict is committal."""
    if verdict not in RECOMMENDATION_SET:
        return None
    return recommendation_correct(verdict, gold)


def action_correct(verdict: str, gold_act: Optional[str]) -> Optional[bool]:
    """Advocate accuracy vs the gold cell's action for its player
    (tmg_games_nb.score_act's "correct" flag); None (excluded) when the game
    has no gold cell. NOVERDICT is wrong and stays in the denominator."""
    sc = score_act(verdict or NOVERDICT, gold_act)
    return None if sc is None else bool(sc["correct"])


def guard_token(round_name: str, role_id: str, verdict) -> str:
    """The CSV's token if it is one of that round's AND seat's committal
    tokens, else NOVERDICT (module docstring, GUARD)."""
    v = (verdict or "").strip()
    return v if v in committal_tokens(round_instrument(round_name, role_id)) else NOVERDICT


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _require(fieldnames: Sequence[str], required: Sequence[str], what: str) -> None:
    missing = [c for c in required if c not in (fieldnames or ())]
    if missing:
        raise ValueError(f"{what}: missing required columns {missing} "
                         f"(module docstring, ROW SCHEMA)")


def _refuse_foreign(r: dict, what: str) -> None:
    """A binding-regime CSV (or any other instrument's) must never be read
    as a non-binding one: scaffold, arm and role ids are checked per row."""
    if r.get("scaffold") != SCAFFOLD_NB:
        raise ValueError(f"{what}: scaffold {r.get('scaffold')!r} is not "
                         f"{SCAFFOLD_NB!r}; this is not a 16.13c run")
    if r.get("arm") not in ARM_OF_NB.values():
        raise ValueError(f"{what}: arm {r.get('arm')!r} is not one of "
                         f"{sorted(ARM_OF_NB.values())}")


def load_debates_nb(rows_path: Path, votes_path: Path, *,
                    model: Optional[str] = None) -> list[dict]:
    """One record per debate (cell) with the sensor fields the grip needs
    (obj_<seat>, label_<seat>, vote_<seat>; undermined_<seat> is filled by
    attach_views from the registered definition). Every debate is returned;
    grip_population applies the committal filter."""
    debates: dict[tuple, dict] = {}
    with open(rows_path, newline="") as f:
        rd = csv.DictReader(f)
        _require(rd.fieldnames or (), ROWS_REQUIRED, rows_path.name)
        for r in rd:
            if model and r.get("model") != model:
                continue
            _refuse_foreign(r, rows_path.name)
            third = r["third_role_id"]
            if third not in THIRD_ROLE_IDS_NB:
                raise ValueError(f"{rows_path.name}: third_role_id {third!r} is not one "
                                 f"of {THIRD_ROLE_IDS_NB}")
            if r["third_seat"] not in THIRD_SEATS or ARM_OF_NB[r["third_seat"]] != r["arm"]:
                raise ValueError(f"{rows_path.name}: third_seat {r['third_seat']!r} does "
                                 f"not match arm {r['arm']!r}")
            key = (r["item_id"], r["arm"], r["sample_idx"])
            debates[key] = {
                "item": r["item_id"], "game_id": r.get("game_id") or r["item_id"],
                "arm": r["arm"], "third_seat": r["third_seat"],
                "third_role_id": third, "seats": (ROW_SEAT_NB, COL_SEAT_NB, third),
                "model": r["model"],
                "moderator_model": r.get("moderator_model") or r["model"],
                "scaffold": r["scaffold"], "sample_idx": int(r["sample_idx"]),
                "gold_type": r["gold_type"], "region": r["region"],
                "gold": r["gold_verdict"],
                "row_perm": r["row_perm"], "col_perm": r["col_perm"],
                "row_name": r["row_name"], "col_name": r["col_name"],
                "verdict": r["verdict"], "synthesis_verdict": r["synthesis_verdict"],
                "correct": recommendation_correct(r["verdict"], r["gold_verdict"]),
                "syn_correct": recommendation_correct(r["synthesis_verdict"], r["gold_verdict"]),
                "syn_ok": committal_recommendation_correct(r["synthesis_verdict"], r["gold_verdict"]),
                "s2_ok": committal_recommendation_correct(r["verdict"], r["gold_verdict"]),
                "r0": {ROW_SEAT_NB: r["r0_verdict_row_advocate"],
                       COL_SEAT_NB: r["r0_verdict_col_advocate"],
                       third: r["r0_verdict_third"]},
                "r1": {ROW_SEAT_NB: r["r1_verdict_row_advocate"],
                       COL_SEAT_NB: r["r1_verdict_col_advocate"],
                       third: r["r1_verdict_third"]},
                "r2": {ROW_SEAT_NB: r["r2_verdict_row_advocate"],
                       COL_SEAT_NB: r["r2_verdict_col_advocate"],
                       third: r["r2_verdict_third"]},
                "n_objectors": int(r.get("n_objectors") or 0),
                "n_reject": _int(r.get("n_reject")),
                "truncated": _int(r.get("truncated")),
                "output_len": _int(r.get("output_len")),
                "synthesis_len": _int(r.get("synthesis_len")),
                "csv_gold_action_row": r.get("gold_action_row"),
                "csv_gold_action_col": r.get("gold_action_col"),
                "n_calls_type_named": _int(r.get("n_calls_type_named")),
            }
    with open(votes_path, newline="") as f:
        rd = csv.DictReader(f)
        _require(rd.fieldnames or (), VOTES_REQUIRED, votes_path.name)
        for v in rd:
            if model and v.get("model") != model:
                continue
            d = debates.get((v["item_id"], v["arm"], v["sample_idx"]))
            if d is None:
                continue
            seat = v["role_id"]
            if seat not in d["seats"]:
                raise ValueError(f"{votes_path.name}: role_id {seat!r} is not a seat of "
                                 f"the {d['arm']} arm {d['seats']}")
            d[f"obj_{seat}"] = v["objected_r3"] == "1"
            d[f"rej_{seat}"] = v["vote"] == "REJECT"
            d[f"label_{seat}"] = v.get("r3_label") or NOVERDICT
            d[f"vote_{seat}"] = v.get("vote") or NOVERDICT
            d[f"vote_truncated_{seat}"] = _int(v.get("truncated"))
            d[f"vote_len_{seat}"] = _int(v.get("output_len"))
    return list(debates.values())


def load_votes_nb(path: Path, *, model: Optional[str] = None) -> list[dict]:
    out = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        _require(rd.fieldnames or (), VOTES_REQUIRED, path.name)
        for r in rd:
            if model and r.get("model") != model:
                continue
            _refuse_foreign({"scaffold": r.get("scaffold", SCAFFOLD_NB), "arm": r.get("arm")},
                            path.name)
            seat = r["role_id"]
            if seat not in ALL_ROLE_IDS_NB:
                raise ValueError(f"{path.name}: role_id {seat!r} is not one of {ALL_ROLE_IDS_NB}")
            stake = r.get("role_stake") or SEAT_STAKE_NB[seat]
            if stake != SEAT_STAKE_NB[seat]:
                raise ValueError(f"{path.name}: role_stake {stake!r} on {seat} (expected "
                                 f"{SEAT_STAKE_NB[seat]!r})")
            out.append({
                "item": r["item_id"], "arm": r["arm"], "role": seat, "stake": stake,
                "sample_idx": int(r["sample_idx"]),
                "gold_type": r["gold_type"], "region": r["region"],
                "gold": r["gold_verdict"],
                "reject": int(r.get("reject") or 0),
                "objected": int(r.get("objected_r3") or 0),
                "r3_label": r.get("r3_label", ""), "vote": r.get("vote", ""),
                "r0_verdict": r.get("r0_verdict", ""),
                "r1_verdict": r.get("r1_verdict", ""),
                "r2_verdict": r.get("r2_verdict", ""),
                "group_verdict": r.get("group_verdict", ""),
                # the CSV's own coding, cross-checked in attach_views; the
                # analyzer's value ("undermined") comes from the definition
                "csv_undermined": _int(r.get("stake_undermined")),
                "csv_gold_action": r.get("gold_action"),
                "undermined": None,
                "truncated": _int(r.get("truncated")),
                "output_len": _int(r.get("output_len")),
            })
    return out


def build_views_nb(debates: Sequence[dict],
                   games: Optional[Sequence[Game]] = None) -> dict[tuple, ItemView]:
    """Re-render every (game_id, sample_idx) and refuse on any disagreement
    with the CSV (permutation, names, gold recommendation)."""
    games = games if games is not None else load_games()
    by_id = {g.game_id: g for g in games}
    views: dict[tuple, ItemView] = {}
    for d in debates:
        key = (d["item"], d["sample_idx"])
        if key in views:
            continue
        g = by_id.get(d["game_id"])
        if g is None:
            raise ValueError(f"game {d['game_id']!r} in the CSV is not in the pinned data")
        v = render_item(g, d["sample_idx"])
        got = (_perm_str(v.row_perm), _perm_str(v.col_perm), v.row_name, v.col_name,
               gold_recommendation(v), v.gold_type, v.region)
        want = (d["row_perm"], d["col_perm"], d["row_name"], d["col_name"], d["gold"],
                d["gold_type"], d["region"])
        if got != want:
            raise ValueError(f"rendering mismatch on {key}: data gives {got}, CSV says "
                             f"{want}; the CSV was not produced from this data / "
                             f"renderer (a binding-regime gold NO_UNIQUE_PLAN is a "
                             f"mismatch here)")
        views[key] = v
    return views


def attach_views(debates: Sequence[dict], votes: Sequence[dict],
                 views: dict[tuple, ItemView]) -> dict:
    """Fill every derived field from the registered definitions and cross-
    check the optional CSV columns. Returns the cross-check counts; raises on
    any disagreement (a runner that codes stake_undermined or gold_action
    differently from the registration is an error, not a variant)."""
    checked = mismatched = 0
    has_col = _has_col(votes, "csv_undermined")
    for v in votes:
        view = views[(v["item"], v["sample_idx"])]
        v["undermined"] = stake_undermined_nb(view, v["group_verdict"], v["stake"])
        # Cross-check the advocate seats only (the ones G3 reads); a runner may
        # leave the third seat's column blank. With the column present, every
        # advocate vote must agree: a value where the definition gives None, a
        # blank where it gives a bool, or a flipped bool is a mismatch.
        if has_col and v["stake"] in ("row", "col"):
            checked += 1
            csv_u = v["csv_undermined"]
            if (csv_u is None) != (v["undermined"] is None) or (
                    csv_u is not None and bool(csv_u) != v["undermined"]):
                mismatched += 1
        if v["stake"] in ("row", "col"):
            ga = gold_action(view, v["stake"])
            v["gold_action"] = ga
            if v["csv_gold_action"] not in (None, "") and v["csv_gold_action"] != (ga or ""):
                mismatched += 1
        else:
            v["gold_action"] = None
    for d in debates:
        view = views[(d["item"], d["sample_idx"])]
        for seat in d["seats"]:
            d[f"undermined_{seat}"] = stake_undermined_nb(view, d["verdict"], SEAT_STAKE_NB[seat])
        for stake, col in (("row", "csv_gold_action_row"), ("col", "csv_gold_action_col")):
            ga = gold_action(view, stake)
            if d.get(col) not in (None, "") and d[col] != (ga or ""):
                mismatched += 1
    if mismatched:
        raise ValueError(f"{mismatched} CSV value(s) of stake_undermined / gold_action "
                         f"disagree with the registered definitions; fix the runner")
    return {"n_checked": checked, "n_mismatched": mismatched,
            "csv_has_stake_undermined": has_col}


def _has_col(votes: Sequence[dict], key: str) -> bool:
    return any(v.get(key) is not None for v in votes)


def grip_population(debates: Sequence[dict]) -> list[dict]:
    """The sibling's filter: both moderator verdicts committal."""
    return [d for d in debates if d["syn_ok"] is not None and d["s2_ok"] is not None]


# ---------------------------------------------------------------------------
# Seat records and the P1' private-action share
# ---------------------------------------------------------------------------


def _seat_sort_key(seat: str) -> tuple:
    order = {ROW_SEAT_NB: 0, COL_SEAT_NB: 1}
    return (order.get(seat, 2), seat)


def seat_records(votes: Sequence[dict], views: dict, seat: str, round_name: str) -> list[dict]:
    """One record per (cell, seat, round) with the seat's verdict scored the
    way its instrument requires (advocates vs gold_action, third seat vs the
    gold recommendation)."""
    out = []
    for v in votes:
        if v["role"] != seat:
            continue
        view = views[(v["item"], v["sample_idx"])]
        verdict = v[f"{round_name}_verdict"]
        rec = {"item": v["item"], "seat": seat, "region": v["region"],
               "gold_type": v["gold_type"], "verdict": verdict,
               "noverdict": verdict not in ACTION_SET + RECOMMENDATION_SET}
        if v["stake"] in ("row", "col"):
            pa = private_action(view, v["stake"])
            da = dominant_action(view, v["stake"])
            rec.update({
                "correct": action_correct(verdict, gold_action(view, v["stake"])),
                "private_defined": pa is not None,
                "is_private": None if pa is None else int(verdict == pa),
                "is_dominant": None if da is None else int(verdict == da),
                "tempted": undermined_at_gold(view, v["stake"]),
                "is_nash": None,
            })
        else:
            rec.update({
                "correct": recommendation_correct(verdict, v["gold"]),
                "private_defined": False, "is_private": None, "is_dominant": None,
                "tempted": None, "is_nash": view.is_nash(verdict),
            })
        out.append(rec)
    return out


def _share_block(recs: Sequence[dict], key: str, *, draws: int, seed: int) -> dict:
    """all / DILEMMA / by tempted, item-clustered, for one 0/1 key."""
    dil = [r for r in recs if r["gold_type"] == DILEMMA_BOTH_PREFER]
    return {
        "all_defined": clustered_mean(recs, lambda r: r[key], draws=draws, seed=seed),
        "dilemma": clustered_mean(dil, lambda r: r[key], draws=draws, seed=seed),
        "dilemma_tempted": clustered_mean([r for r in dil if r["tempted"]], lambda r: r[key],
                                          draws=draws, seed=seed),
        "dilemma_not_tempted": clustered_mean([r for r in dil if r["tempted"] is False],
                                              lambda r: r[key], draws=draws, seed=seed),
        "by_region": {rg: clustered_mean([r for r in recs if r["region"] == rg],
                                         lambda r: r[key], draws=draws, seed=seed)
                      for rg in REGIONS},
        "by_gold_type": {gt: clustered_mean([r for r in recs if r["gold_type"] == gt],
                                            lambda r: r[key], draws=draws, seed=seed)
                         for gt in GOLD_TYPES},
    }


def private_action_share(votes: Sequence[dict], views: dict, *, draws: int, seed: int) -> dict:
    """P1': P(advocate names its player's private action), NOVERDICT counted
    as not private, pooled over both advocates and per seat, at r0 and r2."""
    out: dict = {"definition": "private action = the player's displayed option at the "
                               "game's unique pure Nash cell; NOVERDICT counts as not private",
                 "rounds": SEAT_ROUNDS, "pooled": {}, "per_seat": {}, "dominant_pooled": {}}
    for i, rnd in enumerate(SEAT_ROUNDS):
        recs = [r for s in STAKE_SEATS_NB for r in seat_records(votes, views, s, rnd)]
        out["pooled"][rnd] = _share_block(recs, "is_private", draws=draws, seed=seed + i)
        out["pooled"][rnd]["noverdict_share_dilemma"] = _mean(
            [int(r["noverdict"]) for r in recs if r["gold_type"] == DILEMMA_BOTH_PREFER])
        out["dominant_pooled"][rnd] = _share_block(recs, "is_dominant", draws=draws,
                                                   seed=seed + 10 + i)
        out["per_seat"][rnd] = {}
        for s in STAKE_SEATS_NB:
            sr = [r for r in recs if r["seat"] == s]
            out["per_seat"][rnd][s] = {
                "dilemma": clustered_mean([r for r in sr if r["gold_type"] == DILEMMA_BOTH_PREFER],
                                          lambda r: r["is_private"], draws=draws, seed=seed + i),
                "all_defined": clustered_mean(sr, lambda r: r["is_private"], draws=draws,
                                              seed=seed + i),
            }
    d0 = out["pooled"]["r0"]["dilemma"]
    out["P1_prime"] = {
        "statistic": "pooled advocate private-action share, DILEMMA, r0",
        "point": d0["point"], "lo": d0["lo"], "hi": d0["hi"], "n": d0["n"],
        "n_items": d0["n_items"],
        "holds": bool(d0["point"] is not None and d0["point"] >= 0.5
                      and d0["lo"] is not None and d0["lo"] > 0.5),
    }
    return out


# ---------------------------------------------------------------------------
# Accuracy and the third seat's Nash-pull
# ---------------------------------------------------------------------------


def accuracy_tables(debates: Sequence[dict], votes: Sequence[dict], views: dict, *,
                    draws: int, seed: int) -> dict:
    seats = sorted({v["role"] for v in votes}, key=_seat_sort_key)
    group = {
        "final": _sliced(debates, lambda d: int(bool(d["correct"])), draws=draws, seed=seed),
        "synthesis": _sliced(debates, lambda d: int(bool(d["syn_correct"])),
                             draws=draws, seed=seed),
        "verdict_counts": dict(Counter(d["verdict"] for d in debates)),
        "noncommittal_share": (sum(1 for d in debates if d["verdict"] not in RECOMMENDATION_SET)
                               / len(debates) if debates else None),
        "gold_counts": dict(Counter(d["gold"] for d in debates)),
    }
    per_seat = {}
    for seat in seats:
        per_seat[seat] = {}
        for rnd in SEAT_ROUNDS:
            recs = seat_records(votes, views, seat, rnd)
            per_seat[seat][rnd] = _sliced(
                recs, lambda r: None if r["correct"] is None else int(r["correct"]),
                draws=draws, seed=seed)
    third_nash = {}
    for seat in seats:
        if seat in STAKE_SEATS_NB:
            continue
        third_nash[seat] = {rnd: _sliced(
            seat_records(votes, views, seat, rnd),
            lambda r: None if r["is_nash"] is None else int(r["is_nash"]),
            draws=draws, seed=seed + 5) for rnd in SEAT_ROUNDS}
    games_by_type: Counter = Counter()
    for gt, gid in {(d["gold_type"], d["game_id"]) for d in debates}:
        games_by_type[gt] += 1
    tn = [d["n_calls_type_named"] for d in debates]
    return {"seats": seats, "group": group, "per_seat": per_seat,
            "third_nash_pull": third_nash,
            "cells_by_gold_type": dict(Counter(d["gold_type"] for d in debates)),
            "games_by_gold_type": dict(games_by_type),
            "type_named_cell_share": (None if any(x is None for x in tn) or not tn
                                      else sum(1 for x in tn if x > 0) / len(tn))}


# ---------------------------------------------------------------------------
# Grip (formulas copy-checked against analyze_game_grip / analyze_dilemma_grip)
# ---------------------------------------------------------------------------


def mislocalised(d: dict) -> bool:
    """An advocate objected while its stake was NOT undermined. The sibling
    writes ``not d.get(..., True)``; here the key is always present after
    attach_views and may be None (undefined), which must never read as
    "not undermined", so the test is ``is False``."""
    return any(d.get(f"obj_{s}") and d.get(f"undermined_{s}") is False
               for s in STAKE_SEATS_NB)


def third_objected(d: dict) -> bool:
    return bool(d.get(f"obj_{d['third_role_id']}"))


def flagged(d: dict) -> bool:
    """The composite sensor: third-seat objection OR mis-localised objection.
    An undefined undermined flag (None) reads as not mis-localised, as the
    sibling's .get(..., True) default -- None is a present key here, so it
    is coerced explicitly."""
    return third_objected(d) or mislocalised(d)


def fired(d: dict) -> bool:
    return d["n_objectors"] >= FIRE_MIN_OBJECTORS


def stake_concentration(votes: Sequence[dict], *, draws: int, seed: int) -> dict:
    sv_all = [v for v in votes if v["role"] in STAKE_SEATS_NB]
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
            "undermined": (sum(und) / len(und) if und and s in STAKE_SEATS_NB else None),
            "n_undermined_defined": len(und) if s in STAKE_SEATS_NB else None,
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
# P5: fire rate and error lift by region, composite flag and stake-blind counter
# ---------------------------------------------------------------------------


def _lift(pop: Sequence[dict], pred: Callable[[dict], bool], *, draws: int, seed: int) -> dict:
    fl = [d for d in pop if pred(d)]
    un = [d for d in pop if not pred(d)]
    by_item: dict[str, list] = defaultdict(list)
    for d in pop:
        by_item[d["item"]].append(d)

    def wrong(ds):
        return 1 - sum(bool(d["s2_ok"]) for d in ds) / len(ds)

    def stat(ds):
        f = [d for d in ds if pred(d)]
        u = [d for d in ds if not pred(d)]
        if not f or not u:
            return None
        return wrong(f) - wrong(u)

    p, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed) if by_item else (None, None, None)
    return {
        "n_codable": len(pop), "n_fired": len(fl), "powered": len(fl) >= MIN_FIRED,
        "p_wrong_fired": wrong(fl) if fl else None,
        "p_wrong_unfired": wrong(un) if un else None,
        "delta": p, "lo": lo, "hi": hi,
        "ci_includes_zero": bool(lo is not None and hi is not None and lo <= 0 <= hi),
    }


def flag_readouts(debates: Sequence[dict], *, draws: int, seed: int) -> dict:
    """Fire rate (all / by region / by gold_type) and lift (all / by region)
    for the composite flag (P5) and the stake-blind counter (sibling)."""
    codable = [d for d in debates if d["s2_ok"] is not None]
    pop = grip_population(debates)
    out = {}
    # composite: fire rate and lift on the grip population (both moderator
    # verdicts committal, where flagged() is defined), as G1; stake-blind:
    # fire rate on every debate and lift on the S2-codable ones, as the sibling.
    for name, pred, base, cod in (("composite", flagged, pop, pop),
                                  ("stake_blind", fired, debates, codable)):
        out[name] = {
            "counter": ("third objected OR mis-localised advocate objection"
                        if name == "composite" else f"n_objectors >= {FIRE_MIN_OBJECTORS}"),
            "min_fired": MIN_FIRED,
            "fire_rate": _sliced(base, lambda d, p=pred: int(p(d)), draws=draws, seed=seed),
            "lift": {"all": _lift(cod, pred, draws=draws, seed=seed + 1),
                     "by_region": {rg: _lift([d for d in cod if d["region"] == rg], pred,
                                             draws=draws, seed=seed + 1) for rg in REGIONS}},
        }
    fr = out["composite"]["fire_rate"]["by_gold_type"][DILEMMA_BOTH_PREFER]
    lv = out["composite"]["lift"]["by_region"][REGION_VALUE]
    out["P5"] = {
        "statistic": "composite fire rate on DILEMMA > 0.5 and value-region lift CI includes 0",
        "fire_rate_dilemma": fr, "lift_value": lv,
        "flooding": bool(fr["point"] is not None and fr["point"] > 0.5 and lv["ci_includes_zero"]),
        "computable": bool(fr["point"] is not None and lv["delta"] is not None),
    }
    return out


# ---------------------------------------------------------------------------
# P6: advocate ACCEPT share on gold recommendations
# ---------------------------------------------------------------------------


def accept_on_gold_records(votes: Sequence[dict]) -> list[dict]:
    """Advocate r4 votes on cells whose group verdict IS the gold
    recommendation; a NOVERDICT vote counts as not ACCEPT."""
    return [{"item": v["item"], "region": v["region"], "gold_type": v["gold_type"],
             "seat": v["role"], "accept": int(v["vote"] == "ACCEPT"),
             "undermined": v["undermined"]}
            for v in votes if v["role"] in STAKE_SEATS_NB
            and v["group_verdict"] in RECOMMENDATION_SET and v["group_verdict"] == v["gold"]]


def accept_on_gold(votes: Sequence[dict], *, draws: int, seed: int) -> dict:
    recs = accept_on_gold_records(votes)
    out = {"statistic": "advocate r4 ACCEPT share on cells whose group verdict is the gold "
                        "recommendation", "n_votes": len(recs),
           "pooled": _sliced(recs, lambda r: r["accept"], draws=draws, seed=seed),
           "per_seat": {s: clustered_mean([r for r in recs if r["seat"] == s],
                                          lambda r: r["accept"], draws=draws, seed=seed)
                        for s in STAKE_SEATS_NB},
           "by_undermined": {
               "undermined": clustered_mean([r for r in recs if r["undermined"]],
                                            lambda r: r["accept"], draws=draws, seed=seed),
               "not_undermined": clustered_mean([r for r in recs if r["undermined"] is False],
                                                lambda r: r["accept"], draws=draws, seed=seed)}}
    return out


# ---------------------------------------------------------------------------
# Solo comparator (P2' and the sibling's transfer lift)
# ---------------------------------------------------------------------------


#: The solo comparator's non-binding cache scaffold (run_game_singleagent
#: --regime nonbinding writes it into every row's scaffold column; the
#: selftest pins it against run_game_singleagent.SCAFFOLD_NAME_NB). The
#: BINDING solo run (scaffold tmg_solo, which asked the binding question) is
#: NOT the registered "solo non-binding comparator" and is refused.
SOLO_SCAFFOLD_NB = "tmg_solo_nb"


def load_solo_rows_nb(path: Path, games: Optional[Sequence[Game]] = None) -> list[dict]:
    """The solo NON-BINDING run's verdicts: item_id, verdict, gold_verdict
    (+ optional scaffold, sample_idx, region, gold_type, masked). Masked rows
    (masked = 1 or arm = masked) are counted and dropped. The file is REFUSED
    (ValueError) when its scaffold column carries anything but
    SOLO_SCAFFOLD_NB or when any row's gold is not a recommendation token (a
    NO_UNIQUE_PLAN gold is the binding regime's): P2' must be read against
    a solo model asked the non-binding question, and a binding-regime file
    whose PLAN golds coincide with the nb golds would otherwise pass
    silently."""
    games = games if games is not None else load_games()
    meta = {g.game_id: (g.region, g.gold_type) for g in games}
    out, n_masked = [], 0
    bad_scaffolds: set[str] = set()
    bad_golds: list[tuple[str, str]] = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        has_scaffold = "scaffold" in (rd.fieldnames or ())
        for r in rd:
            if has_scaffold and r.get("scaffold", "") != SOLO_SCAFFOLD_NB:
                bad_scaffolds.add(r.get("scaffold", ""))
            if (r.get("masked") or "0") == "1" or r.get("arm") == "masked":
                n_masked += 1
                continue
            ok = recommendation_correct(r.get("verdict", ""), r.get("gold_verdict", ""))
            if ok is None:
                bad_golds.append((r.get("item_id", ""), r.get("gold_verdict", "")))
                continue
            rg, gt = meta.get(r["item_id"], (None, None))
            out.append({"item": r["item_id"], "sample_idx": _int(r.get("sample_idx")) or 0,
                        "region": r.get("region") or rg, "gold_type": r.get("gold_type") or gt,
                        "correct": int(ok), "wrong": int(not ok)})
    if bad_scaffolds:
        raise ValueError(f"{path.name}: scaffold {sorted(bad_scaffolds)} is not "
                         f"{SOLO_SCAFFOLD_NB!r}; this is not the solo NON-BINDING run "
                         "(run_game_singleagent --regime nonbinding), refusing to read "
                         "it as the P2' comparator")
    if bad_golds:
        raise ValueError(f"{path.name}: {len(bad_golds)} row(s) carry a gold that is not "
                         f"a recommendation token (first {bad_golds[0]}); a NO_UNIQUE_PLAN "
                         "gold is the BINDING regime's -- refusing the file, not dropping "
                         "the rows")
    if n_masked:
        print(f"  solo rows: dropped {n_masked} masked")
    return out


def solo_contrast(arm: dict, solo: Sequence[dict], *, draws: int, seed: int) -> dict:
    """P2': arm group accuracy minus solo accuracy, paired on items, on
    DILEMMA (registered), the value region and all shared items."""
    def sel(recs, pred):
        return _by_item([r for r in recs if pred(r)], lambda r: r["correct"])

    debs = [{"item": d["item"], "region": d["region"], "gold_type": d["gold_type"],
             "correct": int(bool(d["correct"]))} for d in arm["debates"]]
    out = {"direction": f"{arm['tag']} minus solo", "third_seat": arm["third_seat"],
           "n_solo_rows": len(solo), "stats": {}}
    for i, (name, pred) in enumerate((
            ("dilemma", lambda r: r["gold_type"] == DILEMMA_BOTH_PREFER),
            ("value", lambda r: r["region"] == REGION_VALUE),
            ("all", lambda r: True))):
        pd = paired_delta(sel(solo, pred), sel(debs, pred), _mean, draws=draws, seed=seed + i)
        pd["negative"] = bool(pd["delta"] is not None and pd["hi"] is not None and pd["hi"] < 0)
        out["stats"][name] = pd
    d = out["stats"]["dilemma"]
    out["P2_prime"] = {
        "statistic": "neutral-arm group accuracy minus solo non-binding accuracy, DILEMMA, paired",
        "applies": arm["third_seat"] == THIRD_NEUTRAL,
        "delta": d["delta"], "lo": d["lo"], "hi": d["hi"], "n_shared_items": d["n_shared_items"],
        "holds": bool(arm["third_seat"] == THIRD_NEUTRAL and d["negative"]),
    }
    return out


def transfer_to_solo(debates: Sequence[dict], solo: Sequence[dict], *,
                     draws: int, seed: int) -> dict:
    """The sibling's transfer lift: an item is flagged when a majority of its
    cells fired (stake-blind); P(solo wrong | flagged item) - P(solo wrong | not)."""
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


# ---------------------------------------------------------------------------
# Planner-vs-neutral contrast (paired on items)
# ---------------------------------------------------------------------------


def arm_contrast(a: dict, b: dict, *, draws: int, seed: int) -> dict:
    """b minus a on the items both arms carry. When exactly one side is the
    planner arm, b is the planner (every delta reads planner minus neutral)."""
    if a["third_seat"] == THIRD_PLANNER and b["third_seat"] != THIRD_PLANNER:
        a, b = b, a
    val = lambda d: d["region"] == REGION_VALUE  # noqa: E731
    da, db = [d for d in a["debates"] if val(d)], [d for d in b["debates"] if val(d)]
    va, vb = [v for v in a["votes"] if val(v)], [v for v in b["votes"] if val(v)]
    stats: dict = {}
    stats["group_accuracy"] = paired_delta(
        _by_item(da, lambda d: int(bool(d["correct"]))),
        _by_item(db, lambda d: int(bool(d["correct"]))), _mean, draws=draws, seed=seed)
    stats["synthesis_accuracy"] = paired_delta(
        _by_item(da, lambda d: int(bool(d["syn_correct"]))),
        _by_item(db, lambda d: int(bool(d["syn_correct"]))), _mean, draws=draws, seed=seed + 1)
    for i, rnd in enumerate(SEAT_ROUNDS):
        ra = seat_records(va, a["views"], a["third_role_id"], rnd)
        rb = seat_records(vb, b["views"], b["third_role_id"], rnd)
        stats[f"third_{rnd}_accuracy"] = paired_delta(
            _by_item(ra, lambda r: int(bool(r["correct"]))),
            _by_item(rb, lambda r: int(bool(r["correct"]))), _mean, draws=draws, seed=seed + 2 + i)

    def adv_private(side, votes):
        recs = [r for s in STAKE_SEATS_NB for r in seat_records(votes, side["views"], s, "r0")
                if r["gold_type"] == DILEMMA_BOTH_PREFER]
        return _by_item(recs, lambda r: r["is_private"])

    stats["advocate_private_r0_dilemma"] = paired_delta(
        adv_private(a, va), adv_private(b, vb), _mean, draws=draws, seed=seed + 4)
    stats["advocate_accept_on_gold"] = paired_delta(
        _by_item(accept_on_gold_records(va), lambda r: r["accept"]),
        _by_item(accept_on_gold_records(vb), lambda r: r["accept"]), _mean,
        draws=draws, seed=seed + 5)
    stats["advocate_accept_on_gold_all_items"] = paired_delta(
        _by_item(accept_on_gold_records(a["votes"]), lambda r: r["accept"]),
        _by_item(accept_on_gold_records(b["votes"]), lambda r: r["accept"]), _mean,
        draws=draws, seed=seed + 5)
    stats["composite_fire_rate"] = paired_delta(
        _by_item(grip_population(da), lambda d: int(flagged(d))),
        _by_item(grip_population(db), lambda d: int(flagged(d))), _mean, draws=draws, seed=seed + 6)
    stats["stake_blind_fire_rate"] = paired_delta(
        _by_item(da, lambda d: int(fired(d))), _by_item(db, lambda d: int(fired(d))),
        _mean, draws=draws, seed=seed + 7)

    def g3_stat(vs):
        u = [v["reject"] for v in vs if v["undermined"]]
        n = [v["reject"] for v in vs if not v["undermined"]]
        return None if not u or not n else sum(u) / len(u) - sum(n) / len(n)

    def stake_by_item(votes):
        by: dict[str, list] = defaultdict(list)
        for v in votes:
            if v["role"] in STAKE_SEATS_NB and v["undermined"] is not None:
                by[v["item"]].append(v)
        return by

    stats["G3_stake_concentration"] = paired_delta(
        stake_by_item(va), stake_by_item(vb), g3_stat, draws=draws, seed=seed + 8)
    ga, p6 = stats["group_accuracy"], stats["advocate_accept_on_gold"]
    oriented = b["third_seat"] == THIRD_PLANNER and a["third_seat"] == THIRD_NEUTRAL
    return {
        "a": {"tag": a["tag"], "third_seat": a["third_seat"], "n_value_cells": len(da)},
        "b": {"tag": b["tag"], "third_seat": b["third_seat"], "n_value_cells": len(db)},
        "direction": f"{b['tag']} minus {a['tag']}", "region": REGION_VALUE,
        "planner_minus_neutral": oriented, "stats": stats,
        "P3_prime": {"statistic": "planner minus neutral group accuracy, value region, paired",
                     "delta": ga["delta"], "lo": ga["lo"], "hi": ga["hi"],
                     "n_shared_items": ga["n_shared_items"],
                     "holds": bool(oriented and ga["positive"])},
        "P6": {"statistic": "planner minus neutral advocate ACCEPT share on gold "
                            "recommendations, value region, paired",
               "delta": p6["delta"], "lo": p6["lo"], "hi": p6["hi"],
               "n_shared_items": p6["n_shared_items"],
               "holds": bool(oriented and p6["positive"])},
    }


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_CACHE_RE_NB = re.compile(
    rf"^cgd_(?P<model>.+?)_{re.escape(SCAFFOLD_NB)}_t(?P<cap>\d+)_"
    rf"(?P<arm>{'|'.join(re.escape(a) for a in ARM_OF_NB.values())})_"
    rf"(?P<item>[^_]+)_(?P<idx>\d{{2}})_(?P<round>{'|'.join(ROUNDS)})_"
    rf"(?P<role>.+)\.json$")


def index_cache_nb(out_dir: Path) -> dict[tuple, Path]:
    """(model, arm, item, idx, round, role) -> the record at the LARGEST cap."""
    best: dict[tuple, tuple[int, Path]] = {}
    if not out_dir.is_dir():
        return {}
    for p in out_dir.iterdir():
        m = _CACHE_RE_NB.match(p.name)
        if not m:
            continue
        key = (m["model"], m["arm"], m["item"], int(m["idx"]), m["round"], m["role"])
        cap = int(m["cap"])
        if key not in best or cap > best[key][0]:
            best[key] = (cap, p)
    return {k: v[1] for k, v in best.items()}


def guard_rows_from_csv(debates: Sequence[dict]) -> list[dict]:
    """17 guard rows per cell, grouped scaffold/round:role_id as the runner's
    call_guard_row; the committal set is per round AND seat (guard_token)."""
    rows = []
    for d in debates:
        mod_rid = moderator_role_id_nb(d["moderator_model"], d["third_seat"])

        def add(rnd, role, model, verdict, truncated, output_len):
            rows.append({
                "model": model, "scaffold": f"{d['scaffold']}/{rnd}:{role}",
                "arm": d["arm"], "verdict": guard_token(rnd, role, verdict),
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
    idx = index_cache_nb(out_dir)
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


# ---------------------------------------------------------------------------
# Per-arm analysis
# ---------------------------------------------------------------------------


def load_arm(rows_path: Path, votes_path: Path, *, tag: str, model: Optional[str] = None,
             games: Optional[Sequence[Game]] = None) -> dict:
    debates = load_debates_nb(rows_path, votes_path, model=model)
    votes = load_votes_nb(votes_path, model=model)
    if not debates:
        return {"tag": tag, "debates": [], "votes": votes, "views": {},
                "third_seat": None, "third_role_id": None, "cross_check": None}
    third_seats = {d["third_seat"] for d in debates}
    third_ids = {d["third_role_id"] for d in debates}
    if len(third_seats) != 1 or len(third_ids) != 1:
        raise ValueError(f"{tag}: one arm per tag expected, found third seats "
                         f"{sorted(third_seats)} / role ids {sorted(third_ids)}")
    views = build_views_nb(debates, games)
    xc = attach_views(debates, votes, views)
    return {"tag": tag, "debates": debates, "votes": votes, "views": views,
            "third_seat": next(iter(third_seats)),
            "third_role_id": next(iter(third_ids)), "cross_check": xc}


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
        "n_votes": len(votes), "cross_check": arm["cross_check"],
        "guard": guard, "refused": not guard["pass"],
    }
    if not guard["pass"]:
        return res
    res["private_action"] = private_action_share(votes, views, draws=draws, seed=seed)
    res["accuracy"] = accuracy_tables(debates, votes, views, draws=draws, seed=seed + 20)
    res["grip"] = grip(debates, votes, draws=draws, seed=seed + 30)
    res["flags"] = flag_readouts(debates, draws=draws, seed=seed + 40)
    res["accept_on_gold"] = accept_on_gold(votes, draws=draws, seed=seed + 50)
    if solo is not None:
        res["solo"] = solo_contrast(arm, solo, draws=draws, seed=seed + 60)
        res["solo"]["transfer"] = transfer_to_solo(debates, solo, draws=draws, seed=seed + 70)
    return res


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_private_action(pa: dict) -> None:
    print("\nP1' PRIVATE-ACTION SHARE (advocate names its player's option at the unique "
          "pure Nash cell; NOVERDICT = not private), item-clustered 95% CI")
    for rnd in pa["rounds"]:
        p = pa["pooled"][rnd]
        print(f"  {rnd}  pooled  DILEMMA            {_ci(p['dilemma'])}")
        print(f"  {'':<4}        DILEMMA, tempted   {_ci(p['dilemma_tempted'])}")
        print(f"  {'':<4}        DILEMMA, not       {_ci(p['dilemma_not_tempted'])}")
        print(f"  {'':<4}        all defined        {_ci(p['all_defined'])}")
        for s in STAKE_SEATS_NB:
            print(f"  {'':<4}  {s:<22} DILEMMA {_ci(pa['per_seat'][rnd][s]['dilemma'])}")
        print(f"  {'':<4}  NOVERDICT share in DILEMMA {_f(p['noverdict_share_dilemma'])}   "
              f"dominant-action share DILEMMA {_ci(pa['dominant_pooled'][rnd]['dilemma'])}")
    r = pa["P1_prime"]
    print(f"  P1' (>= 0.5, CI lower bound > 0.5): {_f(r['point'])} [{_f(r['lo'])}, "
          f"{_f(r['hi'])}] n={r['n']} items={r['n_items']} -> "
          f"{'HOLDS' if r['holds'] else 'fails'}")


def print_accuracy(acc: dict) -> None:
    g = acc["group"]
    print("\nACCURACY vs gold (group: recommendation; advocates: gold action; "
          "non-committal = wrong), item-clustered 95% CI")
    print(f"  group final      overall   {_ci(g['final']['all'])}")
    print(f"  group synthesis  overall   {_ci(g['synthesis']['all'])}")
    for seat in acc["seats"]:
        for rnd in SEAT_ROUNDS:
            print(f"  {seat:<22} {rnd:<4} {_ci(acc['per_seat'][seat][rnd]['all'])}")
    print("  by region:")
    for rg in REGIONS:
        print(f"    {rg:<16} group final {_ci(g['final']['by_region'][rg])}")
        for seat in acc["seats"]:
            print(f"    {'':<16} {seat:<22} r0 {_ci(acc['per_seat'][seat]['r0']['by_region'][rg])}")
    print("  by gold_type (the extent table): group final [CI]; then synthesis, seat r0 points")
    hdr = "".join(f"{SEAT_SHORT_NB.get(s, s[:7]) + ' r0':>11}" for s in acc["seats"])
    print(f"    {'gold_type':<32}{'games':>6}{'cells':>6}  {'group final':<30}{'synth':>7}{hdr}")
    for gt in GOLD_TYPES:
        c = g["final"]["by_gold_type"][gt]
        seats = "".join(f"{_f(acc['per_seat'][s]['r0']['by_gold_type'][gt]['point']):>11}"
                        for s in acc["seats"])
        print(f"    {gt:<32}{acc['games_by_gold_type'].get(gt, 0):>6}"
              f"{acc['cells_by_gold_type'].get(gt, 0):>6}  "
              f"{_f(c['point']):>7} [{_f(c['lo']):>5}, {_f(c['hi']):>5}]     "
              f"{_f(g['synthesis']['by_gold_type'][gt]['point']):>7}{seats}")
    for seat, tn in acc["third_nash_pull"].items():
        print(f"  third seat Nash-pull ({seat}) r0 {_ci(tn['r0']['all'])}  "
              f"DILEMMA {_ci(tn['r0']['by_gold_type'][DILEMMA_BOTH_PREFER])}")
    print(f"  verdict counts: {g['verdict_counts']}   noncommittal share "
          f"{_f(g['noncommittal_share'])}")
    if acc["type_named_cell_share"] is not None:
        print(f"  cells with a game-type stem quoted into any prompt (transcript leak "
              f"covariate) {_f(acc['type_named_cell_share'])}")


def print_grip(g: dict) -> None:
    print("\nSTAKE-GRIP SCREEN (analyze_dilemma_grip thresholds; stake = strictly "
          "profitable unilateral deviation from the recommended cell)")
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
    print(f"  {'seat':<24}{'objected_r3':>12}{'reject_r4':>10}{'undermined':>12}{'n':>6}")
    for s, v in g["per_seat"].items():
        print(f"  {s:<24}{_f(v['objected_r3']):>12}{_f(v['reject_r4']):>10}"
              f"{_f(v['undermined']):>12}{v['n']:>6}")
    print(f"\n  GRIP: {'yes' if g['grip'] else 'no'}")


def print_flags(fl: dict) -> None:
    for name in ("composite", "stake_blind"):
        fc = fl[name]
        print(f"\nFLAG [{name}] ({fc['counter']}): fire rate and error lift on the group's "
              f"own verdict (lift read at >= {fc['min_fired']} fired)")
        print(f"  fire rate  all              {_ci(fc['fire_rate']['all'])}")
        for rg in REGIONS:
            print(f"  fire rate  {rg:<16} {_ci(fc['fire_rate']['by_region'][rg])}")
        print(f"  fire rate  {'DILEMMA':<16} "
              f"{_ci(fc['fire_rate']['by_gold_type'][DILEMMA_BOTH_PREFER])}")
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
    p5 = fl["P5"]
    print(f"  P5 flooding (composite fire on DILEMMA > 0.5 AND value lift CI includes 0): "
          f"fire {_f(p5['fire_rate_dilemma']['point'])}, lift "
          f"{_f(p5['lift_value']['delta'], '+.3f')} [{_f(p5['lift_value']['lo'], '+.3f')}, "
          f"{_f(p5['lift_value']['hi'], '+.3f')}] -> "
          f"{'FLOODING' if p5['flooding'] else ('not flooding' if p5['computable'] else 'not computable')}")


def print_accept_on_gold(ag: dict) -> None:
    print("\nP6 ADVOCATE ACCEPT SHARE ON GOLD RECOMMENDATIONS (r4 votes on cells whose "
          "group verdict is the gold), item-clustered 95% CI")
    print(f"  pooled  all              {_ci(ag['pooled']['all'])}")
    for rg in REGIONS:
        print(f"  pooled  {rg:<16} {_ci(ag['pooled']['by_region'][rg])}")
    print(f"  pooled  {'DILEMMA':<16} {_ci(ag['pooled']['by_gold_type'][DILEMMA_BOTH_PREFER])}")
    for s in STAKE_SEATS_NB:
        print(f"  {s:<24} {_ci(ag['per_seat'][s])}")
    print(f"  undermined at the gold cell {_ci(ag['by_undermined']['undermined'])}")
    print(f"  not undermined              {_ci(ag['by_undermined']['not_undermined'])}")


def print_solo(so: dict) -> None:
    print(f"\nSOLO NON-BINDING COMPARATOR: {so['direction']} (paired on items; "
          f"{so['n_solo_rows']} solo rows)")
    print(f"  {'slice':<10}{'solo':>8}{'arm':>8}{'delta':>9}{'lo':>9}{'hi':>9}  n_items")
    for name, s in so["stats"].items():
        print(f"  {name:<10}{_f(s['a']):>8}{_f(s['b']):>8}{_f(s['delta'], '+.3f'):>9}"
              f"{_f(s['lo'], '+.3f'):>9}{_f(s['hi'], '+.3f'):>9}  {s['n_shared_items']}"
              f"{'   NEGATIVE' if s['negative'] else ''}")
    p2 = so["P2_prime"]
    print(f"  P2' (neutral arm below solo on DILEMMA, CI excl. 0): "
          f"{'HOLDS' if p2['holds'] else ('fails' if p2['applies'] else 'n/a on this arm')}")
    t = so["transfer"]
    print(f"  transfer lift onto solo (stake-blind flag): P(solo wrong|flagged item)="
          f"{_f(t['p_wrong_flagged'])} vs not={_f(t['p_wrong_unflagged'])}  lift "
          f"{_f(t['lift'], '+.3f')} [{_f(t['lo'], '+.3f')}, {_f(t['hi'], '+.3f')}]  "
          f"items {t['n_items']} flagged {t['n_flagged_items']}")


def print_contrast(c: dict) -> None:
    print("\n" + "=" * 72)
    print(f"THIRD-SEAT CONTRAST on the {c['region']} region: {c['direction']}  "
          f"({c['b']['third_seat']} minus {c['a']['third_seat']}; paired on items)")
    print("=" * 72)
    print(f"  {'statistic':<34}{'a':>8}{'b':>8}{'delta':>9}{'lo':>9}{'hi':>9}  n_items")
    for name, s in c["stats"].items():
        print(f"  {name:<34}{_f(s['a']):>8}{_f(s['b']):>8}{_f(s['delta'], '+.3f'):>9}"
              f"{_f(s['lo'], '+.3f'):>9}{_f(s['hi'], '+.3f'):>9}  {s['n_shared_items']}"
              f"{'   POSITIVE' if s['positive'] else ''}")
    for key in ("P3_prime", "P6"):
        r = c[key]
        print(f"  {key.replace('_prime', chr(39))}: {r['statistic']} -> "
              f"{_f(r['delta'], '+.3f')} [{_f(r['lo'], '+.3f')}, {_f(r['hi'], '+.3f')}] "
              f"{'HOLDS' if r['holds'] else 'fails'}"
              f"{'' if c['planner_minus_neutral'] else '  (not planner-vs-neutral; reading n/a)'}")


def print_arm_report(res: dict) -> None:
    print("\n" + "=" * 72)
    print(f"GAME NON-BINDING -- tag {res['tag']}  arm {res['arm']}  third seat "
          f"{res['third_seat']} ({res['third_role_id']})  models {res['models']}")
    print(f"  rows={res['n_debates']}  items={res['n_items']}  votes={res['n_votes']}")
    xc = res.get("cross_check") or {}
    if xc:
        print(f"  stake_undermined recomputed from the registered definition on every vote; "
              f"CSV column {'present and consistent' if xc['csv_has_stake_undermined'] else 'absent'}")
    print("=" * 72)
    print_guard(res["guard"])
    if res["refused"]:
        print("\n  *** GUARD FAILED: no private-action, accuracy, grip, flag or P-reading "
              "is printed for this arm. Raise the failing round's token cap and "
              "re-run. ***")
        return
    print_private_action(res["private_action"])
    print_accuracy(res["accuracy"])
    print_grip(res["grip"])
    print_flags(res["flags"])
    print_accept_on_gold(res["accept_on_gold"])
    if "solo" in res:
        print_solo(res["solo"])


def readings_summary(results: Sequence[dict], contrast: Optional[dict]) -> dict:
    """The five pre-declared readings in one block, mechanically applied."""
    out: dict = {"P1_prime": {}, "P2_prime": None, "P3_prime": None, "P5": {}, "P6": {}}
    for r in results:
        if r["refused"]:
            continue
        out["P1_prime"][r["tag"]] = r["private_action"]["P1_prime"]
        out["P5"][r["tag"]] = r["flags"]["P5"]
        out["P6"][r["tag"]] = {"accept_on_gold_value":
                               r["accept_on_gold"]["pooled"]["by_region"][REGION_VALUE],
                               "accept_on_gold_all": r["accept_on_gold"]["pooled"]["all"]}
        if "solo" in r and r["solo"]["P2_prime"]["applies"]:
            out["P2_prime"] = r["solo"]["P2_prime"]
    if contrast is not None:
        out["P3_prime"] = contrast["P3_prime"]
        out["P6"]["planner_minus_neutral"] = contrast["P6"]
    return out


def print_readings(rd: dict) -> None:
    print("\n" + "=" * 72)
    print("PRE-DECLARED READINGS (Addendum 16.13c), applied mechanically")
    print("=" * 72)
    for tag, p in rd["P1_prime"].items():
        print(f"  P1' lock reappears   [{tag}]  {_f(p['point'])} [{_f(p['lo'])}, {_f(p['hi'])}]"
              f"  -> {'HOLDS' if p['holds'] else 'fails'}")
    if not rd["P1_prime"]:
        print("  P1' not computable (no arm passed the guard)")
    p = rd["P2_prime"]
    print(f"  P2' collective drifts  " + (
        f"{_f(p['delta'], '+.3f')} [{_f(p['lo'], '+.3f')}, {_f(p['hi'], '+.3f')}]  -> "
        f"{'HOLDS' if p['holds'] else 'fails'}" if p else "not computable (needs the neutral "
        "arm and --solo-rows)"))
    p = rd["P3_prime"]
    print(f"  P3' mediator holds     " + (
        f"{_f(p['delta'], '+.3f')} [{_f(p['lo'], '+.3f')}, {_f(p['hi'], '+.3f')}]  -> "
        f"{'HOLDS' if p['holds'] else 'fails'}   (PRIMARY)" if p else
        "not computable (needs both arms)"))
    for tag, p in rd["P5"].items():
        print(f"  P5  flooding         [{tag}]  fire(DILEMMA) {_f(p['fire_rate_dilemma']['point'])}"
              f"  lift(value) {_f(p['lift_value']['delta'], '+.3f')} -> "
              f"{'FLOODING' if p['flooding'] else ('not flooding' if p['computable'] else 'not computable')}")
    for tag, p in rd["P6"].items():
        if tag == "planner_minus_neutral":
            print(f"  P6  steering (delta)   {_f(p['delta'], '+.3f')} [{_f(p['lo'], '+.3f')}, "
                  f"{_f(p['hi'], '+.3f')}]  -> {'HOLDS' if p['holds'] else 'fails'}")
        else:
            print(f"  P6  accept on gold   [{tag}]  value {_ci(p['accept_on_gold_value'])}")
    if "planner_minus_neutral" not in rd["P6"]:
        print("  P6  delta not computable (needs both arms)")


# ---------------------------------------------------------------------------
# Selftest: synthetic records from the real games written through the
# documented CSV columns, expectations hand-computed from the matrices
# ---------------------------------------------------------------------------


def _wrong_recommendation(view: ItemView) -> str:
    """A committal recommendation that is not the gold: the Nash cell when
    the gold is a non-Nash PLAN (the drift a naive collective would show),
    else a player's best cell, else the first PLAN that differs."""
    gold = gold_recommendation(view)
    nash = unique_nash_cell(view)
    if gold in PLAN_VERDICTS and nash is not None:
        nv = plan_for_cell(nash, view.row_perm, view.col_perm)
        if nv != gold:
            return nv
    for cand in (view.row_best_verdict, view.col_best_verdict) + PLAN_VERDICTS:
        if cand != gold:
            return cand
    raise AssertionError("unreachable")


def _synthetic_arm(games: Sequence[Game], third_seat: str, *, wrong_every: Optional[int],
                   model: str = "stub-model", extra_value_sample: bool = True,
                   noverdict_r0_every: Optional[int] = None,
                   advocates_r0: str = "private", dilemma_wrong_samples: Sequence[int] = (),
                   advocates_persuaded: bool = False,
                   write_stake_col: bool = True) -> tuple[list[dict], list[dict]]:
    """Planted behaviour: advocates' r0 = their player's private action
    ('private') or gold action ('gold'), falling back to the other when
    undefined; r2 = gold action; third r0/r2 = gold; synthesis = gold;
    group = gold except on every `wrong_every`-th game (id order), where it
    is _wrong_recommendation; when `dilemma_wrong_samples` is given, DILEMMA
    cells are wrong exactly on those sample indices (and right otherwise). Objections: R iff wrong, C iff undermined,
    third iff wrong. Rejects: advocates iff undermined (never when
    `advocates_persuaded`), third iff wrong."""
    third_id = THIRD_ROLE_ID_NB[third_seat]
    arm = ARM_OF_NB[third_seat]
    rows, votes = [], []
    for gi, g in enumerate(sorted(games, key=lambda x: x.game_id)):
        samples = (0, 1) if (extra_value_sample and g.region == REGION_VALUE) else (0,)
        wrong_g = wrong_every is not None and gi % wrong_every == wrong_every - 1
        for idx in samples:
            view = render_item(g, idx)
            gold = gold_recommendation(view)
            if g.gold_type == DILEMMA_BOTH_PREFER and dilemma_wrong_samples:
                wrong = idx in dilemma_wrong_samples
            else:
                wrong = wrong_g
            verdict = _wrong_recommendation(view) if wrong else gold
            r0, r2 = {}, {}
            for seat in STAKE_SEATS_NB:
                st = SEAT_STAKE_NB[seat]
                ga, pa = gold_action(view, st), private_action(view, st)
                first, second = (pa, ga) if advocates_r0 == "private" else (ga, pa)
                r0[seat] = first or second or ACT_1
                r2[seat] = ga or ACT_2
            r0[third_id] = gold
            r2[third_id] = gold
            if noverdict_r0_every is not None and gi % noverdict_r0_every == 0:
                r0[ROW_SEAT_NB] = NOVERDICT
            und = {s: stake_undermined_nb(view, verdict, SEAT_STAKE_NB[s]) for s in r0}
            objected = {ROW_SEAT_NB: wrong, COL_SEAT_NB: bool(und[COL_SEAT_NB]), third_id: wrong}
            rejects = {ROW_SEAT_NB: bool(und[ROW_SEAT_NB]) and not advocates_persuaded,
                       COL_SEAT_NB: bool(und[COL_SEAT_NB]) and not advocates_persuaded,
                       third_id: wrong}
            labels = {s: ("ACCEPT_WITH_MODIFICATION" if objected[s] else "ACCEPT") for s in r0}
            vote = {s: ("REJECT" if rejects[s] else "ACCEPT") for s in r0}
            order = (ROW_SEAT_NB, COL_SEAT_NB, third_id)
            ga_r, ga_c = gold_action(view, "row"), gold_action(view, "col")
            rows.append({
                "model": model, "scaffold": SCAFFOLD_NB, "arm": arm,
                "third_seat": third_seat, "third_role_id": third_id,
                "item_id": g.game_id, "game_id": g.game_id, "sample_idx": idx,
                "gold_type": g.gold_type, "region": g.region, "gold_verdict": gold,
                "gold_action_row": ga_r or "", "gold_action_col": ga_c or "",
                "row_perm": _perm_str(view.row_perm), "col_perm": _perm_str(view.col_perm),
                "row_name": view.row_name, "col_name": view.col_name,
                "verdict": verdict, "output_len": 300, "truncated": 0,
                "moderator_model": model, "synthesis_verdict": gold, "synthesis_len": 250,
                "verdict_revised": int(verdict != gold),
                "r0_verdict_row_advocate": r0[ROW_SEAT_NB],
                "r0_verdict_col_advocate": r0[COL_SEAT_NB], "r0_verdict_third": r0[third_id],
                "r1_verdict_row_advocate": r0[ROW_SEAT_NB],
                "r1_verdict_col_advocate": r0[COL_SEAT_NB], "r1_verdict_third": r0[third_id],
                "r2_verdict_row_advocate": r2[ROW_SEAT_NB],
                "r2_verdict_col_advocate": r2[COL_SEAT_NB], "r2_verdict_third": r2[third_id],
                "n_objectors": sum(objected.values()),
                "n_reject": sum(1 for s in order if vote[s] == "REJECT"),
                "n_r3_unparsed": 0, "n_vote_unparsed": 0,
                "unanimous_accept": int(all(vote[s] == "ACCEPT" for s in order)),
                "n_calls_type_named": 0, "type_named_first_round": "",
            })
            for s in order:
                votes.append({
                    "model": model, "scaffold": SCAFFOLD_NB, "arm": arm,
                    "item_id": g.game_id, "sample_idx": idx,
                    "gold_type": g.gold_type, "region": g.region, "gold_verdict": gold,
                    "role_id": s, "role_stake": SEAT_STAKE_NB[s],
                    "group_verdict": verdict, "synthesis_verdict": gold,
                    "gold_action": (ga_r if s == ROW_SEAT_NB else ga_c if s == COL_SEAT_NB else "") or "",
                    "r0_verdict": r0[s], "r1_verdict": r0[s], "r2_verdict": r2[s],
                    "r3_label": labels[s], "objected_r3": int(objected[s]),
                    "vote": vote[s], "reject": int(vote[s] == "REJECT"),
                    "stake_undermined": ("" if (und[s] is None or not write_stake_col)
                                         else int(und[s])),
                    "finish_reason": "stop", "truncated": 0, "output_len": 120,
                })
    return rows, votes


ROW_FIELDS_SYNTH = ROWS_REQUIRED + ROWS_OPTIONAL
VOTE_FIELDS_SYNTH = VOTES_REQUIRED + ("scaffold",) + VOTES_OPTIONAL


def _nb_namespace_files(out_dir: Path = OUT_DIR) -> dict[str, int]:
    """Files in the cache dir carrying any 16.13c name: the tag as a prefix,
    every other name as a substring of a cgd_ call record's name."""
    counts = {f"tag:{TAG_PREFIX_NB}": len(cache_files_with_prefix(TAG_PREFIX_NB, out_dir))}
    cgd = [p.name for p in cache_files_with_prefix("cgd_", out_dir)]
    for name in NB_NAMESPACES:
        if name == TAG_PREFIX_NB:
            continue
        counts[name] = sum(1 for n in cgd if f"_{name}_" in n or n.endswith(f"_{name}.json")
                           or f"_{name}-" in n)
    return counts


def _runner_module():
    """scripts.run_crowdgold_game_nb if it exists on disk, else None (imported
    lazily, only by the selftest, so this module never depends on the runner
    and the runner may import from here without a cycle)."""
    p = Path(__file__).resolve().parent / "run_crowdgold_game_nb.py"
    if not p.is_file():
        return None
    return importlib.import_module("scripts.run_crowdgold_game_nb")


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    ns_before = _nb_namespace_files()
    runner = _runner_module()

    # ---- registered names ----------------------------------------------------------
    check("registered names: scaffold game_action_ncot, arms third_neutral_nb / "
          "third_planner_nb, role ids *_nb, moderator prefix modgamenb, tag cgnb_game "
          "(all imported from tmg_games_nb)",
          SCAFFOLD_NB == "game_action_ncot"
          and ARM_OF_NB == {"neutral": "third_neutral_nb", "planner": "third_planner_nb"}
          and STAKE_SEATS_NB == ("player_r_advocate_nb", "player_c_advocate_nb")
          and THIRD_ROLE_ID_NB == {"neutral": "neutral_reader_nb", "planner": "plan_mediator_nb"}
          and MODERATOR_PREFIX_NB == "modgamenb" and TAG_PREFIX_NB == "cgnb_game"
          and moderator_role_id_nb("m/x", "planner") == "modgamenb-planner-m_x")
    check("names are distinct from the binding regime's (no cache record can be shared); "
          "the tag prefix does not extend the binding 'cg_game'",
          not any(n in ("game_plan_ncot", "third_neutral", "third_planner", "player_r_advocate",
                        "player_c_advocate", "neutral_reader", "plan_mediator", "modgame",
                        "cg_game") for n in NB_NAMESPACES)
          and not TAG_PREFIX_NB.startswith("cg_game"))
    import scripts.tmg_games_nb as tnb
    check("item-level definitions ARE tmg_games_nb's (aliases, not copies): gold "
          "recommendation, gold / private / dominant action, stake_undermined_nb",
          gold_recommendation is tnb.nb_gold_verdict and gold_action is tnb.gold_act_for
          and private_action is tnb.private_act_for and dominant_action is tnb.dominant_act_for
          and stake_undermined_nb is tnb.stake_undermined_nb
          and RECOMMENDATION_SET == tnb.NB_VERDICT_SET and ACTION_SET == tnb.ACT_VERDICTS)
    try:
        import scripts.run_game_singleagent as solo_mod
    except ImportError:
        solo_mod = None
    check("SOLO_SCAFFOLD_NB is run_game_singleagent's non-binding scaffold (tmg_solo_nb) "
          "and differs from its binding one",
          solo_mod is not None and solo_mod.SCAFFOLD_NAME_NB == SOLO_SCAFFOLD_NB
          and solo_mod.SCAFFOLD_NAME != SOLO_SCAFFOLD_NB)
    if runner is None:
        check("every 16.13c namespace is EMPTY in divergence_study_outputs (the runner does "
              f"not exist yet, so nothing may carry these names): {ns_before}",
              all(v == 0 for v in ns_before.values()))
    else:
        print(f"      runner present; namespace counts {ns_before} (emptiness before the "
              "first call is the runner selftest's assertion)")
        want = {"SCAFFOLD_NAME": SCAFFOLD_NB, "TAG_PREFIX": TAG_PREFIX_NB,
                "MODERATOR_PREFIX": MODERATOR_PREFIX_NB, "ARM_OF": ARM_OF_NB,
                "STAKE_SEATS": STAKE_SEATS_NB}
        got = {k: getattr(runner, k, None) for k in want}
        check(f"runner constants match the analyzer's registered names ({got})",
              all(got[k] == v for k, v in want.items()))
        rf = set(getattr(runner, "ROW_FIELDS", ()))
        vf = set(getattr(runner, "VOTE_FIELDS", ()))
        check("runner ROW_FIELDS / VOTE_FIELDS carry every required column "
              f"(missing rows {sorted(set(ROWS_REQUIRED) - rf)}, votes "
              f"{sorted(set(VOTES_REQUIRED) - vf)})",
              set(ROWS_REQUIRED) <= rf and set(VOTES_REQUIRED) <= vf)
        third = getattr(runner, "THIRD_ROLE", {})
        check("runner third-seat role ids are neutral_reader_nb / plan_mediator_nb",
              {k: getattr(v, "role_id", None) for k, v in third.items()} == THIRD_ROLE_ID_NB)
        check("runner instruments carry the same registered names and tokens",
              getattr(runner, "ACT_INSTRUMENT", None) == ACTION_INSTRUMENT
              and getattr(runner, "NB_PLAN_INSTRUMENT", None) == RECOMMENDATION_INSTRUMENT
              and tuple(getattr(runner, "ACT_VERDICTS", ())) == ACTION_SET
              and tuple(getattr(runner, "NB_VERDICT_SET", ())) == RECOMMENDATION_SET
              and getattr(runner, "NO_RECOMMENDATION", None) == NO_RECOMMENDATION)
        check("runner moderator_role_id equals the analyzer's for both variants",
              all(runner.moderator_role_id("m", ts) == moderator_role_id_nb("m", ts)
                  for ts in THIRD_SEATS))
    check("thresholds are the sibling's (0.50 / 0.05 / +0.20)",
          (G1_MAX_FIRE, G2_MIN_REJECT, G3_MIN_CONC) == (0.50, 0.05, 0.20))

    # ---- instruments -----------------------------------------------------------------
    check("instruments: ACT_1 / ACT_2 and PLAN_xy / NO_RECOMMENDATION registered; "
          "committal sets exclude UNRESOLVED; NO_UNIQUE_PLAN is not a token here",
          committal_tokens(ACTION_INSTRUMENT) == ACTION_SET
          and committal_tokens(RECOMMENDATION_INSTRUMENT) == RECOMMENDATION_SET
          and UNRESOLVED not in ACTION_SET + RECOMMENDATION_SET
          and "NO_UNIQUE_PLAN" not in RECOMMENDATION_SET)
    check("round_instrument: advocates ACT on r0/r1/r2, third seat and moderator "
          "recommendation, r3 label, r4 vote",
          all(round_instrument(r, ROW_SEAT_NB) == ACTION_INSTRUMENT for r in ("r0", "r1", "r2"))
          and all(round_instrument(r, "neutral_reader_nb") == RECOMMENDATION_INSTRUMENT
                  for r in ("r0", "r1", "r2"))
          and round_instrument("synthesis", "modgamenb-neutral-m") == RECOMMENDATION_INSTRUMENT
          and round_instrument("integration", "modgamenb-planner-m") == RECOMMENDATION_INSTRUMENT
          and round_instrument("r3_label", ROW_SEAT_NB) == LABEL_INSTRUMENT_NB
          and round_instrument("r4_vote", "plan_mediator_nb") == VOTE_INSTRUMENT_NB)
    strings = list(NB_NAMESPACES) + list(ACTION_SET) + list(RECOMMENDATION_SET) + [
        ACTION_INSTRUMENT, RECOMMENDATION_INSTRUMENT, LABEL_INSTRUMENT_NB, VOTE_INSTRUMENT_NB]
    check("no forbidden word in any name, token or instrument this module defines "
          "(the analyzer renders no prompt; gold_type labels are report-only metadata)",
          not any(forbidden_hits(s) for s in strings))

    # ---- guard token coding (per round AND seat) ----------------------------------------
    check("guard_token: ACT on advocate PLAN rounds, PLAN/NO_RECOMMENDATION on the third "
          "seat and moderator; a PLAN from an advocate, an ACT from the third seat, "
          "NO_UNIQUE_PLAN, UNRESOLVED and blanks are NOVERDICT; r3 / r4 as the sibling",
          guard_token("r0", ROW_SEAT_NB, ACT_1) == ACT_1
          and guard_token("r2", COL_SEAT_NB, ACT_2) == ACT_2
          and guard_token("r0", ROW_SEAT_NB, "PLAN_12") == NOVERDICT
          and guard_token("r0", "neutral_reader_nb", "PLAN_12") == "PLAN_12"
          and guard_token("r1", "plan_mediator_nb", NO_RECOMMENDATION) == NO_RECOMMENDATION
          and guard_token("r0", "neutral_reader_nb", ACT_1) == NOVERDICT
          and guard_token("synthesis", "modgamenb-neutral-m", "NO_UNIQUE_PLAN") == NOVERDICT
          and guard_token("integration", "modgamenb-neutral-m", "PLAN_21") == "PLAN_21"
          and guard_token("r2", ROW_SEAT_NB, "UNRESOLVED") == NOVERDICT
          and guard_token("r4_vote", ROW_SEAT_NB, "ACCEPT") == "ACCEPT"
          and guard_token("r4_vote", ROW_SEAT_NB, "UNRESOLVED") == NOVERDICT
          and guard_token("r4_vote", ROW_SEAT_NB, "ACCEPT_WITH_MODIFICATION") == NOVERDICT
          and guard_token("r3_label", ROW_SEAT_NB, "ACCEPT_WITH_MODIFICATION")
          == "ACCEPT_WITH_MODIFICATION"
          and guard_token("r3_label", ROW_SEAT_NB, "") == NOVERDICT
          and guard_token("r0", ROW_SEAT_NB, None) == NOVERDICT)

    # ---- scoring ----------------------------------------------------------------------
    check("recommendation_correct: NO_RECOMMENDATION vs cannot-assist gold is CORRECT; a "
          "PLAN vs that gold is wrong; NOVERDICT / UNRESOLVED / NO_UNIQUE_PLAN / blank are "
          "wrong (in the denominator); committal_* makes non-committal None",
          recommendation_correct(NO_RECOMMENDATION, NO_RECOMMENDATION) is True
          and recommendation_correct("PLAN_11", NO_RECOMMENDATION) is False
          and recommendation_correct("PLAN_12", "PLAN_12") is True
          and recommendation_correct(NO_RECOMMENDATION, "PLAN_12") is False
          and recommendation_correct(NOVERDICT, "PLAN_12") is False
          and recommendation_correct("UNRESOLVED", NO_RECOMMENDATION) is False
          and recommendation_correct("NO_UNIQUE_PLAN", NO_RECOMMENDATION) is False
          and recommendation_correct("", "PLAN_12") is False
          and recommendation_correct("PLAN_12", "NO_UNIQUE_PLAN") is None
          and committal_recommendation_correct(NOVERDICT, "PLAN_12") is None
          and committal_recommendation_correct("NO_UNIQUE_PLAN", "PLAN_12") is None
          and committal_recommendation_correct(NO_RECOMMENDATION, NO_RECOMMENDATION) is True)
    check("action_correct: vs the gold action; None (excluded) without a gold cell; "
          "NOVERDICT wrong",
          action_correct(ACT_1, ACT_1) is True and action_correct(ACT_2, ACT_1) is False
          and action_correct(NOVERDICT, ACT_1) is False and action_correct(ACT_1, None) is None)

    # ---- data-level definitions ---------------------------------------------------------
    try:
        games = load_games()
    except Exception as e:  # noqa: BLE001
        check(f"load_games() ({type(e).__name__}: {e})", False)
        print(f"\n{len(fails)} FAILED")
        return 1
    check("144 games load", len(games) == 144)
    views_all = {(g.game_id, i): render_item(g, i) for g in games for i in range(3)}
    check("gold_recommendation: PLAN_xy on the 113 games with a gold cell, "
          "NO_RECOMMENDATION on the 31 cannot-assist games, never NO_UNIQUE_PLAN",
          sum(1 for v in views_all.values() if gold_recommendation(v) == NO_RECOMMENDATION) == 31 * 3
          and all((gold_recommendation(v) == NO_RECOMMENDATION) == (v.region == REGION_CANNOT_ASSIST)
                  for v in views_all.values()))
    check("gold_action maps the gold cell through the DISPLAYED permutation (equals the "
          "PLAN's digit for that player) and is None exactly on cannot-assist",
          all((gold_action(v, "row") is None) == (v.region == REGION_CANNOT_ASSIST)
              and (gold_action(v, "row"), gold_action(v, "col"))
              == (plan_action(gold_recommendation(v), "row"), plan_action(gold_recommendation(v), "col"))
              for v in views_all.values()))
    dil = [v for v in views_all.values() if v.gold_type == DILEMMA_BOTH_PREFER]
    asr = [v for v in views_all.values() if v.gold_type == "ASSURANCE"]
    check("private_action is defined on every DILEMMA rendering and differs from gold_action "
          "for BOTH players on every one of them (7 games x 3 samples); undefined on ASSURANCE",
          all(private_action(v, p) is not None and private_action(v, p) != gold_action(v, p)
              for v in dil for p in ("row", "col"))
          and all(private_action(v, p) is None for v in asr for p in ("row", "col")))
    check("private_action is the Nash-cell action: on neutral-region games it EQUALS the "
          "gold action (gold is the unique Nash cell there)",
          all(private_action(v, p) == gold_action(v, p) for v in views_all.values()
              if v.region == "neutral" for p in ("row", "col")))
    n_dom = Counter((v.gold_type, dominant_action(v, "row") is not None,
                     dominant_action(v, "col") is not None) for v in views_all.values()
                    if v.sample_idx == 0)
    check("dominant_action: both players strictly dominant on exactly 1 DILEMMA game, one "
          "player on the other 6, none on ASSURANCE; where defined it equals private_action",
          n_dom[(DILEMMA_BOTH_PREFER, True, True)] == 1
          and n_dom[(DILEMMA_BOTH_PREFER, True, False)] + n_dom[(DILEMMA_BOTH_PREFER, False, True)] == 6
          and not any(k[0] == "ASSURANCE" and (k[1] or k[2]) for k in n_dom)
          and all(dominant_action(v, p) in (None, private_action(v, p))
                  for v in views_all.values() for p in ("row", "col")))
    v111 = views_all[("111", 0)]
    v221 = views_all[("221", 0)]
    v322 = views_all[("322", 0)]
    g111, g221, g322 = (gold_recommendation(x) for x in (v111, v221, v322))
    check("stake_undermined_nb on hand cases: game 111 gold undermines BOTH; game 221 gold "
          "undermines the column player only (row already gets 4); ASSURANCE 322 gold "
          "undermines neither; the Nash cell undermines no one",
          stake_undermined_nb(v111, g111, "row") is True and stake_undermined_nb(v111, g111, "col") is True
          and stake_undermined_nb(v221, g221, "row") is False
          and stake_undermined_nb(v221, g221, "col") is True
          and stake_undermined_nb(v322, g322, "row") is False
          and stake_undermined_nb(v322, g322, "col") is False
          and all(stake_undermined_nb(v111, plan_for_cell(unique_nash_cell(v111), v111.row_perm,
                                                          v111.col_perm), p) is False
                  for p in ("row", "col")))
    check("stake_undermined_nb: None on NO_RECOMMENDATION / NOVERDICT / UNRESOLVED / "
          "NO_UNIQUE_PLAN; the third seat (stake none) is False on any committal verdict "
          "and None otherwise",
          stake_undermined_nb(v111, NO_RECOMMENDATION, "row") is None
          and stake_undermined_nb(v111, NOVERDICT, "col") is None
          and stake_undermined_nb(v111, "UNRESOLVED", "row") is None
          and stake_undermined_nb(v111, "NO_UNIQUE_PLAN", "row") is None
          and stake_undermined_nb(v111, g111, "none") is False
          and stake_undermined_nb(v111, NO_RECOMMENDATION, "none") is False
          and stake_undermined_nb(v111, NOVERDICT, "none") is None)
    both = sum(1 for g in games if g.gold_type == DILEMMA_BOTH_PREFER
               and undermined_at_gold(render_item(g, 0), "row")
               and undermined_at_gold(render_item(g, 0), "col"))
    atleast = sum(1 for g in games if g.gold_type == DILEMMA_BOTH_PREFER
                  and (undermined_at_gold(render_item(g, 0), "row")
                       or undermined_at_gold(render_item(g, 0), "col")))
    check("PINNED against the registration's 'BOTH by construction': under the registered "
          "definition the gold cell undermines at least one advocate on all 7 DILEMMA games "
          f"but BOTH on only {both} of them (game 111); neither on any ASSURANCE game",
          atleast == 7 and both == 1
          and not any(undermined_at_gold(v, p) for v in asr for p in ("row", "col")))
    if runner is not None:
        Item = runner.NBGameItem
        agree = True
        for v in views_all.values():
            it = Item(v)
            agree = agree and it.gold_verdict == gold_recommendation(v)
            for st, seat in (("row", ROW_SEAT_NB), ("col", COL_SEAT_NB)):
                agree = agree and runner.gold_action(it, st) == gold_action(v, st)
                agree = agree and runner.private_action(it, st) == private_action(v, st)
                agree = agree and runner.dominant_action(it, st) == dominant_action(v, st)
                for verdict in RECOMMENDATION_SET + (NOVERDICT, "NO_UNIQUE_PLAN"):
                    agree = agree and (runner.stake_undermined(seat, verdict, it)
                                       == stake_undermined_nb(v, verdict, st))
            for verdict in RECOMMENDATION_SET + (NOVERDICT,):
                agree = agree and (runner.stake_undermined(THIRD_ROLE_ID_NB[THIRD_NEUTRAL], verdict, it)
                                   == stake_undermined_nb(v, verdict, "none"))
        check("runner and analyzer agree on gold recommendation, gold_action, private_action, "
              "dominant_action and stake_undermined on every rendering (144 x 3) and every "
              "verdict token (the runner's dominant-else-Nash private action equals the "
              "Nash-cell action: a strictly dominant action forces a unique Nash cell)", agree)
    check("undermined_at_gold is invariant to the rendering (a permutation cannot change "
          "who is tempted)",
          all(undermined_at_gold(views_all[(gid, 0)], p) == undermined_at_gold(views_all[(gid, 2)], p)
              for gid in {k[0] for k in views_all} for p in ("row", "col")))

    # ---- sensor predicates (as the sibling) -----------------------------------------------
    d_third = {"third_role_id": "neutral_reader_nb", "obj_neutral_reader_nb": True}
    d_misloc = {"third_role_id": "plan_mediator_nb", f"obj_{ROW_SEAT_NB}": True,
                f"undermined_{ROW_SEAT_NB}": False}
    d_clean = {"third_role_id": "neutral_reader_nb", f"obj_{ROW_SEAT_NB}": True,
               f"undermined_{ROW_SEAT_NB}": True, "obj_neutral_reader_nb": False}
    d_undef = {"third_role_id": "neutral_reader_nb", f"obj_{COL_SEAT_NB}": True,
               f"undermined_{COL_SEAT_NB}": None}
    check("third-seat objection alone flags (either variant's role id)", flagged(d_third))
    check("mis-localised advocate objection alone flags", flagged(d_misloc))
    check("a correctly-localised objection with no third objection does not flag",
          not flagged(d_clean))
    check("an objection with an UNDEFINED (None) undermined flag is not mis-localised",
          not flagged(d_undef) and not mislocalised(d_undef))

    # ---- synthetic neutral arm through the documented CSV columns --------------------------
    WRONG_EVERY = 3
    rows_n, votes_n = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY,
                                     dilemma_wrong_samples=(1,))
    check("synthetic rows / votes carry every required column and the optional ones",
          all(k in rows_n[0] for k in ROWS_REQUIRED) and all(k in votes_n[0] for k in VOTES_REQUIRED)
          and not any(forbidden_hits(r["verdict"]) for r in rows_n))
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        _write_csv(tdp / "cgnb_game_neutral_rows.csv", rows_n, ROW_FIELDS_SYNTH)
        _write_csv(tdp / "cgnb_game_neutral_votes.csv", votes_n, VOTE_FIELDS_SYNTH)
        arm = load_arm(tdp / "cgnb_game_neutral_rows.csv", tdp / "cgnb_game_neutral_votes.csv",
                       tag="cgnb_game_neutral", games=games)
        n_cells = len(rows_n)
        check(f"load_arm: {n_cells} debates (144 games + a second sample on the 16 value games), "
              "144 items, views re-rendered and consistent with the CSV, neutral_reader_nb",
              len(arm["debates"]) == n_cells == 160
              and len({d["item"] for d in arm["debates"]}) == 144
              and len(arm["views"]) == 160 and arm["third_role_id"] == "neutral_reader_nb")
        check("attach_views recomputed stake_undermined from the definition on every vote, "
              "cross-checked the CSV column and found it consistent; undefined kept as None",
              arm["cross_check"]["n_mismatched"] == 0 and arm["cross_check"]["csv_has_stake_undermined"]
              and any(v["undermined"] is None for v in arm["votes"] if v["role"] in STAKE_SEATS_NB)
              and all(v["undermined"] is None for v in arm["votes"]
                      if v["role"] in STAKE_SEATS_NB and v["group_verdict"] == NO_RECOMMENDATION)
              and all(v["undermined"] is False for v in arm["votes"]
                      if v["role"] not in STAKE_SEATS_NB and v["group_verdict"] in RECOMMENDATION_SET))
        res = analyse_arm(arm, cache_dir=None, draws=300, seed=3)
        check("guard passes on the clean synthetic arm (17 calls per cell, 0 NOVERDICT)",
              res["guard"]["pass"] and res["guard"]["n_calls"] == CALLS_PER_CELL_NB * n_cells
              and res["guard"]["n_noverdict"] == 0 and not res["refused"])

        # independent expectations straight from the payoff matrices
        by_id = {g.game_id: g for g in games}
        n_wrong = n_val_wrong = n_dil_wrong = n_dil = n_val = 0
        acc_gold_votes: list[int] = []      # advocate accept on gold cells, hand-coded
        acc_gold_value: list[int] = []
        for d in arm["debates"]:
            g = by_id[d["game_id"]]
            wrong = d["verdict"] != d["gold"]
            n_wrong += wrong
            if g.region == REGION_VALUE:
                n_val += 1
                n_val_wrong += wrong
            if g.gold_type == DILEMMA_BOTH_PREFER:
                n_dil += 1
                n_dil_wrong += wrong
            if not wrong:
                gc = g.gold_cell
                for p in ("row", "col"):
                    if gc is None:
                        u = False
                    elif p == "row":
                        u = g.pA[1 - gc[0]][gc[1]] > g.pA[gc[0]][gc[1]]
                    else:
                        u = g.pB[gc[0]][1 - gc[1]] > g.pB[gc[0]][gc[1]]
                    acc_gold_votes.append(int(not u))
                    if g.region == REGION_VALUE:
                        acc_gold_value.append(int(not u))
        m = lambda xs: sum(xs) / len(xs)  # noqa: E731
        pa = res["private_action"]
        check("P1': pooled advocate private-action share in DILEMMA at r0 is exactly 1.0 "
              "(planted), CI lower bound > 0.5, reading HOLDS; per seat 1.0; tempted and "
              "not-tempted splits both 1.0 with the registered tempted count",
              pa["pooled"]["r0"]["dilemma"]["point"] == 1.0 and pa["P1_prime"]["holds"]
              and pa["pooled"]["r0"]["dilemma"]["n"] == 2 * n_dil
              and all(pa["per_seat"]["r0"][s]["dilemma"]["point"] == 1.0 for s in STAKE_SEATS_NB)
              and pa["pooled"]["r0"]["dilemma_tempted"]["n"] == 2 * (7 + 1)   # 8 tempted seats x 2 samples
              and pa["pooled"]["r0"]["dilemma_not_tempted"]["n"] == 2 * 6
              and pa["pooled"]["r0"]["dilemma_tempted"]["point"] == 1.0)
        check("P1' at r2 is exactly 0.0 in DILEMMA (advocates name the gold action there) "
              "and the r0 share on the neutral region is 1.0 (private == gold there)",
              pa["pooled"]["r2"]["dilemma"]["point"] == 0.0
              and pa["pooled"]["r0"]["by_region"]["neutral"]["point"] == 1.0
              and pa["pooled"]["r0"]["by_region"]["value"]["n"] == 2 * n_dil)
        check("dominant-action share in DILEMMA at r0 is 1.0 on the 8 dominant seats "
              "(2 samples each: n = 16) and undefined seats are excluded",
              pa["dominant_pooled"]["r0"]["dilemma"]["point"] == 1.0
              and pa["dominant_pooled"]["r0"]["dilemma"]["n"] == 16)
        acc = res["accuracy"]
        check(f"group final accuracy is exactly 1 - {n_wrong}/{n_cells}; synthesis 1.0; "
              "DILEMMA accuracy exactly 0.5 (sample 1 planted wrong)",
              abs(acc["group"]["final"]["all"]["point"] - (1 - n_wrong / n_cells)) < 1e-9
              and acc["group"]["synthesis"]["all"]["point"] == 1.0
              and abs(acc["group"]["final"]["by_gold_type"][DILEMMA_BOTH_PREFER]["point"] - 0.5) < 1e-9
              and n_dil_wrong == 7 and n_dil == 14)
        check("advocate r0 accuracy vs gold action: 0.0 in DILEMMA (private != gold), 1.0 on "
              "the neutral region, excluded (n=0) on cannot_assist; every seat's r2 = 1.0 "
              "where defined; third seat r0 = 1.0",
              acc["per_seat"][ROW_SEAT_NB]["r0"]["by_gold_type"][DILEMMA_BOTH_PREFER]["point"] == 0.0
              and acc["per_seat"][COL_SEAT_NB]["r0"]["by_region"]["neutral"]["point"] == 1.0
              and acc["per_seat"][ROW_SEAT_NB]["r0"]["by_region"]["cannot_assist"]["n"] == 0
              and all(acc["per_seat"][s]["r2"]["all"]["point"] == 1.0 for s in acc["seats"])
              and acc["per_seat"]["neutral_reader_nb"]["r0"]["all"]["point"] == 1.0)
        check("REGRESSION GUARD on the extent table: cannot_assist NO_RECOMMENDATION group "
              "verdicts score CORRECT (accuracy > 0.5 on all three types)",
              all(acc["group"]["final"]["by_gold_type"][gt]["point"] > 0.5
                  for gt in ("CYCLIC_NO_IDEAL", "CONFLICT_OF_EQUILIBRIA", "ZERO_SUM")))
        check("extent table has the seven registered gold types with the registered game counts",
              acc["games_by_gold_type"] == {"INCENTIVE_COMPATIBLE_EFFICIENT": 70, "HARMONY": 27,
                                            "CYCLIC_NO_IDEAL": 16, "CONFLICT_OF_EQUILIBRIA": 9,
                                            "ASSURANCE": 9, "DILEMMA_BOTH_PREFER": 7, "ZERO_SUM": 6})
        check("third seat Nash-pull is computed on PLAN verdicts only: 0.0 in DILEMMA (it "
              "names the gold), 1.0 on the neutral region",
              acc["third_nash_pull"]["neutral_reader_nb"]["r0"]["by_gold_type"][DILEMMA_BOTH_PREFER]["point"] == 0.0
              and acc["third_nash_pull"]["neutral_reader_nb"]["r0"]["by_region"]["neutral"]["point"] == 1.0)
        gr = res["grip"]
        n_und = sum(1 for v in arm["votes"] if v["undermined"])
        n_rej = sum(v["reject"] for v in arm["votes"])
        check("synthetic design produced undermined advocate votes (else G3 is untestable)",
              n_und > 0)
        check(f"G1 fire rate is exactly the wrong share {n_wrong}/{n_cells} (third objects iff "
              "wrong; R's objections are mis-localised only when wrong)",
              abs(gr["G1_fire_rate"] - n_wrong / n_cells) < 1e-9 and gr["criteria"]["G1"])
        check(f"G2 reject share is exactly {n_rej}/{len(arm['votes'])} and clears 0.05",
              abs(gr["G2_reject_share"] - n_rej / len(arm["votes"])) < 1e-9 and gr["criteria"]["G2"])
        c3 = gr["G3_stake_concentration"]
        check("G3 is exactly +1.0 (advocates reject iff undermined), CI excludes 0, undefined "
              "votes excluded not coded 0",
              abs(c3["delta"] - 1.0) < 1e-9 and c3["lo"] > 0 and gr["criteria"]["G3"]
              and c3["n_excluded_undefined"] > 0
              and c3["n_votes"] + c3["n_excluded_undefined"] == 2 * n_cells)
        check("grip is True when all three pass", gr["grip"] is True)
        fl = res["flags"]
        check(f"composite and stake-blind fire rates are both exactly {n_wrong}/{n_cells}; "
              "lift +1.0 with CI excluding 0 (fired iff wrong by construction); DILEMMA "
              "composite fire rate exactly 0.5",
              abs(fl["composite"]["fire_rate"]["all"]["point"] - n_wrong / n_cells) < 1e-9
              and abs(fl["stake_blind"]["fire_rate"]["all"]["point"] - n_wrong / n_cells) < 1e-9
              and abs(fl["composite"]["lift"]["all"]["delta"] - 1.0) < 1e-9
              and fl["composite"]["lift"]["all"]["lo"] > 0
              and abs(fl["composite"]["fire_rate"]["by_gold_type"][DILEMMA_BOTH_PREFER]["point"] - 0.5) < 1e-9)
        check("P5: fire on DILEMMA is not > 0.5 and the value-region lift CI excludes 0, so "
              "flooding reads False and computable True",
              fl["P5"]["computable"] and not fl["P5"]["flooding"]
              and not fl["P5"]["lift_value"]["ci_includes_zero"])
        check("lift power flag reads >= 60 fired",
              fl["composite"]["lift"]["all"]["powered"] == (n_wrong >= MIN_FIRED))
        ag = res["accept_on_gold"]
        check(f"P6 accept-on-gold share reproduces the matrix computation "
              f"({sum(acc_gold_votes)}/{len(acc_gold_votes)} overall, "
              f"{sum(acc_gold_value)}/{len(acc_gold_value)} value region); undermined "
              "advocates on gold accept 0.0, not-undermined 1.0",
              ag["n_votes"] == len(acc_gold_votes)
              and abs(ag["pooled"]["all"]["point"] - m(acc_gold_votes)) < 1e-9
              and abs(ag["pooled"]["by_region"]["value"]["point"] - m(acc_gold_value)) < 1e-9
              and ag["by_undermined"]["undermined"]["point"] == 0.0
              and ag["by_undermined"]["not_undermined"]["point"] == 1.0
              and ag["pooled"]["by_region"]["value"]["point"] < 1.0)

        # ---- solo comparator: P2' -------------------------------------------------------------
        solo_path = tdp / "game_solo_nb_rows.csv"
        solo_fields = ["scaffold", "item_id", "sample_idx", "verdict", "gold_verdict",
                       "masked", "arm"]

        def _write_solo(path, scaffold, extra=()):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=solo_fields)
                w.writeheader()
                for d in arm["debates"]:
                    w.writerow({"scaffold": scaffold, "item_id": d["item"],
                                "sample_idx": d["sample_idx"], "verdict": d["gold"],
                                "gold_verdict": d["gold"], "masked": 0, "arm": "full"})
                w.writerow({"scaffold": scaffold, "item_id": "111", "sample_idx": 0,
                            "verdict": "PLAN_11", "gold_verdict": "PLAN_11", "masked": 1,
                            "arm": "masked"})
                for row in extra:
                    w.writerow(dict({"scaffold": scaffold}, **row))
        _write_solo(solo_path, SOLO_SCAFFOLD_NB)
        buf = io.StringIO()
        with redirect_stdout(buf):
            solo = load_solo_rows_nb(solo_path, games)
        check("load_solo_rows_nb on the solo NON-BINDING run (scaffold tmg_solo_nb): drops "
              "the masked row, keeps every other row",
              len(solo) == n_cells and "1 masked" in buf.getvalue())
        _write_solo(tdp / "game_solo_rows.csv", "tmg_solo")
        try:
            load_solo_rows_nb(tdp / "game_solo_rows.csv", games)
            solo_bind_refused = False
        except ValueError as e:
            solo_bind_refused = "tmg_solo_nb" in str(e) and "NON-BINDING" in str(e)
        check("load_solo_rows_nb REFUSES the BINDING solo run (scaffold tmg_solo, the Stage A "
              "file, whose PLAN golds coincide with the nb golds): ValueError naming the "
              "expected scaffold, not a silent 'dropped N'", solo_bind_refused)
        _write_solo(tdp / "game_solo_nuq_rows.csv", SOLO_SCAFFOLD_NB,
                    extra=[{"item_id": "111", "sample_idx": 0, "verdict": "NO_UNIQUE_PLAN",
                            "gold_verdict": "NO_UNIQUE_PLAN", "masked": 0, "arm": "full"}])
        try:
            load_solo_rows_nb(tdp / "game_solo_nuq_rows.csv", games)
            solo_gold_refused = False
        except ValueError as e:
            solo_gold_refused = "NO_UNIQUE_PLAN" in str(e) and "refusing" in str(e)
        check("load_solo_rows_nb REFUSES a file with a NO_UNIQUE_PLAN gold (a binding-regime "
              "row) instead of dropping it", solo_gold_refused)
        with open(tdp / "game_solo_noscaf_rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "verdict", "gold_verdict"])
            w.writeheader()
            for d in arm["debates"][:5]:
                w.writerow({"item_id": d["item"], "verdict": d["gold"], "gold_verdict": d["gold"]})
        check("a solo CSV without a scaffold column still loads (the column is checked "
              "only when present)",
              len(load_solo_rows_nb(tdp / "game_solo_noscaf_rows.csv", games)) == 5)
        res_s = analyse_arm(arm, cache_dir=None, solo=solo, draws=300, seed=3)
        so = res_s["solo"]
        check("P2': neutral arm minus an all-correct solo on DILEMMA is exactly -0.5, paired "
              "on 7 items, CI entirely below 0, reading HOLDS",
              abs(so["stats"]["dilemma"]["delta"] + 0.5) < 1e-9
              and so["stats"]["dilemma"]["n_shared_items"] == 7
              and so["stats"]["dilemma"]["hi"] < 0 and so["P2_prime"]["holds"])
        check(f"solo contrast on the value region is exactly -{n_val_wrong}/{n_val}; on all "
              f"items -{n_wrong}/{n_cells}",
              abs(so["stats"]["value"]["delta"] + n_val_wrong / n_val) < 1e-9
              and abs(so["stats"]["all"]["delta"] + n_wrong / n_cells) < 1e-9)
        n_flagged_items = sum(1 for it in {d["item"] for d in arm["debates"]}
                              if 2 * sum(d["verdict"] != d["gold"] for d in arm["debates"]
                                         if d["item"] == it)
                              > sum(1 for d in arm["debates"] if d["item"] == it))
        check("transfer lift onto an all-correct solo run is exactly 0.0 (both conditional "
              "wrong rates 0); flagged items = games wrong on a majority of cells, which "
              "excludes the half-wrong DILEMMA items",
              so["transfer"]["lift"] == 0.0 and so["transfer"]["p_wrong_flagged"] == 0.0
              and so["transfer"]["p_wrong_unflagged"] == 0.0
              and so["transfer"]["n_flagged_items"] == n_flagged_items > 0
              and so["transfer"]["n_items"] == 144)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_arm_report(res_s)
        out = buf.getvalue()
        check("report prints guard PASSED, P1', ACCURACY, extent table, GRIP, both FLAG "
              "sections, P6 and the solo comparator; never GUARD FAILED",
              "guard PASSED" in out and "P1' PRIVATE-ACTION" in out and "ACCURACY" in out
              and "extent table" in out and "GRIP: yes" in out and "FLAG [composite]" in out
              and "FLAG [stake_blind]" in out and "P6 ADVOCATE" in out
              and "SOLO NON-BINDING" in out and "GUARD FAILED" not in out)

        # ---- refusals: foreign CSVs, rendering mismatch, definition mismatch ---------------------
        rows_f = [dict(r, scaffold="game_plan_ncot", arm="third_neutral") for r in rows_n]
        _write_csv(tdp / "foreign_rows.csv", rows_f, ROW_FIELDS_SYNTH)
        _write_csv(tdp / "foreign_votes.csv", votes_n, VOTE_FIELDS_SYNTH)
        try:
            load_arm(tdp / "foreign_rows.csv", tdp / "foreign_votes.csv", tag="foreign", games=games)
            foreign_refused = False
        except ValueError as e:
            foreign_refused = "not a 16.13c run" in str(e)
        check("a binding-regime CSV (scaffold game_plan_ncot) is refused, never read as "
              "non-binding", foreign_refused)
        rows_g = [dict(r, gold_verdict=("NO_UNIQUE_PLAN" if r["gold_verdict"] == NO_RECOMMENDATION
                                        else r["gold_verdict"])) for r in rows_n]
        _write_csv(tdp / "gold_rows.csv", rows_g, ROW_FIELDS_SYNTH)
        try:
            load_arm(tdp / "gold_rows.csv", tdp / "cgnb_game_neutral_votes.csv", tag="g", games=games)
            gold_refused = False
        except ValueError as e:
            gold_refused = "rendering mismatch" in str(e)
        check("a rows CSV whose gold is the binding NO_UNIQUE_PLAN is a rendering mismatch "
              "and is refused", gold_refused)
        rows_p = [dict(r, row_perm=("10" if r["row_perm"] == "01" else "01")) for r in rows_n[:1]] + rows_n[1:]
        _write_csv(tdp / "perm_rows.csv", rows_p, ROW_FIELDS_SYNTH)
        try:
            load_arm(tdp / "perm_rows.csv", tdp / "cgnb_game_neutral_votes.csv", tag="p", games=games)
            perm_refused = False
        except ValueError as e:
            perm_refused = "rendering mismatch" in str(e)
        check("a permutation that disagrees with the renderer is refused", perm_refused)
        votes_u = [dict(v) for v in votes_n]
        flipped = next(v for v in votes_u if v["stake_undermined"] == 1)
        flipped["stake_undermined"] = 0
        _write_csv(tdp / "und_votes.csv", votes_u, VOTE_FIELDS_SYNTH)
        try:
            load_arm(tdp / "cgnb_game_neutral_rows.csv", tdp / "und_votes.csv", tag="u", games=games)
            und_refused = False
        except ValueError as e:
            und_refused = "disagree with the registered definitions" in str(e)
        check("a votes CSV whose stake_undermined disagrees with the registered definition "
              "on one vote is refused", und_refused)
        votes_ga = [dict(v) for v in votes_n]
        ga_v = next(v for v in votes_ga if v["gold_action"] == ACT_1)
        ga_v["gold_action"] = ACT_2
        _write_csv(tdp / "ga_votes.csv", votes_ga, VOTE_FIELDS_SYNTH)
        try:
            load_arm(tdp / "cgnb_game_neutral_rows.csv", tdp / "ga_votes.csv", tag="ga", games=games)
            ga_refused = False
        except ValueError as e:
            ga_refused = "disagree with the registered definitions" in str(e)
        check("a votes CSV whose gold_action disagrees with the gold cell's action is refused",
              ga_refused)
        rows_ns, votes_ns = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY,
                                           dilemma_wrong_samples=(1,), write_stake_col=False)
        _write_csv(tdp / "nostake_rows.csv", rows_ns, ROW_FIELDS_SYNTH)
        _write_csv(tdp / "nostake_votes.csv", votes_ns, VOTES_REQUIRED + ("scaffold",))
        arm_ns = load_arm(tdp / "nostake_rows.csv", tdp / "nostake_votes.csv", tag="ns", games=games)
        g_ns = grip(arm_ns["debates"], arm_ns["votes"], draws=50, seed=1)
        check("without a stake_undermined column the analyzer computes it itself and G3 is "
              "unchanged (+1.0)",
              not arm_ns["cross_check"]["csv_has_stake_undermined"]
              and abs(g_ns["G3_stake_concentration"]["delta"] - 1.0) < 1e-9)
        votes_role = [dict(v, role_id=("player_r_advocate" if v["role_id"] == ROW_SEAT_NB else v["role_id"]))
                      for v in votes_n]
        _write_csv(tdp / "role_votes.csv", votes_role, VOTE_FIELDS_SYNTH)
        try:
            load_arm(tdp / "cgnb_game_neutral_rows.csv", tdp / "role_votes.csv", tag="r", games=games)
            role_refused = False
        except ValueError as e:
            role_refused = "role_id" in str(e)
        check("the binding regime's role id player_r_advocate in a votes CSV is refused",
              role_refused)
        rows_mc = [dict(r) for r in rows_n]
        _write_csv(tdp / "mc_rows.csv", rows_mc, tuple(c for c in ROW_FIELDS_SYNTH if c != "r2_verdict_third"))
        try:
            load_arm(tdp / "mc_rows.csv", tdp / "cgnb_game_neutral_votes.csv", tag="mc", games=games)
            mc_refused = False
        except ValueError as e:
            mc_refused = "missing required columns ['r2_verdict_third']" in str(e)
        check("a rows CSV missing a required column names it and is refused", mc_refused)

        # ---- GUARD FAILED paths -----------------------------------------------------------------
        rows_b, votes_b = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY,
                                         noverdict_r0_every=10)
        _write_csv(tdp / "bad_rows.csv", rows_b, ROW_FIELDS_SYNTH)
        _write_csv(tdp / "bad_votes.csv", votes_b, VOTE_FIELDS_SYNTH)
        bad = load_arm(tdp / "bad_rows.csv", tdp / "bad_votes.csv", tag="bad", games=games)
        res_b = analyse_arm(bad, cache_dir=None, draws=50, seed=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_arm_report(res_b)
        out_b = buf.getvalue()
        check("10% NOVERDICT on the R-advocate's r0 fails the guard; the result carries no "
              "readout keys; the report prints GUARD FAILED and nothing else",
              not res_b["guard"]["pass"] and res_b["refused"]
              and not any(k in res_b for k in ("grip", "private_action", "accuracy", "flags",
                                                "accept_on_gold"))
              and "GUARD FAILED" in out_b and "P1'" not in out_b and "GRIP:" not in out_b
              and "G3" not in out_b)
        check("the failing guard cell is the R-advocate's r0 group",
              any(k.endswith(f"/r0:{ROW_SEAT_NB}|{ARM_OF_NB[THIRD_NEUTRAL]}") and v["noverdict"] > 0.05
                  for k, v in res_b["guard"]["cells"].items()))
        rows_x, votes_x = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY)
        for r in rows_x:
            r["r0_verdict_row_advocate"] = r["gold_verdict"] if r["gold_verdict"] in PLAN_VERDICTS else "PLAN_11"
        for v in votes_x:
            if v["role_id"] == ROW_SEAT_NB:
                v["r0_verdict"] = v["gold_verdict"] if v["gold_verdict"] in PLAN_VERDICTS else "PLAN_11"
        _write_csv(tdp / "x_rows.csv", rows_x, ROW_FIELDS_SYNTH)
        _write_csv(tdp / "x_votes.csv", votes_x, VOTE_FIELDS_SYNTH)
        res_x = analyse_arm(load_arm(tdp / "x_rows.csv", tdp / "x_votes.csv", tag="x", games=games),
                            cache_dir=None, draws=50, seed=1)
        check("an advocate that answers with PLAN tokens (the binding instrument) fails the "
              "guard with NOVERDICT 100% on its r0 group (instrument confusion is not a verdict)",
              not res_x["guard"]["pass"]
              and all(v["noverdict"] == 1.0 for k, v in res_x["guard"]["cells"].items()
                      if f"/r0:{ROW_SEAT_NB}|" in k))
        rows_u2, votes_u2 = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY)
        for v in votes_u2:
            v["vote"], v["reject"] = "UNRESOLVED", 0
        _write_csv(tdp / "unres_rows.csv", rows_u2, ROW_FIELDS_SYNTH)
        _write_csv(tdp / "unres_votes.csv", votes_u2, VOTE_FIELDS_SYNTH)
        res_u = analyse_arm(load_arm(tdp / "unres_rows.csv", tdp / "unres_votes.csv", tag="unres",
                                     games=games), cache_dir=None, draws=50, seed=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_arm_report(res_u)
        check("an arm whose every r4 vote is a stray UNRESOLVED FAILS the guard with NOVERDICT "
              "100% on every r4 group, prints GUARD FAILED and returns no G2",
              not res_u["guard"]["pass"] and res_u["guard"]["worst_noverdict"] == 1.0
              and all(v["noverdict"] == 1.0 for k, v in res_u["guard"]["cells"].items()
                      if "/r4_vote:" in k)
              and "grip" not in res_u and "GUARD FAILED" in buf.getvalue()
              and "G2" not in buf.getvalue())
        rows_l, votes_l = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY)
        for v in votes_l:
            v["r3_label"] = "UNRESOLVED"
        _write_csv(tdp / "unres_l_rows.csv", rows_l, ROW_FIELDS_SYNTH)
        _write_csv(tdp / "unres_l_votes.csv", votes_l, VOTE_FIELDS_SYNTH)
        res_ul = analyse_arm(load_arm(tdp / "unres_l_rows.csv", tdp / "unres_l_votes.csv",
                                      tag="unres_l", games=games), cache_dir=None, draws=50, seed=1)
        check("...and likewise when every r3 label is UNRESOLVED (no silent G1 = 0)",
              not res_ul["guard"]["pass"]
              and all(v["noverdict"] == 1.0 for k, v in res_ul["guard"]["cells"].items()
                      if "/r3_label:" in k))
        rows_m, votes_m = _synthetic_arm(games, THIRD_NEUTRAL, wrong_every=WRONG_EVERY)
        for r in rows_m:
            r["synthesis_verdict"] = "NO_UNIQUE_PLAN"
        _write_csv(tdp / "syn_rows.csv", rows_m, ROW_FIELDS_SYNTH)
        _write_csv(tdp / "syn_votes.csv", votes_m, VOTE_FIELDS_SYNTH)
        res_m = analyse_arm(load_arm(tdp / "syn_rows.csv", tdp / "syn_votes.csv", tag="syn",
                                     games=games), cache_dir=None, draws=50, seed=1)
        check("a moderator that answers NO_UNIQUE_PLAN (the binding token) on every synthesis "
              "fails the guard on the synthesis group",
              not res_m["guard"]["pass"]
              and all(v["noverdict"] == 1.0 for k, v in res_m["guard"]["cells"].items()
                      if "/synthesis:" in k))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "bad", "--dir", str(tdp), "--no-cache", "--n-boot", "20"])
        check("main() exits 4 on a failed guard and prints GUARD FAILED",
              rc == 4 and "GUARD FAILED" in buf.getvalue())

        # ---- cache enrichment of the truncation column ---------------------------------------
        cdir = tdp / "cache"
        cdir.mkdir()
        d0 = arm["debates"][0]
        base_name = (f"cgd_{_safe(d0['model'])}_{SCAFFOLD_NB}_t{{cap}}_{d0['arm']}_"
                     f"{d0['item']}_{d0['sample_idx']:02d}_r0_{ROW_SEAT_NB}.json")
        (cdir / base_name.format(cap=1024)).write_text(json.dumps(
            {"finish_reason": "length", "output": "x" * 50}))
        (cdir / base_name.format(cap=2560)).write_text(json.dumps(
            {"finish_reason": "stop", "output": "y" * 900}))
        syn_name = (f"cgd_{_safe(d0['moderator_model'])}_{SCAFFOLD_NB}_t1024_{d0['arm']}_"
                    f"{d0['item']}_{d0['sample_idx']:02d}_synthesis_"
                    f"{_safe(moderator_role_id_nb(d0['moderator_model'], d0['third_seat']))}.json")
        (cdir / syn_name).write_text(json.dumps({"finish_reason": "length", "output": "z" * 10}))
        # a binding-regime record with the same cell must NOT be picked up
        (cdir / base_name.replace(SCAFFOLD_NB, "game_plan_ncot").replace(d0["arm"], "third_neutral")
         .format(cap=4096)).write_text(json.dumps({"finish_reason": "length", "output": "q"}))
        grows = guard_rows_from_csv(arm["debates"])
        known_before = sum(1 for g in grows if g["truncation_known"])
        cov = enrich_guard_from_cache(grows, cdir)
        r0_row = next(g for g in grows if g["item"] == d0["item"] and g["sample_idx"] == d0["sample_idx"]
                      and g["round"] == "r0" and g["role_id"] == ROW_SEAT_NB)
        syn_row = next(g for g in grows if g["item"] == d0["item"] and g["sample_idx"] == d0["sample_idx"]
                       and g["round"] == "synthesis")
        check("CSV alone knows truncation for 4 of 17 calls per cell (integration + r4)",
              known_before == 4 * n_cells)
        check("cache enrichment: 2 nb records found (the binding-regime record is ignored), "
              "the LARGER cap wins for r0 (finish stop, 900 chars), the synthesis record "
              "marks truncation",
              cov["n_found"] == 2 and r0_row["truncated"] == 0 and r0_row["output_len"] == 900
              and r0_row["truncation_known"] and syn_row["truncated"] == 1
              and sum(1 for g in grows if g["truncation_known"]) == known_before + 2)
        for d in arm["debates"]:
            nm = (f"cgd_{_safe(d['model'])}_{SCAFFOLD_NB}_t2560_{d['arm']}_{d['item']}_"
                  f"{d['sample_idx']:02d}_r1_{COL_SEAT_NB}.json")
            (cdir / nm).write_text(json.dumps({"finish_reason": "length", "output": "q"}))
        res_c = analyse_arm(arm, cache_dir=cdir, draws=50, seed=1)
        check("truncation read from the cache alone (VERDICT lines intact) fails the guard",
              not res_c["guard"]["pass"] and res_c["refused"]
              and res_c["guard"]["cache"]["n_found"] == 2 + n_cells)

        # ---- contrast with a synthetic planner arm (P3', P6) ----------------------------------
        rows_p, votes_p = _synthetic_arm(games, THIRD_PLANNER, wrong_every=None,
                                         advocates_r0="gold", advocates_persuaded=True)
        _write_csv(tdp / "cgnb_game_planner_rows.csv", rows_p, ROW_FIELDS_SYNTH)
        _write_csv(tdp / "cgnb_game_planner_votes.csv", votes_p, VOTE_FIELDS_SYNTH)
        planner = load_arm(tdp / "cgnb_game_planner_rows.csv", tdp / "cgnb_game_planner_votes.csv",
                           tag="cgnb_game_planner", games=games)
        check("planner arm loads with plan_mediator_nb as the third seat and the distinct arm",
              planner["third_role_id"] == "plan_mediator_nb"
              and {d["arm"] for d in planner["debates"]} == {ARM_OF_NB[THIRD_PLANNER]})
        res_p = analyse_arm(planner, cache_dir=None, draws=300, seed=3)
        check("planner arm: P1' share in DILEMMA at r0 is exactly 0.0 (advocates planted on "
              "the gold action) so P1' fails there; P6 accept-on-gold is 1.0 everywhere",
              res_p["private_action"]["pooled"]["r0"]["dilemma"]["point"] == 0.0
              and not res_p["private_action"]["P1_prime"]["holds"]
              and res_p["accept_on_gold"]["pooled"]["all"]["point"] == 1.0)
        con = arm_contrast(planner, arm, draws=300, seed=5)  # order deliberately swapped
        ga = con["stats"]["group_accuracy"]
        check("contrast orients itself planner minus neutral whichever order is given",
              con["b"]["third_seat"] == THIRD_PLANNER and con["a"]["third_seat"] == THIRD_NEUTRAL
              and con["planner_minus_neutral"])
        check(f"P3': 16 shared value items, group accuracy delta exactly {n_val_wrong}/{n_val}, "
              "CI excludes 0, reading HOLDS",
              ga["n_shared_items"] == 16 and abs(ga["delta"] - n_val_wrong / n_val) < 1e-9
              and ga["positive"] and con["P3_prime"]["holds"] and n_val == 32)
        p6 = con["stats"]["advocate_accept_on_gold"]
        check(f"P6: planner (1.0) minus neutral ({m(acc_gold_value):.3f}) accept-on-gold share "
              "on the value region is exactly the complement, CI excludes 0, reading HOLDS",
              abs(p6["delta"] - (1.0 - m(acc_gold_value))) < 1e-9 and p6["positive"]
              and con["P6"]["holds"] and p6["delta"] > 0)
        check("contrast: third-seat accuracy delta 0 (both name the gold); advocate private-"
              "action delta exactly -1.0 (planner advocates on gold); composite fire delta < 0",
              con["stats"]["third_r0_accuracy"]["delta"] == 0.0
              and abs(con["stats"]["advocate_private_r0_dilemma"]["delta"] + 1.0) < 1e-9
              and con["stats"]["composite_fire_rate"]["delta"] < 0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_contrast(con)
        check("contrast prints with the POSITIVE marker and both P3' / P6 lines",
              "THIRD-SEAT CONTRAST" in buf.getvalue() and "POSITIVE" in buf.getvalue()
              and "P3':" in buf.getvalue() and "P6:" in buf.getvalue())
        rd = readings_summary([res_s, res_p], con)
        check("readings summary: P1' holds on the neutral arm and fails on the planner arm, "
              "P2' holds, P3' holds, P5 not flooding, P6 holds",
              rd["P1_prime"]["cgnb_game_neutral"]["holds"]
              and not rd["P1_prime"]["cgnb_game_planner"]["holds"]
              and rd["P2_prime"]["holds"] and rd["P3_prime"]["holds"]
              and not rd["P5"]["cgnb_game_neutral"]["flooding"]
              and rd["P6"]["planner_minus_neutral"]["holds"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "cgnb_game_neutral", "--contrast-tag", "cgnb_game_planner",
                       "--dir", str(tdp), "--no-cache", "--n-boot", "30",
                       "--solo-rows", str(solo_path), "--out", str(tdp / "analysis.json")])
        out_m = buf.getvalue()
        js = json.loads((tdp / "analysis.json").read_text())
        check("main() end to end on both arms + solo exits 0, prints both reports, the "
              "contrast and the readings block, writes JSON with arms / contrast / readings",
              rc == 0 and out_m.count("GAME NON-BINDING -- tag") == 2
              and "THIRD-SEAT CONTRAST" in out_m and "PRE-DECLARED READINGS" in out_m
              and "(PRIMARY)" in out_m
              and js["contrast"]["P3_prime"]["holds"] and js["readings"]["P2_prime"]["holds"]
              and len(js["arms"]) == 2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "nope", "--dir", str(tdp), "--no-cache"])
        check("main() exits 2 when the tag's CSVs are absent", rc == 2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "cgnb_game_neutral", "--dir", str(tdp), "--no-cache",
                       "--n-boot", "20", "--solo-rows", str(tdp / "game_solo_rows.csv")])
        check("main() exits 2 with a clear message when --solo-rows is the BINDING solo run "
              "(scaffold tmg_solo); no arm report is printed",
              rc == 2 and "tmg_solo_nb" in buf.getvalue()
              and "GAME NON-BINDING -- tag" not in buf.getvalue())
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "cgnb_game_neutral", "--dir", str(tdp), "--no-cache",
                       "--n-boot", "20", "--solo-rows", str(tdp / "game_solo_nuq_rows.csv")])
        check("main() exits 2 when --solo-rows carries a NO_UNIQUE_PLAN gold",
              rc == 2 and "NO_UNIQUE_PLAN" in buf.getvalue())
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "cgnb_game_neutral", "--dir", str(tdp), "--no-cache",
                       "--n-boot", "20", "--solo-rows", str(tdp / "absent_solo.csv")])
        check("main() exits 2 when --solo-rows does not exist", rc == 2)

        # ---- integration: the runner's OWN stub output through the analyzer ---------------
        # Offline: generate_any is replaced by the runner's deterministic stub
        # and the cache directory call_cache_path reads is redirected to a
        # temp dir, exactly as the runner's selftest does. No --run anywhere.
        if runner is not None:
            import scripts.run_crowdgold_deliberation as rcd
            real_gen, real_out = rcd.generate_any, rcd.OUT_DIR
            value_games = [g for g in games if g.region == REGION_VALUE]
            got: dict[str, tuple[list, list]] = {}
            try:
                rcd.generate_any = runner._stub_generate  # type: ignore[assignment]
                with tempfile.TemporaryDirectory() as td2:
                    rcd.OUT_DIR = Path(td2)
                    try:
                        for ts in THIRD_SEATS:
                            rr, vv = [], []
                            for it in runner.build_items(value_games, 2):
                                row, vr, _cr = runner.run_deliberation("stub-model", it,
                                                                       third_seat=ts)
                                rr.append(row)
                                vv.extend(vr)
                            got[ts] = (rr, vv)
                    finally:
                        rcd.OUT_DIR = real_out
            finally:
                rcd.generate_any = real_gen
            for ts, (rr, vv) in got.items():
                _write_csv(tdp / f"{TAG_PREFIX_NB}_{ts}_rows.csv", rr, runner.ROW_FIELDS)
                _write_csv(tdp / f"{TAG_PREFIX_NB}_{ts}_votes.csv", vv, runner.VOTE_FIELDS)
            r_arms = {ts: load_arm(tdp / f"{TAG_PREFIX_NB}_{ts}_rows.csv",
                                   tdp / f"{TAG_PREFIX_NB}_{ts}_votes.csv",
                                   tag=f"{TAG_PREFIX_NB}_{ts}", games=games) for ts in THIRD_SEATS}
            r_res = {ts: analyse_arm(r_arms[ts], cache_dir=None, draws=100, seed=1)
                     for ts in THIRD_SEATS}
            n_p12 = sum(1 for g in value_games for i in range(2)
                        if gold_recommendation(render_item(g, i)) == "PLAN_12")
            check("integration: the runner's real ROW_FIELDS / VOTE_FIELDS CSVs (32 value "
                  "cells per arm, stub output) load, re-render consistently, pass the guard "
                  "with 17 calls per cell, and the runner's stake_undermined / gold_action "
                  "columns agree with the analyzer on every vote",
                  all(len(r_arms[ts]["debates"]) == 32 and r_res[ts]["guard"]["pass"]
                      and r_res[ts]["guard"]["n_calls"] == 32 * CALLS_PER_CELL_NB
                      and r_arms[ts]["cross_check"]["n_mismatched"] == 0
                      and r_arms[ts]["cross_check"]["csv_has_stake_undermined"]
                      and r_arms[ts]["cross_check"]["n_checked"] == 64
                      for ts in THIRD_SEATS)
                  and r_arms[THIRD_NEUTRAL]["third_role_id"] == "neutral_reader_nb"
                  and r_arms[THIRD_PLANNER]["third_role_id"] == "plan_mediator_nb")
            check(f"integration: group accuracy on the stub's constant PLAN_12 equals the share "
                  f"of value cells whose gold is PLAN_12 ({n_p12}/32); P1' n = 28 in DILEMMA; "
                  "P6 accept-on-gold is 0.5 (the row advocate always rejects)",
                  all(abs(r_res[ts]["accuracy"]["group"]["final"]["all"]["point"] - n_p12 / 32) < 1e-9
                      and r_res[ts]["private_action"]["pooled"]["r0"]["dilemma"]["n"] == 28
                      and (r_res[ts]["accept_on_gold"]["pooled"]["all"]["point"] == 0.5
                           if n_p12 else True)
                      for ts in THIRD_SEATS))
            r_con = arm_contrast(r_arms[THIRD_NEUTRAL], r_arms[THIRD_PLANNER], draws=100, seed=1)
            check("integration: the paired contrast of two identical stub arms is exactly 0 "
                  "on 16 shared items and P3' / P6 read as failing",
                  r_con["stats"]["group_accuracy"]["delta"] == 0.0
                  and r_con["stats"]["group_accuracy"]["n_shared_items"] == 16
                  and not r_con["P3_prime"]["holds"] and not r_con["P6"]["holds"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["--tag", f"{TAG_PREFIX_NB}_{THIRD_NEUTRAL}", "--contrast-tag",
                           f"{TAG_PREFIX_NB}_{THIRD_PLANNER}", "--dir", str(tdp), "--no-cache",
                           "--n-boot", "20"])
            check("integration: main() on the runner's default tags exits 0 and prints the "
                  "readings block", rc == 0 and "PRE-DECLARED READINGS" in buf.getvalue())
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--tag", "foreign", "--dir", str(tdp), "--no-cache"])
        check("main() exits 2 on a foreign (binding-regime) CSV and says why",
              rc == 2 and "not a 16.13c run" in buf.getvalue())

    # The selftest's own footprint in the real directory would be a cgd_ file
    # of its stub model under the nb scaffold, or a CSV under its synthetic
    # tags; a concurrent real run moves the raw namespace counts, so those
    # are not compared.
    stub_prefix = f"cgd_{_safe('stub-model')}_{SCAFFOLD_NB}_"
    check("the selftest wrote nothing under any 16.13c namespace in the real cache dir "
          "(no stub-model file under the nb scaffold; no CSV of its synthetic tags "
          "appeared)",
          cache_files_with_prefix(stub_prefix) == []
          and not any((OUT_DIR / f"{t}_{k}.csv").exists()
                      for t in ("foreign", "bad", "x", "unres", "syn", "nope")
                      for k in ("rows", "votes")))

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Addendum 16.13c readouts for run_crowdgold_game_nb CSVs (offline; "
                    "reads only; nothing here can generate).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag", default=f"{TAG_PREFIX_NB}_{THIRD_NEUTRAL}")
    ap.add_argument("--contrast-tag", default=None,
                    help="second arm's tag; adds the paired planner-minus-neutral contrast (P3', P6)")
    ap.add_argument("--model", default=None, help="restrict to one agent model")
    ap.add_argument("--dir", type=Path, default=OUT_DIR,
                    help="directory holding <tag>_rows.csv / _votes.csv and the call cache")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not read the call cache for the guard's truncation column")
    ap.add_argument("--solo-rows", type=Path, default=None,
                    help="the solo non-binding run's rows CSV (item_id, verdict, gold_verdict) "
                         "for P2' and the transfer lift")
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
                  "scripts.run_crowdgold_game_nb first.\n")
            return 2
        if games is None:
            games = load_games()
        try:
            arm = load_arm(rows_path, votes_path, tag=tag, model=args.model, games=games)
        except ValueError as e:
            print(f"\nERROR reading {tag}: {e}\n")
            return 2
        if not arm["debates"]:
            print(f"\nERROR: no rows in {rows_path}"
                  f"{' for model ' + args.model if args.model else ''}.\n")
            return 2
        arms.append(arm)

    solo = None
    if args.solo_rows:
        if not args.solo_rows.exists():
            print(f"\nERROR: --solo-rows {args.solo_rows} not found.\n")
            return 2
        try:
            solo = load_solo_rows_nb(args.solo_rows, games)
        except ValueError as e:
            print(f"\nERROR reading --solo-rows: {e}\n")
            return 2
    cache_dir = None if args.no_cache else args.dir
    results = []
    refused = False
    for i, arm in enumerate(arms):
        res = analyse_arm(arm, cache_dir=cache_dir, solo=solo, draws=args.n_boot,
                          seed=args.seed + 100 * i)
        print_arm_report(res)
        results.append(res)
        refused = refused or res["refused"]
    out = {"arms": results, "contrast": None, "readings": None}
    if refused:
        print("\n*** GUARD FAILED on at least one arm: no number is reported; the contrast "
              "and the readings are not computed. ***")
    else:
        if len(arms) == 2:
            if arms[0]["third_seat"] == arms[1]["third_seat"]:
                print(f"\n  both tags carry the same third seat ({arms[0]['third_seat']}); "
                      "the contrast reads as second minus first and P3' / P6 are n/a")
            out["contrast"] = arm_contrast(arms[0], arms[1], draws=args.n_boot,
                                           seed=args.seed + 500)
            print_contrast(out["contrast"])
        out["readings"] = readings_summary(results, out["contrast"])
        print_readings(out["readings"])
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 4 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
