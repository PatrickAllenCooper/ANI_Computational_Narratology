"""
scripts/tmg_games_nb.py -- the NON-BINDING regime of the TMGBench 2x2 games
instrument (prereg Addendum 16.13c): the question text, the two verdict
instruments (joint recommendation PLAN_xy / NO_RECOMMENDATION; advocate
action ACT_1 / ACT_2), the nb rendering of an item, the nb gold, the
stake_undermined definition and the registered cache names of the nb
deliberation runner. THE ONLY DEFINITION SITE: ``run_game_singleagent
--regime nonbinding`` (the solo comparator), ``run_crowdgold_game_nb`` (the
three-seat deliberation) and ``analyze_game_nb`` (the 16.13c readouts) all
IMPORT the question text, the instruments, the parsers, the gold and the
four item-level definitions (gold / private / dominant action,
stake_undermined) from here; none of them re-defines any of them. The
runner keeps only thin NBGameItem / role wrappers, the analyzer only
aliases. The selftests of all three still pin agreement with this module
as regression guards.

This module makes NO model calls and spends nothing. scripts.tmg_games is
not edited: everything here is built on its public surface (render_item,
ItemView, plan_for_cell, normalize_plan_tokens, dominant_strict, _boot_ci)
and the instruments are registered through scripts.verdict_format's
register_instrument, the extension path tmg_games itself uses.

WHY A SECOND REGIME (16.13c)

Under a binding joint plan the mutual-joint cell of a DILEMMA is in the
core and no advocate has an interest to argue against it (16.13 Stage B1:
role-lock 0.09-0.12, Nash-pull 0.000, 48/48 correct in both arms). The
non-binding regime removes the enforcement: each player chooses privately
and simultaneously, a recommendation may be made to both but binds no one,
and in DILEMMA_BOTH_PREFER the private action is dominant, so an advocate
arguing its player's interest argues AGAINST the joint optimum. The
verdict set changes with the regime (NO_RECOMMENDATION replaces
NO_UNIQUE_PLAN) so a record from one regime can never be scored as the
other: a stray NO_UNIQUE_PLAN under the nb instrument is NOVERDICT.

REGIME AND VERDICTS

  question (every seat)   the two players choose privately and at the same
                          time; no agreement, nothing to enforce one; a
                          recommendation may be made to both but neither is
                          bound by it; name the one pair of options that
                          should be recommended to both, or NO_RECOMMENDATION
                          (wording byte-identical to run_crowdgold_game_nb's,
                          pinned by the selftest when that module is present)
  joint verdict           PLAN_11 / PLAN_12 / PLAN_21 / PLAN_22 (displayed
                          option numbers, as tmg_games) or NO_RECOMMENDATION
                          -- the solo comparator, the third seat, the
                          moderator synthesis and the integrated verdict
  advocate verdict        ACT_1 / ACT_2, the advocate's OWN player's
                          displayed option number (r0 / r1 / r2 of the two
                          advocate seats in the nb runner)
  r3 / r4                 the nb runner's label / vote instruments, as in
                          every other runner (not defined here)

GOLD (16.13c: "the cell that is best for all parties"; nb_gold_verdict(view))

  DILEMMA_BOTH_PREFER   the cell both players prefer to the equilibrium
                        (gold_cell)
  ASSURANCE             the payoff-dominant equilibrium (gold_cell)
  neutral region        gold_cell as in 16.13
  cannot_assist         NO_RECOMMENDATION
Advocate accuracy is scored against the gold cell's action for that player
(gold_act_for; score_act is None -- the seat is EXCLUDED, never coded
wrong -- when the game has no gold action). The joint gold is tmg_games'
gold_cell mapped through the item's display permutation, exactly as the
binding gold, with NO_RECOMMENDATION in place of NO_UNIQUE_PLAN:
nb_gold_verdict(view) reads it off the view's binding gold_verdict (one
signature for every consumer); the selftest pins that it maps back to the
canonical gold cell on every view.

STAKE_UNDERMINED (16.13c)

stake_undermined_nb(view, verdict, stake) := the player has a STRICTLY
profitable unilateral deviation from the recommended cell (the cell is not
a best reply for that player). None when the verdict is not a PLAN_xy
(NO_RECOMMENDATION / NOVERDICT: no cell to deviate from); False for a
stake of "none". In ASSURANCE the gold cell (a Nash cell) undermines
neither advocate. In DILEMMA_BOTH_PREFER it undermines the player whose
private action is strictly dominant: BOTH players on game 111 (the one
game where both have a dominant action) and exactly ONE player on the
other six (canonical type one_dominant_inefficient: the other player's
gold action is already its best reply to the gold cell). 16.13c's "in
DILEMMA the gold cell undermines BOTH advocates by construction" therefore
holds on 1 of the 7 games; the selftest pins the actual structure and the
definition is applied as registered, per player, per cell.

The PRIVATE action of a player (private_act_for) is its action at the
game's unique pure Nash cell -- what it would do on its own -- and is None
when the game has no unique pure Nash cell. It equals the strictly
dominant action (dominant_act_for) whenever one exists, and is defined for
both players on every DILEMMA_BOTH_PREFER game (|N| == 1 by the gold rule),
so the registered lock statistic P(advocate recommends its player's
private action) has a value for every advocate on every DILEMMA cell.

VERDICT PARSING (parse_nb_verdict / parse_act_verdict)

Only spacing / punctuation variants of the exact tokens are normalised.
The NO_RECOMMENDATION spacing variant ('no recommendation',
'no-recommendation') is collapsed ONLY in the token position directly after
a VERDICT separator, never in running prose: "VERDICT: PLAN_22 -- no
recommendation is stronger" reads PLAN_22. A verdict line naming two
distinct legal tokens is a hedge and NOVERDICT; UNRESOLVED is a legal token
of every registered instrument here (register_instrument appends it), so
"VERDICT: PLAN_12 (UNRESOLVED)" and "VERDICT: ACT_1 or UNRESOLVED" are
hedges too, and a bare "VERDICT: UNRESOLVED" is NOVERDICT (never offered).
An explanatory parenthetical that names other cells on the verdict line --
"VERDICT: NO_RECOMMENDATION (PLAN_12 and PLAN_21 tie)" -- is coded as a
hedge: conservative (it inflates NOVERDICT toward the guard), never a
misread.

LEAK CONTROLS

Rendering is tmg_games.render_item (permutation, names, masking), with the
question and legend swapped for the nb wording. The nb text carries no
tmg_games.FORBIDDEN_WORDS word and none of the binding runner's
solution-concept vocabulary (nash / pareto / equilibri / dominan / maximin
/ gold); the selftest renders every view (144 games, full and masked) and
scans every prompt string. Masked nb views share the identical surface
template across games, as the binding ones do.

Usage:
  python -m scripts.tmg_games_nb --selftest
  python -m scripts.tmg_games_nb --show 111 --sample-idx 3 [--masked]
  python -m scripts.tmg_games_nb --extract "VERDICT: NO RECOMMENDATION"
  python -m scripts.tmg_games_nb --extract-act "VERDICT: ACT 2"
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from scripts.verdict_format import (
    INSTRUMENTS,
    NOVERDICT,
    UNRESOLVED,
    extract_verdict_detail,
    register_instrument,
    score_against_gold,
    verdict_instruction,
)
from scripts.tmg_games import (
    ASSURANCE,
    CELLS,
    DILEMMA_BOTH_PREFER,
    GAME_INSTRUMENT,
    GAME_VERDICT_INSTRUCTION,
    N_GAMES,
    NO_UNIQUE_PLAN,
    OUT_DIR,
    PLAN_VERDICTS,
    REGION_CANNOT_ASSIST,
    VERDICT_SET as PLAN_VERDICT_SET,
    Cell,
    Game,
    ItemView,
    _boot_ci,
    cache_files_with_prefix,
    dominant_strict,
    forbidden_hits,
    load_games,
    masked_floor_report as _binding_masked_floor_report,
    normalize_plan_tokens,
    parse_plan_verdict,
    plan_for_cell,
    render_item,
    score_plan,
    surface_template,
)

# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

REGIME_BINDING = "binding"
REGIME_NONBINDING = "nonbinding"
REGIMES = (REGIME_BINDING, REGIME_NONBINDING)

# ---------------------------------------------------------------------------
# Joint-recommendation instrument (registered with verdict_format)
# ---------------------------------------------------------------------------

#: Same instrument NAME as scripts.run_crowdgold_game_nb registers (one
#: entry in verdict_format's table; the selftest pins the tokens).
NB_INSTRUMENT = "tmg_recommendation_verdict"
NO_RECOMMENDATION = "NO_RECOMMENDATION"
NB_VERDICT_SET = PLAN_VERDICTS + (NO_RECOMMENDATION,)

#: The forced final line, verbatim (re-derived from verdict_format in the
#: selftest). UNRESOLVED is deliberately NOT offered, as on the binding
#: instrument: NO_RECOMMENDATION is the committal "no answer".
NB_VERDICT_INSTRUCTION = (
    "End your response with exactly one line: VERDICT: PLAN_11, "
    "VERDICT: PLAN_12, VERDICT: PLAN_21, VERDICT: PLAN_22, or "
    "VERDICT: NO_RECOMMENDATION."
)

try:
    register_instrument(NB_INSTRUMENT, NB_VERDICT_SET,
                        legacy_instruction=NB_VERDICT_INSTRUCTION)
except ValueError:
    pass  # already registered (module re-import); the selftest checks tokens

# ---------------------------------------------------------------------------
# Advocate action instrument (the nb runner's r0 / r1 / r2 advocate line)
# ---------------------------------------------------------------------------

ACT_INSTRUMENT = "tmg_action_verdict"
ACT_1, ACT_2 = "ACT_1", "ACT_2"
ACT_VERDICTS = (ACT_1, ACT_2)
ACT_VERDICT_INSTRUCTION = (
    "End your response with exactly one line: VERDICT: ACT_1 or "
    "VERDICT: ACT_2."
)

try:
    register_instrument(ACT_INSTRUMENT, ACT_VERDICTS,
                        legacy_instruction=ACT_VERDICT_INSTRUCTION)
except ValueError:
    pass

# ---------------------------------------------------------------------------
# Registered cache names of the nb deliberation runner (16.13c). Defined
# here so the registration is one auditable place; the runner imports them
# and its selftest asserts every namespace empty before any call. The solo
# comparator's own names (tmg_solo_nb / game_solo_nb) live in
# run_game_singleagent, next to the binding ones.
# ---------------------------------------------------------------------------

NB_SCAFFOLD_NAME = "game_action_ncot"
NB_THIRD_NEUTRAL = "neutral"
NB_THIRD_PLANNER = "planner"
NB_THIRD_SEATS = (NB_THIRD_NEUTRAL, NB_THIRD_PLANNER)
NB_ARM_OF = {NB_THIRD_NEUTRAL: "third_neutral_nb",
             NB_THIRD_PLANNER: "third_planner_nb"}
NB_ROLE_PLAYER_R = "player_r_advocate_nb"
NB_ROLE_PLAYER_C = "player_c_advocate_nb"
NB_ROLE_NEUTRAL_READER = "neutral_reader_nb"
NB_ROLE_PLAN_MEDIATOR = "plan_mediator_nb"
NB_ROLE_IDS = (NB_ROLE_PLAYER_R, NB_ROLE_PLAYER_C, NB_ROLE_NEUTRAL_READER,
               NB_ROLE_PLAN_MEDIATOR)
NB_MODERATOR_PREFIX = "modgamenb"
#: Rows / votes CSV tag prefix of the nb runner (cgnb_game_<variant>). NOT
#: registered by 16.13c (which fixes scaffold / arms / roles / moderator
#: prefix), and deliberately NOT an extension of the binding runner's
#: 'cg_game': that runner counts n.startswith('cg_game') as its own tag
#: namespace, so 'cg_gamenb_*' would have been counted as binding CSVs.
NB_TAG_PREFIX = "cgnb_game"
#: Every name the nb runner writes caches under (substrings of file names).
NB_RUNNER_NAMES = ((NB_SCAFFOLD_NAME,) + tuple(NB_ARM_OF.values()) + NB_ROLE_IDS
                   + (NB_MODERATOR_PREFIX,))
#: The nb runner's own call_cache_path shape (do_call's naming), so that a
#: file carrying one of the names above is "own" only if it has this shape.
_NB_RUNNER_CACHE_RE = re.compile(
    rf"^cgd_.+_{NB_SCAFFOLD_NAME}_t\d+_(?:{'|'.join(NB_ARM_OF.values())})_"
    rf"\d{{3}}_\d{{2}}_[A-Za-z0-9_]+_(?:{'|'.join(NB_ROLE_IDS)}|"
    rf"{NB_MODERATOR_PREFIX}-[A-Za-z0-9_.\-]+)\.json$")

#: Solution-concept vocabulary this module's OWN prompt text must not use
#: (mirrors run_crowdgold_game.STEER_WORDS; kept in step by the selftest of
#: whichever runner imports both).
STEER_WORDS = ("nash", "pareto", "equilibri", "dominan", "maximin", "gold")


def steer_hits(text: str) -> list[str]:
    low = (text or "").lower()
    return [w for w in STEER_WORDS if w in low]


# ---------------------------------------------------------------------------
# The nb item: tmg_games' rendering with the question and legend swapped,
# the nb gold, and the regime carried as metadata (never placed in a prompt)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemViewNB(ItemView):
    """A non-binding rendering. Same fields as ItemView (same permutation,
    names, labels, table and scenario as the binding view of the same
    (game_id, sample_idx)); ``question_text`` / ``legend_text`` /
    ``gold_verdict`` are the nb ones; ``regime`` lets a runner dispatch the
    instrument without a second argument. Never equal to a plain ItemView."""

    regime: str = REGIME_NONBINDING


def nb_question_text(view: ItemView) -> str:
    """The non-binding question, byte-identical to
    scripts.run_crowdgold_game_nb.nb_question_text (the selftest pins it
    whenever that module is importable): the solo comparator and the
    collective must be asked the SAME question (16.13c)."""
    R, C = view.row_name, view.col_name
    return (
        f"{R} and {C} will each choose one option privately and at the same "
        f"time. There is no agreement between them and nothing to enforce "
        f"one: neither player can bind the other, and neither can be held to "
        f"anything said before they choose. A recommendation may be made to "
        f"both players -- one option for {R} and one option for {C} -- but "
        f"neither player is bound by it. If there is one pair of options that "
        f"should be recommended to both players, name it. If there is no such "
        f"pair, or if nothing in the table singles out one pair over another, "
        f"answer {NO_RECOMMENDATION}."
    )


def nb_legend_text(view: ItemView) -> str:
    """The non-binding legend (PLAN_xy pairs, NO_RECOMMENDATION, and the
    ACT_x notation the advocate seats answer in), byte-identical to
    scripts.run_crowdgold_game_nb.nb_legend_text."""
    R, C = view.row_name, view.col_name
    r1, r2 = view.row_labels
    c1, c2 = view.col_labels
    return (
        f"Recommended pairs are written PLAN_xy, where x is {R}'s option "
        f"number and y is {C}'s option number:\n"
        f"  PLAN_11  {R} takes {r1} and {C} takes {c1}\n"
        f"  PLAN_12  {R} takes {r1} and {C} takes {c2}\n"
        f"  PLAN_21  {R} takes {r2} and {C} takes {c1}\n"
        f"  PLAN_22  {R} takes {r2} and {C} takes {c2}\n"
        f"  {NO_RECOMMENDATION}  there is no single pair of options that "
        f"should be recommended to both players\n"
        f"A single player's own option is written ACT_1 or ACT_2 (its option "
        f"number): for {R}, ACT_1 is {r1} and ACT_2 is {r2}; for {C}, ACT_1 is "
        f"{c1} and ACT_2 is {c2}."
    )


def nb_item_text(view: ItemView) -> str:
    """Scenario (tmg_games, verbatim) + the nb question + the nb legend: the
    model-facing item text of the regime (ItemViewNB.text equals it)."""
    return f"{view.scenario_text}\n\n{nb_question_text(view)}\n\n{nb_legend_text(view)}"


def nb_gold_verdict(view: ItemView) -> str:
    """The registered nb gold, read off the view: tmg_games already maps
    gold_cell through the display permutation into ItemView.gold_verdict
    (PLAN_xy, or NO_UNIQUE_PLAN where gold_cell is null); the nb gold is
    that PLAN_xy, and NO_RECOMMENDATION in place of NO_UNIQUE_PLAN (the
    cannot_assist region). One signature for the runner, the analyzer and
    the solo comparator; idempotent on an ItemViewNB."""
    gv = view.gold_verdict
    return gv if gv in PLAN_VERDICTS else NO_RECOMMENDATION


def render_item_nb(game: Game, sample_idx: int, *, masked: bool = False) -> ItemViewNB:
    """The nb rendering: tmg_games.render_item (same permutation, names,
    table, scenario, masking) with the nb question, legend and gold."""
    base = render_item(game, sample_idx, masked=masked)
    kw = {f.name: getattr(base, f.name) for f in fields(base)}
    kw["question_text"] = nb_question_text(base)
    kw["legend_text"] = nb_legend_text(base)
    kw["gold_verdict"] = nb_gold_verdict(base)
    return ItemViewNB(**kw, regime=REGIME_NONBINDING)


def regime_of(view: ItemView) -> str:
    """The regime a view was rendered for; a plain ItemView is binding."""
    return getattr(view, "regime", REGIME_BINDING)


# ---------------------------------------------------------------------------
# Per-regime dispatch (the solo runner is one code path for both regimes)
# ---------------------------------------------------------------------------


def _check_regime(regime: str) -> str:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    return regime


def verdict_set_of(regime: str) -> tuple[str, ...]:
    return PLAN_VERDICT_SET if _check_regime(regime) == REGIME_BINDING else NB_VERDICT_SET


def verdict_instruction_of(regime: str) -> str:
    return (GAME_VERDICT_INSTRUCTION if _check_regime(regime) == REGIME_BINDING
            else NB_VERDICT_INSTRUCTION)


def no_answer_token_of(regime: str) -> str:
    """The regime's committal 'there is no answer' token."""
    return NO_UNIQUE_PLAN if _check_regime(regime) == REGIME_BINDING else NO_RECOMMENDATION


def parse_verdict_of(regime: str, text: str) -> str:
    return (parse_plan_verdict(text) if _check_regime(regime) == REGIME_BINDING
            else parse_nb_verdict(text))


def score_of(regime: str, verdict: str, gold_verdict: str) -> dict[str, bool]:
    return (score_plan(verdict, gold_verdict) if _check_regime(regime) == REGIME_BINDING
            else score_nb(verdict, gold_verdict))


def masked_floor_report_of(regime: str, records: Sequence[Mapping[str, Any]], *,
                           draws: int = 2000, seed: int = 7) -> dict[str, Any]:
    return (_binding_masked_floor_report(records, draws=draws, seed=seed)
            if _check_regime(regime) == REGIME_BINDING
            else masked_floor_report_nb(records, draws=draws, seed=seed))


def verdict_set_for(view: ItemView) -> tuple[str, ...]:
    return verdict_set_of(regime_of(view))


def verdict_instruction_for(view: ItemView) -> str:
    return verdict_instruction_of(regime_of(view))


def no_answer_token_for(view: ItemView) -> str:
    return no_answer_token_of(regime_of(view))


def parse_verdict_for(view: ItemView, text: str) -> str:
    return parse_verdict_of(regime_of(view), text)


def score_for(view: ItemView, verdict: str, gold_verdict: str) -> dict[str, bool]:
    return score_of(regime_of(view), verdict, gold_verdict)


# ---------------------------------------------------------------------------
# Actions, best replies, stake
# ---------------------------------------------------------------------------


def _check_stake(stake: str) -> str:
    if stake not in ("row", "col"):
        raise ValueError(f"stake must be row|col, got {stake!r}")
    return stake


def act_for_cell(view: ItemView, cell: Cell, stake: str) -> str:
    """ACT_x: the player's DISPLAYED option number at a canonical cell."""
    if _check_stake(stake) == "row":
        return f"ACT_{view.row_perm.index(cell[0]) + 1}"
    return f"ACT_{view.col_perm.index(cell[1]) + 1}"


_ACT_TOKEN_RE = re.compile(r"ACT_([12])")


def canonical_action(view: ItemView, act: str, stake: str) -> Optional[int]:
    """ACT_x -> the player's canonical action index (0/1); None otherwise."""
    m = _ACT_TOKEN_RE.fullmatch(act or "")
    if not m:
        return None
    d = int(m.group(1)) - 1
    return view.row_perm[d] if _check_stake(stake) == "row" else view.col_perm[d]


def act_for_plan(view: ItemView, verdict: str, stake: str) -> Optional[str]:
    """The player's ACT_x under a PLAN_xy verdict; None for anything else."""
    c = view.cell_for(verdict)
    return None if c is None else act_for_cell(view, c, stake)


def gold_act_for(view: ItemView, stake: str) -> Optional[str]:
    """The gold cell's action for the player (the advocate's gold); None on
    a NO_RECOMMENDATION / NO_UNIQUE_PLAN gold."""
    return act_for_plan(view, view.gold_verdict, stake)


def dominant_act_for(view: ItemView, stake: str) -> Optional[str]:
    """The player's strictly dominant action as ACT_x; None when it has
    none (both players have one only on DILEMMA game 111 and on no
    ASSURANCE game)."""
    r, c = dominant_strict(view.pA, view.pB)
    a = r if _check_stake(stake) == "row" else c
    if a is None:
        return None
    perm = view.row_perm if stake == "row" else view.col_perm
    return f"ACT_{perm.index(a) + 1}"


def private_act_for(view: ItemView, stake: str) -> Optional[str]:
    """The player's PRIVATE action: its action at the game's unique pure
    Nash cell, as ACT_x; None when the game has no unique pure Nash cell.
    Equals dominant_act_for whenever a dominant action exists (selftest)
    and is defined for both players on every DILEMMA_BOTH_PREFER game."""
    _check_stake(stake)
    if len(view.nash_verdicts) != 1:
        return None
    return act_for_plan(view, next(iter(view.nash_verdicts)), stake)


def player_payoff_at(view: ItemView, cell: Cell, stake: str) -> int:
    M = view.pA if _check_stake(stake) == "row" else view.pB
    return M[cell[0]][cell[1]]


def best_deviation_gain(view: ItemView, cell: Cell, stake: str) -> int:
    """The player's payoff gain from its best unilateral deviation from
    ``cell`` (0 when the cell is a best reply for that player)."""
    i, j = cell
    if _check_stake(stake) == "row":
        return max(0, view.pA[1 - i][j] - view.pA[i][j])
    return max(0, view.pB[i][1 - j] - view.pB[i][j])


def is_best_reply(view: ItemView, cell: Cell, stake: str) -> bool:
    return best_deviation_gain(view, cell, stake) == 0


def stake_undermined_nb(view: ItemView, verdict: str, stake: str) -> Optional[bool]:
    """16.13c: the player has a strictly profitable unilateral deviation
    from the recommended cell. None when ``verdict`` is not a PLAN_xy
    (NO_RECOMMENDATION / NOVERDICT: nothing to deviate from); False for a
    stake of "none"."""
    if stake == "none":
        return None if verdict not in NB_VERDICT_SET else False
    c = view.cell_for(verdict)
    if c is None:
        return None
    return not is_best_reply(view, c, _check_stake(stake))


# ---------------------------------------------------------------------------
# Verdict parsing (spacing variants of the exact tokens only; hedges are
# NOVERDICT, as on the binding instrument)
# ---------------------------------------------------------------------------

#: The NO_RECOMMENDATION spacing variant is collapsed ONLY in the token
#: position directly after a VERDICT separator (the separators
#: verdict_format's forced-line shape accepts, with optional markdown
#: emphasis in between), never in running prose: "no recommendation" is an
#: ordinary English phrase, and rewriting it anywhere would turn
#: "VERDICT: PLAN_22 -- no recommendation is stronger" into a false hedge.
_NO_REC_AFTER_MARKER_RE = re.compile(
    r"(\bVERDICT\b\s*[:\uff1a\-\u2013\u2014][\s*_]*)NO[ _\-]RECOMMENDATION\b", re.I)
#: Every LEGAL nb token as it appears after normalisation, for the hedge
#: check -- the binding parser's rule (two distinct legal tokens on the
#: line). UNRESOLVED is a legal token of the registered instrument
#: (register_instrument appends it) and is never offered, so a line naming
#: it next to a committal token is a hedge, not a committal verdict. The
#: binding token NO_UNIQUE_PLAN is not legal here, so "VERDICT: PLAN_12;
#: NO_UNIQUE_PLAN" reads as PLAN_12 in the solo comparator and the collective.
_ANY_NB_TOKEN_RE = re.compile(r"\bPLAN_[12][12]\b|\bNO_RECOMMENDATION\b|\b" + UNRESOLVED + r"\b")
_ACT_SPACING_RE = re.compile(r"\bACT[ _\-]([12])\b", re.I)
_ANY_ACT_TOKEN_RE = re.compile(r"\bACT_[12]\b|\b" + UNRESOLVED + r"\b")


def normalize_nb_tokens(text: str) -> str:
    """'VERDICT: NO RECOMMENDATION' / 'VERDICT: no-recommendation' ->
    'VERDICT: NO_RECOMMENDATION' (token position only, see
    _NO_REC_AFTER_MARKER_RE), then tmg_games' PLAN spacing normalisation.
    Near-misses stay near-misses; prose is never rewritten."""
    return normalize_plan_tokens(
        _NO_REC_AFTER_MARKER_RE.sub(lambda m: f"{m.group(1)}{NO_RECOMMENDATION}", text or ""))


def parse_nb_verdict(text: str) -> str:
    """One of the five nb tokens, or NOVERDICT. NO_UNIQUE_PLAN (the binding
    token) and a bare UNRESOLVED are NOVERDICT: neither is offered here. A
    line naming more than one distinct LEGAL token (PLAN_xy,
    NO_RECOMMENDATION, UNRESOLVED) is a hedge and is NOVERDICT -- including
    an explanatory parenthetical that names other cells (module docstring,
    VERDICT PARSING). The nb runner and the solo comparator both call this."""
    d = extract_verdict_detail(normalize_nb_tokens(text), NB_INSTRUMENT)
    if d.verdict not in NB_VERDICT_SET:
        return NOVERDICT
    if len(set(_ANY_NB_TOKEN_RE.findall(d.line))) > 1:
        return NOVERDICT
    return d.verdict


def normalize_act_tokens(text: str) -> str:
    """'ACT 1' / 'act-2' / 'Act_1' -> ACT_1 / ACT_2. 'ACT1' is a near-miss
    (mandatory separator), as 'PLAN12' is for the plan tokens."""
    return _ACT_SPACING_RE.sub(lambda m: f"ACT_{m.group(1)}", text or "")


def parse_act_verdict(text: str) -> str:
    """ACT_1 / ACT_2, or NOVERDICT: a bare UNRESOLVED, 'ACT1', 'ACT_3', a PLAN
    token, an option label ('M1'), or a line naming two distinct legal
    tokens (both actions, or an action next to UNRESOLVED) -- never
    coerced."""
    d = extract_verdict_detail(normalize_act_tokens(text), ACT_INSTRUMENT)
    if d.verdict not in ACT_VERDICTS:
        return NOVERDICT
    if len(set(_ANY_ACT_TOKEN_RE.findall(d.line))) > 1:
        return NOVERDICT
    return d.verdict


def score_nb(verdict: str, gold_verdict: str) -> dict[str, bool]:
    """verdict_format.score_against_gold on the nb instrument
    (NO_RECOMMENDATION is a committal gold and can be right or wrong)."""
    return score_against_gold(verdict, gold_verdict, NB_INSTRUMENT)


def score_act(verdict: str, gold_act: Optional[str]) -> Optional[dict[str, bool]]:
    """Advocate accuracy against the gold cell's action for its player:
    verdict_format.score_against_gold on the ACT instrument (NOVERDICT is
    wrong and stays in the denominator). None -- the seat is EXCLUDED, never
    coded wrong -- when the game has no gold action (a NO_RECOMMENDATION
    gold): 16.13c scores advocates "against the gold cell's action", which
    has no value there. The nb runner's score_act and the analyzer's
    action_correct are thin wrappers of this."""
    if gold_act is None:
        return None
    if gold_act not in ACT_VERDICTS:
        raise ValueError(f"gold_act must be one of {ACT_VERDICTS} or None, got {gold_act!r}")
    return score_against_gold(verdict, gold_act, ACT_INSTRUMENT)


# ---------------------------------------------------------------------------
# Masked floor on the nb verdict set. tmg_games.masked_floor_report keeps
# only records whose gold is in the BINDING set, which would drop every
# NO_RECOMMENDATION gold; this is the same computation on the nb set (the
# selftest pins equality with tmg_games' on binding records).
# ---------------------------------------------------------------------------


def _masked_floor_report_on(records: Sequence[Mapping[str, Any]],
                            verdict_set: Sequence[str], *, draws: int,
                            seed: int) -> dict[str, Any]:
    recs = [r for r in records if r.get("gold_verdict") in verdict_set]
    by_item: dict[str, list] = defaultdict(list)
    for r in recs:
        by_item[str(r["item_id"])].append(int(r.get("verdict") == r["gold_verdict"]))
    if not recs:
        return {"n": 0, "n_items": 0, "accuracy": None, "lo": None, "hi": None,
                "floor": None, "chance": 1 / len(verdict_set),
                "leak_suspected": False}

    def stat(xs: list) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    acc, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed)
    gold_counts = Counter(r["gold_verdict"] for r in recs)
    floor = max(gold_counts.values()) / len(recs)
    return {
        "n": len(recs), "n_items": len(by_item), "accuracy": acc,
        "lo": lo, "hi": hi, "floor": floor, "chance": 1 / len(verdict_set),
        "gold_counts": dict(gold_counts),
        "verdict_counts": dict(Counter(r.get("verdict") for r in recs)),
        "leak_suspected": lo is not None and lo > floor,
    }


def masked_floor_report_nb(records: Sequence[Mapping[str, Any]], *,
                           draws: int = 2000, seed: int = 7) -> dict[str, Any]:
    """tmg_games.masked_floor_report on the nb verdict set."""
    return _masked_floor_report_on(records, NB_VERDICT_SET, draws=draws, seed=seed)


# ---------------------------------------------------------------------------
# Cache-namespace helper for the nb runner's registered names
# ---------------------------------------------------------------------------


def cache_files_containing(subs: Sequence[str], out_dir: Path = OUT_DIR) -> list[Path]:
    """Existing files whose NAME contains any of ``subs`` (one directory
    scan). Cache files are cgd_<model>_<scaffold>_..., so a scaffold / arm /
    role name is a substring, never a prefix."""
    if not out_dir.is_dir():
        return []
    return sorted(p for p in out_dir.iterdir() if any(s in p.name for s in subs))


def nb_runner_foreign_files(out_dir: Path = OUT_DIR) -> tuple[list[Path], list[Path]]:
    """(files carrying any nb runner name, those NOT in the runner's own
    cache shape). At introduction both are empty; afterwards only the
    second must be."""
    carrying = cache_files_containing(NB_RUNNER_NAMES, out_dir)
    foreign = [p for p in carrying if not _NB_RUNNER_CACHE_RE.match(p.name)]
    return carrying, foreign


# ---------------------------------------------------------------------------
# Selftest (offline; no API, no network, no spend)
# ---------------------------------------------------------------------------

N_SELFTEST_SAMPLES = 4


def _selftest() -> int:
    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # ---- instruments ---------------------------------------------------------
    check("nb instrument registered with verdict_format: five tokens + UNRESOLVED",
          INSTRUMENTS.get(NB_INSTRUMENT) == NB_VERDICT_SET + (UNRESOLVED,))
    check("act instrument registered: ACT_1, ACT_2 + UNRESOLVED",
          INSTRUMENTS.get(ACT_INSTRUMENT) == ACT_VERDICTS + (UNRESOLVED,))
    check("NB_VERDICT_INSTRUCTION equals verdict_format's derived wording "
          "(allow_unresolved=False) and differs from the binding one only in "
          "the no-answer token",
          verdict_instruction(NB_INSTRUMENT, allow_unresolved=False)
          == NB_VERDICT_INSTRUCTION
          and NB_VERDICT_INSTRUCTION.replace(NO_RECOMMENDATION, NO_UNIQUE_PLAN)
          == GAME_VERDICT_INSTRUCTION)
    check("ACT_VERDICT_INSTRUCTION equals verdict_format's derived wording",
          verdict_instruction(ACT_INSTRUMENT, allow_unresolved=False)
          == ACT_VERDICT_INSTRUCTION)
    check("no nb instruction offers UNRESOLVED or NO_UNIQUE_PLAN",
          all(UNRESOLVED not in s and NO_UNIQUE_PLAN not in s
              for s in (NB_VERDICT_INSTRUCTION, ACT_VERDICT_INSTRUCTION)))
    check("nb instruments are distinct from tmg_games' binding instrument",
          NB_INSTRUMENT != GAME_INSTRUMENT and ACT_INSTRUMENT != GAME_INSTRUMENT
          and NB_VERDICT_SET != PLAN_VERDICT_SET)
    check("instructions and every registered name are free of forbidden and "
          "steer words",
          not any(forbidden_hits(s) or steer_hits(s) for s in
                  (NB_VERDICT_INSTRUCTION, ACT_VERDICT_INSTRUCTION)
                  + NB_RUNNER_NAMES + (NB_INSTRUMENT, ACT_INSTRUMENT)))
    check("regime dispatch: binding -> tmg_games' set/instruction/token, "
          "nonbinding -> the nb ones, anything else refused",
          verdict_set_of(REGIME_BINDING) == PLAN_VERDICT_SET
          and verdict_set_of(REGIME_NONBINDING) == NB_VERDICT_SET
          and verdict_instruction_of(REGIME_BINDING) == GAME_VERDICT_INSTRUCTION
          and verdict_instruction_of(REGIME_NONBINDING) == NB_VERDICT_INSTRUCTION
          and no_answer_token_of(REGIME_BINDING) == NO_UNIQUE_PLAN
          and no_answer_token_of(REGIME_NONBINDING) == NO_RECOMMENDATION
          and _raises(verdict_set_of, "bogus"))

    # ---- registered nb runner names -------------------------------------------
    check("nb runner names: scaffold game_action_ncot, arms third_neutral_nb / "
          "third_planner_nb, four _nb role ids, moderator prefix modgamenb, all "
          "distinct and none equal to a binding-runner name",
          NB_SCAFFOLD_NAME == "game_action_ncot"
          and NB_ARM_OF == {"neutral": "third_neutral_nb", "planner": "third_planner_nb"}
          and NB_ROLE_IDS == ("player_r_advocate_nb", "player_c_advocate_nb",
                              "neutral_reader_nb", "plan_mediator_nb")
          and NB_MODERATOR_PREFIX == "modgamenb"
          and len(set(NB_RUNNER_NAMES)) == len(NB_RUNNER_NAMES)
          and not ({"game_plan_ncot", "third_neutral", "third_planner",
                    "player_r_advocate", "player_c_advocate", "neutral_reader",
                    "plan_mediator", "modgame"} & set(NB_RUNNER_NAMES)))
    check("nb tag prefix cgnb_game does not extend the binding runner's "
          "'cg_game' (its tag namespace is n.startswith('cg_game')), carries no "
          "forbidden word, and no file in the cache dir starts with it yet or "
          "matches an own-shaped CSV of another prefix",
          NB_TAG_PREFIX == "cgnb_game" and not NB_TAG_PREFIX.startswith("cg_game")
          and not forbidden_hits(NB_TAG_PREFIX)
          and all(p.name.startswith(f"{NB_TAG_PREFIX}_")
                  for p in cache_files_with_prefix(NB_TAG_PREFIX)))
    carrying, foreign = nb_runner_foreign_files()
    check(f"no FOREIGN file in {OUT_DIR.name} carries any nb runner name "
          f"({len(carrying)} files carry one, {len(foreign)} foreign)"
          + (f" -- first foreign {foreign[0].name}" if foreign else ""),
          not foreign)
    check("cache_files_with_prefix finds nothing under the nb scaffold, arm or "
          "role names as prefixes (caches are cgd_-prefixed; this is the "
          "tmg_games idiom, the substring scan above is the real check)",
          all(cache_files_with_prefix(n) == [] for n in NB_RUNNER_NAMES))
    check("nb runner cache shape: own-shaped names match, binding-runner names "
          "and a wrong-arm name do not",
          _NB_RUNNER_CACHE_RE.match(
              "cgd_grok-4-1-fast-reasoning_game_action_ncot_t2560_third_neutral_nb_"
              "111_00_r0_player_r_advocate_nb.json")
          and _NB_RUNNER_CACHE_RE.match(
              "cgd_grok-4-1-fast-reasoning_game_action_ncot_t1024_third_planner_nb_"
              "111_02_synthesis_modgamenb-planner-grok-4-1-fast-reasoning.json")
          and not _NB_RUNNER_CACHE_RE.match(
              "cgd_grok-4-1-fast-reasoning_game_plan_ncot_t2560_third_neutral_"
              "111_00_r0_player_r_advocate.json")
          and not _NB_RUNNER_CACHE_RE.match(
              "cgd_m_game_action_ncot_t2560_third_neutral_111_00_r0_"
              "player_r_advocate_nb.json"))

    # ---- data ------------------------------------------------------------------
    try:
        games = load_games()
    except Exception as e:  # noqa: BLE001
        check(f"load_games() ({type(e).__name__}: {e})", False)
        print(f"\n{len(fails)} FAILED")
        return 1
    check(f"load_games returns {N_GAMES} games", len(games) == N_GAMES)

    # ---- rendering -------------------------------------------------------------
    views: dict[tuple[str, int], ItemViewNB] = {}
    masked_views: dict[tuple[str, int], ItemViewNB] = {}
    base_views: dict[tuple[str, int], ItemView] = {}
    for g in games:
        for idx in range(N_SELFTEST_SAMPLES):
            views[(g.game_id, idx)] = render_item_nb(g, idx)
            masked_views[(g.game_id, idx)] = render_item_nb(g, idx, masked=True)
            base_views[(g.game_id, idx)] = render_item(g, idx)
    all_views = list(views.values()) + list(masked_views.values())
    k0 = (games[0].game_id, 0)
    v0, b0 = views[k0], base_views[k0]
    check(f"render_item_nb yields ItemViewNB with regime nonbinding on all "
          f"{len(all_views)} views; a plain ItemView reads as binding",
          all(isinstance(v, ItemViewNB) and regime_of(v) == REGIME_NONBINDING
              for v in all_views)
          and all(regime_of(v) == REGIME_BINDING for v in list(base_views.values())[:20]))
    check("nb view shares permutation, names, labels, table, scenario, Nash set "
          "and best verdicts with the binding view of the same (game_id, idx)",
          all((v.row_perm, v.col_perm, v.row_name, v.col_name, v.row_labels,
               v.col_labels, v.table_text, v.scenario_text, v.nash_verdicts,
               v.row_best_verdict, v.col_best_verdict, v.pA, v.pB)
              == (b.row_perm, b.col_perm, b.row_name, b.col_name, b.row_labels,
                  b.col_labels, b.table_text, b.scenario_text, b.nash_verdicts,
                  b.row_best_verdict, b.col_best_verdict, b.pA, b.pB)
              for (k, v), b in zip(views.items(), (base_views[k] for k in views))))
    check("nb view is never equal to the binding view (question, legend and "
          "class differ) and its text differs",
          all(v != base_views[k] and v.text != base_views[k].text
              for k, v in views.items()))
    check("nb gold is NO_RECOMMENDATION exactly on cannot_assist views and equals "
          "the binding gold on every committal view; nb_gold_verdict(view) is "
          "idempotent on an nb view and never returns NO_UNIQUE_PLAN",
          all((v.gold_verdict == NO_RECOMMENDATION) == (v.region == REGION_CANNOT_ASSIST)
              for v in all_views)
          and all(v.gold_verdict == base_views[k].gold_verdict
                  for k, v in views.items() if v.region != REGION_CANNOT_ASSIST)
          and all(v.gold_verdict in NB_VERDICT_SET for v in all_views)
          and all(nb_gold_verdict(v) == v.gold_verdict for v in all_views)
          and all(nb_gold_verdict(b) == views[k].gold_verdict
                  for k, b in base_views.items()))
    check("nb gold maps back to the canonical gold cell (independent of "
          "tmg_games' gold_verdict: plan_for_cell on the pinned gold_cell)",
          all(v.cell_for(v.gold_verdict)
              == next(g.gold_cell for g in games if g.game_id == v.game_id)
              for v in views.values())
          and all(v.gold_verdict == plan_for_cell(g.gold_cell, v.row_perm, v.col_perm)
                  for v in views.values()
                  for g in (next(x for x in games if x.game_id == v.game_id),)
                  if g.gold_cell is not None))
    leaks = [(v.game_id, v.sample_idx, v.masked, forbidden_hits(s))
             for v in all_views for s in v.prompt_strings if forbidden_hits(s)]
    check(f"no forbidden word in any nb prompt string ({len(all_views)} views x "
          f"{len(all_views[0].prompt_strings)} strings)"
          + (f" -- first leak {leaks[0]}" if leaks else ""), not leaks)
    steer = [(v.game_id, v.sample_idx, steer_hits(s))
             for v in all_views for s in (v.question_text, v.legend_text)
             if steer_hits(s)]
    check(f"no solution-concept vocabulary ({STEER_WORDS}) in any nb question / "
          "legend" + (f" -- first {steer[0]}" if steer else ""), not steer)
    check("nb question states private simultaneous choice, no agreement, nothing "
          "to enforce one, a recommendation neither is bound by; never the "
          "binding question's 'binding agreement' / 'must then follow'; nb text "
          "names NO_RECOMMENDATION, ACT_1 / ACT_2 and never NO_UNIQUE_PLAN; "
          "ItemViewNB.text == nb_item_text",
          all("privately" in v.question_text and "no agreement" in v.question_text
              and "nothing to enforce" in v.question_text
              and "neither player is bound by it" in v.question_text
              and "binding agreement" not in v.text
              and "must then follow" not in v.text
              and NO_RECOMMENDATION in v.question_text
              and NO_RECOMMENDATION in v.legend_text
              and "ACT_1" in v.legend_text and "ACT_2" in v.legend_text
              and NO_UNIQUE_PLAN not in v.text
              and all(f"PLAN_{x}{y}" in v.legend_text for x in "12" for y in "12")
              and v.text == nb_item_text(v)
              for v in all_views))
    check("nb text carries no gold_type / gold / nash / region metadata",
          all(v.gold_type.lower() not in v.text.lower()
              and "gold" not in v.text.lower() and "nash" not in v.text.lower()
              and v.region not in v.text.lower().split() for v in all_views))
    check("nb rendering is deterministic in (game_id, sample_idx)",
          all(render_item_nb(g, 2) == views[(g.game_id, 2)] for g in games[:20]))
    templates = {surface_template(v) for v in masked_views.values()}
    check(f"masked nb surface template is identical across all "
          f"{len(masked_views)} masked views: {len(templates)} template(s)",
          len(templates) == 1)
    check("masked nb view shares permutation, names, labels and gold with its "
          "full twin",
          all((m.row_perm, m.col_perm, m.row_name, m.col_name, m.row_labels,
               m.col_labels, m.gold_verdict)
              == (u.row_perm, u.col_perm, u.row_name, u.col_name, u.row_labels,
                  u.col_labels, u.gold_verdict)
              for (k, m), u in zip(masked_views.items(),
                                   (views[k] for k in masked_views))))
    # ---- actions, best replies, stake ------------------------------------------
    rt_bad = []
    for v in views.values():
        for c in CELLS:
            for st in ("row", "col"):
                a = act_for_cell(v, c, st)
                if a not in ACT_VERDICTS or canonical_action(v, a, st) != (c[0] if st == "row" else c[1]):
                    rt_bad.append((v.game_id, v.sample_idx, c, st))
        for p in PLAN_VERDICTS:
            c = v.cell_for(p)
            if act_for_plan(v, p, "row") != act_for_cell(v, c, "row") or \
               act_for_plan(v, p, "col") != act_for_cell(v, c, "col"):
                rt_bad.append((v.game_id, v.sample_idx, p))
    check("act round trip: cell -> ACT_x -> canonical action on every view / "
          "cell / player; act_for_plan agrees with act_for_cell"
          + (f" -- first bad {rt_bad[0]}" if rt_bad else ""), not rt_bad)
    check("gold_act_for is None exactly on NO_RECOMMENDATION gold and names the "
          "gold cell's action otherwise; canonical_action refuses non-ACT tokens",
          all((gold_act_for(v, "row") is None) == (v.gold_verdict == NO_RECOMMENDATION)
              and (gold_act_for(v, "col") is None) == (v.gold_verdict == NO_RECOMMENDATION)
              for v in views.values())
          and all(gold_act_for(v, "row") == act_for_cell(v, v.cell_for(v.gold_verdict), "row")
                  for v in views.values() if v.gold_verdict != NO_RECOMMENDATION)
          and canonical_action(v0, NO_RECOMMENDATION, "row") is None
          and canonical_action(v0, "ACT_3", "col") is None)
    dil = [v for v in views.values() if v.gold_type == DILEMMA_BOTH_PREFER]
    asr = [v for v in views.values() if v.gold_type == ASSURANCE]
    check(f"DILEMMA_BOTH_PREFER ({len(dil)} views): both players have a private "
          "action (the unique Nash cell's), the pair of private actions IS the "
          "Nash cell, and it differs from the gold action for at least one player",
          len(dil) == 7 * N_SELFTEST_SAMPLES
          and all(private_act_for(v, "row") in ACT_VERDICTS
                  and private_act_for(v, "col") in ACT_VERDICTS
                  and plan_for_cell((canonical_action(v, private_act_for(v, "row"), "row"),
                                     canonical_action(v, private_act_for(v, "col"), "col")),
                                    v.row_perm, v.col_perm) in v.nash_verdicts
                  and (private_act_for(v, "row") != gold_act_for(v, "row")
                       or private_act_for(v, "col") != gold_act_for(v, "col"))
                  for v in dil))
    both_dom = sorted({v.game_id for v in dil
                       if dominant_act_for(v, "row") and dominant_act_for(v, "col")})
    one_dom = sorted({v.game_id for v in dil
                      if (dominant_act_for(v, "row") is None)
                      != (dominant_act_for(v, "col") is None)})
    check(f"DILEMMA structure pinned: both players have a dominant action on "
          f"{both_dom} only; exactly one player on {one_dom}; the dominant "
          "action equals the private action wherever it exists",
          both_dom == ["111"] and len(one_dom) == 6
          and all(dominant_act_for(v, st) in (None, private_act_for(v, st))
                  for v in views.values() for st in ("row", "col")))
    check(f"ASSURANCE ({len(asr)} views): no player has a dominant action and, "
          "with two Nash cells, no private action",
          len(asr) == 9 * N_SELFTEST_SAMPLES
          and all(dominant_act_for(v, "row") is None and dominant_act_for(v, "col") is None
                  and private_act_for(v, "row") is None and private_act_for(v, "col") is None
                  for v in asr))
    check("stake_undermined_nb on the gold cell: DILEMMA undermines exactly the "
          "players with a dominant action (both on 111, one on the other six; "
          "never neither); ASSURANCE undermines neither",
          all(stake_undermined_nb(v, v.gold_verdict, st)
              == (dominant_act_for(v, st) is not None)
              for v in dil for st in ("row", "col"))
          and all(stake_undermined_nb(v, v.gold_verdict, "row")
                  or stake_undermined_nb(v, v.gold_verdict, "col") for v in dil)
          and all(stake_undermined_nb(v, v.gold_verdict, "row") is False
                  and stake_undermined_nb(v, v.gold_verdict, "col") is False for v in asr))
    check("stake_undermined_nb: a Nash cell undermines neither player, a non-Nash "
          "cell undermines at least one, on every view",
          all((stake_undermined_nb(v, p, "row") is False
               and stake_undermined_nb(v, p, "col") is False) == (p in v.nash_verdicts)
              for v in views.values() for p in PLAN_VERDICTS))
    check("stake_undermined_nb: None on NO_RECOMMENDATION / NOVERDICT / "
          "NO_UNIQUE_PLAN for an advocate; stake none is False when committal, "
          "None otherwise",
          all(stake_undermined_nb(v, t, "row") is None
              for v in list(views.values())[:10]
              for t in (NO_RECOMMENDATION, NOVERDICT, NO_UNIQUE_PLAN, ""))
          and stake_undermined_nb(v0, NO_RECOMMENDATION, "none") is False
          and stake_undermined_nb(v0, "PLAN_11", "none") is False
          and stake_undermined_nb(v0, NOVERDICT, "none") is None)
    check("best_deviation_gain is 0 exactly on best replies and equals the payoff "
          "difference to the other action otherwise",
          all(best_deviation_gain(v, c, "row")
              == max(0, v.pA[1 - c[0]][c[1]] - v.pA[c[0]][c[1]])
              and best_deviation_gain(v, c, "col")
              == max(0, v.pB[c[0]][1 - c[1]] - v.pB[c[0]][c[1]])
              and (best_deviation_gain(v, c, "row") == 0) == is_best_reply(v, c, "row")
              for v in list(views.values())[:60] for c in CELLS)
          and _raises(best_deviation_gain, v0, (0, 0), "none"))

    # ---- verdict parsing ---------------------------------------------------------
    cases = (
        ("Both prefer it.\n\nVERDICT: PLAN_12", "PLAN_12"),
        ("**Verdict: plan 21**", "PLAN_21"),
        ("VERDICT: PLAN-22.", "PLAN_22"),
        ("verdict: plan_1_1", "PLAN_11"),
        ("VERDICT: NO_RECOMMENDATION", NO_RECOMMENDATION),
        ("VERDICT: NO RECOMMENDATION", NO_RECOMMENDATION),
        ("> VERDICT: no-recommendation", NO_RECOMMENDATION),
        ("I lean PLAN_11 but\n\nVERDICT: PLAN_22", "PLAN_22"),
        ("VERDICT: PLAN_12 (M1 with Q2)", "PLAN_12"),
        ("VERDICT: NO_UNIQUE_PLAN", NOVERDICT),
        ("VERDICT: NO UNIQUE PLAN", NOVERDICT),
        ("VERDICT: UNRESOLVED", NOVERDICT),
        ("VERDICT: PLAN_1", NOVERDICT),
        ("VERDICT: PLAN12", NOVERDICT),
        ("VERDICT: PLAN_12 or PLAN_21", NOVERDICT),
        ("VERDICT: PLAN_12; NO_RECOMMENDATION", NOVERDICT),
        ("VERDICT: PLAN_12; NO_UNIQUE_PLAN", "PLAN_12"),   # binding token is not legal here
        ("VERDICT: NO RECOMMENDATION, PLAN 12 and PLAN 21 tie", NOVERDICT),
        ("VERDICT: NO_RECOMMENDATION (PLAN_12 and PLAN_21 tie)", NOVERDICT),
        ("VERDICT: ACT_1", NOVERDICT),
        # UNRESOLVED is a legal (never offered) token: next to a committal one it is a hedge
        ("VERDICT: PLAN_12 or UNRESOLVED", NOVERDICT),
        ("VERDICT: PLAN_12 (UNRESOLVED)", NOVERDICT),
        ("VERDICT: NO_RECOMMENDATION or UNRESOLVED", NOVERDICT),
        # 'no recommendation' as PROSE is never rewritten into the token
        ("VERDICT: PLAN_22 -- no recommendation is stronger than this", "PLAN_22"),
        ("There is no recommendation I would rather make.\n\nVERDICT: PLAN_11", "PLAN_11"),
        ("**Verdict: no recommendation**", NO_RECOMMENDATION),
        ("VERDICT - NO RECOMMENDATION", NO_RECOMMENDATION),
        (NB_VERDICT_INSTRUCTION, NOVERDICT),
        ("They should both take option 1.", NOVERDICT),
        ("", NOVERDICT),
    )
    bad_cases = [(t, parse_nb_verdict(t), want) for t, want in cases
                 if parse_nb_verdict(t) != want]
    check(f"parse_nb_verdict: {len(cases) - len(bad_cases)}/{len(cases)} cases"
          + (f" -- first bad {bad_cases[0]!r}" if bad_cases else ""), not bad_cases)
    check("parse_nb_verdict only ever returns an nb token or NOVERDICT",
          all(parse_nb_verdict(t) in NB_VERDICT_SET + (NOVERDICT,) for t, _ in cases))
    act_cases = (
        ("Best for Morgan.\n\nVERDICT: ACT_1", ACT_1),
        ("**Verdict: act 2**", ACT_2),
        ("VERDICT: ACT-1.", ACT_1),
        ("verdict: act_2", ACT_2),
        ("VERDICT: ACT_2 (M2)", ACT_2),
        ("I lean ACT_1 but\n\nVERDICT: ACT_2", ACT_2),
        ("VERDICT: ACT1", NOVERDICT),
        ("VERDICT: ACT_3", NOVERDICT),
        ("VERDICT: ACT_1 or ACT_2", NOVERDICT),
        ("VERDICT: PLAN_12", NOVERDICT),
        ("VERDICT: UNRESOLVED", NOVERDICT),
        ("VERDICT: ACT_1 or UNRESOLVED", NOVERDICT),
        ("VERDICT: ACT_2 (UNRESOLVED)", NOVERDICT),
        (ACT_VERDICT_INSTRUCTION, NOVERDICT),
        ("", NOVERDICT),
    )
    bad_act = [(t, parse_act_verdict(t), want) for t, want in act_cases
               if parse_act_verdict(t) != want]
    check(f"parse_act_verdict: {len(act_cases) - len(bad_act)}/{len(act_cases)} cases"
          + (f" -- first bad {bad_act[0]!r}" if bad_act else ""), not bad_act)
    check("score_nb: NO_RECOMMENDATION gold scores correct/incorrect, NOVERDICT "
          "is noncommittal and never correct; NO_UNIQUE_PLAN never correct",
          score_nb(NO_RECOMMENDATION, NO_RECOMMENDATION) == {"correct": True, "noncommittal": False}
          and score_nb("PLAN_11", NO_RECOMMENDATION) == {"correct": False, "noncommittal": False}
          and score_nb(NOVERDICT, "PLAN_11") == {"correct": False, "noncommittal": True}
          and score_nb("PLAN_11", "PLAN_11")["correct"]
          and not score_nb(NO_UNIQUE_PLAN, NO_RECOMMENDATION)["correct"])
    check("score_act: correct iff the act matches the gold act; NOVERDICT "
          "noncommittal and wrong; None (EXCLUDED, never coded wrong) without a "
          "gold act; a malformed gold act is refused",
          score_act(ACT_1, ACT_1)["correct"] and not score_act(ACT_2, ACT_1)["correct"]
          and score_act(NOVERDICT, ACT_1) == {"correct": False, "noncommittal": True}
          and score_act(ACT_1, None) is None and score_act(NOVERDICT, None) is None
          and _raises(score_act, ACT_1, "PLAN_12"))
    # ---- the sibling deliberation runner, when present, must ask the SAME
    # ---- question and register the SAME instruments (16.13c) ---------------
    try:
        import scripts.run_crowdgold_game_nb as sib
    except ImportError:
        sib = None
    if sib is None:
        print("      (scripts.run_crowdgold_game_nb not importable; cross-runner "
              "wording checks skipped)")
    else:
        def _game(v: ItemView) -> Game:
            return next(g for g in games if g.game_id == v.game_id)
        sib_bad = [(v.game_id, v.sample_idx, v.masked) for v in all_views
                   if v.text != sib.nb_item_text(render_item(_game(v), v.sample_idx,
                                                             masked=v.masked))
                   or v.gold_verdict != sib.nb_gold_verdict(
                       render_item(_game(v), v.sample_idx, masked=v.masked))]
        check(f"run_crowdgold_game_nb asks the SAME nb question: nb_item_text and "
              f"nb_gold_verdict agree byte-for-byte on all {len(all_views)} views"
              + (f" -- first bad {sib_bad[0]}" if sib_bad else ""), not sib_bad)
        check("run_crowdgold_game_nb registers the SAME instruments: names, "
              "verdict sets, instructions, NO_RECOMMENDATION and ACT tokens",
              sib.NB_PLAN_INSTRUMENT == NB_INSTRUMENT
              and sib.ACT_INSTRUMENT == ACT_INSTRUMENT
              and sib.NB_VERDICT_SET == NB_VERDICT_SET
              and sib.ACT_VERDICTS == ACT_VERDICTS
              and sib.NB_VERDICT_INSTRUCTION == NB_VERDICT_INSTRUCTION
              and sib.ACT_INSTRUCTION == ACT_VERDICT_INSTRUCTION
              and sib.NO_RECOMMENDATION == NO_RECOMMENDATION)
        check("run_crowdgold_game_nb uses the registered names this module "
              "carries (scaffold, arms, role ids, moderator prefix, tag prefix)",
              sib.SCAFFOLD_NAME == NB_SCAFFOLD_NAME
              and dict(sib.ARM_OF) == NB_ARM_OF
              and sib.MODERATOR_PREFIX == NB_MODERATOR_PREFIX
              and sib.TAG_PREFIX == NB_TAG_PREFIX
              and tuple(r.role_id for r in sib.ALL_ROLES) == NB_ROLE_IDS)
        # (under `python -m scripts.tmg_games_nb` this file runs as __main__
        # while the runner imports scripts.tmg_games_nb: compare against the
        # imported module object, not this namespace)
        import scripts.tmg_games_nb as me
        check("run_crowdgold_game_nb IMPORTS this module's definitions rather than "
              "re-defining them (single definition site): question / legend / "
              "item text, gold, both parsers, gold / private / dominant action "
              "and the stake test are the same function objects",
              sib.nb_question_text is me.nb_question_text
              and sib.nb_legend_text is me.nb_legend_text
              and sib.nb_item_text is me.nb_item_text
              and sib.nb_gold_verdict is me.nb_gold_verdict
              and sib.parse_recommendation_verdict is me.parse_nb_verdict
              and sib.parse_act_verdict is me.parse_act_verdict
              and sib.gold_act_for is me.gold_act_for
              and sib.private_act_for is me.private_act_for
              and sib.dominant_act_for is me.dominant_act_for
              and sib.stake_undermined_nb is me.stake_undermined_nb
              and sib.score_act_nb is me.score_act and sib.score_nb is me.score_nb)
        keys200 = list(views)[:200]
        sib_items = {k: sib.NBGameItem(base_views[k]) for k in keys200}
        check("run_crowdgold_game_nb's parsers, private action, gold action, "
              "stake test and score_act agree with this module's (200 views x 4 "
              "cells; regression guard on the wrappers)",
              all(sib.parse_recommendation_verdict(t) == parse_nb_verdict(t)
                  for t, _ in cases)
              and all(sib.parse_act_verdict(t) == parse_act_verdict(t)
                      for t, _ in act_cases)
              and all(sib.private_action(sib_items[k], st) == private_act_for(views[k], st)
                      and sib.gold_action(sib_items[k], st) == gold_act_for(views[k], st)
                      and sib.dominant_action(sib_items[k], st) == dominant_act_for(views[k], st)
                      and all(sib.stake_undermined(rid, pv, sib_items[k])
                              == stake_undermined_nb(views[k], pv, st)
                              for pv in PLAN_VERDICTS + (NO_RECOMMENDATION, NOVERDICT,
                                                         NO_UNIQUE_PLAN, UNRESOLVED))
                      for k in keys200
                      for st, rid in (("row", NB_ROLE_PLAYER_R), ("col", NB_ROLE_PLAYER_C)))
              and all(sib.score_act(t, g)
                      == (None if score_act(t, g) is None else bool(score_act(t, g)["correct"]))
                      for t in (ACT_1, ACT_2, NOVERDICT, UNRESOLVED, "PLAN_12", "")
                      for g in (ACT_1, ACT_2, None)))
    check("per-view dispatch: an nb view parses / scores on the nb instrument, "
          "a binding view on tmg_games'",
          parse_verdict_for(v0, "VERDICT: NO RECOMMENDATION") == NO_RECOMMENDATION
          and parse_verdict_for(v0, "VERDICT: NO UNIQUE PLAN") == NOVERDICT
          and parse_verdict_for(b0, "VERDICT: NO UNIQUE PLAN") == NO_UNIQUE_PLAN
          and parse_verdict_for(b0, "VERDICT: NO RECOMMENDATION") == NOVERDICT
          and verdict_instruction_for(v0) == NB_VERDICT_INSTRUCTION
          and verdict_instruction_for(b0) == GAME_VERDICT_INSTRUCTION
          and no_answer_token_for(v0) == NO_RECOMMENDATION
          and verdict_set_for(b0) == PLAN_VERDICT_SET
          and score_for(v0, NO_RECOMMENDATION, NO_RECOMMENDATION)["correct"]
          and not score_for(b0, NO_RECOMMENDATION, NO_UNIQUE_PLAN)["correct"])

    # ---- masked floor on the nb set --------------------------------------------
    rng = random.Random(11)
    at_floor = [{"item_id": v.game_id, "gold_verdict": v.gold_verdict,
                 "verdict": rng.choice(NB_VERDICT_SET)} for v in views.values()]
    r_floor = masked_floor_report_nb(at_floor, draws=300, seed=1)
    want_floor = max(Counter(r["gold_verdict"] for r in at_floor).values()) / len(at_floor)
    check(f"masked_floor_report_nb keeps every record (n={r_floor['n']} of "
          f"{len(at_floor)}, NO_RECOMMENDATION golds included); random guessing "
          f"is not flagged (acc {r_floor['accuracy']:.3f}, floor {r_floor['floor']:.3f})",
          r_floor["n"] == len(at_floor) and not r_floor["leak_suspected"]
          and abs(r_floor["floor"] - want_floor) < 1e-12
          and NO_RECOMMENDATION in r_floor["gold_counts"])
    check("tmg_games.masked_floor_report would DROP the NO_RECOMMENDATION golds "
          "(why the nb report exists)",
          _binding_masked_floor_report(at_floor, draws=20, seed=1)["n"]
          == sum(1 for r in at_floor if r["gold_verdict"] != NO_RECOMMENDATION))
    perfect = [{"item_id": v.game_id, "gold_verdict": v.gold_verdict,
                "verdict": v.gold_verdict} for v in views.values()]
    r_perf = masked_floor_report_nb(perfect, draws=300, seed=1)
    check("masked_floor_report_nb: perfect masked accuracy is flagged LEAK SUSPECTED",
          r_perf["leak_suspected"])
    bind_recs = [{"item_id": v.game_id, "gold_verdict": v.gold_verdict,
                  "verdict": rng.choice(PLAN_VERDICT_SET)} for v in base_views.values()]
    check("on binding records the shared computation equals "
          "tmg_games.masked_floor_report exactly (same draws, same seed)",
          _masked_floor_report_on(bind_recs, PLAN_VERDICT_SET, draws=200, seed=3)
          == _binding_masked_floor_report(bind_recs, draws=200, seed=3)
          and masked_floor_report_of(REGIME_BINDING, bind_recs, draws=50, seed=2)
          == _binding_masked_floor_report(bind_recs, draws=50, seed=2)
          and masked_floor_report_of(REGIME_NONBINDING, at_floor, draws=50, seed=2)
          == masked_floor_report_nb(at_floor, draws=50, seed=2))

    # ---- this module introduces no cache tag ------------------------------------
    check("no 'tmg_' cache file exists (this module writes nothing; runners "
          "assert their own tags)", cache_files_with_prefix("tmg_") == [])

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def _raises(fn: Callable, *a: Any) -> bool:
    try:
        fn(*a)
    except ValueError:
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TMGBench classic 2x2 games, NON-BINDING regime: question, "
                    "verdict instruments, nb gold, stake (no API calls)")
    ap.add_argument("--selftest", action="store_true",
                    help="offline self-check; no API keys, no network")
    ap.add_argument("--show", default=None, metavar="GAME_ID",
                    help="print one nb-rendered item (with metadata)")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--extract", default=None, metavar="TEXT",
                    help="parse a joint nb verdict from TEXT")
    ap.add_argument("--extract-act", default=None, metavar="TEXT",
                    help="parse an advocate ACT verdict from TEXT")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    did = False
    if args.extract is not None:
        did = True
        print(parse_nb_verdict(args.extract))
    if args.extract_act is not None:
        did = True
        print(parse_act_verdict(args.extract_act))
    if args.show:
        did = True
        games = load_games()
        g = next((x for x in games if x.game_id == args.show), None)
        if g is None:
            print(f"unknown game id {args.show!r}")
            return 2
        v = render_item_nb(g, args.sample_idx, masked=args.masked)
        print(v.text)
        print("\n--- metadata (never shown to a model) ---")
        print(f"game {v.game_id} idx {v.sample_idx} masked={v.masked} regime={v.regime} "
              f"gold_type={v.gold_type} region={v.region}")
        print(f"row_perm={v.row_perm} col_perm={v.col_perm} gold={v.gold_verdict} "
              f"nash={sorted(v.nash_verdicts)} gold_act row/col="
              f"{gold_act_for(v, 'row')}/{gold_act_for(v, 'col')} private_act row/col="
              f"{private_act_for(v, 'row')}/{private_act_for(v, 'col')} "
              f"dominant_act row/col={dominant_act_for(v, 'row')}/"
              f"{dominant_act_for(v, 'col')}")
    if not did:
        ap.print_help(sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
