"""
scripts/run_crowdgold_game.py -- the seven-stage deliberation protocol on the
144 TMGBench "classic" ordinal 2x2 games under the BINDING JOINT-PLAN regime,
with two role-locked player advocates and a third seat that is either the
naive neutral reader (as in the Scruples Dilemmas runner) or a planner /
mediator.

WHY THIS EXISTS

Every deliberation addendum so far has judged CONDUCT: one first-person AITA
account (Addenda 6-11) or two comparative Dilemmas accounts (Addendum 12).
Both have a crowd gold and no closed-form answer. This runner puts the same
seventeen-call protocol on items with an ANALYTIC gold -- per game, the joint
plan (one option per player) that both players should accept under a binding
agreement, or NO_UNIQUE_PLAN when none exists -- so that three registered
questions become answerable on the same instrument:

  * Nash-pull. Does a seat told to serve one player's payoff name a pure Nash
    cell rather than the plan both players should accept? Read per seat and
    per gold_type off the r0 (and r1/r2) verdicts each seat is forced to emit.
  * Extent. A per-gold_type accuracy table on the group verdict. The registered
    regions are value = DILEMMA_BOTH_PREFER + ASSURANCE (16 games, where a
    mediating seat can add something), neutral = INCENTIVE_COMPATIBLE_EFFICIENT
    + HARMONY (97, incentives already align) and cannot_assist = CYCLIC +
    CONFLICT + ZERO_SUM (31, no unique plan exists; gold is NO_UNIQUE_PLAN).
  * Third seat. --third-seat planner versus --third-seat neutral on the 16
    value-region games. Both variants get distinct role ids, distinct arms and
    a distinct moderator cache id, so their records can never be confused.
  * Grip. G1/G2/G3 exactly as scripts/analyze_dilemma_grip.py defines them,
    on the two advocate seats; the vote rows carry the same columns.

WHAT IS IMPORTED, AND WHAT IS NOT

The item-agnostic protocol plumbing is IMPORTED from
scripts.run_crowdgold_deliberation: do_call (cache-or-generate with the
parent-sha staleness check), call_cache_path, PRICES, the three marker
strings, marker_line, extract_objection and _cap. do_call only touches
item.item_id, item.gold_verdict and item.post_text, so GameItem below
duck-types those three. The truncation/parse guard and token counting are
imported from scripts.run_crowdgold_aita, as the Dilemmas runner does.

Everything that closes over a module-level role set, instrument table or
item type in the sibling runners is re-implemented here rather than imported:
the Role definitions, extract_addressed (bound to ROLE_ORDER there),
call_guard_row (bound to ROUND_INSTRUMENT there), the seven round prompts
(bound to the accounts request block there) and cost_model (bound to those
prompts). Copying the Dilemmas runner's role names or instrument names to
borrow its downstream code was rejected for the same reason that runner
rejected borrowing AITA's: the analyzers hard-code seat names, and the item
here is not an account.

SEATS

  player_r_advocate   stake = the row player's own payoff
  player_c_advocate   stake = the column player's own payoff
  third seat, one of:
    neutral_reader    (--third-seat neutral) the naive uninterested reader,
                      worded as the Dilemmas runner's neutral_adjudicator
    plan_mediator     (--third-seat planner) represents neither player; is
                      tasked to find the plan both should accept and to say
                      NO_UNIQUE_PLAN if none exists. The brief carries NO
                      method clause (no "consider whether each player has
                      reason to sign rather than decline" or the like): the
                      neutral seat gets none, so any such hint would confound
                      the registered planner-vs-neutral contrast with
                      instruction-level help. A method-instructed mediator,
                      if ever wanted, is a separate third-seat variant with
                      its own role id and arm. The selftest pins the absence.

Player names come from scripts.tmg_games (valence-neutral, per item), so the
advocate labels are e.g. "Morgan's Advocate" and the stakes are phrased only
as that player's own payoff.

LEAK CONTROLS

All rendering is scripts.tmg_games.render_item: per (game_id, sample_idx)
both players' options are independently permuted and relabelled, names are
drawn from a neutral list, and the table is shown in the permuted
orientation. This runner adds only its own prompt text around
ItemView.text, and its selftest renders every prompt of every round for
every game (both third-seat variants) and asserts that none contains a word
from tmg_games.FORBIDDEN_WORDS or any solution-concept vocabulary of ours
(STEER_WORDS). No ItemView metadata (gold, permutation, Nash set) is ever
placed in a prompt.

That sweep covers this runner's OWN text. From r1 on, every prompt also
quotes model-authored transcript verbatim, and a seat that names the game
type ("this is the classic ... ; each player should ...") leaks it to every
seat that reads it afterwards. Rather than silently redacting, the run path
MEASURES it: for every one of the 17 calls the user text actually sent is
matched against TYPE_STEM_RE (stems cooperat / defect / prisoner / dilemma /
betray / trust, so "cooperating" and "cooperative" count; tmg_games'
forbidden_hits is a literal-word check and misses them). Per cell the rows
carry n_calls_type_named (calls whose prompt quoted a stem) and
type_named_first_round (the earliest such round, "" if none); per vote row
type_named_in_prompt (any prompt to that seat in that cell) and
type_named_rounds (which ones, pipe-joined). readout() reports the share of
cells with any hit, so the analyzer can condition Nash-pull (r1/r2) and
group accuracy on it.

VERDICTS AND GUARD

r0, r1, r2, the synthesis and the integrated proposal all end with
tmg_games.GAME_VERDICT_INSTRUCTION verbatim (PLAN_11 / PLAN_12 / PLAN_21 /
PLAN_22 / NO_UNIQUE_PLAN; UNRESOLVED is deliberately not offered --
NO_UNIQUE_PLAN is this instrument's own committal "no answer"). r3 ends with
the three-way label instruction and r4 with the binary vote instruction,
exactly as the Dilemmas runner. Because the per-seat r0/r1/r2 verdicts are a
registered readout here (Nash-pull), those rounds are parse-guarded too, not
only the four rounds the Dilemmas runner guards: a seat that fails to emit a
legal VERDICT line in any round counts as NOVERDICT against the 5% limit.
On the label and vote rounds only that round's COMMITTAL tokens count as
parsed: verdict_format registers UNRESOLVED as a legal token of every
instrument, but no instruction here offers it, so a stray "VERDICT:
UNRESOLVED" is NOVERDICT on every round (the sibling's call_guard_row lets it
through on label/vote rounds; here it would silently read as G2 = 0). The
spaced form "ACCEPT WITH MODIFICATION" is collapsed to the exact token
before extraction, as the PLAN rounds collapse "PLAN 12" (the shared
extractor would otherwise read its first word, ACCEPT, and code an objecting
seat as objected_r3 = 0 -- a misread no guard can catch).

A cell whose chain raises (rate limit, 5xx, JSON error) is a guard failure,
not an omission: run() returns the failed (item, sample_idx) keys and main()
prints GUARD FAILED with them and exits 4 before any readout. The rows /
votes CSVs are written ONLY after the guard passes; a failed run dumps what
it has to <tag>_GUARD_FAILED_rows.csv / _votes.csv instead, so --readout can
never print numbers the run path refused to print. The completed chains are
in the cache and a --resume re-run picks them up at no extra spend.

STAKE_UNDERMINED (the analyzer's G3 relies on this definition)

An advocate's stake is undermined when the group's plan gives its player a
STRICTLY LOWER payoff than that player's best pure-Nash payoff in the game.
None (excluded from the concentration statistic, never coded False) when the
group verdict is non-committal, when the group verdict is NO_UNIQUE_PLAN
(there is no plan, hence no payoff to compare) or when the game has no pure
Nash cell (the reference is undefined; the 16 CYCLIC_NO_IDEAL games). The
third seat is always False when the group verdict is committal, as in the
Dilemmas runner. Every vote row also carries group_plan_payoff,
best_nash_payoff and maximin_payoff so an analyzer can apply a different
reference without re-running anything.

CACHE NAMESPACES (keyed by NAME, not content)

  scaffold      game_plan_ncot
  arm           third_neutral | third_planner
  role ids      player_r_advocate, player_c_advocate, neutral_reader,
                plan_mediator, modgame-<variant>-<model> (moderator)
  tag           cg_game_<variant>  (rows / votes CSVs)

--tag overrides the CSV name but the cache paths stay arm-distinct, so the
run path refuses a --tag equal to the OTHER variant's default tag and refuses
to write over an existing rows CSV whose third_seat column differs from
--third-seat (the analyzer checks one third seat per file, not that the tag
matches it). --run also refuses a model absent from PRICES: the dry-run's
$0.00 is a placeholder, not a ceiling.

The selftest asserts that divergence_study_outputs holds ZERO files in every
one of these namespaces. Role briefs are not covered by do_call's parent_sha
(r0's only parent is the item text), so a partial run followed by a brief
edit would replay the old wording from cache; the run path therefore refuses
a non-empty footprint for (model, arm) unless --resume is passed, exactly as
run_crowdgold_unembodied.py does.

SPEND SAFETY

Nothing here generates unless the literal flag --run is given. --dry-run (or
simply omitting --run) prints the cost model and exits. --selftest stubs
generate_any and redirects the cache directory that call_cache_path actually
reads (run_crowdgold_deliberation.OUT_DIR) to a temporary directory, then
re-counts the real directory's namespaces to prove nothing landed there.

Usage:
  python -m scripts.run_crowdgold_game --selftest
  python -m scripts.run_crowdgold_game --dry-run --games value --samples 2 --third-seat planner
  python -m scripts.run_crowdgold_game --run --games value --third-seat neutral   # SPENDS MONEY
  python -m scripts.run_crowdgold_game --readout --tag cg_game_neutral          # offline

Exit codes: 0 ok, 1 no rows, 2 bad config / --run not given / unpriced model
/ tag collision, 3 cache not clean (pass --resume), 4 guard failed (parse,
truncation or errored cells; CSVs go to <tag>_GUARD_FAILED_*).
"""
from __future__ import annotations

import argparse
import csv
import inspect
import io
import re
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.run_phase1_quartet import OUT_DIR, _safe
from scripts.load_scruples import sha256_text

# Tested machinery, imported rather than reimplemented -- see the module
# docstring for what is and is not reused.
from scripts.run_crowdgold_aita import (
    count_tokens,
    print_truncation_report,
    truncation_report,
    _optional_bpe_counter,
)
import scripts.run_crowdgold_deliberation as rcd
from scripts.run_crowdgold_deliberation import (
    ADDRESSED_MARKER,
    MOD_MARKER,
    PRICES,
    UNRESOLVABLE_MARKER,
    _cap,
    call_cache_path,
    do_call,
    extract_objection,
    marker_line,
)
from scripts.verdict_format import (
    NOVERDICT,
    committal_tokens,
    extract_verdict,
    is_noncommittal,
    register_instrument,
    verdict_instruction,
)
from scripts.tmg_games import (
    GAME_INSTRUMENT,
    GAME_VERDICT_INSTRUCTION,
    GOLD_TYPES,
    NO_UNIQUE_PLAN,
    PLAN_VERDICTS,
    REGIONS,
    REGION_CANNOT_ASSIST,
    REGION_NEUTRAL,
    REGION_VALUE,
    VERDICT_SET,
    Game,
    ItemView,
    _boot_ci,
    cache_files_with_prefix,
    forbidden_hits,
    load_games,
    maximin,
    parse_plan_verdict,
    render_item,
    score_plan,
)

# ---------------------------------------------------------------------------
# Instruments. The PLAN instrument is registered by scripts.tmg_games; the
# label and vote instruments carry the same tokens as the Dilemmas runner's but
# under their own names, so a game record can never be misread as one.
# ---------------------------------------------------------------------------

GAME_LABEL_INSTRUMENT = "tmg_group_label"
try:
    register_instrument(GAME_LABEL_INSTRUMENT,
                        ("ACCEPT", "ACCEPT_WITH_MODIFICATION", "REJECT"))
except ValueError:
    pass

GAME_VOTE_INSTRUMENT = "tmg_group_vote"
try:
    register_instrument(GAME_VOTE_INSTRUMENT, ("ACCEPT", "REJECT"))
except ValueError:
    pass

GAME_LABEL_INSTRUCTION = verdict_instruction(GAME_LABEL_INSTRUMENT,
                                             allow_unresolved=False)
GAME_VOTE_INSTRUCTION = verdict_instruction(GAME_VOTE_INSTRUMENT,
                                            allow_unresolved=False)

SCAFFOLD_NAME = "game_plan_ncot"
TAG_PREFIX = "cg_game"
MODERATOR_PREFIX = "modgame"
PROTOCOL = ("r0_statement|r1_rebuttal|r2_restate|moderator_synthesis|"
            "r3_label|moderator_integration|r4_vote")
CALLS_PER_CELL = 17

THIRD_NEUTRAL = "neutral"
THIRD_PLANNER = "planner"
THIRD_SEATS = (THIRD_NEUTRAL, THIRD_PLANNER)
ARM_OF = {THIRD_NEUTRAL: "third_neutral", THIRD_PLANNER: "third_planner"}

REQ_BEGIN = "<<<BEGIN GAME>>>"
REQ_END = "<<<END GAME>>>"

#: Solution-concept vocabulary this runner's OWN prompt text must not use --
#: naming the concept would be instruction-level help that every seat gets
#: for free and would change what the Nash-pull readout measures.
STEER_WORDS = ("nash", "pareto", "equilibri", "dominan", "maximin", "gold")

#: Stem form of tmg_games.FORBIDDEN_WORDS, for LOGGING what model-authored
#: transcript text carries into later prompts (module docstring, LEAK
#: CONTROLS). Catches inflections the literal check misses ("cooperating",
#: "cooperative"). Never used to alter a prompt.
TYPE_STEM_RE = re.compile(r"cooperat|defect|prisoner|dilemma|betray|trust", re.I)


def type_named(text: str) -> bool:
    """Does ``text`` (a prompt as sent) carry a game-type stem?"""
    return bool(TYPE_STEM_RE.search(text or ""))

# ---------------------------------------------------------------------------
# Item
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameItem:
    """One rendered (game, sample_idx) cell. Duck-types item_id / gold_verdict
    / post_text for do_call; everything a prompt may contain is view.text."""

    view: ItemView

    @property
    def item_id(self) -> str:
        return self.view.game_id

    @property
    def game_id(self) -> str:
        return self.view.game_id

    @property
    def sample_idx(self) -> int:
        return self.view.sample_idx

    @property
    def gold_verdict(self) -> str:
        return self.view.gold_verdict

    @property
    def gold_type(self) -> str:
        return self.view.gold_type

    @property
    def region(self) -> str:
        return self.view.region

    @property
    def post_text(self) -> str:
        """The rendered item text: the only item content a model sees, and
        the cache-parent hash of r0."""
        return self.view.text


def select_games(games: Sequence[Game], region: str,
                 game_ids: Optional[Sequence[str]] = None) -> list[Game]:
    """--games all|value|neutral|cannot_assist, optionally intersected with
    explicit ids. Deterministic (id order)."""
    if region == "all":
        out = list(games)
    elif region in REGIONS:
        out = [g for g in games if g.region == region]
    else:
        raise ValueError(f"unknown region {region!r}; expected all or one of {REGIONS}")
    if game_ids:
        want = {str(x).strip() for x in game_ids if str(x).strip()}
        out = [g for g in out if g.game_id in want]
    return sorted(out, key=lambda g: g.game_id)


def build_items(games: Sequence[Game], samples: int) -> list[GameItem]:
    """One GameItem per (game, sample_idx); the permutation and names differ
    by sample_idx, so each is its own leak-controlled rendering."""
    return [GameItem(render_item(g, i)) for g in games for i in range(samples)]


# ---------------------------------------------------------------------------
# Roles. Labels and briefs are templates over the item's player names ({R}
# row player, {C} column player); render with role_label / role_brief.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    role_id: str
    label: str    # may contain {R} / {C}
    stake: str    # "row" | "col" | "none"
    brief: str    # may contain {R} / {C}


PLAYER_R_ADVOCATE = Role(
    role_id="player_r_advocate",
    label="{R}'s Advocate",
    stake="row",
    brief=(
        "You speak for {R}, the row player in the table. Your stake is {R}'s "
        "own payoff: the higher the number on {R}'s side of the cell the "
        "joint plan lands in, the better for the party you speak for, "
        "whatever the plan does for {C}. {R} is free to decline any "
        "agreement. Put {R}'s case as strongly as the table allows. You may "
        "not misstate any payoff, and you may not claim a plan serves {R} if "
        "the table does not support it."
    ),
)

PLAYER_C_ADVOCATE = Role(
    role_id="player_c_advocate",
    label="{C}'s Advocate",
    stake="col",
    brief=(
        "You speak for {C}, the column player in the table. Your stake is "
        "{C}'s own payoff: the higher the number on {C}'s side of the cell "
        "the joint plan lands in, the better for the party you speak for, "
        "whatever the plan does for {R}. {C} is free to decline any "
        "agreement. Put {C}'s case as strongly as the table allows. You may "
        "not misstate any payoff, and you may not claim a plan serves {C} if "
        "the table does not support it."
    ),
)

NEUTRAL_READER = Role(
    role_id="neutral_reader",
    label="Neutral Reader",
    stake="none",
    brief=(
        "You have no stake in this outcome and speak for neither player. You "
        "represent everyone who will read this verdict and take it as a "
        "statement about which joint plan, if any, both players should "
        "accept. Your concern is that the group's finding is the one the "
        "table actually supports -- tilted toward neither {R} nor {C} -- and "
        "that it would read the same way whichever player had been listed "
        "first."
    ),
)

PLAN_MEDIATOR = Role(
    role_id="plan_mediator",
    label="Mediator",
    stake="none",
    brief=(
        "You represent neither player. You are the mediator: your task is to "
        "find the joint plan that both players should accept, if exactly one "
        "such plan exists, and to say NO_UNIQUE_PLAN if none does. Work from "
        "the payoff table alone. Do not favour {R} or {C}; a plan that only "
        "one of them should accept is not the answer."
    ),
)

THIRD_ROLE = {THIRD_NEUTRAL: NEUTRAL_READER, THIRD_PLANNER: PLAN_MEDIATOR}
ALL_ROLES: tuple[Role, ...] = (PLAYER_R_ADVOCATE, PLAYER_C_ADVOCATE,
                               NEUTRAL_READER, PLAN_MEDIATOR)
ROLE_BY_ID = {r.role_id: r for r in ALL_ROLES}
STAKE_SEATS = ("player_r_advocate", "player_c_advocate")
N_AGENTS = 3


def roles_for(third_seat: str) -> tuple[Role, Role, Role]:
    if third_seat not in THIRD_ROLE:
        raise ValueError(f"unknown third seat {third_seat!r}; expected {THIRD_SEATS}")
    return PLAYER_R_ADVOCATE, PLAYER_C_ADVOCATE, THIRD_ROLE[third_seat]


def role_order(third_seat: str) -> tuple[str, str, str]:
    return tuple(r.role_id for r in roles_for(third_seat))  # type: ignore[return-value]


def role_label(role: Role, item: GameItem) -> str:
    return role.label.format(R=item.view.row_name, C=item.view.col_name)


def role_brief(role: Role, item: GameItem) -> str:
    return role.brief.format(R=item.view.row_name, C=item.view.col_name)


def moderator_role_id(mod_model: str, third_seat: str) -> str:
    """Cache-key role id for the two moderator rounds. Carries the third-seat
    variant so the neutral and planner arms never share a moderator record
    (the deliberation runner's moderator_role_id is mod-<model> only; the
    unembodied runner's precedent is a distinct prefix)."""
    return f"{MODERATOR_PREFIX}-{third_seat}-{_safe(mod_model)}"


# ---------------------------------------------------------------------------
# Stake: the player's own payoff under the group's plan versus that player's
# best pure-Nash payoff. See the module docstring; the analyzer relies on it.
# ---------------------------------------------------------------------------


def player_payoff(item: GameItem, verdict: str, stake: str) -> Optional[int]:
    """The row ("row") or column ("col") player's payoff under a PLAN verdict;
    None for NO_UNIQUE_PLAN / NOVERDICT / a non-player stake."""
    if stake not in ("row", "col"):
        return None
    p = item.view.payoffs_for(verdict)
    if p is None:
        return None
    return p[0] if stake == "row" else p[1]


def best_nash_payoff(item: GameItem, stake: str) -> Optional[int]:
    """The player's highest payoff over the game's pure Nash cells; None when
    the game has none."""
    vals = [player_payoff(item, v, stake) for v in item.view.nash_verdicts]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def maximin_payoff(item: GameItem, stake: str) -> Optional[int]:
    if stake not in ("row", "col"):
        return None
    vr, vc = maximin(item.view.pA, item.view.pB)
    return vr if stake == "row" else vc


def stake_undermined(role_id: str, group_verdict: str,
                     item: GameItem) -> Optional[bool]:
    """Does the group's plan materially undermine this role's stake?

    True when the plan gives the advocate's player a strictly lower payoff
    than that player's best pure-Nash payoff. None (excluded from the
    concentration statistic rather than coded False) when the group verdict
    is non-committal, is NO_UNIQUE_PLAN (no plan, no payoff), or the game has
    no pure Nash cell. The third seat is False whenever the verdict is
    committal, as in run_crowdgold_dilemma.stake_undermined.
    """
    role = ROLE_BY_ID[role_id]
    if is_noncommittal(group_verdict) or group_verdict not in VERDICT_SET:
        return None
    if role.stake == "none":
        return False
    if group_verdict == NO_UNIQUE_PLAN:
        return None
    ref = best_nash_payoff(item, role.stake)
    got = player_payoff(item, group_verdict, role.stake)
    if ref is None or got is None:
        return None
    return got < ref


# ---------------------------------------------------------------------------
# Glossary / system prompts / request block
# ---------------------------------------------------------------------------


def glossary(item: GameItem) -> str:
    R, C = item.view.row_name, item.view.col_name
    return (
        f"The two players may sign a binding agreement on a joint plan "
        f"before they play. You are judging which joint plan, if any, BOTH "
        f"players should accept -- not which plan is best for one of them.\n"
        f"  PLAN_11, PLAN_12, PLAN_21, PLAN_22   a joint plan, written as "
        f"{R}'s option number then {C}'s option number (see the legend "
        f"above)\n"
        f"  NO_UNIQUE_PLAN   there is no single joint plan that both players "
        f"should accept\n"
        f"Judge from the payoff table alone; do not assume anything about the "
        f"players that the table does not state."
    )


AGENT_SYSTEM = (
    "You are one of three participants in a structured deliberation over "
    "which joint plan, if any, two players should both accept under a "
    "binding agreement. You may use only the payoff table given; you may not "
    "invent payoffs or outside facts, and you may not assert a position the "
    "table does not support. Speak plainly and argue your assigned position "
    "as honestly and strongly as the table allows."
)


def synthesis_system(third_seat: str) -> str:
    third = THIRD_ROLE[third_seat].label.lower()
    return (
        "You are a neutral moderator of a three-party deliberation. Two "
        f"advocates, one for each player, and one {third} have each stated a "
        "position, rebutted the others, and restated a final position on "
        "which joint plan, if any, both players should accept. Your task is "
        "to write ONE synthesis position that a reader who had not seen the "
        "deliberation could act on, and to give the group's verdict."
    )


INTEGRATION_SYSTEM = (
    "You are the same neutral moderator. Having read all three "
    "participants' responses to your synthesis, construct ONE integrated "
    "proposal that addresses as many of their requests as can be "
    "reconciled without inventing facts, and give the group's verdict on "
    "it."
)


def request_block(item: GameItem) -> str:
    """The rendered item, quoted verbatim, recoverable byte-identically."""
    return f"{REQ_BEGIN}\n{item.post_text}\n{REQ_END}"


def extract_addressed(text: str, order: Sequence[str]) -> tuple[str, ...]:
    """Role ids (restricted to this arm's seats) the moderator's proposal
    claims to have addressed; () on none / absent / unparseable."""
    line = marker_line(text, ADDRESSED_MARKER)
    if not line:
        return ()
    low = line.lower()
    if low.startswith("none"):
        return ()
    return tuple(rid for rid in order if rid in low)


def _role_header(role: Role, item: GameItem) -> str:
    return (f"You are one of three participants in this deliberation, "
            f"speaking as {role_label(role, item)}.\n\n{role_brief(role, item)}")


def _others_block(texts: dict[str, str], me: str, roles: Sequence[Role],
                  item: GameItem, cap: int) -> str:
    parts = []
    for r in roles:
        if r.role_id == me:
            continue
        t = texts.get(r.role_id) or "[no statement]"
        parts.append(f"--- {role_label(r, item)} ---\n{_cap(t, cap)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompts (structure mirrors run_crowdgold_dilemma's seven stages; the PLAN
# instruction is the LAST line of every plan-bearing round)
# ---------------------------------------------------------------------------


def r0_user(item: GameItem, role: Role) -> str:
    return (
        "Three participants with different stakes will deliberate over the "
        "game below before a group verdict is reached.\n\n"
        f"{request_block(item)}\n\n{glossary(item)}\n\n{_role_header(role, item)}\n\n"
        "Write your opening statement. State the position you hold and "
        "why.\n\n"
        f"{GAME_VERDICT_INSTRUCTION}"
    )


def r1_user(item: GameItem, role: Role, roles: Sequence[Role], own_r0: str,
            r0_texts: dict[str, str], *, cap: int) -> str:
    return (
        f"{request_block(item)}\n\n{glossary(item)}\n\n{_role_header(role, item)}\n\n"
        f"--- YOUR OPENING STATEMENT ---\n{_cap(own_r0, cap)}\n--- END ---\n\n"
        "The other two participants opened as follows.\n\n"
        f"{_others_block(r0_texts, role.role_id, roles, item, cap)}\n\n"
        "Write your rebuttal. Challenge what you believe is wrong or "
        "incomplete, acknowledge what genuinely lands, and change your own "
        "position only if you have actually been persuaded by something in "
        "the table.\n\n"
        f"{GAME_VERDICT_INSTRUCTION}"
    )


def r2_user(item: GameItem, role: Role, roles: Sequence[Role], own_r0: str,
            own_r1: str, r1_texts: dict[str, str], *, cap: int) -> str:
    return (
        f"{request_block(item)}\n\n{glossary(item)}\n\n{_role_header(role, item)}\n\n"
        f"--- YOUR OPENING STATEMENT ---\n{_cap(own_r0, cap)}\n--- END ---\n\n"
        f"--- YOUR REBUTTAL ---\n{_cap(own_r1, cap)}\n--- END ---\n\n"
        "The other two participants rebutted as follows.\n\n"
        f"{_others_block(r1_texts, role.role_id, roles, item, cap)}\n\n"
        "State your final position. Say plainly what the group's finding "
        "should be and what your position rests on.\n\n"
        f"{GAME_VERDICT_INSTRUCTION}"
    )


def synthesis_user(item: GameItem, roles: Sequence[Role],
                   r2_texts: dict[str, str], *, cap: int) -> str:
    parts = []
    for r in roles:
        parts.append(
            f"--- {role_label(r, item)} ({r.role_id}) : final position ---\n"
            f"{_cap(r2_texts.get(r.role_id) or '[no statement]', cap)}"
        )
    return (
        f"{request_block(item)}\n\n{glossary(item)}\n\n"
        "Three participants have deliberated over this game. Their final "
        "positions follow.\n\n" + "\n\n".join(parts)
        + "\n\nWrite one synthesis position. Give it a short name on a line "
        "beginning 'SYNTHESIS:'. In two to four sentences state the "
        "position and say which concern of each participant it answers. "
        "Then give the group's verdict.\n\n"
        f"{GAME_VERDICT_INSTRUCTION}"
    )


def r3_label_user(item: GameItem, role: Role, own_r2: str, synthesis: str, *,
                  cap: int) -> str:
    return (
        f"{request_block(item)}\n\n{glossary(item)}\n\n{_role_header(role, item)}\n\n"
        f"--- YOUR FINAL POSITION ---\n{_cap(own_r2, cap)}\n--- END ---\n\n"
        "The moderator has read all three final positions and written a "
        "single synthesis.\n\n"
        f"--- MODERATOR SYNTHESIS ---\n{_cap(synthesis, cap)}\n--- END ---\n\n"
        "Respond to this synthesis. Accept it as it stands, accept it with "
        "one stated modification, or reject it.\n"
        f"  If you accept it with a modification, put that modification in "
        f"one sentence on its own line beginning '{MOD_MARKER}'.\n"
        f"  If you reject it, name the single concern you believe no "
        f"integration can absorb, in one sentence on its own line beginning "
        f"'{UNRESOLVABLE_MARKER}'.\n"
        "  If you accept it as it stands, write neither line.\n\n"
        f"{GAME_LABEL_INSTRUCTION}"
    )


def integration_user(item: GameItem, roles: Sequence[Role], synthesis: str,
                     labels: dict[str, str], objections: dict[str, str], *,
                     cap: int) -> str:
    parts = []
    for r in roles:
        obj = objections.get(r.role_id) or "[no modification stated]"
        parts.append(
            f"--- {role_label(r, item)} ({r.role_id}) ---\n"
            f"label: {labels.get(r.role_id) or NOVERDICT}\n"
            f"requests: {_cap(obj, cap)}"
        )
    ids = ", ".join(r.role_id for r in roles)
    return (
        f"{request_block(item)}\n\n{glossary(item)}\n\n"
        "You proposed this synthesis to three participants.\n\n"
        f"--- YOUR SYNTHESIS ---\n{_cap(synthesis, cap)}\n--- END ---\n\n"
        "They responded as follows.\n\n" + "\n\n".join(parts)
        + "\n\nConstruct one integrated proposal that addresses as many of "
        "these requests as can be reconciled. Give it a short name on a "
        "line beginning 'PROPOSAL:'. In two to four sentences state the "
        "position and say explicitly how it answers, or why it cannot "
        "answer, each request. Then, on its own line beginning "
        f"'{ADDRESSED_MARKER}', list the participant ids whose request the "
        f"proposal substantially addresses (from: {ids}), "
        "or the word none. Then give the group's verdict.\n\n"
        f"{GAME_VERDICT_INSTRUCTION}"
    )


def r4_vote_user(item: GameItem, role: Role, synthesis: str, own_label: str,
                 own_objection: str, proposal: str, *, cap: int) -> str:
    own = (f"--- YOUR RESPONSE TO THE SYNTHESIS ---\n"
           f"You labelled it: {own_label}\n"
           + (f"You required: {_cap(own_objection, cap)}\n" if own_objection
              else "You stated no modification.\n")
           + "--- END ---")
    return (
        f"{request_block(item)}\n\n{glossary(item)}\n\n{_role_header(role, item)}\n\n"
        f"--- MODERATOR SYNTHESIS ---\n{_cap(synthesis, cap)}\n--- END ---\n\n"
        f"{own}\n\n"
        "The moderator has now read all three responses and built a single "
        "integrated proposal.\n\n"
        f"--- INTEGRATED PROPOSAL ---\n{_cap(proposal, cap)}\n--- END ---\n\n"
        "This is the final round and your vote is binary. Accept the "
        "integrated proposal, or reject it. In two or three sentences say "
        "why, referring to what you required of the synthesis; if you "
        "reject, name the single concern it leaves unresolved.\n\n"
        f"{GAME_VOTE_INSTRUCTION}"
    )


ROUNDS = ("r0", "r1", "r2", "synthesis", "r3_label", "integration", "r4_vote")
AGENT_ROUNDS = ("r0", "r1", "r2", "r3_label", "r4_vote")
MODERATOR_ROUNDS = ("synthesis", "integration")
#: Every round is verdict-bearing here (see the module docstring on why r0-r2
#: are guarded too).
ROUND_INSTRUMENT = {
    "r0": GAME_INSTRUMENT, "r1": GAME_INSTRUMENT, "r2": GAME_INSTRUMENT,
    "synthesis": GAME_INSTRUMENT, "r3_label": GAME_LABEL_INSTRUMENT,
    "integration": GAME_INSTRUMENT, "r4_vote": GAME_VOTE_INSTRUMENT,
}
ROUND_INSTRUCTION = {
    "r0": GAME_VERDICT_INSTRUCTION, "r1": GAME_VERDICT_INSTRUCTION,
    "r2": GAME_VERDICT_INSTRUCTION, "synthesis": GAME_VERDICT_INSTRUCTION,
    "r3_label": GAME_LABEL_INSTRUCTION, "integration": GAME_VERDICT_INSTRUCTION,
    "r4_vote": GAME_VOTE_INSTRUCTION,
}


def all_prompts(item: GameItem, third_seat: str, *, cap: int = 0,
                stand_in: str = "word " * 40) -> dict[str, str]:
    """Every prompt of one cell with stand-in transcript text, keyed
    'round:role_id' (moderator rounds keyed 'round:moderator'), plus the three
    system prompts keyed 'system:*'. Used by the selftest's leak sweep and by
    the cost model."""
    roles = roles_for(third_seat)
    prior = {r.role_id: stand_in for r in roles}
    labels = {r.role_id: "ACCEPT_WITH_MODIFICATION" for r in roles}
    objs = {r.role_id: stand_in for r in roles}
    out: dict[str, str] = {
        "system:agent": AGENT_SYSTEM,
        "system:synthesis": synthesis_system(third_seat),
        "system:integration": INTEGRATION_SYSTEM,
        "synthesis:moderator": synthesis_user(item, roles, prior, cap=cap),
        "integration:moderator": integration_user(item, roles, stand_in, labels,
                                                  objs, cap=cap),
    }
    for r in roles:
        out[f"r0:{r.role_id}"] = r0_user(item, r)
        out[f"r1:{r.role_id}"] = r1_user(item, r, roles, stand_in, prior, cap=cap)
        out[f"r2:{r.role_id}"] = r2_user(item, r, roles, stand_in, stand_in,
                                         prior, cap=cap)
        out[f"r3_label:{r.role_id}"] = r3_label_user(item, r, stand_in, stand_in,
                                                     cap=cap)
        out[f"r4_vote:{r.role_id}"] = r4_vote_user(item, r, stand_in,
                                                   "ACCEPT_WITH_MODIFICATION",
                                                   stand_in, stand_in, cap=cap)
    return out


def steer_hits(text: str) -> list[str]:
    low = (text or "").lower()
    return [w for w in STEER_WORDS if w in low]


# ---------------------------------------------------------------------------
# Parsing / guard
# ---------------------------------------------------------------------------


_AWM_SPACING_RE = re.compile(r"\bACCEPT[ \-_]WITH[ \-_]MODIFICATION\b", re.I)


def parse_round(round_name: str, text: str) -> str:
    """The round's token, or NOVERDICT. PLAN rounds go through
    parse_plan_verdict (which normalises 'PLAN 12'-style spacing before the
    shared extractor). Label / vote rounds collapse the spaced
    'ACCEPT WITH MODIFICATION' to the exact token first, then accept only
    that round's COMMITTAL tokens: a stray UNRESOLVED is NOVERDICT here, as
    it is on the PLAN rounds (module docstring, VERDICTS AND GUARD)."""
    inst = ROUND_INSTRUMENT[round_name]
    if inst == GAME_INSTRUMENT:
        return parse_plan_verdict(text or "")
    norm = _AWM_SPACING_RE.sub("ACCEPT_WITH_MODIFICATION", text or "")
    v = extract_verdict(norm, inst)
    return v if v in committal_tokens(inst) else NOVERDICT


def call_guard_row(rec: dict) -> dict:
    """A per-call row shaped for the IMPORTED truncation_report. Every round
    is consumed as a value here, so every row carries the parsed token and a
    parse failure counts as NOVERDICT."""
    text = rec.get("output") or ""
    parsed = parse_round(rec["round"], text) if text.strip() else NOVERDICT
    return {
        "model": rec["model"],
        "scaffold": f"{rec['scaffold']}/{rec['round']}:{rec['role_id']}",
        "arm": rec["arm"],
        "verdict": parsed,
        "output_len": len(text),
        "finish_reason": rec.get("finish_reason", ""),
        "truncated": int(rec.get("finish_reason", "") in
                         ("max_tokens", "length", "MAX_TOKENS")),
    }


def guard_rows(calls: Sequence[dict]) -> list[dict]:
    return [call_guard_row(c) for c in calls if c.get("round") in ROUND_INSTRUMENT]


def _is_nash_int(item: GameItem, verdict: str):
    v = item.view.is_nash(verdict)
    return "" if v is None else int(v)


def _perm_str(perm: Sequence[int]) -> str:
    return "".join(str(p) for p in perm)


# ---------------------------------------------------------------------------
# One deliberation
# ---------------------------------------------------------------------------


def run_deliberation(model: str, item: GameItem, *, third_seat: str,
                     moderator_model: Optional[str] = None,
                     max_tokens_agent: int = 2560,
                     max_tokens_moderator: int = 1024,
                     max_tokens_label: int = 3072,
                     max_tokens_vote: int = 3072,
                     transcript_cap: int = 0) -> tuple[dict, list[dict], list[dict]]:
    """One full deliberation, the seven stages, 17 calls."""
    mod_model = moderator_model or model
    roles = roles_for(third_seat)
    order = tuple(r.role_id for r in roles)
    arm = ARM_OF[third_seat]
    idx = item.sample_idx
    mod_rid = moderator_role_id(mod_model, third_seat)
    calls: list[dict] = []
    #: (round, role_id) -> the prompt as sent carried a game-type stem
    type_hit: dict[tuple[str, str], bool] = {}

    r0: dict[str, str] = {}
    r0_v: dict[str, str] = {}
    for r in roles:
        user = r0_user(item, r)
        type_hit[("r0", r.role_id)] = type_named(user)
        rec = do_call(
            model=model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
            round_name="r0", role_id=r.role_id, system=AGENT_SYSTEM,
            user=user, max_tokens=max_tokens_agent,
            parents=(item.post_text,), cap=transcript_cap,
        )
        calls.append(rec)
        r0[r.role_id] = rec.get("output") or ""
        r0_v[r.role_id] = parse_round("r0", r0[r.role_id])

    r1: dict[str, str] = {}
    r1_v: dict[str, str] = {}
    r0_parents = tuple(r0[rid] for rid in order)
    for r in roles:
        user = r1_user(item, r, roles, r0[r.role_id], r0, cap=transcript_cap)
        type_hit[("r1", r.role_id)] = type_named(user)
        rec = do_call(
            model=model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
            round_name="r1", role_id=r.role_id, system=AGENT_SYSTEM,
            user=user, max_tokens=max_tokens_agent, parents=r0_parents,
            cap=transcript_cap,
        )
        calls.append(rec)
        r1[r.role_id] = rec.get("output") or ""
        r1_v[r.role_id] = parse_round("r1", r1[r.role_id])

    r2: dict[str, str] = {}
    r2_v: dict[str, str] = {}
    r1_parents = r0_parents + tuple(r1[rid] for rid in order)
    for r in roles:
        user = r2_user(item, r, roles, r0[r.role_id], r1[r.role_id], r1,
                       cap=transcript_cap)
        type_hit[("r2", r.role_id)] = type_named(user)
        rec = do_call(
            model=model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
            round_name="r2", role_id=r.role_id, system=AGENT_SYSTEM,
            user=user, max_tokens=max_tokens_agent, parents=r1_parents,
            cap=transcript_cap,
        )
        calls.append(rec)
        r2[r.role_id] = rec.get("output") or ""
        r2_v[r.role_id] = parse_round("r2", r2[r.role_id])

    r2_parents = r1_parents + tuple(r2[rid] for rid in order)
    user = synthesis_user(item, roles, r2, cap=transcript_cap)
    type_hit[("synthesis", mod_rid)] = type_named(user)
    syn_rec = do_call(
        model=mod_model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
        round_name="synthesis", role_id=mod_rid,
        system=synthesis_system(third_seat), user=user,
        max_tokens=max_tokens_moderator, parents=r2_parents, cap=transcript_cap,
    )
    calls.append(syn_rec)
    synthesis = syn_rec.get("output") or ""
    synthesis_verdict = parse_round("synthesis", synthesis)

    labels: dict[str, str] = {}
    objections: dict[str, str] = {}
    objection_kind: dict[str, str] = {}
    syn_parents = r2_parents + (synthesis,)
    for r in roles:
        user = r3_label_user(item, r, r2[r.role_id], synthesis, cap=transcript_cap)
        type_hit[("r3_label", r.role_id)] = type_named(user)
        rec = do_call(
            model=model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
            round_name="r3_label", role_id=r.role_id, system=AGENT_SYSTEM,
            user=user, max_tokens=max_tokens_label, parents=syn_parents,
            cap=transcript_cap,
        )
        calls.append(rec)
        lab = parse_round("r3_label", rec.get("output") or "")
        kind, text = extract_objection(rec.get("output") or "")
        labels[r.role_id] = lab
        objections[r.role_id] = text
        objection_kind[r.role_id] = kind

    lbl_parents = syn_parents + tuple(
        f"{labels[rid]}|{objections[rid]}" for rid in order)
    user = integration_user(item, roles, synthesis, labels, objections,
                            cap=transcript_cap)
    type_hit[("integration", mod_rid)] = type_named(user)
    int_rec = do_call(
        model=mod_model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
        round_name="integration", role_id=mod_rid, system=INTEGRATION_SYSTEM,
        user=user, max_tokens=max_tokens_moderator, parents=lbl_parents,
        cap=transcript_cap,
    )
    calls.append(int_rec)
    proposal = int_rec.get("output") or ""
    group_verdict = parse_round("integration", proposal)
    addressed = extract_addressed(proposal, order)

    vote_rows: list[dict] = []
    votes: dict[str, str] = {}
    for r in roles:
        lab = labels[r.role_id]
        obj = objections[r.role_id]
        user = r4_vote_user(item, r, synthesis, lab, obj, proposal,
                            cap=transcript_cap)
        type_hit[("r4_vote", r.role_id)] = type_named(user)
        rec = do_call(
            model=model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
            round_name="r4_vote", role_id=r.role_id, system=AGENT_SYSTEM,
            user=user, max_tokens=max_tokens_vote,
            parents=(synthesis, lab, obj, proposal), cap=transcript_cap,
        )
        calls.append(rec)
        seat_hit_rounds = [rd for rd in AGENT_ROUNDS if type_hit.get((rd, r.role_id))]
        vote = parse_round("r4_vote", rec.get("output") or "")
        votes[r.role_id] = vote
        und = stake_undermined(r.role_id, group_verdict, item)
        objected = lab in ("ACCEPT_WITH_MODIFICATION", "REJECT")
        got = player_payoff(item, group_verdict, r.stake)
        ref = best_nash_payoff(item, r.stake)
        mm = maximin_payoff(item, r.stake)
        vote_rows.append({
            "model": model, "scaffold": SCAFFOLD_NAME, "arm": arm,
            "third_seat": third_seat,
            "item_id": item.item_id, "sample_idx": idx,
            "gold_type": item.gold_type, "region": item.region,
            "gold_verdict": item.gold_verdict,
            "role_id": r.role_id, "role_stake": r.stake,
            "group_verdict": group_verdict,
            "synthesis_verdict": synthesis_verdict,
            "r0_verdict": r0_v[r.role_id], "r1_verdict": r1_v[r.role_id],
            "r2_verdict": r2_v[r.role_id],
            "r0_is_nash": _is_nash_int(item, r0_v[r.role_id]),
            "r2_is_nash": _is_nash_int(item, r2_v[r.role_id]),
            "r0_correct": int(score_plan(r0_v[r.role_id], item.gold_verdict)["correct"]),
            "r2_correct": int(score_plan(r2_v[r.role_id], item.gold_verdict)["correct"]),
            "r3_label": lab,
            "r3_label_parsed": int(lab in ("ACCEPT", "ACCEPT_WITH_MODIFICATION",
                                           "REJECT")),
            "r3_objection_kind": objection_kind[r.role_id],
            "r3_objection_len": len(obj),
            "objected_r3": int(objected),
            "objection_unstated": int(objected and not obj),
            "modification_addressed": int(r.role_id in addressed),
            "vote": vote,
            "reject": int(vote == "REJECT"),
            "vote_parsed": int(vote in ("ACCEPT", "REJECT")),
            "revised_r4": int(objected and vote == "ACCEPT"),
            "stake_undermined": "" if und is None else int(und),
            "group_plan_payoff": "" if got is None else got,
            "best_nash_payoff": "" if ref is None else ref,
            "maximin_payoff": "" if mm is None else mm,
            "type_named_in_prompt": int(bool(seat_hit_rounds)),
            "type_named_rounds": "|".join(seat_hit_rounds),
            "finish_reason": rec.get("finish_reason", ""),
            "truncated": int(rec.get("finish_reason", "") in
                             ("max_tokens", "length", "MAX_TOKENS")),
            "output_len": len(rec.get("output") or ""),
        })

    n_type_named = sum(1 for h in type_hit.values() if h)
    type_named_first = next((rd for rd in ROUNDS
                             if any(h for (rr, _), h in type_hit.items() if rr == rd)),
                            "")
    n_reject = sum(1 for v in votes.values() if v == "REJECT")
    n_accept = sum(1 for v in votes.values() if v == "ACCEPT")
    n_awm = sum(1 for v in labels.values() if v == "ACCEPT_WITH_MODIFICATION")
    n_lab_accept = sum(1 for v in labels.values() if v == "ACCEPT")
    n_lab_reject = sum(1 for v in labels.values() if v == "REJECT")
    objectors = [rid for rid in order
                 if labels[rid] in ("ACCEPT_WITH_MODIFICATION", "REJECT")]
    rejecting = [rid for rid in order if votes[rid] == "REJECT"]
    third_id = order[2]
    gp = item.view.payoffs_for(group_verdict)
    gcell = item.view.cell_for(group_verdict)

    row = {
        "model": model, "scaffold": SCAFFOLD_NAME, "arm": arm,
        "third_seat": third_seat, "third_role_id": third_id,
        "item_id": item.item_id, "game_id": item.game_id, "sample_idx": idx,
        "gold_type": item.gold_type, "region": item.region,
        "gold_verdict": item.gold_verdict,
        "nash_verdicts": "|".join(sorted(item.view.nash_verdicts)),
        "n_nash": len(item.view.nash_verdicts),
        "row_perm": _perm_str(item.view.row_perm),
        "col_perm": _perm_str(item.view.col_perm),
        "row_name": item.view.row_name, "col_name": item.view.col_name,
        "row_best_verdict": item.view.row_best_verdict,
        "col_best_verdict": item.view.col_best_verdict,
        "verdict": group_verdict,
        "correct": int(score_plan(group_verdict, item.gold_verdict)["correct"]),
        "noncommittal": int(is_noncommittal(group_verdict)),
        "group_is_nash": _is_nash_int(item, group_verdict),
        "group_cell": "" if gcell is None else f"{gcell[0]},{gcell[1]}",
        "group_payoff_row": "" if gp is None else gp[0],
        "group_payoff_col": "" if gp is None else gp[1],
        "output_len": len(proposal),
        "post_sha256": sha256_text(item.post_text),
        "finish_reason": int_rec.get("finish_reason", ""),
        "truncated": int(int_rec.get("finish_reason", "") in
                         ("max_tokens", "length", "MAX_TOKENS")),
        "protocol": PROTOCOL, "n_agents": N_AGENTS, "moderator_model": mod_model,
        "synthesis_verdict": synthesis_verdict, "synthesis_len": len(synthesis),
        "synthesis_correct": int(score_plan(synthesis_verdict,
                                            item.gold_verdict)["correct"]),
        "verdict_revised": int(synthesis_verdict != group_verdict),
        "r0_verdict_row_advocate": r0_v["player_r_advocate"],
        "r0_verdict_col_advocate": r0_v["player_c_advocate"],
        "r0_verdict_third": r0_v[third_id],
        "r1_verdict_row_advocate": r1_v["player_r_advocate"],
        "r1_verdict_col_advocate": r1_v["player_c_advocate"],
        "r1_verdict_third": r1_v[third_id],
        "r2_verdict_row_advocate": r2_v["player_r_advocate"],
        "r2_verdict_col_advocate": r2_v["player_c_advocate"],
        "r2_verdict_third": r2_v[third_id],
        "r3_labels": "|".join(labels[rid] for rid in order),
        "n_r3_accept": n_lab_accept, "n_r3_accept_with_mod": n_awm,
        "n_r3_reject": n_lab_reject,
        "n_r3_unparsed": N_AGENTS - n_lab_accept - n_awm - n_lab_reject,
        "n_objectors": len(objectors),
        "n_objections_stated": sum(1 for rid in objectors if objections[rid]),
        "n_modifications_addressed": sum(1 for rid in objectors
                                         if rid in addressed),
        "addressed_roles": "|".join(addressed),
        "addressed_parsed": int(bool(marker_line(proposal, ADDRESSED_MARKER))),
        "n_revised": sum(1 for rid in objectors if votes[rid] == "ACCEPT"),
        "n_accept": n_accept, "n_reject": n_reject,
        "n_vote_unparsed": N_AGENTS - n_accept - n_reject,
        "unanimous_accept": int(n_accept == N_AGENTS),
        "rejecting_roles": "|".join(rejecting),
        "vote_row_advocate": votes["player_r_advocate"],
        "vote_col_advocate": votes["player_c_advocate"],
        "vote_third": votes[third_id],
        "n_calls": len(calls),
        "prompt_tokens": sum(int(c.get("prompt_tokens", 0) or 0) for c in calls),
        "completion_tokens": sum(int(c.get("completion_tokens", 0) or 0)
                                 for c in calls),
        "empty_calls": sum(1 for c in calls
                           if not (c.get("output") or "").strip()),
        "transcript_truncated_calls": sum(
            int(c.get("transcript_truncated", 0)) for c in calls),
        "n_calls_type_named": n_type_named,
        "type_named_first_round": type_named_first,
    }
    return row, vote_rows, calls


ROW_FIELDS = (
    "model", "scaffold", "arm", "third_seat", "third_role_id", "item_id",
    "game_id", "sample_idx", "gold_type", "region", "gold_verdict",
    "nash_verdicts", "n_nash", "row_perm", "col_perm", "row_name", "col_name",
    "row_best_verdict", "col_best_verdict", "verdict", "correct",
    "noncommittal", "group_is_nash", "group_cell", "group_payoff_row",
    "group_payoff_col", "output_len", "post_sha256", "finish_reason",
    "truncated", "protocol", "n_agents", "moderator_model",
    "synthesis_verdict", "synthesis_len", "synthesis_correct",
    "verdict_revised", "r0_verdict_row_advocate", "r0_verdict_col_advocate",
    "r0_verdict_third", "r1_verdict_row_advocate", "r1_verdict_col_advocate",
    "r1_verdict_third", "r2_verdict_row_advocate", "r2_verdict_col_advocate",
    "r2_verdict_third", "r3_labels", "n_r3_accept", "n_r3_accept_with_mod",
    "n_r3_reject", "n_r3_unparsed", "n_objectors", "n_objections_stated",
    "n_modifications_addressed", "addressed_roles", "addressed_parsed",
    "n_revised", "n_accept", "n_reject", "n_vote_unparsed", "unanimous_accept",
    "rejecting_roles", "vote_row_advocate", "vote_col_advocate", "vote_third",
    "n_calls", "prompt_tokens", "completion_tokens", "empty_calls",
    "transcript_truncated_calls", "n_calls_type_named",
    "type_named_first_round",
)

VOTE_FIELDS = (
    "model", "scaffold", "arm", "third_seat", "item_id", "sample_idx",
    "gold_type", "region", "gold_verdict", "role_id", "role_stake",
    "group_verdict", "synthesis_verdict", "r0_verdict", "r1_verdict",
    "r2_verdict", "r0_is_nash", "r2_is_nash", "r0_correct", "r2_correct",
    "r3_label", "r3_label_parsed", "r3_objection_kind", "r3_objection_len",
    "objected_r3", "objection_unstated", "modification_addressed", "vote",
    "reject", "vote_parsed", "revised_r4", "stake_undermined",
    "group_plan_payoff", "best_nash_payoff", "maximin_payoff",
    "type_named_in_prompt", "type_named_rounds",
    "finish_reason", "truncated", "output_len",
)


# ---------------------------------------------------------------------------
# Cost model. Completion lengths are ASSUMPTIONS (no run of this instrument
# exists yet); the measured spend read back from cache is authoritative.
# ---------------------------------------------------------------------------

ASSUMED_COMPLETION = {
    "r0": 500, "r1": 500, "r2": 500,
    "synthesis": 450, "r3_label": 320, "integration": 500, "r4_vote": 140,
}
N_CALLS_PER_ROUND = {"r0": N_AGENTS, "r1": N_AGENTS, "r2": N_AGENTS,
                     "synthesis": 1, "r3_label": N_AGENTS,
                     "integration": 1, "r4_vote": N_AGENTS}


def cost_model(model: str, games: Sequence[Game], samples: int, *,
               third_seat: str, transcript_cap: int = 0,
               moderator_model: Optional[str] = None,
               use_bpe: bool = True) -> dict:
    if not games:
        raise ValueError("cost model needs at least one game")
    if samples < 1:
        raise ValueError("samples must be >= 1")
    bpe = _optional_bpe_counter() if use_bpe else None
    probe = GameItem(render_item(
        sorted(games, key=lambda g: len(g.text))[len(games) // 2], 0))
    mod_m = moderator_model or model
    pin, pout = PRICES.get(model, (0.0, 0.0))
    min_, mout_ = PRICES.get(mod_m, (0.0, 0.0))
    cells = len(games) * samples
    roles = roles_for(third_seat)
    r = roles[0]
    stand_in = {rd: "word " * ASSUMED_COMPLETION[rd] for rd in ROUNDS}
    labels0 = {x.role_id: "ACCEPT_WITH_MODIFICATION" for x in roles}
    objs0 = {x.role_id: "word " * 40 for x in roles}
    prior0 = {x.role_id: stand_in["r0"] for x in roles}
    prior1 = {x.role_id: stand_in["r1"] for x in roles}
    prompts = {
        "r0": r0_user(probe, r),
        "r1": r1_user(probe, r, roles, stand_in["r0"], prior0, cap=transcript_cap),
        "r2": r2_user(probe, r, roles, stand_in["r0"], stand_in["r1"], prior1,
                      cap=transcript_cap),
        "synthesis": synthesis_user(probe, roles, prior1, cap=transcript_cap),
        "r3_label": r3_label_user(probe, r, stand_in["r2"], stand_in["synthesis"],
                                  cap=transcript_cap),
        "integration": integration_user(probe, roles, stand_in["synthesis"],
                                        labels0, objs0, cap=transcript_cap),
        "r4_vote": r4_vote_user(probe, r, stand_in["synthesis"],
                                "ACCEPT_WITH_MODIFICATION",
                                objs0[r.role_id], stand_in["integration"],
                                cap=transcript_cap),
    }
    rounds: dict = {}
    tot_in = tot_out = 0
    tot_usd = 0.0
    for rd in ROUNDS:
        ptok = (count_tokens(prompts[rd], tokenizer=bpe) if bpe
                else count_tokens(prompts[rd]))
        n_calls = N_CALLS_PER_ROUND[rd] * cells
        comp = ASSUMED_COMPLETION[rd]
        is_mod = rd in MODERATOR_ROUNDS
        cin, cout = (min_, mout_) if is_mod else (pin, pout)
        in_tok = ptok * n_calls
        out_tok = comp * n_calls
        usd = in_tok / 1e6 * cin + out_tok / 1e6 * cout
        rounds[rd] = {"n_calls": n_calls, "prompt_tokens_per_call": ptok,
                      "assumed_completion": comp, "usd": round(usd, 4)}
        tot_in += in_tok
        tot_out += out_tok
        tot_usd += usd
    return {
        "model": model, "moderator_model": mod_m, "third_seat": third_seat,
        "arm": ARM_OF[third_seat], "n_games": len(games), "samples": samples,
        "cells": cells, "calls_per_cell": CALLS_PER_CELL,
        "total_calls": cells * CALLS_PER_CELL,
        "rounds": rounds, "total_prompt_tokens": tot_in,
        "total_completion_tokens_assumed": tot_out,
        "total_usd": round(tot_usd, 2),
        "priced": model in PRICES and mod_m in PRICES,
        "tokenizer": "tiktoken cl100k_base BPE" if bpe else "regex approximation",
        "assumption": ("every completion length is an UNMEASURED assumption -- "
                       "there is no prior run of this instrument to measure "
                       "from. Treat this as an order-of-magnitude estimate; "
                       "the run's OWN measured spend (read back from the "
                       "cache) is authoritative, not this number."),
    }


def print_cost_model(cm: dict) -> None:
    print("\n" + "=" * 72)
    print(f"COST MODEL (dry run) -- {cm['model']} moderated by "
          f"{cm['moderator_model']}  third seat {cm['third_seat']} "
          f"(arm {cm['arm']})")
    print("=" * 72)
    print(f"  {cm['n_games']} games x {cm['samples']} samples = {cm['cells']} "
          f"cells x {cm['calls_per_cell']} calls = {cm['total_calls']} "
          f"generation calls")
    print(f"  tokenizer: {cm['tokenizer']}")
    for rd, r in cm["rounds"].items():
        print(f"    {rd:<12} {r['n_calls']:>5} calls  "
              f"{r['prompt_tokens_per_call']:>5} prompt tok/call  "
              f"assumed completion {r['assumed_completion']:>4}  "
              f"${r['usd']:.3f}")
    print(f"  TOTAL: ${cm['total_usd']:.2f} "
          f"({cm['total_prompt_tokens']} prompt + "
          f"{cm['total_completion_tokens_assumed']} assumed completion tok)")
    if not cm["priced"]:
        print("  *** WARNING: a model here has no PRICES entry; the $ figure "
              "above is NOT a cost estimate. Add the model to "
              "run_crowdgold_deliberation.PRICES before spending. ***")
    print(f"\n  ASSUMPTION: {cm['assumption']}")


# ---------------------------------------------------------------------------
# Cache namespaces (keyed by name, not content)
# ---------------------------------------------------------------------------


def cache_namespace_counts(out_dir: Path = OUT_DIR) -> dict[str, int]:
    """One pass over the cache directory, counting files in every namespace
    this runner can write to (both third-seat variants)."""
    keys = ([f"scaffold:{SCAFFOLD_NAME}"] + [f"arm:{a}" for a in ARM_OF.values()]
            + [f"role:{r.role_id}" for r in ALL_ROLES]
            + [f"moderator:{MODERATOR_PREFIX}", f"tag:{TAG_PREFIX}"])
    counts = {k: 0 for k in keys}
    if not out_dir.is_dir():
        return counts
    role_suffixes = {f"role:{r.role_id}": f"_{r.role_id}.json" for r in ALL_ROLES}
    # An arm is matched with a boundary (the arm is followed by the three-
    # digit game id), and the tag as a prefix followed by "_": the
    # non-binding sibling's arms (third_neutral_nb / third_planner_nb) and
    # tag (cgnb_game) must not be counted as this runner's. Selftest-only
    # code; the protocol is untouched.
    arm_res = {a: re.compile(rf"_{re.escape(a)}_\d{{3}}_") for a in ARM_OF.values()}
    for p in out_dir.iterdir():
        n = p.name
        if f"_{SCAFFOLD_NAME}_" in n:
            counts[f"scaffold:{SCAFFOLD_NAME}"] += 1
        for a, arm_re in arm_res.items():
            if arm_re.search(n):
                counts[f"arm:{a}"] += 1
        for k, suf in role_suffixes.items():
            if n.endswith(suf):
                counts[k] += 1
        if f"_{MODERATOR_PREFIX}-" in n:
            counts[f"moderator:{MODERATOR_PREFIX}"] += 1
        if n.startswith(f"{TAG_PREFIX}_"):
            counts[f"tag:{TAG_PREFIX}"] += 1
    return counts


def cache_is_clean(model: str, third_seat: str, *,
                   moderator_model: Optional[str] = None,
                   out_dir: Path = OUT_DIR) -> tuple[bool, dict[str, int]]:
    """No cache file may already exist for THIS run's footprint: files whose
    name starts with cgd_<model>_<scaffold>_ and carry this arm, for the
    agent model and for the moderator model."""
    arm = ARM_OF[third_seat]
    models = {model, moderator_model or model}
    counts = {f"cgd_{_safe(m)}_{SCAFFOLD_NAME}_*_{arm}_*": 0 for m in sorted(models)}
    if not out_dir.is_dir():
        return True, counts
    prefixes = {m: f"cgd_{_safe(m)}_{SCAFFOLD_NAME}_" for m in models}
    for p in out_dir.iterdir():
        n = p.name
        for m, pre in prefixes.items():
            if n.startswith(pre) and f"_{arm}_" in n:
                counts[f"cgd_{_safe(m)}_{SCAFFOLD_NAME}_*_{arm}_*"] += 1
    return all(v == 0 for v in counts.values()), counts


# ---------------------------------------------------------------------------
# Readout: per-gold_type accuracy (the extent chart) and Nash-pull by seat,
# with item-clustered bootstrap CIs. The analyzer is the authority; this is
# the first look printed at the end of a run and by --readout.
# ---------------------------------------------------------------------------


def _clustered_mean(recs: Sequence[dict], key: Callable[[dict], Optional[int]],
                    *, draws: int, seed: int) -> dict:
    by_item: dict[str, list] = defaultdict(list)
    for r in recs:
        v = key(r)
        if v is None:
            continue
        by_item[str(r["item_id"])].append(int(v))
    n = sum(len(v) for v in by_item.values())
    if not n:
        return {"n": 0, "point": None, "lo": None, "hi": None}

    def stat(xs: list) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    p, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed)
    return {"n": n, "point": p, "lo": lo, "hi": hi}


def _int_or_none(v) -> Optional[int]:
    if v is None or v == "":
        return None
    return int(v)


def readout(rows: Sequence[dict], votes: Sequence[dict], *, draws: int = 1000,
            seed: int = 7) -> dict:
    correct = lambda r: _int_or_none(r.get("correct"))  # noqa: E731
    acc_by_type = {gt: _clustered_mean([r for r in rows if r.get("gold_type") == gt],
                                       correct, draws=draws, seed=seed)
                   for gt in GOLD_TYPES}
    acc_by_region = {rg: _clustered_mean([r for r in rows if r.get("region") == rg],
                                         correct, draws=draws, seed=seed)
                     for rg in REGIONS}
    overall = _clustered_mean(rows, correct, draws=draws, seed=seed)
    group_nash = _clustered_mean(rows, lambda r: _int_or_none(r.get("group_is_nash")),
                                 draws=draws, seed=seed)
    type_named_cells = _clustered_mean(
        rows, lambda r: (None if _int_or_none(r.get("n_calls_type_named")) is None
                         else int(_int_or_none(r.get("n_calls_type_named")) > 0)),
        draws=draws, seed=seed)
    seats = sorted({v["role_id"] for v in votes})
    pull: dict[str, dict] = {}
    for s in seats:
        sv = [v for v in votes if v["role_id"] == s]
        pull[s] = {
            "r0_all": _clustered_mean(sv, lambda v: _int_or_none(v.get("r0_is_nash")),
                                      draws=draws, seed=seed),
            "r0_correct": _clustered_mean(sv, lambda v: _int_or_none(v.get("r0_correct")),
                                          draws=draws, seed=seed),
        }
        for rg in REGIONS:
            pull[s][f"r0_{rg}"] = _clustered_mean(
                [v for v in sv if v.get("region") == rg],
                lambda v: _int_or_none(v.get("r0_is_nash")), draws=draws, seed=seed)
    return {
        "n_rows": len(rows), "n_items": len({r["item_id"] for r in rows}),
        "n_votes": len(votes),
        "accuracy": overall, "accuracy_by_gold_type": acc_by_type,
        "accuracy_by_region": acc_by_region, "group_is_nash": group_nash,
        "type_named_cells": type_named_cells,
        "nash_pull": pull,
        "verdict_counts": dict(Counter(r.get("verdict") for r in rows)),
    }


def _fmt(c: dict) -> str:
    if not c["n"]:
        return f"{'n/a':>7}   {'':>15}  n=0"
    lo = "n/a" if c["lo"] is None else f"{c['lo']:.3f}"
    hi = "n/a" if c["hi"] is None else f"{c['hi']:.3f}"
    return f"{c['point']:>7.3f}   [{lo:>5}, {hi:>5}]  n={c['n']}"


def print_readout(ro: dict, *, label: str = "") -> None:
    print("\n" + "=" * 72)
    print(f"READOUT {label} -- rows={ro['n_rows']} items={ro['n_items']} "
          f"votes={ro['n_votes']}  (item-clustered 95% CI)")
    print("=" * 72)
    print(f"  group verdict accuracy, overall        {_fmt(ro['accuracy'])}")
    print("  by region:")
    for rg, c in ro["accuracy_by_region"].items():
        print(f"    {rg:<34} {_fmt(c)}")
    print("  by gold_type (the extent chart):")
    for gt, c in ro["accuracy_by_gold_type"].items():
        print(f"    {gt:<34} {_fmt(c)}")
    print(f"  group verdict is a pure Nash cell        {_fmt(ro['group_is_nash'])}")
    print("  Nash-pull: P(seat r0 verdict is a pure Nash cell), by seat:")
    for s, p in ro["nash_pull"].items():
        print(f"    {s:<24} {'all':<13} {_fmt(p['r0_all'])}")
        for rg in REGIONS:
            print(f"    {'':<24} {rg:<13} {_fmt(p[f'r0_{rg}'])}")
        print(f"    {'':<24} {'r0 correct':<13} {_fmt(p['r0_correct'])}")
    print(f"  verdict counts: {ro['verdict_counts']}")
    print(f"  cells with a game-type stem quoted into any prompt "
          f"(transcript leak; condition on it)  {_fmt(ro['type_named_cells'])}")


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Selftest (offline; generate_any stubbed; cache redirected to a tempdir)
# ---------------------------------------------------------------------------


class _Stub:
    def __init__(self, text, finish_reason="stop", ptok=50, ctok=50):
        self.text = text
        self.finish_reason = finish_reason
        self.prompt_tokens = ptok
        self.completion_tokens = ctok


_PLAYERS_RE = re.compile(r"Two players, (\w+) and (\w+), are playing")


def _stub_generate(model, system, user, *, sample_idx=0, max_tokens=0, **kw):
    """Deterministic offline stand-in. Planted behaviour: the row advocate
    objects at r3 and rejects at r4, everyone else accepts; the integration
    writes its verdict as 'PLAN 12' (a spacing variant) to exercise the
    normalising parser on the live path."""
    m = _PLAYERS_RE.search(user)
    R, C = (m.group(1), m.group(2)) if m else ("", "")
    is_row = bool(R) and f"speaking as {R}'s Advocate" in user
    is_col = bool(C) and f"speaking as {C}'s Advocate" in user
    if "This is the final round and your vote is binary" in user:
        return _Stub("I reject; my player is left short.\n\nVERDICT: REJECT"
                     if is_row else "I accept.\n\nVERDICT: ACCEPT")
    if "Respond to this synthesis" in user:
        if is_row:
            return _Stub(f"Needs one change.\n{MOD_MARKER} weigh {R}'s payoff "
                         f"properly.\n\nVERDICT: ACCEPT_WITH_MODIFICATION")
        return _Stub("I accept as written.\n\nVERDICT: ACCEPT")
    if "Write one synthesis position" in user:
        return _Stub("SYNTHESIS: the shared plan.\n\nVERDICT: PLAN_12")
    if "Construct one integrated proposal" in user:
        return _Stub(f"PROPOSAL: refined plan.\n\n{ADDRESSED_MARKER} "
                     f"player_r_advocate\n\nVERDICT: PLAN 12")
    v = "PLAN_11" if is_row else ("PLAN_22" if is_col else "PLAN_12")
    return _Stub(f"My position rests on the table.\n\nVERDICT: {v}")


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # ---- data ---------------------------------------------------------------
    try:
        games = load_games()
    except Exception as e:  # noqa: BLE001
        check(f"load_games() ({type(e).__name__}: {e})", False)
        print(f"\n{len(fails)} FAILED")
        return 1
    check("144 games load with pins verified", len(games) == 144)
    check("select_games: all/value/neutral/cannot_assist = 144/16/97/31",
          (len(select_games(games, "all")), len(select_games(games, REGION_VALUE)),
           len(select_games(games, REGION_NEUTRAL)),
           len(select_games(games, REGION_CANNOT_ASSIST))) == (144, 16, 97, 31))
    check("select_games intersects explicit ids",
          [g.game_id for g in select_games(games, "all", ["112", "111", "zzz"])]
          == ["111", "112"])
    items2 = build_items(select_games(games, REGION_VALUE), 2)
    check("build_items: 16 value games x 2 samples = 32 cells, each its own "
          "(game_id, sample_idx)",
          len(items2) == 32
          and len({(i.item_id, i.sample_idx) for i in items2}) == 32)
    check("GameItem duck-types item_id / gold_verdict / post_text for do_call",
          all(hasattr(items2[0], a) for a in ("item_id", "gold_verdict", "post_text"))
          and items2[0].post_text == items2[0].view.text)

    # ---- seats and namespaces ------------------------------------------------
    check("four role ids are unique and the third seat differs by variant",
          len({r.role_id for r in ALL_ROLES}) == 4
          and role_order(THIRD_NEUTRAL)[2] != role_order(THIRD_PLANNER)[2]
          and role_order(THIRD_NEUTRAL)[:2] == role_order(THIRD_PLANNER)[:2]
          == STAKE_SEATS)
    check("advocates' stakes are their own player's payoff; third seats have none",
          PLAYER_R_ADVOCATE.stake == "row" and PLAYER_C_ADVOCATE.stake == "col"
          and NEUTRAL_READER.stake == "none" and PLAN_MEDIATOR.stake == "none")
    check("arms are distinct per third-seat variant",
          len(set(ARM_OF.values())) == 2)
    mn = moderator_role_id("grok-4-1-fast-reasoning", THIRD_NEUTRAL)
    mp = moderator_role_id("grok-4-1-fast-reasoning", THIRD_PLANNER)
    check("moderator cache ids differ by variant and from the deliberation "
          "runner's mod-<model>",
          mn != mp and mn.startswith(f"{MODERATOR_PREFIX}-{THIRD_NEUTRAL}-")
          and mp.startswith(f"{MODERATOR_PREFIX}-{THIRD_PLANNER}-")
          and rcd.moderator_role_id("grok-4-1-fast-reasoning") not in (mn, mp))
    check("no role id, arm, scaffold, tag or moderator prefix contains a "
          "forbidden word",
          not any(forbidden_hits(s) for s in
                  [r.role_id for r in ALL_ROLES] + list(ARM_OF.values())
                  + [SCAFFOLD_NAME, TAG_PREFIX, MODERATOR_PREFIX, mn, mp]))
    counts_before = cache_namespace_counts()
    # Addendum 16.13 has been RUN (Stage A/B1/B2, 2026-09-12), so this runner's
    # namespaces legitimately hold grok records. The standing check is the
    # sibling runners' idiom: no FOREIGN (stub / selftest / other-regime) file
    # carries any of this runner's names; a real run's files are the registered
    # footprint and are reported, not failed.
    foreign = [q.name for q in OUT_DIR.iterdir()
               if (f"_{SCAFFOLD_NAME}_" in q.name or q.name.startswith(f"{TAG_PREFIX}_"))
               and ("stub-model" in q.name or "_m_" in q.name or "selftest" in q.name)]
    check(f"no foreign (stub / selftest) file carries this runner's names in {OUT_DIR.name} "
          f"(own-run footprint: {counts_before})", foreign == [])
    check("cache_files_with_prefix(TAG_PREFIX) holds only this runner's own rows/votes CSVs",
          all(("_rows.csv" in q.name or "_votes.csv" in q.name or "_analysis" in q.name
               or "_summary" in q.name) for q in cache_files_with_prefix(TAG_PREFIX)))
    for ts in THIRD_SEATS:
        clean, cts = cache_is_clean("grok-4-1-fast-reasoning", ts)
        check(f"cache_is_clean({ts}) is False only because a registered run wrote {cts} "
              f"(the run path refuses without --resume; verified by the B2 refusal on 2026-09-12)",
              clean or all(v > 0 for v in cts.values()))
    stub_before = sorted(q.name for q in OUT_DIR.iterdir() if "stub-model" in q.name or q.name.startswith("cgd_m_"))
    cp = call_cache_path("m", SCAFFOLD_NAME, ARM_OF[THIRD_NEUTRAL], "111", 0,
                         "synthesis", mn, 1024)
    check("cache path carries scaffold, arm and the variant-bearing moderator id",
          SCAFFOLD_NAME in cp.name and ARM_OF[THIRD_NEUTRAL] in cp.name
          and mn in cp.name)

    # ---- instruments ----------------------------------------------------------
    check("label / vote instruments carry the three-way and binary tokens",
          "ACCEPT_WITH_MODIFICATION" in GAME_LABEL_INSTRUCTION
          and "VERDICT: REJECT" in GAME_VOTE_INSTRUCTION
          and "ACCEPT_WITH_MODIFICATION" not in GAME_VOTE_INSTRUCTION)
    check("no instruction offers UNRESOLVED",
          all("UNRESOLVED" not in s for s in
              (GAME_VERDICT_INSTRUCTION, GAME_LABEL_INSTRUCTION, GAME_VOTE_INSTRUCTION)))

    # ---- prompts: leak sweep over every game, both variants -----------------
    leaks: list = []
    steer: list = []
    meta: list = []
    last_ok = True
    n_prompts = 0
    for g in games:
        for idx in (0, 1):
            it = GameItem(render_item(g, idx))
            for ts in THIRD_SEATS:
                ps = all_prompts(it, ts)
                for k, p in ps.items():
                    n_prompts += 1
                    if forbidden_hits(p):
                        leaks.append((g.game_id, idx, ts, k, forbidden_hits(p)))
                    if steer_hits(p):
                        steer.append((g.game_id, idx, ts, k, steer_hits(p)))
                    if it.gold_type in p or "gold" in p.lower():
                        meta.append((g.game_id, idx, ts, k))
                    rd = k.split(":", 1)[0]
                    if rd in ROUND_INSTRUCTION and not p.endswith(ROUND_INSTRUCTION[rd]):
                        last_ok = False
    check(f"no forbidden word in any of {n_prompts} rendered prompts / system "
          f"prompts (144 games x 2 idx x both variants)"
          + (f" -- first leak {leaks[0]}" if leaks else ""), not leaks)
    check("no solution-concept vocabulary of ours in any prompt "
          f"({STEER_WORDS})" + (f" -- first {steer[0]}" if steer else ""),
          not steer)
    check("no gold_type name or the word 'gold' in any prompt", not meta)
    check("the round's verdict instruction is the LAST line of every prompt: "
          "PLAN instruction in r0/r1/r2/synthesis/integration, label in r3, "
          "vote in r4", last_ok)
    it0 = GameItem(render_item(games[0], 0))
    check("r0 and r4 prompts end with their instruction, verbatim",
          r0_user(it0, PLAYER_R_ADVOCATE).endswith(GAME_VERDICT_INSTRUCTION)
          and r4_vote_user(it0, PLAYER_R_ADVOCATE, "s", "ACCEPT", "", "p",
                           cap=0).endswith(GAME_VOTE_INSTRUCTION))
    check("r0 prompt quotes the item text verbatim inside the request block",
          f"{REQ_BEGIN}\n{it0.post_text}\n{REQ_END}" in r0_user(it0, NEUTRAL_READER))
    check("advocate labels use the item's player names",
          role_label(PLAYER_R_ADVOCATE, it0) == f"{it0.view.row_name}'s Advocate"
          and role_label(PLAYER_C_ADVOCATE, it0) == f"{it0.view.col_name}'s Advocate")
    check("both third-seat variants render distinct briefs and system prompts",
          role_brief(NEUTRAL_READER, it0) != role_brief(PLAN_MEDIATOR, it0)
          and synthesis_system(THIRD_NEUTRAL) != synthesis_system(THIRD_PLANNER)
          and "NO_UNIQUE_PLAN" in role_brief(PLAN_MEDIATOR, it0))
    check("mediator brief carries NO method clause (no 'reason to sign' / "
          "'rather than decline' / 'what each player gets'): the registered "
          "planner is task-only, like the neutral seat",
          not any(w in PLAN_MEDIATOR.brief.lower() for w in
                  ("reason to sign", "rather than decline", "what each player gets",
                   "consider"))
          and "Work from the payoff table alone." in PLAN_MEDIATOR.brief)
    check("no game-type STEM in any rendered prompt either (superset of the "
          "literal check: 'cooperating' / 'cooperative' would count)",
          not any(type_named(p) for g in games[:40] for idx in (0, 1)
                  for ts in THIRD_SEATS
                  for p in all_prompts(GameItem(render_item(g, idx)), ts).values()))
    check("type_named: stem matcher catches inflections forbidden_hits misses "
          "and is clean on innocent text",
          type_named("they keep cooperating") and type_named("a cooperative plan")
          and type_named("the classic PRISONER setup") and type_named("Defecting")
          and not forbidden_hits("they keep cooperating")
          and not type_named("both should sign the plan") and not type_named(""))

    # ---- stake_undermined on real games --------------------------------------
    pd_game = next(g for g in games if g.gold_type == "DILEMMA_BOTH_PREFER")
    pdi = GameItem(render_item(pd_game, 0))
    gold = pdi.gold_verdict
    nash = next(iter(pdi.view.nash_verdicts))
    check("value game: gold plan undermines neither advocate (both strictly "
          "above their Nash payoff)",
          stake_undermined("player_r_advocate", gold, pdi) is False
          and stake_undermined("player_c_advocate", gold, pdi) is False)
    check("value game: the Nash plan itself undermines neither (equal, not lower)",
          stake_undermined("player_r_advocate", nash, pdi) is False
          and stake_undermined("player_c_advocate", nash, pdi) is False)
    rb = pdi.view.row_best_verdict
    check("value game: the row player's best cell undermines the column "
          "advocate and not the row advocate",
          stake_undermined("player_c_advocate", rb, pdi) is True
          and stake_undermined("player_r_advocate", rb, pdi) is False)
    check("third seats are False on a committal plan, never None",
          stake_undermined("neutral_reader", rb, pdi) is False
          and stake_undermined("plan_mediator", rb, pdi) is False)
    check("NO_UNIQUE_PLAN -> None for advocates (no plan, no payoff), False "
          "for the third seat",
          stake_undermined("player_r_advocate", NO_UNIQUE_PLAN, pdi) is None
          and stake_undermined("neutral_reader", NO_UNIQUE_PLAN, pdi) is False)
    check("NOVERDICT / UNRESOLVED -> None for every seat",
          all(stake_undermined(rid, v, pdi) is None
              for rid in ROLE_BY_ID for v in (NOVERDICT, "UNRESOLVED")))
    cyc = GameItem(render_item(next(g for g in games
                                    if g.gold_type == "CYCLIC_NO_IDEAL"), 0))
    check("a game with no pure Nash cell -> None for advocates on any plan",
          not cyc.view.nash_verdicts
          and all(stake_undermined(s, v, cyc) is None
                  for s in STAKE_SEATS for v in PLAN_VERDICTS))
    check("best_nash_payoff / player_payoff / maximin_payoff agree with the view",
          best_nash_payoff(pdi, "row") == max(pdi.view.payoffs_for(v)[0]
                                              for v in pdi.view.nash_verdicts)
          and player_payoff(pdi, rb, "row") == 4
          and maximin_payoff(pdi, "col") == maximin(pdi.view.pA, pdi.view.pB)[1]
          and player_payoff(pdi, NO_UNIQUE_PLAN, "row") is None)

    # ---- parsing helpers --------------------------------------------------------
    check("extract_addressed parses this arm's ids and 'none'",
          extract_addressed(f"{ADDRESSED_MARKER} player_c_advocate, plan_mediator",
                            role_order(THIRD_PLANNER))
          == ("player_c_advocate", "plan_mediator")
          and extract_addressed(f"{ADDRESSED_MARKER} neutral_reader",
                                role_order(THIRD_PLANNER)) == ()
          and extract_addressed(f"{ADDRESSED_MARKER} none", role_order(THIRD_NEUTRAL)) == ())
    check("extract_objection (imported) reads both markers",
          extract_objection(f"x\n{MOD_MARKER} fix it")[0] == "modification"
          and extract_objection(f"x\n{UNRESOLVABLE_MARKER} no")[0] == "unresolvable")
    check("parse_round normalises PLAN spacing on plan rounds and uses the "
          "shared extractor on label / vote rounds",
          parse_round("r0", "VERDICT: PLAN 21") == "PLAN_21"
          and parse_round("integration", "VERDICT: NO UNIQUE PLAN") == NO_UNIQUE_PLAN
          and parse_round("r3_label", "VERDICT: ACCEPT_WITH_MODIFICATION")
          == "ACCEPT_WITH_MODIFICATION"
          and parse_round("r3_label", "VERDICT: ACCEPT") == "ACCEPT"
          and parse_round("r3_label", "VERDICT: REJECT") == "REJECT"
          and parse_round("r4_vote", "VERDICT: ACCEPT") == "ACCEPT"
          and parse_round("r4_vote", "VERDICT: REJECT") == "REJECT"
          and parse_round("r4_vote", "VERDICT: PLAN_12") == NOVERDICT)
    check("parse_round collapses the spaced / hyphenated 'ACCEPT WITH "
          "MODIFICATION' to the exact label token (never to ACCEPT), and that "
          "text is NOVERDICT on the vote round",
          parse_round("r3_label", "VERDICT: ACCEPT WITH MODIFICATION")
          == "ACCEPT_WITH_MODIFICATION"
          and parse_round("r3_label", "VERDICT: ACCEPT-WITH-MODIFICATION")
          == "ACCEPT_WITH_MODIFICATION"
          and parse_round("r3_label", "**VERDICT: Accept with modification**")
          == "ACCEPT_WITH_MODIFICATION"
          and parse_round("r4_vote", "VERDICT: ACCEPT WITH MODIFICATION") == NOVERDICT
          and parse_round("r4_vote", "VERDICT: ACCEPT_WITH_MODIFICATION") == NOVERDICT)
    check("a stray 'VERDICT: UNRESOLVED' is NOVERDICT on EVERY round (label and "
          "vote included; verdict_format registers it as legal, no instruction "
          "offers it)",
          all(parse_round(rd, "VERDICT: UNRESOLVED") == NOVERDICT for rd in ROUNDS)
          and all(parse_round(rd, "VERDICT: UNRESOLVED") == NOVERDICT
                  for rd in ("r3_label", "r4_vote")))
    check("ROUND_INSTRUMENT covers all seven rounds (r0-r2 guarded too)",
          set(ROUND_INSTRUMENT) == set(ROUNDS))
    check("call_guard_row: NOVERDICT on empty text, parsed token otherwise",
          call_guard_row({"model": "m", "scaffold": "s", "round": "r0",
                          "role_id": "x", "arm": "a", "output": ""})["verdict"]
          == NOVERDICT
          and call_guard_row({"model": "m", "scaffold": "s", "round": "r4_vote",
                              "role_id": "x", "arm": "a",
                              "output": "VERDICT: REJECT"})["verdict"] == "REJECT")
    unres = lambda rd: call_guard_row({"model": "m", "scaffold": "s", "round": rd,  # noqa: E731
                                       "role_id": "x", "arm": "a",
                                       "output": "VERDICT: UNRESOLVED"})
    check("call_guard_row: 'VERDICT: UNRESOLVED' is NOVERDICT on r3_label and "
          "r4_vote (the sibling's guard lets it through)",
          unres("r3_label")["verdict"] == NOVERDICT
          and unres("r4_vote")["verdict"] == NOVERDICT
          and unres("r0")["verdict"] == NOVERDICT)
    all_unres = [unres("r4_vote") for _ in range(20)] + [unres("r3_label") for _ in range(20)]
    rep_u = truncation_report(all_unres)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_truncation_report(rep_u)
    check("a run whose every r3 label / r4 vote is UNRESOLVED FAILS the guard "
          "and prints GUARD FAILED (no silent G1 = 0 / G2 = 0)",
          not rep_u["pass"] and "GUARD FAILED" in buf.getvalue())
    dirty = [{"model": "m", "scaffold": "s", "arm": "a", "verdict": "PLAN_11",
              "truncated": 0, "output_len": 100, "finish_reason": "stop"}] * 18
    dirty += [{"model": "m", "scaffold": "s", "arm": "a", "verdict": NOVERDICT,
               "truncated": 1, "output_len": 5, "finish_reason": "length"}] * 2
    rep = truncation_report(dirty)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_truncation_report(rep)
    check("imported guard fails at 10% NOVERDICT/truncation and prints GUARD FAILED",
          not rep["pass"] and "GUARD FAILED" in buf.getvalue())
    check("ROW_FIELDS and VOTE_FIELDS have no duplicate columns",
          len(ROW_FIELDS) == len(set(ROW_FIELDS))
          and len(VOTE_FIELDS) == len(set(VOTE_FIELDS)))

    # ---- dry-run cost model --------------------------------------------------
    value = select_games(games, REGION_VALUE)
    for ts in THIRD_SEATS:
        cm = cost_model("grok-4-1-fast-reasoning", value, 2, third_seat=ts,
                        use_bpe=False)
        check(f"dry-run [{ts}]: cells = 16 games x 2 samples = 32, 17 calls each, "
              f"544 total, priced",
              cm["cells"] == 32 and cm["calls_per_cell"] == 17
              and cm["total_calls"] == 544 and cm["priced"]
              and sum(r["n_calls"] for r in cm["rounds"].values()) == 544
              and cm["total_usd"] > 0)
    cm_all = cost_model("grok-4-1-fast-reasoning", games, 1,
                        third_seat=THIRD_PLANNER, use_bpe=False)
    check("dry-run [all]: 144 cells x 17 = 2448 calls",
          cm_all["cells"] == 144 and cm_all["total_calls"] == 2448)
    check("dry-run flags an unpriced model instead of reporting $0 silently",
          not cost_model("no-such-model", value, 1, third_seat=THIRD_NEUTRAL,
                         use_bpe=False)["priced"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_cost_model(cm)
    check("print_cost_model prints the cell arithmetic",
          "32 cells x 17 calls = 544" in buf.getvalue())

    # ---- CLI refusals: no --run means no generation ----------------------------
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--games", "value", "--samples", "1", "--no-bpe"])
    check("main() without --run prints the cost model and REFUSES (exit 2)",
          rc == 2 and "COST MODEL" in buf.getvalue() and "--run" in buf.getvalue())
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--dry-run", "--games", "cannot_assist", "--samples", "3",
                   "--third-seat", "planner", "--no-bpe"])
    check("main(--dry-run) exits 0 with 31 x 3 = 93 cells",
          rc == 0 and "93 cells" in buf.getvalue())
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--dry-run", "--games", "value", "--third-seat", "planner",
                   "--tag", f"{TAG_PREFIX}_{THIRD_NEUTRAL}", "--no-bpe"])
    check("main() refuses --tag equal to the OTHER variant's default tag (exit 2) "
          "before doing anything",
          rc == 2 and "different" in buf.getvalue() and "COST MODEL" not in buf.getvalue())
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--dry-run", "--games", "value", "--model", "no-such-model",
                   "--no-bpe"])
    check("main(--dry-run) on an unpriced model still prints the WARNING and exits "
          "0 (nothing spent); the --run path refuses it (exit 2) -- not exercised "
          "here because the selftest never passes --run",
          rc == 0 and "WARNING" in buf.getvalue()
          and "model not in PRICES" in inspect.getsource(main))
    check("dry runs wrote nothing into any namespace (stub-model / model 'm' files unchanged; "
          "a concurrent registered grok run may add its own files)",
          sorted(q.name for q in OUT_DIR.iterdir() if "stub-model" in q.name or q.name.startswith("cgd_m_")) == stub_before)

    # ---- end-to-end protocol against the stub; cache redirected ----------------
    real_gen = rcd.generate_any
    real_out = rcd.OUT_DIR
    smoke_games = [pd_game,
                   next(g for g in games if g.gold_type == "ASSURANCE"),
                   next(g for g in games if g.gold_type == "ZERO_SUM")]
    rows: list[dict] = []
    votes: list[dict] = []
    calls: list[dict] = []
    n_cache_files = 0
    try:
        rcd.generate_any = _stub_generate  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as td:
            rcd.OUT_DIR = Path(td)
            try:
                for ts in THIRD_SEATS:
                    for it in build_items(smoke_games, 1):
                        row, vr, cr = run_deliberation("stub-model", it,
                                                       third_seat=ts)
                        rows.append(row)
                        votes.extend(vr)
                        calls.extend(cr)
                n_cache_files = len(list(Path(td).iterdir()))
            finally:
                rcd.OUT_DIR = real_out
    finally:
        rcd.generate_any = real_gen
    check("smoke: 6 cells (3 games x 2 variants), 17 calls each, one cache "
          "file per call in the TEMP dir",
          len(rows) == 6 and len(calls) == 6 * CALLS_PER_CELL
          and n_cache_files == 6 * CALLS_PER_CELL)
    check("smoke: nothing landed in the real cache directory (stub-model files unchanged)",
          sorted(q.name for q in OUT_DIR.iterdir() if "stub-model" in q.name or q.name.startswith("cgd_m_")) == stub_before)
    check("smoke: group verdict PLAN_12 parsed from the 'PLAN 12' spacing variant",
          all(r["verdict"] == "PLAN_12" for r in rows))
    check("smoke: per-seat r0 verdicts recorded (row PLAN_11, col PLAN_22, "
          "third PLAN_12)",
          all(r["r0_verdict_row_advocate"] == "PLAN_11"
              and r["r0_verdict_col_advocate"] == "PLAN_22"
              and r["r0_verdict_third"] == "PLAN_12" for r in rows))
    check("smoke: r3 labels, objector, addressed and votes as planted",
          all(r["r3_labels"] == "ACCEPT_WITH_MODIFICATION|ACCEPT|ACCEPT"
              and r["n_objectors"] == 1 and r["n_modifications_addressed"] == 1
              and r["addressed_roles"] == "player_r_advocate"
              and r["n_reject"] == 1 and r["n_accept"] == 2
              and r["rejecting_roles"] == "player_r_advocate"
              and r["unanimous_accept"] == 0 and r["verdict_revised"] == 0
              and r["vote_row_advocate"] == "REJECT" for r in rows))
    check("smoke: third_role_id / arm follow the variant",
          {(r["third_seat"], r["third_role_id"], r["arm"]) for r in rows}
          == {(THIRD_NEUTRAL, "neutral_reader", ARM_OF[THIRD_NEUTRAL]),
              (THIRD_PLANNER, "plan_mediator", ARM_OF[THIRD_PLANNER])})
    check("smoke: rows carry gold_type, region, gold_verdict, nash_verdicts, "
          "display permutation and correctness",
          all(r["gold_type"] and r["region"] in REGIONS
              and r["gold_verdict"] in VERDICT_SET
              and r["row_perm"] in ("01", "10") and r["col_perm"] in ("01", "10")
              and r["correct"] == int(r["verdict"] == r["gold_verdict"])
              for r in rows)
          and any(r["nash_verdicts"] for r in rows))
    check("smoke: every ROW_FIELDS / VOTE_FIELDS key is present",
          all(k in rows[0] for k in ROW_FIELDS)
          and all(k in votes[0] for k in VOTE_FIELDS))
    check("smoke: the clean stub quoted no game-type stem into any prompt "
          "(n_calls_type_named 0, first round '', per-vote 0 / '')",
          all(r["n_calls_type_named"] == 0 and r["type_named_first_round"] == ""
              for r in rows)
          and all(v["type_named_in_prompt"] == 0 and v["type_named_rounds"] == ""
                  for v in votes))
    exp_und = {}
    for v in votes:
        it = next(GameItem(render_item(g, 0)) for g in smoke_games
                  if g.game_id == v["item_id"])
        u = stake_undermined(v["role_id"], v["group_verdict"], it)
        exp_und[(v["item_id"], v["third_seat"], v["role_id"])] = "" if u is None else int(u)
    check("smoke: vote rows' stake_undermined equals stake_undermined() "
          "recomputed from the item",
          all(v["stake_undermined"] == exp_und[(v["item_id"], v["third_seat"],
                                                 v["role_id"])] for v in votes))
    check("smoke: vote rows carry the payoff reference columns and r0 Nash flags",
          all(v["group_plan_payoff"] != "" and v["best_nash_payoff"] != ""
              for v in votes if v["role_id"] in STAKE_SEATS
              and v["gold_type"] != "CYCLIC_NO_IDEAL")
          and all(v["r0_is_nash"] in (0, 1) for v in votes))
    grep = guard_rows(calls)
    check("smoke: guard rows cover all 17 calls per cell", len(grep) == len(calls))
    rep = truncation_report(grep)
    check("smoke: guard passes on clean stub output", rep["pass"])
    ro = readout(rows, votes, draws=100, seed=1)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_readout(ro, label="selftest")
    check("readout: per-gold_type accuracy table and Nash-pull by seat print",
          "extent chart" in buf.getvalue() and "Nash-pull" in buf.getvalue()
          and set(ro["nash_pull"]) == {"player_r_advocate", "player_c_advocate",
                                       "neutral_reader", "plan_mediator"}
          and ro["nash_pull"]["player_r_advocate"]["r0_all"]["n"] == 6)
    check("readout: share of cells with a game-type stem in any prompt is "
          "reported (0 on the clean stub) and printed",
          ro["type_named_cells"]["n"] == 6 and ro["type_named_cells"]["point"] == 0.0
          and "game-type stem" in buf.getvalue())

    # ---- a leaking seat is MEASURED, not silently passed through -------------
    def _leaky_generate(model, system, user, *, sample_idx=0, max_tokens=0, **kw):
        m = _PLAYERS_RE.search(user)
        R = m.group(1) if m else ""
        if (R and f"speaking as {R}'s Advocate" in user
                and "Write your opening statement" in user):
            return _Stub("This is the classic setup where they keep cooperating "
                         "until one defects.\n\nVERDICT: PLAN_11")
        return _stub_generate(model, system, user, sample_idx=sample_idx,
                              max_tokens=max_tokens, **kw)
    try:
        rcd.generate_any = _leaky_generate  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as td:
            rcd.OUT_DIR = Path(td)
            try:
                lrow, lvotes, lcalls = run_deliberation(
                    "stub-model", GameItem(render_item(pd_game, 0)),
                    third_seat=THIRD_NEUTRAL)
            finally:
                rcd.OUT_DIR = real_out
    finally:
        rcd.generate_any = real_gen
    lv = {v["role_id"]: v for v in lvotes}
    check("leaky r0 (row advocate names the type with 'cooperating' / "
          "'defects'): r0 prompts are clean, every r1 prompt quotes it, so "
          "n_calls_type_named >= 3, first round r1, and every seat's vote row "
          "has type_named_in_prompt 1 with r1 in type_named_rounds",
          lrow["n_calls_type_named"] >= 3 and lrow["type_named_first_round"] == "r1"
          and all(v["type_named_in_prompt"] == 1 and "r1" in v["type_named_rounds"]
                  and "r0" not in v["type_named_rounds"] for v in lv.values()))
    check("leak logging is a superset of the literal check: 'cooperating' is "
          "not in forbidden_hits' word list yet is counted ('defects' is caught "
          "by both, as a substring)",
          not forbidden_hits("they keep cooperating")
          and type_named("they keep cooperating")
          and forbidden_hits("until one defects") == ["defect"]
          and type_named("until one defects"))

    # ---- errored cells are a guard failure, never a silent omission ------------
    n_gen = [0]

    def _raising_generate(model, system, user, *, sample_idx=0, max_tokens=0, **kw):
        n_gen[0] += 1
        if ("This is the final round and your vote is binary" in user
                and n_gen[0] % 7 == 0):
            raise RuntimeError("simulated 5xx")
        return _stub_generate(model, system, user, sample_idx=sample_idx,
                              max_tokens=max_tokens, **kw)
    try:
        rcd.generate_any = _raising_generate  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as td:
            rcd.OUT_DIR = Path(td)
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    e_rows, e_votes, e_calls, e_failed = run(
                        "stub-model", build_items(smoke_games, 2),
                        third_seat=THIRD_NEUTRAL, moderator_model=None,
                        max_tokens_agent=64, max_tokens_moderator=64,
                        max_tokens_label=64, max_tokens_vote=64,
                        transcript_cap=0, workers=1)
            finally:
                rcd.OUT_DIR = real_out
    finally:
        rcd.generate_any = real_gen
    check(f"run(): a raising chain is returned as a failed key, dropped from rows "
          f"({len(e_rows)} kept, {len(e_failed)} failed of 6), and the survivors' "
          f"guard rows are 17 per kept cell",
          e_failed and len(e_rows) + len(e_failed) == 6
          and len(e_rows) < 6
          and len(guard_rows(e_calls)) == 17 * len(e_rows)
          and all(k not in {(r["item_id"], r["sample_idx"]) for r in e_rows}
                  for k in e_failed)
          and "ERROR" in buf.getvalue())
    check("survivors alone would PASS the truncation guard (the attrition the "
          "errored-cells guard exists to catch)",
          truncation_report(guard_rows(e_calls))["pass"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok_e = errored_cells_report(e_failed, 6)
        ok_c = errored_cells_report([], 6)
    check("errored_cells_report prints GUARD FAILED naming N of M and the keys, "
          "returns False; passes silently on no failures",
          not ok_e and ok_c
          and f"GUARD FAILED: {len(e_failed)} of 6 cells errored" in buf.getvalue()
          and all(f"{k[0]}/{k[1]}" in buf.getvalue() for k in e_failed))
    gf_r, gf_v = guard_failed_paths("cg_game_neutral", Path("/nonexistent"))
    check("guard-failed dumps go to <tag>_GUARD_FAILED_rows/votes.csv, never "
          "the run tag's own CSVs",
          gf_r.name == "cg_game_neutral_GUARD_FAILED_rows.csv"
          and gf_v.name == "cg_game_neutral_GUARD_FAILED_votes.csv")
    with tempfile.TemporaryDirectory() as td:
        pth = Path(td) / "x_rows.csv"
        _write_csv(pth, [{"third_seat": THIRD_PLANNER}], ("third_seat",))
        check("existing_rows_third_seats reads the third_seat column of an existing "
              "rows CSV (used to refuse a cross-variant overwrite) and is empty "
              "for a missing file",
              existing_rows_third_seats(pth) == {THIRD_PLANNER}
              and existing_rows_third_seats(Path(td) / "none.csv") == set())

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# Run loop / CLI
# ---------------------------------------------------------------------------


def run(model: str, items: Sequence[GameItem], *, third_seat: str,
        moderator_model: Optional[str], max_tokens_agent: int,
        max_tokens_moderator: int, max_tokens_label: int, max_tokens_vote: int,
        transcript_cap: int, workers: int):
    """Returns (rows, votes, calls, failed): ``failed`` is the list of
    (item_id, sample_idx) whose chain raised. An errored cell is a guard
    failure for main(), never a silent omission (module docstring)."""
    print(f"  {len(items)} cells x {CALLS_PER_CELL} calls = "
          f"{len(items) * CALLS_PER_CELL} generation calls")
    rows, votes, calls = [], [], []
    failed: list[tuple[str, int]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(
                run_deliberation, model, it, third_seat=third_seat,
                moderator_model=moderator_model,
                max_tokens_agent=max_tokens_agent,
                max_tokens_moderator=max_tokens_moderator,
                max_tokens_label=max_tokens_label,
                max_tokens_vote=max_tokens_vote,
                transcript_cap=transcript_cap,
            ): (it.item_id, it.sample_idx)
            for it in items
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                row, vr, cr = fut.result()
                rows.append(row)
                votes.extend(vr)
                calls.extend(cr)
                done += 1
                if done % 10 == 0:
                    print(f"    {done}/{len(items)} cells", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR {key}: {e}", flush=True)
                failed.append(key)
    rows.sort(key=lambda r: (r["item_id"], r["sample_idx"]))
    votes.sort(key=lambda v: (v["item_id"], v["sample_idx"], v["role_id"]))
    failed.sort()
    return rows, votes, calls, failed


def errored_cells_report(failed: Sequence[tuple[str, int]], n_items: int) -> bool:
    """Print the errored-cells guard; True when it passes (nothing failed)."""
    if not failed:
        return True
    print(f"\n  *** GUARD FAILED: {len(failed)} of {n_items} cells errored "
          f"(chain raised; dropped from rows, votes and calls). Failed keys: "
          f"{', '.join(f'{k[0]}/{k[1]}' for k in failed)}")
    print("  The completed chains are cached; re-run with --resume to pick "
          "them up at no extra spend. No readout is printed and no rows / "
          "votes CSV is written under the run tag.")
    return False


def guard_failed_paths(tag: str, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    """Where a guard-failed run dumps what it has (never the run tag's own
    CSVs, which --readout and the analyzer read)."""
    return (out_dir / f"{tag}_GUARD_FAILED_rows.csv",
            out_dir / f"{tag}_GUARD_FAILED_votes.csv")


def existing_rows_third_seats(rows_path: Path) -> set[str]:
    """third_seat values in an existing rows CSV (empty set if absent)."""
    if not rows_path.exists():
        return set()
    return {r.get("third_seat", "") for r in _read_csv(rows_path)}


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Seven-stage deliberation on the TMGBench classic 2x2 "
                    "games (binding joint plan; advocates + neutral/planner "
                    "third seat). Generates NOTHING unless --run is passed.")
    ap.add_argument("--selftest", action="store_true",
                    help="offline self-check; stubs generation, no spend")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cost model and exit")
    ap.add_argument("--run", action="store_true",
                    help="REQUIRED to generate; spends money")
    ap.add_argument("--resume", action="store_true",
                    help="proceed despite a non-empty cache footprint for this "
                         "(model, arm); only valid when the briefs are unchanged")
    ap.add_argument("--readout", action="store_true",
                    help="print the readout from an existing --tag's CSVs (offline)")
    ap.add_argument("--model", default="grok-4-1-fast-reasoning")
    ap.add_argument("--moderator-model", default=None)
    ap.add_argument("--third-seat", choices=THIRD_SEATS, default=THIRD_NEUTRAL)
    ap.add_argument("--games", choices=("all",) + REGIONS, default="all")
    ap.add_argument("--game-ids", default=None,
                    help="comma-separated game ids to intersect with --games")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens-agent", type=int, default=2560)
    ap.add_argument("--max-tokens-moderator", type=int, default=1024)
    ap.add_argument("--max-tokens-label", type=int, default=3072)
    ap.add_argument("--max-tokens-vote", type=int, default=3072)
    ap.add_argument("--transcript-cap", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-bpe", action="store_true",
                    help="regex token approximation in the cost model (offline)")
    ap.add_argument("--tag", default=None,
                    help=f"CSV tag; default {TAG_PREFIX}_<third-seat>")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    tag = args.tag or f"{TAG_PREFIX}_{args.third_seat}"
    rows_path = OUT_DIR / f"{tag}_rows.csv"
    votes_path = OUT_DIR / f"{tag}_votes.csv"
    other_default_tags = {f"{TAG_PREFIX}_{ts}" for ts in THIRD_SEATS
                          if ts != args.third_seat}
    if args.tag in other_default_tags and not args.readout:
        print(f"\nERROR: --tag {args.tag!r} is the default tag of a different "
              f"third-seat variant; it would overwrite that arm's CSVs. Use "
              f"--third-seat {args.tag[len(TAG_PREFIX) + 1:]} or another tag.\n")
        return 2

    if args.readout:
        if not rows_path.exists() or not votes_path.exists():
            print(f"\nERROR: {rows_path} / {votes_path} not found.\n")
            return 2
        ro = readout(_read_csv(rows_path), _read_csv(votes_path),
                     draws=args.n_boot, seed=7)
        print_readout(ro, label=tag)
        return 0

    if args.samples < 1:
        print("\nERROR: --samples must be >= 1\n")
        return 2
    try:
        games = load_games()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}\n")
        return 2
    ids = args.game_ids.split(",") if args.game_ids else None
    games = select_games(games, args.games, ids)
    if not games:
        print("\nERROR: no games selected.\n")
        return 2
    by_type = Counter(g.gold_type for g in games)
    print(f"games {args.games}: {len(games)} selected "
          f"{dict(sorted(by_type.items()))}")
    print(f"model {args.model}  moderator "
          f"{args.moderator_model or '(same as agent)'}  third seat "
          f"{args.third_seat} (arm {ARM_OF[args.third_seat]})  tag {tag}")
    print(f"protocol: {PROTOCOL}")

    cm = cost_model(args.model, games, args.samples, third_seat=args.third_seat,
                    transcript_cap=args.transcript_cap,
                    moderator_model=args.moderator_model,
                    use_bpe=not args.no_bpe)
    print_cost_model(cm)
    if args.dry_run:
        return 0
    if not args.run:
        print("\n*** NOT RUNNING: pass --run to generate (this spends money). "
              "Nothing was generated. ***\n")
        return 2
    if not cm["priced"]:
        print("\nERROR: model not in PRICES; add a price to "
              "run_crowdgold_deliberation.PRICES before spending (the $0.00 "
              "above is a placeholder, not a ceiling).\n")
        return 2
    seats_on_disk = existing_rows_third_seats(rows_path)
    if seats_on_disk and seats_on_disk != {args.third_seat}:
        print(f"\nERROR: {rows_path} already holds rows for third seat(s) "
              f"{sorted(seats_on_disk)}; refusing to overwrite them with a "
              f"{args.third_seat} run. Pick another --tag.\n")
        return 2

    clean, counts = cache_is_clean(args.model, args.third_seat,
                                   moderator_model=args.moderator_model)
    if not clean and not args.resume:
        print(f"\nERROR: cache is not clean for this footprint {counts}.\n"
              "A previous partial run under possibly-different briefs would be "
              "replayed silently (role briefs are not covered by parent_sha).\n"
              "Pass --resume only if you are certain the briefs are unchanged.\n")
        return 3
    if not clean:
        print(f"  --resume: {counts} existing records will be served from cache")

    items = build_items(games, args.samples)
    rows, votes, calls, failed = run(
        args.model, items, third_seat=args.third_seat,
        moderator_model=args.moderator_model,
        max_tokens_agent=args.max_tokens_agent,
        max_tokens_moderator=args.max_tokens_moderator,
        max_tokens_label=args.max_tokens_label,
        max_tokens_vote=args.max_tokens_vote,
        transcript_cap=args.transcript_cap, workers=args.workers,
    )
    if not rows:
        print("\nNo rows produced.\n")
        errored_cells_report(failed, len(items))
        return 1

    grep = guard_rows(calls)
    rep = truncation_report(grep)
    print_truncation_report(rep)
    print(f"  calls in guard: {len(grep)} of {len(items) * CALLS_PER_CELL} "
          f"expected ({len(rows)} of {len(items)} cells completed)")
    cells_ok = errored_cells_report(failed, len(items))

    pt = sum(int(c.get("prompt_tokens", 0) or 0) for c in calls)
    ct = sum(int(c.get("completion_tokens", 0) or 0) for c in calls)
    pin, pout = PRICES.get(args.model, (0.0, 0.0))
    print(f"\nmeasured spend: {pt} prompt + {ct} completion tokens "
          f"(mixed agent/moderator pricing not split here) ~= "
          f"${(pt / 1e6 * pin + ct / 1e6 * pout):.2f} at agent-model rates")

    if not rep["pass"] or not cells_ok:
        gf_rows, gf_votes = guard_failed_paths(tag)
        _write_csv(gf_rows, rows, ROW_FIELDS)
        _write_csv(gf_votes, votes, VOTE_FIELDS)
        print(f"\n*** GUARD FAILED. Do not read accuracy, Nash-pull or G1/G2/G3 "
              "off this run. ***")
        print(f"  partial rows / votes dumped to {gf_rows.name} / {gf_votes.name} "
              f"(NOT the run tag's CSVs); fix the failing round's cap or the "
              f"errored cells and re-run with --resume.")
        return 4

    _write_csv(rows_path, rows, ROW_FIELDS)
    _write_csv(votes_path, votes, VOTE_FIELDS)
    print(f"\nwrote {len(rows)} rows -> {rows_path}")
    print(f"wrote {len(votes)} votes -> {votes_path}")

    print_readout(readout(rows, votes, draws=args.n_boot, seed=7), label=tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
