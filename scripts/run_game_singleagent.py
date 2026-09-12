"""
scripts/run_game_singleagent.py -- Stage A of the binding-joint-plan
instrument: the SOLO pilot. One model reads one leak-controlled 2x2 game
COLD (no role, no other seat's text, no deliberation) and names the joint
plan both players should accept, or NO_UNIQUE_PLAN. This is the single-
agent comparator that the three-seat deliberation runner (player_r_advocate
/ player_c_advocate / {neutral, planner}) will later be read against, and
the ``--masked`` hypothesis-only baseline that certifies the item text
carries no game information.

``--regime nonbinding`` (prereg Addendum 16.13c, Stage C1's solo
comparator) is the SAME runner with the question text and the verdict set
swapped: each player chooses privately and simultaneously, no agreement,
nothing to enforce one, a recommendation neither player is bound by; the
model names the one pair of options that should be recommended to both,
PLAN_xy, or NO_RECOMMENDATION -- the SAME words run_crowdgold_game_nb puts
to its three seats (tmg_games_nb pins the two byte-for-byte). See REGIMES
below. The default, ``--regime binding``, is byte-identical to the runner
before the flag existed: same prompts, same cache paths, same rows, same
analysis JSON.

WHY A SEPARATE, MINIMAL RUNNER (same reasoning as run_dilemma_singleagent)

The solo comparator must be structurally incapable of seeing anything a
deliberation produced. Keeping it in its own file, with exactly one call
per (game, sample) and a fixed system prompt, makes that guarantee visible
in the code rather than resting on a mode flag being passed correctly.

WHAT IS REUSED, VERBATIM

  scripts.tmg_games            the 144 sha256-pinned games, first-principles
                               gold, leak-controlled rendering (per-item
                               action permutation + neutral names, seeded by
                               (game_id, sample_idx)), the PLAN verdict
                               instrument, parse_guard, masked_floor_report
  scripts.run_crowdgold_deliberation.do_call / call_cache_path / PRICES
                               cache-or-generate with the parent-sha
                               staleness check, exactly as the Dilemmas
                               runners use it. do_call touches only
                               item.item_id / item.gold_verdict /
                               item.post_text, which GameItem duck-types.
  scripts.run_crowdgold_aita.count_tokens / _optional_bpe_counter
                               the dry-run token estimate

Nothing about the game is computed here: gold, Nash cells, permutations and
verdict parsing all come from tmg_games, whose selftest pins them 144/144
against the canonical gold file.

THE PROMPT, AND WHAT MUST NEVER BE IN IT

system = SOLO_SYSTEM, a neutral analyst who represents neither player and
may work through the table briefly. user = ItemView.text (scenario +
binding-agreement question + PLAN legend, and nothing else) followed by
GAME_VERDICT_INSTRUCTION as the LAST thing in the prompt. No other
ItemView field -- gold_type (whose names contain the word DILEMMA), gold
verdict, Nash set, permutation, canonical payoffs -- is ever placed in a
prompt; run_one additionally refuses to send any prompt in which
tmg_games.forbidden_hits finds a hit (cooperate / defect / cooperation /
defection / prisoner / dilemma / betray / trust), and the selftest scans
every rendered prompt for all 144 games.

MASKED (HYPOTHESIS-ONLY) BASELINE

``--masked`` renders the same (game_id, sample_idx) items with every payoff
number replaced by "?" -- same permutation, same names, same gold -- so a
model can only guess from surface form. Its accuracy must sit at the
base-rate floor (majority-gold rate, 31/144 = 0.215 NO_UNIQUE_PLAN on the
full set); masked_floor_report prints LEAK SUSPECTED if the item-clustered
95% CI sits above it, and no unmasked result may be read until that is
fixed. Masked rows carry arm ``masked`` and default to their own tag, so
they can never be pooled with ``full`` rows by accident.

REGIMES (the --regime flag; scripts.tmg_games_nb holds the nb instrument)

  binding (default)   ItemView from tmg_games.render_item: binding-agreement
                      question, PLAN legend, verdict set PLAN_11 / PLAN_12 /
                      PLAN_21 / PLAN_22 / NO_UNIQUE_PLAN, gold = gold_cell
                      through the permutation or NO_UNIQUE_PLAN; scaffold
                      tmg_solo, tags game_solo / game_solo_masked.
  nonbinding          ItemViewNB from tmg_games_nb.render_item_nb: the SAME
                      permutation, names, table and scenario for the same
                      (game_id, sample_idx), with the non-binding question
                      and legend; verdict set PLAN_11 / PLAN_12 / PLAN_21 /
                      PLAN_22 / NO_RECOMMENDATION (instrument
                      tmg_recommendation_verdict, registered in the
                      tmg_games idiom and shared by name with
                      run_crowdgold_game_nb); gold = gold_cell through
                      the permutation or
                      NO_RECOMMENDATION; scaffold tmg_solo_nb, tags
                      game_solo_nb / game_solo_nb_masked. A stray
                      NO_UNIQUE_PLAN is NOVERDICT here (and vice versa), so
                      a record from one regime can never score as the other.
The system prompt is the same in both regimes; the view carries its regime
(ItemViewNB.regime) so every dispatch -- instruction line, parser, scorer,
no-answer token, masked floor, cache scaffold -- is a function of the view,
and --analyze infers the regime from the rows' scaffold column and refuses
a --regime that contradicts it. The analysis JSON carries a ``regime`` key
under nonbinding only; a JSON without one is binding, exactly as every
file written before the flag existed. The Nash-pull readout is the same
quantity in both regimes (PLAN_xy tokens are shared), and is the registered
"solo non-binding accuracy by region" comparator's companion readout.

CACHE NAMESPACE

Cache files are named by do_call's call_cache_path:
  cgd_{model}_tmg_solo_t{max_tokens}_{full|masked}_{game_id}_{idx:02d}_single_solo_analyst.json
  cgd_{model}_tmg_solo_nb_t{max_tokens}_{full|masked}_{game_id}_{idx:02d}_single_solo_analyst.json
The record's parent sha covers the FULL model-facing text, system prompt
and user prompt (``parents=(system, user)``), not only the item text: an
edit to SOLO_SYSTEM or to the fixed instruction sentence invalidates every
cached record instead of being replayed silently (the selftest pins this).
The item text's own sha is still recorded as post_sha256 for auditing.
The scaffold name ``tmg_solo`` and role id ``solo_analyst`` are NEW: at
introduction the selftest scanned divergence_study_outputs and found zero
files carrying either name (caches are keyed by name, not content). The
same check keeps running after pilots exist, as "no FOREIGN file carries
these names": every file carrying them must match this runner's own
call_cache_path shape exactly (either scaffold). The nb scaffold
``tmg_solo_nb`` was introduced the same way (16.13c): zero files carried it
at introduction, and the same no-FOREIGN check covers it (``tmg_solo`` is a
substring of ``tmg_solo_nb``, so one scan sees both). Row CSVs and the
analysis JSON go to ``{tag}_rows.csv`` / ``{tag}_analysis.json`` with tag
``game_solo`` / ``game_solo_masked`` (``game_solo_nb`` /
``game_solo_nb_masked`` under --regime nonbinding; deliberately not
``tmg_``-prefixed: tmg_games' selftest asserts no ``tmg_``-prefixed file
exists in OUT_DIR, and main() refuses any --tag starting with ``tmg_`` for
the same reason); the selftest asserts every file under those tags is one
of this runner's own outputs.

SPEND GATING

Nothing here generates unless the literal flag ``--run`` is passed.
``--dry-run`` prints the call count (n_games x samples) and a price from
PRICES (and refuses, exit 2, to report $0 for a model PRICES does not
list); ``--run`` refuses an unpriced model the same way before any call;
``--analyze`` reads an existing ``{tag}_rows.csv``; ``--selftest`` runs
fully offline against a stub generator and never passes ``--run``.

A view whose call raises (rate limit, 5xx, JSON error) is a guard failure,
not an omission: run() returns the failed (game_id, sample_idx) keys and
main() prints GUARD FAILED naming them and exits 4 without writing
``{tag}_rows.csv`` (the survivors go to ``{tag}_GUARD_FAILED_rows.csv``
for inspection; the completed calls are cached and a re-run picks them up
at no extra spend). Otherwise a masked baseline on which the model refused
the hard items would look clean on the survivors.

WHAT --analyze REPORTS (the registered read-outs this runner can test)

  accuracy vs gold overall, per gold_type (the extent chart) and per
  region (value / neutral / cannot_assist), each with an item-clustered
  bootstrap CI; NOVERDICT and truncation shares through parse_guard, which
  prints GUARD FAILED above 5% on either; Nash-pull = P(a PLAN_xy verdict
  names a pure Nash cell), by gold_type, the quantity the deliberation
  analyzer will compare per seat; the masked floor check when the rows are
  masked. Written to ``{tag}_analysis.json``.

Usage:
  python -m scripts.run_game_singleagent --selftest
  python -m scripts.run_game_singleagent --dry-run [--samples 3] [--games value]
  python -m scripts.run_game_singleagent --dry-run --masked
  python -m scripts.run_game_singleagent --dry-run --regime nonbinding --samples 3
  python -m scripts.run_game_singleagent --run --samples 1        # SPENDS MONEY
  python -m scripts.run_game_singleagent --run --masked           # SPENDS MONEY
  python -m scripts.run_game_singleagent --run --regime nonbinding --samples 3   # SPENDS MONEY
  python -m scripts.run_game_singleagent --analyze [--tag game_solo]
  python -m scripts.run_game_singleagent --analyze --regime nonbinding [--tag game_solo_nb]

Exit codes: 0 ok, 1 no rows, 2 bad config / refused / data unavailable /
unpriced model / 'tmg_' tag / rows whose regime contradicts --regime /
--run under the other regime's default tag or over an existing rows CSV of
the other regime (run_target_refusal, checked before any call),
4 guard failed (parse, truncation or errored views).
"""
from __future__ import annotations

import argparse
import csv
import inspect
import io
import json
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.run_phase1_quartet import OUT_DIR
from scripts.run_crowdgold_aita import (
    MAX_NOVERDICT_SHARE as AITA_MAX_NOVERDICT_SHARE,
    MAX_TRUNCATION_SHARE as AITA_MAX_TRUNCATION_SHARE,
    count_tokens,
    _optional_bpe_counter,
)
from scripts.run_crowdgold_deliberation import PRICES, call_cache_path, do_call
from scripts.tmg_games import (
    GAME_VERDICT_INSTRUCTION,
    GOLD_TYPES,
    MAX_NOVERDICT_SHARE,
    MAX_TRUNCATION_SHARE,
    N_GAMES,
    NO_UNIQUE_PLAN,
    PLAN_VERDICTS,
    REGION_CANNOT_ASSIST,
    REGION_NEUTRAL,
    REGION_OF,
    REGION_VALUE,
    REGIONS,
    REGISTERED_REGION_COUNTS,
    VERDICT_SET,
    Game,
    ItemView,
    _boot_ci,
    cache_files_with_prefix,
    forbidden_hits,
    load_games,
    parse_guard,
    print_masked_floor_report,
    print_parse_guard,
    render_item,
)
from scripts.tmg_games_nb import (
    NB_VERDICT_INSTRUCTION,
    NB_VERDICT_SET,
    NO_RECOMMENDATION,
    REGIME_BINDING,
    REGIME_NONBINDING,
    REGIMES,
    ItemViewNB,
    masked_floor_report_of,
    no_answer_token_of,
    parse_verdict_for,
    regime_of,
    render_item_nb,
    score_for,
    score_of,
    verdict_instruction_for,
)
from scripts.verdict_format import NOVERDICT

# ---------------------------------------------------------------------------
# Names. All NEW; the selftest asserts no cache file in OUT_DIR carries any
# of them (caches are keyed by name, not content).
# ---------------------------------------------------------------------------

SCAFFOLD_NAME = "tmg_solo"
#: --regime nonbinding (16.13c): its own scaffold, so a non-binding record
#: can never be served from a binding cache path or the reverse.
SCAFFOLD_NAME_NB = "tmg_solo_nb"
SCAFFOLD_OF = {REGIME_BINDING: SCAFFOLD_NAME, REGIME_NONBINDING: SCAFFOLD_NAME_NB}
ARM_FULL = "full"
ARM_MASKED = "masked"
ROUND_NAME = "single"
ROLE_ID = "solo_analyst"
#: Row/analysis file tag. Deliberately NOT 'tmg_'-prefixed: tmg_games'
#: selftest asserts that no 'tmg_'-prefixed file exists in OUT_DIR, and this
#: runner's outputs must not break that once a pilot has been run.
DEFAULT_TAG = "game_solo"
DEFAULT_TAG_MASKED = "game_solo_masked"
DEFAULT_TAG_NB = "game_solo_nb"
DEFAULT_TAG_NB_MASKED = "game_solo_nb_masked"
DEFAULT_TAGS = (DEFAULT_TAG, DEFAULT_TAG_MASKED, DEFAULT_TAG_NB, DEFAULT_TAG_NB_MASKED)
#: Substrings that no FOREIGN cache filename may carry. At introduction the
#: selftest found zero files with either; afterwards every file carrying them
#: must match this runner's own call_cache_path shape (_OWN_CACHE_RE, either
#: scaffold). 'tmg_solo' is a substring of 'tmg_solo_nb', so the one scan
#: covers the nb scaffold too (zero files carried it at ITS introduction).
NEW_CACHE_NAMES = (SCAFFOLD_NAME, ROLE_ID)
_OWN_CACHE_RE = re.compile(
    rf"^cgd_.+_(?:{SCAFFOLD_NAME}|{SCAFFOLD_NAME_NB})_t\d+_"
    rf"(?:{ARM_FULL}|{ARM_MASKED})_\d{{3}}_\d{{2}}_{ROUND_NAME}_{ROLE_ID}\.json$")
_OWN_OUTPUT_SUFFIXES = ("_rows.csv", "_analysis.json")

GAME_SELECTIONS = ("all",) + REGIONS

SOLO_SYSTEM = (
    "You are a careful, impartial analyst answering a single direct question "
    "about a two-player game described by a payoff table. You represent "
    "neither player. Use only what the table and the question say; do not "
    "assume anything about the players beyond what is stated. Work through "
    "the table as briefly as you need, then end with the required verdict "
    "line."
)


def default_tag(masked: bool, regime: str = REGIME_BINDING) -> str:
    if regime == REGIME_NONBINDING:
        return DEFAULT_TAG_NB_MASKED if masked else DEFAULT_TAG_NB
    return DEFAULT_TAG_MASKED if masked else DEFAULT_TAG


def arm_for(masked: bool) -> str:
    return ARM_MASKED if masked else ARM_FULL


def scaffold_for(regime: str) -> str:
    """The cache scaffold of a regime (SCAFFOLD_OF); refuses anything else."""
    try:
        return SCAFFOLD_OF[regime]
    except KeyError:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}") from None


# ---------------------------------------------------------------------------
# Item adapter: do_call reads item.item_id / item.gold_verdict / item.post_text
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameItem:
    """One rendered view, duck-typed for do_call. ``post_text`` is the exact
    model-facing text, so the cache's parent-sha check invalidates a cached
    call whenever the rendering changes."""

    view: ItemView

    @property
    def item_id(self) -> str:
        return self.view.game_id

    @property
    def gold_verdict(self) -> str:
        return self.view.gold_verdict

    @property
    def post_text(self) -> str:
        return self.view.text


def select_games(games: Sequence[Game], which: str) -> list[Game]:
    if which == "all":
        return list(games)
    if which not in REGIONS:
        raise ValueError(f"unknown game selection {which!r}; "
                         f"expected one of {GAME_SELECTIONS}")
    return [g for g in games if g.region == which]


def render_views(games: Sequence[Game], samples: int, *, masked: bool,
                 regime: str = REGIME_BINDING) -> list[ItemView]:
    """Binding: tmg_games.render_item (a plain ItemView). Non-binding:
    tmg_games_nb.render_item_nb (an ItemViewNB carrying its regime)."""
    if regime == REGIME_BINDING:
        return [render_item(g, i, masked=masked) for g in games for i in range(samples)]
    if regime == REGIME_NONBINDING:
        return [render_item_nb(g, i, masked=masked) for g in games for i in range(samples)]
    raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def solo_user(view: ItemView) -> str:
    """ItemView.text (scenario + question + legend) and the forced verdict
    line, LAST. Nothing else from the view ever reaches a prompt. The
    instruction line is the view's regime's (GAME_VERDICT_INSTRUCTION for a
    binding view, NB_VERDICT_INSTRUCTION for an ItemViewNB)."""
    return (
        f"{view.text}\n\n"
        "Answer on your own, from the table alone, with no deliberation and "
        "no other input.\n\n"
        f"{verdict_instruction_for(view)}"
    )


def prompt_pair(view: ItemView) -> tuple[str, str]:
    return SOLO_SYSTEM, solo_user(view)


def assert_prompt_clean(system: str, user: str) -> None:
    """Refuse to send a prompt in which a forbidden word occurs."""
    hits = forbidden_hits(system) + forbidden_hits(user)
    if hits:
        raise RuntimeError(f"forbidden word(s) in prompt, refusing to send: {hits}")


# ---------------------------------------------------------------------------
# One call
# ---------------------------------------------------------------------------

def _perm_str(p: tuple[int, int]) -> str:
    return f"{p[0]}{p[1]}"


def row_from_view(view: ItemView, model: str, rec: Mapping[str, Any]) -> dict:
    """A row of ROW_FIELDS from a call record (cached or fresh)."""
    text = rec.get("output") or ""
    verdict = parse_verdict_for(view, text)
    sc = score_for(view, verdict, view.gold_verdict)
    is_nash = view.is_nash(verdict)
    fr = rec.get("finish_reason", "") or ""
    return {
        "model": model, "scaffold": scaffold_for(regime_of(view)),
        "arm": arm_for(view.masked),
        "item_id": view.game_id, "sample_idx": view.sample_idx,
        "masked": int(view.masked),
        "gold_type": view.gold_type, "region": view.region,
        "gold_verdict": view.gold_verdict, "verdict": verdict,
        "correct": int(sc["correct"]), "noncommittal": int(sc["noncommittal"]),
        "is_nash": "" if is_nash is None else int(is_nash),
        "nash_verdicts": "|".join(sorted(view.nash_verdicts)),
        "row_perm": _perm_str(view.row_perm), "col_perm": _perm_str(view.col_perm),
        "row_name": view.row_name, "col_name": view.col_name,
        "row_best_verdict": view.row_best_verdict,
        "col_best_verdict": view.col_best_verdict,
        "finish_reason": fr,
        "truncated": int(fr in ("max_tokens", "length", "MAX_TOKENS")),
        "output_len": len(text),
        "prompt_tokens": rec.get("prompt_tokens", 0),
        "completion_tokens": rec.get("completion_tokens", 0),
    }


def run_one(model: str, view: ItemView, *, max_tokens: int) -> dict:
    system, user = prompt_pair(view)
    assert_prompt_clean(system, user)
    item = GameItem(view)
    # parents = the FULL model-facing text: a system-prompt or instruction
    # edit invalidates the cache (module docstring, CACHE NAMESPACE). The
    # item text is inside ``user``; do_call records its sha as post_sha256.
    rec = do_call(
        model=model, scaffold=scaffold_for(regime_of(view)), arm=arm_for(view.masked),
        item=item, idx=view.sample_idx, round_name=ROUND_NAME, role_id=ROLE_ID,
        system=system, user=user, max_tokens=max_tokens,
        parents=(system, user), cap=0,
    )
    return row_from_view(view, model, rec)


ROW_FIELDS = (
    "model", "scaffold", "arm", "item_id", "sample_idx", "masked",
    "gold_type", "region", "gold_verdict", "verdict", "correct",
    "noncommittal", "is_nash", "nash_verdicts", "row_perm", "col_perm",
    "row_name", "col_name", "row_best_verdict", "col_best_verdict",
    "finish_reason", "truncated", "output_len", "prompt_tokens",
    "completion_tokens",
)


# ---------------------------------------------------------------------------
# Cost model (dry run). The completion length is an ASSUMPTION: this is a
# new call shape with no measured completions yet.
# ---------------------------------------------------------------------------

#: A brief worked answer plus the verdict line. Reasoning models bill their
#: hidden reasoning as completion tokens too, so treat this as a floor.
ASSUMED_COMPLETION = 350


def cost_model(model: str, views: Sequence[ItemView], *, max_tokens: int) -> dict:
    if not views:
        raise ValueError("cost model needs at least one view")
    regimes = {regime_of(v) for v in views}
    if len(regimes) != 1:
        raise ValueError(f"cost model needs views of one regime, got {sorted(regimes)}")
    regime = regimes.pop()
    scaffold = scaffold_for(regime)
    bpe = _optional_bpe_counter()
    probe = sorted(views, key=lambda v: len(v.text))[len(views) // 2]
    system, user = prompt_pair(probe)
    ptok = (count_tokens(system, tokenizer=bpe) + count_tokens(user, tokenizer=bpe)
            if bpe else count_tokens(system) + count_tokens(user))
    n_calls = len(views)
    priced = model in PRICES
    pin, pout = PRICES.get(model, (0.0, 0.0))
    usd = (ptok * n_calls) / 1e6 * pin + (ASSUMED_COMPLETION * n_calls) / 1e6 * pout
    n_cached = sum(
        1 for v in views
        if call_cache_path(model, scaffold, arm_for(v.masked), v.game_id,
                           v.sample_idx, ROUND_NAME, ROLE_ID, max_tokens).exists())
    n_items = len({v.game_id for v in views})
    return {
        "model": model, "priced": priced, "arm": arm_for(probe.masked),
        "regime": regime, "scaffold": scaffold,
        "n_items": n_items, "samples": n_calls // max(1, n_items),
        "n_calls": n_calls, "n_cached": n_cached,
        "prompt_tokens_per_call": ptok, "assumed_completion": ASSUMED_COMPLETION,
        "max_tokens": max_tokens, "total_usd": round(usd, 4),
        "tokenizer": "tiktoken cl100k_base BPE" if bpe else "regex approximation",
    }


def print_cost_model(cm: dict) -> None:
    print("\n" + "=" * 72)
    # (the regime suffix appears under nonbinding only: the binding header is
    # byte-identical to the runner before the flag existed)
    nb = (f", regime {cm['regime']} (scaffold {cm['scaffold']})"
          if cm.get("regime") == REGIME_NONBINDING else "")
    print(f"COST MODEL (dry run) -- {cm['model']} solo analyst, arm {cm['arm']}, "
          f"max_tokens {cm['max_tokens']}{nb}")
    print("=" * 72)
    print(f"  {cm['n_items']} games x {cm['samples']} samples = {cm['n_calls']} calls"
          f"  ({cm['n_cached']} already cached at this cap; a cached call is "
          f"free unless its parent text changed)")
    print(f"  tokenizer: {cm['tokenizer']}")
    print(f"  prompt tokens/call: {cm['prompt_tokens_per_call']}  "
          f"assumed completion: {cm['assumed_completion']} (unmeasured; a "
          f"reasoning model may bill several times this)")
    print(f"  TOTAL: ${cm['total_usd']:.2f}")
    if not cm["priced"]:
        print(f"\n  *** {cm['model']} is NOT in PRICES: the $0 above is a "
              "placeholder, not an estimate. Add a price before spending. ***")


# ---------------------------------------------------------------------------
# Analysis: accuracy by gold_type / region, Nash-pull, guard, masked floor
# ---------------------------------------------------------------------------

def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in ("1", "true")


#: The JSON key of the no-answer share, per regime. Binding keeps the key
#: every existing analysis file has; nonbinding names its own token.
NO_ANSWER_KEY = {REGIME_BINDING: "no_unique_share",
                 REGIME_NONBINDING: "no_recommendation_share"}


def regime_of_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    """The regime a set of rows was produced under, read off the scaffold
    column (caches and rows are keyed by that name). Rows without a
    scaffold are binding (every file written before the flag existed).
    Refuses (ValueError) rows that mix scaffolds: one run, one regime."""
    scaffolds = {str(r.get("scaffold") or SCAFFOLD_NAME) for r in rows}
    if scaffolds <= {SCAFFOLD_NAME}:
        return REGIME_BINDING
    if scaffolds == {SCAFFOLD_NAME_NB}:
        return REGIME_NONBINDING
    raise ValueError(f"rows mix scaffolds / regimes: {sorted(scaffolds)}")


def _norm_row(r: Mapping[str, Any], regime: str = REGIME_BINDING) -> dict:
    verdict = str(r.get("verdict") or NOVERDICT)
    gold = str(r["gold_verdict"])
    sc = score_of(regime, verdict, gold)
    gt = str(r.get("gold_type") or "")
    region = str(r.get("region") or REGION_OF.get(gt, ""))
    plan = verdict in PLAN_VERDICTS
    return {
        "item": str(r["item_id"]), "gold_type": gt, "region": region,
        "verdict": verdict, "gold": gold,
        "correct": int(sc["correct"]),
        "noverdict": int(verdict == NOVERDICT),
        "no_answer": int(verdict == no_answer_token_of(regime)),
        "plan": int(plan),
        "nash": int(plan and _truthy(r.get("is_nash"))),
        "masked": int(_truthy(r.get("masked", 0))),
    }


def _group_stats(recs: Sequence[Mapping[str, Any]], *, draws: int, seed: int,
                 no_answer_key: str = "no_unique_share") -> dict:
    if not recs:
        return {"n": 0, "n_items": 0, "accuracy": None, "lo": None, "hi": None,
                "noverdict_share": None, no_answer_key: None, "n_plan": 0,
                "n_nash": 0, "nash_pull": None, "nash_pull_lo": None,
                "nash_pull_hi": None, "nash_share_committal": None,
                "verdict_counts": {}}
    by_item: dict[str, list] = defaultdict(list)
    for r in recs:
        by_item[r["item"]].append(r)

    def acc(rs: list) -> Optional[float]:
        return sum(r["correct"] for r in rs) / len(rs) if rs else None

    def pull(rs: list) -> Optional[float]:
        p = [r for r in rs if r["plan"]]
        return sum(r["nash"] for r in p) / len(p) if p else None

    a, alo, ahi = _boot_ci(by_item, acc, draws=draws, seed=seed)
    p, plo, phi = _boot_ci(by_item, pull, draws=draws, seed=seed)
    n = len(recs)
    n_plan = sum(r["plan"] for r in recs)
    n_committal = n - sum(r["noverdict"] for r in recs)
    n_nash = sum(r["nash"] for r in recs)
    return {
        "n": n, "n_items": len(by_item),
        "accuracy": a, "lo": alo, "hi": ahi,
        "noverdict_share": (sum(r["noverdict"] for r in recs) / n) if n else None,
        no_answer_key: (sum(r["no_answer"] for r in recs) / n) if n else None,
        "n_plan": n_plan, "n_nash": n_nash,
        "nash_pull": p, "nash_pull_lo": plo, "nash_pull_hi": phi,
        "nash_share_committal": (n_nash / n_committal) if n_committal else None,
        "verdict_counts": dict(Counter(r["verdict"] for r in recs)),
    }


def analyze_rows(rows: Sequence[Mapping[str, Any]], *, draws: int = 2000,
                 seed: int = 7) -> dict:
    """The registered read-outs on one run's rows (in-memory or CSV-loaded).

    Nash-pull: among PLAN_xy verdicts, the share naming a pure Nash cell of
    the game (denominator excludes the no-answer token and NOVERDICT);
    ``nash_share_committal`` puts the no-answer token back in the
    denominator. Every CI is an item-clustered bootstrap over game_id.

    The regime is read off the rows' scaffold column (regime_of_rows):
    binding rows score on tmg_games' instrument with NO_UNIQUE_PLAN as the
    no-answer token and the JSON of every existing file; nonbinding rows on
    tmg_games_nb's with NO_RECOMMENDATION, the key ``no_recommendation_share``
    in place of ``no_unique_share``, and a top-level ``regime`` key.
    """
    regime = regime_of_rows(rows)
    key = NO_ANSWER_KEY[regime]
    recs = [_norm_row(r, regime) for r in rows]
    rep: dict[str, Any] = {
        "n_rows": len(recs), "n_items": len({r["item"] for r in recs}),
        "models": sorted({str(r.get("model", "")) for r in rows}),
        "arms": sorted({str(r.get("arm", "")) for r in rows}),
        "guard": parse_guard(rows),
        "overall": _group_stats(recs, draws=draws, seed=seed, no_answer_key=key),
        "by_gold_type": {
            gt: _group_stats([r for r in recs if r["gold_type"] == gt],
                             draws=draws, seed=seed, no_answer_key=key)
            for gt in GOLD_TYPES if any(r["gold_type"] == gt for r in recs)},
        "by_region": {
            rg: _group_stats([r for r in recs if r["region"] == rg],
                             draws=draws, seed=seed, no_answer_key=key)
            for rg in REGIONS if any(r["region"] == rg for r in recs)},
        "masked": bool(recs) and all(r["masked"] for r in recs),
    }
    if regime == REGIME_NONBINDING:
        rep["regime"] = regime
    if rep["masked"]:
        rep["masked_floor"] = masked_floor_report_of(regime, rows, draws=draws,
                                                     seed=seed)
    return rep


def regime_of_report(rep: Mapping[str, Any]) -> str:
    """A report without a ``regime`` key is binding (module docstring)."""
    return str(rep.get("regime") or REGIME_BINDING)


def _fmt(x: Optional[float], spec: str = ".3f") -> str:
    return "  n/a" if x is None else format(x, spec)


#: Column header of the no-answer share, per regime.
_NO_ANSWER_COL = {REGIME_BINDING: "NOUNQ", REGIME_NONBINDING: "NOREC"}


def _print_group_table(title: str, groups: Mapping[str, Mapping[str, Any]],
                       *, regime: str = REGIME_BINDING) -> None:
    key, col = NO_ANSWER_KEY[regime], _NO_ANSWER_COL[regime]
    print(f"\n  {title}")
    print(f"  {'group':<32}{'n':>5}{'items':>6}{'acc':>7}{'lo':>7}{'hi':>7}"
          f"{'NOVER':>7}{col:>7}{'nPLAN':>6}{'Nash':>7}")
    for k, g in groups.items():
        print(f"  {k:<32}{g['n']:>5}{g['n_items']:>6}{_fmt(g['accuracy']):>7}"
              f"{_fmt(g['lo']):>7}{_fmt(g['hi']):>7}"
              f"{_fmt(g['noverdict_share']):>7}{_fmt(g[key]):>7}"
              f"{g['n_plan']:>6}{_fmt(g['nash_pull']):>7}")


def print_analysis(rep: Mapping[str, Any], *, label: str = "") -> None:
    regime = regime_of_report(rep)
    key, col = NO_ANSWER_KEY[regime], _NO_ANSWER_COL[regime]
    no_answer = no_answer_token_of(regime)
    # (the regime tag appears under nonbinding only: the binding print is
    # byte-identical to the runner before the flag existed)
    title_regime = " (NON-BINDING regime)" if regime == REGIME_NONBINDING else ""
    print("\n" + "=" * 72)
    print(f"SOLO ANALYST -- accuracy vs gold, Nash-pull{title_regime}"
          f"{(' -- ' + label) if label else ''}")
    print("=" * 72)
    o = rep["overall"]
    print(f"  models {rep['models']}  arms {rep['arms']}  rows {rep['n_rows']}  "
          f"items {rep['n_items']}")
    print(f"  accuracy {_fmt(o['accuracy'])} [{_fmt(o['lo'])}, {_fmt(o['hi'])}]  "
          f"NOVERDICT {_fmt(o['noverdict_share'])}  {no_answer} "
          f"{_fmt(o[key])}")
    print(f"  Nash-pull (PLAN verdicts naming a pure Nash cell) "
          f"{_fmt(o['nash_pull'])} [{_fmt(o['nash_pull_lo'])}, "
          f"{_fmt(o['nash_pull_hi'])}]  n_plan {o['n_plan']}  "
          f"(committal denominator: {_fmt(o['nash_share_committal'])})")
    print(f"  verdicts: {o['verdict_counts']}")
    _print_group_table("by gold_type (the extent chart)", rep["by_gold_type"],
                       regime=regime)
    _print_group_table("by region", rep["by_region"], regime=regime)
    print("\n  acc/lo/hi: item-clustered bootstrap; NOVER: NOVERDICT share; "
          f"{col}: {no_answer} share; Nash: Nash-pull over the nPLAN "
          "PLAN_xy verdicts")
    print_parse_guard(rep["guard"])
    if rep.get("masked_floor") is not None:
        print_masked_floor_report(rep["masked_floor"])


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run(model: str, views: Sequence[ItemView], *, max_tokens: int,
        workers: int) -> tuple[list[dict], list[tuple[str, int]]]:
    """Returns (rows, failed): ``failed`` lists the (game_id, sample_idx)
    whose call raised. An errored view is a guard failure for main(), never
    a silent omission (module docstring, SPEND GATING)."""
    print(f"  {len(views)} calls ({len({v.game_id for v in views})} games x "
          f"{len(views) // max(1, len({v.game_id for v in views}))} samples)")
    rows: list[dict] = []
    failed: list[tuple[str, int]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(run_one, model, v, max_tokens=max_tokens):
                (v.game_id, v.sample_idx) for v in views}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                rows.append(fut.result())
                done += 1
                if done % 40 == 0:
                    print(f"    {done}/{len(views)} calls", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR {key}: {e}", flush=True)
                failed.append(key)
    rows.sort(key=lambda r: (r["item_id"], r["sample_idx"], r["arm"]))
    failed.sort()
    return rows, failed


def errored_views_report(failed: Sequence[tuple[str, int]], n_views: int) -> bool:
    """Print the errored-views guard; True when it passes (nothing failed)."""
    if not failed:
        return True
    print(f"\n  *** GUARD FAILED: {len(failed)} of {n_views} views errored "
          f"(call raised; dropped from rows). Failed keys: "
          f"{', '.join(f'{k[0]}/{k[1]}' for k in failed)}")
    print("  The completed calls are cached; re-run to pick them up at no "
          "extra spend. No analysis is printed and no rows CSV is written "
          "under the run tag.")
    return False


def write_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROW_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ROW_FIELDS})


#: The other regime's default tags, per regime: --run may never write under
#: them (it would replace that regime's rows CSV and analysis JSON).
OTHER_REGIME_TAGS = {REGIME_BINDING: (DEFAULT_TAG_NB, DEFAULT_TAG_NB_MASKED),
                     REGIME_NONBINDING: (DEFAULT_TAG, DEFAULT_TAG_MASKED)}


def run_target_refusal(tag: str, regime: str, rows_path: Path) -> Optional[str]:
    """Why --run may NOT write ``{tag}_rows.csv`` / ``{tag}_analysis.json``
    under ``regime``, or None when it may: the tag is a default tag of the
    OTHER regime, or an existing rows CSV under the tag holds rows of another
    regime (its scaffold column, regime_of_rows). Mirrors
    run_crowdgold_game_nb's binding-tag refusal and existing-rows scaffold
    check; called by main() BEFORE any call is made."""
    scaffold_for(regime)  # refuses an unknown regime
    other = OTHER_REGIME_TAGS[regime]
    if tag in other:
        other_regime = REGIME_NONBINDING if regime == REGIME_BINDING else REGIME_BINDING
        return (f"--tag {tag!r} is a default tag of the {other_regime} regime; a "
                f"{regime} run would overwrite that regime's rows CSV and analysis "
                f"JSON. Use the default tag ({default_tag(tag.endswith('_masked'), regime)}) "
                "or another tag.")
    if rows_path.exists():
        rows = read_rows(rows_path)
        if rows:
            try:
                rows_regime = regime_of_rows(rows)
            except ValueError as e:
                return f"{rows_path} cannot be overwritten: {e}"
            if rows_regime != regime:
                return (f"{rows_path} holds {rows_regime} rows (scaffold "
                        f"{scaffold_for(rows_regime)}); refusing to overwrite them with a "
                        f"{regime} run (scaffold {scaffold_for(regime)}). Pick another --tag.")
    return None


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cache_files_containing(subs: Sequence[str], out_dir: Path = OUT_DIR) -> list[Path]:
    """Existing files whose NAME contains any of ``subs`` (one directory scan)."""
    if not out_dir.is_dir():
        return []
    return sorted(p for p in out_dir.iterdir() if any(s in p.name for s in subs))


# ---------------------------------------------------------------------------
# Selftest (offline; no API, no network, no spend, never passes --run)
# ---------------------------------------------------------------------------

_DIGIT_CELL_RE = re.compile(r"[1-4]\s*\\\s*[1-4]")
_MASKED_CELL_RE = re.compile(r"\?\s*\\\s*\?")
N_SELFTEST_SAMPLES = 3


def _selftest() -> int:
    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # ---- names are new (zero files at introduction; afterwards only this
    # ---- runner's own-shaped files may carry them) --------------------------
    carrying = cache_files_containing(NEW_CACHE_NAMES)
    foreign = [p for p in carrying if not _OWN_CACHE_RE.match(p.name)]
    check(f"no FOREIGN cache file carries scaffold {SCAFFOLD_NAME!r} or role "
          f"{ROLE_ID!r} ({len(carrying)} files carry them, {len(foreign)} foreign)"
          + (f" -- first foreign {foreign[0].name}" if foreign else ""),
          not foreign)
    for tag in (DEFAULT_TAG, DEFAULT_TAG_MASKED):
        tagged = cache_files_with_prefix(tag)
        odd = [p for p in tagged if not p.name.endswith(_OWN_OUTPUT_SUFFIXES)]
        check(f"files starting with tag {tag!r}: {len(tagged)} found, all this "
              f"runner's own _rows.csv / _analysis.json ({len(odd)} foreign)",
              not odd)
    check("own-cache pattern matches call_cache_path's shape for both arms and "
          "a dotted model name",
          all(_OWN_CACHE_RE.match(call_cache_path(m, SCAFFOLD_NAME, a, "007", 3,
                                                  ROUND_NAME, ROLE_ID, 1024).name)
              for m in ("gpt-5.4-nano", "claude-haiku-4-5")
              for a in (ARM_FULL, ARM_MASKED))
          and not _OWN_CACHE_RE.match("cgd_m_tmg_solo_t1024_full_007_03_r0_"
                                      "solo_analyst.json"))
    cp_full = call_cache_path("m", SCAFFOLD_NAME, ARM_FULL, "001", 0, ROUND_NAME,
                              ROLE_ID, 100)
    cp_mask = call_cache_path("m", SCAFFOLD_NAME, ARM_MASKED, "001", 0, ROUND_NAME,
                              ROLE_ID, 100)
    check("cache path carries the new scaffold and role id and is distinct "
          "from every Dilemmas / AITA scaffold",
          SCAFFOLD_NAME in cp_full.name and ROLE_ID in cp_full.name
          and "dilemma" not in cp_full.name and "deliberation" not in cp_full.name)
    check("masked and full arms cache to different paths", cp_full != cp_mask)
    check("ROW_FIELDS has no duplicate columns", len(ROW_FIELDS) == len(set(ROW_FIELDS)))
    check("guard thresholds equal run_crowdgold_aita's (0.05 / 0.05)",
          MAX_TRUNCATION_SHARE == AITA_MAX_TRUNCATION_SHARE == 0.05
          and MAX_NOVERDICT_SHARE == AITA_MAX_NOVERDICT_SHARE == 0.05)

    # ---- --regime nonbinding names (16.13c): own scaffold and tags -----------
    nb_carrying = cache_files_containing((SCAFFOLD_NAME_NB,))
    nb_foreign = [p for p in nb_carrying if not _OWN_CACHE_RE.match(p.name)]
    check(f"nb scaffold {SCAFFOLD_NAME_NB!r}: {len(nb_carrying)} cache files carry "
          f"it (0 at introduction), {len(nb_foreign)} foreign"
          + (f" -- first foreign {nb_foreign[0].name}" if nb_foreign else ""),
          not nb_foreign)
    check("cache_files_with_prefix finds nothing under the nb scaffold or the nb "
          "tags as a cache prefix (the substring scan above is the real check; "
          "tag files are asserted own-shaped below)",
          cache_files_with_prefix(SCAFFOLD_NAME_NB) == []
          and all(not p.name.endswith(".json") or p.name.endswith("_analysis.json")
                  for p in cache_files_with_prefix(DEFAULT_TAG_NB)))
    for tag in (DEFAULT_TAG_NB, DEFAULT_TAG_NB_MASKED):
        tagged = cache_files_with_prefix(tag)
        odd = [p for p in tagged if not p.name.endswith(_OWN_OUTPUT_SUFFIXES)]
        check(f"files starting with nb tag {tag!r}: {len(tagged)} found, all this "
              f"runner's own _rows.csv / _analysis.json ({len(odd)} foreign)",
              not odd)
    check("nb names are distinct from the binding ones, not 'tmg_'-tag-shaped "
          "outputs, free of forbidden words; SCAFFOLD_OF covers both regimes",
          SCAFFOLD_NAME_NB != SCAFFOLD_NAME and DEFAULT_TAG_NB not in (DEFAULT_TAG, DEFAULT_TAG_MASKED)
          and DEFAULT_TAG_NB_MASKED not in (DEFAULT_TAG, DEFAULT_TAG_MASKED, DEFAULT_TAG_NB)
          and not any(t.startswith("tmg_") for t in DEFAULT_TAGS)
          and not any(forbidden_hits(n) for n in DEFAULT_TAGS + (SCAFFOLD_NAME_NB,))
          and set(SCAFFOLD_OF) == set(REGIMES) and scaffold_for(REGIME_NONBINDING) == SCAFFOLD_NAME_NB
          and scaffold_for(REGIME_BINDING) == SCAFFOLD_NAME and _raises(scaffold_for, "bogus")
          and default_tag(False) == DEFAULT_TAG and default_tag(True) == DEFAULT_TAG_MASKED
          and default_tag(False, REGIME_NONBINDING) == DEFAULT_TAG_NB
          and default_tag(True, REGIME_NONBINDING) == DEFAULT_TAG_NB_MASKED)
    cp_nb = call_cache_path("m", SCAFFOLD_NAME_NB, ARM_FULL, "001", 0, ROUND_NAME,
                            ROLE_ID, 100)
    check("nb cache path is own-shaped, carries tmg_solo_nb and differs from the "
          "binding path of the same (model, arm, game, idx, cap)",
          _OWN_CACHE_RE.match(cp_nb.name) and SCAFFOLD_NAME_NB in cp_nb.name
          and cp_nb != cp_full
          and all(_OWN_CACHE_RE.match(call_cache_path(m, SCAFFOLD_NAME_NB, a, "007", 3,
                                                      ROUND_NAME, ROLE_ID, 1024).name)
                  for m in ("gpt-5.4-nano", "claude-haiku-4-5")
                  for a in (ARM_FULL, ARM_MASKED)))

    # ---- data ----------------------------------------------------------------
    try:
        games = load_games()
    except Exception as e:  # noqa: BLE001
        check(f"load_games() ({type(e).__name__}: {e})", False)
        print(f"\n{len(fails)} FAILED")
        return 1
    check(f"load_games returns {N_GAMES} games", len(games) == N_GAMES)
    check("select_games: all / value / neutral / cannot_assist = 144 / 16 / 97 / 31",
          len(select_games(games, "all")) == N_GAMES
          and all(len(select_games(games, r)) == REGISTERED_REGION_COUNTS[r]
                  for r in REGIONS))
    check("select_games rejects an unknown selection",
          _raises(select_games, games, "bogus"))

    # ---- prompt rendering, forbidden words, masked ---------------------------
    full = render_views(games, N_SELFTEST_SAMPLES, masked=False)
    masked = render_views(games, N_SELFTEST_SAMPLES, masked=True)
    check(f"render_views: {N_GAMES} games x {N_SELFTEST_SAMPLES} samples = "
          f"{len(full)} views, masked twin same size",
          len(full) == N_GAMES * N_SELFTEST_SAMPLES and len(masked) == len(full))
    check("system prompt is free of forbidden words", not forbidden_hits(SOLO_SYSTEM))
    leaks = []
    tail_bad = []
    meta_bad = []
    for v in full + masked:
        system, user = prompt_pair(v)
        if forbidden_hits(system) or forbidden_hits(user):
            leaks.append((v.game_id, v.sample_idx, v.masked))
        if not user.rstrip().endswith(GAME_VERDICT_INSTRUCTION):
            tail_bad.append((v.game_id, v.sample_idx))
        low = user.lower()
        if (v.gold_type.lower() in low or "gold" in low or "nash" in low
                or "canonical" in low or v.region in low.split()):
            meta_bad.append((v.game_id, v.sample_idx))
    check(f"no forbidden word in any of the {len(full) + len(masked)} rendered "
          f"system+user prompts" + (f" -- first {leaks[0]}" if leaks else ""),
          not leaks)
    check("GAME_VERDICT_INSTRUCTION is the last thing in every user prompt",
          not tail_bad)
    check("no gold_type / gold / nash / region metadata in any user prompt",
          not meta_bad)
    check("user prompt contains the item text and the 'no deliberation' clause",
          all(v.text in solo_user(v) and "no deliberation" in solo_user(v)
              for v in full[:20]))
    check("every user prompt carries all five verdict tokens",
          all(all(f"VERDICT: {t}" in solo_user(v) for t in VERDICT_SET)
              for v in full[:20] + masked[:20]))
    check("assert_prompt_clean passes every rendered prompt and refuses a "
          "planted forbidden word",
          all(_ok(assert_prompt_clean, *prompt_pair(v)) for v in full[:20])
          and _raises_rt(assert_prompt_clean, SOLO_SYSTEM,
                         solo_user(full[0]) + "\nthey could betray each other"))
    # (the masked scenario sentence itself quotes "? \ ?", so the four-cell
    # count is taken on the table; the absence checks run over the whole
    # user prompt)
    check("masked prompts contain no payoff digit cell anywhere and exactly four "
          "'? \\ ?' table cells; full prompts contain four digit table cells and "
          "no '? \\ ?' anywhere",
          all(not _DIGIT_CELL_RE.search(solo_user(v))
              and len(_MASKED_CELL_RE.findall(v.table_text)) == 4 for v in masked)
          and all(len(_DIGIT_CELL_RE.findall(v.table_text)) == 4
                  and not _MASKED_CELL_RE.search(solo_user(v)) for v in full))
    check("masked view shares permutation, names and gold with its full twin",
          all((m.row_perm, m.col_perm, m.row_name, m.col_name, m.gold_verdict)
              == (u.row_perm, u.col_perm, u.row_name, u.col_name, u.gold_verdict)
              for m, u in zip(masked, full)))
    check("GameItem duck-types item_id / gold_verdict / post_text for do_call",
          GameItem(full[0]).item_id == full[0].game_id
          and GameItem(full[0]).gold_verdict == full[0].gold_verdict
          and GameItem(full[0]).post_text == full[0].text)

    # ---- dry-run count = 144 x samples -------------------------------------
    for s in (1, 2):
        vs = render_views(games, s, masked=False)
        cm = cost_model("grok-4-1-fast-reasoning", vs, max_tokens=1024)
        check(f"dry-run: samples={s} -> {N_GAMES * s} calls, priced, $ > 0",
              cm["n_calls"] == N_GAMES * s and cm["n_items"] == N_GAMES
              and cm["samples"] == s and cm["priced"] and cm["total_usd"] > 0
              and 0 <= cm["n_cached"] <= cm["n_calls"])
    cm_np = cost_model("no-such-model", full[:5], max_tokens=1024)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_cost_model(cm_np)
    check("dry-run flags an unpriced model loudly instead of reporting $0 quietly",
          not cm_np["priced"] and "NOT in PRICES" in buf.getvalue())
    cm_v = cost_model("claude-haiku-4-5",
                      render_views(select_games(games, REGION_VALUE), 2, masked=True),
                      max_tokens=512)
    check("dry-run on --games value --masked --samples 2 = 32 calls, arm masked",
          cm_v["n_calls"] == 32 and cm_v["arm"] == ARM_MASKED)

    # ---- verdict parsing on synthetic responses ----------------------------
    v0 = full[0]
    synth = (
        (f"Both prefer it.\n\nVERDICT: {v0.gold_verdict}", v0.gold_verdict, True),
        ("Nothing singles one out.\n\nVERDICT: NO UNIQUE PLAN", NO_UNIQUE_PLAN,
         v0.gold_verdict == NO_UNIQUE_PLAN),
        ("**Verdict: plan 21**", "PLAN_21", v0.gold_verdict == "PLAN_21"),
        ("I think they should both take option 1.", NOVERDICT, False),
        ("", NOVERDICT, False),
    )
    bad = []
    for text, want, want_correct in synth:
        row = row_from_view(v0, "m", {"output": text, "finish_reason": "stop"})
        if row["verdict"] != want or row["correct"] != int(want_correct):
            bad.append((text, row["verdict"], row["correct"]))
    check(f"row_from_view parses {len(synth)} synthetic responses and scores "
          f"them against gold" + (f" -- first bad {bad[0]!r}" if bad else ""),
          not bad)
    row_t = row_from_view(v0, "m", {"output": "VERDICT: PLAN_12",
                                    "finish_reason": "length"})
    check("row_from_view marks finish_reason=length as truncated",
          row_t["truncated"] == 1 and row_t["is_nash"] in (0, 1))
    check("row_from_view: is_nash is blank for NO_UNIQUE_PLAN / NOVERDICT and "
          "agrees with the view for PLAN verdicts",
          row_from_view(v0, "m", {"output": "VERDICT: NO_UNIQUE_PLAN"})["is_nash"] == ""
          and row_from_view(v0, "m", {"output": "nope"})["is_nash"] == ""
          and all(row_from_view(v0, "m", {"output": f"VERDICT: {p}"})["is_nash"]
                  == int(p in v0.nash_verdicts) for p in PLAN_VERDICTS))

    # ---- analysis arithmetic on engineered rows ----------------------------
    def rows_with(policy) -> list[dict]:
        return [row_from_view(v, "m", {"output": f"VERDICT: {policy(v)}",
                                       "finish_reason": "stop"}) for v in full]

    perfect = rows_with(lambda v: v.gold_verdict)
    rep = analyze_rows(perfect, draws=200, seed=1)
    check("analysis: perfect rows -> accuracy 1.0 overall, per type and per region, "
          "guard passes",
          rep["overall"]["accuracy"] == 1.0 and rep["guard"]["pass"]
          and all(g["accuracy"] == 1.0 for g in rep["by_gold_type"].values())
          and all(g["accuracy"] == 1.0 for g in rep["by_region"].values())
          and set(rep["by_gold_type"]) == set(GOLD_TYPES)
          and set(rep["by_region"]) == set(REGIONS))
    check("analysis: on perfect rows Nash-pull is 0 on DILEMMA_BOTH_PREFER and "
          "1 on the neutral region (gold is non-Nash there, Nash here)",
          rep["by_gold_type"]["DILEMMA_BOTH_PREFER"]["nash_pull"] == 0.0
          and rep["by_region"][REGION_NEUTRAL]["nash_pull"] == 1.0
          and rep["by_region"][REGION_CANNOT_ASSIST]["n_plan"] == 0
          and rep["by_region"][REGION_CANNOT_ASSIST]["nash_pull"] is None)

    def nash_policy(v: ItemView) -> str:
        return sorted(v.nash_verdicts)[0] if v.nash_verdicts else NO_UNIQUE_PLAN
    pulled = analyze_rows(rows_with(nash_policy), draws=200, seed=1)
    check("analysis: a seat that always names a Nash cell has Nash-pull 1.0 on "
          "the value region and accuracy 0 on DILEMMA_BOTH_PREFER",
          pulled["by_region"][REGION_VALUE]["nash_pull"] == 1.0
          and pulled["by_gold_type"]["DILEMMA_BOTH_PREFER"]["accuracy"] == 0.0)

    dirty = [dict(r) for r in perfect]
    n_dirty = 0
    for r in dirty[::10]:                      # ~10% NOVERDICT, above the 5% limit
        r["verdict"], r["correct"], r["noncommittal"], r["is_nash"] = NOVERDICT, 0, 1, ""
        n_dirty += 1
    rep_d = analyze_rows(dirty, draws=100, seed=1)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_analysis(rep_d, label="selftest")
    check(f"analysis: {n_dirty}/{len(dirty)} NOVERDICT fails the guard and "
          "print_analysis prints GUARD FAILED",
          not rep_d["guard"]["pass"] and "GUARD FAILED" in buf.getvalue()
          and abs(rep_d["overall"]["noverdict_share"] - n_dirty / len(dirty)) < 1e-9
          and rep_d["overall"]["noverdict_share"] > MAX_NOVERDICT_SHARE)
    clean_edge = [dict(r) for r in perfect]
    for r in clean_edge[::25][:int(len(clean_edge) * 0.04)]:   # 4% < 5% limit
        r["verdict"], r["correct"], r["noncommittal"], r["is_nash"] = NOVERDICT, 0, 1, ""
    check("analysis: 4% NOVERDICT stays under the 5% limit and passes the guard",
          analyze_rows(clean_edge, draws=20, seed=1)["guard"]["pass"])
    check("analysis: CSV round trip preserves every read-out",
          _csv_roundtrip_equal(perfect))

    import random as _random
    rng = _random.Random(3)
    guess = [row_from_view(v, "m", {"output": f"VERDICT: {rng.choice(VERDICT_SET)}"})
             for v in masked]
    rep_m = analyze_rows(guess, draws=200, seed=1)
    check("analysis: masked rows get a masked floor report; random guessing is "
          "not flagged as a leak",
          rep_m["masked"] and rep_m.get("masked_floor") is not None
          and not rep_m["masked_floor"]["leak_suspected"])
    check("analysis: full rows get no masked floor report",
          not rep["masked"] and rep.get("masked_floor") is None)

    # ---- --regime nonbinding: rendering, prompts, rows, analysis ------------
    full_nb = render_views(games, N_SELFTEST_SAMPLES, masked=False,
                           regime=REGIME_NONBINDING)
    masked_nb = render_views(games, N_SELFTEST_SAMPLES, masked=True,
                             regime=REGIME_NONBINDING)
    check(f"render_views(regime=nonbinding): {len(full_nb)} ItemViewNB views + "
          f"masked twin, regime nonbinding; binding views are plain ItemView; an "
          "unknown regime is refused",
          len(full_nb) == len(full) and len(masked_nb) == len(masked)
          and all(isinstance(v, ItemViewNB) and regime_of(v) == REGIME_NONBINDING
                  for v in full_nb + masked_nb)
          and all(type(v) is ItemView and regime_of(v) == REGIME_BINDING for v in full)
          and _raises_kw(render_views, games, 1, masked=False, regime="bogus"))
    check("nb view of a (game, idx) shares permutation, names and table with the "
          "binding view; gold agrees except NO_UNIQUE_PLAN -> NO_RECOMMENDATION",
          all((n.row_perm, n.col_perm, n.row_name, n.col_name, n.table_text)
              == (b.row_perm, b.col_perm, b.row_name, b.col_name, b.table_text)
              and n.gold_verdict == (NO_RECOMMENDATION if b.gold_verdict == NO_UNIQUE_PLAN
                                     else b.gold_verdict)
              for n, b in zip(full_nb, full)))
    nb_leaks, nb_tail, nb_meta, nb_bind = [], [], [], []
    for v in full_nb + masked_nb:
        system, user = prompt_pair(v)
        if forbidden_hits(system) or forbidden_hits(user):
            nb_leaks.append((v.game_id, v.sample_idx, v.masked))
        if not user.rstrip().endswith(NB_VERDICT_INSTRUCTION):
            nb_tail.append((v.game_id, v.sample_idx))
        low = user.lower()
        if (v.gold_type.lower() in low or "gold" in low or "nash" in low
                or "canonical" in low or v.region in low.split()):
            nb_meta.append((v.game_id, v.sample_idx))
        if (NO_UNIQUE_PLAN in user or "binding agreement" in user
                or GAME_VERDICT_INSTRUCTION in user):
            nb_bind.append((v.game_id, v.sample_idx))
    check(f"nb: no forbidden word in any of the {len(full_nb) + len(masked_nb)} "
          "rendered system+user prompts" + (f" -- first {nb_leaks[0]}" if nb_leaks else ""),
          not nb_leaks)
    check("nb: NB_VERDICT_INSTRUCTION is the last thing in every user prompt "
          "(never GAME_VERDICT_INSTRUCTION)", not nb_tail and not nb_bind)
    check("nb: no gold_type / gold / nash / region metadata in any user prompt",
          not nb_meta)
    check("nb: same system prompt as binding; user prompt = nb item text + the "
          "'no deliberation' clause + the nb instruction; all five nb tokens "
          "present and NO_UNIQUE_PLAN absent; assert_prompt_clean passes",
          all(prompt_pair(v)[0] == SOLO_SYSTEM and v.text in solo_user(v)
              and "no deliberation" in solo_user(v)
              and all(f"VERDICT: {t}" in solo_user(v) for t in NB_VERDICT_SET)
              and _ok(assert_prompt_clean, *prompt_pair(v))
              for v in full_nb[:20] + masked_nb[:20]))
    check("nb: the binding and nb prompts of the same view differ ONLY in the "
          "question, legend and instruction line (scenario identical)",
          all(solo_user(b).split("\n\n")[0] == solo_user(n).split("\n\n")[0]
              and solo_user(b) != solo_user(n)
              for b, n in zip(full[:40], full_nb[:40])))
    check("nb: masked prompts contain no payoff digit cell and four '? \\ ?' table "
          "cells; masked twin shares permutation, names and gold",
          all(not _DIGIT_CELL_RE.search(solo_user(v))
              and len(_MASKED_CELL_RE.findall(v.table_text)) == 4 for v in masked_nb)
          and all((m.row_perm, m.col_perm, m.row_name, m.col_name, m.gold_verdict)
                  == (u.row_perm, u.col_perm, u.row_name, u.col_name, u.gold_verdict)
                  for m, u in zip(masked_nb, full_nb)))
    n0 = full_nb[0]
    ca0 = next(v for v in full_nb if v.region == REGION_CANNOT_ASSIST)
    nb_synth = (
        (n0, f"VERDICT: {n0.gold_verdict}", n0.gold_verdict, True),
        (ca0, "Nothing to recommend.\n\nVERDICT: NO RECOMMENDATION", NO_RECOMMENDATION, True),
        (n0, "VERDICT: NO RECOMMENDATION", NO_RECOMMENDATION, False),
        (ca0, "VERDICT: NO_UNIQUE_PLAN", NOVERDICT, False),        # binding token: illegal here
        (ca0, f"VERDICT: {ca0.row_best_verdict}", ca0.row_best_verdict, False),
        (n0, "**Verdict: plan 21**", "PLAN_21", n0.gold_verdict == "PLAN_21"),
        (n0, "I think they should both take option 1.", NOVERDICT, False),
    )
    nb_bad = []
    for v, text, want, want_correct in nb_synth:
        row = row_from_view(v, "m", {"output": text, "finish_reason": "stop"})
        if (row["verdict"] != want or row["correct"] != int(want_correct)
                or row["scaffold"] != SCAFFOLD_NAME_NB):
            nb_bad.append((text, row["verdict"], row["correct"], row["scaffold"]))
    check(f"nb: row_from_view parses {len(nb_synth)} synthetic responses on the nb "
          "instrument, scores against the nb gold and stamps scaffold tmg_solo_nb"
          + (f" -- first bad {nb_bad[0]!r}" if nb_bad else ""), not nb_bad)
    check("nb: is_nash is blank for NO_RECOMMENDATION / NOVERDICT and agrees with "
          "the view for PLAN verdicts; binding row_from_view still stamps tmg_solo",
          row_from_view(n0, "m", {"output": "VERDICT: NO_RECOMMENDATION"})["is_nash"] == ""
          and row_from_view(n0, "m", {"output": "nope"})["is_nash"] == ""
          and all(row_from_view(n0, "m", {"output": f"VERDICT: {p}"})["is_nash"]
                  == int(p in n0.nash_verdicts) for p in PLAN_VERDICTS)
          and row_from_view(full[0], "m", {"output": "VERDICT: PLAN_12"})["scaffold"]
          == SCAFFOLD_NAME)

    def nb_rows_with(policy) -> list[dict]:
        return [row_from_view(v, "m", {"output": f"VERDICT: {policy(v)}",
                                       "finish_reason": "stop"}) for v in full_nb]

    nb_perfect = nb_rows_with(lambda v: v.gold_verdict)
    check("regime_of_rows: binding rows -> binding, nb rows -> nonbinding, rows "
          "without a scaffold column -> binding, a mix is refused",
          regime_of_rows(perfect) == REGIME_BINDING
          and regime_of_rows(nb_perfect) == REGIME_NONBINDING
          and regime_of_rows([{k: v for k, v in r.items() if k != "scaffold"}
                              for r in perfect[:5]]) == REGIME_BINDING
          and _raises(regime_of_rows, perfect[:3] + nb_perfect[:3]))
    rep_nb = analyze_rows(nb_perfect, draws=200, seed=1)
    check("nb analysis: perfect rows -> accuracy 1.0 overall, per type and per "
          "region; guard passes; report carries regime nonbinding and the "
          "no_recommendation_share key (binding report carries neither)",
          rep_nb["overall"]["accuracy"] == 1.0 and rep_nb["guard"]["pass"]
          and all(g["accuracy"] == 1.0 for g in rep_nb["by_gold_type"].values())
          and all(g["accuracy"] == 1.0 for g in rep_nb["by_region"].values())
          and set(rep_nb["by_region"]) == set(REGIONS)
          and rep_nb.get("regime") == REGIME_NONBINDING
          and "no_recommendation_share" in rep_nb["overall"]
          and "no_unique_share" not in rep_nb["overall"]
          and "regime" not in rep and "no_unique_share" in rep["overall"]
          and "no_recommendation_share" not in rep["overall"])
    check("nb analysis: NO_RECOMMENDATION share is 1.0 on cannot_assist and 0 on "
          "the value region for perfect rows; Nash-pull 0 on DILEMMA_BOTH_PREFER, "
          "1 on the neutral region, undefined on cannot_assist",
          rep_nb["by_region"][REGION_CANNOT_ASSIST]["no_recommendation_share"] == 1.0
          and rep_nb["by_region"][REGION_VALUE]["no_recommendation_share"] == 0.0
          and rep_nb["by_gold_type"]["DILEMMA_BOTH_PREFER"]["nash_pull"] == 0.0
          and rep_nb["by_region"][REGION_NEUTRAL]["nash_pull"] == 1.0
          and rep_nb["by_region"][REGION_CANNOT_ASSIST]["n_plan"] == 0
          and rep_nb["by_region"][REGION_CANNOT_ASSIST]["nash_pull"] is None)

    def nb_nash_policy(v: ItemView) -> str:
        return sorted(v.nash_verdicts)[0] if v.nash_verdicts else NO_RECOMMENDATION
    nb_pulled = analyze_rows(nb_rows_with(nb_nash_policy), draws=200, seed=1)
    asr_nb = [v for v in full_nb if v.gold_type == "ASSURANCE"]
    asr_hit = sum(nb_nash_policy(v) == v.gold_verdict for v in asr_nb) / len(asr_nb)
    check("nb analysis: a solo that always names a Nash cell has Nash-pull 1.0 on "
          "the value region, accuracy 0 on DILEMMA_BOTH_PREFER and, on ASSURANCE "
          f"(two Nash cells), exactly the share of views whose first-sorted Nash "
          f"cell is the gold ({asr_hit:.3f})",
          nb_pulled["by_region"][REGION_VALUE]["nash_pull"] == 1.0
          and nb_pulled["by_gold_type"]["DILEMMA_BOTH_PREFER"]["accuracy"] == 0.0
          and abs(nb_pulled["by_gold_type"]["ASSURANCE"]["accuracy"] - asr_hit) < 1e-12
          and 0.0 < asr_hit < 1.0)
    nb_wrong_token = analyze_rows(nb_rows_with(
        lambda v: NO_UNIQUE_PLAN if v.region == REGION_CANNOT_ASSIST else v.gold_verdict),
        draws=100, seed=1)
    check("nb analysis: a solo answering the BINDING token NO_UNIQUE_PLAN on "
          "cannot_assist scores NOVERDICT there (never correct) and fails the guard",
          nb_wrong_token["by_region"][REGION_CANNOT_ASSIST]["accuracy"] == 0.0
          and nb_wrong_token["by_region"][REGION_CANNOT_ASSIST]["noverdict_share"] == 1.0
          and not nb_wrong_token["guard"]["pass"])
    nb_dirty = [dict(r) for r in nb_perfect]
    n_nb_dirty = 0
    for r in nb_dirty[::10]:
        r["verdict"], r["correct"], r["noncommittal"], r["is_nash"] = NOVERDICT, 0, 1, ""
        n_nb_dirty += 1
    rep_nd = analyze_rows(nb_dirty, draws=100, seed=1)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_analysis(rep_nd, label="selftest-nb")
    check(f"nb analysis: {n_nb_dirty}/{len(nb_dirty)} NOVERDICT fails the guard; "
          "print_analysis prints GUARD FAILED, the NON-BINDING tag, NOREC and "
          "NO_RECOMMENDATION",
          not rep_nd["guard"]["pass"] and "GUARD FAILED" in buf.getvalue()
          and "NON-BINDING" in buf.getvalue() and "NOREC" in buf.getvalue()
          and NO_RECOMMENDATION in buf.getvalue()
          and "NOUNQ" not in buf.getvalue() and NO_UNIQUE_PLAN not in buf.getvalue())
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_analysis(rep, label="selftest")
    check("binding print_analysis carries no NON-BINDING tag, NOREC or "
          "NO_RECOMMENDATION (byte-identical to before the flag)",
          "NON-BINDING" not in buf.getvalue() and "NOREC" not in buf.getvalue()
          and NO_RECOMMENDATION not in buf.getvalue() and "NOUNQ" in buf.getvalue())
    check("nb analysis: CSV round trip preserves every read-out and the regime",
          _csv_roundtrip_equal(nb_perfect)
          and regime_of_rows(read_rows_of(nb_perfect)) == REGIME_NONBINDING)
    nb_guess = [row_from_view(v, "m", {"output": f"VERDICT: {rng.choice(NB_VERDICT_SET)}"})
                for v in masked_nb]
    rep_nm = analyze_rows(nb_guess, draws=200, seed=1)
    check("nb analysis: masked nb rows get a masked floor report on the nb set "
          f"(n={rep_nm.get('masked_floor', {}).get('n')} of {len(nb_guess)}: "
          "NO_RECOMMENDATION golds are kept, not dropped); random guessing is not "
          "flagged; full nb rows get none",
          rep_nm["masked"] and rep_nm.get("masked_floor") is not None
          and rep_nm["masked_floor"]["n"] == len(nb_guess)
          and NO_RECOMMENDATION in rep_nm["masked_floor"]["gold_counts"]
          and not rep_nm["masked_floor"]["leak_suspected"]
          and not rep_nb["masked"] and rep_nb.get("masked_floor") is None)
    cm_nb = cost_model("grok-4-1-fast-reasoning", full_nb, max_tokens=1024)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_cost_model(cm_nb)
    check(f"nb dry-run: {N_GAMES * N_SELFTEST_SAMPLES} calls, priced, scaffold "
          "tmg_solo_nb, header names the regime; a binding+nb mix is refused",
          cm_nb["n_calls"] == N_GAMES * N_SELFTEST_SAMPLES and cm_nb["priced"]
          and cm_nb["scaffold"] == SCAFFOLD_NAME_NB and cm_nb["regime"] == REGIME_NONBINDING
          and "regime nonbinding" in buf.getvalue()
          and _raises_kw(cost_model, "grok-4-1-fast-reasoning", full[:2] + full_nb[:2],
                         max_tokens=1024))

    # ---- end-to-end against a stub generator: no network, no real cache ----
    import scripts.run_crowdgold_deliberation as rcd

    class _Stub:
        def __init__(self, text, finish_reason="stop", ptok=400, ctok=60):
            self.text = text
            self.finish_reason = finish_reason
            self.prompt_tokens = ptok
            self.completion_tokens = ctok

    seen: list[tuple[str, str]] = []

    def stub_generate(model, system, user, *, sample_idx=0, max_tokens=0, **kw):
        seen.append((system, user))
        return _Stub("The efficient cell is the one both prefer.\n\nVERDICT: PLAN 12")

    real_before = len(cache_files_containing(NEW_CACHE_NAMES))
    tmp_out = Path(tempfile.mkdtemp(prefix="_tmg_solo_selftest_"))
    old_gen, old_out = rcd.generate_any, rcd.OUT_DIR
    rcd.generate_any = stub_generate
    rcd.OUT_DIR = tmp_out          # the name call_cache_path actually reads
    try:
        sub = full[:3] + masked[:2]
        rows, failed = run("stub-model", sub, max_tokens=256, workers=2)
        check("smoke: one row per view, sorted by (item, sample), nothing failed",
              len(rows) == len(sub) and failed == []
              and [(r["item_id"], r["sample_idx"]) for r in rows]
              == sorted((v.game_id, v.sample_idx) for v in sub))
        check("smoke: every row carries the parsed verdict PLAN_12 and is committal",
              all(r["verdict"] == "PLAN_12" and r["noncommittal"] == 0 for r in rows))
        check("smoke: rows carry both arms and every ROW_FIELDS key",
              {r["arm"] for r in rows} == {ARM_FULL, ARM_MASKED}
              and all(set(ROW_FIELDS) <= set(r) for r in rows))
        check("smoke: the stub saw the neutral system prompt and a user prompt "
              "ending in the verdict instruction, every call",
              len(seen) == len(sub)
              and all(s == SOLO_SYSTEM and u.rstrip().endswith(GAME_VERDICT_INSTRUCTION)
                      for s, u in seen))
        written = sorted(tmp_out.glob("*.json"))
        check(f"smoke: {len(sub)} cache files written under the temp dir, all "
              f"named with {SCAFFOLD_NAME!r} and {ROLE_ID!r}",
              len(written) == len(sub)
              and all(SCAFFOLD_NAME in p.name and ROLE_ID in p.name for p in written))
        n_seen = len(seen)
        rows2, _ = run("stub-model", sub, max_tokens=256, workers=2)
        check("smoke: a second run is served from cache (no new generate calls) "
              "and yields identical rows",
              len(seen) == n_seen and rows2 == rows)
        rep_s = analyze_rows(rows, draws=50, seed=1)
        check("smoke: analysis runs on the stub rows and the guard passes",
              rep_s["guard"]["pass"] and rep_s["n_rows"] == len(sub))
        # ---- a prompt edit must invalidate the cache (parents = system, user)
        me = sys.modules[__name__]
        saved_system = me.SOLO_SYSTEM
        try:
            me.SOLO_SYSTEM = saved_system + " (edited)"
            run("stub-model", sub, max_tokens=256, workers=2)
            n_after_sys = len(seen)
        finally:
            me.SOLO_SYSTEM = saved_system
        check("smoke: editing SOLO_SYSTEM invalidates every cached record "
              f"({n_after_sys - n_seen} new generate calls for {len(sub)} views, "
              "was 0 when only the item text was the parent)",
              n_after_sys - n_seen == len(sub))
        saved_user = me.solo_user
        try:
            me.solo_user = lambda v: saved_user(v).replace(  # type: ignore[assignment]
                "no deliberation", "no deliberation whatsoever")
            run("stub-model", sub, max_tokens=256, workers=2)
            n_after_user = len(seen)
        finally:
            me.solo_user = saved_user
        check("smoke: editing the fixed instruction sentence invalidates the cache too",
              n_after_user - n_after_sys == len(sub))
        # ---- an errored view is a guard failure, never a silent omission ----
        # (sub[2] is the one full view whose (game_id, sample_idx) has no
        # masked twin in ``sub``; run() keys failures by that pair)
        boom = {(sub[2].game_id, sub[2].sample_idx)}

        def raising_generate(model, system, user, *, sample_idx=0, max_tokens=0, **kw):
            if any(v.text in user for v in sub if (v.game_id, v.sample_idx) in boom):
                raise RuntimeError("simulated 5xx")
            return stub_generate(model, system, user, sample_idx=sample_idx,
                                 max_tokens=max_tokens, **kw)
        rcd.generate_any = raising_generate
        fresh = Path(tempfile.mkdtemp(prefix="_tmg_solo_selftest_err_"))
        rcd.OUT_DIR = fresh
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                rows_e, failed_e = run("stub-model", sub, max_tokens=256, workers=2)
        finally:
            rcd.OUT_DIR = tmp_out
            rcd.generate_any = stub_generate
            shutil.rmtree(fresh, ignore_errors=True)
        check("run(): a raising call is returned as a failed key and dropped from "
              f"rows ({len(rows_e)} kept, {len(failed_e)} failed of {len(sub)}); "
              "the survivors' guard would pass",
              failed_e == sorted(boom) and len(rows_e) == len(sub) - 1
              and analyze_rows(rows_e, draws=20, seed=1)["guard"]["pass"]
              and "ERROR" in buf.getvalue())
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok_e = errored_views_report(failed_e, len(sub))
            ok_c = errored_views_report([], len(sub))
        check("errored_views_report prints GUARD FAILED naming N of M and the keys, "
              "returns False; passes silently on no failures",
              not ok_e and ok_c
              and f"GUARD FAILED: 1 of {len(sub)} views errored" in buf.getvalue()
              and f"{sub[2].game_id}/{sub[2].sample_idx}" in buf.getvalue())
        n_seen = len(seen)
        # ---- --regime nonbinding through the same stub: own scaffold, own
        # ---- cache files, same (game, idx) as the binding views above -------
        sub_nb = full_nb[:3] + masked_nb[:2]
        rows_nb, failed_nb = run("stub-model", sub_nb, max_tokens=256, workers=2)
        check("smoke nb: one row per view, nothing failed, every row stamped "
              "scaffold tmg_solo_nb with verdict PLAN_12 parsed on the nb instrument",
              len(rows_nb) == len(sub_nb) and failed_nb == []
              and all(r["scaffold"] == SCAFFOLD_NAME_NB and r["verdict"] == "PLAN_12"
                      and r["noncommittal"] == 0 for r in rows_nb)
              and {r["arm"] for r in rows_nb} == {ARM_FULL, ARM_MASKED})
        check("smoke nb: the stub saw the SAME system prompt and user prompts "
              "ending in NB_VERDICT_INSTRUCTION, every call",
              len(seen) == n_seen + len(sub_nb)
              and all(s_ == SOLO_SYSTEM and u.rstrip().endswith(NB_VERDICT_INSTRUCTION)
                      for s_, u in seen[n_seen:]))
        written_nb = sorted(p for p in tmp_out.glob("*.json") if SCAFFOLD_NAME_NB in p.name)
        written_b = sorted(p for p in tmp_out.glob("*.json") if SCAFFOLD_NAME_NB not in p.name)
        check(f"smoke nb: {len(written_nb)} cache files under scaffold tmg_solo_nb, "
              f"own-shaped, disjoint from the {len(written_b)} binding files of the "
              "same (game, idx, arm)",
              len(written_nb) == len(sub_nb)
              and all(_OWN_CACHE_RE.match(p.name) for p in written_nb + written_b)
              and not (set(written_nb) & set(written_b))
              and all(f"_{SCAFFOLD_NAME}_t" not in p.name for p in written_nb))
        n_seen2 = len(seen)
        rows_nb2, _ = run("stub-model", sub_nb, max_tokens=256, workers=2)
        check("smoke nb: a second nb run is served from cache (no new generate "
              "calls) and yields identical rows",
              len(seen) == n_seen2 and rows_nb2 == rows_nb)
        rep_snb = analyze_rows(rows_nb, draws=50, seed=1)
        check("smoke nb: analysis runs on the stub rows as nonbinding and the guard passes",
              rep_snb["guard"]["pass"] and rep_snb["n_rows"] == len(sub_nb)
              and rep_snb.get("regime") == REGIME_NONBINDING)
        n_seen = len(seen)
    finally:
        rcd.generate_any, rcd.OUT_DIR = old_gen, old_out
        shutil.rmtree(tmp_out, ignore_errors=True)
    check("smoke: nothing leaked into the real output directory",
          len(cache_files_containing(NEW_CACHE_NAMES)) == real_before)

    # ---- CLI gating: no flag combination other than --run can generate ------
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_none = main(["--games", "value"])
    check("main() without --run/--dry-run/--analyze refuses (exit 2) and says so",
          rc_none == 2 and "--run" in buf.getvalue())
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_dry = main(["--dry-run", "--samples", "2"])
    check(f"main(--dry-run --samples 2) exits 0 and reports {N_GAMES * 2} calls",
          rc_dry == 0 and f"= {N_GAMES * 2} calls" in buf.getvalue())
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_an = main(["--analyze", "--tag", "game_solo_selftest_never_written"])
    check("main(--analyze) on a missing rows file exits 2 without generating",
          rc_an == 2 and "not found" in buf.getvalue() and len(seen) == n_seen)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_tmg = main(["--analyze", "--tag", "tmg_solo"])
    check("main() refuses a 'tmg_'-prefixed --tag (exit 2): its outputs would "
          "break tmg_games' no-'tmg_'-file selftest assertion",
          rc_tmg == 2 and "tmg_" in buf.getvalue() and "not found" not in buf.getvalue()
          and not (OUT_DIR / "tmg_solo_rows.csv").exists())
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_np = main(["--dry-run", "--model", "no-such-model", "--games", "value"])
    check("main(--dry-run) on an unpriced model exits 2; the --run path refuses "
          "it before any call the same way (not exercised: the selftest never "
          "passes --run)",
          rc_np == 2 and "NOT in PRICES" in buf.getvalue()
          and "not in PRICES" in inspect.getsource(main))

    # ---- CLI gating, --regime nonbinding ------------------------------------
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_dry_nb = main(["--dry-run", "--regime", "nonbinding", "--samples", "2"])
    check(f"main(--dry-run --regime nonbinding --samples 2) exits 0, reports "
          f"{N_GAMES * 2} calls, scaffold tmg_solo_nb and tag game_solo_nb",
          rc_dry_nb == 0 and f"= {N_GAMES * 2} calls" in buf.getvalue()
          and f"scaffold {SCAFFOLD_NAME_NB}" in buf.getvalue()
          and f"tag {DEFAULT_TAG_NB}" in buf.getvalue()
          and "regime nonbinding" in buf.getvalue())
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_dry_nbm = main(["--dry-run", "--regime", "nonbinding", "--masked",
                           "--games", "value"])
    check("main(--dry-run --regime nonbinding --masked --games value) exits 0 with "
          "tag game_solo_nb_masked and 16 calls",
          rc_dry_nbm == 0 and f"tag {DEFAULT_TAG_NB_MASKED}" in buf.getvalue()
          and "= 16 calls" in buf.getvalue())
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_none_nb = main(["--regime", "nonbinding", "--games", "value"])
    check("main(--regime nonbinding) without --run/--dry-run/--analyze refuses "
          "(exit 2) and generates nothing",
          rc_none_nb == 2 and "--run" in buf.getvalue() and len(seen) == n_seen)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_an_nb = main(["--analyze", "--regime", "nonbinding", "--tag",
                         "game_solo_nb_selftest_never_written"])
    check("main(--analyze --regime nonbinding) on a missing rows file exits 2 "
          "without generating",
          rc_an_nb == 2 and "not found" in buf.getvalue() and len(seen) == n_seen)
    # ---- --analyze refuses rows whose regime contradicts --regime; the rows
    # ---- live in a temp OUT_DIR so nothing lands in the real one -----------
    me = sys.modules[__name__]
    tmp_an = Path(tempfile.mkdtemp(prefix="_tmg_solo_selftest_an_"))
    saved_out = me.OUT_DIR
    try:
        me.OUT_DIR = tmp_an
        write_rows(perfect[:12], tmp_an / "bind_x_rows.csv")
        write_rows(nb_perfect[:12], tmp_an / "nb_x_rows.csv")
        outs = {}
        for argv in (["--analyze", "--tag", "bind_x"],
                     ["--analyze", "--tag", "bind_x", "--regime", "nonbinding"],
                     ["--analyze", "--tag", "nb_x", "--regime", "nonbinding"],
                     ["--analyze", "--tag", "nb_x"]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                outs[" ".join(argv)] = (main(argv), buf.getvalue())
        check("main(--analyze): binding rows analyse under the default and are "
              "REFUSED (exit 2, naming the regime) under --regime nonbinding",
              outs["--analyze --tag bind_x"][0] == 0
              and outs["--analyze --tag bind_x --regime nonbinding"][0] == 2
              and "binding rows" in outs["--analyze --tag bind_x --regime nonbinding"][1]
              and (tmp_an / "bind_x_analysis.json").exists()
              and "regime" not in json.loads((tmp_an / "bind_x_analysis.json").read_text()))
        check("main(--analyze): nb rows analyse under --regime nonbinding (JSON "
              "carries regime nonbinding) and are REFUSED under the default",
              outs["--analyze --tag nb_x --regime nonbinding"][0] == 0
              and "NON-BINDING" in outs["--analyze --tag nb_x --regime nonbinding"][1]
              and json.loads((tmp_an / "nb_x_analysis.json").read_text()).get("regime")
              == REGIME_NONBINDING
              and outs["--analyze --tag nb_x"][0] == 2
              and "nonbinding rows" in outs["--analyze --tag nb_x"][1]
              and len(seen) == n_seen)
        # ---- the --run path's write-target refusals (run_target_refusal runs
        # ---- BEFORE any call; exercised directly, the selftest never passes
        # ---- the run flag) ------------------------------------------------
        r_cross = run_target_refusal("bind_x", REGIME_NONBINDING, tmp_an / "bind_x_rows.csv")
        r_cross2 = run_target_refusal("nb_x", REGIME_BINDING, tmp_an / "nb_x_rows.csv")
        check("run_target_refusal REFUSES to overwrite an existing rows CSV of the "
              "OTHER regime (binding rows under a nonbinding run and the reverse), "
              "naming both scaffolds",
              r_cross is not None and "binding rows" in r_cross and SCAFFOLD_NAME in r_cross
              and SCAFFOLD_NAME_NB in r_cross
              and r_cross2 is not None and "nonbinding rows" in r_cross2)
        check("run_target_refusal REFUSES the other regime's default tags (game_solo / "
              "game_solo_masked under nonbinding; game_solo_nb / game_solo_nb_masked "
              "under binding) even when no file exists yet",
              all(run_target_refusal(t, REGIME_NONBINDING, tmp_an / f"{t}_rows.csv")
                  is not None for t in (DEFAULT_TAG, DEFAULT_TAG_MASKED))
              and all(run_target_refusal(t, REGIME_BINDING, tmp_an / f"{t}_rows.csv")
                      is not None for t in (DEFAULT_TAG_NB, DEFAULT_TAG_NB_MASKED))
              and DEFAULT_TAG in run_target_refusal(DEFAULT_TAG, REGIME_NONBINDING,
                                                    tmp_an / "none.csv"))
        check("run_target_refusal ALLOWS the own regime's default tags, an existing "
              "rows CSV of the same regime, and an absent file",
              run_target_refusal(DEFAULT_TAG, REGIME_BINDING, tmp_an / "absent_rows.csv") is None
              and run_target_refusal(DEFAULT_TAG_NB, REGIME_NONBINDING, tmp_an / "absent.csv") is None
              and run_target_refusal("bind_x", REGIME_BINDING, tmp_an / "bind_x_rows.csv") is None
              and run_target_refusal("nb_x", REGIME_NONBINDING, tmp_an / "nb_x_rows.csv") is None
              and run_target_refusal("fresh", REGIME_NONBINDING, tmp_an / "fresh_rows.csv") is None)
        src_main = inspect.getsource(main)
        check("main() calls run_target_refusal on the --run path BEFORE run() (so the "
              "refusal precedes any call) and exits 2 on it",
              "run_target_refusal(" in src_main
              and src_main.index("run_target_refusal(") < src_main.index("rows, failed = run(")
              and src_main.index("run_target_refusal(") > src_main.index("if args.run:"))
    finally:
        me.OUT_DIR = saved_out
        shutil.rmtree(tmp_an, ignore_errors=True)
    check("nothing of the --analyze checks landed in the real output directory",
          not (OUT_DIR / "bind_x_rows.csv").exists()
          and not (OUT_DIR / "nb_x_rows.csv").exists()
          and not (OUT_DIR / "nb_x_analysis.json").exists())

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def _csv_roundtrip_equal(rows: Sequence[Mapping[str, Any]]) -> bool:
    tmp = Path(tempfile.mkdtemp(prefix="_tmg_solo_csv_")) / "rows.csv"
    try:
        write_rows(rows, tmp)
        back = read_rows(tmp)
        a = analyze_rows(rows, draws=50, seed=2)
        b = analyze_rows(back, draws=50, seed=2)
        return (len(back) == len(rows)
                and a["overall"] == b["overall"]
                and a["by_gold_type"] == b["by_gold_type"]
                and a["guard"]["pass"] == b["guard"]["pass"])
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def _raises(fn, *a) -> bool:
    try:
        fn(*a)
    except ValueError:
        return True
    return False


def _raises_kw(fn, *a, **kw) -> bool:
    try:
        fn(*a, **kw)
    except ValueError:
        return True
    return False


def read_rows_of(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Rows as they come back from a CSV written by write_rows (strings)."""
    tmp = Path(tempfile.mkdtemp(prefix="_tmg_solo_csv_")) / "rows.csv"
    try:
        write_rows(rows, tmp)
        return read_rows(tmp)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def _raises_rt(fn, *a) -> bool:
    try:
        fn(*a)
    except RuntimeError:
        return True
    return False


def _ok(fn, *a) -> bool:
    try:
        fn(*a)
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage A solo pilot: one neutral analyst per leak-"
                    "controlled 2x2 game, binding joint-plan verdict")
    ap.add_argument("--selftest", action="store_true",
                    help="offline self-check; no API keys, no network")
    ap.add_argument("--dry-run", action="store_true",
                    help="print call count and $ estimate; makes no calls")
    ap.add_argument("--run", action="store_true",
                    help="GENERATE (spends money). Required for any call. "
                         "Implies --analyze on the rows it writes.")
    ap.add_argument("--analyze", action="store_true",
                    help="analyse an existing {tag}_rows.csv; no calls")
    ap.add_argument("--model", default="grok-4-1-fast-reasoning",
                    help="Same default as scripts.run_crowdgold_dilemma: this "
                         "is the same-model solo comparator for the coming "
                         "deliberation, not a cross-vendor actuator.")
    ap.add_argument("--samples", type=int, default=1,
                    help="renderings per game; each sample_idx draws its own "
                         "action permutation and player names")
    ap.add_argument("--masked", action="store_true",
                    help="hypothesis-only baseline: every payoff is '?'")
    ap.add_argument("--regime", choices=REGIMES, default=REGIME_BINDING,
                    help="binding (default): the binding joint-plan question, "
                         "PLAN_xy / NO_UNIQUE_PLAN. nonbinding (16.13c): each "
                         "player chooses privately, a recommendation binds no "
                         "one, PLAN_xy / NO_RECOMMENDATION; own scaffold "
                         f"{SCAFFOLD_NAME_NB} and tag {DEFAULT_TAG_NB}")
    ap.add_argument("--games", choices=GAME_SELECTIONS, default="all")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default=None,
                    help=f"rows/analysis file tag (default {DEFAULT_TAG} or "
                         f"{DEFAULT_TAG_MASKED} with --masked; {DEFAULT_TAG_NB} / "
                         f"{DEFAULT_TAG_NB_MASKED} under --regime nonbinding)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7,
                    help="bootstrap seed only; renderings are seeded by "
                         "(game_id, sample_idx)")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.samples < 1:
        print("\nERROR: --samples must be >= 1\n")
        return 2

    try:
        games = load_games()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}\n")
        return 2
    sel = select_games(games, args.games)
    views = render_views(sel, args.samples, masked=args.masked, regime=args.regime)
    tag = args.tag or default_tag(args.masked, args.regime)
    if tag.startswith("tmg_"):
        print(f"\nERROR: --tag {tag!r} starts with 'tmg_'; scripts.tmg_games' "
              "selftest asserts no 'tmg_'-prefixed file exists in "
              f"{OUT_DIR.name}. Use e.g. {DEFAULT_TAG!r}.\n")
        return 2
    by_region = Counter(g.region for g in sel)
    print(f"model {args.model}  scaffold {scaffold_for(args.regime)}  "
          f"arm {arm_for(args.masked)}"
          f"  games {args.games} ({len(sel)}: "
          + ", ".join(f"{r} {by_region.get(r, 0)}" for r in REGIONS)
          + f")  samples {args.samples}  -> {len(views)} calls  tag {tag}")

    if args.dry_run:
        cm = cost_model(args.model, views, max_tokens=args.max_tokens)
        print_cost_model(cm)
        return 0 if cm["priced"] else 2

    rows_path = OUT_DIR / f"{tag}_rows.csv"
    if args.run:
        if args.model not in PRICES:
            print(f"\nERROR: {args.model} is not in PRICES; add a price to "
                  "run_crowdgold_deliberation.PRICES before spending.\n")
            return 2
        refusal = run_target_refusal(tag, args.regime, rows_path)
        if refusal:
            print(f"\nERROR: {refusal}\n")
            return 2
        rows, failed = run(args.model, views, max_tokens=args.max_tokens,
                           workers=args.workers)
        if not rows:
            print("\nNo rows produced.\n")
            errored_views_report(failed, len(views))
            return 1
        if not errored_views_report(failed, len(views)):
            gf_path = OUT_DIR / f"{tag}_GUARD_FAILED_rows.csv"
            write_rows(rows, gf_path)
            print(f"  survivors dumped to {gf_path} (NOT the run tag's rows CSV)")
            return 4
        write_rows(rows, rows_path)
        print(f"\nwrote {len(rows)} rows -> {rows_path}")
        pt = sum(int(r.get("prompt_tokens", 0) or 0) for r in rows)
        ct = sum(int(r.get("completion_tokens", 0) or 0) for r in rows)
        pin, pout = PRICES.get(args.model, (0.0, 0.0))
        print(f"measured spend: {pt} prompt + {ct} completion tokens ~= "
              f"${(pt / 1e6 * pin + ct / 1e6 * pout):.2f}")
    elif args.analyze:
        if not rows_path.exists():
            print(f"\nERROR: {rows_path} not found; run with --run first.\n")
            return 2
        rows = read_rows(rows_path)
        if not rows:
            print(f"\nERROR: {rows_path} is empty.\n")
            return 1
        try:
            rows_regime = regime_of_rows(rows)
        except ValueError as e:
            print(f"\nERROR: {rows_path}: {e}\n")
            return 2
        if rows_regime != args.regime:
            print(f"\nERROR: {rows_path} holds {rows_regime} rows (scaffold "
                  f"{scaffold_for(rows_regime)}) but --regime {args.regime} was "
                  f"given; pass --regime {rows_regime} so the rows are scored on "
                  "their own instrument.\n")
            return 2
    else:
        print("\nNothing done. Pass --dry-run for the count and $ estimate, "
              "--analyze to read an existing run, or --run to GENERATE "
              "(spends money; refused without the flag).\n")
        return 2

    rep = analyze_rows(rows, draws=args.n_boot, seed=args.seed)
    print_analysis(rep, label=tag)
    out_json = OUT_DIR / f"{tag}_analysis.json"
    out_json.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nwrote analysis -> {out_json}")
    if not rep["guard"]["pass"]:
        print("\n*** GUARD FAILED (truncation or NOVERDICT > 5%). Raise "
              "--max-tokens and re-run before reading any accuracy or "
              "Nash-pull number off this run. ***")
        return 4
    if rep.get("masked_floor", {}).get("leak_suspected"):
        print("\n*** LEAK SUSPECTED on the masked baseline: fix the item text "
              "before reading any unmasked result. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
