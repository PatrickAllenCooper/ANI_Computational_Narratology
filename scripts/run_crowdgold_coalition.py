"""
scripts/run_crowdgold_coalition.py -- Addendum 16.19: coalition advocates.
A FOURTH seat on the writer's side of the registered AITA deliberation.

THE QUESTION

Every embodied cell so far seats one advocate per side. Krishna and Morgan
(2001, "A Model of Expertise", QJE 116(2)) predict that consulting two
experts with LIKE biases is no better than consulting one -- the second
adds no information -- while experts with OPPOSED biases are informative.
On this instrument the prediction is concrete: a second writer-side seat
should object almost exactly when the writer's advocate objects (objection
phi near +1), and pooling its objections into the flag should not raise the
flag's error lift. If instead the ally's objections carry independent
information (phi well below +1 with a lift gain), the sensor scales with
seats rather than with SIDES, and the 16.10 reading "the structure comes
from directional role assignment" needs the word "opposed" in it.

THE ROLE SET (four seats, edges ON, everyone reads everyone)

  writer_advocate_co      the registered writer brief VERBATIM   stake not_at_fault
  writer_ally_co          NEW: speaks for a close friend of the  stake not_at_fault
                          writer whose interest is the same; "a
                          finding that the writer is in the
                          wrong is a finding against the party
                          you speak for"; puts the case as
                          strongly as the account allows; may
                          not invent facts. Mirrors the writer
                          brief sentence for sentence.
  counterparty_co         the registered counterparty brief       stake at_fault
                          VERBATIM
  neutral_adjudicator_co  the registered neutral brief VERBATIM   stake none

  moderator prefix  modcoalition          tag  cg_deliberation_coalition

Panel, arms, model, protocol and caps are the registered AITA panel by
default (grok-4-1-fast-reasoning, --n-yta 99 --n-nta 150, seed 44, the
content-filter screen to 210 items, third_person + as_asker, --samples 1,
narrative_cot scaffold, 2560/1024/1024/512 token caps), inherited from
run_crowdgold_deliberation.main unchanged.

WHAT N_AGENTS = 4 BREAKS IN THE BASE RUNNER, AND WHERE IT IS FIXED (here)

  * every prompt and both moderator system prompts hard-code the participant
    count in words ("one of three participants", "The other two
    participants", "all three final positions", "three-party
    deliberation", ...). The seven prompt builders and the two system
    prompts are re-issued here with ONLY those count words changed; the
    selftest proves each coalition prompt equals the base prompt with the
    count-word substitutions applied and nothing else.
  * ROLE_ORDER loops (_others_block, synthesis_user, integration_user,
    extract_addressed, the R0-R4 loops in run_deliberation, parent_sha
    tuples) are already length-agnostic: they read the module globals that
    install() replaces. _others_block's `cap` is the per-text transcript
    cap (0 = off), not a seat count, so nothing there caps at two.
  * CALLS_PER_CELL becomes 4+4+4+1+4+1+4 = 22; install() recomputes it from
    N_AGENTS, and the cost model reads N_AGENTS/ROLE_ORDER at call time so
    the extra seat and the longer r1/r2/synthesis/integration prompts are
    priced. print_cost_model's hard-coded "3 roles x ..." breakdown line is
    replaced.
  * the outcome row's vote_writer_advocate / vote_counterparty /
    vote_neutral_adjudicator columns key on the embodied ids and would be
    blank; run_deliberation is wrapped to fill them from the `_co` ids and
    to add vote_writer_ally plus n_against_interest_r3 (below). ROW_FIELDS
    is extended so the new columns reach disk.
  * stake_undermined(role, verdict) is stake-driven, so the ally (stake
    not_at_fault) is undermined exactly when the writer's advocate is.
    STAKE_BEARING_ROLES is a module constant computed from the embodied
    ROLES at import and would drop every `_co` seat from the within-role
    concentration; it is re-pointed at the three stake-bearing `_co` seats.
  * n_r3_unparsed / n_vote_unparsed / unanimous_accept / vote_independence
    read N_AGENTS at call time and are correct for four seats.

THE GENERALISED SENSOR

An AGAINST-INTEREST objection is an R3 objection (ACCEPT_WITH_MODIFICATION
or REJECT) lodged against a synthesis verdict that SERVES the seat's own
stake: the writer's advocate or the ally objecting to a not-at-fault
synthesis, the counterparty objecting to an at-fault one. The per-debate
count (0-2: the two writer-side seats on a not-at-fault synthesis, the
counterparty alone on an at-fault one; blank when the synthesis verdict is
not codable) is written to the rows CSV as n_against_interest_r3 and read out by
scripts/analyze_coalition.py together with the like-biased pair's objection
phi, each advocate's role-lock, the pooled flag's lift and the paired S2
accuracy against the embodied cell.

CACHE PLAN (call_cache_path keys on model/scaffold/arm/item/sample/ROLE ID,
not on brief text; the moderator id comes from moderator_role_id)

  EVERY ONE OF THE 22 CALLS PER DEBATE IS A NEW GENERATION. Nothing in this
  cell replays from any existing run: all four seats carry `_co` role ids
  and the moderator the `modcoalition` prefix, so no cache path can
  coincide with an embodied, unembodied, topology or game record. Nor
  could a replay be legitimate: a four-seat prompt says "one of four
  participants" and quotes three neighbours, so not even the writer's
  advocate's r0 is byte-identical to the embodied writer_advocate r0. The
  dry-run therefore counts all 22 calls per debate, and the selftest
  verifies every namespace is empty (or holds only this cell's own
  registered footprint) with the game runner's "no foreign file" idiom.
  cache_is_clean() is enforced on the --run path per model; --resume is the
  only override and is valid only when the briefs are known unchanged.

DISCLOSED DIFFERENCES OUTSIDE THE BRIEFS

  The R0 preamble reads "four participants with different stakes in the
  outcome" -- the single word "three" -> "four". Two of the four hold the
  SAME interest for different parties; the sentence is kept otherwise
  verbatim rather than rewritten, so the manipulation stays one word.

WHAT THE LIKE-PAIR PHI IS A STATISTIC OF (disclosed; read it accordingly)

  Edges are ON: r1_user_co and r2_user_co quote all three neighbours through
  rcd._others_block, so the writer's advocate reads the ally's r0 and r1 and
  the ally reads the advocate's before either labels the synthesis at R3.
  The headline Krishna-Morgan reading -- the objection phi between
  writer_advocate_co and writer_ally_co, measured on objected_r3 by
  scripts/analyze_coalition.py -- is therefore a POST-COMMUNICATION
  statistic: a phi near +1 is a mixture of shared stake and two rounds of
  mutual persuasion, and the opposed-pair contrast under the same regime
  controls that only partly. The PRE-communication reading is the like
  pair's r0 verdict concordance (r0 is written before any seat has read any
  other; the analyzer reports it as like_pair_r0_concordance): if the r0
  concordance and the R3 phi diverge, the clean K-M test is an EDGES-OFF
  coalition cell -- run_crowdgold_topology.r1_user_noedge / r2_user_noedge
  under new `_co_noedge` role ids and a new moderator prefix -- which is not
  built here and would be a separate registration.

GUARDS AFTER THE DELEGATED RUN (this wrapper's, on top of the base runner's)

  * COMPLETENESS. rcd.run swallows an exception raised inside
    run_deliberation with one console ERROR line and omits that debate from
    rows / votes / calls; the base round and outcome guards and every
    analyzer denominator are then computed on the survivors, so a vendor
    outage concentrated in one arm is arm-correlated attrition that nothing
    reports. After rcd.main returns, this wrapper rebuilds the launched key
    set (models x arms x screened items x samples, from the same panel
    arguments it forwarded), compares it with the rows CSV, writes
    <tag>_completeness.json (n_expected, n_rows, n_missing, the missing
    keys, per-(model, arm) counts, the summary's n_items cross-check) and
    prints the literal GUARD FAILED and exits 4 on any shortfall.
    scripts/analyze_coalition.py refuses a cell whose sidecar is missing or
    reports a shortfall.
  * QUARANTINE. The base runner writes its CSVs BEFORE its guard verdict.
    On exit 4 (its round / outcome guard, or the completeness guard) the
    rows and votes CSVs are renamed <tag>_GUARD_FAILED_rows.csv /
    _votes.csv so nothing under the readable tag can be tabulated; re-run
    with --run --resume once the cause is fixed (cached calls replay).
  * BANNED FLAGS. --no-unresolved (and --transcript-cap) are refused
    alongside the stake knobs: each changes the r0-r2 / synthesis /
    integration prompt text WITHOUT changing call_cache_path or the r0
    parent_sha (post text only), so a --run --resume after a partial run
    under the other setting would silently serve records generated for a
    different prompt. The cell is defined at the registered
    allow_unresolved=True, transcript cap 0.

Nothing here generates unless the literal flag --run is given. Omitting it
prints the cost model and exits 2. --dry-run prints the cost model and
exits 0. --selftest stubs generation and redirects the cache directory that
call_cache_path reads (run_crowdgold_deliberation.OUT_DIR) to a tempdir.

Usage
-----
  python -m scripts.run_crowdgold_coalition --selftest
  python -m scripts.run_crowdgold_coalition --verify-cache-clean
  python -m scripts.run_crowdgold_coalition --dry-run
  python -m scripts.run_crowdgold_coalition --run            # SPENDS MONEY
  python -m scripts.run_crowdgold_coalition --run --resume   # briefs unchanged only
  python -m scripts.run_crowdgold_coalition --dry-run --models grok-4-1-fast-reasoning,<new-deployment>

Exit codes: 0 ok, 1 no rows, 2 bad config / --run not given / unpriced model
/ tag not under cg_deliberation_coalition, 3 cache not clean (pass --resume),
4 guard failed (round, outcome or completeness) or a banned flag.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.run_crowdgold_deliberation import POST_DELIM, Role
from scripts.run_phase1_quartet import OUT_DIR, _safe

DEFAULT_TAG = "cg_deliberation_coalition"
MODERATOR_PREFIX = "modcoalition"
SUFFIX = "_co"
DEFAULT_MODELS = "grok-4-1-fast-reasoning"

#: Sibling cells' tags, listed so the selftest can prove the PREFIX RULE
#: (tag_ok: --tag must start with DEFAULT_TAG) refuses every one of them.
#: rcd.main writes <tag>_rows.csv / _votes.csv / _summary.json (and
#: <tag>_dryrun.json on --dry-run) unconditionally, so a deny-list that
#: misses a sibling (cg_deliberation_stdcot, _multivendor, _nperspective,
#: _modsonnet, _modhaiku, _gpt4o ...) would let this runner overwrite it.
SIBLING_TAGS = ("cg_deliberation", "cg_deliberation_noedge", "cg_deliberation_identical",
                "cg_deliberation_identical_noedge", "cg_deliberation_unembodied",
                "cg_deliberation_stdcot", "cg_deliberation_multivendor",
                "cg_deliberation_nperspective", "cg_deliberation_modsonnet",
                "cg_deliberation_modhaiku", "cg_deliberation_gpt4o",
                "cg_deliberation_gpt4o_screen")


def tag_ok(tag: str) -> bool:
    """The only tags this runner may write under: DEFAULT_TAG or a suffix of it."""
    return tag.startswith(DEFAULT_TAG)

#: Captured ONCE at import, before install() mutates rcd. Everything derived
#: from the embodied role set reads these, never rcd.ROLES.
_EMBODIED: tuple[Role, ...] = rcd.ROLES
_EMB_BY_ID = {r.role_id: r for r in _EMBODIED}
_BASE_PROMPTS = {name: getattr(rcd, name) for name in (
    "r0_user", "r1_user", "r2_user", "synthesis_user", "r3_label_user",
    "integration_user", "r4_vote_user")}
_BASE_SYNTHESIS_SYSTEM = rcd.SYNTHESIS_SYSTEM
_BASE_INTEGRATION_SYSTEM = rcd.INTEGRATION_SYSTEM
_BASE_R0_PREAMBLE = rcd.R0_PREAMBLE
_BASE_ROW_FIELDS = rcd.ROW_FIELDS
_BASE_RUN_DELIBERATION = rcd.run_deliberation

WRITER_ALLY_BRIEF = (
    "You speak for a close friend of the writer -- someone who stands with "
    "the person whose first-person account this is. The writer's conduct is "
    "what is being judged, the friend has no power over the outcome, and the "
    "friend's interest is the writer's own. A finding that the writer is in "
    "the wrong is a finding against the party you speak for. Put the writer's "
    "case as strongly as the account itself allows, including whatever "
    "context or constraint the other participants are passing over. You "
    "may not invent facts that are not in the account, and you may not "
    "assert that the writer is not in the wrong if the account itself "
    "does not support it."
)


def build_roles() -> tuple[Role, ...]:
    w, c, n = (_EMB_BY_ID["writer_advocate"], _EMB_BY_ID["counterparty"],
               _EMB_BY_ID["neutral_adjudicator"])
    return (
        Role(role_id="writer_advocate" + SUFFIX, label=w.label, paper_role=w.paper_role,
             stake=w.stake, brief=w.brief),
        Role(role_id="writer_ally" + SUFFIX, label="Writer's Ally", paper_role="unmapped",
             stake="not_at_fault", brief=WRITER_ALLY_BRIEF),
        Role(role_id="counterparty" + SUFFIX, label=c.label, paper_role=c.paper_role,
             stake=c.stake, brief=c.brief),
        Role(role_id="neutral_adjudicator" + SUFFIX, label=n.label, paper_role=n.paper_role,
             stake=n.stake, brief=n.brief),
    )


ROLES_CO: tuple[Role, ...] = build_roles()
ROLE_IDS_CO = tuple(r.role_id for r in ROLES_CO)
STAKE_BEARING_CO = tuple(r.role_id for r in ROLES_CO if r.stake != "none")
WRITER_SIDE_CO = ("writer_advocate" + SUFFIX, "writer_ally" + SUFFIX)

for _r in ROLES_CO:
    assert POST_DELIM not in _r.brief, f"role brief {_r.role_id} has a blank line"
del _r

#: The count-word substitutions, and NOTHING else, that separate a coalition
#: prompt from the base prompt it mirrors. The selftest applies these to the
#: base output and demands byte equality with the coalition output.
COUNT_WORDS = (
    ("one of three participants", "one of four participants"),
    ("three participants with different stakes", "four participants with different stakes"),
    ("The other two participants", "The other three participants"),
    ("Three participants have deliberated", "Four participants have deliberated"),
    ("all three final positions", "all four final positions"),
    ("synthesis to three participants", "synthesis to four participants"),
    ("read all three responses", "read all four responses"),
    # moderator system prompts
    ("three-party deliberation. Three participants", "four-party deliberation. Four participants"),
    ("the concerns the three have revealed", "the concerns the four have revealed"),
    ("not one of the three participants", "not one of the four participants"),
    ("Three participants have each responded", "Four participants have each responded"),
)


def recount(text: str) -> str:
    for a, b in COUNT_WORDS:
        text = text.replace(a, b)
    return text


SYNTHESIS_SYSTEM_CO = (
    "You are a neutral moderator of a four-party deliberation. Four "
    "participants with conflicting stakes have each stated a position, "
    "rebutted the others, and restated a final position. Your task is to write "
    "a SINGLE synthesis position that addresses the concerns the four have "
    "revealed, and to state the group's verdict on the request. The synthesis "
    "must be a concrete position, not a hedge, and the verdict must be the one "
    "the account actually supports. You are not one of the four participants "
    "and you do not represent any of their stakes."
)

INTEGRATION_SYSTEM_CO = (
    "You are a skilled mediator. Four participants have each responded to a "
    "proposed synthesis, accepting it, accepting it with a stated modification, "
    "or rejecting it. Your task is to construct a SINGLE integrated proposal "
    "that addresses as many of their stated modification requests as can be "
    "reconciled without abandoning the position the account supports, to say "
    "explicitly whose request each part of it answers, and to state the group's "
    "verdict. The integrated proposal must be a concrete position, not a hedge. "
    "You are not one of the four participants and you do not represent any of "
    "their stakes."
)

R0_PREAMBLE_CO = (
    "An assistant has received the request below. Before it answers, four "
    "participants with different stakes in the outcome will deliberate.\n\n"
)

COALITION_NOTE = """\
COALITION (Addendum 16.19). A fourth seat, writer_ally_co, speaks for a close
friend of the writer and carries the writer's own stake (not_at_fault). The
other three seats are the registered briefs verbatim under `_co` ids. Krishna
and Morgan (2001) predict the like-biased pair's objections coincide (phi near
+1) and that pooling the ally into the flag adds no lift.
"""


# ---------------------------------------------------------------------------
# Prompts: the base builders with the participant count re-worded to four
# ---------------------------------------------------------------------------

def _role_header_co(role: Role) -> str:
    return (
        f"You are one of four participants in this deliberation, speaking as "
        f"{role.label}.\n\n{role.brief}"
    )


def r0_user_co(arm, item, role, *, allow_unresolved):
    return (
        f"{R0_PREAMBLE_CO}"
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{_role_header_co(role)}\n\n"
        "Write your opening statement. State the position you hold on this "
        "request and why.\n\n"
        f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def r1_user_co(arm, item, role, own_r0, r0_texts, *, allow_unresolved, cap):
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{_role_header_co(role)}\n\n"
        f"--- YOUR OPENING STATEMENT ---\n{rcd._cap(own_r0, cap)}\n--- END ---\n\n"
        "The other three participants opened as follows.\n\n"
        f"{rcd._others_block(r0_texts, role.role_id, cap)}\n\n"
        "Write your rebuttal. Challenge what you believe is wrong or "
        "incomplete, acknowledge what genuinely lands, and change your own "
        "position only if you have actually been persuaded by something in "
        "the account.\n\n"
        f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def r2_user_co(arm, item, role, own_r0, own_r1, r1_texts, *, allow_unresolved, cap):
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{_role_header_co(role)}\n\n"
        f"--- YOUR OPENING STATEMENT ---\n{rcd._cap(own_r0, cap)}\n--- END ---\n\n"
        f"--- YOUR REBUTTAL ---\n{rcd._cap(own_r1, cap)}\n--- END ---\n\n"
        "The other three participants rebutted as follows.\n\n"
        f"{rcd._others_block(r1_texts, role.role_id, cap)}\n\n"
        "State your final position. Say plainly what the group's finding "
        "should be and what your position rests on.\n\n"
        f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def synthesis_user_co(arm, item, r2_texts, *, allow_unresolved, cap):
    parts = []
    for rid in rcd.ROLE_ORDER:
        r = rcd.ROLE_BY_ID[rid]
        parts.append(
            f"--- {r.label} ({rid}) : final position ---\n"
            f"{rcd._cap(r2_texts.get(rid) or '[no statement]', cap)}"
        )
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        "Four participants have deliberated over this request. Their final "
        "positions follow.\n\n"
        + "\n\n".join(parts)
        + "\n\nWrite one synthesis position. Give it a short name on a line "
        "beginning 'SYNTHESIS:'. In two to four sentences state the position "
        "and say which concern of each participant it answers. Then give the "
        "group's verdict on the request.\n\n"
        f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def _refuse_stake_knobs(**knobs) -> None:
    on = [k for k, v in knobs.items() if v]
    if on:
        raise ValueError(f"stake knobs {on} are banned in the coalition cell")


def r3_label_user_co(arm, item, role, own_r2, synthesis, *, cap, stake_nudge=False,
                     stake_cot=False, stake_fewshot=False, stake_fewshot_set="plain"):
    _refuse_stake_knobs(stake_nudge=stake_nudge, stake_cot=stake_cot,
                        stake_fewshot=stake_fewshot)
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{_role_header_co(role)}\n\n"
        f"--- YOUR FINAL POSITION ---\n{rcd._cap(own_r2, cap)}\n--- END ---\n\n"
        "The moderator has read all four final positions and written a single "
        "synthesis.\n\n"
        f"--- MODERATOR SYNTHESIS ---\n{rcd._cap(synthesis, cap)}\n--- END ---\n\n"
        "Respond to this synthesis. Accept it as it stands, accept it with one "
        "stated modification, or reject it.\n"
        f"  If you accept it with a modification, put that modification in one "
        f"sentence on its own line beginning '{rcd.MOD_MARKER}'.\n"
        f"  If you reject it, name the single concern you believe no "
        f"integration can absorb, in one sentence on its own line beginning "
        f"'{rcd.UNRESOLVABLE_MARKER}'.\n"
        "  If you accept it as it stands, write neither line.\n\n"
        f"{rcd.verdict_instruction(rcd.LABEL_INSTRUMENT, allow_unresolved=False)}"
    )


def integration_user_co(arm, item, synthesis, labels, objections, *, allow_unresolved, cap):
    parts = []
    for rid in rcd.ROLE_ORDER:
        r = rcd.ROLE_BY_ID[rid]
        obj = objections.get(rid) or "[no modification stated]"
        parts.append(
            f"--- {r.label} ({rid}) ---\n"
            f"label: {labels.get(rid) or rcd.NOVERDICT}\n"
            f"requests: {rcd._cap(obj, cap)}"
        )
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        "You proposed this synthesis to four participants.\n\n"
        f"--- YOUR SYNTHESIS ---\n{rcd._cap(synthesis, cap)}\n--- END ---\n\n"
        "They responded as follows.\n\n"
        + "\n\n".join(parts)
        + "\n\nConstruct one integrated proposal that addresses as many of "
        "these requests as can be reconciled. Give it a short name on a line "
        "beginning 'PROPOSAL:'. In two to four sentences state the position and "
        "say explicitly how it answers, or why it cannot answer, each request. "
        f"Then, on its own line beginning '{rcd.ADDRESSED_MARKER}', list the "
        "participant ids whose request the proposal substantially addresses "
        f"(from: {', '.join(rcd.ROLE_ORDER)}), or the word none. Then give the "
        "group's verdict on the request.\n\n"
        f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def r4_vote_user_co(arm, item, role, synthesis, own_label, own_objection, proposal, *,
                    cap, stake_nudge=False, stake_cot=False, stake_fewshot=False,
                    stake_fewshot_set="plain"):
    _refuse_stake_knobs(stake_nudge=stake_nudge, stake_cot=stake_cot,
                        stake_fewshot=stake_fewshot)
    own = (f"--- YOUR RESPONSE TO THE SYNTHESIS ---\n"
           f"You labelled it: {own_label}\n"
           + (f"You required: {rcd._cap(own_objection, cap)}\n" if own_objection
              else "You stated no modification.\n")
           + "--- END ---")
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{_role_header_co(role)}\n\n"
        f"--- MODERATOR SYNTHESIS ---\n{rcd._cap(synthesis, cap)}\n--- END ---\n\n"
        f"{own}\n\n"
        "The moderator has now read all four responses and built a single "
        "integrated proposal.\n\n"
        f"--- INTEGRATED PROPOSAL ---\n{rcd._cap(proposal, cap)}\n--- END ---\n\n"
        "This is the final round and your vote is binary. Accept the "
        "integrated proposal, or reject it. In two or three sentences say why, "
        "referring to what you required of the synthesis; if you reject, name "
        "the single concern it leaves unresolved.\n\n"
        f"{rcd.verdict_instruction(rcd.VOTE_INSTRUMENT, allow_unresolved=False)}"
    )


# ---------------------------------------------------------------------------
# Row post-processing: the vote columns and the generalised sensor
# ---------------------------------------------------------------------------

EXTRA_ROW_FIELDS = ("vote_writer_ally", "n_against_interest_r3",
                    "against_interest_roles")


def against_interest_count(vote_rows: Sequence[dict], synthesis_verdict: str):
    """(count, role ids) of stake-bearing seats that objected at R3 to a
    synthesis verdict that SERVES their stake. None when the synthesis
    verdict is not codable, so the debate is excluded rather than coded 0."""
    roles = []
    for v in vote_rows:
        if v.get("role_stake") not in ("at_fault", "not_at_fault"):
            continue
        und = rcd.stake_undermined(v["role_id"], synthesis_verdict)
        if und is None:
            return None, ()
        if int(v.get("objected_r3") or 0) and not und:
            roles.append(v["role_id"])
    return len(roles), tuple(roles)


def run_deliberation_co(*args, **kwargs):
    row, vote_rows, calls = _BASE_RUN_DELIBERATION(*args, **kwargs)
    votes = {v["role_id"]: v["vote"] for v in vote_rows}
    row["vote_writer_advocate"] = votes.get("writer_advocate" + SUFFIX, "")
    row["vote_writer_ally"] = votes.get("writer_ally" + SUFFIX, "")
    row["vote_counterparty"] = votes.get("counterparty" + SUFFIX, "")
    row["vote_neutral_adjudicator"] = votes.get("neutral_adjudicator" + SUFFIX, "")
    n_ai, ai_roles = against_interest_count(vote_rows, row["synthesis_verdict"])
    row["n_against_interest_r3"] = "" if n_ai is None else n_ai
    row["against_interest_roles"] = "|".join(ai_roles)
    return row, vote_rows, calls


def print_cost_model_co(cm: dict, *, label: str) -> None:
    n = rcd.N_AGENTS
    print("\n" + "=" * 78)
    print(f"DRY RUN -- {label}  [coalition: {n} seats, edges on]")
    print("=" * 78)
    print(f"cells                  {cm['cells']:,}")
    print(f"calls per cell         {cm['calls_per_cell']}  "
          f"({n} seats x r0,r1,r2 = {3 * n}, synthesis = 1, {n} labels, "
          f"integration = 1, {n} votes)")
    print(f"TOTAL GENERATION CALLS {cm['total_calls']:,}   "
          "(every one NEW -- no call in this cell replays from cache)")
    print(f"moderator              {cm['moderator_model']}")
    for m, v in cm["per_model"].items():
        print(f"\n  {m}   {'[measured completions]' if v['completion_measured'] else '[completion length ASSUMED]'}"
              f"{'' if v['priced'] else '   [NO PRICE ON FILE]'}")
        print(f"    {'round':<12}{'calls':>9}{'prompt tok':>12}{'compl tok':>11}"
              f"{'in tok':>13}{'out tok':>13}{'usd':>10}  billed to")
        for rd, r in v["rounds"].items():
            print(f"    {rd:<12}{r['calls']:>9,}{r['prompt_tokens_each']:>12,}"
                  f"{r['completion_tokens_each']:>11,}{r['input_tokens']:>13,}"
                  f"{r['output_tokens']:>13,}{r['usd']:>10.2f}  {r['billed_to']}")
        print(f"    {'TOTAL':<12}{v['calls']:>9,}{'':>12}{'':>11}"
              f"{v['input_tokens']:>13,}{v['output_tokens']:>13,}{v['usd']:>10.2f}")
        print(f"    arithmetic: sum of the per-round lines above "
              f"(agent {v['price_in_per_mtok']}/{v['price_out_per_mtok']} per Mtok, "
              f"moderator {v['moderator_price_in_per_mtok']}/"
              f"{v['moderator_price_out_per_mtok']}) = ${v['usd']:.2f}, "
              f"of which ${v['moderator_usd']:.2f} is the moderator")
    print(f"\n  TOTAL ESTIMATED COST  ${cm['total_usd']:.2f}")
    for k, val in cm["assumptions"].items():
        print(f"    assumption[{k}]: {val}")
    unpriced = [m for m, v in cm["per_model"].items() if not v["priced"]]
    if unpriced:
        print(f"\n  *** WARNING: {unpriced} have no PRICES entry; the $ figure above "
              "is a placeholder, not a ceiling. Add a price to "
              "run_crowdgold_deliberation.PRICES before spending. ***")


# ---------------------------------------------------------------------------
# install(): swap the module-level role set and everything that counts seats
# ---------------------------------------------------------------------------

def install() -> None:
    """Total and idempotent. Every patched name is referenced by rcd at call
    time (verified: run_deliberation, run, main, cost_model, _agent_prompts,
    extract_addressed, _others_block all read module globals)."""
    R = ROLES_CO
    rcd.ROLES = R
    rcd.ROLE_BY_ID = {r.role_id: r for r in R}
    rcd.ROLE_ORDER = tuple(r.role_id for r in R)
    rcd.N_AGENTS = len(R)
    rcd.CALLS_PER_CELL = rcd.N_AGENTS * len(rcd.AGENT_ROUNDS) + len(rcd.MODERATOR_ROUNDS)
    rcd.STAKE_BEARING_ROLES = STAKE_BEARING_CO
    rcd.moderator_role_id = lambda m: f"{MODERATOR_PREFIX}-{_safe(m)}"
    rcd.SYNTHESIS_SYSTEM = SYNTHESIS_SYSTEM_CO
    rcd.INTEGRATION_SYSTEM = INTEGRATION_SYSTEM_CO
    rcd.r0_user = r0_user_co
    rcd.r1_user = r1_user_co
    rcd.r2_user = r2_user_co
    rcd.synthesis_user = synthesis_user_co
    rcd.r3_label_user = r3_label_user_co
    rcd.integration_user = integration_user_co
    rcd.r4_vote_user = r4_vote_user_co
    rcd.run_deliberation = run_deliberation_co
    rcd.ROW_FIELDS = tuple(_BASE_ROW_FIELDS) + EXTRA_ROW_FIELDS
    rcd.print_cost_model = print_cost_model_co
    rcd.ROLE_MAPPING_NOTE = rcd.ROLE_MAPPING_NOTE.rstrip("\n") + "\n\n" + COALITION_NOTE \
        if COALITION_NOTE not in rcd.ROLE_MAPPING_NOTE else rcd.ROLE_MAPPING_NOTE


# ---------------------------------------------------------------------------
# Cache namespaces (one listdir pass; the output dir holds ~560k files)
# ---------------------------------------------------------------------------

def namespace_counts(names: Sequence[str], models: Sequence[str],
                     moderator_model: Optional[str]) -> dict[str, int]:
    """Files in every namespace this runner writes: per (model, role id), the
    moderator model's `modcoalition-` records, plus the model-agnostic totals
    and the tag's own CSV/JSON artefacts."""
    counts: dict[str, int] = {}
    mods = sorted({moderator_model or m for m in models})
    for m in models:
        pre = f"cgd_{_safe(m)}_"
        for rid in ROLE_IDS_CO:
            counts[f"{m}:{rid}"] = sum(1 for n in names
                                       if n.startswith(pre) and n.endswith(f"_{rid}.json"))
    for mm in mods:
        pre = f"cgd_{_safe(mm)}_"
        counts[f"{mm}:{MODERATOR_PREFIX}"] = sum(
            1 for n in names if n.startswith(pre) and f"_{MODERATOR_PREFIX}-" in n)
    for rid in ROLE_IDS_CO:
        counts[f"any:{rid}"] = sum(1 for n in names if n.endswith(f"_{rid}.json"))
    counts[f"any:{MODERATOR_PREFIX}"] = sum(1 for n in names if f"_{MODERATOR_PREFIX}-" in n)
    counts[f"tag:{DEFAULT_TAG}"] = sum(1 for n in names if n.startswith(f"{DEFAULT_TAG}_"))
    return counts


def cache_is_clean(models: Sequence[str] = (DEFAULT_MODELS,),
                   moderator_model: Optional[str] = None,
                   out_dir: Optional[Path] = None) -> tuple[bool, dict[str, int]]:
    """Clean = no record for any (listed model, `_co` seat) and none for the
    moderator model under `modcoalition-`. Per model, so a later deployment
    can run into a namespace that already holds another model's records
    without --resume, while a partial re-run of the SAME model under possibly
    edited briefs is refused."""
    out_dir = Path(out_dir) if out_dir is not None else rcd.OUT_DIR
    names = os.listdir(out_dir) if out_dir.is_dir() else []
    counts = namespace_counts(names, list(models), moderator_model)
    own = sum(v for k, v in counts.items() if not k.startswith(("any:", "tag:")))
    return own == 0, counts


def foreign_files(names: Sequence[str]) -> list[str]:
    """Stub / selftest / model-'m' files carrying this runner's names. There
    must never be any: the selftest redirects the cache dir to a tempdir."""
    return sorted(n for n in names
                  if (any(n.endswith(f"_{rid}.json") for rid in ROLE_IDS_CO)
                      or f"_{MODERATOR_PREFIX}-" in n or n.startswith(f"{DEFAULT_TAG}_"))
                  and ("stub-model" in n or n.startswith("cgd_m_") or "selftest" in n))


# ---------------------------------------------------------------------------
# Offline stub for the selftest (never reachable from the run path)
# ---------------------------------------------------------------------------

@dataclass
class _StubResult:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _stub_role(user: str) -> str:
    for r in ROLES_CO:
        if f"speaking as {r.label}." in user:
            return r.role_id
    return "moderator"


def _stub_verdict(text: str) -> str:
    """Ties go to YTA. With four seats the writer's side holds two NTA votes
    against the counterparty's one, so the base runner's tie-to-NTA rule would
    make every stub synthesis NTA and the counterparty could never be planted
    against interest; here the neutral seat's coin decides the tie."""
    return "YTA" if text.count("VERDICT: YTA") >= text.count("VERDICT: NTA") else "NTA"


def _stub_item_hash(user: str) -> int:
    """Hash of the request text alone: identical for every seat and every
    round of one debate, so a plant keyed on it lands on the SAME debates for
    the writer's advocate and the ally (phi stays +1)."""
    body = user.split(rcd.REQ_BEGIN, 1)[1].split(rcd.REQ_END, 1)[0] if rcd.REQ_BEGIN in user else user
    return int(hashlib.sha1(body.encode()).hexdigest()[:8], 16)


def _stub_generate(model, system, user, *, sample_idx=0, max_tokens=512, **kw) -> _StubResult:
    """Deterministic four-seat stand-in. Plants: the ally votes with the
    writer's advocate; both object exactly when the writer's advocate does
    (phi = +1), which is when their shared stake is undermined OR, on an
    item-hash-selected minority of not-at-fault syntheses, AGAINST their
    interest (count 2); the counterparty objects against its interest on an
    item-hash-selected minority of at-fault syntheses (count 1); so
    n_against_interest_r3 takes every value 0-2. Revise-iff-addressed at R4
    with a little noise."""
    h = int(hashlib.sha1(user.encode()).hexdigest()[:8], 16)
    if "This is the final round" in user:
        speaking = _stub_role(user)
        proposal = rcd._stub_block(user, "INTEGRATED PROPOSAL")
        addressed = rcd.marker_line(proposal, rcd.ADDRESSED_MARKER).lower()
        own_label = ("ACCEPT_WITH_MODIFICATION" if "You labelled it: ACCEPT_WITH_MODIFICATION" in user
                     else "REJECT" if "You labelled it: REJECT" in user else "ACCEPT")
        vote = "ACCEPT" if own_label == "ACCEPT" else ("ACCEPT" if speaking in addressed else "REJECT")
        if h % 23 == 0:
            vote = "ACCEPT" if vote == "REJECT" else "REJECT"
        return _StubResult(f"The proposal is workable.\nVERDICT: {vote}", completion_tokens=20)
    if "Construct one integrated proposal" in user:
        v = _stub_verdict(rcd._stub_block(user, "YOUR SYNTHESIS"))
        asked = [rid for rid in rcd.ROLE_ORDER
                 if f"({rid})" in user
                 and "label: ACCEPT_WITH_MODIFICATION" in user.split(f"({rid})", 1)[1][:200]]
        if h % 3 == 0 and asked:
            asked = asked[:-1]
        return _StubResult(
            "PROPOSAL: shared_account_of_the_conflict\n"
            "The position stands, with the constraint each side named folded in.\n"
            f"{rcd.ADDRESSED_MARKER} {', '.join(asked) if asked else 'none'}\n"
            f"VERDICT: {v}", completion_tokens=70)
    if "Write one synthesis position" in user:
        return _StubResult(
            "SYNTHESIS: shared_account_of_the_conflict\n"
            "The group finds the account supports a single reading.\n"
            f"VERDICT: {_stub_verdict(user)}", completion_tokens=60)
    if "Respond to this synthesis" in user:
        speaking = _stub_role(user)
        v = _stub_verdict(rcd._stub_block(user, "MODERATOR SYNTHESIS"))
        role = rcd.ROLE_BY_ID[speaking]
        undermined = rcd.stake_undermined(speaking, v)
        hi = _stub_item_hash(user)
        against = ((speaking == "counterparty" + SUFFIX and not undermined and hi % 3 == 0)
                   or (speaking in WRITER_SIDE_CO and not undermined and hi % 4 == 0))
        if not undermined and not against:
            return _StubResult("This reads correctly to me.\nVERDICT: ACCEPT", completion_tokens=25)
        if h % 11 == 0 and not against:
            return _StubResult(
                "It cannot be reconciled with what I speak for.\n"
                f"{rcd.UNRESOLVABLE_MARKER} the account is read against my party.\n"
                "VERDICT: REJECT", completion_tokens=35)
        return _StubResult(
            "I can accept this if one thing changes.\n"
            f"{rcd.MOD_MARKER} name the constraint {role.label} was under.\n"
            "VERDICT: ACCEPT_WITH_MODIFICATION", completion_tokens=35)
    speaking = _stub_role(user)
    v = ("YTA" if speaking == "counterparty" + SUFFIX
         else "NTA" if speaking in WRITER_SIDE_CO
         else ("YTA" if h % 2 else "NTA"))
    body = ("Section 1 - Protagonist: the writer.\n"
            "Section 2 - Stakeholders: writer, friend, other party, onlookers.\n"
            "Section 3 - Consequences: two steps forward for each.\n"
            "Section 4 - Uncertainty: what the account does not say.\n"
            "Section 5 - Decision: stated plainly.\n"
            f"VERDICT: {v}")
    return _StubResult(body, completion_tokens=120)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

#: --no-unresolved / --transcript-cap: prompt text the cache key and the r0
#: parent_sha cannot see (see GUARDS in the module docstring).
BANNED_FLAGS = ("--stake-nudge", "--stake-cot", "--stake-fewshot", "--stake-fewshot-set",
                "--r3r4-reasoning-effort", "--r3r4-thinking-budget", "--smoke", "--full", "--pilot",
                "--no-unresolved", "--transcript-cap")


# ---------------------------------------------------------------------------
# Completeness guard: every launched debate must be a row
# ---------------------------------------------------------------------------

def panel_args(rest: Sequence[str]) -> argparse.Namespace:
    """The panel-defining flags this wrapper forwards to rcd.main, parsed with
    the base runner's own defaults, so the launched key set can be rebuilt."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--n-yta", type=int, default=rcd.PILOT_N_YTA)
    ap.add_argument("--n-nta", type=int, default=rcd.PILOT_N_NTA)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--min-votes", type=int, default=50)
    ap.add_argument("--min-consensus", type=float, default=0.90)
    ap.add_argument("--arms", default=",".join(rcd.PILOT_ARMS))
    ap.add_argument("--keep-filtered", action="store_true")
    known, _ = ap.parse_known_args(list(rest))
    return known


def expected_keys(models: Sequence[str], rest: Sequence[str]) -> set:
    """(model, arm, item_id, sample_idx) for every debate rcd.main launches
    under these arguments: the registered panel, subset_items (the installed
    one), the content-filter screen as the union over the models, samples."""
    pa = panel_args(rest)
    items = rcd.load_items(source="scruples", n_yta=99, n_nta=150, min_votes=pa.min_votes,
                           min_consensus=pa.min_consensus, seed=pa.seed)
    items = rcd.subset_items(items, pa.n_yta, pa.n_nta)
    if not pa.keep_filtered:
        drop: set = set()
        for m in models:
            drop |= rcd.filter_prone_items(m)
        items = [i for i in items if i.item_id not in drop]
    arms = [a.strip() for a in pa.arms.split(",") if a.strip()]
    return {(m, arm, it.item_id, idx) for m in models for arm in arms for it in items
            for idx in range(pa.samples)}


def completeness_check(tag: str, expected: set, *, out_dir: Optional[Path] = None,
                       summary_n_items: Optional[int] = None) -> dict:
    """Compare <tag>_rows.csv with the launched key set. A shortfall (an
    errored debate rcd.run dropped), an unexpected row, or a summary n_items
    that disagrees with the expected item count fails the guard."""
    import csv
    out_dir = rcd.OUT_DIR if out_dir is None else out_dir
    rows_path = out_dir / f"{tag}_rows.csv"
    present: set = set()
    if rows_path.exists():
        with rows_path.open() as f:
            for r in csv.DictReader(f):
                present.add((r.get("model", ""), r.get("arm", ""), r.get("item_id", ""),
                             int(r.get("sample_idx") or 0)))
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    exp_items = {k[2] for k in expected}
    per = {}
    for m, arm, _, _ in expected:
        d = per.setdefault(f"{m}|{arm}", {"expected": 0, "rows": 0})
        d["expected"] += 1
    for m, arm, i, s in present & expected:
        per[f"{m}|{arm}"]["rows"] += 1
    agree = summary_n_items is None or summary_n_items == len(exp_items)
    return {"tag": tag, "n_expected": len(expected), "n_rows": len(present),
            "n_missing": len(missing), "n_unexpected": len(extra),
            "missing": [list(k) for k in missing], "unexpected": [list(k) for k in extra],
            "per_model_arm": per, "n_items_expected": len(exp_items),
            "summary_n_items": summary_n_items, "summary_n_items_agrees": agree,
            "pass": bool(expected) and not missing and not extra and agree}


def print_completeness(c: dict) -> None:
    print("\nCOMPLETENESS GUARD (every launched debate must be a row; rcd.run drops an errored "
          "debate with one console line)")
    print(f"  expected {c['n_expected']} debates over {c['n_items_expected']} items, rows {c['n_rows']}, "
          f"missing {c['n_missing']}, unexpected {c['n_unexpected']}; "
          + "  ".join(f"{k}={v['rows']}/{v['expected']}" for k, v in c["per_model_arm"].items())
          + f"  {'PASS' if c['pass'] else 'FAIL'}")
    if c["summary_n_items"] is not None and not c["summary_n_items_agrees"]:
        print(f"  the runner's summary reports n_items {c['summary_n_items']} but the rebuilt panel has "
              f"{c['n_items_expected']}: the launched panel could not be reconstructed")
    if c["missing"]:
        print("  missing: " + ", ".join(f"{m}/{a}/{i[:8]}/{s}" for m, a, i, s in c["missing"][:12])
              + (" ..." if len(c["missing"]) > 12 else ""))
    if not c["pass"]:
        print("\n  *** GUARD FAILED (completeness): the surviving rows are arm-correlated attrition "
              "until the missing debates are re-run (--run --resume replays the cached calls). ***")


def quarantine_csvs(tag: str, out_dir: Optional[Path] = None) -> list[str]:
    """The base runner writes its CSVs before its guard verdict; move them out
    of the readable tag. Returns the new names."""
    out_dir = rcd.OUT_DIR if out_dir is None else out_dir
    moved = []
    for kind in ("rows", "votes"):
        p = out_dir / f"{tag}_{kind}.csv"
        if p.exists():
            q = out_dir / f"{tag}_GUARD_FAILED_{kind}.csv"
            p.rename(q)
            moved.append(q.name)
    return moved


def finish_run(tag: str, rc: int, models: Sequence[str], rest: Sequence[str]) -> int:
    """After rcd.main: the completeness guard, then quarantine on any exit 4."""
    if rc in (0, 1, 4):
        sn = None
        sp = rcd.OUT_DIR / f"{tag}_summary.json"
        if sp.exists():
            try:
                sn = json.loads(sp.read_text()).get("n_items")
            except json.JSONDecodeError:
                sn = None
        comp = completeness_check(tag, expected_keys(models, rest), summary_n_items=sn)
        print_completeness(comp)
        (rcd.OUT_DIR / f"{tag}_completeness.json").write_text(json.dumps(comp, indent=2))
        print(f"  wrote {rcd.OUT_DIR / (tag + '_completeness.json')}")
        if not comp["pass"]:
            rc = 4
    if rc == 4:
        moved = quarantine_csvs(tag)
        print(f"\n*** GUARD FAILED -- CSVs moved to {moved or '(none present)'}; nothing under "
              f"{tag}_rows/_votes.csv can be tabulated. Fix the cause and re-run with --run --resume. ***\n")
    return rc


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    from scripts.run_crowdgold_topology import CELLS as TOPO_CELLS, build_roles as topo_roles
    from scripts.run_crowdgold_unembodied import MODERATOR_PREFIX as UNEMB_MOD, UNEMBODIED_ROLES

    # ---- role set ----------------------------------------------------------
    check("four seats", len(ROLES_CO) == 4)
    check("role ids are the embodied ids + '_co' plus writer_ally_co, in the registered order",
          ROLE_IDS_CO == ("writer_advocate_co", "writer_ally_co", "counterparty_co",
                          "neutral_adjudicator_co"))
    check("stakes are not_at_fault, not_at_fault, at_fault, none",
          tuple(r.stake for r in ROLES_CO) == ("not_at_fault", "not_at_fault", "at_fault", "none"))
    for rid in ("writer_advocate", "counterparty", "neutral_adjudicator"):
        co = rcd_role = next(r for r in ROLES_CO if r.role_id == rid + SUFFIX)
        emb = _EMB_BY_ID[rid]
        check(f"{rid}_co carries the embodied brief, label, paper_role and stake VERBATIM",
              co.brief == emb.brief and co.label == emb.label
              and co.paper_role == emb.paper_role and co.stake == emb.stake)
    ally = next(r for r in ROLES_CO if r.role_id == "writer_ally_co")
    writer = _EMB_BY_ID["writer_advocate"]
    check("ally speaks for a close friend of the writer",
          "close friend of the writer" in ally.brief and ally.brief.startswith("You speak for"))
    check("ally carries the writer's stake clause verbatim",
          "A finding that the writer is in the wrong is a finding against the party "
          "you speak for." in ally.brief)
    check("ally keeps the advocacy-intensity clause verbatim",
          "Put the writer's case as strongly as the account itself allows, including "
          "whatever context or constraint the other participants are passing over." in ally.brief)
    check("ally keeps the no-invention clause verbatim",
          ally.brief.endswith("You may not invent facts that are not in the account, and you "
                              "may not assert that the writer is not in the wrong if the account "
                              "itself does not support it."))
    check("ally brief mirrors the writer brief's structure (same sentence count) and length "
          f"(ratio {len(ally.brief) / len(writer.brief):.2f} within [0.85, 1.25])",
          ally.brief.count(". ") + 1 == writer.brief.count(". ") + 1
          and 0.85 <= len(ally.brief) / len(writer.brief) <= 1.25)
    check("ally brief and label are distinct from the writer's",
          ally.brief != writer.brief and ally.label != writer.label)
    check("no brief contains a blank line", all(POST_DELIM not in r.brief for r in ROLES_CO))
    check("every role label is unique (the stub keys on 'speaking as <label>.')",
          len({r.label for r in ROLES_CO}) == 4)

    # ---- namespaces vs every sibling role set -------------------------------
    sibling_ids = {r.role_id for r in _EMBODIED} | {r.role_id for r in UNEMBODIED_ROLES}
    for cell in TOPO_CELLS:
        sibling_ids |= {r.role_id for r in topo_roles(*cell)}
    check("no role id collides with the embodied, unembodied or any topology cell",
          not (set(ROLE_IDS_CO) & sibling_ids))
    check("no role id is a substring of another (extract_addressed matches by substring)",
          not any(a != b and a in b for a in ROLE_IDS_CO for b in ROLE_IDS_CO))
    sib_mods = {"mod", UNEMB_MOD} | {c["mod"] for c in TOPO_CELLS.values()}
    check(f"moderator prefix {MODERATOR_PREFIX!r} is distinct from every sibling's",
          MODERATOR_PREFIX not in sib_mods and not MODERATOR_PREFIX.startswith("mod-"))
    check("tag rule: DEFAULT_TAG and a suffix of it pass; every sibling cell's tag is refused",
          tag_ok(DEFAULT_TAG) and tag_ok(DEFAULT_TAG + "_b") and not any(tag_ok(t) for t in SIBLING_TAGS))
    check("--no-unresolved and --transcript-cap are banned (prompt text the cache key cannot see)",
          "--no-unresolved" in BANNED_FLAGS and "--transcript-cap" in BANNED_FLAGS)

    # ---- install() ---------------------------------------------------------
    install()
    install()  # idempotent
    check("install(): N_AGENTS 4, CALLS_PER_CELL 22 (4+4+4+1+4+1+4)",
          rcd.N_AGENTS == 4 and rcd.CALLS_PER_CELL == 22)
    check("install(): ROLE_ORDER / ROLE_BY_ID resolve every `_co` seat",
          rcd.ROLE_ORDER == ROLE_IDS_CO and all(rcd.ROLE_BY_ID[r.role_id] is r for r in ROLES_CO))
    check("install(): STAKE_BEARING_ROLES re-pointed at the three stake seats",
          rcd.STAKE_BEARING_ROLES == ("writer_advocate_co", "writer_ally_co", "counterparty_co"))
    check("install(): moderator id carries the modcoalition prefix",
          rcd.moderator_role_id("grok-4-1-fast-reasoning") == "modcoalition-grok-4-1-fast-reasoning")
    check("install(): ROW_FIELDS gained the ally vote and the against-interest count exactly once",
          rcd.ROW_FIELDS.count("vote_writer_ally") == 1
          and rcd.ROW_FIELDS.count("n_against_interest_r3") == 1
          and rcd.ROW_FIELDS[:len(_BASE_ROW_FIELDS)] == tuple(_BASE_ROW_FIELDS))
    check("install(): ROLE_MAPPING_NOTE carries the coalition note once",
          rcd.ROLE_MAPPING_NOTE.count("COALITION (Addendum 16.19)") == 1)
    k = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot", "as_asker", "itemX", 0,
                            "r1", rcd.ROLE_ORDER[0], 2560)
    km = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot", "as_asker", "itemX", 0,
                             "synthesis", rcd.moderator_role_id("grok-4-1-fast-reasoning"), 1024)
    check("agent cache path carries the `_co` id and never the embodied id",
          k.name.endswith("_writer_advocate_co.json") and "_writer_advocate.json" not in k.name)
    check("moderator cache path carries modcoalition and never mod-",
          "_modcoalition-grok" in km.name and "_mod-grok" not in km.name)

    # ---- prompts: base + count words, and nothing else ---------------------
    item = rcd.CrowdGoldItem(item_id="X", post_text="[POST]", gold_verdict="YTA")
    role = ROLES_CO[1]
    prior = {rid: f"[{rid} PRIOR]" for rid in ROLE_IDS_CO}
    labels = {rid: "ACCEPT_WITH_MODIFICATION" for rid in ROLE_IDS_CO}
    objs = {rid: f"[{rid} OBJ]" for rid in ROLE_IDS_CO}
    pairs = {
        "r0": (r0_user_co("as_asker", item, role, allow_unresolved=True),
               _BASE_PROMPTS["r0_user"]("as_asker", item, role, allow_unresolved=True)),
        "r1": (r1_user_co("as_asker", item, role, "[OWN0]", prior, allow_unresolved=True, cap=0),
               _BASE_PROMPTS["r1_user"]("as_asker", item, role, "[OWN0]", prior, allow_unresolved=True, cap=0)),
        "r2": (r2_user_co("as_asker", item, role, "[OWN0]", "[OWN1]", prior, allow_unresolved=True, cap=0),
               _BASE_PROMPTS["r2_user"]("as_asker", item, role, "[OWN0]", "[OWN1]", prior, allow_unresolved=True, cap=0)),
        "synthesis": (synthesis_user_co("as_asker", item, prior, allow_unresolved=True, cap=0),
                      _BASE_PROMPTS["synthesis_user"]("as_asker", item, prior, allow_unresolved=True, cap=0)),
        "r3_label": (r3_label_user_co("as_asker", item, role, "[OWN2]", "[SYN]", cap=0),
                     _BASE_PROMPTS["r3_label_user"]("as_asker", item, role, "[OWN2]", "[SYN]", cap=0)),
        "integration": (integration_user_co("as_asker", item, "[SYN]", labels, objs, allow_unresolved=True, cap=0),
                        _BASE_PROMPTS["integration_user"]("as_asker", item, "[SYN]", labels, objs, allow_unresolved=True, cap=0)),
        "r4_vote": (r4_vote_user_co("as_asker", item, role, "[SYN]", "ACCEPT_WITH_MODIFICATION", "[OBJ]", "[PROP]", cap=0),
                    _BASE_PROMPTS["r4_vote_user"]("as_asker", item, role, "[SYN]", "ACCEPT_WITH_MODIFICATION", "[OBJ]", "[PROP]", cap=0)),
    }
    for rd, (co, base) in pairs.items():
        check(f"{rd}: coalition prompt == base prompt with the count words recounted, byte for byte",
              co == recount(base) and co != base)
    check("SYNTHESIS_SYSTEM / INTEGRATION_SYSTEM == base with the count words recounted",
          SYNTHESIS_SYSTEM_CO == recount(_BASE_SYNTHESIS_SYSTEM)
          and INTEGRATION_SYSTEM_CO == recount(_BASE_INTEGRATION_SYSTEM)
          and rcd.SYNTHESIS_SYSTEM is SYNTHESIS_SYSTEM_CO and rcd.INTEGRATION_SYSTEM is INTEGRATION_SYSTEM_CO)
    check("R0 preamble == base with exactly the word three -> four",
          R0_PREAMBLE_CO == _BASE_R0_PREAMBLE.replace("three", "four")
          and R0_PREAMBLE_CO.count("four") == 1)
    check("no coalition prompt still carries any stale three-seat phrase (the left side of "
          "COUNT_WORDS); 'The other three participants' is the correct four-seat wording",
          not any(a in co for co, _ in pairs.values() for a, _ in COUNT_WORDS)
          and not any(a in t for t in (SYNTHESIS_SYSTEM_CO, INTEGRATION_SYSTEM_CO, R0_PREAMBLE_CO)
                      for a, _ in COUNT_WORDS))
    check("r4 keeps 'In two or three sentences' (a count that is NOT a seat count)",
          "In two or three sentences" in pairs["r4_vote"][0])
    check("r1 shows the ally its THREE neighbours and not itself",
          all(f"[{rid} PRIOR]" in pairs["r1"][0] for rid in ROLE_IDS_CO if rid != role.role_id)
          and f"[{role.role_id} PRIOR]" not in pairs["r1"][0])
    check("integration prompt lists the four `_co` ids for the ADDRESSED line",
          f"(from: {', '.join(ROLE_IDS_CO)})" in pairs["integration"][0])
    check("extract_addressed resolves the `_co` ids and ignores the bare embodied ids",
          rcd.extract_addressed(f"{rcd.ADDRESSED_MARKER} writer_ally_co, counterparty")
          == ("writer_ally_co",))
    try:
        r3_label_user_co("as_asker", item, role, "o", "s", cap=0, stake_nudge=True)
        raised = False
    except ValueError:
        raised = True
    check("r3/r4 builders raise on any stake knob (belt to the CLI ban's braces)", raised)
    orth = rcd.assert_arm_orthogonality([item])
    check("arm orthogonality holds on every round for all four seats",
          orth["prompts_checked"] == len(rcd.ARM_ORDER) * 4 * 5 and orth["roles"] == list(ROLE_IDS_CO))

    # ---- stake_undermined for the ally ----------------------------------------
    check("ally is undermined by an at-fault verdict exactly as the writer's advocate is",
          all(rcd.stake_undermined("writer_ally_co", v) == rcd.stake_undermined("writer_advocate_co", v)
              for v in ("YTA", "NTA", "ESH", "NAH", rcd.UNRESOLVED, rcd.NOVERDICT))
          and rcd.stake_undermined("writer_ally_co", "YTA") is True
          and rcd.stake_undermined("writer_ally_co", "NTA") is False
          and rcd.stake_undermined("writer_ally_co", rcd.UNRESOLVED) is None)
    check("counterparty_co is undermined by a not-at-fault verdict; neutral never",
          rcd.stake_undermined("counterparty_co", "NTA") is True
          and rcd.stake_undermined("neutral_adjudicator_co", "YTA") is False)
    vr = [{"role_id": "writer_advocate_co", "role_stake": "not_at_fault", "objected_r3": 1},
          {"role_id": "writer_ally_co", "role_stake": "not_at_fault", "objected_r3": 1},
          {"role_id": "counterparty_co", "role_stake": "at_fault", "objected_r3": 1},
          {"role_id": "neutral_adjudicator_co", "role_stake": "none", "objected_r3": 1}]
    check("against_interest_count: NTA synthesis -> writer + ally against interest, not the counterparty",
          against_interest_count(vr, "NTA") == (2, ("writer_advocate_co", "writer_ally_co")))
    check("against_interest_count: YTA synthesis -> only the counterparty; neutral never counts",
          against_interest_count(vr, "YTA") == (1, ("counterparty_co",)))
    check("against_interest_count is None (excluded) on a non-codable synthesis",
          against_interest_count(vr, rcd.UNRESOLVED) == (None, ()))

    # ---- cache namespaces on the REAL output dir ----------------------------------
    names = os.listdir(OUT_DIR)
    clean, counts = cache_is_clean(out_dir=OUT_DIR)
    own = {k: v for k, v in counts.items() if not k.startswith(("any:", "tag:"))}
    anyc = {k: v for k, v in counts.items() if k.startswith("any:")}
    check(f"no foreign (stub / selftest / model 'm') file carries this runner's names in "
          f"{OUT_DIR.name}", foreign_files(names) == [])
    check(f"every coalition namespace is EMPTY, or holds only this cell's registered "
          f"footprint (all seats + moderator populated together) {anyc}",
          all(v == 0 for v in anyc.values()) or all(v > 0 for v in anyc.values()))
    check(f"cache_is_clean(grok) on the real dir agrees with the namespace counts {own}",
          clean == (sum(own.values()) == 0))
    snapshot = sorted(n for n in names if any(n.endswith(f"_{rid}.json") for rid in ROLE_IDS_CO)
                      or f"_{MODERATOR_PREFIX}-" in n or n.startswith(f"{DEFAULT_TAG}_"))

    # ---- CLI refusals (cache dir redirected; nothing can be spent) ----------
    real_gen, real_out = rcd.generate_any, rcd.OUT_DIR
    try:
        rcd.generate_any = _stub_generate  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as td:
            rcd.OUT_DIR = Path(td)
            try:
                for flag in ("--stake-nudge", "--smoke", "--full", "--r3r4-thinking-budget=1024"):
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        rc = main(["--dry-run", flag])
                    check(f"main() refuses {flag} (exit 4) before building anything",
                          rc == 4 and "DRY RUN" not in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rcs = [main(["--dry-run", "--tag", t]) for t in
                           ("cg_deliberation", "cg_deliberation_stdcot", "cg_deliberation_multivendor",
                            "cg_deliberation_nperspective", "cg_deliberation_modsonnet")]
                check("main() refuses every sibling cell's tag, listed or not (exit 2, nothing written)",
                      rcs == [2] * 5 and "DRY RUN" not in buf.getvalue()
                      and not any(n.endswith("_dryrun.json") for n in os.listdir(td)))
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_nu = main(["--dry-run", "--no-unresolved"])
                    rc_tc = main(["--dry-run", "--transcript-cap", "400"])
                check("main() refuses --no-unresolved and --transcript-cap (exit 4) before building anything",
                      rc_nu == 4 and rc_tc == 4 and "DRY RUN" not in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--dry-run", "--n-yta", "4", "--n-nta", "4"])
                out = buf.getvalue()
                check("main(--dry-run) exits 0, prints the 22-call breakdown and the NEW-calls line",
                      rc == 0 and "calls per cell         22" in out
                      and "4 seats x r0,r1,r2 = 12" in out and "every one NEW" in out)
                check("main(--dry-run) on 8 items x 2 arms counts 16 cells x 22 = 352 calls",
                      "TOTAL GENERATION CALLS 352" in out)
                dj = json.loads((Path(td) / f"{DEFAULT_TAG}_dryrun.json").read_text())
                check("dry-run json: calls_per_cell 22, total_calls 352, priced grok",
                      dj["calls_per_cell"] == 22 and dj["total_calls"] == 352
                      and dj["per_model"]["grok-4-1-fast-reasoning"]["priced"])
                check("dry-run json: r1 prompt is longer than a three-seat r1 would be "
                      "(three neighbours quoted, not two)",
                      dj["per_model"]["grok-4-1-fast-reasoning"]["rounds"]["r1"]["prompt_tokens_each"]
                      > dj["per_model"]["grok-4-1-fast-reasoning"]["rounds"]["r0"]["prompt_tokens_each"] * 2)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--n-yta", "2", "--n-nta", "2"])
                check("main() without --run prints the cost model and REFUSES (exit 2)",
                      rc == 2 and "DRY RUN" in buf.getvalue() and "NOT RUNNING" in buf.getvalue()
                      and "--run" in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--dry-run", "--n-yta", "2", "--n-nta", "2",
                               "--models", "no-such-deployment"])
                check("main(--dry-run) on an unpriced model prints the WARNING and exits 0; "
                      "the --run path refuses it (exit 2), verified in source",
                      rc == 0 and "NO PRICE ON FILE" in buf.getvalue()
                      and "not in PRICES" in inspect.getsource(main))
                check("the run path enforces cache_is_clean with --resume as the only override, "
                      "and prints GUARD FAILED on a failed guard (both verified in source)",
                      "cache_is_clean(" in inspect.getsource(main)
                      and "known.resume" in inspect.getsource(main)
                      and "GUARD FAILED" in inspect.getsource(rcd.main))
                check("dry runs wrote nothing into any coalition namespace of the tempdir",
                      cache_is_clean(out_dir=Path(td))[0])

                # ---- end-to-end protocol against the stub, cache in the tempdir ----
                items = rcd.load_items(source="scruples", n_yta=99, n_nta=150,
                                       min_votes=50, min_consensus=0.90, seed=44)
                sub = rcd.subset_items(items, 8, 8)
                rows, votes, calls = [], [], []
                for it in sub:
                    for arm in (rcd.THIRD_PERSON, rcd.AS_ASKER):
                        row, vr_, cr = rcd.run_deliberation("stub-model", arm, it, 0,
                                                            scaffold="narrative_cot")
                        rows.append(row); votes.extend(vr_); calls.extend(cr)
                n_files = len(os.listdir(td)) - 1  # minus the dryrun json
                check("smoke: 32 debates x 22 calls, one cache file per call in the TEMP dir",
                      len(rows) == 32 and len(calls) == 32 * 22 and n_files == 32 * 22)
                from collections import Counter
                rc_ = Counter(c["round"] for c in calls)
                check("smoke: call mix 4/4/4 + 1 + 4 + 1 + 4",
                      all(rc_[r] == 32 * 4 for r in rcd.AGENT_ROUNDS)
                      and all(rc_[r] == 32 for r in rcd.MODERATOR_ROUNDS))
                check("smoke: every call record carries a `_co` seat or the modcoalition id",
                      all(c["role_id"].endswith(SUFFIX) or c["role_id"].startswith(MODERATOR_PREFIX + "-")
                          for c in calls))
                check("smoke: every group verdict, R3 label and R4 vote parsed",
                      all(r["verdict"] != rcd.NOVERDICT for r in rows)
                      and all(v["vote_parsed"] and v["r3_label_parsed"] for v in votes))
                check("smoke: rows carry n_agents 4 and 22 calls",
                      all(r["n_agents"] == 4 and r["n_calls"] == 22 for r in rows))
                check("smoke: the four vote columns are filled from the `_co` seats",
                      all(r["vote_writer_advocate"] in ("ACCEPT", "REJECT")
                          and r["vote_writer_ally"] in ("ACCEPT", "REJECT")
                          and r["vote_counterparty"] in ("ACCEPT", "REJECT")
                          and r["vote_neutral_adjudicator"] in ("ACCEPT", "REJECT") for r in rows))
                check("smoke: R3 arithmetic sums to 4 on every row",
                      all(r["n_r3_accept"] + r["n_r3_accept_with_mod"] + r["n_r3_reject"]
                          + r["n_r3_unparsed"] == 4 for r in rows))
                by_dbt = {}
                for v in votes:
                    by_dbt.setdefault((v["item_id"], v["arm"]), {})[v["role_id"]] = v
                check("smoke: ally stake_undermined == writer's advocate's in every debate",
                      all(d["writer_ally_co"]["stake_undermined"] == d["writer_advocate_co"]["stake_undermined"]
                          for d in by_dbt.values()))
                check("smoke: planted phi = +1 -- ally objects exactly when the writer's advocate does",
                      all(d["writer_ally_co"]["objected_r3"] == d["writer_advocate_co"]["objected_r3"]
                          for d in by_dbt.values()))
                n_ai = [r["n_against_interest_r3"] for r in rows]
                check("smoke: n_against_interest_r3 is an int on every row and takes every value 0, 1 "
                      f"and 2 (planted counterparty and writer-side against-interest objections) {Counter(n_ai)}",
                      all(isinstance(x, int) for x in n_ai) and {0, 1, 2} <= set(n_ai))
                check("smoke: a count of 2 is always the writer + ally pair on a not-at-fault synthesis, "
                      "a count of 1 on an at-fault synthesis is always the counterparty",
                      all(r["against_interest_roles"] == "writer_advocate_co|writer_ally_co"
                          and r["synthesis_verdict"] in rcd.NOT_AT_FAULT_VERDICTS
                          for r in rows if r["n_against_interest_r3"] == 2)
                      and all(r["against_interest_roles"] == "counterparty_co"
                              for r in rows if r["n_against_interest_r3"] == 1
                              and r["synthesis_verdict"] in rcd.AT_FAULT_VERDICTS))
                check("smoke: both synthesis verdict classes occur (the tie rule lets the neutral decide)",
                      {r["synthesis_verdict"] for r in rows} >= {"YTA", "NTA"})
                check("smoke: against_interest_roles agrees with the count",
                      all(len([s for s in r["against_interest_roles"].split("|") if s])
                          == r["n_against_interest_r3"] for r in rows))
                check("smoke: every row key reaches disk (ROW_FIELDS covers the row)",
                      set().union(*[set(r) for r in rows]) <= set(rcd.ROW_FIELDS))
                conc = rcd.stake_concentration(votes, n_boot=100, seed=1, roles=rcd.STAKE_BEARING_ROLES)
                check("smoke: within-role stake concentration is identified over the three `_co` stake seats",
                      conc is not None and conc["roles"] == sorted(STAKE_BEARING_CO))
                ind = rcd.vote_independence(rows)
                check("smoke: vote_independence counts 0..4 rejections",
                      ind is not None and sorted(ind["observed"]) == [0, 1, 2, 3, 4])
                clean_td, cts = cache_is_clean(models=("stub-model",), out_dir=Path(td))
                check("cache_is_clean(stub-model, tempdir) is DIRTY after the smoke and every "
                      "namespace is populated (the run path would refuse without --resume)",
                      not clean_td and all(v == 32 * (5 if not k.endswith(MODERATOR_PREFIX) else 2)
                                           for k, v in cts.items() if k.startswith("stub-model:")))
                check("cache_is_clean(grok, tempdir) is CLEAN: per-model footprint, another "
                      "deployment can share the namespace",
                      cache_is_clean(models=("grok-4-1-fast-reasoning",), out_dir=Path(td))[0])
                check("foreign_files() catches the stub's files in the tempdir",
                      len(foreign_files(os.listdir(td))) == 32 * 22)
                # guard literal on a planted truncation
                grows = [rcd.call_guard_row(dict(c, finish_reason="max_tokens")) for c in calls]
                rep = rcd.truncation_report(grows)
                check("truncation guard fails on planted truncation (main prints GUARD FAILED "
                      "and exits 4 on this)", not rep["pass"])
            finally:
                rcd.OUT_DIR = real_out

        # ---- the --run path end to end (stubbed, fresh tempdir): an errored debate
        # is dropped by rcd.run; the completeness guard catches it, quarantines the
        # CSVs and exits 4; --run --resume then completes the cell ----------------
        with tempfile.TemporaryDirectory() as td2:
            rcd.OUT_DIR = Path(td2)
            rcd.PRICES["stub-model"] = (1.0, 1.0)
            try:
                two = rcd.subset_items(items, 1, 1)
                victim = two[0]
                as_asker_lead = rcd.build_user_turn(rcd.AS_ASKER, "")[:40]

                def _erroring_stub(model, system, user, *, sample_idx=0, max_tokens=512, **kw):
                    if ("Respond to this synthesis" in user and as_asker_lead in user
                            and victim.post_text[:200] in user):
                        raise RuntimeError("simulated vendor outage on one as_asker r3 call")
                    return _stub_generate(model, system, user, sample_idx=sample_idx,
                                          max_tokens=max_tokens, **kw)
                rcd.generate_any = _erroring_stub  # type: ignore[assignment]
                run_args = ["--run", "--models", "stub-model", "--n-yta", "1", "--n-nta", "1",
                            "--workers", "1", "--n-boot", "20"]
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_a = main(run_args)
                out_a = buf.getvalue()
                names_a = sorted(os.listdir(td2))
                comp_a = json.loads((Path(td2) / f"{DEFAULT_TAG}_completeness.json").read_text()) \
                    if (Path(td2) / f"{DEFAULT_TAG}_completeness.json").exists() else {}
                check("run path: one errored as_asker debate is DROPPED by rcd.run (3 rows for 4 launched); "
                      "the completeness guard fails, prints GUARD FAILED, exits 4",
                      rc_a == 4 and "GUARD FAILED" in out_a and comp_a.get("pass") is False
                      and comp_a.get("n_expected") == 4 and comp_a.get("n_rows") == 3
                      and comp_a.get("missing") == [["stub-model", rcd.AS_ASKER, victim.item_id, 0]]
                      and comp_a.get("summary_n_items") == 2)
                check("run path: the sidecar's per-(model, arm) counts expose the arm-correlated attrition "
                      "(third_person 2/2, as_asker 1/2)",
                      comp_a.get("per_model_arm", {}).get(f"stub-model|{rcd.THIRD_PERSON}") == {"expected": 2, "rows": 2}
                      and comp_a.get("per_model_arm", {}).get(f"stub-model|{rcd.AS_ASKER}") == {"expected": 2, "rows": 1})
                check("run path: on exit 4 the CSVs are quarantined as <tag>_GUARD_FAILED_rows/_votes.csv and "
                      "nothing is left under the readable tag",
                      f"{DEFAULT_TAG}_GUARD_FAILED_rows.csv" in names_a
                      and f"{DEFAULT_TAG}_GUARD_FAILED_votes.csv" in names_a
                      and f"{DEFAULT_TAG}_rows.csv" not in names_a and f"{DEFAULT_TAG}_votes.csv" not in names_a)
                rcd.generate_any = _stub_generate  # type: ignore[assignment]
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_b = main(run_args)
                check("run path: a second --run without --resume is refused as DIRTY (exit 3)",
                      rc_b == 3 and "not clean" in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_c = main(run_args + ["--resume"])
                comp_c = json.loads((Path(td2) / f"{DEFAULT_TAG}_completeness.json").read_text())
                import csv as _csv
                n_rows_c = len(list(_csv.DictReader((Path(td2) / f"{DEFAULT_TAG}_rows.csv").open())))
                check("run path: --run --resume replays the cached calls, completes the missing debate, "
                      "passes both guards (exit 0) and leaves 4 rows under the readable tag",
                      rc_c == 0 and comp_c["pass"] and comp_c["n_rows"] == 4 and n_rows_c == 4
                      and "GUARD FAILED" not in buf.getvalue())
            finally:
                rcd.PRICES.pop("stub-model", None)
                rcd.OUT_DIR = real_out
    finally:
        rcd.generate_any = real_gen
    after = sorted(n for n in os.listdir(OUT_DIR) if any(n.endswith(f"_{rid}.json") for rid in ROLE_IDS_CO)
                   or f"_{MODERATOR_PREFIX}-" in n or n.startswith(f"{DEFAULT_TAG}_"))
    check("nothing landed in the REAL output directory under any coalition name",
          after == snapshot)
    check("registered panel: the grok content-filter screen leaves 210 of 249 items",
          len([i for i in items if i.item_id not in rcd.filter_prone_items("grok-4-1-fast-reasoning")]) == 210)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        add_help=False,
        description="Addendum 16.19 coalition advocates: four seats, edges on. "
                    "Generates NOTHING unless --run is passed.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true", help="REQUIRED to generate; spends money")
    ap.add_argument("--resume", action="store_true",
                    help="proceed despite a non-empty cache footprint for these models; "
                         "only valid when the briefs are known unchanged")
    ap.add_argument("--verify-cache-clean", action="store_true")
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="comma-separated agent models; each must be in "
                         "run_crowdgold_deliberation.PRICES to --run")
    ap.add_argument("--moderator-model", default=None)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    known, rest = ap.parse_known_args(argv)
    if known.selftest:
        return _selftest()

    for a in rest:
        if a.split("=", 1)[0] in BANNED_FLAGS:
            print(f"\nERROR: {a} is banned in the coalition cell (stake knobs inject "
                  "assigned-interest text into the readout rounds; --smoke/--full/--pilot "
                  "change the panel or the generator; --no-unresolved/--transcript-cap change "
                  "prompt text the cache key cannot see). Refusing.\n")
            return 4
    if not tag_ok(known.tag):
        print(f"\nERROR: --tag {known.tag!r} must start with {DEFAULT_TAG!r}; any other tag "
              "could overwrite a sibling cell's CSVs.\n")
        return 2
    models = [m.strip() for m in known.models.split(",") if m.strip()]
    if not models:
        print("\nERROR: --models is empty\n")
        return 2

    clean, counts = cache_is_clean(models, known.moderator_model)
    if known.verify_cache_clean:
        print(f"cache files per coalition namespace: {counts}")
        print("CLEAN -- every call will be a genuine generation." if clean else
              "DIRTY -- refusing; a previous run under different briefs could be "
              "replayed silently.")
        return 0 if clean else 3

    install()
    fwd = list(rest) + ["--models", ",".join(models), "--tag", known.tag]
    if known.moderator_model:
        fwd += ["--moderator-model", known.moderator_model]
    print(f"[coalition] seats {rcd.ROLE_ORDER}  stakes "
          f"{tuple(r.stake for r in rcd.ROLES)}  moderator prefix {MODERATOR_PREFIX}  "
          f"tag {known.tag}  edges ON")
    print(f"[coalition] cache plan: all {rcd.CALLS_PER_CELL} calls per debate are NEW; "
          "no seat or moderator record can replay from any existing run")

    if known.dry_run or not known.run:
        rc = rcd.main(fwd + ["--dry-run"])
        if known.dry_run:
            return rc
        print("\n*** NOT RUNNING: pass --run to generate (this spends money). "
              "Nothing was generated. ***\n")
        return 2

    unpriced = [m for m in models + ([known.moderator_model] if known.moderator_model else [])
                if m not in rcd.PRICES]
    if unpriced:
        print(f"\nERROR: {unpriced} not in PRICES; add a price to "
              "run_crowdgold_deliberation.PRICES before spending (a $0.00 dry-run is a "
              "placeholder, not a ceiling).\n")
        return 2
    if not clean and not known.resume:
        print(f"\nERROR: cache is not clean for {models} {counts}.\n"
              "A previous partial run under possibly-different briefs would be replayed "
              "silently (role briefs are not covered by parent_sha).\n"
              "Pass --resume only if you are certain the briefs are unchanged.\n")
        return 3
    if not clean:
        print(f"  --resume: {counts} existing records will be served from cache")
    print(f"[coalition] delegating to run_crowdgold_deliberation with: {' '.join(fwd)}")
    rc = rcd.main(fwd)
    return finish_run(known.tag, rc, models, rest)


if __name__ == "__main__":
    raise SystemExit(main())
