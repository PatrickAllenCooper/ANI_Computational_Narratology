"""
scripts/run_crowdgold_nperspective.py -- Addendum 16.21: n perspectives, the
k-loser generalisation. A FOURTH seat that speaks for a NAMED THIRD PARTY on
the registered AITA deliberation.

THE QUESTION

The dissent sensor is a deviation detector on an anti-correlated advocate
edge that exists only when a verdict has exactly ONE loser (16.12, 16.15).
With two advocates the number of losers k is 1 on every one-loser verdict and
the theory has only ever been read at k=1 (sensor works) against the
both-party verdicts (flooding) and Dilemmas / games (silence). Seating a
THIRD advocate -- for a person the account itself names, affected by how the
conflict is judged but not the party in conflict -- makes k=2 reachable on a
one-loser verdict: the verdict goes against one of the two parties AND
against the third party's stated interest. The generalised sensor is the
AGAINST-INTEREST objection count: objections from advocates the verdict
favours. The k-loser prediction table (pre-declared; scripts/
analyze_nperspective.py prints it next to the measured cells):

  k = 0   every advocate favoured      SILENCE   nothing deviates, count ~0
  k = 1   n-1 favoured                 SENSOR    count rare, error lift > 0
  k = 2   n-2 = 1 favoured             SENSOR    count from one seat; the
                                                 same lift with less coverage
  k = n   no advocate favoured         FLOODING  plain objections flood, the
                                                 against-interest count is
                                                 structurally empty

On THIS instrument, under the registered published collapse (ESH -> at
fault, NAH -> not at fault) and two opposed party advocates, exactly one of
writer / counterparty is undermined by every codable verdict, so k = 1 +
[third party undermined] and k in {1, 2}: k=0 and k=n=3 are unreachable by
construction and are reported as such, not estimated. Verdict type (one-loser
YTA/NTA vs both-party ESH/NAH) is carried as the second axis, since 16.12
established both-party verdicts as the flooding stratum.

THE ROLE SET (four seats, edges ON, everyone reads everyone)

  writer_advocate_np        the registered writer brief VERBATIM     stake not_at_fault
  counterparty_np           the registered counterparty brief VERBATIM stake at_fault
  third_party_advocate_np   NEW: speaks for the third party the screen stake PER ITEM
                            record names; the party's identity, interest  (YTA / NTA /
                            and which finding serves them are read from   none, from
                            divergence_study_outputs/third_party_screen  the screen)
                            .json (scripts/screen_third_party.py) and
                            quoted in a block after the brief; may not
                            invent facts
  neutral_adjudicator_np    the registered neutral brief VERBATIM      stake none

  moderator prefix  modnp          tag  cg_deliberation_nperspective

Panel: the registered AITA panel (grok-4-1-fast-reasoning, --n-yta 99
--n-nta 150, seed 44, the content-filter screen to 210 items, third_person +
as_asker, --samples 1, narrative_cot, 2560/1024/1024/512 caps) RESTRICTED to
the items the third-party screen selected (install() wraps
rcd.subset_items: registered subset, then intersect with the screened ids;
the runner's own content-filter screen still applies after that). The
per-item stake of the third party (which finding, YTA or NTA, favours them,
or none) is recorded on every row and vote from the screen record.

WHAT N_AGENTS = 4 CHANGES, AND WHERE IT IS FIXED (here; the same fixes as
scripts/run_crowdgold_coalition.py, whose four-seat wording this cell copies
byte for byte -- the selftest asserts equality with that module when it is
importable, so the two four-seat cells differ ONLY in the fourth seat)

  * the seven prompt builders and the two moderator system prompts are
    re-issued with ONLY the participant-count words changed (COUNT_WORDS);
    the selftest proves each equals the base prompt with those substitutions
    applied and nothing else -- for the three verbatim seats. The third-party
    seat's prompt is the same text plus the per-item THIRD PARTY block.
  * CALLS_PER_CELL becomes 4+4+4+1+4+1+4 = 22; install() recomputes it.
  * the outcome row's vote columns are filled from the `_np` ids and the row
    gains vote_third_party_advocate, the third party's description / stake /
    stake source / screen sha, k_losers_synthesis, k_losers_final,
    n_favoured_synthesis, n_against_interest_r3, against_interest_roles and
    verdict_type_synthesis. ROW_FIELDS is extended so they reach disk.
  * stake_undermined() is role-driven and cannot see the item, so the
    third-party seat's Role carries stake "per_item" (the base function
    returns False for it) and run_deliberation is wrapped to overwrite that
    seat's vote-row role_stake / stake_undermined from the screen record
    (blank when the third party has no stake, so it never dilutes G3).
    STAKE_BEARING_ROLES is re-pointed at the three advocate seats.

THE GENERALISED SENSOR

An AGAINST-INTEREST objection is an R3 objection (ACCEPT_WITH_MODIFICATION
or REJECT) lodged against a synthesis verdict that SERVES the seat's own
stake. n_against_interest_r3 counts them over the advocates the synthesis
favours (0..n-k; blank when the synthesis verdict is not codable). k is the
number of advocates the synthesis verdict undermines. Both are written per
row and read out by scripts/analyze_nperspective.py.

CACHE PLAN (call_cache_path keys on model/scaffold/arm/item/sample/ROLE ID
and NOT on brief text; the moderator id comes from moderator_role_id)

  EVERY ONE OF THE 22 CALLS PER DEBATE IS A NEW GENERATION. All four seats
  carry `_np` role ids and the moderator the `modnp` prefix, so no cache path
  can coincide with an embodied, unembodied, topology, coalition or game
  record; nor could a replay be legitimate, since a four-seat prompt says
  "one of four participants" and quotes three neighbours. The dry-run counts
  all 22 calls per debate and the selftest verifies every namespace is empty
  (or holds only this cell's own registered footprint) with the no-foreign-
  file idiom. cache_is_clean() is enforced on the --run path per model;
  --resume is the only override and is valid only when the briefs AND the
  screen file are known unchanged.

  THE SCREEN IS PROMPT TEXT THE CACHE KEY CANNOT SEE. The third-party block
  changes with the screen record, and a re-screened item under the same
  role id would otherwise replay the old seat's r0. install() therefore
  wraps rcd.do_call so every third-party-seat record carries the screen
  record (description | interest | stake) in its parent_sha: a changed
  record misses the cache and regenerates, and every downstream record that
  quoted the old r0 goes stale through the existing parent_sha chain.

GUARDS AFTER THE DELEGATED RUN (this wrapper's, on top of the base runner's)

  * COMPLETENESS. rcd.run swallows an exception raised inside
    run_deliberation with one console ERROR line and omits that debate from
    rows / votes / calls; the base round and outcome guards and every
    analyzer denominator are then computed on the survivors, so a vendor
    outage concentrated in one arm is arm-correlated attrition that nothing
    reports. After rcd.main returns, this wrapper rebuilds the launched key
    set (models x arms x screened-and-filtered items x samples, from the
    panel arguments it forwarded and the installed subset_items), compares
    it with the rows CSV, writes <tag>_completeness.json (n_expected,
    n_rows, n_missing, the missing keys, per-(model, arm) counts, the
    summary's n_items cross-check) and prints the literal GUARD FAILED and
    exits 4 on any shortfall. scripts/analyze_nperspective.py refuses a
    cell whose sidecar or summary is missing or whose guards failed.
  * QUARANTINE. The base runner writes its CSVs BEFORE its guard verdict.
    On exit 4 (its round / outcome guard, or the completeness guard) the
    rows and votes CSVs are renamed <tag>_GUARD_FAILED_rows.csv /
    _votes.csv so nothing under the readable tag can be tabulated; re-run
    with --run --resume once the cause is fixed (cached calls replay).
  * TAG RULE. --tag must start with cg_deliberation_nperspective: rcd.main
    writes <tag>_rows.csv / _votes.csv / _summary.json (and _dryrun.json on
    --dry-run) unconditionally, and a deny-list cannot enumerate every
    sibling cell (stdcot, multivendor, modsonnet, modhaiku, gpt4o, ...).
  * BANNED FLAGS. --no-unresolved (and --transcript-cap) are refused
    alongside the stake knobs and --keep-filtered: each changes the r0-r2 /
    synthesis / integration prompt text WITHOUT changing call_cache_path or
    the r0 parent_sha (post text only for the three verbatim seats), so a
    --run --resume after a partial run under the other setting would
    silently serve records generated for a different prompt. The cell is
    defined at the registered allow_unresolved=True, transcript cap 0.

Nothing here generates unless the literal flag --run is given. Omitting it
prints the cost model and exits 2. --dry-run prints the cost model and exits
0. --selftest stubs generation and redirects the cache directory that
call_cache_path reads (run_crowdgold_deliberation.OUT_DIR) to a tempdir.

Usage
-----
  python -m scripts.screen_third_party                          # first: the $0 screen
  python -m scripts.run_crowdgold_nperspective --selftest
  python -m scripts.run_crowdgold_nperspective --verify-cache-clean
  python -m scripts.run_crowdgold_nperspective --dry-run
  python -m scripts.run_crowdgold_nperspective --run            # SPENDS MONEY
  python -m scripts.run_crowdgold_nperspective --run --resume   # briefs + screen unchanged only
  python -m scripts.run_crowdgold_nperspective --dry-run --models grok-4-1-fast-reasoning,<new-deployment>

Exit codes: 0 ok, 1 no rows, 2 bad config / --run not given / unpriced model
/ tag not under cg_deliberation_nperspective / screen missing, 3 cache not
clean (pass --resume), 4 guard failed (round, outcome or completeness) or a
banned flag.
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
from scripts.screen_third_party import SCREEN_PATH, screened_items
from scripts.load_scruples import AT_FAULT_VERDICTS, NOT_AT_FAULT_VERDICTS, sha256_text

DEFAULT_TAG = "cg_deliberation_nperspective"
MODERATOR_PREFIX = "modnp"
SUFFIX = "_np"
DEFAULT_MODELS = "grok-4-1-fast-reasoning"
THIRD_PARTY_ID = "third_party_advocate" + SUFFIX
THIRD_PARTY_LABEL = "Third Party's Advocate"

#: Sibling cells' tags, listed so the selftest can prove the PREFIX RULE
#: (tag_ok: --tag must start with DEFAULT_TAG) refuses every one of them.
SIBLING_TAGS = ("cg_deliberation", "cg_deliberation_noedge", "cg_deliberation_identical",
                "cg_deliberation_identical_noedge", "cg_deliberation_unembodied",
                "cg_deliberation_coalition", "cg_deliberation_stdcot",
                "cg_deliberation_multivendor", "cg_deliberation_modsonnet",
                "cg_deliberation_modhaiku", "cg_deliberation_gpt4o",
                "cg_deliberation_gpt4o_screen")


def tag_ok(tag: str) -> bool:
    """The only tags this runner may write under: DEFAULT_TAG or a suffix of it."""
    return tag.startswith(DEFAULT_TAG)

#: Captured ONCE at import, before install() mutates rcd.
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
_BASE_DO_CALL = rcd.do_call
_BASE_SUBSET_ITEMS = rcd.subset_items

THIRD_PARTY_BRIEF = (
    "You speak for a third party -- a person the account itself names who is "
    "neither the writer nor the party the writer is in conflict with, but "
    "whose situation turns on how this conflict is judged. Who that person is, "
    "and their interest in the outcome, is stated in the record below; you "
    "speak for no one else. Put their case as strongly as the account itself "
    "allows, including whatever the two parties in conflict are passing over. "
    "You may not invent facts that are not in the account, and you may not "
    "assert an interest the record and the account do not support."
)

#: Which finding about the writer serves the third party -> the stake
#: sentence quoted in the block. YTA and NTA reuse the registered advocates'
#: own sentence structure; none says so explicitly.
STAKE_SENTENCE = {
    "YTA": ("A finding that the writer is not in the wrong is a finding against "
            "the party you speak for."),
    "NTA": ("A finding that the writer is in the wrong is a finding against the "
            "party you speak for."),
    "none": ("The record does not settle which finding serves this person; argue "
             "only for what the account shows their situation requires."),
}
#: screen stake -> Role.stake vocabulary (what stake_undermined reads)
STAKE_TO_ROLE_STAKE = {"YTA": "at_fault", "NTA": "not_at_fault", "none": "none"}
BLOCK_BEGIN = "--- THE PARTY YOU SPEAK FOR (from the screen record) ---"
BLOCK_END = "--- END ---"


def build_roles() -> tuple[Role, ...]:
    w, c, n = (_EMB_BY_ID["writer_advocate"], _EMB_BY_ID["counterparty"],
               _EMB_BY_ID["neutral_adjudicator"])
    return (
        Role(role_id="writer_advocate" + SUFFIX, label=w.label, paper_role=w.paper_role,
             stake=w.stake, brief=w.brief),
        Role(role_id="counterparty" + SUFFIX, label=c.label, paper_role=c.paper_role,
             stake=c.stake, brief=c.brief),
        Role(role_id=THIRD_PARTY_ID, label=THIRD_PARTY_LABEL, paper_role="third_party",
             stake="per_item", brief=THIRD_PARTY_BRIEF),
        Role(role_id="neutral_adjudicator" + SUFFIX, label=n.label, paper_role=n.paper_role,
             stake=n.stake, brief=n.brief),
    )


ROLES_NP: tuple[Role, ...] = build_roles()
ROLE_IDS_NP = tuple(r.role_id for r in ROLES_NP)
ADVOCATES_NP = ("writer_advocate" + SUFFIX, "counterparty" + SUFFIX, THIRD_PARTY_ID)
STAKE_BEARING_NP = ADVOCATES_NP

for _r in ROLES_NP:
    assert POST_DELIM not in _r.brief, f"role brief {_r.role_id} has a blank line"
del _r

#: The count-word substitutions, and NOTHING else, that separate a four-seat
#: prompt from the base prompt it mirrors. Copied from
#: run_crowdgold_coalition.COUNT_WORDS (the selftest asserts equality).
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


SYNTHESIS_SYSTEM_NP = recount(_BASE_SYNTHESIS_SYSTEM)
INTEGRATION_SYSTEM_NP = recount(_BASE_INTEGRATION_SYSTEM)
R0_PREAMBLE_NP = _BASE_R0_PREAMBLE.replace("three", "four")

NPERSPECTIVE_NOTE = """\
N PERSPECTIVES (Addendum 16.21). A fourth seat, third_party_advocate_np, speaks
for a person the account names who is neither the writer nor the counterparty
(divergence_study_outputs/third_party_screen.json, scripts/
screen_third_party.py). Its stake is PER ITEM (which finding serves that
person: YTA, NTA or none) and is recorded on every row. The other three seats
are the registered briefs verbatim under `_np` ids. The k-loser theory
predicts the against-interest objection count works at k=1 and k=2 and is
structurally empty at k=n; k=0 and k=3 are unreachable here by construction.
"""


# ---------------------------------------------------------------------------
# The screen record: what the third-party seat is told, and its sha
# ---------------------------------------------------------------------------

#: item_id -> {"description", "interest", "stake", "stake_source", "sha"}
SCREEN: dict[str, dict] = {}
SCREEN_FILE_SHA = ""


def screen_record(raw: dict) -> dict:
    stake = str(raw.get("stake_final") or "none").upper()
    if stake not in STAKE_SENTENCE:
        stake = "none"
    desc = (raw.get("third_party_description") or "").strip()
    interest = (raw.get("third_party_interest") or "(not stated)").strip()
    rec = {"description": desc, "interest": interest, "stake": stake,
           "stake_source": raw.get("stake_source") or "heuristic"}
    rec["sha"] = sha256_text("\x00".join((desc, interest, stake)))[:16]
    return rec


def set_screen(records: dict[str, dict], file_sha: str = "") -> None:
    """Install a screen map (item_id -> raw screened-in record). The selftest
    uses this with synthetic records; main() uses load_screen()."""
    global SCREEN_FILE_SHA
    SCREEN.clear()
    for iid, raw in records.items():
        rec = screen_record(raw)
        if rec["description"]:
            SCREEN[iid] = rec
    SCREEN_FILE_SHA = file_sha


def load_screen(path: Path = SCREEN_PATH) -> dict:
    """Read the screen JSON; returns the parsed file. Raises FileNotFoundError."""
    text = Path(path).read_text()
    screen = json.loads(text)
    set_screen(screened_items(screen), sha256_text(text)[:16])
    return screen


def third_party_block(item_id: str) -> str:
    rec = SCREEN.get(item_id)
    if rec is None:
        raise KeyError(f"item {item_id} is not in the third-party screen; the panel must "
                       "be restricted to screened items before any prompt is built")
    return (
        f"{BLOCK_BEGIN}\n"
        f"Who: {rec['description']}\n"
        f"Their interest in the outcome: {rec['interest']}\n"
        f"{STAKE_SENTENCE[rec['stake']]}\n"
        f"{BLOCK_END}"
    )


def stake_of(item_id: str) -> str:
    return SCREEN[item_id]["stake"] if item_id in SCREEN else "none"


_NP_ROLE_STAKE = {r.role_id: r.stake for r in ROLES_NP}


def stake_undermined_np(role_id: str, verdict: str, tp_stake: str) -> Optional[bool]:
    """stake_undermined for the four `_np` seats with the third party's stake
    supplied per item. Self-contained (the analysis imports it without
    install()) and byte-for-byte the base rule for the party seats: None when
    the verdict is not codable; for the third party also None when it has no
    stake (not applicable, distinct from the neutral's False)."""
    stake = STAKE_TO_ROLE_STAKE[tp_stake] if role_id == THIRD_PARTY_ID else _NP_ROLE_STAKE[role_id]
    if rcd.is_noncommittal(verdict) or verdict == rcd.NOVERDICT:
        return None
    at_fault = verdict in AT_FAULT_VERDICTS
    not_at_fault = verdict in NOT_AT_FAULT_VERDICTS
    if not (at_fault or not_at_fault):
        return None
    if stake == "not_at_fault":
        return at_fault
    if stake == "at_fault":
        return not_at_fault
    return None if role_id == THIRD_PARTY_ID else False


def losers(verdict: str, tp_stake: str) -> Optional[tuple[str, ...]]:
    """Advocates the verdict goes against; None when not codable."""
    out = []
    for rid in ADVOCATES_NP:
        u = stake_undermined_np(rid, verdict, tp_stake)
        if u is None and rid != THIRD_PARTY_ID:
            return None
        if u:
            out.append(rid)
    return tuple(out)


def favoured(verdict: str, tp_stake: str) -> Optional[tuple[str, ...]]:
    """Advocates whose stake the verdict serves (the third party only when it
    has a stake); None when not codable."""
    if losers(verdict, tp_stake) is None:
        return None
    out = []
    for rid in ADVOCATES_NP:
        u = stake_undermined_np(rid, verdict, tp_stake)
        if u is False:
            out.append(rid)
    return tuple(out)


def verdict_type(verdict: str) -> str:
    if verdict in ("YTA", "NTA"):
        return "one_loser"
    if verdict in ("ESH", "NAH"):
        return "both_party"
    return "none"


# ---------------------------------------------------------------------------
# Prompts: the base builders with the participant count re-worded to four,
# plus the third-party block on that seat
# ---------------------------------------------------------------------------

def _role_header_np(role: Role, item) -> str:
    head = (f"You are one of four participants in this deliberation, speaking as "
            f"{role.label}.\n\n{role.brief}")
    if role.role_id == THIRD_PARTY_ID:
        head += "\n\n" + third_party_block(item.item_id)
    return head


def r0_user_np(arm, item, role, *, allow_unresolved):
    return (
        f"{R0_PREAMBLE_NP}"
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{_role_header_np(role, item)}\n\n"
        "Write your opening statement. State the position you hold on this "
        "request and why.\n\n"
        f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def r1_user_np(arm, item, role, own_r0, r0_texts, *, allow_unresolved, cap):
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{_role_header_np(role, item)}\n\n"
        f"--- YOUR OPENING STATEMENT ---\n{rcd._cap(own_r0, cap)}\n--- END ---\n\n"
        "The other three participants opened as follows.\n\n"
        f"{rcd._others_block(r0_texts, role.role_id, cap)}\n\n"
        "Write your rebuttal. Challenge what you believe is wrong or "
        "incomplete, acknowledge what genuinely lands, and change your own "
        "position only if you have actually been persuaded by something in "
        "the account.\n\n"
        f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def r2_user_np(arm, item, role, own_r0, own_r1, r1_texts, *, allow_unresolved, cap):
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{_role_header_np(role, item)}\n\n"
        f"--- YOUR OPENING STATEMENT ---\n{rcd._cap(own_r0, cap)}\n--- END ---\n\n"
        f"--- YOUR REBUTTAL ---\n{rcd._cap(own_r1, cap)}\n--- END ---\n\n"
        "The other three participants rebutted as follows.\n\n"
        f"{rcd._others_block(r1_texts, role.role_id, cap)}\n\n"
        "State your final position. Say plainly what the group's finding "
        "should be and what your position rests on.\n\n"
        f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def synthesis_user_np(arm, item, r2_texts, *, allow_unresolved, cap):
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
        raise ValueError(f"stake knobs {on} are banned in the n-perspective cell")


def r3_label_user_np(arm, item, role, own_r2, synthesis, *, cap, stake_nudge=False,
                     stake_cot=False, stake_fewshot=False, stake_fewshot_set="plain"):
    _refuse_stake_knobs(stake_nudge=stake_nudge, stake_cot=stake_cot,
                        stake_fewshot=stake_fewshot)
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{_role_header_np(role, item)}\n\n"
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


def integration_user_np(arm, item, synthesis, labels, objections, *, allow_unresolved, cap):
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


def r4_vote_user_np(arm, item, role, synthesis, own_label, own_objection, proposal, *,
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
        f"{_role_header_np(role, item)}\n\n"
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
# Row post-processing: the vote columns, the per-item stake, k and the sensor
# ---------------------------------------------------------------------------

EXTRA_ROW_FIELDS = (
    "vote_third_party_advocate", "third_party_description", "third_party_stake",
    "third_party_stake_source", "screen_sha", "verdict_type_synthesis",
    "k_losers_synthesis", "k_losers_final", "n_favoured_synthesis",
    "n_against_interest_r3", "against_interest_roles",
)


def against_interest_count(vote_rows: Sequence[dict], synthesis_verdict: str,
                           tp_stake: str):
    """(count, role ids) of advocates that objected at R3 to a synthesis
    verdict that SERVES their stake. None when the synthesis verdict is not
    codable, so the debate is excluded rather than coded 0. A third party with
    no stake is neither favoured nor undermined and never counts."""
    fav = favoured(synthesis_verdict, tp_stake)
    if fav is None:
        return None, ()
    roles = [v["role_id"] for v in vote_rows
             if v["role_id"] in fav and int(v.get("objected_r3") or 0)]
    return len(roles), tuple(roles)


def run_deliberation_np(model, arm, item, idx, *args, **kwargs):
    row, vote_rows, calls = _BASE_RUN_DELIBERATION(model, arm, item, idx, *args, **kwargs)
    rec = SCREEN[item.item_id]
    tp_stake = rec["stake"]
    # the third-party seat's per-item stake, overwriting the role-driven blanks
    for v in vote_rows:
        if v["role_id"] == THIRD_PARTY_ID:
            v["role_stake"] = STAKE_TO_ROLE_STAKE[tp_stake]
            und = stake_undermined_np(THIRD_PARTY_ID, v["group_verdict"], tp_stake)
            v["stake_undermined"] = "" if und is None else int(und)
    votes = {v["role_id"]: v["vote"] for v in vote_rows}
    row["vote_writer_advocate"] = votes.get("writer_advocate" + SUFFIX, "")
    row["vote_counterparty"] = votes.get("counterparty" + SUFFIX, "")
    row["vote_third_party_advocate"] = votes.get(THIRD_PARTY_ID, "")
    row["vote_neutral_adjudicator"] = votes.get("neutral_adjudicator" + SUFFIX, "")
    row["third_party_description"] = rec["description"]
    row["third_party_stake"] = tp_stake
    row["third_party_stake_source"] = rec["stake_source"]
    row["screen_sha"] = rec["sha"]
    syn = row["synthesis_verdict"]
    row["verdict_type_synthesis"] = verdict_type(syn)
    ls = losers(syn, tp_stake)
    lf = losers(row["verdict"], tp_stake)
    fav = favoured(syn, tp_stake)
    row["k_losers_synthesis"] = "" if ls is None else len(ls)
    row["k_losers_final"] = "" if lf is None else len(lf)
    row["n_favoured_synthesis"] = "" if fav is None else len(fav)
    n_ai, ai_roles = against_interest_count(vote_rows, syn, tp_stake)
    row["n_against_interest_r3"] = "" if n_ai is None else n_ai
    row["against_interest_roles"] = "|".join(ai_roles)
    return row, vote_rows, calls


def do_call_np(**kw):
    """The third-party seat's records carry the screen record in parent_sha
    (see CACHE PLAN)."""
    if kw.get("role_id") == THIRD_PARTY_ID:
        rec = SCREEN[kw["item"].item_id]
        kw = dict(kw, parents=tuple(kw.get("parents") or ()) + (
            f"screen:{rec['description']}|{rec['interest']}|{rec['stake']}",))
    return _BASE_DO_CALL(**kw)


def subset_items_np(items, n_yta, n_nta):
    """Registered subset, then the third-party screen."""
    base = _BASE_SUBSET_ITEMS(items, n_yta, n_nta)
    keep = [i for i in base if i.item_id in SCREEN]
    print(f"[nperspective] third-party screen: {len(keep)} of {len(base)} panel items "
          f"screened in (screen file sha {SCREEN_FILE_SHA or 'n/a'})")
    return keep


def screen_summary(items) -> dict:
    stakes = {"YTA": 0, "NTA": 0, "none": 0}
    for it in items:
        stakes[stake_of(it.item_id)] += 1
    src: dict[str, int] = {}
    for it in items:
        s = SCREEN[it.item_id]["stake_source"]
        src[s] = src.get(s, 0) + 1
    return {"n_items": len(items), "stake_counts": stakes, "stake_source_counts": src,
            "k2_reachable_items": stakes["YTA"] + stakes["NTA"]}


def print_cost_model_np(cm: dict, *, label: str) -> None:
    n = rcd.N_AGENTS
    print("\n" + "=" * 78)
    print(f"DRY RUN -- {label}  [n-perspective: {n} seats, edges on]")
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
    time (run_deliberation, run, main, cost_model, _agent_prompts,
    extract_addressed, _others_block, do_call and subset_items all read module
    globals)."""
    R = ROLES_NP
    rcd.ROLES = R
    rcd.ROLE_BY_ID = {r.role_id: r for r in R}
    rcd.ROLE_ORDER = tuple(r.role_id for r in R)
    rcd.N_AGENTS = len(R)
    rcd.CALLS_PER_CELL = rcd.N_AGENTS * len(rcd.AGENT_ROUNDS) + len(rcd.MODERATOR_ROUNDS)
    rcd.STAKE_BEARING_ROLES = STAKE_BEARING_NP
    rcd.moderator_role_id = lambda m: f"{MODERATOR_PREFIX}-{_safe(m)}"
    rcd.SYNTHESIS_SYSTEM = SYNTHESIS_SYSTEM_NP
    rcd.INTEGRATION_SYSTEM = INTEGRATION_SYSTEM_NP
    rcd.r0_user = r0_user_np
    rcd.r1_user = r1_user_np
    rcd.r2_user = r2_user_np
    rcd.synthesis_user = synthesis_user_np
    rcd.r3_label_user = r3_label_user_np
    rcd.integration_user = integration_user_np
    rcd.r4_vote_user = r4_vote_user_np
    rcd.run_deliberation = run_deliberation_np
    rcd.do_call = do_call_np
    rcd.subset_items = subset_items_np
    rcd.ROW_FIELDS = tuple(_BASE_ROW_FIELDS) + EXTRA_ROW_FIELDS
    rcd.print_cost_model = print_cost_model_np
    if NPERSPECTIVE_NOTE not in rcd.ROLE_MAPPING_NOTE:
        rcd.ROLE_MAPPING_NOTE = rcd.ROLE_MAPPING_NOTE.rstrip("\n") + "\n\n" + NPERSPECTIVE_NOTE


# ---------------------------------------------------------------------------
# Cache namespaces (one listdir pass; the output dir holds ~570k files)
# ---------------------------------------------------------------------------

def namespace_counts(names: Sequence[str], models: Sequence[str],
                     moderator_model: Optional[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    mods = sorted({moderator_model or m for m in models})
    for m in models:
        pre = f"cgd_{_safe(m)}_"
        for rid in ROLE_IDS_NP:
            counts[f"{m}:{rid}"] = sum(1 for n in names
                                       if n.startswith(pre) and n.endswith(f"_{rid}.json"))
    for mm in mods:
        pre = f"cgd_{_safe(mm)}_"
        counts[f"{mm}:{MODERATOR_PREFIX}"] = sum(
            1 for n in names if n.startswith(pre) and f"_{MODERATOR_PREFIX}-" in n)
    for rid in ROLE_IDS_NP:
        counts[f"any:{rid}"] = sum(1 for n in names if n.endswith(f"_{rid}.json"))
    counts[f"any:{MODERATOR_PREFIX}"] = sum(1 for n in names if f"_{MODERATOR_PREFIX}-" in n)
    counts[f"tag:{DEFAULT_TAG}"] = sum(1 for n in names if n.startswith(f"{DEFAULT_TAG}_"))
    return counts


def cache_is_clean(models: Sequence[str] = (DEFAULT_MODELS,),
                   moderator_model: Optional[str] = None,
                   out_dir: Optional[Path] = None,
                   names: Optional[Sequence[str]] = None) -> tuple[bool, dict[str, int]]:
    """Clean = no record for any (listed model, `_np` seat) and none for the
    moderator model under `modnp-`. Per model, so a later deployment can run
    into a namespace that already holds another model's records without
    --resume, while a partial re-run of the SAME model under possibly edited
    briefs or a re-screened panel is refused."""
    if names is None:
        out_dir = Path(out_dir) if out_dir is not None else rcd.OUT_DIR
        names = os.listdir(out_dir) if out_dir.is_dir() else []
    counts = namespace_counts(names, list(models), moderator_model)
    own = sum(v for k, v in counts.items() if not k.startswith(("any:", "tag:")))
    return own == 0, counts


def foreign_files(names: Sequence[str]) -> list[str]:
    """Stub / selftest / model-'m' files carrying this runner's names. There
    must never be any: the selftest redirects the cache dir to a tempdir."""
    return sorted(n for n in names
                  if (any(n.endswith(f"_{rid}.json") for rid in ROLE_IDS_NP)
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
    for r in ROLES_NP:
        if f"speaking as {r.label}." in user:
            return r.role_id
    return "moderator"


def _stub_tp_stake(user: str) -> str:
    """The third party's stake, read back from the block the prompt carries."""
    if BLOCK_BEGIN not in user:
        return "none"
    block = user.split(BLOCK_BEGIN, 1)[1].split(BLOCK_END, 1)[0]
    for k in ("YTA", "NTA"):
        if STAKE_SENTENCE[k] in block:
            return k
    return "none"


def _stub_verdict(text: str) -> str:
    return "YTA" if text.count("VERDICT: YTA") > text.count("VERDICT: NTA") else "NTA"


def _stub_generate(model, system, user, *, sample_idx=0, max_tokens=512, **kw) -> _StubResult:
    """Deterministic four-seat stand-in. Plants: the writer's advocate opens
    NTA, the counterparty YTA, the third party with its stake (hash when it
    has none), the neutral by hash; at R3 every advocate objects exactly when
    its stake is undermined, the third party with no stake objects on a small
    hash minority, and the favoured advocates object AGAINST interest on a
    hash-selected minority so n_against_interest_r3 is exercised; revise-iff-
    addressed at R4 with a little noise."""
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
        undermined = stake_undermined_np(speaking, v, _stub_tp_stake(user))
        against = (speaking in ADVOCATES_NP and undermined is False and h % 7 == 0)
        no_stake_objects = (undermined is None and speaking == THIRD_PARTY_ID and h % 9 == 0)
        if not undermined and not against and not no_stake_objects:
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
    if speaking == "counterparty" + SUFFIX:
        v = "YTA"
    elif speaking == "writer_advocate" + SUFFIX:
        v = "NTA"
    elif speaking == THIRD_PARTY_ID:
        st = _stub_tp_stake(user)
        v = st if st in ("YTA", "NTA") else ("YTA" if h % 2 else "NTA")
    else:
        v = "YTA" if h % 2 else "NTA"
    body = ("Section 1 - Protagonist: the writer.\n"
            "Section 2 - Stakeholders: writer, third party, other party, onlookers.\n"
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
                "--keep-filtered", "--no-unresolved", "--transcript-cap")


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
    known, _ = ap.parse_known_args(list(rest))
    return known


def expected_keys(models: Sequence[str], rest: Sequence[str]) -> set:
    """(model, arm, item_id, sample_idx) for every debate rcd.main launches
    under these arguments: the registered panel, the INSTALLED subset_items
    (registered subset, then the third-party screen), the content-filter
    screen as the union over the models (--keep-filtered is banned), samples."""
    pa = panel_args(rest)
    items = rcd.load_items(source="scruples", n_yta=99, n_nta=150, min_votes=pa.min_votes,
                           min_consensus=pa.min_consensus, seed=pa.seed)
    items = rcd.subset_items(items, pa.n_yta, pa.n_nta)
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


def _synthetic_screen(items) -> dict[str, dict]:
    """A screen map over the given items with all three stakes represented."""
    out = {}
    for i, it in enumerate(sorted(items, key=lambda x: x.item_id)):
        stake = ("YTA", "NTA", "none")[i % 3]
        out[it.item_id] = {"third_party_description": f"the writer's sibling ({i})",
                           "third_party_interest": "a stable household",
                           "stake_final": stake, "stake_source": "heuristic",
                           "screened_in": True}
    return out


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    from scripts.run_crowdgold_topology import CELLS as TOPO_CELLS, build_roles as topo_roles
    from scripts.run_crowdgold_unembodied import MODERATOR_PREFIX as UNEMB_MOD, UNEMBODIED_ROLES
    try:
        import scripts.run_crowdgold_coalition as co
    except ImportError:  # pragma: no cover
        co = None

    # ---- role set ----------------------------------------------------------
    check("four seats", len(ROLES_NP) == 4)
    check("role ids are writer_advocate_np, counterparty_np, third_party_advocate_np, "
          "neutral_adjudicator_np in that order",
          ROLE_IDS_NP == ("writer_advocate_np", "counterparty_np", "third_party_advocate_np",
                          "neutral_adjudicator_np"))
    check("stakes are not_at_fault, at_fault, per_item, none",
          tuple(r.stake for r in ROLES_NP) == ("not_at_fault", "at_fault", "per_item", "none"))
    for rid in ("writer_advocate", "counterparty", "neutral_adjudicator"):
        np_ = next(r for r in ROLES_NP if r.role_id == rid + SUFFIX)
        emb = _EMB_BY_ID[rid]
        check(f"{rid}_np carries the embodied brief, label, paper_role and stake VERBATIM",
              np_.brief == emb.brief and np_.label == emb.label
              and np_.paper_role == emb.paper_role and np_.stake == emb.stake)
    tp = next(r for r in ROLES_NP if r.role_id == THIRD_PARTY_ID)
    check("third party brief speaks for a person the account names who is neither party",
          tp.brief.startswith("You speak for a third party")
          and "neither the writer nor the party the writer is in conflict with" in tp.brief)
    check("third party brief defers identity and interest to the record and forbids invention",
          "stated in the record below" in tp.brief
          and "You may not invent facts that are not in the account" in tp.brief)
    check("third party brief keeps the advocacy-intensity clause of the registered advocates",
          "as strongly as the account itself allows" in tp.brief)
    check("no brief contains a blank line", all(POST_DELIM not in r.brief for r in ROLES_NP))
    check("every role label is unique (the stub keys on 'speaking as <label>.')",
          len({r.label for r in ROLES_NP}) == 4)
    check("stake sentences: YTA/NTA mirror the registered advocates' sentence; none says so",
          STAKE_SENTENCE["NTA"] in _EMB_BY_ID["writer_advocate"].brief
          and STAKE_SENTENCE["YTA"].replace("is a finding", "is, on most readings of an "
                                            "account like this, a finding")
          in _EMB_BY_ID["counterparty"].brief
          and "does not settle" in STAKE_SENTENCE["none"])

    # ---- namespaces vs every sibling role set -------------------------------
    sibling_ids = {r.role_id for r in _EMBODIED} | {r.role_id for r in UNEMBODIED_ROLES}
    for cell in TOPO_CELLS:
        sibling_ids |= {r.role_id for r in topo_roles(*cell)}
    if co is not None:
        sibling_ids |= set(co.ROLE_IDS_CO)
    check("no role id collides with the embodied, unembodied, coalition or any topology cell",
          not (set(ROLE_IDS_NP) & sibling_ids))
    check("no role id is a substring of another (extract_addressed matches by substring)",
          not any(a != b and a in b for a in ROLE_IDS_NP for b in ROLE_IDS_NP))
    sib_mods = {"mod", UNEMB_MOD} | {c["mod"] for c in TOPO_CELLS.values()}
    if co is not None:
        sib_mods.add(co.MODERATOR_PREFIX)
    check(f"moderator prefix {MODERATOR_PREFIX!r} is distinct from every sibling's",
          MODERATOR_PREFIX not in sib_mods and not MODERATOR_PREFIX.startswith("mod-"))
    check("tag rule: DEFAULT_TAG and a suffix of it pass; every sibling cell's tag is refused",
          tag_ok(DEFAULT_TAG) and tag_ok(DEFAULT_TAG + "_b") and not any(tag_ok(t) for t in SIBLING_TAGS))
    check("--no-unresolved and --transcript-cap are banned (prompt text the cache key cannot see)",
          "--no-unresolved" in BANNED_FLAGS and "--transcript-cap" in BANNED_FLAGS)
    if co is not None:
        check("four-seat wording is byte-identical to the coalition cell's (COUNT_WORDS, "
              "moderator system prompts, R0 preamble)",
              COUNT_WORDS == co.COUNT_WORDS and SYNTHESIS_SYSTEM_NP == co.SYNTHESIS_SYSTEM_CO
              and INTEGRATION_SYSTEM_NP == co.INTEGRATION_SYSTEM_CO
              and R0_PREAMBLE_NP == co.R0_PREAMBLE_CO)

    # ---- install() ---------------------------------------------------------
    install()
    install()  # idempotent
    check("install(): N_AGENTS 4, CALLS_PER_CELL 22 (4+4+4+1+4+1+4)",
          rcd.N_AGENTS == 4 and rcd.CALLS_PER_CELL == 22)
    check("install(): ROLE_ORDER / ROLE_BY_ID resolve every `_np` seat",
          rcd.ROLE_ORDER == ROLE_IDS_NP and all(rcd.ROLE_BY_ID[r.role_id] is r for r in ROLES_NP))
    check("install(): STAKE_BEARING_ROLES re-pointed at the three advocates",
          rcd.STAKE_BEARING_ROLES == ("writer_advocate_np", "counterparty_np",
                                      "third_party_advocate_np"))
    check("install(): moderator id carries the modnp prefix",
          rcd.moderator_role_id("grok-4-1-fast-reasoning") == "modnp-grok-4-1-fast-reasoning")
    check("install(): ROW_FIELDS gained the third-party vote, stake, k and sensor columns once",
          all(rcd.ROW_FIELDS.count(f) == 1 for f in EXTRA_ROW_FIELDS)
          and rcd.ROW_FIELDS[:len(_BASE_ROW_FIELDS)] == tuple(_BASE_ROW_FIELDS))
    check("install(): ROLE_MAPPING_NOTE carries the n-perspective note once",
          rcd.ROLE_MAPPING_NOTE.count("N PERSPECTIVES (Addendum 16.21)") == 1)
    check("install(): do_call and subset_items are wrapped",
          rcd.do_call is do_call_np and rcd.subset_items is subset_items_np)
    k = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot", "as_asker", "itemX", 0,
                            "r1", THIRD_PARTY_ID, 2560)
    km = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot", "as_asker", "itemX", 0,
                             "synthesis", rcd.moderator_role_id("grok-4-1-fast-reasoning"), 1024)
    check("agent cache path carries the `_np` id and never the embodied or `_co` id",
          k.name.endswith("_third_party_advocate_np.json")
          and "_writer_advocate.json" not in k.name and "_co.json" not in k.name)
    check("moderator cache path carries modnp and never mod- / modcoalition",
          "_modnp-grok" in km.name and "_mod-grok" not in km.name and "modcoalition" not in km.name)

    # ---- the screen record and the third-party block ----------------------
    item = rcd.CrowdGoldItem(item_id="X", post_text="[POST]", gold_verdict="YTA")
    set_screen({"X": {"third_party_description": "the writer's younger sister",
                      "third_party_interest": "keeping her place in the family home",
                      "stake_final": "NTA", "stake_source": "heuristic", "screened_in": True}},
               "deadbeef")
    blk = third_party_block("X")
    check("third-party block quotes description, interest and the NTA stake sentence",
          "Who: the writer's younger sister" in blk
          and "Their interest in the outcome: keeping her place" in blk
          and STAKE_SENTENCE["NTA"] in blk and blk.startswith(BLOCK_BEGIN) and blk.endswith(BLOCK_END))
    check("screen record sha changes with description, interest or stake",
          len({screen_record({"third_party_description": d, "third_party_interest": i,
                              "stake_final": s})["sha"]
               for d, i, s in (("a", "b", "YTA"), ("a2", "b", "YTA"), ("a", "b2", "YTA"),
                               ("a", "b", "NTA"))}) == 4)
    check("an unknown or empty stake is coded none",
          screen_record({"third_party_description": "x", "stake_final": ""})["stake"] == "none"
          and screen_record({"third_party_description": "x", "stake_final": "ESH"})["stake"] == "none")
    try:
        third_party_block("not-screened")
        raised = False
    except KeyError:
        raised = True
    check("a prompt for an item outside the screen raises (the panel must be restricted first)",
          raised)

    # ---- prompts: base + count words (+ the block on the third-party seat) ----
    role = ROLES_NP[1]  # counterparty_np: a verbatim seat
    prior = {rid: f"[{rid} PRIOR]" for rid in ROLE_IDS_NP}
    labels = {rid: "ACCEPT_WITH_MODIFICATION" for rid in ROLE_IDS_NP}
    objs = {rid: f"[{rid} OBJ]" for rid in ROLE_IDS_NP}
    pairs = {
        "r0": (r0_user_np("as_asker", item, role, allow_unresolved=True),
               _BASE_PROMPTS["r0_user"]("as_asker", item, role, allow_unresolved=True)),
        "r1": (r1_user_np("as_asker", item, role, "[OWN0]", prior, allow_unresolved=True, cap=0),
               _BASE_PROMPTS["r1_user"]("as_asker", item, role, "[OWN0]", prior, allow_unresolved=True, cap=0)),
        "r2": (r2_user_np("as_asker", item, role, "[OWN0]", "[OWN1]", prior, allow_unresolved=True, cap=0),
               _BASE_PROMPTS["r2_user"]("as_asker", item, role, "[OWN0]", "[OWN1]", prior, allow_unresolved=True, cap=0)),
        "synthesis": (synthesis_user_np("as_asker", item, prior, allow_unresolved=True, cap=0),
                      _BASE_PROMPTS["synthesis_user"]("as_asker", item, prior, allow_unresolved=True, cap=0)),
        "r3_label": (r3_label_user_np("as_asker", item, role, "[OWN2]", "[SYN]", cap=0),
                     _BASE_PROMPTS["r3_label_user"]("as_asker", item, role, "[OWN2]", "[SYN]", cap=0)),
        "integration": (integration_user_np("as_asker", item, "[SYN]", labels, objs, allow_unresolved=True, cap=0),
                        _BASE_PROMPTS["integration_user"]("as_asker", item, "[SYN]", labels, objs, allow_unresolved=True, cap=0)),
        "r4_vote": (r4_vote_user_np("as_asker", item, role, "[SYN]", "ACCEPT_WITH_MODIFICATION", "[OBJ]", "[PROP]", cap=0),
                    _BASE_PROMPTS["r4_vote_user"]("as_asker", item, role, "[SYN]", "ACCEPT_WITH_MODIFICATION", "[OBJ]", "[PROP]", cap=0)),
    }
    for rd, (np_, base) in pairs.items():
        check(f"{rd}: n-perspective prompt == base prompt with the count words recounted, byte for byte "
              "(verbatim seat)", np_ == recount(base) and np_ != base)
    check("SYNTHESIS_SYSTEM / INTEGRATION_SYSTEM == base with the count words recounted",
          SYNTHESIS_SYSTEM_NP == recount(_BASE_SYNTHESIS_SYSTEM)
          and INTEGRATION_SYSTEM_NP == recount(_BASE_INTEGRATION_SYSTEM)
          and rcd.SYNTHESIS_SYSTEM is SYNTHESIS_SYSTEM_NP and rcd.INTEGRATION_SYSTEM is INTEGRATION_SYSTEM_NP)
    check("R0 preamble == base with exactly the word three -> four",
          R0_PREAMBLE_NP == _BASE_R0_PREAMBLE.replace("three", "four") and R0_PREAMBLE_NP.count("four") == 1)
    check("no n-perspective prompt still carries any stale three-seat phrase (the left side of "
          "COUNT_WORDS); 'The other three participants' is the correct four-seat wording",
          not any(a in np_ for np_, _ in pairs.values() for a, _ in COUNT_WORDS)
          and not any(a in t for t in (SYNTHESIS_SYSTEM_NP, INTEGRATION_SYSTEM_NP, R0_PREAMBLE_NP)
                      for a, _ in COUNT_WORDS))
    check("r4 keeps 'In two or three sentences' (a count that is NOT a seat count)",
          "In two or three sentences" in pairs["r4_vote"][0])
    check("r1 shows the counterparty its THREE neighbours and not itself",
          all(f"[{rid} PRIOR]" in pairs["r1"][0] for rid in ROLE_IDS_NP if rid != role.role_id)
          and f"[{role.role_id} PRIOR]" not in pairs["r1"][0])
    check("integration prompt lists the four `_np` ids for the ADDRESSED line",
          f"(from: {', '.join(ROLE_IDS_NP)})" in pairs["integration"][0])
    check("extract_addressed resolves the `_np` ids and ignores the bare embodied ids",
          rcd.extract_addressed(f"{rcd.ADDRESSED_MARKER} third_party_advocate_np, counterparty")
          == ("third_party_advocate_np",))
    tp_r0 = r0_user_np("as_asker", item, tp, allow_unresolved=True)
    tp_base_shape = recount(_BASE_PROMPTS["r0_user"]("as_asker", item, tp, allow_unresolved=True))
    check("third-party r0 == recounted base prompt with ONLY the block inserted after the brief",
          tp_r0 == tp_base_shape.replace(tp.brief, tp.brief + "\n\n" + blk)
          and tp_r0.count(BLOCK_BEGIN) == 1)
    check("the block appears on every agent round of the third-party seat and on no other seat",
          all(BLOCK_BEGIN in p for p in rcd._agent_prompts("third_person", item, tp).values())
          and not any(BLOCK_BEGIN in p for r_ in ROLES_NP if r_ is not tp
                      for p in rcd._agent_prompts("third_person", item, r_).values()))
    check("moderator prompts carry no screen text (the block is the seat's, not the moderator's)",
          BLOCK_BEGIN not in pairs["synthesis"][0] and BLOCK_BEGIN not in pairs["integration"][0])
    try:
        r3_label_user_np("as_asker", item, role, "o", "s", cap=0, stake_nudge=True)
        raised = False
    except ValueError:
        raised = True
    check("r3/r4 builders raise on any stake knob (belt to the CLI ban's braces)", raised)
    orth = rcd.assert_arm_orthogonality([item])
    check("arm orthogonality holds on every round for all four seats (block is arm-invariant)",
          orth["prompts_checked"] == len(rcd.ARM_ORDER) * 4 * 5 and orth["roles"] == list(ROLE_IDS_NP))

    # ---- stake / losers / favoured / the sensor ----------------------------
    check("third party with stake NTA is undermined by YTA and ESH, served by NTA and NAH",
          stake_undermined_np(THIRD_PARTY_ID, "YTA", "NTA") is True
          and stake_undermined_np(THIRD_PARTY_ID, "ESH", "NTA") is True
          and stake_undermined_np(THIRD_PARTY_ID, "NTA", "NTA") is False
          and stake_undermined_np(THIRD_PARTY_ID, "NAH", "NTA") is False)
    check("third party with stake YTA is the mirror; no stake -> None; unresolved -> None",
          stake_undermined_np(THIRD_PARTY_ID, "NTA", "YTA") is True
          and stake_undermined_np(THIRD_PARTY_ID, "YTA", "YTA") is False
          and stake_undermined_np(THIRD_PARTY_ID, "YTA", "none") is None
          and stake_undermined_np(THIRD_PARTY_ID, rcd.UNRESOLVED, "YTA") is None)
    check("the party and neutral seats agree with the base stake_undermined on every verdict "
          "(after install; the function itself needs no install)",
          all(stake_undermined_np(rid, v, "none") == rcd.stake_undermined(rid, v)
              for rid in ("writer_advocate_np", "counterparty_np", "neutral_adjudicator_np")
              for v in ("YTA", "NTA", "ESH", "NAH", rcd.UNRESOLVED, rcd.NOVERDICT, "INFO"))
          and stake_undermined_np("writer_advocate_np", "YTA", "none") is True
          and stake_undermined_np("counterparty_np", "YTA", "none") is False
          and stake_undermined_np("neutral_adjudicator_np", "YTA", "none") is False)
    check("k: YTA with a NTA-stake third party -> 2 losers (writer, third); with YTA-stake -> 1",
          losers("YTA", "NTA") == ("writer_advocate_np", "third_party_advocate_np")
          and losers("YTA", "YTA") == ("writer_advocate_np",)
          and losers("NTA", "YTA") == ("counterparty_np", "third_party_advocate_np")
          and losers("NTA", "none") == ("counterparty_np",))
    check("k in {1, 2} on every codable verdict x stake: k=0 and k=3 unreachable by construction",
          {len(losers(v, s)) for v in ("YTA", "NTA", "ESH", "NAH") for s in ("YTA", "NTA", "none")} == {1, 2})
    check("favoured is the complement among advocates with a stake; none for a non-codable verdict",
          favoured("YTA", "NTA") == ("counterparty_np",)
          and favoured("NTA", "none") == ("writer_advocate_np",)
          and favoured("YTA", "YTA") == ("counterparty_np", "third_party_advocate_np")
          and favoured(rcd.UNRESOLVED, "YTA") is None and losers(rcd.NOVERDICT, "NTA") is None)
    check("verdict_type: YTA/NTA one_loser, ESH/NAH both_party, else none",
          verdict_type("YTA") == "one_loser" and verdict_type("NAH") == "both_party"
          and verdict_type(rcd.UNRESOLVED) == "none")
    vr = [{"role_id": "writer_advocate_np", "objected_r3": 1},
          {"role_id": "counterparty_np", "objected_r3": 1},
          {"role_id": "third_party_advocate_np", "objected_r3": 1},
          {"role_id": "neutral_adjudicator_np", "objected_r3": 1}]
    check("against_interest_count: NTA synthesis, third stake YTA -> only the writer is favoured "
          "and counts; the neutral never counts",
          against_interest_count(vr, "NTA", "YTA") == (1, ("writer_advocate_np",)))
    check("against_interest_count: YTA synthesis, third stake YTA -> counterparty + third party",
          against_interest_count(vr, "YTA", "YTA") == (2, ("counterparty_np", "third_party_advocate_np")))
    check("against_interest_count: third party with no stake never counts",
          against_interest_count(vr, "YTA", "none") == (1, ("counterparty_np",)))
    check("against_interest_count is None (excluded) on a non-codable synthesis",
          against_interest_count(vr, rcd.UNRESOLVED, "YTA") == (None, ()))

    # ---- cache namespaces on the REAL output dir ----------------------------------
    names = os.listdir(OUT_DIR)
    clean, counts = cache_is_clean(names=names)
    own = {k: v for k, v in counts.items() if not k.startswith(("any:", "tag:"))}
    anyc = {k: v for k, v in counts.items() if k.startswith("any:")}
    check(f"no foreign (stub / selftest / model 'm') file carries this runner's names in "
          f"{OUT_DIR.name}", foreign_files(names) == [])
    check(f"every n-perspective namespace is EMPTY, or holds only this cell's registered "
          f"footprint (all seats + moderator populated together) {anyc}",
          all(v == 0 for v in anyc.values()) or all(v > 0 for v in anyc.values()))
    check(f"cache_is_clean(grok) on the real dir agrees with the namespace counts {own}",
          clean == (sum(own.values()) == 0))
    snapshot = sorted(n for n in names if any(n.endswith(f"_{rid}.json") for rid in ROLE_IDS_NP)
                      or f"_{MODERATOR_PREFIX}-" in n or n.startswith(f"{DEFAULT_TAG}_"))
    screen_on_disk = SCREEN_PATH.exists()

    # ---- CLI refusals (cache dir redirected; nothing can be spent) ----------
    real_gen, real_out = rcd.generate_any, rcd.OUT_DIR
    try:
        rcd.generate_any = _stub_generate  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as td:
            rcd.OUT_DIR = Path(td)
            try:
                items = rcd.load_items(source="scruples", n_yta=99, n_nta=150,
                                       min_votes=50, min_consensus=0.90, seed=44)
                # a synthetic screen file over 8 + 8 items, written in the tempdir
                sub = _BASE_SUBSET_ITEMS(items, 8, 8)
                syn_screen = {"items": [dict(v, item_id=k) for k, v in _synthetic_screen(sub).items()]}
                screen_path = Path(td) / "third_party_screen_selftest.json"
                screen_path.write_text(json.dumps(syn_screen))
                for flag in ("--stake-nudge", "--smoke", "--full", "--keep-filtered",
                             "--r3r4-thinking-budget=1024"):
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        rc = main(["--dry-run", flag, "--screen", str(screen_path)])
                    check(f"main() refuses {flag} (exit 4) before building anything",
                          rc == 4 and "DRY RUN" not in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rcs = [main(["--dry-run", "--tag", t, "--screen", str(screen_path)]) for t in
                           ("cg_deliberation", "cg_deliberation_coalition", "cg_deliberation_multivendor",
                            "cg_deliberation_modsonnet", "cg_deliberation_gpt4o")]
                check("main() refuses every sibling cell's tag, listed or not (exit 2, nothing written)",
                      rcs == [2] * 5 and "DRY RUN" not in buf.getvalue()
                      and not any(n.endswith("_dryrun.json") for n in os.listdir(td)))
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_nu = main(["--dry-run", "--no-unresolved", "--screen", str(screen_path)])
                    rc_tc = main(["--dry-run", "--transcript-cap", "400", "--screen", str(screen_path)])
                check("main() refuses --no-unresolved and --transcript-cap (exit 4) before building anything",
                      rc_nu == 4 and rc_tc == 4 and "DRY RUN" not in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--dry-run", "--screen", str(Path(td) / "missing.json")])
                check("main() refuses a missing screen file (exit 2) before building anything",
                      rc == 2 and "DRY RUN" not in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--dry-run", "--n-yta", "4", "--n-nta", "4", "--screen", str(screen_path)])
                out = buf.getvalue()
                check("main(--dry-run) exits 0, prints the 22-call breakdown and the NEW-calls line",
                      rc == 0 and "calls per cell         22" in out
                      and "4 seats x r0,r1,r2 = 12" in out and "every one NEW" in out)
                check("main(--dry-run) restricts the panel to the screen: 8 of 8 items screened in, "
                      "16 cells x 22 = 352 calls",
                      "8 of 8 panel items screened in" in out and "TOTAL GENERATION CALLS 352" in out)
                check("main(--dry-run) prints the screen summary with the k=2-reachable count",
                      "k=2 reachable" in out and "stake counts" in out)
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
                    rc = main(["--n-yta", "2", "--n-nta", "2", "--screen", str(screen_path)])
                check("main() without --run prints the cost model and REFUSES (exit 2)",
                      rc == 2 and "DRY RUN" in buf.getvalue() and "NOT RUNNING" in buf.getvalue()
                      and "--run" in buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--dry-run", "--n-yta", "2", "--n-nta", "2",
                               "--models", "no-such-deployment", "--screen", str(screen_path)])
                check("main(--dry-run) on an unpriced model prints the WARNING and exits 0; "
                      "the --run path refuses it (exit 2), verified in source",
                      rc == 0 and "NO PRICE ON FILE" in buf.getvalue()
                      and "not in PRICES" in inspect.getsource(main))
                check("the run path enforces cache_is_clean with --resume as the only override, "
                      "and prints GUARD FAILED on a failed guard (both verified in source)",
                      "cache_is_clean(" in inspect.getsource(main)
                      and "known.resume" in inspect.getsource(main)
                      and "GUARD FAILED" in inspect.getsource(rcd.main))
                check("dry runs wrote nothing into any n-perspective namespace of the tempdir",
                      cache_is_clean(out_dir=Path(td))[0])

                # ---- end-to-end protocol against the stub, cache in the tempdir ----
                set_screen(_synthetic_screen(sub), "selftest")
                rows, votes, calls = [], [], []
                for it in sub:
                    for arm in (rcd.THIRD_PERSON, rcd.AS_ASKER):
                        row, vr_, cr = rcd.run_deliberation("stub-model", arm, it, 0,
                                                            scaffold="narrative_cot")
                        rows.append(row); votes.extend(vr_); calls.extend(cr)
                n_files = len([n for n in os.listdir(td) if n.startswith("cgd_")])
                check("smoke: 32 debates x 22 calls, one cache file per call in the TEMP dir",
                      len(rows) == 32 and len(calls) == 32 * 22 and n_files == 32 * 22)
                from collections import Counter
                rc_ = Counter(c["round"] for c in calls)
                check("smoke: call mix 4/4/4 + 1 + 4 + 1 + 4",
                      all(rc_[r] == 32 * 4 for r in rcd.AGENT_ROUNDS)
                      and all(rc_[r] == 32 for r in rcd.MODERATOR_ROUNDS))
                check("smoke: every call record carries a `_np` seat or the modnp id",
                      all(c["role_id"].endswith(SUFFIX) or c["role_id"].startswith(MODERATOR_PREFIX + "-")
                          for c in calls))
                check("smoke: every group verdict, R3 label and R4 vote parsed",
                      all(r["verdict"] != rcd.NOVERDICT for r in rows)
                      and all(v["vote_parsed"] and v["r3_label_parsed"] for v in votes))
                check("smoke: rows carry n_agents 4 and 22 calls",
                      all(r["n_agents"] == 4 and r["n_calls"] == 22 for r in rows))
                check("smoke: the four vote columns are filled from the `_np` seats",
                      all(r["vote_writer_advocate"] in ("ACCEPT", "REJECT")
                          and r["vote_counterparty"] in ("ACCEPT", "REJECT")
                          and r["vote_third_party_advocate"] in ("ACCEPT", "REJECT")
                          and r["vote_neutral_adjudicator"] in ("ACCEPT", "REJECT") for r in rows))
                check("smoke: R3 arithmetic sums to 4 on every row",
                      all(r["n_r3_accept"] + r["n_r3_accept_with_mod"] + r["n_r3_reject"]
                          + r["n_r3_unparsed"] == 4 for r in rows))
                check("smoke: every row carries the third party's description, stake and screen sha "
                      "from the screen record",
                      all(r["third_party_description"].startswith("the writer's sibling")
                          and r["third_party_stake"] == stake_of(r["item_id"])
                          and r["screen_sha"] == SCREEN[r["item_id"]]["sha"] for r in rows))
                tp_votes = [v for v in votes if v["role_id"] == THIRD_PARTY_ID]
                check("smoke: third-party vote rows carry the per-item role_stake and a blank "
                      "stake_undermined exactly when the third party has no stake",
                      all(v["role_stake"] == STAKE_TO_ROLE_STAKE[stake_of(v["item_id"])] for v in tp_votes)
                      and all((v["stake_undermined"] == "") == (stake_of(v["item_id"]) == "none")
                              for v in tp_votes))
                und_tp = [v for v in tp_votes
                          if stake_undermined_np(THIRD_PARTY_ID, v["synthesis_verdict"],
                                                 stake_of(v["item_id"])) is True]
                check(f"smoke: the third party objects at R3 whenever the synthesis undermines its "
                      f"stake (planted; {len(und_tp)} such votes)",
                      und_tp and all(v["objected_r3"] == 1 for v in und_tp))
                ks = Counter(r["k_losers_synthesis"] for r in rows)
                check(f"smoke: k_losers_synthesis is 1 or 2 on every row and both occur {dict(ks)}",
                      set(ks) <= {1, 2} and ks[1] > 0 and ks[2] > 0)
                check("smoke: k + n_favoured == number of advocates with a stake, per row",
                      all(r["k_losers_synthesis"] + r["n_favoured_synthesis"]
                          == (2 if r["third_party_stake"] == "none" else 3) for r in rows))
                n_ai = [r["n_against_interest_r3"] for r in rows]
                check("smoke: n_against_interest_r3 is an int on every row and fires somewhere "
                      f"(planted against-interest objections) {Counter(n_ai)}",
                      all(isinstance(x, int) for x in n_ai) and any(x > 0 for x in n_ai))
                check("smoke: against_interest_roles agrees with the count and never names the neutral",
                      all(len([s for s in r["against_interest_roles"].split("|") if s])
                          == r["n_against_interest_r3"] for r in rows)
                      and not any("neutral" in r["against_interest_roles"] for r in rows))
                check("smoke: n_against_interest_r3 <= n_favoured_synthesis on every row",
                      all(r["n_against_interest_r3"] <= r["n_favoured_synthesis"] for r in rows))
                check("smoke: every row key reaches disk (ROW_FIELDS covers the row)",
                      set().union(*[set(r) for r in rows]) <= set(rcd.ROW_FIELDS))
                conc = rcd.stake_concentration(votes, n_boot=100, seed=1, roles=rcd.STAKE_BEARING_ROLES)
                check("smoke: within-role stake concentration is identified over the three advocates",
                      conc is not None and conc["roles"] == sorted(STAKE_BEARING_NP))
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
                # the screen is in the third-party seat's parent_sha: a re-screened
                # item misses the cache and regenerates (nothing else replays)
                it0 = sub[0]
                p0 = rcd.call_cache_path("stub-model", "narrative_cot", rcd.THIRD_PERSON, it0.item_id, 0,
                                         "r0", THIRD_PARTY_ID, 2560)
                sha_before = json.loads(p0.read_text())["parent_sha"]
                changed = dict(_synthetic_screen(sub))
                changed[it0.item_id] = dict(changed[it0.item_id], third_party_interest="a DIFFERENT interest")
                set_screen(changed, "selftest2")
                n_before = len(os.listdir(td))
                rcd.run_deliberation("stub-model", rcd.THIRD_PERSON, it0, 0, scaffold="narrative_cot")
                sha_after = json.loads(p0.read_text())["parent_sha"]
                check("cache landmine: a changed screen record changes the third-party seat's "
                      "parent_sha and the r0 is regenerated, not replayed",
                      sha_before != sha_after and len(os.listdir(td)) == n_before)
                pw = rcd.call_cache_path("stub-model", "narrative_cot", rcd.THIRD_PERSON, it0.item_id, 0,
                                         "r0", "writer_advocate_np", 2560)
                check("cache landmine: the other seats' r0 parent_sha is over the post text only "
                      "(their prompt carries no screen text)",
                      json.loads(pw.read_text())["parent_sha"] == rcd._parent_sha(it0.post_text))
                set_screen(_synthetic_screen(sub), "selftest")
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
                syn_screen2 = {"items": [dict(v, item_id=k) for k, v in _synthetic_screen(sub).items()]}
                screen_path2 = Path(td2) / "third_party_screen_selftest.json"
                screen_path2.write_text(json.dumps(syn_screen2))
                two = _BASE_SUBSET_ITEMS(items, 1, 1)
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
                            "--workers", "1", "--n-boot", "20", "--screen", str(screen_path2)]
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc_a = main(run_args)
                out_a = buf.getvalue()
                names_a = sorted(os.listdir(td2))
                cpath = Path(td2) / f"{DEFAULT_TAG}_completeness.json"
                comp_a = json.loads(cpath.read_text()) if cpath.exists() else {}
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
                comp_c = json.loads(cpath.read_text())
                import csv as _csv
                n_rows_c = len(list(_csv.DictReader((Path(td2) / f"{DEFAULT_TAG}_rows.csv").open())))
                check("run path: --run --resume replays the cached calls, completes the missing debate, "
                      "passes both guards (exit 0) and leaves 4 rows under the readable tag",
                      rc_c == 0 and comp_c["pass"] and comp_c["n_rows"] == 4 and n_rows_c == 4
                      and "GUARD FAILED" not in buf.getvalue())
                check("run path: the screen manifest names the screen file and its sha",
                      json.loads((Path(td2) / f"{DEFAULT_TAG}_screen_manifest.json").read_text())["n_screened_in"] == 16)
            finally:
                rcd.PRICES.pop("stub-model", None)
                rcd.OUT_DIR = real_out
                set_screen(_synthetic_screen(sub), "selftest")
    finally:
        rcd.generate_any = real_gen
    after = sorted(n for n in os.listdir(OUT_DIR) if any(n.endswith(f"_{rid}.json") for rid in ROLE_IDS_NP)
                   or f"_{MODERATOR_PREFIX}-" in n or n.startswith(f"{DEFAULT_TAG}_"))
    check("nothing landed in the REAL output directory under any n-perspective name",
          after == snapshot)
    check("registered panel: the grok content-filter screen leaves 210 of 249 items",
          len([i for i in items if i.item_id not in rcd.filter_prone_items("grok-4-1-fast-reasoning")]) == 210)
    if screen_on_disk:
        screen = load_screen(SCREEN_PATH)
        keep = [i for i in items if i.item_id in SCREEN]
        keep_f = [i for i in keep if i.item_id not in rcd.filter_prone_items("grok-4-1-fast-reasoning")]
        sm = screen_summary(keep_f)
        check(f"the real screen on disk selects a non-empty subset of the registered panel "
              f"({len(keep)} screened, {len(keep_f)} after the content-filter screen; "
              f"stakes {sm['stake_counts']}, k=2 reachable on {sm['k2_reachable_items']})",
              0 < len(keep_f) <= 210 and all(SCREEN[i.item_id]["description"] for i in keep_f))
        check("every screened record builds a third-party block (no missing description / stake)",
              all(BLOCK_BEGIN in third_party_block(i.item_id) for i in keep_f))
    else:
        print("  (no third_party_screen.json on disk; run scripts/screen_third_party.py first)")

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        add_help=False,
        description="Addendum 16.21 n perspectives: four seats incl. a named third party, "
                    "edges on, panel restricted to the third-party screen. Generates "
                    "NOTHING unless --run is passed.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true", help="REQUIRED to generate; spends money")
    ap.add_argument("--resume", action="store_true",
                    help="proceed despite a non-empty cache footprint for these models; "
                         "only valid when the briefs AND the screen are known unchanged")
    ap.add_argument("--verify-cache-clean", action="store_true")
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="comma-separated agent models; each must be in "
                         "run_crowdgold_deliberation.PRICES to --run")
    ap.add_argument("--moderator-model", default=None)
    ap.add_argument("--screen", default=str(SCREEN_PATH),
                    help="third-party screen JSON (scripts/screen_third_party.py)")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    known, rest = ap.parse_known_args(argv)
    if known.selftest:
        return _selftest()

    for a in rest:
        if a.split("=", 1)[0] in BANNED_FLAGS:
            print(f"\nERROR: {a} is banned in the n-perspective cell (stake knobs inject "
                  "assigned-interest text into the readout rounds; --smoke/--full/--pilot/"
                  "--keep-filtered change the panel or the generator; --no-unresolved/"
                  "--transcript-cap change prompt text the cache key cannot see). Refusing.\n")
            return 4
    if not tag_ok(known.tag):
        print(f"\nERROR: --tag {known.tag!r} must start with {DEFAULT_TAG!r}; any other tag "
              "could overwrite a sibling cell's CSVs.\n")
        return 2
    models = [m.strip() for m in known.models.split(",") if m.strip()]
    if not models:
        print("\nERROR: --models is empty\n")
        return 2
    try:
        load_screen(Path(known.screen))
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"\nERROR: third-party screen unreadable at {known.screen}: {e!r}. Run "
              "python -m scripts.screen_third_party first.\n")
        return 2
    if not SCREEN:
        print(f"\nERROR: the screen at {known.screen} selects no items.\n")
        return 2

    clean, counts = cache_is_clean(models, known.moderator_model)
    if known.verify_cache_clean:
        print(f"cache files per n-perspective namespace: {counts}")
        print("CLEAN -- every call will be a genuine generation." if clean else
              "DIRTY -- refusing; a previous run under different briefs or a different "
              "screen could be replayed silently.")
        return 0 if clean else 3

    install()
    fwd = list(rest) + ["--models", ",".join(models), "--tag", known.tag]
    if known.moderator_model:
        fwd += ["--moderator-model", known.moderator_model]
    print(f"[nperspective] seats {rcd.ROLE_ORDER}  stakes "
          f"{tuple(r.stake for r in rcd.ROLES)}  moderator prefix {MODERATOR_PREFIX}  "
          f"tag {known.tag}  edges ON")
    print(f"[nperspective] screen {known.screen} (sha {SCREEN_FILE_SHA}): {len(SCREEN)} items "
          "screened in; the panel below is the registered subset restricted to them")
    print(f"[nperspective] cache plan: all {rcd.CALLS_PER_CELL} calls per debate are NEW; "
          "no seat or moderator record can replay from any existing run; the third-party "
          "seat's parent_sha carries the screen record")
    try:
        panel = rcd.load_items(source="scruples", n_yta=99, n_nta=150, min_votes=50,
                               min_consensus=0.90, seed=44)
        keep = [i for i in panel if i.item_id in SCREEN]
        drop: set = set()
        for m in models:
            drop |= rcd.filter_prone_items(m)
        keep = [i for i in keep if i.item_id not in drop]
        sm = screen_summary(keep)
        print(f"[nperspective] after the content-filter screen: {sm['n_items']} items; "
              f"stake counts {sm['stake_counts']} (source {sm['stake_source_counts']}); "
              f"k=2 reachable on {sm['k2_reachable_items']} items")
    except Exception as e:  # noqa: BLE001 -- the base runner reports panel problems itself
        print(f"[nperspective] screen summary unavailable: {e!r}")

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
              "A previous partial run under possibly-different briefs or a different "
              "screen would be replayed silently (role briefs are not covered by "
              "parent_sha).\nPass --resume only if you are certain the briefs and the "
              "screen are unchanged.\n")
        return 3
    if not clean:
        print(f"  --resume: {counts} existing records will be served from cache")
    manifest = rcd.OUT_DIR / f"{known.tag}_screen_manifest.json"
    manifest.write_text(json.dumps({
        "screen_file": str(known.screen), "screen_file_sha": SCREEN_FILE_SHA,
        "n_screened_in": len(SCREEN),
        "items": {k: v for k, v in sorted(SCREEN.items())},
    }, indent=2, ensure_ascii=False))
    print(f"[nperspective] wrote {manifest}")
    print(f"[nperspective] delegating to run_crowdgold_deliberation with: {' '.join(fwd)}")
    rc = rcd.main(fwd)
    return finish_run(known.tag, rc, models, rest)


if __name__ == "__main__":
    raise SystemExit(main())
