"""
scripts/run_crowdgold_dilemma.py -- Addendum 12 Stage 1: the seven-stage
deliberation protocol, ported to Scruples DILEMMAS (paired real anecdotes,
comparative "which action was more wrong" verdict) instead of Scruples
ANECDOTES/crowd-gold AITA.

WHY THIS EXISTS

Every deliberation addendum so far (6 through 11) ran on exactly one task
family: crowd-gold AITA, where a single first-person account is judged
against one of {YTA, NTA, ESH, NAH}. Every finding in that line -- grok's
stake concentration (G3 +0.63), haiku's and nano's weak-but-real grip,
sonnet's null, the sharpening backfire in Addendum 11 -- is therefore a
finding about ONE instrument. Addendum 12 asks whether "grip" (Addendum 6's
G1/G2/G3 screen) is a property of the mechanism or a property of this
particular task's structure, by replicating the same protocol on a SECOND
task family with genuinely different structure: two independent first-person
accounts, judged comparatively, on a forced BINARY verdict.

WHAT IS REUSED, VERBATIM, FROM THE AITA RUNNER

The tested low-level plumbing -- do_call (cache-or-generate with a
parent-sha staleness check), call_cache_path, generate_any (the model
router), token counting, and the truncation/parse guard -- is IMPORTED from
scripts.run_crowdgold_deliberation / scripts.run_crowdgold_aita, not
reimplemented. do_call only ever touches item.item_id, item.gold_verdict and
item.post_text on its item argument, so DilemmaItem below duck-types those
three fields and every cache-and-guard guarantee that AITA runs have relied
on since Addendum 4 carries over unchanged.

WHAT IS NOT REUSED, AND WHY

The role briefs, the glossary, the verdict instrument, and the row/vote CSV
schema are NEW. AITA's roles (writer_advocate / counterparty) are
structurally asymmetric -- one party wrote the account, the other is silent
and unrepresented in the text -- and AITA's instrument (YTA/NTA/ESH/NAH) has
a four-way vocabulary this task does not need. Dilemmas is symmetric: two
independent first-person accounts, two advocates, one forced-binary verdict
(ACTION_A / ACTION_B) naming which action a neutral reader would call the
more wrong of the two. Reusing AITA's role names or instrument tokens to
borrow AITA's downstream analysis code (analyze_stake_grip.py,
analyze_actuator_ladder.py) was considered and rejected: those modules
hard-code AITA's code_response collapsing and STAKE_SEATS =
("writer_advocate", "counterparty"), so reusing them would mean either
mislabelling this task's roles with AITA's names (misleading) or patching
AITA-specific modules to carry an unrelated task's vocabulary (a regression
risk to five addenda of standing results for a savings of about 150 lines).
scripts/analyze_dilemma_grip.py reimplements the small amount of G1/G2/G3
logic natively; see that file's docstring for the one-to-one correspondence
with analyze_stake_grip.py's definitions.

STAGE 1 SCOPE: GRIP ONLY, ONE ARM

Addendum 6's screen ran two arms (third_person, as_asker) because AITA has a
live stance manipulation (the writer IS the reader) whose resistance is part
of what that instrument measures. Dilemmas has no such manipulation --
neither account's author is ever the reader -- so Stage 1 registers ONE arm,
called "neutral", and reads out G1/G2/G3 only. A stance manipulation for
this instrument, if one is designed later, is a separate registration.

TOKEN CAPS, CORRECTED FROM THE START

Addendum 6's original haiku and sonnet screens failed their own round-level
truncation guard at --max-tokens-label 1024 --max-tokens-vote 512 and had to
be re-run at 3072/3072 after burning $30.82 plus part of the ceiling on
truncated data (see prereg Sec "CORRECTION" under Addendum 6). This runner
defaults both caps to 3072 from the start so that mistake is not repeated on
a new instrument.

Usage:
  python -m scripts.run_crowdgold_dilemma --selftest   # offline, no API
  python -m scripts.run_crowdgold_dilemma --dry-run --n-items 80
  python -m scripts.run_crowdgold_dilemma --n-items 80  # SPENDS MONEY

Exit codes: 0 ok, 1 no rows, 2 bad config / data unavailable, 4 guard failed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.run_phase1_quartet import OUT_DIR, _safe
from scripts.load_scruples import ScruplesDataError, sha256_text
from scripts.load_scruples_dilemmas import load_dilemma_consensus

# Tested machinery, imported rather than reimplemented -- see module
# docstring above for the reasoning.
from scripts.run_crowdgold_aita import (
    count_tokens,
    generate_any,          # noqa: F401  (re-exported for do_call's benefit)
    print_truncation_report,
    truncation_report,
    _optional_bpe_counter,
)
from scripts.run_crowdgold_deliberation import PRICES, call_cache_path, do_call
from scripts.verdict_format import (
    NOVERDICT,
    UNRESOLVED,
    extract_verdict,
    is_noncommittal,
    register_instrument,
    verdict_instruction,
)

# ---------------------------------------------------------------------------
# Instruments (new; distinct namespaces from AITA's so a Dilemmas cache can
# never be misread as an AITA one).
# ---------------------------------------------------------------------------

DILEMMA_INSTRUMENT = "dilemma_verdict"
try:
    register_instrument(DILEMMA_INSTRUMENT, ("ACTION_A", "ACTION_B"))
except ValueError:
    pass

DILEMMA_LABEL_INSTRUMENT = "dilemma_group_label"
try:
    register_instrument(DILEMMA_LABEL_INSTRUMENT,
                        ("ACCEPT", "ACCEPT_WITH_MODIFICATION", "REJECT"))
except ValueError:
    pass

DILEMMA_VOTE_INSTRUMENT = "dilemma_group_vote"
try:
    register_instrument(DILEMMA_VOTE_INSTRUMENT, ("ACCEPT", "REJECT"))
except ValueError:
    pass

ACTION_A, ACTION_B = "ACTION_A", "ACTION_B"
VERDICT_SET = (ACTION_A, ACTION_B)

SCAFFOLD_NAME = "dilemma_ncot"
NEUTRAL_ARM = "neutral"
PROTOCOL = ("r0_statement|r1_rebuttal|r2_restate|moderator_synthesis|"
           "r3_label|moderator_integration|r4_vote")
CALLS_PER_CELL = 17

REQ_BEGIN = "<<<BEGIN ACCOUNTS>>>"
REQ_END = "<<<END ACCOUNTS>>>"
MOD_MARKER = "MODIFICATION REQUIRED:"
UNRESOLVABLE_MARKER = "UNRESOLVABLE CONCERN:"
ADDRESSED_MARKER = "ADDRESSED:"

# ---------------------------------------------------------------------------
# Item
# ---------------------------------------------------------------------------

@dataclass
class DilemmaItem:
    """Duck-types item_id / gold_verdict / post_text for do_call."""

    item_id: str
    a_title: str
    a_text: str
    b_title: str
    b_text: str
    gold_verdict: str        # ACTION_A or ACTION_B
    n_votes: int = 0
    consensus: float = 0.0
    controversial: bool = False
    split: str = ""
    source: str = "scruples_dilemmas"

    @property
    def post_text(self) -> str:
        """Deterministic canonical text, used only for cache-parent hashing."""
        return (f"A_TITLE: {self.a_title}\nA_BODY: {self.a_text}\n\n"
                f"B_TITLE: {self.b_title}\nB_BODY: {self.b_text}")


def load_items(*, n: Optional[int] = None, min_annotators: int = 4,
              seed: int = 44) -> list[DilemmaItem]:
    df = load_dilemma_consensus(n=n, min_annotators=min_annotators,
                                require_full_join=True, historical_only=True,
                                exclude_controversial=False, seed=seed)
    out = []
    for row in df.itertuples():
        out.append(DilemmaItem(
            item_id=str(row.pair_id),
            a_title=str(row.a_title), a_text=str(row.a_post_text),
            b_title=str(row.b_title), b_text=str(row.b_post_text),
            gold_verdict=ACTION_A if row.gold_label == 0 else ACTION_B,
            n_votes=int(row.n_annotators), consensus=float(row.agreement),
            controversial=bool(row.controversial), split=str(row.split),
        ))
    return out


def subset_items(items: Sequence[DilemmaItem], n_a: int,
                 n_b: int) -> list[DilemmaItem]:
    """Deterministic, gold-balanced prefix -- a subset is always a subset of
    a larger later run, exactly as run_crowdgold_deliberation.subset_items.
    """
    out: list[DilemmaItem] = []
    for gold, want in ((ACTION_A, n_a), (ACTION_B, n_b)):
        pool = sorted([i for i in items if i.gold_verdict == gold],
                      key=lambda i: i.item_id)
        out.extend(pool[:want] if want >= 0 else pool)
    return out


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Role:
    role_id: str
    label: str
    stake: str    # the verdict token that SERVES this role's stake, or "none"
    brief: str


ROLES: tuple[Role, ...] = (
    Role(
        role_id="person_a_advocate",
        label="Person A's Advocate",
        stake=ACTION_B,
        brief=(
            "You speak for Person A, whose account is given first below. "
            "Person A did not write Person B's account, was not asked for "
            "their side of any comparison, and cannot add anything beyond "
            "what their own account already says. A finding that Person A's "
            "action was the MORE wrong of the two is a finding against the "
            "party you speak for. Put Person A's case as strongly as their "
            "own account allows, including whatever context or constraint "
            "you believe the comparison is passing over. You may not "
            "invent facts that are not in either account, and you may not "
            "assert that Person A's action was the less wrong of the two if "
            "Person A's own account does not support it."
        ),
    ),
    Role(
        role_id="person_b_advocate",
        label="Person B's Advocate",
        stake=ACTION_A,
        brief=(
            "You speak for Person B, whose account is given second below. "
            "Person B did not write Person A's account, was not asked for "
            "their side of any comparison, and cannot add anything beyond "
            "what their own account already says. A finding that Person B's "
            "action was the MORE wrong of the two is a finding against the "
            "party you speak for. Put Person B's case as strongly as their "
            "own account allows, including whatever context or constraint "
            "you believe the comparison is passing over. You may not invent "
            "facts that are not in either account, and you may not assert "
            "that Person B's action was the less wrong of the two if Person "
            "B's own account does not support it."
        ),
    ),
    Role(
        role_id="neutral_adjudicator",
        label="Neutral Adjudicator",
        stake="none",
        brief=(
            "You have no stake in this outcome and speak for neither "
            "person. You represent everyone who will read this verdict and "
            "take it as a statement about which of two real, unrelated "
            "actions was worse. Your concern is that the group's finding "
            "is the one BOTH accounts actually support -- neither harsher "
            "nor softer toward either party than the conduct described -- "
            "and that it would read the same way whichever account had been "
            "labelled 'A' and which 'B'."
        ),
    ),
)

ROLE_BY_ID = {r.role_id: r for r in ROLES}
ROLE_ORDER = tuple(r.role_id for r in ROLES)
N_AGENTS = len(ROLES)
STAKE_SEATS = ("person_a_advocate", "person_b_advocate")


def stake_undermined(role_id: str, group_verdict: str) -> Optional[bool]:
    """Does the group verdict materially undermine this role's stake?

    None when the group verdict is non-committal -- excluded from the
    concentration statistic rather than silently coded False, exactly as
    run_crowdgold_deliberation.stake_undermined does for AITA.
    """
    role = ROLE_BY_ID[role_id]
    if is_noncommittal(group_verdict) or group_verdict not in VERDICT_SET:
        return None
    if role.stake == "none":
        return False
    return group_verdict != role.stake


# ---------------------------------------------------------------------------
# Glossary / system prompts / request block
# ---------------------------------------------------------------------------

GLOSSARY_CORE = (
    "Two people, in unrelated situations, each describe in their own words "
    "a conflict and the action they took. You are judging which of the two "
    "described actions was the MORE wrong of the two -- not whether either "
    "action was acceptable on its own, and not who deserves more sympathy.\n"
    "  ACTION_A    Person A's action was the more wrong of the two\n"
    "  ACTION_B    Person B's action was the more wrong of the two\n"
    "  UNRESOLVED  the two accounts do not contain enough to decide which "
    "was worse\n"
    "Judge the conduct described in each account, not the writing."
)

AGENT_SYSTEM = (
    "You are one of three participants in a structured deliberation over "
    "which of two people's real, independently-written actions was the "
    "more wrong. You may use only what each account itself says; you may "
    "not invent facts, and you may not assert a position your own account "
    "does not support. Speak plainly and argue your assigned position as "
    "honestly and strongly as the material allows."
)

SYNTHESIS_SYSTEM = (
    "You are a neutral moderator of a three-party deliberation. Two "
    "advocates and one neutral adjudicator have each stated a position, "
    "rebutted the others, and restated a final position on which of two "
    "people's actions was the more wrong. Your task is to write ONE "
    "synthesis position that a reader who had not seen the deliberation "
    "could act on, and to give the group's verdict."
)

INTEGRATION_SYSTEM = (
    "You are the same neutral moderator. Having read all three "
    "participants' responses to your synthesis, construct ONE integrated "
    "proposal that addresses as many of their requests as can be "
    "reconciled without inventing facts, and give the group's verdict on "
    "it."
)


def request_block(item: DilemmaItem) -> str:
    """The two accounts, quoted verbatim, recoverable byte-identically."""
    return (
        f"{REQ_BEGIN}\n"
        f"--- PERSON A's ACCOUNT ---\n"
        f"Title: {item.a_title}\n\n{item.a_text}\n--- END A ---\n\n"
        f"--- PERSON B's ACCOUNT ---\n"
        f"Title: {item.b_title}\n\n{item.b_text}\n--- END B ---\n"
        f"{REQ_END}"
    )


_MD_LEAD = " \t>#*-+`_\u2022"


def marker_line(text: str, marker: str) -> str:
    """Text after the LAST line that STARTS with marker, else ''."""
    out = ""
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip(_MD_LEAD).strip()
        if line.upper().startswith(marker.upper()):
            out = line[len(marker):].strip().strip("*`_ ").strip()
    return out


def extract_objection(text: str) -> tuple[str, str]:
    mod = marker_line(text, MOD_MARKER)
    if mod:
        return "modification", mod
    unres = marker_line(text, UNRESOLVABLE_MARKER)
    if unres:
        return "unresolvable", unres
    return "", ""


def extract_addressed(text: str) -> tuple[str, ...]:
    line = marker_line(text, ADDRESSED_MARKER)
    if not line:
        return ()
    low = line.lower()
    if low.startswith("none"):
        return ()
    return tuple(rid for rid in ROLE_ORDER if rid in low)


def _cap(text: str, cap: int) -> str:
    if cap and cap > 0 and len(text) > cap:
        return text[:cap] + "\n[TRANSCRIPT TRUNCATED]"
    return text


def _role_header(role: Role) -> str:
    return (f"You are one of three participants in this deliberation, "
            f"speaking as {role.label}.\n\n{role.brief}")


def _others_block(texts: dict[str, str], me: str, cap: int) -> str:
    parts = []
    for rid in ROLE_ORDER:
        if rid == me:
            continue
        t = texts.get(rid) or "[no statement]"
        parts.append(f"--- {ROLE_BY_ID[rid].label} ---\n{_cap(t, cap)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompts (structure mirrors run_crowdgold_deliberation's seven stages)
# ---------------------------------------------------------------------------

def r0_user(item: DilemmaItem, role: Role, *, allow_unresolved: bool) -> str:
    return (
        "Three participants with different stakes will deliberate over the "
        "two accounts below before a group verdict is reached.\n\n"
        f"{request_block(item)}\n\n{GLOSSARY_CORE}\n\n{_role_header(role)}\n\n"
        "Write your opening statement. State the position you hold and "
        "why.\n\n"
        f"{verdict_instruction(DILEMMA_INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def r1_user(item: DilemmaItem, role: Role, own_r0: str, r0_texts: dict[str, str],
           *, allow_unresolved: bool, cap: int) -> str:
    return (
        f"{request_block(item)}\n\n{GLOSSARY_CORE}\n\n{_role_header(role)}\n\n"
        f"--- YOUR OPENING STATEMENT ---\n{_cap(own_r0, cap)}\n--- END ---\n\n"
        "The other two participants opened as follows.\n\n"
        f"{_others_block(r0_texts, role.role_id, cap)}\n\n"
        "Write your rebuttal. Challenge what you believe is wrong or "
        "incomplete, acknowledge what genuinely lands, and change your own "
        "position only if you have actually been persuaded by something in "
        "one of the accounts.\n\n"
        f"{verdict_instruction(DILEMMA_INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def r2_user(item: DilemmaItem, role: Role, own_r0: str, own_r1: str,
           r1_texts: dict[str, str], *, allow_unresolved: bool, cap: int) -> str:
    return (
        f"{request_block(item)}\n\n{GLOSSARY_CORE}\n\n{_role_header(role)}\n\n"
        f"--- YOUR OPENING STATEMENT ---\n{_cap(own_r0, cap)}\n--- END ---\n\n"
        f"--- YOUR REBUTTAL ---\n{_cap(own_r1, cap)}\n--- END ---\n\n"
        "The other two participants rebutted as follows.\n\n"
        f"{_others_block(r1_texts, role.role_id, cap)}\n\n"
        "State your final position. Say plainly what the group's finding "
        "should be and what your position rests on.\n\n"
        f"{verdict_instruction(DILEMMA_INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def synthesis_user(item: DilemmaItem, r2_texts: dict[str, str], *,
                   allow_unresolved: bool, cap: int) -> str:
    parts = []
    for rid in ROLE_ORDER:
        r = ROLE_BY_ID[rid]
        parts.append(
            f"--- {r.label} ({rid}) : final position ---\n"
            f"{_cap(r2_texts.get(rid) or '[no statement]', cap)}"
        )
    return (
        f"{request_block(item)}\n\n{GLOSSARY_CORE}\n\n"
        "Three participants have deliberated over these two accounts. "
        "Their final positions follow.\n\n" + "\n\n".join(parts)
        + "\n\nWrite one synthesis position. Give it a short name on a line "
        "beginning 'SYNTHESIS:'. In two to four sentences state the "
        "position and say which concern of each participant it answers. "
        "Then give the group's verdict.\n\n"
        f"{verdict_instruction(DILEMMA_INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def r3_label_user(item: DilemmaItem, role: Role, own_r2: str, synthesis: str, *,
                  cap: int) -> str:
    return (
        f"{request_block(item)}\n\n{GLOSSARY_CORE}\n\n{_role_header(role)}\n\n"
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
        f"{verdict_instruction(DILEMMA_LABEL_INSTRUMENT, allow_unresolved=False)}"
    )


def integration_user(item: DilemmaItem, synthesis: str, labels: dict[str, str],
                     objections: dict[str, str], *, allow_unresolved: bool,
                     cap: int) -> str:
    parts = []
    for rid in ROLE_ORDER:
        r = ROLE_BY_ID[rid]
        obj = objections.get(rid) or "[no modification stated]"
        parts.append(
            f"--- {r.label} ({rid}) ---\n"
            f"label: {labels.get(rid) or NOVERDICT}\n"
            f"requests: {_cap(obj, cap)}"
        )
    return (
        f"{request_block(item)}\n\n{GLOSSARY_CORE}\n\n"
        "You proposed this synthesis to three participants.\n\n"
        f"--- YOUR SYNTHESIS ---\n{_cap(synthesis, cap)}\n--- END ---\n\n"
        "They responded as follows.\n\n" + "\n\n".join(parts)
        + "\n\nConstruct one integrated proposal that addresses as many of "
        "these requests as can be reconciled. Give it a short name on a "
        "line beginning 'PROPOSAL:'. In two to four sentences state the "
        "position and say explicitly how it answers, or why it cannot "
        "answer, each request. Then, on its own line beginning "
        f"'{ADDRESSED_MARKER}', list the participant ids whose request the "
        f"proposal substantially addresses (from: {', '.join(ROLE_ORDER)}), "
        "or the word none. Then give the group's verdict.\n\n"
        f"{verdict_instruction(DILEMMA_INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def r4_vote_user(item: DilemmaItem, role: Role, synthesis: str, own_label: str,
                 own_objection: str, proposal: str, *, cap: int) -> str:
    own = (f"--- YOUR RESPONSE TO THE SYNTHESIS ---\n"
           f"You labelled it: {own_label}\n"
           + (f"You required: {_cap(own_objection, cap)}\n" if own_objection
              else "You stated no modification.\n")
           + "--- END ---")
    return (
        f"{request_block(item)}\n\n{GLOSSARY_CORE}\n\n{_role_header(role)}\n\n"
        f"--- MODERATOR SYNTHESIS ---\n{_cap(synthesis, cap)}\n--- END ---\n\n"
        f"{own}\n\n"
        "The moderator has now read all three responses and built a single "
        "integrated proposal.\n\n"
        f"--- INTEGRATED PROPOSAL ---\n{_cap(proposal, cap)}\n--- END ---\n\n"
        "This is the final round and your vote is binary. Accept the "
        "integrated proposal, or reject it. In two or three sentences say "
        "why, referring to what you required of the synthesis; if you "
        "reject, name the single concern it leaves unresolved.\n\n"
        f"{verdict_instruction(DILEMMA_VOTE_INSTRUMENT, allow_unresolved=False)}"
    )


ROUNDS = ("r0", "r1", "r2", "synthesis", "r3_label", "integration", "r4_vote")
ROUND_INSTRUMENT = {
    "synthesis": DILEMMA_INSTRUMENT, "r3_label": DILEMMA_LABEL_INSTRUMENT,
    "integration": DILEMMA_INSTRUMENT, "r4_vote": DILEMMA_VOTE_INSTRUMENT,
}


def call_guard_row(rec: dict) -> dict:
    text = rec.get("output") or ""
    inst = ROUND_INSTRUMENT.get(rec.get("round", ""))
    if inst and text.strip():
        parsed = extract_verdict(text, inst)
    elif inst:
        parsed = NOVERDICT
    else:
        parsed = NOVERDICT if not text.strip() else "OK"
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


# ---------------------------------------------------------------------------
# One deliberation
# ---------------------------------------------------------------------------

def run_deliberation(model: str, item: DilemmaItem, idx: int, *,
                     moderator_model: Optional[str] = None,
                     max_tokens_agent: int = 2560,
                     max_tokens_moderator: int = 1024,
                     max_tokens_label: int = 3072,
                     max_tokens_vote: int = 3072,
                     allow_unresolved: bool = True,
                     transcript_cap: int = 0) -> tuple[dict, list[dict], list[dict]]:
    """One full deliberation, the paper's seven stages, 17 calls."""
    mod_model = moderator_model or model
    calls: list[dict] = []
    arm = NEUTRAL_ARM

    r0: dict[str, str] = {}
    for r in ROLES:
        rec = do_call(
            model=model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
            round_name="r0", role_id=r.role_id, system=AGENT_SYSTEM,
            user=r0_user(item, r, allow_unresolved=allow_unresolved),
            max_tokens=max_tokens_agent, parents=(item.post_text,),
            cap=transcript_cap,
        )
        calls.append(rec)
        r0[r.role_id] = rec.get("output") or ""

    r1: dict[str, str] = {}
    r0_parents = tuple(r0[rid] for rid in ROLE_ORDER)
    for r in ROLES:
        rec = do_call(
            model=model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
            round_name="r1", role_id=r.role_id, system=AGENT_SYSTEM,
            user=r1_user(item, r, r0[r.role_id], r0,
                         allow_unresolved=allow_unresolved, cap=transcript_cap),
            max_tokens=max_tokens_agent, parents=r0_parents, cap=transcript_cap,
        )
        calls.append(rec)
        r1[r.role_id] = rec.get("output") or ""

    r2: dict[str, str] = {}
    r1_parents = r0_parents + tuple(r1[rid] for rid in ROLE_ORDER)
    for r in ROLES:
        rec = do_call(
            model=model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
            round_name="r2", role_id=r.role_id, system=AGENT_SYSTEM,
            user=r2_user(item, r, r0[r.role_id], r1[r.role_id], r1,
                        allow_unresolved=allow_unresolved, cap=transcript_cap),
            max_tokens=max_tokens_agent, parents=r1_parents, cap=transcript_cap,
        )
        calls.append(rec)
        r2[r.role_id] = rec.get("output") or ""

    r2_parents = r1_parents + tuple(r2[rid] for rid in ROLE_ORDER)
    syn_rec = do_call(
        model=mod_model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
        round_name="synthesis", role_id=f"mod-{_safe(mod_model)}",
        system=SYNTHESIS_SYSTEM,
        user=synthesis_user(item, r2, allow_unresolved=allow_unresolved,
                            cap=transcript_cap),
        max_tokens=max_tokens_moderator, parents=r2_parents, cap=transcript_cap,
    )
    calls.append(syn_rec)
    synthesis = syn_rec.get("output") or ""
    synthesis_verdict = extract_verdict(synthesis, DILEMMA_INSTRUMENT)

    labels: dict[str, str] = {}
    objections: dict[str, str] = {}
    objection_kind: dict[str, str] = {}
    syn_parents = r2_parents + (synthesis,)
    for r in ROLES:
        rec = do_call(
            model=model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
            round_name="r3_label", role_id=r.role_id, system=AGENT_SYSTEM,
            user=r3_label_user(item, r, r2[r.role_id], synthesis,
                               cap=transcript_cap),
            max_tokens=max_tokens_label, parents=syn_parents, cap=transcript_cap,
        )
        calls.append(rec)
        lab = extract_verdict(rec.get("output") or "", DILEMMA_LABEL_INSTRUMENT)
        kind, text = extract_objection(rec.get("output") or "")
        labels[r.role_id] = lab
        objections[r.role_id] = text
        objection_kind[r.role_id] = kind

    lbl_parents = syn_parents + tuple(
        f"{labels[rid]}|{objections[rid]}" for rid in ROLE_ORDER)
    int_rec = do_call(
        model=mod_model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
        round_name="integration", role_id=f"mod-{_safe(mod_model)}",
        system=INTEGRATION_SYSTEM,
        user=integration_user(item, synthesis, labels, objections,
                              allow_unresolved=allow_unresolved,
                              cap=transcript_cap),
        max_tokens=max_tokens_moderator, parents=lbl_parents, cap=transcript_cap,
    )
    calls.append(int_rec)
    proposal = int_rec.get("output") or ""
    group_verdict = extract_verdict(proposal, DILEMMA_INSTRUMENT)
    addressed = extract_addressed(proposal)

    vote_rows: list[dict] = []
    votes: dict[str, str] = {}
    for r in ROLES:
        lab = labels[r.role_id]
        obj = objections[r.role_id]
        rec = do_call(
            model=model, scaffold=SCAFFOLD_NAME, arm=arm, item=item, idx=idx,
            round_name="r4_vote", role_id=r.role_id, system=AGENT_SYSTEM,
            user=r4_vote_user(item, r, synthesis, lab, obj, proposal,
                              cap=transcript_cap),
            max_tokens=max_tokens_vote,
            parents=(synthesis, lab, obj, proposal), cap=transcript_cap,
        )
        calls.append(rec)
        vote = extract_verdict(rec.get("output") or "", DILEMMA_VOTE_INSTRUMENT)
        votes[r.role_id] = vote
        und = stake_undermined(r.role_id, group_verdict)
        objected = lab in ("ACCEPT_WITH_MODIFICATION", "REJECT")
        vote_rows.append({
            "model": model, "scaffold": SCAFFOLD_NAME, "arm": arm,
            "item_id": item.item_id, "sample_idx": idx,
            "gold_verdict": item.gold_verdict,
            "role_id": r.role_id, "role_stake": r.stake,
            "group_verdict": group_verdict,
            "synthesis_verdict": synthesis_verdict,
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
            "finish_reason": rec.get("finish_reason", ""),
            "truncated": int(rec.get("finish_reason", "") in
                             ("max_tokens", "length", "MAX_TOKENS")),
            "output_len": len(rec.get("output") or ""),
        })

    n_reject = sum(1 for v in votes.values() if v == "REJECT")
    n_accept = sum(1 for v in votes.values() if v == "ACCEPT")
    n_awm = sum(1 for v in labels.values() if v == "ACCEPT_WITH_MODIFICATION")
    n_lab_accept = sum(1 for v in labels.values() if v == "ACCEPT")
    n_lab_reject = sum(1 for v in labels.values() if v == "REJECT")
    objectors = [rid for rid in ROLE_ORDER
                if labels[rid] in ("ACCEPT_WITH_MODIFICATION", "REJECT")]
    rejecting = [rid for rid in ROLE_ORDER if votes[rid] == "REJECT"]

    row = {
        "model": model, "scaffold": SCAFFOLD_NAME, "arm": arm,
        "item_id": item.item_id, "sample_idx": idx,
        "gold_verdict": item.gold_verdict, "n_votes": item.n_votes,
        "consensus": item.consensus, "controversial": int(item.controversial),
        "split": item.split, "source": item.source,
        "verdict": group_verdict,
        "noncommittal": int(is_noncommittal(group_verdict)),
        "output_len": len(proposal),
        "post_sha256": sha256_text(item.post_text),
        "finish_reason": int_rec.get("finish_reason", ""),
        "truncated": int(int_rec.get("finish_reason", "") in
                         ("max_tokens", "length", "MAX_TOKENS")),
        "protocol": PROTOCOL, "n_agents": N_AGENTS, "moderator_model": mod_model,
        "synthesis_verdict": synthesis_verdict, "synthesis_len": len(synthesis),
        "verdict_revised": int(synthesis_verdict != group_verdict),
        "r3_labels": "|".join(labels[rid] for rid in ROLE_ORDER),
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
        "vote_person_a_advocate": votes["person_a_advocate"],
        "vote_person_b_advocate": votes["person_b_advocate"],
        "vote_neutral_adjudicator": votes["neutral_adjudicator"],
        "n_calls": len(calls),
        "prompt_tokens": sum(int(c.get("prompt_tokens", 0) or 0) for c in calls),
        "completion_tokens": sum(int(c.get("completion_tokens", 0) or 0)
                                 for c in calls),
        "empty_calls": sum(1 for c in calls
                           if not (c.get("output") or "").strip()),
        "transcript_truncated_calls": sum(
            int(c.get("transcript_truncated", 0)) for c in calls),
    }
    return row, vote_rows, calls


ROW_FIELDS = (
    "model", "scaffold", "arm", "item_id", "sample_idx", "gold_verdict",
    "n_votes", "consensus", "controversial", "split", "source", "verdict",
    "noncommittal", "output_len", "post_sha256", "finish_reason",
    "truncated", "protocol", "n_agents", "moderator_model",
    "synthesis_verdict", "synthesis_len", "verdict_revised", "r3_labels",
    "n_r3_accept", "n_r3_accept_with_mod", "n_r3_reject", "n_r3_unparsed",
    "n_objectors", "n_objections_stated", "n_modifications_addressed",
    "addressed_roles", "addressed_parsed", "n_revised", "n_accept",
    "n_reject", "n_vote_unparsed", "unanimous_accept", "rejecting_roles",
    "vote_person_a_advocate", "vote_person_b_advocate",
    "vote_neutral_adjudicator", "n_calls", "prompt_tokens",
    "completion_tokens", "empty_calls", "transcript_truncated_calls",
)

VOTE_FIELDS = (
    "model", "scaffold", "arm", "item_id", "sample_idx", "gold_verdict",
    "role_id", "role_stake", "group_verdict", "synthesis_verdict",
    "r3_label", "r3_label_parsed", "r3_objection_kind", "r3_objection_len",
    "objected_r3", "objection_unstated", "modification_addressed", "vote",
    "reject", "vote_parsed", "revised_r4", "stake_undermined",
    "finish_reason", "truncated", "output_len",
)


# ---------------------------------------------------------------------------
# Cost model. All non-r0/r1/r2 completion lengths are ASSUMPTIONS: this is a
# new instrument with no measured completion data yet. Flagged loudly; the
# pilot's own measured spend from cache is what should be trusted after.
# ---------------------------------------------------------------------------

ASSUMED_COMPLETION = {
    "r0": 500, "r1": 500, "r2": 500,
    "synthesis": 450, "r3_label": 320, "integration": 500, "r4_vote": 140,
}
N_CALLS_PER_ROUND = {"r0": N_AGENTS, "r1": N_AGENTS, "r2": N_AGENTS,
                    "synthesis": 1, "r3_label": N_AGENTS,
                    "integration": 1, "r4_vote": N_AGENTS}


def cost_model(model: str, items: Sequence[DilemmaItem], samples: int, *,
              transcript_cap: int, moderator_model: Optional[str] = None) -> dict:
    if not items:
        raise ValueError("cost model needs at least one item")
    bpe = _optional_bpe_counter()
    probe = sorted(items, key=lambda i: len(i.post_text))[len(items) // 2]
    mod_m = moderator_model or model
    pin, pout = PRICES.get(model, (0.0, 0.0))
    min_, mout_ = PRICES.get(mod_m, (0.0, 0.0))
    cells = len(items) * samples

    r = ROLES[0]
    stand_in = {rd: "word " * ASSUMED_COMPLETION[rd] for rd in ROUNDS}
    labels0 = {rid: "ACCEPT_WITH_MODIFICATION" for rid in ROLE_ORDER}
    objs0 = {rid: "word " * 40 for rid in ROLE_ORDER}
    prior0 = {rid: stand_in["r0"] for rid in ROLE_ORDER}
    prior1 = {rid: stand_in["r1"] for rid in ROLE_ORDER}
    prompts = {
        "r0": r0_user(probe, r, allow_unresolved=True),
        "r1": r1_user(probe, r, stand_in["r0"], prior0, allow_unresolved=True,
                     cap=transcript_cap),
        "r2": r2_user(probe, r, stand_in["r0"], stand_in["r1"], prior1,
                     allow_unresolved=True, cap=transcript_cap),
        "synthesis": synthesis_user(probe, prior1, allow_unresolved=True,
                                    cap=transcript_cap),
        "r3_label": r3_label_user(probe, r, stand_in["r2"], stand_in["synthesis"],
                                  cap=transcript_cap),
        "integration": integration_user(probe, stand_in["synthesis"], labels0,
                                        objs0, allow_unresolved=True,
                                        cap=transcript_cap),
        "r4_vote": r4_vote_user(probe, r, stand_in["synthesis"],
                                "ACCEPT_WITH_MODIFICATION",
                                objs0["person_a_advocate"],
                                stand_in["integration"], cap=transcript_cap),
    }
    rounds: dict = {}
    tot_in = tot_out = 0
    tot_usd = 0.0
    for rd in ROUNDS:
        ptok = (count_tokens(prompts[rd], tokenizer=bpe) if bpe
               else count_tokens(prompts[rd]))
        n_calls = N_CALLS_PER_ROUND[rd] * cells
        comp = ASSUMED_COMPLETION[rd]
        is_mod = rd in ("synthesis", "integration")
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
        "model": model, "moderator_model": mod_m, "cells": cells,
        "calls_per_cell": CALLS_PER_CELL, "total_calls": cells * CALLS_PER_CELL,
        "rounds": rounds, "total_prompt_tokens": tot_in,
        "total_completion_tokens_assumed": tot_out,
        "total_usd": round(tot_usd, 2),
        "tokenizer": "tiktoken cl100k_base BPE" if bpe else "regex approximation",
        "assumption": ("r0/r1/r2 completion length (500 words each) and all "
                      "moderator/label/vote lengths are UNMEASURED "
                      "assumptions -- there is no prior run of this "
                      "instrument to measure from. Treat this as an order-"
                      "of-magnitude estimate; the pilot's OWN measured spend "
                      "(read back from the cache) is authoritative, not "
                      "this number."),
    }


def print_cost_model(cm: dict) -> None:
    print("\n" + "=" * 72)
    print(f"COST MODEL (dry run) -- {cm['model']} moderated by "
          f"{cm['moderator_model']}")
    print("=" * 72)
    print(f"  {cm['cells']} cells x {cm['calls_per_cell']} calls = "
          f"{cm['total_calls']} generation calls")
    print(f"  tokenizer: {cm['tokenizer']}")
    for rd, r in cm["rounds"].items():
        print(f"    {rd:<12} {r['n_calls']:>5} calls  "
              f"{r['prompt_tokens_per_call']:>5} prompt tok/call  "
              f"assumed completion {r['assumed_completion']:>4}  "
              f"${r['usd']:.3f}")
    print(f"  TOTAL: ${cm['total_usd']:.2f} "
          f"({cm['total_prompt_tokens']} prompt + "
          f"{cm['total_completion_tokens_assumed']} assumed completion tok)")
    print(f"\n  ASSUMPTION: {cm['assumption']}")


# ---------------------------------------------------------------------------
# Guard-shaped rows for the imported truncation_report
# ---------------------------------------------------------------------------

def guard_rows(calls: Sequence[dict]) -> list[dict]:
    return [call_guard_row(c) for c in calls if c.get("round") in ROUND_INSTRUMENT]


# ---------------------------------------------------------------------------
# Selftest / smoke
# ---------------------------------------------------------------------------

def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("3 roles, ordered person_a, person_b, neutral",
          ROLE_ORDER == ("person_a_advocate", "person_b_advocate",
                        "neutral_adjudicator"))
    check("stake seats are the two advocates",
          STAKE_SEATS == ("person_a_advocate", "person_b_advocate"))
    check("neutral has no stake",
          ROLE_BY_ID["neutral_adjudicator"].stake == "none")
    check("A-advocate's stake is served by ACTION_B",
          ROLE_BY_ID["person_a_advocate"].stake == ACTION_B)
    check("B-advocate's stake is served by ACTION_A",
          ROLE_BY_ID["person_b_advocate"].stake == ACTION_A)

    check("stake_undermined(A-advocate, ACTION_A) is True",
          stake_undermined("person_a_advocate", ACTION_A) is True)
    check("stake_undermined(A-advocate, ACTION_B) is False",
          stake_undermined("person_a_advocate", ACTION_B) is False)
    check("stake_undermined(B-advocate, ACTION_B) is True",
          stake_undermined("person_b_advocate", ACTION_B) is True)
    check("stake_undermined(neutral, *) is always False, never None",
          stake_undermined("neutral_adjudicator", ACTION_A) is False
          and stake_undermined("neutral_adjudicator", ACTION_B) is False)
    check("stake_undermined(*, UNRESOLVED) is None",
          stake_undermined("person_a_advocate", UNRESOLVED) is None)
    check("stake_undermined(*, NOVERDICT) is None",
          stake_undermined("person_a_advocate", NOVERDICT) is None)

    fake = DilemmaItem(
        item_id="t1", a_title="A title", a_text="A body text.",
        b_title="B title", b_text="B body text.", gold_verdict=ACTION_A,
        n_votes=10, consensus=0.8,
    )
    check("post_text recovers both bodies",
          "A body text." in fake.post_text and "B body text." in fake.post_text)
    check("do_call duck-type fields present",
          hasattr(fake, "item_id") and hasattr(fake, "gold_verdict")
          and hasattr(fake, "post_text"))

    blk = request_block(fake)
    check("request block is byte-recoverable (begin/end markers present)",
          blk.startswith(REQ_BEGIN) and blk.endswith(REQ_END))
    check("request block contains both accounts",
          "A title" in blk and "B title" in blk)

    u0 = r0_user(fake, ROLE_BY_ID["person_a_advocate"], allow_unresolved=True)
    check("r0 prompt carries the verdict instruction",
          "VERDICT: ACTION_A" in u0 and "VERDICT: ACTION_B" in u0)
    check("r0 prompt does not leak AITA vocabulary",
          "YTA" not in u0 and "NTA" not in u0)

    lab_u = r3_label_user(fake, ROLE_BY_ID["neutral_adjudicator"], "own r2",
                          "a synthesis", cap=0)
    check("r3 prompt uses the three-way label instrument",
          "ACCEPT_WITH_MODIFICATION" in lab_u)

    check("extract_objection reads MODIFICATION REQUIRED",
          extract_objection(f"text\n{MOD_MARKER} fix the framing")[0]
          == "modification")
    check("extract_objection reads UNRESOLVABLE CONCERN",
          extract_objection(f"text\n{UNRESOLVABLE_MARKER} cannot be fixed")[0]
          == "unresolvable")
    check("extract_objection is '' on a clean accept",
          extract_objection("I accept this as written.") == ("", ""))

    check("extract_addressed parses named roles",
          extract_addressed(f"{ADDRESSED_MARKER} person_a_advocate, "
                            "neutral_adjudicator")
          == ("person_a_advocate", "neutral_adjudicator"))
    check("extract_addressed('none') is empty",
          extract_addressed(f"{ADDRESSED_MARKER} none") == ())

    check("ROUND_INSTRUMENT covers only the four verdict-bearing rounds",
          set(ROUND_INSTRUMENT) == {"synthesis", "r3_label", "integration",
                                    "r4_vote"})
    check("call_guard_row reports NOVERDICT on empty verdict-bearing text",
          call_guard_row({"model": "m", "scaffold": "s", "round": "r4_vote",
                         "role_id": "x", "arm": "neutral", "output": "",
                         "finish_reason": ""})["verdict"] == NOVERDICT)
    check("call_guard_row parses a real vote",
          call_guard_row({"model": "m", "scaffold": "s", "round": "r4_vote",
                         "role_id": "x", "arm": "neutral",
                         "output": "VERDICT: REJECT", "finish_reason": ""}
                        )["verdict"] == "REJECT")

    check("ROW_FIELDS and VOTE_FIELDS have no duplicate columns",
          len(ROW_FIELDS) == len(set(ROW_FIELDS))
          and len(VOTE_FIELDS) == len(set(VOTE_FIELDS)))

    cp = call_cache_path("m", SCAFFOLD_NAME, NEUTRAL_ARM, "i1", 0, "r0",
                         "role", 100)
    check("cache path is namespaced by scaffold, distinct from the AITA "
          "deliberation runner's own scaffold name",
          SCAFFOLD_NAME in str(cp))

    # ---- end-to-end protocol against a stub generator: no network ----------
    import scripts.run_crowdgold_dilemma as mod
    import scripts.run_crowdgold_deliberation as rcd
    import scripts.run_phase1_quartet as rpq

    class _Stub:
        def __init__(self, text, finish_reason="stop", ptok=50, ctok=50):
            self.text = text
            self.finish_reason = finish_reason
            self.prompt_tokens = ptok
            self.completion_tokens = ctok

    def stub_generate(model, system, user, *, sample_idx=0, max_tokens=0,
                      **kw):
        if "This is the final round and your vote is binary" in user:
            return _Stub("I accept.\n\nVERDICT: ACCEPT")
        if "Respond to this synthesis" in user:
            return _Stub("I accept as written.\n\nVERDICT: ACCEPT")
        if "Write one synthesis position" in user:
            return _Stub("SYNTHESIS: a balanced reading.\n\n"
                         "VERDICT: ACTION_A")
        if "Construct one integrated proposal" in user:
            return _Stub(f"PROPOSAL: refined reading.\n\n{ADDRESSED_MARKER} "
                        "none\n\nVERDICT: ACTION_A")
        return _Stub("My position is that A's action is worse.\n\n"
                     "VERDICT: ACTION_A")

    mod.generate_any = stub_generate
    rcd.generate_any = stub_generate

    tmp_out = Path("/tmp/_dilemma_selftest_cache")
    tmp_out.mkdir(exist_ok=True)
    old_out = rpq.OUT_DIR
    rpq.OUT_DIR = tmp_out
    try:
        row, votes, calls = run_deliberation("stub-model", fake, 0,
                                            transcript_cap=0)
        check("smoke: 17 calls made", len(calls) == CALLS_PER_CELL)
        check("smoke: row has a committal verdict", row["verdict"] == ACTION_A)
        check("smoke: 3 vote rows", len(votes) == N_AGENTS)
        check("smoke: unanimous accept vote", row["unanimous_accept"] == 1)
        check("smoke: A-advocate's stake IS undermined (verdict=ACTION_A)",
              votes[0]["stake_undermined"] == 1)
        check("smoke: B-advocate's stake is NOT undermined",
              votes[1]["stake_undermined"] == 0)
        check("smoke: neutral's stake_undermined reads 0, not blank",
              votes[2]["stake_undermined"] == 0)
        rows_written = {k: row[k] for k in ROW_FIELDS}
        check("smoke: every ROW_FIELDS key is present on the row",
              len(rows_written) == len(ROW_FIELDS))
        votes_written = {k: votes[0][k] for k in VOTE_FIELDS}
        check("smoke: every VOTE_FIELDS key is present on a vote row",
              len(votes_written) == len(VOTE_FIELDS))
        grep = guard_rows(calls)
        check("smoke: guard_rows keeps only the 4 verdict-bearing rounds per "
              "role (2 mod + 3*2 agent = 8)", len(grep) == 8)
        rep = truncation_report(grep)
        check("smoke: guard passes on clean stub output", rep["pass"])
    finally:
        rpq.OUT_DIR = old_out

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# Run loop / CLI
# ---------------------------------------------------------------------------

def run(model: str, items: Sequence[DilemmaItem], *, samples: int,
       moderator_model: Optional[str], max_tokens_agent: int,
       max_tokens_moderator: int, max_tokens_label: int, max_tokens_vote: int,
       allow_unresolved: bool, transcript_cap: int, workers: int):
    tasks = [(it, i) for it in items for i in range(samples)]
    print(f"  {len(tasks)} cells x {CALLS_PER_CELL} calls = "
          f"{len(tasks) * CALLS_PER_CELL} generation calls")
    rows, votes, calls = [], [], []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(
                run_deliberation, model, it, i,
                moderator_model=moderator_model,
                max_tokens_agent=max_tokens_agent,
                max_tokens_moderator=max_tokens_moderator,
                max_tokens_label=max_tokens_label,
                max_tokens_vote=max_tokens_vote,
                allow_unresolved=allow_unresolved,
                transcript_cap=transcript_cap,
            ): (it.item_id, i)
            for (it, i) in tasks
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
                    print(f"    {done}/{len(tasks)} cells", flush=True)
            except Exception as e:
                print(f"  ERROR {key}: {e}", flush=True)
    return rows, votes, calls


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Addendum 12 Stage 1: deliberation protocol on Scruples "
                    "Dilemmas")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default="grok-4-1-fast-reasoning")
    ap.add_argument("--moderator-model", default=None)
    ap.add_argument("--n-items", type=int, default=80,
                    help="Total items, gold-balanced (half ACTION_A, half "
                         "ACTION_B, deterministic prefix)")
    ap.add_argument("--min-annotators", type=int, default=4)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--max-tokens-agent", type=int, default=2560)
    ap.add_argument("--max-tokens-moderator", type=int, default=1024)
    ap.add_argument("--max-tokens-label", type=int, default=3072,
                    help="Corrected from the start; see module docstring.")
    ap.add_argument("--max-tokens-vote", type=int, default=3072,
                    help="Corrected from the start; see module docstring.")
    ap.add_argument("--transcript-cap", type=int, default=0)
    ap.add_argument("--no-unresolved", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="cg_dilemma")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    try:
        items = load_items(n=None, min_annotators=args.min_annotators,
                           seed=args.seed)
    except ScruplesDataError as e:
        print(f"\nERROR: {e}\n")
        return 2
    panel = len(items)
    n_a, n_b = args.n_items // 2, args.n_items - args.n_items // 2
    items = subset_items(items, n_a, n_b)
    print(f"panel {panel} fully-joined pairs -> using {len(items)} "
          f"({sum(1 for i in items if i.gold_verdict == ACTION_A)} gold-A "
          f"+ {sum(1 for i in items if i.gold_verdict == ACTION_B)} gold-B)")
    print(f"model {args.model}  moderator "
          f"{args.moderator_model or '(same as agent)'}  arm {NEUTRAL_ARM} "
          f"(single arm; Stage 1 measures grip only, see module docstring)")
    print(f"protocol: {PROTOCOL}")

    if args.dry_run:
        cm = cost_model(args.model, items, args.samples,
                        transcript_cap=args.transcript_cap,
                        moderator_model=args.moderator_model)
        print_cost_model(cm)
        return 0

    allow_unresolved = not args.no_unresolved
    rows, votes, calls = run(
        args.model, items, samples=args.samples,
        moderator_model=args.moderator_model,
        max_tokens_agent=args.max_tokens_agent,
        max_tokens_moderator=args.max_tokens_moderator,
        max_tokens_label=args.max_tokens_label,
        max_tokens_vote=args.max_tokens_vote,
        allow_unresolved=allow_unresolved, transcript_cap=args.transcript_cap,
        workers=args.workers,
    )
    if not rows:
        print("\nNo rows produced.\n")
        return 1

    rows_path = OUT_DIR / f"{args.tag}_rows.csv"
    votes_path = OUT_DIR / f"{args.tag}_votes.csv"
    with open(rows_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROW_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ROW_FIELDS})
    with open(votes_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(VOTE_FIELDS))
        w.writeheader()
        for v in votes:
            w.writerow({k: v.get(k, "") for k in VOTE_FIELDS})
    print(f"\nwrote {len(rows)} rows -> {rows_path}")
    print(f"wrote {len(votes)} votes -> {votes_path}")

    rep = truncation_report(guard_rows(calls))
    print_truncation_report(rep)

    pt = sum(int(c.get("prompt_tokens", 0) or 0) for c in calls)
    ct = sum(int(c.get("completion_tokens", 0) or 0) for c in calls)
    pin, pout = PRICES.get(args.model, (0.0, 0.0))
    print(f"\nmeasured spend: {pt} prompt + {ct} completion tokens "
          f"(mixed agent/moderator pricing not split here) ~= "
          f"${(pt / 1e6 * pin + ct / 1e6 * pout):.2f} at agent-model rates")

    if not rep["pass"]:
        print("\n*** GUARD FAILED. Do not read G1/G2/G3 off this run; raise "
              "the failing round's token cap and re-run. ***")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
