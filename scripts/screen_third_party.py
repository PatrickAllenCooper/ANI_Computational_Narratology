"""
scripts/screen_third_party.py -- Addendum 16.21 (n perspectives, the k-loser
generalisation): the THIRD-PARTY SCREEN on the crowd-gold AITA panel.

WHY A SCREEN
------------
The k-loser theory (16.12) says the dissent sensor is a deviation detector on
an anti-correlated advocate edge that exists only when a verdict has exactly
ONE loser. With two advocates the only observable values of k are 1 (YTA/NTA)
and "both" (ESH/NAH). Seating a THIRD advocate -- for a person the account
names who is affected by the verdict but is not the party in conflict -- makes
k=2 reachable on a one-loser verdict, which is the cell the generalisation
needs. That seat only makes sense on accounts that actually name such a
person, so the panel has to be screened for them first.

TWO STAGES, ONE FILE
--------------------
1. ZERO-SPEND heuristic screen (the default invocation). For every item of the
   249-item registered panel (load_items n_yta=99 n_nta=150, seed 44) it reads
   the CACHED grok single-agent narrative_cot outputs
   (cg_gen_grok-4-1-fast-reasoning_narrative_cot_t2560_<arm>_<item>_<k>.json,
   up to 6 per item: third_person x3 and as_asker x3; the as_asker_stance
   arm is EXCLUDED, because its outputs were generated under the
   stance-pressure wrapper and nothing generated under that manipulation
   may feed a prompt of the person-step cell) and parses each one's
   "Section 2 - Stakeholders" list: the narration scaffold asks the model to
   "list every person whose life intersects this decision", which is exactly
   the inventory a third-party screen needs, and it is already paid for.
   Heuristics, all documented in HEURISTICS below and recorded per item:
     * the WRITER is the "Me / Myself / the writer" entry;
     * the COUNTERPARTY is the non-writer person whose role noun (or name) the
       account text mentions most, corroborated where present by the cached
       counterparty seat's r0 opening ("I am the brother ...", also zero
       spend, from the registered cg_deliberation run);
     * a THIRD PARTY is any remaining entry that names a person BY ROLE
       (sibling, friend, boss, child, roommate, ...) or by name, whose role
       noun/name the account text itself mentions (so it is not inferred by
       the model), listed by at least --min-support of the cached outputs;
       ranked by (support, mentions, stake rank, key), where the stake rank
       breaks ties toward a counterparty-attached person over the writer's
       own partner over no signal (the writer's partner is the 16.19 ally in
       different clothes and the least NEW perspective);
     * the third party's STAKE -- which finding about the writer, YTA or NTA,
       serves them -- is a coarse alignment rule (possessive attachment to the
       counterparty -> YTA; the writer's own partner -> NTA; otherwise none)
       and is marked heuristic. The LLM screen is what confirms it;
     * the DESCRIPTION the runner quotes is the modal cleaned head over the
       cached outputs (parentheticals, dash / verb tails and "indirectly"
       qualifiers removed; "my X" -> "the writer's X"; his/her/their kept
       for the advocate to resolve against the account; a proper name reads
       "Name, the writer's <role>" when the account states the role);
     * the INTEREST text the runner quotes verbatim into the third-party
       seat's prompt (both arms, r0-r4) is taken from THIRD_PERSON-arm
       outputs ONLY: the longest Section-2 body for the chosen person among
       those outputs, after the possessive rewrite "my/our X" -> "the
       writer's X" (rewrite_interest). An as_asker body is used only when no
       third_person output lists the person AND the rewritten as_asker body
       carries no first-person pronoun (I / me / my / mine / myself / we /
       our / us); otherwise the interest reads "(not stated)". Before this
       rule the body was the longest over every arm, so first-person writer
       voice from the as_asker arm ("... if I push too hard") reached the
       third_person arm's seat prompt on 63 of 158 items. The source arm,
       output index and whether the rewrite fired are recorded per item as
       third_party_interest_source, so the provenance of every quoted
       interest is in the screen JSON.
   Output: divergence_study_outputs/third_party_screen.json -- the candidate
   list with the extracted third-party description, interest text (and its
   source), stake and every intermediate the heuristic used, plus the
   non-candidates and why. NOTE the runner keys the third-party seat's
   parent_sha on (description | interest | stake), so re-screening under a
   changed rule invalidates that seat's cache by design; the LLM screen, if
   it is ever run, must be run BEFORE the cell for the same reason.

2. LLM screen, --dry-run only in this build: one call per PANEL item (all
   249, not only the candidates, so a third party the heuristic missed can
   still be found) to a cheap model (default gpt-5.4-nano; --model and
   --price keep it config-driven) that names the counterparty, names ONE
   third party or NONE, states their interest in one sentence -- referring
   to the writer as "the writer", never I / me / my, so the sentence can be
   quoted into a third-person seat prompt -- and says which finding favours
   them. --dry-run builds every prompt, prices them from
   run_crowdgold_deliberation.PRICES, writes the prompts to
   third_party_screen_prompts.json and CALLS NOTHING. The --run path exists so
   the confirmation can be authorised later; it refuses without the literal
   flag, refuses an unpriced model, keeps its own cache namespace
   (cg_tpscreen_<model>_<item>.json, verified EMPTY by the selftest with the
   no-foreign-file idiom) and enforces cache_is_clean() with --resume as the
   only override. Its parsed results are merged into the same screen JSON
   under "llm", and the runner (run_crowdgold_nperspective.py) prefers them
   over the heuristic when present.

CACHE
-----
Reads only. Nothing in stage 1 writes a cache file. The single-agent cache it
reads keys on (model, scaffold, cap, arm, item, sample) and is never modified.
The deliberation r0 hint reads cgd_*_r0_counterparty.json from the registered
run and never writes.

Usage
-----
  python -m scripts.screen_third_party --selftest          # offline, no API
  python -m scripts.screen_third_party                     # heuristic screen, $0
  python -m scripts.screen_third_party --dry-run           # + LLM screen cost, prompts written, no call
  python -m scripts.screen_third_party --run               # SPENDS (ceiling $0.30), not in this build's remit

Exit codes: 0 ok, 2 bad config / --run not given / unpriced model,
3 cache not clean (pass --resume), 4 GUARD FAILED.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.run_crowdgold_aita import (
    CrowdGoldItem, _optional_bpe_counter, count_tokens, load_items,
)
from scripts.run_crowdgold_deliberation import PRICES, marker_line
from scripts.run_phase1_quartet import OUT_DIR, _safe
from scripts.load_scruples import sha256_text

REGISTRATION = "prereg_embodiment_community.md Addendum 16.21 (n perspectives)"
SCREEN_PATH = OUT_DIR / "third_party_screen.json"
PROMPTS_PATH = OUT_DIR / "third_party_screen_prompts.json"
SCREEN_MODEL = "grok-4-1-fast-reasoning"          # whose cache is read
SCREEN_CAP = 2560
SINGLE_AGENT_PREFIX = f"cg_gen_{_safe(SCREEN_MODEL)}_narrative_cot_t{SCREEN_CAP}_"
DELIB_R0_SUFFIX = "_00_r0_counterparty.json"
DELIB_R0_PREFIX = f"cgd_{_safe(SCREEN_MODEL)}_narrative_cot_t{SCREEN_CAP}_"

#: registered panel loader arguments (run_crowdgold_deliberation.main)
PANEL_KW = dict(source="scruples", n_yta=99, n_nta=150, min_votes=50,
                min_consensus=0.90, seed=44)

#: LLM screen defaults -- config-driven, never hard-wired into the prompt
DEFAULT_LLM_MODEL = "gpt-5.4-nano"
LLM_MAX_TOKENS = 400
ASSUMED_LLM_COMPLETION = 120
LLM_CEILING_USD = 0.30                            # the registration's ceiling for the screen
TPSCREEN_PREFIX = "cg_tpscreen_"
MAX_PARSE_FAIL_SHARE = 0.05

# ---------------------------------------------------------------------------
# Heuristics -- every rule the screen applies, in one place, recorded in the
# output so a reader can audit the candidate list without re-running it.
# ---------------------------------------------------------------------------

HEURISTICS = {
    "source": "Section 2 (Stakeholders) of every cached grok narrative_cot "
              "output for the item, up to 6 (third_person x3, as_asker x3); "
              "as_asker_stance outputs EXCLUDED (generated under the "
              "stance-pressure wrapper); content-filtered (empty) outputs skipped",
    "interest": "the longest Section-2 body for the chosen person among the "
                "THIRD_PERSON outputs only, after rewrite_interest (my/our X -> "
                "the writer's X); an as_asker body only when no third_person "
                "output lists the person and the rewritten body carries no "
                "first-person pronoun; else '(not stated)'. Source arm, output "
                "index and rewrite flag recorded as interest_source",
    "writer": "entry whose head is Me / Myself / I / OP / the writer / the "
              "poster / the author (or carries one of these in parentheses)",
    "counterparty": "among non-writer PERSON entries (role noun or proper "
                    "name), the one whose role noun / name the ACCOUNT TEXT "
                    "mentions most; corroborated by the cached counterparty "
                    "seat's r0 opening from the registered cg_deliberation run "
                    "when it names a role ('I am the brother'); a hint that "
                    "matches a candidate overrides the mention count",
    "third_party": "any remaining PERSON entry (role noun from ROLE_LEXICON, "
                   "or a proper name) that is not generic (extended family, "
                   "society, onlookers, future selves, pets, institutions), "
                   "whose role noun / name occurs in the account text, listed "
                   "by >= --min-support cached outputs; ranked by (support, "
                   "mentions in account, stake rank, key). STAKE RANK breaks "
                   "ties: a counterparty-attached person (2) over the writer's "
                   "own partner (1) over no alignment signal (0). The k-loser "
                   "readout needs a third party who can lose, and the writer's "
                   "partner is the 16.19 coalition ally in different clothes "
                   "(same household, same stake as the writer), i.e. the least "
                   "NEW perspective on offer",
    "stake": "possessive attachment to the counterparty (his/her/their X, "
             "<counterparty>'s X) -> YTA (a finding against the writer serves "
             "the counterparty's side); the writer's own partner/spouse -> NTA "
             "(household-aligned); everything else -> none. COARSE; marked "
             "stake_source=heuristic; the LLM screen confirms or overrides",
}

#: canonical role -> surface variants (lowercase, singular and plural, common
#: synonyms). Multi-word entries must precede single-word ones in the regex,
#: which _role_regex() guarantees by sorting on length.
ROLE_LEXICON: dict[str, tuple[str, ...]] = {
    "mother": ("mother", "mom", "mum", "mommy", "mama", "stepmother", "stepmom",
               "step-mother", "step-mom", "mother-in-law", "mil"),
    "father": ("father", "dad", "daddy", "papa", "stepfather", "stepdad",
               "step-father", "step-dad", "father-in-law", "fil"),
    "parent": ("parent", "parents", "step-parents", "stepparents", "in-laws",
               "in laws", "inlaws"),
    "sister": ("sister", "sisters", "half-sister", "half-sisters", "half sister",
               "half sisters", "stepsister", "stepsisters", "step-sister",
               "sister-in-law", "sis"),
    "brother": ("brother", "brothers", "half-brother", "half-brothers",
                "half brother", "stepbrother", "step-brother", "brother-in-law",
                "bro"),
    "sibling": ("sibling", "siblings", "half-siblings", "half siblings"),
    "child": ("child", "children", "kid", "kids", "son", "sons", "daughter",
              "daughters", "stepson", "stepdaughter", "step-son",
              "step-daughter", "stepchild", "stepchildren", "step-children",
              "baby", "toddler", "teen", "teenager", "little girl", "little boy"),
    "spouse": ("wife", "husband", "spouse", "partner"),
    "fiance": ("fiance", "fiancee", "fiancé", "fiancée"),
    "girlfriend": ("girlfriend", "gf"),
    "boyfriend": ("boyfriend", "bf"),
    "ex": ("ex", "ex-wife", "ex-husband", "ex-girlfriend", "ex-boyfriend",
           "ex-partner", "ex-fiancé", "ex-fiancée", "ex-fiance", "ex wife",
           "ex husband", "ex girlfriend", "ex boyfriend"),
    "friend": ("friend", "friends", "best friend", "best friends", "close friend",
               "childhood friend", "buddy", "mate"),
    "roommate": ("roommate", "roommates", "flatmate", "flatmates", "housemate",
                 "housemates", "room-mate"),
    "neighbour": ("neighbour", "neighbor", "neighbours", "neighbors"),
    "boss": ("boss", "manager", "supervisor", "employer"),
    "coworker": ("coworker", "co-worker", "colleague"),
    "employee": ("employee", "assistant", "intern", "staff member"),
    "landlord": ("landlord", "landlady"),
    "tenant": ("tenant", "lodger", "subletter"),
    "teacher": ("teacher", "professor", "tutor", "coach", "instructor"),
    "student": ("student", "classmate", "pupil"),
    "cousin": ("cousin", "cousins"),
    "aunt": ("aunt", "auntie"),
    "uncle": ("uncle",),
    "niece": ("niece", "nieces"),
    "nephew": ("nephew", "nephews"),
    "grandparent": ("grandmother", "grandma", "gran", "grams", "nana", "nan",
                    "grandfather", "grandpa", "gramps", "grandparent",
                    "grandparents"),
    "grandchild": ("grandchild", "grandchildren", "granddaughter", "grandson"),
    "doctor": ("doctor", "nurse", "therapist", "physician", "surgeon", "vet"),
    "client": ("client", "customer", "patient"),
    "guest": ("guest", "host", "hostess"),
    "wedding_party": ("bridesmaid", "maid of honor", "maid of honour",
                      "best man", "groomsman", "bride", "groom"),
    "date": ("date", "crush", "hookup"),
    "godparent": ("godmother", "godfather", "godparent", "goddaughter", "godson"),
    "babysitter": ("babysitter", "nanny", "sitter", "caregiver", "carer"),
    "stranger": ("stranger", "passenger", "driver", "cashier", "waiter",
                 "waitress", "server", "bartender", "clerk", "receptionist"),
}

#: heads that name no PERSON -- excluded from both counterparty and third party
GENERIC_PATTERNS = (
    r"\bextended\b", r"\brelatives?\b", r"\bsociety\b", r"\bcommunity\b",
    r"\bonlookers?\b", r"\breaders?\b", r"\bfuture\b", r"\beveryone\b",
    r"\bothers\b", r"\bpublic\b", r"\bcircle\b", r"\bgroup\b", r"\bgeneral\b",
    r"\bwider\b", r"\bbroader\b", r"\bmutual\b", r"\bbystanders?\b",
    r"\bworkplace\b", r"\bcompany\b", r"\bteam\b", r"\bschool\b",
    r"\brelationship\b", r"\bmarriage\b", r"\bcouple\b", r"\bhousehold\b",
    r"\bfamily\s*$", r"^family\b", r"\bfamily unit\b", r"\bfamily dynamics?\b",
    r"\bdog\b", r"\bcat\b", r"\bpets?\b", r"\bsnakes?\b", r"\breptiles?\b",
    r"\bnobody\b", r"\bno one\b", r"\bno direct\b", r"\bnone\b",
    r"\bthe (other )?(side|party) as a whole\b", r"\bindirect\b",
    r"\bboth of us\b", r"\bus both\b", r"\bboth parties\b",
)

WRITER_HEAD_RE = re.compile(
    r"^(me|myself|i|op|the writer|writer|the poster|the author|the narrator|"
    r"the op|the protagonist|protagonist)\b|"
    r"\((me|myself|the writer|writer|op|narrator|the poster|the author)\)|"
    r"\bthe writer\b|\bmyself\b",
    re.I,
)

_MD_STRIP = " \t*_`•-"
_HEAD_SPLIT_RE = re.compile(
    r":\s|\s*[–—]\s*|\s-\s|,\s*whose\b|,\s*who\b|\swhose\b|"
    r"\s(?:is|was|are|were|has|have|had|feels|feel|risks?|would|will|might|may|could)\b")
_PAREN_RE = re.compile(r"\([^)]*\)")
_NAME_RE = re.compile(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)?$")
_PET_WORDS = r"(?:dog|cat|puppy|kitten|hedgehog|rabbit|bunny|hamster|snake|reptile|lizard|bird|parrot|horse|pet)"
_PET_PAREN_RE = re.compile(rf"\((?:the|my|our|his|her|their|a)\s+{_PET_WORDS}\)", re.I)
_NAME_STOP = {"Me", "Myself", "I", "Op", "The", "My", "Our", "His", "Her",
              "Their", "Section", "Stakeholders", "Everyone", "Others",
              "Nobody", "None", "Both", "Family", "Society", "Community"}


def _role_regex() -> re.Pattern:
    variants = []
    for canon, vs in ROLE_LEXICON.items():
        for v in vs:
            variants.append((v, canon))
    variants.sort(key=lambda t: -len(t[0]))
    alt = "|".join(re.escape(v) for v, _ in variants)
    return re.compile(rf"(?<![A-Za-z-])({alt})(?![A-Za-z-])", re.I)


ROLE_RE = _role_regex()
_VARIANT_TO_CANON = {v.lower(): c for c, vs in ROLE_LEXICON.items() for v in vs}


def canon_role(surface: str) -> str:
    return _VARIANT_TO_CANON.get(surface.lower(), surface.lower())


def is_generic(head: str) -> bool:
    low = head.lower().strip()
    return any(re.search(p, low) for p in GENERIC_PATTERNS)


def is_writer_head(head: str) -> bool:
    return bool(WRITER_HEAD_RE.search(head.strip()))


def owner_class(head: str, counterparty_key: Optional[str]) -> str:
    """writer / counterparty / other: whom the possessive in the head attaches
    the person to. 'my X' / 'our X' -> writer; 'his/her/their X' or
    '<counterparty noun>'s X' -> counterparty; else other."""
    low = head.lower().strip()
    low = _PAREN_RE.sub("", low).strip()
    if re.match(r"^(my|our)\b", low):
        return "writer"
    if re.match(r"^(his|her|their)\b", low):
        return "counterparty"
    if counterparty_key:
        cp = counterparty_key.split(":", 1)[1] if ":" in counterparty_key else counterparty_key
        surfaces = ROLE_LEXICON.get(cp, (cp,))
        for s in surfaces:
            if re.search(rf"\b{re.escape(s)}'?s\b", low):
                return "counterparty"
        if counterparty_key.startswith("name:") and re.search(
                rf"\b{re.escape(cp.lower())}'?s\b", low):
            return "counterparty"
    if re.match(r"^the (writer|poster|op|author)'s\b", low):
        return "writer"
    return "other"


THIRD_PERSON_ARM = "third_person"
AS_ASKER_ARM = "as_asker"
EXCLUDED_ARMS = ("as_asker_stance",)      # stance-pressure wrapper: never a screen source
SCREEN_ARMS = (THIRD_PERSON_ARM, AS_ASKER_ARM)
_FIRST_PERSON_RE = re.compile(r"\b(?:I|me|my|mine|myself|we|our|ours|us)\b")


@dataclass
class Entry:
    head: str
    body: str
    output_idx: int
    kind: str = ""            # writer / person / generic / unknown
    role: str = ""            # canonical role, or "" for a name
    name: str = ""            # proper name, if the head is one
    key: str = ""             # "role:<canon>" or "name:<Name>"
    arm: str = THIRD_PERSON_ARM   # which arm's output the entry was parsed from


def rewrite_interest(text: str) -> str:
    """'my/our X' -> \"the writer's X\" (and 'the poster's/OP's' -> the writer's)
    inside an interest sentence; the clean_head convention applied to prose.
    Pronoun subjects (I / me / we) are NOT rewritten -- a regex cannot fix
    verb agreement -- which is why first_person_free() gates the fallback."""
    t = re.sub(r"\b(?:my|our)\s+", "the writer's ", text, flags=re.I)
    t = re.sub(r"\bthe (?:poster|op|author)'s\s+", "the writer's ", t, flags=re.I)
    return t


def first_person_free(text: str) -> bool:
    return _FIRST_PERSON_RE.search(text) is None


def choose_interest(entries: Sequence["Entry"]) -> tuple[str, dict]:
    """The interest text for one person from its Section-2 entries, and its
    provenance. third_person bodies first (longest, after the rewrite); an
    as_asker body only if no third_person body exists and the rewritten body
    is free of first-person pronouns; else '(not stated)'."""
    tp = sorted((e for e in entries if e.arm == THIRD_PERSON_ARM and e.body.strip()),
                key=lambda e: -len(e.body))
    aa = sorted((e for e in entries if e.arm == AS_ASKER_ARM and e.body.strip()),
                key=lambda e: -len(e.body))
    for e in tp:
        body = rewrite_interest(e.body.strip().rstrip("."))
        if first_person_free(body):
            return body, {"arm": e.arm, "output_idx": e.output_idx,
                          "rewritten": body != e.body.strip().rstrip("."),
                          "n_third_person_bodies": len(tp), "n_as_asker_bodies": len(aa),
                          "rule": "third_person body (longest, pronoun-free after rewrite)"}
    for e in aa:
        body = rewrite_interest(e.body.strip().rstrip("."))
        if first_person_free(body):
            return body, {"arm": e.arm, "output_idx": e.output_idx,
                          "rewritten": body != e.body.strip().rstrip("."),
                          "n_third_person_bodies": len(tp), "n_as_asker_bodies": len(aa),
                          "rule": "no third_person body; as_asker body pronoun-free after rewrite"}
    return "(not stated)", {"arm": None, "output_idx": None, "rewritten": False,
                            "n_third_person_bodies": len(tp), "n_as_asker_bodies": len(aa),
                            "rule": ("no pronoun-free body in any third_person output"
                                     if tp else "no third_person body; as_asker bodies carry "
                                                "first-person voice or are empty")}


def _cut_head_at_comma(head: str, rest: str) -> tuple[str, str]:
    """'my fiancé, potentially drawn into family tension' -> head 'my fiancé',
    the tail joins the body so the interest text is kept. A comma inside
    parentheses ('Me (the sister, 28)') is left alone."""
    depth = 0
    for i, ch in enumerate(head):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            before, after = head[:i].strip(), head[i + 1:].strip()
            if before:
                return before, (after + (" " + rest if rest else "")).strip()
            break
    return head, rest


def section2_entries(output: str, output_idx: int, arm: str = THIRD_PERSON_ARM) -> list[Entry]:
    """Parse the Section 2 stakeholder list: bullet form ('- Head: body') and
    the inline form ('Section 2 - Stakeholders: Me (the sister), whose ...;
    my brother, whose ...'). Returns [] when the section is absent."""
    m = re.search(r"section\s*2[^\n]*", output or "", re.I)
    if not m:
        return []
    header = m.group(0)
    start = m.end()
    end_m = re.search(r"\n\s*(?:#+\s*)?(?:\*\*)?section\s*3", output[start:], re.I)
    body = output[start:start + end_m.start()] if end_m else output[start:start + 3000]
    entries: list[Entry] = []
    # inline form: everything after the first ':' on the header line
    inline = ""
    hm = re.search(r"stakeholders?\*{0,2}\s*:\s*(.+)$", header, re.I)
    if hm and hm.group(1).strip():
        inline = hm.group(1).strip()
        # the inline list may continue on the next lines until a blank line
        cont = body.split("\n\n", 1)[0]
        if cont.strip() and not cont.strip().startswith(("-", "*", "•")):
            inline = inline + " " + " ".join(cont.split())
    if inline:
        for chunk in re.split(r";\s*", inline):
            chunk = chunk.strip().strip(".")
            if not chunk:
                continue
            parts = _HEAD_SPLIT_RE.split(chunk, maxsplit=1)
            head = parts[0].strip().strip(_MD_STRIP)
            rest = parts[1].strip() if len(parts) > 1 else ""
            head, rest = _cut_head_at_comma(head, rest)
            if head:
                entries.append(Entry(head=head, body=rest, output_idx=output_idx, arm=arm))
        return entries
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not re.match(r"^(?:[-*•]|\d+[.)])\s+", line):
            continue
        line = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line).strip()
        line = line.replace("**", "")
        parts = _HEAD_SPLIT_RE.split(line, maxsplit=1)
        head = parts[0].strip().strip(_MD_STRIP)
        rest = parts[1].strip() if len(parts) > 1 else ""
        head, rest = _cut_head_at_comma(head, rest)
        if not head or len(head) > 80:
            continue
        entries.append(Entry(head=head, body=rest, output_idx=output_idx, arm=arm))
    return entries


def classify(entry: Entry) -> Entry:
    """Fill kind / role / name / key on an Entry."""
    head = entry.head
    if is_writer_head(head):
        entry.kind = "writer"
        entry.key = "writer"
        return entry
    if is_generic(head):
        entry.kind = "generic"
        return entry
    stripped = _PAREN_RE.sub("", head).strip()
    roles = ROLE_RE.findall(stripped)
    if roles:
        # the head noun is the RIGHTMOST role word ("brother's girlfriend" ->
        # girlfriend); earlier ones are possessive owners
        entry.role = canon_role(roles[-1])
        entry.kind = "person"
        entry.key = f"role:{entry.role}"
        return entry
    # a proper name?
    cand = re.sub(r"^(the|my|our|his|her|their)\s+", "", stripped, flags=re.I).strip()
    if _PET_PAREN_RE.search(head):
        entry.kind = "generic"
        return entry
    cand = cand.rstrip("'s").strip()
    if _NAME_RE.match(cand) and cand.split()[0] not in _NAME_STOP:
        entry.name = cand
        entry.kind = "person"
        entry.key = f"name:{cand}"
        return entry
    entry.kind = "unknown"
    return entry


def mentions(text: str, key: str) -> int:
    """How often the account text mentions this role (any surface variant,
    with plural/possessive tails) or this name."""
    if key.startswith("name:"):
        nm = key[5:]
        return len(re.findall(rf"\b{re.escape(nm)}\b", text))
    canon = key[5:] if key.startswith("role:") else key
    surfaces = ROLE_LEXICON.get(canon, (canon,))
    n = 0
    for s in surfaces:
        n += len(re.findall(rf"(?<![A-Za-z-]){re.escape(s)}(?:'s|s|es)?(?![A-Za-z-])",
                            text, re.I))
    return n


def role_for_name(text: str, name: str) -> str:
    """'my friend Jake' / 'Jake, my brother' / 'Jake (my roommate)' in the
    account -> the role; else ''."""
    pats = [
        rf"\b(?:my|our)\s+(?:\w+\s+){{0,2}}?([\w-]+(?:\s[\w-]+)?)[,\s]+{re.escape(name)}\b",
        rf"\b{re.escape(name)}\b[,\s(]+(?:who is |who's |is )?(?:my|our)\s+(?:\w+\s+){{0,2}}?([\w-]+(?:\s[\w-]+)?)",
        rf"\b{re.escape(name)}\b[^.]{{0,40}}?\b(?:my|our)\s+([\w-]+)",
    ]
    for p in pats:
        for m in re.finditer(p, text, re.I):
            for grp in m.groups():
                if not grp:
                    continue
                r = ROLE_RE.findall(grp)
                if r:
                    return canon_role(r[-1])
    return ""


def counterparty_r0_hint(item_id: str, index: dict[str, list[Path]]) -> Optional[str]:
    """Role noun the cached counterparty seat says it speaks for, from the
    first 400 chars of its r0 opening in either arm; the most common canonical
    role across the arms, or None."""
    found: Counter = Counter()
    for p in index.get(item_id, []):
        try:
            txt = json.loads(p.read_text()).get("output") or ""
        except (OSError, json.JSONDecodeError):
            continue
        head = txt[:400]
        m = re.search(r"\b(?:I am|I'm|As|Speaking as|I speak for|I represent|"
                      r"Speaking for)\b[^.\n]{0,80}", head, re.I)
        if not m:
            continue
        roles = ROLE_RE.findall(m.group(0))
        if roles:
            found[canon_role(roles[0])] += 1
    if not found:
        return None
    return found.most_common(1)[0][0]


@dataclass
class ScreenRecord:
    item_id: str
    gold_verdict: str
    n_outputs: int
    n_outputs_with_section2: int
    writer_heads: list = field(default_factory=list)
    counterparty: dict = field(default_factory=dict)
    third_party: Optional[dict] = None
    alternatives: list = field(default_factory=list)
    stake: dict = field(default_factory=dict)
    generic_heads: list = field(default_factory=list)
    unknown_heads: list = field(default_factory=list)
    candidate: bool = False
    reason: str = ""


def screen_item(item: CrowdGoldItem, outputs: Sequence[str], *,
                r0_hint: Optional[str], min_support: int,
                arms: Optional[Sequence[str]] = None) -> ScreenRecord:
    """`arms[i]` names the arm output i came from (third_person / as_asker);
    None treats every output as third_person (the selftest's synthetic
    outputs). as_asker_stance outputs must not be passed (index_caches never
    indexes them; asserted here)."""
    text = item.post_text
    arms = list(arms) if arms is not None else [THIRD_PERSON_ARM] * len(outputs)
    if len(arms) != len(outputs):
        raise ValueError("arms must be parallel to outputs")
    if any(a in EXCLUDED_ARMS for a in arms):
        raise ValueError(f"{EXCLUDED_ARMS} outputs are not a screen source")
    rec = ScreenRecord(item_id=item.item_id, gold_verdict=item.gold_verdict,
                       n_outputs=len(outputs), n_outputs_with_section2=0)
    all_entries: list[Entry] = []
    for i, (out, arm) in enumerate(zip(outputs, arms)):
        es = [classify(e) for e in section2_entries(out, i, arm)]
        if es:
            rec.n_outputs_with_section2 += 1
        all_entries.extend(es)
    rec.writer_heads = sorted({e.head for e in all_entries if e.kind == "writer"})
    rec.generic_heads = sorted({e.head for e in all_entries if e.kind == "generic"})
    rec.unknown_heads = sorted({e.head for e in all_entries if e.kind == "unknown"})
    persons = [e for e in all_entries if e.kind == "person"]
    if not persons:
        rec.reason = "no person entries parsed from any cached Section 2"
        return rec

    # ---- group person entries by key --------------------------------------
    groups: dict[str, list[Entry]] = defaultdict(list)
    for e in persons:
        groups[e.key].append(e)
    stats = {}
    for key, es in groups.items():
        support = len({e.output_idx for e in es})
        first_pos = min(e.output_idx for e in es)
        men = mentions(text, key)
        if key.startswith("name:") and men == 0:
            continue                       # the model invented or renamed
        if key.startswith("name:") and re.search(
                rf"\b(?:my|our|his|her|their)\s+{_PET_WORDS}[,\s]+(?:named\s+|called\s+)?{re.escape(key[5:])}\b",
                text, re.I):
            continue                       # a pet with a name is not a person
        stats[key] = {"support": support, "mentions": men, "first_output": first_pos,
                      "heads": sorted({e.head for e in es})}
    if not stats:
        rec.reason = "person entries present but none mentioned in the account text"
        return rec

    # ---- counterparty -----------------------------------------------------
    def cp_score(k):
        s = stats[k]
        hint_bonus = 1000 if (r0_hint and k == f"role:{r0_hint}") else 0
        return (hint_bonus, s["mentions"], s["support"], -s["first_output"])
    cp_key = max(stats, key=cp_score)
    rec.counterparty = {
        "key": cp_key, "heads": stats[cp_key]["heads"],
        "mentions_in_account": stats[cp_key]["mentions"],
        "support": stats[cp_key]["support"], "r0_hint": r0_hint,
        "r0_hint_agrees": (r0_hint is not None and cp_key == f"role:{r0_hint}"),
    }

    # ---- third party candidates -------------------------------------------
    cands = []
    for key, s in stats.items():
        if key == cp_key:
            continue
        if s["mentions"] < 1 or s["support"] < min_support:
            continue
        # a second 'role:' key that is the counterparty's synonym class is
        # already merged by canon_role; a name whose account role IS the
        # counterparty role is the counterparty, not a third party
        if key.startswith("name:"):
            r = role_for_name(text, key[5:])
            if r and f"role:{r}" == cp_key:
                continue
            s = dict(s, role_in_account=r)
        heads = groups[key]
        owners = Counter(owner_class(e.head, cp_key) for e in heads)
        interest, interest_src = choose_interest(heads)
        owner = owners.most_common(1)[0][0]
        role_ = key[5:] if key.startswith("role:") else s.get("role_in_account", "")
        cands.append({
            "key": key, "heads": s["heads"], "support": s["support"],
            "of_outputs": rec.n_outputs_with_section2,
            "support_by_arm": dict(Counter(e.arm for e in heads)),
            "mentions_in_account": s["mentions"],
            "owner_class": owner,
            "owner_classes": dict(owners),
            "stake_rank": _STAKE_RANK[_stake_signal(owner, role_)],
            "interest": interest,
            "interest_source": interest_src,
            "role_in_account": s.get("role_in_account", ""),
        })
    cands.sort(key=lambda c: (-c["support"], -c["mentions_in_account"],
                              -c["stake_rank"], c["key"]))
    if not cands:
        rec.reason = (f"no third party: every person entry other than the "
                      f"counterparty has support < {min_support} or is not "
                      f"mentioned in the account")
        return rec
    tp = cands[0]
    rec.third_party = tp
    rec.alternatives = cands[1:4]
    rec.candidate = True
    rec.reason = "candidate"

    # ---- stake heuristic --------------------------------------------------
    role = tp["key"][5:] if tp["key"].startswith("role:") else tp.get("role_in_account", "")
    stake = _stake_signal(tp["owner_class"], role)
    rec.stake = {"heuristic": stake, "basis": _STAKE_BASIS[stake],
                 "confidence": "low", "stake_source": "heuristic"}
    return rec


PARTNER_ROLES = ("spouse", "fiance", "girlfriend", "boyfriend")
#: tiebreak rank of the heuristic stake (HEURISTICS["third_party"])
_STAKE_RANK = {"YTA": 2, "NTA": 1, "none": 0}
_STAKE_BASIS = {
    "YTA": "possessive attachment to the counterparty (a finding against the "
           "writer serves that side)",
    "NTA": "the writer's own partner; household-aligned",
    "none": "no alignment signal in the head; the LLM screen must establish "
            "which finding serves this person",
}


def _stake_signal(owner: str, role: str) -> str:
    """The coarse alignment rule (HEURISTICS['stake']): which finding about
    the writer serves this person, or 'none' when the head carries no signal."""
    if owner == "counterparty":
        return "YTA"
    if owner == "writer" and role in PARTNER_ROLES:
        return "NTA"
    return "none"


_LEAD_JUNK_RE = re.compile(r"^(?:and\s+)?(?:peripherally|indirectly|also|for|plus)\s+", re.I)
_TAIL_JUNK_RE = re.compile(r"\s*\((?:indirect(?:ly)?|mentioned[^)]*|implied[^)]*|hypothetical[^)]*|"
                           r"minimal[^)]*|peripheral[^)]*|unseen|offscreen)\)\s*$", re.I)


def clean_head(head: str) -> str:
    """One stakeholder head -> the short noun phrase a brief can quote:
    parentheticals, dash tails, 'who/whose' clauses and 'indirectly'-type
    qualifiers removed; 'my/our X' -> 'the writer's X' (the block is read next
    to the account, so his/her/their are left for the advocate to resolve)."""
    h = _TAIL_JUNK_RE.sub("", head.strip())
    h = _PAREN_RE.sub("", h)
    h = re.split(r"\s*[–—]\s*|\s-\s|,\s*whose\b|,\s*who\b|\swhose\b|\swho\b|"
                 r"\s(?:is|was|are|were|has|have|had|feels|feel|risks?|would|will|might|may|could)\b",
                 h, maxsplit=1)[0]
    h = _LEAD_JUNK_RE.sub("", h.strip())
    h = h.strip(" .,;:\"'")
    h = re.sub(r"\s+", " ", h)
    if not h:
        return ""
    h = re.sub(r"^(?:my|our)\s+", "the writer's ", h, flags=re.I)
    h = re.sub(r"^the (?:writer|poster|op|author)'s\s+", "the writer's ", h, flags=re.I)
    # lower-case a leading role word / article; keep proper names as written
    first = re.sub(r"'s?$", "", h.split()[0])
    if first.lower() in ("the", "his", "her", "their", "a", "an") or ROLE_RE.fullmatch(first) \
            or first.lower() in _VARIANT_TO_CANON:
        h = h[0].lower() + h[1:]
    return h


def describe_third_party(tp: dict) -> str:
    """Short description used by the runner's THIRD PARTY block: the MODAL
    cleaned head over the cached outputs (ties -> shortest), so a parenthetical
    or a verb tail in one output cannot become the description. A proper-name
    key reads 'Name, the writer's <role>' when the account states the role."""
    key = tp.get("key", "")
    if key.startswith("name:"):
        nm = key[5:]
        role = tp.get("role_in_account") or ""
        return f"{nm}, the writer's {role}" if role else nm
    cleaned = [c for c in (clean_head(h) for h in (tp.get("heads") or [])) if c]
    if not cleaned:
        return key.split(":", 1)[-1]
    counts = Counter(c.lower() for c in cleaned)
    best = min(counts, key=lambda c: (-counts[c], len(c), c))
    # return the first original-cased spelling of the winner
    return next(c for c in cleaned if c.lower() == best)


# ---------------------------------------------------------------------------
# Index of the cached single-agent outputs and the deliberation r0 hints.
# ONE scandir pass: the output directory holds ~560k files and per-item globs
# would take minutes.
# ---------------------------------------------------------------------------

def index_caches(out_dir: Path = OUT_DIR) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    single: dict[str, list[Path]] = defaultdict(list)
    delib: dict[str, list[Path]] = defaultdict(list)
    if not out_dir.is_dir():
        return single, delib
    with os.scandir(out_dir) as it:
        for e in it:
            n = e.name
            if n.startswith(SINGLE_AGENT_PREFIX) and n.endswith(".json"):
                # ..._<arm>_<item32>_<kk>.json ; the item id is the token before _kk
                stem = n[len(SINGLE_AGENT_PREFIX):-5]
                parts = stem.rsplit("_", 2)
                if len(parts) == 3 and parts[2].isdigit() and parts[0] in SCREEN_ARMS:
                    single[parts[1]].append(Path(e.path))
            elif n.startswith(DELIB_R0_PREFIX) and n.endswith(DELIB_R0_SUFFIX):
                stem = n[len(DELIB_R0_PREFIX):-len(DELIB_R0_SUFFIX)]
                parts = stem.rsplit("_", 1)
                if len(parts) == 2:
                    delib[parts[1]].append(Path(e.path))
    return single, delib


def arm_of_path(p: Path) -> str:
    """The arm segment of a cached single-agent path (…_t2560_<arm>_<item>_<kk>.json)."""
    stem = p.name[len(SINGLE_AGENT_PREFIX):-5]
    return stem.rsplit("_", 2)[0]


def load_outputs(paths: Sequence[Path]) -> list[tuple[str, str]]:
    """(arm, output text) per readable, un-refused, non-empty record, in path
    order; as_asker_stance records are never returned."""
    outs = []
    for p in sorted(paths):
        arm = arm_of_path(p)
        if arm in EXCLUDED_ARMS:
            continue
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if r.get("finish_reason") == "content_filter":
            continue
        t = r.get("output") or ""
        if t.strip():
            outs.append((arm, t))
    return outs


# ---------------------------------------------------------------------------
# The LLM screen: prompt, parser, cache, cost. --dry-run builds and prices;
# --run is gated and NOT part of this build's remit.
# ---------------------------------------------------------------------------

LLM_SYSTEM = (
    "You are screening first-person accounts for a research instrument. Read "
    "the account and answer in the exact format requested. Use only what the "
    "account says; do not invent people or interests."
)

TP_MARKER = "THIRD_PARTY:"
CP_MARKER = "COUNTERPARTY:"
INTEREST_MARKER = "INTEREST:"
FAVOURED_MARKER = "FAVOURED_BY:"
FAVOURED_TOKENS = ("YTA", "NTA", "NONE")


def llm_prompt(item: CrowdGoldItem) -> str:
    return (
        "Below is a first-person account posted to a forum where readers judge "
        "whether the WRITER was in the wrong (YTA) or not in the wrong (NTA).\n\n"
        "<<<BEGIN ACCOUNT>>>\n"
        f"{item.post_text}\n"
        "<<<END ACCOUNT>>>\n\n"
        "Identify:\n"
        "  1. the COUNTERPARTY: the person the writer is in conflict with;\n"
        "  2. ONE THIRD PARTY: a person the account itself mentions, by role "
        "or by name, who is neither the writer nor the counterparty, and "
        "whose situation is affected by how the conflict is judged (for "
        "example a sibling, friend, boss, child or roommate). If the account "
        "names no such person, write NONE.\n"
        "  3. that person's INTEREST in the outcome, in one sentence, using "
        "only what the account says, and referring to the writer as \"the "
        "writer\" -- never as I, me or my -- so the sentence reads in the "
        "third person;\n"
        "  4. which finding about the writer serves that interest: YTA (the "
        "writer was in the wrong), NTA (the writer was not in the wrong), or "
        "NONE if the account does not establish it.\n\n"
        "Answer with exactly these four lines and nothing else:\n"
        f"{CP_MARKER} <role or name>\n"
        f"{TP_MARKER} <role or name, or NONE>\n"
        f"{INTEREST_MARKER} <one sentence, or NONE>\n"
        f"{FAVOURED_MARKER} YTA | NTA | NONE"
    )


def parse_llm(text: str) -> dict:
    cp = marker_line(text, CP_MARKER)
    tp = marker_line(text, TP_MARKER)
    interest = marker_line(text, INTEREST_MARKER)
    fav = marker_line(text, FAVOURED_MARKER).upper().strip(" .")
    fav_tok = next((t for t in FAVOURED_TOKENS if fav.startswith(t)), "")
    parsed = bool(tp) and bool(fav_tok)
    tp_none = tp.strip().upper().startswith("NONE")
    return {
        "counterparty": cp, "third_party": "" if tp_none else tp,
        "interest": "" if tp_none else interest,
        "favoured_by": fav_tok, "parsed": parsed, "third_party_named": parsed and not tp_none,
    }


def tpscreen_cache_path(model: str, item_id: str, out_dir: Optional[Path] = None) -> Path:
    return (out_dir or OUT_DIR) / f"{TPSCREEN_PREFIX}{_safe(model)}_{_safe(item_id)}.json"


def cache_is_clean(model: str, out_dir: Path = OUT_DIR) -> tuple[bool, dict]:
    pre = f"{TPSCREEN_PREFIX}{_safe(model)}_"
    n = 0
    if out_dir.is_dir():
        with os.scandir(out_dir) as it:
            for e in it:
                if e.name.startswith(pre):
                    n += 1
    return n == 0, {pre + "*": n}


def llm_cost_model(model: str, items: Sequence[CrowdGoldItem], *,
                   prices: dict, use_bpe: bool = True) -> dict:
    bpe = _optional_bpe_counter() if use_bpe else None
    pin, pout = prices.get(model, (0.0, 0.0))
    ptok = 0
    for it in items:
        p = llm_prompt(it)
        ptok += count_tokens(p, tokenizer=bpe) if bpe else count_tokens(p)
    ctok = ASSUMED_LLM_COMPLETION * len(items)
    usd = ptok / 1e6 * pin + ctok / 1e6 * pout
    return {
        "model": model, "priced": model in prices, "calls": len(items),
        "prompt_tokens": ptok, "completion_tokens_assumed": ctok,
        "price_in_per_mtok": pin, "price_out_per_mtok": pout,
        "usd": round(usd, 3),
        "tokenizer": "tiktoken cl100k_base BPE" if bpe else
                     "regex approximation (undercounts by ~20-35%)",
        "assumption": f"{ASSUMED_LLM_COMPLETION} completion tokens per call "
                      "(four short lines)",
    }


def run_llm_screen(model: str, items: Sequence[CrowdGoldItem], *,
                   max_tokens: int = LLM_MAX_TOKENS) -> tuple[dict, dict]:
    """--run only. One call per item, cached under TPSCREEN_PREFIX. Returns
    (results by item_id, guard)."""
    results: dict = {}
    n_trunc = n_parse_fail = n_cf = 0
    for it in items:
        cache = tpscreen_cache_path(model, it.item_id)
        rec = None
        if cache.exists():
            try:
                rec = json.loads(cache.read_text())
            except json.JSONDecodeError:
                rec = None
            if rec is not None and rec.get("post_sha256") != sha256_text(it.post_text):
                rec = None
        if rec is None:
            res = rcd.generate_any(model, LLM_SYSTEM, llm_prompt(it), sample_idx=0,
                                   max_tokens=max_tokens)
            rec = {"model": model, "item_id": it.item_id,
                   "post_sha256": sha256_text(it.post_text),
                   "output": res.text, "finish_reason": getattr(res, "finish_reason", ""),
                   "prompt_tokens": getattr(res, "prompt_tokens", 0),
                   "completion_tokens": getattr(res, "completion_tokens", 0)}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rec, ensure_ascii=False))
        parsed = parse_llm(rec.get("output") or "")
        parsed["finish_reason"] = rec.get("finish_reason", "")
        parsed["truncated"] = int(parsed["finish_reason"] in ("max_tokens", "length", "MAX_TOKENS"))
        # Azure's content filter returns an EMPTY completion with finish_reason
        # "content_filter": a platform refusal, not a parse failure of the
        # model's answer. Such an item is EXCLUDED from the screen (screened_in
        # False, stake_source "content_filter") and counted separately (16.21
        # amendment, 2026-09-14, recorded in the prereg before any cell call;
        # 14 of the 16 such items are already dropped by the cell's own grok
        # content-filter screen).
        parsed["content_filtered"] = int(parsed["finish_reason"] == "content_filter"
                                         and not (rec.get("output") or "").strip())
        n_cf += parsed["content_filtered"]
        n_trunc += parsed["truncated"]
        n_parse_fail += int(not parsed["parsed"] and not parsed["content_filtered"])
        results[it.item_id] = parsed
    n = max(1, len(items))
    guard = {"n": len(items), "truncated": n_trunc, "parse_failed": n_parse_fail,
             "content_filtered": n_cf,
             "truncated_share": n_trunc / n, "parse_failed_share": n_parse_fail / n,
             "content_filtered_share": n_cf / n,
             "pass": (n_trunc / n <= MAX_PARSE_FAIL_SHARE
                      and n_parse_fail / n <= MAX_PARSE_FAIL_SHARE)}
    return results, guard


# ---------------------------------------------------------------------------
# Assemble the screen file
# ---------------------------------------------------------------------------

def merge_llm(rec: dict, llm: Optional[dict]) -> dict:
    """Attach the LLM result to a heuristic record and settle the fields the
    runner reads: third_party_description, third_party_interest, stake,
    stake_source, screened_in."""
    out = dict(rec)
    out["llm"] = llm
    if llm and llm.get("content_filtered"):
        out["screened_in"] = False
        out["stake_source"] = "content_filter"
        out["stake_final"] = "none"
        return out
    if llm and llm.get("parsed"):
        if llm["third_party_named"]:
            out["screened_in"] = True
            out["third_party_description"] = llm["third_party"]
            out["third_party_interest"] = rewrite_interest(llm["interest"]) if llm["interest"] else "(not stated)"
            out["third_party_interest_source"] = {"arm": None, "output_idx": None, "rule": "llm screen",
                                                  "rewritten": bool(llm["interest"]) and
                                                  rewrite_interest(llm["interest"]) != llm["interest"],
                                                  "first_person_free": first_person_free(
                                                      out["third_party_interest"])}
            out["stake_final"] = "none" if llm["favoured_by"] == "NONE" else llm["favoured_by"]
            out["stake_source"] = "llm"
            out["heuristic_agrees"] = bool(
                rec.get("third_party")
                and _same_person(rec["third_party"], llm["third_party"]))
        else:
            out["screened_in"] = False
            out["stake_source"] = "llm"
            out["stake_final"] = "none"
        return out
    if rec.get("candidate") and rec.get("third_party"):
        out["screened_in"] = True
        out["third_party_description"] = describe_third_party(rec["third_party"])
        out["third_party_interest"] = rec["third_party"]["interest"]
        out["third_party_interest_source"] = rec["third_party"].get("interest_source")
        out["stake_final"] = rec["stake"]["heuristic"]
        out["stake_source"] = "heuristic"
    else:
        out["screened_in"] = False
        out["stake_final"] = "none"
        out["stake_source"] = "heuristic"
    return out


def _same_person(tp: dict, llm_text: str) -> bool:
    key = tp.get("key", "")
    if key.startswith("name:"):
        return key[5:].lower() in llm_text.lower()
    canon = key[5:]
    for s in ROLE_LEXICON.get(canon, (canon,)):
        if re.search(rf"(?<![A-Za-z-]){re.escape(s)}(?:s|'s)?(?![A-Za-z-])", llm_text, re.I):
            return True
    return False


def build_screen(items: Sequence[CrowdGoldItem], single_idx: dict, delib_idx: dict, *,
                 min_support: int, llm_results: Optional[dict] = None,
                 llm_model: Optional[str] = None) -> dict:
    recs = []
    for it in items:
        pairs = load_outputs(single_idx.get(it.item_id, []))
        hint = counterparty_r0_hint(it.item_id, delib_idx)
        r = asdict(screen_item(it, [t for _, t in pairs], r0_hint=hint, min_support=min_support,
                               arms=[a for a, _ in pairs]))
        recs.append(merge_llm(r, (llm_results or {}).get(it.item_id)))
    n_cand = sum(1 for r in recs if r["candidate"])
    n_in = sum(1 for r in recs if r["screened_in"])
    src_counts = Counter(str((r.get("third_party_interest_source") or {}).get("arm"))
                         for r in recs if r["screened_in"])
    n_first_person = sum(1 for r in recs if r["screened_in"]
                         and not first_person_free(r.get("third_party_interest", "")))
    return {
        "registration": REGISTRATION,
        "purpose": "third-party screen for the four-seat (n-perspective) "
                   "deliberation cell; see scripts/screen_third_party.py",
        "panel": {"n_items": len(items), **{k: v for k, v in PANEL_KW.items()}},
        "single_agent_cache": {"model": SCREEN_MODEL, "scaffold": "narrative_cot",
                               "cap": SCREEN_CAP, "prefix": SINGLE_AGENT_PREFIX,
                               "arms": list(SCREEN_ARMS), "excluded_arms": list(EXCLUDED_ARMS)},
        "interest_source_counts": dict(src_counts),
        "n_interest_with_first_person_voice": n_first_person,
        "heuristics": HEURISTICS,
        "min_support": min_support,
        "llm_screen": {"model": llm_model, "applied": bool(llm_results),
                       "n_results": len(llm_results or {})},
        "n_candidates_heuristic": n_cand,
        "n_screened_in": n_in,
        "stake_final_counts": dict(Counter(r["stake_final"] for r in recs if r["screened_in"])),
        "stake_source_counts": dict(Counter(r["stake_source"] for r in recs if r["screened_in"])),
        "items": recs,
    }


def screened_items(screen: dict) -> dict[str, dict]:
    """item_id -> record for every screened-in item. The runner's entry point."""
    return {r["item_id"]: r for r in screen["items"] if r.get("screened_in")}


def print_summary(screen: dict, *, show: int = 8) -> None:
    print("\n" + "=" * 78)
    print("THIRD-PARTY SCREEN (zero spend)")
    print("=" * 78)
    print(f"  panel items                 {screen['panel']['n_items']}")
    print(f"  heuristic candidates        {screen['n_candidates_heuristic']}")
    print(f"  screened in (runner reads)  {screen['n_screened_in']}")
    print(f"  stake (final)               {screen['stake_final_counts']}")
    print(f"  stake source                {screen['stake_source_counts']}")
    print(f"  interest text source arm    {screen.get('interest_source_counts')}   "
          f"(as_asker_stance excluded; first-person voice in "
          f"{screen.get('n_interest_with_first_person_voice')} interest texts)")
    reasons = Counter(r["reason"] for r in screen["items"] if not r["candidate"])
    for k, v in reasons.most_common():
        print(f"  not a candidate: {v:>4}  {k}")
    print(f"\n  first {show} candidates (eyeball these; the LLM screen confirms):")
    n = 0
    for r in screen["items"]:
        if not r["screened_in"]:
            continue
        tp = r.get("third_party") or {}
        print(f"   {r['item_id'][:8]} gold={r['gold_verdict']:<3} "
              f"cp={r['counterparty'].get('key', '?'):<18} "
              f"tp={r.get('third_party_description', '?')[:34]:<34} "
              f"stake={r['stake_final']:<4} support={tp.get('support', '?')}/"
              f"{tp.get('of_outputs', '?')}")
        n += 1
        if n >= show:
            break


# ---------------------------------------------------------------------------
# Selftest -- offline; generation stubbed; cache dir redirected to a tempdir
# ---------------------------------------------------------------------------

_SAMPLE_OUTPUT_BULLETS = """### Section 1 – Protagonist
I am the sister.

### Section 2 – Stakeholders
- Myself (Sarah): My emotional security and wedding excitement are at stake.
- My brother: His autonomy in his 8-year relationship and timing of his proposal.
- Brother's girlfriend: Her chance for a joyful, unpressured engagement moment.
- My fiancé: Our wedding planning focus and family dynamics.
- Both sets of parents and extended family: Potential divided attention.
- Future selves of all: Long-term sibling relationship.

### Section 3 – Consequences
whatever
VERDICT: YTA
"""

_SAMPLE_OUTPUT_INLINE = (
    "Section 1 – Protagonist: I am the sister.\n\n"
    "Section 2 – Stakeholders: Me (the sister), whose emotional security feels "
    "threatened; my brother, whose autonomy is at stake; his girlfriend of eight "
    "years, whose shared future could be delayed; my fiancé, potentially drawn "
    "into family tension; immediate family/parents, who might face divided attention.\n\n"
    "Section 3 – Consequences: whatever\nVERDICT: YTA\n"
)

_SAMPLE_ACCOUNT = (
    "It was a longstanding fear of mine that my brother would get married before "
    "I did. He and his girlfriend have been dating 8 years. Then my boyfriend "
    "proposed. My brother informed me he was planning to propose this summer. "
    "My brother says I have no say. My fiancé thinks I am overreacting."
)


class _Stub:
    def __init__(self, text, finish_reason="stop"):
        self.text = text
        self.finish_reason = finish_reason
        self.prompt_tokens = 300
        self.completion_tokens = 40


def _stub_generate(model, system, user, *, sample_idx=0, max_tokens=0, **kw):
    return _Stub(f"{CP_MARKER} my brother\n{TP_MARKER} the brother's girlfriend\n"
                 f"{INTEREST_MARKER} her engagement should not be delayed.\n"
                 f"{FAVOURED_MARKER} YTA")


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # ---- parser: bullet and inline forms ---------------------------------
    eb = [classify(e) for e in section2_entries(_SAMPLE_OUTPUT_BULLETS, 0)]
    heads = [e.head for e in eb]
    check("bullet form parses six stakeholder heads",
          len(eb) == 6 and heads[0].startswith("Myself"))
    check("writer entry classified as writer", eb[0].kind == "writer")
    check("'My brother' -> role brother", eb[1].key == "role:brother")
    check("'Brother's girlfriend' -> rightmost role noun girlfriend",
          eb[2].key == "role:girlfriend")
    check("'My fiancé' -> role fiance", eb[3].key == "role:fiance")
    check("'parents and extended family' is generic", eb[4].kind == "generic")
    check("'Future selves of all' is generic", eb[5].kind == "generic")
    ei = [classify(e) for e in section2_entries(_SAMPLE_OUTPUT_INLINE, 1)]
    check("inline form parses five entries with the same keys",
          [e.key for e in ei][:4] == ["writer", "role:brother", "role:girlfriend",
                                      "role:fiance"] and len(ei) == 5)
    check("absent Section 2 -> no entries", section2_entries("no sections here", 0) == [])
    check("body captured for the interest text",
          "joyful" in eb[2].body and "autonomy" in eb[1].body)

    # ---- classification helpers -------------------------------------------
    check("owner_class: 'Brother's girlfriend' attaches to the counterparty",
          owner_class("Brother's girlfriend", "role:brother") == "counterparty")
    check("owner_class: 'his girlfriend' -> counterparty; 'My fiancé' -> writer",
          owner_class("his girlfriend of eight years", "role:brother") == "counterparty"
          and owner_class("My fiancé", "role:brother") == "writer")
    check("mentions counts every surface variant with plural/possessive tails",
          mentions(_SAMPLE_ACCOUNT, "role:brother") == 3
          and mentions(_SAMPLE_ACCOUNT, "role:girlfriend") == 1
          and mentions(_SAMPLE_ACCOUNT, "role:boss") == 0)
    check("a proper-name head is a person keyed by name",
          classify(Entry("Jake", "", 0)).key == "name:Jake")
    check("role_for_name reads 'my friend Jake' from the account",
          role_for_name("Last week my best friend Jake called me.", "Jake") == "friend"
          and role_for_name("Jake, my roommate, was upset.", "Jake") == "roommate")
    check("a pet is generic", is_generic("The dog"))

    # ---- screen_item end to end on the sample ----------------------------
    item = CrowdGoldItem(item_id="itemX", post_text=_SAMPLE_ACCOUNT, gold_verdict="YTA")
    rec = screen_item(item, [_SAMPLE_OUTPUT_BULLETS, _SAMPLE_OUTPUT_INLINE],
                      r0_hint=None, min_support=2)
    check("counterparty = brother by account mentions",
          rec.counterparty.get("key") == "role:brother")
    check("third party = the brother's girlfriend (support 2 of 2)",
          rec.candidate and rec.third_party["key"] == "role:girlfriend"
          and rec.third_party["support"] == 2)
    check("fiancé (writer's partner) ties the girlfriend on support and mentions and is "
          "kept as an alternative: the determinate-stake tiebreak picks the "
          "counterparty-attached person",
          any(a["key"] == "role:fiance" for a in rec.alternatives)
          and rec.third_party["stake_rank"] == 2
          and next(a for a in rec.alternatives if a["key"] == "role:fiance")["stake_rank"] == 1)
    check("inline head is cut at a plain comma and the tail kept as the interest text",
          any(e.head == "my fiancé" and "family tension" in e.body for e in ei))
    check("stake heuristic: counterparty-aligned -> YTA, marked heuristic",
          rec.stake["heuristic"] == "YTA" and rec.stake["stake_source"] == "heuristic")
    rec2 = screen_item(item, [_SAMPLE_OUTPUT_BULLETS], r0_hint="girlfriend", min_support=1)
    check("an r0 hint that names a listed person overrides the mention count",
          rec2.counterparty["key"] == "role:girlfriend" and rec2.counterparty["r0_hint_agrees"])
    rec3 = screen_item(item, [_SAMPLE_OUTPUT_BULLETS], r0_hint=None, min_support=3)
    check("min_support above the available outputs -> not a candidate, with a reason",
          not rec3.candidate and "support" in rec3.reason)
    rec4 = screen_item(item, ["no section two here"], r0_hint=None, min_support=1)
    check("no Section 2 anywhere -> not a candidate", not rec4.candidate)
    only_writer = "### Section 2 – Stakeholders\n- Me: my pride.\n- Society: norms.\n### Section 3"
    rec5 = screen_item(item, [only_writer], r0_hint=None, min_support=1)
    check("writer + generic only -> no person entries -> not a candidate",
          not rec5.candidate and "no person" in rec5.reason)

    # ---- the interest text: third_person source, possessive rewrite, no first person ----
    check("rewrite_interest: 'my/our X' -> \"the writer's X\" inside prose; his/her left alone",
          rewrite_interest("Family harmony and her view of my parenting style") ==
          "Family harmony and her view of the writer's parenting style"
          and rewrite_interest("Our wedding planning focus") == "the writer's wedding planning focus"
          and rewrite_interest("His autonomy") == "His autonomy")
    check("first_person_free flags I / me / my / we / our / us and passes third-person prose",
          not first_person_free("she risks feeling undermined if I push too hard")
          and not first_person_free("keeping us together") and first_person_free("Her chance for a joyful moment")
          and first_person_free("the writer's stability"))
    ents = [Entry("my wife", "Family harmony and her view of my parenting style; she risks feeling "
                  "undermined if I push too hard", 3, arm=AS_ASKER_ARM),
            Entry("the writer's wife", "Her view of the writer's parenting style", 0, arm=THIRD_PERSON_ARM),
            Entry("the writer's wife", "Her view of the writer's parenting style and household harmony", 1,
                  arm=THIRD_PERSON_ARM)]
    body, src = choose_interest(ents)
    check("choose_interest prefers the longest THIRD_PERSON body over a longer as_asker body and "
          "records arm / output_idx / rule",
          body == "Her view of the writer's parenting style and household harmony"
          and src["arm"] == THIRD_PERSON_ARM and src["output_idx"] == 1 and src["n_third_person_bodies"] == 2)
    body2, src2 = choose_interest([ents[0]])
    check("choose_interest with only a first-person as_asker body ('... if I push too hard') gives "
          "'(not stated)' rather than injecting writer voice; the rule is recorded",
          body2 == "(not stated)" and src2["arm"] is None and "first-person" in src2["rule"])
    body3, src3 = choose_interest([Entry("his girlfriend", "Her chance for a joyful engagement", 4,
                                         arm=AS_ASKER_ARM)])
    check("choose_interest falls back to a pronoun-free as_asker body when no third_person body exists",
          body3 == "Her chance for a joyful engagement" and src3["arm"] == AS_ASKER_ARM and src3["output_idx"] == 4)
    body4, src4 = choose_interest([Entry("my fiancé", "Our wedding planning focus and my peace of mind", 2,
                                         arm=AS_ASKER_ARM)])
    check("the possessive rewrite makes an as_asker body eligible and is flagged",
          body4 == "the writer's wedding planning focus and the writer's peace of mind" and src4["rewritten"])
    rec_arms = screen_item(item, [_SAMPLE_OUTPUT_BULLETS, _SAMPLE_OUTPUT_INLINE], r0_hint=None,
                           min_support=2, arms=[AS_ASKER_ARM, THIRD_PERSON_ARM])
    check("screen_item with arms: the girlfriend's interest comes from the third_person output "
          "(index 1) even though the as_asker body is longer; support_by_arm recorded",
          rec_arms.candidate and rec_arms.third_party["interest_source"]["arm"] == THIRD_PERSON_ARM
          and rec_arms.third_party["interest_source"]["output_idx"] == 1
          and rec_arms.third_party["support_by_arm"] == {AS_ASKER_ARM: 1, THIRD_PERSON_ARM: 1}
          and "delayed" in rec_arms.third_party["interest"])
    try:
        screen_item(item, [_SAMPLE_OUTPUT_BULLETS], r0_hint=None, min_support=1, arms=["as_asker_stance"])
        stance_refused = False
    except ValueError:
        stance_refused = True
    check("screen_item refuses an as_asker_stance output as a source", stance_refused)
    with tempfile.TemporaryDirectory() as td_idx:
        for arm in ("third_person", "as_asker", "as_asker_stance"):
            (Path(td_idx) / f"{SINGLE_AGENT_PREFIX}{arm}_itemZ_00.json").write_text(
                json.dumps({"output": _SAMPLE_OUTPUT_BULLETS, "finish_reason": "stop"}))
        (Path(td_idx) / f"{SINGLE_AGENT_PREFIX}as_asker_stance_itemZ_01.json").write_text(
            json.dumps({"output": _SAMPLE_OUTPUT_BULLETS, "finish_reason": "stop"}))
        s_idx, _ = index_caches(Path(td_idx))
        pairs = load_outputs(s_idx.get("itemZ", []))
        check("index_caches / load_outputs index third_person and as_asker records only; the two "
              "as_asker_stance records are excluded from the screen source",
              len(s_idx.get("itemZ", [])) == 2 and [a for a, _ in pairs] == ["as_asker", "third_person"]
              and all(arm_of_path(p) in SCREEN_ARMS for p in s_idx["itemZ"]))
    check("the LLM prompt asks for the interest in the third person ('the writer', never I / me / my)",
          "never as I, me or my" in llm_prompt(item))

    # ---- merge_llm precedence ----------------------------------------------
    hrec = asdict(rec)
    m0 = merge_llm(hrec, None)
    check("without an LLM result the heuristic decides: screened_in, stake_source=heuristic",
          m0["screened_in"] and m0["stake_source"] == "heuristic" and m0["stake_final"] == "YTA"
          and "girlfriend" in m0["third_party_description"])
    m1 = merge_llm(hrec, parse_llm(_stub_generate("m", "s", "u").text))
    check("an LLM result overrides: description/interest/stake from the LLM, agreement flagged, "
          "interest source recorded as the llm screen",
          m1["stake_source"] == "llm" and m1["stake_final"] == "YTA"
          and m1["third_party_description"] == "the brother's girlfriend"
          and m1["heuristic_agrees"] is True
          and m1["third_party_interest_source"]["rule"] == "llm screen"
          and m1["third_party_interest_source"]["first_person_free"] is True)
    check("the heuristic record carries the interest's provenance the runner can disclose",
          m0["third_party_interest_source"]["arm"] == THIRD_PERSON_ARM
          and "output_idx" in m0["third_party_interest_source"])
    m2 = merge_llm(hrec, parse_llm(f"{CP_MARKER} brother\n{TP_MARKER} NONE\n{INTEREST_MARKER} NONE\n{FAVOURED_MARKER} NONE"))
    check("LLM 'NONE' screens the item OUT even when the heuristic had a candidate",
          m2["screened_in"] is False and m2["stake_source"] == "llm")
    m3 = merge_llm(hrec, parse_llm("garbage with no markers"))
    check("an unparseable LLM result falls back to the heuristic (and is counted by the guard)",
          m3["screened_in"] and m3["stake_source"] == "heuristic")
    p = parse_llm(f"**{TP_MARKER}** the boss\n{FAVOURED_MARKER} nta.")
    check("parse_llm is line-anchored and tolerant of bold/case/trailing period",
          p["parsed"] and p["third_party"] == "the boss" and p["favoured_by"] == "NTA")

    # ---- descriptions the runner will quote --------------------------------
    check("clean_head strips parentheticals, dash/verb tails and 'indirectly' qualifiers",
          clean_head("my friend—his role in giving honest advice") == "the writer's friend"
          and clean_head("Her mum (minimal intersection)") == "her mum"
          and clean_head("daughter is heartbroken over missing the trip") == "daughter"
          and clean_head("and peripherally the ex-wife") == "the ex-wife"
          and clean_head("my parents risk financial resentment") == "the writer's parents")
    check("clean_head: 'my/our X' -> \"the writer's X\"; his/her/their kept for the advocate to resolve",
          clean_head("My wife") == "the writer's wife" and clean_head("Our unborn child") == "the writer's unborn child"
          and clean_head("His mother") == "his mother")
    check("describe_third_party takes the MODAL cleaned head, not the longest",
          describe_third_party({"key": "role:child", "heads": [
              "Our son", "our son", "our son (family stability, daily access to both parents)"]})
          == "the writer's son"
          and describe_third_party(rec.third_party) == "brother's girlfriend")
    check("describe_third_party keeps a proper name capitalised and adds the account role",
          describe_third_party({"key": "name:Jake", "heads": ["Jake", "Jake (grieving father)"],
                                "role_in_account": "friend"}) == "Jake, the writer's friend"
          and describe_third_party({"key": "name:Mary", "heads": ["Mary"], "role_in_account": ""}) == "Mary")
    check("a named pet is not a person: '(the hedgehog)' head is generic, 'my dog Loki' in the account "
          "screens the name out",
          classify(Entry("Loki (the hedgehog)", "", 0)).kind == "generic"
          and not screen_item(CrowdGoldItem("p", "My dog Loki bit my brother. My brother is furious.", "NTA"),
                              ["### Section 2 – Stakeholders\n- Me: x\n- My brother: y\n- Loki: z\n### Section 3"],
                              r0_hint=None, min_support=1).candidate)

    # ---- prompts and cost --------------------------------------------------
    pr = llm_prompt(item)
    check("LLM prompt carries the account verbatim and the four marker lines",
          _SAMPLE_ACCOUNT in pr and all(m in pr for m in (CP_MARKER, TP_MARKER, INTEREST_MARKER, FAVOURED_MARKER)))
    check("LLM prompt does not leak the heuristic's candidate (independence)",
          "girlfriend" not in pr.split("<<<END ACCOUNT>>>")[1])
    cm = llm_cost_model(DEFAULT_LLM_MODEL, [item] * 10, prices=PRICES, use_bpe=False)
    check("cost model prices from PRICES and counts one call per item",
          cm["priced"] and cm["calls"] == 10 and cm["usd"] > 0)
    cm2 = llm_cost_model("no-such-model", [item], prices=PRICES, use_bpe=False)
    check("an unpriced model is flagged (usd 0 is a placeholder, not a ceiling)",
          not cm2["priced"] and cm2["usd"] == 0)
    cm3 = llm_cost_model("no-such-model", [item], prices={**PRICES, "no-such-model": (1.0, 2.0)},
                         use_bpe=False)
    check("--price style override makes a new deployment priced without a code change",
          cm3["priced"] and cm3["usd"] > 0)

    # ---- cache namespace: no foreign file; --run stubbed into a tempdir ----
    own = f"{TPSCREEN_PREFIX}{_safe(DEFAULT_LLM_MODEL)}_"
    foreign = []
    if OUT_DIR.is_dir():
        with os.scandir(OUT_DIR) as it:
            for e in it:
                if e.name.startswith(TPSCREEN_PREFIX) and (
                        "stub" in e.name or "_m_" in e.name or "selftest" in e.name
                        or "itemX" in e.name):
                    foreign.append(e.name)
    clean, counts = cache_is_clean(DEFAULT_LLM_MODEL)
    check(f"no foreign (stub / selftest) file carries the tpscreen prefix in "
          f"{OUT_DIR.name} (own footprint {counts})", foreign == [])
    # The LLM screen was RUN on 2026-09-13 (249 cached calls, $0.07); the
    # standing check is that the namespace is either empty or holds exactly
    # the panel-sized run, never a partial one from elsewhere.
    check(f"tpscreen namespace for {DEFAULT_LLM_MODEL} is empty or holds the 249-item "
          f"screen {counts}", clean or all(v in (0, 249) for v in counts.values()))
    real_gen, real_out = rcd.generate_any, OUT_DIR
    try:
        rcd.generate_any = _stub_generate  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as td:
            globals()["OUT_DIR"] = Path(td)
            try:
                res, guard = run_llm_screen("stub-model", [item, CrowdGoldItem("itemY", "b", "NTA")])
                n_files = len([q for q in Path(td).iterdir() if q.name.startswith(own.split("_", 2)[0])])
                names = sorted(q.name for q in Path(td).iterdir())
            finally:
                globals()["OUT_DIR"] = real_out
    finally:
        rcd.generate_any = real_gen
    check("stubbed --run path writes one cache file per item in the TEMP dir only",
          n_files == 2 and all(n.startswith(f"{TPSCREEN_PREFIX}stub-model_") for n in names))
    check("stubbed results parse and the guard passes",
          guard["pass"] and res["itemX"]["third_party_named"] and res["itemX"]["favoured_by"] == "YTA")
    clean_after, _ = cache_is_clean("stub-model")
    check("nothing landed in the real output directory", clean_after)

    # ---- CLI refusals ------------------------------------------------------
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--run", "--model", "no-such-model", "--skip-heuristic",
                   "--out", str(Path(tempfile.gettempdir()) / "tp_selftest_never_written.json")])
    check("--run on an unpriced model refuses (exit 2) before any call",
          rc == 2 and "PRICES" in buf.getvalue())

    def _tripwire(*a, **k):
        raise AssertionError("generate_any was called on a --dry-run")
    real_gen = rcd.generate_any
    with tempfile.TemporaryDirectory() as td:
        try:
            rcd.generate_any = _tripwire  # type: ignore[assignment]
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["--dry-run", "--skip-heuristic", "--no-bpe",
                           "--out", str(Path(td) / "screen.json"),
                           "--prompts-out", str(Path(td) / "prompts.json")])
        finally:
            rcd.generate_any = real_gen
        pj = json.loads((Path(td) / "prompts.json").read_text()) if (Path(td) / "prompts.json").exists() else {}
        check("main(--dry-run) exits 0, never calls the generator, writes one prompt per panel item "
              "and prints NOT CALLING",
              rc == 0 and "NOT CALLING" in buf.getvalue() and pj.get("model") == DEFAULT_LLM_MODEL
              and len(pj.get("prompts", {})) == len(load_items(**PANEL_KW))
              and "ceiling" in buf.getvalue())

    # ---- the real caches, if present: index shape -------------------------
    single, delib = index_caches()
    if single:
        check(f"single-agent cache indexes {len(single)} items with up to 6 outputs each "
              "(third_person x3 + as_asker x3; the stance arm is not indexed)",
              len(single) >= 200 and max(len(v) for v in single.values()) <= 6
              and not any(arm_of_path(p) in EXCLUDED_ARMS for v in single.values() for p in v))
        if SCREEN_PATH.exists():
            sj = json.loads(SCREEN_PATH.read_text())
            if "interest_source_counts" in sj:
                ins = [r for r in sj["items"] if r.get("screened_in")]
                check(f"the screen JSON on disk ({len(ins)} screened in) records an interest source for "
                      "every screened-in item, none from the stance arm, and no interest text carries "
                      f"first-person voice (source arms {sj['interest_source_counts']})",
                      all(r.get("third_party_interest_source") is not None for r in ins)
                      and not any((r.get("third_party_interest_source") or {}).get("arm") in EXCLUDED_ARMS
                                  for r in ins)
                      and sj.get("n_interest_with_first_person_voice") == 0
                      and all(first_person_free(r.get("third_party_interest", "")) for r in ins))
            else:
                print("  (screen JSON on disk predates the interest-source rule; re-run the heuristic "
                      "screen to refresh it)")
        check(f"deliberation r0 hint index covers {len(delib)} items (2 arms each)",
              len(delib) >= 100 and max(len(v) for v in delib.values()) <= 2)
    else:
        print("  (no cached single-agent outputs found; index checks skipped)")

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_prices(specs: Sequence[str]) -> dict:
    out = dict(PRICES)
    for s in specs or ():
        try:
            name, pair = s.split("=", 1)
            pin, pout = (float(x) for x in pair.split(","))
        except ValueError:
            raise SystemExit(f"--price expects MODEL=IN,OUT (USD per Mtok), got {s!r}")
        out[name.strip()] = (pin, pout)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Addendum 16.21: third-party screen on the crowd-gold AITA "
                    "panel. Default = zero-spend heuristic screen. --dry-run "
                    "prices and writes the LLM screen prompts without calling. "
                    "Generates NOTHING unless --run is passed.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="also build and price the LLM screen; write prompts; no call")
    ap.add_argument("--run", action="store_true",
                    help="REQUIRED to call the LLM screen; spends money")
    ap.add_argument("--resume", action="store_true",
                    help="proceed despite existing tpscreen cache files")
    ap.add_argument("--verify-cache-clean", action="store_true")
    ap.add_argument("--model", default=DEFAULT_LLM_MODEL,
                    help="LLM screen model (config-driven; add --price for a new deployment)")
    ap.add_argument("--price", action="append", default=[],
                    help="MODEL=IN,OUT USD per Mtok; extends PRICES without a code change")
    ap.add_argument("--min-support", type=int, default=2,
                    help="cached outputs that must list the third party")
    ap.add_argument("--out", default=str(SCREEN_PATH))
    ap.add_argument("--prompts-out", default=str(PROMPTS_PATH))
    ap.add_argument("--no-bpe", action="store_true")
    ap.add_argument("--skip-heuristic", action="store_true",
                    help="(selftest plumbing) do not build the heuristic screen")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    prices = _parse_prices(args.price)
    if args.run and args.model not in prices:
        print(f"\nERROR: {args.model} not in PRICES; pass --price {args.model}=IN,OUT "
              "before spending (a $0.00 dry-run is a placeholder, not a ceiling).\n")
        return 2
    clean, counts = cache_is_clean(args.model)
    if args.verify_cache_clean:
        print(f"tpscreen cache files: {counts}")
        print("CLEAN" if clean else "DIRTY -- refusing")
        return 0 if clean else 3

    try:
        items = load_items(**PANEL_KW)
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: panel unavailable: {e}\n")
        return 2
    print(f"panel: {len(items)} items "
          f"({sum(1 for i in items if i.gold_verdict == 'YTA')} YTA / "
          f"{sum(1 for i in items if i.gold_verdict == 'NTA')} NTA)")

    llm_results = None
    if args.run:
        if not clean and not args.resume:
            print(f"\nERROR: tpscreen cache not clean {counts}; pass --resume only if "
                  "the prompt is known unchanged.\n")
            return 3
        llm_results, guard = run_llm_screen(args.model, items)
        print(f"LLM screen guard: {guard}")
        if not guard["pass"]:
            print("\n  GUARD FAILED -- truncation or parse failure above "
                  f"{MAX_PARSE_FAIL_SHARE:.0%}; the LLM screen is NOT readable. "
                  "Raise --max-tokens or inspect the outputs.\n")
            return 4

    screen = None
    if not args.skip_heuristic:
        single, delib = index_caches()
        print(f"indexed {sum(len(v) for v in single.values())} cached single-agent "
              f"outputs over {len(single)} items; {len(delib)} items carry a "
              "counterparty r0 hint")
        screen = build_screen(items, single, delib, min_support=args.min_support,
                              llm_results=llm_results, llm_model=args.model if llm_results else None)
        print_summary(screen, show=args.show)
        Path(args.out).write_text(json.dumps(screen, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")

    if args.dry_run or (not args.run and not args.skip_heuristic):
        cm = llm_cost_model(args.model, items, prices=prices, use_bpe=not args.no_bpe)
        print("\n" + "=" * 78)
        print("DRY RUN -- LLM screen (one call per panel item; NOT called)")
        print("=" * 78)
        for k, v in cm.items():
            print(f"  {k:<26} {v}")
        print(f"  {'ceiling':<26} ${LLM_CEILING_USD:.2f} -- a reasoning model may bill several times "
              f"the {ASSUMED_LLM_COMPLETION}-token completion assumed above; authorise against the "
              "ceiling, not the point estimate")
        if not cm["priced"]:
            print(f"  WARNING: {args.model} not in PRICES -- $0 above is a placeholder")
        if args.dry_run:
            prompts = {it.item_id: {"system": LLM_SYSTEM, "user": llm_prompt(it)} for it in items}
            Path(args.prompts_out).write_text(json.dumps(
                {"model": args.model, "max_tokens": LLM_MAX_TOKENS, "cost_model": cm,
                 "prompts": prompts}, indent=2, ensure_ascii=False))
            print(f"  wrote {len(prompts)} prompts -> {args.prompts_out}")
        if not args.run:
            print("\n*** NOT CALLING: pass --run to execute the LLM screen (spends "
                  f"~${cm['usd']:.2f}). Nothing was generated. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
