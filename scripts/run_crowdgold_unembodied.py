"""
scripts/run_crowdgold_unembodied.py -- Addendum 16.5: the stake-free
deliberation control, the missing ablation for the programme's central claim.

WHY THIS EXISTS

Every deliberation this programme has ever run seats the SAME embodied triple
(writer_advocate / counterparty / neutral_adjudicator), on both instruments and
all four models. There has never been a control in which the seats deliberate
WITHOUT assigned opposed interests. Without it, "opposed-interest embodiment
makes dissent rare and diagnostic" cannot be distinguished from "multi-agent
disagreement flags hard cases", which the prior literature already occupies.

Addendum 16.4 showed the stake-blind counter `n_objectors >= 2` fires on 19.6%
of grok's embodied debates carrying a +0.320 error lift, while on nano (grip
G3 +0.034) the same counter fires on 76.2% carrying +0.039 -- and that the
ordering tracks GRIP, not accuracy. That is cross-model correlational evidence.
This module is the within-model version: same model, same items, same arms,
same protocol, same scaffold, same token caps, stakes removed.

WHAT IS AND IS NOT REMOVED (stated precisely, after red-team review)

REMOVED from the two advocate briefs, and nothing else:
  (1) the first-person alignment ("You speak for the writer / the other party"
      -> "You examine the writer's / the other party's situation")
  (2) the stake clause ("A finding that X is a finding against the party you
      speak for" -> deleted)
ADDED, one sentence, so the absence of stake is asserted rather than merely
left unsaid:
  (3) "You speak for no one, and the verdict counts neither for nor against
      you."
DELIBERATELY PRESERVED, because deleting it would confound the manipulation:
  the ADVOCACY-INTENSITY directive. The embodied brief says "Put the writer's
  case AS STRONGLY AS THE ACCOUNT ITSELF ALLOWS"; the control says "Set out the
  case that can be made from the writer's situation AS STRONGLY AS THE ACCOUNT
  ITSELF ALLOWS". An earlier draft of this module replaced that with a flat
  "set out whatever context the account gives", which silently removed
  argumentative intensity ALONGSIDE the stake -- and a seat told to stop
  arguing objects less for reasons that have nothing to do with embodiment.
  That would have been the reviewer's best alternative explanation for any
  collapse. The seats still make a case as forcefully as the evidence permits;
  what they no longer have is a side that the verdict counts for or against.
  This isolates the commitment component from the perspective-simulation
  component, which is the separation not_origins_review.md section 2 calls for.
ALSO CHANGED, and disclosed rather than claimed as invariant:
  the three seat LABELS ("Writer's Advocate" -> "Reader A -- the writer's
  situation", etc). Labels are interpolated into every round's prompt via
  _role_header/_others_block/synthesis_user/integration_user, so the third
  seat's BRIEF is byte-identical to the embodied neutral adjudicator's but its
  PROMPT is not: it sees different names for itself and its neighbours. Keeping
  "Advocate" in a stake-free arm would have been the larger distortion.
HELD CONSTANT IN BOTH ARMS, and therefore not a confound but a bound on the
claim: the agent system prompt is PROMPTS["narrative_cot"] throughout, whose
Section 2 asks the model to "List every person whose life intersects this
decision and state what is at stake for each." Stakeholder reasoning is present
in BOTH arms. This experiment is therefore "assigned opposed personal interest
vs none, with stakeholder reasoning retained", NOT "stakes vs no stakes".

This is deliberately NOT "three identical neutral readers". That design would
remove the stake AND the role differentiation at once, and a collapse in
dissent would then be attributable to having made the seats the same agent.
Three identical readers remain available as a follow-on arm if this one is
ambiguous.

CACHE SAFETY -- THE FAILURE MODE THIS MODULE IS BUILT AROUND

`call_cache_path` keys on (model, scaffold, max_tokens, arm, item, idx, round,
ROLE_ID) and NOT on the role brief text. Reusing the embodied role_ids with
rewritten briefs would therefore silently serve the EMBODIED responses back for
every call, and the experiment would return "removing stakes changed nothing"
without ever issuing a request -- a believable wrong answer, and precisely the
answer that would kill the paper's claim. The three role ids below are new
(`reader_*`), verified to have zero pre-existing cache files, so every call is
a genuine generation. The `--verify-cache-clean` flag re-checks this before any
spend and refuses to run if it is ever violated.

Usage
-----
  python -m scripts.run_crowdgold_unembodied --selftest          # offline
  python -m scripts.run_crowdgold_unembodied --verify-cache-clean
  python -m scripts.run_crowdgold_unembodied --dry-run --n-yta 99 --n-nta 150 --samples 2
  python -m scripts.run_crowdgold_unembodied --n-yta 99 --n-nta 150 --samples 2   # SPENDS
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scripts.run_crowdgold_deliberation as rcd
from scripts.run_crowdgold_deliberation import POST_DELIM, Role
from scripts.run_phase1_quartet import OUT_DIR

DEFAULT_TAG = "cg_deliberation_unembodied"

#: The stake-free seats. role_ids are NEW -- see CACHE SAFETY above.
UNEMBODIED_ROLES: tuple[Role, ...] = (
    Role(
        role_id="reader_writer_situation",
        label="Reader A -- the writer's situation",
        paper_role="unembodied_control",
        stake="none",
        brief=(
            "You examine the writer's situation -- the person whose "
            "first-person account this is. Their conduct is what is being "
            "judged and they have no power over the outcome. Set out the case "
            "that can be made from the writer's situation as strongly as the "
            "account itself allows, including whatever context or constraint "
            "the other participants are passing over. You speak for no one, "
            "and the verdict counts neither for nor against you. You may not "
            "invent facts that are not in the account, and you may not assert "
            "that the writer is not in the wrong if the account itself does "
            "not support it."
        ),
    ),
    Role(
        role_id="reader_other_situation",
        label="Reader B -- the other party's situation",
        paper_role="unembodied_control",
        stake="none",
        brief=(
            "You examine the situation of the other party in the conflict -- "
            "the person the writer describes themselves as being in conflict "
            "with. That person did not write this account, was not asked for "
            "their version, and cannot add anything to it. Set out the case "
            "that can be made from that party's situation as strongly as the "
            "writer's own account allows, including what the account concedes "
            "against itself. You speak for no one, and the verdict counts "
            "neither for nor against you. You may not invent facts that are "
            "not in the account."
        ),
    ),
    Role(
        # Byte-identical brief to the embodied neutral_adjudicator: that seat
        # already carried stake="none", so it is unchanged by construction and
        # serves as the internal control on the rest of the protocol.
        role_id="reader_norm",
        label="Reader C -- the norm at issue",
        paper_role="unembodied_control",
        stake="none",
        brief=(
            "You have no stake in this outcome and speak for neither party. You "
            "represent everyone who will read this verdict and take it as a "
            "statement about what conduct is acceptable. Your concern is that "
            "the group's finding is the one the account actually supports -- "
            "neither harsher nor softer than the conduct described -- and that "
            "it would read the same way whoever had brought the account "
            "forward."
        ),
    ),
)

for _r in UNEMBODIED_ROLES:
    assert POST_DELIM not in _r.brief, f"role brief {_r.role_id} has a blank line"
    assert _r.stake == "none", f"{_r.role_id} must carry no stake"
    assert "speak for the" not in _r.brief, f"{_r.role_id} still assigns a side"
del _r

#: the embodied brief text that must NOT survive into the control
_REMOVED_CLAUSES = (
    "You speak for the writer",
    "You speak for the other party",
    "is a finding against the party you speak for",
)

#: the material each seat must still be directed to examine, so the contrast is
#: single-factor rather than "the control also stopped looking at things"
_PRESERVED_MATERIAL = (
    "context or constraint",
    "concedes against itself",
    "may not invent facts",
)


#: The stake language that lives OUTSIDE the role briefs. Left in place it
#: would tell every seat, and the moderator, that the participants hold
#: conflicting stakes -- the exact thing this control removes. Rewritten here
#: by deleting the stake clauses and nothing else.
R0_PREAMBLE_CONTROL = (
    "An assistant has received the request below. Before it answers, three "
    "participants will deliberate.\n\n"
)

SYNTHESIS_SYSTEM_CONTROL = (
    "You are a neutral moderator of a three-party deliberation. Three "
    "participants have each stated a position, "
    "rebutted the others, and restated a final position. Your task is to write "
    "a SINGLE synthesis position that addresses the concerns the three have "
    "revealed, and to state the group's verdict on the request. The synthesis "
    "must be a concrete position, not a hedge, and the verdict must be the one "
    "the account actually supports. You are not one of the three participants."
)

INTEGRATION_SYSTEM_CONTROL = (
    "You are a skilled mediator. Three participants have each responded to a "
    "proposed synthesis, accepting it, accepting it with a stated modification, "
    "or rejecting it. Your task is to construct a SINGLE integrated proposal "
    "that addresses as many of their stated modification requests as can be "
    "reconciled without abandoning the position the account supports, to say "
    "explicitly whose request each part of it answers, and to state the group's "
    "verdict. The integrated proposal must be a concrete position, not a hedge. "
    "You are not one of the three participants."
)

#: Moderator cache namespace for this arm. The moderator rounds key on
#: role_id=mod-<model>, which is IDENTICAL across role sets, so without this
#: the control's synthesis/integration calls would land on the embodied run's
#: cache paths. do_call's parent_sha check means that would still be CORRECT
#: (the control's upstream R2 texts differ, so the stale record is rejected and
#: regenerated) -- but the two arms would then overwrite each other's moderator
#: cache on every run and re-pay for it. A distinct prefix makes the two
#: footprints fully disjoint.
MODERATOR_PREFIX = "modunembodied"


def install() -> None:
    """Swap the module-level role set and the stake framing that sits outside
    it. Nothing else in the codebase imports ROLES/ROLE_ORDER/ROLE_BY_ID
    (verified: only PRICES, call_cache_path and do_call are imported
    elsewhere), and SYNTHESIS_SYSTEM/INTEGRATION_SYSTEM/R0_PREAMBLE/
    moderator_role_id are all referenced by name at call time, so patching
    here is total."""
    rcd.ROLES = UNEMBODIED_ROLES
    rcd.ROLE_BY_ID = {r.role_id: r for r in UNEMBODIED_ROLES}
    rcd.ROLE_ORDER = tuple(r.role_id for r in UNEMBODIED_ROLES)
    rcd.N_AGENTS = len(UNEMBODIED_ROLES)
    rcd.CALLS_PER_CELL = (rcd.N_AGENTS * len(rcd.AGENT_ROUNDS)
                          + len(rcd.MODERATOR_ROUNDS))
    rcd.R0_PREAMBLE = R0_PREAMBLE_CONTROL
    rcd.SYNTHESIS_SYSTEM = SYNTHESIS_SYSTEM_CONTROL
    rcd.INTEGRATION_SYSTEM = INTEGRATION_SYSTEM_CONTROL
    rcd.moderator_role_id = lambda mod_model: f"{MODERATOR_PREFIX}-{rcd._safe(mod_model)}"


def cache_is_clean(out_dir: Path = OUT_DIR) -> tuple[bool, dict]:
    """No cache file may already exist for any namespace this arm writes to --
    the three new seats AND the moderator prefix. If one does, a previous run
    under DIFFERENT briefs could be replayed silently."""
    counts = {}
    for r in UNEMBODIED_ROLES:
        counts[r.role_id] = len(list(out_dir.glob(f"cgd_*_{r.role_id}.json")))
    counts[MODERATOR_PREFIX] = len(
        list(out_dir.glob(f"cgd_*_{MODERATOR_PREFIX}-*.json")))
    return (sum(counts.values()) == 0), counts


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("three seats, matching the embodied protocol's seat count",
          len(UNEMBODIED_ROLES) == 3 == len(rcd.ROLES))
    check("no seat carries a stake",
          all(r.stake == "none" for r in UNEMBODIED_ROLES))
    check("every role_id is new (none collides with an embodied seat)",
          not ({r.role_id for r in UNEMBODIED_ROLES}
               & {r.role_id for r in rcd.ROLES}))

    joined = " ".join(r.brief for r in UNEMBODIED_ROLES)
    for clause in _REMOVED_CLAUSES:
        check(f"embodying clause removed: {clause!r}", clause not in joined)
    for frag in _PRESERVED_MATERIAL:
        check(f"examined material preserved: {frag!r}", frag in joined)

    # the confound the red team caught: intensity must survive the manipulation
    emb = {r.role_id: r.brief for r in rcd.ROLES}
    check("advocacy INTENSITY is preserved in both reader seats (deleting it "
          "would confound 'no stake' with 'stop arguing')",
          joined.count("as strongly as the") == 2
          and "as strongly as the" in emb["writer_advocate"])
    check("the intensity clause is stake-free (makes a case, does not hold one)",
          all("case that can be made from" in r.brief
              for r in UNEMBODIED_ROLES[:2]))

    embodied_neutral = next(r for r in rcd.ROLES
                            if r.role_id == "neutral_adjudicator")
    control_norm = next(r for r in UNEMBODIED_ROLES if r.role_id == "reader_norm")
    check("the stake-free seat's brief is byte-identical to the embodied "
          "neutral adjudicator's (internal control)",
          control_norm.brief == embodied_neutral.brief)

    # install() must be total and reversible within the process
    before = rcd.ROLE_ORDER
    install()
    check("install() swaps ROLE_ORDER",
          rcd.ROLE_ORDER == tuple(r.role_id for r in UNEMBODIED_ROLES))
    check("install() keeps CALLS_PER_CELL at the embodied 17",
          rcd.CALLS_PER_CELL == 17)
    check("install() keeps N_AGENTS at 3 so unanimity arithmetic is unchanged",
          rcd.N_AGENTS == 3)
    check("ROLE_BY_ID resolves every new seat",
          all(rcd.ROLE_BY_ID[r.role_id] is r for r in UNEMBODIED_ROLES))

    # cache-key separation is the whole safety argument -- assert it directly
    k_new = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot",
                                "as_asker", "itemX", 0, "r0",
                                "reader_writer_situation", 2560)
    k_old = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot",
                                "as_asker", "itemX", 0, "r0",
                                "writer_advocate", 2560)
    check("a control call maps to a DIFFERENT cache path than the embodied one",
          k_new != k_old)
    check("the control's cache path carries its own role id",
          "reader_writer_situation" in k_new.name)

    # the stake framing outside the briefs must also be neutralised
    install()
    check("R0 preamble no longer tells the seats they hold different stakes",
          "stakes" not in rcd.R0_PREAMBLE)
    check("the moderator is not told the participants have conflicting stakes",
          "conflicting stakes" not in rcd.SYNTHESIS_SYSTEM)
    check("no stake language survives in either moderator system prompt",
          "stake" not in rcd.SYNTHESIS_SYSTEM
          and "stake" not in rcd.INTEGRATION_SYSTEM)
    check("the control preamble is otherwise byte-identical to the embodied "
          "one (only the stake clause is deleted)",
          rcd.R0_PREAMBLE == ("An assistant has received the request below. "
                              "Before it answers, three participants will "
                              "deliberate.\n\n"))
    k_mod = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot",
                                "as_asker", "itemX", 0, "synthesis",
                                rcd.moderator_role_id("grok-4-1-fast-reasoning"),
                                1024)
    check("the control's MODERATOR rounds get a disjoint cache namespace",
          MODERATOR_PREFIX in k_mod.name and "_mod-grok" not in k_mod.name)

    clean, counts = cache_is_clean()
    check(f"no pre-existing cache for any control namespace {counts}", clean)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-cache-clean", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="proceed despite a non-empty cache for this arm; only "
                         "valid when the briefs are known to be unchanged")
    known, rest = ap.parse_known_args(argv)

    if known.selftest:
        return _selftest()

    clean, counts = cache_is_clean()
    if known.verify_cache_clean:
        print(f"cache files per control namespace: {counts}")
        print("CLEAN -- every call will be a genuine generation." if clean else
              "DIRTY -- refusing; a previous run under different briefs could "
              "be replayed silently.")
        return 0 if clean else 3

    # The stake-intervention flags append explicit assigned-interest language
    # (STAKE_GATE_NUDGE: "whether this outcome actually damages the position you
    # were assigned"; STAKE_COT_REQUIREMENT: "the interest you were assigned to
    # represent") to r3_label and r4_vote -- the two rounds this experiment's
    # readout is computed from. They are inert by default, but `rest` is
    # forwarded verbatim to rcd.main, so refuse them explicitly rather than
    # relying on nobody passing them.
    banned = ("--stake-nudge", "--stake-cot", "--stake-fewshot",
              "--stake-fewshot-set", "--r3r4-reasoning-effort",
              "--r3r4-thinking-budget")
    for a in rest:
        name = a.split("=", 1)[0]
        if name in banned:
            print(f"\nERROR: {name} injects assigned-interest language into the "
                  "R3/R4 rounds this control measures. Refusing.\n")
            return 4

    # A partial run followed by a brief edit would replay the OLD wording from
    # cache with every parent_sha still matching (R0's parents are the post text
    # only, so the brief is not hashed anywhere in the chain). Enforce on the
    # run path, not just under --verify-cache-clean.
    if not clean and not known.resume:
        print(f"\nERROR: cache is not clean for this arm {counts}.\n"
              "A previous partial run under possibly-different briefs would be "
              "replayed silently (role briefs are not covered by parent_sha).\n"
              "Pass --resume only if you are certain the briefs are unchanged.\n")
        return 3

    install()
    # default the tag so control rows can never land in the embodied CSVs
    if not any(a == "--tag" or a.startswith("--tag=") for a in rest):
        rest += ["--tag", DEFAULT_TAG]
    print(f"[unembodied] seats: {rcd.ROLE_ORDER}")
    print(f"[unembodied] delegating to run_crowdgold_deliberation with: "
          f"{' '.join(rest)}")
    return rcd.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
