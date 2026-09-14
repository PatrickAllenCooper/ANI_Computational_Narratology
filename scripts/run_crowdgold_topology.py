"""
scripts/run_crowdgold_topology.py -- Addendum 16.10: the role x connectivity
2x2 on the crowd-gold AITA deliberation. The graph-theory thrust's first
factorial.

THE TWO INDEPENDENT VARIABLES

  ROLES      embodied   = the three registered seats verbatim (two opposed
                          advocates + one stakeless neutral)
             identical  = three seats all carrying the neutral_adjudicator
                          brief byte-for-byte (Addendum 16.9), so role-lock is
                          ~0 by construction
  EDGES      on         = seats read each other's r0 at r1 and r1 at r2 (the
                          registered protocol)
             off        = no seat ever reads another seat; r1 and r2 become
                          "develop your own statement" / "state your final
                          position" over the seat's own prior text only. The
                          moderator still synthesises over all three r2s and
                          r3/r4 still read the moderator's text, so the
                          call structure (17 per debate) and the readout are
                          unchanged; only the peer-to-peer edges are cut.

  cell (embodied, on)  EXISTS -- cg_deliberation (1,680 debates). This module
                       refuses to run it.
  cell (embodied, off) tag cg_deliberation_noedge
  cell (identical, on) tag cg_deliberation_identical      [= Addendum 16.9]
  cell (identical, off) tag cg_deliberation_identical_noedge

WHAT EACH CELL ANSWERS

  (embodied, off) vs (embodied, on): do the peer edges do anything? If the
      dissent structure (localisation excess, fire rate, G3 grip) survives
      with no seat reading any other, then the "graph" contributes nothing
      and the structure is a property of role assignment on independent
      nodes. If it collapses, the edges are load-bearing.
  (identical, on) vs (embodied, on): does directional role assignment
      produce the structure? [16.9]
  (identical, off): the floor -- three independent samples of one
      stakeless brief, no roles, no graph. Pure self-consistency of the
      neutral seat, run through the same moderator.

CACHE SAFETY

call_cache_path keys on role_id but not on topology or brief text, and r1's
parent_sha covers all three r0 texts whether or not the seat sees them --
so a no-edge run under the embodied role ids would silently REPLAY the
embodied r1/r2. Every cell here therefore uses its own role ids and its own
moderator prefix, verified empty before any call, and the clean-cache check
is enforced on the run path (not just under a flag), per the 16.8 fix.

DISCLOSED DIFFERENCES IN THE NO-EDGE ROUNDS

There is no coherent way to ask a seat to "rebut" statements it cannot see,
so the no-edge r1/r2 replace "Write your rebuttal. Challenge what you believe
is wrong or incomplete, acknowledge what genuinely lands" with "Develop your
opening statement. Add whatever the account supports that you have not yet
said." The defeasibility clause -- "change your own position only if you have
actually been persuaded by something in the account" -- is kept verbatim, as
is the verdict instruction. The disclosed manipulation is therefore
"edges cut, and the rebuttal task replaced by its natural single-seat
analogue"; it is not a pure edge deletion, because a pure one is incoherent.

Usage
-----
  python -m scripts.run_crowdgold_topology --selftest
  python -m scripts.run_crowdgold_topology --roles embodied  --edges off --verify-cache-clean
  python -m scripts.run_crowdgold_topology --roles embodied  --edges off --dry-run --n-yta 99 --n-nta 150 --samples 1
  python -m scripts.run_crowdgold_topology --roles embodied  --edges off --n-yta 99 --n-nta 150 --samples 1   # SPENDS
  python -m scripts.run_crowdgold_topology --roles identical --edges on  ...
  python -m scripts.run_crowdgold_topology --roles identical --edges off ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scripts.run_crowdgold_deliberation as rcd
from scripts.run_crowdgold_deliberation import POST_DELIM, Role
from scripts.run_crowdgold_unembodied import (
    INTEGRATION_SYSTEM_CONTROL, R0_PREAMBLE_CONTROL, SYNTHESIS_SYSTEM_CONTROL,
)
from scripts.run_phase1_quartet import OUT_DIR

CELLS = {
    ("embodied", "off"):  {"suffix": "_noedge",       "mod": "modnoedge",
                           "tag": "cg_deliberation_noedge"},
    ("identical", "on"):  {"suffix": "",              "mod": "modidentical",
                           "tag": "cg_deliberation_identical"},
    ("identical", "off"): {"suffix": "_noedge",       "mod": "modidentnoedge",
                           "tag": "cg_deliberation_identical_noedge"},
}

#: Captured ONCE at import, before any install() mutates rcd.ROLES. build_roles
#: must read these, never rcd.ROLES, or a second call double-suffixes the ids
#: (caught by selftest: writer_advocate_noedge_noedge).
_EMBODIED: tuple[Role, ...] = rcd.ROLES
_NEUTRAL = next(r for r in _EMBODIED if r.role_id == "neutral_adjudicator")


def build_roles(roles: str, edges: str) -> tuple[Role, ...]:
    suffix = CELLS[(roles, edges)]["suffix"]
    if roles == "embodied":
        # the three registered seats, briefs and stakes verbatim, ids suffixed
        return tuple(Role(role_id=r.role_id + suffix, label=r.label,
                          paper_role=r.paper_role, stake=r.stake, brief=r.brief)
                     for r in _EMBODIED)
    # identical: the neutral brief three times, distinguished only by label
    return tuple(Role(role_id=f"reader_{g}{suffix}", label=f"Reader {g.capitalize()}",
                      paper_role="identical_control", stake="none",
                      brief=_NEUTRAL.brief)
                 for g in ("alpha", "beta", "gamma"))


def r1_user_noedge(arm, item, role, own_r0, r0_texts, *, allow_unresolved, cap):
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{rcd._role_header(role)}\n\n"
        f"--- YOUR OPENING STATEMENT ---\n{rcd._cap(own_r0, cap)}\n--- END ---\n\n"
        "Develop your opening statement. Add whatever the account supports "
        "that you have not yet said, and change your own position only if you "
        "have actually been persuaded by something in the account.\n\n"
        f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def r2_user_noedge(arm, item, role, own_r0, own_r1, r1_texts, *, allow_unresolved, cap):
    return (
        f"{rcd.request_block(arm, item.post_text)}\n\n"
        f"{rcd.GLOSSARY_CORE}\n\n"
        f"{rcd._role_header(role)}\n\n"
        f"--- YOUR OPENING STATEMENT ---\n{rcd._cap(own_r0, cap)}\n--- END ---\n\n"
        f"--- YOUR DEVELOPED STATEMENT ---\n{rcd._cap(own_r1, cap)}\n--- END ---\n\n"
        "State your final position. Say plainly what the group's finding "
        "should be and what your position rests on.\n\n"
        f"{rcd.verdict_instruction(rcd.INSTRUMENT, allow_unresolved=allow_unresolved)}"
    )


def install(roles: str, edges: str) -> None:
    if (roles, edges) not in CELLS:
        raise SystemExit(f"cell ({roles},{edges}) is the existing embodied run; "
                         "refusing to regenerate it")
    R = build_roles(roles, edges)
    rcd.ROLES = R
    rcd.ROLE_BY_ID = {r.role_id: r for r in R}
    rcd.ROLE_ORDER = tuple(r.role_id for r in R)
    rcd.N_AGENTS = len(R)
    rcd.CALLS_PER_CELL = rcd.N_AGENTS * len(rcd.AGENT_ROUNDS) + len(rcd.MODERATOR_ROUNDS)
    mod = CELLS[(roles, edges)]["mod"]
    rcd.moderator_role_id = lambda m, _p=mod: f"{_p}-{rcd._safe(m)}"
    if roles == "identical":
        rcd.R0_PREAMBLE = R0_PREAMBLE_CONTROL
        rcd.SYNTHESIS_SYSTEM = SYNTHESIS_SYSTEM_CONTROL
        rcd.INTEGRATION_SYSTEM = INTEGRATION_SYSTEM_CONTROL
    if edges == "off":
        rcd.r1_user = r1_user_noedge
        rcd.r2_user = r2_user_noedge


def cache_is_clean(roles: str, edges: str, out_dir: Path = OUT_DIR):
    counts = {}
    for r in build_roles(roles, edges):
        counts[r.role_id] = len(list(out_dir.glob(f"cgd_*_{r.role_id}.json")))
    mod = CELLS[(roles, edges)]["mod"]
    counts[mod] = len(list(out_dir.glob(f"cgd_*_{mod}-*.json")))
    return sum(counts.values()) == 0, counts


def _selftest() -> int:
    fails = []
    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond: fails.append(name)

    emb_ids = {r.role_id for r in _EMBODIED}
    all_ids = set()
    for cell in CELLS:
        R = build_roles(*cell)
        ids = {r.role_id for r in R}
        check(f"{cell}: three seats", len(R) == 3)
        check(f"{cell}: no role id collides with the embodied run", not (ids & emb_ids))
        check(f"{cell}: no role id collides with any other cell", not (ids & all_ids))
        all_ids |= ids
        for r in R:
            assert POST_DELIM not in r.brief
    check("the three cells' moderator prefixes are distinct from each other and "
          "from the embodied 'mod-'",
          len({c["mod"] for c in CELLS.values()}) == 3
          and all(not c["mod"].startswith("mod-") for c in CELLS.values()))

    e_off = build_roles("embodied", "off")
    check("(embodied,off) keeps the embodied briefs and stakes byte-identical",
          all(a.brief == b.brief and a.stake == b.stake for a, b in zip(e_off, _EMBODIED)))
    check("(embodied,off) role ids are exactly the embodied ids + one suffix "
          "(regression: build_roles must not read the mutated rcd.ROLES)",
          [r.role_id for r in e_off] == ["writer_advocate_noedge", "counterparty_noedge",
                                          "neutral_adjudicator_noedge"])
    check("build_roles is idempotent across repeated install() calls",
          [r.role_id for r in build_roles("embodied", "off")]
          == [r.role_id for r in build_roles("embodied", "off")])
    i_on = build_roles("identical", "on")
    check("(identical,on) gives every seat the neutral brief verbatim",
          all(r.brief == _NEUTRAL.brief and r.stake == "none" for r in i_on))

    # no-edge rounds must not reference other participants
    item = rcd.CrowdGoldItem(item_id="X", post_text="[POST]", gold_verdict="YTA")
    role = e_off[0]
    p1 = r1_user_noedge("third_person", item, role, "[R0]", {}, allow_unresolved=True, cap=0)
    p2 = r2_user_noedge("third_person", item, role, "[R0]", "[R1]", {}, allow_unresolved=True, cap=0)
    check("no-edge r1 never mentions the other participants",
          "other two participants" not in p1 and "rebut" not in p1.lower())
    check("no-edge r2 never mentions the other participants",
          "other two participants" not in p2 and "rebut" not in p2.lower())
    check("no-edge r1 keeps the defeasibility clause verbatim",
          "change your own position only if you have actually been persuaded "
          "by something in the account" in p1)
    check("no-edge rounds keep the verdict instruction",
          "VERDICT:" in p1 and "VERDICT:" in p2)

    # install() is total and reversible-per-process; cache keys separate
    for cell in CELLS:
        install(*cell)
        k = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot", "as_asker",
                                "itemX", 0, "r1", rcd.ROLE_ORDER[0], 2560)
        km = rcd.call_cache_path("grok-4-1-fast-reasoning", "narrative_cot", "as_asker",
                                 "itemX", 0, "synthesis",
                                 rcd.moderator_role_id("grok-4-1-fast-reasoning"), 1024)
        check(f"{cell}: agent cache path carries the cell's own role id",
              rcd.ROLE_ORDER[0] in k.name and "_writer_advocate.json" not in k.name)
        check(f"{cell}: moderator cache path carries the cell's own prefix",
              CELLS[cell]["mod"] in km.name and "_mod-grok" not in km.name)
        check(f"{cell}: CALLS_PER_CELL stays 17", rcd.CALLS_PER_CELL == 17)
        if cell[1] == "off":
            check(f"{cell}: r1_user is the no-edge version", rcd.r1_user is r1_user_noedge)
        if cell[0] == "identical":
            check(f"{cell}: stake framing removed from the preamble",
                  "stakes" not in rcd.R0_PREAMBLE)
        clean, counts = cache_is_clean(*cell)
        # Addendum 16.10 has been RUN (2026-09-12): the namespaces hold the
        # registered footprint (2,100 per seat, 840 moderator = 420 debates).
        # The standing check is that no namespace is PARTIAL from elsewhere:
        # either all empty (unrun) or every namespace populated (the run).
        check(f"{cell}: cache namespaces are empty or hold one complete run {counts}",
              clean or all(v > 0 for v in counts.values()))

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--roles", choices=("embodied", "identical"))
    ap.add_argument("--edges", choices=("on", "off"))
    ap.add_argument("--verify-cache-clean", action="store_true")
    ap.add_argument("--resume", action="store_true")
    known, rest = ap.parse_known_args(argv)
    if known.selftest:
        return _selftest()
    if not known.roles or not known.edges:
        print("ERROR: --roles and --edges are required"); return 2
    cell = (known.roles, known.edges)
    if cell not in CELLS:
        print(f"ERROR: cell {cell} is the existing embodied run (cg_deliberation); "
              "refusing to regenerate it"); return 2

    clean, counts = cache_is_clean(*cell)
    if known.verify_cache_clean:
        print(f"cache files per namespace for {cell}: {counts}")
        print("CLEAN" if clean else "DIRTY -- refusing")
        return 0 if clean else 3

    banned = ("--stake-nudge", "--stake-cot", "--stake-fewshot", "--stake-fewshot-set",
              "--r3r4-reasoning-effort", "--r3r4-thinking-budget")
    for a in rest:
        if a.split("=", 1)[0] in banned:
            print(f"ERROR: {a} injects assigned-interest language into the readout rounds"); return 4
    if not clean and not known.resume:
        print(f"ERROR: cache not clean for {cell} {counts}; pass --resume only if the "
              "briefs are known unchanged"); return 3

    install(*cell)
    if not any(a == "--tag" or a.startswith("--tag=") for a in rest):
        rest += ["--tag", CELLS[cell]["tag"]]
    print(f"[topology] cell roles={known.roles} edges={known.edges} seats={rcd.ROLE_ORDER}")
    return rcd.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
