"""
scripts/mine_stake_fewshot_exemplars.py -- Addendum 8, rung 1: mine two real,
symmetric stake-gated exemplar pairs from grok's own cached R3/R4 transcripts
and freeze them into a checked-in artifact.

Each pair is drawn from ONE debate (same synthesis, same proposal) where the
two stake-bearing seats land on opposite sides because their ASSIGNED
INTEREST differs, not because the content differs: one seat's stake was
undermined by the outcome and it rejected; the other's was not and it
accepted. That is exactly the pattern Addendum 7's soft nudge tried to
INSTRUCT and failed to move -- this rung tries to DEMONSTRATE it instead
(in-context distillation of grok's own disposition into a weaker model,
zero training).

Selection is deterministic and disclosed: item/arm/sample_idx are hard-coded
below, found by grep over `cg_deliberation_votes.csv` (grok's registered
Addendum-6 baseline) for the two seat/stake/vote combinations, first hit by
item_id sort order. No cherry-picking on OUTPUT QUALITY beyond "the pattern
holds and the calls are cached" -- this script prints the full candidate
pool sizes so that claim is auditable.

Usage
-----
  python -m scripts.mine_stake_fewshot_exemplars --selftest
  python -m scripts.mine_stake_fewshot_exemplars
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from scripts.analyze_crowdgold_sdt import code_response
from scripts.run_phase1_quartet import OUT_DIR

VOTES_PATH = OUT_DIR / "cg_deliberation_votes.csv"
OUT_PATH = OUT_DIR / "stake_fewshot_exemplars.json"
TRUTHGATED_OUT_PATH = OUT_DIR / "stake_fewshot_truthgated_exemplars.json"
STAKE_SEATS = ("writer_advocate", "counterparty")
MODEL = "grok-4-1-fast-reasoning"
SCAFFOLD = "narrative_cot"

#: (item_id, arm, sample_idx, undermined_role, not_undermined_role) --
#: the first two debates found (by item_id sort order) where the stake
#: seats disagree AND each votes exactly as its own stake predicts.
PAIRS = (
    ("2ijbqwTQ4z0EfCxS0rGF6GqMoNqr0Hdw", "third_person", 1,
     "writer_advocate", "counterparty"),
    ("2zX5IZY4U74RAsvROg9IBm3oUGVYrfQd", "third_person", 0,
     "counterparty", "writer_advocate"),
)
R3_MAX_TOKENS, R4_MAX_TOKENS = 1024, 512


def raw_cache_path(round_name: str, max_tokens: int, arm: str, item_id: str,
                   idx: int, role: str) -> Path:
    return (OUT_DIR / f"cgd_{MODEL}_{SCAFFOLD}_t{max_tokens}_{arm}_{item_id}_"
                      f"{idx:02d}_{round_name}_{role}.json")


def read_output(round_name: str, max_tokens: int, arm: str, item_id: str,
                idx: int, role: str) -> str:
    p = raw_cache_path(round_name, max_tokens, arm, item_id, idx, role)
    return json.loads(p.read_text())["output"].strip()


def verify_pool(votes_path: Path = VOTES_PATH) -> dict:
    """Count how many debates in the pool show the clean pattern each pair
    exemplifies, so the "first two found" selection is auditable, not
    cherry-picked from a near-empty pool.
    """
    with open(votes_path) as f:
        rows = list(csv.DictReader(f))
    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key.setdefault((r["item_id"], r["arm"], r["sample_idx"]), {})[
            r["role_id"]] = r
    clean = 0
    for grp in by_key.values():
        if not all(s in grp for s in STAKE_SEATS):
            continue
        wa, cp = grp["writer_advocate"], grp["counterparty"]
        for u, n in ((wa, cp), (cp, wa)):
            if (u["stake_undermined"] == "1" and u["reject"] == "1"
                    and n["stake_undermined"] == "0" and n["reject"] == "0"):
                clean += 1
    return {"n_debates_with_both_stake_seats": len(by_key), "n_clean_pairs": clean}


def build_exemplars() -> list[dict]:
    out = []
    for item_id, arm, idx, und_role, notund_role in PAIRS:
        r3_und = read_output("r3_label", R3_MAX_TOKENS, arm, item_id, idx, und_role)
        r3_not = read_output("r3_label", R3_MAX_TOKENS, arm, item_id, idx, notund_role)
        r4_und = read_output("r4_vote", R4_MAX_TOKENS, arm, item_id, idx, und_role)
        r4_not = read_output("r4_vote", R4_MAX_TOKENS, arm, item_id, idx, notund_role)
        out.append({
            "item_id": item_id, "arm": arm, "sample_idx": idx,
            "undermined_role": und_role, "not_undermined_role": notund_role,
            "source_model": MODEL,
            "r3_label_undermined": r3_und, "r3_label_not_undermined": r3_not,
            "r4_vote_undermined": r4_und, "r4_vote_not_undermined": r4_not,
        })
    return out


#: (item_id, arm, sample_idx, role) for the first debate found (by item_id
#: sort order) of each of the two TRUTH-GATED patterns (Addendum 11 Phase 1,
#: 1a): the undermined seat's dissent tracks whether the outcome was
#: actually right, not just whether its own stake was hurt. This is a
#: DIFFERENT lesson from the plain pairs above (stake alone predicts the
#: vote): here, stake predicts the vote ONLY when combined with which side
#: gold actually favours.
TRUTHGATED_REJECT_CORRECT = (
    "0hqzpR3kznERjnpcZt7dPaVnujNBYLqd", "as_asker", 1, "counterparty")
TRUTHGATED_ACCEPT_CORRECT = (
    "0QhNF8tvIJOj1UEwwofC3B67eLqta8Xj", "as_asker", 0, "counterparty")


def _correct(verdict: str, gold: str) -> bool | None:
    c = code_response(verdict, "published")
    return None if c is None else c == (1 if gold == "YTA" else 0)


def verify_truthgated_pool(votes_path: Path = VOTES_PATH) -> dict:
    """How many debates show each truth-gated pattern, so the two hard-coded
    picks above are auditable as "first of a real population", not
    cherry-picked from a near-empty set."""
    with open(votes_path) as f:
        rows = list(csv.DictReader(f))
    n_reject_correct = n_accept_correct = 0
    for r in rows:
        if r["role_id"] not in STAKE_SEATS or r.get("stake_undermined") != "1":
            continue
        syn_ok = _correct(r["synthesis_verdict"], r["gold_verdict"])
        if syn_ok is None:
            continue
        if r["reject"] == "1" and not syn_ok:
            n_reject_correct += 1
        if r["reject"] == "0" and syn_ok:
            n_accept_correct += 1
    return {"n_reject_correct": n_reject_correct, "n_accept_correct": n_accept_correct}


def build_truthgated_exemplars() -> dict:
    item_id, arm, idx, role = TRUTHGATED_REJECT_CORRECT
    r3_rc = read_output("r3_label", R3_MAX_TOKENS, arm, item_id, idx, role)
    r4_rc = read_output("r4_vote", R4_MAX_TOKENS, arm, item_id, idx, role)
    item_id2, arm2, idx2, role2 = TRUTHGATED_ACCEPT_CORRECT
    r3_ac = read_output("r3_label", R3_MAX_TOKENS, arm2, item_id2, idx2, role2)
    r4_ac = read_output("r4_vote", R4_MAX_TOKENS, arm2, item_id2, idx2, role2)
    return {
        "reject_correct": {
            "item_id": item_id, "arm": arm, "sample_idx": idx, "role": role,
            "r3_label": r3_rc, "r4_vote": r4_rc,
        },
        "accept_correct": {
            "item_id": item_id2, "arm": arm2, "sample_idx": idx2, "role": role2,
            "r3_label": r3_ac, "r4_vote": r4_ac,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--truthgated-out", type=Path, default=TRUTHGATED_OUT_PATH)
    ap.add_argument("--truthgated", action="store_true",
                    help="also mine the Addendum 11 Phase 1 truth-gated pair")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    pool = verify_pool()
    print(f"pool: {pool['n_debates_with_both_stake_seats']} debates with both "
          f"stake seats, {pool['n_clean_pairs']} show the clean pattern "
          f"(reject iff undermined, on BOTH seats) -- picking the first 2")
    exemplars = build_exemplars()
    for e in exemplars:
        print(f"  {e['item_id'][:12]}.../{e['arm']}/{e['sample_idx']}: "
              f"{e['undermined_role']} (undermined) REJECTs, "
              f"{e['not_undermined_role']} (not) ACCEPTs")
    a.out.write_text(json.dumps({"pool": pool, "exemplars": exemplars}, indent=1))
    print(f"wrote {a.out}")

    if a.truthgated:
        tpool = verify_truthgated_pool()
        print(f"\ntruth-gated pool: {tpool['n_reject_correct']} undermined+"
              f"REJECT+synthesis-was-wrong, {tpool['n_accept_correct']} "
              f"undermined+ACCEPT+synthesis-was-right -- picking the first "
              f"of each")
        texemplars = build_truthgated_exemplars()
        a.truthgated_out.write_text(
            json.dumps({"pool": tpool, "exemplars": texemplars}, indent=1))
        print(f"wrote {a.truthgated_out}")
    return 0


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("VOTES_PATH exists (grok's Addendum-6 baseline is committed)",
          VOTES_PATH.exists())
    if VOTES_PATH.exists():
        pool = verify_pool()
        check("pool has a nontrivial number of clean stake-gated pairs "
              "(selection is not cherry-picked from a near-empty set)",
              pool["n_clean_pairs"] >= 10)
        try:
            exemplars = build_exemplars()
            check("both pairs built, 4 texts each", len(exemplars) == 2
                  and all(len(e["r4_vote_undermined"]) > 0
                          and len(e["r4_vote_not_undermined"]) > 0
                          for e in exemplars))
            check("the undermined side's R4 vote is a REJECT and the "
                  "not-undermined side's is an ACCEPT, verbatim from cache",
                  all("REJECT" in e["r4_vote_undermined"]
                      and "ACCEPT" in e["r4_vote_not_undermined"]
                      for e in exemplars))
            check("the two pairs use different undermined seats (one "
                  "writer_advocate, one counterparty), so the demonstration "
                  "is not seat-specific",
                  {e["undermined_role"] for e in exemplars} == set(STAKE_SEATS))
        except FileNotFoundError as e:
            check(f"raw cache files present for the hard-coded pairs ({e})",
                  False)

        tpool = verify_truthgated_pool()
        check("truth-gated pool has a nontrivial number of each pattern "
              "(not cherry-picked from a near-empty set)",
              tpool["n_reject_correct"] >= 10 and tpool["n_accept_correct"] >= 10)
        try:
            tex = build_truthgated_exemplars()
            check("truth-gated exemplars built, both have r3/r4 text",
                  all(len(tex[k]["r3_label"]) > 0 and len(tex[k]["r4_vote"]) > 0
                      for k in ("reject_correct", "accept_correct")))
            check("reject_correct exemplar's R4 vote is a REJECT, verbatim",
                  "REJECT" in tex["reject_correct"]["r4_vote"])
            check("accept_correct exemplar's R4 vote is an ACCEPT, verbatim",
                  "ACCEPT" in tex["accept_correct"]["r4_vote"]
                  and "REJECT" not in tex["accept_correct"]["r4_vote"])
        except FileNotFoundError as e:
            check(f"raw cache files present for the truth-gated pair ({e})",
                  False)
    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
