"""
scripts/run_loop_iteration.py -- L2: the gated SECOND loop iteration on the
Crowd-Gold deliberation protocol.

Registered in Guidance_Documents/prereg_embodiment_community.md (Addendum 3)
BEFORE any generation. Phase-0 background in scripts/analyze_loop_step.py.

WHAT THIS DOES
--------------
1. REPLAY. Every debate in the rows CSV is re-executed through
   run_crowdgold_deliberation.run_deliberation with generate_any replaced by
   a function that RAISES. Every call must therefore come out of cache; the
   replay provably spends nothing and yields the iteration-1 artifacts
   (R2 positions, synthesis S1, R3 labels, integrated proposal S2).

2. GATE. The composite flag is the frozen Phase-0 sensor, reused directly
   from analyze_loop_step (neutral adjudicator objected at R3, OR a
   stake-bearing seat objected although S1 did not undermine its stake).
   Only flagged debates proceed; everything else keeps S2.

3. ITERATE. For each flagged debate, one further label->integrate->vote
   cycle with the byte-identical iteration-1 templates:
     R3b     each agent labels S2 (r3_label_user, shared across sub-arms)
     integ2  moderator builds a THIRD proposal S3 (integration_user), under
             three sub-arms: blind (no note), generic (flag with no reason),
             informed (flag with the observable reason; never gold)
     r4b     binary vote on S3 (r4_vote_user)
   15 calls per flagged debate: 3 + 3 x (1 + 3).

The audit note is APPENDED to the integration prompt (after the verdict
instruction). Appending keeps the injection deterministic and byte-auditable;
both noted sub-arms are treated identically so the generic/informed contrast
is clean.

Usage
-----
  python -m scripts.run_loop_iteration --selftest
  python -m scripts.run_loop_iteration --dry-run
  python -m scripts.run_loop_iteration --workers 12
  python -m scripts.run_loop_iteration --analyze-only
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_crowdgold_sdt import code_response
from scripts.analyze_loop_step import flagged, mislocalised, neutral_objected
from scripts.run_phase1_quartet import OUT_DIR
from scripts.verdict_format import extract_verdict

ROWS_IN = OUT_DIR / "cg_deliberation_rows.csv"
ROWS_OUT = OUT_DIR / "loop_iter2_rows.csv"
VOTES_OUT = OUT_DIR / "loop_iter2_votes.csv"
ANALYSIS_OUT = OUT_DIR / "loop_iter2_analysis.json"

SUBARMS = ("blind", "generic", "informed")
OBJECTED = ("ACCEPT_WITH_MODIFICATION", "REJECT")

GENERIC_NOTE = (
    "AUDIT NOTE: An automated review has flagged this deliberation for "
    "re-examination before it is finalised. Re-examine the participants' "
    "responses on their merits before settling the group's verdict."
)


def informed_note(flag_neutral: bool, mislocal_seats: Sequence[str]) -> str:
    """The flag reason, from observable iteration-1 structure only. No gold."""
    reasons = []
    if flag_neutral:
        reasons.append(
            "the Neutral Adjudicator, which has no stake in the outcome, "
            "objected to the synthesis"
        )
    for rid in mislocal_seats:
        reasons.append(
            f"the {rcd.ROLE_BY_ID[rid].label} objected although the synthesis "
            "did not go against the party it speaks for"
        )
    assert reasons, "informed_note called on an unflagged debate"
    return (
        "AUDIT NOTE: An automated review has flagged this deliberation: "
        + "; ".join(reasons)
        + ". Objections that are not explained by a participant's stated "
        "interest are often substantive. Re-examine those objections on "
        "their merits before settling the group's verdict."
    )


# ---------------------------------------------------------------------------
# Replay: iteration 1 from cache, spending nothing
# ---------------------------------------------------------------------------

@contextmanager
def no_spend():
    """Any cache miss during replay raises instead of generating."""
    real = rcd.generate_any

    def raiser(*a, **k):
        raise RuntimeError(
            "cache miss during replay -- iteration-1 artifacts are supposed "
            "to be fully cached; refusing to spend"
        )

    rcd.generate_any = raiser  # type: ignore[assignment]
    try:
        yield
    finally:
        rcd.generate_any = real  # type: ignore[assignment]


def replay_debate(model: str, arm: str, item, idx: int, *,
                  scaffold: str = "narrative_cot") -> dict:
    """Re-run one debate out of cache; return the iteration-1 state L2 needs.

    Caller is responsible for wrapping this in no_spend().
    """
    row, vote_rows, calls = rcd.run_deliberation(model, arm, item, idx,
                                                 scaffold=scaffold)
    by_round: dict[tuple[str, str], str] = {
        (c["round"], c["role_id"]): c.get("output") or "" for c in calls
    }
    r2 = {rid: by_round[("r2", rid)] for rid in rcd.ROLE_ORDER}
    r3 = {rid: by_round[("r3_label", rid)] for rid in rcd.ROLE_ORDER}
    proposal = next(t for (rnd, _), t in by_round.items()
                    if rnd == "integration")
    return {
        "row": row,
        "vote_rows": vote_rows,
        "r2": r2,
        "r3": r3,
        "r3_objections": {rid: rcd.extract_objection(r3[rid])[1]
                          for rid in rcd.ROLE_ORDER},
        "synthesis_verdict": row["synthesis_verdict"],
        "proposal": proposal,
        "s2_verdict": row["verdict"],
    }


def gate_state(state: dict) -> Optional[dict]:
    """Build the analyze_loop_step debate dict from a replayed state.

    Returns None when the debate is INELIGIBLE (synthesis or final verdict
    not codable), mirroring Phase 0's load_debates filter exactly.
    """
    row = state["row"]
    gold = row["gold_verdict"]
    syn_code = code_response(row["synthesis_verdict"], "published")
    fin_code = code_response(row["verdict"], "published")
    if syn_code is None or fin_code is None:
        return None
    d = {
        "item": row["item_id"], "arm": row["arm"],
        "syn_ok": syn_code == (1 if gold == "YTA" else 0),
        "fin_ok": fin_code == (1 if gold == "YTA" else 0),
        "syn_code": syn_code,
    }
    for v in state["vote_rows"]:
        seat = v["role_id"]
        d[f"obj_{seat}"] = bool(v["objected_r3"])
        want = v["role_stake"]
        if want in ("at_fault", "not_at_fault"):
            d[f"undermined_{seat}"] = (syn_code == 1) != (want == "at_fault")
    return d


def mislocal_seats(d: dict) -> list[str]:
    return [s for s in ("writer_advocate", "counterparty")
            if d.get(f"obj_{s}") and not d.get(f"undermined_{s}", True)]


# ---------------------------------------------------------------------------
# Iteration 2
# ---------------------------------------------------------------------------

def run_iteration2(model: str, arm: str, item, idx: int, state: dict,
                   gate_d: dict, *, scaffold: str = "narrative_cot",
                   subarms: Sequence[str] = SUBARMS,
                   max_tokens_moderator: int = 1024,
                   max_tokens_label: int = 1024,
                   max_tokens_vote: int = 512,
                   ) -> tuple[list[dict], list[dict]]:
    """One further label->integrate->vote cycle on a flagged debate.

    Returns (outcome_rows, vote_level_rows): one outcome row per sub-arm.
    """
    system = rcd.agent_system(scaffold)
    r2, proposal = state["r2"], state["proposal"]
    gold = item.gold_verdict
    s2_code = code_response(state["s2_verdict"], "published")
    s2_ok = None if s2_code is None else s2_code == (1 if gold == "YTA" else 0)
    flag_n = neutral_objected(gate_d)
    mseats = mislocal_seats(gate_d)

    # ---- R3b: label S2, byte-identical template, shared across sub-arms ---
    labels_b: dict[str, str] = {}
    objections_b: dict[str, str] = {}
    for r in rcd.ROLES:
        rec = rcd.do_call(
            model=model, scaffold=scaffold, arm=arm, item=item, idx=idx,
            round_name="r3b_label", role_id=r.role_id, system=system,
            user=rcd.r3_label_user(arm, item, r, r2[r.role_id], proposal,
                                   cap=0),
            max_tokens=max_tokens_label,
            parents=(r2[r.role_id], proposal), cap=0,
        )
        out = rec.get("output") or ""
        labels_b[r.role_id] = extract_verdict(out, rcd.LABEL_INSTRUMENT)
        _, objections_b[r.role_id] = rcd.extract_objection(out)

    note_by_subarm = {
        "blind": "",
        "generic": GENERIC_NOTE,
        "informed": informed_note(flag_n, mseats),
    }
    base_integ_user = rcd.integration_user(
        arm, item, proposal, labels_b, objections_b,
        allow_unresolved=True, cap=0)

    out_rows: list[dict] = []
    vote_lvl: list[dict] = []
    for sub in subarms:
        note = note_by_subarm[sub]
        integ_user = base_integ_user + (f"\n\n{note}" if note else "")
        lbl_parents = (proposal,) + tuple(
            f"{labels_b[rid]}|{objections_b[rid]}" for rid in rcd.ROLE_ORDER
        ) + (note,)
        int_rec = rcd.do_call(
            model=model, scaffold=scaffold, arm=arm, item=item, idx=idx,
            round_name=f"integ2_{sub}", role_id=f"mod-{rcd._safe(model)}",
            system=rcd.INTEGRATION_SYSTEM, user=integ_user,
            max_tokens=max_tokens_moderator, parents=lbl_parents, cap=0,
        )
        s3 = int_rec.get("output") or ""
        s3_verdict = extract_verdict(s3, rcd.INSTRUMENT)
        addressed = rcd.extract_addressed(s3)
        s3_code = code_response(s3_verdict, "published")
        s3_ok = (None if s3_code is None
                 else s3_code == (1 if gold == "YTA" else 0))

        votes: dict[str, str] = {}
        for r in rcd.ROLES:
            lab, obj = labels_b[r.role_id], objections_b[r.role_id]
            rec = rcd.do_call(
                model=model, scaffold=scaffold, arm=arm, item=item, idx=idx,
                round_name=f"r4b_{sub}", role_id=r.role_id, system=system,
                user=rcd.r4_vote_user(arm, item, r, proposal, lab, obj, s3,
                                      cap=0),
                max_tokens=max_tokens_vote,
                parents=(proposal, lab, obj, s3, note), cap=0,
            )
            vote = extract_verdict(rec.get("output") or "",
                                   rcd.VOTE_INSTRUMENT)
            votes[r.role_id] = vote
            objected = lab in OBJECTED
            vote_lvl.append({
                "model": model, "arm": arm, "item_id": item.item_id,
                "sample_idx": idx, "subarm": sub, "gold_verdict": gold,
                "role_id": r.role_id,
                "r3b_label": lab,
                "objected_r3b": int(objected),
                "modification_addressed": int(r.role_id in addressed),
                "vote": vote,
                "vote_parsed": int(vote in ("ACCEPT", "REJECT")),
                "revised_r4b": int(objected and vote == "ACCEPT"),
            })

        objectors = [rid for rid in rcd.ROLE_ORDER
                     if labels_b[rid] in OBJECTED]
        out_rows.append({
            "model": model, "arm": arm, "item_id": item.item_id,
            "sample_idx": idx, "subarm": sub, "gold_verdict": gold,
            "flag_neutral": int(flag_n),
            "flag_mislocalised": int(bool(mseats)),
            "s1_verdict": state["synthesis_verdict"],
            "s2_verdict": state["s2_verdict"],
            "s3_verdict": s3_verdict,
            "s2_correct": "" if s2_ok is None else int(s2_ok),
            "s3_correct": "" if s3_ok is None else int(s3_ok),
            "changed": int(s3_verdict != state["s2_verdict"]),
            "r3b_labels": "|".join(labels_b[rid] for rid in rcd.ROLE_ORDER),
            "n_objectors_b": len(objectors),
            "n_modifications_addressed_b": sum(
                1 for rid in objectors if rid in addressed),
            "n_accept_b": sum(1 for v in votes.values() if v == "ACCEPT"),
            "n_reject_b": sum(1 for v in votes.values() if v == "REJECT"),
            "note_len": len(note_by_subarm[sub]),
            "s3_len": len(s3),
            "finish_reason": int_rec.get("finish_reason", ""),
        })
    return out_rows, vote_lvl


ROW_FIELDS = (
    "model", "arm", "item_id", "sample_idx", "subarm", "gold_verdict",
    "flag_neutral", "flag_mislocalised", "s1_verdict", "s2_verdict",
    "s3_verdict", "s2_correct", "s3_correct", "changed", "r3b_labels",
    "n_objectors_b", "n_modifications_addressed_b", "n_accept_b",
    "n_reject_b", "note_len", "s3_len", "finish_reason",
)
VOTE_FIELDS = (
    "model", "arm", "item_id", "sample_idx", "subarm", "gold_verdict",
    "role_id", "r3b_label", "objected_r3b", "modification_addressed",
    "vote", "vote_parsed", "revised_r4b",
)


# ---------------------------------------------------------------------------
# Analysis (registered: L2-PRIMARY, L2-INFO, L2-NUDGE, L2-STABILITY)
# ---------------------------------------------------------------------------

def _boot_items(stat_fn, by_item: dict, *, draws: int, seed: int) -> dict:
    ids = sorted(by_item)
    rng = random.Random(seed)

    def ev(sample_ids):
        return stat_fn([r for i in sample_ids for r in by_item[i]])

    point = ev(ids)
    vals = []
    for _ in range(draws):
        v = ev([ids[rng.randrange(len(ids))] for _ in ids])
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return {"point": point, "lo95": None, "hi95": None}
    return {"point": point,
            "lo95": vals[int(0.025 * len(vals))],
            "hi95": vals[min(int(0.975 * len(vals)), len(vals) - 1)]}


def analyze_iter2(rows: Sequence[dict], votes: Sequence[dict], *,
                  draws: int = 4000, seed: int = 11) -> dict:
    rows = [r for r in rows if r["s2_correct"] != "" and r["s3_correct"] != ""]
    for r in rows:
        r["_s2"] = int(r["s2_correct"])
        r["_s3"] = int(r["s3_correct"])
    res: dict = {"n_rows_codable": len(rows)}

    by_item_sub: dict[str, dict[str, list[dict]]] = {
        s: defaultdict(list) for s in SUBARMS}
    for r in rows:
        by_item_sub[r["subarm"]][r["item_id"]].append(r)

    def net(rs):
        return (sum(r["_s3"] - r["_s2"] for r in rs) / len(rs)
                if rs else None)

    def fix(rs):
        w = [r for r in rs if not r["_s2"]]
        return sum(r["_s3"] for r in w) / len(w) if w else None

    def brk(rs):
        c = [r for r in rs if r["_s2"]]
        return sum(1 - r["_s3"] for r in c) / len(c) if c else None

    for i, sub in enumerate(SUBARMS):
        bi = by_item_sub[sub]
        res[f"{sub}_n"] = sum(len(v) for v in bi.values())
        res[f"{sub}_net"] = _boot_items(net, bi, draws=draws, seed=seed + i)
        res[f"{sub}_fix"] = _boot_items(fix, bi, draws=draws,
                                        seed=seed + 10 + i)
        res[f"{sub}_break"] = _boot_items(brk, bi, draws=draws,
                                          seed=seed + 20 + i)

    # Paired contrasts within debate (same item, sample, arm).
    paired: dict[str, dict[tuple, dict[str, dict]]] = defaultdict(dict)
    for r in rows:
        paired[r["item_id"]].setdefault(
            (r["item_id"], r["sample_idx"], r["arm"]), {})[r["subarm"]] = r

    def paired_gap(sub_a: str, sub_b: str):
        def stat(cells):
            deltas = [
                (c[sub_a]["_s3"] - c[sub_a]["_s2"])
                - (c[sub_b]["_s3"] - c[sub_b]["_s2"])
                for c in cells if sub_a in c and sub_b in c
            ]
            return sum(deltas) / len(deltas) if deltas else None
        by_item = {i: list(cells.values())
                   for i, cells in paired.items()}
        return _boot_items(stat, by_item, draws=draws, seed=seed + 30)

    res["info_gap_informed_minus_blind"] = paired_gap("informed", "blind")
    res["nudge_gap_generic_minus_blind"] = paired_gap("generic", "blind")
    res["content_gap_informed_minus_generic"] = paired_gap(
        "informed", "generic")

    # L2-STABILITY: accept-when-addressed gap among R3b objectors, pooled.
    obj = [v for v in votes if v["objected_r3b"] and v["vote_parsed"]]
    acc_addr = [v["vote"] == "ACCEPT" for v in obj
                if v["modification_addressed"]]
    acc_not = [v["vote"] == "ACCEPT" for v in obj
               if not v["modification_addressed"]]
    res["stability_n_objectors"] = len(obj)
    res["stability_accept_when_addressed"] = (
        sum(acc_addr) / len(acc_addr) if acc_addr else None)
    res["stability_accept_when_not"] = (
        sum(acc_not) / len(acc_not) if acc_not else None)
    return res


def _fmt(d: dict) -> str:
    if d.get("point") is None:
        return "n/a"
    lo = f"{d['lo95']:+.3f}" if d.get("lo95") is not None else "?"
    hi = f"{d['hi95']:+.3f}" if d.get("hi95") is not None else "?"
    return f"{d['point']:+.3f} [{lo}, {hi}]"


def print_analysis(res: dict) -> None:
    print(f"\n=== L2 gated second iteration "
          f"({res['n_rows_codable']} codable rows) ===")
    for sub in SUBARMS:
        print(f"  {sub:<9} n={res[f'{sub}_n']:<4} "
              f"net {_fmt(res[f'{sub}_net'])}   "
              f"fix {_fmt(res[f'{sub}_fix'])}   "
              f"break {_fmt(res[f'{sub}_break'])}")
    print(f"\n  L2-INFO  informed - blind (paired): "
          f"{_fmt(res['info_gap_informed_minus_blind'])}")
    print(f"  L2-NUDGE generic - blind (paired):  "
          f"{_fmt(res['nudge_gap_generic_minus_blind'])}")
    print(f"  content  informed - generic:        "
          f"{_fmt(res['content_gap_informed_minus_generic'])}")
    a, b = (res["stability_accept_when_addressed"],
            res["stability_accept_when_not"])
    print(f"\n  L2-STABILITY accept|addressed "
          f"{'n/a' if a is None else f'{a:.3f}'} vs accept|not "
          f"{'n/a' if b is None else f'{b:.3f}'} "
          f"(n objectors {res['stability_n_objectors']})")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_target_debates(rows_path: Path) -> list[tuple[str, str, str, int]]:
    seen = []
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            seen.append((r["model"], r["arm"], r["item_id"],
                         int(r["sample_idx"])))
    return seen


def write_csv(rows: Sequence[dict], path: Path,
              fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L2 gated second loop iteration")
    ap.add_argument("--rows", type=Path, default=ROWS_IN)
    ap.add_argument("--rows-out", type=Path, default=ROWS_OUT)
    ap.add_argument("--votes-out", type=Path, default=VOTES_OUT)
    ap.add_argument("--json", type=Path, default=ANALYSIS_OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of flagged debates (pilot use)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    if a.analyze_only:
        with open(a.rows_out) as f:
            rows = list(csv.DictReader(f))
        with open(a.votes_out) as f:
            votes = list(csv.DictReader(f))
        for v in votes:
            v["objected_r3b"] = int(v["objected_r3b"])
            v["vote_parsed"] = int(v["vote_parsed"])
            v["modification_addressed"] = int(v["modification_addressed"])
        res = analyze_iter2(rows, votes, draws=a.draws)
        print_analysis(res)
        a.json.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {a.json}")
        return 0

    targets = load_target_debates(a.rows)
    print(f"{len(targets)} debates in {a.rows}")

    items = rcd.load_items(source="scruples", n_yta=99, n_nta=150,
                           min_votes=50, min_consensus=0.90, seed=44)
    by_id = {i.item_id: i for i in items}
    missing = [t for t in targets if t[2] not in by_id]
    assert not missing, f"{len(missing)} rows reference unknown items"

    # ---- replay everything from cache, gate on the frozen sensor ----------
    flagged_states: list[tuple] = []
    n_ineligible = n_unflagged = 0
    with no_spend():
        for model, arm, item_id, idx in targets:
            state = replay_debate(model, arm, by_id[item_id], idx)
            d = gate_state(state)
            if d is None:
                n_ineligible += 1
                continue
            if not flagged(d):
                n_unflagged += 1
                continue
            flagged_states.append((model, arm, by_id[item_id], idx, state, d))
    print(f"replayed {len(targets)}: {n_ineligible} ineligible, "
          f"{n_unflagged} unflagged, {len(flagged_states)} FLAGGED")
    if a.limit:
        flagged_states = flagged_states[:a.limit]
        print(f"--limit: running first {len(flagged_states)}")

    n_calls = len(flagged_states) * (3 + 3 * 4)
    print(f"iteration-2 calls: {n_calls} "
          f"({len(flagged_states)} debates x 15)")
    if a.dry_run:
        return 0

    all_rows: list[dict] = []
    all_votes: list[dict] = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {
            ex.submit(run_iteration2, model, arm, item, idx, state, d):
            (item.item_id, idx)
            for model, arm, item, idx, state, d in flagged_states
        }
        done = 0
        for fut in as_completed(futs):
            try:
                rows, votes = fut.result()
                all_rows.extend(rows)
                all_votes.extend(votes)
            except Exception as e:
                print(f"  ERROR {futs[fut]}: {e}")
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(flagged_states)} debates")

    all_rows.sort(key=lambda r: (r["item_id"], r["sample_idx"], r["subarm"]))
    all_votes.sort(key=lambda r: (r["item_id"], r["sample_idx"],
                                  r["subarm"], r["role_id"]))
    write_csv(all_rows, a.rows_out, ROW_FIELDS)
    write_csv(all_votes, a.votes_out, VOTE_FIELDS)
    print(f"wrote {a.rows_out} ({len(all_rows)} rows), "
          f"{a.votes_out} ({len(all_votes)} rows)")

    res = analyze_iter2(all_rows, all_votes, draws=a.draws)
    print_analysis(res)
    a.json.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.json}")
    return 0


# ---------------------------------------------------------------------------
# Selftest: offline, stub generator, temp cache dir
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile

    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # 1. Notes.
    note = informed_note(True, ["writer_advocate"])
    check("informed note names the neutral seat", "Neutral Adjudicator" in note)
    check("informed note names the mislocalised seat",
          "Writer's Advocate" in note)
    check("informed note never mentions gold or verdict correctness",
          "gold" not in note.lower() and "correct" not in note.lower())
    check("generic note has no reason clause", "although" not in GENERIC_NOTE
          and "Adjudicator" not in GENERIC_NOTE)

    # 2. Gate parity with analyze_loop_step on a hand-built state.
    state = {
        "row": {"item_id": "i1", "arm": "third_person",
                "gold_verdict": "YTA", "synthesis_verdict": "NTA",
                "verdict": "YTA"},
        "vote_rows": [
            {"role_id": "writer_advocate", "objected_r3": 1,
             "role_stake": "not_at_fault"},
            {"role_id": "counterparty", "objected_r3": 0,
             "role_stake": "at_fault"},
            {"role_id": "neutral_adjudicator", "objected_r3": 0,
             "role_stake": "none"},
        ],
    }
    d = gate_state(state)
    # synthesis NTA (code 0); writer_advocate wants not_at_fault, so its
    # stake is NOT undermined -> its objection is mislocalised -> flagged.
    check("gate: mislocalised objection flags", d is not None and flagged(d))
    check("gate: mislocal_seats identifies the seat",
          mislocal_seats(d) == ["writer_advocate"])
    state["vote_rows"][0]["objected_r3"] = 0
    d2 = gate_state(state)
    check("gate: no objection, no flag", d2 is not None and not flagged(d2))
    state["row"]["synthesis_verdict"] = "UNRESOLVED"
    check("gate: non-codable synthesis is ineligible",
          gate_state(state) is None)

    # 3. Full offline chain against the deliberation stub, temp cache dir.
    real_out, real_gen = rcd.OUT_DIR, rcd.generate_any
    tmp = Path(tempfile.mkdtemp(prefix="loop_iter2_selftest_"))
    try:
        rcd.OUT_DIR = tmp
        item = rcd.CrowdGoldItem(
            item_id="selftest_item", post_text="AITA for testing my code?",
            gold_verdict="YTA", n_votes=100, consensus=0.95,
            source="selftest")

        # no_spend must raise on an empty cache.
        raised = False
        try:
            with no_spend():
                replay_debate("stub-model", "third_person", item, 0)
        except RuntimeError:
            raised = True
        check("no_spend raises on cache miss", raised)

        # Populate iteration 1 with the stub, then replay must be free.
        rcd.generate_any = rcd._stub_generate  # type: ignore[assignment]
        rcd.run_deliberation("stub-model", "third_person", item, 0)
        rcd.generate_any = real_gen  # type: ignore[assignment]
        with no_spend():
            st = replay_debate("stub-model", "third_person", item, 0)
        check("replay from cache succeeds under no_spend",
              bool(st["proposal"]))

        gd = gate_state(st)
        if gd is None or not flagged(gd):
            # The stub's debate may be non-codable or unflagged; force a
            # flagged dict so iteration 2 is still exercised end to end.
            gd = {"item": item.item_id, "arm": "third_person",
                  "syn_ok": False, "fin_ok": False, "syn_code": 1,
                  "obj_neutral_adjudicator": True}
        rcd.generate_any = rcd._stub_generate  # type: ignore[assignment]
        rows, votes = run_iteration2("stub-model", "third_person", item, 0,
                                     st, gd)
        check("iteration 2 produces one row per sub-arm",
              [r["subarm"] for r in rows] == list(SUBARMS))
        check("iteration 2 produces 3 votes per sub-arm", len(votes) == 9)
        n_files = len(list(tmp.glob("cgd_*_r3b_label_*.json")))
        check("R3b cached once, shared across sub-arms", n_files == 3)
        n_integ = len(list(tmp.glob("cgd_*_integ2_*.json")))
        check("three integ2 caches, one per sub-arm", n_integ == 3)

        # Idempotence: a second pass must be pure cache (guard active).
        rcd.generate_any = real_gen  # type: ignore[assignment]
        with no_spend():
            rows2, _ = run_iteration2("stub-model", "third_person", item, 0,
                                      st, gd)
        check("iteration 2 is cache-idempotent",
              [r["s3_verdict"] for r in rows2]
              == [r["s3_verdict"] for r in rows])
    finally:
        rcd.OUT_DIR = real_out
        rcd.generate_any = real_gen  # type: ignore[assignment]

    # 4. Analysis recovers a planted informed-arm effect.
    rng = random.Random(3)
    rows, votes = [], []
    for i in range(240):
        s2 = rng.random() < 0.5
        for sub in SUBARMS:
            p_fix = {"blind": 0.10, "generic": 0.15, "informed": 0.45}[sub]
            s3 = s2 if s2 else (rng.random() < p_fix)
            if s2 and rng.random() < 0.03:
                s3 = False
            rows.append({
                "item_id": f"i{i % 60}", "sample_idx": i, "arm": "x",
                "subarm": sub, "s2_correct": int(s2), "s3_correct": int(s3),
            })
    res = analyze_iter2(rows, votes, draws=400)
    check("planted informed net positive, CI excludes zero",
          res["informed_net"]["lo95"] is not None
          and res["informed_net"]["lo95"] > 0)
    check("planted info gap recovered (informed > blind)",
          res["info_gap_informed_minus_blind"]["point"] > 0.05)
    check("blind net small", abs(res["blind_net"]["point"]) < 0.10)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
