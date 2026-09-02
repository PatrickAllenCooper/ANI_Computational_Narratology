"""
scripts/run_crux_escalation.py -- S2 of Addendum 5: crux escalation.

For every FLAGGED deliberation debate (composite flag, frozen), a second-vendor
single agent (claude-haiku-4-5, the byte-identical `standard` system prompt
its cached A3a verdicts were produced under) receives the post in the SAME
arm wrapper PLUS the three R3 objection texts from the deliberation -- the
MODIFICATION / UNRESOLVABLE lines -- and returns a verdict. It never sees S1,
S2, or any verdict string from the community. k=3 per debate, majority, tie
falls back to S2.

Paired against A3a (the same model reading the post cold) on the same debates.
Registered hypothesis, from A1's null: the community's content carries no
direction, so crux-informed haiku will NOT beat cold haiku. Two-sided test.

Usage
-----
  python -m scripts.run_crux_escalation --selftest
  python -m scripts.run_crux_escalation --dry-run
  python -m scripts.run_crux_escalation --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import scripts.run_crowdgold_aita as rca
import scripts.run_crowdgold_deliberation as rcd
from scripts.analyze_actuator_ladder import load_debates
from scripts.analyze_loop_step import flagged
from scripts.run_loop_iteration import no_spend, replay_debate
from scripts.run_phase1_quartet import OUT_DIR, _safe
from scripts.verdict_format import extract_verdict

ESCALATION_MODEL = "claude-haiku-4-5"
K_CRUX = 3
MAX_TOKENS = 512          # matches the cached haiku standard cells
ROWS_OUT = OUT_DIR / "crux_escalation_rows.csv"

CRUX_HEADER = (
    "Three reviewers with different briefs have each read this account and "
    "responded to a proposed reading of it. Their objections are reproduced "
    "below, for your consideration only; you are not bound by them and you "
    "are not told what the proposed reading was."
)


def crux_block(objections: dict[str, str]) -> str:
    lines = []
    for rid in rcd.ROLE_ORDER:
        text = (objections.get(rid) or "").strip()
        label = rcd.ROLE_BY_ID[rid].label
        lines.append(f"- {label}: {text if text else '[no objection stated]'}")
    return CRUX_HEADER + "\n" + "\n".join(lines)


def crux_user_turn(arm: str, post_text: str, objections: dict[str, str]) -> str:
    return rca.build_user_turn(arm, post_text) + "\n\n" + crux_block(objections)


def crux_cache_path(model: str, arm: str, item_id: str, debate_idx: int,
                    k: int) -> Path:
    return (OUT_DIR / f"cg_crux_{_safe(model)}_{arm}_{_safe(item_id)}_"
                      f"d{debate_idx:02d}_{k:02d}.json")


def run_crux_cell(model: str, arm: str, item, debate_idx: int, k: int,
                  objections: dict[str, str]) -> dict:
    cache = crux_cache_path(model, arm, item.item_id, debate_idx, k)
    if cache.exists():
        return json.loads(cache.read_text())
    system = rca.build_system(allow_unresolved=True,
                              scaffold=rca.SCAFFOLD_STANDARD,
                              instrument=rca.INSTRUMENT)
    user = crux_user_turn(arm, item.post_text, objections)
    result = rca.generate_any(model, system, user, sample_idx=k,
                              max_tokens=MAX_TOKENS)
    rec = {
        "model": model, "arm": arm, "item_id": item.item_id,
        "debate_idx": debate_idx, "k": k, "gold_verdict": item.gold_verdict,
        "output": result.text,
        "finish_reason": getattr(result, "finish_reason", ""),
        "prompt_tokens": getattr(result, "prompt_tokens", 0),
        "completion_tokens": getattr(result, "completion_tokens", 0),
        "objections_sha": rca.sha256_text(json.dumps(objections, sort_keys=True)),
    }
    cache.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def run_debate(model: str, d: dict, item, objections: dict[str, str],
               k_crux: int = K_CRUX) -> dict:
    verdicts = []
    for k in range(k_crux):
        rec = run_crux_cell(model, d["arm"], item, d["sample_idx"], k, objections)
        verdicts.append(extract_verdict(rec.get("output") or "", rca.INSTRUMENT))
    return {
        "model": model, "arm": d["arm"], "item_id": d["item"],
        "sample_idx": d["sample_idx"], "gold_verdict": d["gold"],
        "s2_verdict": d["s2"],
        "crux_verdicts": "|".join(verdicts),
        "n_objections_stated": sum(1 for v in objections.values() if v),
    }


ROW_FIELDS = ("model", "arm", "item_id", "sample_idx", "gold_verdict",
              "s2_verdict", "crux_verdicts", "n_objections_stated")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S2 crux escalation")
    ap.add_argument("--model", default=ESCALATION_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rows-out", type=Path, default=ROWS_OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    debates = [d for d in load_debates() if flagged(d)]
    items = rcd.load_items(source="scruples", n_yta=99, n_nta=150,
                           min_votes=50, min_consensus=0.90, seed=44)
    by_id = {i.item_id: i for i in items}
    print(f"{len(debates)} flagged debates; replaying for R3 objections ...")
    obj_by_debate: dict[tuple, dict[str, str]] = {}
    with no_spend():
        for d in debates:
            st = replay_debate(d["model"], d["arm"], by_id[d["item"]],
                               d["sample_idx"])
            obj_by_debate[(d["item"], d["arm"], d["sample_idx"])] = st["r3_objections"]
    n_calls = len(debates) * K_CRUX
    print(f"{n_calls} {a.model} calls ({len(debates)} x {K_CRUX})")
    if a.dry_run:
        d0 = debates[0]
        print("\nexample user turn tail:\n"
              + crux_user_turn(d0["arm"], "[post]",
                               obj_by_debate[(d0["item"], d0["arm"], d0["sample_idx"])])[-900:])
        return 0

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_debate, a.model, d, by_id[d["item"]],
                          obj_by_debate[(d["item"], d["arm"], d["sample_idx"])]): d
                for d in debates}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"  ERROR {futs[fut]['item']}/{futs[fut]['sample_idx']}: {e}")
            if i % 50 == 0:
                print(f"  {i}/{len(debates)}")
    rows.sort(key=lambda r: (r["item_id"], r["arm"], r["sample_idx"]))
    a.rows_out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.rows_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROW_FIELDS))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.rows_out} ({len(rows)} rows)")
    return 0


def _selftest() -> int:
    import tempfile
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    objs = {"writer_advocate": "The synthesis ignores that the writer asked first.",
            "counterparty": "", "neutral_adjudicator": "Both parties escalated."}
    ut = crux_user_turn("third_person", "AITA for X?", objs)
    check("user turn starts with the byte-identical arm wrapper",
          ut.startswith(rca.build_user_turn("third_person", "AITA for X?")))
    check("crux block names seats and marks the unstated one",
          "Writer's Advocate:" in ut and "[no objection stated]" in ut)
    check("no verdict tokens injected by us",
          "SYNTHESIS" not in ut and "PROPOSAL" not in ut
          and "VERDICT:" not in crux_block(objs))
    check("wrapper orthogonality: as_asker differs only by wrapper",
          crux_block(objs) in crux_user_turn("as_asker", "AITA for X?", objs))

    global OUT_DIR
    real_gen, real_out = rca.generate_any, OUT_DIR
    tmp = Path(tempfile.mkdtemp(prefix="crux_selftest_"))
    try:
        OUT_DIR = tmp
        calls = {"n": 0}

        class R:
            text = "VERDICT: YTA"
            finish_reason = "stop"
            prompt_tokens = completion_tokens = 1

        def stub(model, system, user, **kw):
            calls["n"] += 1
            return R()

        rca.generate_any = stub  # type: ignore[assignment]
        item = rcd.CrowdGoldItem(item_id="it", post_text="AITA for X?",
                                 gold_verdict="YTA", n_votes=100,
                                 consensus=0.95, source="selftest")
        d = {"arm": "third_person", "sample_idx": 1, "item": "it",
             "gold": "YTA", "s2": "NTA"}
        row = run_debate("stub", d, item, objs)
        check("k=3 calls made", calls["n"] == 3)
        check("verdicts parsed", row["crux_verdicts"] == "YTA|YTA|YTA")
        row2 = run_debate("stub", d, item, objs)
        check("second pass is pure cache", calls["n"] == 3 and row2 == row)
        check("cache namespace is cg_crux_ with debate idx",
              all(p.name.startswith("cg_crux_stub_third_person_it_d01_")
                  for p in tmp.glob("*.json")) and len(list(tmp.glob("*.json"))) == 3)
    finally:
        rca.generate_any = real_gen  # type: ignore[assignment]
        OUT_DIR = real_out

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
