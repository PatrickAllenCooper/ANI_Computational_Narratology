"""
scripts/extract_phase11_r4_votes.py -- per-perspective R4 vote extraction for
the Phase 11 multi-stakeholder head-to-head.

The debate moderator (run_phase5_e2_scaled.py) records only the aggregate R4
tally (accept_count / reject_count) per debate, not which perspective rejected.
To test whether NoT's R4 rejections concentrate in the materially-undermined
role while an optimised-CoT's rejections scatter, we need each perspective's
verdict. This script reads every cached per-perspective R4 vote
(`e2_{arm}_r4_{gen}_{sid}_{pid}_000.json`), classifies it ACCEPT / REJECT /
UNKNOWN with a small cached judge call, and joins the integration record's
`modifications_unaddressed` list to flag whether the rejecting perspective was
one the integrated proposal failed to accommodate ("undermined").

Output: divergence_study_outputs/p11_r4_votes.csv with one row per
(arm, generator, scenario, perspective) that reached R4.

Idempotent: vote classifications cached under p11r4dec_*.

Usage:
  python -m scripts.extract_phase11_r4_votes
         [--arms not,textgrad_cot,std_cot]
         [--generators claude-haiku-4-5,gpt-5.4-nano]
         [--extractor claude-haiku-4-5] [--workers 6]
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.generators import generate
from scripts.run_phase1_quartet import load_daily_dilemmas, _extract_json, _safe
from scripts.run_phase5_e2_scaled import PERSPECTIVE_IDS, PERSPECTIVE_BY_ID

OUT_DIR = Path("./divergence_study_outputs")

DEFAULT_ARMS = ["not", "textgrad_cot", "std_cot"]
DEFAULT_GENERATORS = ["claude-haiku-4-5", "gpt-5.4-nano"]
DEFAULT_EXTRACTOR = "claude-haiku-4-5"

VOTE_SYSTEM = (
    "You are a precise extractor. An agent in a multi-stakeholder debate was "
    "asked to cast a final vote of ACCEPT or REJECT on an integrated proposal, "
    "reasoning before stating its verdict. Read the response and report the "
    "final verdict only. Return valid JSON."
)
VOTE_USER = (
    "Agent response (verdict is near the end):\n{response}\n\n"
    "Return JSON with EXACTLY one key:\n"
    "- \"vote\": \"ACCEPT\" if the agent's final position endorses the proposal, "
    "\"REJECT\" if it refuses or holds a non-negotiable objection, or "
    "\"UNKNOWN\" if no clear verdict is stated."
)


def _r4_path(arm: str, gen: str, sid: str, pid: str) -> Path:
    return OUT_DIR / f"e2_{arm}_r4_{_safe(gen)}_{sid}_{pid}_000.json"


def _integ_path(arm: str, gen: str, sid: str) -> Path:
    return OUT_DIR / f"e2_{arm}_integ_{_safe(gen)}_{sid}_000.json"


def _vote_cache(arm: str, gen: str, sid: str, pid: str) -> Path:
    return OUT_DIR / f"p11r4dec_{_safe(gen)}_{arm}_{sid}_{pid}.json"


def classify_vote(arm: str, gen: str, sid: str, pid: str, extractor: str) -> str:
    cache = _vote_cache(arm, gen, sid, pid)
    if cache.exists():
        return json.loads(cache.read_text()).get("vote", "UNKNOWN")
    r4 = _r4_path(arm, gen, sid, pid)
    text = (json.loads(r4.read_text()).get("output", "") or "")
    raw = generate(
        extractor, VOTE_SYSTEM, VOTE_USER.format(response=text[-3000:]),
        sample_idx=0, max_tokens=120, json_mode=True,
    ).text
    vote = (_extract_json(raw).get("vote") or "UNKNOWN").upper()
    if vote not in {"ACCEPT", "REJECT", "UNKNOWN"}:
        vote = "UNKNOWN"
    cache.write_text(json.dumps({"vote": vote}, ensure_ascii=False))
    return vote


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 11 per-perspective R4 vote extraction")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--generators", default=",".join(DEFAULT_GENERATORS))
    ap.add_argument("--extractor", default=DEFAULT_EXTRACTOR)
    ap.add_argument("--eval-start", type=int, default=30)
    ap.add_argument("--eval-end", type=int, default=60)
    ap.add_argument("--scenarios-n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    scenarios = load_daily_dilemmas(n=args.scenarios_n)[args.eval_start:args.eval_end]

    # Discover (arm, gen, sid, pid) cells that actually reached R4.
    cells = []
    for arm in arms:
        for gen in generators:
            for sc in scenarios:
                if not _integ_path(arm, gen, sc.id).exists():
                    continue  # debate never built an integrated proposal -> no R4
                for pid in PERSPECTIVE_IDS:
                    if _r4_path(arm, gen, sc.id, pid).exists():
                        cells.append((arm, gen, sc.id, pid))
    print(f"R4 per-perspective votes to classify: {len(cells)}", flush=True)

    votes: dict[tuple, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(classify_vote, arm, gen, sid, pid, args.extractor): (arm, gen, sid, pid)
            for (arm, gen, sid, pid) in cells
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                votes[key] = fut.result()
            except Exception as e:
                print(f"  ERROR {key}: {e}", flush=True)
                votes[key] = "UNKNOWN"
            done += 1
            if done % 50 == 0 or done == len(cells):
                print(f"  {done}/{len(cells)} classified", flush=True)

    # Join integration's modifications_unaddressed for the "undermined" flag.
    rows = []
    for (arm, gen, sid, pid) in cells:
        integ = json.loads(_integ_path(arm, gen, sid).read_text())
        unaddressed = set(integ.get("modifications_unaddressed") or [])
        vote = votes[(arm, gen, sid, pid)]
        rows.append({
            "arm": arm,
            "generator": gen,
            "scenario_id": sid,
            "pid": pid,
            "role_label": PERSPECTIVE_BY_ID[pid]["label"],
            "vote": vote,
            "reject": int(vote == "REJECT"),
            "undermined": int(pid in unaddressed),
            "n_unaddressed": len(unaddressed),
        })

    out_csv = OUT_DIR / "p11_r4_votes.csv"
    fieldnames = ["arm", "generator", "scenario_id", "pid", "role_label",
                  "vote", "reject", "undermined", "n_unaddressed"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")

    # Console summary: rejection counts by role per arm.
    print("\n--- R4 rejections by role (per arm) ---")
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        rej = [r for r in arm_rows if r["reject"]]
        by_role = {pid: sum(1 for r in rej if r["pid"] == pid) for pid in PERSPECTIVE_IDS}
        total = len(rej)
        share = (max(by_role.values()) / total) if total else 0.0
        print(f"  {arm:14s} n_votes={len(arm_rows):3d} rejections={total:2d} "
              f"by_role={by_role} modal_share={share:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
