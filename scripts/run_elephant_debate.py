"""
scripts/run_elephant_debate.py -- Phase 12 multi-stakeholder NoT on ELEPHANT prompts.

Runs the R0-R4 integration protocol on OEQ and AITA-YTA subsets, scores the
integrated consensus advice with ELEPHANT metrics.

Output: divergence_study_outputs/elephant_debate_raw.csv

Usage:
  python -m scripts.run_elephant_debate --smoke
  python -m scripts.run_elephant_debate --datasets oeq,aita_yta --n 50 --workers 2
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.elephant_scorers import DEFAULT_JUDGE, score_response
from scripts.load_elephant import ELEPHANT_SEED, load_elephant, metrics_for_dataset
from scripts.run_phase1_quartet import Scenario, OUT_DIR
from scripts.run_phase5_e2_scaled import (
    ARM_TAG, AGENT_SYSTEM, PROMPTS, run_scenario_sample,
)
import scripts.run_phase5_e2_scaled as e2

OUT_CSV = OUT_DIR / "elephant_debate_raw.csv"
DEBATE_GENERATORS = ["claude-haiku-4-5", "gpt-5.4-nano"]
DEFAULT_MODERATOR = "claude-sonnet-4-6"


def _scenario_from_prompt(item_id: str, prompt: str, dataset: str) -> Scenario:
    return Scenario(
        id=f"elephant_{dataset}_{item_id}",
        prompt=prompt,
        topic=dataset,
        action_1="follow_integrated_advice",
        action_2="reject_integrated_advice",
    )


def _advice_text(debate_row: dict) -> str:
    desc = debate_row.get("integrated_description") or ""
    if desc:
        return desc
    return debate_row.get("r2_open_mod_summary") or debate_row.get("synthesis_description") or ""


def _run_one(
    dataset: str,
    item_id: str,
    prompt: str,
    gen_model: str,
    mod_model: str,
    judge: str,
    budget_cap: float,
) -> dict:
    scenario = _scenario_from_prompt(item_id, prompt, dataset)
    debate = run_scenario_sample(scenario, 0, gen_model, mod_model, budget_cap)
    advice = _advice_text(debate)
    row = {
        "dataset": dataset,
        "item_id": item_id,
        "generator": gen_model,
        "arm": "multi_stakeholder_not",
        "moderator": mod_model,
        "prompt_len": len(prompt),
        "response_len": len(advice),
        "response": advice,
        "r4_full_consensus": int(bool(debate.get("r4_full_consensus"))),
        "r4_structural_rejection": int(bool(debate.get("r4_structural_rejection"))),
        "has_synthesis": int(bool(debate.get("has_synthesis"))),
        "skipped_budget": int(bool(debate.get("skipped_budget"))),
    }
    for metric in metrics_for_dataset(dataset):
        if metric == "moral":
            continue
        s = score_response(metric, prompt, advice, judge=judge)
        row[f"score_{metric}"] = s
        row[f"sycophantic_{metric}"] = s
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 12 ELEPHANT multi-stakeholder NoT")
    ap.add_argument("--datasets", default="oeq,aita_yta")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--generators", default=",".join(DEBATE_GENERATORS))
    ap.add_argument("--moderator", default=DEFAULT_MODERATOR)
    ap.add_argument("--judge", default=os.environ.get("ELEPHANT_JUDGE", DEFAULT_JUDGE))
    ap.add_argument("--seed", type=int, default=ELEPHANT_SEED)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget-cap-usd", type=float, default=150.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n = 5
        args.workers = 1

    # Configure NoT agent prompt + cache namespace
    e2.AGENT_SYSTEM = PROMPTS["narrative_cot"]
    e2.ARM_TAG = "elephant_debate"

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    generators = [g.strip() for g in args.generators.split(",") if g.strip()]

    tasks = []
    for ds in datasets:
        items = load_elephant(ds, n=args.n, seed=args.seed)
        for item in items:
            for gen in generators:
                tasks.append((ds, item.id, item.prompt, gen))

    print(f"ELEPHANT debate: {len(tasks)} cells | n={args.n} datasets={datasets}", flush=True)

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _run_one, ds, iid, prompt, gen, args.moderator, args.judge, args.budget_cap_usd,
            ): (ds, iid, gen)
            for ds, iid, prompt, gen in tasks
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"  ERROR {key}: {e}", flush=True)
            done += 1
            if done % 5 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} | est cost ${e2._total_cost_usd:.2f}", flush=True)

    if not rows:
        print("No rows.")
        return 1

    fieldnames = sorted({k for r in rows for k in r})
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
