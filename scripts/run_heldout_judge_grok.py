"""
scripts/run_heldout_judge_grok.py -- Run grok-4-1-fast-reasoning as a held-out
third judge on a 30-scenario subsample from the 100-scenario DailyDilemmas Phase 1
pool to address reviewer W3 (generator/judge circularity).

Design:
  - 30 scenarios drawn deterministically from the 100-scenario pool (seed=99).
  - Generators: grok-4-1-fast-reasoning and claude-sonnet-4-6, chosen because
    both have full j1 (claude-haiku-4-5) and j2 (gpt-5.4-nano) coverage at
    sample_idx=0 across all 100 scenarios.
  - All 3 conditions (baseline_io, standard_cot, narrative_cot) at sample_idx=0.
  - Total: 30 x 3 x 2 = 180 generation outputs judged by grok-4-1-fast-reasoning.
  - Uses the same JUDGE_SYSTEM and JUDGE_USER_TEMPLATE as j1 and j2.

Cache naming:
  judge_grok-4-1-fast-reasoning_gen_{gen_model}_{scenario_id}_{condition}_{idx:03d}.json

Output:
  divergence_study_outputs/coded_results_judge3_grok.csv

Usage (from repo root):
  python -m scripts.run_heldout_judge_grok [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generators import generate as _generate
from scripts.run_phase1_quartet import (
    JUDGE_SYSTEM, JUDGE_USER_TEMPLATE,
    load_daily_dilemmas,
    gen_cache_path, _safe, _extract_json,
    SCALED_SAMPLE_SEED,
    Scenario,
)

OUT_DIR = Path("./divergence_study_outputs")
OUT_DIR.mkdir(exist_ok=True)

JUDGE_MODEL = "grok-4-1-fast-reasoning"
SUBSAMPLE_SEED = 99  # pre-registered for Tier-3; used here for the held-out-judge check
N_SUBSAMPLE = 30
GENERATORS = ["grok-4-1-fast-reasoning", "claude-sonnet-4-6"]
CONDITIONS = ["baseline_io", "standard_cot", "narrative_cot"]
SAMPLE_IDX = 0


# ---------------------------------------------------------------------------
# Subsample selection (deterministic, seed=99)
# ---------------------------------------------------------------------------

def select_subsample(all_scenarios: list[Scenario], n: int, seed: int) -> list[Scenario]:
    """Return a deterministic subsample of n scenarios from the full 100-scenario list."""
    rng = random.Random(seed)
    pool = list(all_scenarios)
    rng.shuffle(pool)
    return pool[:n]


# ---------------------------------------------------------------------------
# Judge cache path for j3
# ---------------------------------------------------------------------------

def j3_cache_path(gen_model: str, scenario_id: str, condition: str, idx: int) -> Path:
    return OUT_DIR / (
        f"judge_{_safe(JUDGE_MODEL)}_gen_{_safe(gen_model)}"
        f"_{scenario_id}_{condition}_{idx:03d}.json"
    )


# ---------------------------------------------------------------------------
# Load existing generation output
# ---------------------------------------------------------------------------

def load_gen_output(gen_model: str, scenario_id: str, condition: str, idx: int) -> Optional[str]:
    p = gen_cache_path(gen_model, scenario_id, condition, idx)
    if p.exists():
        try:
            return json.loads(p.read_text()).get("output", "")
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Run grok as j3 on one sample
# ---------------------------------------------------------------------------

def judge_one(
    scenario: Scenario,
    gen_model: str,
    condition: str,
    idx: int,
) -> Optional[dict]:
    cp = j3_cache_path(gen_model, scenario.id, condition, idx)
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            cp.unlink()

    output_text = load_gen_output(gen_model, scenario.id, condition, idx)
    if output_text is None:
        print(f"  SKIP: no gen cache for {gen_model}/{scenario.id}/{condition}/{idx}")
        return None

    judge_user = JUDGE_USER_TEMPLATE.format(
        scenario=scenario.prompt,
        output=output_text,
    )
    result_text = _generate(
        JUDGE_MODEL,
        JUDGE_SYSTEM,
        judge_user,
        sample_idx=idx,
        max_tokens=1024,
        json_mode=True,
    ).text

    rec = _extract_json(result_text)
    rec["_judge"] = JUDGE_MODEL
    rec["gen_model"] = gen_model
    rec["scenario_id"] = scenario.id
    rec["condition"] = condition
    rec["sample_idx"] = idx
    rec["output_len"] = len(output_text)
    cp.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out third judge (grok) on subsample")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    print("Loading DailyDilemmas 100-scenario pool...")
    all_scenarios = load_daily_dilemmas(n=100)
    subsample = select_subsample(all_scenarios, N_SUBSAMPLE, SUBSAMPLE_SEED)
    print(f"Selected {len(subsample)} scenarios (seed={SUBSAMPLE_SEED})")

    tasks = [
        (s, gen, cond, SAMPLE_IDX)
        for s in subsample
        for gen in GENERATORS
        for cond in CONDITIONS
    ]
    print(f"Judge: {JUDGE_MODEL}")
    print(f"Generators: {GENERATORS}")
    print(f"Tasks: {len(tasks)} ({N_SUBSAMPLE} scenarios x {len(GENERATORS)} gens x {len(CONDITIONS)} conditions)")
    print(f"Workers: {args.workers}")

    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(judge_one, s, gen, cond, idx): (s.id, gen, cond)
            for s, gen, cond, idx in tasks
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                rec = fut.result()
                if rec is not None:
                    rows.append(rec)
            except Exception as e:
                print(f"  ERROR {key}: {e}", flush=True)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)

    out = OUT_DIR / "coded_results_judge3_grok.csv"
    cols = [
        "scenario_id", "condition", "sample_idx", "gen_model",
        "output_len", "refused", "stakeholder_count", "max_causal_hops",
        "uncertainty_score", "framework_names_invoked", "commits_to_action",
        "action_summary", "_judge",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out} ({len(rows)} rows)")

    # Direction-of-effect preview
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        for var in ["stakeholder_count", "max_causal_hops", "uncertainty_score"]:
            if var not in df.columns:
                continue
            narr = df[df["condition"] == "narrative_cot"][var].mean()
            base = df[df["condition"] == "baseline_io"][var].mean()
            std = df[df["condition"] == "standard_cot"][var].mean()
            direction = "narr>base" if narr > base else "narr<base"
            print(f"  {var}: narr={narr:.2f}  std={std:.2f}  base={base:.2f}  [{direction}]")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
