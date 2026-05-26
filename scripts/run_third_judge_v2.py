"""
scripts/run_third_judge_v2.py -- Adversarial third-judge pass for N-CoT v2.

Iterates every cached narrative_cot_v2 generation cell across all four
Phase 1 generators and re-codes each output with grok-4-1-fast-reasoning
using the verbatim Phase 1 judge rubric (same JUDGE_SYSTEM and
JUDGE_USER_TEMPLATE as the primary judge).

Cache convention: judge3_grok_{gen_model}_{scenario_id}_narrative_cot_v2_{idx:03d}.json
(distinct prefix "judge3_" so existing Phase 1 aggregations are unaffected)

Usage:
  python -m scripts.run_third_judge_v2 [--workers 6]
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.generators import generate
from scripts.run_phase1_quartet import (
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    load_daily_dilemmas,
    _extract_json,
    _safe,
    N_PER_GENERATOR,
    DEFAULT_N_FALLBACK,
    gen_cache_path,
    OUT_DIR,
)

THIRD_JUDGE   = "grok-4-1-fast-reasoning"
CONDITION     = "narrative_cot_v2"
GENERATORS    = ["gpt-5.4-nano", "claude-haiku-4-5",
                 "grok-4-1-fast-reasoning", "claude-sonnet-4-6"]


def judge3_cache_path(gen_model: str, scenario_id: str, idx: int) -> Path:
    return OUT_DIR / (
        f"judge3_{_safe(THIRD_JUDGE)}_{_safe(gen_model)}"
        f"_{scenario_id}_{CONDITION}_{idx:03d}.json"
    )


def recode_one(gen_model: str, scenario_id: str, scenario_prompt: str,
               idx: int) -> dict | None:
    """Re-code one cached generation with the third judge.

    Returns the judge record, or None if the generation cache does not exist.
    """
    out_path = judge3_cache_path(gen_model, scenario_id, idx)
    if out_path.exists():
        return json.loads(out_path.read_text())

    gen_path = gen_cache_path(gen_model, scenario_id, CONDITION, idx)
    if not gen_path.exists():
        return None   # generation not yet available; skip silently

    gen_rec = json.loads(gen_path.read_text())
    output_text = gen_rec.get("output", "") or ""

    judge_user = JUDGE_USER_TEMPLATE.format(
        scenario=scenario_prompt, output=output_text
    )
    raw = generate(
        THIRD_JUDGE, JUDGE_SYSTEM, judge_user,
        sample_idx=idx, max_tokens=1024, json_mode=True,
    ).text
    rec = _extract_json(raw)
    rec["_judge"]     = THIRD_JUDGE
    rec["_gen_model"] = gen_model
    rec["_scenario"]  = scenario_id
    rec["_idx"]       = idx
    out_path.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def main(workers: int = 6) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--generators", default=",".join(GENERATORS))
    args = ap.parse_args()

    generators = [g.strip() for g in args.generators.split(",") if g.strip()]

    print(f"Loading 100 DailyDilemmas scenarios...", flush=True)
    scenarios = load_daily_dilemmas(n=100)
    print(f"  {len(scenarios)} scenarios loaded.", flush=True)

    tasks: list[tuple[str, str, str, int]] = []
    for gen_model in generators:
        n = N_PER_GENERATOR.get(gen_model, DEFAULT_N_FALLBACK)
        for scenario in scenarios:
            for idx in range(n):
                tasks.append((gen_model, scenario.id, scenario.prompt, idx))

    total   = len(tasks)
    done    = 0
    skipped = 0
    errors  = 0

    print(f"Third-judge ({THIRD_JUDGE}) pass: {total} cells across "
          f"{len(generators)} generators.", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(recode_one, gen, sid, prompt, idx): (gen, sid, idx)
            for gen, sid, prompt, idx in tasks
        }
        for fut in as_completed(futs):
            gen, sid, idx = futs[fut]
            try:
                rec = fut.result()
                if rec is None:
                    skipped += 1
                else:
                    done += 1
            except Exception as exc:
                errors += 1
                print(f"  ERROR {gen} {sid} idx={idx}: {exc}", flush=True)

            if (done + skipped + errors) % 100 == 0:
                print(f"  progress: {done} coded, {skipped} skipped "
                      f"(gen not yet cached), {errors} errors / {total} total",
                      flush=True)

    print(f"\nDone. coded={done}, skipped={skipped}, errors={errors} / {total}")
    if skipped:
        print(f"  {skipped} cells skipped because generation cache is missing.")
        print("  Re-run after the Phase 1 v2 replication run has completed.")


if __name__ == "__main__":
    main()
