"""
scripts/run_brokenmath.py -- Phase 16 BrokenMath quartet evaluation.

Quartet x {raw, standard_cot, narrative_cot} on adversarial benchmark.
Output: divergence_study_outputs/brokenmath_raw.csv

The UNRESOLVED amendment is opt-in via --allow-unresolved (or
NOT_ALLOW_UNRESOLVED=1) and is OFF by default: with the flag off the prompt,
the request, and every cache filename are byte-identical to the committed
behaviour. With the flag on, the forced VERDICT line is appended to the user
turn, the verdict is parsed back out, and caches are written under a separate
"_unres" key so the two regimes can never collide.

Usage:
  python -m scripts.run_brokenmath --smoke
  python -m scripts.run_brokenmath --workers 6
  python -m scripts.run_brokenmath --allow-unresolved --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.brokenmath_scorer import score_brokenmath_response
from scripts.generators import _is_reasoning, generate
from scripts.load_brokenmath import load_brokenmath
from scripts.run_phase1_quartet import PROMPTS, _safe, OUT_DIR
from scripts.verdict_format import (
    add_unresolved_cli_flag,
    allow_unresolved_enabled,
    augment_user_turn,
    cache_suffix,
    extract_verdict,
    is_noncommittal,
    score_against_gold,
)

BUDGET_GENERATORS = ["gpt-5.4-nano", "claude-haiku-4-5", "grok-4-1-fast-reasoning"]
ALL_GENERATORS = BUDGET_GENERATORS + ["claude-sonnet-4-6"]
ARMS = ["raw", "standard_cot", "narrative_cot"]
OUT_CSV = OUT_DIR / "brokenmath_raw.csv"
DEFAULT_JUDGE = os.environ.get("BROKENMATH_JUDGE", "claude-haiku-4-5")
VERDICT_INSTRUMENT = "brokenmath"


def _gen_cache(
    problem_id: str, gen: str, arm: str, *, allow_unresolved: bool = False,
) -> Path:
    # cache_suffix() is "" unless the amendment is on, so the default filename
    # is byte-identical to every cache file already on disk.
    suf = cache_suffix(allow_unresolved)
    return OUT_DIR / f"bm_gen_{_safe(gen)}_{arm}{suf}_{_safe(problem_id)}.json"


def _system_prompt(arm: str) -> str:
    if arm == "raw":
        return ""
    return PROMPTS.get(arm, PROMPTS["standard_cot"])


def _generate(
    problem: str, gen: str, arm: str, problem_id: str,
    *, allow_unresolved: bool = False,
) -> tuple[str, bool]:
    cache = _gen_cache(problem_id, gen, arm, allow_unresolved=allow_unresolved)
    if cache.exists():
        obj = json.loads(cache.read_text())
        text = obj.get("response", "")
        if text.strip():
            return text, False
    sys_prompt = _system_prompt(arm)
    # Identity when allow_unresolved is False -> the user turn is unchanged.
    problem = augment_user_turn(
        problem, VERDICT_INSTRUMENT, allow_unresolved=allow_unresolved,
    )
    max_tokens = 2048 if arm == "narrative_cot" else 1536
    if _is_reasoning(gen):
        max_tokens = max(max_tokens, 8192)
    text = ""
    kwargs = {}
    if _is_reasoning(gen):
        kwargs["reasoning_effort"] = "minimal"
    for attempt in range(3):
        result = generate(
            gen, sys_prompt, problem, sample_idx=attempt, max_tokens=max_tokens, **kwargs,
        )
        text = (result.text or "").strip()
        if text:
            break
    if text:
        cache.write_text(json.dumps({"response": text, "arm": arm}, ensure_ascii=False))
        return text, False
    return "", True


def _run_cell(
    item, gen: str, arm: str, judge: str, *, allow_unresolved: bool = False,
) -> dict:
    response, empty = _generate(
        item.problem, gen, arm, item.problem_id,
        allow_unresolved=allow_unresolved,
    )
    row = {
        "problem_id": item.problem_id,
        "question_type": item.question_type,
        "generator": gen,
        "arm": arm,
        "response_len": len(response),
        "empty_response": int(empty),
    }
    if empty or not response.strip():
        row["category"] = "UNKNOWN"
        row["is_sycophant"] = -1
    else:
        scored = score_brokenmath_response(
            item.problem, item.original_problem, response,
            problem_id=item.problem_id, judge=judge,
        )
        row.update(scored)
    if allow_unresolved:
        # Extra columns exist only in the amended regime, so the default CSV
        # schema is unchanged.
        verdict = extract_verdict(response, VERDICT_INSTRUMENT)
        # The presented statement is false for adversarial items and true for
        # the originals, so that is the ground truth the forced line is scored
        # against. UNRESOLVED is never correct (see score_against_gold).
        gold = "FALSE" if getattr(item, "is_adversarial", True) else "TRUE"
        scored_v = score_against_gold(verdict, gold, VERDICT_INSTRUMENT)
        row["verdict"] = verdict
        row["verdict_gold"] = gold
        row["verdict_correct"] = int(scored_v["correct"])
        row["verdict_noncommittal"] = int(is_noncommittal(verdict))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 16 BrokenMath eval")
    ap.add_argument("--n", type=int, default=None, help="Subsample (default: full 451)")
    ap.add_argument("--generators", default=",".join(ALL_GENERATORS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=44)
    add_unresolved_cli_flag(ap)
    args = ap.parse_args()
    if args.smoke:
        args.n = 5
        args.workers = 2
        args.generators = "claude-haiku-4-5"
    allow_unresolved = allow_unresolved_enabled(args.allow_unresolved)

    items = load_brokenmath(n=args.n, seed=args.seed)
    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(f"BrokenMath: n={len(items)} gens={generators} arms={arms} judge={args.judge}", flush=True)
    print(f"  forced-verdict UNRESOLVED amendment: {'ON' if allow_unresolved else 'OFF'}", flush=True)
    tasks = [(item, gen, arm) for item in items for gen in generators for arm in arms]
    print(f"  tasks: {len(tasks)}", flush=True)

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _run_cell, item, gen, arm, args.judge,
                allow_unresolved=allow_unresolved,
            ): (item.problem_id, gen, arm)
            for item, gen, arm in tasks
        }
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"  ERROR {futs[fut]}: {e}", flush=True)
            done += 1
            if done % 50 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)}", flush=True)

    if not rows:
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
