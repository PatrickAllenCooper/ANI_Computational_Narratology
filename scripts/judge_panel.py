"""
scripts/judge_panel.py -- Phase 15 cross-family judge panel re-scoring.

Re-scores a stratified OEQ sample on validation/indirectness/framing with
three judge families. Output: divergence_study_outputs/judge_panel_raw.csv

Usage:
  python -m scripts.judge_panel --smoke
  python -m scripts.judge_panel --n 60 --workers 6
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.elephant_scorers import score_response
from scripts.load_elephant import ELEPHANT_SEED, load_elephant
from scripts.run_phase1_quartet import OUT_DIR

PANEL_JUDGES = [
    "gpt-5.4-nano",
    "claude-haiku-4-5",
    "grok-4-1-fast-reasoning",
]
METRICS = ["validation", "indirectness", "framing"]
SA_CSV = OUT_DIR / "elephant_singleagent_raw.csv"
OUT_CSV = OUT_DIR / "judge_panel_raw.csv"
SEED = 1517

# Arms to stratify across (include Phase 14 optimized if present in CSV)
STRAT_ARMS = [
    "raw", "standard_cot", "narrative_cot",
    "narrative_cot_v2", "narrative_cot_v3",
]


def _panel_cache_path(item_id: str, arm: str, gen: str, metric: str, judge: str) -> Path:
    h = hashlib.sha1(f"{item_id}|{arm}|{gen}|{metric}|{judge}".encode()).hexdigest()[:12]
    return OUT_DIR / f"judge_panel_{_safe(judge)}_{metric}_{h}.json"


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:40]


def _prompt_map() -> dict[str, str]:
    items = load_elephant(
        "oeq", n=150, seed=ELEPHANT_SEED,
        data_dir=Path("data/elephant"), allow_sample=True,
    )
    return {item.id: item.prompt for item in items}


def _load_oeq_sample(n: int, seed: int) -> list[dict]:
    if not SA_CSV.exists():
        raise FileNotFoundError(f"Missing {SA_CSV}; run run_elephant first.")
    prompts = _prompt_map()
    with SA_CSV.open(newline="", encoding="utf-8") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r.get("dataset") == "oeq"
            and r.get("generator") not in ("human", "")
            and r.get("response", "").strip()
            and r.get("empty_response") not in ("1", 1)
        ]
    if not rows:
        raise ValueError("No OEQ model rows in elephant CSV.")

    for r in rows:
        r["prompt"] = prompts.get(r.get("item_id", ""), "")

    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        arm = r.get("arm", "")
        if arm in STRAT_ARMS:
            by_arm.setdefault(arm, []).append(r)

    rng = random.Random(seed)
    per_arm = max(1, n // max(1, len(by_arm)))
    sample = []
    for arm, grp in sorted(by_arm.items()):
        rng.shuffle(grp)
        take = min(per_arm, len(grp))
        sample.extend(grp[:take])
    rng.shuffle(sample)
    return sample[:n]


def _score_cell(row: dict, metric: str, judge: str) -> dict:
    item_id = row["item_id"]
    arm = row["arm"]
    gen = row["generator"]
    cache = _panel_cache_path(item_id, arm, gen, metric, judge)
    if cache.exists():
        obj = json.loads(cache.read_text())
        score = int(obj.get("score", -1))
    else:
        prompt = row.get("prompt", "") or ""
        if not prompt:
            # Reconstruct from response context if prompt not stored
            prompt = row.get("response_preview", "")[:200]
        response = row.get("response", "")
        score = score_response(metric, prompt, response, judge=judge)
        cache.write_text(json.dumps({"score": score, "judge": judge}, ensure_ascii=False))

    return {
        "item_id": item_id,
        "generator": gen,
        "arm": arm,
        "metric": metric,
        "judge": judge,
        "score": score,
        "original_score": row.get(f"score_{metric}", row.get(f"sycophantic_{metric}", "")),
        "response_len": row.get("response_len", len(row.get("response", ""))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 15 judge reliability panel")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n = 6
        args.workers = 2

    sample = _load_oeq_sample(args.n, args.seed)
    print(f"Judge panel: {len(sample)} OEQ responses, judges={PANEL_JUDGES}", flush=True)

    tasks = [
        (row, metric, judge)
        for row in sample
        for metric in METRICS
        for judge in PANEL_JUDGES
    ]
    print(f"  scoring tasks: {len(tasks)}", flush=True)

    out_rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_score_cell, row, metric, judge): (row["item_id"], metric, judge)
                for row, metric, judge in tasks}
        for fut in as_completed(futs):
            try:
                out_rows.append(fut.result())
            except Exception as e:
                print(f"  ERROR {futs[fut]}: {e}", flush=True)
            done += 1
            if done % 50 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)}", flush=True)

    if not out_rows:
        print("No panel scores collected.")
        return 1

    fieldnames = sorted({k for r in out_rows for k in r})
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
