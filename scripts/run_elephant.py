"""
scripts/run_elephant.py -- Phase 12 ELEPHANT single-agent evaluation.

Generates advice for baseline_io, standard_cot, narrative_cot across ELEPHANT
datasets, scores with faithful ELEPHANT metrics, and scores human baselines.

Output: divergence_study_outputs/elephant_singleagent_raw.csv

Usage:
  python -m scripts.run_elephant --smoke
  python -m scripts.run_elephant --datasets oeq,aita_yta,ss --n 150 --workers 6
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

from scripts.elephant_scorers import DEFAULT_JUDGE, moral_both_nta, score_response
from scripts.generators import generate
from scripts.load_elephant import (
    ELEPHANT_SEED, _load_flip_pairs, load_elephant, metrics_for_dataset,
)
from scripts.run_phase1_quartet import PROMPTS, _safe, OUT_DIR

BUDGET_GENERATORS = ["gpt-5.4-nano", "claude-haiku-4-5", "grok-4-1-fast-reasoning"]
SINGLE_AGENT_ARMS = ["baseline_io", "standard_cot", "narrative_cot"]
AITA_BINARY_SUFFIX = "\nOutput only YTA or NTA."

OUT_CSV = OUT_DIR / "elephant_singleagent_raw.csv"


def _gen_cache_path(dataset: str, item_id: str, gen: str, arm: str) -> Path:
    return OUT_DIR / f"elephant_gen_{dataset}_{_safe(gen)}_{arm}_{_safe(item_id)}.json"


def _generate_advice(
    prompt: str,
    gen_model: str,
    arm: str,
    dataset: str,
    item_id: str,
    aita_binary: bool = False,
) -> str:
    cache = _gen_cache_path(dataset, item_id, gen_model, arm)
    if cache.exists():
        cached = json.loads(cache.read_text()).get("response", "")
        if cached.strip():
            return cached
    sys_prompt = PROMPTS.get(arm, PROMPTS["standard_cot"])
    user_turn = prompt + (AITA_BINARY_SUFFIX if aita_binary else "")
    text = ""
    for attempt in range(3):
        result = generate(gen_model, sys_prompt, user_turn, sample_idx=attempt, max_tokens=1024)
        text = (result.text or "").strip()
        if text:
            break
    if text:
        cache.write_text(json.dumps({
            "response": text,
            "completion_tokens": result.completion_tokens,
        }, ensure_ascii=False))
    return text


def _score_all_metrics(
    dataset: str,
    prompt: str,
    response: str,
    judge: str,
) -> dict[str, int]:
    scores = {}
    for metric in metrics_for_dataset(dataset):
        if metric == "moral":
            continue
        s = score_response(metric, prompt, response, judge=judge)
        scores[metric] = s
    return scores


def _run_cell(
    dataset: str,
    item_id: str,
    prompt: str,
    human_response: str,
    human_scores: dict,
    gen_model: str,
    arm: str,
    judge: str,
    pair_id: str = "",
    side: str = "",
) -> dict:
    aita_binary = dataset == "flip_pairs"
    response = _generate_advice(
        prompt, gen_model, arm, dataset, item_id,
        aita_binary=aita_binary,
    )
    row = {
        "dataset": dataset,
        "item_id": item_id,
        "pair_id": pair_id,
        "side": side,
        "generator": gen_model,
        "arm": arm,
        "prompt_len": len(prompt),
        "response_len": len(response),
        "response": response,
        "response_preview": response[:200],
    }
    for metric, val in _score_all_metrics(dataset, prompt, response, judge).items():
        row[f"score_{metric}"] = val
        row[f"sycophantic_{metric}"] = val  # 1 = sycophantic in ELEPHANT convention
    for m, v in human_scores.items():
        row[f"human_{m}"] = v
    return row


def _run_human_baseline(
    dataset: str,
    item_id: str,
    prompt: str,
    human_response: str,
    human_scores: dict,
    judge: str,
) -> dict | None:
    if not human_response:
        return None
    row = {
        "dataset": dataset,
        "item_id": item_id,
        "pair_id": "",
        "side": "",
        "generator": "human",
        "arm": "human_baseline",
        "prompt_len": len(prompt),
        "response_len": len(human_response),
        "response_preview": human_response[:200],
    }
    for metric in metrics_for_dataset(dataset):
        if metric == "moral":
            continue
        if metric in human_scores:
            row[f"score_{metric}"] = human_scores[metric]
            row[f"sycophantic_{metric}"] = human_scores[metric]
        else:
            s = score_response(metric, prompt, human_response, judge=judge)
            row[f"score_{metric}"] = s
            row[f"sycophantic_{metric}"] = s
    return row


def _build_flip_moral_rows(
    pair_rows: list[dict],
    judge: str,
) -> list[dict]:
    """Aggregate OG+FLIP responses per (pair, gen, arm) into moral sycophancy rows."""
    by_key: dict[tuple, dict] = {}
    for r in pair_rows:
        key = (r["pair_id"], r["generator"], r["arm"])
        by_key.setdefault(key, {})[r["side"]] = r

    out = []
    for (pair_id, gen, arm), sides in by_key.items():
        if "og" not in sides or "flip" not in sides:
            continue
        og_text = sides["og"].get("response", "") or ""
        flip_text = sides["flip"].get("response", "") or ""
        both_nta = moral_both_nta(og_text, flip_text)
        out.append({
            "dataset": "flip_pairs",
            "item_id": pair_id,
            "pair_id": pair_id,
            "side": "pair",
            "generator": gen,
            "arm": arm,
            "score_moral": both_nta,
            "sycophantic_moral": both_nta,
            "og_yta_nta": sides["og"].get("response_preview", "")[:50],
            "flip_yta_nta": sides["flip"].get("response_preview", "")[:50],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 12 ELEPHANT single-agent eval")
    ap.add_argument("--datasets", default="oeq,aita_yta,ss,flip_pairs")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--generators", default=",".join(BUDGET_GENERATORS))
    ap.add_argument("--arms", default=",".join(SINGLE_AGENT_ARMS))
    ap.add_argument("--judge", default=os.environ.get("ELEPHANT_JUDGE", DEFAULT_JUDGE))
    ap.add_argument("--seed", type=int, default=ELEPHANT_SEED)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="10-item sample datasets")
    args = ap.parse_args()

    if args.smoke:
        args.n = 10

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    judge = args.judge

    print(f"ELEPHANT single-agent: datasets={datasets} n={args.n} arms={arms}", flush=True)
    print(f"  generators={generators} judge={judge}", flush=True)

    tasks = []
    human_tasks = []

    for ds in datasets:
        if ds == "flip_pairs":
            items = _load_flip_pairs(n=args.n, seed=args.seed, data_dir=Path("data/elephant"))
        else:
            items = load_elephant(ds, n=args.n, seed=args.seed)
        for item in items:
            if item.human_response:
                human_tasks.append((ds, item))
            for gen in generators:
                for arm in arms:
                    tasks.append((ds, item, gen, arm))

    print(f"  generation tasks: {len(tasks)} | human baseline: {len(human_tasks)}", flush=True)

    rows: list[dict] = []
    done = 0
    pair_gen_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for ds, item, gen, arm in tasks:
            futs[ex.submit(
                _run_cell,
                ds, item.id, item.prompt, item.human_response, item.human_scores,
                gen, arm, judge, item.pair_id, item.side,
            )] = ("cell", ds, item.id, gen, arm)

        for ds, item in human_tasks:
            futs[ex.submit(
                _run_human_baseline,
                ds, item.id, item.prompt, item.human_response, item.human_scores, judge,
            )] = ("human", ds, item.id)

        for fut in as_completed(futs):
            tag = futs[fut]
            try:
                row = fut.result()
                if row:
                    rows.append(row)
                    if tag[0] == "cell" and tag[1] == "flip_pairs":
                        pair_gen_rows.append(row)
            except Exception as e:
                print(f"  ERROR {tag}: {e}", flush=True)
            done += 1
            if done % 100 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)}", flush=True)

    moral_rows = _build_flip_moral_rows(pair_gen_rows, judge)
    rows.extend(moral_rows)
    print(f"  moral pair rows: {len(moral_rows)}", flush=True)

    if not rows:
        print("No rows collected.")
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
