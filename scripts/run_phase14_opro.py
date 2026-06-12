"""
scripts/run_phase14_opro.py -- Phase 14: OPRO baseline with sycophancy loss.

Optimization by Prompting: meta-prompt over scored prompt trajectory.

Usage:
  python -m scripts.run_phase14_opro [--smoke]
"""
from __future__ import annotations

import argparse
import json
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.syco_loss import (
    BATCH_SIZE,
    DEFAULT_GEN_MODEL,
    DEFAULT_OPTIMISER,
    DEFAULT_TRAIN_JUDGE,
    INIT_PROMPTS,
    MAX_ITERS,
    TRAIN_N,
    batch_for_iter,
    batch_loss,
    eval_holdout_moral,
    eval_holdout_oeq,
    forward_batch,
    iter_path,
    load_holdout_flip_pairs,
    load_holdout_oeq,
    load_train_items,
    opro_next_prompt,
    summary_path,
)

NS = "sg_opro"
INITIAL_PROMPT = INIT_PROMPTS["narrative_cot"]
HAND_COMPARATOR = INITIAL_PROMPT


def run_opro(train, gen_model, judge_model, optimiser_model, n_iters, batch_size, max_tokens):
    prompt = INITIAL_PROMPT
    history = []
    trajectory: list[dict] = []

    t0 = time.time()
    batch0 = batch_for_iter(train, 0, batch_size)
    coded0 = forward_batch(prompt, batch0, gen_model, judge_model, NS, max_tokens)
    stats0 = batch_loss(coded0)
    trajectory.append({
        "prompt": prompt, "loss": stats0["loss"],
        "validation_rate": stats0["validation_rate"],
    })
    history.append({
        "iter": 0, "prompt": prompt, "coded": coded0, "stats": stats0,
        "wall_s": time.time() - t0,
    })
    iter_path(NS, 0).write_text(json.dumps(history[-1], ensure_ascii=False, indent=2))
    print(f"[iter 0] loss={stats0['loss']:.4f}", flush=True)

    for it in range(1, n_iters + 1):
        ts = time.time()
        new_prompt = opro_next_prompt(trajectory, INITIAL_PROMPT, optimiser_model)
        batch = batch_for_iter(train, it, batch_size)
        coded = forward_batch(new_prompt, batch, gen_model, judge_model, NS, max_tokens)
        stats = batch_loss(coded)
        trajectory.append({
            "prompt": new_prompt, "loss": stats["loss"],
            "validation_rate": stats["validation_rate"],
        })
        record = {
            "iter": it, "prompt": new_prompt, "coded": coded, "stats": stats,
            "wall_s": time.time() - ts,
        }
        iter_path(NS, it).write_text(json.dumps(record, ensure_ascii=False, indent=2))
        history.append(record)
        print(f"[iter {it}] loss={stats['loss']:.4f}", flush=True)

    best = min(trajectory, key=lambda r: r["loss"])
    return best["prompt"], history


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 14 OPRO baseline")
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--judge-model", default=DEFAULT_TRAIN_JUDGE)
    ap.add_argument("--optimiser-model", default=DEFAULT_OPTIMISER)
    ap.add_argument("--n-iters", type=int, default=MAX_ITERS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--train-n", type=int, default=TRAIN_N)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-sample", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_iters = 2
        args.batch_size = 3
        args.train_n = 6
        args.allow_sample = True

    train = load_train_items(n=args.train_n, allow_sample=args.allow_sample)
    holdout = load_holdout_oeq(allow_sample=args.allow_sample)
    if args.smoke:
        holdout = holdout[:3]
    flip_items = load_holdout_flip_pairs(n=10 if args.smoke else 150, allow_sample=args.allow_sample)

    final_prompt, history = run_opro(
        train, args.gen_model, args.judge_model, args.optimiser_model,
        args.n_iters, args.batch_size, args.max_tokens,
    )

    hand_holdout = eval_holdout_oeq(
        HAND_COMPARATOR, holdout, args.gen_model, args.judge_model, f"{NS}_hand",
    )
    opt_holdout = eval_holdout_oeq(final_prompt, holdout, args.gen_model, args.judge_model, NS)
    opt_moral = eval_holdout_moral(final_prompt, flip_items, args.gen_model, NS)

    summary = {
        "optimizer": "opro",
        "namespace": NS,
        "final_prompt": final_prompt,
        "n_iters_run": len(history) - 1,
        "training_loss_curve": [{"iter": r["iter"], "loss": r["stats"]["loss"]} for r in history],
        "holdout_oeq": {"hand_ncot": hand_holdout["stats"], "optimised": opt_holdout["stats"]},
        "holdout_moral": {"optimised": opt_moral},
    }
    summary_path(NS).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_path(NS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
