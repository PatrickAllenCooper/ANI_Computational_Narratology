"""
scripts/run_phase14_syco_grad.py -- Phase 14: narrative gradients with sycophancy loss.

Textual-gradient descent initialised at verbatim N-CoT, optimising ELEPHANT
validation+indirectness+framing on disjoint OEQ train split (offset 150).

Usage:
  python -m scripts.run_phase14_syco_grad [--smoke]
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
    DEFAULT_THIRD_JUDGE,
    DEFAULT_TRAIN_JUDGE,
    INIT_PROMPTS,
    MAX_ITERS,
    TRAIN_N,
    batch_for_iter,
    batch_loss,
    deliberative_depth_batch,
    early_stop,
    eval_holdout_moral,
    eval_holdout_oeq,
    forward_batch,
    iter_path,
    load_holdout_flip_pairs,
    load_holdout_oeq,
    load_train_items,
    summary_path,
    textual_gradient,
    update_prompt,
)

NS = "sg"
INITIAL_PROMPT = INIT_PROMPTS["narrative_cot"]


def run_optimization(
    train,
    gen_model: str,
    judge_model: str,
    optimiser_model: str,
    n_iters: int,
    batch_size: int,
    max_tokens: int,
) -> tuple[str, list[dict]]:
    prompt = INITIAL_PROMPT
    history: list[dict] = []

    t0 = time.time()
    batch0 = batch_for_iter(train, 0, batch_size)
    coded0 = forward_batch(prompt, batch0, gen_model, judge_model, NS, max_tokens)
    stats0 = batch_loss(coded0)
    rec0 = {
        "iter": 0,
        "prompt": prompt,
        "coded": coded0,
        "stats": stats0,
        "gradient": None,
        "wall_s": time.time() - t0,
    }
    iter_path(NS, 0).write_text(json.dumps(rec0, ensure_ascii=False, indent=2))
    history.append(rec0)
    print(
        f"[iter 0] loss={stats0['loss']:.4f}  "
        f"v={stats0['validation_rate']:.1%}  "
        f"ind={stats0['indirectness_rate']:.1%}  "
        f"f={stats0['framing_rate']:.1%}",
        flush=True,
    )

    for it in range(1, n_iters + 1):
        ts = time.time()
        prev = history[-1]
        gradient = textual_gradient(
            prev["prompt"], prev["stats"], prev["coded"], optimiser_model,
        )
        new_prompt = update_prompt(prev["prompt"], gradient, optimiser_model)
        batch = batch_for_iter(train, it, batch_size)
        coded = forward_batch(new_prompt, batch, gen_model, judge_model, NS, max_tokens)
        stats = batch_loss(coded)
        record = {
            "iter": it,
            "prompt": new_prompt,
            "coded": coded,
            "stats": stats,
            "gradient": gradient,
            "wall_s": time.time() - ts,
        }
        iter_path(NS, it).write_text(json.dumps(record, ensure_ascii=False, indent=2))
        history.append(record)
        delta = prev["stats"]["loss"] - stats["loss"]
        print(
            f"[iter {it}] loss={stats['loss']:.4f}  delta={delta:+.4f}  "
            f"len={len(new_prompt)}",
            flush=True,
        )
        if early_stop(history):
            print(f"  [early-stop] at iter {it}", flush=True)
            break

    return history[-1]["prompt"], history


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 14 narrative gradients (syco loss)")
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--judge-model", default=DEFAULT_TRAIN_JUDGE)
    ap.add_argument("--third-judge", default=DEFAULT_THIRD_JUDGE)
    ap.add_argument("--optimiser-model", default=DEFAULT_OPTIMISER)
    ap.add_argument("--n-iters", type=int, default=MAX_ITERS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--train-n", type=int, default=TRAIN_N)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-sample", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_iters = 1
        args.batch_size = 3
        args.train_n = 6
        args.allow_sample = True

    train = load_train_items(n=args.train_n, allow_sample=args.allow_sample)
    holdout = load_holdout_oeq(allow_sample=args.allow_sample)
    if args.smoke:
        holdout = holdout[:3]
    flip_items = load_holdout_flip_pairs(
        n=10 if args.smoke else 150, allow_sample=args.allow_sample,
    )

    print(f"Train: {len(train)} OEQ (offset 150) | Holdout: {len(holdout)} OEQ", flush=True)

    final_prompt, history = run_optimization(
        train, args.gen_model, args.judge_model, args.optimiser_model,
        args.n_iters, args.batch_size, args.max_tokens,
    )

    hand_holdout = eval_holdout_oeq(
        INITIAL_PROMPT, holdout, args.gen_model, args.judge_model, f"{NS}_hand",
    )
    opt_holdout = eval_holdout_oeq(
        final_prompt, holdout, args.gen_model, args.judge_model, NS,
    )
    third_holdout = eval_holdout_oeq(
        final_prompt, holdout, args.gen_model, args.third_judge, f"{NS}_third",
    )
    hand_moral = eval_holdout_moral(
        INITIAL_PROMPT, flip_items, args.gen_model, f"{NS}_hand",
    )
    opt_moral = eval_holdout_moral(final_prompt, flip_items, args.gen_model, NS)
    hand_depth = deliberative_depth_batch(
        INITIAL_PROMPT, holdout[:min(30, len(holdout))],
        args.gen_model, args.third_judge, f"{NS}_hand_depth",
    )
    opt_depth = deliberative_depth_batch(
        final_prompt, holdout[:min(30, len(holdout))],
        args.gen_model, args.third_judge, f"{NS}_opt_depth",
    )

    summary = {
        "optimizer": "narrative_grad",
        "namespace": NS,
        "gen_model": args.gen_model,
        "judge_model": args.judge_model,
        "third_judge": args.third_judge,
        "optimiser_model": args.optimiser_model,
        "initial_prompt": INITIAL_PROMPT,
        "final_prompt": final_prompt,
        "n_iters_run": len(history) - 1,
        "training_loss_curve": [
            {"iter": r["iter"], "loss": r["stats"]["loss"], **{
                k: r["stats"][k] for k in (
                    "validation_rate", "indirectness_rate", "framing_rate",
                ) if k in r["stats"]
            }}
            for r in history
        ],
        "holdout_oeq": {
            "hand_ncot": hand_holdout["stats"],
            "optimised": opt_holdout["stats"],
            "third_judge": third_holdout["stats"],
        },
        "holdout_moral": {
            "hand_ncot": hand_moral,
            "optimised": opt_moral,
        },
        "degeneracy_guard": {
            "hand_ncot": hand_depth,
            "optimised": opt_depth,
        },
    }
    summary_path(NS).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {summary_path(NS)}")
    print(
        f"Holdout loss: hand={hand_holdout['stats']['loss']:.3f}  "
        f"opt={opt_holdout['stats']['loss']:.3f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
