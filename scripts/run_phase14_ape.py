"""
scripts/run_phase14_ape.py -- Phase 14: APE baseline with sycophancy loss.

Automatic Prompt Engineering: generate N candidate prompts, score each on
train batch, select argmin.

Usage:
  python -m scripts.run_phase14_ape [--smoke]
"""
from __future__ import annotations

import argparse
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.syco_loss import (
    DEFAULT_GEN_MODEL,
    DEFAULT_OPTIMISER,
    DEFAULT_TRAIN_JUDGE,
    INIT_PROMPTS,
    TRAIN_N,
    ape_candidate,
    batch_for_iter,
    batch_loss,
    eval_holdout_moral,
    eval_holdout_oeq,
    forward_batch,
    load_holdout_flip_pairs,
    load_holdout_oeq,
    load_train_items,
    summary_path,
)

NS = "sg_ape"
INITIAL_PROMPT = INIT_PROMPTS["narrative_cot"]
N_CANDIDATES = 10


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 14 APE baseline")
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--judge-model", default=DEFAULT_TRAIN_JUDGE)
    ap.add_argument("--optimiser-model", default=DEFAULT_OPTIMISER)
    ap.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    ap.add_argument("--train-n", type=int, default=TRAIN_N)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-sample", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_candidates = 3
        args.train_n = 6
        args.allow_sample = True

    train = load_train_items(n=args.train_n, allow_sample=args.allow_sample)
    holdout = load_holdout_oeq(allow_sample=args.allow_sample)
    if args.smoke:
        holdout = holdout[:3]
    flip_items = load_holdout_flip_pairs(n=10 if args.smoke else 150, allow_sample=args.allow_sample)
    batch = batch_for_iter(train, 0, min(10, len(train)))

    candidates = [INITIAL_PROMPT]
    for i in range(1, args.n_candidates):
        cand = ape_candidate(INITIAL_PROMPT, i, args.n_candidates, args.optimiser_model)
        candidates.append(cand)
        print(f"  candidate {i}: len={len(cand)}", flush=True)

    scored = []
    for idx, prompt in enumerate(candidates):
        coded = forward_batch(prompt, batch, args.gen_model, args.judge_model, NS, args.max_tokens)
        stats = batch_loss(coded)
        scored.append({"idx": idx, "prompt": prompt, "stats": stats})
        print(f"  score cand {idx}: loss={stats['loss']:.4f}", flush=True)

    best = min(scored, key=lambda r: r["stats"]["loss"])
    final_prompt = best["prompt"]

    hand_holdout = eval_holdout_oeq(
        INITIAL_PROMPT, holdout, args.gen_model, args.judge_model, f"{NS}_hand",
    )
    opt_holdout = eval_holdout_oeq(final_prompt, holdout, args.gen_model, args.judge_model, NS)
    opt_moral = eval_holdout_moral(final_prompt, flip_items, args.gen_model, NS)

    summary = {
        "optimizer": "ape",
        "namespace": NS,
        "n_candidates": args.n_candidates,
        "initial_prompt": INITIAL_PROMPT,
        "final_prompt": final_prompt,
        "candidate_scores": [
            {"idx": r["idx"], "loss": r["stats"]["loss"]} for r in scored
        ],
        "holdout_oeq": {"hand_ncot": hand_holdout["stats"], "optimised": opt_holdout["stats"]},
        "holdout_moral": {"optimised": opt_moral},
    }
    summary_path(NS).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_path(NS)} | best idx={best['idx']} loss={best['stats']['loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
