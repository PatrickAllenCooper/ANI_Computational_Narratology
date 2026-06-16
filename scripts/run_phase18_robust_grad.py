"""
scripts/run_phase18_robust_grad.py -- Phase 18 panel-robust narrative gradients.

Textual-gradient descent initialised at verbatim N-CoT, optimising the
WORST-CASE-over-judges sycophancy loss on the disjoint OEQ train split
(offset 150). The robust objective only credits a change when every judge in
the optimisation panel agrees it improved, so it cannot Goodhart a single
lenient judge (cf. Phase 18a, where the single-judge indirectness "win"
vanished on held-out judges).

Holdout evaluation compares three prompts -- hand NoT, the Phase 14 single-judge
winner, and this robust prompt -- across the optimisation panel plus a held-out
judge from a distinct family, and reports the Goodhart gap for each.

Usage:
  python -m scripts.run_phase18_robust_grad [--smoke]
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

from scripts.run_phase1_quartet import PROMPTS
from scripts.syco_loss import (
    BATCH_SIZE,
    DEFAULT_GEN_MODEL,
    DEFAULT_OPTIMISER,
    INIT_PROMPTS,
    MAX_ITERS,
    OUT_DIR,
    TRAIN_N,
    batch_for_iter,
    iter_path,
    load_holdout_oeq,
    load_train_items,
    panel_batch_loss,
    panel_early_stop,
    panel_forward_batch,
    robust_textual_gradient,
    summary_path,
    update_prompt,
)

NS = "rg"
INITIAL_PROMPT = INIT_PROMPTS["narrative_cot"]

# Optimisation panel (judges the loss sees) and a held-out judge (eval only).
PANEL = ["grok-4-1-fast-reasoning", "claude-haiku-4-5", "gpt-5.4-nano"]
HELDOUT_JUDGE = "claude-sonnet-4-6"

# Reuse Phase 18a / Phase 14 cached generations for the two reference prompts.
HAND_NS = "sg_ref_narrative_cot"
SINGLE_JUDGE_HOLDOUT_NS = f"sg_holdout_sg_{DEFAULT_GEN_MODEL}"


def run_optimization(
    train,
    gen_model: str,
    judges: list[str],
    optimiser_model: str,
    n_iters: int,
    batch_size: int,
    max_tokens: int,
) -> tuple[str, list[dict]]:
    prompt = INITIAL_PROMPT
    history: list[dict] = []

    t0 = time.time()
    batch0 = batch_for_iter(train, 0, batch_size)
    coded0 = panel_forward_batch(prompt, batch0, gen_model, judges, NS, max_tokens)
    stats0 = panel_batch_loss(coded0, judges)
    rec0 = {
        "iter": 0, "prompt": prompt, "coded": coded0, "stats": stats0,
        "gradient": None, "wall_s": time.time() - t0,
    }
    iter_path(NS, 0).write_text(json.dumps(rec0, ensure_ascii=False, indent=2))
    history.append(rec0)
    _print_iter(0, stats0, None)

    for it in range(1, n_iters + 1):
        ts = time.time()
        prev = history[-1]
        gradient = robust_textual_gradient(
            prev["prompt"], prev["stats"], prev["coded"], judges, optimiser_model,
        )
        new_prompt = update_prompt(prev["prompt"], gradient, optimiser_model)
        batch = batch_for_iter(train, it, batch_size)
        coded = panel_forward_batch(new_prompt, batch, gen_model, judges, NS, max_tokens)
        stats = panel_batch_loss(coded, judges)
        record = {
            "iter": it, "prompt": new_prompt, "coded": coded, "stats": stats,
            "gradient": gradient, "wall_s": time.time() - ts,
        }
        iter_path(NS, it).write_text(json.dumps(record, ensure_ascii=False, indent=2))
        history.append(record)
        _print_iter(it, stats, prev["stats"]["robust_loss"])
        if panel_early_stop(history):
            print(f"  [early-stop] at iter {it}", flush=True)
            break

    best = min(history, key=lambda r: r["stats"]["robust_loss"])
    return best["prompt"], history


def _print_iter(it: int, stats: dict, prev_loss: float | None) -> None:
    pm = stats["panel_max_rate"]
    delta = "" if prev_loss is None else f" delta={prev_loss - stats['robust_loss']:+.4f}"
    print(
        f"[iter {it}] robust_loss={stats['robust_loss']:.4f}{delta}  "
        f"panel-max v={pm['validation']:.0%} ind={pm['indirectness']:.0%} "
        f"f={pm['framing']:.0%}",
        flush=True,
    )


def eval_holdout(prompt: str, items, gen_model: str, ns: str) -> dict:
    judges = [*PANEL, HELDOUT_JUDGE]
    coded = panel_forward_batch(prompt, items, gen_model, judges, ns)
    stats = panel_batch_loss(coded, PANEL)
    heldout = panel_batch_loss(coded, [HELDOUT_JUDGE])
    return {
        "per_judge": stats["per_judge"],
        "panel_max_rate": stats["panel_max_rate"],
        "robust_loss": stats["robust_loss"],
        "heldout_judge": heldout["per_judge"][HELDOUT_JUDGE],
        "n": stats["n"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 18 panel-robust narrative gradients")
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
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
    print(
        f"Train: {len(train)} OEQ (offset 150) | Holdout: {len(holdout)} OEQ | "
        f"panel={PANEL} | held-out={HELDOUT_JUDGE}",
        flush=True,
    )

    final_prompt, history = run_optimization(
        train, args.gen_model, PANEL, args.optimiser_model,
        args.n_iters, args.batch_size, args.max_tokens,
    )

    print("\nHoldout evaluation (panel + held-out judge)...", flush=True)
    single_judge_prompt = json.loads(
        (OUT_DIR / "phase14_holdout.json").read_text()
    )["winner_prompt"]

    configs = {
        "hand_ncot": (INITIAL_PROMPT, HAND_NS),
        "single_judge": (single_judge_prompt, SINGLE_JUDGE_HOLDOUT_NS),
        "robust": (final_prompt, f"{NS}_holdout_{args.gen_model}"),
    }
    holdout_eval = {
        label: eval_holdout(prompt, holdout, args.gen_model, ns)
        for label, (prompt, ns) in configs.items()
    }

    def _gap(label: str, metric: str) -> dict:
        hand = holdout_eval["hand_ncot"]
        cfg = holdout_eval[label]
        train_red = (
            hand["per_judge"]["grok-4-1-fast-reasoning"][metric]
            - cfg["per_judge"]["grok-4-1-fast-reasoning"][metric]
        )
        heldout_red = hand["heldout_judge"][metric] - cfg["heldout_judge"][metric]
        return {
            "train_judge_reduction": train_red,
            "heldout_judge_reduction": heldout_red,
            "goodhart_gap": train_red - heldout_red,
            "panel_max_rate": cfg["panel_max_rate"][metric],
            "heldout_rate": cfg["heldout_judge"][metric],
        }

    goodhart = {
        label: {m: _gap(label, m) for m in ("validation", "indirectness", "framing")}
        for label in ("single_judge", "robust")
    }

    summary = {
        "optimizer": "robust_narrative_grad",
        "namespace": NS,
        "gen_model": args.gen_model,
        "panel": PANEL,
        "heldout_judge": HELDOUT_JUDGE,
        "optimiser_model": args.optimiser_model,
        "initial_prompt": INITIAL_PROMPT,
        "final_prompt": final_prompt,
        "n_iters_run": len(history) - 1,
        "training_loss_curve": [
            {
                "iter": r["iter"],
                "robust_loss": r["stats"]["robust_loss"],
                "panel_max_rate": r["stats"]["panel_max_rate"],
            }
            for r in history
        ],
        "holdout_eval": holdout_eval,
        "goodhart_gap": goodhart,
    }
    summary_path(NS).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT_DIR / "phase18_robust.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    print(f"\nWrote {summary_path(NS)}")
    print("\n=== Holdout panel-max rates (lower = robust) ===")
    print(f"{'config':<14} {'val':>6} {'ind':>6} {'fram':>6} {'r_loss':>7} {'held-v':>7}")
    for label, ev in holdout_eval.items():
        pm = ev["panel_max_rate"]
        print(
            f"{label:<14} {pm['validation']:>6.0%} {pm['indirectness']:>6.0%} "
            f"{pm['framing']:>6.0%} {ev['robust_loss']:>7.3f} "
            f"{ev['heldout_judge']['validation']:>7.0%}"
        )
    print("\n=== Goodhart gap (validation) ===")
    for label in ("single_judge", "robust"):
        g = goodhart[label]["validation"]
        print(
            f"{label:<14} train_red={g['train_judge_reduction']:>6.1%}  "
            f"heldout_red={g['heldout_judge_reduction']:>6.1%}  "
            f"gap={g['goodhart_gap']:>6.1%}  held-rate={g['heldout_rate']:>5.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
