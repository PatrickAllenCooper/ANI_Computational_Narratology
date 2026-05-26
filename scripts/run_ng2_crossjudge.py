"""
scripts/run_ng2_crossjudge.py -- Phase 10b: Cross-judge Narrative-Gradients.

Identical to run_phase10_narrative_grad.py (Phase 10) in all respects except:

  Training judge : grok-4-1-fast-reasoning  (was claude-haiku-4-5)
  Third judge    : claude-haiku-4-5          (was grok-4-1-fast-reasoning)
  Cache namespace: ng2_*                     (never collides with ng_*)
  Summary file   : ng2_summary.json

Motivation: Phase 10 produced v2 with kappa=0.000 between haiku-judge and
grok-judge on the four-generator replication, indicating the optimiser
learned to satisfy the haiku rubric specifically rather than the underlying
deliberative quality.  Using grok as the training judge breaks the
exploitation loop. The resulting prompt (v3) is evaluated under haiku as
primary judge; if kappa between haiku and grok on v3 outputs is now high,
the improvement is genuine and cross-family.

Usage:
  python -m scripts.run_ng2_crossjudge [--n-iters 10] [--batch-size 10]
                                        [--eval-start 30] [--eval-end 60]
                                        [--max-tokens 4096]
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from scripts.run_phase10_narrative_grad import (
    # reused logic -- no cache paths
    INITIAL_PROMPT,
    _prompt_hash,
    _safe,
    pick_subsample,
    batch_loss_continuous,
    early_stop,
    textual_gradient,
    update_prompt,
    cohens_kappa,
    GRADIENT_SYSTEM,
    GRADIENT_USER_TEMPLATE,
    UPDATE_SYSTEM,
    UPDATE_USER_TEMPLATE,
    _cell_loss,
    drift_control_cell,
)
from scripts.generators import generate
from scripts.run_phase1_quartet import (
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    Scenario,
    load_daily_dilemmas,
    _extract_json,
)
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_DIR = Path("./divergence_study_outputs")
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_GEN_MODEL   = "claude-haiku-4-5"
DEFAULT_JUDGE_MODEL = "grok-4-1-fast-reasoning"   # CROSS-FAMILY training judge
DEFAULT_THIRD_JUDGE = "claude-haiku-4-5"           # reversed: haiku is now third
DEFAULT_OPTIMISER   = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Cache helpers (ng2_* namespace)
# ---------------------------------------------------------------------------

def ng2_gen_path(scenario_id: str, prompt_hash: str, gen_model: str) -> Path:
    return OUT_DIR / f"ng2_gen_{_safe(gen_model)}_{scenario_id}_{prompt_hash}.json"


def ng2_judge_primary_path(
    scenario_id: str, prompt_hash: str, gen_model: str, judge_model: str
) -> Path:
    return OUT_DIR / (
        f"ng2_judge_p_{_safe(judge_model)}_{_safe(gen_model)}"
        f"_{scenario_id}_{prompt_hash}.json"
    )


def ng2_judge_third_path(
    scenario_id: str, prompt_hash: str, gen_model: str, third_judge: str
) -> Path:
    return OUT_DIR / (
        f"ng2_judge_t_{_safe(third_judge)}_{_safe(gen_model)}"
        f"_{scenario_id}_{prompt_hash}.json"
    )


def ng2_iter_path(iter_idx: int) -> Path:
    return OUT_DIR / f"ng2_iter_{iter_idx:02d}.json"


def ng2_summary_path() -> Path:
    return OUT_DIR / "ng2_summary.json"


def ng2_drift_path(label: str) -> Path:
    return OUT_DIR / f"ng2_drift_{label}.json"


# ---------------------------------------------------------------------------
# Forward pass with ng2 cache namespace
# ---------------------------------------------------------------------------

def code_one(
    prompt_text: str,
    scenario: Scenario,
    gen_model: str,
    judge_model: str,
    max_tokens: int,
    *,
    is_third_judge: bool = False,
) -> dict:
    phash      = _prompt_hash(prompt_text)
    gen_path   = ng2_gen_path(scenario.id, phash, gen_model)
    judge_path = (
        ng2_judge_third_path(scenario.id, phash, gen_model, judge_model)
        if is_third_judge
        else ng2_judge_primary_path(scenario.id, phash, gen_model, judge_model)
    )

    if gen_path.exists():
        gen_rec = json.loads(gen_path.read_text())
    else:
        result = generate(
            gen_model, prompt_text, scenario.prompt,
            sample_idx=0, max_tokens=max_tokens,
        )
        gen_rec = {
            "model": gen_model,
            "scenario_id": scenario.id,
            "prompt_hash": phash,
            "output": result.text,
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason,
        }
        gen_path.write_text(json.dumps(gen_rec, ensure_ascii=False))

    output_text = gen_rec.get("output", "") or ""

    if judge_path.exists():
        j_rec = json.loads(judge_path.read_text())
    else:
        judge_user = JUDGE_USER_TEMPLATE.format(
            scenario=scenario.prompt, output=output_text
        )
        raw = generate(
            judge_model, JUDGE_SYSTEM, judge_user,
            sample_idx=0, max_tokens=1024, json_mode=True,
        ).text
        j_rec = _extract_json(raw)
        j_rec["_judge"] = judge_model
        judge_path.write_text(json.dumps(j_rec, ensure_ascii=False))

    sc = int(j_rec.get("stakeholder_count", 0) or 0)
    us = int(j_rec.get("uncertainty_score",  0) or 0)
    mh = int(j_rec.get("max_causal_hops",    0) or 0)
    return {
        "scenario_id":       scenario.id,
        "output":            output_text,
        "completion_tokens": int(gen_rec.get("completion_tokens", 0) or 0),
        "stakeholder_count": sc,
        "uncertainty_score": us,
        "max_causal_hops":   mh,
        "collapse_fired":    int(sc <= 1),
        "suppression_fired": int(us == 0),
        "_judge":            judge_model,
    }


def forward_batch(
    prompt_text: str,
    batch: list[Scenario],
    gen_model: str,
    judge_model: str,
    max_tokens: int,
    max_workers: int = 6,
    *,
    is_third_judge: bool = False,
) -> list[dict]:
    out: list[dict] = [None] * len(batch)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                code_one, prompt_text, s, gen_model, judge_model, max_tokens,
                is_third_judge=is_third_judge,
            ): i
            for i, s in enumerate(batch)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
    return out


# ---------------------------------------------------------------------------
# Drift control (ng2 namespace)
# ---------------------------------------------------------------------------

def drift_control_ng2(
    scenario: Scenario,
    gen_model: str,
    judge_model: str,
    max_tokens: int,
    label: str,
) -> dict:
    out_path = ng2_drift_path(label)
    result   = generate(
        gen_model, INITIAL_PROMPT, scenario.prompt,
        sample_idx=0, max_tokens=max_tokens,
    )
    output_text = result.text or ""
    judge_user  = JUDGE_USER_TEMPLATE.format(
        scenario=scenario.prompt, output=output_text
    )
    raw   = generate(
        judge_model, JUDGE_SYSTEM, judge_user,
        sample_idx=0, max_tokens=1024, json_mode=True,
    ).text
    j_rec = _extract_json(raw)
    sc    = int(j_rec.get("stakeholder_count", 0) or 0)
    us    = int(j_rec.get("uncertainty_score",  0) or 0)
    rec   = {
        "label":             label,
        "scenario_id":       scenario.id,
        "gen_model":         gen_model,
        "judge_model":       judge_model,
        "stakeholder_count": sc,
        "uncertainty_score": us,
        "completion_tokens": result.completion_tokens,
    }
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    return rec


# ---------------------------------------------------------------------------
# Optimisation loop (identical logic, ng2 cache)
# ---------------------------------------------------------------------------

def run_optimization(
    gen_model: str,
    judge_model: str,
    optimiser_model: str,
    train: list[Scenario],
    n_iters: int,
    batch_size: int,
    max_tokens: int,
) -> tuple[str, list[dict]]:
    prompt  = INITIAL_PROMPT
    history: list[dict] = []

    def _batch_for(it: int) -> list[Scenario]:
        b_start = (it * batch_size) % len(train)
        b_end   = b_start + batch_size
        if b_end > len(train):
            return train[b_start:] + train[:b_end - len(train)]
        return train[b_start:b_end]

    t0     = time.time()
    batch0 = _batch_for(0)
    coded0 = forward_batch(prompt, batch0, gen_model, judge_model, max_tokens)
    stats0 = batch_loss_continuous(coded0)
    rec0   = {
        "iter": 0, "prompt": prompt,
        "prompt_hash": _prompt_hash(prompt),
        "batch_ids": [s.id for s in batch0],
        "coded": coded0, "stats": stats0,
        "gradient": None, "wall_s": time.time() - t0,
    }
    ng2_iter_path(0).write_text(json.dumps(rec0, ensure_ascii=False, indent=2))
    history.append(rec0)
    print(
        f"[iter 0] loss={stats0['loss']:.4f}  "
        f"mean_sc={stats0['mean_stakeholder_count']:.2f}  "
        f"mean_us={stats0['mean_uncertainty_score']:.2f}  "
        f"(collapse={stats0['collapse_rate']:.1%}, "
        f"suppression={stats0['suppression_rate']:.1%})",
        flush=True,
    )

    for it in range(1, n_iters + 1):
        ts   = time.time()
        prev = history[-1]

        gradient   = textual_gradient(
            prev["prompt"], prev["stats"], prev["coded"], optimiser_model,
        )
        new_prompt = update_prompt(prev["prompt"], gradient, optimiser_model)

        batch  = _batch_for(it)
        coded  = forward_batch(new_prompt, batch, gen_model, judge_model, max_tokens)
        stats  = batch_loss_continuous(coded)

        prev_loss = prev["stats"]["loss"]
        delta     = prev_loss - stats["loss"]
        pct       = delta / max(prev_loss, 1e-9) * 100

        record = {
            "iter": it, "prompt": new_prompt,
            "prompt_hash": _prompt_hash(new_prompt),
            "batch_ids": [s.id for s in batch],
            "coded": coded, "stats": stats,
            "gradient": gradient, "wall_s": time.time() - ts,
        }
        ng2_iter_path(it).write_text(json.dumps(record, ensure_ascii=False, indent=2))
        history.append(record)
        print(
            f"[iter {it}] loss={stats['loss']:.4f}  "
            f"mean_sc={stats['mean_stakeholder_count']:.2f}  "
            f"mean_us={stats['mean_uncertainty_score']:.2f}  "
            f"delta={delta:+.4f} ({pct:+.1f}%)  "
            f"prompt_len={len(new_prompt)} chars",
            flush=True,
        )

        if early_stop(history):
            print(
                f"  [early-stop] last 3 iters each reduced loss by <5%; "
                f"stopping at iter {it}.",
                flush=True,
            )
            break

    return history[-1]["prompt"], history


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

def evaluate_holdout(
    final_prompt: str,
    eval_set: list[Scenario],
    gen_model: str,
    primary_judge: str,
    third_judge: str,
    max_tokens: int,
) -> dict:
    opt_coded = forward_batch(
        final_prompt, eval_set, gen_model, primary_judge, max_tokens,
        is_third_judge=False,
    )
    opt_stats = batch_loss_continuous(opt_coded)

    ncot_coded = forward_batch(
        INITIAL_PROMPT, eval_set, gen_model, primary_judge, max_tokens,
        is_third_judge=False,
    )
    ncot_stats = batch_loss_continuous(ncot_coded)

    # Third-judge re-codes cached v3 generations
    opt_third_coded = forward_batch(
        final_prompt, eval_set, gen_model, third_judge, max_tokens,
        is_third_judge=True,
    )
    opt_third_stats = batch_loss_continuous(opt_third_coded)

    kappa_collapse    = cohens_kappa(
        [c["collapse_fired"]    for c in opt_coded],
        [c["collapse_fired"]    for c in opt_third_coded],
    )
    kappa_suppression = cohens_kappa(
        [c["suppression_fired"] for c in opt_coded],
        [c["suppression_fired"] for c in opt_third_coded],
    )

    return {
        "n_eval_scenarios": len(eval_set),
        "optimised_ncot":   {"stats": opt_stats,       "coded": opt_coded},
        "hand_ncot":        {"stats": ncot_stats,       "coded": ncot_coded},
        "third_judge":      {"stats": opt_third_stats,  "coded": opt_third_coded},
        "inter_judge_kappa": {
            "collapse":    kappa_collapse,
            "suppression": kappa_suppression,
            "primary":     primary_judge,
            "third":       third_judge,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Phase 10b: Cross-judge narrative-gradient descent."
    )
    ap.add_argument("--gen-model",       default=DEFAULT_GEN_MODEL)
    ap.add_argument("--judge-model",     default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--third-judge",     default=DEFAULT_THIRD_JUDGE)
    ap.add_argument("--optimiser-model", default=DEFAULT_OPTIMISER)
    ap.add_argument("--n-iters",         type=int, default=10)
    ap.add_argument("--batch-size",      type=int, default=10)
    ap.add_argument("--eval-start",      type=int, default=30)
    ap.add_argument("--eval-end",        type=int, default=60)
    ap.add_argument("--max-tokens",      type=int, default=4096)
    ap.add_argument("--scenarios-n",     type=int, default=100)
    ap.add_argument("--smoke",           action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_iters    = 1
        args.batch_size = 3

    print(f"Cross-judge narrative-gradient run")
    print(f"  generator : {args.gen_model}")
    print(f"  train judge (grok): {args.judge_model}")
    print(f"  third judge (haiku): {args.third_judge}")
    print(f"  optimiser : {args.optimiser_model}", flush=True)

    all_scenarios = load_daily_dilemmas(n=args.scenarios_n)

    train    = (all_scenarios[:3] if args.smoke
                else pick_subsample(all_scenarios, n=30))
    eval_set = all_scenarios[args.eval_start: args.eval_end]
    if args.smoke:
        eval_set = all_scenarios[args.eval_start: args.eval_start + 3]

    print(f"  Train: {len(train)} scenarios")
    print(f"  Eval:  {len(eval_set)} scenarios "
          f"(indices {args.eval_start}-{args.eval_end - 1})", flush=True)

    # Drift control start
    drift_scenario = all_scenarios[0]
    print(f"\n--- Drift-control START ({drift_scenario.id}) ---", flush=True)
    drift_start = drift_control_ng2(
        drift_scenario, args.gen_model, args.judge_model, args.max_tokens, "start"
    )
    print(f"  sc={drift_start['stakeholder_count']}  "
          f"us={drift_start['uncertainty_score']}", flush=True)

    # Optimisation
    print(f"\n--- Optimisation ({args.n_iters} iters max, batch={args.batch_size}) ---",
          flush=True)
    final_prompt, history = run_optimization(
        gen_model=args.gen_model,
        judge_model=args.judge_model,
        optimiser_model=args.optimiser_model,
        train=train,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    # Drift control end
    print(f"\n--- Drift-control END ({drift_scenario.id}) ---", flush=True)
    drift_end   = drift_control_ng2(
        drift_scenario, args.gen_model, args.judge_model, args.max_tokens, "end"
    )
    drift_delta = drift_end["stakeholder_count"] - drift_start["stakeholder_count"]
    print(f"  sc={drift_end['stakeholder_count']}  delta_sc={drift_delta:+d}",
          flush=True)

    # Held-out evaluation (primary = grok, third = haiku -- note reversed roles)
    print(f"\n--- Held-out evaluation ({len(eval_set)} scenarios) ---", flush=True)
    eval_out = evaluate_holdout(
        final_prompt, eval_set, args.gen_model,
        args.judge_model,   # primary in holdout = same grok used in training
        args.third_judge,   # third = haiku (cross-check)
        args.max_tokens,
    )

    opt   = eval_out["optimised_ncot"]["stats"]
    ncot  = eval_out["hand_ncot"]["stats"]
    third = eval_out["third_judge"]["stats"]
    kappa = eval_out["inter_judge_kappa"]

    print(f"  Optimised v3  (grok-judge): "
          f"loss={opt['loss']:.4f}  "
          f"mean_sc={opt['mean_stakeholder_count']:.2f}  "
          f"mean_us={opt['mean_uncertainty_score']:.2f}  "
          f"mean_tokens={opt['mean_completion_tokens']:.0f}")
    print(f"  Hand N-CoT v1 (grok-judge): "
          f"loss={ncot['loss']:.4f}  "
          f"mean_sc={ncot['mean_stakeholder_count']:.2f}  "
          f"mean_us={ncot['mean_uncertainty_score']:.2f}  "
          f"mean_tokens={ncot['mean_completion_tokens']:.0f}")
    print(f"  Optimised v3  (haiku-judge): "
          f"loss={third['loss']:.4f}  "
          f"mean_sc={third['mean_stakeholder_count']:.2f}  "
          f"mean_us={third['mean_uncertainty_score']:.2f}")
    print(f"  Inter-judge kappa  collapse={kappa['collapse']:.3f}  "
          f"suppression={kappa['suppression']:.3f}  "
          f"(grok vs haiku)")

    summary = {
        "gen_model":        args.gen_model,
        "judge_model":      args.judge_model,
        "third_judge":      args.third_judge,
        "optimiser_model":  args.optimiser_model,
        "n_iters_run":      len(history) - 1,
        "n_iters_max":      args.n_iters,
        "batch_size":       args.batch_size,
        "eval_indices":     [args.eval_start, args.eval_end],
        "scenarios_n":      args.scenarios_n,
        "initial_prompt":   INITIAL_PROMPT,
        "final_prompt":     final_prompt,
        "training_loss_curve": [
            {
                "iter":              r["iter"],
                "loss":              r["stats"]["loss"],
                "mean_sc":           r["stats"]["mean_stakeholder_count"],
                "mean_us":           r["stats"]["mean_uncertainty_score"],
                "collapse_rate":     r["stats"]["collapse_rate"],
                "suppression_rate":  r["stats"]["suppression_rate"],
                "prompt_hash":       r["prompt_hash"],
                "wall_s":            r["wall_s"],
            }
            for r in history
        ],
        "drift_control": {
            "scenario_id": drift_scenario.id,
            "start_sc":    drift_start["stakeholder_count"],
            "end_sc":      drift_end["stakeholder_count"],
            "delta_sc":    drift_delta,
            "confounded":  abs(drift_delta) > 0.5,
        },
        "heldout_eval": {
            "optimised_ncot_stats": opt,
            "hand_ncot_stats":      ncot,
            "third_judge_stats":    third,
            "inter_judge_kappa":    kappa,
        },
    }
    ng2_summary_path().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    curve_csv = OUT_DIR / "ng2_loss_curve.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "iter", "loss", "mean_sc", "mean_us",
            "collapse_rate", "suppression_rate", "prompt_hash", "wall_s",
        ])
        for r in history:
            s = r["stats"]
            w.writerow([
                r["iter"], s["loss"],
                s["mean_stakeholder_count"], s["mean_uncertainty_score"],
                s["collapse_rate"], s["suppression_rate"],
                r["prompt_hash"], r["wall_s"],
            ])

    print("\n=== Phase 10b complete ===")
    print(f"Iters run (incl. baseline): {len(history)}")
    print(f"Initial loss (iter 0):      {history[0]['stats']['loss']:.4f}")
    print(f"Final training loss:        {history[-1]['stats']['loss']:.4f}")
    print(f"Final prompt (v3): {len(final_prompt)} chars")
    print(f"Summary: {ng2_summary_path()}")


if __name__ == "__main__":
    main()
