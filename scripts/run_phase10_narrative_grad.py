"""
scripts/run_phase10_narrative_grad.py -- Phase 10: Narrative-Gradients.

Textual-gradient descent initialised at the verbatim N-CoT prompt with a
continuous loss that rewards stakeholder coverage and uncertainty
acknowledgement.  Mirrors Phase 9 (run_phase9_textgrad.py) but differs on:

  Initial prompt : verbatim N-CoT  (not standard CoT)
  Loss           : continuous  max(0,4-sc)+max(0,2-us)  (not binary failure rate)
  Train set      : 30 scenarios via seed-43 stratified subsample (Phase 2
                   calibration; avoids the Phase 9 train overlap)
  Eval set       : seed-42 indices 30-59  (shared with Phase 9 for cross-phase
                   comparison on a common 30-scenario held-out set)
  Iterations     : 10 (Phase 9 used 5); early-stop if 3 consecutive iters
                   each reduce loss by <5% vs the preceding iter
  Cache namespace: ng_*  (never collides with Phase 9's tg_*)
  Third judge    : grok-4-1-fast-reasoning re-codes the final optimised-prompt
                   eval set; Cohen's kappa vs primary judge reported
  Drift control  : one live (uncached) N-CoT cell at start and end of the run

Pre-registered pilot design (generator = claude-haiku-4-5 only; retains signal
above the binary floor per Experiment 1 haiku cell; full panel deferred).

Usage:
  python -m scripts.run_phase10_narrative_grad [--n-iters 10] [--batch-size 10]
                                               [--eval-start 30] [--eval-end 60]
                                               [--max-tokens 4096]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.generators import generate
from scripts.run_phase1_quartet import (
    PROMPTS,
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    Scenario,
    load_daily_dilemmas,
    _extract_json,
    _safe,
    SUBSAMPLE_SEED,
)

OUT_DIR = Path("./divergence_study_outputs")
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_GEN_MODEL      = "claude-haiku-4-5"
DEFAULT_JUDGE_MODEL    = "claude-haiku-4-5"
DEFAULT_THIRD_JUDGE    = "grok-4-1-fast-reasoning"
DEFAULT_OPTIMISER      = "claude-sonnet-4-6"

INITIAL_PROMPT = PROMPTS["narrative_cot"]   # starting point for Phase 10


# ---------------------------------------------------------------------------
# Seed-43 stratified subsample (mirrors Phase 2 pick_subsample)
# ---------------------------------------------------------------------------

def pick_subsample(scenarios: list[Scenario], n: int = 30) -> list[Scenario]:
    """Deterministic seed-43 stratified subsample from a 100-scenario list."""
    rng = random.Random(SUBSAMPLE_SEED)   # SUBSAMPLE_SEED == 43
    by_topic: dict[str, list[Scenario]] = {}
    for s in scenarios:
        by_topic.setdefault(s.topic, []).append(s)
    topics = sorted(by_topic.keys())
    n_topics = len(topics)
    quota = n // n_topics
    remainder = n - quota * n_topics
    selected: list[Scenario] = []
    for i, topic in enumerate(topics):
        pool = list(by_topic[topic])
        rng.shuffle(pool)
        take = quota + (1 if i < remainder else 0)
        selected.extend(pool[:take])
    return selected[:n]


# ---------------------------------------------------------------------------
# Cache helpers (ng_* namespace; never collides with Phase 9's tg_*)
# ---------------------------------------------------------------------------

def _prompt_hash(prompt_text: str) -> str:
    return hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:10]


def ng_gen_path(scenario_id: str, prompt_hash: str, gen_model: str) -> Path:
    return OUT_DIR / f"ng_gen_{_safe(gen_model)}_{scenario_id}_{prompt_hash}.json"


def ng_judge_primary_path(
    scenario_id: str, prompt_hash: str, gen_model: str, judge_model: str
) -> Path:
    return OUT_DIR / (
        f"ng_judge_p_{_safe(judge_model)}_{_safe(gen_model)}"
        f"_{scenario_id}_{prompt_hash}.json"
    )


def ng_judge_third_path(
    scenario_id: str, prompt_hash: str, gen_model: str, third_judge: str
) -> Path:
    return OUT_DIR / (
        f"ng_judge_t_{_safe(third_judge)}_{_safe(gen_model)}"
        f"_{scenario_id}_{prompt_hash}.json"
    )


def ng_iter_path(iter_idx: int) -> Path:
    return OUT_DIR / f"ng_iter_{iter_idx:02d}.json"


def ng_summary_path() -> Path:
    return OUT_DIR / "ng_summary.json"


# ---------------------------------------------------------------------------
# Continuous loss  L = mean( max(0, 4-sc) + max(0, 2-us) )
# ---------------------------------------------------------------------------

def _cell_loss(sc: int, us: int) -> float:
    return max(0.0, 4.0 - sc) + max(0.0, 2.0 - us)


def batch_loss_continuous(coded: list[dict]) -> dict:
    """Continuous loss; lower is better. Also reports binary failure rates."""
    n = max(1, len(coded))
    losses = [_cell_loss(c["stakeholder_count"], c["uncertainty_score"]) for c in coded]
    mean_loss = sum(losses) / n
    collapse_rate  = sum(c["collapse_fired"]     for c in coded) / n
    suppression_rate = sum(c["suppression_fired"] for c in coded) / n
    mean_sc  = sum(c["stakeholder_count"] for c in coded) / n
    mean_us  = sum(c["uncertainty_score"]  for c in coded) / n
    mean_ct  = sum(c["completion_tokens"]  for c in coded) / n
    return {
        "n": n,
        "loss": mean_loss,               # continuous; this is what we optimise
        "collapse_rate": collapse_rate,  # binary; diagnostic only
        "suppression_rate": suppression_rate,
        "mean_stakeholder_count": mean_sc,
        "mean_uncertainty_score": mean_us,
        "mean_completion_tokens": mean_ct,
        "per_cell_losses": losses,
    }


# ---------------------------------------------------------------------------
# Early-stopping
# ---------------------------------------------------------------------------

def early_stop(history: list[dict]) -> bool:
    """Return True if the last 3 consecutive iters each reduced loss by <5%."""
    if len(history) < 4:   # need iter 0 + at least 3 optimisation iters
        return False
    for i in range(-3, 0):
        prev_loss = history[i - 1]["stats"]["loss"]
        curr_loss = history[i]["stats"]["loss"]
        reduction = (prev_loss - curr_loss) / max(prev_loss, 1e-9)
        if reduction >= 0.05:
            return False
    return True


# ---------------------------------------------------------------------------
# Forward pass: generate + code one (prompt, scenario) cell
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
    """Generate one trace; code with primary judge (or re-code with third judge).

    For is_third_judge=True: re-uses the cached generation (only judge differs).
    Returns {scenario_id, output, completion_tokens, stakeholder_count,
             uncertainty_score, max_causal_hops, collapse_fired,
             suppression_fired, _judge}.
    """
    phash = _prompt_hash(prompt_text)
    gen_path   = ng_gen_path(scenario.id, phash, gen_model)
    judge_path = (
        ng_judge_third_path(scenario.id, phash, gen_model, judge_model)
        if is_third_judge
        else ng_judge_primary_path(scenario.id, phash, gen_model, judge_model)
    )

    # Generation (always the same regardless of judge)
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

    # Judging
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
        "scenario_id":        scenario.id,
        "output":             output_text,
        "completion_tokens":  int(gen_rec.get("completion_tokens", 0) or 0),
        "stakeholder_count":  sc,
        "uncertainty_score":  us,
        "max_causal_hops":    mh,
        "collapse_fired":     int(sc <= 1),
        "suppression_fired":  int(us == 0),
        "_judge":             judge_model,
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
    """Parallel forward pass: generate + judge every (prompt, scenario) cell."""
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
# Drift control (uncached; always makes a live API call)
# ---------------------------------------------------------------------------

def drift_control_cell(
    scenario: Scenario,
    gen_model: str,
    judge_model: str,
    max_tokens: int,
    label: str,           # "start" or "end"
) -> dict:
    """Run a fresh (uncached) N-CoT cell and write ng_drift_{label}.json."""
    out_path = OUT_DIR / f"ng_drift_{label}.json"
    result = generate(
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
# Backward pass: textual gradient (Phase-10-aware template)
# ---------------------------------------------------------------------------

GRADIENT_SYSTEM = (
    "You are an expert prompt engineer diagnosing why a system prompt for "
    "ethical reasoning falls short on specific quality metrics. Be concrete "
    "and concise. Return plain prose, no markdown."
)

GRADIENT_USER_TEMPLATE = """\
Current system prompt being optimised:
<<<
{prompt_text}
>>>

This prompt currently instructs the model to reason through five narrative
sections: Setting (Protagonist), Stakeholders, Consequences (Tensions),
Uncertainty, and Decision (Action).

The loss function is:
  L = mean over batch of: max(0, 4 - stakeholder_count) + max(0, 2 - uncertainty_score)

Lower loss is better. Loss is zero when every response names >=4 distinct
stakeholders AND acknowledges specific uncertainties (uncertainty_score >=2).
Loss grows linearly with shortfall below those targets.

The prompt was tested on {n} ethical-dilemma scenarios. Each response was
coded on a structured rubric. Batch result for this iteration:

  continuous loss              = {loss:.3f}
  mean stakeholder_count       = {mean_sc:.2f}  (target >=4; penalty for <4)
  mean uncertainty_score       = {mean_us:.2f}  (target >=2; penalty for <2)
  stakeholder_collapse rate    = {collapse_rate:.1%}  (sc<=1; binary diagnostic)
  uncertainty_suppression rate = {suppression_rate:.1%}  (us==0; binary diagnostic)
  mean completion tokens       = {mean_ct:.0f}

A representative sample of outputs (up to 5; truncated to 600 chars each):

{samples}

Your task: write a textual gradient -- a concrete diagnosis of what is wrong
with the CURRENT prompt that produces these shortfalls. Reference the
stakeholder_count and uncertainty_score metrics by name. Identify which
sub-instruction (or absence thereof) drives each shortfall. Comment on
whether the five-section scaffold is helping or hindering. Do not write the
new prompt yet; only diagnose. 4-8 sentences."""


UPDATE_SYSTEM = (
    "You are an expert prompt engineer rewriting a system prompt to address "
    "a specific diagnosis. Return ONLY the new system prompt as plain text. "
    "No preamble, no markdown, no quotes around it."
)

UPDATE_USER_TEMPLATE = """\
Current system prompt (the starting-point N-CoT scaffold):
<<<
{prompt_text}
>>>

Diagnosis from the most recent batch (textual gradient):
<<<
{gradient}
>>>

Produce an UPDATED system prompt that addresses the diagnosis. Constraints:
- The prompt is for an LLM reasoning over ethical dilemmas (single turn).
- It must remain a single system prompt addressed to the model in second
  person ("You are...").
- It must NOT exceed 400 words.
- It must NOT list named ethical frameworks (utilitarianism, deontology,
  virtue ethics) by name; instead instruct what the model should do.
- It should aim to increase stakeholder_count (encourage naming multiple
  distinct parties and what is at stake for each) and uncertainty_score
  (encourage explicitly naming what is unknown or uncertain about each
  projected outcome) without prescribing a specific output schema.
- You are FREE to retain, modify, or drop the five-section structure.
  If you retain sections, you may rename or reorder them. If you judge the
  scaffold is already near-optimal, make targeted edits to the failing
  sub-instructions only.

Return ONLY the new system prompt text."""


def textual_gradient(
    prompt_text: str,
    batch_stats: dict,
    coded: list[dict],
    optimiser_model: str,
) -> str:
    samples = "\n\n".join(
        f"--- scenario {c['scenario_id']} "
        f"(sc={c['stakeholder_count']}, us={c['uncertainty_score']}, "
        f"cell_loss={_cell_loss(c['stakeholder_count'], c['uncertainty_score']):.2f}) ---\n"
        f"{(c['output'] or '')[:600]}"
        for c in coded[:5]
    )
    user = GRADIENT_USER_TEMPLATE.format(
        prompt_text=prompt_text,
        n=batch_stats["n"],
        loss=batch_stats["loss"],
        mean_sc=batch_stats["mean_stakeholder_count"],
        mean_us=batch_stats["mean_uncertainty_score"],
        collapse_rate=batch_stats["collapse_rate"],
        suppression_rate=batch_stats["suppression_rate"],
        mean_ct=batch_stats["mean_completion_tokens"],
        samples=samples,
    )
    return generate(
        optimiser_model, GRADIENT_SYSTEM, user,
        sample_idx=0, max_tokens=1024,
    ).text.strip()


def update_prompt(
    prompt_text: str,
    gradient: str,
    optimiser_model: str,
) -> str:
    user = UPDATE_USER_TEMPLATE.format(
        prompt_text=prompt_text, gradient=gradient,
    )
    raw = generate(
        optimiser_model, UPDATE_SYSTEM, user,
        sample_idx=0, max_tokens=1024,
    ).text.strip()
    # Strip surrounding backticks or quotes if the model added them.
    raw = re.sub(r"^```(?:[a-zA-Z]*)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    raw = raw.strip('"').strip("'").strip()
    return raw


# ---------------------------------------------------------------------------
# Main optimisation loop
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
    prompt   = INITIAL_PROMPT
    history: list[dict] = []

    def _batch_for(it: int) -> list[Scenario]:
        b_start = (it * batch_size) % len(train)
        b_end   = b_start + batch_size
        if b_end > len(train):
            return train[b_start:] + train[: b_end - len(train)]
        return train[b_start:b_end]

    # Iter 0: baseline measurement of the hand-written N-CoT prompt
    t0     = time.time()
    batch0 = _batch_for(0)
    coded0 = forward_batch(prompt, batch0, gen_model, judge_model, max_tokens)
    stats0 = batch_loss_continuous(coded0)
    rec0   = {
        "iter": 0,
        "prompt": prompt,
        "prompt_hash": _prompt_hash(prompt),
        "batch_ids": [s.id for s in batch0],
        "coded": coded0,
        "stats": stats0,
        "gradient": None,
        "wall_s": time.time() - t0,
    }
    ng_iter_path(0).write_text(json.dumps(rec0, ensure_ascii=False, indent=2))
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

        # 1. Backward pass: diagnose failures in prev iteration's outputs
        gradient   = textual_gradient(
            prev["prompt"], prev["stats"], prev["coded"], optimiser_model,
        )

        # 2. Optimiser step: rewrite the prompt to address the gradient
        new_prompt = update_prompt(prev["prompt"], gradient, optimiser_model)

        # 3. Forward pass on the next batch with the updated prompt
        batch  = _batch_for(it)
        coded  = forward_batch(new_prompt, batch, gen_model, judge_model, max_tokens)
        stats  = batch_loss_continuous(coded)

        prev_loss = prev["stats"]["loss"]
        delta     = prev_loss - stats["loss"]
        pct       = delta / max(prev_loss, 1e-9) * 100

        record = {
            "iter": it,
            "prompt": new_prompt,
            "prompt_hash": _prompt_hash(new_prompt),
            "batch_ids": [s.id for s in batch],
            "coded": coded,
            "stats": stats,
            "gradient": gradient,
            "wall_s": time.time() - ts,
        }
        ng_iter_path(it).write_text(json.dumps(record, ensure_ascii=False, indent=2))
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
# Held-out evaluation (primary + third judge)
# ---------------------------------------------------------------------------

def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two lists of binary (0/1) labels."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a = sum(a) / n
    p_b = sum(b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if abs(1 - expected) < 1e-9:
        return 1.0
    return (observed - expected) / (1 - expected)


def evaluate_holdout(
    final_prompt: str,
    eval_set: list[Scenario],
    gen_model: str,
    primary_judge: str,
    third_judge: str,
    max_tokens: int,
) -> dict:
    """Evaluate on the held-out eval set with primary + third judge.

    Returns per-prompt stats and inter-judge Cohen's kappa.
    """
    # Optimised N-CoT under primary judge
    opt_coded = forward_batch(
        final_prompt, eval_set, gen_model, primary_judge, max_tokens,
        is_third_judge=False,
    )
    opt_stats = batch_loss_continuous(opt_coded)

    # Verbatim N-CoT control under primary judge
    ncot_coded = forward_batch(
        INITIAL_PROMPT, eval_set, gen_model, primary_judge, max_tokens,
        is_third_judge=False,
    )
    ncot_stats = batch_loss_continuous(ncot_coded)

    # Third-judge re-coding of the optimised prompt's cached generations
    opt_third_coded = forward_batch(
        final_prompt, eval_set, gen_model, third_judge, max_tokens,
        is_third_judge=True,
    )
    opt_third_stats = batch_loss_continuous(opt_third_coded)

    # Cohen's kappa: primary vs third on the optimised prompt's eval set
    kappa_collapse = cohens_kappa(
        [c["collapse_fired"]     for c in opt_coded],
        [c["collapse_fired"]     for c in opt_third_coded],
    )
    kappa_suppression = cohens_kappa(
        [c["suppression_fired"]  for c in opt_coded],
        [c["suppression_fired"]  for c in opt_third_coded],
    )

    return {
        "n_eval_scenarios": len(eval_set),
        "optimised_ncot":   {"stats": opt_stats,       "coded": opt_coded},
        "hand_ncot":        {"stats": ncot_stats,      "coded": ncot_coded},
        "third_judge":      {"stats": opt_third_stats, "coded": opt_third_coded},
        "inter_judge_kappa": {
            "collapse":     kappa_collapse,
            "suppression":  kappa_suppression,
            "primary":      primary_judge,
            "third":        third_judge,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 10 Narrative-Gradients: TextGrad on N-CoT prompt."
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
    ap.add_argument("--scenarios-n",     type=int, default=100,
                    help="Full DailyDilemmas sample size (must match Phase 1).")
    ap.add_argument("--smoke",           action="store_true",
                    help="Smoke-test mode: 1 iter, batch 3, train 3, eval 3+30.")
    args = ap.parse_args()

    if args.smoke:
        args.n_iters    = 1
        args.batch_size = 3

    print(f"Loading {args.scenarios_n} DailyDilemmas scenarios (seed-42 stratified)...",
          flush=True)
    all_scenarios = load_daily_dilemmas(n=args.scenarios_n)

    # Train: seed-43 stratified subsample of 30 (Phase 2 calibration set)
    if args.smoke:
        train = all_scenarios[:3]
    else:
        train = pick_subsample(all_scenarios, n=30)

    # Eval: shared held-out set (seed-42 indices 30-59), same as Phase 9
    eval_set = all_scenarios[args.eval_start: args.eval_end]
    if args.smoke:
        eval_set = all_scenarios[args.eval_start: args.eval_start + 3]

    print(f"  Train: {len(train)} scenarios (seed-43 stratified subsample)")
    print(f"  Eval:  {len(eval_set)} held-out scenarios "
          f"(seed-42 indices {args.eval_start}-{args.eval_end - 1})", flush=True)

    # Drift-control start
    drift_scenario = all_scenarios[0]
    print(f"\n--- Drift-control START cell (scenario {drift_scenario.id}) ---",
          flush=True)
    drift_start = drift_control_cell(
        drift_scenario, args.gen_model, args.judge_model, args.max_tokens, "start"
    )
    print(f"  sc={drift_start['stakeholder_count']}  "
          f"us={drift_start['uncertainty_score']}", flush=True)

    # Optimisation
    print(f"\n--- Optimisation phase ({args.n_iters} iters max, "
          f"batch={args.batch_size}) ---", flush=True)
    final_prompt, history = run_optimization(
        gen_model=args.gen_model,
        judge_model=args.judge_model,
        optimiser_model=args.optimiser_model,
        train=train,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    # Drift-control end
    print(f"\n--- Drift-control END cell (scenario {drift_scenario.id}) ---",
          flush=True)
    drift_end = drift_control_cell(
        drift_scenario, args.gen_model, args.judge_model, args.max_tokens, "end"
    )
    drift_delta = drift_end["stakeholder_count"] - drift_start["stakeholder_count"]
    print(f"  sc={drift_end['stakeholder_count']}  "
          f"us={drift_end['uncertainty_score']}  "
          f"delta_sc={drift_delta:+d}", flush=True)
    if abs(drift_delta) > 0.5:
        print(
            f"  WARNING: drift_delta={drift_delta:+d} exceeds threshold of 0.5; "
            f"compute drift may confound results.",
            flush=True,
        )

    # Held-out evaluation
    print(f"\n--- Held-out evaluation ({len(eval_set)} scenarios) ---", flush=True)
    if args.smoke:
        # In smoke mode skip the third judge to save API calls
        eval_out = evaluate_holdout(
            final_prompt, eval_set, args.gen_model,
            args.judge_model, args.judge_model, args.max_tokens,
        )
    else:
        eval_out = evaluate_holdout(
            final_prompt, eval_set, args.gen_model,
            args.judge_model, args.third_judge, args.max_tokens,
        )

    opt   = eval_out["optimised_ncot"]["stats"]
    ncot  = eval_out["hand_ncot"]["stats"]
    third = eval_out["third_judge"]["stats"]
    kappa = eval_out["inter_judge_kappa"]

    print(f"  Optimised N-CoT (primary judge): "
          f"loss={opt['loss']:.4f}  "
          f"mean_sc={opt['mean_stakeholder_count']:.2f}  "
          f"mean_us={opt['mean_uncertainty_score']:.2f}  "
          f"mean_tokens={opt['mean_completion_tokens']:.0f}")
    print(f"  Hand N-CoT      (primary judge): "
          f"loss={ncot['loss']:.4f}  "
          f"mean_sc={ncot['mean_stakeholder_count']:.2f}  "
          f"mean_us={ncot['mean_uncertainty_score']:.2f}  "
          f"mean_tokens={ncot['mean_completion_tokens']:.0f}")
    print(f"  Optimised N-CoT (third judge):   "
          f"loss={third['loss']:.4f}  "
          f"mean_sc={third['mean_stakeholder_count']:.2f}  "
          f"mean_us={third['mean_uncertainty_score']:.2f}")
    print(f"  Inter-judge kappa  collapse={kappa['collapse']:.3f}  "
          f"suppression={kappa['suppression']:.3f}  "
          f"(primary={kappa['primary']}  third={kappa['third']})")

    # Write summary
    summary = {
        "gen_model":        args.gen_model,
        "judge_model":      args.judge_model,
        "third_judge":      args.third_judge if not args.smoke else args.judge_model,
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
                "iter":         r["iter"],
                "loss":         r["stats"]["loss"],
                "mean_sc":      r["stats"]["mean_stakeholder_count"],
                "mean_us":      r["stats"]["mean_uncertainty_score"],
                "collapse_rate":    r["stats"]["collapse_rate"],
                "suppression_rate": r["stats"]["suppression_rate"],
                "prompt_hash":  r["prompt_hash"],
                "wall_s":       r["wall_s"],
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
    ng_summary_path().write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # Loss-curve CSV for plotting
    curve_csv = OUT_DIR / "ng_loss_curve.csv"
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

    print("\n=== Phase 10 pilot complete ===")
    print(f"Iters run (incl. baseline): {len(history)}")
    print(f"Initial loss (iter 0):      {history[0]['stats']['loss']:.4f}")
    print(f"Final training loss:        {history[-1]['stats']['loss']:.4f}")
    print(f"Summary: {ng_summary_path()}")
    print(f"Loss CSV: {curve_csv}")


if __name__ == "__main__":
    main()
