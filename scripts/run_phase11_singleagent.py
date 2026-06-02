"""
scripts/run_phase11_singleagent.py -- Phase 11 Arm A: single-agent head-to-head
between Narration-of-Thought (N-CoT) and TextGrad-optimised standard CoT.

Question: does hand-written N-CoT still beat a *sufficiently optimised* standard
CoT on the coded structural metrics, or does the optimiser close the gap?

Four conditions are generated and dual-judge-coded on the held-out 30 DailyDilemmas
scenarios (seed-42 indices 30-59, disjoint from the TextGrad training indices 0-29)
across both generators:

  narrative_cot      hand-written N-CoT (PROMPTS["narrative_cot"])
  textgrad_binary    standard CoT optimised under the binary failure-mode loss
  textgrad_continuous standard CoT optimised under the continuous depth loss
  standard_cot       verbatim standard CoT (reference floor)

The two optimised prompts are read from the Phase 11 TextGrad summaries
(tgx_binary_summary.json / tgx_continuous_summary.json, field "final_prompt").

Coding uses the Experiment 1 cross-vendor judge pair: claude-haiku-4-5 (primary)
and gpt-5.4-nano (secondary), on the same six-variable rubric. Deterministic
failure-mode anchors: stakeholder_collapse = (stakeholder_count <= 1),
uncertainty_suppression = (uncertainty_score == 0).

All artefacts are cached under the p11sa_* namespace and the run is idempotent.

Usage:
  python -m scripts.run_phase11_singleagent
         [--generators claude-haiku-4-5,gpt-5.4-nano]
         [--primary-judge claude-haiku-4-5] [--secondary-judge gpt-5.4-nano]
         [--eval-start 30] [--eval-end 60] [--max-tokens 4096] [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
)

OUT_DIR = Path("./divergence_study_outputs")
OUT_DIR.mkdir(exist_ok=True)

NS = "p11sa"
DEFAULT_GENERATORS = ["claude-haiku-4-5", "gpt-5.4-nano"]
DEFAULT_PRIMARY_JUDGE = "claude-haiku-4-5"
DEFAULT_SECONDARY_JUDGE = "gpt-5.4-nano"

TGX_BINARY_SUMMARY = OUT_DIR / "tgx_binary_summary.json"
TGX_CONTINUOUS_SUMMARY = OUT_DIR / "tgx_continuous_summary.json"


def _prompt_hash(prompt_text: str) -> str:
    return hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:10]


def _gen_path(gen_model: str, sid: str, phash: str) -> Path:
    return OUT_DIR / f"{NS}_gen_{_safe(gen_model)}_{sid}_{phash}.json"


def _judge_path(judge: str, gen_model: str, sid: str, phash: str) -> Path:
    return OUT_DIR / f"{NS}_judge_{_safe(judge)}_{_safe(gen_model)}_{sid}_{phash}.json"


def load_conditions() -> dict[str, str]:
    """Return {condition_label: system_prompt} for the four head-to-head arms."""
    conditions = {
        "narrative_cot": PROMPTS["narrative_cot"],
        "standard_cot": PROMPTS["standard_cot"],
    }
    if not TGX_BINARY_SUMMARY.exists():
        raise FileNotFoundError(
            f"{TGX_BINARY_SUMMARY} missing -- run the binary TextGrad optimisation first."
        )
    if not TGX_CONTINUOUS_SUMMARY.exists():
        raise FileNotFoundError(
            f"{TGX_CONTINUOUS_SUMMARY} missing -- run the continuous TextGrad optimisation first."
        )
    conditions["textgrad_binary"] = json.loads(
        TGX_BINARY_SUMMARY.read_text()
    )["final_prompt"]
    conditions["textgrad_continuous"] = json.loads(
        TGX_CONTINUOUS_SUMMARY.read_text()
    )["final_prompt"]
    return conditions


def _judge_one(judge: str, gen_model: str, scenario: Scenario,
               phash: str, output_text: str) -> dict:
    judge_path = _judge_path(judge, gen_model, scenario.id, phash)
    if judge_path.exists():
        return json.loads(judge_path.read_text())
    judge_user = JUDGE_USER_TEMPLATE.format(
        scenario=scenario.prompt, output=output_text
    )
    raw = generate(
        judge, JUDGE_SYSTEM, judge_user,
        sample_idx=0, max_tokens=1024, json_mode=True,
    ).text
    j_rec = _extract_json(raw)
    j_rec["_judge"] = judge
    judge_path.write_text(json.dumps(j_rec, ensure_ascii=False))
    return j_rec


def code_cell(
    condition: str,
    prompt_text: str,
    scenario: Scenario,
    gen_model: str,
    primary_judge: str,
    secondary_judge: str,
    max_tokens: int,
) -> dict:
    """Generate one trace and code it with both judges. Idempotent on disk."""
    phash = _prompt_hash(prompt_text)
    gen_path = _gen_path(gen_model, scenario.id, phash)

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
            "condition": condition,
            "prompt_hash": phash,
            "output": result.text,
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason,
        }
        gen_path.write_text(json.dumps(gen_rec, ensure_ascii=False))

    output_text = gen_rec.get("output", "") or ""
    j1 = _judge_one(primary_judge, gen_model, scenario, phash, output_text)
    j2 = _judge_one(secondary_judge, gen_model, scenario, phash, output_text)

    def _ints(j: dict) -> tuple[int, int, int]:
        return (
            int(j.get("stakeholder_count", 0) or 0),
            int(j.get("uncertainty_score", 0) or 0),
            int(j.get("max_causal_hops", 0) or 0),
        )

    sc1, us1, mh1 = _ints(j1)
    sc2, us2, mh2 = _ints(j2)
    return {
        "generator": gen_model,
        "condition": condition,
        "scenario_id": scenario.id,
        "completion_tokens": int(gen_rec.get("completion_tokens", 0) or 0),
        # primary judge
        "sc_primary": sc1, "us_primary": us1, "mh_primary": mh1,
        "collapse_fired_primary": int(sc1 <= 1),
        "suppression_fired_primary": int(us1 == 0),
        # secondary judge
        "sc_secondary": sc2, "us_secondary": us2, "mh_secondary": mh2,
        "collapse_fired_secondary": int(sc2 <= 1),
        "suppression_fired_secondary": int(us2 == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 11 Arm A single-agent head-to-head")
    ap.add_argument("--generators", default=",".join(DEFAULT_GENERATORS))
    ap.add_argument("--primary-judge", default=DEFAULT_PRIMARY_JUDGE)
    ap.add_argument("--secondary-judge", default=DEFAULT_SECONDARY_JUDGE)
    ap.add_argument("--eval-start", type=int, default=30)
    ap.add_argument("--eval-end", type=int, default=60)
    ap.add_argument("--scenarios-n", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    conditions = load_conditions()

    print(f"Loading {args.scenarios_n} DailyDilemmas scenarios (seed-42 stratified)...", flush=True)
    scenarios = load_daily_dilemmas(n=args.scenarios_n)
    eval_set = scenarios[args.eval_start:args.eval_end]
    print(f"  Held-out eval: {len(eval_set)} scenarios "
          f"(indices {args.eval_start}-{args.eval_end - 1})", flush=True)
    print(f"  Generators: {generators}")
    print(f"  Conditions: {list(conditions.keys())}")
    print(f"  Judges: primary={args.primary_judge} secondary={args.secondary_judge}",
          flush=True)

    tasks = [
        (cond, prompt, sc, gen)
        for gen in generators
        for cond, prompt in conditions.items()
        for sc in eval_set
    ]
    print(f"  Total cells: {len(tasks)}", flush=True)

    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                code_cell, cond, prompt, sc, gen,
                args.primary_judge, args.secondary_judge, args.max_tokens,
            ): (gen, cond, sc.id)
            for cond, prompt, sc, gen in tasks
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"  ERROR {key}: {e}", flush=True)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} cells done", flush=True)

    out_csv = OUT_DIR / "p11_singleagent_raw.csv"
    if rows:
        fieldnames = sorted(set().union(*[r.keys() for r in rows]))
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_csv}")

    # Quick console summary (primary judge means by generator x condition).
    print("\n--- Primary-judge means (stakeholder_count / uncertainty_score) ---")
    for gen in generators:
        print(f"  {gen}")
        for cond in conditions:
            cell = [r for r in rows if r["generator"] == gen and r["condition"] == cond]
            if not cell:
                continue
            n = len(cell)
            msc = sum(r["sc_primary"] for r in cell) / n
            mus = sum(r["us_primary"] for r in cell) / n
            mtk = sum(r["completion_tokens"] for r in cell) / n
            print(f"    {cond:20s} sc={msc:.2f}  us={mus:.2f}  tokens={mtk:.0f}  (n={n})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
