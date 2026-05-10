"""
scripts/run_phase3_debate.py -- Phase 3: multi-agent debate replication
across the new generator trio (claude-haiku-4-5, grok-4-1-fast-reasoning,
claude-sonnet-4-6) plus the strict Standard-CoT-Rounds-0-2 ablation on all
five generators.

This script wraps run_debate_std_cot_ablation.py, routing Rounds 0-2
generation through scripts.generators.generate() so non-Azure models work,
then handing the moderator (R3-R4) to the existing gpt-4o-mini path.

Two arm modes:
  --arm narrative   -- N-CoT in Rounds 0-2 (use run_phase_c for existing generators;
                       this script adds the new trio)
  --arm std_cot     -- Standard CoT in Rounds 0-2 (the strict ablation)

Outputs:
  divergence_study_outputs/debate_arc_quartet.csv
  divergence_study_outputs/debate_full_arc_quartet.pdf

Usage:
  python -m scripts.run_phase3_debate --arm narrative --generators claude-haiku-4-5,...
  python -m scripts.run_phase3_debate --arm std_cot   --generators all
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.generators import generate as _generate

# Import shared data structures from the ablation script
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_debate_std_cot_ablation import (
    SCENARIOS, SCENARIO_BY_ID, PERSPECTIVES, PERSPECTIVES_BY_SCENARIO,
    MP_STD_COT_SYSTEM, MP_STD_COT_USER_TEMPLATE,
    R1_USER_TEMPLATE, R2_OPEN_USER_TEMPLATE,
    OPEN_MOD_SYSTEM, OPEN_MOD_USER_TEMPLATE,
    SYNTH_R3_SYSTEM, SYNTH_R3_USER_TEMPLATE,
    SYNTH_R3_EXTRACT_SYSTEM, SYNTH_R3_EXTRACT_USER_TEMPLATE,
    SYNTH_FINAL_MOD_SYSTEM, SYNTH_FINAL_MOD_USER_TEMPLATE,
    INTEG_MOD_SYSTEM, INTEG_MOD_USER_TEMPLATE,
    R4_FINAL_MOD_SYSTEM, R4_FINAL_MOD_USER_TEMPLATE,
    Scenario, Perspective, PerspectiveGeneration,
    extract_json, _safe,
    MODERATOR_MODEL,
)

# N-CoT system prompt for the narrative arm
NCOT_AGENT_SYSTEM = (
    "You are a thoughtful deliberation participant. When reasoning about an "
    "ethical situation from a specific perspective, structure your response as "
    "a five-part first-person narrative.\n\n"
    "Section 1 – Protagonist: Identify who you are in this situation and what "
    "you know.\n"
    "Section 2 – Stakeholders: Name all parties affected and what is at stake "
    "for each.\n"
    "Section 3 – Consequences: Trace the consequences of each available action "
    "at least two steps forward for each stakeholder.\n"
    "Section 4 – Uncertainty: Articulate what remains genuinely uncertain about "
    "each projected future.\n"
    "Section 5 – Decision: Commit to a specific preferred outcome and explain "
    "why this trajectory is preferable within the narrative frame.\n\n"
    "Work through all five sections before stating your final preferred decision."
)

OUT_DIR = Path("./divergence_study_outputs")
OUT_DIR.mkdir(exist_ok=True)

ALL_GENERATORS = [
    "gpt-5.4-nano",
    "claude-haiku-4-5",
    "grok-4-1-fast-reasoning",
    "claude-sonnet-4-6",
]
NEW_GENERATORS = ["claude-haiku-4-5", "grok-4-1-fast-reasoning", "claude-sonnet-4-6"]


# ---------------------------------------------------------------------------
# Routing wrapper: calls generate() but mirrors the call_chat signature
# ---------------------------------------------------------------------------

def _call_gen(model: str, system: str, user: str, *, seed: int,
              max_tokens: int = 8000) -> str:
    result = _generate(model, system, user, sample_idx=seed, max_tokens=max_tokens)
    return result.text


def _call_mod(system: str, user: str, *, seed: int) -> dict:
    """Call the moderator model in JSON mode."""
    mod_model = os.environ.get("AZURE_AI_MODERATOR_MODEL", "gpt-5.4-nano")
    result = _generate(
        mod_model, system, user,
        sample_idx=seed, max_tokens=2048, json_mode=True,
    )
    if not result.text:
        return {"content_filtered": True}
    return extract_json(result.text) or {}


# ---------------------------------------------------------------------------
# Cache helpers (keyed by arm to avoid collision)
# ---------------------------------------------------------------------------

def _cache(arm: str, round_name: str, model: str, sid: str, pid: str, idx: int) -> Path:
    return OUT_DIR / f"debate_{arm}_{round_name}_{_safe(model)}_{sid}_{pid}_{idx:03d}.json"


def _mod_cache(arm: str, name: str, model: str, sid: str, idx: int) -> Path:
    return OUT_DIR / f"debate_{arm}_{name}_{_safe(model)}_{sid}_{idx:03d}.json"


# ---------------------------------------------------------------------------
# Per-round generation (both arms)
# ---------------------------------------------------------------------------

def gen_r0(s: Scenario, p: Perspective, idx: int, model: str, arm: str) -> PerspectiveGeneration:
    key = _cache(arm, "r0", model, s.id, p.perspective_id, idx)
    if key.exists():
        try:
            return PerspectiveGeneration(**json.loads(key.read_text()))
        except Exception:
            key.unlink()
    system = NCOT_AGENT_SYSTEM if arm == "narrative" else MP_STD_COT_SYSTEM
    user = MP_STD_COT_USER_TEMPLATE.format(
        scenario=s.prompt, perspective_description=p.description,
    )
    out = _call_gen(model, system, user, seed=idx)
    pg = PerspectiveGeneration(
        scenario_id=s.id, perspective_id=p.perspective_id,
        sample_idx=idx, model=model, output=out,
        meta={"arm": arm, "round": "r0"},
    )
    key.write_text(json.dumps(asdict(pg), indent=2))
    return pg


def gen_r1(s: Scenario, p: Perspective, idx: int, model: str, arm: str,
           r0_by_persp: dict) -> PerspectiveGeneration:
    key = _cache(arm, "r1", model, s.id, p.perspective_id, idx)
    if key.exists():
        try:
            return PerspectiveGeneration(**json.loads(key.read_text()))
        except Exception:
            key.unlink()
    system = NCOT_AGENT_SYSTEM if arm == "narrative" else MP_STD_COT_SYSTEM
    own = r0_by_persp.get(p.perspective_id)
    others = "\n\n".join(
        f"[{pid.replace('_', ' ').title()}]:\n{pg.output or '[No statement]'}"
        for pid, pg in r0_by_persp.items() if pid != p.perspective_id
    ) or "[No other statements]"
    user = R1_USER_TEMPLATE.format(
        scenario=s.prompt,
        perspective_description=p.description,
        own_statement=own.output if own else "[No statement]",
        other_statements=others,
    )
    out = _call_gen(model, system, user, seed=idx + 1000)
    pg = PerspectiveGeneration(
        scenario_id=s.id, perspective_id=p.perspective_id,
        sample_idx=idx, model=model, output=out,
        meta={"arm": arm, "round": "r1"},
    )
    key.write_text(json.dumps(asdict(pg), indent=2))
    return pg


def gen_r2(s: Scenario, p: Perspective, idx: int, model: str, arm: str,
           r0_by_persp: dict, r1_by_persp: dict) -> PerspectiveGeneration:
    key = _cache(arm, "r2", model, s.id, p.perspective_id, idx)
    if key.exists():
        try:
            return PerspectiveGeneration(**json.loads(key.read_text()))
        except Exception:
            key.unlink()
    system = NCOT_AGENT_SYSTEM if arm == "narrative" else MP_STD_COT_SYSTEM
    own_r0 = r0_by_persp.get(p.perspective_id)
    own_r1 = r1_by_persp.get(p.perspective_id)
    others_r1 = "\n\n".join(
        f"[{pid.replace('_', ' ').title()}]:\n{pg.output or '[No rebuttal]'}"
        for pid, pg in r1_by_persp.items() if pid != p.perspective_id
    ) or "[No other rebuttals]"
    user = R2_OPEN_USER_TEMPLATE.format(
        scenario=s.prompt,
        perspective_description=p.description,
        own_r0=own_r0.output if own_r0 else "[No statement]",
        own_r1=own_r1.output if own_r1 else "[No rebuttal]",
        other_r1=others_r1,
    )
    out = _call_gen(model, system, user, seed=idx + 3000)
    pg = PerspectiveGeneration(
        scenario_id=s.id, perspective_id=p.perspective_id,
        sample_idx=idx, model=model, output=out,
        meta={"arm": arm, "round": "r2"},
    )
    key.write_text(json.dumps(asdict(pg), indent=2))
    return pg


def run_open_mod(s: Scenario, idx: int, gen_model: str, arm: str,
                 r0, r1, r2) -> dict:
    key = _mod_cache(arm, "open_mod", gen_model, s.id, idx)
    if key.exists():
        try:
            return json.loads(key.read_text())
        except Exception:
            key.unlink()
    ps = PERSPECTIVES_BY_SCENARIO[s.id]

    def _fmt_block(label, rounds):
        parts = []
        for p in ps:
            text = rounds.get(p.perspective_id)
            if text:
                parts.append(f"[{p.perspective_id.replace('_', ' ').title()}]\n{text}")
        return "\n\n".join(parts)

    r0_texts = {p.perspective_id: r0.get(p.perspective_id, PerspectiveGeneration(
        s.id, p.perspective_id, idx, gen_model, "")).output for p in ps}
    r1_texts = {p.perspective_id: r1.get(p.perspective_id, PerspectiveGeneration(
        s.id, p.perspective_id, idx, gen_model, "")).output for p in ps}
    r2_texts = {p.perspective_id: r2.get(p.perspective_id, PerspectiveGeneration(
        s.id, p.perspective_id, idx, gen_model, "")).output for p in ps}

    transcript = (
        "=== Round 0 (Opening statements) ===\n"
        + _fmt_block("r0", r0_texts)
        + "\n\n=== Round 1 (Rebuttals) ===\n"
        + _fmt_block("r1", r1_texts)
        + "\n\n=== Round 2 (Final positions) ===\n"
        + _fmt_block("r2", r2_texts)
    )
    taxonomy_str = "\n".join(
        f"  {k}: {v}" for k, v in s.decision_taxonomy.items()
    )
    user = OPEN_MOD_USER_TEMPLATE.format(
        scenario=s.prompt,
        taxonomy=taxonomy_str,
        transcript=transcript,
    )
    result = _call_mod(OPEN_MOD_SYSTEM, user, seed=idx + 5000)
    key.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Orchestrate one scenario x sample x arm
# ---------------------------------------------------------------------------

def run_debate_sample(s: Scenario, idx: int, gen_model: str, arm: str) -> dict:
    """Run one full four-round debate. Returns a summary record."""
    ps = PERSPECTIVES_BY_SCENARIO.get(s.id, [])
    if not ps:
        return {}

    # Rounds 0-2 with the generator model
    r0: dict[str, PerspectiveGeneration] = {}
    for p in ps:
        r0[p.perspective_id] = gen_r0(s, p, idx, gen_model, arm)

    r1: dict[str, PerspectiveGeneration] = {}
    for p in ps:
        r1[p.perspective_id] = gen_r1(s, p, idx, gen_model, arm, r0)

    r2: dict[str, PerspectiveGeneration] = {}
    for p in ps:
        r2[p.perspective_id] = gen_r2(s, p, idx, gen_model, arm, r0, r1)

    # R3 open moderator pass
    open_mod = run_open_mod(s, idx, gen_model, arm, r0, r1, r2)

    # Extract summary statistics
    consensus_r2 = open_mod.get("consensus_reached", False)
    novel_actions = open_mod.get("novel_actions_proposed", [])
    consensus_kind = open_mod.get("consensus_kind", "none")

    return {
        "generator": gen_model,
        "arm": arm,
        "scenario_id": s.id,
        "sample_idx": idx,
        "r2_consensus_reached": consensus_r2,
        "r2_consensus_kind": consensus_kind,
        "r2_novel_actions_count": len(novel_actions),
        "open_mod_summary": open_mod.get("summary", ""),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def make_arc_figure(df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, arm in zip(axes, ["narrative", "std_cot"]):
        df_arm = df[df["arm"] == arm]
        generators = sorted(df_arm["generator"].unique())
        rates = []
        for gen in generators:
            sub = df_arm[df_arm["generator"] == gen]
            rate = sub["r2_consensus_reached"].astype(float).mean()
            rates.append(rate)
        ax.bar(range(len(generators)), rates, color="#2166ac" if arm == "narrative" else "#d6604d")
        ax.set_xticks(range(len(generators)))
        ax.set_xticklabels(
            [g.replace("claude-", "").replace("grok-4-1-", "grok-").replace("-reasoning", "-r")
             for g in generators],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(
            f"{'Narrative CoT' if arm == 'narrative' else 'Standard CoT'} Rounds 0-2",
            fontsize=10,
        )
        ax.set_ylabel("Round-2 consensus rate", fontsize=9)

    fig.suptitle("Debate arc Round-2 consensus rate: N-CoT vs. Standard-CoT Rounds 0-2\n"
                 "(five generators, five scenarios, N=10)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"  Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3: debate replication + std-CoT ablation")
    parser.add_argument("--arm", choices=["narrative", "std_cot", "both"], default="both",
                        help="Which arm to run")
    parser.add_argument("--generators", default=",".join(NEW_GENERATORS),
                        help="'all' or comma-separated model names")
    parser.add_argument("--scenarios", default="all",
                        help="'all' or comma-separated scenario IDs")
    parser.add_argument("--n", type=int, default=3, help="Samples per cell")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    if args.generators.strip().lower() == "all":
        generators = ALL_GENERATORS
    else:
        generators = [g.strip() for g in args.generators.split(",") if g.strip()]

    arms = ["narrative", "std_cot"] if args.arm == "both" else [args.arm]
    scenarios = SCENARIOS if args.scenarios == "all" else [
        SCENARIO_BY_ID[sid] for sid in args.scenarios.split(",") if sid in SCENARIO_BY_ID
    ]

    print(f"Arms: {arms}  Generators: {generators}  Scenarios: {[s.id for s in scenarios]}  N={args.n}")

    all_rows: list[dict] = []

    for arm in arms:
        for gen in generators:
            print(f"\n{'='*60}")
            print(f"Arm={arm}  Generator={gen}")
            tasks = [
                (s, idx)
                for s in scenarios
                for idx in range(args.n)
            ]
            done = 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {
                    pool.submit(run_debate_sample, s, idx, gen, arm): (s.id, idx)
                    for s, idx in tasks
                }
                for fut in as_completed(futs):
                    key = futs[fut]
                    try:
                        row = fut.result()
                        if row:
                            all_rows.append(row)
                        done += 1
                        if done % 10 == 0:
                            print(f"  {done}/{len(tasks)}", flush=True)
                    except Exception as e:
                        print(f"  ERROR {key}: {e}", flush=True)
            print(f"  Done {done}/{len(tasks)}")

    if not all_rows:
        print("No rows collected.")
        return 1

    df = pd.DataFrame(all_rows)
    raw_path = OUT_DIR / "debate_arc_quartet.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nWrote {raw_path}")

    # Headline: R2 consensus rate by generator x arm
    summary = (
        df.groupby(["generator", "arm"])["r2_consensus_reached"]
        .agg(["mean", "count"])
        .reset_index()
    )
    print(summary.to_string(index=False))

    make_arc_figure(df, OUT_DIR / "debate_full_arc_quartet.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
