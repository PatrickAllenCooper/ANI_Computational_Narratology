"""
scripts/run_phase14_holdout.py -- Phase 14 holdout evaluation across optimizers.

Reads sg_*, sg_tg_*, sg_opro_*, sg_ape_* summaries; picks winner by holdout
OEQ loss; replicates winner across quartet on OEQ validation + flip_pairs moral.

Usage:
  python -m scripts.run_phase14_holdout [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.run_phase1_quartet import PROMPTS
from scripts.syco_loss import (
    DEFAULT_GEN_MODEL,
    DEFAULT_TRAIN_JUDGE,
    OUT_DIR,
    eval_holdout_moral,
    eval_holdout_oeq,
    load_holdout_flip_pairs,
    load_holdout_oeq,
    summary_path,
)

QUARTET = [
    "gpt-5.4-nano", "claude-haiku-4-5", "claude-sonnet-4-6", "grok-4-1-fast-reasoning",
]
OPTIMIZER_NS = {
    "narrative_grad": "sg",
    "textgrad_cot": "sg_tg",
    "opro": "sg_opro",
    "ape": "sg_ape",
}
REFERENCE_ARMS = {
    "narrative_cot": PROMPTS["narrative_cot"],
    "standard_cot": PROMPTS["standard_cot"],
}


def _load_summary(ns: str) -> dict | None:
    p = summary_path(ns)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _holdout_loss(summary: dict) -> float:
    return summary.get("holdout_oeq", {}).get("optimised", {}).get("loss", 999.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 14 holdout + quartet replication")
    ap.add_argument("--judge-model", default=DEFAULT_TRAIN_JUDGE)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-sample", action="store_true")
    args = ap.parse_args()

    summaries = {}
    for name, ns in OPTIMIZER_NS.items():
        s = _load_summary(ns)
        if s:
            summaries[name] = s
            print(f"  {name}: holdout loss={_holdout_loss(s):.4f}", flush=True)

    if not summaries:
        print("No optimizer summaries found; run Phase 14 optimizers first.")
        return 1

    winner_name = min(summaries, key=lambda k: _holdout_loss(summaries[k]))
    winner = summaries[winner_name]
    winner_prompt = winner["final_prompt"]
    print(f"Winner: {winner_name} (loss={_holdout_loss(winner):.4f})", flush=True)

    holdout = load_holdout_oeq(allow_sample=args.allow_sample)
    flip_n = 10 if args.smoke else 150
    flip_items = load_holdout_flip_pairs(n=flip_n, allow_sample=args.allow_sample)
    if args.smoke:
        holdout = holdout[:3]

    results = {
        "winner": winner_name,
        "winner_namespace": OPTIMIZER_NS[winner_name],
        "winner_prompt": winner_prompt,
        "optimizer_comparison": {},
        "quartet_replication": {},
        "reference_arms": {},
    }

    for name, s in summaries.items():
        results["optimizer_comparison"][name] = {
            "holdout_loss": _holdout_loss(s),
            "holdout_oeq": s.get("holdout_oeq", {}),
            "holdout_moral": s.get("holdout_moral", {}),
        }

    gens = [DEFAULT_GEN_MODEL] if args.smoke else QUARTET
    for gen in gens:
        ns = f"sg_holdout_{OPTIMIZER_NS[winner_name]}"
        oeq = eval_holdout_oeq(winner_prompt, holdout, gen, args.judge_model, f"{ns}_{gen}")
        moral = eval_holdout_moral(winner_prompt, flip_items, gen, f"{ns}_{gen}")
        results["quartet_replication"][gen] = {
            "oeq": oeq["stats"],
            "moral": moral,
        }

    for arm_name, prompt in REFERENCE_ARMS.items():
        ref = eval_holdout_oeq(
            prompt, holdout, DEFAULT_GEN_MODEL, args.judge_model, f"sg_ref_{arm_name}",
        )
        results["reference_arms"][arm_name] = ref["stats"]

    out_path = OUT_DIR / "phase14_holdout.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
