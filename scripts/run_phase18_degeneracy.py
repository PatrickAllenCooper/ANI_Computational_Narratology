"""
scripts/run_phase18_degeneracy.py -- Degeneracy guard for robust prompt (Phase 18d).

Runs deliberative-depth coding (stakeholder count, uncertainty score) on hand NoT
vs robust prompt over holdout OEQ subsample. Writes into rg_summary.json.

Usage:
  python -m scripts.run_phase18_degeneracy [--n 30]
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
    INIT_PROMPTS,
    OUT_DIR,
    deliberative_depth_batch,
    load_holdout_oeq,
    summary_path,
)

NS = "rg"


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 18 degeneracy guard")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--judge-model", default="claude-haiku-4-5")
    args = ap.parse_args()

    items = load_holdout_oeq()[: args.n]
    robust_prompt = json.loads((OUT_DIR / "rg_summary.json").read_text())["final_prompt"]
    hand_prompt = INIT_PROMPTS["narrative_cot"]

    print(f"Degeneracy guard n={len(items)} gen={args.gen_model}", flush=True)
    hand = deliberative_depth_batch(
        hand_prompt, items, args.gen_model, args.judge_model, f"{NS}_deg_hand",
    )
    robust = deliberative_depth_batch(
        robust_prompt, items, args.gen_model, args.judge_model, f"{NS}_deg_robust",
    )

    summary_path_p = summary_path(NS)
    summary = json.loads(summary_path_p.read_text())
    summary["degeneracy_guard"] = {"hand_ncot": hand, "robust": robust}
    summary_path_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    phase18 = OUT_DIR / "phase18_robust.json"
    if phase18.exists():
        p18 = json.loads(phase18.read_text())
        p18["degeneracy_guard"] = summary["degeneracy_guard"]
        phase18.write_text(json.dumps(p18, ensure_ascii=False, indent=2))

    print(
        f"hand: stakeholders={hand['mean_stakeholder_count']:.2f} "
        f"uncertainty={hand['mean_uncertainty_score']:.2f}",
        flush=True,
    )
    print(
        f"robust: stakeholders={robust['mean_stakeholder_count']:.2f} "
        f"uncertainty={robust['mean_uncertainty_score']:.2f}",
        flush=True,
    )
    print(f"Updated {summary_path_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
