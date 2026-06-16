"""
scripts/run_phase18_quartet.py -- Replicate robust prompt across quartet generators.

Evaluates hand NoT vs Phase 18b robust prompt on holdout OEQ (n=150) for each
generator in the quartet. Scores with optimisation panel plus held-out judges
(claude-sonnet-4-6, deepseek-r1). Haiku results are merged from phase18_robust.json.

Output: divergence_study_outputs/phase18_quartet.json

Usage:
  python -m scripts.run_phase18_quartet [--smoke] [--generators grok,...]
"""
from __future__ import annotations

import argparse
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.run_phase1_quartet import PROMPTS
from scripts.syco_loss import (
    INIT_PROMPTS,
    OUT_DIR,
    load_holdout_oeq,
    panel_batch_loss,
    panel_forward_batch,
)

QUARTET = [
    "gpt-5.4-nano",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "grok-4-1-fast-reasoning",
]
PANEL = ["grok-4-1-fast-reasoning", "claude-haiku-4-5", "gpt-5.4-nano"]
HELDOUT_JUDGES = ["claude-sonnet-4-6", "deepseek-r1"]
ALL_JUDGES = [*PANEL, *HELDOUT_JUDGES]
NS = "rg"


def _robust_prompt() -> str:
    summary = json.loads((OUT_DIR / "rg_summary.json").read_text())
    return summary["final_prompt"]


def eval_config(prompt: str, items, gen: str, ns: str) -> dict:
    coded = panel_forward_batch(prompt, items, gen, ALL_JUDGES, ns)
    panel_stats = panel_batch_loss(coded, PANEL)
    heldout = {j: panel_batch_loss(coded, [j])["per_judge"][j] for j in HELDOUT_JUDGES}
    return {
        "n": panel_stats["n"],
        "robust_loss": panel_stats["robust_loss"],
        "panel_max_rate": panel_stats["panel_max_rate"],
        "per_judge": panel_stats["per_judge"],
        "heldout_judges": heldout,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 18 quartet replication")
    ap.add_argument("--generators", default=",".join(QUARTET))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-sample", action="store_true")
    args = ap.parse_args()

    items = load_holdout_oeq(allow_sample=args.allow_sample)
    if args.smoke:
        items = items[:3]
        args.generators = "claude-sonnet-4-6"

    robust_prompt = _robust_prompt()
    hand_prompt = INIT_PROMPTS["narrative_cot"]
    generators = [g.strip() for g in args.generators.split(",") if g.strip()]

    results = {"generators": {}, "panel": PANEL, "heldout_judges": HELDOUT_JUDGES}
    out_path = OUT_DIR / "phase18_quartet.json"
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            results["generators"].update(prev.get("generators", {}))
        except Exception:
            pass

    for gen in generators:
        print(f"\n=== {gen} ===", flush=True)
        hand_ns = f"{NS}_quartet_hand_{gen}"
        robust_ns = f"{NS}_quartet_robust_{gen}"
        hand = eval_config(hand_prompt, items, gen, hand_ns)
        robust = eval_config(robust_prompt, items, gen, robust_ns)
        results["generators"][gen] = {"hand_ncot": hand, "robust": robust}
        pm_h = hand["panel_max_rate"]
        pm_r = robust["panel_max_rate"]
        print(
            f"  panel-max hand v={pm_h['validation']:.0%} ind={pm_h['indirectness']:.0%} "
            f"f={pm_h['framing']:.0%} | robust v={pm_r['validation']:.0%} "
            f"ind={pm_r['indirectness']:.0%} f={pm_r['framing']:.0%}",
            flush=True,
        )
        for hj in HELDOUT_JUDGES:
            h = hand["heldout_judges"][hj]
            r = robust["heldout_judges"][hj]
            print(
                f"  held-out {hj}: hand v={h['validation']:.0%} f={h['framing']:.0%} | "
                f"robust v={r['validation']:.0%} f={r['framing']:.0%}",
                flush=True,
            )

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
