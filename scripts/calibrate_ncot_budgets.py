"""
scripts/calibrate_ncot_budgets.py -- Compute per-generator N-CoT median
output token budgets from the existing Phase 1 generation cache.

Reads all gen_{model}_*_narrative_cot_*.json files, computes median
completion_tokens per generator, and writes
divergence_study_outputs/ncot_median_budgets.json.

Zero API spend -- pure file reads.

Usage:
  python -m scripts.calibrate_ncot_budgets
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT_DIR = Path("./divergence_study_outputs")


def main() -> None:
    print("Scanning gen_*_narrative_cot_*.json files in", OUT_DIR)

    per_model: dict[str, list[int]] = {}
    word_per_model: dict[str, list[int]] = {}

    for p in sorted(OUT_DIR.glob("gen_*_narrative_cot_*.json")):
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        model = rec.get("model", "")
        if not model:
            # try to parse from filename: gen_{model}_{sid}_{cond}_{idx}_{hash}.json
            parts = p.stem.split("_")
            # condition is known to be 'narrative_cot'; reconstruct model from prefix
            # This is a fallback; real records always have "model" key
            continue
        tokens = rec.get("completion_tokens", 0)
        text = rec.get("output", "")
        words = len(text.split()) if text else 0
        per_model.setdefault(model, []).append(tokens)
        word_per_model.setdefault(model, []).append(words)

    if not per_model:
        print("ERROR: no narrative_cot generation files found.")
        print("       Make sure divergence_study_outputs/ contains gen_*_narrative_cot_*.json files.")
        return

    budgets: dict[str, dict] = {}
    print("\nPer-generator N-CoT output statistics:")
    for model in sorted(per_model.keys()):
        toks = per_model[model]
        words = word_per_model[model]
        median_tokens = int(statistics.median(toks))
        p90_tokens = int(sorted(toks)[int(0.90 * len(toks))])
        median_words = int(statistics.median(words))
        # Many models report 0 completion_tokens via some Azure routes;
        # fall back to a word-count proxy: 1 word ≈ 1.3 tokens.
        if median_tokens == 0:
            median_tokens = int(median_words * 1.3)
            p90_tokens = int(sorted(words)[int(0.90 * len(words))] * 1.3)
        # Cap at 8192 to stay within typical context limits for generation
        budget = min(p90_tokens + 512, 8192)
        budgets[model] = {
            "n_samples": len(toks),
            "median_completion_tokens": median_tokens,
            "p90_completion_tokens": p90_tokens,
            "matched_budget_max_tokens": budget,
            "median_words": median_words,
        }
        print(f"  {model}:")
        print(f"    n={len(toks)}, median_tokens={median_tokens}, "
              f"p90_tokens={p90_tokens}, budget={budget}")

    out_path = OUT_DIR / "ncot_median_budgets.json"
    out_path.write_text(json.dumps(budgets, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
