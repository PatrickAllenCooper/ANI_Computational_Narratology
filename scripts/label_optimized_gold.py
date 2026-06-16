"""
scripts/label_optimized_gold.py -- Author validation labels on model outputs.

Interactive CLI for hand-labeling validation (0/1) on stratified holdout
responses from hand NoT and robust prompts. Resumable via data/optimized_gold.jsonl.

Usage:
  python -m scripts.label_optimized_gold
  python -m scripts.label_optimized_gold --export-only   # write sample, no prompts
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.syco_loss import (
    DEFAULT_GEN_MODEL,
    _prompt_hash,
    gen_path,
    load_holdout_oeq,
)
from scripts.run_phase1_quartet import PROMPTS

GOLD_PATH = Path("data/optimized_gold.jsonl")
SAMPLE_N_PER_CONFIG = 20
SEED = 44


def _robust_prompt() -> str:
    from scripts.syco_loss import OUT_DIR
    return json.loads((OUT_DIR / "rg_summary.json").read_text())["final_prompt"]


def _load_labeled() -> dict[str, dict]:
    if not GOLD_PATH.exists():
        return {}
    out = {}
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["uid"]] = row
    return out


def _build_sample(items) -> list[dict]:
    hand_prompt = PROMPTS["narrative_cot"]
    robust_prompt = _robust_prompt()
    gen = DEFAULT_GEN_MODEL
    configs = [
        ("hand_ncot", hand_prompt, "sg_ref_narrative_cot"),
        ("robust", robust_prompt, f"rg_holdout_{gen}"),
    ]
    pool: list[dict] = []
    for config, prompt, ns in configs:
        phash = _prompt_hash(prompt)
        for item in items:
            cache = gen_path(ns, item.id, phash, gen)
            if not cache.exists():
                continue
            resp = (json.loads(cache.read_text()).get("response") or "").strip()
            if not resp:
                continue
            pool.append({
                "uid": f"{config}|{item.id}",
                "config": config,
                "item_id": item.id,
                "question": item.prompt,
                "response": resp,
            })
    rng = random.Random(SEED)
    by_cfg: dict[str, list] = {"hand_ncot": [], "robust": []}
    for row in pool:
        by_cfg[row["config"]].append(row)
    sample = []
    for cfg in ("hand_ncot", "robust"):
        rows = by_cfg[cfg]
        rng.shuffle(rows)
        sample.extend(rows[:SAMPLE_N_PER_CONFIG])
    rng.shuffle(sample)
    return sample


def _save_row(row: dict) -> None:
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_labeled()
    existing[row["uid"]] = row
    with GOLD_PATH.open("w", encoding="utf-8") as f:
        for r in existing.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Label optimized outputs (validation)")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    items = load_holdout_oeq()
    sample = _build_sample(items)
    labeled = _load_labeled()
    pending = [r for r in sample if r["uid"] not in labeled]

    if args.export_only:
        GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GOLD_PATH.open("w", encoding="utf-8") as f:
            for row in sample:
                row = {**row, "human_validation": labeled.get(row["uid"], {}).get("human_validation")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(sample)} sample rows to {GOLD_PATH} ({len(labeled)} already labeled)")
        return 0

    print(f"Sample: {len(sample)} items | labeled: {len(labeled)} | pending: {len(pending)}")
    print("Commands: 0/1 = label, s = skip, q = quit\n")

    for row in pending:
        print("=" * 72)
        print(f"[{row['config']}] item {row['item_id']}")
        print("-" * 72)
        print("QUESTION:")
        print(row["question"][:1500])
        print("-" * 72)
        print("RESPONSE:")
        print(row["response"][:2500])
        print("-" * 72)
        while True:
            ans = input("Validation (1=sycophantic, 0=not, s=skip, q=quit): ").strip().lower()
            if ans == "q":
                print(f"Saved progress to {GOLD_PATH}")
                return 0
            if ans == "s":
                break
            if ans in ("0", "1"):
                out = {**row, "human_validation": int(ans), "labeler": "author"}
                _save_row(out)
                print("  saved.")
                break
            print("  enter 0, 1, s, or q")

    print(f"Done. {len(_load_labeled())} labels in {GOLD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
