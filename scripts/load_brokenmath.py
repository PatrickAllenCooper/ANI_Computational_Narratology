"""
scripts/load_brokenmath.py -- Load BrokenMath benchmark split.

Caches INSAIT-Institute/BrokenMath benchmark (451 adversarial problems) to
data/brokenmath/benchmark.jsonl (gitignored).

Usage:
  python -m scripts.load_brokenmath
  python -m scripts.load_brokenmath --smoke
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path("data/brokenmath")
CACHE_FILE = CACHE_DIR / "benchmark.jsonl"
HF_DATASET = "INSAIT-Institute/BrokenMath"


@dataclass
class BrokenMathItem:
    problem_id: str
    problem: str
    original_problem: str
    gold_answer: str
    solution: str
    question_type: str
    is_adversarial: bool

    @property
    def id(self) -> str:
        return self.problem_id


def _download_benchmark() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Install datasets: pip install datasets"
        ) from e
    ds = load_dataset(HF_DATASET, split="benchmark")
    return [dict(row) for row in ds]


def load_brokenmath(
    *,
    adversarial_only: bool = True,
    n: int | None = None,
    seed: int = 44,
    force_refresh: bool = False,
) -> list[BrokenMathItem]:
    """Load BrokenMath benchmark items from cache or HuggingFace."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if force_refresh or not CACHE_FILE.exists():
        rows = _download_benchmark()
        with CACHE_FILE.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Cached {len(rows)} rows to {CACHE_FILE}")

    items = []
    with CACHE_FILE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if adversarial_only and not row.get("is_adversarial", True):
                continue
            items.append(BrokenMathItem(
                problem_id=str(row.get("problem_id", "")),
                problem=row.get("problem", ""),
                original_problem=row.get("original_problem", ""),
                gold_answer=str(row.get("gold_answer", "")),
                solution=row.get("solution", ""),
                question_type=row.get("question_type", "proof"),
                is_adversarial=bool(row.get("is_adversarial", True)),
            ))

    if n is not None and n < len(items):
        import random
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Load BrokenMath benchmark")
    ap.add_argument("--smoke", action="store_true", help="Load 5 items only")
    ap.add_argument("--force-refresh", action="store_true")
    args = ap.parse_args()
    n = 5 if args.smoke else None
    items = load_brokenmath(n=n, force_refresh=args.force_refresh)
    print(f"Loaded {len(items)} BrokenMath items")
    if items:
        print(f"  sample id={items[0].problem_id} type={items[0].question_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
