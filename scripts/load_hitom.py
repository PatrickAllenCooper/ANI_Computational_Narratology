"""
scripts/load_hitom.py -- Hi-ToM higher-order Theory-of-Mind benchmark loader.

Source: https://huggingface.co/datasets/Hi-ToM/Hi-ToM_Dataset (public, no auth).
Paper: He et al. 2023, "Hi-ToM: A Benchmark for Evaluating Higher-Order
Theory of Mind Reasoning in Large Language Models".

Each item is a short multi-character story (objects moved between rooms
while some characters are absent, with optional deception) followed by a
question of a given ToM order:
  order 0 -- "Where is X really?"                       (ground truth, no ToM)
  order 1 -- "Where does A think X is?"                 (first-order belief)
  order 2-4 -- "Where does A think B thinks ... X is?"  (nested belief chains)

The dataset ships two `prompting_type` copies (CoTP/VP) of the identical
600 (story, question, choices, answer) items; they differ only in how the
original paper wrapped the instruction, which we discard since our own
runner supplies its own system prompts. We load CoTP only (600 unique
items) and dedupe defensively.

Usage:
  python -m scripts.load_hitom --order 3 --n 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

DATA_DIR = Path("data/hitom")
DATA_FILE = DATA_DIR / "Hi-ToM_data.json"
HF_REPO = "Hi-ToM/Hi-ToM_Dataset"
HF_FILENAME = "Hi-ToM_data.json"


def _download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(HF_REPO, HF_FILENAME, repo_type="dataset")
    DATA_FILE.write_bytes(Path(cached).read_bytes())


def _raw() -> list[dict]:
    if not DATA_FILE.exists():
        _download()
    return json.loads(DATA_FILE.read_text())


def load_hitom(
    order: Optional[int] = None,
    story_length: Optional[int] = None,
    deception: Optional[bool] = None,
    n: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Return a list of item dicts, optionally filtered and subsampled.

    Fields: id, story, question, choices (list[str]), answer, order,
    story_length, deception.
    """
    raw = [d for d in _raw() if d["prompting_type"] == "CoTP"]
    items = []
    for d in raw:
        choices = [c.strip() for c in d["choices"].split(",")]
        items.append({
            "id": f"hitom_{d['story_length']}_{d['question_order']}_{d['deception']}_{d['sample_id']}",
            "story": d["story"],
            "question": d["question"],
            "choices": choices,
            "answer": d["answer"],
            "order": d["question_order"],
            "story_length": d["story_length"],
            "deception": d["deception"],
        })

    if order is not None:
        items = [it for it in items if it["order"] == order]
    if story_length is not None:
        items = [it for it in items if it["story_length"] == story_length]
    if deception is not None:
        items = [it for it in items if it["deception"] == deception]

    if n is not None and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=None)
    ap.add_argument("--story-length", type=int, default=None)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    items = load_hitom(order=args.order, story_length=args.story_length, n=args.n)
    print(f"Loaded {len(items)} items")
    for it in items[:3]:
        print(json.dumps(it, indent=2)[:600])
