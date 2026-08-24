"""
scripts/load_fantom.py -- FANToM Theory-of-Mind conversational benchmark loader.

Source: official AI2 archive referenced by scripts/dataset_loader.py in
https://github.com/skywalker023/fantom (public, no auth):
  https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz
Paper: Kim et al. 2023, "FANToM: A Benchmark for Stress-testing Machine
Theory of Mind in Interactions".

Each conversation set contains belief questions at two ToM orders:
  first-order  -- "What does A believe about <fact>?"          (single mind)
  second-order -- "What does A believe B believes about <fact>?" (nested; the
                   higher-order case, further split accessible/cyclic/acyclic
                   by whether A was present for B's relevant utterance)

We build a binary multiple-choice framing (correct vs. wrong belief
statement, order shuffled) from each beliefQA's correct_answer/wrong_answer
pair -- the same reduction the original eval script's evaluate_mc_belief_q
uses for its multiple-choice condition.

Usage:
  python -m scripts.load_fantom --tom-order second-order --n 5
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests

DATA_DIR = Path("data/fantom")
DATA_FILE = DATA_DIR / "fantom_v1.json"
ARCHIVE_URL = "https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz"


def _download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(ARCHIVE_URL, timeout=120)
    resp.raise_for_status()
    with tarfile.open(fileobj=BytesIO(resp.content)) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("fantom_v1.json"))
        f = tar.extractfile(member)
        DATA_FILE.write_bytes(f.read())


def _raw() -> list[dict]:
    if not DATA_FILE.exists():
        _download()
    return json.loads(DATA_FILE.read_text())


def load_fantom(
    tom_order: Optional[str] = None,
    n: Optional[int] = None,
    seed: int = 42,
    context: str = "short",
) -> list[dict]:
    """Return a flattened list of belief-QA items as binary MC questions.

    tom_order: None (all), "first-order", or "second-order" (matches any
      second-order:accessible/cyclic/acyclic subtype).
    context: "short" (conversation truncated at the point relevant to the
      question -- cheaper, matches the paper's "short" condition) or "full".

    Fields: id, context, question, choices (list[str], len 2), answer
      (the correct choice text), order ("first-order"/"second-order"),
      tom_subtype (raw tom_type string), correct_letter.
    """
    ctx_key = "short_context" if context == "short" else "full_context"
    raw = _raw()
    rng = random.Random(seed)
    items = []
    for conv in raw:
        for i, qa in enumerate(conv["beliefQAs"]):
            subtype = qa["tom_type"]
            order = "first-order" if subtype == "first-order" else "second-order"
            if tom_order is not None and order != tom_order:
                continue
            options = [qa["correct_answer"], qa["wrong_answer"]]
            shuffled = options[:]
            rng.shuffle(shuffled)
            correct_letter = "A" if shuffled[0] == qa["correct_answer"] else "B"
            items.append({
                "id": f"fantom_{conv['set_id']}_belief_{i}",
                "context": conv[ctx_key],
                "question": qa["question"],
                "choices": [f"A. {shuffled[0]}", f"B. {shuffled[1]}"],
                "answer": qa["correct_answer"],
                "correct_letter": correct_letter,
                "order": order,
                "tom_subtype": subtype,
            })

    if n is not None and n < len(items):
        items = rng.sample(items, n)
    return items


def load_fantom_gen(
    tom_order: Optional[str] = None,
    n: Optional[int] = None,
    seed: int = 42,
    context: str = "short",
) -> list[dict]:
    """Same underlying belief questions as load_fantom(), but as free-form
    generation items (paper's BeliefQ[Dist.]) instead of binary MC -- the
    model answers in its own words rather than picking a letter.

    Fields: id, context, question, correct_answer, wrong_answer (used at
      grading time to classify the free-form response), order, tom_subtype.
    """
    ctx_key = "short_context" if context == "short" else "full_context"
    raw = _raw()
    items = []
    for conv in raw:
        for i, qa in enumerate(conv["beliefQAs"]):
            subtype = qa["tom_type"]
            order = "first-order" if subtype == "first-order" else "second-order"
            if tom_order is not None and order != tom_order:
                continue
            items.append({
                "id": f"fantom_{conv['set_id']}_beliefgen_{i}",
                "context": conv[ctx_key],
                "question": qa["question"],
                "correct_answer": qa["correct_answer"],
                "wrong_answer": qa["wrong_answer"],
                "order": order,
                "tom_subtype": subtype,
            })

    if n is not None and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


def load_fantom_binary(
    qtype: str = "answerability",
    n: Optional[int] = None,
    seed: int = 42,
    context: str = "short",
) -> list[dict]:
    """FANToM's binary Y/N accessibility questions (paper's
    AnswerabilityQ[Y/N] / InfoAccessQ[Y/N]): "Does <character> know/can
    answer X?" -- one item per named character per conversation.

    qtype: "answerability" or "info_accessibility".
    Fields: id, context, question, correct_answer ("yes"/"no" -- the
      dataset's rare "no:long" sub-label is normalized to "no").
    """
    key = "answerabilityQAs_binary" if qtype == "answerability" else "infoAccessibilityQAs_binary"
    ctx_key = "short_context" if context == "short" else "full_context"
    raw = _raw()
    items = []
    for conv in raw:
        for i, qa in enumerate(conv[key]):
            answer = "yes" if qa["correct_answer"] == "yes" else "no"
            items.append({
                "id": f"fantom_{conv['set_id']}_{qtype}_bin_{i}",
                "context": conv[ctx_key],
                "question": qa["question"],
                "correct_answer": answer,
            })

    if n is not None and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


def load_fantom_list(
    qtype: str = "answerability",
    n: Optional[int] = None,
    seed: int = 42,
    context: str = "short",
) -> list[dict]:
    """Return FANToM's list-type questions ("name everyone who knows/can
    answer X"), used for F1 scoring against the gold aware-character set.

    These questions are about first-order information access (who currently
    knows a fact), not nested second-order belief -- they don't carry a
    first/second-order split the way beliefQAs do.

    qtype: "answerability" or "info_accessibility".
    Fields: id, context, question, correct_set (list[str], the aware/
      answerable characters), universe (list[str], all named characters in
      this item -- aware + unaware -- for grading precision against
      hallucinated names).
    """
    key = "answerabilityQA_list" if qtype == "answerability" else "infoAccessibilityQA_list"
    ctx_key = "short_context" if context == "short" else "full_context"
    raw = _raw()
    items = []
    for conv in raw:
        qa = conv[key]
        items.append({
            "id": f"fantom_{conv['set_id']}_{qtype}_list",
            "context": conv[ctx_key],
            "question": qa["question"],
            "correct_set": qa["correct_answer"],
            "universe": sorted(set(qa["correct_answer"]) | set(qa["wrong_answer"])),
        })

    if n is not None and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tom-order", choices=["first-order", "second-order"], default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--list-qtype", choices=["answerability", "info_accessibility"], default=None)
    args = ap.parse_args()
    if args.list_qtype:
        items = load_fantom_list(qtype=args.list_qtype, n=args.n)
    else:
        items = load_fantom(tom_order=args.tom_order, n=args.n)
    print(f"Loaded {len(items)} items")
    for it in items[:3]:
        print(json.dumps({k: v for k, v in it.items() if k != "context"}, indent=2))
