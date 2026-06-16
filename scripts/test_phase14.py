"""
scripts/test_phase14.py -- Offline unit tests for Phase 14 sycophancy loss helpers.

Usage:
  python -m scripts.test_phase14
"""
from __future__ import annotations

import sys

from scripts.load_elephant import load_elephant
from scripts.syco_loss import batch_loss, batch_for_iter


def test_batch_loss() -> None:
    coded = [
        {"validation": 1, "indirectness": 0, "framing": 1, "cell_loss": 2, "n_metrics": 3},
        {"validation": 0, "indirectness": 1, "framing": 0, "cell_loss": 1, "n_metrics": 3},
    ]
    stats = batch_loss(coded)
    assert abs(stats["loss"] - (2 / 3 + 1 / 3) / 2) < 1e-6, stats
    assert abs(stats["validation_rate"] - 0.5) < 1e-6
    print("  batch_loss OK")


def test_train_holdout_disjoint() -> None:
    holdout = load_elephant("oeq", n=10, offset=0, allow_sample=True)
    train = load_elephant("oeq", n=10, offset=10, allow_sample=True)
    holdout_ids = {x.id for x in holdout}
    train_ids = {x.id for x in train}
    assert holdout_ids.isdisjoint(train_ids), (holdout_ids & train_ids)
    print("  train/holdout disjoint OK")


def test_batch_for_iter_wraps() -> None:
    items = [f"i{i}" for i in range(7)]
    b = batch_for_iter(items, 1, 5)  # type: ignore[arg-type]
    assert len(b) == 5
    print("  batch_for_iter OK")


def test_krippendorff_alpha() -> None:
    from scripts.krippendorff import cohens_kappa, krippendorff_alpha_nominal

    perfect = krippendorff_alpha_nominal([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    assert perfect > 0.99, perfect
    disagree = krippendorff_alpha_nominal([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    assert disagree < perfect, (disagree, perfect)
    kappa = cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0])
    assert -1.0 <= kappa <= 1.0
    print("  krippendorff_alpha OK")


def test_verdict_extraction() -> None:
    from scripts.elephant_scorers import extract_yta_nta, moral_both_nta

    assert extract_yta_nta("NTA") == "NTA"
    assert extract_yta_nta("You are the asshole. YTA.") == "YTA"
    assert moral_both_nta("NTA", "NTA") == 1
    assert moral_both_nta("YTA", "NTA") == 0
    print("  verdict_extraction OK")


def test_judge_text_reasoning_budget() -> None:
    from scripts.elephant_scorers import _judge_text
    from scripts.generators import _is_reasoning

    assert _is_reasoning("gpt-5.4-nano")
    assert not _is_reasoning("claude-haiku-4-5")
    # Offline: helper exists and is callable (live API tested separately)
    assert callable(_judge_text)
    print("  judge_text_reasoning_budget OK")


def main() -> int:
    print("Phase 14 offline tests:")
    test_batch_loss()
    test_train_holdout_disjoint()
    test_batch_for_iter_wraps()
    test_krippendorff_alpha()
    test_verdict_extraction()
    test_judge_text_reasoning_budget()
    print("All passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
