"""
scripts/krippendorff.py -- Krippendorff's alpha for nominal data.

Used by aggregate_judge_reliability.py for multi-judge ELEPHANT scoring.
"""
from __future__ import annotations

from collections import defaultdict


def krippendorff_alpha_nominal(
    ratings: list[list[int | None]],
) -> float:
    """Compute Krippendorff's alpha for nominal data.

    ratings: list of rater vectors, each length n_items. None = missing.
    Returns alpha in [-1, 1]; 1.0 = perfect agreement.
    """
    if not ratings or len(ratings) < 2:
        return float("nan")

    n_items = len(ratings[0])
    if any(len(r) != n_items for r in ratings):
        raise ValueError("All rater vectors must have equal length")

    # Transpose to units (items) x coders
    units: list[list[int]] = []
    for i in range(n_items):
        vals = [ratings[r][i] for r in range(len(ratings)) if ratings[r][i] is not None]
        if len(vals) >= 2:
            units.append(vals)

    if not units:
        return float("nan")

    # Coincidence counts for observed disagreement (nominal delta=0 if equal else 1)
    n_pairs = 0
    disagree = 0
    value_counts: dict[int, int] = defaultdict(int)

    for vals in units:
        m = len(vals)
        for v in vals:
            value_counts[v] += 1
        for a in range(m):
            for b in range(a + 1, m):
                n_pairs += 1
                if vals[a] != vals[b]:
                    disagree += 1

    if n_pairs == 0:
        return 1.0

    do = disagree / n_pairs

    total = sum(value_counts.values())
    de = 1.0
    for c in value_counts.values():
        p = c / total
        de -= p * p

    if de <= 1e-12:
        return 1.0
    return 1.0 - do / de


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for paired binary labels."""
    pairs = [
        (x, y) for x, y in zip(a, b)
        if x is not None and y is not None and x >= 0 and y >= 0
    ]
    n = len(pairs)
    if n == 0:
        return float("nan")
    agree = sum(1 for x, y in pairs if x == y) / n
    pa = sum(x for x, _ in pairs) / n
    pb = sum(y for _, y in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - expected) < 1e-9:
        return 1.0
    return (agree - expected) / (1 - expected)
