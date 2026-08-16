"""
scripts/aggregate_s2_rationale.py -- analysis for the S2 judge-rationale study.

Reads divergence_study_outputs/s2_rationale_raw.jsonl (written by
scripts/run_s2_rationale.py) and computes:

PRIMARY
  G_pin = alpha(span-pinned) - alpha(free), per metric, over the items the
  design randomly assigned to a pin arm.  Cluster-bootstrap CIs resample
  responses (the clustering unit), and the randomisation p-value comes from the
  experimental assignment itself: the sharp null "a real pin is no different
  from a placebo pin" is tested by permuting the pin-arm labels across assigned
  items, preserving counts, and recomputing the difference-in-differences
  (G_pin_real - G_pin_placebo).

CO-PRIMARY (purely mechanical -- string offsets only, zero judgment)
  D = P(different span | judges disagree) - P(different span | judges agree),
  where "different span" is IoU over character offsets below a threshold.  This
  is the non-circular disagreement test: span identity is a pointer into the
  response, not a restatement of the label.  The headline D is computed on the
  free condition only -- in the pin arms every judge is handed the same passage,
  so overlap is forced by construction -- and the pin arms are reported
  separately as a manipulation check.

SECONDARY
  * alpha within attention-aligned items, WITH a permutation null.  Alignment is
    not randomised and correlates mechanically with agreement (judges who agree
    tend to quote the same sentence), so the raw aligned-vs-unaligned gain is
    uninterpretable without permuting the alignment labels across items at a
    fixed aligned count.
  * per-judge hallucinated-span rate, and the pre-registered >25% exclusion.
  * intra-judge test-retest self-consistency (the self-consistency ceiling).
  * tetrachoric judge-pair correlation and alpha after per-judge quantile
    equating: tetrachoric is threshold-free, so if pairwise tetrachoric r is
    high while raw alpha is low, judges share a latent construct and differ in
    threshold (threshold hypothesis); if tetrachoric r is also low, they are
    measuring different things (conflation hypothesis).
  * alpha split by text population (model vs human), the domain-shift check.

EXPLORATORY ONLY (--exploratory-clustering)
  Rationale-prose clustering is NOT a primary test: a post-hoc rationale
  restates its own label, so label vocabulary leaks into cluster assignment and
  any "different reasons" result comes out mechanically positive.  When enabled,
  cluster stability is reported (bootstrap ARI and cross-representation ARI) and
  the result is labelled exploratory in the output.

Usage
  python -m scripts.aggregate_s2_rationale
  python -m scripts.aggregate_s2_rationale --n-bootstrap 2000 --n-perm 5000
  python -m scripts.aggregate_s2_rationale --exploratory-clustering
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Optional, Sequence

from scripts.bootstrap_reliability_cis import _bootstrap_ci_items, _percentile
from scripts.krippendorff import cohens_kappa, krippendorff_alpha_nominal

OUT_DIR = Path("divergence_study_outputs")
RAW_PATH = OUT_DIR / "s2_rationale_raw.jsonl"
CORPUS_PATH = OUT_DIR / "s2_rationale_corpus.jsonl"
SUMMARY_PATH = OUT_DIR / "s2_rationale_summary.json"
CELLS_CSV = OUT_DIR / "s2_rationale_span_cells.csv"
JUDGE_CSV = OUT_DIR / "s2_rationale_judge_diagnostics.csv"

# Pre-registered exclusion rule (mirrors run_s2_rationale).
HALLUCINATION_EXCLUSION_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.5
IOU_SWEEP = (0.0, 0.25, 0.5)

NAN = float("nan")


# ---------------------------------------------------------------------------
# Span geometry
# ---------------------------------------------------------------------------

def span_iou(a: Optional[Sequence], b: Optional[Sequence]) -> float:
    """Intersection-over-union of two character intervals [start, end).

    Returns NaN when either span is missing/degenerate; 0.0 for disjoint spans.
    Touching-but-not-overlapping intervals ([0,5) and [5,9)) score 0.0.
    """
    if a is None or b is None:
        return NAN
    try:
        a0, a1 = int(a[0]), int(a[1])
        b0, b1 = int(b[0]), int(b[1])
    except (TypeError, ValueError, IndexError):
        return NAN
    if a1 <= a0 or b1 <= b0:
        return NAN
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    if union <= 0:
        return NAN
    return inter / union


def row_span(row: dict) -> Optional[tuple]:
    s, e = row.get("span_start"), row.get("span_end")
    if s is None or e is None:
        return None
    try:
        s, e = int(s), int(e)
    except (TypeError, ValueError):
        return None
    if e <= s:
        return None
    return (s, e)


# ---------------------------------------------------------------------------
# Row loading / selection
# ---------------------------------------------------------------------------

def load_rows(path: Path = RAW_PATH) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def has_label(row: dict) -> bool:
    return row.get("label") in (0, 1)


def select(rows: Iterable[dict], **eq) -> list[dict]:
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in eq.items()):
            out.append(r)
    return out


def judges_in(rows: Iterable[dict]) -> list[str]:
    return sorted({r["judge"] for r in rows})


def items_from_rows(
    rows: Iterable[dict],
    judges: Sequence[str],
) -> list[tuple[str, dict]]:
    """[(response_id, {judge: label})], one entry per response, labels only."""
    bucket: dict[str, dict] = defaultdict(dict)
    for r in rows:
        if not has_label(r):
            continue
        if r["judge"] not in judges:
            continue
        bucket[r["response_id"]][r["judge"]] = int(r["label"])
    return sorted(bucket.items())


def score_items_from_rows(
    rows: Iterable[dict],
    judges: Sequence[str],
) -> list[tuple[str, dict]]:
    """[(response_id, {judge: mean label})] -- graded scores for quantile equating.

    Averaging a judge's labels for a response over the rows supplied (protocol
    variants and test-retest replicates) yields values in [0, 1] instead of a
    bare 0/1, so the per-judge ranking used by quantile_equate is only partly
    tied rather than fully tied.
    """
    bucket: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not has_label(r) or r["judge"] not in judges:
            continue
        bucket[r["response_id"]][r["judge"]].append(int(r["label"]))
    out = []
    for rid, per_judge in sorted(bucket.items()):
        out.append((rid, {j: sum(v) / len(v) for j, v in per_judge.items() if v}))
    return out


def alpha_for_items(
    items: Sequence[tuple[str, dict]],
    judges: Sequence[str],
) -> float:
    """Krippendorff alpha over (response -> judge -> label) units."""
    if len(items) < 2:
        return NAN
    rater_vecs = []
    for judge in judges:
        rater_vecs.append([it[1].get(judge) for it in items])
    if len(rater_vecs) < 2:
        return NAN
    return krippendorff_alpha_nominal(rater_vecs)


# ---------------------------------------------------------------------------
# Co-primary: agreement x span-identity decomposition
# ---------------------------------------------------------------------------

def build_pairs(
    rows: Sequence[dict],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    require_located: bool = True,
) -> list[dict]:
    """Judge pairs within each (response, metric, condition, protocol, rep) cell.

    Each pair carries: response_id (the bootstrap cluster), agree (labels equal),
    same_span (IoU >= threshold), and the raw IoU.  ``iou_threshold=0`` means
    "any overlap at all" (strictly IoU > 0), so disjoint spans still count as
    different; higher thresholds are inclusive.
    """
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if not has_label(r):
            continue
        key = (r["response_id"], r["metric"], r.get("condition", "free"),
               r.get("protocol", "label_first"), r.get("rep", 0))
        cells[key].append(r)

    pairs: list[dict] = []
    for key, group in sorted(cells.items()):
        group = sorted(group, key=lambda r: r["judge"])
        for a, b in combinations(group, 2):
            if a["judge"] == b["judge"]:
                continue
            sa, sb = row_span(a), row_span(b)
            iou = span_iou(sa, sb)
            located = sa is not None and sb is not None
            if require_located and not located:
                continue
            overlapping = located and not math.isnan(iou) and (
                iou > 0.0 if iou_threshold <= 0.0 else iou >= iou_threshold
            )
            pairs.append({
                "response_id": key[0],
                "metric": key[1],
                "condition": key[2],
                "protocol": key[3],
                "rep": key[4],
                "judge_a": a["judge"],
                "judge_b": b["judge"],
                "agree": int(a["label"]) == int(b["label"]),
                "iou": iou,
                "located": located,
                "same_span": bool(overlapping),
            })
    return pairs


def decompose_agreement(pairs: Sequence[dict]) -> dict:
    """Four-cell agree/disagree x same/different-span table plus D.

    D = P(different span | disagree) - P(different span | agree).
    """
    cells = {
        "agree_same": 0,
        "agree_diff": 0,
        "disagree_same": 0,
        "disagree_diff": 0,
    }
    for p in pairs:
        agree = "agree" if p["agree"] else "disagree"
        same = "same" if p["same_span"] else "diff"
        cells[f"{agree}_{same}"] += 1
    n_agree = cells["agree_same"] + cells["agree_diff"]
    n_disagree = cells["disagree_same"] + cells["disagree_diff"]
    p_diff_agree = cells["agree_diff"] / n_agree if n_agree else NAN
    p_diff_disagree = cells["disagree_diff"] / n_disagree if n_disagree else NAN
    d = (
        p_diff_disagree - p_diff_agree
        if n_agree and n_disagree else NAN
    )
    ious = [p["iou"] for p in pairs if not math.isnan(p["iou"])]
    return {
        **cells,
        "n_pairs": len(pairs),
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "p_diff_given_agree": p_diff_agree,
        "p_diff_given_disagree": p_diff_disagree,
        "D": d,
        "mean_iou": sum(ious) / len(ious) if ious else NAN,
        "mean_iou_agree": _mean([p["iou"] for p in pairs
                                 if p["agree"] and not math.isnan(p["iou"])]),
        "mean_iou_disagree": _mean([p["iou"] for p in pairs
                                    if not p["agree"] and not math.isnan(p["iou"])]),
    }


def d_statistic(pairs: Sequence[dict]) -> float:
    return decompose_agreement(pairs)["D"]


def _mean(vals: Sequence[float]) -> float:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else NAN


def cluster_bootstrap_d(
    pairs: Sequence[dict],
    *,
    n_bootstrap: int = 2000,
    seed: int = 44,
) -> dict:
    """Bootstrap D by resampling RESPONSES (clusters), not pairs."""
    by_resp: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_resp[p["response_id"]].append(p)
    clusters = [v for _, v in sorted(by_resp.items())]

    def stat(draw: Sequence[Sequence[dict]]) -> float:
        flat = [p for cl in draw for p in cl]
        return d_statistic(flat)

    return _bootstrap_ci_items(clusters, stat, n_bootstrap=n_bootstrap, seed=seed)


# ---------------------------------------------------------------------------
# Primary: G_pin
# ---------------------------------------------------------------------------

def _pin_units(
    rows: Sequence[dict],
    metric: str,
    judges: Sequence[str],
) -> list[dict]:
    """One unit per pin-assigned response: its free labels and its pinned labels."""
    free: dict[str, dict] = defaultdict(dict)
    pinned: dict[str, dict] = defaultdict(dict)
    arm: dict[str, str] = {}
    for r in rows:
        if r.get("metric") != metric or not has_label(r):
            continue
        if r.get("rep", 0) != 0 or r.get("protocol") != "label_first":
            continue
        if r["judge"] not in judges:
            continue
        rid = r["response_id"]
        cond = r.get("condition", "free")
        if cond == "free":
            free[rid][r["judge"]] = int(r["label"])
        elif cond in ("pin_real", "pin_placebo"):
            pinned[rid][r["judge"]] = int(r["label"])
            arm[rid] = cond
    units = []
    for rid in sorted(pinned):
        if rid not in free:
            continue
        units.append({
            "response_id": rid,
            "arm": arm[rid],
            "free": free[rid],
            "pin": pinned[rid],
        })
    return units


def g_pin(units: Sequence[dict], judges: Sequence[str], *, arm: Optional[str] = None) -> float:
    """alpha(pinned) - alpha(free) on the same set of responses."""
    sel = [u for u in units if arm is None or u["arm"] == arm]
    if len(sel) < 2:
        return NAN
    free_items = [(u["response_id"], u["free"]) for u in sel]
    pin_items = [(u["response_id"], u["pin"]) for u in sel]
    return alpha_for_items(pin_items, judges) - alpha_for_items(free_items, judges)


def pin_did(units: Sequence[dict], judges: Sequence[str]) -> float:
    """(G_pin | real pin) - (G_pin | placebo pin): the decisive causal contrast."""
    return g_pin(units, judges, arm="pin_real") - g_pin(units, judges, arm="pin_placebo")


def randomization_p_pin(
    units: Sequence[dict],
    judges: Sequence[str],
    *,
    n_perm: int = 2000,
    seed: int = 91,
) -> dict:
    """Design-based p-value: permute the randomised pin-arm labels across items.

    The assignment of an item to pin_real vs pin_placebo was randomised by
    run_s2_rationale.assign_design, so under the sharp null "real and placebo
    pins are interchangeable" the arm labels are exchangeable across items.
    """
    arms = [u["arm"] for u in units]
    n_real = sum(1 for a in arms if a == "pin_real")
    if n_real == 0 or n_real == len(arms) or len(arms) < 4:
        return {"observed": pin_did(units, judges), "p_value": NAN, "n_perm": 0,
                "null_mean": NAN, "null_sd": NAN}
    observed = pin_did(units, judges)
    rng = random.Random(seed)
    shuffled = list(arms)
    null = []
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        permuted = [dict(u, arm=a) for u, a in zip(units, shuffled)]
        val = pin_did(permuted, judges)
        if not math.isnan(val):
            null.append(val)
    if not null or math.isnan(observed):
        return {"observed": observed, "p_value": NAN, "n_perm": len(null),
                "null_mean": NAN, "null_sd": NAN}
    extreme = sum(1 for v in null if abs(v) >= abs(observed) - 1e-12)
    return {
        "observed": observed,
        "p_value": (extreme + 1) / (len(null) + 1),
        "n_perm": len(null),
        "null_mean": _mean(null),
        "null_sd": _sd(null),
        "null_ci95": [_percentile(null, 2.5), _percentile(null, 97.5)],
    }


def _sd(vals: Sequence[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 2:
        return NAN
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def bootstrap_g_pin(
    units: Sequence[dict],
    judges: Sequence[str],
    *,
    arm: Optional[str] = None,
    n_bootstrap: int = 2000,
    seed: int = 45,
) -> dict:
    sel = [u for u in units if arm is None or u["arm"] == arm]

    def stat(draw: Sequence[dict]) -> float:
        return g_pin(draw, judges, arm=None)

    out = _bootstrap_ci_items(sel, stat, n_bootstrap=n_bootstrap, seed=seed)
    out["n_responses"] = len(sel)
    return out


# ---------------------------------------------------------------------------
# Secondary: attention alignment with a permutation null
# ---------------------------------------------------------------------------

def alignment_flags(
    rows: Sequence[dict],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, bool]:
    """response_id -> all located judge spans mutually overlap at >= threshold."""
    pairs = build_pairs(rows, iou_threshold=iou_threshold, require_located=True)
    by_resp: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_resp[p["response_id"]].append(p)
    return {
        rid: bool(ps) and all(p["same_span"] for p in ps)
        for rid, ps in by_resp.items()
    }


def alignment_gain(
    items: Sequence[tuple[str, dict]],
    aligned: Sequence[bool],
    judges: Sequence[str],
) -> float:
    """alpha(aligned items) - alpha(unaligned items)."""
    a_items = [it for it, fl in zip(items, aligned) if fl]
    u_items = [it for it, fl in zip(items, aligned) if not fl]
    if len(a_items) < 2 or len(u_items) < 2:
        return NAN
    return alpha_for_items(a_items, judges) - alpha_for_items(u_items, judges)


def permutation_null_alignment(
    items: Sequence[tuple[str, dict]],
    aligned: Sequence[bool],
    judges: Sequence[str],
    *,
    n_perm: int = 2000,
    seed: int = 17,
) -> dict:
    """Permute alignment labels across items, preserving the aligned count.

    Required because alignment is not randomised: judges who agree mechanically
    tend to quote the same sentence, so an aligned-item alpha is inflated even
    under a null in which alignment carries no information.
    """
    if len(items) != len(aligned):
        raise ValueError("items and aligned must be the same length")
    observed = alignment_gain(items, aligned, judges)
    flags = list(aligned)
    n_aligned = sum(1 for f in flags if f)
    if n_aligned < 2 or (len(flags) - n_aligned) < 2:
        return {"gain": observed, "p_value": NAN, "n_perm": 0,
                "null_mean": NAN, "null_sd": NAN,
                "n_aligned": n_aligned, "n_unaligned": len(flags) - n_aligned}
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        rng.shuffle(flags)
        val = alignment_gain(items, flags, judges)
        if not math.isnan(val):
            null.append(val)
    if not null or math.isnan(observed):
        p = NAN
    else:
        extreme = sum(1 for v in null if abs(v) >= abs(observed) - 1e-12)
        p = (extreme + 1) / (len(null) + 1)
    return {
        "gain": observed,
        "p_value": p,
        "n_perm": len(null),
        "null_mean": _mean(null),
        "null_sd": _sd(null),
        "n_aligned": n_aligned,
        "n_unaligned": len(flags) - n_aligned,
    }


# ---------------------------------------------------------------------------
# Secondary: hallucination, self-consistency, threshold vs conflation
# ---------------------------------------------------------------------------

def hallucination_rates(rows: Sequence[dict]) -> dict[str, dict]:
    """Per judge: fraction of quoted spans that could not be found in the source."""
    stats: dict[str, dict] = defaultdict(
        lambda: {"n_attempted": 0, "n_hallucinated": 0, "n_declined": 0, "n_rows": 0}
    )
    for r in rows:
        s = stats[r["judge"]]
        s["n_rows"] += 1
        if r.get("span_match") == "none_declared":
            s["n_declined"] += 1
            continue
        s["n_attempted"] += 1
        if r.get("hallucinated"):
            s["n_hallucinated"] += 1
    out = {}
    for judge, s in stats.items():
        rate = s["n_hallucinated"] / s["n_attempted"] if s["n_attempted"] else NAN
        out[judge] = {
            **s,
            "hallucination_rate": rate,
            "excluded": bool(not math.isnan(rate) and rate > HALLUCINATION_EXCLUSION_THRESHOLD),
        }
    return dict(sorted(out.items()))


def excluded_judges(rows: Sequence[dict]) -> list[str]:
    return [j for j, s in hallucination_rates(rows).items() if s["excluded"]]


def self_consistency(rows: Sequence[dict]) -> dict[str, dict]:
    """Intra-judge test-retest: rep 0 vs rep 1 on the same decision cell."""
    cells: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if not has_label(r):
            continue
        key = (r["judge"], r["response_id"], r["metric"],
               r.get("condition", "free"), r.get("protocol", "label_first"))
        cells[key][r.get("rep", 0)] = int(r["label"])
    per_judge: dict[str, list[tuple[int, int]]] = defaultdict(list)
    span_iou_by_judge: dict[str, list[float]] = defaultdict(list)
    spans: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        key = (r["judge"], r["response_id"], r["metric"],
               r.get("condition", "free"), r.get("protocol", "label_first"))
        s = row_span(r)
        if s is not None:
            spans[key][r.get("rep", 0)] = s
    for key, reps in cells.items():
        if 0 in reps and 1 in reps:
            per_judge[key[0]].append((reps[0], reps[1]))
            sp = spans.get(key, {})
            if 0 in sp and 1 in sp:
                iou = span_iou(sp[0], sp[1])
                if not math.isnan(iou):
                    span_iou_by_judge[key[0]].append(iou)
    out = {}
    for judge, pairs in sorted(per_judge.items()):
        if not pairs:
            continue
        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        out[judge] = {
            "n_retest_pairs": len(pairs),
            "percent_agreement": agree,
            "kappa": cohens_kappa([a for a, _ in pairs], [b for _, b in pairs]),
            "mean_span_iou": _mean(span_iou_by_judge.get(judge, [])),
        }
    return out


def _bvn_cdf(h: float, k: float, rho: float, *, n_steps: int = 400) -> float:
    """P(X <= h, Y <= k) for standard bivariate normal with correlation rho."""
    nd = NormalDist()
    base = nd.cdf(h) * nd.cdf(k)
    rho = max(-0.999999, min(0.999999, rho))
    if abs(rho) < 1e-12:
        return base

    def integrand(r: float) -> float:
        one = 1.0 - r * r
        if one <= 1e-12:
            return 0.0
        expo = -(h * h - 2 * r * h * k + k * k) / (2 * one)
        if expo < -700:
            return 0.0
        return math.exp(expo) / (2 * math.pi * math.sqrt(one))

    # Composite Simpson over [0, rho].
    n = n_steps if n_steps % 2 == 0 else n_steps + 1
    step = rho / n
    total = integrand(0.0) + integrand(rho)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * integrand(i * step)
    return base + total * step / 3.0


def tetrachoric_corr(n00: int, n01: int, n10: int, n11: int) -> float:
    """Tetrachoric correlation for a 2x2 table (0.5 continuity correction).

    Threshold-free: estimates the latent correlation between two binary raters
    net of their different marginal base rates.
    """
    if min(n00, n01, n10, n11) == 0:
        n00, n01, n10, n11 = n00 + 0.5, n01 + 0.5, n10 + 0.5, n11 + 0.5
    n = n00 + n01 + n10 + n11
    if n <= 0:
        return NAN
    p_a0 = (n00 + n01) / n           # rater A says 0
    p_b0 = (n00 + n10) / n           # rater B says 0
    if not (0 < p_a0 < 1) or not (0 < p_b0 < 1):
        return NAN
    nd = NormalDist()
    h = nd.inv_cdf(p_a0)
    k = nd.inv_cdf(p_b0)
    target = n00 / n
    lo, hi = -0.999, 0.999
    f_lo = _bvn_cdf(h, k, lo) - target
    f_hi = _bvn_cdf(h, k, hi) - target
    if f_lo > 0:
        return -0.999
    if f_hi < 0:
        return 0.999
    for _ in range(80):
        mid = (lo + hi) / 2
        f_mid = _bvn_cdf(h, k, mid) - target
        if f_mid < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return (lo + hi) / 2


def judge_pair_tetrachoric(
    items: Sequence[tuple[str, dict]],
    judges: Sequence[str],
) -> list[dict]:
    out = []
    for ja, jb in combinations(judges, 2):
        n00 = n01 = n10 = n11 = 0
        for _, labels in items:
            a, b = labels.get(ja), labels.get(jb)
            if a is None or b is None:
                continue
            if a == 0 and b == 0:
                n00 += 1
            elif a == 0 and b == 1:
                n01 += 1
            elif a == 1 and b == 0:
                n10 += 1
            else:
                n11 += 1
        n = n00 + n01 + n10 + n11
        out.append({
            "judge_a": ja,
            "judge_b": jb,
            "n": n,
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "kappa": cohens_kappa(
                [0] * n00 + [0] * n01 + [1] * n10 + [1] * n11,
                [0] * n00 + [1] * n01 + [0] * n10 + [1] * n11,
            ) if n else NAN,
            "tetrachoric_r": tetrachoric_corr(n00, n01, n10, n11) if n else NAN,
        })
    return out


def quantile_equate(
    items: Sequence[tuple[str, dict]],
    judges: Sequence[str],
    *,
    target_rate: Optional[float] = None,
    seed: int = 5,
) -> list[tuple[str, dict]]:
    """Re-dichotomise each judge at a common base rate (per-judge quantile equating).

    Each judge's per-item score is their mean label over the rows supplied
    (values in [0, 1]); ties are broken by a seeded jitter so the ranking is
    total and reproducible.  The top ``target_rate`` fraction of each judge's
    ranking is relabelled 1.  If judges differ only in threshold, alpha rises
    sharply after equating; if they conflate different constructs, it does not.
    """
    scored: dict[str, list[tuple[float, str]]] = {j: [] for j in judges}
    for rid, labels in items:
        for j in judges:
            if j in labels:
                scored[j].append((float(labels[j]), rid))
    rates = [
        sum(v for v, _ in vals) / len(vals)
        for vals in scored.values() if vals
    ]
    if target_rate is None:
        target_rate = sum(rates) / len(rates) if rates else 0.5

    equated: dict[str, dict[str, int]] = {}
    for j, vals in scored.items():
        rng = random.Random(f"{seed}|{j}")
        ranked = sorted(vals, key=lambda t: (-t[0], rng.random(), t[1]))
        k = int(round(target_rate * len(ranked)))
        equated[j] = {}
        for i, (_, rid) in enumerate(ranked):
            equated[j][rid] = 1 if i < k else 0

    out = []
    for rid, labels in items:
        new = {j: equated[j][rid] for j in judges if j in labels and rid in equated[j]}
        out.append((rid, new))
    return out


# ---------------------------------------------------------------------------
# Exploratory: rationale prose clustering with stability
# ---------------------------------------------------------------------------

_LABEL_LEAK_TOKENS = {
    "0", "1", "label", "yes", "no", "not", "does", "validating", "validation",
    "valid", "suggestive", "suggest", "suggestion", "premise", "challenge",
    "challenges", "challenging", "emotional", "emotionally", "indirect",
    "indirectness", "framing",
}
_TOKEN_RE = re.compile(r"[a-z']+")


def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> float:
    """ARI between two clusterings of the same items."""
    if len(a) != len(b):
        raise ValueError("clusterings must be the same length")
    n = len(a)
    if n < 2:
        return NAN
    table: dict[tuple, int] = defaultdict(int)
    ra: dict[int, int] = defaultdict(int)
    rb: dict[int, int] = defaultdict(int)
    for x, y in zip(a, b):
        table[(x, y)] += 1
        ra[x] += 1
        rb[y] += 1

    def c2(m: int) -> float:
        return m * (m - 1) / 2.0

    sum_ij = sum(c2(v) for v in table.values())
    sum_a = sum(c2(v) for v in ra.values())
    sum_b = sum(c2(v) for v in rb.values())
    total = c2(n)
    expected = sum_a * sum_b / total if total else 0.0
    maxi = (sum_a + sum_b) / 2.0
    if abs(maxi - expected) < 1e-12:
        return 1.0 if abs(sum_ij - expected) < 1e-12 else 0.0
    return (sum_ij - expected) / (maxi - expected)


def _hash_vectors(texts: Sequence[str], *, dim: int = 256, mode: str = "word"):
    """Hashed TF vectors (numpy). ``mode`` is 'word' or 'char4'."""
    import numpy as np  # lazy: optional heavy dep
    mat = np.zeros((len(texts), dim), dtype="float64")
    for i, t in enumerate(texts):
        low = (t or "").lower()
        if mode == "word":
            toks = [w for w in _TOKEN_RE.findall(low) if w not in _LABEL_LEAK_TOKENS]
        else:
            stripped = re.sub(r"\s+", " ", low)
            toks = [stripped[j:j + 4] for j in range(max(0, len(stripped) - 3))]
        for tok in toks:
            h = _token_hash(tok) % dim
            mat[i, h] += 1.0
    # sublinear tf + L2 normalise (cosine geometry)
    mat = np.log1p(mat)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _token_hash(tok: str) -> int:
    """Stable across processes (unlike hash())."""
    return int(hashlib.sha1(tok.encode("utf-8", "ignore")).hexdigest()[:8], 16)


def _kmeans(mat, k: int, *, seed: int = 0, iters: int = 30) -> list[int]:
    import numpy as np
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    if n == 0:
        return []
    k = max(1, min(k, n))
    idx = rng.choice(n, size=k, replace=False)
    centers = mat[idx].copy()
    labels = [0] * n
    for _ in range(iters):
        sims = mat @ centers.T
        new = sims.argmax(axis=1)
        if list(new) == labels:
            break
        labels = list(int(x) for x in new)
        for c in range(k):
            members = mat[new == c]
            if len(members):
                v = members.mean(axis=0)
                nrm = float(np.linalg.norm(v)) or 1.0
                centers[c] = v / nrm
    return labels


def rationale_clustering(
    rows: Sequence[dict],
    *,
    k: int = 4,
    n_bootstrap: int = 20,
    seed: int = 3,
) -> dict:
    """EXPLORATORY. Cluster rationale prose and report cluster stability.

    Not a primary test: a post-hoc rationale restates its own label, so label
    vocabulary leaks into the clustering and a "judges gave different reasons"
    result comes out mechanically positive.  Reported with bootstrap ARI and
    cross-representation ARI so the instability is visible.
    """
    texts = [r.get("rationale", "") for r in rows if (r.get("rationale") or "").strip()]
    if len(texts) < max(8, k * 2):
        return {"status": "skipped", "reason": "too few rationales", "n": len(texts)}
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return {"status": "skipped", "reason": "numpy unavailable", "n": len(texts)}

    word = _hash_vectors(texts, mode="word")
    char = _hash_vectors(texts, mode="char4")
    lab_word = _kmeans(word, k, seed=seed)
    lab_char = _kmeans(char, k, seed=seed)
    cross_ari = adjusted_rand_index(lab_word, lab_char)

    rng = random.Random(seed)
    boots = []
    n = len(texts)
    for _ in range(n_bootstrap):
        draw = sorted({rng.randrange(n) for _ in range(n)})
        if len(draw) < k * 2:
            continue
        sub = word[draw]
        lab_sub = _kmeans(sub, k, seed=rng.randrange(10_000))
        boots.append(adjusted_rand_index(lab_sub, [lab_word[i] for i in draw]))
    return {
        "status": "exploratory",
        "n_rationales": n,
        "k": k,
        "cross_representation_ari": cross_ari,
        "bootstrap_ari_mean": _mean(boots),
        "bootstrap_ari_ci95": [_percentile(boots, 2.5), _percentile(boots, 97.5)] if boots else [NAN, NAN],
        "n_bootstrap": len(boots),
        "caveat": (
            "Post-hoc rationales restate their labels; cluster assignment leaks "
            "label vocabulary. Exploratory only -- never the primary test."
        ),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def analyse(
    rows: Sequence[dict],
    *,
    n_bootstrap: int = 2000,
    n_perm: int = 2000,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    clustering: bool = False,
    seed: int = 44,
) -> dict:
    metrics = sorted({r["metric"] for r in rows})
    all_judges = judges_in(rows)
    hall = hallucination_rates(rows)
    dropped = [j for j, s in hall.items() if s["excluded"]]
    span_judges = [j for j in all_judges if j not in dropped]

    summary: dict = {
        "n_decisions": len(rows),
        "n_responses": len({r["response_id"] for r in rows}),
        "judges": all_judges,
        "metrics": metrics,
        "iou_threshold": iou_threshold,
        "n_bootstrap": n_bootstrap,
        "n_perm": n_perm,
        "hallucination": hall,
        "excluded_judges_for_span_analyses": dropped,
        "exclusion_rule": (
            f"pre-registered: judge excluded from rationale/span analyses when "
            f">{HALLUCINATION_EXCLUSION_THRESHOLD:.0%} of quoted spans cannot be "
            f"located in the source text"
        ),
        "self_consistency": self_consistency(rows),
        "primary_G_pin": {},
        "coprimary_D": {},
        "secondary": {"alignment": {}, "threshold_vs_conflation": {}, "population": {}},
        "protocol_effect": {},
    }

    # --- PRIMARY: G_pin -----------------------------------------------------
    for metric in metrics:
        units = _pin_units(rows, metric, all_judges)
        if len(units) < 4:
            summary["primary_G_pin"][metric] = {"status": "insufficient", "n_units": len(units)}
            continue
        entry = {
            "n_units": len(units),
            "n_real": sum(1 for u in units if u["arm"] == "pin_real"),
            "n_placebo": sum(1 for u in units if u["arm"] == "pin_placebo"),
            "alpha_free_all": alpha_for_items(
                [(u["response_id"], u["free"]) for u in units], all_judges),
            "alpha_pinned_all": alpha_for_items(
                [(u["response_id"], u["pin"]) for u in units], all_judges),
            "G_pin_all": g_pin(units, all_judges),
            "G_pin_real": bootstrap_g_pin(units, all_judges, arm="pin_real",
                                          n_bootstrap=n_bootstrap, seed=seed),
            "G_pin_placebo": bootstrap_g_pin(units, all_judges, arm="pin_placebo",
                                             n_bootstrap=n_bootstrap, seed=seed + 1),
            "G_pin_overall": bootstrap_g_pin(units, all_judges, arm=None,
                                             n_bootstrap=n_bootstrap, seed=seed + 2),
            "did_real_minus_placebo": randomization_p_pin(
                units, all_judges, n_perm=n_perm, seed=seed + 3),
        }
        summary["primary_G_pin"][metric] = entry

    # --- CO-PRIMARY: D ------------------------------------------------------
    span_rows = [r for r in rows if r["judge"] in span_judges]
    for metric in metrics:
        mrows = select(span_rows, metric=metric)
        # Headline D uses the FREE condition only: there each judge chooses its
        # own evidence, so span identity is informative. In the pin arms every
        # judge is handed the same passage, which mechanically forces overlap.
        free_rows_all = select(mrows, condition="free")
        per_threshold = {}
        for tau in sorted({iou_threshold, *IOU_SWEEP}):
            pairs = build_pairs(free_rows_all, iou_threshold=tau)
            cells = decompose_agreement(pairs)
            if tau == iou_threshold:
                cells["bootstrap"] = cluster_bootstrap_d(
                    pairs, n_bootstrap=n_bootstrap, seed=seed + 4)
            per_threshold[f"iou_{tau:g}"] = cells
        # Condition breakdown at the primary threshold.
        by_condition = {}
        for cond in sorted({r.get("condition", "free") for r in mrows}):
            pairs = build_pairs(select(mrows, condition=cond),
                                iou_threshold=iou_threshold)
            by_condition[cond] = decompose_agreement(pairs)
        summary["coprimary_D"][metric] = {
            "scope": "condition=free (judge-chosen evidence)",
            "by_iou_threshold": per_threshold,
            "by_condition": by_condition,
        }

    # --- SECONDARY ----------------------------------------------------------
    for metric in metrics:
        free_rows = [r for r in select(span_rows, metric=metric, condition="free",
                                       protocol="label_first")
                     if r.get("rep", 0) == 0]
        flags = alignment_flags(free_rows, iou_threshold=iou_threshold)
        items = items_from_rows(free_rows, span_judges)
        items = [(rid, labels) for rid, labels in items if rid in flags]
        aligned = [flags[rid] for rid, _ in items]
        summary["secondary"]["alignment"][metric] = {
            **permutation_null_alignment(items, aligned, span_judges,
                                         n_perm=n_perm, seed=seed + 5),
            "alpha_aligned": alpha_for_items(
                [it for it, f in zip(items, aligned) if f], span_judges),
            "alpha_unaligned": alpha_for_items(
                [it for it, f in zip(items, aligned) if not f], span_judges),
            "alpha_all": alpha_for_items(items, span_judges),
            "note": (
                "Alignment is not randomised and correlates mechanically with "
                "agreement; only the permutation p-value is interpretable."
            ),
        }

        all_free = [r for r in select(rows, metric=metric, condition="free",
                                      protocol="label_first")
                    if r.get("rep", 0) == 0]
        full_items = items_from_rows(all_free, all_judges)
        # Equating ranks judges on graded scores pooled over protocol variants
        # and retest replicates (pin arms excluded: they change the object of
        # judgment), then re-dichotomises every judge at one common base rate.
        graded = score_items_from_rows(select(rows, metric=metric, condition="free"),
                                       all_judges)
        eq_items = quantile_equate(graded, all_judges)
        summary["secondary"]["threshold_vs_conflation"][metric] = {
            "alpha_raw": alpha_for_items(full_items, all_judges),
            "alpha_quantile_equated": alpha_for_items(eq_items, all_judges),
            "base_rates": {
                j: _mean([lab[j] for _, lab in full_items if j in lab])
                for j in all_judges
            },
            "judge_pairs": judge_pair_tetrachoric(full_items, all_judges),
            "interpretation": (
                "high tetrachoric r with low raw alpha => threshold difference; "
                "low tetrachoric r => conflation of different constructs"
            ),
        }

        pop = {}
        for population in sorted({r.get("population", "model") for r in all_free}):
            prows = select(all_free, population=population)
            pit = items_from_rows(prows, all_judges)
            pop[population] = {
                "n_responses": len(pit),
                "alpha": alpha_for_items(pit, all_judges),
            }
        summary["secondary"]["population"][metric] = pop

        # Protocol order effect (label_first vs rationale_first, same items).
        lf = {r["response_id"]: r for r in all_free}
        rf_rows = [r for r in select(rows, metric=metric, condition="free",
                                     protocol="rationale_first")
                   if r.get("rep", 0) == 0]
        flips = 0
        matched = 0
        per_judge_flip: dict[str, list[int]] = defaultdict(list)
        lf_by_key = {(r["response_id"], r["judge"]): r for r in all_free}
        for r in rf_rows:
            base = lf_by_key.get((r["response_id"], r["judge"]))
            if base is None or not has_label(base) or not has_label(r):
                continue
            matched += 1
            flipped = int(base["label"]) != int(r["label"])
            flips += int(flipped)
            per_judge_flip[r["judge"]].append(int(flipped))
        rf_items = items_from_rows(rf_rows, all_judges)
        summary["protocol_effect"][metric] = {
            "n_matched_decisions": matched,
            "flip_rate": flips / matched if matched else NAN,
            "flip_rate_by_judge": {j: _mean(v) for j, v in sorted(per_judge_flip.items())},
            "alpha_label_first": alpha_for_items(items_from_rows(all_free, all_judges),
                                                 all_judges),
            "alpha_rationale_first": alpha_for_items(rf_items, all_judges),
            "n_label_first_items": len(lf),
        }

    if clustering:
        summary["exploratory_clustering"] = {
            metric: rationale_clustering(select(span_rows, metric=metric), seed=seed)
            for metric in metrics
        }
    else:
        summary["exploratory_clustering"] = {
            "status": "disabled",
            "reason": (
                "Rationale-prose clustering is exploratory only (label vocabulary "
                "leaks into cluster assignment). Enable with --exploratory-clustering."
            ),
        }
    return summary


def write_cells_csv(summary: dict, path: Path = CELLS_CSV) -> None:
    fields = [
        "metric", "scope", "n_pairs", "agree_same", "agree_diff",
        "disagree_same", "disagree_diff", "p_diff_given_agree",
        "p_diff_given_disagree", "D", "mean_iou_agree", "mean_iou_disagree",
    ]
    rows = []
    for metric, blk in summary.get("coprimary_D", {}).items():
        for scope, cells in blk.get("by_iou_threshold", {}).items():
            rows.append({"metric": metric, "scope": scope,
                         **{k: cells.get(k) for k in fields if k not in ("metric", "scope")}})
        for scope, cells in blk.get("by_condition", {}).items():
            rows.append({"metric": metric, "scope": f"condition:{scope}",
                         **{k: cells.get(k) for k in fields if k not in ("metric", "scope")}})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_judge_csv(summary: dict, path: Path = JUDGE_CSV) -> None:
    fields = [
        "judge", "n_rows", "n_attempted", "n_declined", "n_hallucinated",
        "hallucination_rate", "excluded", "n_retest_pairs",
        "retest_percent_agreement", "retest_kappa", "retest_mean_span_iou",
    ]
    sc = summary.get("self_consistency", {})
    rows = []
    for judge, s in summary.get("hallucination", {}).items():
        c = sc.get(judge, {})
        rows.append({
            "judge": judge,
            "n_rows": s.get("n_rows"),
            "n_attempted": s.get("n_attempted"),
            "n_declined": s.get("n_declined"),
            "n_hallucinated": s.get("n_hallucinated"),
            "hallucination_rate": s.get("hallucination_rate"),
            "excluded": s.get("excluded"),
            "n_retest_pairs": c.get("n_retest_pairs"),
            "retest_percent_agreement": c.get("percent_agreement"),
            "retest_kappa": c.get("kappa"),
            "retest_mean_span_iou": c.get("mean_span_iou"),
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate the S2 judge-rationale study")
    ap.add_argument("--raw", type=Path, default=RAW_PATH)
    ap.add_argument("--out", type=Path, default=SUMMARY_PATH)
    ap.add_argument("--cells-csv", type=Path, default=CELLS_CSV)
    ap.add_argument("--judge-csv", type=Path, default=JUDGE_CSV)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--exploratory-clustering", action="store_true",
                    help="add rationale-prose clustering (exploratory, with stability)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_bootstrap = 50
        args.n_perm = 50

    rows = load_rows(args.raw)
    if not rows:
        print(f"No rows in {args.raw}; run scripts.run_s2_rationale first.")
        return 1

    summary = analyse(
        rows,
        n_bootstrap=args.n_bootstrap,
        n_perm=args.n_perm,
        iou_threshold=args.iou_threshold,
        clustering=args.exploratory_clustering,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    write_cells_csv(summary, args.cells_csv)
    write_judge_csv(summary, args.judge_csv)
    print(f"Wrote {args.out}, {args.cells_csv}, {args.judge_csv}")

    for metric, blk in summary.get("primary_G_pin", {}).items():
        if blk.get("status") == "insufficient":
            print(f"  G_pin {metric}: insufficient units ({blk['n_units']})")
            continue
        real = blk["G_pin_real"]
        did = blk["did_real_minus_placebo"]
        print(
            f"  G_pin {metric}: real={real['point']:.3f} "
            f"[{real['ci95'][0]:.3f}, {real['ci95'][1]:.3f}] "
            f"DiD={did['observed']:.3f} p={did['p_value']}"
        )
    for metric, blk in summary.get("coprimary_D", {}).items():
        cells = blk["by_iou_threshold"].get(f"iou_{args.iou_threshold:g}", {})
        boot = cells.get("bootstrap", {})
        ci = boot.get("ci95", [NAN, NAN])
        print(
            f"  D {metric}: {cells.get('D')} "
            f"[{ci[0]}, {ci[1]}] over {cells.get('n_pairs')} judge pairs"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
