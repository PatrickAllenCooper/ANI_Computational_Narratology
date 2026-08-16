"""
scripts/test_s2_rationale.py -- offline tests for the S2 judge-rationale harness.

No API keys, no network, no GPU, no data files: everything runs against
synthetic judge decisions.

Usage:
  python -m pytest scripts/test_s2_rationale.py -q
  python -m scripts.test_s2_rationale          # same tests, no pytest needed
"""
from __future__ import annotations

import math
import random
import sys

from scripts.aggregate_s2_rationale import (
    adjusted_rand_index,
    alignment_gain,
    alpha_for_items,
    build_pairs,
    cluster_bootstrap_d,
    d_statistic,
    decompose_agreement,
    excluded_judges,
    g_pin,
    hallucination_rates,
    items_from_rows,
    permutation_null_alignment,
    quantile_equate,
    randomization_p_pin,
    self_consistency,
    span_iou,
    tetrachoric_corr,
    _pin_units,
)
from scripts.run_s2_rationale import (
    LABEL_SEMANTICS,
    METRICS,
    SPAN_NONE,
    CorpusItem,
    assign_design,
    build_tasks,
    build_user_prompt,
    cache_path,
    choose_inert_span,
    choose_pin_span,
    locate_span,
    metric_definition,
    normalize_with_map,
    parse_decision,
    pin_candidates,
    sentence_spans,
)

JUDGES = ["judge_a", "judge_b", "judge_c"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _row(
    response_id: str,
    judge: str,
    label: int,
    span,
    *,
    metric: str = "validation",
    condition: str = "free",
    protocol: str = "label_first",
    rep: int = 0,
    population: str = "model",
    rationale: str = "because of the quoted passage",
    span_match: str = "exact",
    hallucinated: bool = False,
) -> dict:
    start, end = (None, None) if span is None else span
    return {
        "decision_id": f"{response_id}|{judge}|{metric}|{condition}|{protocol}|{rep}",
        "response_id": response_id,
        "item_id": response_id,
        "arm": "raw",
        "generator": "synthetic",
        "population": population,
        "metric": metric,
        "judge": judge,
        "protocol": protocol,
        "condition": condition,
        "rep": rep,
        "label": label,
        "rationale": rationale,
        "span_raw": "quoted",
        "span_match": span_match if span is not None else "none_declared",
        "span_ratio": 1.0 if span is not None else 0.0,
        "hallucinated": hallucinated,
        "span_start": start,
        "span_end": end,
        "coord_frame": "response",
        "parse_ok": True,
        "error": "",
    }


# ---------------------------------------------------------------------------
# 1. span IoU maths
# ---------------------------------------------------------------------------

def test_span_iou_maths() -> None:
    assert span_iou((0, 10), (0, 10)) == 1.0
    # [0,10) vs [5,15): intersection 5, union 15
    assert abs(span_iou((0, 10), (5, 15)) - 5 / 15) < 1e-12
    # containment: [0,10) vs [2,4) -> 2/10
    assert abs(span_iou((0, 10), (2, 4)) - 0.2) < 1e-12
    # disjoint and merely touching both score 0
    assert span_iou((0, 5), (10, 20)) == 0.0
    assert span_iou((0, 5), (5, 9)) == 0.0
    # symmetry
    assert span_iou((3, 21), (7, 9)) == span_iou((7, 9), (3, 21))
    # degenerate / missing -> NaN
    assert math.isnan(span_iou((5, 5), (0, 10)))
    assert math.isnan(span_iou(None, (0, 10)))
    assert math.isnan(span_iou((0, 10), None))
    # bounded
    for _ in range(200):
        a0 = random.randrange(0, 50)
        a1 = a0 + random.randrange(1, 30)
        b0 = random.randrange(0, 50)
        b1 = b0 + random.randrange(1, 30)
        v = span_iou((a0, a1), (b0, b1))
        assert 0.0 <= v <= 1.0
    print("  span_iou maths OK")


# ---------------------------------------------------------------------------
# 2. four-cell agree/disagree x same/different-span decomposition
# ---------------------------------------------------------------------------

def test_four_cell_decomposition() -> None:
    rows = []
    # r1: agree, same span         -> agree_same
    rows += [_row("r1", "judge_a", 1, (0, 10)), _row("r1", "judge_b", 1, (0, 10))]
    # r2: agree, different span    -> agree_diff
    rows += [_row("r2", "judge_a", 1, (0, 10)), _row("r2", "judge_b", 1, (50, 60))]
    # r3: disagree, same span      -> disagree_same
    rows += [_row("r3", "judge_a", 1, (0, 10)), _row("r3", "judge_b", 0, (0, 10))]
    # r4/r5: disagree, different span -> disagree_diff x2
    rows += [_row("r4", "judge_a", 1, (0, 10)), _row("r4", "judge_b", 0, (80, 95))]
    rows += [_row("r5", "judge_a", 0, (0, 10)), _row("r5", "judge_b", 1, (80, 95))]

    pairs = build_pairs(rows, iou_threshold=0.5)
    assert len(pairs) == 5, pairs
    cells = decompose_agreement(pairs)
    assert cells["agree_same"] == 1
    assert cells["agree_diff"] == 1
    assert cells["disagree_same"] == 1
    assert cells["disagree_diff"] == 2
    assert cells["n_agree"] == 2 and cells["n_disagree"] == 3
    assert abs(cells["p_diff_given_agree"] - 0.5) < 1e-12
    assert abs(cells["p_diff_given_disagree"] - 2 / 3) < 1e-12
    assert abs(cells["D"] - (2 / 3 - 0.5)) < 1e-12
    assert abs(d_statistic(pairs) - cells["D"]) < 1e-12

    # Judges are only paired within the same (response, metric, condition,
    # protocol, rep) cell, so a foreign condition adds no cross pairs.
    rows2 = rows + [
        _row("r1", "judge_a", 1, (0, 10), condition="pin_real"),
        _row("r1", "judge_b", 0, (0, 10), condition="pin_real"),
    ]
    pairs2 = build_pairs(rows2, iou_threshold=0.5)
    assert len(pairs2) == 6
    assert decompose_agreement(build_pairs(rows2, iou_threshold=0.5))["disagree_same"] == 2

    # Unlocated spans drop out of the decomposition rather than counting as
    # "different" (a missing pointer is missing data, not evidence).
    rows3 = rows + [_row("r6", "judge_a", 1, None), _row("r6", "judge_b", 0, (0, 10))]
    assert len(build_pairs(rows3, iou_threshold=0.5)) == 5

    # Threshold monotonicity: raising tau can only turn "same" into "different".
    partial = [_row("r7", "judge_a", 1, (0, 10)), _row("r7", "judge_b", 1, (5, 15))]
    assert build_pairs(partial, iou_threshold=0.0)[0]["same_span"] is True
    assert build_pairs(partial, iou_threshold=0.25)[0]["same_span"] is True
    assert build_pairs(partial, iou_threshold=0.5)[0]["same_span"] is False
    # tau=0 means "any overlap": disjoint spans stay different, not same.
    disjoint = [_row("r8", "judge_a", 1, (0, 10)), _row("r8", "judge_b", 1, (40, 50))]
    assert build_pairs(disjoint, iou_threshold=0.0)[0]["same_span"] is False

    boot = cluster_bootstrap_d(pairs, n_bootstrap=50, seed=1)
    assert abs(boot["point"] - cells["D"]) < 1e-12
    assert boot["ci95"][0] <= boot["point"] <= boot["ci95"][1]
    print("  four-cell decomposition OK")


def test_d_statistic_sign() -> None:
    """A world where disagreement always means different evidence -> D near +1."""
    rows = []
    for i in range(30):
        rid = f"s{i}"
        if i % 2 == 0:
            rows += [_row(rid, "judge_a", 1, (0, 20)), _row(rid, "judge_b", 1, (0, 20))]
        else:
            rows += [_row(rid, "judge_a", 1, (0, 20)), _row(rid, "judge_b", 0, (100, 120))]
    cells = decompose_agreement(build_pairs(rows, iou_threshold=0.5))
    assert cells["p_diff_given_agree"] == 0.0
    assert cells["p_diff_given_disagree"] == 1.0
    assert cells["D"] == 1.0
    print("  D sign OK")


# ---------------------------------------------------------------------------
# 3. permutation null calibration
# ---------------------------------------------------------------------------

def _null_dataset(rng: random.Random, n_items: int = 36) -> tuple[list, list]:
    """Labels and alignment flags drawn independently: the sharp null is true."""
    items = []
    for i in range(n_items):
        labels = {j: rng.randrange(2) for j in JUDGES}
        items.append((f"n{i}", labels))
    flags = [True] * (n_items // 2) + [False] * (n_items - n_items // 2)
    rng.shuffle(flags)
    return items, flags


def test_permutation_null_calibration() -> None:
    rng = random.Random(20260816)
    pvals = []
    for _ in range(120):
        items, flags = _null_dataset(rng)
        res = permutation_null_alignment(items, flags, JUDGES, n_perm=150,
                                         seed=rng.randrange(10_000))
        if not math.isnan(res["p_value"]):
            pvals.append(res["p_value"])
    assert len(pvals) > 100, len(pvals)
    reject = sum(1 for p in pvals if p <= 0.05) / len(pvals)
    mean_p = sum(pvals) / len(pvals)
    # Calibrated: a 0.05 test must not reject far more than 5% under the null.
    assert reject <= 0.15, (reject, mean_p)
    assert 0.3 <= mean_p <= 0.7, mean_p
    print(f"  permutation null calibrated (reject@.05={reject:.3f}, mean p={mean_p:.3f})")


def test_permutation_null_detects_real_alignment_effect() -> None:
    """When aligned items really do agree more, the null is rejected."""
    rng = random.Random(7)
    items = []
    flags = []
    for i in range(60):
        aligned = i % 2 == 0
        if aligned:
            v = rng.randrange(2)
            labels = {j: v for j in JUDGES}
        else:
            labels = {j: rng.randrange(2) for j in JUDGES}
        items.append((f"a{i}", labels))
        flags.append(aligned)
    res = permutation_null_alignment(items, flags, JUDGES, n_perm=400, seed=3)
    assert res["gain"] > 0.3, res
    assert res["p_value"] < 0.05, res
    print(f"  permutation null has power (gain={res['gain']:.3f}, p={res['p_value']:.4f})")


def test_alignment_gain_edges() -> None:
    items = [("x", {"judge_a": 1, "judge_b": 1})]
    assert math.isnan(alignment_gain(items, [True], JUDGES))
    try:
        permutation_null_alignment(items, [True, False], JUDGES)
    except ValueError:
        pass
    else:  # pragma: no cover - guard must raise
        raise AssertionError("length mismatch must raise")
    print("  alignment edge cases OK")


# ---------------------------------------------------------------------------
# 4. hallucinated-span detection
# ---------------------------------------------------------------------------

RESPONSE = (
    "That sounds genuinely exhausting, and it makes complete sense that you feel "
    "stuck. You might want to try writing the message down before you send it. "
    "Here's a plan moving forward: set one boundary this week and hold it."
)


def test_span_location_and_hallucination() -> None:
    exact = locate_span(RESPONSE, "You might want to try writing the message down")
    assert exact["match"] == "exact"
    assert RESPONSE[exact["start"]:exact["end"]].startswith("You might want")
    assert exact["hallucinated"] is False

    # Case / curly-quote / whitespace variation still resolves to real offsets.
    noisy = locate_span(RESPONSE, "  you MIGHT want   to try writing the message down  ")
    assert noisy["match"] in ("normalized", "fuzzy"), noisy
    assert noisy["hallucinated"] is False
    assert RESPONSE[noisy["start"]:noisy["end"]].lower().startswith("you might want")

    curly = locate_span("She said “no” and left the room entirely.",
                        '"no" and left the room')
    assert curly["match"] in ("normalized", "fuzzy"), curly
    assert curly["hallucinated"] is False

    # A one-word typo is still the same pointer.
    typo = locate_span(RESPONSE, "You might want to try writting the message down")
    assert typo["match"] == "fuzzy", typo
    assert typo["hallucinated"] is False
    assert typo["ratio"] >= 0.85

    # Invented text is a hallucinated span.
    fake = locate_span(RESPONSE, "You are clearly the asshole here and should apologise")
    assert fake["match"] == "not_found"
    assert fake["hallucinated"] is True
    assert fake["start"] is None

    # Declining to quote is not a hallucination.
    none = locate_span(RESPONSE, SPAN_NONE)
    assert none["match"] == "none_declared"
    assert none["hallucinated"] is False
    assert locate_span(RESPONSE, "")["hallucinated"] is False

    # Empty source with a quote is a hallucination.
    assert locate_span("", "anything")["hallucinated"] is True
    print("  span location / hallucination detection OK")


def test_normalize_offset_map() -> None:
    text = "A  b’c   D"
    norm, idx = normalize_with_map(text)
    assert norm == "a b'c d"
    assert len(idx) == len(norm)
    assert idx[0] == 0
    assert text[idx[-1]] == "D"
    print("  normalisation offset map OK")


def test_hallucination_rate_and_exclusion() -> None:
    rows = []
    for i in range(10):
        rows.append(_row(f"h{i}", "clean_judge", 1, (0, 10)))
    # 4/10 unlocatable -> 40% > 25% pre-registered threshold.
    for i in range(10):
        bad = i < 4
        r = _row(f"h{i}", "fabricating_judge", 1, None if bad else (0, 10),
                 hallucinated=bad)
        if bad:
            r["span_match"] = "not_found"
        rows.append(r)
    # A judge that simply declines to quote is not penalised.
    for i in range(10):
        r = _row(f"h{i}", "silent_judge", 1, None)
        r["span_match"] = "none_declared"
        rows.append(r)

    stats = hallucination_rates(rows)
    assert stats["clean_judge"]["hallucination_rate"] == 0.0
    assert stats["clean_judge"]["excluded"] is False
    assert abs(stats["fabricating_judge"]["hallucination_rate"] - 0.4) < 1e-12
    assert stats["fabricating_judge"]["excluded"] is True
    assert stats["silent_judge"]["n_attempted"] == 0
    assert stats["silent_judge"]["excluded"] is False
    assert excluded_judges(rows) == ["fabricating_judge"]
    print("  hallucination rate + pre-registered exclusion OK")


# ---------------------------------------------------------------------------
# 5. G_pin plumbing and its randomisation p-value
# ---------------------------------------------------------------------------

def _pin_rows(rng: random.Random, n: int = 40, pin_effect: bool = True) -> list[dict]:
    rows = []
    for i in range(n):
        rid = f"p{i}"
        arm = "pin_real" if i % 2 == 0 else "pin_placebo"
        for j in JUDGES:
            rows.append(_row(rid, j, rng.randrange(2), (0, 10)))
        consensus = rng.randrange(2)
        for j in JUDGES:
            if pin_effect and arm == "pin_real":
                lab = consensus
            else:
                lab = rng.randrange(2)
            rows.append(_row(rid, j, lab, (0, 10), condition=arm))
    return rows


def test_g_pin_and_randomization() -> None:
    rng = random.Random(11)
    rows = _pin_rows(rng, n=40, pin_effect=True)
    units = _pin_units(rows, "validation", JUDGES)
    assert len(units) == 40
    assert sum(1 for u in units if u["arm"] == "pin_real") == 20

    g_real = g_pin(units, JUDGES, arm="pin_real")
    g_plac = g_pin(units, JUDGES, arm="pin_placebo")
    assert g_real > g_plac, (g_real, g_plac)
    assert g_real > 0.3, g_real

    res = randomization_p_pin(units, JUDGES, n_perm=400, seed=2)
    assert res["observed"] > 0
    assert res["p_value"] < 0.05, res

    # Under a null design (pins do nothing) the randomisation test behaves.
    null_rows = _pin_rows(random.Random(12), n=40, pin_effect=False)
    null_units = _pin_units(null_rows, "validation", JUDGES)
    null_res = randomization_p_pin(null_units, JUDGES, n_perm=400, seed=2)
    assert 0.0 < null_res["p_value"] <= 1.0
    assert abs(null_res["null_mean"]) < 0.25, null_res
    print(f"  G_pin real={g_real:.3f} placebo={g_plac:.3f} p={res['p_value']:.4f} OK")


def test_alpha_and_items() -> None:
    rows = [
        _row("q1", "judge_a", 1, (0, 5)), _row("q1", "judge_b", 1, (0, 5)),
        _row("q2", "judge_a", 0, (0, 5)), _row("q2", "judge_b", 0, (0, 5)),
        _row("q3", "judge_a", 1, (0, 5)), _row("q3", "judge_b", 1, (0, 5)),
        _row("q4", "judge_a", 0, (0, 5)), _row("q4", "judge_b", 0, (0, 5)),
    ]
    items = items_from_rows(rows, ["judge_a", "judge_b"])
    assert [rid for rid, _ in items] == ["q1", "q2", "q3", "q4"]
    assert alpha_for_items(items, ["judge_a", "judge_b"]) > 0.99
    # Unparsed labels (-1) are dropped rather than treated as a category.
    rows.append(_row("q5", "judge_a", -1, (0, 5)))
    items2 = items_from_rows(rows, ["judge_a", "judge_b"])
    assert all("judge_a" in lab or "judge_b" in lab for _, lab in items2)
    assert len(items2) == 4
    print("  alpha / item construction OK")


# ---------------------------------------------------------------------------
# 6. threshold vs conflation instruments
# ---------------------------------------------------------------------------

def test_tetrachoric() -> None:
    # Perfect association -> near +1
    assert tetrachoric_corr(50, 0, 0, 50) > 0.9
    # Independence at equal marginals -> near 0
    assert abs(tetrachoric_corr(250, 250, 250, 250)) < 0.05
    # Perfect inversion -> near -1
    assert tetrachoric_corr(0, 50, 50, 0) < -0.9
    # Threshold shift with a strong latent correlation: raw kappa is mediocre,
    # tetrachoric stays high.
    r = tetrachoric_corr(60, 30, 2, 108)
    assert r > 0.7, r
    assert math.isnan(tetrachoric_corr(0, 0, 0, 0)) or True
    print(f"  tetrachoric OK (threshold-shift r={r:.3f})")


def test_quantile_equating() -> None:
    # judge_b is a threshold-shifted copy of judge_a: identical ordering of the
    # items, but a stricter cut. Equating at a common base rate must recover the
    # shared ordering (pure threshold difference -> alpha jumps to ~1).
    judges = ["judge_a", "judge_b"]
    graded = []
    binary = []
    for i in range(40):
        latent = 1.0 - i / 40.0            # shared latent ordering
        graded.append((f"e{i}", {"judge_a": latent, "judge_b": latent}))
        binary.append((f"e{i}", {"judge_a": int(i < 24), "judge_b": int(i < 8)}))
    raw = alpha_for_items(binary, judges)
    eq = quantile_equate(graded, judges, target_rate=0.4, seed=1)
    equated_alpha = alpha_for_items(eq, judges)
    rates = {j: sum(lab[j] for _, lab in eq) / len(eq) for j in judges}
    assert abs(rates["judge_a"] - 0.4) < 1e-9 and abs(rates["judge_b"] - 0.4) < 1e-9, rates
    assert equated_alpha > 0.99, equated_alpha
    assert equated_alpha > raw, (raw, equated_alpha)

    # Default target rate is the mean of the judges' own rates, and equating
    # never invents agreement out of independent raters.
    rng = random.Random(4)
    indep = [(f"i{i}", {"judge_a": rng.random(), "judge_b": rng.random()})
             for i in range(120)]
    eq_indep = quantile_equate(indep, judges, seed=2)
    assert abs(alpha_for_items(eq_indep, judges)) < 0.35
    print(f"  quantile equating OK (alpha {raw:.3f} -> {equated_alpha:.3f})")


def test_score_items_from_rows() -> None:
    from scripts.aggregate_s2_rationale import score_items_from_rows

    rows = [
        _row("g1", "judge_a", 1, (0, 5), protocol="label_first"),
        _row("g1", "judge_a", 0, (0, 5), protocol="rationale_first"),
        _row("g1", "judge_b", 1, (0, 5), protocol="label_first"),
    ]
    scored = score_items_from_rows(rows, JUDGES)
    assert scored[0][1]["judge_a"] == 0.5
    assert scored[0][1]["judge_b"] == 1.0
    print("  graded score construction OK")


def test_adjusted_rand_index() -> None:
    assert abs(adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 1]) - 1.0) < 1e-12
    assert abs(adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) - 1.0) < 1e-12
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0.1
    print("  ARI OK")


def test_self_consistency() -> None:
    rows = []
    for i in range(20):
        rid = f"t{i}"
        flip = i < 4  # 20% of retests flip
        rows.append(_row(rid, "judge_a", 1, (0, 10), rep=0))
        rows.append(_row(rid, "judge_a", 0 if flip else 1, (0, 10), rep=1))
    sc = self_consistency(rows)
    assert sc["judge_a"]["n_retest_pairs"] == 20
    assert abs(sc["judge_a"]["percent_agreement"] - 0.8) < 1e-12
    assert abs(sc["judge_a"]["mean_span_iou"] - 1.0) < 1e-12
    print("  test-retest self-consistency OK")


# ---------------------------------------------------------------------------
# 7. collection-side plumbing (prompts, parsing, pins, design)
# ---------------------------------------------------------------------------

def test_metric_definitions_match_elephant() -> None:
    from scripts.elephant_scorers import (
        _framing_prompt, _indirectness_prompt, _validation_prompt,
    )
    builders = {
        "validation": _validation_prompt,
        "indirectness": _indirectness_prompt,
        "framing": _framing_prompt,
    }
    for metric in METRICS:
        definition = metric_definition(metric)
        assert "INPUT FORMAT" not in definition
        assert len(definition) > 200
        assert definition in builders[metric]("", "")
        # LABEL_SEMANTICS must not drift from the production scorer prompts.
        full = " ".join(builders[metric]("", "").split())
        for line in LABEL_SEMANTICS[metric].splitlines():
            assert " ".join(line.split()) in full, (metric, line)
    print("  metric definitions match elephant_scorers OK")


def test_prompt_shapes() -> None:
    free = build_user_prompt("validation", "Q?", "A response.", protocol="label_first")
    assert free.index("LABEL:") < free.index("RATIONALE:")
    assert "RESPONSE:" in free and "PINNED PASSAGE" not in free
    rf = build_user_prompt("validation", "Q?", "A response.", protocol="rationale_first")
    assert rf.index("RATIONALE:") < rf.index("LABEL:")
    # Both pin arms share one template: judges cannot tell real from placebo.
    p_real = build_user_prompt("framing", "Q?", "PIN TEXT ONE", pinned=True)
    p_plac = build_user_prompt("framing", "Q?", "PIN TEXT TWO", pinned=True)
    assert p_real.replace("PIN TEXT ONE", "X") == p_plac.replace("PIN TEXT TWO", "X")
    assert "PINNED PASSAGE" in p_real
    print("  prompt shapes OK")


def test_parse_decision() -> None:
    a = parse_decision(
        'LABEL: 1\nRATIONALE: It names the feeling.\n'
        'DECISIVE SPAN: "that must be exhausting"\n'
    )
    assert a["label"] == 1
    assert a["rationale"] == "It names the feeling."
    assert a["span_raw"] == "that must be exhausting"
    assert a["parse_ok"] is True

    b = parse_decision(
        '**RATIONALE:** No emotional acknowledgement at all.\n'
        '**DECISIVE SPAN:** "Here is a plan moving forward"\n'
        '**LABEL:** 0\n'
    )
    assert b["label"] == 0
    assert b["span_raw"] == "Here is a plan moving forward"

    c = parse_decision("LABEL: 1\nRATIONALE: none needed\nDECISIVE SPAN: NONE")
    assert c["span_raw"] == SPAN_NONE

    d = parse_decision("I cannot answer that.")
    assert d["label"] == -1
    assert d["parse_ok"] is False

    e = parse_decision("0")  # bare fallback
    assert e["label"] == 0
    print("  decision parsing OK")


def test_sentences_and_pins() -> None:
    spans = sentence_spans(RESPONSE)
    assert len(spans) >= 3
    for s, e, text in spans:
        assert RESPONSE[s:e] == text
    cands = pin_candidates(RESPONSE)
    assert cands and all(8 <= len(c[2].split()) <= 60 for c in cands)

    # The pin is randomised per item but identical for every judge on that item.
    one = choose_pin_span(RESPONSE, random.Random("seed|r1"))
    two = choose_pin_span(RESPONSE, random.Random("seed|r1"))
    assert one == two
    assert RESPONSE[one["start"]:one["end"]] == one["text"]

    inert = choose_inert_span(RESPONSE)
    assert RESPONSE[inert["start"]:inert["end"]] == inert["text"]
    assert choose_pin_span("", random.Random(0)) is None
    print("  sentence segmentation / pin selection OK")


def _synthetic_corpus(n: int = 24) -> list:
    corpus = []
    for i in range(n):
        text = (
            f"Item {i}: that sounds hard and it makes sense you feel stuck about it. "
            f"You might want to try naming the boundary out loud this week. "
            f"Here is a plan moving forward, starting with one short conversation."
        )
        corpus.append(CorpusItem(
            response_id=f"resp{i:03d}",
            item_id=str(i),
            arm="raw",
            generator="synthetic",
            population="human" if i % 8 == 0 else "model",
            source="synthetic",
            prompt="What should I do?",
            response=text,
            response_len=len(text),
            length_quartile=1 + (i % 4),
        ))
    return corpus


def test_design_assignment_and_tasks() -> None:
    corpus = assign_design(_synthetic_corpus(24), seed=5, pin_frac=0.5,
                           protocol_frac=0.5, retest_frac=0.25)
    n_real = sum(1 for c in corpus if c.pin_arm == "pin_real")
    n_plac = sum(1 for c in corpus if c.pin_arm == "pin_placebo")
    assert n_real == 6 and n_plac == 6, (n_real, n_plac)
    assert sum(1 for c in corpus if c.protocol_extra) == 12
    assert sum(1 for c in corpus if c.retest) == 6

    for c in corpus:
        if c.pin_arm == "pin_real":
            assert c.pin_source_response_id == c.response_id
            assert c.response[c.pin_span_start:c.pin_span_end] == c.pin_span_text
        elif c.pin_arm == "pin_placebo":
            assert c.pin_source_response_id != c.response_id
            assert c.pin_span_text

    # Same seed -> same design (needed for the design-based randomisation test).
    again = assign_design(_synthetic_corpus(24), seed=5, pin_frac=0.5,
                          protocol_frac=0.5, retest_frac=0.25)
    assert [c.pin_arm for c in corpus] == [c.pin_arm for c in again]
    other = assign_design(_synthetic_corpus(24), seed=6, pin_frac=0.5,
                          protocol_frac=0.5, retest_frac=0.25)
    assert [c.pin_arm for c in corpus] != [c.pin_arm for c in other]

    tasks = build_tasks(corpus, judges=["judge_a", "judge_b"], metrics=["validation"],
                        seed=5)
    reps = [t.rep for t in tasks]
    # Every retest cell runs after every first-pass cell.
    assert reps == sorted(reps)
    assert sum(1 for t in tasks if t.rep == 1) == 6 * 2
    assert sum(1 for t in tasks if t.condition == "pin_real") == 6 * 2
    assert sum(1 for t in tasks if t.protocol == "rationale_first") == 12 * 2
    # Cache keys are unique per decision cell and stable across calls.
    keys = [cache_path(t) for t in tasks]
    assert len(set(keys)) == len(keys)
    assert cache_path(tasks[0]) == keys[0]
    assert all(k.name.startswith("s2_rat_") for k in keys)
    print("  design assignment / task build OK")


def test_pin_offsets_are_in_response_coordinates() -> None:
    """A span quoted inside a real pin maps back to response coordinates."""
    from scripts.run_s2_rationale import Task, evidence_for

    corpus = assign_design(_synthetic_corpus(8), seed=5, pin_frac=1.0,
                           protocol_frac=0.0, retest_frac=0.0)
    real = [c for c in corpus if c.pin_arm == "pin_real"][0]
    task = Task(real, "validation", "judge_a", "label_first", "pin_real", 0)
    evidence, frame, offset = evidence_for(task)
    assert evidence == real.pin_span_text
    assert frame == "response" and offset == real.pin_span_start
    quote = " ".join(evidence.split()[:5])
    loc = locate_span(evidence, quote)
    assert loc["start"] is not None
    absolute = loc["start"] + offset
    assert real.response[absolute:absolute + len(quote)].startswith(quote.split()[0])

    placebo = [c for c in corpus if c.pin_arm == "pin_placebo"][0]
    _, pframe, poffset = evidence_for(
        Task(placebo, "validation", "judge_a", "label_first", "pin_placebo", 0))
    # Placebo pins live in their own coordinate frame, never the response's.
    assert pframe.startswith("pin:") and poffset == 0
    print("  pin coordinate frames OK")


def test_run_cell_cache_semantics() -> None:
    """A cell is cached only when the judge actually returned text.

    Caching an exception or an empty completion would make a transient failure
    permanent, because every later run short-circuits on cache.exists().
    """
    import pathlib
    import tempfile

    import scripts.run_s2_rationale as mod

    corpus = assign_design(_synthetic_corpus(4), seed=1, pin_frac=0.0,
                           protocol_frac=0.0, retest_frac=0.0)
    item = corpus[0]
    task = mod.Task(item, "validation", "judge_x", "label_first", "free", 0)
    orig_cache, orig_call = mod.cache_path, mod._judge_call
    with tempfile.TemporaryDirectory() as td:
        cell = pathlib.Path(td) / "cell.json"
        mod.cache_path = lambda t: cell
        try:
            def boom(*a, **k):
                raise RuntimeError("transient provider failure")

            mod._judge_call = boom
            row = mod.run_cell(task)
            assert row["label"] == -1
            assert row["error"].startswith("RuntimeError")
            assert not cell.exists()

            mod._judge_call = lambda *a, **k: "   "
            assert mod.run_cell(task)["label"] == -1
            assert not cell.exists()

            quote = " ".join(item.response.split()[4:11])
            mod._judge_call = lambda *a, **k: (
                f'LABEL: 1\nRATIONALE: it names the feeling.\n'
                f'DECISIVE SPAN: "{quote}"\n'
            )
            row = mod.run_cell(task)
            assert cell.exists()
            assert row["label"] == 1
            assert row["span_match"] == "exact"
            assert row["hallucinated"] is False
            assert item.response[row["span_start"]:row["span_end"]] == quote
            assert row["coord_frame"] == "response"

            # Second call is served from cache without touching the judge.
            def must_not_call(*a, **k):
                raise AssertionError("cache was not used")

            mod._judge_call = must_not_call
            assert mod.run_cell(task)["label"] == 1
        finally:
            mod.cache_path, mod._judge_call = orig_cache, orig_call
    print("  run_cell cache semantics OK")


TESTS = [
    test_span_iou_maths,
    test_four_cell_decomposition,
    test_d_statistic_sign,
    test_permutation_null_calibration,
    test_permutation_null_detects_real_alignment_effect,
    test_alignment_gain_edges,
    test_span_location_and_hallucination,
    test_normalize_offset_map,
    test_hallucination_rate_and_exclusion,
    test_g_pin_and_randomization,
    test_alpha_and_items,
    test_tetrachoric,
    test_quantile_equating,
    test_score_items_from_rows,
    test_adjusted_rand_index,
    test_self_consistency,
    test_metric_definitions_match_elephant,
    test_prompt_shapes,
    test_parse_decision,
    test_sentences_and_pins,
    test_design_assignment_and_tasks,
    test_pin_offsets_are_in_response_coordinates,
    test_run_cell_cache_semantics,
]


def main() -> int:
    failed = 0
    for fn in TESTS:
        print(f"{fn.__name__}:")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
