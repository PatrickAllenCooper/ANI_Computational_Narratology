"""
scripts/test_propensity.py -- offline unit tests for the exact teacher-forced
verdict-propensity instrument.

Everything here runs with no GPU, no API keys, and no network: synthetic
logprob arrays are fed through the numerical core, and a stub tokenizer drives
the tokenization audit.  Any test that would need real weights is absent by
design -- the live checks live in the CURC job scripts.

Usage:
  python -m scripts.test_propensity          # standalone runner
  python -m pytest scripts/test_propensity.py -q
"""
from __future__ import annotations

import hashlib
import math
import re
import tempfile
from pathlib import Path

from scripts import propensity as P
from scripts.propensity import (
    DEFAULT_PREFIX,
    _entry_logprob,
    PropensityRecord,
    audit_from_token_ids,
    cache_path,
    cell_hash,
    compare_propensities,
    label_bias_check,
    longest_common_prefix_len,
    propensity_from_token_logprobs,
    rankdata,
    renormalise,
    resolve_repo,
    spearman_rho,
    sum_token_logprobs,
    tokenization_audit,
    variance_decomposition,
    verdicts_for_instrument,
)


# ---------------------------------------------------------------------------
# Stub tokenizer: three deterministic regimes that reproduce the real failure
# modes without downloading anything.
#
#   "clean"  -- whitespace-aligned pieces; the prefix's trailing space is its
#               own token, so verdicts tokenize identically with and without it.
#   "merge"  -- BPE-like: leading whitespace is absorbed into the following
#               piece, so "VERDICT: " + "YTA" re-tokenizes across the boundary.
#   "char"   -- one token per character; makes shared leading tokens and nested
#               token sequences unavoidable.
# ---------------------------------------------------------------------------

class StubTokenizer:
    def __init__(self, mode: str = "clean") -> None:
        if mode not in {"clean", "merge", "char"}:
            raise ValueError(mode)
        self.mode = mode
        self._vocab: dict[str, int] = {}
        self.chat_template = None

    def _pieces(self, text: str) -> list[str]:
        if self.mode == "char":
            return list(text)
        if self.mode == "clean":
            return re.findall(r"\S+|\s+", text)
        return re.findall(r"\s*\S{1,3}|\s+$", text)

    def _id(self, piece: str) -> int:
        if piece not in self._vocab:
            self._vocab[piece] = len(self._vocab) + 1000
        return self._vocab[piece]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._id(p) for p in self._pieces(text)]
        if add_special_tokens:
            ids = [self._id("<bos>")] + ids
        return ids

    def convert_ids_to_tokens(self, ids):
        rev = {v: k for k, v in self._vocab.items()}
        return [rev.get(int(i), "<unk>") for i in ids]


# ---------------------------------------------------------------------------
# Simplex renormalisation
# ---------------------------------------------------------------------------

def test_renormalise_is_a_simplex() -> None:
    probs = renormalise({"YTA": -1.0, "NTA": -2.0, "ESH": -5.5, "UNRESOLVED": -11.0})
    assert abs(sum(probs.values()) - 1.0) < 1e-12, probs
    assert all(0.0 <= p <= 1.0 for p in probs.values()), probs
    assert probs["YTA"] > probs["NTA"] > probs["ESH"] > probs["UNRESOLVED"]
    # Exact two-way check against the closed form.
    two = renormalise({"a": -1.0, "b": -3.0})
    expected_a = 1.0 / (1.0 + math.exp(-2.0))
    assert abs(two["a"] - expected_a) < 1e-12, two


def test_renormalise_is_shift_invariant() -> None:
    """A constant shared by every candidate (the shared prefix) must cancel."""
    base = {"a": -3.0, "b": -4.25, "c": -9.0}
    shifted = {k: v - 137.5 for k, v in base.items()}
    p0, p1 = renormalise(base), renormalise(shifted)
    for k in base:
        assert abs(p0[k] - p1[k]) < 1e-12, (k, p0, p1)


def test_renormalise_handles_impossible_and_degenerate() -> None:
    probs = renormalise({"a": 0.0, "b": float("-inf")})
    assert abs(probs["a"] - 1.0) < 1e-12 and probs["b"] == 0.0, probs
    allinf = renormalise({"a": float("-inf"), "b": float("-inf")})
    assert abs(allinf["a"] - 0.5) < 1e-12, allinf
    assert renormalise({}) == {}


def test_renormalise_survives_huge_magnitudes() -> None:
    """Log-sum-exp stability: naive exp() would underflow to 0/0 here."""
    probs = renormalise({"a": -4000.0, "b": -4002.0})
    assert abs(sum(probs.values()) - 1.0) < 1e-12, probs
    assert abs(probs["a"] - 1.0 / (1.0 + math.exp(-2.0))) < 1e-12, probs


# ---------------------------------------------------------------------------
# The no-length-normalisation property
# ---------------------------------------------------------------------------

def test_no_length_normalisation() -> None:
    """Equal raw logprob SUMS must give equal probability, whatever the length.

    A length-normalised (per-token mean) score would rank the 3-token verdict
    far above the 1-token one here; the raw sum must not.
    """
    verdicts = ["A", "LONGER"]
    cand_ids = {"A": [7, 11], "LONGER": [7, 21, 22, 23]}
    cand_lps = {"A": [None, -1.2], "LONGER": [None, -0.4, -0.4, -0.4]}
    core = propensity_from_token_logprobs(verdicts, cand_ids, cand_lps)
    assert abs(core["logprob_sums"]["A"] - (-1.2)) < 1e-12
    assert abs(core["logprob_sums"]["LONGER"] - (-1.2)) < 1e-12
    assert abs(core["probs"]["A"] - 0.5) < 1e-12, core["probs"]
    assert abs(core["probs"]["LONGER"] - 0.5) < 1e-12, core["probs"]
    # Token counts differ -- so a per-token mean would NOT have tied.
    assert core["n_scored_tokens"] == {"A": 1, "LONGER": 3}, core["n_scored_tokens"]
    mean_a = core["logprob_sums"]["A"] / core["n_scored_tokens"]["A"]
    mean_b = core["logprob_sums"]["LONGER"] / core["n_scored_tokens"]["LONGER"]
    assert mean_a < mean_b, "test is vacuous unless the per-token means differ"


def test_multi_token_verdict_beats_top1_reading() -> None:
    """A verdict can win on the full string while losing at token 0.

    "UNRESOLVED" starts with a low-probability first token but is nearly
    deterministic afterwards; "NTA" starts strong and then decays.  A top-k
    next-token read would call this for NTA; exact scoring calls it for
    UNRESOLVED.  This is the whole reason the instrument exists.
    """
    verdicts = ["NTA", "UNRESOLVED"]
    cand_ids = {"NTA": [1, 2, 30, 31], "UNRESOLVED": [1, 2, 40, 41, 42]}
    cand_lps = {
        "NTA": [None, None, -0.20, -3.00],   # first token strong, then decays
        "UNRESOLVED": [None, None, -1.00, -0.05, -0.05],
    }
    # position 1 is inside the shared prefix, so its None is never summed
    cand_lps["NTA"][1] = -0.5
    cand_lps["UNRESOLVED"][1] = -0.5
    core = propensity_from_token_logprobs(verdicts, cand_ids, cand_lps)
    assert core["split"] == 2, core["split"]
    assert core["probs"]["UNRESOLVED"] > core["probs"]["NTA"], core["probs"]
    # ...while the first differing token favours NTA.
    assert cand_lps["NTA"][2] > cand_lps["UNRESOLVED"][2]


# ---------------------------------------------------------------------------
# Shared-prefix split
# ---------------------------------------------------------------------------

def test_longest_common_prefix_len() -> None:
    assert longest_common_prefix_len([[1, 2, 3], [1, 2, 9]]) == 2
    assert longest_common_prefix_len([[1, 2], [1, 2]]) == 2
    assert longest_common_prefix_len([[1], [2]]) == 0
    assert longest_common_prefix_len([[1, 2, 3]]) == 3
    assert longest_common_prefix_len([]) == 0
    assert longest_common_prefix_len([[1, 2, 3], [1, 2], [1, 2, 3]]) == 2


def test_split_makes_shared_context_cancel() -> None:
    """Whatever the shared prefix's logprobs are, the answer must not move."""
    verdicts = ["YTA", "NTA"]
    cand_ids = {"YTA": [5, 6, 7, 100], "NTA": [5, 6, 7, 200]}
    a = {"YTA": [None, -2.0, -3.0, -0.5], "NTA": [None, -2.0, -3.0, -1.5]}
    b = {"YTA": [None, -90.0, -80.0, -0.5], "NTA": [None, -90.0, -80.0, -1.5]}
    pa = propensity_from_token_logprobs(verdicts, cand_ids, a)["probs"]
    pb = propensity_from_token_logprobs(verdicts, cand_ids, b)["probs"]
    for v in verdicts:
        assert abs(pa[v] - pb[v]) < 1e-12, (v, pa, pb)
    assert abs(pa["YTA"] - 1.0 / (1.0 + math.exp(-1.0))) < 1e-12, pa


def test_boundary_merge_still_conditions_identically() -> None:
    """When BPE eats the prefix's last token, the split retreats for everyone."""
    verdicts = ["YTA", "NTA"]
    #                       shared     merged-boundary token differs
    cand_ids = {"YTA": [5, 6, 71, 100], "NTA": [5, 6, 72, 200]}
    lps = {"YTA": [None, -2.0, -0.7, -0.5], "NTA": [None, -2.0, -0.9, -0.5]}
    core = propensity_from_token_logprobs(verdicts, cand_ids, lps)
    assert core["split"] == 2, core["split"]
    # Both candidates score their merged token plus their verdict token.
    assert core["n_scored_tokens"] == {"YTA": 2, "NTA": 2}
    assert abs(core["logprob_sums"]["YTA"] - (-1.2)) < 1e-12
    assert core["probs"]["YTA"] > core["probs"]["NTA"]


def test_sum_token_logprobs_skips_position_zero() -> None:
    assert abs(sum_token_logprobs([None, -1.0, -2.0], 0) - (-3.0)) < 1e-12
    assert abs(sum_token_logprobs([None, -1.0, -2.0], 1) - (-3.0)) < 1e-12
    assert abs(sum_token_logprobs([None, -1.0, -2.0], 2) - (-2.0)) < 1e-12
    assert sum_token_logprobs([None, -1.0], 5) == 0.0


def test_core_rejects_malformed_input() -> None:
    for bad, exc in [
        (lambda: propensity_from_token_logprobs([], {}, {}), ValueError),
        (
            lambda: propensity_from_token_logprobs(
                ["a", "a"], {"a": [1]}, {"a": [None]}
            ),
            ValueError,
        ),
        (
            lambda: propensity_from_token_logprobs(
                ["a", "b"], {"a": [1, 2]}, {"a": [None, -1.0]}
            ),
            KeyError,
        ),
        (
            lambda: propensity_from_token_logprobs(
                ["a"], {"a": [1, 2]}, {"a": [None]}
            ),
            ValueError,
        ),
        (
            lambda: propensity_from_token_logprobs(
                ["a"], {"a": [1, 2]}, {"a": [-0.1, -1.0]}
            ),
            ValueError,
        ),
    ]:
        try:
            bad()
        except exc:
            continue
        raise AssertionError(f"expected {exc.__name__}")


def test_core_allows_none_before_split_but_not_after() -> None:
    """Unscored prefix positions may be None; a hole in the scored range may not."""
    verdicts = ["YTA", "NTA"]
    cand_ids = {"YTA": [5, 6, 7, 100], "NTA": [5, 6, 7, 200]}
    # Backend reported nothing for the shared prefix -- fine, it cancels anyway.
    sparse = {"YTA": [None, None, None, -0.5], "NTA": [None, None, None, -1.5]}
    core = propensity_from_token_logprobs(verdicts, cand_ids, sparse)
    assert core["split"] == 3
    assert abs(core["probs"]["YTA"] - 1.0 / (1.0 + math.exp(-1.0))) < 1e-12, core["probs"]

    holed = {"YTA": [None, None, None, None], "NTA": [None, None, None, -1.5]}
    try:
        propensity_from_token_logprobs(verdicts, cand_ids, holed)
    except ValueError as e:
        assert "missing logprobs" in str(e), e
    else:
        raise AssertionError("expected ValueError for a hole in the scored range")


def test_core_flags_nested_and_identical_tokenizations() -> None:
    nested = propensity_from_token_logprobs(
        ["A", "AB"],
        {"A": [9, 100], "AB": [9, 100, 101]},
        {"A": [None, -0.3], "AB": [None, -0.3, -0.8]},
    )
    assert "nested_tokenization" in nested["flags"], nested["flags"]
    assert "empty_scored_suffix" in nested["flags"], nested["flags"]

    identical = propensity_from_token_logprobs(
        ["YTA", "yta"],
        {"YTA": [9, 100], "yta": [9, 100]},
        {"YTA": [None, -0.3], "yta": [None, -0.3]},
    )
    assert "identical_tokenization" in identical["flags"], identical["flags"]

    clean = propensity_from_token_logprobs(
        ["YTA", "NTA"],
        {"YTA": [9, 100], "NTA": [9, 200]},
        {"YTA": [None, -0.3], "NTA": [None, -0.8]},
    )
    assert clean["flags"] == [], clean["flags"]


# ---------------------------------------------------------------------------
# Tokenization audit
# ---------------------------------------------------------------------------

def test_audit_clean_tokenizer_passes() -> None:
    tok = StubTokenizer("clean")
    rep = tokenization_audit("stub", ["YTA", "NTA", "ESH"], tokenizer=tok)
    assert rep["passed"], rep
    assert rep["inconsistent_with_space"] == [], rep["inconsistent_with_space"]
    assert rep["shared_leading_token"] == [], rep["shared_leading_token"]
    assert rep["nested_prefix"] == [] and rep["identical_tokenization"] == []
    for v in ("YTA", "NTA", "ESH"):
        assert rep["per_verdict"][v]["n_tokens_in_context"] == 1
        assert rep["per_verdict"][v]["boundary_merge"] is False
    assert rep["prefix"] == DEFAULT_PREFIX


def test_audit_flags_space_absorption() -> None:
    """BPE-like merge: the prefix's trailing space is eaten by the verdict."""
    tok = StubTokenizer("merge")
    rep = tokenization_audit("stub", ["YTA", "NTA"], tokenizer=tok)
    assert set(rep["inconsistent_with_space"]) == {"YTA", "NTA"}, rep
    for v in ("YTA", "NTA"):
        assert rep["per_verdict"][v]["boundary_merge"] is True, rep["per_verdict"][v]
    # Boundary merging is a warning, not a hard failure: the shared-prefix split
    # keeps the measurement exact.
    assert rep["passed"], rep["hard_failures"]
    assert any("trailing space" in w for w in rep["warnings"]), rep["warnings"]


def test_audit_flags_shared_leading_token_and_multi_token() -> None:
    tok = StubTokenizer("char")
    rep = tokenization_audit("stub", ["YTA", "YTB"], tokenizer=tok)
    assert rep["shared_leading_token"] == [["YTA", "YTB"]], rep["shared_leading_token"]
    assert set(rep["multi_token"]) == {"YTA", "YTB"}, rep["multi_token"]
    assert rep["passed"], rep["hard_failures"]
    assert any("top-k" in w for w in rep["warnings"]), rep["warnings"]


def test_audit_hard_fails_on_nested_tokenization() -> None:
    tok = StubTokenizer("char")
    rep = tokenization_audit("stub", ["A", "AB"], tokenizer=tok)
    assert rep["nested_prefix"] == [["A", "AB"]], rep["nested_prefix"]
    assert not rep["passed"], rep
    assert any("ill-posed" in h for h in rep["hard_failures"]), rep["hard_failures"]


def test_audit_hard_fails_on_identical_tokenization() -> None:
    rep = audit_from_token_ids(
        ["OPTION_A", "OPTION_A "],
        {"OPTION_A": [1, 2], "OPTION_A ": [1, 2]},
        [9],
        {"OPTION_A": [9, 1, 2], "OPTION_A ": [9, 1, 2]},
    )
    assert rep["identical_tokenization"] == [["OPTION_A", "OPTION_A "]], rep
    assert not rep["passed"], rep


def test_audit_pure_logic_matches_hand_worked_example() -> None:
    """prefix=[10,11]; "X" merges the boundary, "Y" does not."""
    rep = audit_from_token_ids(
        ["X", "Y"],
        {"X": [50], "Y": [60]},
        [10, 11],
        {"X": [10, 77, 50], "Y": [10, 11, 60]},
    )
    assert rep["per_verdict"]["X"]["split"] == 1
    assert rep["per_verdict"]["X"]["boundary_merge"] is True
    assert rep["per_verdict"]["X"]["suffix_ids"] == [77, 50]
    assert rep["per_verdict"]["Y"]["split"] == 2
    assert rep["per_verdict"]["Y"]["boundary_merge"] is False
    assert rep["per_verdict"]["Y"]["space_consistent"] is True
    assert rep["inconsistent_with_space"] == ["X"]
    assert rep["passed"]


# ---------------------------------------------------------------------------
# Variance decomposition (the T0-B pilot statistic)
# ---------------------------------------------------------------------------

def test_variance_decomposition_hand_computed() -> None:
    """Hand-worked: q = [0.2, 0.8] over two traces.

      within  = mean(0.2*0.8, 0.8*0.2) = 0.16
      between = ((0.2-0.5)^2 + (0.8-0.5)^2) / (2-1) = 0.18
      gain    = 1 + 0.16/0.18 = 1.888888...
    """
    vd = variance_decomposition({"i1": [0.2, 0.8]})
    item = vd["items"]["i1"]
    assert abs(item["mean_q"] - 0.5) < 1e-12
    assert abs(item["within_trace"] - 0.16) < 1e-12, item
    assert abs(item["between_trace"] - 0.18) < 1e-12, item
    assert abs(item["total"] - 0.34) < 1e-12, item
    assert abs(item["efficiency_gain"] - (1.0 + 0.16 / 0.18)) < 1e-12, item
    assert abs(vd["efficiency_gain"] - (1.0 + 0.16 / 0.18)) < 1e-12, vd


def test_variance_decomposition_determinative_traces_give_gain_one() -> None:
    """The documented degenerate case: traces pin q to 0/1, so gain -> 1."""
    vd = variance_decomposition({"i1": [0.0, 1.0, 0.0, 1.0]})
    item = vd["items"]["i1"]
    assert item["within_trace"] == 0.0, item
    assert item["between_trace"] > 0.0, item
    assert abs(item["efficiency_gain"] - 1.0) < 1e-12, item
    assert item["determinative"] is True
    assert "determinative" in vd["interpretation"], vd["interpretation"]


def test_variance_decomposition_zero_between_trace() -> None:
    """No trace-to-trace variation at all: sampling noise is all there is."""
    vd = variance_decomposition({"i1": [0.5, 0.5, 0.5]})
    assert vd["items"]["i1"]["between_trace"] == 0.0
    assert math.isinf(vd["items"]["i1"]["efficiency_gain"])
    assert math.isinf(vd["efficiency_gain"])
    # Fully degenerate (no variance anywhere) is reported as gain 1, not inf.
    flat = variance_decomposition({"i1": [1.0, 1.0]})
    assert flat["efficiency_gain"] == 1.0, flat


def test_variance_decomposition_pooling_and_k() -> None:
    """Pooled over two items, hand-computed; and k scales the within term."""
    vd = variance_decomposition({"det": [0.0, 1.0], "mid": [0.2, 0.8]})
    assert abs(vd["pooled_within_trace"] - (0.0 + 0.16) / 2) < 1e-12, vd
    assert abs(vd["pooled_between_trace"] - (0.5 + 0.18) / 2) < 1e-12, vd
    assert abs(vd["efficiency_gain"] - (1.0 + 0.08 / 0.34)) < 1e-12, vd
    vd4 = variance_decomposition({"det": [0.0, 1.0], "mid": [0.2, 0.8]}, k=4)
    assert abs(vd4["efficiency_gain"] - (1.0 + 0.08 / (4 * 0.34))) < 1e-12, vd4
    assert vd4["efficiency_gain"] < vd["efficiency_gain"]


def test_variance_decomposition_accepts_sequences_and_skips_singletons() -> None:
    vd = variance_decomposition([[0.2, 0.8], [0.5]])
    assert vd["n_items"] == 1, vd
    assert vd["skipped_items"] == ["item0001"], vd
    assert abs(vd["items"]["item0000"]["within_trace"] - 0.16) < 1e-12


def test_variance_decomposition_rejects_bad_input() -> None:
    for bad in (
        lambda: variance_decomposition({}),
        lambda: variance_decomposition({"i": [0.5, 1.5]}),
        lambda: variance_decomposition({"i": [0.5, 0.5]}, k=0),
        lambda: variance_decomposition({"i": [0.5]}),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# Rank statistics + label-bias gate
# ---------------------------------------------------------------------------

def test_rankdata_and_spearman() -> None:
    assert rankdata([10.0, 20.0, 30.0]) == [1.0, 2.0, 3.0]
    assert rankdata([10.0, 10.0, 30.0]) == [1.5, 1.5, 3.0]
    assert abs(spearman_rho([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) - 1.0) < 1e-12
    assert abs(spearman_rho([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) + 1.0) < 1e-12
    # Monotone but non-linear: Spearman 1, Pearson would not be.
    assert abs(spearman_rho([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) - 1.0) < 1e-12
    assert math.isnan(spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def _unit_hash(key: str) -> float:
    """Deterministic pseudo-random value in [0, 1) -- NOT hash(), which is salted."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) % 1_000_003) / 1_000_003.0


def _fake_propensity_factory(label_shift: float):
    """Stub propensity_fn blending item difficulty with a label-driven component.

    label_shift = 0  -> the verdict vocabulary is irrelevant; every labeling
                        returns the same per-item propensity (rho = 1).
    label_shift -> 1 -> the label strings, not the scenario, decide the answer,
                        which is exactly what this gate must catch.
    """
    def fn(model, system, user, *, verdicts, trace="", prefix="", item_id="", **kw):
        scenario = _unit_hash(f"scenario|{user}")
        label_driven = _unit_hash(f"label|{verdicts[0]}|{user}")
        p0 = (1.0 - label_shift) * scenario + label_shift * label_driven
        p0 = min(max(p0, 0.01), 0.99)
        rest = (1.0 - p0) / max(len(verdicts) - 1, 1)
        out = {verdicts[0]: p0}
        for v in verdicts[1:]:
            out[v] = rest
        return out
    return fn


def test_label_bias_check_passes_when_vocabulary_is_irrelevant() -> None:
    items = [{"id": f"i{i}", "system": "", "user": f"scenario {i}"} for i in range(30)]
    rep = label_bias_check(
        "stub",
        items,
        [("YTA", "NTA"), ("A", "B"), ("1", "2")],
        propensity_fn=_fake_propensity_factory(0.0),
    )
    assert rep["n_classes"] == 2 and rep["n_items"] == 30
    assert len(rep["pairs"]) == 3, rep["pairs"]
    assert abs(rep["min_rho"] - 1.0) < 1e-12, rep
    assert all(p["max_abs_diff"] < 1e-12 for p in rep["pairs"]), rep["pairs"]
    assert rep["passed"], rep
    assert rep["per_item"] == [], "per-item rho is degenerate with 2 classes"


def test_label_bias_check_fails_when_labels_drive_the_answer() -> None:
    items = [{"id": f"i{i}", "system": "", "user": f"scenario {i}"} for i in range(30)]
    rep = label_bias_check(
        "stub",
        items,
        [("YTA", "NTA"), ("A", "B"), ("1", "2")],
        propensity_fn=_fake_propensity_factory(1.0),
    )
    assert rep["gate"] == 0.8
    assert rep["min_rho"] < 0.8, rep["pairs"]
    if rep["passed"]:
        raise AssertionError(f"label bias should have tripped the gate: {rep['pairs']}")


def test_label_bias_check_reports_per_item_rho_for_three_classes() -> None:
    items = [{"id": f"i{i}", "system": "", "user": "x" * (i + 1)} for i in range(5)]
    rep = label_bias_check(
        "stub",
        items,
        [("TRUE", "FALSE", "UNRESOLVED"), ("A", "B", "C")],
        propensity_fn=_fake_propensity_factory(0.0),
    )
    assert rep["n_classes"] == 3
    assert len(rep["per_item"]) == 5, rep["per_item"]
    assert all(abs(row["rho_mean"] - 1.0) < 1e-12 for row in rep["per_item"]), rep["per_item"]


def test_label_bias_check_validates_labelings() -> None:
    items = [{"id": "i0", "system": "", "user": "u"}]
    for bad in (
        lambda: label_bias_check("stub", items, [("A", "B")]),
        lambda: label_bias_check("stub", items, [("A", "B"), ("A", "B", "C")]),
        lambda: label_bias_check("stub", items, [("A", "A"), ("B", "C")]),
        lambda: label_bias_check("stub", items, [("A",), ("B",)]),
        lambda: label_bias_check("stub", [], [("A", "B"), ("C", "D")]),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_label_bias_check_accepts_tuples_and_objects() -> None:
    class Item:
        id = "obj0"
        system = ""
        user = "hello"
        trace = ""

    rep = label_bias_check(
        "stub",
        [("", "abc"), Item()],
        [("A", "B"), ("1", "2")],
        propensity_fn=_fake_propensity_factory(0.0),
    )
    assert rep["n_items"] == 2, rep


# ---------------------------------------------------------------------------
# vLLM response parsing (offline: the HTTP layer is stubbed)
# ---------------------------------------------------------------------------

def test_vllm_entry_logprob_variants() -> None:
    # Keyed by token id as a JSON string, the shape vLLM actually returns.
    entry = {"100": {"logprob": -0.25, "rank": 3, "decoded_token": "YTA"}}
    assert abs(_entry_logprob(entry, 100) - (-0.25)) < 1e-12
    # Integer keys (some clients deserialise them that way).
    assert abs(_entry_logprob({100: {"logprob": -0.5}}, 100) - (-0.5)) < 1e-12
    # Unknown id, single entry -> unambiguous.
    assert abs(_entry_logprob({"7": {"logprob": -1.5}}, None) - (-1.5)) < 1e-12
    # Unknown id, several entries -> fall back to the rank-1 token.
    multi = {"7": {"logprob": -1.5, "rank": 2}, "9": {"logprob": -0.1, "rank": 1}}
    assert abs(_entry_logprob(multi, None) - (-0.1)) < 1e-12
    # Genuinely ambiguous -> None, which the core then rejects loudly.
    assert _entry_logprob({"7": {"logprob": -1.5}, "9": {"logprob": -0.1}}, None) is None
    assert _entry_logprob({}, 3) is None
    assert _entry_logprob(None, 3) is None


def test_vllm_token_logprobs_parses_prompt_logprobs() -> None:
    ids_by_text = {"ctx YTA": [5, 6, 7, 100], "ctx NTA": [5, 6, 7, 200]}
    lp_by_text = {"ctx YTA": [-2.0, -0.5, -0.25], "ctx NTA": [-2.0, -0.5, -1.25]}

    def fake_post(path, payload, timeout=600):
        text = payload["prompt"]
        if path == "/tokenize":
            return {"tokens": ids_by_text[text], "count": len(ids_by_text[text])}
        ids, lps = ids_by_text[text], lp_by_text[text]
        entries = [None]
        for tid, lp in zip(ids[1:], lps):
            entries.append({str(tid): {"logprob": lp, "rank": 1}})
        return {"choices": [{"prompt_logprobs": entries}]}

    original = P._vllm_post
    P._vllm_post = fake_post
    try:
        enc, lps = P.vllm_token_logprobs("stub", ["ctx YTA", "ctx NTA"])
    finally:
        P._vllm_post = original

    assert enc == [[5, 6, 7, 100], [5, 6, 7, 200]]
    assert lps[0][0] is None and lps[1][0] is None
    assert lps[0] == [None, -2.0, -0.5, -0.25], lps[0]
    core = propensity_from_token_logprobs(
        ["YTA", "NTA"],
        {"YTA": enc[0], "NTA": enc[1]},
        {"YTA": lps[0], "NTA": lps[1]},
    )
    assert core["split"] == 3, core["split"]
    assert abs(core["probs"]["YTA"] - 1.0 / (1.0 + math.exp(-1.0))) < 1e-12, core["probs"]


def test_vllm_raises_on_token_stream_mismatch() -> None:
    """A BOS added on one endpoint but not the other must not be papered over."""
    def fake_post(path, payload, timeout=600):
        if path == "/tokenize":
            return {"tokens": [5, 6, 7]}
        return {"choices": [{"prompt_logprobs": [None, {"6": {"logprob": -1.0}}]}]}

    original = P._vllm_post
    P._vllm_post = fake_post
    try:
        P.vllm_token_logprobs("stub", ["ctx"])
    except RuntimeError as e:
        assert "token stream mismatch" in str(e), e
    else:
        raise AssertionError("expected RuntimeError")
    finally:
        P._vllm_post = original


def test_vllm_payload_matches_tokenize_special_tokens() -> None:
    """add_special_tokens must be sent to both endpoints identically."""
    seen: list[tuple] = []

    def fake_post(path, payload, timeout=600):
        seen.append((path, payload.get("add_special_tokens")))
        if path == "/tokenize":
            return {"tokens": [5, 6]}
        return {"choices": [{"prompt_logprobs": [None, {"6": {"logprob": -1.0}}]}]}

    original = P._vllm_post
    P._vllm_post = fake_post
    try:
        P.vllm_token_logprobs("stub", ["ctx"], add_special_tokens=False)
    finally:
        P._vllm_post = original
    assert [p for _, p in seen] == [False, False], seen
    assert [path for path, _ in seen] == ["/tokenize", "/v1/completions"], seen


def test_dtype_kwarg_for_transformers_version() -> None:
    assert P._dtype_kwarg_for_version("4.56.0") == "dtype"
    assert P._dtype_kwarg_for_version("4.57.1") == "dtype"
    assert P._dtype_kwarg_for_version("5.0.0") == "dtype"
    assert P._dtype_kwarg_for_version("4.55.2") == "torch_dtype"
    assert P._dtype_kwarg_for_version("4.44.0") == "torch_dtype"
    # Unparseable version -> the conservative choice that works on old builds.
    assert P._dtype_kwarg_for_version("") == "torch_dtype"
    assert P._dtype_kwarg_for_version("dev") == "torch_dtype"


def test_vllm_raises_when_prompt_logprobs_missing() -> None:
    def fake_post(path, payload, timeout=600):
        if path == "/tokenize":
            return {"tokens": [1, 2, 3]}
        return {"choices": [{}]}

    original = P._vllm_post
    P._vllm_post = fake_post
    try:
        P.vllm_token_logprobs("stub", ["anything"])
    except RuntimeError as e:
        assert "prompt_logprobs" in str(e), e
    else:
        raise AssertionError("expected RuntimeError")
    finally:
        P._vllm_post = original


# ---------------------------------------------------------------------------
# End-to-end record path + per-cell caching (backend stubbed)
# ---------------------------------------------------------------------------

def _stub_backend(calls: list):
    """Stand in for score_sequences: NTA beats YTA by exactly 1 nat."""
    def fn(model, texts, *, backend, add_special_tokens=True):
        calls.append(list(texts))
        enc, lps = [], []
        for i, _ in enumerate(texts):
            enc.append([11, 12, 13, 900 + i])
            lps.append([None, -4.0, -3.0, -1.0 - i])
        return enc, lps
    return fn


def test_record_end_to_end_and_cache_roundtrip() -> None:
    calls: list = []
    original = P.score_sequences
    P.score_sequences = _stub_backend(calls)
    try:
        with tempfile.TemporaryDirectory() as td:
            kw = dict(
                verdicts=("YTA", "NTA"),
                trace="I weighed both sides. ",
                backend="hf",
                item_id="aita_0007",
                cache_dir=Path(td),
            )
            rec = P.verdict_propensity_record("stub-model", "sys", "user text", **kw)
            assert len(calls) == 1, calls
            # Candidates are context + verdict, and the context is shared.
            assert calls[0][0].endswith("YTA") and calls[0][1].endswith("NTA")
            assert calls[0][0][: -len("YTA")] == calls[0][1][: -len("NTA")]
            assert "I weighed both sides. " in calls[0][0]
            assert calls[0][0].endswith(DEFAULT_PREFIX + "YTA")

            assert abs(sum(rec.probs.values()) - 1.0) < 1e-12, rec.probs
            assert rec.argmax == "YTA", rec.probs
            assert abs(rec.probs["YTA"] - 1.0 / (1.0 + math.exp(-1.0))) < 1e-12
            assert rec.split == 3 and rec.n_scored_tokens == {"YTA": 1, "NTA": 1}
            assert rec.backend == "hf"

            written = list(Path(td).glob("prop_*.json"))
            assert len(written) == 1, written
            assert "aita_0007" in written[0].name, written[0].name

            # Second call must be served from cache, not the backend.
            again = P.verdict_propensity_record("stub-model", "sys", "user text", **kw)
            assert len(calls) == 1, "cache miss: backend was called twice"
            assert again.probs == rec.probs
            assert again.split == rec.split

            # A different trace is a different cell.
            kw2 = dict(kw)
            kw2["trace"] = "Actually, on reflection. "
            P.verdict_propensity_record("stub-model", "sys", "user text", **kw2)
            assert len(calls) == 2, calls
            assert len(list(Path(td).glob("prop_*.json"))) == 2
    finally:
        P.score_sequences = original


def test_verdict_propensity_contract_wrapper() -> None:
    calls: list = []
    original = P.score_sequences
    P.score_sequences = _stub_backend(calls)
    try:
        with tempfile.TemporaryDirectory() as td:
            probs = P.verdict_propensity(
                "stub-model",
                "sys",
                "user text",
                verdicts=("YTA", "NTA", "ESH"),
                backend="hf",
                cache_dir=Path(td),
            )
        assert isinstance(probs, dict)
        assert set(probs) == {"YTA", "NTA", "ESH"}
        assert abs(sum(probs.values()) - 1.0) < 1e-12, probs
        assert all(isinstance(v, float) for v in probs.values())
    finally:
        P.score_sequences = original


def test_singleton_verdict_set_is_degenerate_not_an_error() -> None:
    probs = P.verdict_propensity(
        "stub-model", "", "u", verdicts=("ONLY",), backend="hf", use_cache=False
    )
    assert probs == {"ONLY": 1.0}, probs


def test_render_context_falls_back_to_plain_offline() -> None:
    ctx, templated = P.render_context(
        "stub-model", "SYS", "USER", trace="TRACE ", prefix=DEFAULT_PREFIX
    )
    assert templated is False
    assert ctx.startswith("SYS")
    assert "USER" in ctx
    assert ctx.endswith("TRACE " + DEFAULT_PREFIX), ctx
    # Explicit chat template with no tokenizer available must fail loudly.
    try:
        P.render_context("stub-model", "", "u", chat_template="chat")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for chat_template='chat'")
    try:
        P.render_context("stub-model", "", "u", chat_template="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a bad chat_template")


def test_render_context_uses_chat_template_when_present() -> None:
    class TemplatedTokenizer(StubTokenizer):
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):
            body = "".join(f"<{m['role']}>{m['content']}" for m in msgs)
            return body + ("<assistant>" if add_generation_prompt else "")

    tok = TemplatedTokenizer("clean")
    tok.chat_template = "stub"  # StubTokenizer.__init__ clears it by default
    ctx, templated = P.render_context(
        "stub-model", "SYS", "USER", trace="T ", prefix="VERDICT: ", tokenizer=tok
    )
    assert templated is True
    assert ctx == "<system>SYS<user>USER<assistant>T VERDICT: ", ctx


def test_resolve_backend() -> None:
    assert P.resolve_backend("hf") == "hf"
    assert P.resolve_backend("vllm") == "vllm"
    try:
        P.resolve_backend("tensorrt")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown backend")


# ---------------------------------------------------------------------------
# Cross-backend comparison helper
# ---------------------------------------------------------------------------

def test_compare_propensities() -> None:
    a = {"YTA": 0.7, "NTA": 0.3}
    same = compare_propensities(a, {"YTA": 0.7005, "NTA": 0.2995})
    assert same["agree"] and same["argmax_agree"], same
    assert abs(same["total_variation"] - 0.0005) < 1e-9, same
    far = compare_propensities(a, {"YTA": 0.3, "NTA": 0.7})
    assert not far["agree"] and not far["argmax_agree"], far
    assert abs(far["max_abs_diff"] - 0.4) < 1e-12, far


# ---------------------------------------------------------------------------
# Record, cache naming, instrument plumbing
# ---------------------------------------------------------------------------

def test_record_roundtrip_and_derived_fields() -> None:
    rec = PropensityRecord(
        model="Qwen3-8B",
        backend="hf",
        verdicts=("YTA", "NTA"),
        logprob_sums={"YTA": -0.5, "NTA": -1.5},
        probs=renormalise({"YTA": -0.5, "NTA": -1.5}),
        n_scored_tokens={"YTA": 2, "NTA": 2},
        split=41,
    )
    assert rec.argmax == "YTA"
    assert 0.0 < rec.entropy < math.log(2) + 1e-12
    back = PropensityRecord.from_dict(rec.to_dict())
    assert back.verdicts == rec.verdicts
    assert back.probs == rec.probs
    assert back.split == 41 and back.backend == "hf"


def test_cache_path_follows_repo_convention() -> None:
    h = cell_hash(
        system="s", user="u", trace="", prefix=DEFAULT_PREFIX,
        suffix="", verdicts=("YTA", "NTA"), chat_template="auto",
    )
    assert len(h) == 16 and h.isalnum()
    with tempfile.TemporaryDirectory() as td:
        p = cache_path("Qwen/Qwen3-8B", "hf", "aita_0001", h, cache_dir=Path(td))
        assert p.name.startswith("prop_")
        assert p.suffix == ".json"
        assert "Qwen_Qwen3-8B" in p.name, p.name
        assert h in p.name
        assert not re.search(r"[^A-Za-z0-9._\-/]", p.name), p.name
        # Model aliases must collapse onto one cache file, not three.
        for alias in ("qwen3-8b", "Qwen3-8B", "local/qwen3-8b"):
            assert cache_path(alias, "hf", "aita_0001", h, cache_dir=Path(td)) == p, alias
        # Backends keep separate files so a cross-backend check compares two
        # real measurements rather than one cached one.
        assert cache_path("Qwen/Qwen3-8B", "vllm", "aita_0001", h, cache_dir=Path(td)) != p


def test_cell_hash_is_sensitive_to_every_field() -> None:
    base = dict(
        system="s", user="u", trace="t", prefix=DEFAULT_PREFIX,
        suffix="", verdicts=("YTA", "NTA"), chat_template="auto",
    )
    h0 = cell_hash(**base)
    for field_name, new in [
        ("system", "s2"), ("user", "u2"), ("trace", "t2"),
        ("prefix", "ANSWER: "), ("suffix", "\n"),
        ("verdicts", ("NTA", "YTA")), ("chat_template", "plain"),
    ]:
        alt = dict(base)
        alt[field_name] = new
        assert cell_hash(**alt) != h0, field_name


def test_verdicts_for_instrument() -> None:
    aita = verdicts_for_instrument("aita")
    assert "YTA" in aita and "NTA" in aita
    assert "NOVERDICT" not in aita
    no_unres = verdicts_for_instrument("aita", allow_unresolved=False)
    assert "UNRESOLVED" not in no_unres
    assert len(no_unres) < len(aita)


def test_resolve_repo() -> None:
    assert resolve_repo("Qwen3-8B").endswith("Qwen3-8B")
    assert "/" in resolve_repo("Llama-3.1-8B-Instruct")
    assert resolve_repo("some/unknown-repo") == "some/unknown-repo"


def test_heavy_deps_are_only_imported_lazily() -> None:
    """torch/transformers/vllm must never appear at module scope.

    Checked by parsing the source rather than inspecting sys.modules, so the
    test still means something on a GPU node where torch is already loaded.
    """
    import ast

    src = Path(__file__).with_name("propensity.py").read_text()
    tree = ast.parse(src)
    heavy = {"torch", "transformers", "vllm", "datasets", "accelerate"}
    offenders = []
    for node in tree.body:  # module scope only -- nested imports are the point
        if isinstance(node, ast.Import):
            offenders += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            offenders.append(node.module.split(".")[0])
        elif isinstance(node, ast.Try):
            for sub in node.body:
                if isinstance(sub, ast.Import):
                    offenders += [a.name.split(".")[0] for a in sub.names]
                elif isinstance(sub, ast.ImportFrom) and sub.module:
                    offenders.append(sub.module.split(".")[0])
    bad = heavy & set(offenders)
    assert not bad, f"heavy dependencies imported at module scope: {sorted(bad)}"


def test_requests_is_only_imported_lazily() -> None:
    """The vLLM backend must not drag requests in at import time either."""
    import ast

    src = Path(__file__).with_name("propensity.py").read_text()
    tree = ast.parse(src)
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
    assert "requests" not in top, "requests imported at module scope"


# ---------------------------------------------------------------------------
# Standalone runner (mirrors scripts/test_phase14.py)
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    print(f"propensity offline tests ({len(tests)}):")
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  {name} OK")
        except Exception as e:  # noqa: BLE001 - reporting harness
            failed += 1
            print(f"  {name} FAILED: {type(e).__name__}: {e}")
    if failed:
        print(f"{failed} test(s) failed.")
        return 1
    print("All passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
