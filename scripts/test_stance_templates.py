"""
scripts/test_stance_templates.py -- Offline unit tests for the manipulation
template library (scripts/stance_templates.py) and the nuisance reference
distribution (scripts/nuisance.py).

No network, no API keys, no GPU, no data downloads: everything under test is
pure string manipulation and arithmetic.

Usage:
  python -m scripts.test_stance_templates
  python -m pytest scripts/test_stance_templates.py -q
"""
from __future__ import annotations

from scripts.nuisance import (
    DEFAULT_PROBE_TEXTS,
    NUISANCE_PERTURBATIONS,
    NUISANCE_SPECS,
    applicable_specs,
    apply_nuisance,
    apply_nuisance_by_name,
    content_delta,
    content_words,
    nuisance_floor,
    nuisance_percentile,
    nuisance_specs,
    report_effect,
    summarize_nuisance,
    verify_content_preservation,
    verify_registry,
)
from scripts.stance_templates import (
    AUDIENCE_ARM_IDS,
    AUDIENCE_ARMS,
    AUDIENCE_LENGTH_MATCHED_ARMS,
    AUDIENCE_SCRUTINY_2X2,
    AUDIENCE_TOKEN_TOLERANCE,
    CHANNELS,
    DOSE_DIMENSION,
    DOSES,
    NULL_DIMENSION,
    PARAPHRASE_DIMENSIONS,
    PARAPHRASE_TOKEN_TOLERANCE,
    READ_CLAUSES,
    REGISTERS,
    TOKEN_TOLERANCE,
    assert_token_matched,
    audience_arm_supports_will_read,
    audience_cells,
    audience_scrutiny_cell,
    audience_sentence,
    cell_paraphrases,
    evidence_terms,
    held_out_split,
    instrument_for_register,
    paraphrase_family,
    stance_sentence,
    stance_terms,
    token_count,
    verify_all,
    verify_will_read_orthogonality,
)

_STANCE_PAIRS = (
    ("preference", "endorse"),
    ("preference", "doubt"),
    ("testimony", "endorse"),
    ("testimony", "doubt"),
    ("neutral_filler", "none"),
)


# ---------------------------------------------------------------------------
# Token matching
# ---------------------------------------------------------------------------

def test_token_matching_within_register_dose_block() -> None:
    """Every (register, dose) block is token-matched within tolerance."""
    for register in REGISTERS:
        for dose in DOSES:
            block = [
                text
                for channel, direction in _STANCE_PAIRS
                for text in cell_paraphrases(
                    channel=channel, direction=direction, register=register, dose=dose
                )
            ]
            assert len(block) == 10, (register, dose, len(block))
            rep = assert_token_matched(
                block, tolerance=TOKEN_TOLERANCE, label=f"{register}/dose{dose}"
            )
            assert rep["spread"] <= 2 * TOKEN_TOLERANCE, rep
    print("  token_matching_within_register_dose_block OK")


def test_every_cell_has_at_least_two_paraphrases() -> None:
    for register in REGISTERS:
        for dose in DOSES:
            for channel, direction in _STANCE_PAIRS:
                cell = cell_paraphrases(
                    channel=channel, direction=direction, register=register, dose=dose
                )
                assert len(cell) >= 2, (channel, direction, register, dose)
                assert len(set(cell)) == len(cell), cell
    print("  every_cell_has_at_least_two_paraphrases OK")


def test_assert_token_matched_catches_a_mismatch() -> None:
    """The helper must actually fail on an unmatched arm set."""
    bad = ["one two three four five", "one"]
    try:
        assert_token_matched(bad, tolerance=1, label="deliberately-bad")
    except AssertionError:
        pass
    else:  # pragma: no cover - guards a silent-pass regression
        raise AssertionError("assert_token_matched accepted an unmatched arm set")
    assert_token_matched(["one two three", "four five six"], tolerance=1, label="ok")
    print("  assert_token_matched_catches_a_mismatch OK")


def test_audience_arms_token_matched() -> None:
    for register in REGISTERS:
        arms = [AUDIENCE_ARMS[a][register] for a in AUDIENCE_LENGTH_MATCHED_ARMS]
        assert_token_matched(
            arms, tolerance=AUDIENCE_TOKEN_TOLERANCE, label=f"audience/{register}"
        )
    print("  audience_arms_token_matched OK")


def test_paraphrase_dimensions_token_matched() -> None:
    """Dimension realisations, NULL family included, share one length band."""
    for register in REGISTERS:
        for channel in ("preference", "testimony"):
            for direction in ("endorse", "doubt"):
                fam = paraphrase_family(
                    channel=channel, direction=direction, register=register
                )
                texts = [
                    str(a["text"]) for a in fam if a["dimension"] != DOSE_DIMENSION
                ]
                assert_token_matched(
                    texts,
                    tolerance=PARAPHRASE_TOKEN_TOLERANCE,
                    label=f"{register}/{channel}/{direction}",
                )
    print("  paraphrase_dimensions_token_matched OK")


def test_stance_sentence_validates_and_wraps() -> None:
    s0 = stance_sentence(channel="preference", direction="endorse", dose=1, register="aita")
    s1 = stance_sentence(
        channel="preference", direction="endorse", dose=1, register="aita", paraphrase_idx=1
    )
    assert s0 != s1
    # paraphrase_idx wraps modulo the cell size
    assert stance_sentence(
        channel="preference", direction="endorse", dose=1, register="aita", paraphrase_idx=2
    ) == s0
    for kwargs in (
        {"channel": "nope", "direction": "endorse"},
        {"channel": "preference", "direction": "none"},
        {"channel": "neutral_filler", "direction": "endorse"},
        {"channel": "preference", "direction": "endorse", "dose": 9},
        {"channel": "preference", "direction": "endorse", "register": "nope"},
    ):
        try:
            stance_sentence(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"stance_sentence accepted illegal kwargs: {kwargs}")
    print("  stance_sentence_validates_and_wraps OK")


# ---------------------------------------------------------------------------
# Channel separation (the two-channel methodological commitment)
# ---------------------------------------------------------------------------

def test_preference_channel_is_evidence_free() -> None:
    for register in REGISTERS:
        for direction in ("endorse", "doubt"):
            for dose in DOSES:
                for text in cell_paraphrases(
                    channel="preference", direction=direction, register=register, dose=dose
                ):
                    assert not evidence_terms(text), (register, direction, dose, text)
    print("  preference_channel_is_evidence_free OK")


def test_testimony_channel_asserts_evidence() -> None:
    for register in REGISTERS:
        for direction in ("endorse", "doubt"):
            for dose in DOSES:
                for text in cell_paraphrases(
                    channel="testimony", direction=direction, register=register, dose=dose
                ):
                    assert evidence_terms(text), (register, direction, dose, text)
    print("  testimony_channel_asserts_evidence OK")


def test_neutral_filler_is_stance_free() -> None:
    """Filler is the length-matched zero: no stance, no evidence, no audience."""
    for register in REGISTERS:
        for dose in DOSES:
            for text in cell_paraphrases(
                channel="neutral_filler", direction="none", register=register, dose=dose
            ):
                assert not stance_terms(text), (register, dose, text, stance_terms(text))
                assert not evidence_terms(text), (register, dose, text)
    print("  neutral_filler_is_stance_free OK")


# ---------------------------------------------------------------------------
# will_read x view-alignment orthogonality
# ---------------------------------------------------------------------------

def test_will_read_is_orthogonal_to_view_alignment() -> None:
    """All four cells exist and differ only in the read clause / the stem."""
    aligned, opposed = "A5_PANEL_KNOWN_ALIGNED", "A6_PANEL_KNOWN_OPPOSED"
    for register in REGISTERS:
        cells = {
            (arm, flag): audience_sentence(arm, register=register, will_read=flag)
            for arm in (aligned, opposed)
            for flag in (True, False)
        }
        assert len(set(cells.values())) == 4, (register, cells)

        # 1. Within an arm, the ONLY difference is the read clause.
        for arm in (aligned, opposed):
            read, noread = cells[(arm, True)], cells[(arm, False)]
            assert read.endswith(READ_CLAUSES[True]), (register, arm, read)
            assert noread.endswith(READ_CLAUSES[False]), (register, arm, noread)
            assert read[: -len(READ_CLAUSES[True])] == noread[: -len(READ_CLAUSES[False])]
            assert read.replace(READ_CLAUSES[True], READ_CLAUSES[False]) == noread

        # 2. The alignment contrast is the same stem swap under both read levels.
        stems = {
            (arm, flag): cells[(arm, flag)][: -len(READ_CLAUSES[flag]) - 1]
            for arm in (aligned, opposed)
            for flag in (True, False)
        }
        assert stems[(aligned, True)] == stems[(aligned, False)], register
        assert stems[(opposed, True)] == stems[(opposed, False)], register
        assert stems[(aligned, True)] != stems[(opposed, True)], register

        # 3. Orthogonality proper: the read factor is a constant suffix and the
        #    alignment factor a constant stem, so the 2x2 is fully crossed.
        for flag in (True, False):
            assert cells[(aligned, flag)] == f"{stems[(aligned, flag)]} {READ_CLAUSES[flag]}"
            assert cells[(opposed, flag)] == f"{stems[(opposed, flag)]} {READ_CLAUSES[flag]}"

        verify_will_read_orthogonality(register=register)
    print("  will_read_is_orthogonal_to_view_alignment OK")


def test_audience_arm_ids_and_will_read_applicability() -> None:
    for arm in AUDIENCE_ARM_IDS:
        assert arm in AUDIENCE_ARMS, arm
        assert set(AUDIENCE_ARMS[arm]) == set(REGISTERS), arm
    assert AUDIENCE_ARMS["A0_NONE"]["aita"] == ""
    assert "no one else will read your reply" in AUDIENCE_ARMS["A2_PRIVATE"]["aita"]
    for arm in ("A0_NONE", "A1_FILLER", "A2_PRIVATE"):
        assert not audience_arm_supports_will_read(arm), arm
        # will_read is ignored, not silently duplicated, for these arms
        assert audience_sentence(arm, will_read=True) == audience_sentence(
            arm, will_read=False
        )
    for arm in ("A3_NAMED_REVIEWER_UNKNOWN", "A4_PANEL_UNKNOWN", "A7_PUBLIC"):
        assert audience_arm_supports_will_read(arm), arm
        assert audience_sentence(arm, will_read=True) != audience_sentence(
            arm, will_read=False
        )
    cells = audience_cells(register="aita")
    ids = [c["id"] for c in cells]
    assert len(ids) == len(set(ids)) == 3 + 2 * 5, ids
    print("  audience_arm_ids_and_will_read_applicability OK")


def test_audience_scrutiny_2x2_is_factorial_and_length_controlled() -> None:
    """Every cell is two clauses; length is held fixed across the 2x2."""
    for register in REGISTERS:
        texts = {
            cid: AUDIENCE_SCRUTINY_2X2[cid][register]
            for cid in ("NEITHER", "AUDIENCE_ONLY", "SCRUTINY_ONLY", "BOTH")
        }
        assert len(set(texts.values())) == 4, texts
        assert_token_matched(
            list(texts.values()), tolerance=TOKEN_TOLERANCE, label=f"2x2/{register}"
        )
        # BOTH is the audience clause of AUDIENCE_ONLY plus the scrutiny clause
        # of SCRUTINY_ONLY, so the factors are additive by construction.
        aud_clause = texts["AUDIENCE_ONLY"].split(". ")[0]
        scr_clause = texts["SCRUTINY_ONLY"].split(". ")[1]
        assert texts["BOTH"] == f"{aud_clause}. {scr_clause}", texts
        assert audience_scrutiny_cell(
            audience=True, scrutiny=True, register=register
        ) == texts["BOTH"]
    print("  audience_scrutiny_2x2_is_factorial_and_length_controlled OK")


# ---------------------------------------------------------------------------
# Paraphrase dimensions and the signed pseudo-target NULL family
# ---------------------------------------------------------------------------

def test_null_family_carries_no_stance_lexicon() -> None:
    """Signed pseudo-targets must be stance-free, or the floor is not a floor."""
    seen = 0
    for register in REGISTERS:
        for channel, direction in _STANCE_PAIRS:
            nulls = [
                a
                for a in paraphrase_family(
                    channel=channel, direction=direction, register=register
                )
                if a["is_null"]
            ]
            assert len(nulls) == len(PARAPHRASE_DIMENSIONS[NULL_DIMENSION]) >= 3
            for arm in nulls:
                assert arm["dimension"] == NULL_DIMENSION, arm
                assert not arm["stance_terms"], (arm["id"], arm["stance_terms"])
                assert not arm["evidence_terms"], (arm["id"], arm["evidence_terms"])
                # the sign is metadata only
                assert arm["direction"] == direction, arm
                seen += 1
    assert seen == len(REGISTERS) * len(_STANCE_PAIRS) * 3, seen
    print("  null_family_carries_no_stance_lexicon OK")


def test_null_text_is_identical_across_signs() -> None:
    """Endorse-labelled and doubt-labelled nulls must be the SAME string."""
    for register in REGISTERS:
        endorse = {
            a["realisation"]: a["text"]
            for a in paraphrase_family(
                channel="preference", direction="endorse", register=register
            )
            if a["is_null"]
        }
        doubt = {
            a["realisation"]: a["text"]
            for a in paraphrase_family(
                channel="preference", direction="doubt", register=register
            )
            if a["is_null"]
        }
        assert endorse == doubt, (register, endorse, doubt)
    print("  null_text_is_identical_across_signs OK")


def test_paraphrase_family_is_dimension_tagged() -> None:
    fam = paraphrase_family(channel="preference", direction="endorse", register="aita")
    dims = {a["dimension"] for a in fam}
    expected = set(PARAPHRASE_DIMENSIONS) | {DOSE_DIMENSION}
    assert dims == expected, (dims ^ expected)
    assert len({a["id"] for a in fam}) == len(fam)
    for arm in fam:
        assert arm["dimensions"] == (arm["dimension"],)
        assert arm["n_tokens"] == token_count(str(arm["text"]))
        assert arm["register"] == "aita"
    # nine interpretable dimensions plus the null pseudo-dimension
    assert len(PARAPHRASE_DIMENSIONS) == 10, sorted(PARAPHRASE_DIMENSIONS)
    for dim, reals in PARAPHRASE_DIMENSIONS.items():
        assert len(reals) >= 3, (dim, reals)
    print("  paraphrase_family_is_dimension_tagged OK")


def test_held_out_dimension_split() -> None:
    held = ["urgency", "sunk_cost"]
    train, test = held_out_split(
        channel="preference", direction="endorse", register="aita", held_out=held
    )
    train_dims = {a["dimension"] for a in train if not a["is_null"]}
    test_dims = {a["dimension"] for a in test if not a["is_null"]}
    assert train_dims.isdisjoint(held), train_dims
    assert test_dims == set(held), test_dims
    # the brittleness floor must be present on BOTH sides
    assert any(a["is_null"] for a in train)
    assert any(a["is_null"] for a in test)
    try:
        held_out_split(
            channel="preference", direction="endorse", register="aita", held_out=["bogus"]
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("held_out_split accepted an unknown dimension")
    print("  held_out_dimension_split OK")


def test_registers_map_to_instruments() -> None:
    assert instrument_for_register("math") == "brokenmath"
    assert instrument_for_register("aita") == "aita"
    assert instrument_for_register("ledger") == "ledger"
    print("  registers_map_to_instruments OK")


def test_library_self_checks() -> None:
    verify_all()
    assert set(CHANNELS) == {"preference", "testimony", "neutral_filler"}
    print("  library_self_checks OK")


# ---------------------------------------------------------------------------
# Nuisance perturbations
# ---------------------------------------------------------------------------

def test_nuisance_registry_shape() -> None:
    info = verify_registry()
    assert info["n_perturbations"] >= 20, info
    assert len(NUISANCE_PERTURBATIONS) == len(NUISANCE_SPECS)
    names = [n for n, _ in NUISANCE_PERTURBATIONS]
    assert len(names) == len(set(names))
    kinds = {s.kind for s in NUISANCE_SPECS}
    for required in ("whitespace", "reorder", "aside", "typo", "location", "time"):
        assert required in kinds, (required, kinds)
    print("  nuisance_registry_shape OK")


def test_nuisance_preserves_item_content() -> None:
    """No content word of the item is removed; none is added off-vocabulary."""
    verify_content_preservation(DEFAULT_PROBE_TEXTS)
    for text in DEFAULT_PROBE_TEXTS:
        base = content_words(text)
        for spec in NUISANCE_SPECS:
            delta = content_delta(text, spec.fn(text))
            assert not delta["removed"], (spec.name, delta["removed"])
            illegal = [w for w in delta["added"] if w not in spec.allowed_added]
            assert not illegal, (spec.name, illegal)
            # the item's own words survive with their counts intact
            after = content_words(spec.fn(text))
            for word, count in base.items():
                assert after.get(word, 0) >= count, (spec.name, word, count)
    print("  nuisance_preserves_item_content OK")


def test_purely_typographic_perturbations_add_nothing() -> None:
    typographic = [
        s for s in NUISANCE_SPECS if s.kind in ("whitespace", "typography", "formatting", "reorder")
    ]
    assert len(typographic) >= 10, len(typographic)
    for spec in typographic:
        assert not spec.allowed_added, (spec.name, spec.allowed_added)
        for text in DEFAULT_PROBE_TEXTS:
            delta = content_delta(text, spec.fn(text))
            assert not delta["added"] and not delta["removed"], (spec.name, delta)
    print("  purely_typographic_perturbations_add_nothing OK")


def test_typos_only_touch_function_words() -> None:
    text = "The dispute with my roommate was that the invoice arrived late."
    base = content_words(text)
    for name in ("typo_the", "typo_and", "typo_with", "typo_that"):
        out = apply_nuisance_by_name(text, name)
        after = content_words(out)
        for word, count in base.items():
            assert after.get(word, 0) >= count, (name, word)
    assert "teh" in apply_nuisance_by_name(text, "typo_the")
    assert "wtih" in apply_nuisance_by_name(text, "typo_with")
    print("  typos_only_touch_function_words OK")


def test_apply_nuisance_wraps_and_is_pure() -> None:
    text = DEFAULT_PROBE_TEXTS[0]
    n = len(NUISANCE_PERTURBATIONS)
    assert apply_nuisance(text, 0) == apply_nuisance(text, n)
    assert apply_nuisance(text, 3) == apply_nuisance(text, n + 3)
    before = text
    apply_nuisance(text, 5)
    assert text == before  # perturbations must not mutate their input
    print("  apply_nuisance_wraps_and_is_pure OK")


def test_conditional_perturbations_are_flagged_and_filterable() -> None:
    conditional = {s.name for s in NUISANCE_SPECS if s.conditional}
    assert {"location_denver", "location_glasgow", "time_morning", "time_late_night"} <= conditional
    strict = {s.name for s in nuisance_specs(strict=True)}
    assert strict.isdisjoint(conditional)
    assert len(strict) >= 20, len(strict)
    print("  conditional_perturbations_are_flagged_and_filterable OK")


def test_applicable_specs_drops_noops() -> None:
    text = "Short question with no lists."
    usable = applicable_specs(text)
    assert all(s.fn(text) != text for s in usable)
    assert len(usable) < len(NUISANCE_SPECS)
    assert len(usable) >= 5, len(usable)
    print("  applicable_specs_drops_noops OK")


# ---------------------------------------------------------------------------
# Nuisance reference distribution
# ---------------------------------------------------------------------------

def test_nuisance_percentile_bounds_and_order() -> None:
    ref = [0.004, 0.011, 0.002, 0.019, 0.007, 0.031, 0.005, 0.014]
    assert nuisance_percentile(0.0, ref) == 0.0
    assert nuisance_percentile(1.0, ref) == 100.0
    assert 0.0 <= nuisance_percentile(0.008, ref) <= 100.0
    prev = -1.0
    for effect in (0.0, 0.003, 0.008, 0.012, 0.020, 0.040):
        pct = nuisance_percentile(effect, ref)
        assert pct >= prev, (effect, pct, prev)
        prev = pct
    # magnitude, not sign: the floor is two-sided by construction
    assert nuisance_percentile(-0.020, ref) == nuisance_percentile(0.020, ref)
    # ties use the mid-rank convention
    assert nuisance_percentile(0.5, [0.5, 0.5]) == 50.0
    try:
        nuisance_percentile(0.1, [])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("nuisance_percentile accepted an empty reference set")
    print("  nuisance_percentile_bounds_and_order OK")


def test_nuisance_floor_and_summary() -> None:
    ref = [0.01, -0.02, 0.03, 0.04, 0.05]
    assert nuisance_floor(ref, q=0.0) == 0.01
    assert nuisance_floor(ref, q=1.0) == 0.05
    assert 0.01 <= nuisance_floor(ref, q=0.5) <= 0.05
    summary = summarize_nuisance(ref)
    assert summary["n"] == 5.0
    assert summary["max_abs"] == 0.05
    assert summary["median_abs"] == 0.03
    assert summary["p90"] <= summary["p99"]
    print("  nuisance_floor_and_summary OK")


def test_report_effect_requires_floor_and_threshold() -> None:
    ref = [0.004, 0.011, 0.002, 0.019, 0.007, 0.031, 0.005, 0.014]
    tiny = report_effect(0.006, ref, absolute_threshold=0.10)
    assert tiny["clears_floor"] is False and tiny["reportable"] is False
    mid = report_effect(0.045, ref, absolute_threshold=0.10)
    assert mid["clears_floor"] is True
    assert mid["clears_threshold"] is False
    assert mid["reportable"] is False, "clearing the floor alone must not suffice"
    big = report_effect(0.120, ref, absolute_threshold=0.10)
    assert big["clears_floor"] and big["clears_threshold"] and big["reportable"]
    unbounded = report_effect(0.120, ref)
    assert unbounded["clears_threshold"] is None and unbounded["reportable"] is True
    print("  report_effect_requires_floor_and_threshold OK")


_TESTS = (
    test_token_matching_within_register_dose_block,
    test_every_cell_has_at_least_two_paraphrases,
    test_assert_token_matched_catches_a_mismatch,
    test_audience_arms_token_matched,
    test_paraphrase_dimensions_token_matched,
    test_stance_sentence_validates_and_wraps,
    test_preference_channel_is_evidence_free,
    test_testimony_channel_asserts_evidence,
    test_neutral_filler_is_stance_free,
    test_will_read_is_orthogonal_to_view_alignment,
    test_audience_arm_ids_and_will_read_applicability,
    test_audience_scrutiny_2x2_is_factorial_and_length_controlled,
    test_null_family_carries_no_stance_lexicon,
    test_null_text_is_identical_across_signs,
    test_paraphrase_family_is_dimension_tagged,
    test_held_out_dimension_split,
    test_registers_map_to_instruments,
    test_library_self_checks,
    test_nuisance_registry_shape,
    test_nuisance_preserves_item_content,
    test_purely_typographic_perturbations_add_nothing,
    test_typos_only_touch_function_words,
    test_apply_nuisance_wraps_and_is_pure,
    test_conditional_perturbations_are_flagged_and_filterable,
    test_applicable_specs_drops_noops,
    test_nuisance_percentile_bounds_and_order,
    test_nuisance_floor_and_summary,
    test_report_effect_requires_floor_and_threshold,
)


def main() -> int:
    print("Stance-template / nuisance offline tests:")
    for fn in _TESTS:
        fn()
    print(f"All {len(_TESTS)} passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
