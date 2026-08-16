"""
scripts/test_runner_flagoff_identity.py -- differential proof that the
UNRESOLVED amendment is invisible when its flag is off.

WHAT THIS PINS
--------------
``scripts/run_brokenmath.py`` and ``scripts/run_elephant.py`` were amended to
support an opt-in forced-verdict regime (``--allow-unresolved``). The
non-negotiable invariant is that with the flag OFF the runners behave exactly
as they did before the amendment, because ``divergence_study_outputs/*.csv``
and the per-cell JSON caches underpin already-submitted results.

This test proves that invariant by DIFFERENTIAL EXECUTION rather than by
inspection:

  1. Recover the last pre-amendment revision of each runner from git (the most
     recent revision of the file that mentions neither ``verdict_format`` nor
     ``allow_unresolved``) and import it as a separate module.
  2. Point both the old and the new module at their own scratch ``OUT_DIR`` and
     replace ``generate`` with a deterministic recording stub.
  3. Drive both over the REAL corpora -- all 451 rows of
     data/brokenmath/benchmark.jsonl, and the ELEPHANT sample datasets
     including both flip-pairs moral modes -- with the flag off.
  4. Assert that every recorded generate() call is identical (model, system
     turn, user turn, sample_idx, max_tokens, kwargs) and that the resulting
     cache trees are byte-identical, filename for filename.
  5. Additionally assert that the per-cell ROW dicts (the CSV schema) are
     identical, with generation and judging stubbed out.

Everything runs offline: no API keys, no network, no GPU, and nothing is
written outside a temporary directory.

Usage:
  python -m scripts.test_runner_flagoff_identity
  python -m scripts.test_runner_flagoff_identity --verbose
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from scripts.generators import GenerationResult

REPO_ROOT = Path(__file__).resolve().parent.parent

#: A revision of a runner is "pre-amendment" when its source mentions none of
#: these. The amendment is the only thing that introduced them.
AMENDMENT_MARKERS: Tuple[str, ...] = ("verdict_format", "allow_unresolved")

VERBOSE = False


class Skipped(Exception):
    """Raised when the git history needed for the differential is unavailable."""


# ---------------------------------------------------------------------------
# Recovering and importing the pre-amendment module
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
    ).decode("utf-8")


def pre_amendment_source(rel_path: str) -> Tuple[str, str]:
    """Return ``(rev, source)`` of the newest pre-amendment revision.

    Walking history (rather than hard-coding ``HEAD~1``) is what keeps this
    test meaningful after further commits land.
    """
    try:
        revs = _git("log", "--format=%H", "--", rel_path).split()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Skipped(f"git unavailable: {exc}") from None
    if not revs:
        raise Skipped(f"no git history for {rel_path}")
    for rev in revs:
        try:
            src = _git("show", f"{rev}:{rel_path}")
        except subprocess.CalledProcessError:
            continue
        if not any(marker in src for marker in AMENDMENT_MARKERS):
            return rev, src
    raise Skipped(f"no pre-amendment revision of {rel_path} in history")


def import_source_as(module_name: str, source: str, workdir: Path) -> Any:
    """Import ``source`` as ``module_name`` from a file under ``workdir``."""
    path = workdir / f"{module_name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise Skipped(f"cannot load {module_name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# The deterministic generate() spy
# ---------------------------------------------------------------------------

def _digest(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def make_spy(records: List[tuple]) -> Callable[..., GenerationResult]:
    """A stand-in for ``scripts.generators.generate``.

    Deterministic in its inputs only, so any difference between the two
    modules' recorded outputs is a real behavioural difference. One in eleven
    prompts returns empty text on the first attempt, so the runners' retry
    loops are exercised identically on both sides.
    """

    def fake_generate(
        model: str, system: str, user: str, *,
        sample_idx: int, max_tokens: int = 4096, **kwargs: Any,
    ) -> GenerationResult:
        records.append(
            (model, system, user, sample_idx, max_tokens,
             tuple(sorted(kwargs.items()))),
        )
        d = _digest(model, system, user)
        if int(d[:4], 16) % 11 == 0 and sample_idx == 0:
            text = ""            # force the retry path
        else:
            text = f"reasoning {d[:16]}\n\nVERDICT: FALSE"
        return GenerationResult(
            text=text, finish_reason="stop", model=model,
            prompt_tokens=len(user) // 4, completion_tokens=len(text) // 4,
        )

    return fake_generate


def cache_tree(root: Path) -> Dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(root.glob("*.json"))}


def assert_same_calls(old: List[tuple], new: List[tuple], label: str) -> None:
    assert old, f"{label}: the differential recorded no generate() calls"
    assert len(old) == len(new), (
        f"{label}: call count differs -- pre-amendment {len(old)}, "
        f"amended {len(new)}"
    )
    for i, (a, b) in enumerate(zip(old, new)):
        if a != b:
            fields = ("model", "system", "user", "sample_idx", "max_tokens", "kwargs")
            diffs = [f"{f}: {x!r} != {y!r}" for f, x, y in zip(fields, a, b) if x != y]
            raise AssertionError(
                f"{label}: generate() call #{i} differs\n  " + "\n  ".join(diffs)
            )


def assert_same_caches(old: Path, new: Path, label: str) -> None:
    a, b = cache_tree(old), cache_tree(new)
    assert a, f"{label}: no cache files were written"
    assert set(a) == set(b), (
        f"{label}: cache filenames differ\n"
        f"  only pre-amendment: {sorted(set(a) - set(b))[:5]}\n"
        f"  only amended:       {sorted(set(b) - set(a))[:5]}"
    )
    for name in sorted(a):
        assert a[name] == b[name], f"{label}: cache bytes differ for {name}"


# ---------------------------------------------------------------------------
# BrokenMath
# ---------------------------------------------------------------------------

def _prepare(mod: Any, out_dir: Path, records: List[tuple]) -> Any:
    out_dir.mkdir(parents=True, exist_ok=True)
    mod.OUT_DIR = out_dir
    mod.generate = make_spy(records)
    return mod


def _brokenmath_cells(items: List[Any]) -> List[Tuple[Any, str, str]]:
    # One reasoning and one non-reasoning generator, so both the max_tokens
    # branch and the reasoning_effort kwarg branch are covered.
    gens = ["claude-haiku-4-5", "grok-4-1-fast-reasoning"]
    arms = ["raw", "standard_cot", "narrative_cot"]
    return [(it, g, a) for it in items for g in gens for a in arms]


def test_brokenmath_flag_off_is_byte_identical() -> None:
    from scripts import run_brokenmath as new_mod
    from scripts.load_brokenmath import load_brokenmath

    rev, src = pre_amendment_source("scripts/run_brokenmath.py")
    items = load_brokenmath(n=None, seed=44)
    assert len(items) == 451, f"expected the full 451-item corpus, got {len(items)}"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_mod = import_source_as("_pre_amendment_run_brokenmath", src, tmp)

        old_calls: List[tuple] = []
        new_calls: List[tuple] = []
        old_dir, new_dir = tmp / "old_bm", tmp / "new_bm"
        _prepare(old_mod, old_dir, old_calls)
        _prepare(new_mod, new_dir, new_calls)
        try:
            cells = _brokenmath_cells(items)
            for item, gen, arm in cells:
                old_mod._generate(item.problem, gen, arm, item.problem_id)
                # Flag off, both by omission and by explicit False.
                new_mod._generate(item.problem, gen, arm, item.problem_id)
                new_mod._generate(
                    item.problem, gen, arm, item.problem_id,
                    allow_unresolved=False,
                )
        finally:
            del sys.modules["_pre_amendment_run_brokenmath"]

        # The amended module was driven twice per cell; the second pass is a
        # pure cache hit, so it must add no calls at all.
        assert_same_calls(old_calls, new_calls, f"brokenmath vs {rev[:8]}")
        assert_same_caches(old_dir, new_dir, f"brokenmath vs {rev[:8]}")
        if VERBOSE:
            print(f"    {len(cells)} cells, {len(old_calls)} generate() calls, "
                  f"{len(cache_tree(old_dir))} cache files, rev {rev[:8]}")


def test_brokenmath_flag_off_row_schema_identical() -> None:
    """The default CSV schema (row keys and values) must not move either."""
    from scripts import run_brokenmath as new_mod
    from scripts.load_brokenmath import load_brokenmath

    rev, src = pre_amendment_source("scripts/run_brokenmath.py")
    items = load_brokenmath(n=25, seed=44)

    def fake_score(problem, original, response, *, problem_id, judge):
        return {
            "category": "Detected",
            "is_sycophant": 0,
            "judge": judge,
            "raw": f"stub {len(response)}",
        }

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_mod = import_source_as("_pre_amendment_run_brokenmath_rows", src, tmp)
        old_rows, new_rows = [], []
        try:
            for mod, out, sink in (
                (old_mod, tmp / "old_rows", old_rows),
                (new_mod, tmp / "new_rows", new_rows),
            ):
                _prepare(mod, out, [])
                mod.score_brokenmath_response = fake_score
                for item in items:
                    for arm in ("raw", "narrative_cot"):
                        sink.append(
                            mod._run_cell(item, "claude-haiku-4-5", arm, "judge-x"),
                        )
        finally:
            del sys.modules["_pre_amendment_run_brokenmath_rows"]
            new_mod.score_brokenmath_response = (
                __import__("scripts.brokenmath_scorer", fromlist=["x"])
                .score_brokenmath_response
            )

    assert old_rows and len(old_rows) == len(new_rows)
    for i, (a, b) in enumerate(zip(old_rows, new_rows)):
        assert list(a.keys()) == list(b.keys()), (
            f"row {i} key order/schema differs vs {rev[:8]}:\n"
            f"  pre-amendment {list(a.keys())}\n  amended       {list(b.keys())}"
        )
        assert a == b, f"row {i} values differ vs {rev[:8]}"


# ---------------------------------------------------------------------------
# ELEPHANT
# ---------------------------------------------------------------------------

def _elephant_cells() -> List[Tuple[str, Any, str, str, str]]:
    from scripts.load_elephant import _load_flip_pairs, load_elephant

    data_dir = Path("data/elephant")
    cells: List[Tuple[str, Any, str, str, str]] = []
    gens = ["gpt-5.4-nano", "claude-haiku-4-5"]
    arms = ["raw", "baseline_io", "narrative_cot"]
    for ds in ("oeq", "aita_yta", "ss"):
        items = load_elephant(
            ds, n=6, seed=44, data_dir=data_dir, allow_sample=True,
        )
        for it in items:
            for g in gens:
                for a in arms:
                    cells.append((ds, it, g, a, "binary"))
    pairs = _load_flip_pairs(n=6, seed=44, data_dir=data_dir, allow_sample=True)
    # Both moral modes: 'binary' is the branch that appends AITA_BINARY_SUFFIX
    # and is therefore the one the amendment had to leave alone.
    for mode in ("binary", "free_form"):
        for it in pairs:
            for g in gens:
                for a in arms:
                    cells.append(("flip_pairs", it, g, a, mode))
    return cells


def test_elephant_flag_off_is_byte_identical() -> None:
    from scripts import run_elephant as new_mod

    rev, src = pre_amendment_source("scripts/run_elephant.py")
    cells = _elephant_cells()
    assert cells, "no ELEPHANT cells built"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_mod = import_source_as("_pre_amendment_run_elephant", src, tmp)
        old_calls: List[tuple] = []
        new_calls: List[tuple] = []
        old_dir, new_dir = tmp / "old_el", tmp / "new_el"
        _prepare(old_mod, old_dir, old_calls)
        _prepare(new_mod, new_dir, new_calls)
        try:
            for ds, item, gen, arm, mode in cells:
                aita_binary = ds == "flip_pairs" and mode == "binary"
                out_ds = (
                    "flip_pairs_free"
                    if ds == "flip_pairs" and mode == "free_form" else ds
                )
                old_mod._generate_advice(
                    item.prompt, gen, arm, out_ds, item.id,
                    aita_binary=aita_binary, moral_mode=mode,
                )
                new_mod._generate_advice(
                    item.prompt, gen, arm, out_ds, item.id,
                    aita_binary=aita_binary, moral_mode=mode,
                )
                new_mod._generate_advice(
                    item.prompt, gen, arm, out_ds, item.id,
                    aita_binary=aita_binary, moral_mode=mode,
                    allow_unresolved=False,
                )
        finally:
            del sys.modules["_pre_amendment_run_elephant"]

        assert_same_calls(old_calls, new_calls, f"elephant vs {rev[:8]}")
        assert_same_caches(old_dir, new_dir, f"elephant vs {rev[:8]}")
        if VERBOSE:
            print(f"    {len(cells)} cells, {len(old_calls)} generate() calls, "
                  f"{len(cache_tree(old_dir))} cache files, rev {rev[:8]}")


def test_elephant_flag_off_row_and_moral_schema_identical() -> None:
    """Cell rows and moral pair rows must be identical with the flag off."""
    from scripts import run_elephant as new_mod
    from scripts.load_elephant import _load_flip_pairs

    rev, src = pre_amendment_source("scripts/run_elephant.py")
    pairs = _load_flip_pairs(
        n=8, seed=44, data_dir=Path("data/elephant"), allow_sample=True,
    )

    def fake_scores(dataset, prompt, response, judge, *, empty):
        return {"moral_placeholder": 0} if False else {}

    def fake_both_nta(a, b, *, free_form=False, judge=""):
        return int(a.strip().upper().startswith("NTA") and
                   b.strip().upper().startswith("NTA"))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_mod = import_source_as("_pre_amendment_run_elephant_rows", src, tmp)
        results = {}
        try:
            for tag, mod, out in (
                ("old", old_mod, tmp / "old_rows"),
                ("new", new_mod, tmp / "new_rows"),
            ):
                _prepare(mod, out, [])
                mod._score_all_metrics = fake_scores
                mod.moral_both_nta = fake_both_nta
                rows = []
                for it in pairs:
                    for arm in ("baseline_io", "narrative_cot"):
                        rows.append(mod._run_cell(
                            "flip_pairs", it.id, it.prompt, it.human_response,
                            it.human_scores, "claude-haiku-4-5", arm, "judge-x",
                            it.pair_id, it.side, "binary",
                        ))
                results[tag] = (rows, mod._build_flip_moral_rows(rows, "judge-x"))
        finally:
            del sys.modules["_pre_amendment_run_elephant_rows"]
            import scripts.elephant_scorers as _es
            new_mod.moral_both_nta = _es.moral_both_nta

    old_rows, old_moral = results["old"]
    new_rows, new_moral = results["new"]
    assert old_rows and len(old_rows) == len(new_rows)
    for i, (a, b) in enumerate(zip(old_rows, new_rows)):
        assert list(a.keys()) == list(b.keys()), (
            f"cell row {i} schema differs vs {rev[:8]}:\n"
            f"  pre-amendment {list(a.keys())}\n  amended       {list(b.keys())}"
        )
        assert a == b, f"cell row {i} values differ vs {rev[:8]}"
    assert old_moral and len(old_moral) == len(new_moral)
    key = lambda r: (r["pair_id"], r["generator"], r["arm"])
    for a, b in zip(sorted(old_moral, key=key), sorted(new_moral, key=key)):
        assert list(a.keys()) == list(b.keys()), (
            f"moral row schema differs vs {rev[:8]}:\n"
            f"  pre-amendment {list(a.keys())}\n  amended       {list(b.keys())}"
        )
        assert a == b, f"moral row values differ vs {rev[:8]}"


TESTS = [
    test_brokenmath_flag_off_is_byte_identical,
    test_brokenmath_flag_off_row_schema_identical,
    test_elephant_flag_off_is_byte_identical,
    test_elephant_flag_off_row_and_moral_schema_identical,
]


def main(argv: Optional[List[str]] = None) -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(
        description="Differential flag-off byte-identity tests for the runners",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="alias for the full suite")
    args = ap.parse_args(argv)
    VERBOSE = args.verbose
    print("flag-off differential identity tests (offline):")
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  {fn.__name__} OK")
        except Skipped as exc:
            print(f"  {fn.__name__} SKIPPED ({exc})")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"  {fn.__name__} FAILED [{type(exc).__name__}]\n{exc}")
    if failed:
        print(f"{failed} test(s) failed.")
        return 1
    print("All passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
