"""
scripts/build_manifest.py -- canonical work-manifest builder for CURC array jobs.

`slurm/array_generate.sbatch` consumes a JSONL manifest with ONE FLAT CELL PER
LINE (top-level ``model`` and ``user`` required). Nothing else in the repo emits
that schema, so this module is the single writer: every experiment that wants to
run on the cluster builds its manifest here, and every manifest is validated
before it is written.

Why validation is not optional: the worker skips malformed lines with a warning
and exits 0. A manifest in the wrong shape therefore produces a "successful" run
that generated nothing -- exactly the silent-underproduction failure the array
sharding guard exists to prevent. `validate_cells()` refuses to write instead.

Usage:
  python -m scripts.build_manifest --experiment stance --out work.jsonl
  python -m scripts.build_manifest --experiment crowdgold --out cg.jsonl --n 20
  python -m scripts.build_manifest --experiment t0b --out t0b.jsonl --models llama-3.1-8b
  python -m scripts.build_manifest --validate work.jsonl      # check an existing file
  python -m scripts.build_manifest --experiment stance --out /dev/null --dry-run

The report printed on write includes the number of cells ALREADY CACHED, so the
remaining work is visible before any GPU time is requested.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from scripts.run_phase1_quartet import PROMPTS, _safe, OUT_DIR

# Keys the worker reads. Anything else belongs in `record_extra`.
REQUIRED_KEYS = ("model", "user")
OPTIONAL_KEYS = (
    "system", "arm", "item_id", "problem_id", "id", "cache_kind", "cache_path",
    "max_tokens", "sample_idx", "temperature", "seed", "instrument",
    "append_verdict_instruction", "allow_unresolved", "record_extra",
    "chat_template_kwargs",
)
KNOWN_KEYS = frozenset(REQUIRED_KEYS + OPTIONAL_KEYS)

# Local open-weight families that default to emitting a <think>...</think>
# block before the answer (Qwen3). An earlier 8192 floor here was extrapolated
# from a trivial verdict on Qwen3-0.6B (165 completion tokens with thinking on)
# and was WRONG for real BrokenMath-difficulty items: CONFIRMED 2026-08-18 on
# CURC (job 31425222, Qwen3-8B, standard_cot arm, a real olympiad item) -- at
# an 8192 budget every single T0-B pilot cell (10/10) hit finish_reason="length"
# mid-<think>-block and produced NOVERDICT; raising the budget to 28000 for
# that same item let it finish naturally at finish_reason="stop" using only
# 12449 completion tokens (44%), with a real verdict extracted. 28000 also
# stays safely under Qwen3-8B's ~32k native context alongside these prompts
# (all under 350 tokens). Only one item has been directly measured this way --
# per-problem variance is real (see array_generate.sbatch's cache_is_complete,
# which now regenerates any cell that hit finish_reason=="length" rather than
# trusting a merely-non-empty cached response) -- so watch the next full T0-B
# smoke rerun across all pilot items/arms before assuming 28000 has zero
# truncations left. The hosted pipeline's own 8192 floor (scripts/generators.py,
# scripts/run_brokenmath.py) was NOT the confirmed-good precedent this comment
# used to claim -- it has the same untested floor and doesn't even record
# finish_reason in its cache, so it cannot be checked for this failure mode.
LOCAL_REASONING_FAMILIES = ("qwen3",)
REASONING_MAX_TOKENS = 28000


def is_local_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(fam in m for fam in LOCAL_REASONING_FAMILIES)


def budget_max_tokens(model: str, base: int) -> int:
    """Reasoning-aware max_tokens: never truncate a local reasoning model
    before it can reach its own thinking block's end."""
    if is_local_reasoning_model(model):
        return max(base, REASONING_MAX_TOKENS)
    return base


class ManifestError(ValueError):
    """Raised when a manifest would be unusable by the array worker."""


def cell_cache_path(cell: dict[str, Any]) -> Path:
    """Mirror the worker's cache-path derivation so we can count cache hits.

    Kept byte-compatible with `slurm/array_generate.sbatch::cell_cache_path`.
    """
    explicit = cell.get("cache_path")
    if explicit:
        return Path(explicit)
    kind = str(cell.get("cache_kind", "bm_gen"))
    model = str(cell.get("model", "unknown"))
    arm = str(cell.get("arm", "raw"))
    item = str(
        cell.get("item_id") or cell.get("problem_id") or cell.get("id") or "0"
    )
    return OUT_DIR / f"{kind}_{_safe(model)}_{arm}_{_safe(item)}.json"


def validate_cells(cells: Sequence[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable problems; empty means the manifest is good."""
    problems: list[str] = []
    for i, cell in enumerate(cells, 1):
        if not isinstance(cell, dict):
            problems.append(f"line {i}: not a JSON object")
            continue
        for key in REQUIRED_KEYS:
            val = cell.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                problems.append(f"line {i}: missing or empty required key {key!r}")
        unknown = set(cell) - KNOWN_KEYS
        if unknown:
            problems.append(
                f"line {i}: unknown key(s) {sorted(unknown)} -- the worker ignores "
                f"these; put per-cell metadata in 'record_extra'"
            )
        if cell.get("append_verdict_instruction") and not cell.get("instrument"):
            problems.append(
                f"line {i}: append_verdict_instruction set without 'instrument'"
            )
        for numeric in ("max_tokens", "sample_idx", "seed"):
            if numeric in cell and cell[numeric] is not None:
                if not isinstance(cell[numeric], int):
                    problems.append(f"line {i}: {numeric} must be an int")
    return problems


def _write_one(cells: Sequence[dict[str, Any]], out: Path, *, dry_run: bool) -> dict[str, Any]:
    cached = sum(1 for c in cells if cell_cache_path(c).exists())
    report = {
        "cells": len(cells),
        "already_cached": cached,
        "remaining": len(cells) - cached,
        "models": sorted({str(c.get("model")) for c in cells}),
        "arms": sorted({str(c.get("arm", "raw")) for c in cells}),
        "out": str(out),
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for cell in cells:
                fh.write(json.dumps(cell, ensure_ascii=False) + "\n")
    return report


def write_manifest(
    cells: Sequence[dict[str, Any]], out: Path, *, dry_run: bool = False,
    split_by_model: bool = False,
) -> dict[str, Any]:
    """Validate then write. Refuses to write an invalid manifest.

    ``split_by_model``: emit one file per model instead of one combined file.

    Why this matters for local (open-weight) runs, and does not matter for
    hosted ones: `array_generate.sbatch` uses round-robin striping (line %
    shards == shard) so that no single shard gets stuck with all the
    expensive cells. That is exactly right for hosted per-token billing, but
    it means a single array TASK's assigned cells interleave every model in
    the manifest -- and `local_backend.load_hf_model` caches every distinct
    model it is asked for in `_HF_BUNDLES` for the lifetime of the process.
    A combined manifest therefore makes every local worker load and hold
    EVERY model in the sweep simultaneously (two 8B models in bf16 is already
    ~32GB; mixing in a 70B is worse), rather than the one model its GPU
    allocation was sized for. Splitting by model, and submitting one array
    job per model with a GPU request sized to that model, is what the CURC
    runbook's `--gres` guidance already assumes.
    """
    problems = validate_cells(cells)
    if problems:
        head = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ManifestError(
            f"manifest failed validation ({len(problems)} problem(s)):\n  {head}{more}"
        )

    if not split_by_model:
        return _write_one(cells, out, dry_run=dry_run)

    by_model: dict[str, list[dict[str, Any]]] = {}
    for c in cells:
        by_model.setdefault(str(c.get("model")), []).append(c)

    per_file: dict[str, Any] = {}
    for model, model_cells in sorted(by_model.items()):
        model_out = out.with_name(f"{out.stem}.{_safe(model)}{out.suffix}")
        per_file[model] = _write_one(model_cells, model_out, dry_run=dry_run)

    return {
        "split_by_model": True,
        "n_models": len(by_model),
        "cells": len(cells),
        "already_cached": sum(r["already_cached"] for r in per_file.values()),
        "remaining": sum(r["remaining"] for r in per_file.values()),
        "per_model": per_file,
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    cells = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cells.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ManifestError(f"line {lineno}: invalid JSON: {e}") from e
    return cells


def coverage_check(n_cells: int, shards: int) -> dict[str, Any]:
    """Prove the worker's round-robin stripe covers every cell exactly once.

    The worker uses `line_index % shards == shard`. This reproduces it and
    asserts the union of shards is the whole manifest with no duplicates --
    the check `slurm/array_generate.sbatch` tells you to run before a big sweep.
    """
    seen: dict[int, int] = {}
    for shard in range(shards):
        for idx in range(n_cells):
            if idx % shards == shard:
                seen[idx] = seen.get(idx, 0) + 1
    missing = [i for i in range(n_cells) if i not in seen]
    duplicated = [i for i, c in seen.items() if c > 1]
    return {
        "cells": n_cells,
        "shards": shards,
        "covered": len(seen),
        "missing": missing[:10],
        "n_missing": len(missing),
        "duplicated": duplicated[:10],
        "n_duplicated": len(duplicated),
        "ok": not missing and not duplicated,
    }


# --------------------------------------------------------------------------
# Experiment builders. Each returns a list of flat cells.
# --------------------------------------------------------------------------

def _system_for(arm: str) -> str:
    return "" if arm == "raw" else PROMPTS.get(arm, PROMPTS["standard_cot"])


def build_stance(
    *, models: Sequence[str], arms: Sequence[str], n: int | None,
    allow_unresolved: bool, k: int,
) -> list[dict[str, Any]]:
    """BrokenMath stance factorial cells."""
    from scripts.load_brokenmath import load_brokenmath
    from scripts.stance_templates import stance_sentence

    items = load_brokenmath(n=n, seed=44)
    stance_arms = [
        ("neutral_filler", "neutral_filler", "none"),
        ("pref_endorse", "preference", "endorse"),
        ("pref_doubt", "preference", "doubt"),
        ("test_endorse", "testimony", "endorse"),
        ("test_doubt", "testimony", "doubt"),
    ]
    cells: list[dict[str, Any]] = []
    for item in items:
        for stance_id, channel, direction in stance_arms:
            sentence = stance_sentence(
                channel=channel, direction=direction, register="math",
            )
            user = f"{sentence}\n\n{item.problem}" if sentence else item.problem
            for arm in arms:
                for model in models:
                    for s in range(k):
                        cells.append({
                            "model": model,
                            "system": _system_for(arm),
                            "user": user,
                            "arm": f"{arm}__{stance_id}",
                            "item_id": f"{item.problem_id}__s{s}",
                            "cache_kind": "bm_stance_gen",
                            "instrument": "brokenmath",
                            "append_verdict_instruction": True,
                            "allow_unresolved": allow_unresolved,
                            "sample_idx": s,
                            "max_tokens": budget_max_tokens(model, 2048),
                            "record_extra": {
                                "stance_channel": channel,
                                "stance_direction": direction,
                                "base_arm": arm,
                                "problem_id": item.problem_id,
                            },
                        })
    return cells


def build_crowdgold(
    *, models: Sequence[str], arms: Sequence[str], n: int | None,
    allow_unresolved: bool, k: int,
) -> list[dict[str, Any]]:
    """Crowd-Gold AITA requester-identity cells (byte-identical post text)."""
    from scripts.run_crowdgold_aita import ARMS as CG_ARMS, build_user_turn, load_items

    want = n or 20
    items = load_items(
        source="scruples", n_yta=want, n_nta=want,
        min_votes=50, min_consensus=0.90, seed=44,
    )
    cells: list[dict[str, Any]] = []
    for item in items:
        for wrapper in CG_ARMS:
            user = build_user_turn(wrapper, item.post_text)
            for arm in arms:
                for model in models:
                    for s in range(k):
                        cells.append({
                            "model": model,
                            "system": _system_for(arm),
                            "user": user,
                            "arm": f"{arm}__{wrapper}",
                            "item_id": f"{item.item_id}__s{s}",
                            "cache_kind": "cg_gen",
                            "instrument": "aita",
                            "append_verdict_instruction": True,
                            "allow_unresolved": allow_unresolved,
                            "sample_idx": s,
                            "max_tokens": budget_max_tokens(model, 1536),
                            "record_extra": {
                                "wrapper": wrapper,
                                "gold_verdict": item.gold_verdict,
                                "n_votes": item.n_votes,
                                "consensus": item.consensus,
                                "post_id": item.item_id,
                            },
                        })
    return cells


def build_t0b(
    *, models: Sequence[str], arms: Sequence[str], n: int | None,
    allow_unresolved: bool, k: int,
) -> list[dict[str, Any]]:
    """T0-B variance-components pilot: m traces per item, exact scoring later."""
    from scripts.load_brokenmath import load_brokenmath

    items = load_brokenmath(n=n or 100, seed=44)
    cells: list[dict[str, Any]] = []
    for item in items:
        for arm in arms:
            for model in models:
                for trace_idx in range(k):
                    cells.append({
                        "model": model,
                        "system": _system_for(arm),
                        "user": item.problem,
                        "arm": arm,
                        "item_id": f"{item.problem_id}__t{trace_idx}",
                        "cache_kind": "t0b_trace",
                        "instrument": "brokenmath",
                        "append_verdict_instruction": True,
                        "allow_unresolved": allow_unresolved,
                        "sample_idx": trace_idx,
                        "temperature": 0.7,
                        "max_tokens": budget_max_tokens(model, 2048),
                        "record_extra": {
                            "trace_idx": trace_idx,
                            "problem_id": item.problem_id,
                        },
                    })
    return cells


BUILDERS = {
    "stance": build_stance,
    "crowdgold": build_crowdgold,
    "t0b": build_t0b,
}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a validated JSONL work manifest for slurm/array_generate.sbatch",
    )
    ap.add_argument("--experiment", choices=sorted(BUILDERS), help="Which study to build for")
    ap.add_argument("--out", type=Path, help="Output .jsonl path")
    ap.add_argument("--models", default="llama-3.1-8b",
                    help="Comma-separated model aliases/names")
    ap.add_argument("--arms", default="standard_cot,narrative_cot")
    ap.add_argument("--n", type=int, default=None, help="Item subsample")
    ap.add_argument("--k", type=int, default=3, help="Samples/traces per cell")
    ap.add_argument("--allow-unresolved", action="store_true", default=True,
                    help="Forced verdict line offers UNRESOLVED (default: on for new studies)")
    ap.add_argument("--no-allow-unresolved", dest="allow_unresolved",
                    action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="Validate and report, do not write")
    ap.add_argument("--validate", type=Path, help="Validate an existing manifest and exit")
    ap.add_argument("--coverage", type=int, metavar="SHARDS",
                    help="Prove shard coverage for this many shards and exit")
    ap.add_argument("--split-by-model", dest="split_by_model", action="store_true",
                    default=None,
                    help="One manifest file per model (default: on iff any "
                         "requested model is local -- see write_manifest docstring)")
    ap.add_argument("--no-split-by-model", dest="split_by_model", action="store_false")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Qwen3-family models only: chat_template_kwargs="
                         "{'enable_thinking': False}. Off by default -- the "
                         "reasoning trace is usually the object of study; use "
                         "this only for cheap verdict-only pilots.")
    args = ap.parse_args(argv)

    if args.validate:
        cells = read_manifest(args.validate)
        problems = validate_cells(cells)
        if problems:
            print(f"INVALID ({len(problems)} problem(s)):", file=sys.stderr)
            for p in problems[:20]:
                print(f"  {p}", file=sys.stderr)
            return 1
        cached = sum(1 for c in cells if cell_cache_path(c).exists())
        print(f"VALID: {len(cells)} cells, {cached} already cached, "
              f"{len(cells) - cached} remaining")
        if args.coverage:
            print(json.dumps(coverage_check(len(cells), args.coverage), indent=2))
        return 0

    if not args.experiment or not args.out:
        ap.error("--experiment and --out are required (or use --validate)")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cells = BUILDERS[args.experiment](
        models=models, arms=arms, n=args.n,
        allow_unresolved=args.allow_unresolved, k=args.k,
    )

    if args.disable_thinking:
        for c in cells:
            if is_local_reasoning_model(str(c.get("model", ""))):
                c["chat_template_kwargs"] = {"enable_thinking": False}
                # Thinking off means no thinking-budget tax to pay.
                c["max_tokens"] = min(int(c.get("max_tokens", 2048)), 2048)

    split_by_model = args.split_by_model
    if split_by_model is None:
        try:
            from scripts.local_backend import is_local_model
            split_by_model = any(is_local_model(m) for m in models)
        except Exception:
            split_by_model = False

    try:
        report = write_manifest(
            cells, args.out, dry_run=args.dry_run, split_by_model=split_by_model,
        )
    except ManifestError as e:
        print(f"REFUSED TO WRITE: {e}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    if args.coverage:
        cov = coverage_check(report["cells"], args.coverage)
        print(json.dumps(cov, indent=2))
        if not cov["ok"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
