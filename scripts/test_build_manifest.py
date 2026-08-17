"""
scripts/test_build_manifest.py -- offline regression tests for the manifest
builder, focused on the CURC-readiness fixes:

  1. reasoning-aware max_tokens (Qwen3 default-thinking truncation risk)
  2. split-by-model manifest emission (local-model GPU memory interleaving)
  3. --disable-thinking wiring
  4. hosted-only sweeps are unaffected (single combined file, as before)

No network, no GPU, no API keys. BrokenMath's benchmark.jsonl is a committed
local file, so build_t0b/build_stance run offline.

Run: python -m scripts.test_build_manifest
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.build_manifest import (
    budget_max_tokens,
    coverage_check,
    is_local_reasoning_model,
    main as build_manifest_main,
    read_manifest,
    validate_cells,
    write_manifest,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def test_reasoning_detection():
    print("is_local_reasoning_model / budget_max_tokens")
    check("qwen3-8b is reasoning", is_local_reasoning_model("qwen3-8b"))
    check("qwen3-32b is reasoning", is_local_reasoning_model("qwen3-32b"))
    check("Qwen/Qwen3-8B (raw repo id) is reasoning", is_local_reasoning_model("Qwen/Qwen3-8B"))
    check("llama-3.1-8b is NOT reasoning", not is_local_reasoning_model("llama-3.1-8b"))
    check("claude-haiku-4-5 is NOT reasoning", not is_local_reasoning_model("claude-haiku-4-5"))

    check("qwen3 budget floors at 8192", budget_max_tokens("qwen3-8b", 2048) == 8192)
    check("qwen3 budget never LOWERS an already-larger base",
          budget_max_tokens("qwen3-8b", 16000) == 16000)
    check("llama budget passes base through unchanged",
          budget_max_tokens("llama-3.1-8b", 2048) == 2048)


def test_split_by_model_isolates_files():
    print("write_manifest(split_by_model=True)")
    cells = [
        {"model": "llama-3.1-8b", "user": "u1", "arm": "raw", "item_id": "a"},
        {"model": "qwen3-8b", "user": "u2", "arm": "raw", "item_id": "a"},
        {"model": "llama-3.1-8b", "user": "u3", "arm": "raw", "item_id": "b"},
        {"model": "qwen3-8b", "user": "u4", "arm": "raw", "item_id": "b"},
    ]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "work.jsonl"
        report = write_manifest(cells, out, split_by_model=True)
        check("report flags split_by_model", report.get("split_by_model") is True)
        check("report counts 2 models", report.get("n_models") == 2)
        check("total cells preserved", report.get("cells") == 4)

        llama_file = out.with_name("work.llama-3.1-8b.jsonl")
        qwen_file = out.with_name("work.qwen3-8b.jsonl")
        check("llama file exists", llama_file.exists())
        check("qwen file exists", qwen_file.exists())
        check("no combined file was written", not out.exists())

        llama_rows = read_manifest(llama_file)
        qwen_rows = read_manifest(qwen_file)
        check("llama file has only llama cells",
              all(r["model"] == "llama-3.1-8b" for r in llama_rows) and len(llama_rows) == 2)
        check("qwen file has only qwen cells",
              all(r["model"] == "qwen3-8b" for r in qwen_rows) and len(qwen_rows) == 2)

        # Coverage self-test must hold independently on each split file.
        cov = coverage_check(len(llama_rows), 2)
        check("split file passes its own coverage check", cov["ok"])


def test_split_by_model_off_writes_one_file():
    print("write_manifest(split_by_model=False) -- unchanged single-file path")
    cells = [
        {"model": "claude-haiku-4-5", "user": "u1", "arm": "raw", "item_id": "a"},
        {"model": "claude-haiku-4-5", "user": "u2", "arm": "raw", "item_id": "b"},
    ]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "hosted.jsonl"
        report = write_manifest(cells, out, split_by_model=False)
        check("no split flag on unsplit report", "split_by_model" not in report)
        check("single file written", out.exists())
        check("file has both cells", len(read_manifest(out)) == 2)


def test_invalid_manifest_still_refused_with_new_key():
    print("validate_cells recognises chat_template_kwargs as a known key")
    good = [{"model": "qwen3-8b", "user": "u", "chat_template_kwargs": {"enable_thinking": False}}]
    bad = [{"model": "qwen3-8b", "user": "u", "totally_unknown_key": 1}]
    check("chat_template_kwargs cell is valid", validate_cells(good) == [])
    problems = validate_cells(bad)
    check("unknown key is still rejected", len(problems) == 1 and "totally_unknown_key" in problems[0])


def test_cli_t0b_split_and_reasoning_budget(tmp_out: Path):
    print("CLI: t0b experiment, mixed local models, default split-by-model")
    rc = build_manifest_main([
        "--experiment", "t0b", "--models", "llama-3.1-8b,qwen3-8b",
        "--n", "2", "--k", "1", "--out", str(tmp_out),
    ])
    check("CLI exits 0", rc == 0)
    qwen_file = tmp_out.with_name(f"{tmp_out.stem}.qwen3-8b{tmp_out.suffix}")
    llama_file = tmp_out.with_name(f"{tmp_out.stem}.llama-3.1-8b{tmp_out.suffix}")
    check("qwen manifest exists (auto-split on local models)", qwen_file.exists())
    check("llama manifest exists", llama_file.exists())
    if qwen_file.exists():
        rows = read_manifest(qwen_file)
        check("qwen cells all carry the reasoning token budget",
              all(r["max_tokens"] >= 8192 for r in rows),
              detail=str({r["max_tokens"] for r in rows}))
    if llama_file.exists():
        rows = read_manifest(llama_file)
        check("llama cells keep the base budget (not bumped)",
              all(r["max_tokens"] < 8192 for r in rows),
              detail=str({r["max_tokens"] for r in rows}))


def test_cli_disable_thinking(tmp_out: Path):
    print("CLI: --disable-thinking sets chat_template_kwargs and caps tokens")
    rc = build_manifest_main([
        "--experiment", "t0b", "--models", "qwen3-8b",
        "--n", "1", "--k", "1", "--disable-thinking", "--out", str(tmp_out),
    ])
    check("CLI exits 0", rc == 0)
    qwen_file = tmp_out.with_name(f"{tmp_out.stem}.qwen3-8b{tmp_out.suffix}")
    if qwen_file.exists():
        rows = read_manifest(qwen_file)
        check("thinking disabled on every row",
              all(r.get("chat_template_kwargs") == {"enable_thinking": False} for r in rows))
        check("token budget not inflated when thinking is off",
              all(r["max_tokens"] <= 2048 for r in rows))


def test_cli_hosted_only_still_single_file(tmp_out: Path):
    print("CLI: hosted-only crowdgold sweep stays a single combined file")
    rc = build_manifest_main([
        "--experiment", "crowdgold", "--models", "claude-haiku-4-5",
        "--n", "2", "--out", str(tmp_out),
    ])
    check("CLI exits 0", rc == 0)
    check("combined file exists (no split for hosted-only)", tmp_out.exists())
    split_variant = tmp_out.with_name(f"{tmp_out.stem}.claude-haiku-4-5{tmp_out.suffix}")
    check("no split variant was created", not split_variant.exists())


def main() -> int:
    test_reasoning_detection()
    test_split_by_model_isolates_files()
    test_split_by_model_off_writes_one_file()
    test_invalid_manifest_still_refused_with_new_key()
    with tempfile.TemporaryDirectory() as td:
        test_cli_t0b_split_and_reasoning_budget(Path(td) / "t0b.jsonl")
        test_cli_disable_thinking(Path(td) / "dt.jsonl")
        test_cli_hosted_only_still_single_file(Path(td) / "hosted.jsonl")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
