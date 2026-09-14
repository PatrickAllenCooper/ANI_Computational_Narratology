"""
scripts/probe_foundry.py -- one-call probe of a Foundry v1 deployment.

THE QUESTION. Does a named Azure AI Foundry deployment (Llama-3.3-70B-
Instruct, Mistral-Large-3-2, DeepSeek-V4-Pro -- the fourth-vendor arrivals
in scripts/generators.FOUNDRY_V1_MODELS) actually answer through the
generators.generate route, on which OpenAI-compatible surface
({base}/openai/v1 or {base}/models), and with the response fields the
runners read: finish_reason, prompt/completion usage, and text? A
deployment that returns finish_reason "length" with empty text at 64 tokens
is spending its budget on hidden reasoning and belongs in
generators.FOUNDRY_V1_REASONING (the 8192 floor) before any runner uses it.

WHAT IT SERVES. The graph ablation programme's cross-vendor addenda keep
model lists config-driven so a new vendor is a name plus a price, not a
code path; this probe is the ONE authorised call per deployment that turns
"plumbed" into "answers", so the first real run is not the first test.

WHAT IT DOES, AND ONLY WITH --run. Sends one fixed ~30-token prompt to one
deployment with max_tokens 64 (sample_idx 0, no system prompt) through
scripts.generators.generate, prints finish_reason, usage and the first 80
characters, and writes divergence_study_outputs/probe_foundry_<model>_<surface>.json.
With --surface both it tries openai/v1 first, then models, each as its own
call and its own artefact (a failure on one surface does not stop the other).
Without --run it prints the plan and refuses (exit 2). --selftest never
passes --run and never reaches generators.generate: the runner takes the
generator as an argument and the selftest hands it a stub.

    python -m scripts.probe_foundry --selftest
    python -m scripts.probe_foundry --model DeepSeek-V4-Pro                 # plan only, refuses
    python -m scripts.probe_foundry --model DeepSeek-V4-Pro --surface both --run   # 2 calls, ~$0.001

Cost ceiling per call: ~30 prompt + <=64 completion tokens, so under a
tenth of a cent at any of the three list prices in
run_crowdgold_deliberation.PRICES.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

OUT_DIR = Path("divergence_study_outputs")

#: The fixed probe prompt. ~30 tokens; short enough that a 64-token cap can
#: hold a complete answer from a non-reasoning model, so finish_reason
#: "stop" vs "length" separates "answers" from "burns budget on reasoning".
PROBE_PROMPT = ("In one short sentence, name the capital of France and one "
                "river that flows through it. Then stop.")
PROBE_MAX_TOKENS = 64
PROBE_SYSTEM = ""
SURFACE_CHOICES = ("openai/v1", "models", "both")
HEAD_CHARS = 80


def _safe(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _safe_surface(surface: str) -> str:
    return surface.replace("/", "_")


def plan(model: str, surface: str) -> list[dict]:
    """The call(s) a probe would make, in order: one dict per surface with the
    exact generate() arguments. Pure; no client, no network."""
    if not model or not model.strip():
        raise ValueError("--model is required")
    if surface not in SURFACE_CHOICES:
        raise ValueError(f"--surface must be one of {SURFACE_CHOICES}, got {surface!r}")
    surfaces = ("openai/v1", "models") if surface == "both" else (surface,)
    return [{"model": model.strip(), "surface": s, "system": PROBE_SYSTEM,
             "user": PROBE_PROMPT, "sample_idx": 0, "max_tokens": PROBE_MAX_TOKENS}
            for s in surfaces]


def run_probe(calls: Sequence[dict], *, gen: Callable, out_dir: Optional[Path] = None,
              write: bool = True) -> list[dict]:
    """Execute the planned calls through ``gen`` (generators.generate under
    --run; a stub under --selftest). Each call sets FOUNDRY_V1_SURFACE for
    its duration so the route posts to the requested surface, and restores
    the previous value afterwards. Never raises on a failed call: the
    failure is recorded and the next surface is still tried."""
    out_dir = OUT_DIR if out_dir is None else out_dir
    results: list[dict] = []
    for c in calls:
        prev = os.environ.get("FOUNDRY_V1_SURFACE")
        os.environ["FOUNDRY_V1_SURFACE"] = c["surface"]
        rec: dict = {"model": c["model"], "surface": c["surface"],
                     "max_tokens": c["max_tokens"], "prompt": c["user"],
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        t0 = time.monotonic()
        try:
            r = gen(c["model"], c["system"], c["user"],
                    sample_idx=c["sample_idx"], max_tokens=c["max_tokens"])
            rec.update({
                "ok": True,
                "finish_reason": getattr(r, "finish_reason", ""),
                "prompt_tokens": getattr(r, "prompt_tokens", 0),
                "completion_tokens": getattr(r, "completion_tokens", 0),
                "latency_s": round(getattr(r, "latency_s", 0.0) or (time.monotonic() - t0), 3),
                "text_head": (getattr(r, "text", "") or "")[:HEAD_CHARS],
                "text_chars": len(getattr(r, "text", "") or ""),
                "meta": dict(getattr(r, "meta", {}) or {}),
            })
        except Exception as e:  # noqa: BLE001 -- the probe reports, it does not crash
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:600],
                        "latency_s": round(time.monotonic() - t0, 3)})
        finally:
            if prev is None:
                os.environ.pop("FOUNDRY_V1_SURFACE", None)
            else:
                os.environ["FOUNDRY_V1_SURFACE"] = prev
        results.append(rec)
        print_result(rec)
        if write:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"probe_foundry_{_safe(c['model'])}_{_safe_surface(c['surface'])}.json"
            path.write_text(json.dumps(rec, indent=2))
            print(f"  wrote {path}")
    return results


def print_result(rec: dict) -> None:
    print(f"\n[probe] {rec['model']} via {rec['surface']}  (max_tokens {rec['max_tokens']})")
    if rec.get("ok"):
        print(f"  finish_reason : {rec['finish_reason']!r}")
        print(f"  usage         : prompt {rec['prompt_tokens']}  completion {rec['completion_tokens']}"
              f"  latency {rec['latency_s']}s")
        print(f"  text[:{HEAD_CHARS}]     : {rec['text_head']!r}  ({rec['text_chars']} chars)")
        if rec["finish_reason"] == "length" and rec["text_chars"] == 0:
            print("  NOTE: budget exhausted with no visible text -- candidate for "
                  "generators.FOUNDRY_V1_REASONING (8192 floor)")
    else:
        print(f"  FAILED        : {rec.get('error')}  (after {rec.get('latency_s')}s)")


def print_plan(calls: Sequence[dict]) -> None:
    print("probe plan (no call made):")
    for c in calls:
        print(f"  generate(model={c['model']!r}, system={c['system']!r}, user=<{len(c['user'])} chars>, "
              f"sample_idx={c['sample_idx']}, max_tokens={c['max_tokens']})  "
              f"with FOUNDRY_V1_SURFACE={c['surface']}")


# ---------------------------------------------------------------------------
# Selftest: refusal and argument wiring, with a stub generator; never --run
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import contextlib
    import inspect
    import io
    import tempfile

    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")

    print("probe_foundry selftest (offline; the generator is a stub; --run is never passed)\n")
    import scripts.generators as generators
    real_probe_files_before = sorted(p.name for p in OUT_DIR.glob("probe_foundry_*"))

    # -- 1. the prompt and cap are what the docstring promises ---------------
    n_words = len(PROBE_PROMPT.split())
    check("fixed prompt is short (~30 tokens: 15-30 words) and asks for one sentence",
          15 <= n_words <= 30 and "one short sentence" in PROBE_PROMPT, f"{n_words} words")
    check("max_tokens is 64 and the system prompt is empty",
          PROBE_MAX_TOKENS == 64 and PROBE_SYSTEM == "")

    # -- 2. plan wiring -------------------------------------------------------
    p1 = plan("DeepSeek-V4-Pro", "openai/v1")
    check("--surface openai/v1 plans exactly one call on openai/v1",
          len(p1) == 1 and p1[0]["surface"] == "openai/v1")
    pm = plan("DeepSeek-V4-Pro", "models")
    check("--surface models plans exactly one call on models",
          len(pm) == 1 and pm[0]["surface"] == "models")
    pb = plan(" Llama-3.3-70B-Instruct ", "both")
    check("--surface both plans openai/v1 THEN models, model name stripped",
          [c["surface"] for c in pb] == ["openai/v1", "models"]
          and all(c["model"] == "Llama-3.3-70B-Instruct" for c in pb))
    check("every planned call carries the fixed prompt, sample_idx 0, max_tokens 64, empty system",
          all(c["user"] == PROBE_PROMPT and c["sample_idx"] == 0 and c["max_tokens"] == 64
              and c["system"] == "" for c in p1 + pm + pb))
    for bad_model, bad_surface in (("", "openai/v1"), ("x", "nope")):
        try:
            plan(bad_model, bad_surface)
            check(f"plan refuses model={bad_model!r} surface={bad_surface!r}", False)
        except ValueError:
            check(f"plan refuses model={bad_model!r} surface={bad_surface!r}", True)
    check("the planned model names are Foundry v1 names in generators (they would route there)",
          all(generators._is_foundry_v1(c["model"]) for c in p1 + pb))

    # -- 3. run_probe with a stub: surface env set per call and restored ------
    seen: list[dict] = []

    class _R:
        def __init__(self, text, fr, pt, ct):
            self.text, self.finish_reason = text, fr
            self.prompt_tokens, self.completion_tokens = pt, ct
            self.latency_s, self.meta = 0.01, {"surface": os.environ.get("FOUNDRY_V1_SURFACE")}

    def stub(model, system, user, *, sample_idx, max_tokens, **kw):
        seen.append({"model": model, "system": system, "user": user, "sample_idx": sample_idx,
                     "max_tokens": max_tokens, "surface_env": os.environ.get("FOUNDRY_V1_SURFACE")})
        if os.environ.get("FOUNDRY_V1_SURFACE") == "models":
            raise RuntimeError("stub: 404 on /models")
        return _R("Paris is the capital of France; the Seine flows through it." * 3, "stop", 31, 18)

    prev_surface = os.environ.get("FOUNDRY_V1_SURFACE")
    os.environ["FOUNDRY_V1_SURFACE"] = "openai/v1"
    with tempfile.TemporaryDirectory() as td:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = run_probe(plan("DeepSeek-V4-Pro", "both"), gen=stub, out_dir=Path(td))
        check("--surface both makes two stub calls, openai/v1 then models, each with the fixed "
              "prompt, sample_idx 0, max_tokens 64",
              [s["surface_env"] for s in seen] == ["openai/v1", "models"]
              and all(s["user"] == PROBE_PROMPT and s["sample_idx"] == 0 and s["max_tokens"] == 64
                      and s["model"] == "DeepSeek-V4-Pro" and s["system"] == "" for s in seen))
        check("a failure on one surface is recorded and the other surface's result stands",
              res[0]["ok"] and res[0]["finish_reason"] == "stop"
              and res[0]["prompt_tokens"] == 31 and res[0]["completion_tokens"] == 18
              and not res[1]["ok"] and "404" in res[1]["error"])
        check(f"text is truncated to {HEAD_CHARS} chars in the record and the full length is kept",
              len(res[0]["text_head"]) == HEAD_CHARS and res[0]["text_chars"] > HEAD_CHARS)
        check("finish_reason, usage and the text head are printed",
              "finish_reason : 'stop'" in buf.getvalue() and "prompt 31  completion 18" in buf.getvalue()
              and "text[:80]" in buf.getvalue() and "FAILED" in buf.getvalue())
        files = sorted(p.name for p in Path(td).iterdir())
        check("one JSON artefact per surface, named by model and surface",
              files == ["probe_foundry_DeepSeek-V4-Pro_models.json",
                        "probe_foundry_DeepSeek-V4-Pro_openai_v1.json"], str(files))
        check("FOUNDRY_V1_SURFACE is restored after the probe",
              os.environ.get("FOUNDRY_V1_SURFACE") == "openai/v1")
    if prev_surface is None:
        os.environ.pop("FOUNDRY_V1_SURFACE", None)
    else:
        os.environ["FOUNDRY_V1_SURFACE"] = prev_surface
    seen.clear()
    with tempfile.TemporaryDirectory() as td:
        with contextlib.redirect_stdout(io.StringIO()):
            res2 = run_probe(plan("Mistral-Large-3-2", "openai/v1"), gen=stub, out_dir=Path(td), write=False)
        check("write=False writes nothing", res2[0]["ok"] and not list(Path(td).iterdir()))
        check("FOUNDRY_V1_SURFACE unset before the probe stays unset after it",
              (prev_surface is None) == ("FOUNDRY_V1_SURFACE" not in os.environ))

    # -- 4. main() refuses without --run and never touches generators.generate -
    real_generate = generators.generate
    calls: list = []

    def trap(*a, **kw):
        calls.append((a, kw))
        raise AssertionError("generators.generate must not be reached without --run")

    generators.generate = trap  # type: ignore[assignment]
    try:
        for argv in (["--model", "DeepSeek-V4-Pro"],
                     ["--model", "DeepSeek-V4-Pro", "--surface", "both"],
                     ["--model", "Llama-3.3-70B-Instruct", "--surface", "models"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(argv)
            check(f"main({argv}) without --run prints the plan, refuses (exit 2), makes no call",
                  rc == 2 and "NOT RUNNING" in buf.getvalue() and "probe plan" in buf.getvalue()
                  and not calls, f"rc={rc}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                rc = main(["--surface", "both"])
            except SystemExit as e:
                rc = e.code
        check("main() without --model is refused", rc in (2,) and not calls, f"rc={rc}")
    finally:
        generators.generate = real_generate  # type: ignore[assignment]
    check("generators.generate restored", generators.generate is real_generate)
    src = inspect.getsource(main)
    check("main() hands generators.generate to run_probe exactly once, and only under `if args.run`",
          src.count("gen=generators.generate") == 1 and "if args.run" in src
          and src.index("if args.run") < src.index("gen=generators.generate")
          and src.count("run_probe(") == 1)
    check("no artefact was written to the real output directory by the selftest",
          sorted(p.name for p in OUT_DIR.glob("probe_foundry_*")) == real_probe_files_before)

    print("\n" + ("ALL OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="One-call probe of a Foundry v1 deployment through "
                    "scripts.generators.generate. Makes NO call unless --run is passed.")
    ap.add_argument("--selftest", action="store_true",
                    help="Refusal and argument wiring against a stub generator. Never calls a model.")
    ap.add_argument("--model", default=None,
                    help="Exact Foundry deployment name (e.g. DeepSeek-V4-Pro). Required unless --selftest.")
    ap.add_argument("--surface", choices=SURFACE_CHOICES, default="openai/v1",
                    help="Which OpenAI-compatible path to post to; 'both' tries openai/v1 then models "
                         "(two calls).")
    ap.add_argument("--run", action="store_true",
                    help="REQUIRED to send the probe; spends a fraction of a cent per surface.")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.model:
        print("ERROR: --model is required (or --selftest)")
        return 2
    try:
        calls = plan(args.model, args.surface)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2
    print_plan(calls)
    if not any(generators_is_foundry_v1(c["model"]) for c in calls):
        print(f"  WARNING: {args.model!r} is not a Foundry v1 name in scripts.generators "
              "(FOUNDRY_V1_MODELS or the FOUNDRY_V1_MODELS env var); generate() would route it "
              "elsewhere and FOUNDRY_V1_SURFACE would be ignored.")
    if args.run:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:  # noqa: BLE001 -- dotenv is optional
            pass
        import scripts.generators as generators
        print(f"\n*** --run: sending {len(calls)} probe call(s). ***")
        results = run_probe(calls, gen=generators.generate)
        return 0 if all(r.get("ok") for r in results) else 1
    print("\n*** NOT RUNNING: pass --run to send the probe (this spends money). "
          "Nothing was sent. ***")
    return 2


def generators_is_foundry_v1(model: str) -> bool:
    """Routing predicate lookup kept out of main() so main() names
    generators.generate exactly once (the selftest asserts that)."""
    import scripts.generators as generators
    return generators._is_foundry_v1(model)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
