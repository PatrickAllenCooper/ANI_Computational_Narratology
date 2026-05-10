"""
scripts/smoke_generators.py -- Verify all four generator models produce
non-empty output before running any large-scale phase.

Usage:
    cd /path/to/ANI_Examination
    python -m scripts.smoke_generators

Reads credentials from .env in the working directory (via python-dotenv).
Prints per-model latency and token usage; exits non-zero if any model fails.
"""
from __future__ import annotations

import os
import sys
import time

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.generators import generate

MODELS = [
    "gpt-5.4-nano",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "grok-4-1-fast-reasoning",
]

SYSTEM = (
    "You are a helpful assistant. Answer in one sentence."
)
USER = (
    "A doctor has one dose of a life-saving drug and two equally ill patients. "
    "What is the most important ethical principle to consider?"
)

def main() -> int:
    failures = []
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        t0 = time.monotonic()
        try:
            result = generate(
                model, SYSTEM, USER,
                sample_idx=0,
                max_tokens=256,
            )
        except Exception as e:
            print(f"  FAIL: {e}")
            failures.append(model)
            continue
        elapsed = time.monotonic() - t0
        if not result.text.strip():
            print(f"  FAIL: empty output (finish_reason={result.finish_reason!r})")
            failures.append(model)
            continue
        print(f"  OK  finish={result.finish_reason!r}  "
              f"tokens={result.prompt_tokens}+{result.completion_tokens}  "
              f"latency={elapsed:.2f}s")
        print(f"  Output: {result.text[:200]!r}")

    print("\n" + "="*60)
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("All models OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
