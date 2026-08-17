#!/usr/bin/env bash
#SBATCH --job-name=ani-gen
#SBATCH --partition=ah200
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --qos=gpu-normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-15%8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#
# =============================================================================
# slurm/array_generate.sbatch -- resumable single-GPU generation array.
# =============================================================================
#
# WHAT IT DOES
#   Reads a JSONL work manifest, takes this task's stripe of it, and for each
#   cell writes ONE per-cell JSON file into the repo's existing cache dir
#   (divergence_study_outputs/).  Cells whose cache file already exists are
#   skipped.  That single fact is what makes the job resumable, idempotent, and
#   safe to requeue: kill it at any point, resubmit the identical array, and it
#   picks up exactly where it stopped.  It never rewrites an existing cache
#   file, so committed results cannot be invalidated by a re-run.
#
# SUBMIT
#   mkdir -p slurm_logs                     # Slurm will not create this for you
#   export SBATCH_ACCOUNT=<your-account>    # from slurm/preflight.sh (U1)
#   sbatch --export=ALL,ANI_MANIFEST=/path/work.jsonl slurm/array_generate.sbatch
#
#   Size the array to the manifest, e.g. 2706 cells / ~170 per task:
#   sbatch --array=0-15%8 --export=ALL,ANI_MANIFEST=... slurm/array_generate.sbatch
#
# MANIFEST FORMAT (one JSON object per line)
#   Required:  model, user
#   Optional:  system (default ""), arm (default "raw"), item_id, max_tokens,
#              sample_idx, temperature, seed, cache_kind, cache_path,
#              instrument, append_verdict_instruction, record_extra
#
#   {"model":"Qwen3-8B","arm":"raw","item_id":"bm_0007","cache_kind":"bm_gen",
#    "system":"","user":"Prove that ...","max_tokens":2048,"sample_idx":0}
#
#   Cache path (matching the repo's verified convention):
#     OUT_DIR/{cache_kind}_{safe(model)}_{arm}_{safe(item_id)}.json
#   or the literal "cache_path" if the cell supplies one.
#
# ALTERNATE ALLOCATIONS -- override on the sbatch command line, or edit above.
#
#   Debug / smoke (1 h cap, 5 concurrent jobs, billed at 10%; gpu-testing is
#   valid ONLY on aa100 and ami100):
#     sbatch --partition=aa100 --qos=gpu-testing --gres=gpu:a100-40gb:1 \
#            --time=01:00:00 --array=0-1 ...
#
#   RTX Pro 6000 MIG (needs CUDA >12.8):
#     sbatch --partition=artxpro6000 --gres=gpu:rtx_pro_6000_2g.48gb:1 ...
#
#   Larger MIG slice for a 8B model with long contexts:
#     sbatch --partition=ah200 --gres=gpu:h200_3g.71gb:1 ...
#
#   32B, whole GPUs, tensor-parallel 2:
#     sbatch --partition=ah200 --gres=gpu:h200:2 --cpus-per-task=16 --mem=200G \
#            --qos=gpu-long --time=24:00:00 --export=ALL,ANI_TP_SIZE=2,ANI_MANIFEST=...
#
#   70B, whole GPUs, tensor-parallel 4:
#     sbatch --partition=ah200 --gres=gpu:h200:4 --cpus-per-task=32 --mem=400G \
#            --qos=gpu-long --time=48:00:00 --export=ALL,ANI_TP_SIZE=4,ANI_MANIFEST=...
#
# WHY MIG SLICES AND NOT ONE LONG MULTI-GPU JOB
#   An 8B model in bf16 needs ~16 GB, which fits an h200_2g.35gb slice with room
#   for KV cache.  A standing 4xH200 gpu-long reservation bills roughly 94k SU
#   per week at the A100 factor even while idle -- about a fifth of an
#   allocation spent on waiting.  Many small resumable array tasks bill only for
#   the work actually done and backfill into the scheduler far more easily.
#
# WHY NOT QUANTIZED
#   8B bf16 fits.  Quantization perturbs logprobs enough to contaminate the
#   small propensity shifts this study measures.
#
# ENVIRONMENT KNOBS (all optional; pass via --export=ALL,VAR=value)
#   ANI_MANIFEST         path to the JSONL work manifest (REQUIRED)
#   ANI_SHARD            override this task's shard index
#   ANI_SHARDS           override the total shard count
#   ANI_OUT_DIR          per-cell cache dir (default: repo divergence_study_outputs)
#   ANI_CKPT_DIR         checkpoint dir (default: $ANI_SCRATCH/ani_checkpoints)
#   ANI_RUN_NAME         groups checkpoints (default: manifest basename)
#   ANI_MAX_CELLS        stop after N cells (smoke testing)
#   ANI_DRY_RUN=1        resolve + report work, generate nothing
#   ANI_STRICT_EXISTS=1  skip on mere cache existence, even if response is empty
#                        (default: an empty cached response is regenerated,
#                         matching scripts/run_brokenmath.py)
#   ANI_CHECKPOINT_EVERY number of cells between checkpoint writes (default 10)
#   ANI_GRACE_S          seconds before walltime to stop cleanly (default 240)
#   ANI_AUTO_REQUEUE=1   self-requeue when stopped early with work remaining
#   ANI_TP_SIZE          tensor-parallel size, forwarded to the local backend
#   NOT_LOCAL_BACKEND    vllm | hf | auto  (see slurm/env_setup.sh)
#
# Nothing here hardcodes a username, an account, or a home path.
# =============================================================================
set -euo pipefail

# --- locate the repo --------------------------------------------------------
# SLURM_SUBMIT_DIR is where sbatch was invoked; fall back to this file's parent.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/slurm" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
export ANI_REPO="${ANI_REPO:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"

export ANI_QUIET="${ANI_QUIET:-0}"
# shellcheck source=./env_setup.sh
. "${SCRIPT_DIR}/env_setup.sh"

cd "${ANI_REPO}"

# --- job identity -----------------------------------------------------------
JOB_A="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_MIN="${SLURM_ARRAY_TASK_MIN:-${TASK_ID}}"
TASK_MAX="${SLURM_ARRAY_TASK_MAX:-${TASK_ID}}"

# --- shard derivation (MUST NOT silently under-cover) -----------------------
# The striping rule is `line_index % SHARDS == SHARD`, which covers the manifest
# exactly once ONLY when the task ids are the dense range [TASK_MIN, TASK_MAX].
# Slurm reports TASK_MIN/TASK_MAX as the *endpoints* regardless of stride or
# enumeration, so deriving SHARDS from the endpoints is wrong for specs like
# `--array=1-8:2` (drops ~43% of a 100-line manifest) or `--array=1,2,5` (~40%).
# Those runs complete "successfully" with a partial cache, which is the worst
# possible failure mode. So: prefer SLURM_ARRAY_TASK_COUNT (the true number of
# tasks), and refuse to start when the array is not a dense contiguous range.
ARRAY_COUNT="${SLURM_ARRAY_TASK_COUNT:-}"
SPAN=$(( TASK_MAX - TASK_MIN + 1 ))

if [ -n "${ANI_SHARDS:-}" ]; then
    # Explicit override: trust it, but validate against what Slurm reports.
    SHARDS="${ANI_SHARDS}"
    SHARD="${ANI_SHARD:-$(( TASK_ID - TASK_MIN ))}"
    if [ -n "${ARRAY_COUNT}" ] && [ "${ARRAY_COUNT}" -ne "${SHARDS}" ]; then
        printf 'array_generate: FATAL: ANI_SHARDS=%s but the array has %s tasks.\n' \
            "${SHARDS}" "${ARRAY_COUNT}" >&2
        printf 'array_generate: shards must equal the task count or cells are dropped.\n' >&2
        exit 2
    fi
elif [ -n "${ARRAY_COUNT}" ]; then
    if [ "${ARRAY_COUNT}" -ne "${SPAN}" ]; then
        printf 'array_generate: FATAL: non-contiguous array (%s tasks spanning %s..%s).\n' \
            "${ARRAY_COUNT}" "${TASK_MIN}" "${TASK_MAX}" >&2
        printf 'array_generate: strided (1-8:2) and enumerated (1,2,5) arrays would\n' >&2
        printf 'array_generate: silently process a SUBSET of the manifest.\n' >&2
        printf 'array_generate: use a dense range (--array=0-%s) or set ANI_SHARD/ANI_SHARDS.\n' \
            "$(( ARRAY_COUNT - 1 ))" >&2
        exit 2
    fi
    SHARDS="${ARRAY_COUNT}"
    SHARD="${ANI_SHARD:-$(( TASK_ID - TASK_MIN ))}"
else
    # No SLURM_ARRAY_TASK_COUNT (very old Slurm, or a non-array run).
    SHARDS="${SPAN}"
    SHARD="${ANI_SHARD:-$(( TASK_ID - TASK_MIN ))}"
fi

[ "${SHARDS}" -ge 1 ] || SHARDS=1
if [ "${SHARD}" -lt 0 ] || [ "${SHARD}" -ge "${SHARDS}" ]; then
    printf 'array_generate: FATAL: shard %s out of range for %s shards.\n' \
        "${SHARD}" "${SHARDS}" >&2
    exit 2
fi

# --- manifest ---------------------------------------------------------------
MANIFEST="${ANI_MANIFEST:-}"
if [ -z "${MANIFEST}" ]; then
    printf 'array_generate: ANI_MANIFEST is not set.\n' >&2
    printf 'array_generate: submit with --export=ALL,ANI_MANIFEST=/path/work.jsonl\n' >&2
    exit 2
fi
if [ ! -r "${MANIFEST}" ]; then
    printf 'array_generate: manifest not readable: %s\n' "${MANIFEST}" >&2
    exit 2
fi

RUN_NAME="${ANI_RUN_NAME:-$(basename -- "${MANIFEST}" .jsonl)}"
CKPT_DIR="${ANI_CKPT_DIR}/${RUN_NAME}_${JOB_A}"
mkdir -p "${CKPT_DIR}"
CKPT_FILE="${CKPT_DIR}/task_${TASK_ID}.json"
export ANI_CKPT_FILE="${CKPT_FILE}"

# --- graceful-stop deadline -------------------------------------------------
# Prefer the signal trap below, but also self-limit: if Slurm's signal is lost
# or delivered late, the worker still stops and checkpoints on its own.
GRACE_S="${ANI_GRACE_S:-240}"
DEADLINE=0
if [ -n "${SLURM_JOB_END_TIME:-}" ] && [ "${SLURM_JOB_END_TIME}" -gt 0 ] 2>/dev/null; then
    DEADLINE=$(( SLURM_JOB_END_TIME - GRACE_S ))
elif [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    _end="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
            | tr ' ' '\n' | awk -F= '/^EndTime=/{print $2; exit}')" || _end=""
    if [ -n "${_end}" ]; then
        _end_epoch="$(date -d "${_end}" '+%s' 2>/dev/null || true)"
        if [ -n "${_end_epoch}" ]; then DEADLINE=$(( _end_epoch - GRACE_S )); fi
    fi
fi
export ANI_DEADLINE_EPOCH="${DEADLINE}"

# --- banner -----------------------------------------------------------------
cat <<BANNER
==============================================================================
 ani-gen  array task ${TASK_ID}  (job ${JOB_A})
==============================================================================
  node          $(hostname -s 2>/dev/null || hostname)
  repo          ${ANI_REPO}
  manifest      ${MANIFEST}
  shard         ${SHARD} of ${SHARDS}
  out dir       ${ANI_OUT_DIR}
  checkpoint    ${CKPT_FILE}
  backend       ${NOT_LOCAL_BACKEND} -> ${ANI_RESOLVED_BACKEND}
  tp size       ${ANI_TP_SIZE:-1}
  cpus          ${SLURM_CPUS_PER_TASK:-?}
  gres          ${SLURM_JOB_GRES:-${SLURM_STEP_GRES:-?}}
  deadline      $( [ "${DEADLINE}" -gt 0 ] && printf '%s' "${DEADLINE}" || printf 'none (signal only)' )
  started       $(date -u '+%Y-%m-%dT%H:%M:%SZ')
==============================================================================
BANNER

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
else
    printf '  (nvidia-smi unavailable)\n'
fi

# --- emit the worker --------------------------------------------------------
# The worker is written out at runtime rather than kept as a separate module:
# this .sbatch is self-contained, so a job cannot half-run against a worker
# from a different commit.
WORK_TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ani_gen_${JOB_A}_${TASK_ID}"
mkdir -p "${WORK_TMP}"
WORKER="${WORK_TMP}/array_worker.py"
cleanup() { rm -rf "${WORK_TMP}" 2>/dev/null || true; }
trap cleanup EXIT

cat > "${WORKER}" <<'PYWORKER'
"""Array-task worker emitted by slurm/array_generate.sbatch.

Streams a JSONL work manifest, keeps this task's stripe, and generates only
the cells whose per-cell JSON cache is missing.  Writes each cache file
atomically and checkpoints progress so a requeued or resubmitted task resumes
exactly where it stopped.
"""
from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

# --- repo import path -------------------------------------------------------
REPO = os.environ.get("ANI_REPO") or os.getcwd()
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

try:
    from scripts.run_phase1_quartet import _safe
except Exception:  # pragma: no cover - repo layout fallback
    def _safe(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", s)

OUT_DIR = Path(os.environ.get("ANI_OUT_DIR") or (Path(REPO) / "divergence_study_outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- backend ----------------------------------------------------------------
# scripts/local_backend.py is the contracted router (local + hosted).  Until it
# lands, fall back to the hosted-only generator so this harness still runs.
BACKEND_NAME = "unknown"
try:
    from scripts.local_backend import generate_any as _generate
    BACKEND_NAME = "local_backend.generate_any"
except Exception as _exc:  # noqa: BLE001
    from scripts.generators import generate as _generate
    BACKEND_NAME = f"generators.generate (local_backend unavailable: {type(_exc).__name__})"

# --- optional verdict formatting -------------------------------------------
try:
    from scripts import verdict_format as _vf
except Exception:  # noqa: BLE001
    _vf = None

# --- config -----------------------------------------------------------------
STRICT_EXISTS = os.environ.get("ANI_STRICT_EXISTS", "0") == "1"
DRY_RUN = os.environ.get("ANI_DRY_RUN", "0") == "1"
CHECKPOINT_EVERY = int(os.environ.get("ANI_CHECKPOINT_EVERY", "10") or 10)
MAX_CELLS = int(os.environ.get("ANI_MAX_CELLS", "0") or 0)
DEADLINE = int(os.environ.get("ANI_DEADLINE_EPOCH", "0") or 0)
CKPT_FILE = Path(os.environ.get("ANI_CKPT_FILE") or (OUT_DIR / "ani_ckpt.json"))
ATTEMPTS = int(os.environ.get("ANI_ATTEMPTS", "3") or 3)

# --- graceful stop ----------------------------------------------------------
_STOP: dict[str, Any] = {"flag": False, "why": ""}


def _on_signal(signum: int, _frame: Any) -> None:
    if not _STOP["flag"]:
        _STOP["flag"] = True
        try:
            _STOP["why"] = signal.Signals(signum).name
        except Exception:  # noqa: BLE001
            _STOP["why"] = f"signal {signum}"
        print(f"\n[worker] {_STOP['why']} received -- finishing current cell, "
              f"then checkpointing and exiting.", flush=True)


for _sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _on_signal)
    except (ValueError, OSError):  # pragma: no cover
        pass


def _deadline_hit() -> bool:
    return DEADLINE > 0 and time.time() >= DEADLINE


# --- cache convention -------------------------------------------------------
def cell_cache_path(cell: dict) -> Path:
    """Resolve a cell's cache file.

    Matches the repo's verified naming, e.g.
    ``bm_gen_{safe(model)}_{arm}_{safe(problem_id)}.json``.
    """
    explicit = cell.get("cache_path")
    if explicit:
        p = Path(str(explicit))
        return p if p.is_absolute() else OUT_DIR / p
    kind = str(cell.get("cache_kind", "bm_gen"))
    model = str(cell.get("model", "unknown"))
    arm = str(cell.get("arm", "raw"))
    item = str(cell.get("item_id") or cell.get("problem_id") or cell.get("id") or "0")
    return OUT_DIR / f"{kind}_{_safe(model)}_{arm}_{_safe(item)}.json"


def cache_is_complete(path: Path) -> bool:
    """True when an existing cache file counts as done.

    Default matches scripts/run_brokenmath.py: a cached-but-empty response is
    treated as NOT done and is regenerated.  ANI_STRICT_EXISTS=1 makes mere
    existence sufficient.
    """
    if not path.exists():
        return False
    if STRICT_EXISTS:
        return True
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return bool(str(obj.get("response", "")).strip())


def write_cache_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# --- generation -------------------------------------------------------------
def build_prompts(cell: dict) -> tuple[str, str, Optional[str]]:
    """Return (system, user, instrument).

    The verdict instruction is appended ONLY when the cell opts in with
    ``append_verdict_instruction: true`` -- prompts are never silently altered,
    because the committed results were produced without them.
    """
    system = str(cell.get("system", "") or "")
    user = str(cell.get("user", "") or "")
    instrument = cell.get("instrument")
    if cell.get("append_verdict_instruction") and instrument and _vf is not None:
        try:
            instr = _vf.verdict_instruction(
                str(instrument),
                allow_unresolved=bool(cell.get("allow_unresolved", True)),
            )
            system = f"{system}\n\n{instr}".strip() if system else instr
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] verdict_instruction failed ({exc}); prompt left unchanged",
                  flush=True)
    return system, user, (str(instrument) if instrument else None)


def call_backend(cell: dict, system: str, user: str, attempt: int):
    """Call the backend, degrading gracefully if it rejects optional kwargs."""
    model = str(cell["model"])
    kwargs: dict[str, Any] = {
        "sample_idx": int(cell.get("sample_idx", 0)) + attempt,
        "max_tokens": int(cell.get("max_tokens", 2048)),
    }
    # Only kwargs the contract actually declares.  Tensor-parallel size is a
    # property of how the SERVER (or the in-process HF load) was started, not
    # of a single request, so it travels in the environment as ANI_TP_SIZE and
    # is never injected here -- generate_any forwards unknown kwargs straight
    # into the vLLM request body, where an unexpected field is a 400.
    optional: dict[str, Any] = {}
    if "temperature" in cell:
        optional["temperature"] = float(cell["temperature"])
    if cell.get("seed") is not None:
        optional["seed"] = int(cell["seed"])
    if cell.get("chat_template_kwargs") is not None:
        # Local (HF/vLLM) only -- e.g. {"enable_thinking": False} to switch a
        # Qwen3-family model out of its default reasoning mode. Ignored for
        # hosted models via the same TypeError degrade path as temperature/seed.
        optional["chat_template_kwargs"] = dict(cell["chat_template_kwargs"])

    try:
        return _generate(model, system, user, **kwargs, **optional)
    except TypeError:
        # Backend does not accept the optional kwargs (e.g. the hosted-only
        # generators.generate signature); retry with the contracted minimum.
        return _generate(model, system, user, **kwargs)


def generate_cell(cell: dict) -> tuple[dict, bool, str]:
    """Generate one cell.

    Returns ``(record, ok, failure_kind)`` where failure_kind is "" on success,
    "error" when every attempt raised, and "empty" when the backend returned
    successfully but with no text.  The two are counted separately because
    they call for different fixes: "error" is usually a dead endpoint or an OOM,
    "empty" is usually a max_tokens or chat-template problem.
    """
    system, user, instrument = build_prompts(cell)
    text = ""
    result = None
    used_attempts = 0
    last_error = ""
    for attempt in range(ATTEMPTS):
        used_attempts = attempt + 1
        try:
            result = call_backend(cell, system, user, attempt)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[worker]   attempt {attempt + 1}/{ATTEMPTS} failed: {last_error}",
                  flush=True)
            time.sleep(min(2 ** attempt, 8))
            continue
        text = (getattr(result, "text", "") or "").strip()
        if text:
            break

    record: dict[str, Any] = {
        # "response" and "arm" are the keys existing aggregators read; keep
        # them first and unchanged in meaning.
        "response": text,
        "arm": str(cell.get("arm", "raw")),
        "model": str(cell.get("model", "")),
        "item_id": str(cell.get("item_id") or cell.get("problem_id") or cell.get("id") or ""),
        "finish_reason": getattr(result, "finish_reason", "") if result else "",
        "prompt_tokens": getattr(result, "prompt_tokens", 0) if result else 0,
        "completion_tokens": getattr(result, "completion_tokens", 0) if result else 0,
        "latency_s": round(float(getattr(result, "latency_s", 0.0) or 0.0), 3) if result else 0.0,
        "backend": BACKEND_NAME,
        "attempts": used_attempts,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if last_error and not text:
        record["error"] = last_error
    if instrument and _vf is not None and text:
        try:
            record["instrument"] = instrument
            record["verdict"] = _vf.extract_verdict(text, instrument)
        except Exception as exc:  # noqa: BLE001
            record["verdict_error"] = f"{type(exc).__name__}: {exc}"
    extra = cell.get("record_extra")
    if isinstance(extra, dict):
        for k, v in extra.items():
            record.setdefault(k, v)
    if text:
        failure_kind = ""
    elif last_error:
        failure_kind = "error"
    else:
        failure_kind = "empty"
    return record, bool(text), failure_kind


# --- checkpointing ----------------------------------------------------------
def write_checkpoint(state: dict) -> None:
    state = dict(state)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    CKPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_FILE.with_name(f"{CKPT_FILE.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, CKPT_FILE)


# --- manifest streaming -----------------------------------------------------
def iter_shard(manifest: Path, shard: int, shards: int):
    """Yield (manifest_line_number, cell) for this task's stripe.

    Round-robin striping (line % shards == shard) rather than contiguous
    blocks: manifests are usually grouped by model/arm, so contiguous blocks
    would give one task all the cheap cells and another all the expensive ones.
    """
    with manifest.open(encoding="utf-8") as fh:
        idx = -1
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            idx += 1
            if idx % shards != shard:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[worker] manifest line {lineno}: bad JSON ({exc}); skipping",
                      flush=True)


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: array_worker.py MANIFEST SHARD SHARDS", file=sys.stderr)
        return 2
    manifest = Path(sys.argv[1])
    shard = int(sys.argv[2])
    shards = max(int(sys.argv[3]), 1)

    print(f"[worker] backend: {BACKEND_NAME}", flush=True)
    print(f"[worker] verdict_format: {'available' if _vf else 'NOT available'}", flush=True)
    print(f"[worker] out dir: {OUT_DIR}", flush=True)
    print(f"[worker] strict-exists: {STRICT_EXISTS}  dry-run: {DRY_RUN}", flush=True)

    state = {
        "run": os.environ.get("ANI_RUN_NAME", manifest.stem),
        "array_job": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "manifest": str(manifest),
        "shard": shard,
        "shards": shards,
        "backend": BACKEND_NAME,
        "assigned": 0,
        "skipped": 0,
        "generated": 0,
        "empty": 0,
        "errors": 0,
        "last_manifest_line": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": False,
        "stopped_by": "",
    }
    write_checkpoint(state)

    t0 = time.monotonic()
    processed = 0
    for lineno, cell in iter_shard(manifest, shard, shards):
        state["assigned"] += 1
        state["last_manifest_line"] = lineno

        if _STOP["flag"]:
            state["stopped_by"] = str(_STOP["why"])
            break
        if _deadline_hit():
            state["stopped_by"] = "deadline"
            print("[worker] walltime deadline reached -- stopping cleanly.", flush=True)
            break

        if not cell.get("model") or cell.get("user") is None:
            print(f"[worker] line {lineno}: cell missing 'model' or 'user'; skipping",
                  flush=True)
            state["errors"] += 1
            continue

        cache = cell_cache_path(cell)
        if cache_is_complete(cache):
            state["skipped"] += 1
            continue

        if DRY_RUN:
            print(f"[worker] would generate -> {cache.name}", flush=True)
            state["generated"] += 1
            processed += 1
        else:
            try:
                record, ok, failure_kind = generate_cell(cell)
            except Exception as exc:  # noqa: BLE001
                print(f"[worker] line {lineno}: unrecoverable: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                state["errors"] += 1
                continue
            if ok:
                write_cache_atomic(cache, record)
                state["generated"] += 1
            else:
                # Deliberately NOT cached: a failed cell must stay retryable on
                # the next pass rather than poisoning the cache with "".
                state["errors" if failure_kind == "error" else "empty"] += 1
                reason = ("every attempt raised: " + str(record.get("error", "")))\
                    if failure_kind == "error" else "backend returned no text"
                print(f"[worker] line {lineno}: {reason} after {ATTEMPTS} "
                      f"attempts; leaving {cache.name} uncached", flush=True)
            processed += 1

        if processed and processed % CHECKPOINT_EVERY == 0:
            rate = processed / max(time.monotonic() - t0, 1e-6)
            state["cells_per_s"] = round(rate, 4)
            write_checkpoint(state)
            print(f"[worker] {state['generated']} generated, {state['skipped']} skipped, "
                  f"{state['empty']} empty, {state['errors']} errors "
                  f"({rate * 3600:.0f} cells/h)", flush=True)

        if MAX_CELLS and processed >= MAX_CELLS:
            state["stopped_by"] = "max_cells"
            print(f"[worker] ANI_MAX_CELLS={MAX_CELLS} reached.", flush=True)
            break

    # How much of the stripe is still uncached?  Always recount (a stat per
    # cell) rather than trusting the counters, so a concurrent task's writes
    # are reflected too.
    remaining = 0
    for _lineno, cell in iter_shard(manifest, shard, shards):
        if not cache_is_complete(cell_cache_path(cell)):
            remaining += 1

    state["remaining"] = remaining
    # "traversed": we reached the end of the stripe instead of stopping early.
    state["traversed"] = not state["stopped_by"]
    # "complete": traversed AND nothing is left uncached.
    state["complete"] = state["traversed"] and remaining == 0
    # Only a clock-driven stop is worth retrying automatically.  A stripe that
    # traversed fully but still has uncached cells has cells that FAIL, and
    # requeueing it would spin forever burning SU.
    state["resumable"] = state["stopped_by"] in ("SIGUSR1", "SIGTERM", "SIGINT", "deadline")
    state["elapsed_s"] = round(time.monotonic() - t0, 1)
    write_checkpoint(state)

    print("-" * 78, flush=True)
    print(f"[worker] shard {shard}/{shards}: "
          f"assigned={state['assigned']} generated={state['generated']} "
          f"skipped={state['skipped']} empty={state['empty']} errors={state['errors']}",
          flush=True)
    if state["complete"]:
        print("[worker] shard COMPLETE -- every cell cached.", flush=True)
    elif state["traversed"]:
        print(f"[worker] shard TRAVERSED but {remaining} cell(s) remain uncached "
              f"({state['empty']} empty, {state['errors']} errored). These will NOT "
              f"fix themselves on a requeue -- inspect the log above before "
              f"resubmitting.", flush=True)
    else:
        print(f"[worker] shard STOPPED EARLY (stopped_by={state['stopped_by']}, "
              f"{remaining} cells left). Resubmit the identical array to resume; "
              f"cached cells will be skipped.", flush=True)
    print(f"[worker] checkpoint: {CKPT_FILE}", flush=True)
    # Exit 0 even when incomplete: stopping at a walltime cap is a normal,
    # expected outcome, not a job failure.  The checkpoint and the per-cell
    # cache carry the state.
    return 0


if __name__ == "__main__":
    sys.exit(main())
PYWORKER

# --- run with signal forwarding --------------------------------------------
PYBIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [ -z "${PYBIN}" ]; then
    printf 'array_generate: no python interpreter found\n' >&2
    exit 127
fi

"${PYBIN}" -u "${WORKER}" "${MANIFEST}" "${SHARD}" "${SHARDS}" &
CHILD=$!

forward() {
    printf '\n[batch] forwarding %s to worker pid %s\n' "$1" "${CHILD}"
    kill -"$1" "${CHILD}" 2>/dev/null || true
}
trap 'forward USR1' USR1
trap 'forward TERM' TERM
trap 'forward INT'  INT

# `wait` returns >128 when a trapped signal interrupts it, WITHOUT the child
# having exited.  Re-wait until the child is genuinely gone, so its final
# checkpoint is on disk before this script (and its EXIT trap) finishes.
#
# Note the two traps in this loop:
#   * `wait ... || st=$?` rather than `if wait ...; then`, because an `if` with
#     no else branch yields status 0 when its condition FAILS -- reading `$?`
#     after the `fi` would silently report success for an interrupted wait.
#   * liveness is decided by `kill -0`, not by the status value, so the loop is
#     correct regardless of the platform's 128+signum convention.
RC=0
while true; do
    st=0
    wait "${CHILD}" || st=$?
    if kill -0 "${CHILD}" 2>/dev/null; then
        continue
    fi
    RC="${st}"
    break
done

printf '[batch] worker exited rc=%s at %s\n' "${RC}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# --- optional self-requeue --------------------------------------------------
# Requeue ONLY when the worker stopped because it ran out of clock
# ("resumable": true).  A shard that traversed its whole stripe and still has
# uncached cells has cells that genuinely fail; requeueing that would spin
# forever and burn SU on a bug.
if [ "${ANI_AUTO_REQUEUE:-0}" = "1" ] && [ -r "${CKPT_FILE}" ] && command -v scontrol >/dev/null 2>&1; then
    if grep -q '"resumable": true' "${CKPT_FILE}"; then
        REQ_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"
        if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
            REQ_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        fi
        if [ -n "${REQ_ID}" ]; then
            printf '[batch] work remains; requeueing %s\n' "${REQ_ID}"
            scontrol requeue "${REQ_ID}" || printf '[batch] requeue failed\n' >&2
        fi
    fi
fi

exit "${RC}"
