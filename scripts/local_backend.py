"""
scripts/local_backend.py -- Open-weights (local) generation backend.

One interface, two engines, so every existing runner in this repo can be
pointed at a CURC-hosted open-weights model with a one-line change:

    -from scripts.generators import generate
    +from scripts.local_backend import generate_any as generate

``generate_any`` routes local aliases (``llama-3.1-8b``, ``qwen3-8b``,
``qwen3-32b``, ``llama-3.3-70b``) to ``local_generate`` and everything else
to the unchanged hosted router in ``scripts.generators.generate``.  Hosted
behaviour is untouched: nothing here changes a prompt, a cache key, or a
hosted call, so committed results and on-disk caches stay valid.

Engines
-------
vllm (default)
    Talks to a vLLM server over its OpenAI-compatible HTTP endpoint using the
    ``openai`` client that is already a dependency of this repo.  The base URL
    comes from ``NOT_VLLM_BASE_URL`` or from the host:port file the Slurm serve
    job writes (``NOT_VLLM_ENDPOINT_FILE``, default
    ``./divergence_study_outputs/vllm_endpoint.txt``).  No new hard dependency:
    vLLM itself is never imported in this process.

hf
    ``transformers`` in-process.  ``torch`` / ``transformers`` are imported
    lazily inside functions, so this module imports fine on a laptop with no
    GPU and no ML stack (that property is asserted by
    ``scripts/test_local_backend.py``).

Backend selection is ``NOT_LOCAL_BACKEND`` (``vllm`` default, ``hf``
fallback).  When ``vllm`` is requested but no endpoint answers, the module
falls back to ``hf`` and prints a warning; set ``NOT_LOCAL_NO_FALLBACK=1`` to
turn that silent degradation into a hard, actionable error instead.

DETERMINISM -- READ THIS BEFORE WRITING AN EXACT-REPRODUCTION TEST
------------------------------------------------------------------
vLLM is NOT bitwise-deterministic across batch compositions.  Continuous
batching means a request is reduced together with whatever other requests
happen to be in flight, and floating-point reduction order changes with batch
shape.  Identical prompt + ``temperature=0`` + identical ``seed`` can therefore
still produce a different token at some position, and one divergent token can
change the whole continuation.  Greedy decoding on vLLM is *statistically*
stable, not *bitwise* reproducible.

Consequence: any test that asserts exact string equality between two
generations -- notably the AGT check that "steering coefficient 0.0 must
reproduce the unsteered output" -- MUST use the HF path, batch size 1, greedy:

    os.environ["NOT_LOCAL_BACKEND"] = "hf"
    enable_hf_determinism()                      # torch.use_deterministic_algorithms(True)
    a = local_generate("qwen3-8b", sys_p, usr_p, sample_idx=0,
                       temperature=0.0, seed=0, deterministic=True)
    b = local_generate("qwen3-8b", sys_p, usr_p, sample_idx=0,
                       temperature=0.0, seed=0, deterministic=True)
    assert a.text == b.text

``deterministic=True`` sets the seeds, enables deterministic algorithms, and
records ``meta["deterministic"]=True`` on the result.  It also requires
``CUBLAS_WORKSPACE_CONFIG=:4096:8`` on CUDA; the module sets that env var
itself if it is unset, but it only takes effect if it is set *before* the
first CUDA context is created, so prefer setting it in the Slurm script.

Environment variables
---------------------
  NOT_LOCAL_BACKEND        "vllm" (default) | "hf" | "auto"
  ANI_LOCAL_BACKEND        already-resolved backend from slurm/env_setup.sh;
                           used only when NOT_LOCAL_BACKEND is unset/"auto"
  NOT_LOCAL_NO_FALLBACK    "1" -> error instead of falling back to hf
  NOT_VLLM_BASE_URL        e.g. http://c3gpu-c2-u17:8000/v1
  NOT_VLLM_ENDPOINT_FILE   host:port file (or directory of them) from the serve job
  ANI_ENDPOINT_DIR         serve-job endpoint directory (slurm/env_setup.sh)
  NOT_VLLM_API_KEY         token for the vLLM server (default "EMPTY")
  NOT_VLLM_SERVED_NAME     --served-model-name if it differs from the repo id
  NOT_VLLM_TIMEOUT         per-request timeout, seconds (default 600)
  NOT_VLLM_PROBE_TIMEOUT   health-probe timeout, seconds (default 10)
  NOT_VLLM_JSON_MODE       "1" -> honour json_mode via response_format
  NOT_LOCAL_MAX_ATTEMPTS   retry attempts per call (default 5, as generators.py)
  NOT_LOCAL_EXTRA_MODELS   JSON dict of extra alias -> HF repo id
  NOT_LOCAL_ALLOW_ANY_REPO "1" -> treat any "org/name" string as a local model
  NOT_HF_DEVICE            "cuda" | "cpu" | "auto" (default "auto")
  NOT_HF_DTYPE             "bfloat16" (default on CUDA) | "float16" | "float32"
  HF_HUB_OFFLINE=1         recommended on compute nodes (stage weights first)

Usage:
  python -m scripts.local_backend --smoke        # offline; no network, no GPU
  python -m scripts.local_backend --list
  python -m scripts.local_backend --model qwen3-8b --user "2+2=?"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from scripts.generators import GenerationResult

try:  # optional, mirrors the other runners
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


# ---------------------------------------------------------------------------
# Alias table
# ---------------------------------------------------------------------------

#: Local alias -> canonical HuggingFace repo id.  Aliases are what appears in
#: cache filenames and result CSVs, so they are short and filesystem-safe.
LOCAL_MODELS: dict[str, str] = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
}

#: Prefixes a caller may use to force the local route, e.g. "local/qwen3-8b".
_LOCAL_PREFIXES = ("local/", "local:", "hf/", "hf:")

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_VLLM_TIMEOUT = 600.0
DEFAULT_PROBE_TIMEOUT = 10.0
DEFAULT_ENDPOINT_FILES = (
    "./divergence_study_outputs/vllm_endpoint.txt",
    "~/.not_vllm_endpoint",
)


class LocalBackendError(RuntimeError):
    """Base class for local-backend failures."""


class LocalEndpointUnavailable(LocalBackendError):
    """Raised when the vLLM endpoint cannot be reached and fallback is off."""


class UnknownLocalModel(LocalBackendError):
    """Raised when an alias is not in LOCAL_MODELS / NOT_LOCAL_EXTRA_MODELS."""


# ---------------------------------------------------------------------------
# Small env helpers (read at call time so tests and Slurm scripts can flip them)
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _max_attempts() -> int:
    try:
        n = int(os.environ.get("NOT_LOCAL_MAX_ATTEMPTS", "") or DEFAULT_MAX_ATTEMPTS)
    except ValueError:
        n = DEFAULT_MAX_ATTEMPTS
    return max(1, n)


def _extra_models() -> dict[str, str]:
    raw = os.environ.get("NOT_LOCAL_EXTRA_MODELS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k).strip().lower(): str(v) for k, v in parsed.items()}


def _alias_table() -> dict[str, str]:
    table = dict(LOCAL_MODELS)
    table.update(_extra_models())
    return table


def _normalise(model: str) -> str:
    m = (model or "").strip()
    low = m.lower()
    for pref in _LOCAL_PREFIXES:
        if low.startswith(pref):
            return m[len(pref):].strip()
    return m


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def is_local_model(model: str) -> bool:
    """True if ``model`` names an open-weights model served by this module.

    Accepts an alias (``qwen3-8b``, case-insensitive), an explicitly prefixed
    alias (``local/qwen3-8b``), or a canonical HF repo id already present in
    the alias table.  Hosted names (``gpt-5.4-nano``, ``claude-haiku-4-5``,
    ``grok-4-1-fast-reasoning``, ``DeepSeek-V3``) return False, so existing
    runners keep their current routing unless they opt in.
    """
    raw = (model or "").strip()
    if not raw:
        return False
    if raw.lower().startswith(_LOCAL_PREFIXES):
        return True
    name = _normalise(raw)
    table = _alias_table()
    if name.lower() in table:
        return True
    if name in table.values():
        return True
    if _env_flag("NOT_LOCAL_ALLOW_ANY_REPO") and name.count("/") == 1:
        return True
    return False


def resolve_repo_id(model: str) -> str:
    """Map an alias (or repo id) to its canonical HF repo id."""
    name = _normalise(model)
    table = _alias_table()
    hit = table.get(name.lower())
    if hit:
        return hit
    if name in table.values():
        return name
    if _env_flag("NOT_LOCAL_ALLOW_ANY_REPO") and name.count("/") == 1:
        return name
    raise UnknownLocalModel(
        f"{model!r} is not a known local model. Known aliases: "
        f"{sorted(table)}. Add one via NOT_LOCAL_EXTRA_MODELS "
        '(e.g. \'{"my-model": "org/My-Model"}\') or set '
        "NOT_LOCAL_ALLOW_ANY_REPO=1 to pass raw repo ids through."
    )


def canonical_alias(model: str) -> str:
    """Return the short alias for a model, or the input if it has none.

    Used for cache keys so that ``qwen3-8b`` and ``Qwen/Qwen3-8B`` do not
    produce two different cache files for the same cell.
    """
    name = _normalise(model)
    table = _alias_table()
    if name.lower() in table:
        return name.lower()
    for alias, repo in table.items():
        if repo == name:
            return alias
    return name


# ---------------------------------------------------------------------------
# vLLM endpoint discovery
# ---------------------------------------------------------------------------

def _normalise_base_url(raw: str) -> str:
    """Turn ``host:port`` / ``http://host:port`` / ``.../v1/`` into a base URL."""
    url = (raw or "").strip().strip('"').strip("'")
    if not url:
        return ""
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _read_endpoint_file(path: Path) -> str:
    try:
        text = path.read_text()
    except Exception:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""


def _files_in_endpoint_dir(directory: Path) -> list[Path]:
    """Regular files in an endpoint directory, newest mtime first.

    The serve job writes one file per running server; when several are present
    (a previous job's file has not been cleaned up) the newest wins, and a
    stale entry is caught by the health probe and falls through.
    """
    try:
        entries = [p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")]
    except Exception:
        return []
    return sorted(entries, key=lambda p: p.stat().st_mtime, reverse=True)


def endpoint_file_candidates() -> list[Path]:
    """Files searched for the host:port written by the vLLM serve job.

    Order: ``NOT_VLLM_ENDPOINT_FILE`` (a file, or a directory to scan), then
    ``ANI_ENDPOINT_DIR`` (set by ``slurm/env_setup.sh``), then
    ``$ANI_OUT_DIR/vllm_endpoint.txt``, then the repo-local defaults.
    """
    paths: list[Path] = []
    explicit = os.environ.get("NOT_VLLM_ENDPOINT_FILE", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        paths.extend(_files_in_endpoint_dir(p) if p.is_dir() else [p])
    endpoint_dir = os.environ.get("ANI_ENDPOINT_DIR", "").strip()
    if endpoint_dir:
        paths.extend(_files_in_endpoint_dir(Path(endpoint_dir).expanduser()))
    out_dir = os.environ.get("ANI_OUT_DIR", "").strip()
    if out_dir:
        paths.append(Path(out_dir).expanduser() / "vllm_endpoint.txt")
    for cand in DEFAULT_ENDPOINT_FILES:
        paths.append(Path(cand).expanduser())
    seen: set[str] = set()
    unique: list[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def vllm_base_url() -> str:
    """Resolve the vLLM OpenAI-compatible base URL, or "" if undiscoverable.

    Order: ``NOT_VLLM_BASE_URL`` env, then the host:port file written by the
    Slurm serve job (``NOT_VLLM_ENDPOINT_FILE``, then the default locations).
    """
    env_url = os.environ.get("NOT_VLLM_BASE_URL", "").strip()
    if env_url:
        return _normalise_base_url(env_url)
    for path in endpoint_file_candidates():
        raw = _read_endpoint_file(path)
        if raw:
            return _normalise_base_url(raw)
    return ""


# ---------------------------------------------------------------------------
# vLLM client + health probe
# ---------------------------------------------------------------------------

_VLLM_CLIENTS: dict[str, Any] = {}
_PROBE_CACHE: dict[str, Optional[list[str]]] = {}
#: (requested backend, base url) -> resolved backend, so a fallback decision is
#: made once per configuration rather than once per call.
_BACKEND_CHOICE: dict[tuple[str, str], str] = {}
_HF_BUNDLES: dict[str, "HFBundle"] = {}
_DETERMINISM_ENABLED = False


def reset_state() -> None:
    """Drop cached clients, probes, backend choice and loaded HF weights.

    Tests call this between cases; long-running jobs can call it after a
    serve-job restart moves the endpoint to a different node.
    """
    _VLLM_CLIENTS.clear()
    _PROBE_CACHE.clear()
    _HF_BUNDLES.clear()
    _BACKEND_CHOICE.clear()


def get_vllm_client(base_url: str = ""):
    """Return a cached ``openai.OpenAI`` client bound to the vLLM server.

    ``openai`` is imported lazily so this module stays importable (and this
    function stays stubbable) with no SDK installed.
    """
    url = base_url or vllm_base_url()
    if not url:
        raise LocalEndpointUnavailable(
            "No vLLM endpoint configured. Set NOT_VLLM_BASE_URL=http://<node>:<port>/v1 "
            "or point NOT_VLLM_ENDPOINT_FILE at the host:port file written by the "
            f"serve job (searched: {[str(p) for p in endpoint_file_candidates()]})."
        )
    client = _VLLM_CLIENTS.get(url)
    if client is not None:
        return client
    from openai import OpenAI

    client = OpenAI(
        base_url=url,
        api_key=os.environ.get("NOT_VLLM_API_KEY", "EMPTY"),
        timeout=_env_float("NOT_VLLM_TIMEOUT", DEFAULT_VLLM_TIMEOUT),
        max_retries=0,  # we manage retries ourselves, as generators.py does
    )
    _VLLM_CLIENTS[url] = client
    return client


def _probe_vllm(base_url: str) -> Optional[list[str]]:
    """Return the served model ids, or None if the endpoint does not answer.

    Result is cached per base URL; ``reset_state()`` clears it.
    """
    if base_url in _PROBE_CACHE:
        return _PROBE_CACHE[base_url]
    ids: Optional[list[str]] = None
    try:
        from openai import OpenAI

        probe = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("NOT_VLLM_API_KEY", "EMPTY"),
            timeout=_env_float("NOT_VLLM_PROBE_TIMEOUT", DEFAULT_PROBE_TIMEOUT),
            max_retries=0,
        )
        listed = probe.models.list()
        ids = [getattr(m, "id", "") for m in getattr(listed, "data", []) or []]
    except Exception:
        ids = None
    _PROBE_CACHE[base_url] = ids
    return ids


def vllm_available() -> bool:
    """True if a vLLM endpoint is discoverable and answering ``GET /v1/models``."""
    url = vllm_base_url()
    if not url:
        return False
    return _probe_vllm(url) is not None


def _served_model_name(repo_id: str, base_url: str) -> str:
    """Name to send as ``model=``: the served name, not necessarily the repo id."""
    override = os.environ.get("NOT_VLLM_SERVED_NAME", "").strip()
    if override:
        return override
    served = _PROBE_CACHE.get(base_url)
    if served:
        if repo_id in served:
            return repo_id
        alias = canonical_alias(repo_id)
        for name in served:
            if name == alias or name.rsplit("/", 1)[-1] == repo_id.rsplit("/", 1)[-1]:
                return name
        if len(served) == 1:
            return served[0]
    return repo_id


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def requested_backend() -> str:
    """The engine the environment asks for, before any health probe.

    ``NOT_LOCAL_BACKEND`` is the contracted switch: ``vllm`` (default) or
    ``hf``.  ``auto`` and the empty string mean "vllm if a server answers, else
    hf" and are treated as the default.  When the switch is unset or ``auto``,
    ``ANI_LOCAL_BACKEND`` (exported already-resolved by
    ``slurm/env_setup.sh``, never ``auto``) is honoured, so a node with no vLLM
    installed skips the HTTP probe entirely.
    """
    raw = (os.environ.get("NOT_LOCAL_BACKEND", "") or "").strip().lower()
    if raw in ("", "auto"):
        resolved = (os.environ.get("ANI_LOCAL_BACKEND", "") or "").strip().lower()
        raw = resolved if resolved in ("vllm", "hf") else "vllm"
    if raw not in ("vllm", "hf"):
        raise LocalBackendError(
            f"NOT_LOCAL_BACKEND={raw!r} is not valid; use 'vllm', 'hf' or 'auto'."
        )
    return raw


def local_backend_name(*, probe: bool = True) -> str:
    """Resolve the active local backend: ``"vllm"`` or ``"hf"``.

    ``NOT_LOCAL_BACKEND`` picks the requested engine (default ``vllm``).  When
    ``vllm`` is requested and ``probe`` is True, the endpoint is health-checked
    once; if it does not answer we fall back to ``hf`` with a warning, unless
    ``NOT_LOCAL_NO_FALLBACK=1``, in which case we raise with the URLs and files
    that were searched.
    """
    requested = requested_backend()
    if requested == "hf":
        return "hf"
    if not probe:
        return "vllm"
    url = vllm_base_url()
    key = (requested, url)
    cached = _BACKEND_CHOICE.get(key)
    if cached is not None:
        return cached
    if vllm_available():
        _BACKEND_CHOICE[key] = "vllm"
        return "vllm"
    detail = (
        f"vLLM endpoint {url!r} did not answer GET /models"
        if url
        else "no vLLM endpoint was discoverable "
        f"(NOT_VLLM_BASE_URL unset; searched {[str(p) for p in endpoint_file_candidates()]})"
    )
    if _env_flag("NOT_LOCAL_NO_FALLBACK"):
        raise LocalEndpointUnavailable(
            f"{detail}. Start the vLLM serve job, or set NOT_VLLM_BASE_URL, or set "
            "NOT_LOCAL_BACKEND=hf to run transformers in-process."
        )
    print(
        f"[local_backend] WARNING: {detail}; falling back to the in-process HF "
        "backend (slower, one model per process). Set NOT_LOCAL_NO_FALLBACK=1 "
        "to make this an error instead.",
        file=sys.stderr,
        flush=True,
    )
    _BACKEND_CHOICE[key] = "hf"
    return "hf"


# ---------------------------------------------------------------------------
# Retry (mirrors scripts/generators.py behaviour)
# ---------------------------------------------------------------------------

def _retry_sleep(attempt: int, err: Exception) -> None:
    """Exponential back-off with jitter; honours Retry-After on 429."""
    if getattr(err, "status_code", None) == 429 or "429" in str(err):
        resp = getattr(err, "response", None)
        wait = 30.0
        if resp is not None:
            try:
                wait = float(resp.headers.get("Retry-After", 30))
            except Exception:
                pass
        time.sleep(wait + random.uniform(0, 2))
    else:
        time.sleep(2 ** attempt + random.uniform(0, 1))


def _looks_like_connection_error(err: Exception) -> bool:
    name = type(err).__name__.lower()
    text = str(err).lower()
    return (
        "connect" in name
        or "timeout" in name
        or "connection refused" in text
        or "failed to establish" in text
        or "connection error" in text
        or "name or service not known" in text
    )


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def build_messages(system: str, user: str) -> list[dict[str, str]]:
    """Chat messages, omitting an empty system turn (the "raw" arm sends "")."""
    messages: list[dict[str, str]] = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


# ---------------------------------------------------------------------------
# vLLM path
# ---------------------------------------------------------------------------

def _call_vllm(
    alias: str,
    repo_id: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
    stop: Optional[Iterable[str]] = None,
    json_mode: bool = False,
    chat_template_kwargs: Optional[dict] = None,
    extra_body: Optional[dict] = None,
) -> GenerationResult:
    base_url = vllm_base_url()
    client = get_vllm_client(base_url)
    served = _served_model_name(repo_id, base_url)
    messages = build_messages(system, user)

    body: dict[str, Any] = {}
    if chat_template_kwargs:
        body["chat_template_kwargs"] = dict(chat_template_kwargs)
    if temperature <= 0.0:
        # Greedy: pin the samplers vLLM would otherwise apply from its defaults.
        body["top_k"] = -1
    if extra_body:
        body.update(extra_body)

    kwargs: dict[str, Any] = dict(
        model=served,
        messages=messages,
        max_tokens=max_tokens,
        temperature=float(temperature),
    )
    if seed is not None:
        kwargs["seed"] = int(seed)
    if stop:
        kwargs["stop"] = list(stop)
    if json_mode and _env_flag("NOT_VLLM_JSON_MODE"):
        # Off by default: guided decoding is not enabled on every vLLM build,
        # and a hard 400 here would burn the whole retry budget.
        kwargs["response_format"] = {"type": "json_object"}
    if body:
        kwargs["extra_body"] = body

    attempts = _max_attempts()
    last_err: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(**kwargs)
            latency = time.monotonic() - t0
            choice = resp.choices[0]
            message = getattr(choice, "message", None)
            text = getattr(message, "content", None) or ""
            usage = getattr(resp, "usage", None)
            return GenerationResult(
                text=text,
                finish_reason=getattr(choice, "finish_reason", "") or "",
                model=alias,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                latency_s=latency,
                meta={
                    "backend": "vllm",
                    "repo_id": repo_id,
                    "served_model": served,
                    "base_url": base_url,
                    "temperature": float(temperature),
                    "seed": seed,
                    "deterministic": False,  # see module docstring
                },
            )
        except Exception as e:  # noqa: BLE001 - classified below
            last_err = e
            if attempt < attempts - 1:
                _retry_sleep(attempt, e)
    assert last_err is not None
    hint = (
        "The vLLM server is not answering. Check the serve job is still running "
        "(squeue), that the host:port file is current, and that the compute node "
        "is reachable from here."
        if _looks_like_connection_error(last_err)
        else "The vLLM server answered but the request failed."
    )
    raise LocalEndpointUnavailable(
        f"vLLM generation failed after {attempts} attempts against {base_url!r} "
        f"(model={served!r}). {hint} Last error: {last_err}"
    ) from last_err


# ---------------------------------------------------------------------------
# HF transformers path (lazy imports; nothing heavy at module scope)
# ---------------------------------------------------------------------------

def _torch():
    """Import torch lazily.  Seam: tests monkeypatch this."""
    import torch  # noqa: PLC0415 - deliberately lazy

    return torch


def _transformers():
    """Import transformers lazily.  Seam: tests monkeypatch this."""
    import transformers  # noqa: PLC0415 - deliberately lazy

    return transformers


@dataclass
class HFBundle:
    """A loaded transformers model plus its tokenizer and target device."""

    model: Any
    tokenizer: Any
    device: str
    repo_id: str
    meta: dict = field(default_factory=dict)


def _hf_device() -> str:
    want = (os.environ.get("NOT_HF_DEVICE", "auto") or "auto").strip().lower()
    if want in ("cuda", "cpu", "mps"):
        return want
    try:
        torch = _torch()
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _hf_dtype(device: str):
    torch = _torch()
    want = (os.environ.get("NOT_HF_DTYPE", "") or "").strip().lower()
    if want:
        return getattr(torch, want)
    # bf16 fits an 8B in ~16GB and preserves logprob fidelity; do NOT quantize
    # 8B models -- quantization perturbs logprobs enough to contaminate the
    # small propensity shifts this project measures.
    return torch.bfloat16 if device == "cuda" else torch.float32


def enable_hf_determinism(seed: int = 0) -> None:
    """Put torch into deterministic mode for exact-reproduction tests.

    Sets ``torch.use_deterministic_algorithms(True)``, disables cuDNN
    benchmarking, and seeds python/torch RNGs.  ``CUBLAS_WORKSPACE_CONFIG`` is
    set if unset, but CUDA reads it at context creation, so set it in the Slurm
    script for a guarantee.
    """
    global _DETERMINISM_ENABLED
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _torch()
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:  # some ops have no deterministic kernel
        print(f"[local_backend] use_deterministic_algorithms failed: {e}", file=sys.stderr)
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None) if backends is not None else None
    if cudnn is not None:
        try:
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
    random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass
    _DETERMINISM_ENABLED = True


def load_hf_model(model: str) -> HFBundle:
    """Load (and cache) a transformers model + tokenizer for ``model``.

    Heavy: one model per process.  Exposed so sibling modules (e.g. activation
    steering / logprob propensity) can reuse the same weights instead of
    loading a second copy.
    """
    repo_id = resolve_repo_id(model)
    cached = _HF_BUNDLES.get(repo_id)
    if cached is not None:
        return cached
    transformers = _transformers()
    torch = _torch()
    device = _hf_device()
    dtype = _hf_dtype(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)
    kwargs: dict[str, Any] = {"dtype": dtype}
    try:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
    except TypeError:
        # transformers < 4.56 spells it torch_dtype
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            repo_id, torch_dtype=dtype
        )
    try:
        hf_model = hf_model.to(device)
    except Exception:
        pass
    try:
        hf_model.eval()
    except Exception:
        pass
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass
    bundle = HFBundle(
        model=hf_model,
        tokenizer=tokenizer,
        device=device,
        repo_id=repo_id,
        meta={"dtype": str(dtype), "torch": getattr(torch, "__version__", "?")},
    )
    _HF_BUNDLES[repo_id] = bundle
    return bundle


def apply_chat_template(
    tokenizer: Any,
    system: str,
    user: str,
    *,
    chat_template_kwargs: Optional[dict] = None,
) -> str:
    """Render the chat messages to a prompt string using the model's template.

    Falls back to a plain ``System:/User:/Assistant:`` layout for tokenizers
    that carry no chat template (base models).
    """
    messages = build_messages(system, user)
    fn = getattr(tokenizer, "apply_chat_template", None)
    if fn is not None and getattr(tokenizer, "chat_template", None):
        kwargs = dict(chat_template_kwargs or {})
        try:
            return fn(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        except TypeError:
            return fn(messages, tokenize=False, add_generation_prompt=True)
    parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _call_hf(
    alias: str,
    repo_id: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
    top_p: float = 1.0,
    stop: Optional[Iterable[str]] = None,
    deterministic: bool = False,
    chat_template_kwargs: Optional[dict] = None,
) -> GenerationResult:
    torch = _torch()
    if deterministic and not _DETERMINISM_ENABLED:
        enable_hf_determinism(seed if seed is not None else 0)
    bundle = load_hf_model(repo_id)
    tokenizer = bundle.tokenizer
    prompt = apply_chat_template(
        tokenizer, system, user, chat_template_kwargs=chat_template_kwargs
    )

    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_tokens)}
    if temperature and temperature > 0.0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
    else:
        gen_kwargs["do_sample"] = False  # greedy
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    attempts = _max_attempts()
    last_err: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            if seed is not None:
                torch.manual_seed(int(seed))
            t0 = time.monotonic()
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            enc = {k: _to_device(v, bundle.device) for k, v in dict(enc).items()}
            input_len = int(enc["input_ids"].shape[-1])
            with torch.inference_mode():
                out = bundle.model.generate(**enc, **gen_kwargs)
            latency = time.monotonic() - t0
            sequence = out[0]
            completion_ids = sequence[input_len:]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            completion_tokens = int(len(completion_ids))
            finish_reason = "length" if completion_tokens >= int(max_tokens) else "stop"
            if stop:
                text, finish_reason = _apply_stop(text, stop, finish_reason)
            return GenerationResult(
                text=text,
                finish_reason=finish_reason,
                model=alias,
                prompt_tokens=input_len,
                completion_tokens=completion_tokens,
                latency_s=latency,
                meta={
                    "backend": "hf",
                    "repo_id": repo_id,
                    "device": bundle.device,
                    "temperature": float(temperature),
                    "seed": seed,
                    "greedy": not gen_kwargs.get("do_sample", False),
                    "deterministic": bool(deterministic),
                },
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < attempts - 1:
                _retry_sleep(attempt, e)
    assert last_err is not None
    raise LocalBackendError(
        f"HF generation failed after {attempts} attempts for {repo_id!r}: {last_err}"
    ) from last_err


def _to_device(value: Any, device: str) -> Any:
    to = getattr(value, "to", None)
    if to is None:
        return value
    try:
        return to(device)
    except Exception:
        return value


def _apply_stop(text: str, stop: Iterable[str], finish_reason: str) -> tuple[str, str]:
    cut = len(text)
    hit = False
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and idx < cut:
            cut = idx
            hit = True
    return (text[:cut], "stop") if hit else (text, finish_reason)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def local_generate(
    model: str,
    system: str,
    user: str,
    *,
    sample_idx: int,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    **kw: Any,
) -> GenerationResult:
    """Generate from an open-weights model, returning a ``GenerationResult``.

    Args:
        model: local alias (``qwen3-8b``) or canonical HF repo id.
        system: system prompt ("" for the raw arm -> no system turn).
        user: user turn text.
        sample_idx: sample index; also the default RNG seed, mirroring how
            ``scripts.generators.generate`` forwards ``sample_idx`` as ``seed``.
        max_tokens: max NEW tokens to generate.
        temperature: 0.0 (default) means greedy decoding.
        seed: explicit RNG seed; defaults to ``sample_idx``.

    Extra keyword arguments:
        backend: "vllm" | "hf" to force an engine for this call.
        deterministic: HF only; enable ``torch.use_deterministic_algorithms``.
        top_p, stop, chat_template_kwargs, extra_body, json_mode.
        Hosted-only kwargs (``reasoning_effort``) are accepted and ignored so
        existing call sites do not need editing.

    Determinism: greedy + fixed seed is bitwise reproducible only on the HF
    path with batch size 1.  See the module docstring -- vLLM's continuous
    batching changes reduction order, so exact-equality assertions must not run
    against the vLLM backend.
    """
    alias = canonical_alias(model)
    repo_id = resolve_repo_id(model)
    effective_seed = int(sample_idx) if seed is None else int(seed)

    forced = kw.pop("backend", None)
    if forced:
        backend = str(forced).strip().lower()
        if backend not in ("vllm", "hf"):
            raise LocalBackendError(f"backend={forced!r} is not valid; use 'vllm' or 'hf'.")
    else:
        backend = local_backend_name()

    deterministic = bool(kw.pop("deterministic", False))
    top_p = float(kw.pop("top_p", 1.0))
    stop = kw.pop("stop", None)
    json_mode = bool(kw.pop("json_mode", False))
    chat_template_kwargs = kw.pop("chat_template_kwargs", None)
    extra_body = kw.pop("extra_body", None)
    kw.pop("reasoning_effort", None)  # hosted-only; harmless here
    if kw:
        # Loud enough to catch a typo (temperture=0.9), soft enough that a
        # runner passing a future hosted-only kwarg still runs.
        print(
            f"[local_backend] WARNING: ignoring unrecognised kwargs {sorted(kw)} "
            f"for local model {alias!r}.",
            file=sys.stderr,
            flush=True,
        )

    if backend == "vllm":
        if deterministic:
            print(
                "[local_backend] WARNING: deterministic=True was requested but the "
                "vLLM backend is active. vLLM is not bitwise-deterministic across "
                "batch compositions; use NOT_LOCAL_BACKEND=hf for exact-reproduction "
                "checks.",
                file=sys.stderr,
                flush=True,
            )
        return _call_vllm(
            alias,
            repo_id,
            system,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=effective_seed,
            stop=stop,
            json_mode=json_mode,
            chat_template_kwargs=chat_template_kwargs,
            extra_body=extra_body,
        )
    return _call_hf(
        alias,
        repo_id,
        system,
        user,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=effective_seed,
        top_p=top_p,
        stop=stop,
        deterministic=deterministic,
        chat_template_kwargs=chat_template_kwargs,
    )


#: kwargs that only the local backends understand; stripped before a hosted call.
_LOCAL_ONLY_KWARGS = (
    "temperature",
    "seed",
    "backend",
    "deterministic",
    "top_p",
    "stop",
    "chat_template_kwargs",
    "extra_body",
)


def generate_any(model: str, system: str, user: str, **kw: Any) -> GenerationResult:
    """Route to the local backend for local aliases, else to the hosted router.

    This is the single seam that lets every existing runner work on CURC:

        from scripts.local_backend import generate_any as generate

    Hosted calls are forwarded verbatim to ``scripts.generators.generate``
    except that local-only kwargs (``temperature``, ``seed``, ``backend``, ...)
    are dropped, since the hosted router does not accept them.  ``sample_idx``
    defaults to 0 when a caller omits it.
    """
    kw.setdefault("sample_idx", 0)
    if is_local_model(model):
        return local_generate(model, system, user, **kw)
    hosted_kw = {k: v for k, v in kw.items() if k not in _LOCAL_ONLY_KWARGS}
    from scripts import generators as _generators  # late bind so tests can patch

    return _generators.generate(model, system, user, **hosted_kw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _smoke() -> int:
    """Offline self-check: no network, no GPU, no API keys."""
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'ok' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")

    print("local_backend smoke")
    check("alias table has 4 aliases", len(LOCAL_MODELS) == 4, str(sorted(LOCAL_MODELS)))
    for alias, repo in LOCAL_MODELS.items():
        check(f"{alias} -> {repo}", resolve_repo_id(alias) == repo)
        check(f"is_local_model({alias})", is_local_model(alias))
        check(f"is_local_model({repo})", is_local_model(repo))
    for hosted in ("gpt-5.4-nano", "claude-haiku-4-5", "grok-4-1-fast-reasoning", "DeepSeek-V3"):
        check(f"not is_local_model({hosted})", not is_local_model(hosted))
    check("prefixed alias routes local", is_local_model("local/qwen3-8b"))
    check("canonical_alias(repo) -> alias", canonical_alias("Qwen/Qwen3-8B") == "qwen3-8b")
    check(
        "base url normalisation",
        _normalise_base_url("node01:8000") == "http://node01:8000/v1",
        _normalise_base_url("node01:8000"),
    )
    check("messages omit empty system", len(build_messages("", "hi")) == 1)
    check("messages keep system", len(build_messages("sys", "hi")) == 2)
    url = vllm_base_url()
    print(f"  vllm base url: {url or '(none discovered)'}")
    print(f"  requested backend: {os.environ.get('NOT_LOCAL_BACKEND', 'vllm')} "
          f"(no probe performed in smoke)")
    for mod in ("torch", "vllm", "transformers"):
        check(f"{mod} not imported at module scope", mod not in sys.modules)
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Local (open-weights) generation backend")
    parser.add_argument("--smoke", action="store_true",
                        help="Offline self-check: aliases, routing, URL parsing")
    parser.add_argument("--list", action="store_true", help="Print the alias table")
    parser.add_argument("--status", action="store_true",
                        help="Probe the vLLM endpoint and report the backend that would be used")
    parser.add_argument("--model", default="", help="Alias or repo id to generate with")
    parser.add_argument("--system", default="", help="System prompt")
    parser.add_argument("--user", default="What is 2+2? Answer with a number only.",
                        help="User turn")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="", choices=["", "vllm", "hf"],
                        help="Force a backend for this call")
    args = parser.parse_args(argv)

    if args.smoke:
        return _smoke()
    if args.list:
        for alias, repo in sorted(_alias_table().items()):
            print(f"{alias:<16} {repo}")
        return 0
    if args.status:
        url = vllm_base_url()
        print(f"NOT_LOCAL_BACKEND = {os.environ.get('NOT_LOCAL_BACKEND', 'vllm')}")
        print(f"vLLM base URL     = {url or '(none discovered)'}")
        if url:
            served = _probe_vllm(url)
            print(f"served models     = {served if served is not None else '(endpoint down)'}")
        print(f"effective backend = {local_backend_name()}")
        return 0
    if not args.model:
        parser.print_help()
        return 0

    kw: dict[str, Any] = {}
    if args.backend:
        kw["backend"] = args.backend
    result = local_generate(
        args.model, args.system, args.user,
        sample_idx=0,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        **kw,
    )
    print(f"--- {result.model} ({result.meta.get('backend')}) ---")
    print(result.text)
    print(
        f"[prompt_tokens={result.prompt_tokens} completion_tokens={result.completion_tokens} "
        f"finish={result.finish_reason} latency={result.latency_s:.2f}s]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
