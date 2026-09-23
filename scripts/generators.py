"""
scripts/generators.py -- Multi-vendor generator router.

Provides a single ``generate()`` function that dispatches across five
API surfaces based on the model name:

  * Azure AI Foundry v1 OpenAI-compatible surface ({base}/openai/v1) for the
    fourth-vendor deployments Llama-3.3-70B-Instruct, Mistral-Large-3-2 and
    DeepSeek-V4-Pro (exact deployment names, matched case-insensitively
    against FOUNDRY_V1_MODELS; the env var of the same name adds more).
    Checked FIRST in generate(), because DeepSeek-V4-Pro would otherwise
    fall into the legacy _call_deepseek route below.
  * OpenAI / Azure-OpenAI-compatible (gpt-5.4-nano, gpt-4o, ...)
  * Anthropic Messages API on Azure Foundry (claude-haiku-4-5, claude-sonnet-4-6)
  * xAI OpenAI-compatible endpoint (grok-4-1-fast-reasoning)
  * DeepSeek on Azure AI Foundry unified inference (/models/chat/completions)

All surfaces return a uniform GenerationResult dataclass so the caller does
not need to know which backend was used.

The Foundry v1 route (added 2026-09-14, no call has been made through it
yet; scripts/probe_foundry.py --run is the one-call probe) serves the graph
ablation programme's fourth-vendor arrival: model lists stay config-driven,
so a new deployment is a name in a tuple plus a PRICES entry, not a code
path. Its request is deliberately plain -- chat.completions.create with
messages, max_tokens and temperature=0.7 (dropped when
FOUNDRY_V1_NO_TEMPERATURE is set) and NO seed, because the /openai/v1
surface does not promise seed support for third-party deployments. The
reasoning floor max(max_tokens, 8192) that the gpt-5 and deepseek-r branches
apply is applied here ONLY to names listed in FOUNDRY_V1_REASONING, which is
empty until the probe says which deployments burn completion tokens on
hidden reasoning (DeepSeek-V4-Pro is the candidate).

  python -m scripts.generators --selftest     # routing only; no network

Environment variables (load from .env with python-dotenv before calling):
  AZURE_AI_PROJECT_ENDPOINT -- Azure Foundry resource endpoint
  AZURE_AI_API_KEY          -- Foundry key (covers OpenAI + Anthropic surfaces)
  AZURE_AI_API_VERSION      -- optional, defaults to 2025-04-01-preview
  XAI_API_KEY               -- xAI key for Grok models
  DEEPSEEK_ENDPOINT         -- Azure Foundry unified inference URL (optional;
                               defaults to {base}/models/chat/completions)
  DEEPSEEK_API_KEY          -- api-key for DeepSeek endpoint (optional;
                               defaults to AZURE_AI_API_KEY)
  DEEPSEEK_MODEL            -- deployment name for deepseek-* judge aliases
  ANTHROPIC_ENDPOINT        -- override the Anthropic Messages URL if needed
                               (defaults to {AZURE_AI_PROJECT_ENDPOINT}/anthropic/v1/messages)
  FOUNDRY_V1_MODELS         -- comma list of extra deployment names to send
                               through the Foundry v1 route (additive to the
                               built-in tuple; case-insensitive)
  FOUNDRY_V1_SURFACE        -- "openai/v1" (default) or "models": which
                               OpenAI-compatible path under the Foundry
                               resource the v1 route posts to
  FOUNDRY_V1_NO_TEMPERATURE -- set (any non-empty value) to omit temperature
                               from Foundry v1 requests

Usage:
    from scripts.generators import generate
    result = generate("claude-haiku-4-5", system_prompt, user_prompt,
                      sample_idx=0, max_tokens=2048)
    print(result.text)

Notes:
- Reasoning models (names containing gpt-5, o1, o3, o4, grok.*reason) are
  sent with reasoning_effort="medium" instead of temperature.
- Retry logic: up to 5 attempts with exponential back-off; 429 responses
  honour the Retry-After header; content-filter errors return empty string.
- The cache-key convention from the notebook is preserved: the caller
  supplies sample_idx which is forwarded as seed for OpenAI/xAI surfaces;
  Anthropic has no seed parameter but the caller should vary user content.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Model classification helpers
# ---------------------------------------------------------------------------

_REASONING_HINTS = ("gpt-5", "o1", "o3", "o4", "reasoning")
_ANTHROPIC_PREFIXES = ("claude",)
_XAI_PREFIXES = ("grok",)
_DEEPSEEK_PREFIXES = ("deepseek",)

#: Exact Foundry deployment names served through the /openai/v1 route.
#: Matched case-insensitively; FOUNDRY_V1_MODELS (env, comma list) adds more.
#: Keep this tuple the single place a fourth-vendor deployment is named.
FOUNDRY_V1_MODELS: tuple[str, ...] = (
    "Llama-3.3-70B-Instruct",
    "Mistral-Large-3-2",
    "DeepSeek-V4-Pro",
)

#: Foundry v1 deployments that spend completion tokens on hidden reasoning
#: and therefore get the max(max_tokens, 8192) floor. EMPTY until
#: scripts/probe_foundry.py --run has shown which do; DeepSeek-V4-Pro is the
#: candidate to add after the probe. Nothing is floored by name pattern here.
FOUNDRY_V1_REASONING: tuple[str, ...] = ()

#: The two OpenAI-compatible paths under a Foundry resource the v1 route can
#: post to. "openai/v1" is the Azure OpenAI v1 surface; "models" is the
#: Azure AI Model Inference surface that _get_xai_via_azure_client uses.
FOUNDRY_V1_SURFACES = ("openai/v1", "models")


# --------------------------------------------------------------------------
# SDK client timeouts.
#
# CONFIRMED 2026-08-19 the hard way: two long grok runs wedged at ~95% with
# workers pinned at 0% CPU and no log output for 20+ minutes. None of the three
# OpenAI-SDK clients below set a timeout, so each request inherits the SDK
# default (600s) AND the SDK's own internal retries -- which then sit underneath
# this module's 5-attempt retry loop. One unresponsive request could therefore
# block a worker for hours, and enough of them stall the whole run.
#
# Fix: bound every request, and set max_retries=0 because retry/backoff is
# already this module's job (_retry_sleep). Override with NOT_HTTP_TIMEOUT.
# --------------------------------------------------------------------------
SDK_TIMEOUT = float(os.environ.get("NOT_HTTP_TIMEOUT", "180"))

def _is_reasoning(model: str) -> bool:
    m = model.lower()
    return any(h in m for h in _REASONING_HINTS) or "deepseek-r" in m


def _is_anthropic(model: str) -> bool:
    return any(model.lower().startswith(p) for p in _ANTHROPIC_PREFIXES)


def _is_xai(model: str) -> bool:
    return any(model.lower().startswith(p) for p in _XAI_PREFIXES)


def _is_deepseek(model: str) -> bool:
    return any(model.lower().startswith(p) for p in _DEEPSEEK_PREFIXES)


def _foundry_v1_models() -> tuple[str, ...]:
    """Built-in tuple plus the FOUNDRY_V1_MODELS env comma list, read at call
    time so a probe can add a name without editing this file."""
    extra = tuple(x.strip() for x in os.environ.get("FOUNDRY_V1_MODELS", "").split(",")
                  if x.strip())
    return FOUNDRY_V1_MODELS + extra


def _is_foundry_v1(model: str) -> bool:
    """Exact (case-insensitive) match on a listed deployment name. Not a
    prefix test: 'deepseek-r1' must keep routing to _call_deepseek."""
    m = (model or "").strip().lower()
    return bool(m) and m in {x.lower() for x in _foundry_v1_models()}


def _is_foundry_v1_reasoning(model: str) -> bool:
    m = (model or "").strip().lower()
    return bool(m) and m in {x.lower() for x in FOUNDRY_V1_REASONING}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    text: str
    finish_reason: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _is_content_filter(err: Exception) -> bool:
    s = str(err).lower()
    return (
        "content_filter" in s
        or "responsibleaipolicy" in s
        or "content management policy" in s
    )


def _retry_sleep(attempt: int, err: Exception) -> None:
    if getattr(err, "status_code", None) == 429 or "429" in str(err):
        resp = getattr(err, "response", None)
        wait = 30
        if resp is not None:
            try:
                wait = int(resp.headers.get("Retry-After", 30))
            except Exception:
                pass
        time.sleep(wait + random.uniform(0, 2))
    else:
        time.sleep(2 ** attempt + random.uniform(0, 1))


# ---------------------------------------------------------------------------
# OpenAI / Azure-OpenAI surface
# ---------------------------------------------------------------------------

_OPENAI_CLIENT: object = None


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    from openai import AzureOpenAI
    api_key = os.environ.get("AZURE_AI_API_KEY")
    endpoint_raw = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
    base = endpoint_raw.split("/api/projects")[0].split("/openai")[0].rstrip("/")
    if api_key:
        _OPENAI_CLIENT = AzureOpenAI(
            api_key=api_key,
            api_version=os.environ.get("AZURE_AI_API_VERSION", "2025-04-01-preview"),
            azure_endpoint=base,
            timeout=SDK_TIMEOUT,
            max_retries=0,
        )
    else:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
        project = AIProjectClient(endpoint=endpoint_raw, credential=DefaultAzureCredential())
        _OPENAI_CLIENT = project.get_openai_client(
            api_version=os.environ.get("AZURE_AI_API_VERSION", "2025-04-01-preview")
        )
    return _OPENAI_CLIENT


def _history_messages(history) -> list:
    """Prior turns as chat messages, oldest first: [(role, content), ...] with role in
    {"user", "assistant"}. None or empty -> []. Used by multi-turn probes (Addendum 17.9);
    every single-turn caller passes nothing and builds byte-identical requests."""
    if not history:
        return []
    out = []
    for role, content in history:
        if role not in ("user", "assistant"):
            raise ValueError(f"history role must be user or assistant, got {role!r}")
        out.append({"role": role, "content": content})
    return out


def _call_openai(
    model: str,
    system: str,
    user: str,
    *,
    sample_idx: int,
    max_tokens: int,
    reasoning_effort: Optional[str] = None,
    json_mode: bool = False,
    history=None,
) -> GenerationResult:
    client = _get_openai_client()
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            t0 = time.monotonic()
            messages = []
            if system and system.strip():
                messages.append({"role": "system", "content": system})
            messages.extend(_history_messages(history))
            messages.append({"role": "user", "content": user})
            kwargs: dict = dict(
                model=model,
                messages=messages,
                seed=sample_idx,
            )
            if _is_reasoning(model):
                kwargs["max_completion_tokens"] = max_tokens
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
            else:
                kwargs["max_tokens"] = max_tokens
                kwargs["temperature"] = 0.7
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            latency = time.monotonic() - t0
            choice = resp.choices[0]
            usage = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
            return GenerationResult(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason or "",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                latency_s=latency,
            )
        except Exception as e:
            last_err = e
            if _is_content_filter(e):
                return GenerationResult(text="", finish_reason="content_filter", model=model)
            _retry_sleep(attempt, e)
    raise RuntimeError(f"OpenAI generation failed after 5 attempts: {last_err}") from last_err


# ---------------------------------------------------------------------------
# Anthropic Messages surface (Azure Foundry)
# ---------------------------------------------------------------------------

def _anthropic_endpoint() -> str:
    override = os.environ.get("ANTHROPIC_ENDPOINT", "")
    if override:
        return override
    base = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "").rstrip("/")
    # Strip /api/projects/... suffix if present
    base = base.split("/api/projects")[0].rstrip("/")
    return f"{base}/anthropic/v1/messages"


_ANTHROPIC_SESSION: object = None


def _get_anthropic_session():
    """Return a long-lived requests.Session with connection pooling."""
    global _ANTHROPIC_SESSION
    if _ANTHROPIC_SESSION is not None:
        return _ANTHROPIC_SESSION
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    # pool_connections / pool_maxsize avoid per-call DNS lookups under high concurrency
    adapter = HTTPAdapter(
        pool_connections=32,
        pool_maxsize=32,
        max_retries=Retry(total=0),  # we manage retries ourselves
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _ANTHROPIC_SESSION = session
    return session


def _call_anthropic(
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
    thinking_budget: int = 0,
    history=None,
) -> GenerationResult:
    """thinking_budget > 0 enables Claude's manual extended thinking
    (Messages API `thinking: {type: enabled, budget_tokens}`), CONFIRMED
    supported on Haiku 4.5 as of 2026-09 (Anthropic docs; Opus/Sonnet 4.7+
    moved to adaptive-only and reject manual mode with a 400).  Anthropic
    requires max_tokens > budget_tokens and (per the same constraint as
    every other reasoning branch in this module) forbids setting
    temperature/top_p while thinking is enabled -- this payload never sets
    either, so that's satisfied by omission, not by an explicit check.
    """
    session = _get_anthropic_session()
    endpoint = _anthropic_endpoint()
    api_key = os.environ.get("AZURE_AI_API_KEY", "")
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    eff_tokens = max(max_tokens, thinking_budget + 1024) if thinking_budget else max_tokens
    payload = {
        "model": model,
        "max_tokens": eff_tokens,
        "messages": _history_messages(history) + [{"role": "user", "content": user}],
    }
    if thinking_budget:
        payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    if system and system.strip():
        payload["system"] = system
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            t0 = time.monotonic()
            r = session.post(endpoint, json=payload, headers=headers, timeout=300)
            latency = time.monotonic() - t0
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                time.sleep(wait + random.uniform(0, 2))
                continue
            if r.status_code >= 400:
                msg = r.text[:500]
                if any(k in msg.lower() for k in ("content_filter", "policy")):
                    return GenerationResult(text="", finish_reason="content_filter", model=model)
                raise RuntimeError(f"Anthropic HTTP {r.status_code}: {msg}")
            data = r.json()
            # With thinking enabled, content[0] is a "thinking" block, not
            # "text" -- blindly indexing [0] silently returned the model's
            # scratchpad reasoning as if it were the answer. Scan for the
            # (last) text block instead; also capture the thinking block so
            # it's inspectable, not just discarded.
            blocks = data.get("content") or []
            text = next((b.get("text", "") for b in reversed(blocks)
                        if b.get("type") == "text"), "")
            thinking_text = "".join(b.get("thinking", "") for b in blocks
                                    if b.get("type") == "thinking")
            usage = data.get("usage", {})
            return GenerationResult(
                text=text,
                finish_reason=data.get("stop_reason", ""),
                model=model,
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                latency_s=latency,
                meta={"thinking": thinking_text} if thinking_text else {},
            )
        except Exception as e:
            last_err = e
            if _is_content_filter(e):
                return GenerationResult(text="", finish_reason="content_filter", model=model)
            _retry_sleep(attempt, e)
    raise RuntimeError(f"Anthropic generation failed after 5 attempts: {last_err}") from last_err


# ---------------------------------------------------------------------------
# xAI / Grok surface
#
# Grok models may be served either directly from api.x.ai (requires a real
# xAI console key) or from an Azure AI Foundry deployment via the
# /models OpenAI-compatible endpoint.  We try the Azure Foundry path first
# because that is the configuration used in this project; if XAI_API_KEY
# differs from AZURE_AI_API_KEY the caller can override by setting
# XAI_USE_NATIVE=1 to force the api.x.ai route.
# ---------------------------------------------------------------------------

_XAI_AZURE_CLIENT: object = None


def _get_xai_via_azure_client():
    """Return an OpenAI-SDK client pointed at the Azure Foundry /models endpoint."""
    global _XAI_AZURE_CLIENT
    if _XAI_AZURE_CLIENT is not None:
        return _XAI_AZURE_CLIENT
    from openai import OpenAI
    api_key = os.environ.get("XAI_API_KEY") or os.environ.get("AZURE_AI_API_KEY")
    if not api_key:
        raise RuntimeError("Neither XAI_API_KEY nor AZURE_AI_API_KEY is set")
    endpoint_raw = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
    base = endpoint_raw.split("/api/projects")[0].rstrip("/")
    # Azure AI Foundry Models endpoint is OpenAI-compatible at {base}/models
    _XAI_AZURE_CLIENT = OpenAI(
        api_key=api_key,
        base_url=f"{base}/models",
        timeout=SDK_TIMEOUT,
        max_retries=0,
    )
    return _XAI_AZURE_CLIENT


def _call_xai(
    model: str,
    system: str,
    user: str,
    *,
    sample_idx: int,
    max_tokens: int,
    json_mode: bool = False,
    history=None,
) -> GenerationResult:
    """Call a Grok model.  Routes via Azure Foundry /models unless XAI_USE_NATIVE=1."""
    use_native = os.environ.get("XAI_USE_NATIVE", "0") == "1"
    if use_native:
        from openai import OpenAI
        xai_key = os.environ.get("XAI_API_KEY")
        if not xai_key:
            raise RuntimeError("XAI_API_KEY not set")
        client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1",
                        timeout=SDK_TIMEOUT, max_retries=0)
    else:
        client = _get_xai_via_azure_client()

    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            t0 = time.monotonic()
            messages = []
            if system and system.strip():
                messages.append({"role": "system", "content": system})
            messages.extend(_history_messages(history))
            messages.append({"role": "user", "content": user})
            kwargs: dict = dict(model=model, messages=messages)
            if _is_reasoning(model):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
                kwargs["temperature"] = 0.7
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            latency = time.monotonic() - t0
            choice = resp.choices[0]
            usage = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
            return GenerationResult(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason or "",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                latency_s=latency,
            )
        except Exception as e:
            last_err = e
            if _is_content_filter(e):
                return GenerationResult(text="", finish_reason="content_filter", model=model)
            _retry_sleep(attempt, e)
    raise RuntimeError(f"Grok generation failed after 5 attempts: {last_err}") from last_err


# ---------------------------------------------------------------------------
# Foundry v1 surface (fourth-vendor deployments: Llama / Mistral / DeepSeek-V4)
#
# One OpenAI-SDK client per surface, pointed at {base}/openai/v1 (default) or
# {base}/models (FOUNDRY_V1_SURFACE=models), where {base} is the Foundry
# resource endpoint stripped exactly as _get_openai_client strips it. The
# request shape is the plainest chat.completions.create: messages,
# max_tokens, temperature=0.7 (omitted under FOUNDRY_V1_NO_TEMPERATURE), no
# seed, optional response_format. Retry, content-filter detection and the
# GenerationResult fields mirror _call_xai.
# ---------------------------------------------------------------------------

_FOUNDRY_V1_CLIENTS: dict = {}


def _foundry_v1_surface(surface: Optional[str] = None) -> str:
    s = (surface or os.environ.get("FOUNDRY_V1_SURFACE", "") or "openai/v1").strip().strip("/")
    if s not in FOUNDRY_V1_SURFACES:
        raise ValueError(f"FOUNDRY_V1_SURFACE must be one of {FOUNDRY_V1_SURFACES}, got {s!r}")
    return s


def _foundry_v1_base_url(surface: Optional[str] = None) -> str:
    endpoint_raw = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
    base = endpoint_raw.split("/api/projects")[0].split("/openai")[0].rstrip("/")
    return f"{base}/{_foundry_v1_surface(surface)}"


def _get_foundry_v1_client(surface: Optional[str] = None):
    """Return (and cache per surface) an OpenAI-SDK client for the Foundry v1
    route. Constructing the client makes no network call."""
    key = _foundry_v1_surface(surface)
    client = _FOUNDRY_V1_CLIENTS.get(key)
    if client is not None:
        return client
    from openai import OpenAI
    api_key = os.environ.get("AZURE_AI_API_KEY")
    if not api_key:
        raise RuntimeError("AZURE_AI_API_KEY not set (Foundry v1 route)")
    client = OpenAI(
        base_url=_foundry_v1_base_url(key),
        api_key=api_key,
        timeout=SDK_TIMEOUT,
        max_retries=0,
    )
    _FOUNDRY_V1_CLIENTS[key] = client
    return client


def _foundry_v1_request_kwargs(
    model: str, system: str, user: str, *, max_tokens: int, json_mode: bool = False,
    history=None,
) -> dict:
    """The exact kwargs sent to chat.completions.create. Kept separate so the
    selftest can assert the shape (no seed; temperature present unless
    FOUNDRY_V1_NO_TEMPERATURE) without a client."""
    messages = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    messages.extend(_history_messages(history))
    messages.append({"role": "user", "content": user})
    kwargs: dict = dict(model=model, messages=messages, max_tokens=max_tokens)
    if not os.environ.get("FOUNDRY_V1_NO_TEMPERATURE", ""):
        kwargs["temperature"] = 0.7
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def _call_foundry_v1(
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
    json_mode: bool = False,
    surface: Optional[str] = None,
    history=None,
) -> GenerationResult:
    """Call a Foundry v1 deployment by its exact deployment name."""
    surf = _foundry_v1_surface(surface)
    client = _get_foundry_v1_client(surf)
    kwargs = _foundry_v1_request_kwargs(model, system, user,
                                        max_tokens=max_tokens, json_mode=json_mode,
                                        history=history)
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(**kwargs)
            latency = time.monotonic() - t0
            choice = resp.choices[0]
            usage = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
            return GenerationResult(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason or "",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_s=latency,
                meta={"surface": surf, "deployment": model},
            )
        except Exception as e:
            last_err = e
            if _is_content_filter(e):
                return GenerationResult(text="", finish_reason="content_filter", model=model,
                                        meta={"surface": surf, "deployment": model})
            _retry_sleep(attempt, e)
    raise RuntimeError(f"Foundry v1 generation failed after 5 attempts "
                       f"({surf}, {model}): {last_err}") from last_err


# ---------------------------------------------------------------------------
# DeepSeek surface (Azure AI Foundry unified inference)
# ---------------------------------------------------------------------------

_DEEPSEEK_SESSION: object = None
_THINKING_RE = re.compile(
    r"<(?:redacted_thinking|think)>.*?</(?:redacted_thinking|think)>",
    re.DOTALL | re.IGNORECASE,
)


def _deepseek_endpoint() -> str:
    override = os.environ.get("DEEPSEEK_ENDPOINT", "").strip()
    if override:
        return override.split("?")[0]
    endpoint_raw = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "").rstrip("/")
    base = endpoint_raw.split("/api/projects")[0].rstrip("/")
    return f"{base}/models/chat/completions"


def _deepseek_api_version() -> str:
    override = os.environ.get("DEEPSEEK_ENDPOINT", "")
    if "api-version=" in override:
        return override.split("api-version=")[-1].split("&")[0]
    return os.environ.get("DEEPSEEK_API_VERSION", "2024-05-01-preview")


def _deepseek_deployment(model: str) -> str:
    """Map logical judge alias to Azure deployment name."""
    explicit = os.environ.get("DEEPSEEK_MODEL", "").strip()
    if explicit:
        return explicit
    m = model.lower()
    if "v3" in m:
        return "DeepSeek-V3"
    if "r1" in m:
        return "DeepSeek-R1"
    return explicit or "DeepSeek-V3"


def _get_deepseek_session():
    global _DEEPSEEK_SESSION
    if _DEEPSEEK_SESSION is not None:
        return _DEEPSEEK_SESSION
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=Retry(total=0))
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _DEEPSEEK_SESSION = session
    return session


def _strip_deepseek_reasoning(text: str) -> str:
    if not text:
        return ""
    cleaned = _THINKING_RE.sub("", text).strip()
    return cleaned


def _call_deepseek(
    model: str,
    system: str,
    user: str,
    *,
    sample_idx: int,
    max_tokens: int,
    json_mode: bool = False,
) -> GenerationResult:
    session = _get_deepseek_session()
    endpoint = _deepseek_endpoint()
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("AZURE_AI_API_KEY", "")
    deployment = _deepseek_deployment(model)
    params = {"api-version": _deepseek_api_version()}
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    messages = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload: dict = {
        "model": deployment,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "seed": sample_idx,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            t0 = time.monotonic()
            r = session.post(
                endpoint, params=params, json=payload, headers=headers, timeout=300,
            )
            latency = time.monotonic() - t0
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                time.sleep(wait + random.uniform(0, 2))
                continue
            if r.status_code >= 400:
                msg = r.text[:500]
                if any(k in msg.lower() for k in ("content_filter", "policy")):
                    return GenerationResult(text="", finish_reason="content_filter", model=model)
                raise RuntimeError(f"DeepSeek HTTP {r.status_code}: {msg}")
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            raw = (choice.get("message") or {}).get("content") or ""
            text = _strip_deepseek_reasoning(raw)
            usage = data.get("usage") or {}
            return GenerationResult(
                text=text,
                finish_reason=choice.get("finish_reason", ""),
                model=model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_s=latency,
                meta={"deployment": deployment},
            )
        except Exception as e:
            last_err = e
            if _is_content_filter(e):
                return GenerationResult(text="", finish_reason="content_filter", model=model)
            _retry_sleep(attempt, e)
    raise RuntimeError(f"DeepSeek generation failed after 5 attempts: {last_err}") from last_err


# ---------------------------------------------------------------------------
# Public dispatch interface
# ---------------------------------------------------------------------------

def generate(
    model: str,
    system: str,
    user: str,
    *,
    sample_idx: int,
    max_tokens: int = 4096,
    reasoning_effort: Optional[str] = "medium",
    json_mode: bool = False,
    thinking_budget: int = 0,
    history=None,
) -> GenerationResult:
    """Single entry point for all model families.

    history: optional prior turns [(role, content), ...] placed between the
    system prompt and `user` (multi-turn probes). Supported on the OpenAI,
    Anthropic, xAI and Foundry v1 routes; the legacy DeepSeek route refuses it.

    Args:
        model: model name string (determines routing).
        system: system prompt text.
        user: user turn text.
        sample_idx: used as RNG seed for OpenAI/xAI surfaces; also
            identifies this call in cache key construction.
        max_tokens: maximum output tokens (completion only).
        reasoning_effort: passed to reasoning models; ignored otherwise.
        json_mode: request JSON output format (OpenAI/xAI only).
        thinking_budget: Anthropic manual extended-thinking token budget;
            0 disables it. Ignored for non-Anthropic models.
    """
    # Foundry v1 deployments are matched by exact name and checked FIRST:
    # DeepSeek-V4-Pro starts with "deepseek" and would otherwise be sent to
    # the legacy _call_deepseek route (and its DeepSeek-V3 deployment map).
    if _is_foundry_v1(model):
        eff_tokens = max(max_tokens, 8192) if _is_foundry_v1_reasoning(model) else max_tokens
        return _call_foundry_v1(
            model, system, user,
            max_tokens=eff_tokens,
            json_mode=json_mode,
            history=history,
        )
    if _is_anthropic(model):
        return _call_anthropic(model, system, user, max_tokens=max_tokens,
                               thinking_budget=thinking_budget, history=history)
    if _is_deepseek(model):
        if history:
            raise NotImplementedError("history is not supported on the legacy DeepSeek route")
        eff_tokens = max(max_tokens, 8192) if _is_reasoning(model) else max_tokens
        return _call_deepseek(
            model, system, user,
            sample_idx=sample_idx,
            max_tokens=eff_tokens,
            json_mode=json_mode,
        )
    if _is_xai(model):
        return _call_xai(
            model, system, user,
            sample_idx=sample_idx,
            max_tokens=max_tokens,
            json_mode=json_mode,
            history=history,
        )
    # Default: OpenAI / Azure-Foundry-OpenAI.  CONFIRMED 2026-08-21: reasoning
    # models on this branch (gpt-5 family) consume completion tokens for
    # internal reasoning before any visible text, and `reasoning_effort` is
    # passed here whenever `_is_reasoning(model)` -- so an un-floored
    # max_tokens can be exhausted entirely by reasoning with zero visible
    # output. The deepseek branch above already floors at 8192 for exactly
    # this reason; this branch silently didn't, and a real run against
    # gpt-5.4-nano at max_tokens=2048 came back 77.5% genuinely empty (186/240
    # calls) as a direct result -- not truncated, not NOVERDICT-with-content,
    # completely empty completions with reasoning_effort quietly eating the
    # entire budget. Mirrors the deepseek floor rather than inventing a new one.
    eff_tokens = max(max_tokens, 8192) if _is_reasoning(model) else max_tokens
    return _call_openai(
        model, system, user,
        sample_idx=sample_idx,
        max_tokens=eff_tokens,
        reasoning_effort=reasoning_effort if _is_reasoning(model) else None,
        json_mode=json_mode,
        history=history,
    )


# ---------------------------------------------------------------------------
# Convenience: judge/extractor calls (Anthropic-only path, as in notebook)
# ---------------------------------------------------------------------------

def call_judge(
    judge_model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 1024,
    sample_idx: int = 0,
) -> str:
    """Call a judge model and return raw text.  Accepts any model family."""
    result = generate(
        judge_model, system, user,
        sample_idx=sample_idx,
        max_tokens=max_tokens,
        json_mode=True,
    )
    return result.text


# ---------------------------------------------------------------------------
# Selftest (routing only; no client is constructed, no network)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import sys as _sys

    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")

    print("generators selftest (routing predicates and dispatch order; no network)\n")
    me = _sys.modules[__name__]
    new_names = ("Llama-3.3-70B-Instruct", "Mistral-Large-3-2", "DeepSeek-V4-Pro")
    old_names = ("claude-haiku-4-5", "claude-sonnet-4-6", "gpt-5.4-nano", "gpt-4o",
                 "grok-4-1-fast-reasoning", "DeepSeek-V3", "deepseek-r1", "DeepSeek-R1")

    # -- 1. the three names route to the Foundry v1 surface, case-insensitively
    for n in new_names:
        check(f"_is_foundry_v1({n!r})", _is_foundry_v1(n))
        check(f"_is_foundry_v1({n.lower()!r}) (case-insensitive)", _is_foundry_v1(n.lower()))
        check(f"_is_foundry_v1({n.upper()!r}) (case-insensitive)", _is_foundry_v1(n.upper()))
    check("FOUNDRY_V1_MODELS holds exactly the three deployment names",
          FOUNDRY_V1_MODELS == new_names, str(FOUNDRY_V1_MODELS))
    check("no existing name is captured by the Foundry v1 predicate",
          not any(_is_foundry_v1(n) for n in old_names), str(old_names))
    check("the predicate is exact, not prefix (a DeepSeek-V4-Pro-x variant is not routed)",
          not _is_foundry_v1("DeepSeek-V4-Pro-x") and not _is_foundry_v1("DeepSeek-V4")
          and not _is_foundry_v1(""))

    # -- 2. existing predicates unchanged ---------------------------------
    check("claude-* -> anthropic", _is_anthropic("claude-haiku-4-5") and _is_anthropic("claude-sonnet-4-6"))
    check("grok-* -> xai", _is_xai("grok-4-1-fast-reasoning"))
    check("deepseek-* -> deepseek (legacy route)", _is_deepseek("DeepSeek-V3") and _is_deepseek("deepseek-r1"))
    check("DeepSeek-V4-Pro ALSO satisfies _is_deepseek (which is why dispatch order matters)",
          _is_deepseek("DeepSeek-V4-Pro"))
    check("gpt-5.4-nano is reasoning; gpt-4o is not",
          _is_reasoning("gpt-5.4-nano") and not _is_reasoning("gpt-4o"))
    check("none of the three new names is anthropic/xai; only DeepSeek-V4-Pro is deepseek-prefixed",
          not any(_is_anthropic(n) or _is_xai(n) for n in new_names)
          and [_is_deepseek(n) for n in new_names] == [False, False, True])
    check("no new name is floored by _is_reasoning (no 'reasoning'/'gpt-5' hint)",
          not any(_is_reasoning(n) for n in new_names))

    # -- 3. FOUNDRY_V1_MODELS env var is additive and case-insensitive -----
    saved = os.environ.get("FOUNDRY_V1_MODELS")
    try:
        os.environ["FOUNDRY_V1_MODELS"] = " Foo-Bar-7B , , baz-instruct "
        check("env FOUNDRY_V1_MODELS adds names (whitespace and empties tolerated)",
              _is_foundry_v1("foo-bar-7b") and _is_foundry_v1("BAZ-INSTRUCT")
              and _foundry_v1_models() == new_names + ("Foo-Bar-7B", "baz-instruct"),
              str(_foundry_v1_models()))
        check("built-in names still route with the env var set", _is_foundry_v1("Mistral-Large-3-2"))
        os.environ["FOUNDRY_V1_MODELS"] = ""
        check("empty env var adds nothing", _foundry_v1_models() == new_names)
    finally:
        if saved is None:
            os.environ.pop("FOUNDRY_V1_MODELS", None)
        else:
            os.environ["FOUNDRY_V1_MODELS"] = saved

    # -- 4. is_local_model is False for the three names ---------------------
    try:
        from scripts.local_backend import is_local_model  # noqa: WPS433 (lazy: circular at module scope)
        for n in new_names:
            check(f"is_local_model({n!r}) is False", not is_local_model(n))
        check("the local alias llama-3.3-70b is still local (unchanged)", is_local_model("llama-3.3-70b"))
        check("the local alias llama-3.3-70b is NOT a Foundry v1 name", not _is_foundry_v1("llama-3.3-70b"))
    except Exception as e:  # noqa: BLE001
        check("scripts.local_backend importable", False, repr(e))

    # -- 5. surface selection and base URL (string only; no client) -------
    saved_env = {k: os.environ.get(k) for k in
                 ("FOUNDRY_V1_SURFACE", "AZURE_AI_PROJECT_ENDPOINT", "FOUNDRY_V1_NO_TEMPERATURE")}
    try:
        os.environ["AZURE_AI_PROJECT_ENDPOINT"] = "https://example.services.ai.azure.com/api/projects/proj"
        os.environ.pop("FOUNDRY_V1_SURFACE", None)
        check("default surface is openai/v1", _foundry_v1_surface() == "openai/v1")
        check("base url strips /api/projects/... exactly as _get_openai_client does",
              _foundry_v1_base_url() == "https://example.services.ai.azure.com/openai/v1",
              _foundry_v1_base_url())
        os.environ["FOUNDRY_V1_SURFACE"] = "models"
        check("FOUNDRY_V1_SURFACE=models selects the /models path",
              _foundry_v1_base_url() == "https://example.services.ai.azure.com/models",
              _foundry_v1_base_url())
        check("an explicit surface argument overrides the env var",
              _foundry_v1_base_url("openai/v1").endswith("/openai/v1"))
        os.environ["FOUNDRY_V1_SURFACE"] = "bogus"
        try:
            _foundry_v1_surface()
            check("an unknown surface is refused", False)
        except ValueError:
            check("an unknown surface is refused", True)
        os.environ.pop("FOUNDRY_V1_SURFACE", None)
        os.environ["AZURE_AI_PROJECT_ENDPOINT"] = "https://example.openai.azure.com/openai/"
        check("a /openai suffix on the endpoint is stripped too",
              _foundry_v1_base_url() == "https://example.openai.azure.com/openai/v1",
              _foundry_v1_base_url())

        # -- 6. request shape: no seed; temperature 0.7 unless disabled ----
        os.environ.pop("FOUNDRY_V1_NO_TEMPERATURE", None)
        kw = _foundry_v1_request_kwargs("DeepSeek-V4-Pro", "sys", "usr", max_tokens=64)
        check("request carries model/messages/max_tokens/temperature=0.7 and NO seed",
              kw["model"] == "DeepSeek-V4-Pro" and kw["max_tokens"] == 64
              and kw.get("temperature") == 0.7 and "seed" not in kw
              and "max_completion_tokens" not in kw
              and [m["role"] for m in kw["messages"]] == ["system", "user"],
              str(sorted(kw)))
        kw2 = _foundry_v1_request_kwargs("DeepSeek-V4-Pro", "", "usr", max_tokens=64, json_mode=True)
        check("empty system prompt is omitted; json_mode sets response_format",
              [m["role"] for m in kw2["messages"]] == ["user"]
              and kw2.get("response_format") == {"type": "json_object"})
        os.environ["FOUNDRY_V1_NO_TEMPERATURE"] = "1"
        kw3 = _foundry_v1_request_kwargs("DeepSeek-V4-Pro", "sys", "usr", max_tokens=64)
        check("FOUNDRY_V1_NO_TEMPERATURE drops temperature", "temperature" not in kw3)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # -- 7. dispatch order in generate(), with every backend stubbed -------
    hits: list = []

    def _stub(name):
        def f(model, system, user, **kw):
            hits.append((name, model, kw.get("max_tokens")))
            return GenerationResult(text="stub", finish_reason="stop", model=model)
        return f

    real = {n: getattr(me, n) for n in
            ("_call_foundry_v1", "_call_deepseek", "_call_openai", "_call_xai", "_call_anthropic")}
    real_reasoning = FOUNDRY_V1_REASONING
    try:
        for n in real:
            setattr(me, n, _stub(n))
        generate("DeepSeek-V4-Pro", "s", "u", sample_idx=0, max_tokens=64)
        check("generate('DeepSeek-V4-Pro') dispatches to _call_foundry_v1, NOT _call_deepseek",
              hits[-1][0] == "_call_foundry_v1", str(hits[-1]))
        check("no reasoning floor by default (FOUNDRY_V1_REASONING is empty): max_tokens 64 stays 64",
              hits[-1][2] == 64 and FOUNDRY_V1_REASONING == (), str(hits[-1]))
        generate("Llama-3.3-70B-Instruct", "s", "u", sample_idx=0, max_tokens=64)
        check("generate('Llama-3.3-70B-Instruct') -> _call_foundry_v1", hits[-1][0] == "_call_foundry_v1")
        generate("Mistral-Large-3-2", "s", "u", sample_idx=0, max_tokens=64)
        check("generate('Mistral-Large-3-2') -> _call_foundry_v1", hits[-1][0] == "_call_foundry_v1")
        generate("DeepSeek-V3", "s", "u", sample_idx=0, max_tokens=64)
        check("generate('DeepSeek-V3') still -> _call_deepseek", hits[-1][0] == "_call_deepseek", str(hits[-1]))
        generate("deepseek-r1", "s", "u", sample_idx=0, max_tokens=64)
        check("generate('deepseek-r1') still -> _call_deepseek with the 8192 floor",
              hits[-1][0] == "_call_deepseek" and hits[-1][2] == 8192, str(hits[-1]))
        generate("claude-haiku-4-5", "s", "u", sample_idx=0, max_tokens=64)
        check("generate('claude-haiku-4-5') still -> _call_anthropic", hits[-1][0] == "_call_anthropic")
        generate("grok-4-1-fast-reasoning", "s", "u", sample_idx=0, max_tokens=64)
        check("generate('grok-4-1-fast-reasoning') still -> _call_xai", hits[-1][0] == "_call_xai")
        generate("gpt-4o", "s", "u", sample_idx=0, max_tokens=64)
        check("generate('gpt-4o') still -> _call_openai, unfloored", hits[-1][0] == "_call_openai" and hits[-1][2] == 64)
        generate("gpt-5.4-nano", "s", "u", sample_idx=0, max_tokens=64)
        check("generate('gpt-5.4-nano') still -> _call_openai with the 8192 floor",
              hits[-1][0] == "_call_openai" and hits[-1][2] == 8192)
        me.FOUNDRY_V1_REASONING = ("DeepSeek-V4-Pro",)
        generate("deepseek-v4-pro", "s", "u", sample_idx=0, max_tokens=64)
        check("listing a name in FOUNDRY_V1_REASONING applies max(max_tokens, 8192) (case-insensitive)",
              hits[-1][0] == "_call_foundry_v1" and hits[-1][2] == 8192, str(hits[-1]))
        generate("Llama-3.3-70B-Instruct", "s", "u", sample_idx=0, max_tokens=64)
        check("an unlisted Foundry v1 name stays unfloored", hits[-1][2] == 64)
    finally:
        me.FOUNDRY_V1_REASONING = real_reasoning
        for n, f in real.items():
            setattr(me, n, f)
    check("backends restored after the dispatch checks",
          all(getattr(me, n) is f for n, f in real.items()))
    check("no Foundry v1 client was constructed during the selftest", not _FOUNDRY_V1_CLIENTS)

    print("\n" + ("ALL OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Multi-vendor generator router (library; "
                                             "--selftest checks routing offline)")
    ap.add_argument("--selftest", action="store_true",
                    help="Routing predicates and dispatch order with every backend stubbed. No network.")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
