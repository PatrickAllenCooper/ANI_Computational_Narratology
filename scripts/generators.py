"""
scripts/generators.py -- Multi-vendor generator router.

Provides a single ``generate()`` function that dispatches across three
API surfaces based on the model name:

  * OpenAI / Azure-OpenAI-compatible (gpt-5.4-nano, gpt-4o, ...)
  * Anthropic Messages API on Azure Foundry (claude-haiku-4-5, claude-sonnet-4-6)
  * xAI OpenAI-compatible endpoint (grok-4-1-fast-reasoning)

All three surfaces return a uniform GenerationResult dataclass so the
caller does not need to know which backend was used.

Environment variables (load from .env with python-dotenv before calling):
  AZURE_AI_PROJECT_ENDPOINT -- Azure Foundry resource endpoint
  AZURE_AI_API_KEY          -- Foundry key (covers OpenAI + Anthropic surfaces)
  AZURE_AI_API_VERSION      -- optional, defaults to 2025-04-01-preview
  XAI_API_KEY               -- xAI key for Grok models
  ANTHROPIC_ENDPOINT        -- override the Anthropic Messages URL if needed
                               (defaults to {AZURE_AI_PROJECT_ENDPOINT}/anthropic/v1/messages)

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


def _is_reasoning(model: str) -> bool:
    return any(h in model.lower() for h in _REASONING_HINTS)


def _is_anthropic(model: str) -> bool:
    return any(model.lower().startswith(p) for p in _ANTHROPIC_PREFIXES)


def _is_xai(model: str) -> bool:
    return any(model.lower().startswith(p) for p in _XAI_PREFIXES)


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
        )
    else:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
        project = AIProjectClient(endpoint=endpoint_raw, credential=DefaultAzureCredential())
        _OPENAI_CLIENT = project.get_openai_client(
            api_version=os.environ.get("AZURE_AI_API_VERSION", "2025-04-01-preview")
        )
    return _OPENAI_CLIENT


def _call_openai(
    model: str,
    system: str,
    user: str,
    *,
    sample_idx: int,
    max_tokens: int,
    reasoning_effort: Optional[str] = None,
    json_mode: bool = False,
) -> GenerationResult:
    client = _get_openai_client()
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            t0 = time.monotonic()
            kwargs: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
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
) -> GenerationResult:
    session = _get_anthropic_session()
    endpoint = _anthropic_endpoint()
    api_key = os.environ.get("AZURE_AI_API_KEY", "")
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
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
            text = data.get("content", [{}])[0].get("text", "") if data.get("content") else ""
            usage = data.get("usage", {})
            return GenerationResult(
                text=text,
                finish_reason=data.get("stop_reason", ""),
                model=model,
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                latency_s=latency,
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
) -> GenerationResult:
    """Call a Grok model.  Routes via Azure Foundry /models unless XAI_USE_NATIVE=1."""
    use_native = os.environ.get("XAI_USE_NATIVE", "0") == "1"
    if use_native:
        from openai import OpenAI
        xai_key = os.environ.get("XAI_API_KEY")
        if not xai_key:
            raise RuntimeError("XAI_API_KEY not set")
        client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")
    else:
        client = _get_xai_via_azure_client()

    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            t0 = time.monotonic()
            kwargs: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
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
) -> GenerationResult:
    """Single entry point for all model families.

    Args:
        model: model name string (determines routing).
        system: system prompt text.
        user: user turn text.
        sample_idx: used as RNG seed for OpenAI/xAI surfaces; also
            identifies this call in cache key construction.
        max_tokens: maximum output tokens (completion only).
        reasoning_effort: passed to reasoning models; ignored otherwise.
        json_mode: request JSON output format (OpenAI/xAI only).
    """
    if _is_anthropic(model):
        return _call_anthropic(model, system, user, max_tokens=max_tokens)
    if _is_xai(model):
        return _call_xai(
            model, system, user,
            sample_idx=sample_idx,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    # Default: OpenAI / Azure-Foundry-OpenAI
    return _call_openai(
        model, system, user,
        sample_idx=sample_idx,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort if _is_reasoning(model) else None,
        json_mode=json_mode,
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
