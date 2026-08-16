"""
scripts/test_local_backend.py -- Offline unit tests for the local backend.

Runs with no GPU, no network, no API keys and no ML stack: the HTTP layer and
the torch/transformers layer are both stubbed through the seams
``local_backend.get_vllm_client``, ``local_backend._probe_vllm``,
``local_backend._torch`` and ``local_backend.load_hf_model``.

Usage:
  python -m pytest scripts/test_local_backend.py -q     # if pytest installed
  python -m scripts.test_local_backend                  # standalone, no pytest
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Optional

from scripts.generators import GenerationResult
from scripts import local_backend as lb

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every env var this module reads; cleared before each case so a developer's
# real .env / shell cannot make a test pass or fail spuriously.
_MANAGED_ENV = (
    "NOT_LOCAL_BACKEND",
    "ANI_LOCAL_BACKEND",
    "ANI_ENDPOINT_DIR",
    "ANI_OUT_DIR",
    "NOT_LOCAL_NO_FALLBACK",
    "NOT_LOCAL_MAX_ATTEMPTS",
    "NOT_LOCAL_EXTRA_MODELS",
    "NOT_LOCAL_ALLOW_ANY_REPO",
    "NOT_VLLM_BASE_URL",
    "NOT_VLLM_ENDPOINT_FILE",
    "NOT_VLLM_API_KEY",
    "NOT_VLLM_SERVED_NAME",
    "NOT_VLLM_JSON_MODE",
    "NOT_VLLM_TIMEOUT",
    "NOT_VLLM_PROBE_TIMEOUT",
    "NOT_HF_DEVICE",
    "NOT_HF_DTYPE",
)


# ---------------------------------------------------------------------------
# Test helpers (no pytest fixtures, so the module also runs standalone)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def env(**overrides: Optional[str]):
    """Clear all managed env vars, apply overrides, restore afterwards."""
    saved = {k: os.environ.get(k) for k in set(_MANAGED_ENV) | set(overrides)}
    for k in _MANAGED_ENV:
        os.environ.pop(k, None)
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    lb.reset_state()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        lb.reset_state()


@contextlib.contextmanager
def patch(obj: Any, name: str, value: Any):
    """Temporarily replace an attribute (stand-in for pytest's monkeypatch)."""
    missing = object()
    old = getattr(obj, name, missing)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if old is missing:
            delattr(obj, name)
        else:
            setattr(obj, name, old)


@contextlib.contextmanager
def no_sleep():
    """Neutralise retry back-off so failure paths run instantly."""
    with patch(lb.time, "sleep", lambda *_a, **_k: None):
        yield


# --- vLLM stubs ------------------------------------------------------------

class _StubUsage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = _StubMessage(content)
        self.finish_reason = finish_reason


class _StubResponse:
    def __init__(self, content: str, finish_reason: str, p: int, c: int) -> None:
        self.choices = [_StubChoice(content, finish_reason)]
        self.usage = _StubUsage(p, c)


class StubCompletions:
    """Records the kwargs it was called with; optionally fails N times first."""

    def __init__(self, text: str = "stub answer", fail_times: int = 0,
                 error: Optional[Exception] = None) -> None:
        self.text = text
        self.fail_times = fail_times
        self.error = error or ConnectionError("Connection refused")
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        if len(self.calls) <= self.fail_times:
            raise self.error
        return _StubResponse(self.text, "stop", 11, 7)


class StubVLLMClient:
    def __init__(self, completions: StubCompletions) -> None:
        self.chat = types.SimpleNamespace(completions=completions)


# --- HF stubs --------------------------------------------------------------

class _StubTensor:
    """Minimal stand-in for a torch tensor of shape (1, n)."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.ids))

    def to(self, _device: str) -> "_StubTensor":
        return self

    def __getitem__(self, key: Any) -> Any:
        return self.ids[key]


class StubTokenizer:
    chat_template = "{{ messages }}"
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, prompt_ids: int = 3, completion: str = "hf stub answer") -> None:
        self.prompt_ids = prompt_ids
        self.completion = completion
        self.template_calls: list[dict] = []
        self.decoded: list[Any] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        self.template_calls.append(
            {"messages": messages, "tokenize": tokenize,
             "add_generation_prompt": add_generation_prompt, "kw": kw}
        )
        return "PROMPT:" + "|".join(m["role"] for m in messages)

    def __call__(self, prompt: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        return {
            "input_ids": _StubTensor(list(range(self.prompt_ids))),
            "attention_mask": _StubTensor([1] * self.prompt_ids),
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        self.decoded.append(list(ids))
        return self.completion


class StubHFModel:
    def __init__(self, prompt_ids: int = 3, new_tokens: int = 5) -> None:
        self.prompt_ids = prompt_ids
        self.new_tokens = new_tokens
        self.calls: list[dict] = []

    def generate(self, **kwargs: Any):
        self.calls.append(kwargs)
        total = self.prompt_ids + self.new_tokens
        return [list(range(total))]


def stub_torch(seeds: list[int], flags: Optional[dict] = None) -> types.SimpleNamespace:
    """A torch stand-in exposing only what ``_call_hf`` touches."""
    flags = {} if flags is None else flags

    @contextlib.contextmanager
    def inference_mode():
        flags["inference_mode"] = True
        yield

    return types.SimpleNamespace(
        inference_mode=inference_mode,
        manual_seed=seeds.append,
        use_deterministic_algorithms=lambda v: flags.__setitem__("deterministic", v),
        backends=types.SimpleNamespace(
            cudnn=types.SimpleNamespace(deterministic=False, benchmark=True)
        ),
        __version__="stub",
    )


def stub_bundle(prompt_ids: int = 3, new_tokens: int = 5,
                completion: str = "hf stub answer") -> lb.HFBundle:
    return lb.HFBundle(
        model=StubHFModel(prompt_ids=prompt_ids, new_tokens=new_tokens),
        tokenizer=StubTokenizer(prompt_ids=prompt_ids, completion=completion),
        device="cpu",
        repo_id="Qwen/Qwen3-8B",
    )


# ---------------------------------------------------------------------------
# Alias table / routing
# ---------------------------------------------------------------------------

def test_alias_table_contract() -> None:
    with env():
        assert lb.LOCAL_MODELS == {
            "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
            "qwen3-8b": "Qwen/Qwen3-8B",
            "qwen3-32b": "Qwen/Qwen3-32B",
            "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        }
        for alias, repo in lb.LOCAL_MODELS.items():
            assert repo.count("/") == 1, (alias, repo)
            assert lb.resolve_repo_id(alias) == repo


def test_is_local_model_routing() -> None:
    with env():
        for alias in lb.LOCAL_MODELS:
            assert lb.is_local_model(alias)
            assert lb.is_local_model(alias.upper())
            assert lb.is_local_model("local/" + alias)
            assert lb.is_local_model(lb.LOCAL_MODELS[alias])
        # Hosted quartet must keep routing to scripts.generators.
        for hosted in ("gpt-5.4-nano", "claude-haiku-4-5", "claude-sonnet-4-6",
                       "grok-4-1-fast-reasoning", "DeepSeek-V3", "gpt-4o", ""):
            assert not lb.is_local_model(hosted), hosted
        # A bare repo id that is not in the table stays hosted-routed by default.
        assert not lb.is_local_model("mistralai/Mistral-7B-Instruct-v0.3")
    with env(NOT_LOCAL_ALLOW_ANY_REPO="1"):
        assert lb.is_local_model("mistralai/Mistral-7B-Instruct-v0.3")
        assert lb.resolve_repo_id("mistralai/Mistral-7B-Instruct-v0.3") == \
            "mistralai/Mistral-7B-Instruct-v0.3"
        assert not lb.is_local_model("gpt-5.4-nano")


def test_alias_resolution_and_extras() -> None:
    with env():
        assert lb.canonical_alias("Qwen/Qwen3-8B") == "qwen3-8b"
        assert lb.canonical_alias("QWEN3-8B") == "qwen3-8b"
        assert lb.canonical_alias("local/qwen3-32b") == "qwen3-32b"
        assert lb.canonical_alias("gpt-5.4-nano") == "gpt-5.4-nano"
        try:
            lb.resolve_repo_id("no-such-model")
        except lb.UnknownLocalModel as e:
            assert "qwen3-8b" in str(e)
        else:
            raise AssertionError("expected UnknownLocalModel")
    with env(NOT_LOCAL_EXTRA_MODELS='{"olmo-7b": "allenai/OLMo-7B-Instruct"}'):
        assert lb.is_local_model("olmo-7b")
        assert lb.resolve_repo_id("olmo-7b") == "allenai/OLMo-7B-Instruct"
        assert lb.is_local_model("qwen3-8b"), "extras must not displace built-ins"


def test_build_messages_omits_empty_system() -> None:
    # PROMPTS["raw"] is "" -- the raw arm must not gain a system turn.
    assert lb.build_messages("", "hi") == [{"role": "user", "content": "hi"}]
    assert lb.build_messages("   ", "hi") == [{"role": "user", "content": "hi"}]
    assert lb.build_messages("sys", "hi")[0] == {"role": "system", "content": "sys"}


# ---------------------------------------------------------------------------
# Endpoint discovery
# ---------------------------------------------------------------------------

def test_base_url_from_env_and_normalisation() -> None:
    for raw, expected in [
        ("http://node01:8000/v1", "http://node01:8000/v1"),
        ("http://node01:8000", "http://node01:8000/v1"),
        ("node01:8000", "http://node01:8000/v1"),
        ("http://node01:8000/v1/", "http://node01:8000/v1"),
        ("https://node01:8443", "https://node01:8443/v1"),
    ]:
        with env(NOT_VLLM_BASE_URL=raw):
            assert lb.vllm_base_url() == expected, raw


def test_base_url_from_serve_job_file() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "vllm_endpoint.txt"
        path.write_text("# written by serve job\nc3gpu-c2-u17:8000\n")
        with env(NOT_VLLM_ENDPOINT_FILE=str(path)):
            assert lb.vllm_base_url() == "http://c3gpu-c2-u17:8000/v1"
        # env var beats the file
        with env(NOT_VLLM_ENDPOINT_FILE=str(path),
                 NOT_VLLM_BASE_URL="http://other:9000/v1"):
            assert lb.vllm_base_url() == "http://other:9000/v1"
        missing = Path(td) / "nope.txt"
        with env(NOT_VLLM_ENDPOINT_FILE=str(missing)):
            assert lb.vllm_base_url() == ""


def test_base_url_from_endpoint_directory() -> None:
    """slurm/env_setup.sh exports ANI_ENDPOINT_DIR, one file per serve job."""
    import tempfile
    import time as _time

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        old = d / "job_111.endpoint"
        old.write_text("stale-node:8000\n")
        os.utime(old, (1_000_000, 1_000_000))
        new = d / "job_222.endpoint"
        new.write_text("fresh-node:8001\n")
        os.utime(new, (_time.time(), _time.time()))
        with env(ANI_ENDPOINT_DIR=str(d)):
            assert lb.vllm_base_url() == "http://fresh-node:8001/v1", "newest file wins"
            assert [p.name for p in lb.endpoint_file_candidates()][:2] == \
                ["job_222.endpoint", "job_111.endpoint"]
        # A directory passed via NOT_VLLM_ENDPOINT_FILE is scanned too.
        with env(NOT_VLLM_ENDPOINT_FILE=str(d)):
            assert lb.vllm_base_url() == "http://fresh-node:8001/v1"
        # ANI_OUT_DIR is searched for the conventional filename.
        (d / "vllm_endpoint.txt").write_text("out-node:9000\n")
        with env(ANI_OUT_DIR=str(d)):
            assert lb.vllm_base_url() == "http://out-node:9000/v1"


def test_get_vllm_client_errors_without_endpoint() -> None:
    with env():
        try:
            lb.get_vllm_client()
        except lb.LocalEndpointUnavailable as e:
            assert "NOT_VLLM_BASE_URL" in str(e)
        else:
            raise AssertionError("expected LocalEndpointUnavailable")


# ---------------------------------------------------------------------------
# Backend selection and fallback
# ---------------------------------------------------------------------------

def test_backend_selection_and_fallback() -> None:
    # Explicit hf: never probes.
    with env(NOT_LOCAL_BACKEND="hf"):
        with patch(lb, "_probe_vllm", lambda url: (_ for _ in ()).throw(
                AssertionError("must not probe when hf is requested"))):
            assert lb.local_backend_name() == "hf"

    # vLLM up -> vllm.
    with env(NOT_VLLM_BASE_URL="http://node01:8000/v1"):
        with patch(lb, "_probe_vllm", lambda url: ["Qwen/Qwen3-8B"]):
            assert lb.local_backend_name() == "vllm"

    # vLLM down -> automatic fallback to hf.
    with env(NOT_VLLM_BASE_URL="http://node01:8000/v1"):
        with patch(lb, "_probe_vllm", lambda url: None):
            assert lb.local_backend_name() == "hf"

    # No endpoint discoverable at all -> fallback to hf.
    with env():
        assert lb.local_backend_name() == "hf"

    # Fallback disabled -> hard, actionable error.
    with env(NOT_VLLM_BASE_URL="http://node01:8000/v1", NOT_LOCAL_NO_FALLBACK="1"):
        with patch(lb, "_probe_vllm", lambda url: None):
            try:
                lb.local_backend_name()
            except lb.LocalEndpointUnavailable as e:
                assert "node01:8000" in str(e)
                assert "NOT_LOCAL_BACKEND=hf" in str(e)
            else:
                raise AssertionError("expected LocalEndpointUnavailable")

    # Invalid value is rejected loudly rather than silently defaulting.
    with env(NOT_LOCAL_BACKEND="ollama"):
        try:
            lb.local_backend_name()
        except lb.LocalBackendError as e:
            assert "ollama" in str(e)
        else:
            raise AssertionError("expected LocalBackendError")


def test_backend_honours_slurm_env_setup_contract() -> None:
    """slurm/env_setup.sh defaults NOT_LOCAL_BACKEND=auto and resolves ANI_LOCAL_BACKEND."""
    # "auto" must behave like the default, not blow up.
    with env(NOT_LOCAL_BACKEND="auto", NOT_VLLM_BASE_URL="http://node01:8000/v1"):
        with patch(lb, "_probe_vllm", lambda url: ["Qwen/Qwen3-8B"]):
            assert lb.requested_backend() == "vllm"
            assert lb.local_backend_name() == "vllm"
    # auto + a node with no vLLM installed -> env_setup resolved hf; skip the probe.
    with env(NOT_LOCAL_BACKEND="auto", ANI_LOCAL_BACKEND="hf"):
        with patch(lb, "_probe_vllm", lambda url: (_ for _ in ()).throw(
                AssertionError("must not probe when env_setup resolved hf"))):
            assert lb.requested_backend() == "hf"
            assert lb.local_backend_name() == "hf"
    # An explicit switch still beats the resolved mirror.
    with env(NOT_LOCAL_BACKEND="hf", ANI_LOCAL_BACKEND="vllm"):
        assert lb.requested_backend() == "hf"
    with env(ANI_LOCAL_BACKEND="vllm", NOT_VLLM_BASE_URL="http://node01:8000/v1"):
        with patch(lb, "_probe_vllm", lambda url: ["Qwen/Qwen3-8B"]):
            assert lb.local_backend_name() == "vllm"


def test_backend_probe_is_cached() -> None:
    calls: list[str] = []

    def counting_probe(url: str):
        calls.append(url)
        return ["Qwen/Qwen3-8B"]

    with env(NOT_VLLM_BASE_URL="http://node01:8000/v1"):
        with patch(lb, "_probe_vllm", counting_probe):
            assert lb.local_backend_name() == "vllm"
            assert lb.local_backend_name() == "vllm"
            assert len(calls) == 1, calls


# ---------------------------------------------------------------------------
# vLLM generation path
# ---------------------------------------------------------------------------

def test_local_generate_vllm_shape_and_params() -> None:
    completions = StubCompletions(text="VERDICT: TRUE")
    with env(NOT_VLLM_BASE_URL="http://node01:8000/v1"):
        with patch(lb, "_probe_vllm", lambda url: ["Qwen/Qwen3-8B"]), \
                patch(lb, "get_vllm_client", lambda base_url="": StubVLLMClient(completions)):
            lb._PROBE_CACHE["http://node01:8000/v1"] = ["Qwen/Qwen3-8B"]
            res = lb.local_generate(
                "qwen3-8b", "sys prompt", "user prompt",
                sample_idx=3, max_tokens=128, temperature=0.0,
            )
    assert isinstance(res, GenerationResult)
    assert res.text == "VERDICT: TRUE"
    assert res.model == "qwen3-8b"
    assert res.finish_reason == "stop"
    assert res.prompt_tokens == 11 and res.completion_tokens == 7
    assert res.total_tokens == 18
    assert res.latency_s >= 0.0
    assert res.meta["backend"] == "vllm"
    assert res.meta["repo_id"] == "Qwen/Qwen3-8B"
    assert res.meta["deterministic"] is False

    call = completions.calls[0]
    assert call["model"] == "Qwen/Qwen3-8B"
    assert call["max_tokens"] == 128
    assert call["temperature"] == 0.0
    assert call["seed"] == 3, "sample_idx must become the default seed"
    assert call["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert call["extra_body"]["top_k"] == -1, "greedy must pin top_k"
    assert "response_format" not in call, "json_mode is opt-in via NOT_VLLM_JSON_MODE"


def test_local_generate_vllm_explicit_seed_and_served_name() -> None:
    completions = StubCompletions()
    with env(NOT_VLLM_BASE_URL="http://node01:8000/v1", NOT_VLLM_SERVED_NAME="served-qwen"):
        with patch(lb, "_probe_vllm", lambda url: ["served-qwen"]), \
                patch(lb, "get_vllm_client", lambda base_url="": StubVLLMClient(completions)):
            lb.local_generate("qwen3-8b", "", "u", sample_idx=0, seed=99, temperature=0.7)
    call = completions.calls[0]
    assert call["seed"] == 99
    assert call["model"] == "served-qwen"
    assert call["temperature"] == 0.7
    assert "top_k" not in call.get("extra_body", {}), "sampling must not pin top_k"
    assert call["messages"] == [{"role": "user", "content": "u"}]


def test_local_generate_vllm_retries_then_succeeds() -> None:
    completions = StubCompletions(fail_times=2, error=RuntimeError("transient 500"))
    with env(NOT_VLLM_BASE_URL="http://node01:8000/v1"), no_sleep():
        with patch(lb, "_probe_vllm", lambda url: ["Qwen/Qwen3-8B"]), \
                patch(lb, "get_vllm_client", lambda base_url="": StubVLLMClient(completions)):
            res = lb.local_generate("qwen3-8b", "", "u", sample_idx=0)
    assert res.text == "stub answer"
    assert len(completions.calls) == 3, "must retry with back-off like generators.py"


def test_local_generate_vllm_endpoint_down_error_is_clear() -> None:
    completions = StubCompletions(fail_times=99, error=ConnectionError("Connection refused"))
    with env(NOT_VLLM_BASE_URL="http://node01:8000/v1", NOT_LOCAL_MAX_ATTEMPTS="2"), no_sleep():
        with patch(lb, "_probe_vllm", lambda url: ["Qwen/Qwen3-8B"]), \
                patch(lb, "get_vllm_client", lambda base_url="": StubVLLMClient(completions)):
            try:
                lb.local_generate("qwen3-8b", "", "u", sample_idx=0)
            except lb.LocalEndpointUnavailable as e:
                msg = str(e)
                assert "http://node01:8000/v1" in msg
                assert "2 attempts" in msg
                assert "serve job" in msg
            else:
                raise AssertionError("expected LocalEndpointUnavailable")
    assert len(completions.calls) == 2, "NOT_LOCAL_MAX_ATTEMPTS must be honoured"


def test_local_generate_vllm_json_mode_opt_in() -> None:
    completions = StubCompletions()
    with env(NOT_VLLM_BASE_URL="http://node01:8000/v1", NOT_VLLM_JSON_MODE="1"):
        with patch(lb, "_probe_vllm", lambda url: ["Qwen/Qwen3-8B"]), \
                patch(lb, "get_vllm_client", lambda base_url="": StubVLLMClient(completions)):
            lb.local_generate("qwen3-8b", "", "u", sample_idx=0, json_mode=True)
    assert completions.calls[0]["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# HF generation path
# ---------------------------------------------------------------------------

def test_local_generate_hf_shape_and_greedy() -> None:
    seeds: list[int] = []
    bundle = stub_bundle(prompt_ids=3, new_tokens=5, completion="hf stub answer")
    with env(NOT_LOCAL_BACKEND="hf"):
        with patch(lb, "_torch", lambda: stub_torch(seeds)), \
                patch(lb, "load_hf_model", lambda m: bundle):
            res = lb.local_generate(
                "qwen3-8b", "sys", "user", sample_idx=7, max_tokens=64, temperature=0.0,
            )
    assert isinstance(res, GenerationResult)
    assert res.text == "hf stub answer"
    assert res.model == "qwen3-8b"
    assert res.prompt_tokens == 3, "prompt tokens come from the encoded input length"
    assert res.completion_tokens == 5, "completion tokens come from the generated span"
    assert res.finish_reason == "stop"
    assert res.meta["backend"] == "hf"
    assert res.meta["greedy"] is True
    assert res.meta["repo_id"] == "Qwen/Qwen3-8B"
    assert seeds == [7], "sample_idx must seed torch when no explicit seed is given"

    gen_call = bundle.model.calls[0]
    assert gen_call["do_sample"] is False, "temperature=0 must be greedy"
    assert gen_call["max_new_tokens"] == 64
    assert "temperature" not in gen_call
    assert bundle.tokenizer.template_calls[0]["add_generation_prompt"] is True
    assert bundle.tokenizer.decoded == [[3, 4, 5, 6, 7]], "decode only the new tokens"


def test_local_generate_hf_sampling_and_length_finish() -> None:
    seeds: list[int] = []
    bundle = stub_bundle(prompt_ids=4, new_tokens=16)
    with env(NOT_LOCAL_BACKEND="hf"):
        with patch(lb, "_torch", lambda: stub_torch(seeds)), \
                patch(lb, "load_hf_model", lambda m: bundle):
            res = lb.local_generate(
                "llama-3.1-8b", "", "u", sample_idx=0, seed=1234,
                max_tokens=16, temperature=0.8, top_p=0.95,
            )
    gen_call = bundle.model.calls[0]
    assert gen_call["do_sample"] is True
    assert gen_call["temperature"] == 0.8
    assert gen_call["top_p"] == 0.95
    assert res.finish_reason == "length", "hitting max_new_tokens is a length stop"
    assert res.meta["greedy"] is False
    assert seeds == [1234]


def test_local_generate_hf_repeatable_with_fixed_seed() -> None:
    """The exact-reproduction contract used by the AGT coefficient-0.0 check."""
    seeds: list[int] = []
    flags: dict = {}
    torch = stub_torch(seeds, flags)
    with env(NOT_LOCAL_BACKEND="hf"):
        with patch(lb, "_torch", lambda: torch), \
                patch(lb, "_DETERMINISM_ENABLED", False):
            outs = []
            for _ in range(2):
                bundle = stub_bundle()
                with patch(lb, "load_hf_model", lambda m, _b=bundle: _b):
                    outs.append(lb.local_generate(
                        "qwen3-8b", "sys", "user", sample_idx=0,
                        temperature=0.0, seed=0, deterministic=True,
                    ))
    assert outs[0].text == outs[1].text
    assert outs[0].meta["deterministic"] is True
    assert outs[0].meta["greedy"] is True
    assert flags["deterministic"] is True, "torch.use_deterministic_algorithms(True) must run"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert seeds and set(seeds) == {0}, seeds
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_hf_stop_strings_truncate() -> None:
    seeds: list[int] = []
    bundle = stub_bundle(completion="answer A\nEND\ntrailing junk")
    with env(NOT_LOCAL_BACKEND="hf"):
        with patch(lb, "_torch", lambda: stub_torch(seeds)), \
                patch(lb, "load_hf_model", lambda m: bundle):
            res = lb.local_generate("qwen3-8b", "", "u", sample_idx=0, stop=["\nEND"])
    assert res.text == "answer A"
    assert res.finish_reason == "stop"


def test_hosted_only_kwargs_are_tolerated() -> None:
    """Existing runners pass reasoning_effort/json_mode; those must not explode."""
    seeds: list[int] = []
    bundle = stub_bundle()
    with env(NOT_LOCAL_BACKEND="hf"):
        with patch(lb, "_torch", lambda: stub_torch(seeds)), \
                patch(lb, "load_hf_model", lambda m: bundle):
            res = lb.local_generate(
                "qwen3-8b", "s", "u", sample_idx=0,
                reasoning_effort="medium", json_mode=True,
            )
    assert res.text == "hf stub answer"
    assert "reasoning_effort" not in bundle.model.calls[0]
    assert "json_mode" not in bundle.model.calls[0]


def test_unknown_local_model_raises_clearly() -> None:
    with env():
        try:
            lb.local_generate("gpt-5.4-nano", "s", "u", sample_idx=0)
        except lb.UnknownLocalModel as e:
            assert "gpt-5.4-nano" in str(e)
        else:
            raise AssertionError("expected UnknownLocalModel")


def test_chat_template_fallback_without_template() -> None:
    tok = StubTokenizer()
    tok.chat_template = None
    rendered = lb.apply_chat_template(tok, "sys", "user")
    assert "System: sys" in rendered and "User: user" in rendered
    assert rendered.rstrip().endswith("Assistant:")
    assert tok.template_calls == [], "no template -> must not call apply_chat_template"


def test_chat_template_kwargs_forwarded() -> None:
    tok = StubTokenizer()
    lb.apply_chat_template(tok, "sys", "u", chat_template_kwargs={"enable_thinking": False})
    assert tok.template_calls[0]["kw"] == {"enable_thinking": False}


# ---------------------------------------------------------------------------
# generate_any routing seam
# ---------------------------------------------------------------------------

def test_generate_any_routes_local() -> None:
    seen: list[tuple] = []

    def fake_local(model, system, user, **kw):
        seen.append((model, system, user, kw))
        return GenerationResult(text="local", finish_reason="stop", model=model)

    with env():
        with patch(lb, "local_generate", fake_local):
            res = lb.generate_any("qwen3-8b", "s", "u", sample_idx=5, max_tokens=32,
                                  temperature=0.0, seed=1)
    assert res.text == "local"
    model, system, user, kw = seen[0]
    assert (model, system, user) == ("qwen3-8b", "s", "u")
    assert kw == {"sample_idx": 5, "max_tokens": 32, "temperature": 0.0, "seed": 1}


def test_generate_any_routes_hosted_and_strips_local_kwargs() -> None:
    from scripts import generators as gens

    seen: list[tuple] = []

    def fake_hosted(model, system, user, **kw):
        seen.append((model, system, user, kw))
        return GenerationResult(text="hosted", finish_reason="stop", model=model)

    with env():
        with patch(gens, "generate", fake_hosted):
            res = lb.generate_any(
                "claude-haiku-4-5", "s", "u",
                sample_idx=2, max_tokens=256, json_mode=True,
                temperature=0.0, seed=9, backend="hf", deterministic=True,
            )
    assert res.text == "hosted"
    model, system, user, kw = seen[0]
    assert model == "claude-haiku-4-5"
    # Hosted router does not accept temperature/seed/backend/deterministic.
    assert kw == {"sample_idx": 2, "max_tokens": 256, "json_mode": True}


def test_generate_any_defaults_sample_idx() -> None:
    from scripts import generators as gens

    seen: list[dict] = []
    with env():
        with patch(gens, "generate",
                   lambda m, s, u, **kw: (seen.append(kw),
                                          GenerationResult(text="", finish_reason="", model=m))[1]):
            lb.generate_any("gpt-5.4-nano", "s", "u")
    assert seen[0]["sample_idx"] == 0


def test_generate_any_returns_generators_dataclass() -> None:
    """Downstream aggregators unpickle/inspect the shared GenerationResult."""
    completions = StubCompletions()
    with env(NOT_VLLM_BASE_URL="http://node01:8000/v1"):
        with patch(lb, "_probe_vllm", lambda url: ["Qwen/Qwen3-8B"]), \
                patch(lb, "get_vllm_client", lambda base_url="": StubVLLMClient(completions)):
            res = lb.generate_any("qwen3-8b", "", "u", sample_idx=0, max_tokens=16)
    assert type(res) is GenerationResult
    for fieldname in ("text", "finish_reason", "model", "prompt_tokens",
                      "completion_tokens", "latency_s", "meta"):
        assert hasattr(res, fieldname), fieldname


# ---------------------------------------------------------------------------
# Import hygiene: nothing heavy at module scope
# ---------------------------------------------------------------------------

_IMPORT_PROBE = """
import sys
import scripts.local_backend as lb
heavy = [m for m in ("torch", "vllm", "transformers", "datasets") if m in sys.modules]
assert not heavy, "imported at module scope: %s" % heavy
assert lb.is_local_model("qwen3-8b")
assert not lb.is_local_model("gpt-5.4-nano")
print("IMPORT_CLEAN")
"""


def test_module_import_pulls_no_heavy_deps() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "IMPORT_CLEAN" in proc.stdout, proc.stdout


def test_lazy_import_seams_exist() -> None:
    # The seams that keep torch/transformers out of module scope must stay
    # functions, not module-level bindings.
    assert callable(lb._torch)
    assert callable(lb._transformers)
    src = (REPO_ROOT / "scripts" / "local_backend.py").read_text()
    head = src.split("# ---------------------------------------------------------------------------\n# Alias table")[0]
    for banned in ("\nimport torch", "\nimport transformers", "\nimport vllm", "\nfrom openai"):
        assert banned not in head, f"{banned.strip()} must not be at module scope"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_smoke_cli_runs_offline() -> None:
    with env():
        assert lb.main(["--smoke"]) == 0
        assert lb.main(["--list"]) == 0
        assert lb.main([]) == 0


# ---------------------------------------------------------------------------
# Standalone runner (repo convention: python -m scripts.test_local_backend)
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Offline tests for scripts/local_backend.py")
    parser.add_argument("--smoke", action="store_true",
                        help="Run the fast subset (routing + import hygiene only)")
    args = parser.parse_args(argv)

    fast = {
        "test_alias_table_contract",
        "test_is_local_model_routing",
        "test_alias_resolution_and_extras",
        "test_module_import_pulls_no_heavy_deps",
    }
    tests = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    if args.smoke:
        tests = [(n, o) for n, o in tests if n in fast]

    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  {name} OK")
        except Exception as e:  # noqa: BLE001 - test harness
            failures += 1
            print(f"  {name} FAILED: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
