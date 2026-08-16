"""
scripts/propensity.py -- exact teacher-forced verdict-propensity instrument.

WHAT THIS MEASURES
------------------
For a closed verdict set V (e.g. ("YTA", "NTA", "ESH", "NAH", "UNRESOLVED")) we
want q(v) = P(model emits verdict v | system, user, trace, prefix), measured
*exactly* rather than estimated by sampling k completions and counting.

The naive approach -- read the top-k logprob list returned by a chat API and
look up each verdict -- is wrong for two independent reasons:

  1. The verdict token can fall outside the top-k cap (k is typically 5-20).
     A verdict with q = 0.001 is invisible, and the renormalisation over the
     visible subset is then silently biased.
  2. Multi-token verdict words break it entirely.  "UNRESOLVED" is 2-4 tokens
     under every tokenizer we use; a single next-token distribution cannot
     express its probability, and two verdicts that share a leading token
     ("OPTION_A" / "OPTION_B", "YTA" / "YTB") are indistinguishable at
     position 0.

So instead, for each candidate v in V we build the FULL token sequence

        [system][user][trace][prefix][v]

run a single forward pass that scores the whole sequence, and **sum the raw
token logprobs of v's tokens only**.  There is deliberately NO length
normalisation: we are comparing complete strings under an identical prefix, so

        sum_t log P(v_t | prefix, v_<t)  ==  log P(v | prefix)

exactly.  Dividing by token count would compute a per-token mean, which is a
different (and non-probabilistic) quantity and would systematically favour long
verdict strings.  The raw sums are then renormalised across V onto the simplex.

THE SHARED-PREFIX SPLIT
-----------------------
We do not assume that tokenize(prefix) is a prefix of tokenize(prefix + v):
BPE happily merges the prefix's trailing space into the verdict's first
character.  Instead we tokenize each candidate string in full and score from
the longest common token prefix (LCP) across all candidates.  Every candidate
is then conditioned on a byte-for-byte identical token prefix, so the shared
head contributes an identical additive constant to every l_v and cancels in the
softmax.  This is exactly equivalent to scoring from any deeper anchor and is
robust to boundary merging.

Two pathologies are detected rather than papered over:
  * nested tokenizations (one verdict's token sequence is a prefix of
    another's, e.g. "A" vs "AB") make raw-sum scoring ill-posed -- the shorter
    candidate scores log P = 0 for an empty suffix.  Pass ``suffix="\\n"`` (an
    opt-in argument, default off) to disambiguate, or choose a better vocabulary.
  * identical tokenizations for two distinct verdict strings.

BACKENDS
--------
  hf    -- transformers + torch, direct logits.  This is the reference
           implementation and the arbiter in any disagreement.
  vllm  -- OpenAI-compatible endpoint, ``prompt_logprobs`` on /v1/completions
           plus /tokenize for the token ids.

Both backends funnel into one shared, unit-tested numerical core
(``propensity_from_token_logprobs``), so the maths cannot drift between them.
``cross_backend_check`` runs both and reports agreement.

Backends may return None for positions *before* the shared split (those never
enter the sum), but a None at or after the split is a backend failure and the
core raises rather than quietly summing around the hole.

VALIDATION GATES (pre-registered)
---------------------------------
  tokenization_audit(model, verdicts)      -- how each verdict tokenizes;
      flags shared leading tokens, prefix-space inconsistency, nesting.
  label_bias_check(model, items, labelings) -- recompute propensities under
      several verdict vocabularies (YTA/NTA vs A/B vs 1/2); gate is
      Spearman rho > 0.8 between labelings.
  variance_decomposition(...)              -- the T0-B pilot statistic; splits
      total variance into within-trace q(1-q) and between-trace sigma^2 and
      reports the realised efficiency gain 1 + E[q(1-q)]/sigma^2_trace.

Caching: one JSON per cell under OUT_DIR, ``prop_{model}_{backend}_{item}_{hash}.json``.

Environment:
  PROPENSITY_BACKEND        hf | vllm | auto        (default auto)
  PROPENSITY_VLLM_URL       e.g. http://localhost:8000  (presence selects vllm under auto)
  PROPENSITY_VLLM_MODEL     served model name if it differs from the HF repo id
  PROPENSITY_TEMPLATE_KWARGS  JSON dict passed to apply_chat_template (e.g. {"enable_thinking": false})
  PROPENSITY_CACHE          0 to disable the on-disk cache

Usage:
  python -m scripts.propensity --smoke
  python -m scripts.propensity --model Qwen3-8B --audit --instrument aita
  python -m scripts.propensity --model Qwen3-8B --instrument aita \\
      --user "AITA for eating the last slice?" --backend hf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.run_phase1_quartet import OUT_DIR, _safe

# ---------------------------------------------------------------------------
# Contract-module imports.  These are owned by other agents; guard them so this
# module still imports (and its offline tests still run) before they land.
# ---------------------------------------------------------------------------

_FALLBACK_INSTRUMENTS: dict[str, tuple[str, ...]] = {
    "brokenmath": ("TRUE", "FALSE", "UNRESOLVED"),
    "aita": ("YTA", "NTA", "ESH", "NAH", "UNRESOLVED"),
    "ledger": ("OPTION_A", "OPTION_B", "UNRESOLVED"),
}
_FALLBACK_NOVERDICT = "NOVERDICT"

try:  # pragma: no cover - exercised only once verdict_format.py exists
    from scripts.verdict_format import (  # type: ignore
        INSTRUMENTS as _VF_INSTRUMENTS,
        NOVERDICT as _VF_NOVERDICT,
        verdict_tokens as _vf_verdict_tokens,
    )
    HAVE_VERDICT_FORMAT = True
except Exception:  # ImportError, or partial module during parallel development
    _VF_INSTRUMENTS = _FALLBACK_INSTRUMENTS  # type: ignore[assignment]
    _VF_NOVERDICT = _FALLBACK_NOVERDICT
    _vf_verdict_tokens = None  # type: ignore[assignment]
    HAVE_VERDICT_FORMAT = False

try:  # pragma: no cover - exercised only once local_backend.py exists
    from scripts.local_backend import (  # type: ignore
        LOCAL_MODELS as _LOCAL_MODELS,
        resolve_repo_id as _lb_resolve_repo_id,
        vllm_base_url as _lb_vllm_base_url,
    )
    HAVE_LOCAL_BACKEND = True
except Exception:
    _LOCAL_MODELS = {}  # type: ignore[assignment]
    _lb_resolve_repo_id = None  # type: ignore[assignment]
    _lb_vllm_base_url = None  # type: ignore[assignment]
    HAVE_LOCAL_BACKEND = False


DEFAULT_PREFIX = "VERDICT: "
DEFAULT_VLLM_URL = "http://localhost:8000"

# Fallback alias -> HF repo id map, used only when scripts/local_backend.py is
# unavailable.  local_backend.LOCAL_MODELS wins whenever it is importable.
_FALLBACK_REPOS: dict[str, str] = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
}


def resolve_repo(model: str) -> str:
    """Map a model alias to an HF repo id (identity for anything unrecognised).

    Defers to ``local_backend.resolve_repo_id`` when that module is importable,
    so "local/qwen3-8b", "qwen3-8b" and "Qwen/Qwen3-8B" all land on one repo and
    one cache key.  Unknown models fall through to identity rather than raising:
    a propensity measurement on an arbitrary HF repo is perfectly meaningful.
    """
    if _lb_resolve_repo_id is not None:
        try:
            return str(_lb_resolve_repo_id(model))
        except Exception:
            pass  # UnknownLocalModel etc. -- fall through to the local table
    if model in _LOCAL_MODELS:
        return str(_LOCAL_MODELS[model])
    low = model.lower()
    for alias, repo in _LOCAL_MODELS.items():
        if alias.lower() == low:
            return str(repo)
    if model in _FALLBACK_REPOS:
        return _FALLBACK_REPOS[model]
    if low in _FALLBACK_REPOS:
        return _FALLBACK_REPOS[low]
    return model


def verdicts_for_instrument(instrument: str, *, allow_unresolved: bool = True) -> tuple[str, ...]:
    """Closed candidate set for an instrument.

    Prefers scripts.verdict_format when available.  NOVERDICT is an extraction
    sentinel, never a scoring candidate, so it is always excluded.
    """
    if _vf_verdict_tokens is not None:
        toks = tuple(_vf_verdict_tokens(instrument, allow_unresolved=allow_unresolved))
    else:
        toks = tuple(_VF_INSTRUMENTS[instrument])
        if not allow_unresolved:
            toks = tuple(t for t in toks if t != "UNRESOLVED")
    return tuple(t for t in toks if t != _VF_NOVERDICT)


# ===========================================================================
# Numerical core -- pure, backend-free, unit-tested offline
# ===========================================================================

def longest_common_prefix_len(seqs: Sequence[Sequence[int]]) -> int:
    """Length of the longest token prefix shared by every sequence."""
    seqs = [list(s) for s in seqs]
    if not seqs:
        return 0
    limit = min(len(s) for s in seqs)
    i = 0
    while i < limit:
        first = seqs[0][i]
        if any(s[i] != first for s in seqs[1:]):
            break
        i += 1
    return i


def renormalise(logprob_sums: Mapping[str, float]) -> dict[str, float]:
    """Softmax the raw logprob sums onto the simplex.

    No temperature, no length term: the inputs are already log P(v | prefix)
    up to one additive constant shared by every candidate, and that constant
    cancels here.  Returns a dict summing to 1.0 (to floating-point).
    """
    keys = list(logprob_sums)
    if not keys:
        return {}
    vals = [float(logprob_sums[k]) for k in keys]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        u = 1.0 / len(keys)
        return {k: u for k in keys}
    top = max(finite)
    exps = [math.exp(v - top) if math.isfinite(v) else 0.0 for v in vals]
    total = math.fsum(exps)
    if total <= 0.0:
        u = 1.0 / len(keys)
        return {k: u for k in keys}
    return {k: e / total for k, e in zip(keys, exps)}


def sum_token_logprobs(token_logprobs: Sequence[Optional[float]], start: int) -> float:
    """Sum raw logprobs from ``start`` to the end, skipping None entries.

    Index 0 is None by construction (no token precedes the first token), so a
    ``start`` of 0 is silently treated as 1.  Deliberately unnormalised.
    """
    lo = max(int(start), 1)
    vals = [lp for lp in token_logprobs[lo:] if lp is not None]
    if not vals:
        return 0.0
    return math.fsum(float(v) for v in vals)


def propensity_from_token_logprobs(
    verdicts: Sequence[str],
    cand_ids: Mapping[str, Sequence[int]],
    cand_token_logprobs: Mapping[str, Sequence[Optional[float]]],
    *,
    split: Optional[int] = None,
) -> dict[str, Any]:
    """The whole measurement, given per-candidate token ids and token logprobs.

    Args:
        verdicts: ordered candidate strings.
        cand_ids: verdict -> full token id sequence of [context][verdict].
        cand_token_logprobs: verdict -> per-position logprob of the *actual*
            token at that position given everything before it.  Position 0 must
            be None (nothing conditions it).
        split: override the shared-prefix split index; defaults to the longest
            common token prefix across all candidates.

    Returns a dict with logprob_sums (raw, unnormalised), probs (simplex),
    n_scored_tokens, split, and structural flags.
    """
    verdicts = list(verdicts)
    if not verdicts:
        raise ValueError("verdicts must be non-empty")
    if len(set(verdicts)) != len(verdicts):
        raise ValueError(f"verdicts must be unique, got {verdicts}")
    missing = [v for v in verdicts if v not in cand_ids or v not in cand_token_logprobs]
    if missing:
        raise KeyError(f"missing scores for verdicts: {missing}")

    for v in verdicts:
        ids = list(cand_ids[v])
        lps = list(cand_token_logprobs[v])
        if len(ids) != len(lps):
            raise ValueError(
                f"token/logprob length mismatch for {v!r}: {len(ids)} ids vs {len(lps)} logprobs"
            )
        if lps and lps[0] is not None:
            raise ValueError(
                f"position 0 logprob must be None for {v!r} (nothing conditions the first token)"
            )

    seqs = [list(cand_ids[v]) for v in verdicts]
    auto_split = longest_common_prefix_len(seqs)
    use_split = auto_split if split is None else int(split)

    # A backend may legitimately return None for positions *before* the shared
    # split (they are never summed).  A None at or after the split means the
    # backend failed to report a logprob we actually need -- summing around it
    # would silently understate that candidate, so fail loudly instead.
    lo = max(use_split, 1)
    for v in verdicts:
        gaps = [
            i for i, lp in enumerate(cand_token_logprobs[v])
            if i >= lo and lp is None
        ]
        if gaps:
            raise ValueError(
                f"missing logprobs for {v!r} at scored positions {gaps[:8]} "
                f"(split={use_split}); the backend did not report every token "
                "in the scored range"
            )

    logprob_sums: dict[str, float] = {}
    n_scored: dict[str, int] = {}
    for v in verdicts:
        logprob_sums[v] = sum_token_logprobs(cand_token_logprobs[v], use_split)
        # Tokens actually scored are indices [max(split, 1), len): position 0 is
        # never scored because nothing conditions it.
        n_scored[v] = max(len(cand_ids[v]) - max(use_split, 1), 0)

    flags: list[str] = []
    if len(verdicts) > 1:
        for v in verdicts:
            if n_scored[v] == 0:
                flags.append("empty_scored_suffix")
                break
        id_tuples = {v: tuple(cand_ids[v]) for v in verdicts}
        seen: dict[tuple[int, ...], str] = {}
        for v, t in id_tuples.items():
            if t in seen:
                flags.append("identical_tokenization")
                break
            seen[t] = v
        for a in verdicts:
            for b in verdicts:
                if a is b or a == b:
                    continue
                ta, tb = id_tuples[a], id_tuples[b]
                if len(ta) < len(tb) and tb[: len(ta)] == ta:
                    flags.append("nested_tokenization")
                    break
            if "nested_tokenization" in flags:
                break

    return {
        "verdicts": tuple(verdicts),
        "logprob_sums": logprob_sums,
        "probs": renormalise(logprob_sums),
        "n_scored_tokens": n_scored,
        "split": use_split,
        "auto_split": auto_split,
        "flags": sorted(set(flags)),
    }


# ===========================================================================
# Record type + cache
# ===========================================================================

@dataclass
class PropensityRecord:
    """One measured cell: exact propensities over a closed verdict set."""

    model: str
    backend: str
    verdicts: tuple[str, ...]
    logprob_sums: dict[str, float]
    probs: dict[str, float]
    n_scored_tokens: dict[str, int]
    split: int
    context_tokens: int = 0
    chat_template_applied: bool = False
    prefix: str = DEFAULT_PREFIX
    suffix: str = ""
    item_id: str = "anon"
    flags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def argmax(self) -> str:
        return max(self.probs, key=lambda k: self.probs[k])

    @property
    def entropy(self) -> float:
        """Shannon entropy of the renormalised simplex, in nats."""
        return -math.fsum(p * math.log(p) for p in self.probs.values() if p > 0.0)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "backend": self.backend,
            "verdicts": list(self.verdicts),
            "logprob_sums": self.logprob_sums,
            "probs": self.probs,
            "n_scored_tokens": self.n_scored_tokens,
            "split": self.split,
            "context_tokens": self.context_tokens,
            "chat_template_applied": self.chat_template_applied,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "item_id": self.item_id,
            "flags": list(self.flags),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "PropensityRecord":
        return cls(
            model=obj["model"],
            backend=obj["backend"],
            verdicts=tuple(obj["verdicts"]),
            logprob_sums=dict(obj["logprob_sums"]),
            probs=dict(obj["probs"]),
            n_scored_tokens=dict(obj.get("n_scored_tokens", {})),
            split=int(obj.get("split", 0)),
            context_tokens=int(obj.get("context_tokens", 0)),
            chat_template_applied=bool(obj.get("chat_template_applied", False)),
            prefix=obj.get("prefix", DEFAULT_PREFIX),
            suffix=obj.get("suffix", ""),
            item_id=obj.get("item_id", "anon"),
            flags=list(obj.get("flags", [])),
            meta=dict(obj.get("meta", {})),
        )


def cell_hash(
    *,
    system: str,
    user: str,
    trace: str,
    prefix: str,
    suffix: str,
    verdicts: Sequence[str],
    chat_template: str,
) -> str:
    raw = "|".join(
        [
            system,
            user,
            trace,
            prefix,
            suffix,
            ",".join(verdicts),
            chat_template,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_path(
    model: str,
    backend: str,
    item_id: str,
    key_hash: str,
    *,
    cache_dir: Optional[Path] = None,
) -> Path:
    """One JSON per cell, per the repo's ``{prefix}_{model}_{...}.json`` convention.

    The model is canonicalised to its HF repo id first, so "qwen3-8b",
    "local/qwen3-8b" and "Qwen/Qwen3-8B" share one cache file rather than
    silently re-measuring the same cell three times.
    """
    base = Path(cache_dir) if cache_dir is not None else OUT_DIR
    name = _safe(resolve_repo(model))
    return base / f"prop_{name}_{_safe(backend)}_{_safe(item_id)}_{key_hash}.json"


def _cache_enabled() -> bool:
    return os.environ.get("PROPENSITY_CACHE", "1") != "0"


# ===========================================================================
# Context rendering
# ===========================================================================

def _template_kwargs() -> dict:
    raw = os.environ.get("PROPENSITY_TEMPLATE_KWARGS", "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def render_context(
    model: str,
    system: str,
    user: str,
    *,
    trace: str = "",
    prefix: str = DEFAULT_PREFIX,
    chat_template: str = "auto",
    tokenizer: Any = None,
) -> tuple[str, bool]:
    """Build the exact conditioning string ``[system][user][trace][prefix]``.

    With ``chat_template="auto"`` the model's own chat template is applied with
    ``add_generation_prompt=True``, so the assistant turn is left open and the
    trace + prefix continue it -- which is what teacher forcing requires.
    Falls back to a plain concatenation when transformers or a template is
    unavailable.  Returns (context, chat_template_applied).
    """
    if chat_template not in {"auto", "plain", "chat"}:
        raise ValueError(f"chat_template must be auto|plain|chat, got {chat_template!r}")
    if chat_template != "plain":
        tok = tokenizer
        if tok is None:
            try:
                tok = load_tokenizer(model)
            except Exception:
                tok = None
        template = getattr(tok, "chat_template", None) if tok is not None else None
        if tok is not None and template:
            msgs: list[dict] = []
            if system and system.strip():
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": user})
            try:
                head = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **_template_kwargs()
                )
                return f"{head}{trace}{prefix}", True
            except Exception:
                pass
        if chat_template == "chat":
            raise RuntimeError(
                f"chat_template='chat' requested but no usable chat template for {model!r}"
            )
    head = f"{system.strip()}\n\n" if system and system.strip() else ""
    return f"{head}{user}\n\n{trace}{prefix}", False


# ===========================================================================
# Tokenizer loading (HF; lazy, optional dependency)
# ===========================================================================

_TOKENIZER_CACHE: dict[str, Any] = {}
_MODEL_CACHE: dict[str, Any] = {}


def load_tokenizer(model: str) -> Any:
    """Load (and memoise) the HF tokenizer for ``model``.  Requires transformers."""
    repo = resolve_repo(model)
    if repo in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[repo]
    try:
        from transformers import AutoTokenizer  # lazy: heavy, GPU box only
    except ImportError as e:  # pragma: no cover - depends on host
        raise RuntimeError(
            "transformers is required for tokenizer access; install it on the "
            "compute node (it is intentionally not a laptop dependency)"
        ) from e
    tok = AutoTokenizer.from_pretrained(repo)
    _TOKENIZER_CACHE[repo] = tok
    return tok


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def _ids_to_tokens(tokenizer: Any, ids: Sequence[int]) -> list[str]:
    fn = getattr(tokenizer, "convert_ids_to_tokens", None)
    if fn is None:
        return [str(i) for i in ids]
    try:
        return [str(t) for t in fn(list(ids))]
    except Exception:
        return [str(i) for i in ids]


# ===========================================================================
# Backend: HF transformers (reference implementation)
# ===========================================================================

def _dtype_kwarg_for_version(version: str) -> str:
    """transformers renamed from_pretrained(torch_dtype=) to dtype= in 4.56.

    from_pretrained swallows **kwargs, so guessing wrong does not raise -- it
    silently loads in float32 and doubles the memory.  Hence the version check.
    """
    try:
        parts = str(version).split(".")
        major, minor = int(parts[0]), int(parts[1])
    except Exception:
        return "torch_dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def _load_hf_model(model: str) -> tuple[Any, Any]:
    repo = resolve_repo(model)
    if repo in _MODEL_CACHE:
        return _MODEL_CACHE[repo]
    try:
        import torch  # lazy
        import transformers  # lazy
        from transformers import AutoModelForCausalLM  # lazy
    except ImportError as e:  # pragma: no cover - depends on host
        raise RuntimeError(
            "torch + transformers are required for the hf propensity backend"
        ) from e
    tok = load_tokenizer(model)
    cuda = bool(torch.cuda.is_available())
    kwargs: dict = {
        _dtype_kwarg_for_version(getattr(transformers, "__version__", "")): (
            torch.bfloat16 if cuda else torch.float32
        )
    }
    if cuda:
        kwargs["device_map"] = "auto"
    mdl = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    mdl.eval()
    _MODEL_CACHE[repo] = (tok, mdl)
    return tok, mdl


def _logits_keep_kwarg(mdl: Any) -> Optional[str]:
    """Name of the 'only compute the last N logit rows' kwarg, if supported.

    transformers renamed num_logits_to_keep -> logits_to_keep; without either,
    a forward pass materialises [1, seq_len, vocab] logits, which for a long
    AITA context and a 150k vocab is gigabytes we do not need.
    """
    import inspect

    try:
        params = inspect.signature(mdl.forward).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


def hf_token_logprobs(
    model: str,
    texts: Sequence[str],
    *,
    add_special_tokens: bool = True,
    score_from: Optional[int] = None,
) -> tuple[list[list[int]], list[list[Optional[float]]]]:
    """Score full sequences with transformers -- the reference implementation.

    One forward pass per candidate, batch size 1: candidates differ in length,
    and right-padding a batch would misalign the "keep only the last N logit
    rows" optimisation while left-padding would corrupt RoPE offsets without
    hand-built position ids.  Since candidates share a long prefix and differ
    only in a short tail, correctness is worth far more here than the constant
    factor.

    Only positions from the shared split onward are scored; earlier positions
    come back as None because they cancel in the renormalisation anyway.

    Returns (token id sequences, per-position logprobs), position 0 always None.
    """
    import torch  # lazy

    tok, mdl = _load_hf_model(model)
    enc = [_encode(tok, t, add_special_tokens=add_special_tokens) for t in texts]
    if any(len(e) < 2 for e in enc):
        raise ValueError("every scored sequence needs at least 2 tokens")
    base = longest_common_prefix_len(enc) if score_from is None else int(score_from)
    first = max(base, 1)
    device = getattr(mdl, "device", None) or "cpu"
    keep_kwarg = _logits_keep_kwarg(mdl)

    out: list[list[Optional[float]]] = []
    for ids in enc:
        n = len(ids)
        start = min(first, n - 1)  # always score at least the final token
        keep = n - start + 1       # logit rows for positions [start-1, n)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        kwargs: dict = {}
        if keep_kwarg is not None:
            kwargs[keep_kwarg] = keep
        with torch.no_grad():
            logits = mdl(input_ids=input_ids, **kwargs).logits
        if logits.shape[1] != keep:
            # No keep-kwarg support (or the model ignored it): slice before the
            # float() upcast so we never hold two full-vocab copies.
            logits = logits[:, -keep:, :]
        lp = torch.log_softmax(logits[0].float(), dim=-1)
        row: list[Optional[float]] = [None] * n
        for j in range(keep - 1):
            pos = start + j  # token predicted by logit row j
            row[pos] = float(lp[j, ids[pos]])
        out.append(row)
    return enc, out


# ===========================================================================
# Backend: vLLM OpenAI-compatible endpoint
# ===========================================================================

def _vllm_base_url() -> str:
    """PROPENSITY_VLLM_URL wins; otherwise reuse local_backend's discovery.

    On CURC the serve job writes a host:port endpoint file that local_backend
    already knows how to find, so array jobs need no extra configuration.
    """
    explicit = os.environ.get("PROPENSITY_VLLM_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    if _lb_vllm_base_url is not None:
        try:
            discovered = str(_lb_vllm_base_url() or "").strip()
            if discovered:
                return discovered.rstrip("/")
        except Exception:
            pass
    return DEFAULT_VLLM_URL


def _vllm_model_name(model: str) -> str:
    explicit = os.environ.get("PROPENSITY_VLLM_MODEL", "").strip()
    return explicit or resolve_repo(model)


def _vllm_post(path: str, payload: dict, *, timeout: int = 600) -> dict:
    import requests  # lazy

    url = f"{_vllm_base_url()}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"vLLM HTTP {r.status_code} at {path}: {r.text[:500]}")
    return r.json()


def vllm_tokenize(model: str, text: str, *, add_special_tokens: bool = True) -> list[int]:
    data = _vllm_post(
        "/tokenize",
        {
            "model": _vllm_model_name(model),
            "prompt": text,
            "add_special_tokens": bool(add_special_tokens),
        },
    )
    return [int(t) for t in data.get("tokens", [])]


def _entry_logprob(entry: Mapping[Any, Any], token_id: Optional[int]) -> Optional[float]:
    """Pull the actual token's logprob out of one vLLM prompt_logprobs entry."""
    if not entry:
        return None
    if token_id is not None:
        for key in (str(token_id), token_id):
            if key in entry:
                val = entry[key]
                return float(val["logprob"] if isinstance(val, Mapping) else val)
    # No id available: prefer the rank-1 entry, else the single entry present.
    for val in entry.values():
        if isinstance(val, Mapping) and val.get("rank") == 1:
            return float(val["logprob"])
    if len(entry) == 1:
        val = next(iter(entry.values()))
        return float(val["logprob"] if isinstance(val, Mapping) else val)
    return None


def vllm_token_logprobs(
    model: str,
    texts: Sequence[str],
    *,
    add_special_tokens: bool = True,
) -> tuple[list[list[int]], list[list[Optional[float]]]]:
    """Score full sequences through vLLM's ``prompt_logprobs``.

    ``max_tokens=1`` because some vLLM builds reject 0; the sampled token is
    discarded.  ``prompt_logprobs=0`` returns exactly the prompt tokens' own
    logprobs, which is all the measurement needs.
    """
    all_ids: list[list[int]] = []
    all_lps: list[list[Optional[float]]] = []
    for text in texts:
        ids = vllm_tokenize(model, text, add_special_tokens=add_special_tokens)
        data = _vllm_post(
            "/v1/completions",
            {
                "model": _vllm_model_name(model),
                "prompt": text,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": 0,
                "echo": False,
                # Must match the /tokenize call, or the two token streams shift
                # by the BOS and the shared-prefix split lands in the wrong place.
                "add_special_tokens": bool(add_special_tokens),
            },
        )
        choice = (data.get("choices") or [{}])[0]
        entries = choice.get("prompt_logprobs") or []
        if not entries:
            raise RuntimeError(
                "vLLM returned no prompt_logprobs; the server must be started "
                "without --disable-logprobs and with a build that supports "
                "prompt_logprobs on /v1/completions"
            )
        if len(entries) != len(ids):
            raise RuntimeError(
                f"vLLM token stream mismatch: /tokenize returned {len(ids)} tokens "
                f"but /v1/completions returned {len(entries)} prompt_logprobs. "
                "The usual cause is the server ignoring add_special_tokens on one "
                "of the two endpoints; refusing to guess the alignment."
            )
        row: list[Optional[float]] = []
        for i, entry in enumerate(entries):
            if entry is None:
                row.append(None)
                continue
            row.append(_entry_logprob(entry, ids[i]))
        if row and row[0] is not None:
            row[0] = None
        all_ids.append(ids)
        all_lps.append(row)
    return all_ids, all_lps


# ===========================================================================
# Backend dispatch
# ===========================================================================

VALID_BACKENDS = ("hf", "vllm")


def resolve_backend(backend: Optional[str] = None) -> str:
    """Pick a backend.  "auto" chooses vllm iff an endpoint is discoverable.

    Discovery is env/file only -- never a network probe -- so resolution stays
    cheap and deterministic inside a Slurm array element.
    """
    b = (backend or os.environ.get("PROPENSITY_BACKEND", "auto")).lower()
    if b == "auto":
        if os.environ.get("PROPENSITY_VLLM_URL", "").strip():
            return "vllm"
        if _lb_vllm_base_url is not None:
            try:
                if str(_lb_vllm_base_url() or "").strip():
                    return "vllm"
            except Exception:
                pass
        return "hf"
    if b not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {VALID_BACKENDS} or auto, got {backend!r}")
    return b


def score_sequences(
    model: str,
    texts: Sequence[str],
    *,
    backend: str,
    add_special_tokens: bool = True,
) -> tuple[list[list[int]], list[list[Optional[float]]]]:
    if backend == "hf":
        return hf_token_logprobs(model, texts, add_special_tokens=add_special_tokens)
    if backend == "vllm":
        return vllm_token_logprobs(model, texts, add_special_tokens=add_special_tokens)
    raise ValueError(f"unknown backend {backend!r}")


# ===========================================================================
# The public instrument
# ===========================================================================

def verdict_propensity_record(
    model: str,
    system: str,
    user: str,
    *,
    verdicts: Sequence[str],
    trace: str = "",
    prefix: str = DEFAULT_PREFIX,
    suffix: str = "",
    backend: Optional[str] = None,
    chat_template: str = "auto",
    item_id: str = "anon",
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> PropensityRecord:
    """Full-fidelity exact propensity measurement (see module docstring).

    ``suffix`` is opt-in and defaults to "" so this never silently changes any
    previously measured cell; set it (e.g. "\\n") only when the tokenization
    audit reports nested verdict tokenizations.
    """
    verdicts = tuple(verdicts)
    if not verdicts:
        raise ValueError("verdicts must be non-empty")
    if len(set(verdicts)) != len(verdicts):
        raise ValueError(f"verdicts must be unique, got {verdicts}")

    bk = resolve_backend(backend)
    key = cell_hash(
        system=system,
        user=user,
        trace=trace,
        prefix=prefix,
        suffix=suffix,
        verdicts=verdicts,
        chat_template=chat_template,
    )
    path = cache_path(model, bk, item_id, key, cache_dir=cache_dir)
    if use_cache and _cache_enabled() and path.exists():
        try:
            return PropensityRecord.from_dict(json.loads(path.read_text()))
        except Exception:
            pass  # corrupt cache entry: recompute and overwrite

    if len(verdicts) == 1:
        rec = PropensityRecord(
            model=model,
            backend=bk,
            verdicts=verdicts,
            logprob_sums={verdicts[0]: 0.0},
            probs={verdicts[0]: 1.0},
            n_scored_tokens={verdicts[0]: 0},
            split=0,
            prefix=prefix,
            suffix=suffix,
            item_id=item_id,
            flags=["singleton_verdict_set"],
        )
        return rec

    context, templated = render_context(
        model, system, user, trace=trace, prefix=prefix, chat_template=chat_template
    )
    # A chat template already emits BOS/turn markers; adding specials again
    # would double them and shift every logprob.
    add_special = not templated
    texts = [f"{context}{v}{suffix}" for v in verdicts]

    ids_list, lp_list = score_sequences(
        model, texts, backend=bk, add_special_tokens=add_special
    )
    cand_ids = {v: ids_list[i] for i, v in enumerate(verdicts)}
    cand_lps = {v: lp_list[i] for i, v in enumerate(verdicts)}
    core = propensity_from_token_logprobs(verdicts, cand_ids, cand_lps)

    flags = list(core["flags"])
    context_tokens = 0
    try:
        ctx_ids = score_context_tokens(model, context, backend=bk, add_special_tokens=add_special)
        context_tokens = len(ctx_ids)
        if core["split"] < context_tokens:
            flags.append("boundary_merge")
    except Exception:
        context_tokens = 0

    rec = PropensityRecord(
        model=model,
        backend=bk,
        verdicts=verdicts,
        logprob_sums=core["logprob_sums"],
        probs=core["probs"],
        n_scored_tokens=core["n_scored_tokens"],
        split=int(core["split"]),
        context_tokens=context_tokens,
        chat_template_applied=templated,
        prefix=prefix,
        suffix=suffix,
        item_id=item_id,
        flags=sorted(set(flags)),
        meta={
            "context_chars": len(context),
            "trace_chars": len(trace),
            "add_special_tokens": add_special,
            "repo": resolve_repo(model),
        },
    )
    if use_cache and _cache_enabled():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec.to_dict(), ensure_ascii=False))
        except Exception:
            pass
    return rec


def score_context_tokens(
    model: str, context: str, *, backend: str, add_special_tokens: bool
) -> list[int]:
    """Token ids of the bare context, for the boundary-merge diagnostic only."""
    if backend == "vllm":
        return vllm_tokenize(model, context, add_special_tokens=add_special_tokens)
    tok = load_tokenizer(model)
    return _encode(tok, context, add_special_tokens=add_special_tokens)


def verdict_propensity(
    model: str,
    system: str,
    user: str,
    *,
    verdicts: Sequence[str],
    trace: str = "",
    prefix: str = DEFAULT_PREFIX,
    **kwargs: Any,
) -> dict[str, float]:
    """Exact teacher-forced propensity over a closed verdict set.

    Returns a dict verdict -> probability summing to 1.0.  Use
    ``verdict_propensity_record`` when the diagnostics matter.
    """
    rec = verdict_propensity_record(
        model, system, user, verdicts=verdicts, trace=trace, prefix=prefix, **kwargs
    )
    return dict(rec.probs)


# ===========================================================================
# Gate 1 -- tokenization audit
# ===========================================================================

def audit_from_token_ids(
    verdicts: Sequence[str],
    bare_ids: Mapping[str, Sequence[int]],
    ctx_prefix_ids: Sequence[int],
    ctx_cand_ids: Mapping[str, Sequence[int]],
    *,
    tokens: Optional[Mapping[str, Sequence[str]]] = None,
) -> dict:
    """Pure tokenization-audit logic (no tokenizer needed; unit-tested offline).

    Args:
        verdicts: candidate strings.
        bare_ids: verdict -> ids of the verdict tokenized on its own.
        ctx_prefix_ids: ids of the prefix alone (e.g. "VERDICT: ").
        ctx_cand_ids: verdict -> ids of prefix + verdict.
        tokens: optional verdict -> readable token pieces of ctx_cand_ids.
    """
    verdicts = list(verdicts)
    per: dict[str, dict] = {}
    shared: list[list[str]] = []
    inconsistent: list[str] = []
    multi: list[str] = []
    nested: list[list[str]] = []
    identical: list[list[str]] = []

    suffix_ids: dict[str, tuple[int, ...]] = {}
    for v in verdicts:
        cand = list(ctx_cand_ids[v])
        split = longest_common_prefix_len([list(ctx_prefix_ids), cand])
        suf = tuple(cand[split:])
        suffix_ids[v] = suf
        bare = tuple(bare_ids[v])
        boundary_merge = split < len(ctx_prefix_ids)
        space_consistent = (suf == bare) and not boundary_merge
        per[v] = {
            "bare_ids": list(bare),
            "n_tokens_bare": len(bare),
            "context_ids": cand,
            "suffix_ids": list(suf),
            "n_tokens_in_context": len(suf),
            "split": split,
            "prefix_tokens": len(ctx_prefix_ids),
            "boundary_merge": boundary_merge,
            "space_consistent": space_consistent,
            "first_token_in_context": suf[0] if suf else None,
            "multi_token": len(suf) > 1 or len(bare) > 1,
        }
        if tokens is not None and v in tokens:
            per[v]["context_tokens"] = list(tokens[v])
        if not space_consistent:
            inconsistent.append(v)
        if per[v]["multi_token"]:
            multi.append(v)

    for i, a in enumerate(verdicts):
        for b in verdicts[i + 1:]:
            sa, sb = suffix_ids[a], suffix_ids[b]
            if sa and sb and sa[0] == sb[0]:
                shared.append([a, b])
            if sa == sb:
                identical.append([a, b])
            elif len(sa) < len(sb) and sb[: len(sa)] == sa:
                nested.append([a, b])
            elif len(sb) < len(sa) and sa[: len(sb)] == sb:
                nested.append([b, a])

    warnings: list[str] = []
    hard: list[str] = []
    if shared:
        warnings.append(
            "verdicts share a leading token -- a top-k next-token read would "
            "conflate them; exact full-sequence scoring is unaffected"
        )
    if inconsistent:
        warnings.append(
            "verdict tokenizes differently with the prefix's trailing space "
            "than on its own -- expected for BPE; scoring uses the shared-prefix "
            "split so it stays exact"
        )
    if multi:
        warnings.append(
            "multi-token verdicts present -- a single next-token distribution "
            "cannot express these, which is why this instrument exists"
        )
    if identical:
        hard.append("two distinct verdict strings tokenize identically")
    if nested:
        hard.append(
            "one verdict's token sequence is a prefix of another's; raw-sum "
            "scoring is ill-posed -- pass suffix='\\n' or change the vocabulary"
        )

    return {
        "verdicts": verdicts,
        "per_verdict": per,
        "shared_leading_token": shared,
        "inconsistent_with_space": inconsistent,
        "multi_token": multi,
        "nested_prefix": nested,
        "identical_tokenization": identical,
        "warnings": warnings,
        "hard_failures": hard,
        "passed": not hard,
    }


def tokenization_audit(
    model: str,
    verdicts: Sequence[str],
    *,
    prefix: str = DEFAULT_PREFIX,
    tokenizer: Any = None,
) -> dict:
    """Gate 1: report how each verdict tokenizes under this model's tokenizer.

    Flags verdicts that share a leading token and verdicts that tokenize
    inconsistently with vs without the prefix's trailing space, plus the two
    hard failures (nested / identical tokenizations).  Pass ``tokenizer`` to
    audit an arbitrary tokenizer object (used by the offline tests).
    """
    tok = tokenizer if tokenizer is not None else load_tokenizer(model)
    verdicts = list(verdicts)
    bare = {v: _encode(tok, v, add_special_tokens=False) for v in verdicts}
    ctx_prefix = _encode(tok, prefix, add_special_tokens=False)
    ctx_cand = {v: _encode(tok, f"{prefix}{v}", add_special_tokens=False) for v in verdicts}
    pieces = {v: _ids_to_tokens(tok, ctx_cand[v]) for v in verdicts}
    out = audit_from_token_ids(verdicts, bare, ctx_prefix, ctx_cand, tokens=pieces)
    out["model"] = model
    out["repo"] = resolve_repo(model)
    out["prefix"] = prefix
    return out


# ===========================================================================
# Gate 2 -- label-bias check (Spearman across verdict vocabularies)
# ===========================================================================

def rankdata(xs: Sequence[float]) -> list[float]:
    """Average ranks, ties shared (the ranking Spearman's rho needs)."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson_r(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n != len(b):
        raise ValueError("length mismatch")
    if n < 2:
        return float("nan")
    ma = math.fsum(a) / n
    mb = math.fsum(b) / n
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    num = math.fsum(x * y for x, y in zip(da, db))
    va = math.fsum(x * x for x in da)
    vb = math.fsum(y * y for y in db)
    if va <= 0.0 or vb <= 0.0:
        return float("nan")
    return num / math.sqrt(va * vb)


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation; nan when either input is constant."""
    return pearson_r(rankdata(a), rankdata(b))


def _normalise_item(item: Any, idx: int) -> dict:
    """Accept dicts, objects with attributes, or (system, user[, trace]) tuples."""
    if isinstance(item, Mapping):
        return {
            "id": str(item.get("id", item.get("item_id", f"item{idx:04d}"))),
            "system": str(item.get("system", "")),
            "user": str(item.get("user", item.get("prompt", ""))),
            "trace": str(item.get("trace", "")),
        }
    if isinstance(item, (tuple, list)):
        seq = list(item)
        return {
            "id": f"item{idx:04d}",
            "system": str(seq[0]) if len(seq) > 0 else "",
            "user": str(seq[1]) if len(seq) > 1 else "",
            "trace": str(seq[2]) if len(seq) > 2 else "",
        }
    return {
        "id": str(getattr(item, "id", f"item{idx:04d}")),
        "system": str(getattr(item, "system", "")),
        "user": str(getattr(item, "user", getattr(item, "prompt", ""))),
        "trace": str(getattr(item, "trace", "")),
    }


def label_bias_check(
    model: str,
    items: Sequence[Any],
    labelings: Sequence[Sequence[str]],
    *,
    prefix: str = DEFAULT_PREFIX,
    gate: float = 0.8,
    propensity_fn: Optional[Callable[..., Mapping[str, float]]] = None,
    **kwargs: Any,
) -> dict:
    """Gate 2: does the measurement survive relabelling the verdict vocabulary?

    ``labelings`` are parallel tuples in the same semantic order, e.g.
    [("YTA","NTA"), ("A","B"), ("1","2")] -- index j denotes the same class in
    every labeling.  For each pair of labelings we correlate the per-item,
    per-class propensities (flattened over items x classes) and report Spearman
    rho.  The pre-registered gate is rho > 0.8 for every pair.

    Also reports the per-item across-class rho when there are >= 3 classes;
    with 2 classes that quantity is degenerate (always +/-1) and is skipped.
    """
    labelings = [tuple(lab) for lab in labelings]
    if len(labelings) < 2:
        raise ValueError("need at least two labelings to compare")
    widths = {len(lab) for lab in labelings}
    if len(widths) != 1:
        raise ValueError(f"all labelings must have the same length, got {sorted(widths)}")
    n_classes = widths.pop()
    if n_classes < 2:
        raise ValueError("labelings need at least two classes")
    for lab in labelings:
        if len(set(lab)) != len(lab):
            raise ValueError(f"labeling has duplicate labels: {lab}")

    fn = propensity_fn if propensity_fn is not None else verdict_propensity
    norm_items = [_normalise_item(it, i) for i, it in enumerate(items)]
    if not norm_items:
        raise ValueError("items must be non-empty")

    # per_labeling[li][item_index] = [p(class0), p(class1), ...]
    per_labeling: list[list[list[float]]] = []
    for li, lab in enumerate(labelings):
        rows: list[list[float]] = []
        for it in norm_items:
            probs = fn(
                model,
                it["system"],
                it["user"],
                verdicts=lab,
                trace=it["trace"],
                prefix=prefix,
                item_id=f"{it['id']}_lab{li}",
                **kwargs,
            )
            rows.append([float(probs.get(c, 0.0)) for c in lab])
        per_labeling.append(rows)

    pairs: list[dict] = []
    for i in range(len(labelings)):
        for j in range(i + 1, len(labelings)):
            flat_i = [p for row in per_labeling[i] for p in row]
            flat_j = [p for row in per_labeling[j] for p in row]
            rho = spearman_rho(flat_i, flat_j)
            diffs = [abs(a - b) for a, b in zip(flat_i, flat_j)]
            # Class-0 propensity across items -- the headline sycophancy channel.
            c0_i = [row[0] for row in per_labeling[i]]
            c0_j = [row[0] for row in per_labeling[j]]
            pairs.append(
                {
                    "labeling_a": list(labelings[i]),
                    "labeling_b": list(labelings[j]),
                    "rho_flat": rho,
                    "rho_class0": spearman_rho(c0_i, c0_j),
                    "mean_abs_diff": (math.fsum(diffs) / len(diffs)) if diffs else 0.0,
                    "max_abs_diff": max(diffs) if diffs else 0.0,
                }
            )

    per_item: list[dict] = []
    if n_classes >= 3:
        for k, it in enumerate(norm_items):
            rhos = []
            for i in range(len(labelings)):
                for j in range(i + 1, len(labelings)):
                    rhos.append(spearman_rho(per_labeling[i][k], per_labeling[j][k]))
            usable = [r for r in rhos if not math.isnan(r)]
            per_item.append(
                {
                    "id": it["id"],
                    "rho_min": min(usable) if usable else float("nan"),
                    "rho_mean": (math.fsum(usable) / len(usable)) if usable else float("nan"),
                }
            )

    considered = [p["rho_flat"] for p in pairs if not math.isnan(p["rho_flat"])]
    min_rho = min(considered) if considered else float("nan")
    return {
        "model": model,
        "n_items": len(norm_items),
        "n_classes": n_classes,
        "labelings": [list(lab) for lab in labelings],
        "pairs": pairs,
        "per_item": per_item,
        "min_rho": min_rho,
        "gate": gate,
        "passed": bool(considered) and min_rho > gate,
        "note": (
            "rho_flat correlates per-item, per-class propensities across the two "
            "vocabularies; the pre-registered gate is rho > %.2f. A failure means "
            "the label strings, not the scenario, are driving the measurement."
            % gate
        ),
    }


# ===========================================================================
# Gate 3 -- variance decomposition (the T0-B pilot statistic)
# ===========================================================================

def _as_item_map(per_item_traces: Any) -> dict[str, list[float]]:
    if isinstance(per_item_traces, Mapping):
        return {str(k): [float(x) for x in v] for k, v in per_item_traces.items()}
    return {
        f"item{i:04d}": [float(x) for x in row]
        for i, row in enumerate(per_item_traces)
    }


def variance_decomposition(
    per_item_traces: Any,
    *,
    k: int = 1,
    ddof: int = 1,
) -> dict:
    """Split propensity variance into within-trace and between-trace parts.

    Input: per-item, per-trace propensities q[i][t] in [0, 1] -- the exact
    probability that item i under trace t yields the target verdict.  Either a
    mapping item_id -> [q, ...] or a sequence of sequences.

    For each item, over its traces:
        within_i  = E_t[ q(1-q) ]        the Bernoulli sampling variance you
                                         pay when you *draw* a verdict instead
                                         of computing its probability
        between_i = Var_t(q)             genuine trace-to-trace variation
        total_i   = within_i + between_i (law of total variance for one draw)

    A k-sample binary estimator has per-trace variance between + within/k; the
    exact estimator has between only.  So the realised efficiency gain is

        gain = 1 + E[q(1-q)] / (k * sigma^2_trace)

    which for k = 1 is exactly the pre-registered statistic
    ``1 + E[q(1-q)]/sigma^2_trace``.

    PLAINLY: if traces are highly determinative -- q sits at 0 or 1 once the
    trace is fixed -- then E[q(1-q)] -> 0 and the gain -> 1, i.e. exact scoring
    buys nothing over just sampling one completion per trace.  That is not a
    bug in this helper; measuring which regime we are actually in is the entire
    point of the T0-B pilot.  A large gain only appears when the model is
    genuinely undecided *given* the trace.

    Items with fewer than two traces cannot support a between-trace variance
    and are reported under ``skipped_items``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    data = _as_item_map(per_item_traces)
    if not data:
        raise ValueError("per_item_traces must be non-empty")
    for item_id, qs in data.items():
        for q in qs:
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"propensity out of [0,1] for {item_id}: {q}")

    items: dict[str, dict] = {}
    skipped: list[str] = []
    withins: list[float] = []
    betweens: list[float] = []
    for item_id, qs in data.items():
        n = len(qs)
        if n < 2:
            skipped.append(item_id)
            continue
        mean_q = math.fsum(qs) / n
        within = math.fsum(q * (1.0 - q) for q in qs) / n
        denom = max(n - ddof, 1)
        between = math.fsum((q - mean_q) ** 2 for q in qs) / denom
        gain = (1.0 + within / (k * between)) if between > 0.0 else float("inf")
        items[item_id] = {
            "n_traces": n,
            "mean_q": mean_q,
            "within_trace": within,
            "between_trace": between,
            "total": within + between,
            "efficiency_gain": gain,
            "determinative": between > 0.0 and within / between < 0.05,
        }
        withins.append(within)
        betweens.append(between)

    if not items:
        raise ValueError("no item had >= 2 traces; cannot decompose variance")

    pooled_within = math.fsum(withins) / len(withins)
    pooled_between = math.fsum(betweens) / len(betweens)
    pooled_gain = (
        1.0 + pooled_within / (k * pooled_between)
        if pooled_between > 0.0
        else float("inf")
    )
    if math.isinf(pooled_gain) and pooled_within == 0.0:
        pooled_gain = 1.0  # no variance anywhere: degenerate, not infinitely efficient

    if math.isinf(pooled_gain):
        verdict = (
            "between-trace variance is zero: every trace pins the same propensity, "
            "so the sampling noise is all there is and exact scoring removes it entirely"
        )
    elif pooled_gain < 1.10:
        verdict = (
            "traces are highly determinative -- exact scoring buys <10%% effective "
            "sample size over 1-sample binary scoring (gain %.3f)" % pooled_gain
        )
    elif pooled_gain < 2.0:
        verdict = "modest gain: exact scoring is worth %.2f binary samples per trace" % pooled_gain
    else:
        verdict = (
            "substantial gain: exact scoring is worth %.2f binary samples per trace, "
            "so the model is genuinely undecided given the trace" % pooled_gain
        )

    return {
        "k": k,
        "ddof": ddof,
        "n_items": len(items),
        "skipped_items": skipped,
        "pooled_within_trace": pooled_within,
        "pooled_between_trace": pooled_between,
        "pooled_total": pooled_within + pooled_between,
        "efficiency_gain": pooled_gain,
        "equivalent_traces_multiplier": pooled_gain,
        "items": items,
        "interpretation": verdict,
    }


# ===========================================================================
# Cross-backend agreement
# ===========================================================================

def compare_propensities(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    tol: float = 1e-2,
) -> dict:
    """Compare two simplices over the same support (pure; unit-tested offline)."""
    keys = sorted(set(a) | set(b))
    diffs = [abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys]
    max_abs = max(diffs) if diffs else 0.0
    tv = 0.5 * math.fsum(diffs)
    return {
        "keys": keys,
        "max_abs_diff": max_abs,
        "total_variation": tv,
        "argmax_a": max(a, key=lambda k: a[k]) if a else None,
        "argmax_b": max(b, key=lambda k: b[k]) if b else None,
        "argmax_agree": (
            (max(a, key=lambda k: a[k]) == max(b, key=lambda k: b[k])) if a and b else False
        ),
        "tol": tol,
        "agree": max_abs <= tol,
    }


def cross_backend_check(
    model: str,
    system: str,
    user: str,
    *,
    verdicts: Sequence[str],
    trace: str = "",
    prefix: str = DEFAULT_PREFIX,
    tol: float = 1e-2,
    **kwargs: Any,
) -> dict:
    """Run the same cell through hf and vllm and report agreement.

    Both backends share the numerical core, so a disagreement means a
    conditioning difference (chat template, special tokens, dtype) rather than
    a maths difference.  hf is the arbiter.
    """
    kwargs.pop("backend", None)
    rec_hf = verdict_propensity_record(
        model, system, user, verdicts=verdicts, trace=trace, prefix=prefix,
        backend="hf", **kwargs,
    )
    rec_vllm = verdict_propensity_record(
        model, system, user, verdicts=verdicts, trace=trace, prefix=prefix,
        backend="vllm", **kwargs,
    )
    cmp = compare_propensities(rec_hf.probs, rec_vllm.probs, tol=tol)
    ref = list(verdicts)[0]
    gap_hf = {v: rec_hf.logprob_sums[v] - rec_hf.logprob_sums[ref] for v in verdicts}
    gap_vllm = {v: rec_vllm.logprob_sums[v] - rec_vllm.logprob_sums[ref] for v in verdicts}
    max_gap_diff = max(abs(gap_hf[v] - gap_vllm[v]) for v in verdicts)
    return {
        "model": model,
        "hf": rec_hf.to_dict(),
        "vllm": rec_vllm.to_dict(),
        "comparison": cmp,
        "max_logprob_gap_diff": max_gap_diff,
        "token_counts_match": rec_hf.n_scored_tokens == rec_vllm.n_scored_tokens,
        "agree": bool(cmp["agree"] and rec_hf.n_scored_tokens == rec_vllm.n_scored_tokens),
    }


# ===========================================================================
# CLI
# ===========================================================================

def _smoke() -> int:
    """Offline self-check of the numerical core -- no model, no network."""
    print("propensity smoke (offline, no model):")

    # Simplex renormalisation.
    sums = {"YTA": -1.0, "NTA": -2.0, "UNRESOLVED": -8.0}
    probs = renormalise(sums)
    assert abs(sum(probs.values()) - 1.0) < 1e-12, probs
    assert probs["YTA"] > probs["NTA"] > probs["UNRESOLVED"]
    print(f"  renormalise -> {({k: round(v, 5) for k, v in probs.items()})}")

    # No length normalisation: equal totals -> equal probability.
    cand_ids = {"A": [7, 11], "LONGER": [7, 21, 22, 23]}
    cand_lps = {"A": [None, -1.0], "LONGER": [None, -0.4, -0.3, -0.3]}
    core = propensity_from_token_logprobs(["A", "LONGER"], cand_ids, cand_lps)
    assert abs(core["probs"]["A"] - core["probs"]["LONGER"]) < 1e-12, core["probs"]
    print(f"  no-length-normalisation -> {({k: round(v, 6) for k, v in core['probs'].items()})}")

    # Variance decomposition, hand-checkable.
    vd = variance_decomposition({"det": [0.0, 1.0], "undecided": [0.2, 0.8]})
    print(
        "  variance: within=%.4f between=%.4f gain=%.4f"
        % (vd["pooled_within_trace"], vd["pooled_between_trace"], vd["efficiency_gain"])
    )
    print(f"  {vd['interpretation']}")
    print("smoke OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[1])
    ap.add_argument("--smoke", action="store_true", help="offline numerical self-check")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--instrument", default="aita", help=f"one of {sorted(_VF_INSTRUMENTS)}")
    ap.add_argument("--verdicts", default="", help="comma-separated override for --instrument")
    ap.add_argument("--no-unresolved", action="store_true")
    ap.add_argument("--system", default="")
    ap.add_argument("--user", default="")
    ap.add_argument("--trace", default="")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--suffix", default="", help="opt-in verdict terminator (e.g. '\\n')")
    ap.add_argument("--backend", default=None, choices=["hf", "vllm", "auto"])
    ap.add_argument("--chat-template", default="auto", choices=["auto", "plain", "chat"])
    ap.add_argument("--item-id", default="cli")
    ap.add_argument("--audit", action="store_true", help="run the tokenization audit and exit")
    ap.add_argument("--cross-backend", action="store_true", help="run hf vs vllm agreement check")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        return _smoke()

    if args.verdicts.strip():
        verdicts = tuple(v.strip() for v in args.verdicts.split(",") if v.strip())
    else:
        verdicts = verdicts_for_instrument(
            args.instrument, allow_unresolved=not args.no_unresolved
        )

    if args.audit:
        rep = tokenization_audit(args.model, verdicts, prefix=args.prefix)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0 if rep["passed"] else 1

    if not args.user.strip():
        ap.error("--user is required (or use --smoke / --audit)")

    if args.cross_backend:
        rep = cross_backend_check(
            args.model, args.system, args.user, verdicts=verdicts,
            trace=args.trace, prefix=args.prefix, suffix=args.suffix,
            chat_template=args.chat_template, item_id=args.item_id,
            use_cache=not args.no_cache,
        )
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0 if rep["agree"] else 1

    rec = verdict_propensity_record(
        args.model, args.system, args.user, verdicts=verdicts,
        trace=args.trace, prefix=args.prefix, suffix=args.suffix,
        backend=args.backend, chat_template=args.chat_template,
        item_id=args.item_id, use_cache=not args.no_cache,
    )
    print(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
