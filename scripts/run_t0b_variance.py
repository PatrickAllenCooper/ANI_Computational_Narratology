"""
scripts/run_t0b_variance.py -- T0-B variance-components pilot.

THE HIGHEST-PRIORITY MEASUREMENT IN THE PROGRAM.

Every sample size in both proposal documents (the scenario redesign and the
audience/paraphrase taxonomy) rests on one unmeasured assumption: that scoring a
generated trace with an *exact* verdict propensity q_t = P(verdict | trace) buys
5-20x the statistical efficiency of drawing k binary samples and counting hits.
This module measures the realised gain instead of assuming it.

The decomposition
-----------------
For one item, let q_t = P(target verdict | trace t) be the exact propensity under
trace t, with traces drawn i.i.d. from the model's trace distribution.  Write
p = E_t[q_t] for the item's marginal verdict probability.  Then, exactly,

    p(1 - p) = E_t[q_t(1 - q_t)]  +  Var_t(q_t)
               \\_ within-trace _/    \\_ between-trace _/

A k-sample binary estimator of p has variance p(1-p)/k.  An m-trace exact-scored
estimator has variance Var_t(q)/m.  At equal generation count the efficiency gain
is therefore

    G = p(1 - p) / Var_t(q) = 1 + E[q(1-q)] / Var_t(q)

If traces are highly determinative -- the likely case for scaffolded CoT/NoT on a
forced verdict line -- q is bimodal, E[q(1-q)] -> 0, Var_t(q) -> p(1-p), and
G -> 1: exact scoring buys nothing per generation and every n in the program must
be re-planned.  If traces leave the verdict genuinely open, G is large.

Estimators used here (both unbiased for their target, per item):
    s2      = sample variance of q over traces (ddof=1)   -> Var_t(q)
    w       = mean of q_t(1 - q_t) over traces            -> E[q(1-q)]
    pq_hat  = w + s2                                      -> p(1-p)
    gain_i  = pq_hat / s2 = 1 + w / s2
The headline is the POOLED ratio (sum of numerators / sum of denominators), not
the mean of per-item ratios: per-item ratios are unbounded when s2 is small and
their mean is dominated by near-degenerate items.  Item-level bootstrap CIs.

Pre-gates (both must pass before any gain is reported)
------------------------------------------------------
  1. tokenization audit  -- scripts.propensity.tokenization_audit
  2. label-bias check    -- scripts.propensity.label_bias_check, recomputing the
     propensities under three verdict vocabularies and requiring per-item
     Spearman rho above --rho-threshold (default 0.8)
If either gate fails, the summary is written with status GATES_FAILED and the
gain fields are null.  An indeterminate gate (the gate returned a dict this
module cannot interpret) also withholds the gain unless
--allow-indeterminate-gates is passed, which is recorded in the summary.

Exact propensities require logprob access, i.e. a local open-weight model routed
through scripts.local_backend.  Hosted quartet models are accepted only with
--allow-hosted and are flagged in the output; their "exact" propensities are
whatever scripts.propensity can produce for them.

Outputs
-------
  divergence_study_outputs/t0b_variance_summary.json   (the consumable summary)
  divergence_study_outputs/t0b_per_item_{safe(model)}.json
Per-cell caches (one JSON per cell, existing repo convention, never invalidated):
  divergence_study_outputs/t0b_trace_{safe(model)}_{safe(item_id)}_{idx:02d}.json
  divergence_study_outputs/t0b_prop_{safe(model)}_{safe(item_id)}_{idx:02d}.json

Usage
-----
  python -m scripts.run_t0b_variance --selftest          # offline, no keys, no GPU
  python -m scripts.run_t0b_variance --smoke             # offline synthetic end-to-end
  python -m scripts.run_t0b_variance --models Qwen3-8B --n-items 100 --m-traces 8
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.run_phase1_quartet import PROMPTS, _safe, OUT_DIR

# --------------------------------------------------------------------------
# Contract modules owned by other agents.  Guarded so this module always
# imports on a laptop with no GPU, no API keys, and no sibling modules yet.
# --------------------------------------------------------------------------

try:  # scripts/verdict_format.py
    from scripts.verdict_format import (  # type: ignore
        INSTRUMENTS,
        NOVERDICT,
        extract_verdict,
        is_noncommittal,
        verdict_instruction,
        verdict_tokens,
    )
    _HAVE_VERDICT_FORMAT = True
except Exception:  # pragma: no cover - exercised only before that module lands
    _HAVE_VERDICT_FORMAT = False
    INSTRUMENTS = {}  # type: ignore[assignment]
    NOVERDICT = "NOVERDICT"
    extract_verdict = None  # type: ignore[assignment]
    is_noncommittal = None  # type: ignore[assignment]
    verdict_instruction = None  # type: ignore[assignment]
    verdict_tokens = None  # type: ignore[assignment]

try:  # scripts/propensity.py
    from scripts.propensity import (  # type: ignore
        label_bias_check,
        tokenization_audit,
        verdict_propensity,
    )
    _HAVE_PROPENSITY = True
except Exception:  # pragma: no cover
    _HAVE_PROPENSITY = False
    label_bias_check = None  # type: ignore[assignment]
    tokenization_audit = None  # type: ignore[assignment]
    verdict_propensity = None  # type: ignore[assignment]

try:  # independent second implementation of the same decomposition
    from scripts.propensity import variance_decomposition as _propensity_vardecomp  # type: ignore
except Exception:  # pragma: no cover
    _propensity_vardecomp = None  # type: ignore[assignment]

try:  # scripts/local_backend.py
    from scripts.local_backend import (  # type: ignore
        LOCAL_MODELS,
        generate_any,
        is_local_model,
    )
    _HAVE_LOCAL_BACKEND = True
except Exception:  # pragma: no cover
    _HAVE_LOCAL_BACKEND = False
    LOCAL_MODELS = {}  # type: ignore[assignment]
    generate_any = None  # type: ignore[assignment]
    is_local_model = None  # type: ignore[assignment]


OUT_DIR.mkdir(exist_ok=True)

SUMMARY_PATH = OUT_DIR / "t0b_variance_summary.json"
SCHEMA_VERSION = "t0b/1"

# The proposals' assumed efficiency gain.  Both documents plan on 5-20x; the
# planning default here is the geometric-ish middle of that band.
ASSUMED_GAIN_DEFAULT = 10.0
ASSUMED_GAIN_BAND = (5.0, 20.0)

# Per-cell measurement SE the program wants on the propensity scale.
TARGET_SE_DEFAULT = 0.05

# Below this, the between-trace variance is treated as numerically degenerate.
DEGENERATE_S2 = 1e-9

DEFAULT_INSTRUMENT = "brokenmath"
DEFAULT_SCAFFOLD = "standard_cot"

# Alternative verdict vocabularies for the label-bias gate.  Position-matched to
# the instrument's canonical tuple; the third slot stays the non-commitment label
# so the gate never silently drops the UNRESOLVED option.
ALT_LABELINGS: dict[str, list[tuple[str, ...]]] = {
    "brokenmath": [("A", "B", "UNRESOLVED"), ("1", "2", "UNRESOLVED")],
    "aita": [
        ("A", "B", "C", "D", "UNRESOLVED"),
        ("1", "2", "3", "4", "UNRESOLVED"),
    ],
    "ledger": [("A", "B", "UNRESOLVED"), ("1", "2", "UNRESOLVED")],
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _require(flag: bool, module: str, what: str) -> None:
    if not flag:
        raise RuntimeError(
            f"{what} needs {module}, which is not importable yet. "
            f"That module is owned by another agent; run --selftest or --smoke "
            f"until it lands."
        )


_VERDICT_LINE_RE = re.compile(r"(?im)^\s*VERDICT\s*:\s*[A-Za-z0-9_\- ]+\s*$")


def strip_trailing_verdict(text: str) -> tuple[str, bool]:
    """Remove a trailing ``VERDICT: X`` line from a generated trace.

    The propensity must be measured on a trace that has NOT already committed to
    a verdict; otherwise q_t is trivially 1 for whatever the model already wrote
    and the between-trace variance is an artefact of the format instruction.
    Returns (trace_without_verdict, had_explicit_verdict).
    """
    if not text:
        return "", False
    matches = list(_VERDICT_LINE_RE.finditer(text))
    if not matches:
        return text.strip(), False
    last = matches[-1]
    # Only strip if it is genuinely trailing (nothing but whitespace after it).
    if text[last.end():].strip():
        return text.strip(), True
    return text[: last.start()].rstrip(), True


def trace_cache_path(model: str, item_id: str, idx: int) -> Path:
    return OUT_DIR / f"t0b_trace_{_safe(model)}_{_safe(item_id)}_{idx:02d}.json"


def prop_cache_path(model: str, item_id: str, idx: int) -> Path:
    return OUT_DIR / f"t0b_prop_{_safe(model)}_{_safe(item_id)}_{idx:02d}.json"


def instrument_verdicts(instrument: str, *, allow_unresolved: bool = True) -> tuple[str, ...]:
    """Legal verdict tuple for an instrument, via the contract module."""
    _require(_HAVE_VERDICT_FORMAT, "scripts.verdict_format", "verdict vocabulary lookup")
    if verdict_tokens is not None:
        return tuple(verdict_tokens(instrument, allow_unresolved=allow_unresolved))
    return tuple(INSTRUMENTS[instrument])


# --------------------------------------------------------------------------
# Variance decomposition (pure, offline, self-tested)
# --------------------------------------------------------------------------

def decompose_item(q: Sequence[float]) -> dict:
    """Variance components for one item from its per-trace propensities.

    Args:
        q: propensities q_t = P(target verdict | trace t), one per trace.

    Returns a dict with:
        m            number of traces
        p_hat        mean propensity (the item's marginal verdict probability)
        s2_between   unbiased between-trace variance of q  (Var_t(q))
        w_within     mean of q(1-q)                        (E[q(1-q)])
        pq_unbiased  w_within + s2_between                 (p(1-p))
        gain         pq_unbiased / s2_between, or None when degenerate
        degenerate   True when s2_between is numerically zero
    """
    vals = [float(x) for x in q]
    m = len(vals)
    if m < 2:
        raise ValueError("variance decomposition needs at least 2 traces per item")
    for v in vals:
        if not (0.0 - 1e-9 <= v <= 1.0 + 1e-9):
            raise ValueError(f"propensity out of [0,1]: {v}")
    p_hat = statistics.fmean(vals)
    s2 = statistics.variance(vals)  # ddof=1, unbiased for Var_t(q)
    w = statistics.fmean([v * (1.0 - v) for v in vals])  # unbiased for E[q(1-q)]
    pq = w + s2
    degenerate = s2 <= DEGENERATE_S2
    return {
        "m": m,
        "p_hat": p_hat,
        "s2_between": s2,
        "w_within": w,
        "pq_unbiased": pq,
        "gain": (None if degenerate else pq / s2),
        "degenerate": degenerate,
    }


def pooled_gain(components: Sequence[dict]) -> Optional[float]:
    """Pooled efficiency gain: sum(p(1-p)) / sum(Var_t(q)) over items.

    Returns None when every item is degenerate (zero between-trace variance),
    which means exact scoring has zero sampling variance and the ratio is not a
    finite number -- a result that must be reported as such, not as a huge gain.
    """
    num = sum(c["pq_unbiased"] for c in components)
    den = sum(c["s2_between"] for c in components)
    if den <= DEGENERATE_S2:
        return None
    return num / den


def bootstrap_pooled_gain(
    components: Sequence[dict],
    *,
    n_boot: int = 2000,
    seed: int = 7,
    alpha: float = 0.05,
) -> Optional[tuple[float, float]]:
    """Item-level percentile bootstrap CI for the pooled gain."""
    if len(components) < 3:
        return None
    rng = random.Random(seed)
    n = len(components)
    draws: list[float] = []
    for _ in range(n_boot):
        sample = [components[rng.randrange(n)] for _ in range(n)]
        g = pooled_gain(sample)
        if g is not None:
            draws.append(g)
    if len(draws) < max(20, n_boot // 10):
        return None
    draws.sort()
    lo = draws[max(0, int(math.floor((alpha / 2) * len(draws))))]
    hi = draws[min(len(draws) - 1, int(math.ceil((1 - alpha / 2) * len(draws))) - 1)]
    return (lo, hi)


def recommendation_block(
    components: Sequence[dict],
    *,
    realised_gain: Optional[float],
    assumed_gain: float,
    target_se: float,
) -> dict:
    """Turn the decomposition into the numbers the rest of the program consumes.

    recommended_generations_per_cell
        traces per cell needed to reach ``target_se`` under exact scoring
        (ceil(mean Var_t(q) / target_se^2)), floor 1.
    equivalent_k_binary
        the k that k-sample binary scoring would need for the same SE
        (ceil(mean p(1-p) / target_se^2)).
    n_multiplier
        how much every pre-planned n must be inflated because the realised gain
        fell short of the assumed one: assumed_gain / realised_gain.  This is an
        UPPER BOUND on the required inflation -- it is exact only when per-cell
        measurement variance dominates the item-level total variance; whenever
        the true between-item shift variance is non-zero the needed inflation is
        smaller.  Reported as such, and clipped at 1.0 in n_multiplier_planning.
    """
    n_items = len(components)
    mean_s2 = statistics.fmean([c["s2_between"] for c in components]) if n_items else 0.0
    mean_pq = statistics.fmean([c["pq_unbiased"] for c in components]) if n_items else 0.0
    se2 = target_se * target_se

    m_exact = max(1, int(math.ceil(mean_s2 / se2))) if se2 > 0 else None
    k_binary = max(1, int(math.ceil(mean_pq / se2))) if se2 > 0 else None

    block: dict[str, Any] = {
        "target_per_cell_se": target_se,
        "recommended_generations_per_cell": m_exact,
        "equivalent_k_binary": k_binary,
        "mean_between_trace_variance": mean_s2,
        "mean_p_times_1_minus_p": mean_pq,
        "assumed_gain": assumed_gain,
        "assumed_gain_band": list(ASSUMED_GAIN_BAND),
    }
    if realised_gain is None or realised_gain <= 0:
        block["n_multiplier"] = None
        block["n_multiplier_planning"] = None
        block["n_multiplier_vs_band"] = None
        block["interpretation"] = (
            "Between-trace variance is numerically zero on every item: traces do "
            "not move the verdict propensity at all. Exact scoring has no "
            "sampling variance here, so the ratio is undefined rather than large. "
            "Re-check the trace sampling temperature before consuming this."
        )
        return block

    block["n_multiplier"] = assumed_gain / realised_gain
    block["n_multiplier_planning"] = max(1.0, assumed_gain / realised_gain)
    block["n_multiplier_vs_band"] = {
        "vs_5x": max(1.0, ASSUMED_GAIN_BAND[0] / realised_gain),
        "vs_20x": max(1.0, ASSUMED_GAIN_BAND[1] / realised_gain),
    }
    if realised_gain < 1.5:
        verdict = (
            "Traces are effectively determinative: exact scoring buys under 1.5x "
            "per generation. Every n in both proposals must be re-planned as if "
            "k-sample binary scoring were being used."
        )
    elif realised_gain < ASSUMED_GAIN_BAND[0]:
        verdict = (
            "Realised gain is below the 5x floor both proposals assumed. Inflate "
            "planned n by n_multiplier_planning (an upper bound) before "
            "committing compute."
        )
    elif realised_gain <= ASSUMED_GAIN_BAND[1]:
        verdict = "Realised gain is inside the assumed 5-20x band; planned n stand."
    else:
        verdict = (
            "Realised gain exceeds the assumed band; planned n are conservative. "
            "Do not shrink n on this evidence alone -- the pilot measures "
            "measurement variance only, not between-item shift variance."
        )
    block["interpretation"] = verdict
    return block


# --------------------------------------------------------------------------
# Pre-gates
# --------------------------------------------------------------------------

_TRUE_KEYS = ("ok", "passed", "pass", "clean", "all_single_token", "single_token_all")
_PROBLEM_KEYS = ("collisions", "ambiguous", "problems", "errors", "failures", "conflicts")
_RHO_KEYS = ("min_rho", "min_spearman", "min_spearman_rho", "spearman_min", "rho_min")


def _interpret_boolean_gate(raw: Any) -> tuple[str, str]:
    """Map an audit dict onto pass / fail / indeterminate.

    Deliberately conservative: a dict this module cannot read is *indeterminate*,
    never a silent pass.  The raw dict is always retained in the summary so a
    human can adjudicate.
    """
    if not isinstance(raw, dict):
        return "indeterminate", f"audit returned {type(raw).__name__}, not a dict"
    for key in _TRUE_KEYS:
        if key in raw and isinstance(raw[key], bool):
            return ("pass" if raw[key] else "fail"), f"{key}={raw[key]}"
    for key in _PROBLEM_KEYS:
        if key in raw:
            val = raw[key]
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                return ("fail" if val else "pass"), f"{key}={val}"
            if isinstance(val, (list, tuple, set, dict)):
                return ("fail" if len(val) else "pass"), f"{key} has {len(val)} entries"
    return "indeterminate", f"no interpretable status key in {sorted(raw)[:8]}"


def _min_rho(raw: Any) -> Optional[float]:
    """Pull the minimum per-item Spearman rho out of a label-bias result."""
    if not isinstance(raw, dict):
        return None
    for key in _RHO_KEYS:
        val = raw.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    for key in ("spearman", "rho", "per_item_rho", "rhos", "spearman_rho"):
        val = raw.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            nums = [float(v) for v in val.values() if isinstance(v, (int, float))]
            if nums:
                return min(nums)
        if isinstance(val, (list, tuple)):
            nums = [float(v) for v in val if isinstance(v, (int, float))]
            if nums:
                return min(nums)
    return None


def _interpret_label_bias(raw: Any, threshold: float) -> tuple[str, str]:
    """Pass only if the audit passed AND min rho clears the threshold set here.

    The audit carries its own gate constant; this pilot's --rho-threshold may be
    stricter, and the stricter of the two must win.  Never the looser.
    """
    bool_status, bool_detail = _interpret_boolean_gate(raw)
    rho = _min_rho(raw)
    if rho is None:
        return bool_status, bool_detail
    if math.isnan(rho):
        return "fail", "min Spearman rho is NaN: no usable vocabulary pair was compared"
    detail = f"min Spearman rho={rho:.4f} vs threshold {threshold}"
    if bool_status == "fail":
        return "fail", f"{detail}; audit itself reported failure ({bool_detail})"
    return ("pass" if rho > threshold else "fail"), detail


def run_gates(
    model: str,
    verdicts: Sequence[str],
    items: Sequence[Any],
    *,
    instrument: str,
    rho_threshold: float,
) -> dict:
    """Run the tokenization audit and the label-bias check for one model."""
    _require(_HAVE_PROPENSITY, "scripts.propensity", "the T0-B pre-gates")
    out: dict[str, Any] = {"model": model, "rho_threshold": rho_threshold}

    try:
        tok_raw = tokenization_audit(model, list(verdicts))
        tok_status, tok_detail = _interpret_boolean_gate(tok_raw)
    except Exception as e:
        tok_raw, tok_status, tok_detail = {"exception": str(e)}, "fail", f"raised: {e}"
    out["tokenization_audit"] = {
        "status": tok_status, "detail": tok_detail, "raw": _jsonable(tok_raw),
    }

    labelings: list[Sequence[str]] = [tuple(verdicts)]
    for alt in ALT_LABELINGS.get(instrument, []):
        if len(alt) == len(verdicts):
            labelings.append(alt)
    try:
        lab_raw = label_bias_check(model, list(items), labelings)
        lab_status, lab_detail = _interpret_label_bias(lab_raw, rho_threshold)
    except Exception as e:
        lab_raw, lab_status, lab_detail = {"exception": str(e)}, "fail", f"raised: {e}"
    out["label_bias_check"] = {
        "status": lab_status,
        "detail": lab_detail,
        "labelings": [list(x) for x in labelings],
        "raw": _jsonable(lab_raw),
    }

    statuses = {out["tokenization_audit"]["status"], out["label_bias_check"]["status"]}
    if "fail" in statuses:
        out["overall"] = "fail"
    elif "indeterminate" in statuses:
        out["overall"] = "indeterminate"
    else:
        out["overall"] = "pass"
    return out


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of a foreign audit payload to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return repr(obj)


# --------------------------------------------------------------------------
# Items and prompts
# --------------------------------------------------------------------------

def load_pilot_items(instrument: str, n: int, seed: int) -> list[dict]:
    """Load n pilot items as plain dicts: {id, user_text, gold_verdict}."""
    if instrument == "brokenmath":
        from scripts.load_brokenmath import load_brokenmath
        raw = load_brokenmath(n=n, seed=seed)
        return [
            {
                "id": it.problem_id,
                "user_text": it.problem,
                # BrokenMath's adversarial rows are all false statements.
                "gold_verdict": "FALSE" if it.is_adversarial else "UNRESOLVED",
            }
            for it in raw
        ]
    if instrument == "aita":
        from scripts.load_elephant import load_elephant
        raw = load_elephant("aita", n=n, allow_sample=True)
        out = []
        for it in raw:
            gold = getattr(it, "gold", "") or getattr(it, "ytanta", "") or "UNRESOLVED"
            out.append({
                "id": getattr(it, "id", ""),
                "user_text": getattr(it, "prompt", "") or getattr(it, "text", ""),
                "gold_verdict": str(gold).upper() or "UNRESOLVED",
            })
        return out
    raise ValueError(f"unsupported instrument for the T0-B pilot: {instrument!r}")


def build_user_text(item_text: str, instrument: str, *, allow_unresolved: bool) -> str:
    _require(_HAVE_VERDICT_FORMAT, "scripts.verdict_format", "forced-verdict prompts")
    instruction = verdict_instruction(instrument, allow_unresolved=allow_unresolved)
    return f"{item_text.strip()}\n\n{instruction.strip()}"


# --------------------------------------------------------------------------
# Trace collection and exact scoring
# --------------------------------------------------------------------------

def _model_is_local(model: str) -> bool:
    if _HAVE_LOCAL_BACKEND and is_local_model is not None:
        try:
            return bool(is_local_model(model))
        except Exception:
            return False
    return False


def generate_trace(
    model: str,
    system: str,
    user: str,
    *,
    item_id: str,
    idx: int,
    max_tokens: int,
    temperature: float,
    seed_base: int,
) -> dict:
    """Generate (or reuse a cached) trace for one (item, trace-index) cell."""
    cache = trace_cache_path(model, item_id, idx)
    if cache.exists():
        try:
            rec = json.loads(cache.read_text())
            if rec.get("text", "").strip():
                return rec
        except Exception:
            pass  # corrupt cache -> regenerate
    _require(_HAVE_LOCAL_BACKEND, "scripts.local_backend", "trace generation")
    kwargs: dict[str, Any] = {"sample_idx": idx, "max_tokens": max_tokens}
    if _model_is_local(model):
        kwargs["temperature"] = temperature
        kwargs["seed"] = seed_base + idx
    result = generate_any(model, system, user, **kwargs)
    text = (getattr(result, "text", "") or "").strip()
    rec = {
        "model": model,
        "item_id": item_id,
        "trace_idx": idx,
        "text": text,
        "finish_reason": getattr(result, "finish_reason", ""),
        "prompt_tokens": getattr(result, "prompt_tokens", 0),
        "completion_tokens": getattr(result, "completion_tokens", 0),
        "temperature": temperature,
        "is_local": _model_is_local(model),
    }
    cache.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def score_trace(
    model: str,
    system: str,
    user: str,
    trace_text: str,
    *,
    item_id: str,
    idx: int,
    verdicts: Sequence[str],
    target_verdict: str,
) -> dict:
    """Exact propensity for one trace, cached per cell."""
    cache = prop_cache_path(model, item_id, idx)
    if cache.exists():
        try:
            rec = json.loads(cache.read_text())
            if "propensities" in rec:
                return rec
        except Exception:
            pass
    _require(_HAVE_PROPENSITY, "scripts.propensity", "exact verdict propensities")
    trace, had_verdict = strip_trailing_verdict(trace_text)
    props = verdict_propensity(
        model, system, user, verdicts=list(verdicts), trace=trace, prefix="VERDICT: ",
    )
    props = {str(k): float(v) for k, v in dict(props).items()}
    rec = {
        "model": model,
        "item_id": item_id,
        "trace_idx": idx,
        "propensities": props,
        "q_target": float(props.get(target_verdict, 0.0)),
        "target_verdict": target_verdict,
        "trace_had_explicit_verdict": had_verdict,
        "trace_chars": len(trace),
    }
    cache.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def collect_item_propensities(
    model: str,
    item: dict,
    *,
    instrument: str,
    scaffold: str,
    m_traces: int,
    max_tokens: int,
    temperature: float,
    seed_base: int,
    verdicts: Sequence[str],
    allow_unresolved: bool,
) -> dict:
    """m traces + m exact scorings for one item.  Returns the per-item record."""
    system = PROMPTS.get(scaffold, PROMPTS["standard_cot"])
    user = build_user_text(item["user_text"], instrument, allow_unresolved=allow_unresolved)
    target = item.get("gold_verdict") or verdicts[0]

    qs: list[float] = []
    binaries: list[int] = []
    empties = 0
    for idx in range(m_traces):
        tr = generate_trace(
            model, system, user,
            item_id=item["id"], idx=idx,
            max_tokens=max_tokens, temperature=temperature, seed_base=seed_base,
        )
        text = tr.get("text", "")
        if not text.strip():
            empties += 1
            continue
        sc = score_trace(
            model, system, user, text,
            item_id=item["id"], idx=idx,
            verdicts=verdicts, target_verdict=target,
        )
        qs.append(float(sc["q_target"]))
        if _HAVE_VERDICT_FORMAT and extract_verdict is not None:
            binaries.append(int(extract_verdict(text, instrument) == target))
    return {
        "item_id": item["id"],
        "target_verdict": target,
        "q": qs,
        "binary_hits": binaries,
        "n_empty_traces": empties,
    }


# --------------------------------------------------------------------------
# Synthetic mode (offline smoke / self-test)
# --------------------------------------------------------------------------

def synthetic_propensities(
    n_items: int, m_traces: int, *, seed: int, determinative: float = 0.5,
) -> list[dict]:
    """Deterministic synthetic per-item propensity sets for offline exercise.

    ``determinative`` in [0,1] is the fraction of items whose traces pin the
    verdict (bimodal q, gain -> 1); the rest have diffuse q (gain >> 1).  This
    exists only so the arithmetic and the reporting path can be exercised with
    no GPU and no API keys.  Output produced from it is stamped synthetic=true
    and status SMOKE so it can never be mistaken for a measurement.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n_items):
        pinned = rng.random() < determinative
        qs = []
        for _ in range(m_traces):
            if pinned:
                qs.append(rng.uniform(0.97, 0.999) if rng.random() < 0.7
                          else rng.uniform(0.001, 0.03))
            else:
                qs.append(min(0.999, max(0.001, rng.gauss(0.6, 0.05))))
        out.append({
            "item_id": f"synth_{i:03d}",
            "target_verdict": "FALSE",
            "q": qs,
            "binary_hits": [int(rng.random() < q) for q in qs],
            "n_empty_traces": 0,
        })
    return out


# --------------------------------------------------------------------------
# Model-level analysis
# --------------------------------------------------------------------------

def analyse_model(
    per_item: Sequence[dict],
    *,
    assumed_gain: float,
    target_se: float,
    n_boot: int,
    seed: int,
    min_traces: int,
) -> dict:
    """Decompose, pool, bootstrap, and recommend for one model."""
    components: list[dict] = []
    skipped: list[str] = []
    for rec in per_item:
        qs = rec.get("q", [])
        if len(qs) < min_traces:
            skipped.append(rec.get("item_id", "?"))
            continue
        comp = decompose_item(qs)
        comp["item_id"] = rec.get("item_id")
        comp["target_verdict"] = rec.get("target_verdict")
        comp["binary_rate"] = (
            statistics.fmean(rec["binary_hits"]) if rec.get("binary_hits") else None
        )
        components.append(comp)

    if not components:
        return {
            "n_items_used": 0,
            "n_items_skipped": len(skipped),
            "realised_gain": None,
            "error": "no item had enough non-empty traces to decompose",
        }

    gain = pooled_gain(components)
    ci = bootstrap_pooled_gain(components, n_boot=n_boot, seed=seed) if gain is not None else None
    finite_gains = sorted(c["gain"] for c in components if c["gain"] is not None)
    n_degen = sum(1 for c in components if c["degenerate"])

    quantiles: dict[str, Optional[float]] = {"p25": None, "median": None, "p75": None}
    if finite_gains:
        quantiles["median"] = statistics.median(finite_gains)
        if len(finite_gains) >= 4:
            quantiles["p25"] = finite_gains[int(0.25 * (len(finite_gains) - 1))]
            quantiles["p75"] = finite_gains[int(0.75 * (len(finite_gains) - 1))]

    out: dict[str, Any] = {
        "n_items_used": len(components),
        "n_items_skipped": len(skipped),
        "skipped_item_ids": skipped[:20],
        "traces_per_item": statistics.fmean([c["m"] for c in components]),
        "realised_gain": gain,
        "realised_gain_ci95": list(ci) if ci else None,
        "per_item_gain_quantiles": quantiles,
        "n_degenerate_items": n_degen,
        "degenerate_fraction": n_degen / len(components),
        "components_pooled": {
            "sum_p_times_1_minus_p": sum(c["pq_unbiased"] for c in components),
            "sum_between_trace_variance": sum(c["s2_between"] for c in components),
            "sum_within_trace_variance": sum(c["w_within"] for c in components),
            "between_share": (
                sum(c["s2_between"] for c in components)
                / sum(c["pq_unbiased"] for c in components)
                if sum(c["pq_unbiased"] for c in components) > 0 else None
            ),
        },
        "mean_p_hat": statistics.fmean([c["p_hat"] for c in components]),
        "recommendation": recommendation_block(
            components,
            realised_gain=gain,
            assumed_gain=assumed_gain,
            target_se=target_se,
        ),
    }
    cross = cross_check_gain([c for c in per_item if len(c.get("q", [])) >= min_traces], gain)
    if cross is not None:
        out["cross_check"] = cross
    out["_components"] = components
    return out


def cross_check_gain(per_item: Sequence[dict], gain: Optional[float]) -> Optional[dict]:
    """Recompute the pooled gain with scripts.propensity's own decomposition.

    Two independent implementations of the same estimator agreeing is a real
    check on the single most load-bearing number in the program; a disagreement
    is a defect in one of them and must surface in the summary rather than be
    silently averaged away.
    """
    if _propensity_vardecomp is None or not per_item:
        return None
    try:
        other = _propensity_vardecomp(
            {r["item_id"]: r["q"] for r in per_item}, k=1, ddof=1,
        )
    except Exception as e:
        return {"available": True, "error": str(e)}
    theirs = None
    for key in ("pooled_efficiency_gain", "efficiency_gain", "pooled_gain", "gain"):
        val = other.get(key)
        if isinstance(val, (int, float)):
            theirs = float(val)
            break
    if theirs is None or gain is None:
        return {"available": True, "their_gain": theirs, "our_gain": gain, "agrees": None}
    agrees = math.isclose(theirs, gain, rel_tol=1e-6, abs_tol=1e-9)
    return {
        "available": True,
        "source": "scripts.propensity.variance_decomposition",
        "their_gain": theirs,
        "our_gain": gain,
        "abs_diff": abs(theirs - gain),
        "agrees": agrees,
        "note": None if agrees else (
            "the two implementations of the pooled efficiency gain DISAGREE; "
            "resolve before consuming either number"
        ),
    }


# --------------------------------------------------------------------------
# Self-test (offline, no keys, no GPU)
# --------------------------------------------------------------------------

def selftest() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"  OK   {name}")
        else:
            print(f"  FAIL {name} {detail}")
            failures.append(name)

    # Identity: E[q(1-q)] + Var(q) == p(1-p) up to the finite-sample correction.
    q = [0.1, 0.4, 0.5, 0.9, 0.75, 0.2]
    d = decompose_item(q)
    p = statistics.fmean(q)
    s2 = statistics.variance(q)
    w = statistics.fmean([x * (1 - x) for x in q])
    check("decompose identity", abs(d["pq_unbiased"] - (w + s2)) < 1e-12)
    check(
        "plugin p(1-p) relation",
        abs((w + s2 * (len(q) - 1) / len(q)) - p * (1 - p)) < 1e-12,
        f"{w + s2 * (len(q) - 1) / len(q)} vs {p * (1 - p)}",
    )
    check("gain formula", abs(d["gain"] - (1.0 + w / s2)) < 1e-12)

    # Determinative traces (bimodal q) -> gain approaches 1.
    bimodal = [0.999, 0.001, 0.998, 0.002, 0.999, 0.001]
    gb = decompose_item(bimodal)["gain"]
    check("bimodal gain ~ 1", 1.0 <= gb < 1.02, f"gain={gb}")

    # Diffuse traces (q clustered mid-range) -> large gain.
    diffuse = [0.50, 0.51, 0.49, 0.52, 0.48, 0.50]
    gd = decompose_item(diffuse)["gain"]
    check("diffuse gain large", gd > 100, f"gain={gd}")

    # Degenerate: zero between-trace variance.
    deg = decompose_item([0.5] * 5)
    check("degenerate flagged", deg["degenerate"] and deg["gain"] is None)
    check("pooled gain all-degenerate is None", pooled_gain([deg, deg]) is None)

    # Pooled gain is a ratio of sums, not a mean of ratios.
    comps = [decompose_item(bimodal), decompose_item(diffuse)]
    pg = pooled_gain(comps)
    num = sum(c["pq_unbiased"] for c in comps)
    den = sum(c["s2_between"] for c in comps)
    check("pooled = ratio of sums", abs(pg - num / den) < 1e-12)
    check("pooled below mean-of-ratios", pg < statistics.fmean([gb, gd]))

    # Bootstrap CI brackets the point estimate.
    synth = synthetic_propensities(40, 8, seed=1)
    comps2 = [decompose_item(r["q"]) for r in synth]
    g2 = pooled_gain(comps2)
    ci = bootstrap_pooled_gain(comps2, n_boot=400, seed=3)
    check("bootstrap returns CI", ci is not None)
    check("CI brackets point estimate", ci is not None and ci[0] <= g2 <= ci[1], f"{ci} vs {g2}")

    # Recommendation arithmetic.
    rec = recommendation_block(comps2, realised_gain=2.0, assumed_gain=10.0, target_se=0.05)
    check("n_multiplier = assumed/realised", abs(rec["n_multiplier"] - 5.0) < 1e-12)
    check("planning multiplier >= 1", rec["n_multiplier_planning"] >= 1.0)
    check("k_binary >= m_exact", rec["equivalent_k_binary"] >= rec["recommended_generations_per_cell"])
    rec_none = recommendation_block(comps2, realised_gain=None, assumed_gain=10.0, target_se=0.05)
    check("null gain -> null multiplier", rec_none["n_multiplier"] is None)

    # Cross-implementation agreement on the program's load-bearing number.
    if _propensity_vardecomp is not None:
        cc = cross_check_gain(synth, g2)
        check("cross-check available", cc is not None and cc.get("their_gain") is not None)
        check("two implementations agree", cc is not None and cc.get("agrees") is True,
              f"{cc}")
        check("analyse_model carries the cross-check",
              "cross_check" in analyse_model(
                  synth, assumed_gain=10.0, target_se=0.05, n_boot=0, seed=5, min_traces=2))
    else:
        print("  SKIP cross-check (scripts.propensity.variance_decomposition absent)")

    # Gate interpretation is conservative.
    check("bool gate pass", _interpret_boolean_gate({"ok": True})[0] == "pass")
    check("bool gate fail", _interpret_boolean_gate({"ok": False})[0] == "fail")
    check("collision gate fail", _interpret_boolean_gate({"collisions": ["a"]})[0] == "fail")
    check("collision gate pass", _interpret_boolean_gate({"collisions": []})[0] == "pass")
    check("unknown gate indeterminate", _interpret_boolean_gate({"foo": 1})[0] == "indeterminate")
    check("non-dict gate indeterminate", _interpret_boolean_gate(None)[0] == "indeterminate")
    check("rho above threshold passes", _interpret_label_bias({"min_rho": 0.91}, 0.8)[0] == "pass")
    check("rho below threshold fails", _interpret_label_bias({"min_rho": 0.5}, 0.8)[0] == "fail")
    check(
        "rho dict min used",
        _interpret_label_bias({"spearman": {"a": 0.95, "b": 0.4}}, 0.8)[0] == "fail",
    )
    # The stricter of (audit's own gate, --rho-threshold) must win, never the looser.
    check("stricter local threshold wins",
          _interpret_label_bias({"passed": True, "min_rho": 0.85}, 0.9)[0] == "fail")
    check("audit failure wins over a loose threshold",
          _interpret_label_bias({"passed": False, "min_rho": 0.85}, 0.5)[0] == "fail")
    check("both passing passes",
          _interpret_label_bias({"passed": True, "min_rho": 0.95}, 0.9)[0] == "pass")
    check("NaN rho fails",
          _interpret_label_bias({"passed": True, "min_rho": float("nan")}, 0.8)[0] == "fail")
    # An audit that raises is a failure to measure, hence a gate failure.
    if _HAVE_PROPENSITY:
        gates = run_gates(
            "definitely-not-a-real-model", ("TRUE", "FALSE", "UNRESOLVED"), [],
            instrument="brokenmath", rho_threshold=0.8,
        )
        check("unmeasurable gates fail closed", gates["overall"] == "fail", str(gates)[:200])

    # Verdict stripping.
    t, had = strip_trailing_verdict("reasoning here\nVERDICT: FALSE\n")
    check("verdict stripped", t == "reasoning here" and had)
    t2, had2 = strip_trailing_verdict("no verdict at all")
    check("no verdict untouched", t2 == "no verdict at all" and not had2)
    t3, had3 = strip_trailing_verdict("VERDICT: TRUE\nbut then more text")
    check("non-trailing verdict kept", "but then more text" in t3 and had3)

    # analyse_model end-to-end on synthetic data, and the refusal path.
    res = analyse_model(
        synth, assumed_gain=10.0, target_se=0.05, n_boot=200, seed=5, min_traces=2,
    )
    check("analyse_model produces gain", res["realised_gain"] is not None)
    check("analyse_model counts items", res["n_items_used"] == 40)
    short = analyse_model(
        [{"item_id": "x", "q": [0.5], "binary_hits": [1]}],
        assumed_gain=10.0, target_se=0.05, n_boot=10, seed=5, min_traces=2,
    )
    check("short item skipped", short["n_items_used"] == 0 and short["realised_gain"] is None)

    # The refusal contract: a failed gate must null out every route to the gain.
    full_block = analyse_model(
        synth, assumed_gain=10.0, target_se=0.05, n_boot=0, seed=5, min_traces=2,
    )
    full_block.pop("_components", None)
    withheld = _apply_gate_refusal(full_block, overall="fail", allow_indeterminate=False)
    check("failed gate withholds gain", withheld["realised_gain"] is None)
    check("failed gate records reason", bool(withheld.get("gain_withheld_reason")))
    check("failed gate withholds pooled components",
          withheld.get("components_pooled") is None)
    check("failed gate withholds the recommendation",
          all(withheld["recommendation"].get(f) is None for f in RECOMMENDATION_FIELDS))
    check("failed gate withholds the variance inputs",
          withheld["recommendation"].get("mean_between_trace_variance") is None)
    if "cross_check" in withheld:
        check("failed gate withholds the cross-check gains",
              all(withheld["cross_check"].get(f) is None for f in CROSS_CHECK_GAIN_FIELDS))
    check("no numeric gain survives anywhere in the withheld block",
          _no_gain_leak(withheld), json.dumps(withheld, default=str)[:400])
    kept = _apply_gate_refusal(
        {"realised_gain": 3.0, "recommendation": {"n_multiplier": 3.0}},
        overall="pass", allow_indeterminate=False,
    )
    check("passed gate keeps gain", kept["realised_gain"] == 3.0)
    indet = _apply_gate_refusal(
        {"realised_gain": 3.0, "recommendation": {}},
        overall="indeterminate", allow_indeterminate=False,
    )
    check("indeterminate withholds by default", indet["realised_gain"] is None)
    indet_ok = _apply_gate_refusal(
        {"realised_gain": 3.0, "recommendation": {}},
        overall="indeterminate", allow_indeterminate=True,
    )
    check("indeterminate override keeps gain", indet_ok["realised_gain"] == 3.0)

    print()
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("SELFTEST PASSED")
    return 0


# --------------------------------------------------------------------------
# Refusal logic
# --------------------------------------------------------------------------

GAIN_FIELDS = (
    "realised_gain",
    "realised_gain_ci95",
    "per_item_gain_quantiles",
    # The pooled component sums are one division away from the gain, so they are
    # withheld too. The raw per-trace propensities stay on disk at
    # per_item_path: collected data is never destroyed, only the report is.
    "components_pooled",
)
RECOMMENDATION_FIELDS = (
    "recommended_generations_per_cell",
    "equivalent_k_binary",
    "n_multiplier",
    "n_multiplier_planning",
    "n_multiplier_vs_band",
)
CROSS_CHECK_GAIN_FIELDS = ("their_gain", "our_gain", "abs_diff")

_LEAKY_KEYS = frozenset(
    GAIN_FIELDS + RECOMMENDATION_FIELDS + CROSS_CHECK_GAIN_FIELDS
    + ("mean_between_trace_variance", "mean_p_times_1_minus_p")
)


def _no_gain_leak(obj: Any) -> bool:
    """True when no numeric value survives under any gain-bearing key.

    Guards the refusal contract against a future field being added to a nested
    block and quietly re-exposing the number the gates said may not be reported.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _LEAKY_KEYS:
                if isinstance(v, bool):
                    pass
                elif isinstance(v, (int, float)):
                    return False
                elif isinstance(v, (list, tuple, dict)) and v:
                    return False
            if not _no_gain_leak(v):
                return False
        return True
    if isinstance(obj, (list, tuple)):
        return all(_no_gain_leak(v) for v in obj)
    return True


def _apply_gate_refusal(model_block: dict, *, overall: str, allow_indeterminate: bool) -> dict:
    """Null out every gain / recommendation field when the pre-gates did not pass.

    This is the hard requirement of the pilot: a realised-efficiency number
    measured through a tokenizer that splits verdict words, or under a verdict
    vocabulary the model is biased between, is not a measurement.  The raw
    variance components are left in place (they are data), but everything the
    rest of the program would consume is nulled with an explicit reason.
    """
    block = dict(model_block)
    if overall == "pass":
        return block
    if overall == "indeterminate" and allow_indeterminate:
        block["gain_gate_override"] = True
        block["gain_withheld_reason"] = None
        return block
    reason = (
        "pre-gates FAILED (tokenization audit and/or label-bias check); "
        "no efficiency gain may be reported"
        if overall == "fail" else
        "pre-gate results were INDETERMINATE and --allow-indeterminate-gates "
        "was not passed; no efficiency gain may be reported"
    )
    for f in GAIN_FIELDS:
        if f in block:
            block[f] = None
    rec = block.get("recommendation")
    if isinstance(rec, dict):
        rec = dict(rec)
        for f in RECOMMENDATION_FIELDS:
            if f in rec:
                rec[f] = None
        for f in ("mean_between_trace_variance", "mean_p_times_1_minus_p"):
            if f in rec:
                rec[f] = None
        rec["interpretation"] = reason
        block["recommendation"] = rec
    cc = block.get("cross_check")
    if isinstance(cc, dict):
        cc = dict(cc)
        for f in CROSS_CHECK_GAIN_FIELDS:
            if f in cc:
                cc[f] = None
        block["cross_check"] = cc
    block["gain_withheld_reason"] = reason
    return block


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

# The two primary 8B targets; the 32B/70B rungs are not pilot models.
PILOT_MODEL_PREFERENCE = ("llama-3.1-8b", "qwen3-8b")


def _default_models() -> list[str]:
    if _HAVE_LOCAL_BACKEND and LOCAL_MODELS:
        preferred = [m for m in PILOT_MODEL_PREFERENCE if m in LOCAL_MODELS]
        return preferred or sorted(LOCAL_MODELS.keys())[:2]
    return list(PILOT_MODEL_PREFERENCE)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="T0-B variance-components pilot: realised gain of exact "
                    "propensity scoring over k-sample binary scoring.",
    )
    ap.add_argument("--models", default="", help="Comma-separated models (default: local models)")
    ap.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="brokenmath | aita")
    ap.add_argument("--scaffold", default=DEFAULT_SCAFFOLD,
                    help=f"System-prompt arm from run_phase1_quartet.PROMPTS (default {DEFAULT_SCAFFOLD})")
    ap.add_argument("--n-items", type=int, default=100, help="Items in the pilot (default 100)")
    ap.add_argument("--m-traces", type=int, default=8, help="Traces per item (default 8, min 2)")
    ap.add_argument("--min-traces", type=int, default=3,
                    help="Minimum usable traces before an item enters the decomposition")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Trace sampling temperature. MUST match the temperature the "
                         "production runs use, or the decomposition describes a "
                         "different trace distribution than the one being planned for.")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--assumed-gain", type=float, default=ASSUMED_GAIN_DEFAULT,
                    help="Gain the proposals assumed (default 10; band is 5-20)")
    ap.add_argument("--target-se", type=float, default=TARGET_SE_DEFAULT,
                    help="Per-cell measurement SE the recommendation targets")
    ap.add_argument("--rho-threshold", type=float, default=0.8,
                    help="Label-bias gate: minimum per-item Spearman rho")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-unresolved", action="store_true",
                    help="Drop UNRESOLVED from the legal verdict set (default: keep it)")
    ap.add_argument("--allow-hosted", action="store_true",
                    help="Permit hosted (non-logprob) models; results are flagged")
    ap.add_argument("--allow-indeterminate-gates", action="store_true",
                    help="Report the gain even when a pre-gate could not be interpreted")
    ap.add_argument("--skip-gates", action="store_true",
                    help="Skip the pre-gates entirely. The gain is ALWAYS withheld in "
                         "this mode; use it only to collect traces before the gates exist.")
    ap.add_argument("--out", default=str(SUMMARY_PATH))
    ap.add_argument("--smoke", action="store_true",
                    help="Offline synthetic end-to-end run: no GPU, no API keys, "
                         "output stamped synthetic and status SMOKE")
    ap.add_argument("--selftest", action="store_true",
                    help="Run the offline arithmetic and refusal-contract checks")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.m_traces < 2:
        print("ERROR: --m-traces must be at least 2 (variance needs 2 traces)")
        return 2

    models = [m.strip() for m in args.models.split(",") if m.strip()] or _default_models()
    allow_unresolved = not args.no_unresolved

    summary: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "pilot": "T0-B variance components",
        "instrument": args.instrument,
        "scaffold": args.scaffold,
        "n_items_requested": args.n_items,
        "m_traces": args.m_traces,
        "temperature": args.temperature,
        "allow_unresolved": allow_unresolved,
        "assumed_gain": args.assumed_gain,
        "target_per_cell_se": args.target_se,
        "seed": args.seed,
        "synthetic": bool(args.smoke),
        "status": "SMOKE" if args.smoke else "MEASURED",
        "models": {},
    }

    if args.smoke:
        print("SMOKE: synthetic propensities, no model calls. "
              "Output is NOT a measurement.")
        n_items = min(args.n_items, 30)
        for model in models[:1]:
            per_item = synthetic_propensities(n_items, args.m_traces, seed=args.seed)
            block = analyse_model(
                per_item,
                assumed_gain=args.assumed_gain,
                target_se=args.target_se,
                n_boot=min(args.n_boot, 300),
                seed=args.seed,
                min_traces=args.min_traces,
            )
            block.pop("_components", None)
            block["gates"] = {"overall": "skipped-smoke"}
            block = _apply_gate_refusal(block, overall="fail", allow_indeterminate=False)
            block["gain_withheld_reason"] = "synthetic smoke run; not a measurement"
            block["is_local"] = None
            summary["models"][model] = block
        # Keep the consumable key present in every mode so downstream planning
        # code can read it unconditionally and see that nothing was measured.
        summary["program_recommendation"] = _program_recommendation(
            [], assumed_gain=args.assumed_gain,
        )
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Wrote {args.out}")
        return 0

    verdicts = instrument_verdicts(args.instrument, allow_unresolved=allow_unresolved)
    print(f"Instrument={args.instrument} verdicts={verdicts}")
    items = load_pilot_items(args.instrument, args.n_items, args.seed)
    print(f"Loaded {len(items)} pilot items")
    summary["n_items_loaded"] = len(items)
    summary["verdicts"] = list(verdicts)

    for model in models:
        local = _model_is_local(model)
        if not local and not args.allow_hosted:
            print(f"SKIP {model}: exact propensities need logprob access "
                  f"(local model). Pass --allow-hosted to force.")
            summary["models"][model] = {
                "skipped": "hosted model without --allow-hosted",
                "is_local": False,
                "realised_gain": None,
            }
            continue

        print(f"\n=== {model} (local={local}) ===")
        if args.skip_gates:
            gates = {"overall": "skipped", "note": "--skip-gates was passed"}
        else:
            gates = run_gates(
                model, verdicts, items,
                instrument=args.instrument, rho_threshold=args.rho_threshold,
            )
            print(f"  gates: tokenization={gates['tokenization_audit']['status']} "
                  f"label_bias={gates['label_bias_check']['status']} "
                  f"-> {gates['overall']}")

        per_item: list[dict] = []
        for i, item in enumerate(items):
            try:
                rec = collect_item_propensities(
                    model, item,
                    instrument=args.instrument,
                    scaffold=args.scaffold,
                    m_traces=args.m_traces,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed_base=args.seed * 1000,
                    verdicts=verdicts,
                    allow_unresolved=allow_unresolved,
                )
                per_item.append(rec)
            except Exception as e:
                print(f"  ERROR item={item['id']}: {e}", flush=True)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(items)} items", flush=True)

        per_item_path = OUT_DIR / f"t0b_per_item_{_safe(model)}.json"
        per_item_path.write_text(json.dumps(per_item, indent=2, ensure_ascii=False))

        block = analyse_model(
            per_item,
            assumed_gain=args.assumed_gain,
            target_se=args.target_se,
            n_boot=args.n_boot,
            seed=args.seed,
            min_traces=args.min_traces,
        )
        components = block.pop("_components", [])
        block["is_local"] = local
        block["gates"] = gates
        block["per_item_path"] = str(per_item_path)
        overall = gates.get("overall", "indeterminate")
        if overall == "skipped":
            block = _apply_gate_refusal(block, overall="fail", allow_indeterminate=False)
            block["gain_withheld_reason"] = (
                "--skip-gates was passed: the tokenization audit and label-bias "
                "check are preconditions for reporting any efficiency gain"
            )
        else:
            block = _apply_gate_refusal(
                block, overall=overall,
                allow_indeterminate=args.allow_indeterminate_gates,
            )
        summary["models"][model] = block

        gain = block.get("realised_gain")
        if gain is None:
            print(f"  GAIN WITHHELD: {block.get('gain_withheld_reason')}")
        else:
            print(f"  realised gain = {gain:.3f}x  CI95={block.get('realised_gain_ci95')}")
            print(f"  {block['recommendation']['interpretation']}")

    reported = [
        b.get("realised_gain") for b in summary["models"].values()
        if b.get("realised_gain") is not None
    ]
    summary["program_recommendation"] = _program_recommendation(
        reported, assumed_gain=args.assumed_gain,
    )
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out}")
    return 0


def _program_recommendation(gains: Sequence[float], *, assumed_gain: float) -> dict:
    """The single block the rest of the program consumes."""
    if not gains:
        return {
            "status": "NO_GAIN_REPORTED",
            "recommended_n_multiplier": None,
            "note": "No model produced a gate-passing gain. Do not re-plan any n "
                    "on this run; fix the gates or the trace collection first.",
        }
    worst = min(gains)
    return {
        "status": "OK",
        "n_models": len(gains),
        "min_realised_gain": worst,
        "max_realised_gain": max(gains),
        # Plan against the worst model: an n that is adequate for the model with
        # the least determinative traces is adequate for all of them.
        "recommended_n_multiplier": max(1.0, assumed_gain / worst),
        "note": "recommended_n_multiplier is an upper bound on the inflation "
                "planned sample sizes need; it is exact only when per-cell "
                "measurement variance dominates item-level variance.",
    }


if __name__ == "__main__":
    sys.exit(main())
