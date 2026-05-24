"""
scripts/kc_proxy.py -- Algorithmic Causal Complexity (K_C) proxy module.

Exposes three measurable proxies for the minimum description length of the
causal model implicit in a model output:

  kc_graph(text, extractor_model)
      Calls a relation-extraction prompt and returns a structural complexity
      score: n_nodes + n_edges + mean_degree.  Extractor defaults to
      claude-sonnet-4-6 (flagged as cheapest path that produces reliable JSON).

  kc_gzip(text)
      Byte length after gzip.compress(text.encode()).  Fast, no API calls.

  kc_lm(text, scorer_model)
      Average negative log-probability under a small reference scorer using
      claude-haiku-4-5 in token-scoring mode (cheapest Foundry path).
      Approximated via a prompt asking for per-sentence perplexity ratings.

All three can be run as a batch via kc_score_all(texts, ...).

Usage:
  from scripts.kc_proxy import kc_graph, kc_gzip, kc_lm, kc_score_all
  scores = kc_score_all(["The doctor decided to...", ...])
"""
from __future__ import annotations

import gzip
import json
import math
import re
from typing import Optional

from scripts.generators import generate


# ---------------------------------------------------------------------------
# kc_graph
# ---------------------------------------------------------------------------

_RELATION_EXTRACTION_SYSTEM = (
    "You are a relation extractor. Given a short text, extract all causal "
    "and consequential relations as triples: (subject, relation, object). "
    "Return only valid JSON with exactly one key: 'triples', whose value is "
    "a list of [subject_str, relation_str, object_str] arrays. Use short "
    "noun-phrase strings for subjects and objects."
)

_RELATION_EXTRACTION_USER = (
    "Extract all causal / consequential triples from the following text.\n\n"
    "{text}"
)


def kc_graph(
    text: str,
    extractor_model: str = "claude-sonnet-4-6",
    sample_idx: int = 0,
) -> dict:
    """Extract a causal relation graph and return structural complexity metrics.

    Returns a dict with keys:
      n_nodes: number of unique nodes
      n_edges: number of unique edges (triples)
      mean_degree: average number of edges per node
      kc_graph_score: n_nodes + n_edges + mean_degree (combined proxy)
      triples: list of [subject, relation, object] extracted
    """
    prompt = _RELATION_EXTRACTION_USER.format(text=text[:3000])
    result = generate(
        extractor_model,
        _RELATION_EXTRACTION_SYSTEM,
        prompt,
        sample_idx=sample_idx,
        max_tokens=1024,
        json_mode=True,
    )
    try:
        data = json.loads(result.text) if result.text else {}
    except Exception:
        # Try to extract JSON from partial response
        m = re.search(r"\{.*\}", result.text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}

    triples = data.get("triples", [])
    # Deduplicate
    unique_triples = list({(t[0], t[1], t[2]) for t in triples if len(t) == 3})
    nodes = set()
    for s, r, o in unique_triples:
        nodes.add(s)
        nodes.add(o)
    n_nodes = len(nodes)
    n_edges = len(unique_triples)
    mean_degree = (2 * n_edges / n_nodes) if n_nodes > 0 else 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": round(mean_degree, 4),
        "kc_graph_score": round(n_nodes + n_edges + mean_degree, 4),
        "triples": [list(t) for t in unique_triples],
    }


# ---------------------------------------------------------------------------
# kc_gzip
# ---------------------------------------------------------------------------

def kc_gzip(text: str) -> dict:
    """Compress text with gzip and return byte length as proxy for K_C.

    Returns a dict with keys:
      raw_bytes: len(text.encode('utf-8'))
      gzip_bytes: compressed length
      compression_ratio: gzip_bytes / raw_bytes
      kc_gzip_score: gzip_bytes (smaller = simpler causal model)
    """
    raw = text.encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=9)
    raw_bytes = len(raw)
    gzip_bytes = len(compressed)
    ratio = gzip_bytes / raw_bytes if raw_bytes > 0 else 1.0
    return {
        "raw_bytes": raw_bytes,
        "gzip_bytes": gzip_bytes,
        "compression_ratio": round(ratio, 4),
        "kc_gzip_score": gzip_bytes,
    }


# ---------------------------------------------------------------------------
# kc_lm (approximated via LM scoring prompt)
# ---------------------------------------------------------------------------

_LM_SCORER_SYSTEM = (
    "You are a text complexity scorer. Given a passage, estimate the average "
    "negative log-probability per token under a generic language model. "
    "Concretely: for each sentence, rate how surprising / low-probability "
    "it would be relative to common English text (0=very common, "
    "10=very unusual/technical). Return only valid JSON with exactly one key: "
    "'sentence_scores', a list of floats."
)

_LM_SCORER_USER = (
    "Score each sentence in the following passage for average token-level "
    "negative log-probability (0-10 scale).\n\n"
    "{text}"
)


def kc_lm(
    text: str,
    scorer_model: str = "claude-haiku-4-5",
    sample_idx: int = 0,
) -> dict:
    """Approximate average negative log-probability using a cheap LM scorer.

    Returns a dict with keys:
      mean_nlp: mean of per-sentence scores
      n_sentences: number of sentences scored
      kc_lm_score: mean_nlp (higher = more surprising = higher complexity)
    """
    # Split into sentences (simple heuristic)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return {"mean_nlp": 0.0, "n_sentences": 0, "kc_lm_score": 0.0}

    prompt = _LM_SCORER_USER.format(text=text[:3000])
    result = generate(
        scorer_model,
        _LM_SCORER_SYSTEM,
        prompt,
        sample_idx=sample_idx,
        max_tokens=512,
        json_mode=True,
    )
    try:
        data = json.loads(result.text) if result.text else {}
    except Exception:
        m = re.search(r"\{.*\}", result.text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}

    scores = [float(s) for s in data.get("sentence_scores", []) if isinstance(s, (int, float))]
    if not scores:
        return {"mean_nlp": 0.0, "n_sentences": len(sentences), "kc_lm_score": 0.0}
    mean_nlp = sum(scores) / len(scores)
    return {
        "mean_nlp": round(mean_nlp, 4),
        "n_sentences": len(scores),
        "kc_lm_score": round(mean_nlp, 4),
    }


# ---------------------------------------------------------------------------
# kc_graph_mdl and kc_struct_entropy (E7 SCM-level proxies)
# ---------------------------------------------------------------------------

def kc_graph_mdl(n_nodes: int, n_edges: int, text_len: int) -> dict:
    """Minimum-description-length proxy over an extracted SCM.

    MDL = log2(n_nodes) + log2(n_edges + 1), normalised by log2(text_len + 1)
    so that longer texts don't trivially score higher.

    Returns a dict with keys:
      kc_graph_mdl: the MDL score (higher = more complex causal model)
      mdl_raw: unnormalised MDL
    """
    import math
    mdl_raw = math.log2(n_nodes + 1) + math.log2(n_edges + 1)
    normaliser = math.log2(text_len + 1) if text_len > 0 else 1.0
    mdl_norm = mdl_raw / normaliser if normaliser > 0 else 0.0
    return {
        "kc_graph_mdl": round(mdl_norm, 6),
        "mdl_raw": round(mdl_raw, 4),
    }


def kc_struct_entropy(triples: list[list[str]]) -> dict:
    """Shannon entropy of the row-normalised adjacency distribution over extracted triples.

    Treats each (subject, object) pair as a graph edge. Computes the out-degree
    distribution of the subject nodes, normalises to a probability distribution,
    and returns its Shannon entropy (nats). Higher entropy = more branching causal
    structure = higher structural complexity.

    Returns a dict with keys:
      kc_struct_entropy: entropy in nats
      n_subjects: number of unique subject nodes
    """
    import math, collections
    out_degree: dict = collections.Counter()
    for t in triples:
        if len(t) >= 3:
            out_degree[t[0]] += 1
    if not out_degree:
        return {"kc_struct_entropy": 0.0, "n_subjects": 0}
    total = sum(out_degree.values())
    entropy = -sum(
        (cnt / total) * math.log(cnt / total)
        for cnt in out_degree.values()
        if cnt > 0
    )
    return {
        "kc_struct_entropy": round(entropy, 6),
        "n_subjects": len(out_degree),
    }


# ---------------------------------------------------------------------------
# Batch scorer
# ---------------------------------------------------------------------------

def kc_score_all(
    texts: list[str],
    *,
    extractor_model: str = "claude-sonnet-4-6",
    scorer_model: str = "claude-haiku-4-5",
    skip_graph: bool = False,
    skip_lm: bool = False,
) -> list[dict]:
    """Score a list of texts on all three K_C proxies.

    Args:
        texts: list of generation output strings.
        extractor_model: model for kc_graph relation extraction.
        scorer_model: model for kc_lm scoring.
        skip_graph: omit the kc_graph proxy (saves API calls).
        skip_lm: omit the kc_lm proxy.

    Returns a list of dicts, one per text, with merged proxy results.
    """
    results = []
    for idx, text in enumerate(texts):
        row: dict = {"text_idx": idx, "text_len": len(text)}
        # gzip is always computed (no API)
        row.update(kc_gzip(text))
        if not skip_graph:
            try:
                row.update(kc_graph(text, extractor_model=extractor_model, sample_idx=idx))
            except Exception as e:
                row.update({"kc_graph_score": float("nan"), "kc_graph_error": str(e)})
        if not skip_lm:
            try:
                row.update(kc_lm(text, scorer_model=scorer_model, sample_idx=idx))
            except Exception as e:
                row.update({"kc_lm_score": float("nan"), "kc_lm_error": str(e)})
        results.append(row)
    return results
