"""
scripts/blind_label_optimized_gold.py -- Out-of-family proxy labels on optimized outputs.

Scores the n=40 optimized-gold sample with DeepSeek-R1 (held out from optimisation
panel) as an independent, non-author validation anchor. Caches per-item scores.

Output: divergence_study_outputs/optimized_gold_blinded.json

Usage:
  python -m scripts.blind_label_optimized_gold
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.elephant_scorers import score_response
from scripts.krippendorff import cohens_kappa

GOLD_PATH = Path("data/optimized_gold.jsonl")
CACHE_PATH = Path("divergence_study_outputs/optimized_gold_blinded_scores.jsonl")
OUT_PATH = Path("divergence_study_outputs/optimized_gold_blinded.json")
PROXY_JUDGE = "deepseek-r1"
FALLBACK_JUDGE = "claude-sonnet-4-6"


def _read_gold() -> list[dict]:
    rows = []
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r.get("human_validation") in (0, 1)]


def _load_cache() -> dict[str, int]:
    if not CACHE_PATH.exists():
        return {}
    out = {}
    with CACHE_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("score", -1)) >= 0:
                out[row["uid"]] = int(row["score"])
    return out


def _write_cache(rows: list[dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    gold = _read_gold()
    if not gold:
        print(f"No labeled rows in {GOLD_PATH}")
        return 1

    cache = _load_cache()
    cache_rows = []
    for row in gold:
        uid = row["uid"]
        if uid in cache and cache[uid] >= 0:
            score = cache[uid]
        else:
            score = -1
            for judge in (PROXY_JUDGE, FALLBACK_JUDGE):
                for attempt in range(5):
                    score = score_response(
                        "validation", row["question"], row["response"], judge=judge,
                    )
                    if score >= 0:
                        break
                if score >= 0:
                    break
            cache[uid] = score
        cache_rows.append({"uid": uid, "judge": PROXY_JUDGE, "score": score})
    _write_cache(cache_rows)

    by_config: dict[str, list[int]] = {"hand_ncot": [], "robust": []}
    author, proxy = [], []
    items = []
    for row in gold:
        uid = row["uid"]
        hv = int(row["human_validation"])
        pv = cache[uid]
        cfg = row["config"]
        if pv >= 0:
            by_config.setdefault(cfg, []).append(pv)
            author.append(hv)
            proxy.append(pv)
        items.append({
            "uid": uid,
            "config": cfg,
            "author_validation": hv,
            "proxy_validation": pv,
        })

    summary = {
        "proxy_judge": PROXY_JUDGE,
        "n_labeled": len(gold),
        "proxy_validation_rate": {
            cfg: sum(v) / len(v) if v else None for cfg, v in by_config.items()
        },
        "author_validation_rate": {
            cfg: sum(int(r["human_validation"]) for r in gold if r["config"] == cfg)
            / max(1, sum(1 for r in gold if r["config"] == cfg))
            for cfg in by_config
        },
        "kappa_proxy_vs_author": cohens_kappa(author, proxy) if author else None,
        "items": items,
    }
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_PATH} (n={len(gold)}, judge={PROXY_JUDGE})")
    for cfg, rate in summary["proxy_validation_rate"].items():
        print(f"  proxy {cfg}: {rate:.0%}")
    print(f"  kappa proxy vs author: {summary['kappa_proxy_vs_author']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
