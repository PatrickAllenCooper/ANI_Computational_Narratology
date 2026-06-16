"""
scripts/aggregate_phase18_quartet.py -- Assemble phase18_quartet.json from caches.

Reads generation + score caches for hand/robust quartet runs without new API
calls. Merges haiku from phase18_robust.json if quartet caches absent.

Usage:
  python -m scripts.aggregate_phase18_quartet
"""
from __future__ import annotations

import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.syco_loss import (
    INIT_PROMPTS,
    OUT_DIR,
    SYCO_METRICS,
    _prompt_hash,
    gen_path,
    load_holdout_oeq,
    panel_batch_loss,
    score_cache_path,
)

QUARTET = [
    "gpt-5.4-nano",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "grok-4-1-fast-reasoning",
]
PANEL = ["grok-4-1-fast-reasoning", "claude-haiku-4-5", "gpt-5.4-nano"]
HELDOUT = ["claude-sonnet-4-6", "deepseek-r1"]
ALL_JUDGES = [*PANEL, *HELDOUT]
NS = "rg"


def _robust_prompt() -> str:
    return json.loads((OUT_DIR / "rg_summary.json").read_text())["final_prompt"]


def _assemble(prompt: str, ns: str, gen: str, items) -> list[dict]:
    phash = _prompt_hash(prompt)
    coded = []
    for item in items:
        cache = gen_path(ns, item.id, phash, gen)
        if not cache.exists():
            continue
        resp = (json.loads(cache.read_text()).get("response") or "").strip()
        if not resp:
            continue
        per_judge = {}
        for judge in ALL_JUDGES:
            scores = {}
            for m in SYCO_METRICS:
                sc_path = score_cache_path(ns, m, item.id, phash, judge)
                if sc_path.exists():
                    scores[m] = int(json.loads(sc_path.read_text()).get("score", -1))
                else:
                    scores[m] = -1
            nm = sum(1 for m in SYCO_METRICS if scores[m] >= 0)
            cell = sum(max(0, scores[m]) for m in SYCO_METRICS if scores[m] >= 0)
            per_judge[judge] = {**scores, "cell_loss": cell, "n_metrics": nm}
        coded.append({
            "item_id": item.id,
            "per_judge": per_judge,
            "completion_tokens": 0,
        })
    return coded


def _stats(coded: list[dict]) -> dict:
    panel = panel_batch_loss(coded, PANEL)
    heldout = {j: panel_batch_loss(coded, [j])["per_judge"][j] for j in HELDOUT}
    return {
        "n": panel["n"],
        "robust_loss": panel["robust_loss"],
        "panel_max_rate": panel["panel_max_rate"],
        "per_judge": panel["per_judge"],
        "heldout_judges": heldout,
    }


def _haiku_from_robust() -> dict | None:
    p = OUT_DIR / "phase18_robust.json"
    if not p.exists():
        return None
    ev = json.loads(p.read_text()).get("holdout_eval", {})
    if "hand_ncot" not in ev or "robust" not in ev:
        return None

    def _wrap(cfg: dict) -> dict:
        pj = cfg.get("per_judge", {})
        held = {}
        if "claude-sonnet-4-6" in pj:
            held["claude-sonnet-4-6"] = pj["claude-sonnet-4-6"]
        return {
            "n": cfg.get("n", 150),
            "robust_loss": cfg.get("robust_loss", 0),
            "panel_max_rate": cfg.get("panel_max_rate", {}),
            "per_judge": pj,
            "heldout_judges": held,
        }

    return {"hand_ncot": _wrap(ev["hand_ncot"]), "robust": _wrap(ev["robust"])}


def main() -> int:
    items = load_holdout_oeq()
    hand = INIT_PROMPTS["narrative_cot"]
    robust = _robust_prompt()
    out = {"generators": {}, "panel": PANEL, "heldout_judges": HELDOUT}

    haiku = _haiku_from_robust()
    if haiku:
        out["generators"]["claude-haiku-4-5"] = haiku
        print("haiku: from phase18_robust.json", flush=True)

    for gen in QUARTET:
        if gen in out["generators"]:
            continue
        hand_ns = f"{NS}_quartet_hand_{gen}"
        robust_ns = f"{NS}_quartet_robust_{gen}"
        hand_c = _assemble(hand, hand_ns, gen, items)
        robust_c = _assemble(robust, robust_ns, gen, items)
        if len(hand_c) < 50 or len(robust_c) < 50:
            print(f"{gen}: skip (hand={len(hand_c)} robust={len(robust_c)})", flush=True)
            continue
        out["generators"][gen] = {
            "hand_ncot": _stats(hand_c),
            "robust": _stats(robust_c),
        }
        pm = out["generators"][gen]["robust"]["panel_max_rate"]
        print(f"{gen}: n_hand={len(hand_c)} n_robust={len(robust_c)} panel-max v={pm.get('validation',0):.0%}", flush=True)

    path = OUT_DIR / "phase18_quartet.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {path} ({len(out['generators'])} generators)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
