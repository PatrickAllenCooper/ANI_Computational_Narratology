"""Aggregate Phase 9 (TextGrad) results: compute Cliff's delta between
TextGrad-optimised and N-CoT outputs on the held-out 30 scenarios,
plus matched-pair stats for the appendix table.
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
from pathlib import Path

OUT = Path("./divergence_study_outputs")

GEN_MODEL = "gpt-5.4-nano"
JUDGE_MODEL = "claude-haiku-4-5"


def _hash(p: str) -> str:
    return hashlib.sha1(p.encode("utf-8")).hexdigest()[:10]


def cliffs_delta(a: list[float], b: list[float]) -> tuple[float, tuple[float, float]]:
    """Cliff's delta of A vs B; positive means A > B in stochastic order."""
    n, m = len(a), len(b)
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    delta = (gt - lt) / (n * m) if n * m else 0.0
    # bootstrap CI
    rng = random.Random(42)
    deltas = []
    for _ in range(500):
        ai = [a[rng.randrange(n)] for _ in range(n)]
        bi = [b[rng.randrange(m)] for _ in range(m)]
        g = l = 0
        for x in ai:
            for y in bi:
                if x > y:
                    g += 1
                elif x < y:
                    l += 1
        deltas.append((g - l) / (n * m))
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1]
    return delta, (lo, hi)


def main():
    summary = json.loads((OUT / "tg_summary.json").read_text())
    tg_prompt = summary["final_prompt"]
    ncot_prompt_hash = None
    # The N-CoT held-out cells were judged under the verbatim N-CoT prompt
    from scripts.run_phase1_quartet import PROMPTS
    ncot_hash = _hash(PROMPTS["narrative_cot"])
    tg_hash = _hash(tg_prompt)

    eval_start, eval_end = summary["eval_indices"]
    # Load all judge files for tg and ncot prompts on the held-out scenarios
    tg_codes = []
    ncot_codes = []
    import re as _re
    sid_pat = _re.compile(rf"tg_judge_{_re.escape(JUDGE_MODEL)}_{_re.escape(GEN_MODEL)}_(.+?)_([0-9a-f]{{10}})\.json$")
    for path in sorted(glob.glob(f"{OUT}/tg_judge_{JUDGE_MODEL}_{GEN_MODEL}_*_{tg_hash}.json")):
        m = sid_pat.match(Path(path).name)
        if not m:
            continue
        d = json.loads(Path(path).read_text())
        tg_codes.append({
            "scenario_id": m.group(1),
            "sc": int(d.get("stakeholder_count", 0) or 0),
            "us": int(d.get("uncertainty_score", 0) or 0),
            "mh": int(d.get("max_causal_hops", 0) or 0),
        })
    for path in sorted(glob.glob(f"{OUT}/tg_judge_{JUDGE_MODEL}_{GEN_MODEL}_*_{ncot_hash}.json")):
        m = sid_pat.match(Path(path).name)
        if not m:
            continue
        d = json.loads(Path(path).read_text())
        ncot_codes.append({
            "scenario_id": m.group(1),
            "sc": int(d.get("stakeholder_count", 0) or 0),
            "us": int(d.get("uncertainty_score", 0) or 0),
            "mh": int(d.get("max_causal_hops", 0) or 0),
        })

    # Restrict to held-out scenarios (intersection)
    from scripts.run_phase1_quartet import load_daily_dilemmas
    scenarios = load_daily_dilemmas(n=summary.get("scenarios_n", 100) if False else 100)
    eval_ids = {s.id for s in scenarios[eval_start:eval_end]}
    tg_codes = [c for c in tg_codes if c["scenario_id"] in eval_ids]
    ncot_codes = [c for c in ncot_codes if c["scenario_id"] in eval_ids]

    print(f"TG codes on held-out: {len(tg_codes)}; N-CoT codes on held-out: {len(ncot_codes)}")

    # Cliff's delta: N-CoT vs TextGrad-optimised (positive favours N-CoT)
    nc_sc = [c["sc"] for c in ncot_codes]
    nc_us = [c["us"] for c in ncot_codes]
    nc_mh = [c["mh"] for c in ncot_codes]
    tg_sc = [c["sc"] for c in tg_codes]
    tg_us = [c["us"] for c in tg_codes]
    tg_mh = [c["mh"] for c in tg_codes]

    print("\n=== Cliff's delta: N-CoT vs TextGrad-optimised (positive = N-CoT wins) ===")
    for name, a, b in [
        ("stakeholder_count", nc_sc, tg_sc),
        ("uncertainty_score", nc_us, tg_us),
        ("max_causal_hops", nc_mh, tg_mh),
    ]:
        d, (lo, hi) = cliffs_delta(a, b)
        print(f"  {name:25s}: delta={d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # Also save as JSON for paper rendering
    out_json = {
        "n_eval_scenarios": min(len(tg_codes), len(ncot_codes)),
        "ncot_means": {
            "stakeholder_count": sum(nc_sc) / len(nc_sc),
            "uncertainty_score": sum(nc_us) / len(nc_us),
            "max_causal_hops": sum(nc_mh) / len(nc_mh),
        },
        "tg_means": {
            "stakeholder_count": sum(tg_sc) / len(tg_sc),
            "uncertainty_score": sum(tg_us) / len(tg_us),
            "max_causal_hops": sum(tg_mh) / len(tg_mh),
        },
        "ncot_vs_tg_cliffs_delta": {
            "stakeholder_count": cliffs_delta(nc_sc, tg_sc),
            "uncertainty_score": cliffs_delta(nc_us, tg_us),
            "max_causal_hops": cliffs_delta(nc_mh, tg_mh),
        },
        "ncot_mean_tokens": summary["heldout_eval"]["ncot_stats"]["mean_completion_tokens"],
        "tg_mean_tokens": summary["heldout_eval"]["tg_stats"]["mean_completion_tokens"],
        "ncot_failure_rates": {
            "collapse": summary["heldout_eval"]["ncot_stats"]["collapse_rate"],
            "suppression": summary["heldout_eval"]["ncot_stats"]["suppression_rate"],
        },
        "tg_failure_rates": {
            "collapse": summary["heldout_eval"]["tg_stats"]["collapse_rate"],
            "suppression": summary["heldout_eval"]["tg_stats"]["suppression_rate"],
        },
    }
    (OUT / "tg_aggregate.json").write_text(json.dumps(out_json, indent=2, default=str))
    print(f"\nAggregate JSON: {OUT}/tg_aggregate.json")


if __name__ == "__main__":
    main()
