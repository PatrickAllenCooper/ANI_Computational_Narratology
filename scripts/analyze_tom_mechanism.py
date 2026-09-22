"""
analyze_tom_mechanism.py -- diagnostic for Addendum 18 stage 1: why does NoT cost
accuracy on ToMBench (pooled delta -0.044 [-0.066, -0.022])?

Three checks, all recomputed from the cache (no new calls):
  1. McNemar-style flip asymmetry: CoT-right/NoT-wrong vs CoT-wrong/NoT-right.
  2. Task-type stratification: FACTUAL (False Belief, Scalar Implicature -- pure
     state-tracking or logic, no coherent "preferable" reading) vs INTENT (Strange
     Story, Faux-pas, Persuasion, Hinting -- inferring deception, tact or strategy,
     where a charitable reading competes with the true one) vs AMBIG (the other two).
  3. Signal-word rate: does NoT's wrong-flip text use the normative vocabulary
     ("preferable", "trust", "honest*", "fair*", "morale", "reassur*", "benefit of
     the doubt") more than its both-right text.

  python -m scripts.analyze_tom_mechanism
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "divergence_study_outputs"
MODELS = ("gpt-5.4-nano", "grok-4-1-fast-reasoning", "Mistral-Large-3-2", "DeepSeek-V4-Pro")
FACTUAL = {"False Belief Task", "Scalar Implicature Test"}
INTENT = {"Strange Story Task", "Faux-pas Recognition Test", "Persuasion Story Task", "Hinting Task Test"}
SIGNAL = re.compile(r"\bpreferable\b|\bcharitable\b|\bbenefit of the doubt\b|\btrust\b|\bmorale\b|"
                    r"\breassur\w*\b|\bhonest\w*\b|\bfair\w*\b", re.I)


def load():
    recs = defaultdict(dict)
    for f in glob.glob(str(OUT / "tom_gen_*.json")):
        d = json.load(open(f))
        recs[(d["model"], d["arm"])][(d["task"], d["row"])] = d
    return recs


def boot(x, draws=4000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    m = x[idx].mean(1)
    return {"point": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": len(x)}


def main() -> dict:
    recs = load()
    flip_to_wrong = flip_to_right = both_right = both_wrong = 0
    by_bucket = defaultdict(list)
    signal = {"wrong_flip": [0, 0], "both_right": [0, 0]}  # [hits, n]
    for model in MODELS:
        cot, nott = recs[(model, "standard_cot")], recs[(model, "narrative_cot")]
        for k in set(cot) & set(nott):
            cr = cot[k]["answer"] == cot[k]["gold"]
            nr = nott[k]["answer"] == nott[k]["gold"]
            task = k[0]
            bucket = "FACTUAL" if task in FACTUAL else "INTENT" if task in INTENT else "AMBIG"
            by_bucket[bucket].append(int(nr) - int(cr))
            hit = bool(SIGNAL.search(nott[k]["text"]))
            if cr and not nr:
                flip_to_wrong += 1; signal["wrong_flip"][0] += hit; signal["wrong_flip"][1] += 1
            elif not cr and nr:
                flip_to_right += 1
            elif cr and nr:
                both_right += 1; signal["both_right"][0] += hit; signal["both_right"][1] += 1
            else:
                both_wrong += 1

    res = {"flip_to_wrong": flip_to_wrong, "flip_to_right": flip_to_right,
           "both_right": both_right, "both_wrong": both_wrong,
           "asymmetry_ratio": flip_to_wrong / flip_to_right if flip_to_right else None,
           "by_task_type": {b: boot(xs) for b, xs in by_bucket.items()},
           "signal_word_rate": {k: v[0] / v[1] for k, v in signal.items()},
           "signal_word_counts": signal}
    print(f"CoT-right -> NoT-wrong: {flip_to_wrong}   CoT-wrong -> NoT-right: {flip_to_right}   "
          f"ratio {res['asymmetry_ratio']:.2f}x")
    for b, v in res["by_task_type"].items():
        print(f"  {b:10s} n={v['n']:4d} delta {v['point']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    for k, v in res["signal_word_rate"].items():
        n = signal[k][1]
        print(f"  signal-word rate, {k:12s} {v:.3f} (n={n})")
    out = OUT / "tom_mechanism_analysis.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)
    return res


if __name__ == "__main__":
    main()
