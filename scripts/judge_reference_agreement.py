"""
scripts/judge_reference_agreement.py -- 17.8 amendment 2 (registered before computing):
each judge's agreement with ELEPHANT's own reference validation labels on the benchmark's
150 human-written OEQ answers (`validation_human` in the release), a check of the judges
that is independent of the NoT and CoT arms. Descriptive only; no judge is excluded.

Every answer is scored through the same full-text path as 17.8
(`rescore_elephant_untruncated.score_at(..., limit=None)`), so the cache is shared.

Run:
  python -m scripts.judge_reference_agreement            # scores (cached calls replay) and reports
  python -m scripts.judge_reference_agreement --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scripts.load_elephant import load_elephant
from scripts.rescore_elephant_full_judge import DEFAULT_JUDGES
from scripts.rescore_elephant_untruncated import score_at

OUT = Path("divergence_study_outputs/judge_reference_agreement.json")


def kappa(a: list[int], b: list[int]) -> float:
    a, b = np.array(a), np.array(b)
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_kappa(a, b, draws=4000, seed=20260922):
    a, b = np.array(a), np.array(b)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(draws):
        i = rng.integers(0, len(a), len(a))
        ks.append(kappa(list(a[i]), list(b[i])))
    return float(np.nanpercentile(ks, 2.5)), float(np.nanpercentile(ks, 97.5))


def run(judges, workers=8, metric="validation") -> dict:
    items = [it for it in load_elephant("oeq", n=150)
             if it.human_response and it.human_scores.get(metric) in (0, 1)]
    res = {"metric": metric, "n_items": len(items), "reference_rate": float(np.mean([it.human_scores[metric] for it in items])),
           "judges": {}, "judge_scores": {}}
    for j in judges:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            scores = list(ex.map(lambda it: score_at(metric, it.prompt, it.human_response, j, None), items))
        res["judge_scores"][j] = {it.id: s for it, s in zip(items, scores)}
        pairs = [(it.human_scores[metric], s) for it, s in zip(items, scores) if s in (0, 1)]
        ref, jud = [p[0] for p in pairs], [p[1] for p in pairs]
        lo, hi = boot_kappa(ref, jud)
        res["judges"][j] = {"n": len(pairs), "unparsed": len(items) - len(pairs),
                            "judge_rate": float(np.mean(jud)), "agreement": float(np.mean(np.array(ref) == np.array(jud))),
                            "kappa": kappa(ref, jud), "kappa_lo": lo, "kappa_hi": hi}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--metric", default="validation", choices=("validation", "framing", "indirectness"),
                    help="17.11 amendment 2 scores 'framing' on the human answers")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        ok = abs(kappa([1, 0, 1, 0], [1, 0, 1, 0]) - 1) < 1e-9 and abs(kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9
        print("selftest:", "ALL OK" if ok else "FAILED")
        return 0 if ok else 1
    res = run([j for j in a.judges.split(",") if j], a.workers, a.metric)
    print(f"reference validating rate {res['reference_rate']:.3f} on {res['n_items']} human answers")
    for j, v in res["judges"].items():
        print(f"  {j:26s} judge rate {v['judge_rate']:.3f}  agreement {v['agreement']:.3f}  "
              f"kappa {v['kappa']:.3f} [{v['kappa_lo']:.3f}, {v['kappa_hi']:.3f}]  n {v['n']}")
    out = OUT if a.metric == "validation" else OUT.with_name(OUT.stem + f"_{a.metric}.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
