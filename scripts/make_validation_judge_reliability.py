"""
make_validation_judge_reliability.py -- Z5 of Addendum 17: one artefact for the
paper's appendix collecting what the record holds about the reliability of the
judge-scored VALIDATION construct (the pillar-1 outcome), recomputed from
judge_panel_raw.csv where a number can be recomputed and copied from the named
artefacts where it cannot (with the source recorded).

  MTMM on the 60-unit, 3-judge, 3-construct panel: convergent r (same construct,
    different judges), monomethod-heterotrait r (same judge, different constructs)
  Krippendorff alpha per construct and its bootstrap CI (judge_reliability_summary.json,
    reliability_bootstrap_cis.json)
  kappa of each judge against the 29/30 human gold items (reliability_bootstrap_cis.json)
  the calibration result on the Goodhart re-score (theory_grounding §S0.1, copied)

  python -m scripts.make_validation_judge_reliability
"""
from __future__ import annotations

import csv
import itertools
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "divergence_study_outputs"


def pearson(x, y):
    if len(x) < 3 or st.pstdev(x) == 0 or st.pstdev(y) == 0:
        return None
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) * st.pstdev(x) * st.pstdev(y))


def mtmm(rows):
    """units = (arm, generator, item_id); cell = (judge, metric) -> score."""
    cell = defaultdict(dict)
    for r in rows:
        if r["score"] in ("0", "1"):
            cell[(r["arm"], r["generator"], r["item_id"])][(r["judge"], r["metric"])] = int(r["score"])
    judges = sorted({r["judge"] for r in rows}); metrics = sorted({r["metric"] for r in rows})
    complete = [u for u, d in cell.items() if len(d) == len(judges) * len(metrics)]
    conv, mono = {}, {}
    for m in metrics:
        rs = []
        for j1, j2 in itertools.combinations(judges, 2):
            r = pearson([cell[u][(j1, m)] for u in complete], [cell[u][(j2, m)] for u in complete])
            if r is not None:
                rs.append({"judges": [j1, j2], "r": r})
        conv[m] = {"pairs": rs, "mean_r": st.mean([p["r"] for p in rs]) if rs else None}
    rs = []
    for j in judges:
        for m1, m2 in itertools.combinations(metrics, 2):
            r = pearson([cell[u][(j, m1)] for u in complete], [cell[u][(j, m2)] for u in complete])
            if r is not None:
                rs.append({"judge": j, "metrics": [m1, m2], "r": r})
    mono = {"pairs": rs, "mean_r": st.mean([p["r"] for p in rs]) if rs else None}
    base = {j: {m: st.mean([cell[u][(j, m)] for u in complete]) for m in metrics} for j in judges}
    return {"n_complete_units": len(complete), "judges": judges, "metrics": metrics,
            "convergent": conv, "monomethod_heterotrait": mono, "base_rates": base}


def main() -> int:
    rows = list(csv.DictReader(open(OUT / "judge_panel_raw.csv")))
    res = {"construct": "validation (ELEPHANT OEQ, the pillar-1 outcome)", "mtmm": mtmm(rows)}
    rel = json.loads((OUT / "judge_reliability_summary.json").read_text())
    boot = json.loads((OUT / "reliability_bootstrap_cis.json").read_text())
    res["krippendorff_alpha"] = {m: {"alpha": v["point"], "ci95": v["ci95"], "n_items": v["n_items"],
                                      "threshold": rel["alpha_threshold"]}
                                 for m, v in boot["panel_alpha"].items()}
    res["kappa_vs_human_gold"] = [k for k in boot["kappa_vs_human"]]
    res["calibration_S0_1"] = {
        "source": "Guidance_Documents/theory_grounding_and_judge_rehabilitation.md §S0.1 (copied, not recomputed)",
        "validation_cross_judge_spread_of_reductions_pp": 54.3, "absorbed_by_per_judge_threshold": 0.92,
        "threshold_model_rejected_G2_df4_p": {"validation": [16.63, 0.0023]},
        "note": "calibration absorbs the level gap on validation, which is the one quantity a threshold moves; "
                "the judge x arm interaction remains (threshold model rejected)"}
    res["sources"] = ["judge_panel_raw.csv", "judge_reliability_summary.json", "reliability_bootstrap_cis.json",
                      "theory_grounding_and_judge_rehabilitation.md §7, §S0.1"]
    m = res["mtmm"]
    print(f"MTMM on {m['n_complete_units']} complete units, judges {m['judges']}")
    for k, v in m["convergent"].items():
        print(f"  convergent {k:13s} mean r {v['mean_r']:+.3f}  pairs " + ", ".join(f"{p['r']:+.2f}" for p in v["pairs"]))
    print(f"  monomethod-heterotrait mean r {m['monomethod_heterotrait']['mean_r']:+.3f}")
    for k, v in res["krippendorff_alpha"].items():
        print(f"  alpha {k:13s} {v['alpha']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    for k in res["kappa_vs_human_gold"]:
        if k["metric"] == "validation":
            print(f"  kappa vs human, validation, {k['judge']:26s} {k['point']:+.3f} [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}] n={k['n']}")
    out = OUT / "validation_judge_reliability.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
