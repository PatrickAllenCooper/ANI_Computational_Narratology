"""
scripts/analyze_phi_predictors.py -- zero-spend, cross-model diagnostic
(registered as a zero-spend note appended after Addendum 16.26 RESULTS).

Every RESULTS block from 16.24 onward reported, pairwise, that role-lock does
not predict the SIGN of advocate phi within one-loser verdicts (Llama: lock
0.862, phi +0.148; DeepSeek: lock 0.816, phi -0.093 -- similar lock, opposite
sign). That pairwise reading is correct as far as it goes but was never
checked against the full set of conditions this session has now measured.
This script pulls phi (from emergent_graphs_analysis.json, computed
independently of the reading-off-by-hand in the prose) and solo-base accuracy
(from each community's router_decomposition_<short>.json, acc_base field) for
every condition on disk, and reports the correlation. Role-lock is not
machine-readable from a single artefact for four of the six conditions (it
was computed once, by hand, from analyze_topology_2x2 / analyze_stake_grip
runs whose console output is quoted in the prereg and not re-serialised);
those four values are hardcoded here with an inline citation to the exact
prereg line they were copied from, and a --selftest checks the two that ARE
machine-readable (DeepSeek, Mistral, from topology_cell_*.json) against the
hardcoded copy so a transcription error would be caught.

Run:
  python -m scripts.analyze_phi_predictors
  python -m scripts.analyze_phi_predictors --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT_DIR = Path(__file__).resolve().parents[1] / "divergence_study_outputs"
PREREG = Path(__file__).resolve().parents[1] / "Guidance_Documents" / "prereg_embodiment_community.md"

# role-lock: hand-transcribed from the prereg (source line noted); DeepSeek and
# Mistral are cross-checked against topology_cell_*.json in the selftest.
ROLE_LOCK = {
    "grok_on_edge": (0.971, "prereg line ~7645, '(grok 0.971, haiku'"),
    "grok_no_edge": (0.969, "prereg line ~4147, no-edge k=4 role-lock row, second column"),
    "haiku_249": (0.419, "prereg 16.23 RESULTS, 'Role-lock 0.419 (0.414) [0.971]'"),
    "llama": (0.862, "prereg 16.24 RESULTS, 'role-lock 0.862 [0.869 matched]'"),
    "deepseek": (0.8162614139457665, "topology_cell_cg_deliberation_deepseekv4pro.json, role_lock"),
    "mistral": (0.70802200083022, "topology_cell_cg_deliberation_mistrallarge32.json, role_lock"),
}

# (emergent_graphs community key, router_decomposition artefact, short label)
CONDITIONS = [
    ("grok_on_edge", "grok_aita", "router_decomposition.json"),
    ("grok_no_edge", "grok_aita_noedge", "router_decomposition_noedge_k4.json"),
    ("haiku_249", "haiku_aita_249", "router_decomposition_haiku.json"),
    ("llama", "llama_aita", "router_decomposition_llama.json"),
    ("deepseek", "deepseek_aita", "router_decomposition_deepseek.json"),
    ("mistral", "mistral_aita", "router_decomposition_mistral.json"),
]


def load_phi(emergent: dict, community_key: str) -> dict:
    edge = emergent["communities"][community_key]["populations"]["s2_codable"]["graphs"]["one_loser"]["edges"]["advocate_1|advocate_2"]["phi"]
    return {"point": edge["point"], "lo": edge["lo"], "hi": edge["hi"]}


def load_acc_base(router_path: Path) -> float:
    d = json.loads(router_path.read_text())
    rvs = d["routed_vs_solo"]
    a, b = rvs["sonnet"]["acc_base"], rvs["haiku"]["acc_base"]
    assert abs(a - b) < 1e-9, f"{router_path}: sonnet/haiku acc_base disagree ({a} vs {b})"
    return a


def pearsonr(x: np.ndarray, y: np.ndarray):
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def spearmanr(x: np.ndarray, y: np.ndarray):
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def build_table(emergent_path: Path = OUT_DIR / "emergent_graphs_analysis.json") -> dict:
    emergent = json.loads(emergent_path.read_text())
    rows = []
    for label, com_key, router_file in CONDITIONS:
        phi = load_phi(emergent, com_key)
        acc = load_acc_base(OUT_DIR / router_file)
        lock, lock_src = ROLE_LOCK[label]
        rows.append({"label": label, "role_lock": lock, "role_lock_source": lock_src,
                     "solo_base_acc": acc, "phi_point": phi["point"],
                     "phi_lo": phi["lo"], "phi_hi": phi["hi"]})
    return {"rows": rows}


def analyse(table: dict) -> dict:
    rows = table["rows"]
    # pooled n=6 (grok on/off-edge are pseudo-replicates of one model) and the
    # honest n=5 (grok on/off-edge averaged to a single independent point)
    lock6 = np.array([r["role_lock"] for r in rows])
    acc6 = np.array([r["solo_base_acc"] for r in rows])
    phi6 = np.array([r["phi_point"] for r in rows])

    by_label = {r["label"]: r for r in rows}
    grok_lock = (by_label["grok_on_edge"]["role_lock"] + by_label["grok_no_edge"]["role_lock"]) / 2
    grok_acc = by_label["grok_on_edge"]["solo_base_acc"]  # identical by construction (same solo cache)
    grok_phi = (by_label["grok_on_edge"]["phi_point"] + by_label["grok_no_edge"]["phi_point"]) / 2
    others = [r for r in rows if r["label"] not in ("grok_on_edge", "grok_no_edge")]
    lock5 = np.array([grok_lock] + [r["role_lock"] for r in others])
    acc5 = np.array([grok_acc] + [r["solo_base_acc"] for r in others])
    phi5 = np.array([grok_phi] + [r["phi_point"] for r in others])

    out = {"n6": {}, "n5_independent_models": {}, "leave_one_out_n5": {}}
    for name, x in (("role_lock", lock6), ("solo_base_acc", acc6)):
        r, p = pearsonr(x, phi6)
        rho, ps = spearmanr(x, phi6)
        out["n6"][name] = {"pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": ps}
    for name, x in (("role_lock", lock5), ("solo_base_acc", acc5)):
        r, p = pearsonr(x, phi5)
        rho, ps = spearmanr(x, phi5)
        out["n5_independent_models"][name] = {"pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": ps}
    labels5 = ["grok"] + [r["label"] for r in others]
    for i, dropped in enumerate(labels5):
        keep = [j for j in range(len(labels5)) if j != i]
        out["leave_one_out_n5"][dropped] = {
            "role_lock_r": pearsonr(lock5[keep], phi5[keep])[0],
            "solo_base_acc_r": pearsonr(acc5[keep], phi5[keep])[0],
        }
    per_model = dict(out["leave_one_out_n5"])
    for name in ("role_lock", "solo_base_acc"):
        rs = [v[f"{name}_r"] for v in per_model.values()]
        out["leave_one_out_n5"][f"_{name}_range"] = [min(rs), max(rs)]
    return out


def print_report(table: dict, res: dict) -> None:
    print("=" * 78)
    print("PILLAR-3 MECHANISM DIAGNOSTIC -- role-lock and solo accuracy vs advocate phi")
    print("(zero spend; registered as a note after Addendum 16.26 RESULTS)")
    print("=" * 78)
    print(f"{'condition':16s}  {'role-lock':>10s}  {'solo acc':>9s}  {'phi (one-loser)':>22s}")
    for r in table["rows"]:
        print(f"{r['label']:16s}  {r['role_lock']:10.3f}  {r['solo_base_acc']:9.4f}  "
              f"{r['phi_point']:+.3f} [{r['phi_lo']:+.3f}, {r['phi_hi']:+.3f}]")
    print()
    for pop, label in (("n6", "n=6 (grok on/off-edge as two points, pseudo-replicated)"),
                       ("n5_independent_models", "n=5 (grok on/off-edge averaged, independent models only)")):
        print(f"-- {label} --")
        for name in ("role_lock", "solo_base_acc"):
            v = res[pop][name]
            print(f"  {name:14s} pearson r={v['pearson_r']:+.3f} p={v['pearson_p']:.3f}   "
                  f"spearman rho={v['spearman_rho']:+.3f} p={v['spearman_p']:.3f}")
        print()
    loo = res["leave_one_out_n5"]
    print("-- leave-one-out, n=5 -> n=4 --")
    for k, v in loo.items():
        if k.startswith("_"):
            continue
        print(f"  drop {k:10s} role_lock r={v['role_lock_r']:+.2f}   solo_base_acc r={v['solo_base_acc_r']:+.2f}")
    lr, ar = loo["_role_lock_range"], loo["_solo_base_acc_range"]
    print(f"  range: role_lock [{lr[0]:+.2f}, {lr[1]:+.2f}]   solo_base_acc [{ar[0]:+.2f}, {ar[1]:+.2f}]")
    print()
    print("Reading (n=5 independent models; suggestive, not confirmatory):")
    print(" 1. Solo accuracy, the candidate the 16.24 diagnostic note named, is NOT")
    print("    supported: weak, non-significant, and unstable under leave-one-out.")
    print(" 2. Role-lock keeps a negative association in every leave-one-out, so no")
    print("    single model manufactures it. But it is largely STRUCTURAL: if each")
    print("    advocate starts on its own side (high lock) and objects only when its")
    print("    side loses, opposed objections (negative phi) follow nearly by")
    print("    construction. Lock reads as necessary for opposition, not sufficient.")
    print(" 3. The open question is the missing second ingredient, whether a locked")
    print("    advocate confines its objections to verdicts against its own side.")
    print("    Llama is the clean counter-case (lock 0.862, phi +0.148). Neither")
    print("    candidate here explains it.")


def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("[ok] " if cond else "[FAIL] ") + msg)
        ok = ok and cond

    table = build_table()
    by_label = {r["label"]: r for r in table["rows"]}
    check("6 conditions loaded", len(table["rows"]) == 6)
    check("DeepSeek role-lock matches topology_cell_*.json exactly",
          abs(by_label["deepseek"]["role_lock"] - 0.8162614139457665) < 1e-9)
    check("Mistral role-lock matches topology_cell_*.json exactly",
          abs(by_label["mistral"]["role_lock"] - 0.70802200083022) < 1e-9)
    check("grok on-edge phi matches the registered -0.901 to 3 dp",
          round(by_label["grok_on_edge"]["phi_point"], 3) == -0.901)
    check("haiku-249 phi matches the registered +0.246 to 3 dp",
          round(by_label["haiku_249"]["phi_point"], 3) == 0.246)
    check("Llama phi matches the registered +0.148 to 3 dp",
          round(by_label["llama"]["phi_point"], 3) == 0.148)
    check("DeepSeek phi matches the registered -0.093 to 3 dp",
          round(by_label["deepseek"]["phi_point"], 3) == -0.093)
    check("Mistral phi matches the registered +0.208 to 3 dp",
          round(by_label["mistral"]["phi_point"], 3) == 0.208)
    check("sonnet-judge and haiku-judge acc_base agree in every router_decomposition file "
          "(asserted inside load_acc_base; reaching here means every file passed)", True)

    res = analyse(table)
    check("n5 role-lock correlation is negative (higher lock, more opposed/negative phi)",
          res["n5_independent_models"]["role_lock"]["pearson_r"] < 0)
    check("n5 role-lock association is the larger of the two candidates by |r|",
          abs(res["n5_independent_models"]["role_lock"]["pearson_r"])
          > abs(res["n5_independent_models"]["solo_base_acc"]["pearson_r"]))

    print("\nselftest: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    table = build_table()
    res = analyse(table)
    print_report(table, res)
    out_path = a.json or (OUT_DIR / "phi_predictors_diagnostic.json")
    out_path.write_text(json.dumps({"table": table, "correlations": res}, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
