"""
scripts/aggregate_phase11.py -- Phase 11 aggregation and figures.

Arm A (single-agent head-to-head): reads p11_singleagent_raw.csv. Per generator,
computes Cliff's delta with bootstrap 95% CI between N-CoT and each TextGrad-optimised
CoT (and standard CoT) on stakeholder_count, uncertainty_score, max_causal_hops,
plus binary fire rates and mean completion tokens. Picks the best TextGrad-CoT
(highest held-out mean sc+us, pooled over generators) and records its source summary
so Arm B can consume it.

Arm B (multi-stakeholder head-to-head): reads debate_dd_headtohead_{arm}.csv for
arms not / textgrad_cot / std_cot. Per arm: R4 full-consensus rate with Wilson 95% CI,
combined (R2-or-R4) convergence, structural-rejection rate. Between-arm Fisher exact
(N-CoT vs TextGrad-CoT) on R4 full consensus. Arm B is skipped gracefully if its
CSVs are absent (so the script can run after Arm A alone to pick the winner).

Emits divergence_study_outputs/p11_summary.json and two figures:
  p11_arm_a_depth.pdf   grouped bars: mean sc / us by condition, per generator
  p11_arm_b_consensus.pdf  R4 full-consensus by arm with Wilson CIs

Usage:
  python -m scripts.aggregate_phase11
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.aggregate_tg_results import cliffs_delta
from scripts import figure_style as fs

OUT = Path("./divergence_study_outputs")

ARM_A_CSV = OUT / "p11_singleagent_raw.csv"
ARM_B_CSVS = {
    "not": OUT / "debate_dd_headtohead_not.csv",
    "textgrad_cot": OUT / "debate_dd_headtohead_textgrad_cot.csv",
    "std_cot": OUT / "debate_dd_headtohead_std_cot.csv",
}
R4_VOTES_CSV = OUT / "p11_r4_votes.csv"
ROLE_ORDER = ["stakeholder_a", "stakeholder_b", "stakeholder_c"]
ROLE_LABELS = {
    "stakeholder_a": "Primary Decision Maker",
    "stakeholder_b": "Affected Third Party",
    "stakeholder_c": "External Advisor",
}
TGX_SUMMARIES = {
    "textgrad_binary": OUT / "tgx_binary_summary.json",
    "textgrad_continuous": OUT / "tgx_continuous_summary.json",
}

CONDITION_ORDER = ["narrative_cot", "textgrad_binary", "textgrad_continuous", "standard_cot"]
COND_LABELS = {
    "narrative_cot": "NoT",
    "textgrad_binary": "TG-CoT (binary)",
    "textgrad_continuous": "TG-CoT (cont.)",
    "standard_cot": "Std CoT",
}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval. Returns (point, lo, hi) as proportions."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def fisher_two_proportion(s1: int, n1: int, s2: int, n2: int) -> dict:
    """Fisher exact test on a 2x2 success/failure table; also a two-proportion z."""
    from scipy.stats import fisher_exact
    table = [[s1, n1 - s1], [s2, n2 - s2]]
    odds, p_fisher = fisher_exact(table)
    # Two-proportion z (normal approx) for cross-check.
    p1 = s1 / n1 if n1 else 0.0
    p2 = s2 / n2 if n2 else 0.0
    p_pool = (s1 + s2) / (n1 + n2) if (n1 + n2) else 0.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if n1 and n2 else 0.0
    z = (p1 - p2) / se if se > 0 else 0.0
    from math import erf, sqrt
    p_z = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return {
        "fisher_odds_ratio": float(odds),
        "fisher_p": float(p_fisher),
        "z": float(z),
        "z_p": float(p_z),
        "p1": p1, "p2": p2, "diff_pp": (p1 - p2) * 100,
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Arm A
# ---------------------------------------------------------------------------

def aggregate_arm_a() -> dict:
    rows = _read_csv(ARM_A_CSV)
    generators = sorted({r["generator"] for r in rows})
    conditions = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in rows)]

    def _col(rs, key):
        return [int(r[key]) for r in rs]

    per_gen: dict[str, dict] = {}
    for gen in generators:
        grows = [r for r in rows if r["generator"] == gen]
        by_cond = {c: [r for r in grows if r["condition"] == c] for c in conditions}
        cond_stats = {}
        for c, rs in by_cond.items():
            if not rs:
                continue
            n = len(rs)
            cond_stats[c] = {
                "n": n,
                "mean_sc": sum(_col(rs, "sc_primary")) / n,
                "mean_us": sum(_col(rs, "us_primary")) / n,
                "mean_mh": sum(_col(rs, "mh_primary")) / n,
                "collapse_rate": sum(_col(rs, "collapse_fired_primary")) / n,
                "suppression_rate": sum(_col(rs, "suppression_fired_primary")) / n,
                "mean_tokens": sum(int(r["completion_tokens"]) for r in rs) / n,
                # secondary judge means for robustness
                "mean_sc_secondary": sum(_col(rs, "sc_secondary")) / n,
                "mean_us_secondary": sum(_col(rs, "us_secondary")) / n,
            }
        # Cliff's delta: N-CoT vs each TextGrad arm (positive = N-CoT richer)
        deltas = {}
        nc = by_cond.get("narrative_cot", [])
        for c in conditions:
            if c == "narrative_cot" or not by_cond.get(c):
                continue
            comp = by_cond[c]
            d = {}
            for metric, key in [("stakeholder_count", "sc_primary"),
                                 ("uncertainty_score", "us_primary"),
                                 ("max_causal_hops", "mh_primary")]:
                delta, (lo, hi) = cliffs_delta(_col(nc, key), _col(comp, key))
                d[metric] = {"delta": delta, "ci": [lo, hi]}
            deltas[c] = d
        per_gen[gen] = {"condition_stats": cond_stats, "ncot_vs_textgrad_cliffs_delta": deltas}

    # Pick best TextGrad condition: highest pooled held-out mean (sc + us), primary judge.
    tg_conditions = [c for c in conditions if c.startswith("textgrad")]
    pooled_depth = {}
    for c in tg_conditions:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        n = len(rs)
        pooled_depth[c] = (sum(_col(rs, "sc_primary")) + sum(_col(rs, "us_primary"))) / n
    best_tg = max(pooled_depth, key=pooled_depth.get) if pooled_depth else None

    return {
        "n_generators": len(generators),
        "generators": generators,
        "conditions": conditions,
        "per_generator": per_gen,
        "pooled_textgrad_depth": pooled_depth,
        "best_textgrad_condition": best_tg,
        "best_textgrad_summary_path": (
            str(TGX_SUMMARIES[best_tg]) if best_tg in TGX_SUMMARIES else None
        ),
    }


# ---------------------------------------------------------------------------
# Arm B
# ---------------------------------------------------------------------------

def _truthy(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def aggregate_arm_b() -> dict | None:
    present = {a: p for a, p in ARM_B_CSVS.items() if p.exists()}
    if not present:
        return None

    arm_stats: dict[str, dict] = {}
    for arm, path in present.items():
        rows = [r for r in _read_csv(path) if not _truthy(r.get("skipped_budget"))]
        n = len(rows)
        if n == 0:
            continue
        r4 = sum(1 for r in rows if _truthy(r.get("r4_full_consensus")))
        r2 = sum(1 for r in rows if _truthy(r.get("r2_consensus_reached")))
        combined = sum(
            1 for r in rows
            if _truthy(r.get("r4_full_consensus")) or _truthy(r.get("r2_consensus_reached"))
        )
        struct = sum(1 for r in rows if _truthy(r.get("r4_structural_rejection")))
        has_synth = sum(1 for r in rows if _truthy(r.get("has_synthesis")))
        p, lo, hi = wilson_ci(r4, n)
        arm_stats[arm] = {
            "n": n,
            "r4_full_consensus": r4,
            "r4_full_consensus_rate": p,
            "r4_wilson_ci": [lo, hi],
            "r2_consensus": r2,
            "combined_convergence_rate": combined / n,
            "structural_rejection_rate": struct / n,
            "has_synthesis_rate": has_synth / n,
        }

    tests = {}
    if "not" in arm_stats and "textgrad_cot" in arm_stats:
        a, b = arm_stats["not"], arm_stats["textgrad_cot"]
        tests["not_vs_textgrad_cot_r4"] = fisher_two_proportion(
            a["r4_full_consensus"], a["n"], b["r4_full_consensus"], b["n"]
        )
    return {"arms": arm_stats, "tests": tests}


# ---------------------------------------------------------------------------
# Arm B: role concentration of R4 rejections
# ---------------------------------------------------------------------------

def _norm_entropy(counts: list[int]) -> float:
    """Shannon entropy over roles, normalised to [0,1]. 0 = all in one role
    (concentrated); 1 = uniform across the three roles (scattered)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    if len(ps) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(len(ROLE_ORDER))


def aggregate_role_concentration() -> dict | None:
    """Where do each arm's R4 rejections land? NoT's hypothesis: rejections
    concentrate in the principle-check role (External Advisor); an optimised
    CoT scatters across roles, including the Primary Decision Maker rejecting
    the very synthesis it helped build."""
    if not R4_VOTES_CSV.exists():
        return None
    rows = _read_csv(R4_VOTES_CSV)
    arms = [a for a in ["not", "textgrad_cot", "std_cot"]
            if any(r["arm"] == a for r in rows)]
    out: dict[str, dict] = {}
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        rej = [r for r in arm_rows if r.get("reject") == "1"]
        by_role = {pid: sum(1 for r in rej if r["pid"] == pid) for pid in ROLE_ORDER}
        counts = [by_role[pid] for pid in ROLE_ORDER]
        total = sum(counts)
        modal_pid = max(by_role, key=by_role.get) if total else None
        out[arm] = {
            "n_votes": len(arm_rows),
            "n_rejections": total,
            "rejections_by_role": by_role,
            "role_share": {pid: (by_role[pid] / total if total else 0.0) for pid in ROLE_ORDER},
            "modal_role": modal_pid,
            "modal_role_share": (max(counts) / total) if total else 0.0,
            "normalised_entropy": _norm_entropy(counts),
            "advisor_share": (by_role["stakeholder_c"] / total) if total else 0.0,
            "decider_share": (by_role["stakeholder_a"] / total) if total else 0.0,
            # secondary (sparse) signal: integrator-unaddressed roles
            "undermined_rejections": sum(1 for r in rej if r.get("undermined") == "1"),
            "unknown_votes": sum(1 for r in arm_rows if r.get("vote") == "UNKNOWN"),
        }
    return {"arms": out}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_arm_a(arm_a: dict) -> None:
    import numpy as np
    gens = arm_a["generators"]
    conds = arm_a["conditions"]
    fig, axes, _ = fs.panel_grid(len(gens), panel_w=2.4, panel_h=2.6, sharey=True)
    x = np.arange(len(conds))
    width = 0.38
    for ax, gen in zip(axes, gens):
        cs = arm_a["per_generator"][gen]["condition_stats"]
        sc = [cs.get(c, {}).get("mean_sc", 0) for c in conds]
        us = [cs.get(c, {}).get("mean_us", 0) for c in conds]
        ax.bar(x - width / 2, sc, width, label="stakeholder count",
               color=fs.CONDITION_COLORS["narrative_cot"])
        ax.bar(x + width / 2, us, width, label="uncertainty score",
               color=fs.CONDITION_COLORS["standard_cot"])
        ax.set_title(fs.short_generator_label(gen))
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS.get(c, c) for c in conds], rotation=30, ha="right",
                           fontsize=6.5)
    axes[0].set_ylabel("mean (primary judge)")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="upper right")
    fs.save(fig, OUT / "p11_arm_a_depth.pdf")


def figure_arm_b(arm_b: dict) -> None:
    import numpy as np
    arms = [a for a in ["not", "textgrad_cot", "std_cot"] if a in arm_b["arms"]]
    if not arms:
        return
    fig, ax = fs.single_panel(width=fs.COL_WIDTH, height=2.6)
    x = np.arange(len(arms))
    rates = [arm_b["arms"][a]["r4_full_consensus_rate"] for a in arms]
    los = [arm_b["arms"][a]["r4_wilson_ci"][0] for a in arms]
    his = [arm_b["arms"][a]["r4_wilson_ci"][1] for a in arms]
    yerr = [[r - lo for r, lo in zip(rates, los)], [hi - r for r, hi in zip(rates, his)]]
    colors = [fs.CONDITION_COLORS["narrative_cot"],
              fs.VENDOR_COLORS["openai"], fs.CONDITION_COLORS["standard_cot"]]
    ax.bar(x, rates, 0.6, yerr=yerr, capsize=3, color=colors[:len(arms)])
    ax.set_xticks(x)
    ax.set_xticklabels({"not": "NoT", "textgrad_cot": "TG-CoT", "std_cot": "Std CoT"}[a]
                       for a in arms)
    fs.percent_axis(ax, 1.0)
    ax.set_ylabel("R4 full-consensus rate")
    ax.set_title("Multi-stakeholder consensus by arm")
    fs.save(fig, OUT / "p11_arm_b_consensus.pdf")


def figure_role_concentration(rc: dict) -> None:
    """Stacked bars: R4 rejection counts by role per arm."""
    import numpy as np
    arms = [a for a in ["not", "textgrad_cot", "std_cot"] if a in rc["arms"]]
    if not arms:
        return
    fig, ax = fs.single_panel(width=fs.COL_WIDTH, height=2.7)
    x = np.arange(len(arms))
    role_colors = {
        "stakeholder_a": fs.CONDITION_COLORS["standard_cot"],
        "stakeholder_b": fs.VENDOR_COLORS["openai"],
        "stakeholder_c": fs.CONDITION_COLORS["narrative_cot"],
    }
    bottom = np.zeros(len(arms))
    for pid in ROLE_ORDER:
        vals = np.array([rc["arms"][a]["rejections_by_role"][pid] for a in arms], dtype=float)
        ax.bar(x, vals, 0.6, bottom=bottom, label=ROLE_LABELS[pid], color=role_colors[pid])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels({"not": "NoT", "textgrad_cot": "TG-CoT", "std_cot": "Std CoT"}[a]
                       for a in arms)
    ax.set_ylabel("R4 rejections (per-perspective)")
    ax.set_title("Where R4 rejections land, by role")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fs.save(fig, OUT / "p11_arm_b_rejection_roles.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fmt_delta(d: dict) -> str:
    return f"{d['delta']:+.3f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]"


def main() -> int:
    summary: dict = {}

    if ARM_A_CSV.exists():
        arm_a = aggregate_arm_a()
        summary["arm_a"] = arm_a
        print("=== Arm A: single-agent head-to-head (primary judge) ===")
        for gen in arm_a["generators"]:
            print(f"\n{gen}")
            cs = arm_a["per_generator"][gen]["condition_stats"]
            for c in arm_a["conditions"]:
                if c not in cs:
                    continue
                s = cs[c]
                print(f"  {COND_LABELS.get(c, c):16s} sc={s['mean_sc']:.2f} us={s['mean_us']:.2f} "
                      f"mh={s['mean_mh']:.2f} collapse={s['collapse_rate']:.0%} "
                      f"suppr={s['suppression_rate']:.0%} tok={s['mean_tokens']:.0f}")
            print("  Cliff delta (NoT vs TextGrad; + = NoT richer):")
            for c, d in arm_a["per_generator"][gen]["ncot_vs_textgrad_cliffs_delta"].items():
                print(f"    vs {COND_LABELS.get(c, c):16s} "
                      f"sc={_fmt_delta(d['stakeholder_count'])} "
                      f"us={_fmt_delta(d['uncertainty_score'])}")
        print(f"\nBest TextGrad condition (pooled depth): {arm_a['best_textgrad_condition']}")
        print(f"  -> {arm_a['best_textgrad_summary_path']}")
        try:
            figure_arm_a(arm_a)
        except Exception as e:
            print(f"  [figure_arm_a skipped: {e}]")
    else:
        print(f"Arm A CSV not found ({ARM_A_CSV}); skipping Arm A.")

    arm_b = aggregate_arm_b()
    if arm_b:
        summary["arm_b"] = arm_b
        print("\n=== Arm B: multi-stakeholder head-to-head ===")
        for arm, s in arm_b["arms"].items():
            lo, hi = s["r4_wilson_ci"]
            print(f"  {arm:14s} n={s['n']:3d} R4-consensus={s['r4_full_consensus_rate']:.0%} "
                  f"[{lo:.0%},{hi:.0%}] combined={s['combined_convergence_rate']:.0%} "
                  f"struct-reject={s['structural_rejection_rate']:.0%}")
        for name, t in arm_b["tests"].items():
            print(f"  test {name}: diff={t['diff_pp']:+.1f}pp  Fisher p={t['fisher_p']:.3f}  "
                  f"z p={t['z_p']:.3f}")
        try:
            figure_arm_b(arm_b)
        except Exception as e:
            print(f"  [figure_arm_b skipped: {e}]")
    else:
        print("\nArm B CSVs not found; skipping Arm B (run the debates first).")

    rc = aggregate_role_concentration()
    if rc:
        summary["arm_b_role_concentration"] = rc
        print("\n=== Arm B: role concentration of R4 rejections ===")
        for arm in ["not", "textgrad_cot", "std_cot"]:
            if arm not in rc["arms"]:
                continue
            s = rc["arms"][arm]
            by = s["rejections_by_role"]
            modal = ROLE_LABELS.get(s["modal_role"], s["modal_role"]) if s["modal_role"] else "-"
            print(f"  {arm:14s} rej={s['n_rejections']:2d}/{s['n_votes']:3d}  "
                  f"a={by['stakeholder_a']} b={by['stakeholder_b']} c={by['stakeholder_c']}  "
                  f"modal={modal} ({s['modal_role_share']:.0%})  "
                  f"advisor={s['advisor_share']:.0%} decider={s['decider_share']:.0%}  "
                  f"entropy={s['normalised_entropy']:.2f}  unknown={s['unknown_votes']}")
        try:
            figure_role_concentration(rc)
        except Exception as e:
            print(f"  [figure_role_concentration skipped: {e}]")

    (OUT / "p11_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {OUT / 'p11_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
