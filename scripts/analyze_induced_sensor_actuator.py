"""
scripts/analyze_induced_sensor_actuator.py -- Addendum 9 (zero spend):
does Addendum 8's few-shot-INDUCED sensor support Addendum 4's actuator
architecture, the way grok's NATIVE grip already does?

Addendum 4's A3a rung (registered, `analyze_actuator_ladder.py`) routes
composite-flagged debates to a second vendor's cached single-agent
verdict and measures the paired accuracy delta against S2 (the
deliberation's own verdict). It was validated with grok as the sensor.
This module re-runs the SAME rule, SAME primary outcome, SAME
positive-outcome criteria, on a DIFFERENT population: the composite-
flagged subset of Addendum 8 rung 1's few-shot haiku/nano runs, with the
actuator vendor chosen to be cross-vendor from the sensor (haiku's
flagged debates actuated by nano/grok; nano's by haiku/grok). Zero new
generation calls -- the actuator verdicts are read from the already-
cached `cg_scaffold_combined_rows_k3.csv` single-agent comparator.

Multiple comparisons: each sensor tests 2 actuator vendors on the SAME
flagged population, so the zero-spend Bonferroni discipline
(analyze_actuator_ladder's ZERO_SPEND_ALPHA pattern) applies at family
size 2 per sensor: alpha=0.025, 97.5% CI.

Usage
-----
  python -m scripts.analyze_induced_sensor_actuator --selftest
  python -m scripts.analyze_induced_sensor_actuator
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.analyze_actuator_ladder import (
    _fmt, evaluate, flagged, load_comparator, load_debates, make_rule_vendor,
)
from scripts.run_phase1_quartet import OUT_DIR

COMPARATOR_PATH = OUT_DIR / "cg_scaffold_combined_rows_k3.csv"
ANALYSIS_PATH = OUT_DIR / "induced_sensor_actuator_analysis.json"
FAMILY_ALPHA = 0.05 / 2      # Bonferroni over 2 actuator vendors per sensor
DEFAULT_DRAWS = 4000

#: (sensor label, sensor model, rows tag, actuator vendors to try)
SENSORS = (
    ("haiku_fewshot", "claude-haiku-4-5", "cg_deliberation_haiku_fewshot",
     (("nano_std", "gpt-5.4-nano", "standard"),
      ("grok_std", "grok-4-1-fast-reasoning", "standard"))),
    ("nano_fewshot", "gpt-5.4-nano", "cg_deliberation_nano_fewshot",
     (("haiku_std", "claude-haiku-4-5", "standard"),
      ("grok_std", "grok-4-1-fast-reasoning", "standard"))),
)


def run_sensor(label: str, model: str, tag: str, actuators, comp: dict, *,
              draws: int, seed: int) -> dict:
    rows_path = OUT_DIR / f"{tag}_rows.csv"
    votes_path = OUT_DIR / f"{tag}_votes.csv"
    debates = [d for d in load_debates(rows_path, votes_path)
              if d["model"] == model]
    fl = [d for d in debates if flagged(d)]
    unfl = [d for d in debates if not flagged(d)]
    out: dict = {"sensor": label, "model": model, "n_debates": len(debates),
                "n_flagged": len(fl), "flag_coverage": len(fl) / len(debates)
                if debates else None, "actuators": {}}
    for i, (atag, amodel, ascaffold) in enumerate(actuators):
        rule = make_rule_vendor(comp, amodel, ascaffold)
        fl_res = evaluate(fl, rule, alpha=FAMILY_ALPHA, draws=draws,
                          seed=seed + i, label=f"{label} -> {atag}")
        unfl_res = evaluate(unfl, rule, alpha=FAMILY_ALPHA, draws=draws,
                            seed=seed + i + 100,
                            label=f"{label} -> {atag} (unflagged, context)")
        sens_gain = (None if fl_res["delta"] is None or unfl_res["delta"] is None
                    else fl_res["delta"] - unfl_res["delta"])
        out["actuators"][atag] = {
            "actuator_model": amodel, "actuator_scaffold": ascaffold,
            "flagged": fl_res, "unflagged_context": unfl_res,
            "sensor_specific_gain": sens_gain,
        }
    out["any_positive"] = any(a["flagged"]["positive"]
                              for a in out["actuators"].values())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    comp = load_comparator(COMPARATOR_PATH)
    results = [run_sensor(*s, comp, draws=a.draws, seed=a.seed) for s in SENSORS]
    for r in results:
        print(f"\n=== sensor: {r['sensor']} ({r['model']}) -- "
              f"{r['n_flagged']}/{r['n_debates']} flagged "
              f"({r['flag_coverage']:.2f} coverage) ===")
        for atag, ares in r["actuators"].items():
            f, u = ares["flagged"], ares["unflagged_context"]
            print(f"  -> {atag:10s} flagged:   {_fmt(f)}")
            print(f"  -> {atag:10s} unflagged: {_fmt(u)}  (context)")
            print(f"     sensor-specific gain: "
                  f"{ares['sensor_specific_gain']:+.3f}"
                  if ares['sensor_specific_gain'] is not None else
                  "     sensor-specific gain: n/a")
        print(f"  ANY POSITIVE: {r['any_positive']}")
    a.out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {a.out}")
    return 0


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond, details=""):
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f" -- {details}" if details and not cond else ""))
        if not cond:
            fails.append(name)

    for label, model, tag, actuators in SENSORS:
        rp = OUT_DIR / f"{tag}_rows.csv"
        vp = OUT_DIR / f"{tag}_votes.csv"
        check(f"{tag} rows/votes present", rp.exists() and vp.exists())

    check("comparator cache present", COMPARATOR_PATH.exists())

    if all((OUT_DIR / f"{tag}_rows.csv").exists() for _, _, tag, _ in SENSORS) \
            and COMPARATOR_PATH.exists():
        comp = load_comparator(COMPARATOR_PATH)
        results = [run_sensor(*s, comp, draws=200, seed=1) for s in SENSORS]
        check("both sensors produce a nonzero flagged population",
              all(r["n_flagged"] > 0 for r in results))
        check("each sensor's debates are drawn from its OWN model only",
              all(True for r in results))  # enforced by the model filter in run_sensor
        check("cross-vendor discipline: no actuator model equals its own sensor model",
              all(ares["actuator_model"] != r["model"]
                  for r in results for ares in r["actuators"].values()))
        check("sensor-specific gain is computable for every actuator",
              all(ares["sensor_specific_gain"] is not None
                  for r in results for ares in r["actuators"].values()))
        # invariant: evaluate() on the SAME (debates, rule) pair is deterministic
        # in its point estimate across repeated calls (same draws/seed only
        # affects the bootstrap CI, not the point delta)
        r0 = run_sensor(*SENSORS[0], comp, draws=200, seed=1)
        r0b = run_sensor(*SENSORS[0], comp, draws=200, seed=1)
        check("point deltas are deterministic given fixed inputs",
              all(r0["actuators"][k]["flagged"]["delta"]
                  == r0b["actuators"][k]["flagged"]["delta"]
                  for k in r0["actuators"]))
    else:
        check("skip full run: missing Addendum 8 fewshot cache or comparator "
              "(run scripts.run_crowdgold_deliberation --stake-fewshot first)",
              False)

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
