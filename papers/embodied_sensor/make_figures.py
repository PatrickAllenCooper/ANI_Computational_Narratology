"""Figures for the ICLR submission. results.tex includes fig1, fig3, fig4 and fig5;
fig6 (the roles-by-edges two-by-two of Table tab-topology) is drawn for the record
and has no \\includegraphics yet. fig2 (grip against lift) is not referenced by
results.tex (header note (d)) and is not generated.

House rule. Every number that appears in a figure is read from a JSON artefact in
divergence_study_outputs/ at build time through one accessor (V for a stored value,
D for a value derived from stored values) that records the artefact file and key path
of everything drawn. Nothing numeric is hard-coded except pure reference constants
(chance 0.5, ratio 1, lift 0) and layout. `--check` rebuilds every figure without
saving, prints every plotted value with its artefact key and asserts that each equals a
fresh read of the artefact to 1e-6 (derived values are recomputed from fresh
components), then runs the cross-artefact consistency checks listed under CROSS.

Sources (JSON = divergence_study_outputs/<file>)
  fig1  rolelock_by_model.json (P(at-fault), seat accuracy, role-lock and n_r0 per
        model; the scripted regeneration of papers/embodied_sensor/figures/
        fig1_rolelock.json, which --check asserts it equals field by field, as it does
        effective_voter.json panels.*.fig1_reference); effective_voter.json
        panels.*.role_lock lo / hi for the lock intervals in panel B;
        crowdgold_aita_summary.json n_gold_yta / (n_gold_yta + n_gold_nta) for the
        gold base rate line in panel A.
  fig3  transfer_readouts.json targets.*.by_sensor.counter (the stake-blind counter,
        two or more seats objecting, the registered sensor of 16.15) with the
        artefact's own intervals and solo accuracies for the four targets;
        router_decomposition.json counter_lift_by_stratum.pooled and
        composed_haiku.routers.collective_alone for the collective's own row.
  fig4  unembodied_ablation_analysis.json baseline_full_panel.localisation (observed,
        independence_poisson_binomial, per_seat_marginals) and
        localisation_excess_ci; effective_voter.json panels.grok (neutral seat,
        majority and CV oracle on the triples, with the oracle's gap and interval);
        router_decomposition.json composed_haiku.routers.collective_alone;
        transfer_readouts.json targets.grok_solo_standard.solo_accuracy.
  fig5  router_decomposition.json composed_haiku.routers and
        composed_sonnet.block.routers (collective alone, counter, verdict type, union,
        each with coverage and interval) and each judge's random_at_union_coverage;
        the two solo grok baselines from transfer_readouts.json. The former "route
        everything to the judge" point (haiku everywhere 0.850, prereg L891) has no
        artefact field and is no longer drawn.
  fig6  topology_2x2_analysis.json cells.*.role_lock, objection.fire_rate,
        objection.localisation_excess, aggregation.group_verdict_s2, n_debates and
        lift.n_fired (the item-matched `cells` block that Table tab-topology cites).

Form. Accuracy readouts on a zoomed axis (fig4 B, fig6 D) are drawn as points with their
artefact interval where one exists, never as bars from a truncated baseline; fig1 A's
legend sits in a band above the tallest bar.

Zero model or API calls. Run from anywhere:
    .venv/bin/python papers/embodied_sensor/make_figures.py          # writes figures/fig{1,3,4,5,6}.png
    .venv/bin/python papers/embodied_sensor/make_figures.py --check  # prints and verifies every value
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DSO = os.path.join(ROOT, "divergence_study_outputs")
OUT = os.path.join(HERE, "figures")
TOL = 1e-6
PRODUCED = ["fig1.png", "fig3.png", "fig4.png", "fig5.png", "fig6.png"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"emb": "#1f4e79", "ctl": "#b35806", "gray": "#7f7f7f", "acc": "#2b8a3e", "warn": "#c0392b"}

# artefact short names
RL = "rolelock_by_model.json"
EV = "effective_voter.json"
CG = "crowdgold_aita_summary.json"
RD = "router_decomposition.json"
TR = "transfer_readouts.json"
UA = "unembodied_ablation_analysis.json"
TP = "topology_2x2_analysis.json"
FIG1_SCRATCH = os.path.join(HERE, "figures", "fig1_rolelock.json")  # the un-scripted original

# ---------------------------------------------------------------- artefact access
_cache = {}


def _path(fname):
    return fname if os.path.isabs(fname) else os.path.join(DSO, fname)


def _load(fname, fresh=False):
    p = _path(fname)
    if fresh:
        with open(p) as f:
            return json.load(f)
    if p not in _cache:
        with open(p) as f:
            _cache[p] = json.load(f)
    return _cache[p]


def _walk(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def _key(fname, path):
    return f"{os.path.basename(fname)}:{'.'.join(str(k) for k in path)}"


REGISTRY = []   # every plotted value: fig, label, file, path or formula, value
CROSS = []      # cross-artefact equalities asserted under --check


def V(fig, label, fname, *path):
    """Read one stored value and register it."""
    val = _walk(_load(fname), path)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"{_key(fname, path)} is not a number: {val!r}")
    REGISTRY.append({"fig": fig, "label": label, "file": fname, "path": path, "value": val})
    return val


def D(fig, label, fname, formula, fn, *paths):
    """Register a value derived from stored values; --check recomputes it from fresh reads."""
    comps = [_walk(_load(fname), p) for p in paths]
    val = fn(*comps)
    REGISTRY.append({"fig": fig, "label": label, "file": fname, "path": None,
                     "formula": formula, "fn": fn, "paths": paths, "value": val})
    return val


def X(label, a, b):
    """Assert under --check that two artefact fields agree (a, b are (file, path) pairs)."""
    CROSS.append({"label": label, "a": a, "b": b})


MODELS = ["grok", "sonnet", "nano", "haiku"]
SEATS = ["writer_advocate", "counterparty", "neutral_adjudicator"]
SEAT_LABELS = ["writer's advocate", "counterparty", "neutral"]
SEAT_COLORS = [C["emb"], C["warn"], C["gray"]]


def _ci_bar(ax, y, lo, hi, color, lw=2, tick=0.15, horizontal=True):
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=2)
        ax.plot([lo, lo], [y - tick, y + tick], color=color, lw=1.2, zorder=2)
        ax.plot([hi, hi], [y - tick, y + tick], color=color, lw=1.2, zorder=2)
    else:
        ax.plot([y, y], [lo, hi], color=color, lw=lw, solid_capstyle="butt", zorder=2)
        ax.plot([y - tick, y + tick], [lo, lo], color=color, lw=1.2, zorder=2)
        ax.plot([y - tick, y + tick], [hi, hi], color=color, lw=1.2, zorder=2)


def _save(fig, name, save):
    if save:
        fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1
def fig1(save=True):
    F = "fig1"
    p_at = {m: {s: V(F, f"A P(at-fault) {m} {s}", RL, m, "p_at_fault", s) for s in SEATS} for m in MODELS}
    acc = {m: {s: V(F, f"B seat accuracy {m} {s}", RL, m, "seat_acc", s) for s in SEATS} for m in MODELS}
    lock = {m: V(F, f"A/B role-lock {m}", RL, m, "role_lock") for m in MODELS}
    n_r0 = {m: V(F, f"A n_r0 {m}", RL, m, "n_r0") for m in MODELS}
    lock_lo = {m: V(F, f"B role-lock lo {m}", EV, "panels", m, "role_lock", "lo") for m in MODELS}
    lock_hi = {m: V(F, f"B role-lock hi {m}", EV, "panels", m, "role_lock", "hi") for m in MODELS}
    n_gold = V(F, "A gold at-fault items", CG, "n_gold_yta")
    n_items = V(F, "A corpus items", CG, "n_items")
    base = D(F, "A gold base rate", CG, "n_gold_yta / (n_gold_yta + n_gold_nta)",
             lambda y, n: y / (y + n), ("n_gold_yta",), ("n_gold_nta",))
    # the drawn lock is the centre of the interval drawn around it, and the scripted
    # artefact equals the original scratch JSON and effective_voter's fig1_reference
    for m in MODELS:
        X(f"role-lock {m}: rolelock_by_model == effective_voter point", (RL, (m, "role_lock")),
          (EV, ("panels", m, "role_lock", "point")))
        X(f"role-lock {m}: rolelock_by_model == fig1_rolelock.json", (RL, (m, "role_lock")),
          (FIG1_SCRATCH, (m, "role_lock")))
        X(f"n_r0 {m}: rolelock_by_model == fig1_rolelock.json", (RL, (m, "n_r0")), (FIG1_SCRATCH, (m, "n_r0")))
        X(f"n_r0 {m}: rolelock_by_model == effective_voter fig1_reference", (RL, (m, "n_r0")),
          (EV, ("panels", m, "fig1_reference", "n_r0")))
        for s in SEATS:
            X(f"P(at-fault) {m} {s}: rolelock_by_model == fig1_rolelock.json",
              (RL, (m, "p_at_fault", s)), (FIG1_SCRATCH, (m, "p_at_fault", s)))
            X(f"seat_acc {m} {s}: rolelock_by_model == fig1_rolelock.json",
              (RL, (m, "seat_acc", s)), (FIG1_SCRATCH, (m, "seat_acc", s)))
            X(f"P(at-fault) {m} {s}: rolelock_by_model == effective_voter fig1_reference",
              (RL, (m, "p_at_fault", s)), (EV, ("panels", m, "fig1_reference", "p_at_fault", s)))
            X(f"seat_acc {m} {s}: rolelock_by_model == effective_voter fig1_reference",
              (RL, (m, "seat_acc", s)), (EV, ("panels", m, "fig1_reference", "seat_acc_per_seat_codable", s)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(MODELS)); w = 0.26
    for i, (s, lab) in enumerate(zip(SEATS, SEAT_LABELS)):
        a1.bar(x + (i - 1) * w, [p_at[m][s] for m in MODELS], w, label=lab, color=SEAT_COLORS[i], alpha=0.9)
    a1.axhline(base, ls=":", color="k", lw=0.8, label=f"gold base rate {base:.3f} ({n_gold} of {n_items} items)")
    a1.set_xticks(x)
    a1.set_xticklabels([f"{m}\nlock {lock[m]:.2f}, n {n_r0[m]:,}" for m in MODELS], fontsize=8.5)
    a1.set_ylabel("P(verdict = at-fault) at opening statement")
    # ylim 1.30 leaves a clear band above the tallest bar for the legend, so it never
    # overprints the grok / sonnet counterparty bars; ticks stay on the probability scale
    a1.set_ylim(0, 1.30); a1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a1.set_title("A. Assigned interest pins the verdict", loc="left", fontsize=10.5, fontweight="bold")
    a1.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    # one lock interval per model, drawn once under the points on two staggered rows
    row = {"haiku": 0.365, "nano": 0.365, "sonnet": 0.345, "grok": 0.345}
    for m in MODELS:
        a2.plot([lock_lo[m], lock_hi[m]], [row[m], row[m]], color="#bbbbbb", lw=3, solid_capstyle="butt", zorder=1)
        up = row[m] > 0.35
        a2.text(lock[m], row[m] + (0.012 if up else -0.012), m, ha="center", va="bottom" if up else "top",
                fontsize=7.5, color="#444")
    for i, (s, lab) in enumerate(zip(SEATS, SEAT_LABELS)):
        a2.plot([lock[m] for m in MODELS], [acc[m][s] for m in MODELS], "o", ms=6, color=SEAT_COLORS[i], label=lab, zorder=3)
    a2.set_xlabel("role-lock  |P(at-fault) writer − counterparty|\n(grey bars, 95% item-clustered CI)")
    a2.set_ylabel("that seat's verdict accuracy")
    a2.set_title("B. …and destroys its verdict-information", loc="left", fontsize=10.5, fontweight="bold")
    a2.set_ylim(0.30, 1.03); a2.set_xlim(0.3, 1.02); a2.axhline(0.5, ls=":", color="k", lw=0.8)
    a2.text(0.31, 0.512, "chance", fontsize=7.5)
    fig.tight_layout(); _save(fig, "fig1.png", save)


# ---------------------------------------------------------------- Fig 3 (headline)
TARGETS = [("grok_solo_standard", "grok solo, standard"), ("grok_solo_narrative", "grok solo, narrated"),
           ("nano_standard", "nano solo"), ("haiku_standard", "haiku solo")]


def fig3(save=True):
    F = "fig3"
    pw_f = V(F, "A collective P(wrong | fired)", RD, "counter_lift_by_stratum", "pooled", "p_wrong_fired")
    pw_u = V(F, "A collective P(wrong | unfired)", RD, "counter_lift_by_stratum", "pooled", "p_wrong_unfired")
    ratio = [V(F, "A collective ratio", RD, "counter_lift_by_stratum", "pooled", "ratio")]
    n_fired = V(F, "A collective n_fired", RD, "counter_lift_by_stratum", "pooled", "n_fired")
    n_deb = V(F, "A collective n debates", RD, "counter_lift_by_stratum", "pooled", "n")
    coll_acc = V(F, "A collective accuracy", RD, "composed_haiku", "routers", "collective_alone", "composed", "acc")
    solo_acc, n_cells, lift, lo, hi = [], [], [], [], []
    for k, lab in TARGETS:
        solo_acc.append(V(F, f"A {lab} solo accuracy", TR, "targets", k, "solo_accuracy"))
        n_cells.append(V(F, f"A/B {lab} codable cells", TR, "targets", k, "by_sensor", "counter", "n"))
        ratio.append(V(F, f"A {lab} ratio", TR, "targets", k, "by_sensor", "counter", "ratio"))
        lift.append(V(F, f"B {lab} lift", TR, "targets", k, "by_sensor", "counter", "lift"))
        lo.append(V(F, f"B {lab} lift lo", TR, "targets", k, "by_sensor", "counter", "lo"))
        hi.append(V(F, f"B {lab} lift hi", TR, "targets", k, "by_sensor", "counter", "hi"))
    draws = V(F, "B bootstrap draws", TR, "draws")
    gens = [f"collective itself\n(acc {coll_acc:.3f}, {n_deb:,} debates)"] + \
           [f"{lab}\n(acc {solo_acc[i]:.3f}, {n_cells[i]} cells)" for i, (k, lab) in enumerate(TARGETS)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(gens))
    a1.barh(y, ratio, 0.55, color=[C["warn"]] + [C["emb"]] * len(TARGETS))
    for i in range(len(gens)):
        a1.text(ratio[i] + 0.08, y[i], f"{ratio[i]:.2f}×", va="center", fontsize=9, fontweight="bold")
    a1.axvline(1, color="k", lw=0.8, ls=":")
    a1.set_yticks(y); a1.set_yticklabels(gens, fontsize=8.5); a1.invert_yaxis()
    a1.set_xlabel("P(wrong | counter fired) / P(wrong | not fired)"); a1.set_xlim(0, 6)
    a1.set_title("A. Error ratio on flagged cells, by generator", loc="left", fontsize=10.5, fontweight="bold")
    a1.axhspan(y[0] - 0.5, y[0] + 0.5, color="#f2f2f2", zorder=0)
    a1.text(0.05, y[0] + 0.49, f"reference, the counter's own collective ({pw_f:.3f} vs {pw_u:.3f}, {n_fired} fired)",
            fontsize=7, va="bottom", color="#666")
    for i in range(len(TARGETS)):
        _ci_bar(a2, y[i + 1], lo[i], hi[i], C["emb"])
        a2.plot(lift[i], y[i + 1], "o", color=C["emb"], ms=6, zorder=3)
        a2.text(hi[i] + 0.01, y[i + 1], f"{lift[i]:+.3f}", va="center", fontsize=7.5, color="#444")
    a2.axvline(0, color="k", lw=0.8)
    a2.set_yticks(y[1:]); a2.set_yticklabels([g.split("\n")[0] for g in gens[1:]], fontsize=8.5); a2.invert_yaxis()
    a2.set_ylim(y[-1] + 0.6, y[1] - 0.6); a2.set_xlim(-0.08, 0.52)
    a2.set_xlabel(f"lift, 95% item-clustered CI ({draws:,} draws)")
    a2.set_title("B. Transfer lifts", loc="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(); _save(fig, "fig3.png", save)


# ---------------------------------------------------------------- Fig 4
def fig4(save=True):
    F = "fig4"
    bfp = ("baseline_full_panel",)  # not "control" (stake-free) nor "baseline" (item-matched)
    ks = ["0", "1", "2", "3"]
    obs = [V(F, f"A observed share, {k} objecting", UA, *bfp, "localisation", "observed", k) for k in ks]
    pb = [V(F, f"A Poisson-binomial null, {k} objecting", UA, *bfp, "localisation", "independence_poisson_binomial", k)
          for k in ks]
    marg = [V(F, f"A per-seat marginal {s}", UA, *bfp, "localisation", "per_seat_marginals", s) for s in SEATS]
    excess = V(F, "A localisation excess", UA, *bfp, "localisation_excess_ci", "point")
    ex_lo = V(F, "A localisation excess lo", UA, *bfp, "localisation_excess_ci", "lo")
    ex_hi = V(F, "A localisation excess hi", UA, *bfp, "localisation_excess_ci", "hi")
    n_ua = V(F, "A debates", UA, *bfp, "n_debates")
    g = ("panels", "grok")
    n_tri = V(F, "B triples", EV, *g, "n_triples")
    neu = [V(F, f"B neutral seat {q}", EV, *g, "seat_acc_on_triples", "neutral_adjudicator", q) for q in ("point", "lo", "hi")]
    maj = [V(F, f"B majority {q}", EV, *g, "majority_acc_on_triples", q) for q in ("point", "lo", "hi")]
    cv = V(F, "B CV oracle accuracy", EV, *g, "cv_oracle_bits", "cv_oracle_acc")
    gap = [V(F, f"B CV oracle gap {q}", EV, *g, "cv_oracle_bits", q) for q in ("gap", "lo", "hi")]
    ca = ("composed_haiku", "routers", "collective_alone", "composed")
    coll = [V(F, f"B collective verdict {q}", RD, *ca, q) for q in ("acc", "lo", "hi")]
    n_deb = V(F, "B collective debates", RD, "composed_haiku", "n_debates")
    solo = V(F, "B grok solo accuracy", TR, "targets", "grok_solo_standard", "solo_accuracy")
    n_cells = V(F, "B grok solo cells", TR, "n_cells")
    X("CV oracle neutral_acc == neutral seat point", (EV, (*g, "cv_oracle_bits", "neutral_acc")),
      (EV, (*g, "seat_acc_on_triples", "neutral_adjudicator", "point")))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.2, 1.35]})
    x = np.arange(4)
    a1.bar(x - 0.18, pb, 0.36, color=C["gray"], label="independence (Poisson-binomial,\nper-seat marginals)")
    a1.bar(x + 0.18, obs, 0.36, color=C["emb"], label=f"observed ({n_ua:,} debates)")
    a1.set_xticks(x); a1.set_xticklabels(ks); a1.set_xlabel("number of seats objecting")
    a1.set_ylabel("share of debates"); a1.set_ylim(0, 1.0)
    a1.legend(frameon=False, fontsize=7.5, loc="upper right")
    a1.set_title("A. Dissent concentrates in ONE seat", loc="left", fontsize=10.5, fontweight="bold")
    a1.annotate(f"localisation excess\n{excess:+.3f} [{ex_lo:+.3f}, {ex_hi:+.3f}]", (1.18, obs[1]), xytext=(1.45, 0.66),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    a1.text(1.95, 0.36, "per-seat objection rates\n" + "\n".join(f"{lab}  {v:.3f}" for lab, v in zip(SEAT_LABELS, marg)),
            fontsize=7.5, va="bottom", color="#444")
    rows = [(f"neutral seat alone ({n_tri:,} debates)", neu[0], neu[1], neu[2], C["gray"]),
            ("majority of 3 seats", maj[0], maj[1], maj[2], C["gray"]),
            ("best fixed rule over the 3 opening\nverdicts (cross-validated)", cv, None, None, C["gray"]),
            (f"full protocol verdict ({n_deb:,} debates)", coll[0], coll[1], coll[2], C["emb"]),
            (f"grok solo, 3 calls ({n_cells} cells)", solo, None, None, C["acc"])]
    # dot-and-interval, not bars: the axis is zoomed to the 0.78-0.97 band, and a bar
    # drawn from a truncated baseline would exaggerate the differences between rows
    for i, (lab, v, lo, hi, col) in enumerate(rows):
        a2.axhline(i, color="#e6e6e6", lw=0.8, zorder=0)
        if lo is not None:
            _ci_bar(a2, i, lo, hi, col, lw=1.6, tick=0.12)
            a2.text(hi + 0.004, i, f"{v:.3f} [{lo:.3f}, {hi:.3f}]", va="center", fontsize=7.5)
        elif i == 2:
            a2.text(v + 0.006, i, f"{v:.3f}, gap to neutral\n{gap[0]:+.3f} [{gap[1]:+.3f}, {gap[2]:+.3f}]",
                    va="center", fontsize=7.5)
        else:
            a2.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=7.5)
        a2.plot(v, i, "o", ms=8, color=col, mec="k", mew=0.6, zorder=3)
    a2.set_yticks(range(len(rows))); a2.set_yticklabels([r[0] for r in rows], fontsize=8); a2.invert_yaxis()
    a2.set_ylim(len(rows) - 0.5, -0.5)
    a2.set_xlim(0.78, 0.97); a2.set_xlabel("accuracy, 95% item-clustered CI where one exists\n(zoomed axis)")
    a2.set_title("B. One effective voter", loc="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(); _save(fig, "fig4.png", save)


# ---------------------------------------------------------------- Fig 5
ROUTERS = [("collective_alone", "collective alone"), ("counter", "counter, 2+ seats object"),
           ("verdict_type", "verdict type, both parties"), ("union", "union of the two")]
JUDGES = [("A", "haiku", ("composed_haiku",)), ("B", "sonnet", ("composed_sonnet", "block"))]


def fig5(save=True):
    F = "fig5"
    solo_std = V(F, "A/B grok solo standard", TR, "targets", "grok_solo_standard", "solo_accuracy")
    solo_nar = V(F, "A/B grok solo narrated", TR, "targets", "grok_solo_narrative", "solo_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    x_text = 0.33
    box = dict(facecolor="white", edgecolor="none", pad=1.0)
    for ax, (P, judge, base) in zip(axes, JUDGES):
        n_deb = V(F, f"{P} {judge} debates", RD, *base, "n_debates")
        pts = {}
        for key, lab in ROUTERS:
            r = (*base, "routers", key)
            pts[key] = dict(cov=V(F, f"{P} {judge} {lab} coverage", RD, *r, "coverage"),
                            acc=V(F, f"{P} {judge} {lab} accuracy", RD, *r, "composed", "acc"),
                            lo=V(F, f"{P} {judge} {lab} lo", RD, *r, "composed", "lo"),
                            hi=V(F, f"{P} {judge} {lab} hi", RD, *r, "composed", "hi"),
                            n=V(F, f"{P} {judge} {lab} routed", RD, *r, "n_fire"), lab=lab)
        rn = (*base, "random_at_union_coverage")
        rnd = {q: V(F, f"{P} {judge} random {q}", RD, *rn, q) for q in ("coverage", "mean", "best", "p05", "p95", "n_seeds")}
        ax.axhline(solo_std, ls="--", color=C["acc"], lw=1.2, zorder=1)
        ax.text(0.005, solo_std + 0.003, f"grok solo  {solo_std:.3f}", fontsize=7.6, color=C["acc"])
        ax.axhline(solo_nar, ls="--", color=C["acc"], lw=0.8, alpha=0.7, zorder=1)
        ax.text(0.005, solo_nar + 0.003, f"narrated  {solo_nar:.3f}", fontsize=7.6, color=C["acc"], alpha=0.85)
        for key, p in pts.items():
            col = C["gray"] if key == "collective_alone" else C["emb"]
            _ci_bar(ax, p["cov"], p["lo"], p["hi"], col, lw=1.2, tick=0.006, horizontal=False)
            ax.scatter(p["cov"], p["acc"], s=60, color=col, zorder=3, edgecolor="k")
            if key == "collective_alone":
                ax.annotate(f"{p['lab']}  {p['acc']:.3f}", (p["cov"], p["acc"]), xytext=(8, -12),
                            textcoords="offset points", fontsize=7.6)
            else:
                ax.plot([p["cov"] + 0.012, x_text - 0.006], [p["acc"], p["acc"]], color="#bbbbbb", lw=0.6, zorder=2)
                ax.text(x_text, p["acc"], f"{p['lab']}  {p['acc']:.3f}  ({p['n']} routed)", va="center",
                        fontsize=7.6, bbox=box, zorder=4)
        ax.plot([rnd["coverage"]] * 2, [rnd["p05"], rnd["p95"]], color=C["warn"], lw=1.2, zorder=2)
        ax.scatter(rnd["coverage"], rnd["mean"], marker="x", s=70, color=C["warn"], zorder=3)
        ax.plot([rnd["coverage"] + 0.012, x_text - 0.006], [rnd["mean"]] * 2, color="#e8b4ad", lw=0.6, zorder=2)
        ax.text(x_text, rnd["mean"], f"random routing at the union's coverage\nmean {rnd['mean']:.3f}, "
                f"best {rnd['best']:.3f}, {rnd['n_seeds']} seeds", va="center", fontsize=7.2, color=C["warn"],
                bbox=box, zorder=4)
        ax.set_xlabel("fraction of debates routed to the judge")
        ax.set_xlim(-0.03, 0.70); ax.set_ylim(0.79, 0.99)
        ax.set_title(f"{P}. {judge} judge, majority-of-3, {n_deb:,} debates", loc="left", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("composed-system accuracy, 95% CI")
    fig.tight_layout(); _save(fig, "fig5.png", save)


# ---------------------------------------------------------------- Fig 6 (the 2x2)
CELLS = [("embodied", "on"), ("embodied", "off"), ("identical", "on"), ("identical", "off")]
# (title, artefact path under cells.<cell>, label format, ylim, mark). Panels A-C are bars
# from zero. Panel D is drawn as points on a zoomed axis (the artefact carries no interval
# for group_verdict_s2): a bar from a truncated 0.78 baseline would exaggerate the gaps.
READOUTS = [("A. role-lock", ("role_lock",), "{:.3f}", (0, 1.08), "bar"),
            ("B. fire rate, two or more object", ("objection", "fire_rate"), "{:.3f}", (0, 0.26), "bar"),
            ("C. localisation excess", ("objection", "localisation_excess"), "{:+.3f}", (0, 0.40), "bar"),
            ("D. collective's verdict accuracy", ("aggregation", "group_verdict_s2"), "{:.3f}", (0.78, 0.905), "dot")]


def fig6(save=True):
    F = "fig6"
    n_deb = {c: V(F, f"debates {c[0]}/{c[1]}", TP, "cells", f"{c[0]}/{c[1]}", "n_debates") for c in CELLS}
    n_fired = {c: V(F, f"B fired {c[0]}/{c[1]}", TP, "cells", f"{c[0]}/{c[1]}", "lift", "n_fired") for c in CELLS}
    n_items = V(F, "items matched", TP, "n_items_matched")
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    x = np.array([0, 1]); w = 0.36
    for ax, (title, path, fmt, ylim, mark) in zip(axes, READOUTS):
        for j, roles in enumerate(["embodied", "identical"]):
            vals = [V(F, f"{title[:1]} {path[-1]} {roles}/{e}", TP, "cells", f"{roles}/{e}", *path) for e in ("on", "off")]
            col = C["emb"] if roles == "embodied" else C["gray"]
            if mark == "bar":
                ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=f"{roles} roles")
            else:
                ax.plot(x + (j - 0.5) * w, vals, "o", ms=9, color=col, mec="k", mew=0.6, label=f"{roles} roles", zorder=3)
            for xi, v, e in zip(x, vals, ("on", "off")):
                txt = fmt.format(v)
                if path[-1] == "fire_rate":
                    txt += f"\n{n_fired[(roles, e)]} fired"
                ax.text(xi + (j - 0.5) * w, max(v, ylim[0]) + (ylim[1] - ylim[0]) * (0.015 if mark == "bar" else 0.06),
                        txt, ha="center", va="bottom", fontsize=7.2)
        ax.set_xticks(x); ax.set_xticklabels(["edges on", "edges off"]); ax.set_ylim(*ylim); ax.set_xlim(-0.5, 1.5)
        if mark == "dot":
            ax.text(0.02, 0.98, "zoomed axis", transform=ax.transAxes, fontsize=7, color="#666", va="top")
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.text(0.01, 0.01, "grok, " + f"{n_items} items; debates per cell: " +
             ", ".join(f"{r} / edges {e} {n_deb[(r, e)]:,}" for r, e in CELLS), fontsize=7.2, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); _save(fig, "fig6.png", save)


# ---------------------------------------------------------------- driver
def build_all(save=True):
    REGISTRY.clear(); CROSS.clear()
    for fn in (fig1, fig3, fig4, fig5, fig6):
        fn(save=save)


def verify():
    """Print every registered value with its artefact key, re-read fresh, count mismatches."""
    bad = 0
    print(f"{'figure':6s} {'plotted value':44s} {'artefact key':78s} value")
    for r in REGISTRY:
        if r["path"] is not None:
            fresh = _walk(_load(r["file"], fresh=True), r["path"])
            key = _key(r["file"], r["path"])
        else:
            fresh = r["fn"](*[_walk(_load(r["file"], fresh=True), p) for p in r["paths"]])
            key = f"{os.path.basename(r['file'])}:{r['formula']}"
        ok = abs(float(fresh) - float(r["value"])) <= TOL
        bad += not ok
        print(f"{r['fig']:6s} {r['label'][:44]:44s} {key[:78]:78s} {r['value']!r}{'' if ok else '  MISMATCH ' + repr(fresh)}")
    print(f"\n{len(REGISTRY)} plotted values checked against fresh artefact reads, tolerance {TOL}")
    for c in CROSS:
        a = _walk(_load(c["a"][0], fresh=True), c["a"][1]); b = _walk(_load(c["b"][0], fresh=True), c["b"][1])
        ok = abs(float(a) - float(b)) <= TOL
        bad += not ok
        print(f"cross  {c['label'][:70]:70s} {_key(*c['a'])[:60]:60s} == {_key(*c['b'])[:60]:60s} "
              f"{'ok' if ok else 'MISMATCH ' + repr((a, b))}")
    print(f"{len(CROSS)} cross-artefact equalities checked")
    return bad


def _warn_stale():
    stale = sorted(set(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "fig*.png"))) - set(PRODUCED))
    if stale:
        print(f"WARNING stale PNGs in {OUT} not produced by this script: {stale}")


def check():
    build_all(save=False)
    bad = verify()
    _warn_stale()
    assert bad == 0, f"{bad} value(s) do not match their artefact"
    print("CHECK OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="rebuild without saving, print every plotted value with its artefact key, assert equality to 1e-6")
    args = ap.parse_args(argv)
    if args.check:
        check()
        return
    os.makedirs(OUT, exist_ok=True)
    build_all(save=True)
    print("figures written: " + ", ".join(PRODUCED))
    _warn_stale()


if __name__ == "__main__":
    main()
