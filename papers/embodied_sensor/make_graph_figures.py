"""Graph figures for the embodied-sensor paper, drawn from the emergent-graph
artefact. Every number on every figure is read from a JSON artefact under
divergence_study_outputs/ (chiefly emergent_graphs.json, the output of
scripts/analyze_emergent_graphs.py, with effective_voter.json,
flooding_analysis.json, the games artefacts and the Dilemmas actuator artefact
for g5, g5n and g6) through one accessor that records the file and key path of
everything drawn, and `--check` rebuilds each figure without saving and asserts
that every registered value equals a fresh read of its artefact to 1e-6 (674
values, 25 cross-artefact equalities and one prereg record quote on 2026-09-13).
Nothing numeric is hard-coded except layout, the reference constants 0 and 1,
and the two record quotes described below.

Figures, written to papers/embodied_sensor/figures/

  g1_protocol.png    the deployed protocol as a graph. Three seat nodes, a
                     moderator node and a judge node. Directed edges for the r1
                     and r2 reads among the seats, the seat to moderator
                     synthesis at S1, the moderator to seat labels at r3, the
                     seat to moderator votes at r4, the flag as the three-seat
                     r3 objection set the stake-blind counter reads (drawn as
                     the three undirected objection edges W C, W N, C N, the
                     advocate pair thick because its against-interest crossings
                     carry the signal) and the routing edge from the moderator
                     to the judge. Annotated with the raw fire rate, the split
                     of the one-loser firings between both-advocate and
                     neutral-plus-loser objections, the within-one-loser error
                     ratio and the routing gain.
  g2_cells.png       the four cells of the 16.10 roles x edges factorial as four
                     small seat graphs. Node area follows the seat's objection
                     rate, edge width follows |phi|, colour follows the sign.
                     The identical cells are nearly edgeless.
  g3_by_verdict.png  the grok objection graph within one-loser and within
                     both-party verdicts, with the who-loses table beside it.
                     The anti-correlated advocate edge is present only under
                     one-loser verdicts.
  g4_instruments.png the same three-node graph on grok AITA, the Scruples
                     Dilemmas, the binding games (B1) and the non-binding games
                     (C1), showing where the graph goes silent.
  g5_models.png      extra, not requested. The same graph on grok, nano and
                     haiku, the model boundary of the edge. Role-lock in the
                     subtitles reads effective_voter.json panels.*.role_lock.
  g5_n_perspectives.png
                     the k-loser generalisation of section 4 of
                     graph_theory_design.md as a schematic star of n advocates
                     over the moderator. Three panels, k = 1 (the sensor
                     regime), k = 0 (silence) and k = n (flooding). The k
                     advocates the verdict goes against are shaded, their
                     with-interest objection edges are thick grey, and the
                     against-interest objection edges from the n - k favoured
                     advocates are highlighted red. Each panel carries the
                     measured n = 2 values from the artefacts (grok AITA
                     one-loser and both-party strata, games B1, Dilemmas).
  g6_taxonomy.png    the game-theoretic taxonomy of section 5 of
                     graph_theory_design.md as a grid. One row per interest
                     structure, a small graph glyph, which edge exists, what
                     the flag does, what the record measured (every number
                     read from an artefact) and the status of the experiment.
                     Nine rows since 2026-09-13, when the one-shot prisoner's
                     dilemma (C1) was moved out of the flooding row, which the
                     record contradicts (the sensor is SILENT there, prereg
                     L5069, and the counter fires 0 of 48), into its own
                     silence row.

Record quotes. Two numbers on g6 have no artefact field: the 16.11a routing
gain over grok solo (carried in emergent_graphs.json protocol.annotations as a
flagged record quote, drawn as such) and the interval of the B2 planner minus
solo contrast on the cannot-assist games (prereg L5039). The B2 point is
recomputed from game_B2_analysis.json and game_solo_analysis.json and `--check`
asserts it rounds to the quoted point and that the quoted prereg line still
carries the quoted interval. Both are labelled as record quotes on the figure.

Drawn with matplotlib only. networkx is installed in this environment but is
not imported, so the script has no dependency on it.

Zero model or API calls. Run from anywhere
    .venv/bin/python papers/embodied_sensor/make_graph_figures.py          writes the PNGs
    .venv/bin/python papers/embodied_sensor/make_graph_figures.py --check  verifies every value
"""
import argparse
import json
import math
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DSO = os.path.join(ROOT, "divergence_study_outputs")
OUT = os.path.join(HERE, "figures")
PREREG = os.path.join(ROOT, "Guidance_Documents", "prereg_embodiment_community.md")
EG = "emergent_graphs.json"
RD = "router_decomposition.json"
EV = "effective_voter.json"
FL = "flooding_analysis.json"
GB1 = "game_B1_analysis.json"
GC1 = "game_C1_analysis.json"
GCO = "game_coordination_analysis.json"
GB2 = "game_B2_analysis.json"
GSO = "game_solo_analysis.json"
DA = "dilemma_stage2_actuator_analysis.json"
TOL = 1e-6
PRODUCED = ["g1_protocol.png", "g2_cells.png", "g3_by_verdict.png", "g4_instruments.png", "g5_models.png",
            "g5_n_perspectives.png", "g6_taxonomy.png"]

# The only numbers on any figure that no artefact holds. Each is a verbatim
# prereg quote with its line, drawn with a "record" label, and `--check` reads
# the prereg (never writes it) to assert the quoted line still carries the text.
RECORD_QUOTES = {
    "b2_planner_minus_solo_cannot_assist": {
        "point": 0.151, "lo": 0.043, "hi": 0.280, "n_games": 31,
        "line": 5039, "text": "cannot-assist +0.151 [+0.043, +0.280]",
        "source": "16.13 B2 extent map RESULTS, planner - solo, cannot-assist region, 31 games, 3,000 draws, seed 7",
    },
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"neg": "#1f4e79", "pos": "#b35806", "gray": "#8a8a8a", "node": "#dfe8f1", "nodeedge": "#1f4e79",
     "mod": "#f2e3c9", "judge": "#e4d5ee", "flag": "#c0392b", "text": "#222222"}

# ---------------------------------------------------------------- artefact access
_cache = {}
REGISTRY = []      # every plotted value: fig, label, file, path, value
CROSS = []         # cross-artefact equalities asserted under --check


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


def V(fig, label, *path, fname=EG, allow_none=False):
    """Read one stored value from the artefact and register it. None is
    registered as None (an undefined phi) only when allow_none is set."""
    val = _walk(_load(fname), path)
    if val is None and allow_none:
        REGISTRY.append({"fig": fig, "label": label, "file": fname, "path": path, "value": None})
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"{_key(fname, path)} is not a number: {val!r}")
    REGISTRY.append({"fig": fig, "label": label, "file": fname, "path": path, "value": val})
    return val


def X(label, a, b):
    CROSS.append({"label": label, "a": a, "b": b})


def CI(fig, label, *path, fname=EG):
    """Point, lo, hi of a {point, lo, hi} block, each registered; None allowed."""
    return tuple(V(fig, f"{label} {k}", *path, k, fname=fname, allow_none=True) for k in ("point", "lo", "hi"))


# ---------------------------------------------------------------- formatting
def _f(x, fmt=".3f"):
    if x is None:
        return "n/a"
    s = format(x, fmt)
    return s.replace("-", "−")


def _ci_txt(p, lo, hi, fmt=".3f"):
    if p is None:
        return "undefined"
    if lo is None:
        return _f(p, fmt)
    return f"{_f(p, fmt)} [{_f(lo, fmt)}, {_f(hi, fmt)}]"


SEAT_SHORT = {
    "writer_advocate": "writer's\nadvocate", "counterparty": "counter-\nparty", "neutral_adjudicator": "neutral",
    "writer_advocate_noedge": "writer's\nadvocate", "counterparty_noedge": "counter-\nparty",
    "neutral_adjudicator_noedge": "neutral",
    "reader_alpha": "reader\nalpha", "reader_beta": "reader\nbeta", "reader_gamma": "reader\ngamma",
    "reader_alpha_noedge": "reader\nalpha", "reader_beta_noedge": "reader\nbeta", "reader_gamma_noedge": "reader\ngamma",
    "person_a_advocate": "A's\nadvocate", "person_b_advocate": "B's\nadvocate",
    "player_r_advocate": "row\nadvocate", "player_c_advocate": "column\nadvocate", "neutral_reader": "neutral\nreader",
    "plan_mediator": "plan\nmediator",
    "player_r_advocate_nb": "row\nadvocate", "player_c_advocate_nb": "column\nadvocate",
    "neutral_reader_nb": "neutral\nreader", "plan_mediator_nb": "plan\nmediator",
}
POS = {"advocate_1": (0.0, 0.0), "advocate_2": (2.0, 0.0), "third": (1.0, 1.55)}
PAIRS = ("advocate_1|advocate_2", "advocate_1|third", "advocate_2|third")
SEAT_XLIM = (-1.05, 3.05)
SEAT_YLIM = (-0.95, 2.45)


# ---------------------------------------------------------------- the seat-graph primitive
def _node_radius(rate):
    return 0.24 + 0.28 * (rate if rate is not None else 0.0)


def draw_seat_graph(ax, fig_id, com_key, block_path, *, title, subtitle=None, foot=None,
                    label_fmt=".3f", edge_fmt="+.3f", show_ci=True, xlim=SEAT_XLIM, ylim=SEAT_YLIM):
    """Draw one three-seat objection graph from the artefact block at
    `block_path` (a graphs.<stratum> block). Registers every number drawn.
    Every label sits inside the axes limits so that no layout pass has to
    shrink the axes to fit it."""
    eg = _load(EG)
    com = eg["communities"][com_key]
    nodes, edges = {}, {}
    for p in ("advocate_1", "advocate_2", "third"):
        pt, lo, hi = CI(fig_id, f"{com_key} {block_path[-1]} node {p}", *block_path, "nodes", p, "objection_rate")
        nodes[p] = {"seat": com["seats"][p], "rate": pt, "lo": lo, "hi": hi, "r": _node_radius(pt)}
    for e in PAIRS:
        pt, lo, hi = CI(fig_id, f"{com_key} {block_path[-1]} edge {e}", *block_path, "edges", e, "phi")
        edges[e] = {"phi": pt, "lo": lo, "hi": hi}
    n = V(fig_id, f"{com_key} {block_path[-1]} n", *block_path, "n")
    fire, flo, fhi = CI(fig_id, f"{com_key} {block_path[-1]} fire", *block_path, "fire_rate")
    lines = []
    if subtitle:
        lines += subtitle.split("\n")
    lines.append(f"n = {n:,} debates, fire P(≥2 object) {_ci_txt(fire, flo, fhi)}")
    if foot:
        lines += foot.split("\n")
    ylim = (ylim[0], ylim[1] + 0.21 * max(0, len(lines) - 2))

    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_anchor("N"); ax.axis("off")
    bb = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95)
    # edges first
    for e, ed in edges.items():
        a, b = e.split("|")
        (x1, y1), (x2, y2) = POS[a], POS[b]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
        ra, rb = nodes[a]["r"], nodes[b]["r"]
        sx, sy, ex, ey = x1 + ux * ra, y1 + uy * ra, x2 - ux * rb, y2 - uy * rb
        phi = ed["phi"]
        if phi is None:
            ax.plot([sx, ex], [sy, ey], color=C["gray"], lw=0.9, ls=(0, (1.5, 2.5)), zorder=1)
            col = C["gray"]
        else:
            col = C["neg"] if phi < 0 else C["pos"]
            ax.plot([sx, ex], [sy, ey], color=col, lw=0.8 + 7.0 * abs(phi), alpha=0.9, zorder=1,
                    solid_capstyle="round")
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        if phi is None:
            head, tail = "φ undefined", "(no objection)"
        elif show_ci:
            head, tail = f"φ {_f(phi, edge_fmt)}", f"[{_f(ed['lo'], edge_fmt)}, {_f(ed['hi'], edge_fmt)}]"
        else:
            head, tail = f"φ {_f(phi, edge_fmt)}", ""
        if e == "advocate_1|advocate_2":
            ax.text(mx, my + 0.16, (head + "  " + tail).strip(), ha="center", va="bottom", fontsize=7.2, color=col,
                    bbox=bb, zorder=3)
        elif e == "advocate_1|third":
            ax.text(mx - 0.22, my + 0.10, head + "\n" + tail, ha="right", va="center", fontsize=7.2, color=col,
                    bbox=bb, zorder=3, linespacing=1.15)
        else:
            ax.text(mx + 0.22, my + 0.10, head + "\n" + tail, ha="left", va="center", fontsize=7.2, color=col,
                    bbox=bb, zorder=3, linespacing=1.15)
    # nodes
    for p, nd in nodes.items():
        x, y = POS[p]
        ax.add_patch(Circle((x, y), nd["r"], fc=C["node"], ec=C["nodeedge"], lw=1.0, zorder=2))
        ax.text(x, y, SEAT_SHORT.get(nd["seat"], nd["seat"]), ha="center", va="center",
                fontsize=(7.0 if nd["r"] >= 0.30 else 6.2), zorder=4, color=C["text"], linespacing=1.0)
        if show_ci and nd["lo"] is not None:
            rate_txt = f"objects {_f(nd['rate'], label_fmt)}\n[{_f(nd['lo'], label_fmt)}, {_f(nd['hi'], label_fmt)}]"
        else:
            rate_txt = f"objects {_f(nd['rate'], label_fmt)}"
        if p == "third":
            ax.text(x, y + nd["r"] + 0.08, rate_txt, ha="center", va="bottom", fontsize=6.9, color=C["text"],
                    zorder=4, linespacing=1.15)
        else:
            ax.text(x, y - nd["r"] - 0.08, rate_txt, ha="center", va="top", fontsize=6.9, color=C["text"],
                    zorder=4, linespacing=1.15)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=2)
    ax.text(0.0, 0.995, "\n".join(lines), transform=ax.transAxes, fontsize=7.0, va="top", ha="left", color="#444",
            linespacing=1.25)
    return nodes, edges


LEGEND = ("node area follows the seat's r3 objection rate. edge width follows |φ| between the two seats' objection "
          "indicators, blue negative (anti-correlated), orange positive, dotted grey undefined. brackets are "
          "item-clustered 95% bootstrap intervals")


def _legend_text(fig, y=0.012, x=0.01):
    fig.text(x, y, LEGEND, fontsize=6.9, color="#555", va="bottom", ha="left")


# ---------------------------------------------------------------- Fig g1, the protocol
def g1(save=True):
    F = "g1"
    eg = _load(EG)
    fire = V(F, "fire rate raw (grok)", "protocol", "annotations", "fire_rate_raw")
    ratio = V(F, "one-loser ratio", "protocol", "annotations", "one_loser_ratio")
    lift, llo, lhi = CI(F, "one-loser lift", "protocol", "annotations", "one_loser_lift")
    phi_ol, plo, phi_hi = CI(F, "phi W,C one-loser", "communities", "grok_aita", "populations", "s2_codable",
                             "graphs", "one_loser", "edges", "advocate_1|advocate_2", "phi")
    phi_bp, blo, bhi = CI(F, "phi W,C both-party", "communities", "grok_aita", "populations", "s2_codable",
                          "graphs", "both_party", "edges", "advocate_1|advocate_2", "phi")
    g_h = V(F, "routing gain haiku judge", "protocol", "annotations", "routing_gain", "haiku_judge", "gain", "delta")
    g_h_lo = V(F, "routing gain haiku lo", "protocol", "annotations", "routing_gain", "haiku_judge", "gain", "lo")
    g_h_hi = V(F, "routing gain haiku hi", "protocol", "annotations", "routing_gain", "haiku_judge", "gain", "hi")
    g_s = V(F, "routing gain sonnet judge", "protocol", "annotations", "routing_gain", "sonnet_judge", "gain", "delta")
    g_s_lo = V(F, "routing gain sonnet lo", "protocol", "annotations", "routing_gain", "sonnet_judge", "gain", "lo")
    g_s_hi = V(F, "routing gain sonnet hi", "protocol", "annotations", "routing_gain", "sonnet_judge", "gain", "hi")
    routed = V(F, "debates routed", "protocol", "annotations", "routing_gain", "haiku_judge", "routed")
    n_deb = V(F, "n debates raw", "communities", "grok_aita", "raw", "n_debates")
    n_sens = V(F, "n sensor population", "protocol", "annotations", "n_sensor_population")
    q = ("protocol", "annotations", "routing_gain_over_solo_16_11a")
    q_d = V(F, "16.11a routed minus solo (record quote)", *q, "routed_embodied", "delta_vs_solo")
    q_lo = V(F, "16.11a routed minus solo lo (record quote)", *q, "routed_embodied", "lo")
    q_hi = V(F, "16.11a routed minus solo hi (record quote)", *q, "routed_embodied", "hi")
    q_cov = V(F, "16.11a coverage (record quote)", *q, "routed_embodied", "coverage")
    q_dom = V(F, "16.11a routed minus sonnet everywhere (record quote)", *q, "sonnet_everywhere",
              "routed_embodied_minus_this")
    assert _walk(_load(EG), q)["kind"].startswith("record quote"), "16.11a block must be flagged as a record quote"
    sp = ("protocol", "annotations", "one_loser_firings_split")
    sp_fired = V(F, "one-loser firings", *sp, "n_fired")
    sp_both = V(F, "one-loser firings with both advocates", *sp, "both_advocates_object")
    sp_neu = V(F, "one-loser firings with the neutral seat", *sp, "neutral_with_one_advocate")
    sp_n_obj = V(F, "neutral objected within one-loser", *sp, "neutral_objected")
    sp_n = V(F, "one-loser n", *sp, "n")
    ol = ("communities", "grok_aita", "populations", "s2_codable", "graphs", "one_loser")
    X("firings split n_fired == one-loser graph n_fired", (EG, (*sp, "n_fired")), (EG, (*ol, "fire_rate", "n_fired")))
    X("firings split both == one-loser advocate edge both", (EG, (*sp, "both_advocates_object")),
      (EG, (*ol, "edges", "advocate_1|advocate_2", "table", "both")))
    X("firings split neutral objected == one-loser third n_objected", (EG, (*sp, "neutral_objected")),
      (EG, (*ol, "nodes", "third", "n_objected")))
    assert sp_fired == sp_both + sp_neu, "one-loser firings split does not add up"
    flag_edge = [e for e in eg["protocol"]["edges"] if e["kind"] == "flag"]
    assert len(flag_edge) == 1 and flag_edge[0].get("seats") == ["W", "C", "N"], "flag must be the three-seat hyperedge"
    w_rate = V(F, "W raw objection rate", "communities", "grok_aita", "raw", "objection_rate", "advocate_1", "objection_rate")
    c_rate = V(F, "C raw objection rate", "communities", "grok_aita", "raw", "objection_rate", "advocate_2", "objection_rate")
    n_rate = V(F, "N raw objection rate", "communities", "grok_aita", "raw", "objection_rate", "third", "objection_rate")
    # the artefact's copies of the routing gain must equal router_decomposition.json
    X("routing gain haiku == router_decomposition", (EG, ("protocol", "annotations", "routing_gain", "haiku_judge", "gain", "delta")),
      (RD, ("composed_haiku", "paired_deltas", "counter_minus_collective_alone", "delta")))
    X("routing gain sonnet == router_decomposition", (EG, ("protocol", "annotations", "routing_gain", "sonnet_judge", "gain", "delta")),
      (RD, ("composed_sonnet", "block", "paired_deltas", "counter_minus_collective_alone", "delta")))
    X("one-loser ratio == router_decomposition", (EG, ("protocol", "annotations", "one_loser_ratio")),
      (RD, ("counter_lift_by_stratum", "one_loser", "ratio")))
    X("fire rate annotation == grok raw fire rate", (EG, ("protocol", "annotations", "fire_rate_raw")),
      (EG, ("communities", "grok_aita", "raw", "fire_rate")))
    X("sensor population n == router_decomposition pooled n", (EG, ("protocol", "annotations", "n_sensor_population")),
      (RD, ("counter_lift_by_stratum", "pooled", "n")))
    X("debates routed == sensor pooled n_fired", (EG, ("protocol", "annotations", "routing_gain", "haiku_judge", "routed")),
      (EG, ("communities", "grok_aita", "populations", "s1_s2_codable", "sensor", "strata", "all", "n_fired")))
    n_edges = len(eg["protocol"]["edges"])
    n_read = sum(1 for e in eg["protocol"]["edges"] if e["kind"] == "read")
    assert n_edges == 17 and n_read == 6, "protocol block edge count changed"

    fig = plt.figure(figsize=(12.0, 8.7))
    ax = fig.add_axes((0.02, 0.235, 0.96, 0.735))
    ax.set_xlim(-1.1, 7.3); ax.set_ylim(-2.9, 2.8); ax.set_aspect("equal"); ax.axis("off")
    P = {"W": (0.0, 0.0), "C": (3.4, 0.0), "N": (1.7, 2.1), "M": (1.7, -2.2), "J": (6.3, -2.2)}
    R = {"W": 0.42, "C": 0.42, "N": 0.42, "M": 0.46, "J": 0.46}
    COL = {"read": "#6b7f95", "syn": "#2e6b3e", "lab": "#7a4f9a", "vote": "#8c5a1e"}

    def arrow(a, b, *, rad=0.0, color="#333", lw=1.2, ls="-", zorder=2, head=11, gap=0.05):
        """Directed edge from the boundary of a to the boundary of b, so the
        head is drawn outside the target circle and stays visible."""
        (x1, y1), (x2, y2) = P[a], P[b]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
        s = (x1 + ux * (R[a] + gap), y1 + uy * (R[a] + gap))
        e = (x2 - ux * (R[b] + gap), y2 - uy * (R[b] + gap))
        ax.add_patch(FancyArrowPatch(s, e, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=head, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                     zorder=zorder))

    # r1, r2 reads among the seats, both directions on gentle arcs
    for a, b in (("W", "C"), ("W", "N"), ("C", "N")):
        arrow(a, b, rad=0.25, color=COL["read"], lw=1.0)
        arrow(b, a, rad=0.25, color=COL["read"], lw=1.0)
    # seat -> M synthesis (solid green, straight); M -> seat r3 label (dotted purple, one side);
    # seat -> M r4 vote (dashed brown, other side)
    for s in ("W", "C", "N"):
        arrow(s, "M", rad=0.0, color=COL["syn"], lw=1.4)
        arrow("M", s, rad=0.33, color=COL["lab"], lw=1.2, ls=(0, (1.5, 2.0)))
        arrow(s, "M", rad=0.33, color=COL["vote"], lw=1.2, ls=(0, (5, 2.5)))
    # the flag, a three-seat hyperedge over the r3 objections of W, C and N, drawn as the three
    # undirected objection edges the stake-blind counter reads. the advocate pair is thick because its
    # against-interest crossings carry the signal; the two neutral edges are thinner but are inside the
    # counter (44 of the 93 one-loser firings are the neutral seat with the losing advocate)
    ax.plot([P["W"][0] + R["W"], P["C"][0] - R["C"]], [0, 0], color=C["flag"], lw=4.5, alpha=0.95, zorder=1,
            solid_capstyle="round")
    for a in ("W", "C"):
        (x1, y1), (x2, y2) = P[a], P["N"]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
        ax.plot([x1 + ux * R[a], x2 - ux * R["N"]], [y1 + uy * R[a], y2 - uy * R["N"]], color=C["flag"], lw=2.0,
                alpha=0.9, zorder=1, solid_capstyle="round")
    # the routing edge M -> J
    arrow("M", "J", rad=0.0, color=C["flag"], lw=3.0, head=16)

    # nodes
    for k, (x, y) in P.items():
        fc = C["mod"] if k == "M" else (C["judge"] if k == "J" else C["node"])
        ax.add_patch(Circle((x, y), R[k], fc=fc, ec="#333", lw=1.1, zorder=5))
    ax.text(*P["W"], "W\nwriter's\nadvocate", ha="center", va="center", fontsize=8.2, zorder=6)
    ax.text(*P["C"], "C\ncounter-\nparty", ha="center", va="center", fontsize=8.2, zorder=6)
    ax.text(*P["N"], "N\nneutral", ha="center", va="center", fontsize=8.2, zorder=6)
    ax.text(*P["M"], "M\nmoderator", ha="center", va="center", fontsize=8.2, zorder=6)
    ax.text(*P["J"], "J\njudge", ha="center", va="center", fontsize=8.2, zorder=6)
    bb = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=1.0)
    ax.text(P["W"][0] - R["W"] - 0.08, P["W"][1], f"objects\non {_f(w_rate)}", ha="right", va="center", fontsize=7.8,
            color="#333", bbox=bb, zorder=7)
    ax.text(P["C"][0] + R["C"] + 0.08, P["C"][1], f"objects\non {_f(c_rate)}", ha="left", va="center", fontsize=7.8,
            color="#333", bbox=bb, zorder=7)
    ax.text(P["N"][0], P["N"][1] + 0.60, f"objects on {_f(n_rate)}", ha="center", fontsize=7.8, color="#333",
            bbox=bb, zorder=7)

    # stage labels, each placed clear of the arcs
    ax.text(0.05, 1.65, "r1, r2 reads", ha="center", va="center", fontsize=8.0, color=COL["read"], bbox=bb, zorder=7)
    ax.text(-0.85, -1.35, "S1 synthesis\nseat → M", ha="center", va="center", fontsize=7.8, color=COL["syn"], bbox=bb,
            zorder=7)
    ax.text(4.55, -0.75, "r3 label  M → seat\naccept / modify / reject", ha="center", va="center", fontsize=7.8,
            color=COL["lab"], bbox=bb, zorder=7)
    ax.text(4.25, -1.55, "r4 vote  seat → M", ha="center", va="center", fontsize=7.8, color=COL["vote"], bbox=bb,
            zorder=7)
    ax.text(1.7, -0.52, "the flag: two or more of the three seats object at r3 (stake-blind counter)", ha="center",
            va="center", fontsize=8.2, color=C["flag"], fontweight="bold", bbox=bb, zorder=7)
    ax.text(1.15, -0.86, f"{sp_fired} one-loser firings: {sp_both} both advocates, {sp_neu} N with the losing advocate",
            ha="center", va="center", fontsize=7.4, color=C["flag"], bbox=bb, zorder=7)
    ax.text(1.7, 0.50, f"φ(W, C) = {_ci_txt(phi_ol, plo, phi_hi)} within one-loser verdicts\n"
                       f"φ(W, C) = {_ci_txt(phi_bp, blo, bhi)} within both-party verdicts",
            ha="center", va="center", fontsize=7.8, color=C["flag"], bbox=bb, zorder=7)
    ax.text(4.0, -2.62, "route to the judge when two or more seats object", ha="center", va="center", fontsize=8.0,
            color=C["flag"], fontweight="bold", bbox=bb, zorder=7)
    ax.set_title(f"The deployed protocol as a graph (grok AITA, {n_deb:,} debates, seven stages, 17 calls per debate)",
                 loc="left", fontsize=11, fontweight="bold")

    # annotation band (numbers) and legend band, below the graph and clear of each other
    fig.text(0.02, 0.205,
             f"fire rate {_f(fire)} ({routed:,} of {n_sens:,} codable debates routed). within one-loser verdicts the counter "
             f"fired {sp_fired} times, {sp_both} with both advocates objecting and {sp_neu} with the neutral seat objecting "
             f"beside the losing advocate (N objected on {sp_n_obj} of {sp_n:,})\n"
             f"within one-loser verdicts the flagged debate is wrong {_f(ratio, '.2f')}x as often as the unflagged "
             f"(lift {_ci_txt(lift, llo, lhi, '+.3f')})\n"
             f"composed gain over the collective's own verdict {_ci_txt(g_h, g_h_lo, g_h_hi, '+.3f')} with the haiku judge, "
             f"{_ci_txt(g_s, g_s_lo, g_s_hi, '+.3f')} with the sonnet judge (router_decomposition.json)\n"
             f"16.11a per-cell record quote, routed minus grok solo {_ci_txt(q_d, q_lo, q_hi, '+.3f')} at coverage "
             f"{_f(q_cov)}. Sonnet everywhere beats routing by {_f(-q_dom)}, so criterion 2 is not met",
             fontsize=8.2, color=C["flag"], va="top", ha="left", linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff5f5", ec=C["flag"], lw=0.7))
    fig.text(0.02, 0.075,
             "solid green is the S1 synthesis from the seats' r2 positions.   dotted purple, the r3 labels on that "
             "synthesis.   dashed brown, the r4 votes on the integrated S2 verdict.\n"
             "grey arcs are the r1, r2 reads, the six edges cut in the edges-off cell.   red marks the flag (the three "
             "r3 objection edges the counter reads, the advocate pair thick) and the routing edge, the only edges the "
             "sensor uses.   objection rates are the raw 16.10 seat marginals.",
             fontsize=7.6, color="#555", va="top", ha="left", linespacing=1.4)
    _save(fig, "g1_protocol.png", save)


# ---------------------------------------------------------------- Fig g2, the 2x2 cells
def g2(save=True):
    F = "g2"
    cells = [("grok_aita", "embodied roles, edges on"), ("grok_aita_noedge", "embodied roles, edges off"),
             ("grok_aita_identical", "identical readers, edges on"),
             ("grok_aita_identical_noedge", "identical readers, edges off")]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8))
    n_items = V(F, "grok items", "communities", "grok_aita", "raw", "n_items")
    for key, _ in cells[1:]:
        X(f"{key} items == grok items", (EG, ("communities", key, "raw", "n_items")),
          (EG, ("communities", "grok_aita", "raw", "n_items")))
    for ax, (key, title) in zip(axes.ravel(), cells):
        raw_fire = V(F, f"{key} raw fire", "communities", key, "raw", "fire_rate")
        n_raw = V(F, f"{key} raw n", "communities", key, "raw", "n_debates")
        draw_seat_graph(ax, F, key, ("communities", key, "populations", "s2_codable", "graphs", "all"),
                        title=title,
                        subtitle=f"all debates, fire P(≥2) {_f(raw_fire)} on {n_raw:,} debates (16.10)")
    # the identical cells' small phis are the floor for rare, near-exclusive objections, not a structured edge
    notes = []
    for key, name in (("grok_aita_identical", "identical, edges on"), ("grok_aita_identical_noedge", "identical, edges off")):
        b = ("communities", key, "populations", "s2_codable", "graphs", "all")
        n_a = V(F, f"{key} alpha objected", *b, "nodes", "advocate_1", "n_objected")
        n_b = V(F, f"{key} beta objected", *b, "nodes", "advocate_2", "n_objected")
        n_g = V(F, f"{key} gamma objected", *b, "nodes", "third", "n_objected")
        n = V(F, f"{key} n", *b, "n")
        both = [V(F, f"{key} both {e}", *b, "edges", e, "table", "both") for e in PAIRS]
        notes.append(f"{name}, the readers objected on {n_a}, {n_b} and {n_g} of {n} debates and coincided on "
                     f"{both[0]}, {both[1]} and {both[2]} pairs")
    note = ("the identical cells' small φ values are the floor for rare, near-exclusive objections, not a structured "
            "edge. " + ". ".join(notes))
    fig.text(0.01, 0.038, "\n".join(textwrap.wrap(note, width=190, break_long_words=False, break_on_hyphens=False)),
             fontsize=6.9, color="#555", va="bottom", linespacing=1.35)
    fig.suptitle(f"The objection graph in the four cells of the roles x edges factorial (grok, {n_items} items)",
                 x=0.01, y=0.985, ha="left", fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.06, wspace=0.08, hspace=0.18)
    _legend_text(fig)
    _save(fig, "g2_cells.png", save)


# ---------------------------------------------------------------- Fig g3, by verdict type
def g3(save=True):
    F = "g3"
    key = "grok_aita"
    base = ("communities", key, "populations")
    n_s2 = V(F, "S2-codable n", *base, "s2_codable", "n")
    phi_ol = V(F, "phi W,C one-loser point", *base, "s2_codable", "graphs", "one_loser", "edges", "advocate_1|advocate_2", "phi", "point")
    phi_bp = V(F, "phi W,C both-party point", *base, "s2_codable", "graphs", "both_party", "edges", "advocate_1|advocate_2", "phi", "point")
    fig = plt.figure(figsize=(13.4, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], left=0.015, right=0.985, top=0.88, bottom=0.14, wspace=0.12)
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for ax, stratum, title in ((ax1, "one_loser", "within one-loser verdicts (YTA or NTA)"),
                               (ax2, "both_party", "within both-party verdicts (ESH or NAH)")):
        ratio = V(F, f"{stratum} ratio", *base, "s1_s2_codable", "sensor", "strata", stratum, "ratio")
        pwf = V(F, f"{stratum} p wrong fired", *base, "s1_s2_codable", "sensor", "strata", stratum, "p_wrong_fired")
        pwu = V(F, f"{stratum} p wrong unfired", *base, "s1_s2_codable", "sensor", "strata", stratum, "p_wrong_unfired")
        lift, lo, hi = CI(F, f"{stratum} lift", *base, "s1_s2_codable", "sensor", "strata", stratum, "lift")
        draw_seat_graph(ax, F, key, (*base, "s2_codable", "graphs", stratum), title=title,
                        foot=f"P(wrong | fired) {_f(pwf)} vs {_f(pwu)} unfired, ratio {_f(ratio, '.2f')}x\n"
                             f"lift {_ci_txt(lift, lo, hi, '+.3f')} (16.15 population)")
    # who loses: objection rate by seat for each verdict type
    types = ["YTA", "NTA", "ESH", "NAH"]
    seats = ["advocate_1", "advocate_2", "third"]
    labels = ["writer's advocate", "counterparty", "neutral"]
    cols = [C["neg"], C["pos"], C["gray"]]
    w = 0.26
    ticks = []
    for j, t in enumerate(types):
        n = V(F, f"who-loses {t} n", *base, "s2_codable", "who_loses", t, "n")
        ticks.append(f"n = {n:,}")
        for i, p in enumerate(seats):
            pt, lo, hi = CI(F, f"who-loses {t} {p}", *base, "s2_codable", "who_loses", t, "objection_rate", p)
            x = j + (i - 1) * w
            ax3.bar(x, pt, w, color=cols[i], alpha=0.9, label=labels[i] if j == 0 else None)
            ax3.plot([x, x], [lo, hi], color="k", lw=0.9)
            ax3.text(x, hi + 0.02, _f(pt), ha="center", va="bottom", fontsize=6.6)
    names = ["YTA\nwriter at fault", "NTA\nwriter not at fault", "ESH\neveryone at fault", "NAH\nno one at fault"]
    ax3.set_xticks(range(4)); ax3.set_xticklabels([f"{a}\n{b}" for a, b in zip(names, ticks)], fontsize=7.6,
                                                   linespacing=1.3)
    ax3.set_ylim(0, 1.18); ax3.set_ylabel("P(seat objects at r3)")
    ax3.set_title(f"who loses, who objects (S2-codable, {n_s2:,} debates)", loc="left", fontsize=9.5, fontweight="bold",
                  pad=2)
    ax3.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    ax3.spines["left"].set_position(("outward", 4))
    ax3.tick_params(axis="x", length=0, pad=6)
    fig.suptitle(f"The grok objection graph by verdict type. The advocate edge is φ {_f(phi_ol)} where exactly one "
                 f"seat loses and collapses to {_f(phi_bp)} where both do", x=0.01, y=0.975, ha="left", fontsize=10.5,
                 fontweight="bold")
    _legend_text(fig)
    _save(fig, "g3_by_verdict.png", save)


# ---------------------------------------------------------------- Fig g4, across instruments
def _sensor_foot(F, key, stratum="all"):
    base = ("communities", key, "populations", "s1_s2_codable", "sensor", "strata", stratum)
    n_f = V(F, f"{key} sensor fired", *base, "n_fired")
    n = V(F, f"{key} sensor n", *base, "n")
    ratio = V(F, f"{key} sensor ratio", *base, "ratio", allow_none=True)
    lift, lo, hi = CI(F, f"{key} sensor lift", *base, "lift")
    if n_f == 0:
        return f"counter fired on 0 of {n:,} scored debates\nlift undefined (silent)"
    if ratio is None:
        return f"counter fired on {n_f:,} of {n:,} scored debates, ratio undefined\nlift {_ci_txt(lift, lo, hi, '+.3f')}"
    return (f"counter fired on {n_f:,} of {n:,} scored debates, error ratio {_f(ratio, '.2f')}x\n"
            f"lift {_ci_txt(lift, lo, hi, '+.3f')}")


def g4(save=True):
    F = "g4"
    panels = [("grok_aita", "grok AITA (one-loser verdicts exist)", "16.10 embodied roles, edges on, {ni} items"),
              ("grok_dilemmas", "Scruples Dilemmas (one loser by construction)", "Addendum 12 Stage 1, {ni} items"),
              ("grok_games_binding_neutral", "games, binding plan (B1, neutral reader)", "16.13 value region, {ni} games x 3"),
              ("grok_games_nonbinding_neutral", "games, non-binding (C1, neutral reader)", "16.13c value region, {ni} games x 3")]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 5.0))
    for ax, (key, title, sub) in zip(axes, panels):
        sub = sub.format(ni=V(F, f"{key} items", "communities", key, "raw", "n_items"))
        foot = _sensor_foot(F, key)
        draw_seat_graph(ax, F, key, ("communities", key, "populations", "s2_codable", "graphs", "all"),
                        title=title, subtitle=sub, foot=foot)
    # the planner arms of the two games instruments, as a footnote read from the artefact
    notes = []
    for key, name in (("grok_games_binding_planner", "B1 plan-mediator arm"),
                      ("grok_games_nonbinding_planner", "C1 plan-mediator arm")):
        b = ("communities", key, "populations", "s2_codable", "graphs", "all")
        rates = [V(F, f"{key} node {p}", *b, "nodes", p, "objection_rate", "point") for p in ("advocate_1", "advocate_2", "third")]
        fire = V(F, f"{key} fire", *b, "fire_rate", "point")
        phi = V(F, f"{key} phi adv", *b, "edges", "advocate_1|advocate_2", "phi", "point", allow_none=True)
        notes.append(f"{name}, seats object on {_f(rates[0])} / {_f(rates[1])} / {_f(rates[2])}, "
                     f"fire {_f(fire)}, φ(advocates) {_f(phi) if phi is not None else 'undefined'}")
    # the C1 phi floor: the rare objections never coincide
    c1 = ("communities", "grok_games_nonbinding_neutral", "populations", "s2_codable", "graphs", "all")
    both = V(F, "C1 advocates both object", *c1, "edges", "advocate_1|advocate_2", "table", "both")
    n_r = V(F, "C1 row objected", *c1, "nodes", "advocate_1", "n_objected")
    n_c = V(F, "C1 column objected", *c1, "nodes", "advocate_2", "n_objected")
    n_t = V(F, "C1 neutral objected", *c1, "nodes", "third", "n_objected")
    n_c1 = V(F, "C1 n", *c1, "n")
    notes.append(f"C1 neutral-reader arm, the seats objected on {n_r}, {n_c} and {n_t} of {n_c1} debates and the two "
                 f"advocates coincided on {both}, so the small negative φ is the floor for those margins, not a "
                 f"structured edge")
    # the Dilemmas caution and the non-codable firing concentration
    share = V(F, "dilemmas codable share", "communities", "grok_dilemmas", "guards", "checks", "s2_codable_share", "value")
    n_nc = V(F, "dilemmas noncodable n", "communities", "grok_dilemmas", "raw", "n_s2_noncodable")
    f_nc = V(F, "dilemmas fired among noncodable", "communities", "grok_dilemmas", "raw", "n_fired_among_noncodable")
    raw_f = V(F, "dilemmas raw fire", "communities", "grok_dilemmas", "raw", "fire_rate")
    n_raw = V(F, "dilemmas raw n", "communities", "grok_dilemmas", "raw", "n_debates")
    notes.append(f"Dilemmas caution. S2-codable share {_f(share)} (below 0.95, no registered exemption for UNRESOLVED). "
                 f"On all {n_raw:,} debates the counter fired on {_f(raw_f)}, and {f_nc} of those firings sit on the "
                 f"{n_nc} UNRESOLVED or NOVERDICT group verdicts")
    fig.text(0.01, 0.045, "\n".join(notes), fontsize=6.9, color="#555", va="bottom", linespacing=1.35)
    fig.suptitle("The same three-node graph across instruments. The edge needs a verdict that one interested seat "
                 "loses and can deviate from", x=0.01, y=0.975, ha="left", fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.20, wspace=0.08)
    _legend_text(fig)
    _save(fig, "g4_instruments.png", save)


# ---------------------------------------------------------------- Fig g5, across models (extra)
def g5(save=True):
    F = "g5"
    panels = [("grok_aita", "grok", "grok-4-1-fast-reasoning"),
              ("nano_aita", "nano", "gpt-5.4-nano"),
              ("haiku_aita", "haiku", "claude-haiku-4-5")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.4))
    for ax, (key, ev_key, title) in zip(axes, panels):
        nd = V(F, f"{key} debates", "communities", key, "raw", "n_debates")
        ni = V(F, f"{key} items", "communities", key, "raw", "n_items")
        lock, llo, lhi = CI(F, f"{key} role-lock", "panels", ev_key, "role_lock", fname=EV)
        X(f"{key} debates == effective_voter n_debates", (EG, ("communities", key, "raw", "n_debates")),
          (EV, ("panels", ev_key, "n_debates")))
        sub = f"{nd:,} debates, {ni} items, role-lock {_ci_txt(lock, llo, lhi)}"
        foot = _sensor_foot(F, key)
        draw_seat_graph(ax, F, key, ("communities", key, "populations", "s2_codable", "graphs", "all"),
                        title=title, subtitle=sub, foot=foot)
    fig.suptitle("The objection graph on three models (AITA, embodied roles, edges on). The anti-correlated edge "
                 "is grok's, nano floods and haiku's edge is positive", x=0.01, y=0.975, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.text(0.01, 0.045, "role-lock in the subtitles is effective_voter.json panels.*.role_lock (the 16.15.4 "
                          "readout with its interval). every rate, φ, fire rate and lift is emergent_graphs.json",
             fontsize=6.9, color="#555", va="bottom")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.12, wspace=0.08)
    _legend_text(fig)
    _save(fig, "g5_models.png", save)


# ---------------------------------------------------------------- Fig g5n, the k-loser generalisation
K_ADV_X = (0.15, 1.25, 2.35, 3.85)        # A1, A2, A3 and An, with an ellipsis between A3 and An
K_ADV_Y = 2.05
K_M = (2.0, 0.0)
K_N = (4.45, -0.15)
K_R = 0.34
C_LOSER = "#5b6b7c"
C_WITH = "#8593a3"


def _kloser_panel(ax, k, *, title, subtitle):
    """One schematic panel. k is 0, 1 or 'n'. Returns nothing numeric, the
    schematic itself carries no artefact value. Every node is an advocate
    A_i objecting (or not) to the moderator M's verdict v at r3."""
    ax.set_xlim(-0.75, 5.15); ax.set_ylim(-1.15, 3.15); ax.set_aspect("equal"); ax.set_anchor("N"); ax.axis("off")
    names = ("A1", "A2", "A3", "An")
    losers = {0: set(), 1: {0}, "n": {0, 1, 2, 3}}[k]
    bb = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95)

    def arrow(s, e, *, color, lw, ls="-", head=11):
        (x1, y1), (x2, y2) = s, e
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
        a = (x1 + ux * (K_R + 0.04), y1 + uy * (K_R + 0.04))
        b = (x2 - ux * (K_R + 0.10), y2 - uy * (K_R + 0.10))
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw, ls=ls,
                                     shrinkA=0, shrinkB=0, zorder=2))

    for i, x in enumerate(K_ADV_X):
        loser = i in losers
        if k == 0:
            arrow((x, K_ADV_Y), K_M, color=C["gray"], lw=0.9, ls=(0, (1.5, 2.5)), head=8)
        elif loser:
            arrow((x, K_ADV_Y), K_M, color=C_WITH, lw=3.8)
        else:
            arrow((x, K_ADV_Y), K_M, color=C["flag"], lw=2.2)
    arrow(K_N, K_M, color=C["gray"], lw=0.9, ls=(0, (1.5, 2.5)), head=8)
    for i, x in enumerate(K_ADV_X):
        loser = i in losers
        ax.add_patch(Circle((x, K_ADV_Y), K_R, fc=(C_LOSER if loser else C["node"]), ec=C["nodeedge"], lw=1.0,
                            zorder=4))
        ax.text(x, K_ADV_Y, names[i] + ("\nloses" if loser else "\nfavoured"), ha="center", va="center",
                fontsize=7.0, color=("white" if loser else C["text"]), zorder=5, linespacing=1.0)
    ax.text(3.10, K_ADV_Y, "···", ha="center", va="center", fontsize=12, color=C["text"], zorder=5)
    ax.add_patch(Circle(K_M, K_R + 0.04, fc=C["mod"], ec="#333", lw=1.0, zorder=4))
    ax.text(*K_M, "M\nverdict v", ha="center", va="center", fontsize=7.0, zorder=5, linespacing=1.0)
    ax.add_patch(Circle(K_N, K_R - 0.04, fc="white", ec=C["gray"], lw=1.0, ls=(0, (2, 2)), zorder=4))
    ax.text(*K_N, "N\ndisinter-\nested", ha="center", va="center", fontsize=6.0, color="#555", zorder=5,
            linespacing=1.0)
    # edge-class labels, one per class per panel
    if k == 1:
        ax.text(0.05, 1.05, "with interest\nnear-certain,\ncarries nothing", ha="center", va="center", fontsize=6.8,
                color=C_LOSER, bbox=bb, zorder=6, linespacing=1.15)
        ax.text(3.75, 0.95, "against interest\nrare, carries the case", ha="center", va="center", fontsize=6.8,
                color=C["flag"], fontweight="bold", bbox=bb, zorder=6, linespacing=1.15)
    elif k == 0:
        ax.text(0.10, 1.05, "no one loses,\nno objection", ha="center", va="center", fontsize=6.8, color="#555",
                bbox=bb, zorder=6, linespacing=1.15)
    else:
        ax.text(0.05, 1.05, "every objection\nis with interest", ha="center", va="center", fontsize=6.8,
                color=C_LOSER, bbox=bb, zorder=6, linespacing=1.15)
        ax.text(3.85, 0.95, "no against-interest\nobjection exists", ha="center", va="center", fontsize=6.8,
                color=C["flag"], bbox=bb, zorder=6, linespacing=1.15)
    ax.text(K_N[0], K_N[1] - K_R - 0.08, "a second objector\nwithout a stake", ha="center", va="top", fontsize=6.0,
            color="#555", zorder=6, linespacing=1.1)
    ax.set_title(title, loc="left", fontsize=9.6, fontweight="bold", pad=2)
    ax.text(0.0, 0.995, subtitle, transform=ax.transAxes, fontsize=7.0, va="top", ha="left", color="#444",
            linespacing=1.25)


def g5n(save=True):
    F = "g5n"
    base = ("communities", "grok_aita", "populations")
    ol = (*base, "s2_codable", "graphs", "one_loser")
    bp = (*base, "s2_codable", "graphs", "both_party")
    wl = (*base, "s2_codable", "who_loses")
    sen = (*base, "s1_s2_codable", "sensor", "strata")
    # k = 1, the one-loser stratum
    n_ol = V(F, "one-loser n", *ol, "n")
    n_yta = V(F, "YTA n", *wl, "YTA", "n"); n_nta = V(F, "NTA n", *wl, "NTA", "n")
    w_yta = CI(F, "W objects on YTA, with interest", *wl, "YTA", "objection_rate", "advocate_1")
    c_nta = CI(F, "C objects on NTA, with interest", *wl, "NTA", "objection_rate", "advocate_2")
    c_yta = CI(F, "C objects on YTA, against interest", *wl, "YTA", "objection_rate", "advocate_2")
    w_nta = CI(F, "W objects on NTA, against interest", *wl, "NTA", "objection_rate", "advocate_1")
    p_one = V(F, "exactly one objects, one-loser", *ol, "p_exactly_one")
    fire_ol = CI(F, "fire one-loser", *ol, "fire_rate")
    nf_ol = V(F, "n fired one-loser", *ol, "fire_rate", "n_fired")
    both_ol = V(F, "both advocates object, one-loser", *ol, "edges", "advocate_1|advocate_2", "table", "both")
    n_third_ol = V(F, "neutral objected, one-loser", *ol, "nodes", "third", "n_objected")
    phi_ol = CI(F, "phi W,C one-loser", *ol, "edges", "advocate_1|advocate_2", "phi")
    pwf = V(F, "one-loser p wrong fired", *sen, "one_loser", "p_wrong_fired")
    pwu = V(F, "one-loser p wrong unfired", *sen, "one_loser", "p_wrong_unfired")
    ratio = V(F, "one-loser ratio", *sen, "one_loser", "ratio")
    lift_ol = CI(F, "one-loser lift", *sen, "one_loser", "lift")
    X("one-loser p_exactly_one == flooding_analysis", (EG, (*ol, "p_exactly_one")),
      (FL, ("aita", "table_16_12", "rows", "one_loser", "p_exactly_one", "point")))
    # k = n, the both-party stratum. ESH is k = n. NAH is in the stratum by the registered grouping of
    # 16.12 and 16.15.1 (verdict token), not by the stake definition, under which it has one loser; the
    # design document (section 3) records it as the anomaly the credible-signal principle does not predict
    n_bp = V(F, "both-party n", *bp, "n")
    n_esh = V(F, "ESH n", *wl, "ESH", "n"); n_nah = V(F, "NAH n", *wl, "NAH", "n")
    w_esh = V(F, "W objects on ESH", *wl, "ESH", "objection_rate", "advocate_1", "point")
    c_esh = V(F, "C objects on ESH", *wl, "ESH", "objection_rate", "advocate_2", "point")
    w_nah = V(F, "W objects on NAH", *wl, "NAH", "objection_rate", "advocate_1", "point")
    c_nah = V(F, "C objects on NAH", *wl, "NAH", "objection_rate", "advocate_2", "point")
    n_nah_r = V(F, "N objects on NAH", *wl, "NAH", "objection_rate", "third", "point")
    fire_bp = CI(F, "fire both-party", *bp, "fire_rate")
    fire_esh = CI(F, "fire ESH", "aita", "table_16_12", "rows", "ESH", "p_two_plus", fname=FL)
    fire_nah = CI(F, "fire NAH", "aita", "table_16_12", "rows", "NAH", "p_two_plus", fname=FL)
    phi_bp = CI(F, "phi W,C both-party", *bp, "edges", "advocate_1|advocate_2", "phi")
    lift_bp = CI(F, "both-party lift", *sen, "both_party", "lift")
    X("both-party fire == flooding_analysis", (EG, (*bp, "fire_rate", "point")),
      (FL, ("aita", "table_16_12", "rows", "both_party", "p_two_plus", "point")))
    X("ESH n == flooding_analysis", (EG, (*wl, "ESH", "n")), (FL, ("aita", "table_16_12", "rows", "ESH", "n")))
    nano_fire = V(F, "nano fire", "communities", "nano_aita", "populations", "s2_codable", "graphs", "all",
                  "fire_rate", "point")
    # k = 0, the binding games B1 (neutral-reader arm)
    b1 = ("communities", "grok_games_binding_neutral", "populations", "s2_codable", "graphs", "all")
    n_b1 = V(F, "B1 n", *b1, "n")
    n_b1_items = V(F, "B1 items", "communities", "grok_games_binding_neutral", "raw", "n_items")
    b1_obj = [V(F, f"B1 objected {p}", *b1, "nodes", p, "n_objected") for p in ("advocate_1", "advocate_2", "third")]
    b1_fire = V(F, "B1 fire", *b1, "fire_rate", "point")
    b1_acc = [CI(F, f"B1 group accuracy arm {i}", "arms", i, "accuracy", "group", "final", "all", fname=GB1)
              for i in (0, 1)]
    assert [_walk(_load(GB1), ("arms", i, "third_seat")) for i in (0, 1)] == ["neutral", "planner"]
    X("B1 n == game_B1_analysis arm 0", (EG, (*b1, "n")), (GB1, ("arms", 0, "n_debates")))
    # k = 1 with nothing to deviate to, the Scruples Dilemmas
    dl = ("communities", "grok_dilemmas", "populations", "s2_codable")
    n_dl = V(F, "Dilemmas n", *dl, "n")
    a_on_a = V(F, "A's advocate objects on ACTION_A", *dl, "who_loses", "ACTION_A", "objection_rate", "advocate_1", "point")
    b_on_b = V(F, "B's advocate objects on ACTION_B", *dl, "who_loses", "ACTION_B", "objection_rate", "advocate_2", "point")
    b_on_a = V(F, "B's advocate objects on ACTION_A", *dl, "who_loses", "ACTION_A", "objection_rate", "advocate_2", "point")
    a_on_b = V(F, "A's advocate objects on ACTION_B", *dl, "who_loses", "ACTION_B", "objection_rate", "advocate_1", "point")
    phi_dl = CI(F, "phi Dilemmas", *dl, "graphs", "all", "edges", "advocate_1|advocate_2", "phi")
    dl_fire = CI(F, "Dilemmas fire", *dl, "graphs", "all", "fire_rate")
    dl_nf = V(F, "Dilemmas n fired", *dl, "graphs", "all", "fire_rate", "n_fired")
    dls = ("communities", "grok_dilemmas", "populations", "s1_s2_codable", "sensor", "strata", "all")
    dl_sn = V(F, "Dilemmas sensor n", *dls, "n"); dl_snf = V(F, "Dilemmas sensor fired", *dls, "n_fired")
    dl_lift = CI(F, "Dilemmas lift", *dls, "lift")
    dl_share = V(F, "Dilemmas codable share", "communities", "grok_dilemmas", "guards", "checks", "s2_codable_share", "value")
    assert _walk(_load(EG), ("communities", "grok_dilemmas", "status")) == "READ_WITH_CAUTION"

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.9))
    _kloser_panel(axes[0], 1, title="k = 1 loser, the sensor regime",
                  subtitle="one with-interest objection is expected. any second objector is against interest or "
                           "disinterested,\nso the stake-blind counter (two or more objectors) inherits the "
                           "against-interest signal")
    _kloser_panel(axes[1], 0, title="k = 0 losers, silence",
                  subtitle="the verdict undermines no stake. nothing is spoken against interest because\nnothing "
                           "is spoken at all. both counters are silent and the lift is undefined")
    _kloser_panel(axes[2], "n", title="k = n losers, flooding",
                  subtitle="every advocate objects with interest, so with-interest objections alone reach any\n"
                           "stake-blind threshold up to n. the counter fires on most debates and carries nothing")
    box = dict(boxstyle="round,pad=0.45", fc="#f7f7f7", ec="#bbb", lw=0.6)
    t1 = (f"measured at n = 2, grok AITA one-loser verdicts, {n_ol:,} debates\n"
          f"with interest, W objects on {_f(w_yta[0])} of YTA (n {n_yta}) and C on {_f(c_nta[0])} of NTA "
          f"(n {n_nta}). against interest, C objects on {_ci_txt(*c_yta)} of YTA and W on "
          f"{_ci_txt(*w_nta)} of NTA\n"
          f"exactly one seat objects on {_f(p_one)}. the counter fires on {_ci_txt(*fire_ol)}, {nf_ol} debates, "
          f"{both_ol} with both advocates and {nf_ol - both_ol} with the neutral seat, which objected on "
          f"{n_third_ol}\n"
          f"φ(W, C) {_ci_txt(*phi_ol, '+.3f')}. fired debates are wrong {_f(pwf)} vs {_f(pwu)} unfired, "
          f"{_f(ratio, '.2f')}x, lift {_ci_txt(*lift_ol, '+.3f')}")
    t0 = (f"measured at n = 2, games under a binding joint plan (B1, neutral reader), {n_b1_items} games x 3, "
          f"{n_b1} debates\n"
          f"objections {b1_obj[0]} / {b1_obj[1]} / {b1_obj[2]} of {n_b1} on the row advocate, column advocate and "
          f"neutral reader. counter fires {_f(b1_fire)}, lift undefined. group verdict {_f(b1_acc[0][0])} in the "
          f"neutral arm and {_f(b1_acc[1][0])} in the planner arm, the solo answer already suffices\n"
          f"silence also arrives at k = 1 when the favoured advocate has nowhere to go (Scruples Dilemmas, "
          f"a binary verdict, {n_dl} codable debates, READ_WITH_CAUTION at codable share {_f(dl_share)}). "
          f"the loser objects on {_f(a_on_a)} / {_f(b_on_b)}, the favoured seat on {_f(b_on_a)} / {_f(a_on_b)}, "
          f"φ {_ci_txt(*phi_dl, '+.3f')}, the counter fires {dl_nf} of {n_dl} ({_f(dl_fire[0])}), {dl_snf} of "
          f"{dl_sn} scored, lift {_ci_txt(*dl_lift, '+.3f')}")
    tn = (f"measured at n = 2, grok AITA both-party verdicts, {n_bp} debates. ESH (everyone at fault, n {n_esh}) "
          f"is k = n. NAH (no one at fault, n {n_nah}) sits in the stratum by the registered grouping, not by "
          f"stake: it has one loser (C) yet does not behave like NTA, the anomaly of section 3\n"
          f"with interest, on ESH W objects on {_f(w_esh)} and C on {_f(c_esh)}. on NAH C objects on "
          f"{_f(c_nah)}, but W, favoured by stake, objects on {_f(w_nah)} and N on {_f(n_nah_r)}\n"
          f"the counter fires on {_ci_txt(*fire_bp)}, ESH {_ci_txt(*fire_esh)} and NAH "
          f"{_ci_txt(*fire_nah)}. φ(W, C) {_ci_txt(*phi_bp, '+.3f')}, lift {_ci_txt(*lift_bp, '+.3f')}. "
          f"the verdict type is the signal\n"
          f"a model whose advocates ignore the with-interest rule floods at every k (nano, fire {_f(nano_fire)})")
    for ax, t in zip(axes, (t1, t0, tn)):
        lines = []
        for para in t.split("\n"):
            lines += textwrap.wrap(para, width=86, break_long_words=False, break_on_hyphens=False)
        ax.text(0.0, -0.02, "\n".join(lines), transform=ax.transAxes, fontsize=6.9, va="top", ha="left",
                color="#333", linespacing=1.38, bbox=box)
    fig.suptitle("The k-loser generalisation. n advocates object to the moderator's verdict, the k it goes "
                 "against are shaded, and only the n − k favoured advocates can object against interest",
                 x=0.01, y=0.985, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, 0.055,
             "the flag statistic for general (n, k) is the count of against-interest objections from the n − k "
             "favoured advocates. a stake-blind threshold must exceed k, because the k with-interest objections "
             "alone reach k (arithmetic, not a cited result),\nso the deployed counter (threshold 2) is the special "
             "case n = 2, k = 1, where any second objector is against interest or disinterested. at k = 0 a "
             "one-dimensional stake predicts silence, and NAH (k = 0 by the at-fault subset) floods instead.\nat "
             "k = n no against-interest objection exists and a stake-blind counter floods. "
             "the 16.21 three-party instrument (section 4 of graph_theory_design.md) runs k from 0 to 3 and tests "
             "both counters at each k.",
             fontsize=7.4, color="#444", va="bottom", ha="left", linespacing=1.4)
    fig.text(0.01, 0.012,
             "thick grey arrows are with-interest objections (from a shaded loser). red arrows are against-interest "
             "objections (from a favoured advocate). dotted arrows carry no objection. N is a disinterested seat, "
             "optional, whose objection counts as a second objector.\nthe schematic draws n = 4 for generality. "
             "every number in the grey boxes is measured at n = 2 and read from emergent_graphs.json, "
             "flooding_analysis.json and game_B1_analysis.json under --check",
             fontsize=6.8, color="#666", va="bottom", ha="left", linespacing=1.35)
    # the axes box matches the drawing's aspect (5.9 by 4.3 units) so no blank band opens above or below it
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.41, wspace=0.05)
    _save(fig, "g5_n_perspectives.png", save)


# ---------------------------------------------------------------- Fig g6, the taxonomy grid
BODY_PT = 8.0    # the grid's body font size
CPI = 16.6 * 7.6 / BODY_PT   # characters per inch of DejaVu Sans at BODY_PT, used only for wrapping
LH = BODY_PT / 72 * 1.3      # line height in inches at 1.3 linespacing
G6_COLS = [("interest structure", 1.95), ("graph", 1.30), ("which edge exists", 2.80), ("what the flag does", 2.35),
           ("what the record measured", 3.85), ("status", 1.45)]


def _glyph(ax, spec):
    """A small three-or-four-node graph. spec keys: nodes (list of (name, x, y,
    kind)), edges (list of (a, b, kind)), flag (name of edge carrying the red
    flag mark, or None), flag_style ('on', 'off', 'flood')."""
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.2, 1.4); ax.set_aspect("equal"); ax.axis("off")
    pos = {n[0]: (n[1], n[2]) for n in spec["nodes"]}
    kinds = {n[0]: n[3] for n in spec["nodes"]}
    R = 0.27
    style = {"neg": dict(color=C["neg"], lw=2.6, ls="-"), "weak": dict(color=C["neg"], lw=0.9, ls="-"),
             "pos": dict(color=C["pos"], lw=2.2, ls="-"), "none": dict(color=C["gray"], lw=0.8, ls=(0, (1.5, 2.5))),
             "with": dict(color=C_WITH, lw=2.6, ls="-"), "thin": dict(color="#777", lw=0.9, ls="-"),
             "rec": dict(color=C["pos"], lw=1.4, ls="-"), "verify": dict(color="#555", lw=0.9, ls=(0, (2, 2)))}
    for a, b, kind in spec["edges"]:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
        s = (x1 + ux * R, y1 + uy * R); e = (x2 - ux * R, y2 - uy * R)
        st = style[kind]
        if kind in ("rec", "thin", "with"):
            ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=7, color=st["color"], lw=st["lw"],
                                         ls=st["ls"], shrinkA=0, shrinkB=0, zorder=1))
        else:
            ax.plot([s[0], e[0]], [s[1], e[1]], color=st["color"], lw=st["lw"], ls=st["ls"], zorder=1,
                    solid_capstyle="round")
        if spec.get("flag") == (a, b):
            mx, my = (s[0] + e[0]) / 2, (s[1] + e[1]) / 2
            fs = spec.get("flag_style", "on")
            fc = {"on": C["flag"], "off": "white", "flood": C_WITH}[fs]
            ax.plot([mx], [my], marker="D", ms=6, mfc=fc, mec=C["flag"], mew=1.0, zorder=3)
    for name, (x, y) in pos.items():
        kind = kinds[name]
        if kind == "loser":
            fc, ec, ls, tc = C_LOSER, C["nodeedge"], "-", "white"
        elif kind == "adv":
            fc, ec, ls, tc = C["node"], C["nodeedge"], "-", C["text"]
        elif kind == "mod":
            fc, ec, ls, tc = C["mod"], "#333", "-", C["text"]
        elif kind == "dashed":
            fc, ec, ls, tc = "white", "#555", (0, (2, 2)), C["text"]
        else:
            fc, ec, ls, tc = "white", C["gray"], "-", C["text"]
        ax.add_patch(Circle((x, y), R, fc=fc, ec=ec, lw=0.9, ls=ls, zorder=2))
        ax.text(x, y, name, ha="center", va="center", fontsize=6.4, color=tc, zorder=4)
    if spec.get("note"):
        ax.text(0.0, -1.12, spec["note"], ha="center", va="bottom", fontsize=5.8, color="#555")


def _wrap(text, width_in):
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width=max(8, int(width_in * CPI)), break_long_words=False,
                               break_on_hyphens=False) or [""]
    return lines


def g6(save=True):
    F = "g6"
    eg = _load(EG)
    base = ("communities", "grok_aita", "populations")
    ol = (*base, "s2_codable", "graphs", "one_loser")
    bp = (*base, "s2_codable", "graphs", "both_party")
    sen = (*base, "s1_s2_codable", "sensor", "strata")
    wl = (*base, "s2_codable", "who_loses")
    # row 1, one loser
    n_ol = V(F, "one-loser n", *ol, "n")
    phi_ol = CI(F, "phi one-loser", *ol, "edges", "advocate_1|advocate_2", "phi")
    p_one = V(F, "exactly one objects", *ol, "p_exactly_one")
    fire_ol = V(F, "fire one-loser", *ol, "fire_rate", "point")
    pwf = V(F, "p wrong fired", *sen, "one_loser", "p_wrong_fired")
    pwu = V(F, "p wrong unfired", *sen, "one_loser", "p_wrong_unfired")
    ratio = V(F, "ratio", *sen, "one_loser", "ratio")
    lift_ol = CI(F, "lift one-loser", *sen, "one_loser", "lift")
    # row 2, Dilemmas
    dl = ("communities", "grok_dilemmas", "populations", "s2_codable")
    n_dl = V(F, "Dilemmas n", *dl, "n")
    phi_dl = CI(F, "phi Dilemmas", *dl, "graphs", "all", "edges", "advocate_1|advocate_2", "phi")
    dl_fire = V(F, "Dilemmas fire", *dl, "graphs", "all", "fire_rate", "point")
    dl_nf = V(F, "Dilemmas n fired", *dl, "graphs", "all", "fire_rate", "n_fired")
    dls = ("communities", "grok_dilemmas", "populations", "s1_s2_codable", "sensor", "strata", "all")
    dl_sn = V(F, "Dilemmas sensor n", *dls, "n"); dl_snf = V(F, "Dilemmas sensor fired", *dls, "n_fired")
    dl_lift = CI(F, "Dilemmas lift", *dls, "lift")
    dl_share = V(F, "Dilemmas codable share", "communities", "grok_dilemmas", "guards", "checks", "s2_codable_share", "value")
    act_n = V(F, "Dilemmas actuator flagged n", "flagged", "n", fname=DA)
    act_d = V(F, "Dilemmas actuator delta", "flagged", "delta", fname=DA)
    act_lo = V(F, "Dilemmas actuator lo", "flagged", "lo", fname=DA)
    act_hi = V(F, "Dilemmas actuator hi", "flagged", "hi", fname=DA)
    X("Dilemmas sensor n == actuator n_debates", (EG, (*dls, "n")), (DA, ("n_debates",)))
    X("Dilemmas sensor fired == actuator n_flagged", (EG, (*dls, "n_fired")), (DA, ("n_flagged",)))
    # row 3, everyone loses
    fire_esh = CI(F, "fire ESH", "aita", "table_16_12", "rows", "ESH", "p_two_plus", fname=FL)
    fire_nah = CI(F, "fire NAH", "aita", "table_16_12", "rows", "NAH", "p_two_plus", fname=FL)
    w_nah = V(F, "W objects on NAH", *wl, "NAH", "objection_rate", "advocate_1", "point")
    phi_esh = CI(F, "phi W,C within ESH", "aita", "table_16_12", "rows", "ESH", "phi", fname=FL)
    phi_nah = CI(F, "phi W,C within NAH", "aita", "table_16_12", "rows", "NAH", "phi", fname=FL)
    X("flooding_analysis both-party phi == emergent_graphs both-party phi",
      (FL, ("aita", "table_16_12", "rows", "both_party", "phi", "point")), (EG, (*bp, "edges", "advocate_1|advocate_2", "phi", "point")))
    phi_bp = CI(F, "phi both-party", *bp, "edges", "advocate_1|advocate_2", "phi")
    lift_bp = CI(F, "lift both-party", *sen, "both_party", "lift")
    c1_priv = CI(F, "C1 advocates private at r0", "readings", "P1_prime", "cgnb_game_neutral", fname=GC1)
    c1_priv_n = V(F, "C1 private n", "readings", "P1_prime", "cgnb_game_neutral", "n", fname=GC1)
    p3 = ("contrast", "P3_prime")
    p3_d = V(F, "C1 P3' delta", *p3, "delta", fname=GC1)
    p3_lo = V(F, "C1 P3' lo", *p3, "lo", fname=GC1); p3_hi = V(F, "C1 P3' hi", *p3, "hi", fname=GC1)
    c1 = ("communities", "grok_games_nonbinding_neutral", "populations", "s2_codable", "graphs", "all")
    c1_nf = V(F, "C1 fired", *c1, "fire_rate", "n_fired"); c1_n = V(F, "C1 n", *c1, "n")
    c1_acc = CI(F, "C1 neutral-arm group accuracy", "arms", 0, "accuracy", "group", "final", "all", fname=GC1)
    c1p = ("communities", "grok_games_nonbinding_planner", "populations", "s2_codable", "graphs", "all")
    c1_nf_p = V(F, "C1 planner fired", *c1p, "fire_rate", "n_fired"); c1_n_p = V(F, "C1 planner n", *c1p, "n")
    assert _walk(_load(GC1), ("arms", 0, "third_seat")) == "neutral", "game_C1_analysis arm order changed"
    X("C1 n == game_C1_analysis arm 0", (EG, (*c1, "n")), (GC1, ("arms", 0, "n_debates")))
    # row 4, no one loses
    b1 = ("communities", "grok_games_binding_neutral", "populations", "s2_codable", "graphs", "all")
    n_b1 = V(F, "B1 n", *b1, "n")
    b1_obj = [V(F, f"B1 objected {p}", *b1, "nodes", p, "n_objected") for p in ("advocate_1", "advocate_2", "third")]
    b1_acc = [CI(F, f"B1 group accuracy arm {i}", "arms", i, "accuracy", "group", "final", "all", fname=GB1)
              for i in (0, 1)]
    # row 5, multiple equilibria
    co_n = V(F, "coordination cells", "arms", "neutral", "n_cells", fname=GCO)
    co_np = V(F, "coordination cells planner", "arms", "planner", "n_cells", fname=GCO)
    co_neu = CI(F, "COORD neutral", "arms", "neutral", "COORD", fname=GCO)
    co_pla = CI(F, "COORD planner", "arms", "planner", "COORD", fname=GCO)
    co_solo = V(F, "COORD solo", "solo", "COORD", "point", fname=GCO)
    co_solo_n = V(F, "solo rows", "solo", "n_rows", fname=GCO)
    d2 = CI(F, "D2 neutral minus solo", "contrasts", "D2 neutral - solo, COORD", fname=GCO)
    d1 = CI(F, "D1 planner minus neutral", "contrasts", "D1 planner - neutral, COORD", fname=GCO)
    lock_neu = V(F, "coordination lock neutral", "arms", "neutral", "LOCK_r0_prefers_own_equilibrium", "point", fname=GCO)
    lock_pla = V(F, "coordination lock planner", "arms", "planner", "LOCK_r0_prefers_own_equilibrium", "point", fname=GCO)
    acc_oth = V(F, "accept other's equilibrium neutral", "arms", "neutral", "ACCEPT_favours_other", "point", fname=GCO)
    acc_oth_p = V(F, "accept other's equilibrium planner", "arms", "planner", "ACCEPT_favours_other", "point", fname=GCO)
    co_fired = V(F, "coordination flag fired neutral", "arms", "neutral", "FLAG_n_fired", fname=GCO)
    co_fired_p = V(F, "coordination flag fired planner", "arms", "planner", "FLAG_n_fired", fname=GCO)
    co_cnt_neu = round(co_neu[0] * co_n); co_cnt_pla = round(co_pla[0] * co_np)
    assert abs(co_cnt_neu / co_n - co_neu[0]) < 1e-9 and abs(co_cnt_pla / co_np - co_pla[0]) < 1e-9
    X("coordination cells == emergent_graphs n", (GCO, ("arms", "neutral", "n_cells")),
      (EG, ("communities", "grok_games_coordination_neutral", "populations", "s2_codable", "graphs", "all", "n")))
    # row 6, no unique answer (B2 cannot-assist)
    b2p = ("arms", 1, "accuracy", "group", "final", "by_region", "cannot_assist")
    b2n = ("arms", 0, "accuracy", "group", "final", "by_region", "cannot_assist")
    b2_pla = CI(F, "B2 planner cannot-assist", *b2p, fname=GB2)
    b2_neu = CI(F, "B2 neutral cannot-assist", *b2n, fname=GB2)
    b2_ng = V(F, "B2 cannot-assist games", *b2p, "n_items", fname=GB2)
    so_ca = ("by_region", "cannot_assist")
    so_acc = V(F, "solo cannot-assist accuracy", *so_ca, "accuracy", fname=GSO)
    so_lo = V(F, "solo cannot-assist lo", *so_ca, "lo", fname=GSO); so_hi = V(F, "solo cannot-assist hi", *so_ca, "hi", fname=GSO)
    so_ng = V(F, "solo cannot-assist games", *so_ca, "n_items", fname=GSO)
    b2_fired = V(F, "B2 planner flag fired n", "arms", 1, "flag_concentration", "fire_rate", "all", "n", fname=GB2)
    b2_fire = V(F, "B2 planner flag fire", "arms", 1, "flag_concentration", "fire_rate", "all", "point", fname=GB2)
    assert _walk(_load(GB2), ("arms", 1, "third_seat")) == "planner" and _walk(_load(GB2), ("arms", 0, "third_seat")) == "neutral"
    X("B2 cannot-assist games == solo cannot-assist games", (GB2, (*b2p, "n_items")), (GSO, (*so_ca, "n_items")))
    q = RECORD_QUOTES["b2_planner_minus_solo_cannot_assist"]
    b2_delta = b2_pla[0] - so_acc
    assert abs(round(b2_delta, 3) - q["point"]) < 1e-9, (b2_delta, q)
    assert b2_ng == q["n_games"]
    # row 8, verifiable claims (the 16.11a record quote carried in the artefact)
    r = ("protocol", "annotations", "routing_gain_over_solo_16_11a")
    assert eg["protocol"]["annotations"]["routing_gain_over_solo_16_11a"]["kind"].startswith("record quote")
    q_d = V(F, "16.11a routed minus solo (record quote)", *r, "routed_embodied", "delta_vs_solo")
    q_lo = V(F, "16.11a lo (record quote)", *r, "routed_embodied", "lo")
    q_hi = V(F, "16.11a hi (record quote)", *r, "routed_embodied", "hi")
    q_cov = V(F, "16.11a coverage (record quote)", *r, "routed_embodied", "coverage")
    q_dom = V(F, "16.11a sonnet everywhere (record quote)", *r, "sonnet_everywhere", "routed_embodied_minus_this")

    A, B, M = ("A", -0.72, 0.62), ("B", 0.72, 0.62), ("M", 0.0, -0.55)
    rows = [
        dict(structure="strictly opposed, exactly one loser\n(AITA verdicts YTA and NTA)",
             glyph=dict(nodes=[(*A, "loser"), (*B, "adv"), (*M, "mod")],
                        edges=[("A", "B", "neg"), ("A", "M", "with"), ("B", "M", "thin")], flag=("A", "B"),
                        flag_style="on"),
             edge="one advocate per side, a moderator whose r3 labels create the objection round, a naive verdict "
                  "node. the flag sits on the advocate edge, which is anti-correlated because exactly one seat "
                  "loses\n(Milgrom and Roberts 1986, Lipman and Seppi 1995, Shin 1998)",
             flag="the loser objects with interest, so a second objector is against interest or disinterested "
                  "and the stake-blind counter inherits the signal",
             record=f"φ {_ci_txt(*phi_ol, '+.3f')} on {n_ol:,} one-loser debates. exactly one seat objects on "
                    f"{_f(p_one)} and the counter fires on {_f(fire_ol)}. fired debates are wrong {_f(pwf)} vs "
                    f"{_f(pwu)} unfired, {_f(ratio, '.2f')}x, lift {_ci_txt(*lift_ol, '+.3f')}",
             status="done. g3"),
        dict(structure="one loser, binary verdict space\n(Scruples Dilemmas)",
             glyph=dict(nodes=[(*A, "loser"), (*B, "adv"), (*M, "mod")],
                        edges=[("A", "B", "neg"), ("A", "M", "with"), ("B", "M", "none")], flag=("A", "B"),
                        flag_style="off", note="favoured seat has nowhere to go"),
             edge="the same graph. the account that the favoured advocate has no partial concession to move to, "
                  "so the edge exists and is never crossed, is post hoc: the record expected about 30 firings at "
                  "the Stage 1 rate and got 6 (prereg L2634 to L2641). no cited theory predicted the silence",
             flag="almost never fires. the loser objects, the favoured seat accepts, and there is no second "
                  "objector to read",
             record=f"φ {_ci_txt(*phi_dl, '+.3f')} on {n_dl} codable debates. the counter fires {dl_nf} of "
                    f"{n_dl} ({_f(dl_fire)}), {dl_snf} of {dl_sn} scored, lift {_ci_txt(*dl_lift, '+.3f')}. "
                    f"the actuator on the {act_n} flagged debates {_ci_txt(act_d, act_lo, act_hi, '+.3f')}. "
                    f"S2-codable share {_f(dl_share)}, READ_WITH_CAUTION",
             status="done. g4. 16.21 tests whether a richer verdict space restores the signal"),
        dict(structure="everyone loses\n(ESH, and NAH by the registered grouping)",
             glyph=dict(nodes=[(*A, "loser"), (*B, "loser"), (*M, "mod")],
                        edges=[("A", "B", "weak"), ("A", "M", "with"), ("B", "M", "with")], flag=("A", "B"),
                        flag_style="flood", note="both object with interest"),
             edge="no dissent edge survives. every objection is with interest and the verdict type is the "
                  "signal\n(Battaglini 2002 read as an interpretation, a two-party fault verdict is "
                  "two-dimensional and two opposed advocates are collinear)",
             flag="floods. with-interest objections alone reach the threshold, so the counter fires on most "
                  "debates and its lift is not positive",
             record=f"the counter fires on {_ci_txt(*fire_esh)} of ESH and {_ci_txt(*fire_nah)} of NAH. "
                    f"both-party φ {_ci_txt(*phi_bp, '+.3f')}, lift {_ci_txt(*lift_bp, '+.3f')}. NAH is the "
                    f"anomaly of section 3: by the stake definition it has one loser, yet the writer's advocate "
                    f"objects on {_f(w_nah)} of it, so its place in this row is the registered grouping by "
                    f"verdict token and not a derivation from who loses",
             status="done. g3"),
        dict(structure="no one loses, common interest\n(binding joint plan B1, assurance under binding)",
             glyph=dict(nodes=[(*A, "adv"), (*B, "adv"), (*M, "mod")],
                        edges=[("A", "B", "none"), ("A", "M", "none"), ("B", "M", "none")], flag=None,
                        note="nothing to argue"),
             edge="nothing to argue over, a solo answer suffices and advocates do not earn their cost\n"
                  "(Dewatripont and Tirole 1999, advocacy pays only under conflict)",
             flag="silent. no seat objects, the counter never fires and its lift is undefined",
             record=f"objections {b1_obj[0]} / {b1_obj[1]} / {b1_obj[2]} of {n_b1} on the row advocate, column "
                    f"advocate and neutral reader. group verdict {_ci_txt(*b1_acc[0])} in the neutral arm and "
                    f"{_ci_txt(*b1_acc[1])} in the planner arm",
             status="done. g4"),
        dict(structure="no advocate loses because the verdict is the equilibrium every node recommends\n"
                       "(one-shot prisoner's dilemma without enforcement, C1)",
             glyph=dict(nodes=[(*A, "adv"), (*B, "adv"), (*M, "mod")],
                        edges=[("A", "B", "none"), ("A", "M", "thin"), ("B", "M", "thin")], flag=None,
                        note="every node names Nash"),
             edge="none that helps. no seat holds a recommendation the verdict undermines, so there is no edge "
                  "to read, and no third node can make the better outcome rational\n(Aumann 1974 as read in "
                  "16.13c, the correlated set of a one-shot PD is its Nash set, a reading the record marks as "
                  "post hoc, L5090. Myerson 1986 only for the revelation-principle reading)",
             flag="silent, not flooded. the advocates and the verdict agree on the equilibrium, so nothing is "
                  "objected to (prereg L5069, 'not flooding; the sensor is SILENT')",
             record=f"advocates name the private action at r0 on {_ci_txt(*c1_priv)} ({c1_priv_n} votes), "
                    f"group verdict {_ci_txt(*c1_acc)}, planner minus neutral "
                    f"{_ci_txt(p3_d, p3_lo, p3_hi, '+.3f')} (P3' fails), the stake-blind counter fires "
                    f"{c1_nf} of {c1_n} in the neutral arm and {c1_nf_p} of {c1_n_p} in the planner arm",
             status="done. g4. an earlier draft of this grid placed C1 in the flooding row. open, repeated play "
                    "or a game whose correlated set exceeds Nash"),
        dict(structure="multiple equilibria\n(battle of the sexes, chicken, 16.13d)",
             glyph=dict(nodes=[(*A, "adv"), (*B, "adv"), ("R", 0.0, -0.55, "mod")],
                        edges=[("A", "B", "pos"), ("R", "A", "rec"), ("R", "B", "rec")], flag=None,
                        note="R publishes a recommendation"),
             edge="a public recommendation node with edges to both advocates. the advocate edge is a "
                  "coordination edge, any third seat serves\n(Aumann 1974, a recommendation is self-enforcing "
                  "once both hear it. Myerson 1986 only for the revelation-principle reading, a mediator achieves "
                  "nothing a direct mechanism with the same commitment cannot)",
             flag="no flag fires. the advocates lock at r0 and then accept the other player's equilibrium on "
                  "every vote, so nothing is objected to",
             record=f"the collective coordinates on {_f(co_neu[0])} ({co_cnt_neu} of {co_n} cells) with the "
                    f"neutral reader and {_f(co_pla[0])} ({co_cnt_pla} of {co_np}) with the planner, the solo "
                    f"on {_f(co_solo)} ({co_solo_n} rows). D2 neutral minus solo {_ci_txt(*d2, '+.3f')}. "
                    f"D1 planner minus neutral {_ci_txt(*d1, '+.3f')}. r0 lock {_f(lock_neu)} / "
                    f"{_f(lock_pla)}, accept the other's equilibrium {_f(acc_oth)} / {_f(acc_oth_p)}, flag fired "
                    f"{co_fired} and {co_fired_p} of {co_n}",
             status=f"done at {co_n} cells. the coordination gain belongs to the collective with either third "
                    f"seat, not to the mediator (D1 fails). powered replication is future work"),
        dict(structure="no unique answer\n(cannot-assist games under a binding plan, B2)",
             glyph=dict(nodes=[(*A, "adv"), (*B, "adv"), ("T", 0.0, -0.55, "dashed")],
                        edges=[("A", "B", "none"), ("A", "T", "thin"), ("B", "T", "thin")], flag=None,
                        note="T carries restraint in its brief"),
             edge="a collective that declines where the solo names a plan. whether the restraint is the third "
                  "seat's brief or the deliberation is not separable, the record says so (prereg L5055 to "
                  "L5057), and planner minus neutral on these games includes 0 (L5037)\n(selective prediction)",
             flag="silent. no seat objects on any game in the extent map",
             record=f"on the {b2_ng} cannot-assist games the planner arm scores {_ci_txt(*b2_pla)}, the neutral "
                    f"arm {_ci_txt(*b2_neu)} and the solo {_ci_txt(so_acc, so_lo, so_hi)} ({so_ng} games). "
                    f"planner minus solo {_f(b2_delta, '+.3f')}, record interval [{_f(q['lo'], '+.3f')}, "
                    f"{_f(q['hi'], '+.3f')}] (prereg L{q['line']}). the counter fires {_f(b2_fire)} on "
                    f"{b2_fired} games",
             status="observed against the registered P4 (neither arm beats the solo on cannot-assist games, "
                    "CONTRADICTED for the planner arm, L5034 to L5045). unreplicated"),
        dict(structure="partially aligned coalitions\n(shared interests among n perspectives)",
             glyph=dict(nodes=[("A1", -0.85, 0.55, "loser"), ("A2", 0.0, 1.0, "loser"), ("B", 0.85, 0.55, "adv"),
                               (*M, "mod")],
                        edges=[("A1", "A2", "pos"), ("A1", "B", "weak"), ("A2", "B", "weak"), ("A1", "M", "with"),
                               ("A2", "M", "with"), ("B", "M", "thin")], flag=("A1", "A2"), flag_style="on",
                        note="flag reads inside the coalition"),
             edge="hypothesised: within-coalition edges positive, cross-coalition edges negative, the flag reads "
                  "against-interest objections from inside a coalition\n(Krishna and Morgan 2001, like-biased "
                  "experts add nothing. Chwe 1994 as a pointer to farsighted coalition formation; the "
                  "co-objection claim is the design's own)",
             flag="untested. predicted to flood if it counts objectors stake-blind, since a coalition of k "
                  "losers reaches k on its own",
             record=f"none at n > 2. the n = 2 null is that co-losers do not object together: within ESH "
                    f"φ {_ci_txt(*phi_esh, '+.3f')} and within NAH φ {_ci_txt(*phi_nah, '+.3f')}",
             status="16.19 ally seat and the k = 2 stratum of 16.21, approved, to register"),
        dict(structure="verifiable against unverifiable claims",
             glyph=dict(nodes=[(*A, "adv"), (*B, "adv"), ("D", 0.0, -0.55, "mod"), ("✓", 0.95, -0.55, "plain")],
                        edges=[("A", "B", "none"), ("A", "D", "thin"), ("B", "D", "thin"), ("D", "✓", "verify")],
                        flag=None, note="D checks what it can"),
             edge="the same star, with what the decider can check as the variable\n(Lipman and Seppi 1995, "
                  "little provability is needed given conflict. Okuno-Fujiwara, Postlewaite and Suzumura 1990. "
                  "Glazer and Rubinstein 2004, a verification budget)",
             flag="a sceptical decider that can check the case needs the sensor less",
             record=f"our advocates argue over a fixed shared account and may not invent facts. the sonnet judge "
                    f"routed on the embodied flag beats grok solo by {_ci_txt(q_d, q_lo, q_hi, '+.3f')} at "
                    f"coverage {_f(q_cov)} and loses to sonnet everywhere by {_f(-q_dom)} (16.11a record quote "
                    f"carried in the artefact protocol block)",
             status="to register, not in the approved sequence"),
    ]

    # layout. column widths in inches, row heights from the wrapped text
    left, right, top = 0.25, 0.25, 0.30
    xs = []; x = left
    for _, w in G6_COLS:
        xs.append(x); x += w
    W = x + right
    text_cols = [("structure", 0), ("edge", 2), ("flag", 3), ("record", 4), ("status", 5)]
    wrapped = []
    heights = []
    for row in rows:
        wr = {}
        n_lines = 1
        for key, ci in text_cols:
            wr[key] = _wrap(row[key], G6_COLS[ci][1] - 0.18)
            n_lines = max(n_lines, len(wr[key]))
        wrapped.append(wr)
        heights.append(max(1.30, n_lines * LH + 0.28))
    header_h = 0.42
    title_h = 0.55
    footer = (
        "glyphs. shaded nodes lose under the verdict. thick blue is an anti-correlated advocate edge, thin blue "
        "a collapsed one, orange a positive (coalition or coordination) edge, dotted grey no edge. thick grey "
        "arrows are with-interest objections, thin arrows other reports to the moderator M, the recommender R, "
        "the restrained seat T or the decider D. the red diamond is the flag on the edge the counter reads, "
        "hollow where it never fires, grey where it floods.\n"
        "every number is read from emergent_graphs.json, flooding_analysis.json, game_B1_analysis.json, "
        "game_C1_analysis.json, game_coordination_analysis.json, game_B2_analysis.json, game_solo_analysis.json "
        "and dilemma_stage2_actuator_analysis.json under --check, except the two record quotes named in their "
        "cells (the B2 interval at prereg L5039, whose point is recomputed from the two game artefacts, and the "
        "16.11a routing gain carried as a flagged quote in the artefact). author and year in brackets are the "
        "verified entries of references.bib and advocacy_refs.bib. brackets on numbers are item-clustered 95% "
        "bootstrap intervals.")
    footer_lines = []
    for para in footer.split("\n"):
        footer_lines += textwrap.wrap(para, width=int((W - left - right) * CPI * BODY_PT / 6.9),
                                      break_long_words=False, break_on_hyphens=False)
    footer_h = 0.30 + 0.125 * len(footer_lines)
    H = top + title_h + header_h + sum(heights) + footer_h
    fig = plt.figure(figsize=(W, H))
    fx = lambda v: v / W
    fy = lambda v: 1.0 - v / H
    y = top
    fig.text(fx(left), fy(y), "The game-theoretic taxonomy of section 5. The structure of interests over the verdict "
                              "decides which edge must exist,\nwhat the flag does, and what the record has measured",
             fontsize=10.5, fontweight="bold", va="top", ha="left", linespacing=1.3)
    y += title_h
    for (name, w), x0 in zip(G6_COLS, xs):
        fig.text(fx(x0 + 0.06), fy(y + 0.08), name, fontsize=8.2, fontweight="bold", va="top", ha="left", color="#222")
    y += header_h
    fig.add_artist(Line2D([fx(left), fx(W - right)], [fy(y), fy(y)], color="#333", lw=0.9))
    for row, wr, h in zip(rows, wrapped, heights):
        for key, ci in text_cols:
            x0 = xs[ci]
            fig.text(fx(x0 + 0.06), fy(y + 0.12), "\n".join(wr[key]), fontsize=BODY_PT, va="top", ha="left",
                     color=(C["text"] if key != "structure" else "#111"), linespacing=1.3,
                     fontweight=("bold" if key == "structure" else "normal"))
        gx, gw = xs[1] + 0.05, G6_COLS[1][1] - 0.10
        gh = min(h - 0.12, 1.25)
        ax = fig.add_axes((fx(gx), fy(y + 0.06 + gh), gw / W, gh / H))
        _glyph(ax, row["glyph"])
        y += h
        fig.add_artist(Line2D([fx(left), fx(W - right)], [fy(y), fy(y)], color="#bbb", lw=0.6))
    y += 0.12
    fig.text(fx(left), fy(y), "\n".join(footer_lines), fontsize=6.9, color="#555", va="top", ha="left",
             linespacing=1.4)
    _save(fig, "g6_taxonomy.png", save)


# ---------------------------------------------------------------- driver
def _save(fig, name, save):
    if save:
        fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_all(save=True):
    REGISTRY.clear(); CROSS.clear()
    for fn in (g1, g2, g3, g4, g5, g5n, g6):
        fn(save=save)


def verify_record_quotes():
    """Read-only check that each quoted prereg line still carries the quoted
    text. The prereg is never written by this module."""
    bad = 0
    if not os.path.exists(PREREG):
        print("record quotes: prereg not found, quotes not checked")
        return 0
    with open(PREREG) as f:
        lines = f.read().split("\n")
    for name, q in RECORD_QUOTES.items():
        line = lines[q["line"] - 1] if q["line"] - 1 < len(lines) else ""
        ok = q["text"] in line
        bad += not ok
        print(f"quote  {name[:60]:60s} prereg L{q['line']} carries {q['text']!r}  "
              f"{'ok' if ok else 'MISMATCH ' + repr(line[:80])}")
    print(f"{len(RECORD_QUOTES)} record quote(s) checked against the prereg text")
    return bad


def verify():
    bad = 0
    print(f"{'figure':6s} {'plotted value':50s} {'artefact key':92s} value")
    for r in REGISTRY:
        fresh = _walk(_load(r["file"], fresh=True), r["path"])
        key = _key(r["file"], r["path"])
        if r["value"] is None or fresh is None:
            ok = r["value"] is None and fresh is None
        else:
            ok = abs(float(fresh) - float(r["value"])) <= TOL
        bad += not ok
        print(f"{r['fig']:6s} {r['label'][:50]:50s} {key[:92]:92s} {r['value']!r}{'' if ok else '  MISMATCH ' + repr(fresh)}")
    print(f"\n{len(REGISTRY)} plotted values checked against fresh artefact reads, tolerance {TOL}")
    for c in CROSS:
        a = _walk(_load(c["a"][0], fresh=True), c["a"][1]); b = _walk(_load(c["b"][0], fresh=True), c["b"][1])
        ok = abs(float(a) - float(b)) <= TOL
        bad += not ok
        print(f"cross  {c['label'][:60]:60s} {_key(*c['a'])[:70]:70s} == {_key(*c['b'])[:70]:70s} "
              f"{'ok' if ok else 'MISMATCH ' + repr((a, b))}")
    print(f"{len(CROSS)} cross-artefact equalities checked")
    bad += verify_record_quotes()
    return bad


def check():
    build_all(save=False)
    bad = verify()
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


if __name__ == "__main__":
    main()
