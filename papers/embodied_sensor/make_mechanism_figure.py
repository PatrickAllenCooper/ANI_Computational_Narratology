"""
make_mechanism_figure.py -- Figure 1 of the paper, the mechanism, drawn with
Graphviz (dot) for automatic, clean layout. Three separate sub-figures, each
one idea, large type, minimal text; the motivation goes in the caption, not
on the figure.

  fig_mechanism_a  embodiment: one account, three seats, one brief each
  fig_mechanism_b  the collective: seats argue, a moderator decides, seats object and vote
  fig_mechanism_c  the flag: exactly one seat is expected to object; a second objector is the flag

The numbers on C are read from divergence_study_outputs at build time and
asserted under --check.

  python papers/embodied_sensor/make_mechanism_figure.py            # writes figures/fig_mechanism_{a,b,c}.{pdf,png}
  python papers/embodied_sensor/make_mechanism_figure.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import graphviz

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
ART = ROOT / "divergence_study_outputs"

FONT = "Helvetica"
BLUE, ORANGE, GREY, GOLD, RED, PURPLE = "#1f4e79", "#b4561e", "#555555", "#8a6d1f", "#b3261e", "#5b3d8a"


def numbers() -> dict:
    rd = json.loads((ART / "router_decomposition.json").read_text())
    fl = json.loads((ART / "flooding_analysis.json").read_text())
    ol = rd["counter_lift_by_stratum"]["one_loser"]
    row = fl["aita"]["table_16_12"]["rows"]["one_loser"]
    val = lambda x: float(x["point"]) if isinstance(x, dict) else float(x)
    return {"p_wrong_fired": float(ol["p_wrong_fired"]), "p_wrong_unfired": float(ol["p_wrong_unfired"]),
            "ratio": float(ol["ratio"]), "coverage": float(rd["counter_lift_by_stratum"]["pooled"]["coverage"]),
            "exactly_one": val(row["p_exactly_one"]), "two_plus": val(row["p_two_plus"])}


def base(name: str, rankdir: str = "TB") -> graphviz.Digraph:
    g = graphviz.Digraph(name, format="pdf")
    g.attr(rankdir=rankdir, nodesep="0.45", ranksep="0.55", pad="0.15", dpi="200", bgcolor="white")
    g.attr("node", fontname=FONT, fontsize="16", shape="box", style="rounded", penwidth="1.4",
           margin="0.18,0.10", color="#333333", fontcolor="#111111")
    g.attr("edge", fontname=FONT, fontsize="13", color="#555555", penwidth="1.3", arrowsize="0.8")
    return g


def seat(g, nid, label, color):
    g.node(nid, label, shape="circle", width="0.85", fixedsize="true", fontsize="20",
           penwidth="2.4", color=color, fontcolor=color, style="")


# ---------------------------------------------------------------- A
def fig_a() -> graphviz.Digraph:
    g = base("fig_mechanism_a")
    g.node("acct", "one account of a dispute\n(the same text to every seat)", style="rounded,filled", fillcolor="#f2f2f2")
    with g.subgraph() as s:
        s.attr(rank="same")
        seat(s, "W", "W", BLUE); seat(s, "C", "C", ORANGE); seat(s, "N", "N", GREY)
    g.node("bw", "speaks for the writer\nstake: writer not at fault", shape="plaintext", fontsize="14", fontcolor=BLUE)
    g.node("bc", "speaks for the other party\nstake: writer at fault", shape="plaintext", fontsize="14", fontcolor=ORANGE)
    g.node("bn", "speaks for no one\nno stake", shape="plaintext", fontsize="14", fontcolor=GREY)
    for a, b in (("acct", "W"), ("acct", "C"), ("acct", "N")):
        g.edge(a, b)
    for a, b in (("W", "bw"), ("C", "bc"), ("N", "bn")):
        g.edge(a, b, style="invis")
    g.node("note", "each seat is a copy of the same model under its own brief", shape="plaintext", fontsize="13", fontcolor="#666666")
    g.edge("bc", "note", style="invis")
    return g


# ---------------------------------------------------------------- B
def fig_b() -> graphviz.Digraph:
    g = base("fig_mechanism_b")
    g.attr(ranksep="0.5")
    with g.subgraph() as s:
        s.attr(rank="same")
        seat(s, "W", "W", BLUE); seat(s, "C", "C", ORANGE); seat(s, "N", "N", GREY)
    g.edge("W", "C", dir="both", style="dashed", color="#888888", label=" argue ", fontcolor="#666666", constraint="false")
    g.edge("C", "N", dir="both", style="dashed", color="#888888", label=" argue ", fontcolor="#666666", constraint="false")
    g.node("M", "M\nmoderator\nno stake", shape="circle", width="1.25", fixedsize="true", fontsize="14",
           penwidth="2.4", color=GOLD, fontcolor=GOLD, style="filled", fillcolor="#fbf4e3")
    for x in ("W", "C", "N"):
        g.edge(x, "M", color=GOLD, label=" position " if x == "C" else "", fontcolor=GOLD)
    g.node("V", "verdict", style="rounded,filled", fillcolor="#f2f2f2")
    g.edge("M", "V", color=GOLD, label=" writes ", fontcolor=GOLD)
    g.node("O", "W, C and N each accept it,\nask for a change, or reject it", style="rounded", color=RED, fontcolor=RED)
    g.edge("V", "O", color=RED)
    g.node("V2", "revised verdict", style="rounded,filled", fillcolor="#f2f2f2")
    g.edge("O", "V2", color=GOLD, label=" M revises ", fontcolor=GOLD)
    g.node("T", "W, C and N each vote\naccept or reject", style="rounded", color=RED, fontcolor=RED)
    g.edge("V2", "T", color=RED)
    return g


# ---------------------------------------------------------------- C
def fig_c(n: dict) -> graphviz.Digraph:
    g = base("fig_mechanism_c")
    g.attr(ranksep="0.45")
    g.node("V", "the verdict goes against one party", style="rounded,filled", fillcolor="#f2f2f2")
    g.node("L", f"that party's seat objects\nexpected, so it carries nothing\nexactly one objector on {n['exactly_one']*100:.0f}% of cases",
           style="rounded", color=ORANGE, fontcolor=ORANGE, fontsize="14")
    g.node("L2", f"a second seat objects as well\nagainst its side, or with no side\n{n['two_plus']*100:.0f}% of cases",
           style="rounded", color=RED, fontcolor=RED, fontsize="14", penwidth="2")
    g.node("F", f"FLAG\nflagged cases are wrong {n['ratio']:.1f}x as often\n({n['p_wrong_fired']:.2f} vs {n['p_wrong_unfired']:.2f})",
           style="rounded,filled", fillcolor="#fdecea", color=RED, fontcolor=RED, penwidth="2")
    g.node("J", "a stronger judge\nwith no side", style="rounded,filled", fillcolor="#efe6f5", color=PURPLE, fontcolor=PURPLE)
    g.edge("V", "L", color="#777777")
    g.edge("V", "L2", color=RED, penwidth="2")
    with g.subgraph() as s:
        s.attr(rank="same"); s.node("L"); s.node("L2")
    g.edge("L2", "F", color=RED, penwidth="2")
    g.edge("F", "J", color=PURPLE, penwidth="2", label=f" route only the flagged {n['coverage']*100:.0f}% ", fontcolor=PURPLE)
    return g


def build(check: bool) -> int:
    n = numbers()
    if check:
        ok = (0.38 < n["p_wrong_fired"] < 0.40 and 0.03 < n["p_wrong_unfired"] < 0.05 and 9.5 < n["ratio"] < 10.2
              and 0.19 < n["coverage"] < 0.21 and 0.90 < n["exactly_one"] < 0.94 and 0.06 < n["two_plus"] < 0.08)
        print("numbers", {k: round(v, 4) for k, v in n.items()})
        print("CHECK OK" if ok else "CHECK FAILED"); return 0 if ok else 1
    OUT.mkdir(exist_ok=True)
    for g in (fig_a(), fig_b(), fig_c(n)):
        for fmt in ("pdf", "png"):
            g.format = fmt
            g.render(OUT / g.name, cleanup=True)
    print("wrote", ", ".join(f"figures/{x}.pdf/.png" for x in ("fig_mechanism_a", "fig_mechanism_b", "fig_mechanism_c")))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    sys.exit(build(ap.parse_args().check))
