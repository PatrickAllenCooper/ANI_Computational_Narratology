#!/usr/bin/env python3
"""Figure 2 of the ICLR draft as an annotated diverging heatmap (2026-09-24 redesign).

Rows are the seven generators; columns are judges. Panel (a) is judged emotional validation under
the five judges, panel (b) acceptance of the asker's framing under the two judges that scored it,
both on the same-day 4,096-token responses (the registered headline, prereg 17.10 and 17.11).
Each cell prints NoT minus CoT in points. A cell whose 95 percent interval excludes zero is filled
from a binned diverging scale (blue when NoT lowers the rate, red when it raises it; four
magnitude bins per arm, validated as ordinal ramps with the dataviz skill's validate_palette.js);
a cell whose interval includes zero is neutral gray with the value in muted ink.

Sources: divergence_study_outputs/headline_rep4k_readout.json (per_generator.<g>.per_judge.<j>:
drop, lo, hi) and framing_17_11_readout.json (judges.<j>.design_a_per_generator.<g>.drop_arm:
point, lo, hi).

  python3 scripts/iclr_fig_judge_heatmap.py --out papers/iclr2027/figures/fig_judge_panel.tex
  python3 scripts/iclr_fig_judge_heatmap.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT_DIR = Path("divergence_study_outputs")
GENS = [("claude-haiku-4-5", "haiku"), ("claude-sonnet-4-6", "sonnet"), ("gpt-5.4-nano", "nano"),
        ("grok-4-1-fast-reasoning", "grok"), ("Llama-3.3-70B-Instruct", "Llama"),
        ("Mistral-Large-3-2", "Mistral"), ("DeepSeek-V4-Pro", "DeepSeek")]
VAL_JUDGES = [("claude-haiku-4-5", "haiku", "(production)"), ("gpt-4o", "gpt-4o", ""),
              ("gpt-5.4-nano", "nano", ""), ("Llama-3.3-70B-Instruct", "Llama", ""),
              ("grok-4-1-fast-reasoning", "grok", "")]
FRAME_JUDGES = [("claude-haiku-4-5", "haiku", "(production)"), ("gpt-4o", "gpt-4o", "")]
# Binned diverging scale: bin edges in points of |difference|; blue arm (NoT lowers) and red arm
# (NoT raises), each validated with validate_palette.js --ordinal (all checks pass, light end
# >= 2:1 on the page); neutral gray for intervals that include zero.
EDGES = (10, 25, 45)
BLUE = ("86B6EF", "3987E5", "1C5CAB", "0D366B")
RED = ("F09794", "E34948", "B23130", "7A1D1D")
NEUTRAL = "F0EFEC"
INK, MUTED = "0B0B0B", "52514E"
# CELL_H 0.42 (was 0.50) and the colour bar at y = LEGEND_Y (was -0.50): vertical compaction only, for the
# 9-page budget with the pillar-2 stand-in (2026-09-24 wiring pass); layout, colours and labels unchanged
CELL_W, CELL_H, GAP = 1.30, 0.42, 0.06        # cm
LABEL_W, PANEL_GAP = 1.55, 0.55                # cm
# LEGEND_Y -0.32 (was -0.36): the 8pt tick labels (were 7pt) made the figure 0.64pt taller, which pushed three
# lines of Section 4 to the next page in the 9-page budget; raising the colour bar 0.04cm (1.1pt) pays for it,
# leaving 0.20cm between the last row and the bar (legibility review, 2026-09-25)
LEGEND_Y = -0.32                               # cm, centre of the colour bar below the rows


def bin_index(mag: float) -> int:
    return sum(mag >= e for e in EDGES)


def cell_style(d: float, lo: float, hi: float) -> tuple[str, str, bool]:
    """(fill hex, text hex, bold) for a difference d with interval [lo, hi], all in points."""
    if lo <= 0 <= hi:
        return NEUTRAL, MUTED, False
    i = bin_index(abs(d))
    fill = (BLUE if d < 0 else RED)[i]
    # Near-black text on the two lighter bins of each arm, white on the two darker ones. White on the second
    # bins (3987E5, E34948) had contrast 3.6:1 and 3.9:1; near-black there has 5.4:1 and 5.0:1, and the
    # validated colours are unchanged (legibility review 2026-09-25).
    return fill, ("FFFFFF" if i >= 2 else INK), True


def load() -> tuple[dict, dict]:
    rep = json.loads((OUT_DIR / "headline_rep4k_readout.json").read_text())
    fr = json.loads((OUT_DIR / "framing_17_11_readout.json").read_text())
    val = {}
    for g, _ in GENS:
        for j, _, _ in VAL_JUDGES:
            c = rep["per_generator"][g]["per_judge"][j]
            val[(g, j)] = (100 * c["drop"], 100 * c["lo"], 100 * c["hi"])
    frame = {}
    for g, _ in GENS:
        for j, _, _ in FRAME_JUDGES:
            c = fr["judges"][j]["design_a_per_generator"][g]["drop_arm"]
            frame[(g, j)] = (100 * c["point"], 100 * c["lo"], 100 * c["hi"])
    return val, frame


def fmt(d: float) -> str:
    s = f"{d:+.1f}".replace("-", "$-$").replace("+", "$+$")
    return s


def panel(x0: float, judges, data, title: str) -> list[str]:
    n = len(GENS)
    out = []
    width = len(judges) * CELL_W
    top = n * CELL_H
    out.append(f"\\node[anchor=south, font=\\small\\bfseries] at ({x0 + width / 2:.3f},{top + 0.72:.3f}) {{{title}}};")
    for k, (j, short, note) in enumerate(judges):
        xc = x0 + (k + 0.5) * CELL_W
        # the production marker at 8pt (was \\scriptsize, 7pt in this template, below the paper's 8pt floor;
        # legibility review 2026-09-25)
        head = f"\\strut {short}" if not note else f"\\strut {short}\\\\[-2pt]{{\\fontsize{{8}}{{9}}\\selectfont {note}}}"
        out.append(f"\\node[anchor=south, align=center, font=\\footnotesize, inner sep=1pt] at ({xc:.3f},{top + 0.05:.3f}) {{{head}}};")
    for r, (g, _) in enumerate(GENS):
        y = top - (r + 1) * CELL_H
        for k, (j, _, _) in enumerate(judges):
            d, lo, hi = data[(g, j)]
            fill, ink, bold = cell_style(d, lo, hi)
            x = x0 + k * CELL_W
            out.append(f"\\fill[fill={{rgb,255:red,{int(fill[0:2],16)};green,{int(fill[2:4],16)};blue,{int(fill[4:6],16)}}}] "
                       f"({x + GAP / 2:.3f},{y + GAP / 2:.3f}) rectangle ({x + CELL_W - GAP / 2:.3f},{y + CELL_H - GAP / 2:.3f});")
            txt = fmt(d)
            if bold:
                txt = f"\\textbf{{{txt}}}"
            out.append(f"\\node[font=\\footnotesize, text={{rgb,255:red,{int(ink[0:2],16)};green,{int(ink[2:4],16)};blue,{int(ink[4:6],16)}}}] "
                       f"at ({x + CELL_W / 2:.3f},{y + CELL_H / 2:.3f}) {{{txt}}};")
    return out


def _rgb(h: str) -> str:
    return f"{{rgb,255:red,{int(h[0:2],16)};green,{int(h[2:4],16)};blue,{int(h[4:6],16)}}}"


def legend(x0: float, y: float) -> list[str]:
    """A binned colour bar: four blue bins (NoT lowers the rate), four red bins (NoT raises it),
    tick labels at the bin edges, and a separate neutral swatch for intervals that include zero."""
    w, h = 0.62, 0.26
    blocks = list(reversed(BLUE)) + list(RED)       # darkest blue ... darkest red
    edges = ["$-45$", "$-25$", "$-10$", "0", "$+10$", "$+25$", "$+45$"]
    out = [f"\\node[anchor=east, font=\\footnotesize, align=right] at ({x0 - 0.12:.3f},{y:.3f}) {{NoT lowers the rate}};"]
    for k, fill in enumerate(blocks):
        x = x0 + k * w
        out.append(f"\\fill[fill={_rgb(fill)}] ({x + 0.02:.3f},{y - h / 2:.3f}) rectangle ({x + w - 0.02:.3f},{y + h / 2:.3f});")
    for k, lab in enumerate(edges, start=1):
        x = x0 + k * w
        out.append(f"\\draw[line width=0.4pt, color={_rgb(MUTED)}] ({x:.3f},{y - h / 2 - 0.02:.3f}) -- ({x:.3f},{y - h / 2 - 0.09:.3f});")
        out.append(f"\\node[anchor=north, font=\\fontsize{{8}}{{9}}\\selectfont, inner sep=1pt] at ({x:.3f},{y - h / 2 - 0.09:.3f}) {{{lab}}};")
    xe = x0 + len(blocks) * w
    out.append(f"\\node[anchor=west, font=\\footnotesize] at ({xe + 0.10:.3f},{y:.3f}) {{NoT raises it}};")
    xn = xe + 2.35
    out.append(f"\\fill[fill={_rgb(NEUTRAL)}] ({xn:.3f},{y - h / 2:.3f}) rectangle ({xn + w - 0.04:.3f},{y + h / 2:.3f});")
    out.append(f"\\node[anchor=west, font=\\footnotesize, align=left] at ({xn + w:.3f},{y:.3f}) {{interval\\\\[-2pt]includes 0}};")
    return out


def tikz(val: dict, frame: dict) -> str:
    n = len(GENS)
    lines = ["% generated by scripts/iclr_fig_judge_heatmap.py from headline_rep4k_readout.json (validation,",
             "% prereg 17.10) and framing_17_11_readout.json (accepting framing, prereg 17.11); do not hand-edit",
             "\\begin{tikzpicture}[x=1cm, y=1cm]"]
    for r, (_, short) in enumerate(GENS):
        y = n * CELL_H - (r + 0.5) * CELL_H
        lines.append(f"\\node[anchor=east, font=\\small] at ({LABEL_W - 0.08:.3f},{y:.3f}) {{{short}}};")
    # corner label at 8pt (was \\scriptsize, 7pt) and naming the rows "generator" as the caption and the tables
    # do (was "model", a word that also covers the judges; legibility review 2026-09-25)
    lines.append(f"\\node[anchor=south east, align=right, font=\\fontsize{{8}}{{9}}\\selectfont, text={_rgb(MUTED)}, inner sep=1pt] "
                 f"at ({LABEL_W - 0.08:.3f},{n * CELL_H + 0.05:.3f}) {{judge $\\rightarrow$\\\\ generator $\\downarrow$}};")
    xa = LABEL_W
    lines += panel(xa, VAL_JUDGES, val, "(a) Judged emotional validation")
    xb = xa + len(VAL_JUDGES) * CELL_W + PANEL_GAP
    lines += panel(xb, FRAME_JUDGES, frame, "(b) Accepting the asker's framing")
    lines += legend(LABEL_W + 2.6, LEGEND_Y)
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("[ok] " if cond else "[FAIL] ") + msg)
        ok = ok and cond
    check("interval through zero is neutral", cell_style(-5, -12, 2)[0] == NEUTRAL)
    check("large reduction takes the darkest blue", cell_style(-63.4, -71.8, -54.9)[0] == BLUE[3])
    check("small significant rise takes the lightest red", cell_style(8.0, 1.3, 15.3)[0] == RED[0])
    check("bin edges: 10 goes to the second bin", bin_index(10) == 1 and bin_index(9.9) == 0)
    val, frame = load()
    check("loads 35 validation and 14 framing cells", len(val) == 35 and len(frame) == 14)
    check("production judge Llama validation is the registered -63.4", abs(val[("Llama-3.3-70B-Instruct", "claude-haiku-4-5")][0] + 63.4) < 0.06)
    check("gpt-4o sonnet framing is the registered -26.7", abs(frame[("claude-sonnet-4-6", "gpt-4o")][0] + 26.7) < 0.06)
    print("\nselftest: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    t = tikz(*load())
    if a.out:
        a.out.write_text(t)
        print(f"wrote {a.out}")
    else:
        sys.stdout.write(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
